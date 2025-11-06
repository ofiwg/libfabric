/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>

#include "libcxi_test_common.h"
#include "uapi/misc/cxi.h"

#define WIN_LENGTH        (0x1000)
#define EAGER_SEND_LENGTH (16)
#define SEND_BUFFER_ID    (0xb00)
#define RECV_BUFFER_ID    (0xb0f)
#define EAGER_BUFFER_ID   (0xb10)
#define UX_BUFFER_ID      (0xb1f)
#define LONG_SEND_LENGTH  (2048)
#define ACK_USER_PTR      (0xf00dUL)
#define RPLY_USER_PTR     (0xca5cadeUL)
#define TEST_RDV_ID       (0x54)

#define RDV_PROT_SHIFT    (63)
#define RDV_PROT_MASK     (1ULL << RDV_PROT_SHIFT)
#define RDV_PROT_BIT      (1ULL << (RDV_PROT_SHIFT))
#define RDV_ID_SHIFT      (56)
#define RDV_ID_MASK       (0x7fULL << RDV_ID_SHIFT)
#define RDV_ID(id)        (RDV_ID_MASK & (((uint64_t)id) << RDV_ID_SHIFT))
#define GET_RDV_ID(match) ((RDV_ID_MASK & match) >> RDV_ID_SHIFT)

#define INIT_PID(init) ((init) >> 24)
#define INIT_NID(init) ((init) & ((1 << 24) - 1))

struct pte_info {
	struct cxil_pte *pte;
	struct cxil_pte_map *map;
	uint8_t ptl_list;
	struct cxi_eq *evtq;
};

static inline void fill_buffer(void *buffer, size_t length, uint16_t seed)
{
	uint16_t *buff = (uint16_t *)buffer;

	length /= sizeof(uint16_t);
	for (int i = 0; i < length; i += 2, seed++) {
		buff[i] = seed;
		buff[i + 1] = ~seed;
	}
}

static void pte_info_init(int pid_idx, struct pte_info *pte_info)
{
	int rc;
	struct cxi_pt_alloc_opts pt_opts = {.is_matching = 1};
	union c_cmdu cmd = {};
	union c_event event;

	/* Allocate PTE */
	rc = cxil_alloc_pte(lni, pte_info->evtq, &pt_opts, &pte_info->pte);
	cr_assert_eq(rc, 0, "cxil_alloc_pte failed %d", rc);

	/* Map */
	rc = cxil_map_pte(pte_info->pte, domain, pid_idx, false,
			  &pte_info->map);
	cr_assert_eq(rc, 0, "cxil_map_pte failed %d", rc);

	/* Enable */
	cmd.set_state.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = pte_info->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;

	rc = cxi_cq_emit_target(target_cmdq, &cmd);
	cr_assert_eq(rc, 0, "cxi_cq_emit_target failed %d", rc);

	cxi_cq_ring(target_cmdq);

	/* Wait for enable response EQ event */
	process_eqe(pte_info->evtq, EQE_TGT_LONG, C_EVENT_STATE_CHANGE,
		    -1, &event);

	cr_assert_eq(event.tgt_long.initiator.state_change.ptlte_state,
		     C_PTLTE_ENABLED, "Invalid state, got %d",
		     event.tgt_long.initiator.state_change.ptlte_state);
	cr_assert_eq(event.tgt_long.ptlte_index, pte_info->pte->ptn,
		     "Invalid ptn, %d != %d", event.tgt_long.ptlte_index,
		     pte_info->pte->ptn);
}

static void pte_info_teardown(struct pte_info *info)
{
	int rc;

	rc = cxil_unmap_pte(info->map);
	cr_assert_eq(rc, 0, "TX cxil_unmap_pte failed %d", rc);

	rc = cxil_destroy_pte(info->pte);
	cr_assert_eq(rc, 0, "RX cxil_destroy_pte failed %d", rc);
}

TestSuite(rendezvous, .init = data_xfer_setup, .fini = data_xfer_teardown);

/* Test eager rendezvous send */
Test(rendezvous, eager, .timeout = 5)
{
	uint8_t rx_idx = 0;
	uint8_t tx_idx = 255;
	int off = 0;
	struct mem_window snd_mem;
	struct mem_window rcv_mem;
	struct mem_window eager_mem;
	struct mem_window ux_mem;
	union c_event event;
	uint8_t *start;
	struct pte_info tx_pte_info = {
		.evtq = transmit_evtq,
		.ptl_list = C_PTL_LIST_PRIORITY
	};
	uint64_t match = 0;

	/* TX/RX: Allocate buffers and Initialize Memory */
	alloc_iobuf(WIN_LENGTH, &snd_mem, CXI_MAP_READ | CXI_MAP_WRITE);
	alloc_iobuf(WIN_LENGTH, &rcv_mem, CXI_MAP_WRITE);
	alloc_iobuf(WIN_LENGTH, &eager_mem, CXI_MAP_WRITE);
	alloc_iobuf(1, &ux_mem, CXI_MAP_READ);

	fill_buffer(snd_mem.buffer, snd_mem.length, 0x9A52);
	memset(rcv_mem.buffer, 0, rcv_mem.length);
	memset(eager_mem.buffer, 0, eager_mem.length);

	/* Initialize TX/RX PtlTEs */
	pte_info_init(tx_idx, &tx_pte_info);
	ptlte_setup(rx_idx, true, false);

	/* TX: Post Send buffer */
	append_le(tx_pte_info.pte, &snd_mem, C_PTL_LIST_PRIORITY,
		  SEND_BUFFER_ID, match, ~(RDV_ID_MASK | RDV_PROT_MASK),
		  CXI_MATCH_ID_ANY, 0, false, false, true, false, true, false,
		  true);
	process_eqe(transmit_evtq, EQE_TGT_LONG, C_EVENT_LINK,
		    SEND_BUFFER_ID, NULL);

	/* RX: Post eager buffer to match all eager length puts */
	append_le_sync(rx_pte, &eager_mem, C_PTL_LIST_OVERFLOW,
		       EAGER_BUFFER_ID, 0, ~RDV_PROT_MASK, CXI_MATCH_ID_ANY,
		       1024, false, false, false, true, true, true, false,
		       NULL);

	/* RX: Post unexpected header buffer to match all rendezvous puts */
	append_le_sync(rx_pte, &ux_mem, C_PTL_LIST_OVERFLOW,
		       UX_BUFFER_ID, RDV_PROT_BIT, ~RDV_PROT_MASK,
		       CXI_MATCH_ID_ANY, 0, false, true, false, true, false,
		       true, false, NULL);

	/* TX: Initiate Eager Put Operation */
	do_put_sync(snd_mem, EAGER_SEND_LENGTH, 0, 0, rx_idx, false, match,
		    0, 0, false);

	/* RX: Gather PUT event from overflow buffer */
	process_eqe(target_evtq, EQE_TGT_LONG, C_EVENT_PUT,
		    EAGER_BUFFER_ID, NULL);

	/* RX: Post matching recv to Priority list */
	append_le(rx_pte, &rcv_mem, C_PTL_LIST_PRIORITY,
		  RECV_BUFFER_ID, match, 0, CXI_MATCH_ID_ANY, 0,
		  false, true, true, false, false, true, false);

	/* RX: Process PUT_OVERFLOW event */
	process_eqe(target_evtq, EQE_TGT_LONG, C_EVENT_PUT_OVERFLOW,
		    RECV_BUFFER_ID, &event);

	/* TX/RX: Validate send and overflow data match */
	start = (uint8_t *)eager_mem.md->va +
			(event.tgt_long.start - eager_mem.md->iova);
	for (int i = 0; i < EAGER_SEND_LENGTH; i++)
		cr_expect_eq(snd_mem.buffer[i],
			     start[i],
			     "Data mismatch: idx %2d - %02x != %02x",
			     off + i, snd_mem.buffer[i],
			     start[i]);

	/* TX/RX: Free overflow LE and PtlTE */
	unlink_le_sync(rx_pte, C_PTL_LIST_OVERFLOW, UX_BUFFER_ID);
	unlink_le_sync(rx_pte, C_PTL_LIST_OVERFLOW, EAGER_BUFFER_ID);
	unlink_le(tx_pte_info.pte, C_PTL_LIST_PRIORITY, SEND_BUFFER_ID);
	process_eqe(transmit_evtq, EQE_TGT_LONG, C_EVENT_UNLINK,
		    SEND_BUFFER_ID, NULL);

	ptlte_teardown();
	pte_info_teardown(&tx_pte_info);

	/* TX/RX: Deallocate buffers */
	free_iobuf(&snd_mem);
	free_iobuf(&rcv_mem);
	free_iobuf(&eager_mem);
	free_iobuf(&ux_mem);
}

/* Test long rendezvous send */
Test(rendezvous, long, .timeout = 5)
{
	uint8_t rx_idx = 5;
	uint8_t tx_idx = 255;
	size_t off, errs;
	struct mem_window snd_mem;
	struct mem_window rcv_mem;
	struct mem_window eager_mem;
	struct mem_window ux_mem;
	union c_event event;
	const union c_event *evt;
	struct pte_info tx_pte_info = {
		.evtq = transmit_evtq,
		.ptl_list = C_PTL_LIST_PRIORITY
	};
	uint64_t match = (RDV_PROT_BIT | RDV_ID(TEST_RDV_ID));
	uint32_t init_nid;
	uint32_t init_pid;

	/* TX/RX: Allocate buffers and Initialize Memory */
	alloc_iobuf(WIN_LENGTH, &snd_mem, CXI_MAP_READ);
	alloc_iobuf(WIN_LENGTH, &eager_mem, CXI_MAP_WRITE);
	alloc_iobuf(1, &ux_mem, CXI_MAP_READ | CXI_MAP_WRITE);

	fill_buffer(snd_mem.buffer, snd_mem.length, 0x56BC);
	memset(eager_mem.buffer, 0, eager_mem.length);

	/* TX/RX: Initialize PtlTEs */
	pte_info_init(tx_idx, &tx_pte_info);
	ptlte_setup(rx_idx, true, false);

	/* TX: Post Send buffer */
	append_le(tx_pte_info.pte, &snd_mem, C_PTL_LIST_PRIORITY,
		  SEND_BUFFER_ID, match, ~(RDV_ID_MASK | RDV_PROT_MASK),
		  CXI_MATCH_ID_ANY, 0, false, false, true, false, true, false,
		  true);
	process_eqe(transmit_evtq, EQE_TGT_LONG, C_EVENT_LINK,
		    SEND_BUFFER_ID, NULL);

	/* RX: Post eager buffer to match all eager length puts */
	append_le_sync(rx_pte, &eager_mem, C_PTL_LIST_OVERFLOW,
		       EAGER_BUFFER_ID, 0, ~RDV_PROT_MASK, CXI_MATCH_ID_ANY,
		       1024, false, false, false, true, true, true, false,
		       NULL);

	/* RX: Post unexpected header buffer to match all rendezvous puts */
	append_le_sync(rx_pte, &ux_mem, C_PTL_LIST_OVERFLOW,
		       UX_BUFFER_ID, RDV_PROT_BIT, ~RDV_PROT_MASK,
		       CXI_MATCH_ID_ANY, 0, false, true, false, true, false,
		       true, false, NULL);

	/* TX:
	 * Initiate Rendezvous Operation
	 */
	do_put(snd_mem, LONG_SEND_LENGTH, 0, 0, rx_idx, 0, match, ACK_USER_PTR,
	       (domain->pid << 24) | dev->info.nid, false);

	/* RX:
	 * Notification of match of the unexpected buffer from the PUT event
	 */
	process_eqe(target_evtq, EQE_TGT_LONG, C_EVENT_PUT,
		    UX_BUFFER_ID, &event);
	cr_assert_eq(event.tgt_long.return_code, C_RC_OK,
		     "Invalid return_code, got %x", event.tgt_long.return_code);

	/* RX:
	 * Allocate the recv memory for the rendezvous put
	 */
	alloc_iobuf(event.tgt_long.rlength, &rcv_mem, CXI_MAP_WRITE);
	memset(rcv_mem.buffer, 0, rcv_mem.length);

	/* RX:
	 * Post the recv memory to the priority list
	 */
	append_le(rx_pte, &rcv_mem, C_PTL_LIST_PRIORITY,
		  RECV_BUFFER_ID, event.tgt_long.match_bits, 0,
		  CXI_MATCH_ID_ANY, 0, false, true, true, false, false, true,
		  false);

	/* RX:
	 * Notification of a match to the RECV_BUFFER_ID that was just posted
	 * from the PUT OVERFLOW event
	 */
	process_eqe(target_evtq, EQE_TGT_LONG, C_EVENT_PUT_OVERFLOW,
		    RECV_BUFFER_ID, &event);
	cr_assert_eq(event.tgt_long.return_code, C_RC_OK,
		     "Invalid return_code, got %x",
		     event.tgt_long.return_code);

	/* TX:
	 * The ACK event notification that not all data was xfered, a GET event
	 * is to be expected at a later point to indicate the xfer is complete
	 */
	while (!(evt = cxi_eq_get_event(transmit_evtq)))
		sched_yield();
	cr_assert_not_null(evt, "Got NULL event");
	cr_assert_eq(evt->hdr.event_type, C_EVENT_ACK,
		     "Invalid event_type, got %u", evt->hdr.event_type);
	cr_assert_eq(evt->init_short.user_ptr, ACK_USER_PTR,
		     "Invalid user_ptr, got %lx", evt->init_short.user_ptr);
	cr_assert_eq(evt->init_short.return_code, C_RC_OK,
		     "Invalid return_code, got %x",
		     evt->init_short.return_code);
	cr_assert_eq(evt->init_short.user_ptr, ACK_USER_PTR,
		     "ACK event user_ptr not expected %lx",
		     evt->init_short.user_ptr);
	cxi_eq_ack_events(transmit_evtq);

	/* Make sure initiator NID/PID match our only Domain */
	init_nid = INIT_NID(event.tgt_long.initiator.initiator.process);
	init_pid = INIT_PID(event.tgt_long.initiator.initiator.process);
	cr_assert_eq(init_nid, dev->info.nid,
		     "Invalid initiator NID, expected: %u, got: %u\n",
		     dev->info.nid, init_nid);
	cr_assert_eq(init_pid, domain->pid,
		     "Invalid initiator PID, expected: %u, got: %u\n",
		     domain->pid, init_pid);

	/* RX:
	 * Issue a Get for the remaining data
	 */
	do_get(rcv_mem, event.tgt_long.rlength, 0, tx_idx, 0,
	       event.tgt_long.match_bits, RPLY_USER_PTR,
	       0, target_evtq);

	/* RX:
	 * Get the REPLY event indicating the data has arrived in the receive
	 * buffer. The operation on the receive side is now complete.
	 */
	while (!(evt = cxi_eq_get_event(target_evtq)))
		sched_yield();
	cr_assert_not_null(evt, "Got NULL event");
	cr_assert_eq(evt->hdr.event_type, C_EVENT_REPLY,
		     "Invalid event_type, got %u", evt->hdr.event_type);
	cr_assert_eq(evt->init_short.user_ptr, RPLY_USER_PTR,
		     "Invalid user_ptr, got %lx", evt->init_short.user_ptr);
	cr_assert_eq(evt->init_short.return_code, C_RC_OK,
		    "Invalid return_code, got %x", evt->init_short.return_code);
	cr_assert_eq(evt->init_short.user_ptr, RPLY_USER_PTR,
		    "REPLY event user_ptr not expected %lx",
		    evt->init_short.user_ptr);
	cxi_eq_ack_events(target_evtq);

	/* UNLINK is occurring before GET. */
	process_eqe(transmit_evtq, EQE_TGT_LONG, C_EVENT_UNLINK,
		    SEND_BUFFER_ID, NULL);

	/* TX:
	 * The GET event notifies the initiator that the Rendezvous PUT
	 * operation is complete
	 */
	process_eqe(transmit_evtq, EQE_TGT_SHORT, C_EVENT_GET,
		    SEND_BUFFER_ID, &event);
	cr_assert_eq(event.tgt_long.return_code, C_RC_OK,
		     "Invalid return_code, got %x",
		     event.tgt_long.return_code);

	/* TX/RX: Validate send and receive data buffers match */
	for (off = 0, errs = 0; off < LONG_SEND_LENGTH; off++) {
		if (errs < 8)
			cr_expect_eq(snd_mem.buffer[off], rcv_mem.buffer[off],
				     "Mismatch: idx %2ld - %02X != %02X", off,
				     snd_mem.buffer[off], rcv_mem.buffer[off]);

		if (snd_mem.buffer[off] != rcv_mem.buffer[off])
			errs++;
	}
	if (errs)
		cr_log_info("Total bytes with mismatch: %ld", errs);

	/* TX/RX: Free overflow LE and PtlTE */
	unlink_le_sync(rx_pte, C_PTL_LIST_OVERFLOW, UX_BUFFER_ID);
	unlink_le_sync(rx_pte, C_PTL_LIST_OVERFLOW, EAGER_BUFFER_ID);

	ptlte_teardown();
	pte_info_teardown(&tx_pte_info);

	/* TX/RX: Deallocate buffers */
	free_iobuf(&snd_mem);
	free_iobuf(&rcv_mem);
	free_iobuf(&eager_mem);
	free_iobuf(&ux_mem);
}
