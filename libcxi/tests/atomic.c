/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <poll.h>

#include "libcxi_test_common.h"

#define WIN_BUFFER_ID 0xb0f

TestSuite(atomic, .init = data_xfer_setup, .fini = data_xfer_teardown);

static void set_cstate(struct cxi_cq *cmdq, int write_lac, int restricted)
{
	union c_cmdu cmd = {};
	int rc;

	cmd.c_state.write_lac = write_lac;
	cmd.c_state.restricted = restricted;
	cmd.c_state.event_send_disable = 1;
	cmd.c_state.eq = transmit_evtq->eqn;

	rc = cxi_cq_emit_c_state(cmdq, &cmd.c_state);
	cr_assert_eq(rc, 0, "cxi_cq_emit_c_state failed: %d", rc);

	cxi_cq_ring(cmdq);
}

/* Do Atomic Add64 IDC transaction. */
static void do_idc_amo(struct cxi_md *md, void *buf, uint64_t inc,
		       uint64_t r_off, uint32_t pid_idx)
{
	union c_cmdu cmd = {};
	union c_fab_addr dfa;
	uint8_t idx_ext;
	const union c_event *event;
	bool fetching = (md && buf);
	int rc;

	cxi_build_dfa(dev->info.nid, domain->pid, dev->info.pid_bits,
		      pid_idx, &dfa, &idx_ext);

	cmd.idc_amo.idc_header.dfa = dfa;
	cmd.idc_amo.idc_header.remote_offset = r_off;

	cmd.idc_amo.atomic_op = C_AMO_OP_SUM;
	cmd.idc_amo.atomic_type = C_AMO_TYPE_UINT64_T;
	cmd.idc_amo.local_addr = (md && buf) ? CXI_VA_TO_IOVA(md, buf) : 0L;
	cmd.idc_amo.op1_word1 = inc;

	rc = cxi_cq_emit_idc_amo(transmit_cmdq, &cmd.idc_amo, fetching);
	cr_assert_eq(rc, 0, "During %s CQ emit failed %d", __func__, rc);

	cxi_cq_ring(transmit_cmdq);

	/* Wait for EQ event */
	while (!(event = cxi_eq_get_event(transmit_evtq)))
		sched_yield();

	if (fetching) {
		cr_assert_eq(event->hdr.event_type, C_EVENT_REPLY,
			     "Invalid event_type, expected: %u got %u",
			     C_EVENT_REPLY, event->hdr.event_type);
	} else {
		cr_assert_eq(event->hdr.event_type, C_EVENT_ACK,
			     "Invalid event_type, expected: %u got %u",
			     C_EVENT_ACK, event->hdr.event_type);
	}

	/* Make sure the queue is empty before acknowledging */
	event = cxi_eq_get_event(transmit_evtq);
	cr_assert_null(event, "NULL event expected\n");

	cxi_eq_ack_events(transmit_evtq);
}

/* Test basic IDC atomic */
Test(atomic, idc, .timeout = 10)
{
	int win_len = 0x1000;
	int amo_off = 0;
	int pid_idx = 0;
	struct mem_window rem_mem;
	struct mem_window loc_mem;
	uint64_t *remote_value;
	uint64_t *local_value;
	uint64_t increment;
	uint64_t rem_expected;
	uint64_t rtn_expected;

	/* Allocate buffers */
	alloc_iobuf(win_len, &rem_mem, CXI_MAP_WRITE | CXI_MAP_READ);
	alloc_iobuf(win_len, &loc_mem, CXI_MAP_WRITE | CXI_MAP_READ);

	/* IDC AMO performs AMO on remote node */
	remote_value = (uint64_t *)&rem_mem.buffer[amo_off];

	/* IDC AMO places fetching AMO here */
	local_value = (uint64_t *)loc_mem.buffer;

	/* Set write_lac, restricted commands */
	set_cstate(transmit_cmdq, loc_mem.md->lac, 1);

	/* Initialize RMA PtlTE */
	ptlte_setup(pid_idx, false, false);

	/* Post RMA Buffer */
	append_le(rx_pte, &rem_mem, C_PTL_LIST_PRIORITY, WIN_BUFFER_ID,
		  0, 0, CXI_MATCH_ID_ANY, 0, true, false, false, false, true,
		  true, true);
	cr_assert(!((uint64_t)rem_mem.buffer & 0x7));
	process_eqe(target_evtq, EQE_TGT_LONG, C_EVENT_LINK,
		    WIN_BUFFER_ID, NULL);

	/* Test initial AMO remote value */
	rtn_expected = 0;
	rem_expected = 0;
	cr_assert_eq(*remote_value, rem_expected, "Remote value = %ld != %ld\n",
		     *remote_value, rem_expected);

	/* Test IDC AMO add64 +1 */
	increment = 1;
	rtn_expected = rem_expected;
	rem_expected += increment;
	do_idc_amo(NULL, NULL, increment, amo_off, pid_idx);
	cr_assert_eq(*remote_value, rem_expected, "Remote value = %ld != %ld\n",
		     *remote_value, rem_expected);

	/* Test IDC AMO add64 +5 */
	increment = 5;
	rtn_expected = rem_expected;
	rem_expected += increment;
	do_idc_amo(NULL, NULL, increment, amo_off, pid_idx);
	cr_assert_eq(*remote_value, rem_expected, "Remote value = %ld != %ld\n",
		     *remote_value, rem_expected);

	/* Test IDC FETCHING add64 +7 */
	increment = 7;
	rtn_expected = rem_expected;
	rem_expected += increment;
	do_idc_amo(loc_mem.md, loc_mem.buffer, increment, amo_off, pid_idx);
	cr_assert_eq(*remote_value, rem_expected, "Remote value = %ld != %ld\n",
		     *remote_value, rem_expected);
	cr_assert_eq(*local_value, rtn_expected, "Result value = %ld != %ld\n",
		     *local_value, rtn_expected);

	/* Clean up */
	unlink_le_sync(rx_pte, C_PTL_LIST_PRIORITY, WIN_BUFFER_ID);

	ptlte_teardown();

	free_iobuf(&loc_mem);
	free_iobuf(&rem_mem);
}

/* Do Atomic Add64 DMA transaction. */
static void do_dma_amo(struct cxi_md *md, void *inc, void *res,
		       uint64_t r_off, uint32_t pid_idx, bool fetching)
{
	union c_cmdu cmd = {};
	union c_fab_addr dfa;
	uint8_t idx_ext;
	const union c_event *event;
	int rc;

	cxi_build_dfa(dev->info.nid, domain->pid, dev->info.pid_bits,
		      pid_idx, &dfa, &idx_ext);

	cmd.dma_amo.index_ext = idx_ext;
	cmd.dma_amo.lac = md->lac;
	cmd.dma_amo.event_send_disable = 1;
	cmd.dma_amo.restricted = 1;
	cmd.dma_amo.dfa = dfa;
	cmd.dma_amo.remote_offset = r_off;
	cmd.dma_amo.local_read_addr = CXI_VA_TO_IOVA(md, inc);
	cmd.dma_amo.local_write_addr = CXI_VA_TO_IOVA(md, res);

	cmd.dma_amo.eq = transmit_evtq->eqn;

	cmd.dma_amo.atomic_op = C_AMO_OP_SUM;
	cmd.dma_amo.atomic_type = C_AMO_TYPE_UINT64_T;

	rc = cxi_cq_emit_dma_amo(transmit_cmdq, &cmd.dma_amo, fetching);
	cr_assert_eq(rc, 0, "During %s CQ emit failed %d", __func__, rc);

	cxi_cq_ring(transmit_cmdq);

	/* Wait for EQ event */
	while (!(event = cxi_eq_get_event(transmit_evtq)))
		sched_yield();

	if (fetching) {
		cr_assert_eq(event->hdr.event_type, C_EVENT_REPLY,
			     "Invalid event_type, expected: %u got %u",
			     C_EVENT_REPLY, event->hdr.event_type);
	} else {
		cr_assert_eq(event->hdr.event_type, C_EVENT_ACK,
			     "Invalid event_type, expected: %u got %u",
			     C_EVENT_ACK, event->hdr.event_type);
	}

	/* Make sure the queue is empty before acknowledging */
	event = cxi_eq_get_event(transmit_evtq);
	cr_assert_null(event, "NULL event expected\n");

	cxi_eq_ack_events(transmit_evtq);
}

/* Test basic DMA atomic */
/* NOTE: nic-emu does not yet support DMA AMO operations. */
Test(atomic, dma, .timeout = 5)
{
	int win_len = 0x1000;
	int amo_off = 8;
	int pid_idx = 0;
	struct mem_window loc_mem;
	struct mem_window rem_mem;
	uint64_t *remote_value;
	uint64_t *increment;
	uint64_t *result;
	uint64_t rem_expected;
	uint64_t inc_expected;
	uint64_t rtn_expected;

	/* Allocate buffers */
	alloc_iobuf(win_len, &rem_mem, CXI_MAP_WRITE|CXI_MAP_READ);
	alloc_iobuf(win_len, &loc_mem, CXI_MAP_WRITE|CXI_MAP_READ);

	/* DMA AMO performs AMO on remote node */
	remote_value = (uint64_t *)&rem_mem.buffer[amo_off];

	/* DMA AMO takes op1 via DMA on this node */
	increment = (uint64_t *)&loc_mem.buffer[0];

	/* DMA AMO deposits results here */
	result = (uint64_t *)&loc_mem.buffer[8];

	/* Initialize RMA PtlTE */
	ptlte_setup(pid_idx, false, false);

	/* Post RMA Buffer */
	append_le(rx_pte, &rem_mem, C_PTL_LIST_PRIORITY, WIN_BUFFER_ID,
		  0, 0, CXI_MATCH_ID_ANY, 0, true, false, false, false, true,
		  true, true);
	process_eqe(target_evtq, EQE_TGT_LONG, C_EVENT_LINK,
		    WIN_BUFFER_ID, NULL);

	/* Test initialization of result */
	rem_expected = 0;
	cr_assert_eq(*remote_value, rem_expected, "Remote value = %ld != %ld\n",
		     *remote_value, rem_expected);

	*increment = 2;
	*result = -1;
	inc_expected = *increment;
	rtn_expected = *result;
	rem_expected += *increment;
	do_dma_amo(loc_mem.md, increment, result, amo_off, pid_idx, false);
	cr_assert_eq(*remote_value, rem_expected, "Remote value = %ld != %ld\n",
		     *remote_value, rem_expected);
	cr_assert_eq(*increment, inc_expected, "Increment value = %ld != %ld\n",
		     *increment, inc_expected);
	cr_assert_eq(*result, rtn_expected, "Return value = %ld != %ld\n",
		     *result, rtn_expected);

	*increment = 7;
	*result = -1;
	inc_expected = *increment;
	rtn_expected = *result;
	rem_expected += *increment;
	do_dma_amo(loc_mem.md, increment, result, amo_off, pid_idx, false);
	cr_assert_eq(*remote_value, rem_expected, "Remote value = %ld != %ld\n",
		     *remote_value, rem_expected);
	cr_assert_eq(*increment, inc_expected, "Increment value = %ld != %ld\n",
		     *increment, inc_expected);
	cr_assert_eq(*result, rtn_expected, "Return value = %ld != %ld\n",
		     *result, rtn_expected);

	*increment = 8;
	*result = -1;
	inc_expected = *increment;
	rtn_expected = *remote_value;
	rem_expected += *increment;
	do_dma_amo(loc_mem.md, increment, result, amo_off, pid_idx, true);
	cr_assert_eq(*remote_value, rem_expected, "Remote value = %ld != %ld\n",
		     *remote_value, rem_expected);
	cr_assert_eq(*increment, inc_expected, "Increment value = %ld != %ld\n",
		     *increment, inc_expected);
	cr_assert_eq(*result, rtn_expected, "Return value = %ld != %ld\n",
		     *result, rtn_expected);

	/* Clean up */
	unlink_le_sync(rx_pte, C_PTL_LIST_PRIORITY, WIN_BUFFER_ID);

	ptlte_teardown();

	free_iobuf(&loc_mem);
	free_iobuf(&rem_mem);
}
