/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>

#include "libcxi_test_common.h"
#include "uapi/misc/cxi.h"

#define WIN_LENGTH (0x1000)
#define SEND_LENGTH (16)
#define RECV_BUFFER_ID 0xb0f
#define OFLOW_BUFFER_ID 0xb10

TestSuite(msg, .init = data_xfer_setup, .fini = data_xfer_teardown);

/* Test basic send/recv */
Test(msg, simple_send, .timeout = 10)
{
	int pid_idx = 0;
	int xfer_len;
	struct mem_window snd_mem;
	struct mem_window rcv_mem;

	/* Allocate buffers */
	alloc_iobuf(WIN_LENGTH, &snd_mem, CXI_MAP_READ);
	alloc_iobuf(WIN_LENGTH, &rcv_mem, CXI_MAP_WRITE);

	/* Initialize Send Memory */
	for (int i = 0; i < snd_mem.length; i++)
		snd_mem.buffer[i] = i;
	memset(rcv_mem.buffer, 0, rcv_mem.length);

	/* Initialize recv PtlTE */
	ptlte_setup(pid_idx, true, false);

	for (xfer_len = 1; xfer_len <= snd_mem.length; xfer_len <<= 1) {
		/* Post receive buffer */
		append_le_sync(rx_pte, &rcv_mem, C_PTL_LIST_PRIORITY,
			       RECV_BUFFER_ID, 0, 0, CXI_MATCH_ID_ANY, 0,
			       false, false, true, false, true, true, false,
			       NULL);

		/* Initiate Put Operation */
		do_put_sync(snd_mem, xfer_len, 0, 0, pid_idx, false, 0, 0, 0,
			    false);

		/* Gather Unlink event from recv buffer */
		process_eqe(target_evtq, EQE_TGT_LONG, C_EVENT_UNLINK,
			    RECV_BUFFER_ID, NULL);

		/* Gather PUT event from recv buffer */
		process_eqe(target_evtq, EQE_TGT_SHORT, C_EVENT_PUT,
			    RECV_BUFFER_ID, NULL);

		/* Validate send and receive data match */
		for (int i = 0; i < xfer_len; i++)
			cr_expect_eq(snd_mem.buffer[i],
				     rcv_mem.buffer[i],
				     "Data mismatch: idx %2d - %02x != %02x",
				     i, snd_mem.buffer[i],
				     rcv_mem.buffer[i]);
	}

	/* Free receive queue PtlTE */
	ptlte_teardown();

	/* Deallocate buffers */
	free_iobuf(&snd_mem);
	free_iobuf(&rcv_mem);
}

/* Test unexpected send/recv */
Test(msg, ux_send, .timeout = 10)
{
	int pid_idx = 0;
	int xfer_len;
	struct mem_window snd_mem;
	struct mem_window rcv_mem;
	struct mem_window oflow_mem;
	const union c_event *event;
	uint8_t *start;

	/* Allocate buffers */
	alloc_iobuf(WIN_LENGTH, &snd_mem, CXI_MAP_READ);
	alloc_iobuf(WIN_LENGTH, &rcv_mem, CXI_MAP_WRITE);
	alloc_iobuf(WIN_LENGTH * WIN_LENGTH, &oflow_mem, CXI_MAP_WRITE);

	/* Initialize Memory */
	for (int i = 0; i < snd_mem.length; i++)
		snd_mem.buffer[i] = i;
	memset(rcv_mem.buffer, 0, rcv_mem.length);
	memset(oflow_mem.buffer, 0, oflow_mem.length);

	/* Initialize recv PtlTE */
	ptlte_setup(pid_idx, true, false);

	/* Post overflow buffer to match all sends */
	append_le_sync(rx_pte, &oflow_mem, C_PTL_LIST_OVERFLOW,
		       OFLOW_BUFFER_ID, 0, -1ULL, CXI_MATCH_ID_ANY, 1024,
		       false, false, false, true, true, true, false, NULL);

	for (xfer_len = 1; xfer_len <= snd_mem.length; xfer_len <<= 1) {
		/* Initiate Put Operation */
		do_put_sync(snd_mem, xfer_len, 0, 0, pid_idx, false, 0, 0, 0,
			    false);

		/* Gather PUT event from overflow buffer */
		process_eqe(target_evtq, EQE_TGT_LONG, C_EVENT_PUT,
			    OFLOW_BUFFER_ID, NULL);
	}

	for (xfer_len = 1; xfer_len <= snd_mem.length; xfer_len <<= 1) {
		/* Post matching recv to Priority list */
		append_le(rx_pte, &rcv_mem, C_PTL_LIST_PRIORITY,
			  RECV_BUFFER_ID, 0, 0, CXI_MATCH_ID_ANY, 0,
			  false, true, true, false, true, true, false);

		/* Process PUT_OVERFLOW event */
		while (!(event = cxi_eq_get_event(target_evtq)))
			sched_yield();

		cr_assert_eq(event->hdr.event_type, C_EVENT_PUT_OVERFLOW,
			     "Invalid event_type, expected: %u got %u",
			     C_EVENT_PUT_OVERFLOW, event->hdr.event_type);
		cr_assert_eq(event->tgt_long.event_size, C_EVENT_SIZE_64_BYTE);
		cr_assert_eq(event->tgt_long.buffer_id, RECV_BUFFER_ID,
			     "Invalid buffer_id, expected: %u got %u",
			     RECV_BUFFER_ID, event->tgt_long.buffer_id);

		/* Validate send and overflow data match */
		start = (uint8_t *)oflow_mem.md->va +
				(event->tgt_long.start - oflow_mem.md->iova);
		for (int i = 0; i < xfer_len; i++)
			cr_expect_eq(snd_mem.buffer[i],
				     start[i],
				     "Data mismatch: idx %2d - %02x != %02x",
				     i, snd_mem.buffer[i],
				     start[i]);
	}

	/* Free overflow LE and PtlTE */
	unlink_le_sync(rx_pte, C_PTL_LIST_OVERFLOW, OFLOW_BUFFER_ID);
	ptlte_teardown();

	/* Deallocate buffers */
	free_iobuf(&snd_mem);
	free_iobuf(&rcv_mem);
	free_iobuf(&oflow_mem);
}
