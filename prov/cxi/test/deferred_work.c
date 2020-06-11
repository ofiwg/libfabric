/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2020 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <pthread.h>

#include "cxip.h"
#include "cxip_test_common.h"

static void poll_counter_assert(struct fid_cntr *cntr, uint64_t expected_value,
				unsigned int timeout)
{
	int ret;
	struct timespec cur = {};
	struct timespec end;
	uint64_t value;

	ret = clock_gettime(CLOCK_MONOTONIC, &end);
	cr_assert_eq(ret, 0);

	end.tv_sec += timeout;

	while (true) {
		ret = clock_gettime(CLOCK_MONOTONIC, &cur);
		cr_assert_eq(ret, 0);

		value = fi_cntr_read(cntr);
		if (value == expected_value)
			break;

		if (cur.tv_sec > end.tv_sec) {
			// cr_fail doesn't work so fake it
			cr_assert_eq(value, expected_value,
				     "Counter failed to reach expected value: expected=%lu, got=%lu\n",
				     expected_value, value);
			break;
		}
	}
}

void deferred_msg_op_test(bool comp_event, size_t xfer_size,
			  uint64_t trig_thresh, bool is_tagged, uint64_t tag)
{
	int i;
	int ret;
	uint8_t *recv_buf;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	int err = 0;
	fi_addr_t from;
	struct iovec iov = {};
	struct fi_op_msg msg = {};
	struct fi_op_tagged tagged = {};
	struct fi_deferred_work work = {};
	uint64_t expected_rx_flags =
		is_tagged ? FI_TAGGED | FI_RECV : FI_MSG | FI_RECV;
	uint64_t expected_rx_tag = is_tagged ? tag : 0;
	uint64_t expected_tx_flags =
		is_tagged ? FI_TAGGED | FI_SEND : FI_MSG | FI_SEND;

	recv_buf = calloc(1, xfer_size);
	cr_assert(recv_buf);

	send_buf = calloc(1, xfer_size);
	cr_assert(send_buf);

	for (i = 0; i < xfer_size; i++)
		send_buf[i] = i + 0xa0;

	/* Post RX buffer */
	if (is_tagged)
		ret = fi_trecv(cxit_ep, recv_buf, xfer_size, NULL,
			       FI_ADDR_UNSPEC, tag, 0, NULL);
	else
		ret = fi_recv(cxit_ep, recv_buf, xfer_size, NULL,
			      FI_ADDR_UNSPEC, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_recv failed %d", ret);

	/* Send deferred op to self */
	iov.iov_base = send_buf;
	iov.iov_len = xfer_size;

	work.threshold = trig_thresh;
	work.triggering_cntr = cxit_send_cntr;
	work.completion_cntr = cxit_send_cntr;

	if (is_tagged) {
		tagged.ep = cxit_ep;
		tagged.msg.msg_iov = &iov;
		tagged.msg.iov_count = 1;
		tagged.msg.addr = cxit_ep_fi_addr;
		tagged.msg.tag = tag;
		tagged.flags = comp_event ? FI_COMPLETION : 0;

		work.op_type = FI_OP_TSEND;
		work.op.tagged = &tagged;
	} else {
		msg.ep = cxit_ep;
		msg.msg.msg_iov = &iov;
		msg.msg.iov_count = 1;
		msg.msg.addr = cxit_ep_fi_addr;
		msg.flags = comp_event ? FI_COMPLETION : 0;

		work.op_type = FI_OP_SEND;
		work.op.msg = &msg;
	}

	ret = fi_control(&cxit_domain->fid, FI_QUEUE_WORK, &work);
	cr_assert_eq(ret, FI_SUCCESS, "FI_QUEUE_WORK failed %d", ret);

	/* Verify no target event has occurred. */
	ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
	cr_assert_eq(ret, -FI_EAGAIN, "fi_cq_read unexpected value %d", ret);

	ret = fi_cntr_add(cxit_send_cntr, work.threshold);
	cr_assert_eq(ret, FI_SUCCESS, "fi_cntr_add failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, xfer_size, expected_rx_flags, NULL, 0,
			  expected_rx_tag);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	if (comp_event) {
		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

		validate_tx_event(&tx_cqe, expected_tx_flags, NULL);
	} else {
		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
		cr_assert_eq(ret, -FI_EAGAIN, "fi_cq_read unexpected value %d",
			     ret);
	}

	/* Validate sent data */
	for (i = 0; i < xfer_size; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	poll_counter_assert(cxit_send_cntr, work.threshold + 1, 5);

	free(send_buf);
	free(recv_buf);
}


TestSuite(deferred_work, .init = cxit_setup_msg, .fini = cxit_teardown_msg,
	  .timeout = CXIT_DEFAULT_TIMEOUT);


Test(deferred_work, eager_message_comp_event)
{
	deferred_msg_op_test(true, 1024, 123546, false, 0);
}

Test(deferred_work, rendezvous_message_comp_event)
{
	deferred_msg_op_test(true, 1024 * 1024, 123546, false, 0);
}

Test(deferred_work, eager_message_no_comp_event)
{
	deferred_msg_op_test(false, 1024, 123546, false, 0);
}

Test(deferred_work, rendezvous_message_no_comp_event)
{
	deferred_msg_op_test(false, 1024 * 1024, 123546, false, 0);
}

Test(deferred_work, tagged_eager_message_comp_event)
{
	deferred_msg_op_test(true, 1024, 123546, true, 987654321);
}

Test(deferred_work, tagged_rendezvous_message_comp_event)
{
	deferred_msg_op_test(true, 1024 * 1024, 123546, true, 987654321);
}

Test(deferred_work, tagged_eager_message_no_comp_event)
{
	deferred_msg_op_test(false, 1024, 123546, true, 987654321);
}

Test(deferred_work, tagged_rendezvous_message_no_comp_event)
{
	deferred_msg_op_test(false, 1024 * 1024, 123546, true, 987654321);
}

Test(deferred_work, flush_work)
{
	int i;
	int ret;
	uint8_t *recv_buf;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	struct iovec iov = {};
	struct fi_op_msg msg = {};
	struct fi_deferred_work msg_work = {};
	unsigned int trig_thresh;
	size_t xfer_size = 1;
	uint64_t key = 0xbeef;
	struct mem_region mem_window;
	struct fi_rma_iov rma_iov = {};
	struct fi_op_rma rma = {};
	struct fi_deferred_work rma_work = {};
	struct fi_ioc ioc = {};
	struct fi_rma_ioc rma_ioc = {};
	struct fi_op_atomic amo = {};
	struct fi_deferred_work amo_work = {};

	recv_buf = calloc(1, xfer_size);
	cr_assert(recv_buf);

	send_buf = calloc(1, xfer_size);
	cr_assert(send_buf);

	ret = mr_create(xfer_size, FI_REMOTE_WRITE | FI_REMOTE_READ, 0xa0, key,
			&mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed %d", ret);

	for (i = 0; i < xfer_size; i++)
		send_buf[i] = i + 0xa0;

	/* Post RX buffer */
	ret = fi_recv(cxit_ep, recv_buf, xfer_size, NULL, FI_ADDR_UNSPEC, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_recv failed %d", ret);

	/* Send deferred 64 bytes to self */
	msg.ep = cxit_ep;
	iov.iov_base = send_buf;
	iov.iov_len = xfer_size;
	msg.msg.msg_iov = &iov;
	msg.msg.iov_count = 1;
	msg.msg.addr = cxit_ep_fi_addr;
	msg.flags = FI_COMPLETION;

	msg_work.triggering_cntr = cxit_send_cntr;
	msg_work.completion_cntr = cxit_send_cntr;
	msg_work.op_type = FI_OP_SEND;
	msg_work.op.msg = &msg;

	/* Deferred RMA op to be cancelled. */
	rma_iov.key = key;

	rma.ep = cxit_ep;
	rma.msg.msg_iov = &iov;
	rma.msg.iov_count = 1;
	rma.msg.addr = cxit_ep_fi_addr;
	rma.msg.rma_iov = &rma_iov;
	rma.msg.rma_iov_count = 1;
	rma.flags = FI_COMPLETION;

	rma_work.triggering_cntr = cxit_send_cntr;
	msg_work.completion_cntr = cxit_send_cntr;
	rma_work.op_type = FI_OP_READ;
	rma_work.op.rma = &rma;

	/* Deferred AMO op to be cancelled. */
	ioc.addr = &send_buf;
	ioc.count = 1;

	rma_ioc.key = key;
	rma_ioc.count = 1;

	amo.ep = cxit_ep;

	amo.msg.msg_iov = &ioc;
	amo.msg.iov_count = 1;
	amo.msg.addr = cxit_ep_fi_addr;
	amo.msg.rma_iov = &rma_ioc;
	amo.msg.rma_iov_count = 1;
	amo.msg.datatype = FI_UINT8;
	amo.msg.op = FI_SUM;

	amo_work.triggering_cntr = cxit_send_cntr;
	amo_work.completion_cntr = cxit_send_cntr;
	amo_work.op_type = FI_OP_ATOMIC;
	amo_work.op.atomic = &amo;

	/* Queue up multiple trigger requests to be cancelled. */
	for (i = 0, trig_thresh = 12345; i < 9; i++, trig_thresh++) {
		struct fi_deferred_work *work;

		if (i < 3)
			work = &msg_work;
		else if (i < 6)
			work = &rma_work;
		else
			work = &amo_work;

		work->threshold = trig_thresh;

		ret = fi_control(&cxit_domain->fid, FI_QUEUE_WORK, work);
		cr_assert_eq(ret, FI_SUCCESS, "FI_QUEUE_WORK failed %d", ret);
	}

	/* Verify no source or target event has occurred. */
	ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
	cr_assert_eq(ret, -FI_EAGAIN, "fi_cq_read unexpected value %d", ret);

	ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
	cr_assert_eq(ret, -FI_EAGAIN, "fi_cq_read unexpected value %d", ret);

	/* Flush all work requests. */
	ret = fi_control(&cxit_domain->fid, FI_FLUSH_WORK, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "FI_FLUSH_WORK failed %d", ret);

	ret = fi_cntr_add(cxit_send_cntr, trig_thresh);
	cr_assert_eq(ret, FI_SUCCESS, "fi_cntr_add failed %d", ret);

	/* Verify no source or target event has occurred. */
	ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
	cr_assert_eq(ret, -FI_EAGAIN, "fi_cq_read unexpected value %d", ret);

	ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
	cr_assert_eq(ret, -FI_EAGAIN, "fi_cq_read unexpected value %d", ret);

	poll_counter_assert(cxit_send_cntr, trig_thresh, 5);

	free(send_buf);
	free(recv_buf);
	mr_destroy(&mem_window);
}

static void deferred_rma_test(enum fi_op_type op, size_t xfer_size,
			      uint64_t trig_thresh, uint64_t key,
			      bool comp_event)
{
	int ret;
	struct mem_region mem_window;
	struct fi_cq_tagged_entry cqe;
	struct iovec iov = {};
	struct fi_rma_iov rma_iov = {};
	struct fi_op_rma rma = {};
	struct fi_deferred_work work = {};
	struct fid_cntr *trig_cntr = cxit_write_cntr;
	struct fid_cntr *comp_cntr = cxit_read_cntr;
	uint8_t *send_buf;
	uint64_t expected_flags =
		op == FI_OP_WRITE ? FI_RMA | FI_WRITE : FI_RMA | FI_READ;

	send_buf = calloc(1, xfer_size);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	mr_create(xfer_size, FI_REMOTE_WRITE | FI_REMOTE_READ, 0xa0, key,
		  &mem_window);

	iov.iov_base = send_buf;
	iov.iov_len = xfer_size;

	rma_iov.key = key;

	rma.ep = cxit_ep;
	rma.msg.msg_iov = &iov;
	rma.msg.iov_count = 1;
	rma.msg.addr = cxit_ep_fi_addr;
	rma.msg.rma_iov = &rma_iov;
	rma.msg.rma_iov_count = 1;
	rma.flags = comp_event ? FI_COMPLETION : 0;

	work.threshold = trig_thresh;
	work.triggering_cntr = trig_cntr;
	work.completion_cntr = comp_cntr;
	work.op_type = op;
	work.op.rma = &rma;

	ret = fi_control(&cxit_domain->fid, FI_QUEUE_WORK, &work);
	cr_assert_eq(ret, FI_SUCCESS, "FI_QUEUE_WORK failed %d", ret);

	/* Verify no target event has occurred. */
	ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
	cr_assert_eq(ret, -FI_EAGAIN, "fi_cq_read unexpected value %d", ret);

	ret = fi_cntr_add(trig_cntr, work.threshold);
	cr_assert_eq(ret, FI_SUCCESS, "fi_cntr_add failed %d", ret);

	if (comp_event) {
		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

		validate_tx_event(&cqe, expected_flags, NULL);
	} else {
		ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
		cr_assert_eq(ret, -FI_EAGAIN, "fi_cq_read unexpected value %d",
			     ret);
	}

	poll_counter_assert(trig_cntr, work.threshold, 5);
	poll_counter_assert(comp_cntr, 1, 5);

	/* Validate RMA data */
	for (size_t i = 0; i < xfer_size; i++)
		cr_assert_eq(mem_window.mem[i], send_buf[i],
			     "data mismatch, element: (%ld) %02x != %02x\n", i,
			     mem_window.mem[i], send_buf[i]);

	mr_destroy(&mem_window);
	free(send_buf);
}

Test(deferred_work, rma_write)
{
	deferred_rma_test(FI_OP_WRITE, 12345, 54321, 0xbeef, true);
}

Test(deferred_work, rma_write_no_event)
{
	deferred_rma_test(FI_OP_WRITE, 12345, 54321, 0xbeef, false);
}

Test(deferred_work, rma_read)
{
	deferred_rma_test(FI_OP_READ, 12345, 54321, 0xbeef, true);
}

Test(deferred_work, rma_read_no_event)
{
	deferred_rma_test(FI_OP_READ, 12345, 54321, 0xbeef, false);
}

static void deferred_amo_test(bool comp_event)
{
	int ret;
	struct mem_region mem_window;
	struct fi_cq_tagged_entry cqe;
	struct fi_ioc iov = {};
	struct fi_rma_ioc rma_iov = {};
	struct fi_op_atomic amo = {};
	struct fi_deferred_work work = {};
	struct fid_cntr *trig_cntr = cxit_write_cntr;
	struct fid_cntr *comp_cntr = cxit_read_cntr;
	uint64_t expected_flags = FI_ATOMIC | FI_WRITE;
	uint64_t source_buf = 1;
	uint64_t *target_buf;
	uint64_t result;
	uint64_t key = 0xbbbbb;
	uint64_t trig_thresh = 12345;

	ret = mr_create(sizeof(*target_buf), FI_REMOTE_WRITE | FI_REMOTE_READ,
			0, key, &mem_window);
	assert(ret == FI_SUCCESS);

	target_buf = (uint64_t *)mem_window.mem;
	*target_buf = 0x7FFFFFFFFFFFFFFF;

	result = source_buf + *target_buf;

	iov.addr = &source_buf;
	iov.count = 1;

	rma_iov.key = key;
	rma_iov.count = 1;

	amo.ep = cxit_ep;

	amo.msg.msg_iov = &iov;
	amo.msg.iov_count = 1;
	amo.msg.addr = cxit_ep_fi_addr;
	amo.msg.rma_iov = &rma_iov;
	amo.msg.rma_iov_count = 1;
	amo.msg.datatype = FI_UINT64;
	amo.msg.op = FI_SUM;

	amo.flags = comp_event ? FI_COMPLETION : 0;

	work.threshold = trig_thresh;
	work.triggering_cntr = trig_cntr;
	work.completion_cntr = comp_cntr;
	work.op_type = FI_OP_ATOMIC;
	work.op.atomic = &amo;

	ret = fi_control(&cxit_domain->fid, FI_QUEUE_WORK, &work);
	cr_assert_eq(ret, FI_SUCCESS, "FI_QUEUE_WORK failed %d", ret);

	/* Verify no target event has occurred. */
	ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
	cr_assert_eq(ret, -FI_EAGAIN, "fi_cq_read unexpected value %d", ret);

	ret = fi_cntr_add(trig_cntr, work.threshold);
	cr_assert_eq(ret, FI_SUCCESS, "fi_cntr_add failed %d", ret);

	if (comp_event) {
		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

		validate_tx_event(&cqe, expected_flags, NULL);
	} else {
		ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
		cr_assert_eq(ret, -FI_EAGAIN, "fi_cq_read unexpected value %d",
			     ret);
	}

	poll_counter_assert(trig_cntr, work.threshold, 5);
	poll_counter_assert(comp_cntr, 1, 5);

	/* Validate RMA data */
	cr_assert_eq(*target_buf, result, "Invalid target result");

	mr_destroy(&mem_window);
}

Test(deferred_work, amo_no_event)
{
	deferred_amo_test(false);
}

Test(deferred_work, amo_event)
{
	deferred_amo_test(true);
}
