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
			  uint64_t trig_thresh)
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
	struct fi_deferred_work work = {};

	recv_buf = calloc(1, xfer_size);
	cr_assert(recv_buf);

	send_buf = calloc(1, xfer_size);
	cr_assert(send_buf);

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
	msg.flags = comp_event ? FI_COMPLETION : 0;

	work.threshold = trig_thresh;
	work.triggering_cntr = cxit_send_cntr;
	work.completion_cntr = cxit_send_cntr;
	work.op_type = FI_OP_SEND;
	work.op.msg = &msg;

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

	validate_rx_event(&rx_cqe, NULL, xfer_size, FI_MSG | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	if (comp_event) {
		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

		validate_tx_event(&tx_cqe, FI_MSG | FI_SEND, NULL);
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
	deferred_msg_op_test(true, 1024, 123546);
}

Test(deferred_work, rendezvous_message_comp_event)
{
	deferred_msg_op_test(true, 1024 * 1024, 123546);
}

Test(deferred_work, eager_message_no_comp_event)
{
	deferred_msg_op_test(false, 1024, 123546);
}

Test(deferred_work, rendezvous_message_no_comp_event)
{
	deferred_msg_op_test(false, 1024 * 1024, 123546);
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
	struct fi_deferred_work work = {};
	unsigned int trig_thresh;
	size_t xfer_size = 1;

	recv_buf = calloc(1, xfer_size);
	cr_assert(recv_buf);

	send_buf = calloc(1, xfer_size);
	cr_assert(send_buf);

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

	work.triggering_cntr = cxit_send_cntr;
	work.completion_cntr = cxit_send_cntr;
	work.op_type = FI_OP_SEND;
	work.op.msg = &msg;

	/* Queue up multiple trigger requests to be cancelled. */
	for (i = 0, trig_thresh = 12345; i < 15; i++, trig_thresh++) {
		work.threshold = trig_thresh;
		ret = fi_control(&cxit_domain->fid, FI_QUEUE_WORK, &work);
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
}
