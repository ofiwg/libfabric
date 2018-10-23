/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <pthread.h>

#include "cxip.h"
#include "cxip_test_common.h"

TestSuite(tagged, .init = cxit_setup_tagged, .fini = cxit_teardown_tagged);

/* Test basic send/recv */
Test(tagged, ping, .timeout = 3)
{
	int i, ret;
	uint8_t *recv_buf,
		*send_buf;
	int recv_len = 64;
	int send_len = 64;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	/* Post RX buffer */
	ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL, FI_ADDR_UNSPEC, 0,
		       0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);

	/* Send 64 bytes to FI address 0 (self) */
	ret = fi_tsend(cxit_ep, send_buf, send_len, NULL, 0, 0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);

	/* Wait for async event indicating data has been received */
	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	/* Validate RX event fields */
	cr_assert(rx_cqe.op_context == NULL, "RX CQE Context mismatch");
	cr_assert(rx_cqe.flags == (FI_TAGGED | FI_RECV),
		  "RX CQE flags mismatch");
	cr_assert(rx_cqe.len == send_len, "Invalid RX CQE length");
	cr_assert(rx_cqe.buf == 0, "Invalid RX CQE address");
	cr_assert(rx_cqe.data == 0, "Invalid RX CQE data");
	cr_assert(rx_cqe.tag == 0, "Invalid RX CQE tag");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	/* Validate TX event fields */
	cr_assert(tx_cqe.op_context == NULL, "TX CQE Context mismatch");
	cr_assert(tx_cqe.flags == (FI_TAGGED | FI_SEND),
		  "TX CQE flags mismatch");
	cr_assert(tx_cqe.len == 0, "Invalid TX CQE length");
	cr_assert(tx_cqe.buf == 0, "Invalid TX CQE address");
	cr_assert(tx_cqe.data == 0, "Invalid TX CQE data");
	cr_assert(tx_cqe.tag == 0, "Invalid TX CQE tag");

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	free(send_buf);
	free(recv_buf);
}

/* Test unexpected send/recv */
Test(tagged, ux_ping, .timeout = 3)
{
	int i, ret;
	uint8_t *recv_buf,
		*send_buf;
	int recv_len = 64;
	int send_len = 64;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	/* Send 64 bytes to FI address 0 (self) */
	ret = fi_tsend(cxit_ep, send_buf, send_len, NULL, 0, 0, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Give some time for the message to move */
	sleep(1);

	/* Post RX buffer */
	ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL, FI_ADDR_UNSPEC, 0,
		       0, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Wait for async event indicating data has been received */
	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert(ret == 1);

	/* Validate RX event fields */
	cr_assert(rx_cqe.op_context == NULL, "RX CQE Context mismatch");
	cr_assert(rx_cqe.flags == (FI_TAGGED | FI_RECV),
		  "RX CQE flags mismatch");
	cr_assert(rx_cqe.len == send_len, "Invalid RX CQE length");
	cr_assert(rx_cqe.buf == 0, "Invalid RX CQE address");
	cr_assert(rx_cqe.data == 0, "Invalid RX CQE data");
	cr_assert(rx_cqe.tag == 0, "Invalid RX CQE tag");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert(ret == 1);

	/* Validate TX event fields */
	cr_assert(tx_cqe.op_context == NULL, "TX CQE Context mismatch");
	cr_assert(tx_cqe.flags == (FI_TAGGED | FI_SEND),
		  "TX CQE flags mismatch");
	cr_assert(tx_cqe.len == 0, "Invalid TX CQE length");
	cr_assert(tx_cqe.buf == 0, "Invalid TX CQE address");
	cr_assert(tx_cqe.data == 0, "Invalid TX CQE data");
	cr_assert(tx_cqe.tag == 0, "Invalid TX CQE tag");

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_assert(recv_buf[i] == send_buf[i],
			  "data mismatch, element: %d\n", i);
	}

	free(send_buf);
	free(recv_buf);
}

/* Test DIRECTED_RECV send/recv */
Test(tagged, directed, .timeout = 3)
{
	int i, ret;
	uint8_t *recv_buf,
		*fake_recv_buf,
		*send_buf;
	int recv_len = 64;
	int send_len = 64;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
	struct cxip_addr fake_ep_addr = { .nic = 0xbad, .pid = 0xba };
	fi_addr_t from;

	/* Insert non-existent peer addr into AV (addr 1) */
	ret = fi_av_insert(cxit_av, (void *)&fake_ep_addr, 1, NULL, 0, NULL);
	cr_assert(ret == 1);

	recv_buf = calloc(recv_len, 1);
	cr_assert(recv_buf);

	fake_recv_buf = calloc(recv_len, 1);
	cr_assert(fake_recv_buf);

	send_buf = malloc(send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	/* Post RX buffer matching non-existent peer (addr 2) */
	ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL, 2, 0, 0, NULL);
	cr_assert(ret == -FI_EINVAL);

	/* Post RX buffer matching fake peer (addr 1) */
	ret = fi_trecv(cxit_ep, fake_recv_buf, recv_len, NULL, 1, 0, 0, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Post RX buffer matching self (addr 0) */
	ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL, 0, 0, 0, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Send 64 bytes to FI address 0 (self) */
	ret = fi_tsend(cxit_ep, send_buf, send_len, NULL, 0, 0, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == 1);

	/* Validate RX event fields */
	cr_assert(rx_cqe.op_context == NULL, "RX CQE Context mismatch");
	cr_assert(rx_cqe.flags == (FI_TAGGED | FI_RECV),
		  "RX CQE flags mismatch");
	cr_assert(rx_cqe.len == send_len, "Invalid RX CQE length");
	cr_assert(rx_cqe.buf == 0, "Invalid RX CQE address");
	cr_assert(rx_cqe.data == 0, "Invalid RX CQE data");
	cr_assert(rx_cqe.tag == 0, "Invalid RX CQE tag");
	cr_assert(from == 0, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	do {
		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == 1);

	/* Validate TX event fields */
	cr_assert(tx_cqe.op_context == NULL, "TX CQE Context mismatch");
	cr_assert(tx_cqe.flags == (FI_TAGGED | FI_SEND),
		  "TX CQE flags mismatch");
	cr_assert(tx_cqe.len == 0, "Invalid TX CQE length");
	cr_assert(tx_cqe.buf == 0, "Invalid TX CQE address");
	cr_assert(tx_cqe.data == 0, "Invalid TX CQE data");
	cr_assert(tx_cqe.tag == 0, "Invalid TX CQE tag");

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			     "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			     i, send_buf[i], recv_buf[i], err++);
		cr_expect_eq(fake_recv_buf[i], 0,
			     "fake data corrupted, element[%d] err=%d\n",
			     i, err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	free(send_buf);
	free(fake_recv_buf);
	free(recv_buf);
}

/* Test unexpected send/recv */
#define RDVS_TAG (46)

struct tagged_thread_args {
	uint8_t *buf;
	size_t len;
	struct fi_cq_tagged_entry *cqe;
	size_t io_num;
	size_t tag_offset;
};

static void *tsend_worker(void *data)
{
	int ret;
	struct tagged_thread_args *args;
	uint64_t tag;

	args = (struct tagged_thread_args *)data;
	tag = RDVS_TAG + args->tag_offset;

	/* Send 64 bytes to FI address 0 (self) */
	ret = fi_tsend(cxit_ep, args->buf, args->len, NULL, 0, tag, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "%s %ld: unexpected ret %d", __func__,
		     args->io_num, ret);

	/* Wait for async event indicating data has been sent */
	do {
		ret = fi_cq_read(cxit_tx_cq, args->cqe, 1);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "%s %ld: unexpected ret %d", __func__,
		     args->io_num, ret);

	pthread_exit(NULL);
}

static void *trecv_worker(void *data)
{
	int ret;
	struct tagged_thread_args *args;
	uint64_t tag;

	args = (struct tagged_thread_args *)data;
	tag = RDVS_TAG + args->tag_offset;

	/* Post RX buffer */
	ret = fi_trecv(cxit_ep, args->buf, args->len, NULL, FI_ADDR_UNSPEC, tag,
		       0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "%s %ld: unexpected ret %d", __func__,
		     args->io_num, ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_read(cxit_rx_cq, args->cqe, 1);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "%s %ld: unexpected ret %d", __func__,
		     args->io_num, ret);

	pthread_exit(NULL);
}

Test(tagged, ux_sw_rdvs, .timeout = 10)
{
	size_t i;
	int ret;
	uint8_t *recv_buf, *send_buf;
	size_t buf_len = 2 * 1024 * 1024;
	int recv_len = 4 * 1024;
	int send_len = 4 * 1024;
	struct fi_cq_tagged_entry rx_cqe, tx_cqe;
	pthread_t threads[2];
	struct tagged_thread_args args[2];
	pthread_attr_t attr;

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	recv_buf = aligned_alloc(C_PAGE_SIZE, buf_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, buf_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, buf_len);
	cr_assert(send_buf);

	for (i = 0; i < buf_len; i++)
		send_buf[i] = i + 0xa0;

	args[0].buf = send_buf;
	args[0].len = send_len;
	args[0].cqe = &tx_cqe;
	args[0].io_num = 0;
	args[0].tag_offset = 0;
	args[1].buf = recv_buf;
	args[1].len = recv_len;
	args[1].cqe = &rx_cqe;
	args[1].io_num = 1;
	args[1].tag_offset = 0;

	/* Give some time for the message to move */
	cr_assert_arr_neq(recv_buf, send_buf, buf_len);

	/* start tsend thread */
	ret = pthread_create(&threads[0], &attr, tsend_worker,
			     (void *)&args[0]);
	cr_assert_eq(ret, 0, "Send thread create failed %d", ret);

	sleep(1);

	/* start trecv thread */
	ret = pthread_create(&threads[1], &attr, trecv_worker,
			     (void *)&args[1]);
	cr_assert_eq(ret, 0, "Recv thread create failed %d", ret);

	/* Wait for the threads to complete */
	ret = pthread_join(threads[0], NULL);
	cr_assert_eq(ret, 0, "Send thread join failed %d", ret);
	ret = pthread_join(threads[1], NULL);
	cr_assert_eq(ret, 0, "Recv thread join failed %d", ret);

	pthread_attr_destroy(&attr);

	/* Validate sent data */
	cr_expect_arr_eq(recv_buf, send_buf, recv_len);

	/* Validate TX event fields */
	cr_assert_null(tx_cqe.op_context, "TX CQE Context mismatch");
	cr_assert_null(tx_cqe.buf, "Invalid TX CQE address %p", tx_cqe.buf);
	cr_assert_eq(tx_cqe.flags, (FI_TAGGED | FI_SEND),
		     "TX CQE flags mismatch %lx", tx_cqe.flags);
	cr_assert_eq(tx_cqe.len, 0, "Invalid TX CQE length %lx", tx_cqe.len);
	cr_assert_eq(tx_cqe.data, 0, "Invalid TX CQE data %lx", tx_cqe.data);
	cr_assert_eq(tx_cqe.tag, 0, "Invalid TX CQE tag %lx",
		     tx_cqe.tag);

	/* Validate RX event fields */
	cr_assert_null(rx_cqe.op_context, "RX CQE Context mismatch");
	cr_assert_null(rx_cqe.buf, "Invalid RX CQE address %p", rx_cqe.buf);
	cr_assert_eq(rx_cqe.flags, (FI_TAGGED | FI_RECV),
		    "RX CQE flags mismatch %lx", rx_cqe.flags);
	cr_assert_eq(rx_cqe.len, recv_len, "Invalid RX CQE length %lx",
		     rx_cqe.len);
	cr_assert_eq(rx_cqe.data, 0, "Invalid RX CQE data %lx", rx_cqe.data);
	cr_assert_eq(rx_cqe.tag, 0, "Invalid RX CQE tag %lx",
		     rx_cqe.tag);

	free(send_buf);
	free(recv_buf);
}

Test(tagged, expected_sw_rdvs, .timeout = 10)
{
	size_t i;
	int ret;
	uint8_t *recv_buf, *send_buf;
	size_t buf_len = 2 * 1024 * 1024;
	int recv_len = 4 * 1024;
	int send_len = 4 * 1024;
	struct fi_cq_tagged_entry rx_cqe, tx_cqe;
	pthread_t threads[2];
	struct tagged_thread_args args[2];
	pthread_attr_t attr;

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	recv_buf = aligned_alloc(C_PAGE_SIZE, buf_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, buf_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, buf_len);
	cr_assert(send_buf);

	for (i = 0; i < buf_len; i++)
		send_buf[i] = i + 0xa0;

	args[0].buf = send_buf;
	args[0].len = send_len;
	args[0].cqe = &tx_cqe;
	args[0].io_num = 0;
	args[0].tag_offset = 0;
	args[1].buf = recv_buf;
	args[1].len = recv_len;
	args[1].cqe = &rx_cqe;
	args[1].io_num = 1;
	args[1].tag_offset = 0;

	/* Give some time for the message to move */
	cr_assert_arr_neq(recv_buf, send_buf, buf_len);

	/* Start trecv thread first so the buffer is ready when the data is sent
	 */
	ret = pthread_create(&threads[1], &attr, trecv_worker,
			     (void *)&args[1]);
	cr_assert_eq(ret, 0, "Recv thread create failed %d", ret);

	sleep(1);

	/* Start tsend thread to send the data into the ready buffer */
	ret = pthread_create(&threads[0], &attr, tsend_worker,
			     (void *)&args[0]);
	cr_assert_eq(ret, 0, "Send thread create failed %d", ret);

	/* Wait for the threads to complete */
	ret = pthread_join(threads[0], NULL);
	cr_assert_eq(ret, 0, "Send thread join failed %d", ret);
	ret = pthread_join(threads[1], NULL);
	cr_assert_eq(ret, 0, "Recv thread join failed %d", ret);

	pthread_attr_destroy(&attr);

	/* Validate sent data */
	cr_expect_arr_eq(recv_buf, send_buf, recv_len);

	/* Validate TX event fields */
	cr_assert_null(tx_cqe.op_context, "TX CQE Context mismatch");
	cr_assert_null(tx_cqe.buf, "Invalid TX CQE address %p", tx_cqe.buf);
	cr_assert_eq(tx_cqe.flags, (FI_TAGGED | FI_SEND),
		     "TX CQE flags mismatch %lx", tx_cqe.flags);
	cr_assert_eq(tx_cqe.len, 0, "Invalid TX CQE length %lx", tx_cqe.len);
	cr_assert_eq(tx_cqe.data, 0, "Invalid TX CQE data %lx", tx_cqe.data);
	cr_assert_eq(tx_cqe.tag, 0, "Invalid TX CQE tag %lx",
		     tx_cqe.tag);

	/* Validate RX event fields */
	cr_assert_null(rx_cqe.op_context, "RX CQE Context mismatch");
	cr_assert_null(rx_cqe.buf, "Invalid RX CQE address %p", rx_cqe.buf);
	cr_assert_eq(rx_cqe.flags, (FI_TAGGED | FI_RECV),
		    "RX CQE flags mismatch %lx", rx_cqe.flags);
	cr_assert_eq(rx_cqe.len, recv_len, "Invalid RX CQE length %lx",
		     rx_cqe.len);
	cr_assert_eq(rx_cqe.data, 0, "Invalid RX CQE data %lx", rx_cqe.data);
	cr_assert_eq(rx_cqe.tag, 0, "Invalid RX CQE tag %lx",
		     rx_cqe.tag);

	free(send_buf);
	free(recv_buf);
}

Test(tagged, rdvs_id, .timeout = 1)
{
	int rc;
	struct cxip_tx_ctx tx_ctx = {};

	/* Allocate all the IDs for the tx_ctx */
	for (int i = 0; i < 128; i++) {
		rc = cxip_tx_ctx_alloc_rdvs_id(&tx_ctx);
		cr_assert_eq(rc, i, "Expected %d Got %d", i, rc);
	}

	/* Allocate one more expecting it to fail */
	rc = cxip_tx_ctx_alloc_rdvs_id(&tx_ctx);
	cr_assert_eq(rc, -FI_ENOSPC, "Got rc %d", rc);

	/* Put ID 67 back */
	rc = cxip_tx_ctx_free_rdvs_id(&tx_ctx, 67);
	cr_assert_eq(rc, FI_SUCCESS, "Got rc %d", rc);

	/* Allocate one more expecting the one just put back */
	rc = cxip_tx_ctx_alloc_rdvs_id(&tx_ctx);
	cr_assert_eq(rc, 67, "Got ID %d instead", rc);

	/* Allocate one more expecting it to fail */
	rc = cxip_tx_ctx_alloc_rdvs_id(&tx_ctx);
	cr_assert_eq(rc, -FI_ENOSPC, "Got rc %d", rc);

	/* Allocate all the IDs for the tx_ctx */
	for (int i = 0; i < 128; i++) {
		rc = cxip_tx_ctx_free_rdvs_id(&tx_ctx, i);
		cr_assert_eq(rc, FI_SUCCESS, "Got rc %d", rc);
	}

	/* Free out of bounds */
	rc = cxip_tx_ctx_free_rdvs_id(&tx_ctx, 325);
	cr_assert_eq(rc, -FI_EINVAL, "Got rc %d", rc);
}

#define NUM_IOS (12)

struct tagged_event_args {
	struct fid_cq *cq;
	struct fi_cq_tagged_entry *cqe;
	size_t io_num;
};

static void *tagged_evt_worker(void *data)
{
	int ret;
	struct tagged_event_args *args;

	args = (struct tagged_event_args *)data;

	for (size_t i = 0; i < args->io_num; i++) {
		/* Wait for async event indicating data has been sent */
		do {
			ret = fi_cq_read(args->cq, &args->cqe[i], 1);
		} while (ret == -FI_EAGAIN);
		cr_assert_eq(ret, 1, "%ld: unexpected ret %d", i,
			     ret);
	}

	pthread_exit(NULL);
}

Test(tagged, multitudes_sw_rdvs, .timeout = 10)
{
	int ret;
	size_t buf_len = 4 * 1024;
	struct fi_cq_tagged_entry rx_cqe[NUM_IOS];
	struct fi_cq_tagged_entry tx_cqe[NUM_IOS];
	struct tagged_thread_args tx_args[NUM_IOS];
	struct tagged_thread_args rx_args[NUM_IOS];
	pthread_t tx_thread;
	pthread_t rx_thread;
	pthread_attr_t attr;
	struct tagged_event_args tx_evt_args = {
		.cq = cxit_tx_cq,
		.cqe = tx_cqe,
		.io_num = NUM_IOS,
	};
	struct tagged_event_args rx_evt_args = {
		.cq = cxit_rx_cq,
		.cqe = rx_cqe,
		.io_num = NUM_IOS,
	};

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	/* Issue the Sends */
	for (size_t tx_io = 0; tx_io < NUM_IOS; tx_io++) {
		tx_args[tx_io].len = buf_len;
		tx_args[tx_io].tag_offset = RDVS_TAG + tx_io;
		tx_args[tx_io].buf = aligned_alloc(C_PAGE_SIZE, buf_len);
		cr_assert_not_null(tx_args[tx_io].buf);
		for (size_t i = 0; i < buf_len; i++)
			tx_args[tx_io].buf[i] = i + 0xa0 + tx_io;

		ret = fi_tsend(cxit_ep, tx_args[tx_io].buf, tx_args[tx_io].len,
			       NULL, 0, tx_args[tx_io].tag_offset, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsend %ld: unexpected ret %d",
			     tx_io, ret);
	}

	/* Start processing Send events */
	ret = pthread_create(&tx_thread, &attr, tagged_evt_worker,
				(void *)&tx_evt_args);
	cr_assert_eq(ret, 0, "Send thread create failed %d", ret);

	sleep(1);

	/* Issue the Receives */
	for (size_t rx_io = 0; rx_io < NUM_IOS; rx_io++) {
		rx_args[rx_io].len = buf_len;
		rx_args[rx_io].tag_offset = RDVS_TAG + rx_io;
		rx_args[rx_io].buf = aligned_alloc(C_PAGE_SIZE, buf_len);
		cr_assert_not_null(rx_args[rx_io].buf);
		memset(rx_args[rx_io].buf, 0, buf_len);

		ret = fi_trecv(cxit_ep, rx_args[rx_io].buf, rx_args[rx_io].len,
			       NULL, FI_ADDR_UNSPEC, rx_args[rx_io].tag_offset,
			       0, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_trecv %ld: unexpected ret %d",
			     rx_io, ret);
	}

	/* Start processing Receive events */
	ret = pthread_create(&rx_thread, &attr, tagged_evt_worker,
			     (void *)&rx_evt_args);
	cr_assert_eq(ret, 0, "Receive thread create failed %d", ret);

	/* Wait for the RX/TX event threads to complete */
	ret = pthread_join(tx_thread, NULL);
	cr_assert_eq(ret, 0, "Send thread join failed %d", ret);

	ret = pthread_join(rx_thread, NULL);
	cr_assert_eq(ret, 0, "Recv thread join failed %d", ret);

	/* Validate results */
	for (size_t io = 0; io < NUM_IOS; io++) {
		/* Validate sent data */
		cr_expect_arr_eq(rx_args[io].buf, tx_args[io].buf, buf_len);

		/* Validate TX event fields */
		cr_expect_null(tx_cqe[io].op_context,
			       "TX CQE %ld Context mismatch", io);
		cr_expect_null(tx_cqe[io].buf, "Invalid TX CQE %ld address %p",
			      io, tx_cqe[io].buf);
		cr_expect_eq(tx_cqe[io].flags, (FI_TAGGED | FI_SEND),
			     "TX CQE %ld flags mismatch %lx", io,
			     tx_cqe[io].flags);
		cr_expect_eq(tx_cqe[io].len, 0, "Invalid TX CQE %ld length %lx",
			     io, buf_len);
		cr_expect_eq(tx_cqe[io].data, 0, "Invalid TX CQE %ld data %lx",
			     io, tx_cqe[io].data);
		cr_expect_eq(tx_cqe[io].tag, 0, "Invalid TX CQE %ld tag %lx",
			     io, tx_cqe[io].tag);

		/* Validate RX event fields */
		cr_expect_null(rx_cqe[io].op_context,
			       "RX CQE %ld Context mismatch", io);
		cr_expect_null(rx_cqe[io].buf, "Invalid RX CQE %ld address %p",
			       io, rx_cqe[io].buf);
		cr_expect_eq(rx_cqe[io].flags, (FI_TAGGED | FI_RECV),
			     "RX CQE %ld flags mismatch %lx", io,
			     rx_cqe[io].flags);
		cr_expect_eq(rx_cqe[io].len, buf_len,
			     "Invalid RX CQE %ld length %lx", io,
			     rx_cqe[io].len);
		cr_expect_eq(rx_cqe[io].data, 0, "Invalid RX CQE %ld data %lx",
			     io, rx_cqe[io].data);
		cr_expect_eq(rx_cqe[io].tag, 0, "Invalid RX CQE %ld tag %lx",
			     io, rx_cqe[io].tag);

		free(tx_args[io].buf);
		free(rx_args[io].buf);
	}

	pthread_attr_destroy(&attr);
}

