/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>

#include "cxip.h"
#include "cxip_test_common.h"

TestSuite(cntr, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

Test(cntr, mod)
{
	int ret;
	int i;
	uint64_t val = 0;
	uint64_t errval = 0;

	cr_assert(!fi_cntr_read(cxit_write_cntr));

	/* Test invalid values */
	ret = fi_cntr_add(cxit_write_cntr, CXIP_CNTR_SUCCESS_MAX + 1);
	cr_assert(ret == -FI_EINVAL);

	ret = fi_cntr_set(cxit_write_cntr, CXIP_CNTR_SUCCESS_MAX + 1);
	cr_assert(ret == -FI_EINVAL);

	ret = fi_cntr_adderr(cxit_write_cntr, CXIP_CNTR_FAILURE_MAX + 1);
	cr_assert(ret == -FI_EINVAL);

	ret = fi_cntr_seterr(cxit_write_cntr, CXIP_CNTR_FAILURE_MAX + 1);
	cr_assert(ret == -FI_EINVAL);

	for (i = 0; i < 10; i++) {
		val += 10;
		ret = fi_cntr_add(cxit_write_cntr, 10);
		cr_assert(ret == FI_SUCCESS);

		while (fi_cntr_read(cxit_write_cntr) != val)
			sched_yield();

		errval += 30;
		ret = fi_cntr_adderr(cxit_write_cntr, 30);
		cr_assert(ret == FI_SUCCESS);

		while (fi_cntr_readerr(cxit_write_cntr) != errval)
			sched_yield();

		val = 5;
		ret = fi_cntr_set(cxit_write_cntr, val);
		cr_assert(ret == FI_SUCCESS);

		while (fi_cntr_read(cxit_write_cntr) != val)
			sched_yield();

		errval = 15;
		ret = fi_cntr_seterr(cxit_write_cntr, errval);
		cr_assert(ret == FI_SUCCESS);

		while (fi_cntr_readerr(cxit_write_cntr) != errval)
			sched_yield();
	}
}

/* Test RMA with counters */
Test(cntr, write)
{
	int ret;
	uint8_t *send_buf;
	int win_len = 0x1000;
	int send_len = 8;
	struct mem_region mem_window;
	int key_val = 0x1f;
	struct fi_cq_tagged_entry cqe;
	int writes = 10;
	int i;

	send_buf = calloc(1, win_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	for (i = 0; i < send_len; i++)
		send_buf[i] = 0xab + i;

	mr_create(win_len, FI_REMOTE_WRITE, 0xa0, key_val, &mem_window);

	cr_assert(!fi_cntr_read(cxit_write_cntr));

	for (i = 0; i < writes; i++) {
		int off = i * send_len;

		ret = fi_inject_write(cxit_ep, send_buf + off, send_len,
				      cxit_ep_fi_addr, off, key_val);
		cr_assert(ret == FI_SUCCESS);
	}

	while (fi_cntr_read(cxit_write_cntr) != writes)
		sched_yield();

	/* Validate sent data */
	for (int i = 0; i < writes * send_len; i++)
		cr_assert_eq(mem_window.mem[i], send_buf[i],
			     "data mismatch, element: (%d) %02x != %02x\n", i,
			     mem_window.mem[i], send_buf[i]);

	/* Make sure no events were delivered */
	ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
	cr_assert(ret == -FI_EAGAIN);

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test all sizes of RMA transactions with counters */
Test(cntr, write_sizes)
{
	int ret;
	uint8_t *send_buf;
	int win_len = 16 * 1024;
	int send_len = 8;
	struct mem_region mem_window;
	int key_val = 0x1f;
	struct fi_cq_tagged_entry cqe;
	int writes = 0;

	send_buf = calloc(1, win_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	mr_create(win_len, FI_REMOTE_WRITE, 0xa0, key_val, &mem_window);

	cr_assert(!fi_cntr_read(cxit_write_cntr));

	for (send_len = 1; send_len <= win_len; send_len <<= 1) {
		ret = fi_write(cxit_ep, send_buf, send_len, NULL,
			       cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert(ret == FI_SUCCESS);

		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

		validate_tx_event(&cqe, FI_RMA | FI_WRITE, NULL);
		writes++;

		/* Validate sent data */
		for (int i = 0; i < send_len; i++)
			cr_assert_eq(mem_window.mem[i], send_buf[i],
				     "data mismatch, element: (%d) %02x != %02x\n", i,
				     mem_window.mem[i], send_buf[i]);
	}

	while (fi_cntr_read(cxit_write_cntr) != writes)
		sched_yield();

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test fi_read with counters */
Test(cntr, read)
{
	int ret;
	uint8_t *local;
	int remote_len = 0x1000;
	int local_len = 8;
	int key_val = 0xa;
	struct fi_cq_tagged_entry cqe;
	struct mem_region remote;

	local = calloc(1, local_len);
	cr_assert_not_null(local, "local alloc failed");

	mr_create(remote_len, FI_REMOTE_READ, 0xc0, key_val, &remote);

	cr_assert(!fi_cntr_read(cxit_read_cntr));

	/* Get 8 bytes from the source buffer to the receive buffer */
	ret = fi_read(cxit_ep, local, local_len, NULL, cxit_ep_fi_addr, 0,
		      key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_read() failed (%d)", ret);

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read() failed (%d)", ret);

	validate_tx_event(&cqe, FI_RMA | FI_READ, NULL);

	/* Validate sent data */
	for (int i = 0; i < local_len; i++)
		cr_expect_eq(local[i], remote.mem[i],
			     "data mismatch, element: (%d) %02x != %02x\n", i,
			     local[i], remote.mem[i]);

	while (fi_cntr_read(cxit_read_cntr) != 1)
		sched_yield();

	mr_destroy(&remote);
	free(local);
}

/* Test send/recv counters */
Test(cntr, ping)
{
	int i, ret;
	uint8_t *recv_buf,
		*send_buf;
	int recv_len = 64;
	int send_len = 64;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
	fi_addr_t from;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	cr_assert(!fi_cntr_read(cxit_send_cntr));
	cr_assert(!fi_cntr_read(cxit_recv_cntr));

	/* Post RX buffer */
	ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL, FI_ADDR_UNSPEC, 0,
		       0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);

	/* Send 64 bytes to self */
	ret = fi_tsend(cxit_ep, send_buf, send_len, NULL, cxit_ep_fi_addr, 0,
		       NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	while (fi_cntr_read(cxit_send_cntr) != 1)
		sched_yield();

	while (fi_cntr_read(cxit_recv_cntr) != 1)
		sched_yield();

	free(send_buf);
	free(recv_buf);
}
