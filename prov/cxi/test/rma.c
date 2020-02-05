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

#define RMA_WIN_KEY 0x1f

TestSuite(rma, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test fi_write simple case. Test IDC sizes to multi-packe sizes. */
Test(rma, simple_write)
{
	int ret;
	uint8_t *send_buf;
	int win_len = 16 * 1024;
	int send_len = 8;
	struct mem_region mem_window;
	int key_val = RMA_WIN_KEY;
	struct fi_cq_tagged_entry cqe;

	send_buf = calloc(1, win_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	mr_create(win_len, FI_REMOTE_WRITE, 0xa0, key_val, &mem_window);

	for (send_len = 1; send_len <= win_len; send_len <<= 1) {
		ret = fi_write(cxit_ep, send_buf, send_len, NULL,
			       cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert(ret == FI_SUCCESS);

		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

		validate_tx_event(&cqe, FI_RMA | FI_WRITE, NULL);

		/* Validate sent data */
		for (int i = 0; i < send_len; i++)
			cr_assert_eq(mem_window.mem[i], send_buf[i],
				     "data mismatch, element: (%d) %02x != %02x\n", i,
				     mem_window.mem[i], send_buf[i]);
	}

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test simple writes to a standard MR. */
Test(rma, simple_write_std_mr)
{
	int ret;
	uint8_t *send_buf;
	int win_len = 16 * 1024;
	int send_len = 8;
	struct mem_region mem_window;
	uint64_t key_val = 0xabcdef;
	struct fi_cq_tagged_entry cqe;

	send_buf = calloc(1, win_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	mr_create(win_len, FI_REMOTE_WRITE, 0xa0, key_val, &mem_window);

	for (send_len = 1; send_len <= win_len; send_len <<= 1) {
		ret = fi_write(cxit_ep, send_buf, send_len, NULL,
			       cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert(ret == FI_SUCCESS);

		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

		validate_tx_event(&cqe, FI_RMA | FI_WRITE, NULL);

		/* Validate sent data */
		for (int i = 0; i < send_len; i++)
			cr_assert_eq(mem_window.mem[i], send_buf[i],
				     "data mismatch, element: (%d) %02x != %02x\n", i,
				     mem_window.mem[i], send_buf[i]);
	}

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test fi_writev simple case */
Test(rma, simple_writev)
{
	int ret;
	uint8_t *send_buf;
	int win_len = 0x1000;
	int send_len = 8;
	struct mem_region mem_window;
	int key_val = RMA_WIN_KEY;
	struct fi_cq_tagged_entry cqe;
	struct iovec iov[1];

	send_buf = calloc(1, win_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	mr_create(win_len, FI_REMOTE_WRITE, 0x44, key_val, &mem_window);

	iov[0].iov_base = send_buf;
	iov[0].iov_len = send_len;

	/* Send 8 bytes from send buffer data to RMA window 0 */
	ret = fi_writev(cxit_ep, iov, NULL, 1, cxit_ep_fi_addr, 0, key_val,
			NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writev failed %d", ret);

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

	validate_tx_event(&cqe, FI_RMA | FI_WRITE, NULL);

	/* Validate sent data */
	for (int i = 0; i < send_len; i++)
		cr_assert_eq(mem_window.mem[i], send_buf[i],
			     "data mismatch, element: (%d) %02x != %02x\n", i,
			     mem_window.mem[i], send_buf[i]);

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test fi_writemsg simple case */
Test(rma, simple_writemsg)
{
	int ret;
	uint8_t *send_buf;
	int win_len = 0x1000;
	int send_len = 8;
	struct mem_region mem_window;
	int key_val = RMA_WIN_KEY;
	struct fi_cq_tagged_entry cqe;
	struct fi_msg_rma msg = {};
	struct iovec iov[1];
	struct fi_rma_iov rma[1];
	uint64_t flags = 0;

	send_buf = calloc(1, win_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	mr_create(win_len, FI_REMOTE_WRITE, 0x44, key_val, &mem_window);

	iov[0].iov_base = send_buf;
	iov[0].iov_len = send_len;

	rma[0].addr = 0;
	rma[0].len = send_len;
	rma[0].key = key_val;

	msg.msg_iov = iov;
	msg.iov_count = 1;
	msg.rma_iov = rma;
	msg.rma_iov_count = 1;
	msg.addr = cxit_ep_fi_addr;

	/* Send 8 bytes from send buffer data to RMA window 0 at FI address 0
	 * (self)
	 */
	ret = fi_writemsg(cxit_ep, &msg, flags);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writemsg failed %d", ret);

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

	validate_tx_event(&cqe, FI_RMA | FI_WRITE, NULL);

	/* Validate sent data */
	for (int i = 0; i < send_len; i++)
		cr_assert_eq(mem_window.mem[i], send_buf[i],
			     "data mismatch, element: (%d) %02x != %02x\n", i,
			     mem_window.mem[i], send_buf[i]);

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Perform a write that uses a flushing ZBR at the target. Validate flush with
 * pycxi:
 *
 *    $ pycxi/utils/csrutil dump csr ixe_cntr | grep dmawr_flush_reqs
 */
Test(rma, flush)
{
	int ret;
	uint8_t *send_buf;
	int win_len = 0x1000;
	int send_len = 8;
	struct mem_region mem_window;
	int key_val = RMA_WIN_KEY;
	struct fi_cq_tagged_entry cqe;
	struct fi_msg_rma msg = {};
	struct iovec iov[1];
	struct fi_rma_iov rma[1];
	uint64_t flags = FI_DELIVERY_COMPLETE;

	send_buf = calloc(1, win_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	mr_create(win_len, FI_REMOTE_WRITE, 0x44, key_val, &mem_window);

	iov[0].iov_base = send_buf;
	iov[0].iov_len = send_len;

	rma[0].addr = 0;
	rma[0].len = send_len;
	rma[0].key = key_val;

	msg.msg_iov = iov;
	msg.iov_count = 1;
	msg.rma_iov = rma;
	msg.rma_iov_count = 1;
	msg.addr = cxit_ep_fi_addr;

	/* Send 8 bytes from send buffer data to RMA window 0 at FI address 0
	 * (self)
	 */
	ret = fi_writemsg(cxit_ep, &msg, flags);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writemsg failed %d", ret);

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

	validate_tx_event(&cqe, FI_RMA | FI_WRITE, NULL);

	/* Validate sent data */
	for (int i = 0; i < send_len; i++)
		cr_assert_eq(mem_window.mem[i], send_buf[i],
			     "data mismatch, element: (%d) %02x != %02x\n", i,
			     mem_window.mem[i], send_buf[i]);

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test fi_writemsg with FI_INJECT flag */
Test(rma, simple_writemsg_inject)
{
	int ret;
	uint8_t *send_buf;
	int win_len = 0x1000;
	int send_len = 8;
	struct mem_region mem_window;
	int key_val = RMA_WIN_KEY;
	struct fi_cq_tagged_entry cqe;
	struct fi_msg_rma msg = {};
	struct iovec iov[1];
	struct fi_rma_iov rma[1];
	uint64_t flags = FI_INJECT;

	send_buf = calloc(1, win_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	mr_create(win_len, FI_REMOTE_WRITE, 0x44, key_val, &mem_window);

	iov[0].iov_base = send_buf;
	iov[0].iov_len = send_len;

	rma[0].addr = 0;
	rma[0].len = send_len;
	rma[0].key = key_val;

	msg.msg_iov = iov;
	msg.iov_count = 1;
	msg.rma_iov = rma;
	msg.rma_iov_count = 1;
	msg.addr = cxit_ep_fi_addr;

	/* Send 8 bytes from send buffer data to RMA window 0 at FI address 0
	 * (self)
	 */
	ret = fi_writemsg(cxit_ep, &msg, flags);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writemsg failed %d", ret);

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

	validate_tx_event(&cqe, FI_RMA | FI_WRITE, NULL);

	/* Validate sent data */
	for (int i = 0; i < send_len; i++)
		cr_assert_eq(mem_window.mem[i], send_buf[i],
			     "data mismatch, element: (%d) %02x != %02x\n", i,
			     mem_window.mem[i], send_buf[i]);

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test fi_inject_write simple case */
Test(rma, simple_inject_write)
{
	int ret;
	uint8_t *send_buf;
	int win_len = 0x1000;
	int send_len = 8;
	struct mem_region mem_window;
	int key_val = RMA_WIN_KEY;
	struct fi_cq_tagged_entry cqe;

	send_buf = calloc(1, win_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	mr_create(win_len, FI_REMOTE_WRITE, 0xa0, key_val, &mem_window);

	cr_assert(!fi_cntr_read(cxit_write_cntr));

	/* Test invalid inject length */
	ret = fi_inject_write(cxit_ep, send_buf,
			      cxit_fi->tx_attr->inject_size + 100,
			      cxit_ep_fi_addr, 0, key_val);
	cr_assert(ret == -FI_EMSGSIZE);

	/* Send 8 bytes from send buffer data to RMA window 0 */
	ret = fi_inject_write(cxit_ep, send_buf, send_len, cxit_ep_fi_addr, 0,
			      key_val);
	cr_assert(ret == FI_SUCCESS);

	while (fi_cntr_read(cxit_write_cntr) != 1)
		sched_yield();

	/* Validate sent data */
	for (int i = 0; i < send_len; i++)
		cr_assert_eq(mem_window.mem[i], send_buf[i],
			     "data mismatch, element: (%d) %02x != %02x\n", i,
			     mem_window.mem[i], send_buf[i]);

	/* Make sure an event wasn't delivered */
	ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
	cr_assert(ret == -FI_EAGAIN);

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test fi_read simple case */
Test(rma, simple_read)
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

	while (fi_cntr_read(cxit_read_cntr) != 1)
		sched_yield();

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read() failed (%d)", ret);

	validate_tx_event(&cqe, FI_RMA | FI_READ, NULL);

	/* Validate sent data */
	for (int i = 0; i < local_len; i++)
		cr_expect_eq(local[i], remote.mem[i],
			     "data mismatch, element: (%d) %02x != %02x\n", i,
			     local[i], remote.mem[i]);

	mr_destroy(&remote);
	free(local);
}

/* Test fi_readv simple case */
Test(rma, simple_readv)
{
	int ret;
	uint8_t *local;
	int remote_len = 0x1000;
	int local_len = 8;
	int key_val = 0x2a;
	struct fi_cq_tagged_entry cqe;
	struct mem_region remote;
	struct iovec iov[1];

	local = calloc(1, local_len);
	cr_assert_not_null(local, "local alloc failed");

	mr_create(remote_len, FI_REMOTE_READ, 0x3c, key_val, &remote);

	iov[0].iov_base = local;
	iov[0].iov_len = local_len;

	/* Get 8 bytes from the source buffer to the receive buffer */
	ret = fi_readv(cxit_ep, iov, NULL, 1, cxit_ep_fi_addr, 0, key_val,
		       NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_readv() failed (%d)", ret);

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read() failed (%d)", ret);

	validate_tx_event(&cqe, FI_RMA | FI_READ, NULL);

	/* Validate sent data */
	for (int i = 0; i < local_len; i++)
		cr_expect_eq(local[i], remote.mem[i],
			     "data mismatch, element: (%d) %02x != %02x\n", i,
			     local[i], remote.mem[i]);

	mr_destroy(&remote);
	free(local);
}

/* Test fi_readmsg simple case */
Test(rma, simple_readmsg)
{
	int ret;
	uint8_t *local;
	int remote_len = 0x1000;
	int local_len = 8;
	int key_val = 0x2a;
	struct fi_cq_tagged_entry cqe;
	struct mem_region remote;
	struct fi_msg_rma msg = {};
	struct iovec iov[1];
	struct fi_rma_iov rma[1];
	uint64_t flags = 0;

	local = calloc(1, local_len);
	cr_assert_not_null(local, "local alloc failed");

	mr_create(remote_len, FI_REMOTE_READ, 0xd9, key_val, &remote);

	iov[0].iov_base = local;
	iov[0].iov_len = local_len;

	rma[0].addr = 0;
	rma[0].len = local_len;
	rma[0].key = key_val;

	msg.msg_iov = iov;
	msg.iov_count = 1;
	msg.rma_iov = rma;
	msg.rma_iov_count = 1;
	msg.addr = cxit_ep_fi_addr;

	/* Get 8 bytes from the source buffer to the receive buffer */
	ret = fi_readmsg(cxit_ep, &msg, flags);
	cr_assert_eq(ret, FI_SUCCESS, "fi_readv() failed (%d)", ret);

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read() failed (%d)", ret);

	validate_tx_event(&cqe, FI_RMA | FI_READ, NULL);

	/* Validate sent data */
	for (int i = 0; i < local_len; i++)
		cr_expect_eq(local[i], remote.mem[i],
			     "data mismatch, element: (%d) %02x != %02x\n", i,
			     local[i], remote.mem[i]);

	mr_destroy(&remote);
	free(local);
}

/* Test fi_readmsg failure cases */
Test(rma, readmsg_failures)
{
	int ret;
	struct iovec iov[1];
	struct fi_rma_iov rma[1];
	struct fi_msg_rma msg = {
		.msg_iov = iov,
		.rma_iov = rma,
		.iov_count = 1,
		.rma_iov_count = 1,
	};
	uint64_t flags = 0;

	/* Invalid msg value */
	ret = fi_readmsg(cxit_ep, NULL, flags);
	cr_assert_eq(ret, -FI_EINVAL, "NULL msg return %d", ret);

	msg.iov_count = cxit_fi->tx_attr->rma_iov_limit + 1;
	ret = fi_readmsg(cxit_ep, &msg, flags);
	cr_assert_eq(ret, -FI_EINVAL, "Invalid iov_count return %d", ret);

	msg.iov_count = cxit_fi->tx_attr->rma_iov_limit;
	flags = FI_DIRECTED_RECV; /* Invalid flag value */
	ret = fi_readmsg(cxit_ep, &msg, flags);
	cr_assert_eq(ret, -FI_EBADFLAGS, "Invalid flag unexpected return %d",
		     ret);
}

/* Test fi_writemsg failure cases */
Test(rma, writemsg_failures)
{
	int ret;
	struct iovec iov[1];
	struct fi_rma_iov rma[1];
	struct fi_msg_rma msg = {
		.msg_iov = iov,
		.rma_iov = rma,
		.iov_count = 1,
		.rma_iov_count = 1,
	};
	uint64_t flags = 0;
	size_t send_len = 10;
	char send_buf[send_len];

	/* Invalid msg value */
	ret = fi_writemsg(cxit_ep, NULL, flags);
	cr_assert_eq(ret, -FI_EINVAL, "NULL msg return %d", ret);

	msg.iov_count = cxit_fi->tx_attr->rma_iov_limit + 1;
	ret = fi_writemsg(cxit_ep, &msg, flags);
	cr_assert_eq(ret, -FI_EINVAL, "Invalid iov_count return %d", ret);

	msg.iov_count = cxit_fi->tx_attr->rma_iov_limit;
	flags = FI_DIRECTED_RECV; /* Invalid flag value */
	ret = fi_writemsg(cxit_ep, &msg, flags);
	cr_assert_eq(ret, -FI_EBADFLAGS, "Invalid flag return %d", ret);

	/* Invalid length */
	iov[0].iov_base = send_buf;
	iov[0].iov_len = 1024*1024*1024+1;

	rma[0].addr = 0;
	rma[0].len = send_len;
	rma[0].key = 0xa;
	msg.msg_iov = iov;
	msg.iov_count = 1;
	msg.rma_iov = rma;
	msg.rma_iov_count = 1;

	ret = fi_writemsg(cxit_ep, &msg, 0);
	cr_assert_eq(ret, -FI_EMSGSIZE, "Invalid flag return %d", ret);

	/* Invalid inject length */
	iov[0].iov_len = C_MAX_IDC_PAYLOAD_RES+1;

	ret = fi_writemsg(cxit_ep, &msg, FI_INJECT);
	cr_assert_eq(ret, -FI_EMSGSIZE, "Invalid flag return %d", ret);
}

/* Test fi_readv failure cases */
Test(rma, readv_failures)
{
	int ret;
	struct iovec iov = {};

	 /* Invalid count value */
	ret = fi_readv(cxit_ep, &iov, NULL,
		       cxit_fi->tx_attr->rma_iov_limit + 1,
		       cxit_ep_fi_addr, 0, 0, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "Invalid count return %d", ret);
}

/* Test fi_writev failure cases */
Test(rma, writev_failures)
{
	int ret;
	struct iovec iov = {};

	 /* Invalid count value */
	ret = fi_writev(cxit_ep, &iov, NULL,
			cxit_fi->tx_attr->rma_iov_limit + 1,
			cxit_ep_fi_addr, 0, 0, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "Invalid count return %d", ret);
}

/* Perform an RMA write spanning a page */
Test(rma, write_spanning_page)
{
	int ret;
	uint8_t *send_buf;
	uint8_t *send_addr;
	int win_len = 0x2000;
	int send_len = 8;
	struct mem_region mem_window;
	int key_val = RMA_WIN_KEY;
	struct fi_cq_tagged_entry cqe;

	send_buf = calloc(1, win_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	send_addr = (uint8_t *)FLOOR(send_buf + C_PAGE_SIZE, C_PAGE_SIZE) - 4;
	memset(send_addr, 0xcc, send_len);

	mr_create(win_len, FI_REMOTE_WRITE, 0xa0, key_val, &mem_window);
	memset(mem_window.mem, 0x33, win_len);

	/* Send 8 bytes from send buffer data to RMA window 0 */
	ret = fi_write(cxit_ep, send_addr, send_len, NULL, cxit_ep_fi_addr, 0,
		       key_val, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

	validate_tx_event(&cqe, FI_RMA | FI_WRITE, NULL);

	/* Validate sent data */
	for (int i = 0; i < send_len; i++)
		cr_assert_eq(mem_window.mem[i], send_addr[i],
			     "data mismatch, element: (%d) %02x != %02x\n", i,
			     mem_window.mem[i], send_addr[i]);

	mr_destroy(&mem_window);
	free(send_buf);
}

Test(rma, rma_cleanup)
{
	int ret;
	long i;
	uint8_t *send_buf;
	int win_len = 0x1000;
	int send_len = 8;
	struct mem_region mem_window;
	int key_val = RMA_WIN_KEY;
	int writes = 50;

	send_buf = calloc(1, win_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	for (i = 0; i < win_len; i++)
		send_buf[i] = 0xb1 * i;

	mr_create(win_len, FI_REMOTE_WRITE, 0xa0, key_val, &mem_window);

	/* Send 8 bytes from send buffer data to RMA window 0 */
	for (i = 0; i < writes; i++) {
		ret = fi_write(cxit_ep, send_buf, send_len, NULL,
				cxit_ep_fi_addr, 0, key_val, (void *)i);
		cr_assert(ret == FI_SUCCESS);
	}

	mr_destroy(&mem_window);

	/* Exit without gathering events. */
}

void cxit_setup_rma_selective_completion(void)
{
	cxit_tx_cq_bind_flags |= FI_SELECTIVE_COMPLETION;

	cxit_setup_getinfo();
	cxit_fi_hints->tx_attr->op_flags = FI_COMPLETION;
	cxit_setup_rma();
}

/* Test selective completion behavior with RMA. */
Test(rma_sel, selective_completion,
     .init = cxit_setup_rma_selective_completion,
     .fini = cxit_teardown_rma)
{
	int ret;
	uint8_t *loc_buf;
	int win_len = 0x1000;
	int loc_len = 8;
	struct mem_region mem_window;
	int key_val = RMA_WIN_KEY;
	struct fi_cq_tagged_entry cqe;
	struct fi_msg_rma msg = {};
	struct iovec iov;
	struct fi_rma_iov rma;
	int count = 0;

	loc_buf = calloc(1, win_len);
	cr_assert_not_null(loc_buf, "loc_buf alloc failed");

	mr_create(win_len, FI_REMOTE_WRITE | FI_REMOTE_READ, 0xa0, key_val,
		  &mem_window);

	iov.iov_base = loc_buf;
	iov.iov_len = loc_len;

	rma.addr = 0;
	rma.key = key_val;

	msg.msg_iov = &iov;
	msg.iov_count = 1;
	msg.rma_iov = &rma;
	msg.rma_iov_count = 1;
	msg.addr = cxit_ep_fi_addr;

	/* Puts */

	/* Completion requested by default. */
	for (loc_len = 1; loc_len <= win_len; loc_len <<= 1) {
		ret = fi_write(cxit_ep, loc_buf, loc_len, NULL,
			       cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert(ret == FI_SUCCESS);
		count++;

		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

		validate_tx_event(&cqe, FI_RMA | FI_WRITE, NULL);

		/* Validate sent data */
		for (int i = 0; i < loc_len; i++)
			cr_assert_eq(mem_window.mem[i], loc_buf[i],
				     "data mismatch, element: (%d) %02x != %02x\n", i,
				     mem_window.mem[i], loc_buf[i]);
	}

	/* Completion explicitly requested. */
	for (loc_len = 1; loc_len <= win_len; loc_len <<= 1) {
		iov.iov_len = loc_len;
		ret = fi_writemsg(cxit_ep, &msg, FI_COMPLETION);
		cr_assert(ret == FI_SUCCESS);
		count++;

		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

		validate_tx_event(&cqe, FI_RMA | FI_WRITE, NULL);

		/* Validate sent data */
		for (int i = 0; i < loc_len; i++)
			cr_assert_eq(mem_window.mem[i], loc_buf[i],
				     "data mismatch, element: (%d) %02x != %02x\n", i,
				     mem_window.mem[i], loc_buf[i]);
	}

	/* Suppress completion. */
	for (loc_len = 1; loc_len <= win_len; loc_len <<= 1) {
		iov.iov_len = loc_len;
		ret = fi_writemsg(cxit_ep, &msg, 0);
		cr_assert(ret == FI_SUCCESS);
		count++;

		while (fi_cntr_read(cxit_write_cntr) != count)
			sched_yield();

		/* Validate sent data */
		for (int i = 0; i < loc_len; i++)
			while (mem_window.mem[i] != loc_buf[i])
				sched_yield();

		/* Ensure no events were generated */
		ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
		cr_assert(ret == -FI_EAGAIN);
	}

	/* Inject never generates an event */
	loc_len = 8;
	ret = fi_inject_write(cxit_ep, loc_buf, loc_len, cxit_ep_fi_addr, 0,
			      key_val);
	cr_assert(ret == FI_SUCCESS);

	/* Validate sent data */
	for (int i = 0; i < loc_len; i++)
		while (mem_window.mem[i] != loc_buf[i])
			sched_yield();

	/* Make sure an event wasn't delivered */
	ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
	cr_assert(ret == -FI_EAGAIN);

	/* Gets */
	memset(loc_buf, 0, win_len);
	count = 0;

	/* Completion requested by default. */
	for (loc_len = 1; loc_len <= win_len; loc_len <<= 1) {
		memset(loc_buf, 0, loc_len);
		ret = fi_read(cxit_ep, loc_buf, loc_len, NULL,
			      cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert(ret == FI_SUCCESS);
		count++;

		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

		validate_tx_event(&cqe, FI_RMA | FI_READ, NULL);

		/* Validate sent data */
		for (int i = 0; i < loc_len; i++)
			cr_assert_eq(mem_window.mem[i], loc_buf[i],
				     "data mismatch, element: (%d) %02x != %02x\n", i,
				     mem_window.mem[i], loc_buf[i]);
	}

	/* Completion explicitly requested. */
	for (loc_len = 1; loc_len <= win_len; loc_len <<= 1) {
		memset(loc_buf, 0, loc_len);
		iov.iov_len = loc_len;
		ret = fi_readmsg(cxit_ep, &msg, FI_COMPLETION);
		cr_assert(ret == FI_SUCCESS);
		count++;

		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

		validate_tx_event(&cqe, FI_RMA | FI_READ, NULL);

		/* Validate sent data */
		for (int i = 0; i < loc_len; i++)
			cr_assert_eq(mem_window.mem[i], loc_buf[i],
				     "data mismatch, element: (%d) %02x != %02x\n", i,
				     mem_window.mem[i], loc_buf[i]);
	}

	/* Suppress completion. */
	for (loc_len = 1; loc_len <= win_len; loc_len <<= 1) {
		memset(loc_buf, 0, loc_len);
		iov.iov_len = loc_len;
		ret = fi_readmsg(cxit_ep, &msg, 0);
		cr_assert(ret == FI_SUCCESS);
		count++;

		while (fi_cntr_read(cxit_read_cntr) != count)
			sched_yield();

		/* Validate sent data */
		for (int i = 0; i < loc_len; i++)
			cr_assert_eq(mem_window.mem[i], loc_buf[i],
				     "data mismatch, element: (%d) %02x != %02x\n", i,
				     mem_window.mem[i], loc_buf[i]);

		/* Ensure no events were generated */
		ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
		cr_assert(ret == -FI_EAGAIN);
	}

	mr_destroy(&mem_window);
	free(loc_buf);
}

void cxit_setup_rma_selective_completion_suppress(void)
{
	cxit_tx_cq_bind_flags |= FI_SELECTIVE_COMPLETION;

	cxit_setup_getinfo();
	cxit_fi_hints->tx_attr->op_flags = 0;
	cxit_setup_rma();
}

/* Test selective completion behavior with RMA. */
Test(rma_sel, selective_completion_suppress,
     .init = cxit_setup_rma_selective_completion_suppress,
     .fini = cxit_teardown_rma)
{
	int ret;
	uint8_t *send_buf;
	int win_len = 0x1000;
	int send_len = 8;
	struct mem_region mem_window;
	int key_val = RMA_WIN_KEY;
	struct fi_cq_tagged_entry cqe;
	struct fi_msg_rma msg = {};
	struct iovec iov;
	struct fi_rma_iov rma;
	int write_count = 0;

	send_buf = calloc(1, win_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	mr_create(win_len, FI_REMOTE_WRITE, 0xa0, key_val, &mem_window);

	iov.iov_base = send_buf;
	iov.iov_len = send_len;

	rma.addr = 0;
	rma.key = key_val;

	msg.msg_iov = &iov;
	msg.iov_count = 1;
	msg.rma_iov = &rma;
	msg.rma_iov_count = 1;
	msg.addr = cxit_ep_fi_addr;

	/* Normal writes do not generate completions */
	for (send_len = 1; send_len <= win_len; send_len <<= 1) {
		memset(mem_window.mem, 0, send_len);
		ret = fi_write(cxit_ep, send_buf, send_len, NULL,
			       cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert(ret == FI_SUCCESS);
		write_count++;

		while (fi_cntr_read(cxit_write_cntr) != write_count)
			sched_yield();

		/* Validate sent data */
		for (int i = 0; i < send_len; i++)
			while (mem_window.mem[i] != send_buf[i])
				sched_yield();

		/* Ensure no events were generated */
		ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
		cr_assert(ret == -FI_EAGAIN);
	}

	/* Request completions from fi_writemsg */
	for (send_len = 1; send_len <= win_len; send_len <<= 1) {
		memset(mem_window.mem, 0, send_len);
		iov.iov_len = send_len;
		ret = fi_writemsg(cxit_ep, &msg, FI_COMPLETION);
		cr_assert(ret == FI_SUCCESS);
		write_count++;

		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

		validate_tx_event(&cqe, FI_RMA | FI_WRITE, NULL);

		/* Validate sent data */
		for (int i = 0; i < send_len; i++)
			cr_assert_eq(mem_window.mem[i], send_buf[i],
				     "data mismatch, element: (%d) %02x != %02x\n", i,
				     mem_window.mem[i], send_buf[i]);
	}

	/* Suppress completions using fi_writemsg */
	for (send_len = 1; send_len <= win_len; send_len <<= 1) {
		memset(mem_window.mem, 0, send_len);
		iov.iov_len = send_len;
		ret = fi_writemsg(cxit_ep, &msg, 0);
		cr_assert(ret == FI_SUCCESS);
		write_count++;

		while (fi_cntr_read(cxit_write_cntr) != write_count)
			sched_yield();

		/* Validate sent data */
		for (int i = 0; i < send_len; i++)
			while (mem_window.mem[i] != send_buf[i])
				sched_yield();

		/* Ensure no events were generated */
		ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
		cr_assert(ret == -FI_EAGAIN);
	}

	/* Inject never generates an event */
	send_len = 8;
	memset(mem_window.mem, 0, send_len);
	ret = fi_inject_write(cxit_ep, send_buf, send_len, cxit_ep_fi_addr, 0,
			      key_val);
	cr_assert(ret == FI_SUCCESS);
	write_count++;

	while (fi_cntr_read(cxit_write_cntr) != write_count)
		sched_yield();

	/* Validate sent data */
	for (int i = 0; i < send_len; i++)
		while (mem_window.mem[i] != send_buf[i])
			sched_yield();

	/* Make sure an event wasn't delivered */
	ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
	cr_assert(ret == -FI_EAGAIN);

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test remote counter events with RMA */
Test(rma, rem_cntr)
{
	int ret;
	uint8_t *send_buf;
	int win_len = 16 * 1024;
	int send_len = 8;
	struct mem_region mem_window;
	int key_val = RMA_WIN_KEY;
	struct fi_cq_tagged_entry cqe;
	int count = 0;

	send_buf = calloc(1, win_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	mr_create(win_len, FI_REMOTE_WRITE, 0xa0, key_val, &mem_window);

	for (send_len = 1; send_len <= win_len; send_len <<= 1) {
		ret = fi_write(cxit_ep, send_buf, send_len, NULL,
			       cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert(ret == FI_SUCCESS);

		/* Wait for remote counter event, then check data */
		count++;
		while (fi_cntr_read(cxit_rem_cntr) != count)
			sched_yield();

		/* Validate sent data */
		for (int i = 0; i < send_len; i++)
			cr_assert_eq(mem_window.mem[i], send_buf[i],
				     "data mismatch, element: (%d) %02x != %02x\n", i,
				     mem_window.mem[i], send_buf[i]);

		/* Gather source completion after data */
		ret = cxit_await_completion(cxit_tx_cq, &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

		validate_tx_event(&cqe, FI_RMA | FI_WRITE, NULL);
	}

	mr_destroy(&mem_window);
	free(send_buf);
}
