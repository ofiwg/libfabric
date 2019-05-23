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

struct mem_region {
	uint8_t *mem;
	struct fid_mr *mr;
};

static void mr_create(size_t len, uint64_t access, uint8_t seed, uint64_t key,
		      struct mem_region *mr)
{
	int ret;

	cr_assert_not_null(mr);

	mr->mem = calloc(1, len);
	cr_assert_not_null(mr->mem, "Error allocating memory window");

	for (size_t i = 0; i < len; i++)
		mr->mem[i] = i + seed;

	ret = fi_mr_reg(cxit_domain, mr->mem, len, access, 0, key, 0, &mr->mr,
			NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_reg failed %d", ret);

	ret = fi_mr_bind(mr->mr, &cxit_ep->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind failed %d", ret);

	ret = fi_mr_enable(mr->mr);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_enable failed %d", ret);
}

static void mr_destroy(struct mem_region *mr)
{
	fi_close(&mr->mr->fid);
	free(mr->mem);
}

TestSuite(rma, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test fi_write simple case */
Test(rma, simple_write)
{
	int ret;
	uint8_t *send_buf;
	int win_len = 0x1000;
	int send_len = 8;
	struct mem_region mem_window;
	int key_val = 0x1f;
	struct fi_cq_tagged_entry cqe;

	send_buf = calloc(1, win_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	mr_create(win_len, FI_REMOTE_WRITE, 0xa0, key_val, &mem_window);

	/* Send 8 bytes from send buffer data to RMA window 0 */
	ret = fi_write(cxit_ep, send_buf, send_len, NULL, cxit_ep_fi_addr, 0,
		       key_val, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

	/* Validate event fields */
	cr_assert(cqe.op_context == NULL, "CQE Context mismatch");
	cr_assert(cqe.flags == (FI_RMA | FI_WRITE), "CQE flags mismatch (%lx)",
		  cqe.flags);
	cr_assert(cqe.len == 0, "Invalid CQE length");
	cr_assert(cqe.buf == 0, "Invalid CQE address");
	cr_assert(cqe.data == 0, "Invalid CQE data");
	cr_assert(cqe.tag == 0, "Invalid CQE tag");

	/* Validate sent data */
	for (int i = 0; i < send_len; i++)
		cr_assert_eq(mem_window.mem[i], send_buf[i],
			     "data mismatch, element: (%d) %02x != %02x\n", i,
			     mem_window.mem[i], send_buf[i]);

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
	int key_val = 0x1f;
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

	/* Validate event fields */
	cr_assert(cqe.op_context == NULL, "CQE Context mismatch");
	cr_assert(cqe.flags == (FI_RMA | FI_WRITE), "CQE flags mismatch (%lx)",
		  cqe.flags);
	cr_assert(cqe.len == 0, "Invalid CQE length");
	cr_assert(cqe.buf == 0, "Invalid CQE address");
	cr_assert(cqe.data == 0, "Invalid CQE data");
	cr_assert(cqe.tag == 0, "Invalid CQE tag");

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
	int key_val = 0x1f;
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

	/* Validate event fields */
	cr_assert(cqe.op_context == NULL, "CQE Context mismatch");
	cr_assert(cqe.flags == (FI_RMA | FI_WRITE), "CQE flags mismatch (%lx)",
		  cqe.flags);
	cr_assert(cqe.len == 0, "Invalid CQE length");
	cr_assert(cqe.buf == 0, "Invalid CQE address");
	cr_assert(cqe.data == 0, "Invalid CQE data");
	cr_assert(cqe.tag == 0, "Invalid CQE tag");

	/* Validate sent data */
	for (int i = 0; i < send_len; i++)
		cr_assert_eq(mem_window.mem[i], send_buf[i],
			     "data mismatch, element: (%d) %02x != %02x\n", i,
			     mem_window.mem[i], send_buf[i]);

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

	/* Get 8 bytes from the source buffer to the receive buffer */
	ret = fi_read(cxit_ep, local, local_len, NULL, cxit_ep_fi_addr, 0,
		      key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_read() failed (%d)", ret);

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read() failed (%d)", ret);

	/* Validate event fields */
	cr_assert_null(cqe.op_context, "CQE Context mismatch");
	cr_assert_eq(cqe.flags, (FI_RMA | FI_READ), "CQE flags mismatch (%lx)",
		     cqe.flags);
	cr_assert_eq(cqe.len, 0UL, "Invalid CQE length (%lx)", cqe.len);
	cr_assert_null(cqe.buf, "Invalid CQE address (%p)", cqe.buf);
	cr_assert_eq(cqe.data, 0UL, "Invalid CQE data (%lx)", cqe.data);
	cr_assert_eq(cqe.tag, 0UL, "Invalid CQE tag (%lx)", cqe.tag);

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

	/* Validate event fields */
	cr_assert_null(cqe.op_context, "CQE Context mismatch");
	cr_assert_eq(cqe.flags, (FI_RMA | FI_READ), "CQE flags mismatch (%lx)",
		     cqe.flags);
	cr_assert_eq(cqe.len, 0UL, "Invalid CQE length (%lx)", cqe.len);
	cr_assert_null(cqe.buf, "Invalid CQE address (%p)", cqe.buf);
	cr_assert_eq(cqe.data, 0UL, "Invalid CQE data (%lx)", cqe.data);
	cr_assert_eq(cqe.tag, 0UL, "Invalid CQE tag (%lx)", cqe.tag);

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

	/* Validate event fields */
	cr_assert_null(cqe.op_context, "CQE Context mismatch");
	cr_assert_eq(cqe.flags, (FI_RMA | FI_READ), "CQE flags mismatch (%lx)",
		     cqe.flags);
	cr_assert_eq(cqe.len, 0UL, "Invalid CQE length (%lx)", cqe.len);
	cr_assert_null(cqe.buf, "Invalid CQE address (%p)", cqe.buf);
	cr_assert_eq(cqe.data, 0UL, "Invalid CQE data (%lx)", cqe.data);
	cr_assert_eq(cqe.tag, 0UL, "Invalid CQE tag (%lx)", cqe.tag);

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
	struct fi_msg_rma msg = {
		.iov_count = CXIP_RMA_MAX_IOV,
	};
	uint64_t flags = 0;

	/* Invalid msg value */
	ret = fi_readmsg(cxit_ep, NULL, flags);
	cr_assert_eq(ret, -FI_EINVAL, "NULL msg return %d", ret);

	msg.iov_count = CXIP_RMA_MAX_IOV + 1; /* Invalid iov_count value */
	ret = fi_readmsg(cxit_ep, &msg, flags);
	cr_assert_eq(ret, -FI_EINVAL, "Invalid iov_count return %d", ret);

	msg.iov_count = CXIP_RMA_MAX_IOV;
	flags = FI_DIRECTED_RECV; /* Invalid flag value */
	ret = fi_readmsg(cxit_ep, &msg, flags);
	cr_assert_eq(ret, -FI_EBADFLAGS, "NULL msg unexpected return %d", ret);

	flags = FI_COMPLETION; /* Unsupported flag value */
	ret = fi_readmsg(cxit_ep, &msg, flags);
	cr_assert_eq(ret, -FI_EINVAL, "NULL msg unexpected return %d", ret);
}

/* Test fi_writemsg failure cases */
Test(rma, writemsg_failures)
{
	int ret;
	struct fi_msg_rma msg = {
		.iov_count = CXIP_RMA_MAX_IOV,
	};
	uint64_t flags = 0;

	/* Invalid msg value */
	ret = fi_writemsg(cxit_ep, NULL, flags);
	cr_assert_eq(ret, -FI_EINVAL, "NULL msg return %d", ret);

	msg.iov_count = CXIP_RMA_MAX_IOV + 1; /* Invalid iov_count value */
	ret = fi_writemsg(cxit_ep, &msg, flags);
	cr_assert_eq(ret, -FI_EINVAL, "Invalid iov_count return %d", ret);

	msg.iov_count = CXIP_RMA_MAX_IOV;
	flags = FI_DIRECTED_RECV; /* Invalid flag value */
	ret = fi_writemsg(cxit_ep, &msg, flags);
	cr_assert_eq(ret, -FI_EBADFLAGS, "Invalid flag return %d", ret);

	flags = FI_COMPLETION; /* Unsupported flag value */
	ret = fi_writemsg(cxit_ep, &msg, flags);
	cr_assert_eq(ret, -FI_EINVAL, "Unsupported flag return %d", ret);
}

/* Test fi_readv failure cases */
Test(rma, readv_failures)
{
	int ret;
	struct iovec iov = {};

	 /* Invalid count value */
	ret = fi_readv(cxit_ep, &iov, NULL, CXIP_RMA_MAX_IOV + 1,
		       cxit_ep_fi_addr, 0, 0, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "Invalid count return %d", ret);
}

/* Test fi_writev failure cases */
Test(rma, writev_failures)
{
	int ret;
	struct iovec iov = {};

	 /* Invalid count value */
	ret = fi_writev(cxit_ep, &iov, NULL, CXIP_RMA_MAX_IOV + 1,
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
	int key_val = 0x1f;
	struct fi_cq_tagged_entry cqe;

	send_buf = calloc(1, win_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	send_addr = (uint8_t *)FLOOR(send_buf + C_PAGE_SIZE, C_PAGE_SIZE) - 4;
	memset(send_addr, 0xcc, send_len);
	printf("buf: %p addr: %p\n", send_buf, send_addr);

	mr_create(win_len, FI_REMOTE_WRITE, 0xa0, key_val, &mem_window);
	memset(mem_window.mem, 0x33, win_len);

	/* Send 8 bytes from send buffer data to RMA window 0 */
	ret = fi_write(cxit_ep, send_addr, send_len, NULL, cxit_ep_fi_addr, 0,
		       key_val, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

	/* Validate event fields */
	cr_assert(cqe.op_context == NULL, "CQE Context mismatch");
	cr_assert(cqe.flags == (FI_RMA | FI_WRITE), "CQE flags mismatch (%lx)",
		  cqe.flags);
	cr_assert(cqe.len == 0, "Invalid CQE length");
	cr_assert(cqe.buf == 0, "Invalid CQE address");
	cr_assert(cqe.data == 0, "Invalid CQE data");
	cr_assert(cqe.tag == 0, "Invalid CQE tag");

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
	int key_val = 0x1f;
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
