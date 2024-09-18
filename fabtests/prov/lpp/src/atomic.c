/*
 * Copyright (c) 2018-2024 GigaIO, Inc. All Rights Reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "test_util.h"

static const uint64_t context = 0xabce;

int run_simple_atomic_write(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	const size_t buffer_len = 64;
	const size_t count = buffer_len / 2;
	struct rank_info *pri = NULL;

	TRACE(ri, util_simple_setup(ri, &pri, buffer_len, FI_WRITE, FI_REMOTE_WRITE));

	if (my_node == NODE_A) {
		struct fi_ioc msg_iov = {
			.addr = ri->mr_info[0].uaddr,
			.count = count,
		};
		struct fi_rma_ioc rma_iov = {
			.addr = (uint64_t)pri->mr_info[0].uaddr,
			.count = count,
			.key = pri->mr_info[0].key,
		};
		struct fi_msg_atomic msg = {
			.msg_iov = &msg_iov,
			.desc = NULL,
			.iov_count = 1,
			.addr = pri->ep_info[0].fi_addr,
			.rma_iov = &rma_iov,
			.rma_iov_count = 1,
			.datatype = FI_UINT16,
			.op = FI_ATOMIC_WRITE,
			.context = get_ctx_simple(ri, context),
			.data = 0,
		};
		INSIST_FI_EQ(ri, fi_atomicmsg(ri->ep_info[0].fid, &msg, 0), 0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_ATOMIC;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		TRACE(ri, util_barrier(ri));
	} else {
		TRACE(ri, util_barrier(ri));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_simple_atomic_write2(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	const size_t buffer_len = 64;
	const size_t count = buffer_len / 2;
	struct rank_info *pri = NULL;

	TRACE(ri, util_simple_setup(ri, &pri, buffer_len, FI_WRITE, FI_REMOTE_WRITE));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_atomic(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				       count, NULL, pri->ep_info[0].fi_addr,
				       (uint64_t)pri->mr_info[0].uaddr,
				       pri->mr_info[0].key, FI_UINT16,
				       FI_ATOMIC_WRITE,
				       get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_ATOMIC;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		TRACE(ri, util_barrier(ri));
	} else {
		TRACE(ri, util_barrier(ri));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_simple_atomic_fetch_write(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 64;
	const size_t count = buffer_len / 2;
	struct rank_info *pri = NULL;

	TRACE(ri, util_init(ri));

	mr_params.length = buffer_len;
	if (my_node == NODE_A) {
		mr_params.idx = 0;
		mr_params.access = FI_WRITE;
		mr_params.seed = seed_node_a;
		TRACE(ri, util_create_mr(ri, &mr_params));

		mr_params.idx = 1;
		mr_params.access = FI_READ;
		mr_params.seed = seed_node_a + 1;
		TRACE(ri, util_create_mr(ri, &mr_params));
	} else {
		mr_params.idx = 0;
		mr_params.access = FI_REMOTE_WRITE | FI_REMOTE_READ;
		mr_params.seed = seed_node_b;
		TRACE(ri, util_create_mr(ri, &mr_params));
	}

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));
	TRACE(ri, util_sync(ri, &pri));

	if (my_node == NODE_A) {
		struct fi_ioc msg_iov = {
			.addr = ri->mr_info[0].uaddr,
			.count = count,
		};
		struct fi_rma_ioc rma_iov = {
			.addr = (uint64_t)pri->mr_info[0].uaddr,
			.count = count,
			.key = pri->mr_info[0].key,
		};
		struct fi_msg_atomic msg = {
			.msg_iov = &msg_iov,
			.desc = NULL,
			.iov_count = 1,
			.addr = pri->ep_info[0].fi_addr,
			.rma_iov = &rma_iov,
			.rma_iov_count = 1,
			.datatype = FI_UINT16,
			.op = FI_ATOMIC_WRITE,
			.context = get_ctx_simple(ri, context),
			.data = 0,
		};
		struct fi_ioc result_iov = {
			.addr = ri->mr_info[1].uaddr,
			.count = count,
		};
		INSIST_FI_EQ(ri,
			     fi_fetch_atomicmsg(ri->ep_info[0].fid, &msg,
						&result_iov, NULL, 1, 0),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_ATOMIC;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		TRACE(ri, util_barrier(ri));

		verify_buf_params.mr_idx = 1;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_b;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	} else {
		TRACE(ri, util_barrier(ri));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_simple_atomic_fetch_write2(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 64;
	const size_t count = buffer_len / 2;
	struct rank_info *pri = NULL;

	TRACE(ri, util_init(ri));

	mr_params.length = buffer_len;
	if (my_node == NODE_A) {
		mr_params.idx = 0;
		mr_params.access = FI_WRITE;
		mr_params.seed = seed_node_a;
		TRACE(ri, util_create_mr(ri, &mr_params));

		mr_params.idx = 1;
		mr_params.access = FI_READ;
		mr_params.seed = seed_node_a + 1;
		TRACE(ri, util_create_mr(ri, &mr_params));
	} else {
		mr_params.idx = 0;
		mr_params.access = FI_REMOTE_WRITE | FI_REMOTE_READ;
		mr_params.seed = seed_node_b;
		TRACE(ri, util_create_mr(ri, &mr_params));
	}

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));
	TRACE(ri, util_sync(ri, &pri));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_fetch_atomic(ri->ep_info[0].fid,
					     ri->mr_info[0].uaddr, count, NULL,
					     ri->mr_info[1].uaddr, NULL,
					     pri->ep_info[0].fi_addr,
					     (uint64_t)pri->mr_info[0].uaddr,
					     pri->mr_info[0].key, FI_UINT16,
					     FI_ATOMIC_WRITE,
					     get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_ATOMIC;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		TRACE(ri, util_barrier(ri));

		verify_buf_params.mr_idx = 1;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_b;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	} else {
		TRACE(ri, util_barrier(ri));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_simple_atomic_fetch_read(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	const size_t buffer_len = 64;
	const size_t count = buffer_len / 2;
	struct rank_info *pri = NULL;

	TRACE(ri,
	      util_simple_setup(ri, &pri, buffer_len, FI_READ, FI_REMOTE_READ));

	if (my_node == NODE_A) {
		struct fi_rma_ioc rma_iov = {
			.addr = (uint64_t)pri->mr_info[0].uaddr,
			.count = count,
			.key = pri->mr_info[0].key,
		};
		struct fi_msg_atomic msg = {
			.msg_iov = NULL,
			.desc = NULL,
			.iov_count = 1,
			.addr = pri->ep_info[0].fi_addr,
			.rma_iov = &rma_iov,
			.rma_iov_count = 1,
			.datatype = FI_UINT16,
			.op = FI_ATOMIC_READ,
			.context = get_ctx_simple(ri, context),
			.data = 0,
		};
		struct fi_ioc result_iov = {
			.addr = ri->mr_info[0].uaddr,
			.count = count,
		};
		INSIST_FI_EQ(ri,
			     fi_fetch_atomicmsg(ri->ep_info[0].fid, &msg,
						&result_iov, NULL, 1, 0),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_ATOMIC;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		TRACE(ri, util_barrier(ri));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_b;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	} else {
		TRACE(ri, util_barrier(ri));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_simple_atomic_fetch_read2(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	const size_t buffer_len = 64;
	const size_t count = buffer_len / 2;
	struct rank_info *pri = NULL;

	TRACE(ri,
	      util_simple_setup(ri, &pri, buffer_len, FI_READ, FI_REMOTE_READ));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_fetch_atomic(ri->ep_info[0].fid, NULL, count,
					     NULL, ri->mr_info[0].uaddr, NULL,
					     pri->ep_info[0].fi_addr,
					     (uint64_t)pri->mr_info[0].uaddr,
					     pri->mr_info[0].key, FI_UINT16,
					     FI_ATOMIC_READ,
					     get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_ATOMIC;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		TRACE(ri, util_barrier(ri));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_b;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	} else {
		TRACE(ri, util_barrier(ri));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_simple_atomic_cswap(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 64;
	const size_t count = buffer_len / 4;
	struct rank_info *pri = NULL;

	TRACE(ri, util_init(ri));

	mr_params.length = buffer_len;
	if (my_node == NODE_A) {
		mr_params.idx = 0;
		mr_params.access = FI_WRITE;
		mr_params.seed = seed_node_a;
		TRACE(ri, util_create_mr(ri, &mr_params));

		mr_params.idx = 1;
		mr_params.access = FI_WRITE;
		mr_params.seed = seed_node_a + 1;
		TRACE(ri, util_create_mr(ri, &mr_params));

		mr_params.idx = 2;
		mr_params.access = FI_READ;
		mr_params.seed = seed_node_a + 2;
		TRACE(ri, util_create_mr(ri, &mr_params));
	} else {
		mr_params.idx = 0;
		mr_params.access = FI_REMOTE_WRITE | FI_REMOTE_READ;
		// The compare buffer on NODE_A is index 1. We'll match its
		// initial data here so all the cswap operations match and do
		// the swap.
		mr_params.seed = seed_node_a + 1;
		TRACE(ri, util_create_mr(ri, &mr_params));
	}

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));
	TRACE(ri, util_sync(ri, &pri));

	if (my_node == NODE_A) {
		struct fi_ioc msg_iov = {
			.addr = ri->mr_info[0].uaddr,
			.count = count,
		};
		struct fi_rma_ioc rma_iov = {
			.addr = (uint64_t)pri->mr_info[0].uaddr,
			.count = count,
			.key = pri->mr_info[0].key,
		};
		struct fi_msg_atomic msg = {
			.msg_iov = &msg_iov,
			.desc = NULL,
			.iov_count = 1,
			.addr = pri->ep_info[0].fi_addr,
			.rma_iov = &rma_iov,
			.rma_iov_count = 1,
			.datatype = FI_UINT16,
			.op = FI_CSWAP,
			.context = get_ctx_simple(ri, context),
			.data = 0,
		};
		struct fi_ioc compare_iov = {
			.addr = ri->mr_info[1].uaddr,
			.count = count,
		};
		struct fi_ioc result_iov = {
			.addr = ri->mr_info[2].uaddr,
			.count = count,
		};
		INSIST_FI_EQ(ri,
			     fi_compare_atomicmsg(ri->ep_info[0].fid, &msg,
						  &compare_iov, NULL, 1,
						  &result_iov, NULL, 1, 0),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_ATOMIC;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		TRACE(ri, util_barrier(ri));

		verify_buf_params.mr_idx = 1;
		verify_buf_params.length = count;
		verify_buf_params.expected_seed = seed_node_a + 1;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	} else {
		TRACE(ri, util_barrier(ri));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = count;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_simple_atomic_cswap2(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 64;
	const size_t count = buffer_len / 2;
	struct rank_info *pri = NULL;

	TRACE(ri, util_init(ri));

	mr_params.length = buffer_len;
	if (my_node == NODE_A) {
		mr_params.idx = 0;
		mr_params.access = FI_WRITE;
		mr_params.seed = seed_node_a;
		TRACE(ri, util_create_mr(ri, &mr_params));

		mr_params.idx = 1;
		mr_params.access = FI_WRITE;
		mr_params.seed = seed_node_a + 1;
		TRACE(ri, util_create_mr(ri, &mr_params));

		mr_params.idx = 2;
		mr_params.access = FI_READ;
		mr_params.seed = seed_node_a + 2;
		TRACE(ri, util_create_mr(ri, &mr_params));
	} else {
		mr_params.idx = 0;
		mr_params.access = FI_REMOTE_WRITE | FI_REMOTE_READ;
		// The compare buffer on NODE_A is index 1. We'll match its
		// initial data here so all the cswap operations match and do
		// the swap.
		mr_params.seed = seed_node_a + 1;
		TRACE(ri, util_create_mr(ri, &mr_params));
	}

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));
	TRACE(ri, util_sync(ri, &pri));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_compare_atomic(
				     ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     count, NULL, ri->mr_info[1].uaddr, NULL,
				     ri->mr_info[2].uaddr, NULL,
				     pri->ep_info[0].fi_addr,
				     (uint64_t)pri->mr_info[0].uaddr,
				     pri->mr_info[0].key, FI_UINT8, FI_CSWAP,
				     get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_ATOMIC;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		TRACE(ri, util_barrier(ri));

		verify_buf_params.mr_idx = 1;
		verify_buf_params.length = count;
		verify_buf_params.expected_seed = seed_node_a + 1;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	} else {
		TRACE(ri, util_barrier(ri));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = count;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}
