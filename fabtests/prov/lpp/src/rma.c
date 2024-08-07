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

const static uint64_t context = 0xabcd;

static int simple_rma_write_common(struct rank_info *ri, size_t buffer_len)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct rank_info *pri = NULL;

	TRACE(ri, util_simple_setup(ri, &pri, buffer_len, FI_WRITE,
				    FI_REMOTE_WRITE));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_write(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				      buffer_len, NULL, pri->ep_info[0].fi_addr,
				      (uint64_t)pri->mr_info[0].uaddr,
				      pri->mr_info[0].key,
				      get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_RMA | FI_WRITE;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	}

	TRACE(ri, util_barrier(ri));

	if (my_node == NODE_B) {
		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

static int simple_rma_read_common(struct rank_info *ri, size_t buffer_len)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct rank_info *pri = NULL;

	TRACE(ri,
	      util_simple_setup(ri, &pri, buffer_len, FI_READ, FI_REMOTE_READ));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_read(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, pri->ep_info[0].fi_addr,
				     (uint64_t)pri->mr_info[0].uaddr,
				     pri->mr_info[0].key,
				     get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_RMA | FI_READ;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	}

	if (my_node == NODE_A) {
		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_b;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_simple_rma_write(struct rank_info *ri)
{
	return simple_rma_write_common(ri, 1024);
}

int run_offset_rma_write(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct rank_info *pri = NULL;
	const size_t buffer_len = 1024;
	const size_t offset = 12;
	const size_t write_len = buffer_len - offset;

	TRACE(ri, util_simple_setup(ri, &pri, buffer_len, FI_WRITE,
				    FI_REMOTE_WRITE));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_write(ri->ep_info[0].fid,
				      (uint8_t *)ri->mr_info[0].uaddr + offset, write_len,
				      NULL, pri->ep_info[0].fi_addr,
				      (uint64_t)pri->mr_info[0].uaddr + offset,
				      pri->mr_info[0].key,
				      get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_RMA | FI_WRITE;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	}

	TRACE(ri, util_barrier(ri));

	if (my_node == NODE_B) {
		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = write_len;
		verify_buf_params.expected_seed = seed_node_a;
		verify_buf_params.expected_seed_offset = offset;
		verify_buf_params.offset = offset;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_inject_rma_write(struct rank_info *ri)
{
	struct wait_cntr_params wait_cntr_params = { 0 };
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	const size_t buffer_len = 64;
	struct rank_info *pri = NULL;

	TRACE(ri, util_simple_setup(ri, &pri, buffer_len, FI_WRITE,
				    FI_REMOTE_WRITE));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_inject_write(ri->ep_info[0].fid,
					     ri->mr_info[0].uaddr, buffer_len,
					     pri->ep_info[0].fi_addr,
					     (uint64_t)pri->mr_info[0].uaddr,
					     pri->mr_info[0].key),
			     0);

		// Make sure no completion was generated for the inject.
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_RMA | FI_WRITE;
		wait_tx_cq_params.expect_empty = true;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		wait_cntr_params.ep_idx = 0;
		wait_cntr_params.val = 1;
		wait_cntr_params.which = WAIT_CNTR_TX;
		TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));
	}

	TRACE(ri, util_barrier(ri));

	if (my_node == NODE_B) {
		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_simple_rma_read(struct rank_info *ri)
{
	return simple_rma_read_common(ri, 1024);
}

int run_large_rma_write(struct rank_info *ri)
{
	return simple_rma_write_common(ri, 129 * 1024 * 1024);
}

int run_large_rma_read(struct rank_info *ri)
{
	return simple_rma_read_common(ri, 129 * 1024 * 1024);
}

int run_os_bypass_rma(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_cntr_params wait_cntr_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 64;
	struct rank_info *pri = NULL;
	uint8_t *write_buf = NULL;
	uint8_t *peer_buf;
	bool is_lpp_write_only = util_is_write_only();

	TRACE(ri, util_init(ri));

	if (my_node == NODE_A) {
		for (int i = 0; i < 2; i++) {
			mr_params.idx = i;
			mr_params.length = buffer_len;
			mr_params.access = FI_WRITE;
			mr_params.seed = seed_node_a + i;
			TRACE(ri, util_create_mr(ri, &mr_params));
		}
		write_buf = ri->mr_info[0].uaddr;
	} else {
		mr_params.idx = 0;
		mr_params.length = buffer_len;
		// Note: FI_REMOTE_READ is required for PIO, even if
		// the MR is used only for writes. This is because
		// write-only mappings are not possible with x86 MMU.
		mr_params.access = FI_REMOTE_READ | FI_REMOTE_WRITE;
		mr_params.seed = seed_node_b;
		TRACE(ri, util_create_mr(ri, &mr_params));

	}

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	peer_buf = pri->mr_info[0].uaddr;

	// First write will not be an OS bypass; the kernel needs to first get
	// the MR info from the remote side.
	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_write(ri->ep_info[0].fid, &write_buf[0], 8,
				      NULL, pri->ep_info[0].fi_addr,
				      (uint64_t)&peer_buf[0],
				      pri->mr_info[0].key,
				      get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_RMA | FI_WRITE;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		wait_cntr_params.ep_idx = 0;
		wait_cntr_params.val = 1;
		wait_cntr_params.which = WAIT_CNTR_TX;
		TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));
	}

	TRACE(ri, util_barrier(ri));

	if (my_node == NODE_B) {
		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = 8;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_barrier(ri));

	// Second write will be an OS bypass, however, we'll still enter the
	// kernel the first time we write to the mapped region since the page
	// will not yet be present.
	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_write(ri->ep_info[0].fid, &write_buf[8], 8,
				      NULL, pri->ep_info[0].fi_addr,
				      (uint64_t)&peer_buf[8],
				      pri->mr_info[0].key,
				      get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_RMA | FI_WRITE | FI_OS_BYPASS;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		wait_cntr_params.ep_idx = 0;
		wait_cntr_params.val = 2;
		wait_cntr_params.which = WAIT_CNTR_TX;
		TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));
	}

	TRACE(ri, util_barrier(ri));

	if (my_node == NODE_B) {
		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = 8;
		verify_buf_params.offset = 8;
		verify_buf_params.expected_seed = seed_node_a;
		verify_buf_params.expected_seed_offset = 8;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_barrier(ri));

	// Third write will be full OS bypass.
	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_write(ri->ep_info[0].fid, &write_buf[16], 8,
				      NULL, pri->ep_info[0].fi_addr,
				      (uint64_t)&peer_buf[16],
				      pri->mr_info[0].key,
				      get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_RMA | FI_WRITE | FI_OS_BYPASS;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		wait_cntr_params.ep_idx = 0;
		wait_cntr_params.val = 3;
		wait_cntr_params.which = WAIT_CNTR_TX;
		TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));
	}

	TRACE(ri, util_barrier(ri));

	if (my_node == NODE_B) {
		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = 8;
		verify_buf_params.offset = 16;
		verify_buf_params.expected_seed = seed_node_a;
		verify_buf_params.expected_seed_offset = 16;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_barrier(ri));

	// Attempt an OS bypass read into our second MR. The latter 32 bytes
	// have not been written, so they still have the original node_b random
	// seed.
	// os_bypass read is not possible when lpp is in write only mode
	if (!is_lpp_write_only && my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_read(ri->ep_info[0].fid, ri->mr_info[1].uaddr,
				     32, NULL, pri->ep_info[0].fi_addr,
				     (uint64_t)&peer_buf[32],
				     pri->mr_info[0].key,
				     get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_RMA | FI_READ | FI_OS_BYPASS;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		wait_cntr_params.ep_idx = 0;
		wait_cntr_params.val = 4;
		wait_cntr_params.which = WAIT_CNTR_TX;
		TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));

		verify_buf_params.mr_idx = 1;
		verify_buf_params.length = 32;
		verify_buf_params.offset = 0;
		verify_buf_params.expected_seed = seed_node_b;
		verify_buf_params.expected_seed_offset = 32;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));
	return 0;
}

int run_os_bypass_offset_rma(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_cntr_params wait_cntr_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 4096;
	const size_t offsets[] = { 4072, 4080, 4088 };
	struct rank_info *pri = NULL;
	uint8_t *write_buf;
	uint8_t *peer_buf;

	TRACE(ri, util_init(ri));

	if (my_node == NODE_A) {
		mr_params.access = FI_WRITE;
		mr_params.seed = seed_node_a;
	} else {
		mr_params.access = FI_REMOTE_READ | FI_REMOTE_WRITE;
		mr_params.seed = seed_node_b;
	}
	mr_params.idx = 0;
	mr_params.length = buffer_len;
	// We'd like to test reaching the end of the page. To do this, we must
	// know that our MR is aligned, hence page_align.
	mr_params.page_align = true;
	TRACE(ri, util_create_mr(ri, &mr_params));

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	write_buf = ri->mr_info[0].uaddr;
	peer_buf = pri->mr_info[0].uaddr;

	for (int i = 0; i < 3; i++) {
		if (my_node == NODE_A) {
			INSIST_FI_EQ(ri,
				     fi_write(ri->ep_info[0].fid,
					      &write_buf[offsets[i]], 8, NULL,
					      pri->ep_info[0].fi_addr,
					      (uint64_t)&peer_buf[offsets[i]],
					      pri->mr_info[0].key,
					      get_ctx_simple(ri, context)),
				     0);

			wait_tx_cq_params.ep_idx = 0;
			wait_tx_cq_params.context_val = context;
			wait_tx_cq_params.flags = FI_RMA | FI_WRITE;
			// First write must go to kernel to fetch MR info.
			// Subsequent writes should be OS bypass.
			if (i > 0) {
				wait_tx_cq_params.flags |= FI_OS_BYPASS;
			}
			TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

			wait_cntr_params.ep_idx = 0;
			wait_cntr_params.val = i;
			wait_cntr_params.which = WAIT_CNTR_TX;
			TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));
		}

		TRACE(ri, util_barrier(ri));

		if (my_node == NODE_B) {
			verify_buf_params.mr_idx = 0;
			verify_buf_params.length = 8;
			verify_buf_params.offset = offsets[i];
			verify_buf_params.expected_seed = seed_node_a;
			verify_buf_params.expected_seed_offset = offsets[i];
			TRACE(ri, util_verify_buf(ri, &verify_buf_params));
		}

		TRACE(ri, util_barrier(ri));
	}

	TRACE(ri, util_teardown(ri, pri));
	return 0;
}

int run_os_bypass_outofbounds_rma(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 4096;
	const size_t offsets[] = { 4080, 4088 };
	struct rank_info *pri = NULL;
	uint8_t *write_buf;
	uint8_t *peer_buf;

	TRACE(ri, util_init(ri));

	if (my_node == NODE_A) {
		mr_params.access = FI_WRITE;
		mr_params.seed = seed_node_a;
		mr_params.length = buffer_len;
	} else {
		mr_params.access = FI_REMOTE_READ | FI_REMOTE_WRITE;
		mr_params.seed = seed_node_b;
		// This is the key to this test: the B side buffer is a byte
		// short.
		mr_params.length = buffer_len - 1;
	}
	mr_params.idx = 0;
	// We'd like to test reaching the end of the page. To do this, we must
	// know that our MR is aligned, hence page_align.
	mr_params.page_align = true;
	TRACE(ri, util_create_mr(ri, &mr_params));

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	write_buf = ri->mr_info[0].uaddr;
	peer_buf = pri->mr_info[0].uaddr;

	for (int i = 0; i < 2; i++) {
		if (my_node == NODE_A) {
			INSIST_FI_EQ(ri,
				     fi_write(ri->ep_info[0].fid,
					      &write_buf[offsets[i]], 8, NULL,
					      pri->ep_info[0].fi_addr,
					      (uint64_t)&peer_buf[offsets[i]],
					      pri->mr_info[0].key,
					      get_ctx_simple(ri, context)),
				     0);

			wait_tx_cq_params.ep_idx = 0;
			wait_tx_cq_params.context_val = context;
			wait_tx_cq_params.flags = FI_RMA | FI_WRITE;
			// The second write should go off the end of the B side
			// buffer and fail.
			if (i > 0) {
				wait_tx_cq_params.expect_error = true;
			}
			TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
		}

		TRACE(ri, util_barrier(ri));

		// Only first write will make it through (and need
		// verification).
		if (i == 0 && my_node == NODE_B) {
			verify_buf_params.mr_idx = 0;
			verify_buf_params.length = 8;
			verify_buf_params.offset = offsets[i];
			verify_buf_params.expected_seed = seed_node_a;
			verify_buf_params.expected_seed_offset = offsets[i];
			TRACE(ri, util_verify_buf(ri, &verify_buf_params));
		}

		TRACE(ri, util_barrier(ri));
	}

	TRACE(ri, util_teardown(ri, pri));
	return 0;
}

int run_selective_completion_osbypass_error(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_cntr_params wait_cntr_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 4096;
	struct rank_info *pri = NULL;

	TRACE(ri, util_init(ri));

	if (my_node == NODE_A) {
		mr_params.access = FI_WRITE;
		mr_params.seed = seed_node_a;
	} else {
		mr_params.access = FI_REMOTE_READ | FI_REMOTE_WRITE;
		mr_params.seed = seed_node_b;
	}
	mr_params.idx = 0;
	mr_params.length = buffer_len;
	TRACE(ri, util_create_mr(ri, &mr_params));

	ep_params.idx = 0;
	// Here's the main bit for this test: bind the CQ with selective
	// completion.
	ep_params.cq_bind_flags = FI_SELECTIVE_COMPLETION;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	if (my_node == NODE_A) {
		// Issue a couple of writes to warm up the mappings. We want to
		// ensure we're doing OS bypass writes (which cannot occur on
		// the very first write). We'll do the second one with
		// FI_COMPLETION so we can ensure the flags show an OS bypass.
		for (int i = 0; i < 2; i++) {
			struct iovec iov = {
				// XXX: WRONG! But this caused the host to crash!
				//.iov_base = pri->mr_info[0].uaddr,
				.iov_base = ri->mr_info[0].uaddr,
				.iov_len = 16,
			};
			struct fi_rma_iov rma_iov = {
				.addr = (uint64_t)pri->mr_info[0].uaddr,
				.len = buffer_len,
				.key = pri->mr_info[0].key,
			};
			struct fi_msg_rma msg_rma = {
				.msg_iov = &iov,
				.desc = NULL,
				.iov_count = 1,
				.addr = pri->ep_info[0].fi_addr,
				.rma_iov = &rma_iov,
				.rma_iov_count = 1,
				.context = get_ctx_simple(ri, context),
				.data = 0,
			};
			INSIST_FI_EQ(ri,
				     fi_writemsg(ri->ep_info[0].fid, &msg_rma,
						 i == 0 ? 0 : FI_COMPLETION),
				     0);
			if (i == 0) {
				wait_cntr_params.ep_idx = 0;
				wait_cntr_params.val = 1;
				wait_cntr_params.which = WAIT_CNTR_TX;
				TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));

				memset(&wait_tx_cq_params, 0, sizeof(wait_tx_cq_params));
				wait_tx_cq_params.ep_idx = 0;
				wait_tx_cq_params.expect_empty = true;
				TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
			} else {
				memset(&wait_tx_cq_params, 0, sizeof(wait_tx_cq_params));
				wait_tx_cq_params.ep_idx = 0;
				wait_tx_cq_params.context_val = context;
				wait_tx_cq_params.flags = FI_RMA | FI_WRITE | FI_OS_BYPASS;
				TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
			}
		}

		// Issue a write that goes past the end of the remote buffer
		// (should fail).
		INSIST_FI_EQ(ri,
			     fi_write(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				      16, NULL, pri->ep_info[0].fi_addr,
				      (uint64_t)pri->mr_info[0].uaddr +
					      buffer_len - 2,
				      pri->mr_info[0].key,
				      get_ctx_simple(ri, context)),
			     0);
		// Error completion should be present.
		memset(&wait_tx_cq_params, 0, sizeof(wait_tx_cq_params));
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.expect_error = true;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	}

	TRACE(ri, util_teardown(ri, pri));
	return 0;
}

int run_rma_write_auto_reg_mr(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 4096;
	struct rank_info *pri = NULL;

	TRACE(ri, util_init(ri));

	mr_params.idx = 0;
	mr_params.length = buffer_len;
	if (my_node == NODE_A) {
		mr_params.access = FI_WRITE;
		mr_params.seed = seed_node_a;
		// Key to the test: the local side will create a buffer, but
		// will not register it with libfabric.
		mr_params.skip_reg = true;
	} else {
		// Remote side has to register the MR still; we need libfabric
		// to know about it so we can shoot data into it.
		mr_params.access = FI_REMOTE_WRITE;
		mr_params.seed = seed_node_b;
	}
	TRACE(ri, util_create_mr(ri, &mr_params));

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_write(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				      buffer_len, NULL, pri->ep_info[0].fi_addr,
				      (uint64_t)pri->mr_info[0].uaddr,
				      pri->mr_info[0].key,
				      get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_RMA | FI_WRITE;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	}

	TRACE(ri, util_barrier(ri));

	if (my_node == NODE_B) {
		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_rma_read_auto_reg_mr(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 4096;
	struct rank_info *pri = NULL;

	TRACE(ri, util_init(ri));

	mr_params.idx = 0;
	mr_params.length = buffer_len;
	if (my_node == NODE_A) {
		mr_params.access = FI_WRITE;
		mr_params.seed = seed_node_a;
		// Key to the test: the local side will create a buffer, but
		// will not register it with libfabric.
		mr_params.skip_reg = true;
	} else {
		// Remote side has to register the MR still; we need libfabric
		// to know about it so we can shoot data into it.
		mr_params.access = FI_REMOTE_WRITE;
		mr_params.seed = seed_node_b;
	}
	TRACE(ri, util_create_mr(ri, &mr_params));

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_read(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, pri->ep_info[0].fi_addr,
				     (uint64_t)pri->mr_info[0].uaddr,
				     pri->mr_info[0].key,
				     get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_RMA | FI_READ;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	}

	if (my_node == NODE_A) {
		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_b;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

static int loopback_write_common(struct rank_info *ri, size_t buffer_len)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };

	TRACE(ri, util_init(ri));

	// This is a loopback test on NODE_A only. NODE_B gets to take 5.
	if (my_node == NODE_A) {
		for (int i = 0; i < 2; i++) {
			mr_params.idx = i;
			mr_params.length = buffer_len;
			mr_params.access = FI_WRITE | FI_REMOTE_WRITE;
			mr_params.seed = seed_node_a + i;
			TRACE(ri, util_create_mr(ri, &mr_params));

			ep_params.idx = i;
			TRACE(ri, util_create_ep(ri, &ep_params));
		}

		TRACE(ri, util_av_insert_all(ri, ri));

		// Write from endpoint 0 to endpoint 1.
		INSIST_FI_EQ(ri,
			     fi_write(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				      buffer_len, NULL, ri->ep_info[1].fi_addr,
				      (uint64_t)ri->mr_info[1].uaddr,
				      ri->mr_info[1].key,
				      get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_RMA | FI_WRITE;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		verify_buf_params.mr_idx = 1;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, NULL));

	return 0;
}

int run_loopback_write(struct rank_info *ri)
{
	return loopback_write_common(ri, 4096);
}

int run_loopback_small_write(struct rank_info *ri)
{
	return loopback_write_common(ri, 32);
}

static int loopback_read_common(struct rank_info *ri, size_t buffer_len)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };

	TRACE(ri, util_init(ri));

	// This is a loopback test on NODE_A only. NODE_B gets to take 5.
	if (my_node == NODE_A) {
		for (int i = 0; i < 2; i++) {
			mr_params.idx = i;
			mr_params.length = buffer_len;
			mr_params.access = FI_READ | FI_REMOTE_READ;
			mr_params.seed = seed_node_a + i;
			TRACE(ri, util_create_mr(ri, &mr_params));

			ep_params.idx = i;
			TRACE(ri, util_create_ep(ri, &ep_params));
		}

		TRACE(ri, util_av_insert_all(ri, ri));

		// Read from endpoint 1 into endpoint 0.
		INSIST_FI_EQ(ri,
			     fi_read(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				      buffer_len, NULL, ri->ep_info[1].fi_addr,
				      (uint64_t)ri->mr_info[1].uaddr,
				      ri->mr_info[1].key,
				      get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_RMA | FI_READ;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a + 1;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, NULL));

	return 0;
}

int run_loopback_read(struct rank_info *ri)
{
	return loopback_read_common(ri, 4096);
}

int run_loopback_small_read(struct rank_info *ri)
{
	return loopback_read_common(ri, 32);
}
