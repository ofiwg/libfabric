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

#include <unistd.h>

#include "test_util.h"

static const uint64_t context = 0xabcd;

int run_simple_msg(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	const size_t buffer_len = 4096;
	struct rank_info *pri = NULL;

	TRACE(ri, util_simple_setup(ri, &pri, buffer_len, FI_SEND, FI_RECV));

	if (my_node == NODE_B) {
		INSIST_FI_EQ(ri,
			     fi_recv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, FI_ADDR_UNSPEC,
				     get_ctx_simple(ri, context)),
			     0);
	}

	// We sync here to ensure the recv side has setup its buffer before we
	// send. We don't want an unexpected msg here (that's a separate test).
	TRACE(ri, util_barrier(ri));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_send(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, pri->ep_info[0].fi_addr,
				     get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_MSG | FI_SEND;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	} else {
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_MSG | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_simple_small_msg(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	const size_t buffer_len = 1024;
	const size_t send_len = 16;
	struct rank_info *pri = NULL;

	TRACE(ri, util_simple_setup(ri, &pri, buffer_len, FI_SEND, FI_RECV));

	if (my_node == NODE_B) {
		INSIST_FI_EQ(ri,
			     fi_recv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, FI_ADDR_UNSPEC,
				     get_ctx_simple(ri, context)),
			     0);
	}

	// We sync here to ensure the recv side has setup its buffer before we
	// send. We don't want an unexpected msg here (that's a separate test).
	TRACE(ri, util_barrier(ri));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_send(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     send_len, NULL, pri->ep_info[0].fi_addr,
				     get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_MSG | FI_SEND;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	}

	if (my_node == NODE_B) {
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_MSG | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = send_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_inject_msg(struct rank_info *ri)
{
	struct wait_cntr_params wait_cntr_params = { 0 };
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	const size_t buffer_len = 1024;
	const size_t send_len = 64;
	struct rank_info *pri = NULL;
	int ret;

	TRACE(ri, util_simple_setup(ri, &pri, buffer_len, FI_SEND, FI_RECV));

	if (my_node == NODE_B) {
		INSIST_FI_EQ(ri,
			     fi_recv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, FI_ADDR_UNSPEC,
				     get_ctx_simple(ri, context)),
			     0);
	}

	// We sync here to ensure the recv side has setup its buffer before we
	// send. We don't want an unexpected msg here (that's a separate test).
	TRACE(ri, util_barrier(ri));

	if (my_node == NODE_A) {
		SEND_AND_INSIST_EQ(ri, ret,
					   fi_inject(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
					   send_len, pri->ep_info[0].fi_addr),
					   0);

		// Make sure no completion was generated for the inject.
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_MSG | FI_SEND;
		wait_tx_cq_params.expect_empty = true;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		wait_cntr_params.ep_idx = 0;
		wait_cntr_params.val = 1;
		wait_cntr_params.which = WAIT_CNTR_TX;
		TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));
	}

	if (my_node == NODE_B) {
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_MSG | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = send_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_tagged_msg(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 4096;
	struct rank_info *pri = NULL;
	const uint64_t contexts[] = { context, context + 1, context + 2 };
	const uint64_t tags[] = { 0xfff10001, 0xfff20002 };

	TRACE(ri, util_init(ri));

	for (int i = 0; i < 3; i++) {
		mr_params.idx = i;
		mr_params.length = buffer_len;
		if (my_node == NODE_A) {
			mr_params.access = FI_SEND;
			mr_params.seed = seed_node_a + i;
		} else {
			mr_params.access = FI_RECV;
			mr_params.seed = seed_node_b + i;
		}
		TRACE(ri, util_create_mr(ri, &mr_params));
	}

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	// Set up 3 recv buffers. Two tagged (with different tags) and one
	// untagged.
	if (my_node == NODE_B) {
		INSIST_FI_EQ(ri,
			     fi_recv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, FI_ADDR_UNSPEC,
				     get_ctx_simple(ri, contexts[0])),
			     0);
		INSIST_FI_EQ(ri,
			     fi_trecv(ri->ep_info[0].fid, ri->mr_info[1].uaddr,
				      buffer_len, NULL, FI_ADDR_UNSPEC, tags[0],
				      0xffff, get_ctx_simple(ri, contexts[1])),
			     0);
		INSIST_FI_EQ(ri,
			     fi_trecv(ri->ep_info[0].fid, ri->mr_info[2].uaddr,
				      buffer_len, NULL, FI_ADDR_UNSPEC, tags[1],
				      0xffff, get_ctx_simple(ri, contexts[2])),
			     0);
	}

	// We sync here to ensure the recv side has setup its buffer before we
	// send. We don't want an unexpected msg here (that's a separate test).
	TRACE(ri, util_barrier(ri));

	// Target the second tagged buffer (tags[0]) with the first message.
	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_tsend(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				      buffer_len, NULL, pri->ep_info[0].fi_addr,
				      tags[0], get_ctx_simple(ri, contexts[0])),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = contexts[0];
		wait_tx_cq_params.flags = FI_TAGGED | FI_SEND;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	} else {
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = contexts[1];
		wait_rx_cq_params.flags = FI_TAGGED | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		verify_buf_params.mr_idx = 1;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	// Target the third tagged buffer (tags[1]) with the second message.
	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_tsend(ri->ep_info[0].fid, ri->mr_info[1].uaddr,
				      buffer_len, NULL, pri->ep_info[0].fi_addr,
				      tags[1], get_ctx_simple(ri, contexts[0])),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = contexts[0];
		wait_tx_cq_params.flags = FI_TAGGED | FI_SEND;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	} else {
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = contexts[2];
		wait_rx_cq_params.flags = FI_TAGGED | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		verify_buf_params.mr_idx = 2;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a + 1;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	// Target the first buffer (untagged) with the third message.
	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_send(ri->ep_info[0].fid, ri->mr_info[2].uaddr,
				     buffer_len, NULL, pri->ep_info[0].fi_addr,
				     get_ctx_simple(ri, contexts[0])),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = contexts[0];
		wait_tx_cq_params.flags = FI_MSG | FI_SEND;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	} else {
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = contexts[0];
		wait_rx_cq_params.flags = FI_MSG | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a + 2;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_directed_recv_msg(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 4096;
	struct rank_info *pri = NULL;
	const uint64_t contexts[] = { context, context + 1, context + 2 };
	fi_addr_t filter_addrs[3] = { 0 };

	TRACE(ri, util_init(ri));

	for (int i = 0; i < 3; i++) {
		mr_params.idx = i;
		mr_params.length = buffer_len;
		if (my_node == NODE_A) {
			mr_params.access = FI_SEND;
			mr_params.seed = seed_node_a + i;
		} else {
			mr_params.access = FI_RECV;
			mr_params.seed = seed_node_b + i;
		}
		TRACE(ri, util_create_mr(ri, &mr_params));
	}

	ep_params.idx = 0;
	if (my_node == NODE_B) {
		ep_params.rx_attr.additional_caps = FI_DIRECTED_RECV;
	}
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	// Set up 3 recv buffers. The first will have a dummy address, and
	// should never match a message. The second will explicitly match the
	// other node. The third will be a wildcard match.
	if (my_node == NODE_B) {
		filter_addrs[0] = 0x1234;
		filter_addrs[1] = pri->ep_info[0].fi_addr;
		filter_addrs[2] = FI_ADDR_UNSPEC;

		for (int i = 0; i < 3; i++) {
			INSIST_FI_EQ(ri,
				     fi_recv(ri->ep_info[0].fid,
					     ri->mr_info[i].uaddr, buffer_len,
					     NULL, filter_addrs[i],
					     get_ctx_simple(ri, contexts[i])),
				     0);
		}
	}

	// We sync here to ensure the recv side has setup its buffer before we
	// send. We don't want an unexpected msg here (that's a separate test).
	TRACE(ri, util_barrier(ri));

	// Target the second buffer (matches NODE_A) with the first message.
	// (The first buffer should be skipped since its address doesn't
	// match.)
	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_send(ri->ep_info[0].fid, ri->mr_info[1].uaddr,
				     buffer_len, NULL, pri->ep_info[0].fi_addr,
				     get_ctx_simple(ri, contexts[1])),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = contexts[1];
		wait_tx_cq_params.flags = FI_MSG | FI_SEND;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	} else {
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = contexts[1];
		wait_rx_cq_params.flags = FI_MSG | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		verify_buf_params.mr_idx = 1;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a + 1;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	// Target the third buffer (matches any node) with the second message.
	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_send(ri->ep_info[0].fid, ri->mr_info[2].uaddr,
				     buffer_len, NULL, pri->ep_info[0].fi_addr,
				     get_ctx_simple(ri, contexts[2])),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = contexts[2];
		wait_tx_cq_params.flags = FI_MSG | FI_SEND;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	} else {
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = contexts[2];
		wait_rx_cq_params.flags = FI_MSG | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		verify_buf_params.mr_idx = 2;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a + 2;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

static int simple_multi_recv_common(struct rank_info *ri, size_t buffer_len)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct rank_info *pri = NULL;
	size_t min_multi_recv = 0;

	TRACE(ri, util_simple_setup(ri, &pri, buffer_len, FI_SEND, FI_RECV));

	if (my_node == NODE_B) {
		INSIST_FI_EQ(ri,
			     fi_setopt(&ri->ep_info[0].fid->fid,
				       FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
				       &min_multi_recv, sizeof(min_multi_recv)),
			     0);

		struct iovec iov = {
			.iov_base = ri->mr_info[0].uaddr,
			.iov_len = buffer_len
		};
		struct fi_msg msg = {
			.msg_iov = &iov,
			.desc = NULL,
			.iov_count = 1,
			.addr = FI_ADDR_UNSPEC,
			.context = get_ctx(ri, context, 0),
			.data = 0,
		};
		INSIST_FI_EQ(
			ri, fi_recvmsg(ri->ep_info[0].fid, &msg, FI_MULTI_RECV),
			0);
	}

	TRACE(ri, util_barrier(ri));

	// Send the data it two halves (it should arrive in the same multi recv
	// buffer at the destination).
	if (my_node == NODE_A) {
		for (int i = 0; i < 2; i++) {
			INSIST_FI_EQ(ri,
				     fi_send(ri->ep_info[0].fid,
					     (uint8_t *)ri->mr_info[0].uaddr +
						     (i * buffer_len / 2),
					     buffer_len / 2, NULL,
					     pri->ep_info[0].fi_addr,
					     get_ctx_simple(ri, context)),
				     0);

			wait_tx_cq_params.ep_idx = 0;
			wait_tx_cq_params.context_val = context;
			wait_tx_cq_params.flags = FI_MSG | FI_SEND;
			TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
		}
	} else {
		for (int i = 0; i < 2; i++) {
			wait_rx_cq_params.ep_idx = 0;
			wait_rx_cq_params.context_val = context;
			wait_rx_cq_params.flags = FI_MSG | FI_RECV;
			wait_rx_cq_params.multi_recv = true;
			wait_rx_cq_params.buf_offset = (i * buffer_len / 2);
			TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

			verify_buf_params.mr_idx = 0;
			verify_buf_params.length = buffer_len / 2;
			verify_buf_params.expected_seed = seed_node_a;
			verify_buf_params.expected_seed_offset = (i * buffer_len / 2);
			verify_buf_params.offset = (i * buffer_len / 2);
			TRACE(ri, util_verify_buf(ri, &verify_buf_params));
		}
		// Reap the completion indicating the multi recv buffer has
		// been consumed.
		memset(&wait_rx_cq_params, 0, sizeof(wait_rx_cq_params));
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_MULTI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_multi_recv_msg(struct rank_info *ri)
{
	return simple_multi_recv_common(ri, 4096);
}

int run_multi_recv_small_msg(struct rank_info *ri)
{
	return simple_multi_recv_common(ri, 64);
}

int run_unexpected_msg(struct rank_info *ri)
{
	struct wait_cntr_params wait_cntr_params = { 0 };
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t recv_buffer_len = 2048;
	struct rank_info *pri = NULL;
	const size_t send_buf_lens[] = { 64, 64, 2048, 64, 64 };
	const uint64_t contexts[] = { context, context + 1, context + 2,
				      context + 3, context + 4 };
	const uint64_t tags[] = { 1, 2, 3, 4, 5 };

	TRACE(ri, util_init(ri));

	if (my_node == NODE_A) {
		for (int i = 0; i < 5; i++) {
			mr_params.idx = i;
			mr_params.length = send_buf_lens[i];
			mr_params.access = FI_SEND;
			mr_params.seed = seed_node_a + i;
			TRACE(ri, util_create_mr(ri, &mr_params));
		}
	} else {
			mr_params.idx = 0;
			mr_params.length = recv_buffer_len;
			mr_params.access = FI_RECV;
			mr_params.seed = seed_node_b;
			TRACE(ri, util_create_mr(ri, &mr_params));
	}

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	if (my_node == NODE_A) {
		// Send all 5 messages. Since no recv has been posted yet,
		// these should be unexpected msgs.
		for (int i = 0; i < 5; i++) {
			INSIST_FI_EQ(ri,
				     fi_tsend(ri->ep_info[0].fid,
					      ri->mr_info[i].uaddr,
					      send_buf_lens[i], NULL,
					      pri->ep_info[0].fi_addr, tags[i],
					      get_ctx_simple(ri, contexts[i])),
				     0);
		}
	}

	TRACE(ri, util_barrier(ri));

	if (my_node == NODE_B) {
		// Recv the last message (tag[4]). Recving the last msg ensures
		// all the previous ones have arrived and therefore must be in
		// the unexpected queue.
		INSIST_FI_EQ(ri,
			     fi_trecv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				      recv_buffer_len, NULL, FI_ADDR_UNSPEC,
				      tags[4], 0, get_ctx(ri, context, 0)),
			     0);

		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_TAGGED | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = send_buf_lens[4];
		verify_buf_params.expected_seed = seed_node_a + 4;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));

		// Try to peek a non-existent message (should fail).
		struct fi_msg_tagged msg = {
			.msg_iov = NULL,
			.desc = NULL,
			.iov_count = 0,
			.addr = FI_ADDR_UNSPEC,
			.tag = 0xBAD,
			.ignore = 0,
			.context = get_ctx_simple(ri, context),
			.data = 0,
		};
		INSIST_FI_EQ(ri, fi_trecvmsg(ri->ep_info[0].fid, &msg, FI_PEEK),
			     0);

		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = 0;
		wait_rx_cq_params.expect_error = true;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		// Try to PEEK and CLAIM and non-existent message (also should
		// fail).
		INSIST_FI_EQ(ri,
			     fi_trecvmsg(ri->ep_info[0].fid, &msg,
					 FI_PEEK | FI_CLAIM),
			     0);
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		// Recv the third message.
		INSIST_FI_EQ(ri,
			     fi_trecv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				      recv_buffer_len, NULL, FI_ADDR_UNSPEC,
				      tags[2], 0, get_ctx_simple(ri, context)),
			     0);

		memset(&wait_rx_cq_params, 0, sizeof(wait_rx_cq_params));
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_TAGGED | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = send_buf_lens[2];
		verify_buf_params.expected_seed = seed_node_a + 2;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));

		// Claim the first message.
		msg.tag = tags[0];
		INSIST_FI_EQ(ri,
			     fi_trecvmsg(ri->ep_info[0].fid, &msg,
					 FI_PEEK | FI_CLAIM),
			     0);
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_TAGGED | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		// Try to peek the first message. This should fail as it was already claimed.
		INSIST_FI_EQ(ri, fi_trecvmsg(ri->ep_info[0].fid, &msg, FI_PEEK),
			     0);
		wait_rx_cq_params.flags = 0;
		wait_rx_cq_params.expect_error = true;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		// Retrieve the first message (which we previously claimed).
		// The context value must be the same as when we claimed it.
		struct iovec iov = {
			.iov_base = ri->mr_info[0].uaddr,
			.iov_len = recv_buffer_len,
		};
		msg.msg_iov = &iov;
		msg.iov_count = 1;
		INSIST_FI_EQ(
			ri, fi_trecvmsg(ri->ep_info[0].fid, &msg, FI_CLAIM), 0);

		memset(&wait_rx_cq_params, 0, sizeof(wait_rx_cq_params));
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_TAGGED | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = send_buf_lens[0];
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));

		// Peek and discard the second message.
		msg.tag = tags[1];
		INSIST_FI_EQ(ri,
			     fi_trecvmsg(ri->ep_info[0].fid, &msg,
					 FI_PEEK | FI_DISCARD),
			     0);
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		// Claim the fourth message.
		msg.tag = tags[3];
		INSIST_FI_EQ(ri,
			     fi_trecvmsg(ri->ep_info[0].fid, &msg,
					 FI_PEEK | FI_CLAIM),
			     0);
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		// Discard the fourth message (which we just claimed).
		INSIST_FI_EQ(ri,
			     fi_trecvmsg(ri->ep_info[0].fid, &msg,
					 FI_CLAIM | FI_DISCARD),
			     0);
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));
	} else {
		// Reap completions for all the sent messages.
		wait_cntr_params.ep_idx = 0;
		wait_cntr_params.val = 5;
		wait_cntr_params.which = WAIT_CNTR_TX;
		TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_unexpected_multi_recv_msg(struct rank_info *ri)
{
	struct wait_cntr_params wait_cntr_params = { 0 };
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t recv_buffer_len = 128;
	const size_t send_buffer_len = 64;
	size_t min_multi_recv = 64;
	struct rank_info *pri = NULL;
	const uint64_t contexts[] = { context, context + 1, context + 2 };
	const uint64_t tags[] = { 0xffff0001, 0xffff0002, 0xdeadb0b0 };

	TRACE(ri, util_init(ri));

	if (my_node == NODE_A) {
		for (int i = 0; i < 3; i++) {
			mr_params.idx = i;
			mr_params.length = send_buffer_len;
			mr_params.access = FI_SEND;
			mr_params.seed = seed_node_a + i;
			TRACE(ri, util_create_mr(ri, &mr_params));
		}
	} else {
			mr_params.idx = 0;
			mr_params.length = recv_buffer_len;
			mr_params.access = FI_RECV;
			mr_params.seed = seed_node_b;
			TRACE(ri, util_create_mr(ri, &mr_params));
	}

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));

	if (my_node == NODE_B) {
		INSIST_FI_EQ(ri,
			     fi_setopt(&ri->ep_info[0].fid->fid,
				       FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
				       &min_multi_recv, sizeof(min_multi_recv)),
			     0);
	}

	TRACE(ri, util_sync(ri, &pri));

	if (my_node == NODE_A) {
		// Send all 3 messages. Since no recv has been posted yet,
		// these should be unexpected msgs.
		for (int i = 0; i < 3; i++) {
			INSIST_FI_EQ(ri,
				     fi_tsend(ri->ep_info[0].fid,
					      ri->mr_info[i].uaddr,
					      send_buffer_len, NULL,
					      pri->ep_info[0].fi_addr, tags[i],
					      get_ctx_simple(ri, contexts[i])),
				     0);
		}
	}

	TRACE(ri, util_barrier(ri));

	if (my_node == NODE_B) {
		// Recving the last msg ensures all the previous ones have
		// arrived and therefore must be in the unexpected queue.
		INSIST_FI_EQ(ri,
			     fi_trecv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				      recv_buffer_len, NULL, FI_ADDR_UNSPEC,
				      tags[2], 0, get_ctx_simple(ri, context)),
			     0);

		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_TAGGED | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = send_buffer_len;
		verify_buf_params.expected_seed = seed_node_a + 2;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));

		// Post a multi recv buffer which matches both the unexpected
		// msgs.
		struct iovec iov = {
			.iov_base = ri->mr_info[0].uaddr,
			.iov_len = recv_buffer_len,
		};
		struct fi_msg_tagged msg = {
			.msg_iov = &iov,
			.desc = NULL,
			.iov_count = 1,
			.tag = 0xffff0000,
			.ignore = 0xffff,
			.addr = FI_ADDR_UNSPEC,
			.context = get_ctx(ri, context, 0),
			.data = 0,
		};
		INSIST_FI_EQ(ri,
			     fi_trecvmsg(ri->ep_info[0].fid, &msg,
					 FI_MULTI_RECV),
			     0);

		// Both unexpected messages should be matched and recvd.
		for (int i = 0; i < 2; i++) {
			wait_rx_cq_params.ep_idx = 0;
			wait_rx_cq_params.context_val = context;
			wait_rx_cq_params.flags = FI_TAGGED | FI_RECV;
			wait_rx_cq_params.multi_recv = true;
			wait_rx_cq_params.buf_offset = i * send_buffer_len;
			TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

			verify_buf_params.mr_idx = 0;
			verify_buf_params.length = send_buffer_len;
			verify_buf_params.expected_seed = seed_node_a + i;
			verify_buf_params.offset = i * send_buffer_len;
			TRACE(ri, util_verify_buf(ri, &verify_buf_params));
		}
		// Reap the completion indicating the multi recv buffer has
		// been consumed.
		memset(&wait_rx_cq_params, 0, sizeof(wait_rx_cq_params));
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_MULTI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));
	} else {
		// Reap completions for all the sent messages.
		wait_cntr_params.ep_idx = 0;
		wait_cntr_params.val = 3;
		wait_cntr_params.which = WAIT_CNTR_TX;
		TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

// TODO: actually includes RMA ops, even though we're in msg.c.
int run_selective_completion(struct rank_info *ri)
{
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_cntr_params wait_cntr_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 4096;
	struct rank_info *pri = NULL;

	TRACE(ri, util_init(ri));

	if (my_node == NODE_A) {
		mr_params.access = FI_SEND | FI_WRITE;
		mr_params.seed = seed_node_a;
	} else {
		mr_params.access = FI_RECV | FI_REMOTE_WRITE;
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
		INSIST_FI_EQ(ri,
			     fi_send(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, pri->ep_info[0].fi_addr,
				     get_ctx_simple(ri, context)),
			     0);

		// Counter shows operation has completed.
		wait_cntr_params.ep_idx = 0;
		wait_cntr_params.val = 1;
		wait_cntr_params.which = WAIT_CNTR_TX;
		TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));

		//...but there should be no CQ entry generated.
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.expect_empty = true;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		// Issue a write.
		INSIST_FI_EQ(ri,
			     fi_write(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				      buffer_len, NULL, pri->ep_info[0].fi_addr,
				      (uint64_t)pri->mr_info[0].uaddr,
				      pri->mr_info[0].key,
				      get_ctx_simple(ri, context)),
			     0);
		// Counter shows operation has completed.
		wait_cntr_params.ep_idx = 0;
		wait_cntr_params.val = 2;
		wait_cntr_params.which = WAIT_CNTR_TX;
		TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));

		//...but there should be no CQ entry generated.
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.expect_empty = true;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		// Do a send with FI_COMPLETION.
		struct iovec iov = {
			.iov_base = ri->mr_info[0].uaddr,
			.iov_len = buffer_len
		};
		struct fi_msg msg = {
			.msg_iov = &iov,
			.desc = NULL,
			.iov_count = 1,
			.addr = pri->ep_info[0].fi_addr,
			.context = get_ctx_simple(ri, context),
			.data = 0,
		};
		INSIST_FI_EQ(
			ri, fi_sendmsg(ri->ep_info[0].fid, &msg, FI_COMPLETION),
			0);
		// This one should generate a completion.
		memset(&wait_tx_cq_params, 0, sizeof(wait_tx_cq_params));
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_MSG | FI_SEND;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		// Do a write with FI_COMPLETION.
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
					 FI_COMPLETION),
			     0);
		// This one should generate a completion.
		memset(&wait_tx_cq_params, 0, sizeof(wait_tx_cq_params));
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_RMA | FI_WRITE;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	} else {
		INSIST_FI_EQ(ri,
			     fi_recv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, FI_ADDR_UNSPEC,
				     get_ctx_simple(ri, context)),
			     0);
		wait_cntr_params.ep_idx = 0;
		wait_cntr_params.val = 1;
		wait_cntr_params.which = WAIT_CNTR_RX;
		TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));

		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.expect_empty = true;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		// Do a recv with FI_COMPLETION.
		struct iovec iov = {
			.iov_base = ri->mr_info[0].uaddr,
			.iov_len = buffer_len
		};
		struct fi_msg msg = {
			.msg_iov = &iov,
			.desc = NULL,
			.iov_count = 1,
			.addr = FI_ADDR_UNSPEC,
			.context = get_ctx_simple(ri, context),
			.data = 0,
		};
		INSIST_FI_EQ(
			ri, fi_recvmsg(ri->ep_info[0].fid, &msg, FI_COMPLETION),
			0);
		// This one should generate a completion.
		memset(&wait_rx_cq_params, 0, sizeof(wait_rx_cq_params));
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_MSG | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));
	}

	TRACE(ri, util_teardown(ri, pri));
	return 0;
}

int run_selective_completion2(struct rank_info *ri)
{
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_cntr_params wait_cntr_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 4096;
	uint64_t cntr_before;
	struct rank_info *pri = NULL;

	TRACE(ri, util_init(ri));

	if (my_node == NODE_A) {
		mr_params.access = FI_SEND | FI_WRITE;
		mr_params.seed = seed_node_a;
	} else {
		mr_params.access = FI_RECV | FI_REMOTE_WRITE;
		mr_params.seed = seed_node_b;
	}
	mr_params.idx = 0;
	mr_params.length = buffer_len;
	TRACE(ri, util_create_mr(ri, &mr_params));

	ep_params.idx = 0;
	// Here's the main bit for this test: bind CQ with selective completion
	// but then bind the contexts with FI_COMPLETION.
	ep_params.cq_bind_flags = FI_SELECTIVE_COMPLETION;
	ep_params.tx_attr.op_flags = FI_COMPLETION;
	ep_params.rx_attr.op_flags = FI_COMPLETION;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_send(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, pri->ep_info[0].fi_addr,
				     get_ctx_simple(ri, context)),
			     0);

		// Completion present due to bind flags.
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_MSG | FI_SEND;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		// Issue a write.
		INSIST_FI_EQ(ri,
			     fi_write(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				      buffer_len, NULL, pri->ep_info[0].fi_addr,
				      (uint64_t)pri->mr_info[0].uaddr,
				      pri->mr_info[0].key,
				      get_ctx_simple(ri, context)),
			     0);
		// Completion present due to bind flags.
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_RMA | FI_WRITE;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		// Do a send with FI_COMPLETION.
		struct iovec iov = {
			.iov_base = ri->mr_info[0].uaddr,
			.iov_len = buffer_len
		};
		struct fi_msg msg = {
			.msg_iov = &iov,
			.desc = NULL,
			.iov_count = 1,
			.addr = pri->ep_info[0].fi_addr,
			.context = get_ctx_simple(ri, context),
			.data = 0,
		};
		INSIST_FI_EQ(
			ri, fi_sendmsg(ri->ep_info[0].fid, &msg, FI_COMPLETION),
			0);
		// Completion present due to bind flags.
		memset(&wait_tx_cq_params, 0, sizeof(wait_tx_cq_params));
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_MSG | FI_SEND;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		// Do a write with FI_COMPLETION.
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
					 FI_COMPLETION),
			     0);
		// Completion present due to bind flags.
		memset(&wait_tx_cq_params, 0, sizeof(wait_tx_cq_params));
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_RMA | FI_WRITE;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		TRACE(ri, (cntr_before = util_read_tx_cntr(ri, 0)));
		// Do a send with explicitly 0 flags.
		INSIST_FI_EQ(ri, fi_sendmsg(ri->ep_info[0].fid, &msg, 0), 0);
		// Counter shows operation has completed.
		wait_cntr_params.ep_idx = 0;
		wait_cntr_params.val = cntr_before + 1;
		wait_cntr_params.which = WAIT_CNTR_TX;
		TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));
		// ... but 0 flags should prevent completion.
		memset(&wait_tx_cq_params, 0, sizeof(wait_tx_cq_params));
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.expect_empty = true;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	} else {
		INSIST_FI_EQ(ri,
			     fi_recv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, FI_ADDR_UNSPEC,
				     get_ctx_simple(ri, context)),
			     0);
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_MSG | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		// Do a recv with FI_COMPLETION.
		struct iovec iov = {
			.iov_base = ri->mr_info[0].uaddr,
			.iov_len = buffer_len
		};
		struct fi_msg msg = {
			.msg_iov = &iov,
			.desc = NULL,
			.iov_count = 1,
			.addr = FI_ADDR_UNSPEC,
			.context = get_ctx_simple(ri, context),
			.data = 0,
		};
		INSIST_FI_EQ(
			ri, fi_recvmsg(ri->ep_info[0].fid, &msg, FI_COMPLETION),
			0);
		// Completion present due to bind flags.
		memset(&wait_rx_cq_params, 0, sizeof(wait_rx_cq_params));
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_MSG | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		TRACE(ri, (cntr_before = util_read_rx_cntr(ri, 0)));
		// Do a recv with explicitly 0 flags.
		INSIST_FI_EQ(ri, fi_recvmsg(ri->ep_info[0].fid, &msg, 0), 0);
		// Counter shows operation has completed.
		wait_cntr_params.ep_idx = 0;
		wait_cntr_params.val = cntr_before + 1;
		wait_cntr_params.which = WAIT_CNTR_RX;
		TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));
		// ... but 0 flags should prevent completion.
		memset(&wait_rx_cq_params, 0, sizeof(wait_rx_cq_params));
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.expect_empty = true;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));
	}

	TRACE(ri, util_teardown(ri, pri));
	return 0;
}

int run_selective_completion_error(struct rank_info *ri)
{
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_cntr_params wait_cntr_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 2048;
	struct rank_info *pri = NULL;

	TRACE(ri, util_init(ri));

	if (my_node == NODE_A) {
		mr_params.access = FI_SEND | FI_WRITE;
		mr_params.seed = seed_node_a;
	} else {
		mr_params.access = FI_RECV | FI_REMOTE_WRITE;
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
		// Post a send for the full buffer size. Recver will have a
		// half size buffer, so this should fail.
		INSIST_FI_EQ(ri,
			     fi_send(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, pri->ep_info[0].fi_addr,
				     get_ctx_simple(ri, context)),
			     0);
		// Cntr shows operation has completed.
		wait_cntr_params.ep_idx = 0;
		wait_cntr_params.val = 1;
		wait_cntr_params.which = WAIT_CNTR_TX;
		TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));
		// But no CQ completion.
		//
		// XXX: For now, we drop oversized messages. It's also allowed,
		// however, to truncate, in which case we'd get an error entry
		// here with FI_ETRUNC.
		//wait_tx_cq_params.expect_error = true;
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.expect_empty = true;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		// Issue an oversized write (2x buffer size). This should fail.
		INSIST_FI_EQ(ri,
			     fi_write(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				      buffer_len * 2, NULL,
				      pri->ep_info[0].fi_addr,
				      (uint64_t)pri->mr_info[0].uaddr,
				      pri->mr_info[0].key,
				      get_ctx_simple(ri, context)),
			     0);
		// This should generate an error completion.
		memset(&wait_tx_cq_params, 0, sizeof(wait_tx_cq_params));
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.expect_error = true;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	} else {
		// Post a recv for 1/2 size buffer. Sender will send full size
		// message, which should fail.
		INSIST_FI_EQ(ri,
			     fi_recv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len / 2, NULL, FI_ADDR_UNSPEC,
				     get_ctx_simple(ri, context)),
			     0);
		// We should get a "message too long" error.
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.expect_error = true;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));
	}

	TRACE(ri, util_teardown(ri, pri));
	return 0;
}

int run_rsrc_mgmt_cq_overrun(struct rank_info *ri)
{
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_cntr_params wait_cntr_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 4096;
	struct rank_info *pri = NULL;

	TRACE(ri, util_init(ri));

	mr_params.idx = 0;
	mr_params.length = buffer_len;
	if (my_node == NODE_A) {
		mr_params.access = FI_SEND;
		mr_params.seed = seed_node_a;
	} else {
		mr_params.access = FI_RECV;
		mr_params.seed = seed_node_b;
	}
	TRACE(ri, util_create_mr(ri, &mr_params));

	ep_params.idx = 0;
	// Key to this test: only 4 entries in the CQs.
	ep_params.cq_size = 4;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	// Do 4 sends (filling up all CQ slots with completions).
	for (int i = 0; i < 4; i++) {
		if (my_node == NODE_A) {
			INSIST_FI_EQ(ri,
				     fi_send(ri->ep_info[0].fid,
					     ri->mr_info[0].uaddr, buffer_len,
					     NULL, pri->ep_info[0].fi_addr,
					     get_ctx_simple(ri, context)),
				     0);
		} else {
			INSIST_FI_EQ(ri,
				     fi_recv(ri->ep_info[0].fid,
					     ri->mr_info[0].uaddr, buffer_len,
					     NULL, FI_ADDR_UNSPEC,
					     get_ctx_simple(ri, context)),
				     0);
		}
	}
	wait_cntr_params.ep_idx = 0;
	wait_cntr_params.val = 4;
	if (my_node == NODE_A) {
		wait_cntr_params.which = WAIT_CNTR_TX;
	} else {
		wait_cntr_params.which = WAIT_CNTR_RX;
	}
	TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));

	// Try to send/recv a 5th msg. This should EAGAIN, since there is no
	// room in the CQ.
	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_send(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, pri->ep_info[0].fi_addr,
				     get_ctx_simple(ri, context)),
			     -FI_EAGAIN);
	} else {
		INSIST_FI_EQ(ri,
			     fi_recv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, FI_ADDR_UNSPEC,
				     get_ctx_simple(ri, context)),
			     -FI_EAGAIN);
	}

	// Reap a CQ entry to make some room.
	if (my_node == NODE_A) {
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_MSG | FI_SEND;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	} else {
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_MSG | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));
	}

	// Retry send/recv. This should go through now.
	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_send(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, pri->ep_info[0].fi_addr,
				     get_ctx_simple(ri, context)),
			     0);
	} else {
		INSIST_FI_EQ(ri,
			     fi_recv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, FI_ADDR_UNSPEC,
				     get_ctx_simple(ri, context)),
			     0);
	}

	// Reap the remaining CQ entries.
	for (int i = 0; i < 4; i++) {
		if (my_node == NODE_A) {
			wait_tx_cq_params.ep_idx = 0;
			wait_tx_cq_params.context_val = context;
			wait_tx_cq_params.flags = FI_MSG | FI_SEND;
			TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
		} else {
			wait_rx_cq_params.ep_idx = 0;
			wait_rx_cq_params.context_val = context;
			wait_rx_cq_params.flags = FI_MSG | FI_RECV;
			TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));
		}
	}

	TRACE(ri, util_teardown(ri, pri));
	return 0;
}

static int msg_auto_reg_mr_common(struct rank_info *ri, size_t buffer_len)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	struct rank_info *pri = NULL;

	TRACE(ri, util_init(ri));

	mr_params.idx = 0;
	mr_params.length = buffer_len;
	// Key to the test: create the buffers but don't register them with
	// libfabric.
	mr_params.skip_reg = true;
	if (my_node == NODE_A) {
		mr_params.access = FI_SEND;
		mr_params.seed = seed_node_a;
	} else {
		mr_params.access = FI_RECV;
		mr_params.seed = seed_node_b;
	}
	TRACE(ri, util_create_mr(ri, &mr_params));

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	if (my_node == NODE_B) {
		INSIST_FI_EQ(ri,
			     fi_recv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, FI_ADDR_UNSPEC,
				     get_ctx_simple(ri, context)),
			     0);
	}

	TRACE(ri, util_barrier(ri));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_send(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, pri->ep_info[0].fi_addr,
				     get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_MSG | FI_SEND;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	}

	if (my_node == NODE_B) {
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_MSG | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_msg_auto_reg_mr(struct rank_info *ri)
{
	return msg_auto_reg_mr_common(ri, 4096);
}

int run_small_msg_auto_reg_mr(struct rank_info *ri)
{
	return msg_auto_reg_mr_common(ri, 64);
}

int run_zero_length(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buffer_len = 0; // Main part of this test: buffers are 0 length.
	struct rank_info *pri = NULL;

	TRACE(ri, util_init(ri));

	mr_params.length = buffer_len;
	for (int i = 0; i < 2; i++) {
		if (my_node == NODE_A) {
			mr_params.access = FI_SEND;
			mr_params.seed = seed_node_a + i;
		} else {
			mr_params.access = FI_RECV;
			mr_params.seed = seed_node_b + i;
		}
		mr_params.idx = i;
		if (i == 1) {
			// Second MR will be generated.
			mr_params.skip_reg = true;
		}
		TRACE(ri, util_create_mr(ri, &mr_params));
	}

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	// Do the zero-length send/recvs (one with registered MRs, one
	// without).
	for (int i = 0; i < 2; i++) {
		if (my_node == NODE_A) {
			INSIST_FI_EQ(ri,
				     fi_send(ri->ep_info[0].fid,
					     ri->mr_info[i].uaddr, buffer_len,
					     NULL, pri->ep_info[0].fi_addr,
					     get_ctx_simple(ri, context)),
				     0);

			wait_tx_cq_params.ep_idx = 0;
			wait_tx_cq_params.context_val = context;
			wait_tx_cq_params.flags = FI_MSG | FI_SEND;
			TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
		} else {
			INSIST_FI_EQ(ri,
				     fi_recv(ri->ep_info[0].fid,
					     ri->mr_info[i].uaddr, buffer_len,
					     NULL, FI_ADDR_UNSPEC,
					     get_ctx_simple(ri, context)),
				     0);

			wait_rx_cq_params.ep_idx = 0;
			wait_rx_cq_params.context_val = context;
			wait_rx_cq_params.flags = FI_MSG | FI_RECV;
			TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

			verify_buf_params.mr_idx = i;
			verify_buf_params.length = buffer_len;
			verify_buf_params.expected_seed = seed_node_a + i;
			TRACE(ri, util_verify_buf(ri, &verify_buf_params));
		}
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

static int loopback_msg_common(struct rank_info *ri, size_t buffer_len)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct wait_cntr_params wait_cntr_params[2] = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const uint64_t contexts[] = { context, context + 1 };

	TRACE(ri, util_init(ri));

	// This is a loopback test on NODE_A only. NODE_B gets to take 5.
	if (my_node == NODE_A) {
		for (int i = 0; i < 2; i++) {
			mr_params.idx = i;
			mr_params.length = buffer_len;
			mr_params.access = FI_SEND | FI_RECV;
			mr_params.skip_reg = true;
			mr_params.seed = seed_node_a + i;
			TRACE(ri, util_create_mr(ri, &mr_params));

			ep_params.idx = i;
			TRACE(ri, util_create_ep(ri, &ep_params));
		}

		TRACE(ri, util_av_insert_all(ri, ri));

		INSIST_FI_EQ(ri,
			     fi_recv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, FI_ADDR_UNSPEC,
				     get_ctx_simple(ri, contexts[0])),
			     0);
		INSIST_FI_EQ(ri,
			     fi_send(ri->ep_info[1].fid, ri->mr_info[1].uaddr,
				     buffer_len, NULL, ri->ep_info[0].fi_addr,
				     get_ctx_simple(ri, contexts[1])),
			     0);

		// Both endpoints will need to make progress for the above
		// send/recv to complete. So we must wait on counters from both
		// endpoint.
		wait_cntr_params[0].ep_idx = 0;
		wait_cntr_params[0].val = 1;
		wait_cntr_params[0].which = WAIT_CNTR_RX;
		wait_cntr_params[1].ep_idx = 1;
		wait_cntr_params[1].val = 1;
		wait_cntr_params[1].which = WAIT_CNTR_TX;
		TRACE(ri, util_wait_cntr_many(ri, wait_cntr_params, 2));

		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = contexts[0];
		wait_rx_cq_params.flags = FI_MSG | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		wait_tx_cq_params.ep_idx = 1;
		wait_tx_cq_params.context_val = contexts[1];
		wait_tx_cq_params.flags = FI_MSG | FI_SEND;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		verify_buf_params.mr_idx = 1;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a + 1;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, NULL));

	return 0;
}

int run_loopback_msg(struct rank_info *ri)
{
	return loopback_msg_common(ri, 4096);
}

int run_loopback_small_msg(struct rank_info *ri)
{
	return loopback_msg_common(ri, 32);
}

int run_cq_sread(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	const size_t buffer_len = 4096;
	struct rank_info *pri = NULL;

	TRACE(ri, util_simple_setup(ri, &pri, buffer_len, FI_SEND, FI_RECV));

	if (my_node == NODE_B) {
		INSIST_FI_EQ(ri,
			     fi_recv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, FI_ADDR_UNSPEC,
				     get_ctx_simple(ri, context)),
			     0);
	}

	// We sync here to ensure the recv side has setup its buffer before we
	// send. We don't want an unexpected msg here (that's a separate test).
	TRACE(ri, util_barrier(ri));

	const uint64_t cq_timeout_ms = 2000;
	if (my_node == NODE_A) {
		TRACE(ri, sleep((cq_timeout_ms * 2) / 1000));

		INSIST_FI_EQ(ri,
			     fi_send(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, pri->ep_info[0].fi_addr,
				     get_ctx_simple(ri, context)),
			     0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_MSG | FI_SEND;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	} else {
		struct fid_cq *cq = ri->ep_info[0].rx_cq_fid;
		struct fi_cq_tagged_entry entry;
		struct timespec start, end;
		util_get_time(&start);
		// Our timeout of 2s should be hit before the other side is done sleeping.
		INSIST_FI_EQ(ri, fi_cq_sread(cq, &entry, 1, NULL, cq_timeout_ms), -FI_EAGAIN);
		util_get_time(&end);
		uint64_t delta_ms = util_time_delta_ms(&start, &end);
		if (delta_ms < cq_timeout_ms * 0.75 || delta_ms > cq_timeout_ms * 1.25) {
			ERRORX(ri, "fi_cq_sread took %lu ms, expected %lu\n",
			       delta_ms, cq_timeout_ms);
		}

		INSIST_FI_EQ(ri, fi_cq_sread(cq, &entry, 1, NULL, cq_timeout_ms * 2), 1);
		util_validate_cq_entry(ri, &entry, NULL, FI_MSG | FI_RECV, 0,
				       context, false, 0);

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}
