/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"
#include "ofi_util.h"
#include "efa_rdm_ep.h"

#define MSG_SIZE 10

void test_efa_rdm_msg_send_to_local_peer_with_null_desc(struct efa_resource **state)
{
        struct efa_resource *resource = *state;
        char buf[MSG_SIZE];
        int i;
        struct iovec iov;
        struct efa_ep_addr raw_addr;
	size_t raw_addr_len = sizeof(raw_addr);
        fi_addr_t addr;
        int ret;
        struct fi_msg msg = {0};
        struct fi_msg_tagged tmsg = {0};

        efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

        ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);

	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0 /* flags */, NULL /* context */);
	assert_int_equal(ret, 1);

        for (i = 0; i < MSG_SIZE; i++)
                buf[i] = 'a' + i;

        iov.iov_base = buf;
        iov.iov_len = MSG_SIZE;

        efa_unit_test_construct_msg(&msg, &iov, 1, addr, NULL, 0, NULL);

        efa_unit_test_construct_tmsg(&tmsg, &iov, 1, addr, NULL, 0, NULL, 0, 0);

        /* The peer won't be verified by shm so it is expected that EAGAIN will be returned */
        ret = fi_send(resource->ep, buf, MSG_SIZE, NULL, addr, NULL);
        assert_int_equal(ret, -FI_EAGAIN);

        ret = fi_sendv(resource->ep, &iov, NULL, 1, addr, NULL);
        assert_int_equal(ret, -FI_EAGAIN);

        ret = fi_senddata(resource->ep, buf, MSG_SIZE, NULL, 0, addr, NULL);
        assert_int_equal(ret, -FI_EAGAIN);

        ret = fi_sendmsg(resource->ep, &msg, 0);
        assert_int_equal(ret, -FI_EAGAIN);

        ret = fi_tsend(resource->ep, buf, MSG_SIZE, NULL, addr, 0, NULL);
        assert_int_equal(ret, -FI_EAGAIN);

        ret = fi_tsendv(resource->ep, &iov, NULL, 1, addr, 0, NULL);
        assert_int_equal(ret, -FI_EAGAIN);

        ret = fi_tsenddata(resource->ep, buf, MSG_SIZE, NULL, 0, addr, 0, NULL);
        assert_int_equal(ret, -FI_EAGAIN);

        ret = fi_tsendmsg(resource->ep, &tmsg, 0);
        assert_int_equal(ret, -FI_EAGAIN);
}

/**
 * @brief Test that a partial multi-packet send failure leaves nothing in-flight
 *
 * Unlike RDMA write/read — which post segments individually and ring the
 * doorbell (ibv_wr_complete) on each one — the send path batches all
 * packets via efa_rdm_pke_sendv() using FI_MORE + a single final
 * ibv_wr_complete. Per-pkt efa_qp_post_send calls with FI_MORE only
 * stage work requests in the qp; only the last call (without FI_MORE)
 * actually submits the whole batch, atomically. As a consequence,
 * efa_rdm_pke_sendv() defers every efa_rdm_ep_record_tx_op_submitted()
 * call to a single post-loop pass that only runs when the batch fully
 * succeeds.
 *
 * With mtu=8k and a 16k message, efa_rdm_pke_sendv() is invoked with
 * exactly two packets. This test mocks call #1 (FI_MORE, ibv_wr_send
 * staging) to succeed and call #2 (ibv_wr_complete doorbell) to return
 * ENOMEM — the exact partial-post scenario. It asserts that neither
 * packet is tracked as in-flight and the txe is cleanly released.
 *
 * If a future refactor moved efa_rdm_ep_record_tx_op_submitted() into
 * the per-pkt loop (the shape that makes RDMA write/read vulnerable to
 * the partial-post double-free), call #1's "success" would record its
 * packet as in-flight and this test would fail.
 */
void test_efa_rdm_msg_send_multi_pkt_sendv_fail_no_inflight(
		struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_unit_test_buff send_buff;
	fi_addr_t addr;
	struct efa_ep_addr raw_addr;
	size_t raw_addr_len = sizeof(raw_addr);
	size_t ep_tx_ops_before, peer_tx_ops_before;
	int peer_outstanding_pkts_before;
	void *desc;
	int ret;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	/* Set up peer */
	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL);
	assert_int_equal(ret, 1);

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	/*
	 * Create a buffer larger than eager max. Use 16k to force a
	 * multi-packet send path (medium or longcts RTM).
	 */
	efa_unit_test_buff_construct(&send_buff, resource, 16384);
	desc = fi_mr_desc(send_buff.mr);

	/*
	 * Mock the partial-post scenario (mtu=8k, msg=16k -> exactly 2 pkts):
	 *   call #1 (FI_MORE, ibv_wr_send staging):      returns 0
	 *   call #2 (final, ibv_wr_complete doorbell):   returns ENOMEM
	 * efa_rdm_pke_sendv() must break on the call-#2 failure and skip the
	 * post-loop efa_rdm_ep_record_tx_op_submitted() pass entirely, so
	 * neither packet ends up tracked as in-flight.
	 */
	g_efa_unit_test_mocks.efa_qp_post_send = &efa_mock_efa_qp_post_send_return_mock;
	will_return(efa_mock_efa_qp_post_send_return_mock, 0);
	will_return(efa_mock_efa_qp_post_send_return_mock, ENOMEM);

	/* Capture baselines right before fi_send to ignore any setup state. */
	ep_tx_ops_before = efa_rdm_ep->efa_outstanding_tx_ops;
	peer_tx_ops_before = peer->efa_outstanding_tx_ops;
	peer_outstanding_pkts_before = efa_unit_test_get_dlist_length(&peer->outstanding_tx_pkts);

	ret = fi_send(resource->ep, send_buff.buff, send_buff.size, desc,
		      addr, NULL);
	assert_int_equal(ret, -FI_EAGAIN);

	/*
	 * Key regression invariants: nothing in the batch became in-flight,
	 * even though at least one earlier call "succeeded" (returned 0 from
	 * the FI_MORE staging path). If efa_rdm_pke_sendv() ever ran
	 * efa_rdm_ep_record_tx_op_submitted() without waiting for the whole
	 * batch to succeed, these counters would advance. (Note: we don't
	 * check g_ibv_submitted_wr_id_cnt because the mock records the wr_id
	 * on every invocation regardless of its return value; it tracks mock
	 * calls, not successful submissions.)
	 */
	assert_int_equal(efa_rdm_ep->efa_outstanding_tx_ops, ep_tx_ops_before);
	assert_int_equal(peer->efa_outstanding_tx_ops, peer_tx_ops_before);
	assert_int_equal(efa_unit_test_get_dlist_length(&peer->outstanding_tx_pkts),
			 peer_outstanding_pkts_before);
	/* txe was released by the caller (safe: no segments were committed). */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 0);

	efa_unit_test_buff_destruct(&send_buff);
}
