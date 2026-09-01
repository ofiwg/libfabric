/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_unit_tests.h"
#include "rdm/efa_rdm_pke_rtm.h"
#include "rdm/efa_rdm_proto.h"
#include "rdm/efa_rdm_proto_eager.h"
#include "rdm/efa_rdm_proto_medium.h"

/*
 * A size that is too large for a single eager packet (the device MTU is a few
 * KB) but well within the medium threshold (64KB for system memory), so the
 * medium protocol is selected and has to split the message into several
 * packets.
 */
#define EFA_UNIT_TEST_PROTO_MEDIUM_LEN 16384

/*
 * Upper bound for the per-test arrays that snapshot a medium message's packet
 * entries. The medium protocol needs ceil(len / (mtu - hdr)) of them, which is
 * a handful for the size above on any EFA device; the tests assert the fit
 * rather than trusting it.
 */
#define EFA_UNIT_TEST_PROTO_MAX_PKES 64

/* Tests from efa_unit_test_proto_select.c */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */


/**
 * @brief Helper to set up an endpoint, peer, and TXE for protocol selection
 * tests.
 *
 * Returns the efa_rdm_ep. Caller must provide a peer_addr output and a
 * pre-allocated txe pointer output.
 */
static struct efa_rdm_ep *setup_proto_select_test(struct efa_resource *resource,
						  fi_addr_t *peer_addr)
{
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_rdm_ep *ep;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, peer_addr, 0, NULL),
		1);

	return ep;
}

/**
 * @brief Test that eager protocol is selected for small messages.
 */
void test_proto_select_eager_for_small_msg(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	fi_addr_t peer_addr;
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	ep = setup_proto_select_test(resource, &peer_addr);
	peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	iov.iov_base = NULL;
	iov.iov_len = 64; /* Small message, fits in eager */
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0, NULL);

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	assert_int_equal(err, 0);
	assert_non_null(proto);
	assert_ptr_equal(proto, &efa_rdm_proto_eager);

	ofi_buf_free(txe);
}

/**
 * @brief Test that zero-length messages select eager protocol.
 */
void test_proto_select_eager_for_zero_len_msg(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	fi_addr_t peer_addr;
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	ep = setup_proto_select_test(resource, &peer_addr);
	peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	iov.iov_base = NULL;
	iov.iov_len = 0;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0, NULL);

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	assert_int_equal(err, 0);
	assert_ptr_equal(proto, &efa_rdm_proto_eager);

	ofi_buf_free(txe);
}

/**
 * @brief Test that eager construct_tx_pkes produces exactly 1 PKE with
 * the correct callback set.
 */
void test_proto_eager_construct_pkes_single_pke(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_unit_test_buff_construct(&send_buff, resource, 64);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(txe);

	/* Fill TXE as generic_send would before calling construct_tx_pkes */
	txe->ep = ep;
	efa_rdm_proto_txe_fill(txe, ep, peer, &msg, ofi_op_msg, 0, 0, 0,
			       &efa_rdm_proto_eager);
	txe->msg_id = peer->next_msg_id++;

	err = efa_rdm_proto_eager.construct_tx_pkes(ep, peer, &msg, ofi_op_msg,
						    0, 0, 0, txe);
	assert_int_equal(err, 0);
	assert_int_equal(ep->send_pkt_entry_vec_size, 1);
	assert_non_null(ep->send_pkt_entry_vec[0]);
	assert_non_null(ep->send_pkt_entry_vec[0]->handle_pke);
	assert_ptr_equal(ep->send_pkt_entry_vec[0]->ope, txe);

	/* Clean up */
	efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[0]);
	efa_rdm_txe_release(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Test that eager send completion callback releases TXE and PKE
 * for non-DC messages.
 */
void test_proto_eager_send_completion_releases_txe(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *pkt_entry;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	int err;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_unit_test_buff_construct(&send_buff, resource, 64);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	/* Mock efa_qp_post_send to succeed */
	g_efa_unit_test_mocks.efa_qp_post_send =
		&efa_mock_efa_qp_post_send_return_mock;
	will_return_int_maybe(efa_mock_efa_qp_post_send_return_mock, 0);

	/* Send a message via fi_send which goes through the new code path */
	err = fi_send(resource->ep, send_buff.buff, send_buff.size,
		      fi_mr_desc(send_buff.mr), peer_addr, NULL);
	assert_int_equal(err, 0);
	assert_int_equal(efa_unit_test_get_ope_list_length(ep, EFA_RDM_TXE), 1);

	/* Get the TXE and PKE */
	pkt_entry = ep->send_pkt_entry_vec[0];
	assert_non_null(pkt_entry->handle_pke);

	/* Simulate send completion: record_tx_op_completed + callback */
	efa_rdm_ep_record_tx_op_completed(ep, pkt_entry);
	pkt_entry->handle_pke(pkt_entry);

	/* TXE should be released */
	assert_int_equal(efa_unit_test_get_ope_list_length(ep, EFA_RDM_TXE), 0);

	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Test that eager assigns msg_id from peer->next_msg_id.
 */
void test_proto_eager_assigns_msg_id(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	uint32_t initial_msg_id;
	int err;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_unit_test_buff_construct(&send_buff, resource, 64);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	initial_msg_id = peer->next_msg_id;

	g_efa_unit_test_mocks.efa_qp_post_send =
		&efa_mock_efa_qp_post_send_return_mock;
	will_return_int_maybe(efa_mock_efa_qp_post_send_return_mock, 0);

	err = fi_send(resource->ep, send_buff.buff, send_buff.size,
		      fi_mr_desc(send_buff.mr), peer_addr, NULL);
	assert_int_equal(err, 0);

	/* msg_id should have been assigned and next_msg_id incremented */
	struct efa_rdm_ope *txe =
		efa_unit_test_get_first_ope(ep, EFA_RDM_TXE);
	assert_int_equal(txe->msg_id, initial_msg_id);
	assert_int_equal(peer->next_msg_id, initial_msg_id + 1);

	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Test that eager construct_tx_pkes produces a headerless PKE with
 * EFA_RDM_PKE_SEND_TO_USER_RECV_QP flag when the peer expects zero-copy
 * (headerless) data transfer.
 */
void test_proto_eager_construct_pkes_zero_copy(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_unit_test_buff_construct(&send_buff, resource, 64);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	/* Mark peer as expecting zero-copy transfer */
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_REQUEST_USER_RECV_QP;
	peer->user_recv_qp.qpn = 99;
	peer->user_recv_qp.qkey = 0xABCD;

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(txe);
	txe->ep = ep;
	efa_rdm_proto_txe_fill(txe, ep, peer, &msg, ofi_op_msg, 0, 0, 0,
			       &efa_rdm_proto_eager);
	txe->msg_id = peer->next_msg_id++;

	err = efa_rdm_proto_eager.construct_tx_pkes(ep, peer, &msg, ofi_op_msg,
						    0, 0, 0, txe);
	assert_int_equal(err, 0);
	assert_int_equal(ep->send_pkt_entry_vec_size, 1);

	/* Verify headerless packet properties */
	struct efa_rdm_pke *pke = ep->send_pkt_entry_vec[0];
	assert_true(pke->flags & EFA_RDM_PKE_SEND_TO_USER_RECV_QP);
	assert_true(pke->flags & EFA_RDM_PKE_HAS_NO_BASE_HDR);
	assert_int_equal(pke->pkt_size, 64);
	assert_int_equal(pke->payload_size, 64);

	ofi_buf_free(pke);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Test that a send is queued before handshake and dequeued after
 * handshake completes when the peer may have zero-copy mode enabled.
 */
void test_proto_eager_queue_dequeue_handshake(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct fi_cq_tagged_entry cq_entry;
	int ret;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_unit_test_buff_construct(&send_buff, resource, 64);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	peer->flags &= ~EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	ep->peer_may_have_zcpy_rx = true;

	/*
	 * The handshake trigger and the queued send's repost each post at
	 * least once, and the progress engine may retry, so let every post
	 * succeed rather than queueing a fixed number of return values. The
	 * mock reads its value with mock_int(), so it needs the int variant.
	 */
	g_efa_unit_test_mocks.efa_qp_post_send = &efa_mock_efa_qp_post_send_return_mock;
	will_return_int_always(efa_mock_efa_qp_post_send_return_mock, 0);

	ret = fi_send(resource->ep, send_buff.buff, send_buff.size,
		      fi_mr_desc(send_buff.mr), peer_addr, NULL);
	assert_int_equal(ret, 0);

	/* Verify the OPE is in the queued list */
	assert_int_equal(ep->ope_queued_before_handshake_cnt, 1);
	txe = container_of(ep->ope_queued_list.next,
			   struct efa_rdm_ope, queued_entry);
	assert_true(dlist_entry_in_list(&txe->queued_entry,
					&ep->ope_queued_list));

	/* Simulate handshake received */
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	/* Progress via fi_cq_read which calls efa_domain_progress */
	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, -FI_EAGAIN);

	/* Verify the OPE was dequeued and sent */
	assert_int_equal(ep->ope_queued_before_handshake_cnt, 0);
	assert_true(dlist_empty(&ep->ope_queued_list));

	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief An eager txe queued before the handshake survives the MR generation
 *        check when it is reposted.
 *
 * efa_rdm_ope_process_queued_ope() gates the repost on
 * efa_rdm_mr_gen_check_ope(), which asserts the dispatch-time snapshot in
 * ope->desc_gen[] was initialized and compares it against the live MR
 * generation. The refactored TXE setup must take that snapshot, exactly as
 * efa_rdm_txe_construct() does; otherwise the check reads a stale value from
 * the recycled ope pool slot and cancels a healthy transfer with a spurious
 * FI_EFA_ERR_PEER_ABORTED.
 */
void test_proto_eager_queued_before_handshake_survives_mr_gen_check(
	void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct fi_cq_tagged_entry cq_entry;
	struct fi_cq_err_entry err_entry;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	unsigned int i;
	int ret;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_unit_test_buff_construct(&send_buff, resource, 64);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	peer->flags &= ~EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	ep->peer_may_have_zcpy_rx = true;

	/* Let every post succeed; see test_proto_eager_queue_dequeue_handshake. */
	g_efa_unit_test_mocks.efa_qp_post_send =
		&efa_mock_efa_qp_post_send_return_mock;
	will_return_int_always(efa_mock_efa_qp_post_send_return_mock, 0);

	/* The send is queued because the handshake has not arrived yet. */
	ret = fi_send(resource->ep, send_buff.buff, send_buff.size,
		      fi_mr_desc(send_buff.mr), peer_addr, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(ep->ope_queued_before_handshake_cnt, 1);

	txe = container_of(ep->ope_queued_list.next, struct efa_rdm_ope,
			   queued_entry);

	/*
	 * The MR generation snapshot must be populated, and must match the
	 * live generation of every source MR -- the transfer is healthy.
	 */
	for (i = 0; i < txe->iov_count; i++) {
		if (!txe->desc[i])
			break;
		assert_true(efa_rdm_mr_gen_value_is_valid(txe->desc_gen[i]));
	}
	assert_true(efa_rdm_mr_gen_check_ope(txe));

	/* Handshake arrives; the queued send is reposted, not canceled. */
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, -FI_EAGAIN);

	assert_int_equal(ep->ope_queued_before_handshake_cnt, 0);
	assert_true(dlist_empty(&ep->ope_queued_list));

	/* No error completion: the repost must not be mistaken for an abort. */
	memset(&err_entry, 0, sizeof(err_entry));
	assert_int_equal(fi_cq_readerr(resource->cq, &err_entry, 0), -FI_EAGAIN);

	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief A failure inside the eager construct_tx_pkes rolls peer->next_msg_id
 *        back by exactly one.
 *
 * The txe is owned by efa_rdm_msg_generic_send(), which releases it and rolls
 * back the msg_id when posting fails. construct_tx_pkes() must therefore not
 * do the same cleanup itself: doing so releases a pooled txe twice and
 * decrements next_msg_id twice, desynchronizing the per-peer message sequence
 * the receiver's reorder window depends on.
 */
void test_proto_eager_construct_pkes_failure_rolls_back_msg_id(
	void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	uint32_t initial_msg_id;
	int ret;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_unit_test_buff_construct(&send_buff, resource, 64);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	initial_msg_id = peer->next_msg_id;

	/* Fail the device post so the send path takes its error branch. */
	g_efa_unit_test_mocks.efa_qp_post_send =
		&efa_mock_efa_qp_post_send_return_mock;
	will_return_int_maybe(efa_mock_efa_qp_post_send_return_mock, -FI_ENOMEM);

	ret = fi_send(resource->ep, send_buff.buff, send_buff.size,
		      fi_mr_desc(send_buff.mr), peer_addr, NULL);
	assert_int_not_equal(ret, 0);

	/* Rolled back exactly once, and the txe was released exactly once. */
	assert_int_equal(peer->next_msg_id, initial_msg_id);
	assert_int_equal(efa_unit_test_get_ope_list_length(ep, EFA_RDM_TXE), 0);

	efa_unit_test_buff_destruct(&send_buff);
}

/*
 * Simulate the source MR being closed mid-transfer (its generation bumped)
 * so efa_rdm_mr_gen_check_ope() reports the MR was canceled -- the
 * precondition the sender peer-abort path
 * (efa_rdm_txe_mark_peer_abort_if_needed) requires before it will mark the
 * txe and drive the PEER_ERROR emit/drain.
 */
static void efa_unit_test_proto_simulate_source_mr_canceled(struct efa_rdm_ope *txe)
{
	static struct efa_rdm_mr stale_source_mr;

	stale_source_mr.gen = 1;	/* current MR generation */
	txe->iov_count = 1;
	txe->desc[0] = &stale_source_mr;
	txe->desc_gen[0] = 2;		/* dispatch-time snapshot, now stale */
}

/**
 * @brief Set up an endpoint and a handshake-completed peer for a medium
 *        protocol test, and register a source buffer too large for eager.
 */
static struct efa_rdm_ep *
setup_proto_medium_test(struct efa_resource *resource,
			struct efa_unit_test_buff *send_buff,
			fi_addr_t *peer_addr, struct efa_rdm_peer **peer)
{
	struct efa_rdm_ep *ep;

	ep = setup_proto_select_test(resource, peer_addr);
	efa_unit_test_buff_construct(send_buff, resource,
				     EFA_UNIT_TEST_PROTO_MEDIUM_LEN);

	*peer = efa_rdm_ep_get_peer_explicit(ep, *peer_addr);
	(*peer)->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	/*
	 * The medium protocol only makes sense for a message that needs more
	 * than one packet but still fits in the REQ packets, so make sure the
	 * test size really is in that band on this device.
	 */
	assert_true(EFA_UNIT_TEST_PROTO_MEDIUM_LEN > ep->mtu_size);
	assert_true(EFA_UNIT_TEST_PROTO_MEDIUM_LEN <=
		    g_efa_hmem_info[FI_HMEM_SYSTEM].max_medium_msg_size);

	return ep;
}

/**
 * @brief Test that the medium protocol is selected for a message that is too
 *        large for eager but within the interface's medium threshold.
 */
void test_proto_select_medium_for_medium_msg(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	fi_addr_t peer_addr;
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	ep = setup_proto_medium_test(resource, &send_buff, &peer_addr, &peer);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	assert_int_equal(err, 0);
	assert_non_null(proto);
	assert_ptr_equal(proto, &efa_rdm_proto_medium);

	ofi_buf_free(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief The medium protocol fans the message out over several packets whose
 *        segments tile the whole message exactly once.
 *
 * Each packet must carry the callback and the ope back-reference, the total
 * message length, and its own segment offset -- the receiver reassembles from
 * those two header fields alone, so a gap or an overlap silently corrupts the
 * peer's copy.
 */
void test_proto_medium_construct_pkes_multiple_pkes(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	struct efa_rdm_medium_rtm_base_hdr *rtm_hdr;
	fi_addr_t peer_addr;
	struct fi_msg msg = {0};
	struct iovec iov;
	size_t i, expected_offset = 0;
	int err;

	ep = setup_proto_medium_test(resource, &send_buff, &peer_addr, &peer);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(txe);

	/* Drive the same sequence efa_rdm_msg_generic_send() drives. */
	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	assert_int_equal(err, 0);
	assert_ptr_equal(proto, &efa_rdm_proto_medium);

	efa_rdm_proto_txe_fill(txe, ep, peer, &msg, ofi_op_msg, 0, 0, 0, proto);
	txe->msg_id = peer->next_msg_id++;

	err = proto->construct_tx_pkes(ep, peer, NULL, txe->op, txe->tag,
				       txe->fi_flags, txe->internal_flags, txe);
	assert_int_equal(err, 0);

	/* A single packet would mean eager should have won the selection. */
	assert_true(ep->send_pkt_entry_vec_size > 1);

	/*
	 * The peer-abort protocol reads txe->protocol to tell a two-sided RTM
	 * from an operation it does not handle.
	 */
	assert_int_equal(txe->protocol, EFA_RDM_MEDIUM_MSGRTM_PKT);

	for (i = 0; i < ep->send_pkt_entry_vec_size; ++i) {
		struct efa_rdm_pke *pke = ep->send_pkt_entry_vec[i];

		assert_non_null(pke);
		assert_ptr_equal(pke->handle_pke,
				 &efa_rdm_proto_medium_handle_rtm_send_completion);
		assert_ptr_equal(pke->ope, txe);
		assert_true(pke->payload_size > 0);

		rtm_hdr = efa_rdm_pke_get_medium_rtm_base_hdr(pke);
		assert_int_equal(rtm_hdr->msg_length, txe->total_len);
		assert_int_equal(rtm_hdr->seg_offset, expected_offset);
		expected_offset += pke->payload_size;
	}

	/* The REQ packets carry the whole message, with no gaps. */
	assert_int_equal(expected_offset, txe->total_len);

	for (i = 0; i < ep->send_pkt_entry_vec_size; ++i)
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[i]);
	efa_rdm_txe_release(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief The medium protocol's construct_tx_pkes() is idempotent, so a txe
 *        queued before the handshake can be reposted.
 *
 * efa_rdm_msg_repost_rtm_proto() re-enters construct_tx_pkes() on a txe the
 * first attempt already set up, and efa_rdm_ope_process_queued_ope() retries a
 * txe that returned -FI_EAGAIN without clearing its queued state, so the second
 * call must produce exactly the same packets. An accumulating write (the
 * mainline medium path did bytes_sent += payload_size per packet) would make the
 * message look partly sent and the segment offsets drift past the buffer.
 */
void test_proto_medium_construct_pkes_is_idempotent(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	fi_addr_t peer_addr;
	struct fi_msg msg = {0};
	struct iovec iov;
	size_t i, first_pke_cnt, first_seg_offsets[EFA_UNIT_TEST_PROTO_MAX_PKES];
	uint64_t first_bytes_sent;
	uint32_t first_protocol;
	int err;

	ep = setup_proto_medium_test(resource, &send_buff, &peer_addr, &peer);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	assert_int_equal(err, 0);
	assert_ptr_equal(proto, &efa_rdm_proto_medium);

	efa_rdm_proto_txe_fill(txe, ep, peer, &msg, ofi_op_msg, 0, 0, 0, proto);
	txe->msg_id = peer->next_msg_id++;

	err = proto->construct_tx_pkes(ep, peer, NULL, txe->op, txe->tag,
				       txe->fi_flags, txe->internal_flags, txe);
	assert_int_equal(err, 0);

	first_pke_cnt = ep->send_pkt_entry_vec_size;
	assert_true(first_pke_cnt > 1);
	assert_true(first_pke_cnt <= EFA_UNIT_TEST_PROTO_MAX_PKES);
	for (i = 0; i < first_pke_cnt; ++i) {
		first_seg_offsets[i] =
			efa_rdm_pke_get_medium_rtm_base_hdr(
				ep->send_pkt_entry_vec[i])->seg_offset;
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[i]);
	}
	first_protocol = txe->protocol;

	/*
	 * The first attempt never reached the device, but the post-send hook
	 * runs on the repost, so mimic a first attempt that did publish its
	 * accounting -- that is the state an accumulating write would corrupt.
	 */
	proto->handle_tx_pkes_posted(ep, txe);
	first_bytes_sent = txe->bytes_sent;
	assert_int_equal(first_bytes_sent, txe->total_len);

	/* The repost, with no fi_msg, exactly as efa_rdm_msg_repost_rtm_proto. */
	err = proto->construct_tx_pkes(ep, peer, NULL, txe->op, txe->tag,
				       txe->fi_flags, txe->internal_flags, txe);
	assert_int_equal(err, 0);

	assert_int_equal(ep->send_pkt_entry_vec_size, first_pke_cnt);
	assert_int_equal(txe->protocol, first_protocol);
	for (i = 0; i < first_pke_cnt; ++i)
		assert_int_equal(efa_rdm_pke_get_medium_rtm_base_hdr(
					 ep->send_pkt_entry_vec[i])->seg_offset,
				 first_seg_offsets[i]);

	proto->handle_tx_pkes_posted(ep, txe);
	assert_int_equal(txe->bytes_sent, first_bytes_sent);

	for (i = 0; i < ep->send_pkt_entry_vec_size; ++i)
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[i]);
	efa_rdm_txe_release(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief A medium txe whose source MR was closed mid-transfer is completed
 *        exactly once by the peer-abort drain helper, even though its remaining
 *        packets complete successfully.
 *
 * This is the scenario the EFA_RDM_OPE_PEER_ABORT_PENDING branch in the medium
 * send completion callback exists for, and the reason that branch is not dead
 * code the way it would be for the single packet eager protocol: one medium
 * message is several packets sharing one txe, so an early failure can mark the
 * txe peer-aborting while the later packets are still in flight and go on to
 * report success.
 *
 * Without the branch, the first such success walks into
 * efa_rdm_ope_handle_send_completed(), which asserts the flag is clear (debug)
 * and double-completes the txe (release). Verifies instead that every success
 * routes to the drain helper, which stays silent until the last data WR
 * retires, then emits one PEER_ERROR_PKT so the peer can unblock its reorder
 * window, and finally writes exactly one FI_ECANCELED /
 * FI_EFA_ERR_PEER_ABORTED CQ error entry.
 */
void test_proto_medium_send_completion_peer_abort(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_pke *data_pkes[EFA_UNIT_TEST_PROTO_MAX_PKES];
	struct fi_cq_err_entry err_entry;
	fi_addr_t peer_addr;
	size_t i, pke_cnt;
	int err, ret;

	ep = setup_proto_medium_test(resource, &send_buff, &peer_addr, &peer);
	/* The peer must advertise PEER_ERROR support or the emit is skipped. */
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	g_efa_unit_test_mocks.efa_qp_post_send =
		&efa_mock_efa_qp_post_send_return_mock;
	will_return_int_maybe(efa_mock_efa_qp_post_send_return_mock, 0);

	err = fi_send(resource->ep, send_buff.buff, send_buff.size,
		      fi_mr_desc(send_buff.mr), peer_addr, NULL);
	assert_int_equal(err, 0);
	assert_int_equal(efa_unit_test_get_ope_list_length(ep, EFA_RDM_TXE), 1);

	/*
	 * Snapshot the data packets: emitting the PEER_ERROR_PKT reuses
	 * ep->send_pkt_entry_vec, so it is only safe to read before the drain.
	 */
	pke_cnt = ep->send_pkt_entry_vec_size;
	assert_true(pke_cnt > 1);
	assert_true(pke_cnt <= EFA_UNIT_TEST_PROTO_MAX_PKES);
	for (i = 0; i < pke_cnt; ++i) {
		data_pkes[i] = ep->send_pkt_entry_vec[i];
		assert_non_null(data_pkes[i]->handle_pke);
	}

	txe = data_pkes[0]->ope;
	assert_non_null(txe);
	assert_true(efa_rdm_pkt_type_is_rtm(txe->protocol));

	/*
	 * The application closes the source MR while the send is in flight.
	 * The error path marks the txe peer-aborting; because every data WR is
	 * still outstanding, the drain helper defers the emit.
	 */
	efa_unit_test_proto_simulate_source_mr_canceled(txe);
	efa_rdm_txe_handle_error(txe, FI_ECANCELED, FI_EFA_ERR_PKT_POST);
	assert_true(txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_false(txe->internal_flags & EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);

	memset(&err_entry, 0, sizeof(err_entry));
	assert_int_equal(fi_cq_readerr(resource->cq, &err_entry, 0), -FI_EAGAIN);

	/*
	 * Every in-flight data WR now completes successfully. Only the last one
	 * drains the txe, so only it may emit; the earlier ones must be silent
	 * rather than taking the normal completion path.
	 */
	for (i = 0; i < pke_cnt; ++i) {
		efa_rdm_ep_record_tx_op_completed(ep, data_pkes[i]);
		data_pkes[i]->handle_pke(data_pkes[i]);

		if (i + 1 < pke_cnt) {
			assert_false(txe->internal_flags &
				     EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);
			assert_int_equal(
				fi_cq_readerr(resource->cq, &err_entry, 0),
				-FI_EAGAIN);
		}
	}

	/* The last data WR emitted the PEER_ERROR_PKT and kept the txe alive. */
	assert_true(txe->internal_flags & EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);
	assert_int_equal(efa_unit_test_get_ope_list_length(ep, EFA_RDM_TXE), 1);
	/* Still withheld: the completion waits for the PEER_ERROR_PKT to drain. */
	assert_int_equal(fi_cq_readerr(resource->cq, &err_entry, 0), -FI_EAGAIN);

	/*
	 * The PEER_ERROR_PKT's own send completion releases the txe and writes
	 * the single peer-abort error completion.
	 */
	pkt_entry = ep->send_pkt_entry_vec[0];
	efa_rdm_pke_handle_send_completion(pkt_entry);

	ret = fi_cq_readerr(resource->cq, &err_entry, 0);
	assert_int_equal(ret, 1);
	assert_int_equal(err_entry.err, FI_ECANCELED);
	assert_int_equal(err_entry.prov_errno, FI_EFA_ERR_PEER_ABORTED);

	/* Exactly one completion, and the txe is reaped. */
	assert_int_equal(fi_cq_readerr(resource->cq, &err_entry, 0), -FI_EAGAIN);
	assert_int_equal(efa_unit_test_get_ope_list_length(ep, EFA_RDM_TXE), 0);

	efa_unit_test_buff_destruct(&send_buff);
}
