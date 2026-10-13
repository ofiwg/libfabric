/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_unit_tests.h"
#include "rdm/efa_rdm_pke_nonreq.h"
#include "rdm/efa_rdm_pke_rta.h"
#include "rdm/efa_rdm_pke_utils.h"
#include "rdm/efa_rdm_proto.h"
#include "rdm/protocols/efa_rdm_proto_atomic.h"
#include "rdm/protocols/efa_rdm_proto_eager.h"
#include "rdm/protocols/efa_rdm_proto_zero_copy.h"
#include <errno.h>

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

	ep = setup_proto_select_test(resource, &peer_addr);
	peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	iov.iov_base = NULL;
	iov.iov_len = 64; /* Small message, fits in eager */
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0, NULL);

	txe = ofi_buf_alloc(ep->base_ep.txe_pool);
	assert_non_null(txe);

	efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0, txe,
					   &proto);
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

	ep = setup_proto_select_test(resource, &peer_addr);
	peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	iov.iov_base = NULL;
	iov.iov_len = 0;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0, NULL);

	txe = ofi_buf_alloc(ep->base_ep.txe_pool);
	assert_non_null(txe);

	efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0, txe,
					   &proto);
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
	struct efa_rdm_proto *proto = NULL;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;
	uint64_t pke_send_flags;

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

	txe = ofi_buf_alloc(ep->base_ep.txe_pool);
	assert_non_null(txe);

	/*
	 * Fill the TXE as generic_send would before calling construct_tx_pkes.
	 * The selection step is what populates txe->req_pkt_type, txe->desc[]
	 * and the MR generation snapshot, all of which construct_tx_pkes()
	 * reads, so run it rather than setting up a subset by hand: a freshly
	 * allocated txe_pool slot holds whatever its previous occupant left
	 * behind.
	 */
	efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0, txe,
					   &proto);
	assert_ptr_equal(proto, &efa_rdm_proto_eager);
	efa_rdm_proto_txe_fill(txe, ep, peer, &msg, ofi_op_msg, 0, 0, 0, proto);
	txe->msg_id = peer->next_msg_id++;

	err = efa_rdm_proto_eager.construct_tx_pkes(ep, txe, &pke_send_flags);
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

/**
 * @brief Test that the zero-copy protocol's construct_tx_pkes produces a
 * headerless PKE with the EFA_RDM_PKE_SEND_TO_USER_RECV_QP flag.
 */
void test_proto_zero_copy_construct_pkes(void **state)
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
	uint64_t pke_send_flags;

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

	txe = ofi_buf_alloc(ep->base_ep.txe_pool);
	assert_non_null(txe);

	/* Fill TXE as generic_send would before calling construct_tx_pkes */
	efa_rdm_proto_txe_init_buffers(ep, &msg, txe);
	efa_rdm_proto_txe_fill(txe, ep, peer, &msg, ofi_op_msg, 0, 0, 0,
			       &efa_rdm_proto_zero_copy);
	txe->msg_id = peer->next_msg_id++;

	err = efa_rdm_proto_zero_copy.construct_tx_pkes(ep, txe,
							&pke_send_flags);
	assert_int_equal(err, 0);
	assert_int_equal(ep->send_pkt_entry_vec_size, 1);

	/* Verify headerless packet properties */
	struct efa_rdm_pke *pke = ep->send_pkt_entry_vec[0];
	assert_true(pke->flags & EFA_RDM_PKE_SEND_TO_USER_RECV_QP);
	assert_true(pke->flags & EFA_RDM_PKE_HAS_NO_BASE_HDR);
	assert_int_equal(pke->pkt_size, 64);
	assert_int_equal(pke->payload_size, 64);

	efa_rdm_pke_release_tx(pke);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief A send queued before the handshake switches to the zero-copy protocol
 *        when the handshake reveals a headerless peer.
 *
 * Whether a peer accepts only headerless packets is carried in its handshake,
 * so a send dispatched before the handshake arrived selected a protocol that
 * writes a REQ header. The repost must revisit that choice; otherwise the
 * headered packet lands on the peer's control QP, which a peer in zero-copy
 * receive mode rejects.
 */
void test_proto_zero_copy_reselected_after_handshake(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *pkt_entry;
	struct fi_cq_tagged_entry cq_entry;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
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
	assert_ptr_equal(txe->proto, &efa_rdm_proto_eager);

	/* The handshake arrives and reports the peer is in zero-copy mode. */
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_REQUEST_USER_RECV_QP;
	peer->user_recv_qp.qpn = 99;
	peer->user_recv_qp.qkey = 0xABCD;

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, -FI_EAGAIN);

	assert_int_equal(ep->ope_queued_before_handshake_cnt, 0);
	assert_true(dlist_empty(&ep->ope_queued_list));

	/* The reposted packet is headerless and bound for the user_recv_qp. */
	assert_ptr_equal(txe->proto, &efa_rdm_proto_zero_copy);
	pkt_entry = ep->send_pkt_entry_vec[0];
	assert_ptr_equal(pkt_entry->ope, txe);
	assert_true(pkt_entry->flags & EFA_RDM_PKE_SEND_TO_USER_RECV_QP);
	assert_true(pkt_entry->flags & EFA_RDM_PKE_HAS_NO_BASE_HDR);

	efa_unit_test_buff_destruct(&send_buff);
}

#define EFA_UNIT_TEST_ATOMIC_REMOTE_ADDR 0x12345000ULL
#define EFA_UNIT_TEST_ATOMIC_REMOTE_KEY  0xabcdu
#define EFA_UNIT_TEST_ATOMIC_LEN         8

struct efa_unit_test_atomic_ctx {
	struct efa_resource *resource;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct fi_msg_atomic msg;
	struct fi_ioc ioc;
	struct fi_rma_ioc rma_ioc;
	void *desc;
	fi_addr_t peer_addr;
	size_t tx_pkt_free;
	size_t txe_free;
	uint32_t next_msg_id;
};

static size_t test_proto_atomic_bufpool_free_count(struct ofi_bufpool *pool)
{
	struct slist_entry *item;
	size_t count = 0;

	for (item = pool->free_list.entries.head; item; item = item->next)
		count++;

	return count;
}

static void test_proto_atomic_setup(struct efa_resource *resource,
				    struct efa_unit_test_atomic_ctx *ctx)
{
	struct efa_rdm_pke *pke;
	struct efa_rdm_ope *txe;

	memset(ctx, 0, sizeof(*ctx));
	ctx->resource = resource;
	ctx->ep = setup_proto_select_test(resource, &ctx->peer_addr);
	ctx->peer = efa_rdm_ep_get_peer_explicit(ctx->ep, ctx->peer_addr);
	ctx->peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	efa_unit_test_buff_construct(&ctx->send_buff, resource,
				     ctx->ep->mtu_size);
	ctx->desc = fi_mr_desc(ctx->send_buff.mr);
	ctx->ioc.addr = ctx->send_buff.buff;
	ctx->rma_ioc.addr = EFA_UNIT_TEST_ATOMIC_REMOTE_ADDR;
	ctx->rma_ioc.key = EFA_UNIT_TEST_ATOMIC_REMOTE_KEY;
	ctx->msg.msg_iov = &ctx->ioc;
	ctx->msg.desc = &ctx->desc;
	ctx->msg.iov_count = 1;
	ctx->msg.addr = ctx->peer_addr;
	ctx->msg.rma_iov = &ctx->rma_ioc;
	ctx->msg.rma_iov_count = 1;
	ctx->msg.datatype = FI_UINT8;
	ctx->msg.op = FI_ATOMIC_WRITE;
	ctx->msg.context = ctx;

	/* Warm lazy pools so free-count checks detect leaked entries. */
	pke = efa_rdm_pke_alloc(ctx->ep, ctx->ep->efa_tx_pkt_pool,
				ctx->ep->efa_tx_bounce_pool,
				EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pke);
	efa_rdm_pke_release_tx(pke);
	txe = ofi_buf_alloc(ctx->ep->base_ep.txe_pool);
	assert_non_null(txe);
	ofi_buf_free(txe);

	ctx->tx_pkt_free =
		test_proto_atomic_bufpool_free_count(ctx->ep->efa_tx_pkt_pool);
	ctx->txe_free =
		test_proto_atomic_bufpool_free_count(ctx->ep->base_ep.txe_pool);
	ctx->next_msg_id = ctx->peer->next_msg_id;
}

static ssize_t test_proto_atomic_post(struct efa_unit_test_atomic_ctx *ctx,
				      size_t count, uint64_t flags)
{
	ctx->ioc.count = count;
	ctx->rma_ioc.count = count;
	return fi_atomicmsg(ctx->resource->ep, &ctx->msg, flags);
}

static struct efa_rdm_pke *
test_proto_atomic_posted_pke(struct efa_unit_test_atomic_ctx *ctx)
{
	assert_int_equal(ctx->ep->send_pkt_entry_vec_size, 1);
	assert_non_null(ctx->ep->send_pkt_entry_vec[0]);
	return ctx->ep->send_pkt_entry_vec[0];
}

static struct efa_rdm_pke *
test_proto_atomic_receipt(struct efa_unit_test_atomic_ctx *ctx,
			  uint32_t tx_id)
{
	struct efa_rdm_pke *pke;
	struct efa_rdm_receipt_hdr *hdr;

	pke = efa_rdm_pke_alloc(ctx->ep, ctx->ep->rx_unexp_pkt_pool,
				ctx->ep->rx_unexp_bounce_pool,
				EFA_RDM_PKE_FROM_UNEXP_POOL);
	assert_non_null(pke);
	pke->ep = ctx->ep;
	pke->peer = ctx->peer;
	hdr = efa_rdm_pke_get_receipt_hdr(pke);
	memset(hdr, 0, sizeof(*hdr));
	hdr->type = EFA_RDM_RECEIPT_PKT;
	hdr->version = EFA_RDM_PROTOCOL_VERSION;
	hdr->tx_id = tx_id;
	return pke;
}

static void test_proto_atomic_assert_clean(
	struct efa_unit_test_atomic_ctx *ctx, uint32_t next_msg_id)
{
	assert_int_equal(
		efa_unit_test_get_ope_list_length(ctx->ep, EFA_RDM_TXE), 0);
	assert_int_equal(ctx->ep->efa_outstanding_tx_ops, 0);
	assert_int_equal(ctx->peer->next_msg_id, next_msg_id);
	assert_int_equal(
		test_proto_atomic_bufpool_free_count(ctx->ep->efa_tx_pkt_pool),
		ctx->tx_pkt_free);
	assert_int_equal(
		test_proto_atomic_bufpool_free_count(ctx->ep->base_ep.txe_pool),
		ctx->txe_free);
}

static void test_proto_atomic_assert_cq_empty(
	struct efa_unit_test_atomic_ctx *ctx)
{
	struct fi_cq_data_entry entry;
	struct fi_cq_err_entry err_entry = {0};

	assert_int_equal(fi_cq_read(ctx->resource->cq, &entry, 1),
			 -FI_EAGAIN);
	assert_int_equal(fi_cq_readerr(ctx->resource->cq, &err_entry, 0),
			 -FI_EAGAIN);
}

static void test_proto_atomic_read_success(
	struct efa_unit_test_atomic_ctx *ctx, size_t len)
{
	struct fi_cq_data_entry entry;

	assert_int_equal(fi_cq_read(ctx->resource->cq, &entry, 1), 1);
	assert_ptr_equal(entry.op_context, ctx);
	assert_int_equal(entry.flags, FI_ATOMIC | FI_WRITE);
	assert_int_equal(entry.len, len);
}

static void test_proto_atomic_read_error(
	struct efa_unit_test_atomic_ctx *ctx, int err, int prov_errno)
{
	struct fi_cq_data_entry entry;
	struct fi_cq_err_entry err_entry = {0};

	assert_int_equal(fi_cq_read(ctx->resource->cq, &entry, 1),
			 -FI_EAVAIL);
	assert_int_equal(fi_cq_readerr(ctx->resource->cq, &err_entry, 0), 1);
	assert_ptr_equal(err_entry.op_context, ctx);
	assert_int_equal(err_entry.flags, FI_ATOMIC | FI_WRITE);
	assert_int_equal(err_entry.err, err);
	assert_int_equal(err_entry.prov_errno, prov_errno);
}

void test_proto_atomic_write_post_and_complete(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_atomic_ctx ctx;
	struct efa_rdm_ope probe = {0};
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *pke;
	struct efa_rdm_rta_hdr *hdr;
	uint8_t expected[EFA_UNIT_TEST_ATOMIC_LEN];
	size_t exact;
	size_t hdr_size;
	int post_count;

	test_proto_atomic_setup(resource, &ctx);
	probe.ep = ctx.ep;
	probe.peer = ctx.peer;
	probe.rma_iov_count = 1;
	exact = efa_rdm_txe_max_req_data_capacity(
		ctx.ep, &probe, EFA_RDM_WRITE_RTA_PKT);
	assert_true(exact > 0);
	assert_true(exact + 1 <= ctx.send_buff.size);
	memset(ctx.send_buff.buff, 0x5a, ctx.send_buff.size);

	g_efa_unit_test_mocks.efa_qp_post_send =
		efa_mock_efa_qp_post_send_return_mock;
	will_return_int_always(efa_mock_efa_qp_post_send_return_mock, 0);

	assert_int_equal(
		test_proto_atomic_post(&ctx, exact, FI_COMPLETION), 0);
	pke = test_proto_atomic_posted_pke(&ctx);
	txe = pke->ope;
	hdr = efa_rdm_pke_get_rta_hdr(pke);
	hdr_size = efa_rdm_pke_get_req_hdr_size(pke);
	assert_int_equal(efa_rdm_pkt_type_of(pke), EFA_RDM_WRITE_RTA_PKT);
	assert_ptr_equal(txe->atomic_proto, &efa_rdm_atomic_write_proto);
	assert_null(txe->proto);
	assert_ptr_equal(pke->handle_pke,
			 txe->atomic_proto->handle_send_completion);
	assert_int_equal(hdr->atomic_datatype, FI_UINT8);
	assert_int_equal(hdr->atomic_op, FI_ATOMIC_WRITE);
	assert_memory_equal(pke->wiredata + hdr_size,
			    ctx.send_buff.buff, exact);

	efa_rdm_pke_handle_send_completion(pke);
	test_proto_atomic_read_success(&ctx, exact);
	test_proto_atomic_assert_clean(&ctx, ctx.next_msg_id + 1);

	post_count = g_ibv_submitted_wr_id_cnt;
	assert_int_equal(
		test_proto_atomic_post(&ctx, exact + 1, FI_COMPLETION),
		-FI_EMSGSIZE);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, post_count);
	test_proto_atomic_assert_clean(&ctx, ctx.next_msg_id + 1);
	test_proto_atomic_assert_cq_empty(&ctx);

	memset(ctx.send_buff.buff, 0xa5, sizeof(expected));
	memcpy(expected, ctx.send_buff.buff, sizeof(expected));
	assert_int_equal(
		fi_inject_atomic(resource->ep, ctx.send_buff.buff,
				 sizeof(expected), ctx.peer_addr,
				 EFA_UNIT_TEST_ATOMIC_REMOTE_ADDR,
				 EFA_UNIT_TEST_ATOMIC_REMOTE_KEY,
				 FI_UINT8, FI_ATOMIC_WRITE),
		0);
	pke = test_proto_atomic_posted_pke(&ctx);
	txe = pke->ope;
	hdr_size = efa_rdm_pke_get_req_hdr_size(pke);
	assert_true(txe->internal_flags & EFA_RDM_TXE_NO_COMPLETION);
	assert_ptr_equal(txe->atomic_proto, &efa_rdm_atomic_write_proto);
	assert_ptr_equal(pke->handle_pke,
			 txe->atomic_proto->handle_send_completion);
	memset(ctx.send_buff.buff, 0, sizeof(expected));
	assert_memory_equal(pke->wiredata + hdr_size,
			    expected, sizeof(expected));

	efa_rdm_pke_handle_send_completion(pke);
	test_proto_atomic_assert_clean(&ctx, ctx.next_msg_id + 2);
	test_proto_atomic_assert_cq_empty(&ctx);
	efa_unit_test_buff_destruct(&ctx.send_buff);
}

void test_proto_atomic_dc_write_completion_orders(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_atomic_ctx ctx;
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *req_pke;
	struct efa_rdm_pke *receipt_pke;
	int send_first;

	test_proto_atomic_setup(resource, &ctx);
	memset(ctx.send_buff.buff, 0x5a, EFA_UNIT_TEST_ATOMIC_LEN);
	g_efa_unit_test_mocks.efa_qp_post_send =
		efa_mock_efa_qp_post_send_return_mock;
	will_return_int_always(efa_mock_efa_qp_post_send_return_mock, 0);

	for (send_first = 0; send_first <= 1; send_first++) {
		assert_int_equal(
			test_proto_atomic_post(
				&ctx, EFA_UNIT_TEST_ATOMIC_LEN,
				FI_DELIVERY_COMPLETE | FI_COMPLETION),
			0);
		req_pke = test_proto_atomic_posted_pke(&ctx);
		txe = req_pke->ope;
		assert_int_equal(efa_rdm_pkt_type_of(req_pke),
				 EFA_RDM_DC_WRITE_RTA_PKT);
		assert_int_equal(efa_rdm_pke_get_rta_hdr(req_pke)->send_id,
				 txe->tx_id);
		assert_ptr_equal(req_pke->handle_pke,
				 txe->atomic_proto->handle_send_completion);
		receipt_pke = test_proto_atomic_receipt(&ctx, txe->tx_id);

		if (send_first) {
			efa_rdm_pke_handle_send_completion(req_pke);
			assert_int_equal(efa_unit_test_get_ope_list_length(
						 ctx.ep, EFA_RDM_TXE), 1);
			test_proto_atomic_assert_cq_empty(&ctx);
			efa_rdm_proto_handle_receipt_recv(receipt_pke);
			test_proto_atomic_read_success(
				&ctx, EFA_UNIT_TEST_ATOMIC_LEN);
		} else {
			efa_rdm_proto_handle_receipt_recv(receipt_pke);
			assert_true(txe->internal_flags &
				    EFA_RDM_TXE_REMOTE_ACK_RECEIVED);
			assert_int_equal(efa_unit_test_get_ope_list_length(
						 ctx.ep, EFA_RDM_TXE), 1);
			test_proto_atomic_read_success(
				&ctx, EFA_UNIT_TEST_ATOMIC_LEN);
			efa_rdm_pke_handle_send_completion(req_pke);
		}

		test_proto_atomic_assert_clean(
			&ctx, ctx.next_msg_id + send_first + 1);
		test_proto_atomic_assert_cq_empty(&ctx);
	}

	efa_unit_test_buff_destruct(&ctx.send_buff);
}

void test_proto_atomic_write_sync_failures(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_atomic_ctx ctx;
	uint64_t flags[] = {
		FI_COMPLETION,
		FI_COMPLETION | FI_DELIVERY_COMPLETE,
	};
	size_t i;

	test_proto_atomic_setup(resource, &ctx);
	memset(ctx.send_buff.buff, 0x5a, EFA_UNIT_TEST_ATOMIC_LEN);
	g_efa_unit_test_mocks.efa_copy_from_hmem_iov =
		efa_mock_efa_copy_from_hmem_iov_return_mock;
	will_return_count(efa_mock_efa_copy_from_hmem_iov_return_mock,
			  -FI_EIO, ARRAY_SIZE(flags));

	for (i = 0; i < ARRAY_SIZE(flags); i++) {
		assert_int_equal(
			test_proto_atomic_post(
				&ctx, EFA_UNIT_TEST_ATOMIC_LEN, flags[i]),
			-FI_EIO);
		test_proto_atomic_assert_clean(&ctx, ctx.next_msg_id);
		test_proto_atomic_assert_cq_empty(&ctx);
	}

	g_efa_unit_test_mocks.efa_copy_from_hmem_iov =
		__real_efa_copy_from_hmem_iov;
	g_efa_unit_test_mocks.efa_qp_post_send =
		efa_mock_efa_qp_post_send_return_mock;
	will_return_int(efa_mock_efa_qp_post_send_return_mock, ENOMEM);
	assert_int_equal(
		test_proto_atomic_post(
			&ctx, EFA_UNIT_TEST_ATOMIC_LEN, FI_COMPLETION),
		-FI_EAGAIN);
	test_proto_atomic_assert_clean(&ctx, ctx.next_msg_id);
	test_proto_atomic_assert_cq_empty(&ctx);
	efa_unit_test_buff_destruct(&ctx.send_buff);
}

void test_proto_atomic_write_async_release_paths(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_atomic_ctx ctx;
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *pke;

	test_proto_atomic_setup(resource, &ctx);
	memset(ctx.send_buff.buff, 0x5a, EFA_UNIT_TEST_ATOMIC_LEN);
	g_efa_unit_test_mocks.efa_qp_post_send =
		efa_mock_efa_qp_post_send_return_mock;

	will_return_int(efa_mock_efa_qp_post_send_return_mock, 0);
	assert_int_equal(
		test_proto_atomic_post(
			&ctx, EFA_UNIT_TEST_ATOMIC_LEN, FI_COMPLETION),
		0);
	pke = test_proto_atomic_posted_pke(&ctx);
	efa_rdm_pke_handle_tx_error(
		pke, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);
	test_proto_atomic_read_error(
		&ctx, FI_EHOSTUNREACH,
		EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);
	test_proto_atomic_assert_clean(&ctx, ctx.next_msg_id + 1);
	test_proto_atomic_assert_cq_empty(&ctx);

	will_return_int(efa_mock_efa_qp_post_send_return_mock, 0);
	assert_int_equal(
		test_proto_atomic_post(
			&ctx, EFA_UNIT_TEST_ATOMIC_LEN, FI_COMPLETION),
		0);
	pke = test_proto_atomic_posted_pke(&ctx);
	txe = pke->ope;
	efa_rdm_pke_handle_tx_error(
		pke, EFA_IO_COMP_STATUS_REMOTE_ERROR_RNR);
	assert_true(txe->internal_flags & EFA_RDM_OPE_QUEUED_RNR);
	assert_int_equal(
		efa_unit_test_get_ope_list_length(ctx.ep, EFA_RDM_TXE), 1);

	will_return_int(efa_mock_efa_qp_post_send_return_mock, EIO);
	assert_int_equal(efa_rdm_ope_process_queued_ope(txe), -FI_EIO);
	test_proto_atomic_read_error(&ctx, FI_EIO, FI_EFA_ERR_PKT_POST);
	test_proto_atomic_assert_clean(&ctx, ctx.next_msg_id + 2);
	test_proto_atomic_assert_cq_empty(&ctx);
	efa_unit_test_buff_destruct(&ctx.send_buff);
}

static void test_proto_atomic_complete_dc(struct efa_unit_test_atomic_ctx *ctx,
					  struct efa_rdm_pke *req_pke,
					  size_t len)
{
	struct efa_rdm_pke *receipt_pke;

	receipt_pke = test_proto_atomic_receipt(ctx, req_pke->ope->tx_id);
	efa_rdm_pke_handle_send_completion(req_pke);
	test_proto_atomic_assert_cq_empty(ctx);
	efa_rdm_proto_handle_receipt_recv(receipt_pke);
	test_proto_atomic_read_success(ctx, len);
}

/**
 * @brief A send error after RECEIPT keeps the reported success and releases
 * the PKE and TXE without writing an error CQ entry.
 */
void test_proto_atomic_dc_receipt_then_send_error(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_atomic_ctx ctx;
	struct efa_rdm_pke *pke;

	test_proto_atomic_setup(resource, &ctx);
	memset(ctx.send_buff.buff, 0x5a, EFA_UNIT_TEST_ATOMIC_LEN);
	g_efa_unit_test_mocks.efa_qp_post_send =
		efa_mock_efa_qp_post_send_return_mock;
	will_return_int(efa_mock_efa_qp_post_send_return_mock, 0);

	assert_int_equal(
		test_proto_atomic_post(&ctx, EFA_UNIT_TEST_ATOMIC_LEN,
				       FI_COMPLETION | FI_DELIVERY_COMPLETE),
		0);
	pke = test_proto_atomic_posted_pke(&ctx);
	efa_rdm_proto_handle_receipt_recv(
		test_proto_atomic_receipt(&ctx, pke->ope->tx_id));
	test_proto_atomic_read_success(&ctx, EFA_UNIT_TEST_ATOMIC_LEN);

	efa_rdm_pke_handle_tx_error(
		pke, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);
	test_proto_atomic_assert_cq_empty(&ctx);
	test_proto_atomic_assert_clean(&ctx, ctx.next_msg_id + 1);
	efa_unit_test_buff_destruct(&ctx.send_buff);
}

/**
 * @brief FI_INJECT does not downgrade FI_DELIVERY_COMPLETE: the operand is
 * copied before return and success is reported once, on RECEIPT.
 */
void test_proto_atomic_inject_delivery_complete(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_atomic_ctx ctx;
	uint8_t expected[EFA_UNIT_TEST_ATOMIC_LEN];
	struct efa_rdm_pke *pke;

	test_proto_atomic_setup(resource, &ctx);
	memset(ctx.send_buff.buff, 0xa5, sizeof(expected));
	memcpy(expected, ctx.send_buff.buff, sizeof(expected));
	g_efa_unit_test_mocks.efa_qp_post_send =
		efa_mock_efa_qp_post_send_return_mock;
	will_return_int(efa_mock_efa_qp_post_send_return_mock, 0);

	assert_int_equal(
		test_proto_atomic_post(&ctx, sizeof(expected),
				       FI_INJECT | FI_DELIVERY_COMPLETE |
					       FI_COMPLETION),
		0);
	memset(ctx.send_buff.buff, 0, sizeof(expected));
	pke = test_proto_atomic_posted_pke(&ctx);
	assert_int_equal(efa_rdm_pkt_type_of(pke), EFA_RDM_DC_WRITE_RTA_PKT);
	assert_int_equal(efa_rdm_pke_get_rta_hdr(pke)->send_id,
			 pke->ope->tx_id);
	assert_memory_equal(pke->wiredata + efa_rdm_pke_get_req_hdr_size(pke),
			    expected, sizeof(expected));

	test_proto_atomic_complete_dc(&ctx, pke, sizeof(expected));
	test_proto_atomic_assert_clean(&ctx, ctx.next_msg_id + 1);
	test_proto_atomic_assert_cq_empty(&ctx);
	efa_unit_test_buff_destruct(&ctx.send_buff);
}
