/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_unit_tests.h"
#include "rdm/efa_rdm_proto.h"
#include "rdm/efa_rdm_proto_eager.h"

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
 * @brief An eager txe whose source MR was closed reaches send completion with
 *        EFA_RDM_OPE_PEER_ABORT_PENDING set, and is completed exactly once by
 *        the peer-abort drain helper.
 *
 * Regression test for the refactored eager send completion callback. Without
 * the PEER_ABORT_PENDING branch the callback walks into
 * efa_rdm_ope_handle_send_completed(), which asserts the flag is clear (debug)
 * and double-completes the txe (release). Verifies the callback instead routes
 * to the drain helper, which emits one PEER_ERROR_PKT so the peer can unblock
 * its reorder window, then writes exactly one FI_ECANCELED /
 * FI_EFA_ERR_PEER_ABORTED CQ error entry.
 */
void test_proto_eager_send_completion_peer_abort(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *pkt_entry;
	struct fi_cq_err_entry err_entry;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	size_t outstanding_before;
	int err, ret;

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
	/* The peer must advertise PEER_ERROR support or the emit is skipped. */
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	g_efa_unit_test_mocks.efa_qp_post_send =
		&efa_mock_efa_qp_post_send_return_mock;
	will_return_int_maybe(efa_mock_efa_qp_post_send_return_mock, 0);

	err = fi_send(resource->ep, send_buff.buff, send_buff.size,
		      fi_mr_desc(send_buff.mr), peer_addr, NULL);
	assert_int_equal(err, 0);
	assert_int_equal(efa_unit_test_get_ope_list_length(ep, EFA_RDM_TXE), 1);

	pkt_entry = ep->send_pkt_entry_vec[0];
	txe = pkt_entry->ope;
	assert_non_null(txe);
	assert_non_null(pkt_entry->handle_pke);

	/*
	 * The refactored path must record the wire protocol on the txe;
	 * efa_rdm_txe_mark_peer_abort_if_needed() only marks two-sided RTMs.
	 */
	assert_true(efa_rdm_pkt_type_is_rtm(txe->protocol));

	/*
	 * The application closes the source MR while the send is in flight.
	 * The error path marks the txe peer-aborting; because the single data
	 * WR is still outstanding, the drain helper defers the emit.
	 */
	efa_unit_test_proto_simulate_source_mr_canceled(txe);
	efa_rdm_txe_handle_error(txe, FI_ECANCELED, FI_EFA_ERR_PKT_POST);
	assert_true(txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_false(txe->internal_flags & EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);

	/* No user CQ entry yet -- the completion is withheld until drain. */
	memset(&err_entry, 0, sizeof(err_entry));
	assert_int_equal(fi_cq_readerr(resource->cq, &err_entry, 0), -FI_EAGAIN);

	outstanding_before = ep->efa_outstanding_tx_ops;

	/*
	 * The in-flight data WR now completes successfully. The callback must
	 * route the aborting txe to the drain helper, which emits the
	 * PEER_ERROR_PKT (one new WR) and keeps the txe alive for it.
	 */
	efa_rdm_ep_record_tx_op_completed(ep, pkt_entry);
	pkt_entry->handle_pke(pkt_entry);

	assert_true(txe->internal_flags & EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before);
	assert_int_equal(efa_unit_test_get_ope_list_length(ep, EFA_RDM_TXE), 1);
	/* Still withheld: the completion waits for the PEER_ERROR_PKT to drain. */
	assert_int_equal(fi_cq_readerr(resource->cq, &err_entry, 0), -FI_EAGAIN);

	/* The PEER_ERROR_PKT's own send completion releases the txe and
	 * writes the single peer-abort error completion. */
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

	/*
	 * The handshake trigger and the queued send's repost each post at least
	 * once, and the progress engine may retry, so let every post succeed
	 * rather than queueing a fixed number of return values. The mock reads
	 * its value with mock_int(), so it needs the int variant.
	 */
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
