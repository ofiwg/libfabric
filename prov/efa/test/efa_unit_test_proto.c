/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_unit_tests.h"
#include "ofi_atomic.h"
#include "rdm/efa_rdm_pke_nonreq.h"
#include "rdm/efa_rdm_pke_rta.h"
#include "rdm/efa_rdm_pke_utils.h"
#include "rdm/efa_rdm_proto.h"
#include "rdm/protocols/efa_rdm_proto_atomic.h"
#include "rdm/efa_rdm_pke_rtm.h"
#include "rdm/protocols/efa_rdm_proto_eager.h"
#include "rdm/protocols/efa_rdm_proto_runtread.h"
#include "rdm/protocols/efa_rdm_proto_zero_copy.h"
#include <errno.h>

/*
 * Sizes for the runt read tests: past the medium threshold so medium declines,
 * and comfortably larger than the runt below so the receiver still has a tail to
 * read -- which is what makes this the runt read protocol rather than a plain
 * multi-packet send.
 */
#define EFA_UNIT_TEST_PROTO_RUNTREAD_LEN 131072
#define EFA_UNIT_TEST_PROTO_RUNT_SIZE	 32768

/*
 * Upper bound for the per-test arrays that snapshot a multi-packet message's
 * packet entries. The runt portion needs ceil(runt_size / (mtu - hdr)) of them,
 * which is a handful for the size above on any EFA device; the tests assert the
 * fit rather than trusting it.
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

	ofi_buf_free(pke);
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
	size_t rxe_free;
	size_t atomrsp_free;
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

static size_t test_proto_atomic_bufpool_used_count(struct ofi_bufpool *pool){
	assert(!(pool->attr.flags & OFI_BUFPOOL_INDEXED));
	return pool->entry_cnt - test_proto_atomic_bufpool_free_count(pool);
}

static void test_proto_atomic_setup_with_mr_access(
	struct efa_resource *resource, struct efa_unit_test_atomic_ctx *ctx,
	uint64_t mr_access)
{
	struct efa_rdm_pke *pke;
	struct efa_rdm_ope *txe;
	void *atomrsp;

	memset(ctx, 0, sizeof(*ctx));
	ctx->resource = resource;
	ctx->ep = setup_proto_select_test(resource, &ctx->peer_addr);
	ctx->peer = efa_rdm_ep_get_peer_explicit(ctx->ep, ctx->peer_addr);
	ctx->peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	efa_unit_test_buff_construct(&ctx->send_buff, resource,
				     3 * ctx->ep->mtu_size);
	assert_int_equal(fi_close(&ctx->send_buff.mr->fid), 0);
	assert_int_equal(fi_mr_reg(resource->domain, ctx->send_buff.buff,
				   ctx->send_buff.size,
				   FI_SEND | FI_RECV | mr_access, 0, 0, 0,
				   &ctx->send_buff.mr, NULL),
			 0);
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
				EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pke);
	efa_rdm_pke_release_tx(pke);
	txe = ofi_buf_alloc(ctx->ep->base_ep.txe_pool);
	assert_non_null(txe);
	ofi_buf_free(txe);
	txe = ofi_buf_alloc(ctx->ep->base_ep.rxe_pool);
	assert_non_null(txe);
	ofi_buf_free(txe);
	atomrsp = ofi_buf_alloc(ctx->ep->rx_atomrsp_pool);
	assert_non_null(atomrsp);
	ofi_buf_free(atomrsp);

	ctx->tx_pkt_free =
		test_proto_atomic_bufpool_free_count(ctx->ep->efa_tx_pkt_pool);
	ctx->txe_free =
		test_proto_atomic_bufpool_free_count(ctx->ep->base_ep.txe_pool);
	ctx->rxe_free =
		test_proto_atomic_bufpool_free_count(ctx->ep->base_ep.rxe_pool);
	ctx->atomrsp_free =
		test_proto_atomic_bufpool_free_count(ctx->ep->rx_atomrsp_pool);
	ctx->next_msg_id = ctx->peer->next_msg_id;
}

static void test_proto_atomic_setup(struct efa_resource *resource,
				    struct efa_unit_test_atomic_ctx *ctx)
{
	test_proto_atomic_setup_with_mr_access(
		resource, ctx, FI_REMOTE_READ | FI_REMOTE_WRITE);
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

static struct efa_rdm_pke *test_proto_atomic_request_rx(
	struct efa_unit_test_atomic_ctx *ctx, struct efa_rdm_pke *req_pke)
{
	struct efa_rdm_pke *req_rx;

	req_rx = efa_rdm_pke_alloc(ctx->ep, ctx->ep->rx_unexp_pkt_pool,
				   EFA_RDM_PKE_FROM_UNEXP_POOL);
	assert_non_null(req_rx);
	req_rx->ep = ctx->ep;
	req_rx->peer = ctx->peer;
	req_rx->pkt_size = req_pke->pkt_size;
	memcpy(req_rx->wiredata, req_pke->wiredata, req_pke->pkt_size);
	return req_rx;
}

static struct efa_rdm_pke *
test_proto_atomic_receipt(struct efa_unit_test_atomic_ctx *ctx,
			  uint32_t tx_id)
{
	struct efa_rdm_pke *pke;
	struct efa_rdm_receipt_hdr *hdr;

	pke = efa_rdm_pke_alloc(ctx->ep, ctx->ep->rx_unexp_pkt_pool,
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

static struct efa_rdm_pke *test_proto_atomic_atomrsp(
	struct efa_unit_test_atomic_ctx *ctx, uint32_t tx_id, const void *data,
	size_t len) {
	struct efa_rdm_atomrsp_pkt *pkt;
	struct efa_rdm_pke *pke;

	pke = efa_rdm_pke_alloc(ctx->ep, ctx->ep->rx_unexp_pkt_pool,
				EFA_RDM_PKE_FROM_UNEXP_POOL);
	assert_non_null(pke);
	pke->ep = ctx->ep;
	pke->peer = ctx->peer;
	pkt = (struct efa_rdm_atomrsp_pkt *) pke->wiredata;
	memset(pkt, 0, sizeof(pkt->hdr));
	pkt->hdr.type = EFA_RDM_ATOMRSP_PKT;
	pkt->hdr.version = EFA_RDM_PROTOCOL_VERSION;
	pkt->hdr.recv_id = tx_id;
	pkt->hdr.seg_length = len;
	memcpy(pkt->data, data, len);
	pke->pkt_size = sizeof(pkt->hdr) + len;
	return pke;
}

static ssize_t test_proto_atomic_fetch_compare_post(
	struct efa_unit_test_atomic_ctx *ctx, enum fi_op op, size_t count,
	void *compare, void *result, uint64_t flags) {
	struct fi_ioc compare_ioc = {.addr = compare, .count = count};
	struct fi_ioc result_ioc = {.addr = result, .count = count};
	void *compare_desc = ctx->desc;
	void *result_desc = ctx->desc;

	ctx->ioc.count = count;
	ctx->rma_ioc.count = count;
	ctx->msg.op = op;
	if (op == FI_CSWAP)
		return fi_compare_atomicmsg(
			ctx->resource->ep, &ctx->msg, &compare_ioc,
			&compare_desc, 1, &result_ioc, &result_desc, 1, flags);
	return fi_fetch_atomicmsg(ctx->resource->ep, &ctx->msg, &result_ioc,
				  &result_desc, 1, flags);
}

static int test_proto_atomic_process_fetch_compare(
	struct efa_rdm_pke *req_rx, enum fi_op op)
{
	return op == FI_CSWAP ? efa_rdm_pke_proc_compare_rta(req_rx) :
			       efa_rdm_pke_proc_fetch_rta(req_rx);
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
	assert_int_equal(
		test_proto_atomic_bufpool_free_count(ctx->ep->base_ep.rxe_pool),
		ctx->rxe_free);
	assert_int_equal(
		test_proto_atomic_bufpool_free_count(ctx->ep->rx_atomrsp_pool),
		ctx->atomrsp_free);
	assert_int_equal(
		test_proto_atomic_bufpool_used_count(
			ctx->ep->rx_unexp_pkt_pool),
		0);
	assert_int_equal(ctx->ep->efa_rnr_queued_pkt_cnt, 0);
	assert_int_equal(ctx->ep->ope_queued_before_handshake_cnt, 0);
	assert_true(dlist_empty(&ctx->ep->ope_queued_list));
	assert_int_equal(ctx->peer->efa_outstanding_tx_ops, 0);
	assert_int_equal(ctx->peer->rnr_queued_pkt_cnt, 0);
	assert_true(dlist_empty(&ctx->peer->outstanding_tx_pkts));
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

static void test_proto_atomic_read_success(struct efa_unit_test_atomic_ctx *ctx,
					   size_t len, uint64_t flags)
{
	struct fi_cq_data_entry entry;

	assert_int_equal(fi_cq_read(ctx->resource->cq, &entry, 1), 1);
	assert_ptr_equal(entry.op_context, ctx);
	assert_int_equal(entry.flags, flags);
	assert_int_equal(entry.len, len);
}

static void test_proto_atomic_read_error(struct efa_unit_test_atomic_ctx *ctx,
					 int err, int prov_errno,
					 uint64_t flags)
{
	struct fi_cq_data_entry entry;
	struct fi_cq_err_entry err_entry = {0};

	assert_int_equal(fi_cq_read(ctx->resource->cq, &entry, 1),
			 -FI_EAVAIL);
	assert_int_equal(fi_cq_readerr(ctx->resource->cq, &err_entry, 0), 1);
	assert_ptr_equal(err_entry.op_context, ctx);
	assert_int_equal(err_entry.flags, flags);
	assert_int_equal(err_entry.err, err);
	assert_int_equal(err_entry.prov_errno, prov_errno);
}

static void test_proto_atomic_read_eq_error(
	struct efa_unit_test_atomic_ctx *ctx, int err, int prov_errno)
{
	struct fi_eq_err_entry err_entry = {0};

	assert_int_equal(fi_eq_readerr(ctx->resource->eq, &err_entry, 0),
			 sizeof(err_entry));
	assert_int_equal(err_entry.err, err);
	assert_int_equal(err_entry.prov_errno, prov_errno);
}

static void test_proto_atomic_fail_request(
	struct efa_unit_test_atomic_ctx *ctx, struct efa_rdm_pke *req_pke,
	uint64_t cq_flags, uint32_t next_msg_id)
{
	efa_rdm_pke_handle_tx_error(
		req_pke, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);
	test_proto_atomic_read_error(
		ctx, FI_EHOSTUNREACH,
		EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE, cq_flags);
	test_proto_atomic_assert_clean(ctx, next_msg_id);
	test_proto_atomic_assert_cq_empty(ctx);
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
	test_proto_atomic_read_success(&ctx, exact, FI_ATOMIC | FI_WRITE);
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
			test_proto_atomic_read_success(&ctx,
						       EFA_UNIT_TEST_ATOMIC_LEN,
						       FI_ATOMIC | FI_WRITE);
		} else {
			efa_rdm_proto_handle_receipt_recv(receipt_pke);
			assert_true(txe->internal_flags &
				    EFA_RDM_TXE_REMOTE_ACK_RECEIVED);
			assert_int_equal(efa_unit_test_get_ope_list_length(
						 ctx.ep, EFA_RDM_TXE), 1);
			test_proto_atomic_read_success(&ctx,
						       EFA_UNIT_TEST_ATOMIC_LEN,
						       FI_ATOMIC | FI_WRITE);
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
		EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE,
		FI_ATOMIC | FI_WRITE);
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
	test_proto_atomic_read_error(&ctx, FI_EIO, FI_EFA_ERR_PKT_POST,
				     FI_ATOMIC | FI_WRITE);
	test_proto_atomic_assert_clean(&ctx, ctx.next_msg_id + 2);
	test_proto_atomic_assert_cq_empty(&ctx);
	efa_unit_test_buff_destruct(&ctx.send_buff);
}

void test_proto_atomic_write_responder_rejects_invalid_request(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_atomic_ctx ctx;
	struct efa_rdm_rta_hdr *rta_hdr;
	struct efa_rdm_pke *req_pke, *req_rx;
	const struct {
		uint32_t op;
		uint32_t datatype;
		uint64_t flags;
	} invalid_hdrs[] = {
		{FI_ATOMIC_READ, FI_UINT8, FI_COMPLETION},
		{FI_BOR, FI_FLOAT, FI_COMPLETION},
		{OFI_SWAP_OP_LAST, FI_UINT8,
		 FI_COMPLETION | FI_DELIVERY_COMPLETE},
		{FI_SUM, OFI_DATATYPE_CNT, FI_COMPLETION},
	};
	uint8_t *target, *source;
	uint32_t next_msg_id;
	int submitted_before;
	size_t i;

	test_proto_atomic_setup(resource, &ctx);
	target = ctx.send_buff.buff;
	source = target + 16;
	ctx.ioc.addr = source;
	ctx.rma_ioc.addr = (uint64_t) target;
	g_efa_unit_test_mocks.efa_qp_post_send =
		efa_mock_efa_qp_post_send_return_mock;
	will_return_int_always(efa_mock_efa_qp_post_send_return_mock, 0);

	for (i = 0; i < ARRAY_SIZE(invalid_hdrs); i++) {
		target[0] = 1;
		source[0] = 2;
		next_msg_id = ctx.peer->next_msg_id;

		assert_int_equal(
			test_proto_atomic_post(
				&ctx, 1, invalid_hdrs[i].flags),
			0);
		req_pke = test_proto_atomic_posted_pke(&ctx);
		req_rx = test_proto_atomic_request_rx(&ctx, req_pke);
		rta_hdr = efa_rdm_pke_get_rta_hdr(req_rx);
		rta_hdr->atomic_op = invalid_hdrs[i].op;
		rta_hdr->atomic_datatype = invalid_hdrs[i].datatype;
		submitted_before = g_ibv_submitted_wr_id_cnt;

		if (invalid_hdrs[i].flags & FI_DELIVERY_COMPLETE)
			assert_int_equal(
				efa_rdm_pke_proc_dc_write_rta(req_rx),
				-FI_EINVAL);
		else
			assert_int_equal(
				efa_rdm_pke_proc_write_rta(req_rx),
				-FI_EINVAL);
		assert_int_equal(target[0], 1);
		assert_int_equal(g_ibv_submitted_wr_id_cnt, submitted_before);
		test_proto_atomic_read_eq_error(
			&ctx, FI_EINVAL, FI_EFA_ERR_INVALID_DATATYPE);
		test_proto_atomic_fail_request(
			&ctx, req_pke, FI_ATOMIC | FI_WRITE,
			next_msg_id + 1);
	}

	/* Keep one case for WRITE_RTA's distinct RMA verification path. */
	target[0] = 1;
	source[0] = 2;
	/* EFA never assigns FI_KEY_NOTAVAIL to a registered MR. */
	ctx.rma_ioc.key = FI_KEY_NOTAVAIL;
	next_msg_id = ctx.peer->next_msg_id;
	assert_int_equal(test_proto_atomic_post(&ctx, 1, FI_COMPLETION), 0);
	req_pke = test_proto_atomic_posted_pke(&ctx);
	submitted_before = g_ibv_submitted_wr_id_cnt;
	req_rx = test_proto_atomic_request_rx(&ctx, req_pke);
	assert_int_equal(efa_rdm_pke_proc_write_rta(req_rx), -FI_EINVAL);
	assert_int_equal(target[0], 1);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, submitted_before);
	test_proto_atomic_read_eq_error(
		&ctx, FI_EINVAL, FI_EFA_ERR_RMA_ADDR);
	efa_rdm_pke_handle_send_completion(req_pke);
	test_proto_atomic_read_success(&ctx, 1, FI_ATOMIC | FI_WRITE);
	test_proto_atomic_assert_clean(&ctx, next_msg_id + 1);
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
	test_proto_atomic_read_success(ctx, len, FI_ATOMIC | FI_WRITE);
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
	test_proto_atomic_read_success(&ctx, EFA_UNIT_TEST_ATOMIC_LEN,
				       FI_ATOMIC | FI_WRITE);

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

void test_proto_atomic_fetch_compare_request(void **state) {
	struct efa_resource *resource = *state;
	const struct efa_rdm_atomic_proto *protos[] = {
		&efa_rdm_atomic_fetch_proto,
		&efa_rdm_atomic_compare_proto,
	};
	enum fi_op ops[] = {FI_SUM, FI_CSWAP};
	int pkt_types[] = {EFA_RDM_FETCH_RTA_PKT, EFA_RDM_COMPARE_RTA_PKT};
	struct efa_unit_test_atomic_ctx ctx;
	struct efa_rdm_ope probe = {0};
	struct efa_rdm_pke *pke;
	struct efa_rdm_rta_hdr *hdr;
	uint8_t *source, *compare, *result;
	size_t capacity, count, hdr_size, i;
	uint32_t next_msg_id, tx_id;
	int post_count;

	test_proto_atomic_setup(resource, &ctx);
	source = ctx.send_buff.buff;
	compare = source + ctx.ep->mtu_size;
	result = compare + ctx.ep->mtu_size;
	probe.ep = ctx.ep;
	probe.peer = ctx.peer;
	probe.rma_iov_count = 1;
	next_msg_id = ctx.next_msg_id;
	g_efa_unit_test_mocks.efa_qp_post_send =
		efa_mock_efa_qp_post_send_return_mock;
	will_return_int_always(efa_mock_efa_qp_post_send_return_mock, 0);

	for (i = 0; i < ARRAY_SIZE(ops); i++) {
		capacity = efa_rdm_txe_max_req_data_capacity(ctx.ep, &probe,
							     pkt_types[i]);
		count = i ? capacity / 2 : capacity;
		memset(source, 0x10 + i, count);
		memset(compare, 0x20 + i, count);
		memset(result, 0, count);
		assert_int_equal(test_proto_atomic_fetch_compare_post(
					 &ctx, ops[i], count, compare, result,
					 FI_COMPLETION),
				 0);
		pke = test_proto_atomic_posted_pke(&ctx);
		hdr = efa_rdm_pke_get_rta_hdr(pke);
		hdr_size = efa_rdm_pke_get_req_hdr_size(pke);
		assert_int_equal(efa_rdm_pkt_type_of(pke), pkt_types[i]);
		assert_ptr_equal(pke->ope->atomic_proto, protos[i]);
		assert_null(pke->ope->proto);
		assert_ptr_equal(
			pke->handle_pke,
			pke->ope->atomic_proto->handle_send_completion);
		assert_int_equal(hdr->recv_id, pke->ope->tx_id);
		assert_int_equal(hdr->rma_iov[0].len, count);
		assert_memory_equal(pke->wiredata + hdr_size, source, count);
		if (i)
			assert_memory_equal(pke->wiredata + hdr_size + count,
					    compare, count);
		assert_int_equal(pke->pkt_size, hdr_size + count * (i + 1));

		tx_id = pke->ope->tx_id;
		efa_rdm_pke_handle_send_completion(pke);
		efa_rdm_pke_handle_atomrsp_recv(
			test_proto_atomic_atomrsp(&ctx, tx_id, compare, count));
		test_proto_atomic_read_success(&ctx, count,
					       FI_ATOMIC | FI_READ);
		test_proto_atomic_assert_clean(&ctx, ++next_msg_id);

		post_count = g_ibv_submitted_wr_id_cnt;
		assert_int_equal(test_proto_atomic_fetch_compare_post(
					 &ctx, ops[i], count + 1, compare,
					 result, FI_COMPLETION),
				 -FI_EMSGSIZE);
		assert_int_equal(g_ibv_submitted_wr_id_cnt, post_count);
		test_proto_atomic_assert_clean(&ctx, next_msg_id);
		test_proto_atomic_assert_cq_empty(&ctx);
	}
	efa_unit_test_buff_destruct(&ctx.send_buff);
}

void test_proto_atomic_fetch_compare_completion(void **state) {
	struct efa_resource *resource = *state;
	struct efa_unit_test_atomic_ctx ctx;
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *req_pke, *atomrsp_pke;
	uint8_t *source, *compare, *result;
	uint8_t old[EFA_UNIT_TEST_ATOMIC_LEN];
	struct fi_eq_err_entry eq_err_entry = {0};
	uint32_t next_msg_id, tx_id;

	test_proto_atomic_setup(resource, &ctx);
	source = ctx.send_buff.buff;
	compare = source + ctx.ep->mtu_size;
	result = compare + ctx.ep->mtu_size;
	memset(source, 0x11, sizeof(old));
	memset(compare, 0x22, sizeof(old));
	memset(old, 0x33, sizeof(old));
	next_msg_id = ctx.next_msg_id;
	g_efa_unit_test_mocks.efa_qp_post_send =
		efa_mock_efa_qp_post_send_return_mock;
	will_return_int_always(efa_mock_efa_qp_post_send_return_mock, 0);

	memset(result, 0, sizeof(old));
	assert_int_equal(test_proto_atomic_fetch_compare_post(
				 &ctx, FI_CSWAP, sizeof(old), compare, result,
				 FI_COMPLETION),
			 0);
	req_pke = test_proto_atomic_posted_pke(&ctx);
	txe = req_pke->ope;
	efa_rdm_pke_handle_atomrsp_recv(test_proto_atomic_atomrsp(
		&ctx, txe->tx_id, old, sizeof(old)));
	assert_memory_equal(result, old, sizeof(old));
	test_proto_atomic_read_success(&ctx, sizeof(old),
				       FI_ATOMIC | FI_READ);
	assert_int_equal(
		efa_unit_test_get_ope_list_length(ctx.ep, EFA_RDM_TXE), 1);
	efa_rdm_pke_handle_send_completion(req_pke);
	test_proto_atomic_assert_clean(&ctx, ++next_msg_id);
	test_proto_atomic_assert_cq_empty(&ctx);

	memset(result, 0, sizeof(old));
	assert_int_equal(test_proto_atomic_fetch_compare_post(
				 &ctx, FI_SUM, sizeof(old), compare, result,
				 FI_COMPLETION),
			 0);
	req_pke = test_proto_atomic_posted_pke(&ctx);
	txe = req_pke->ope;
	efa_rdm_pke_handle_atomrsp_recv(test_proto_atomic_atomrsp(
		&ctx, txe->tx_id, old, sizeof(old) - 1));
	assert_true(txe->internal_flags & EFA_RDM_TXE_REMOTE_ACK_RECEIVED);
	assert_memory_equal(result, (uint8_t[sizeof(old)]) {0}, sizeof(old));
	test_proto_atomic_read_error(
		&ctx, FI_ETRUNC, EFA_IO_COMP_STATUS_LOCAL_ERROR_BAD_LENGTH,
		FI_ATOMIC | FI_READ);
	efa_rdm_pke_handle_send_completion(req_pke);
	test_proto_atomic_assert_clean(&ctx, ++next_msg_id);
	test_proto_atomic_assert_cq_empty(&ctx);

	memset(result, 0, sizeof(old));
	assert_int_equal(test_proto_atomic_fetch_compare_post(
				 &ctx, FI_CSWAP, sizeof(old), compare, result,
				 FI_COMPLETION),
			 0);
	req_pke = test_proto_atomic_posted_pke(&ctx);
	tx_id = req_pke->ope->tx_id;
	efa_rdm_pke_handle_send_completion(req_pke);
	atomrsp_pke = test_proto_atomic_atomrsp(
		&ctx, tx_id, old, sizeof(old));
	atomrsp_pke->pkt_size--;
	efa_rdm_pke_handle_atomrsp_recv(atomrsp_pke);
	assert_memory_equal(result, (uint8_t[sizeof(old)]) {0}, sizeof(old));
	test_proto_atomic_read_error(
		&ctx, FI_ETRUNC, EFA_IO_COMP_STATUS_LOCAL_ERROR_BAD_LENGTH,
		FI_ATOMIC | FI_READ);
	test_proto_atomic_assert_clean(&ctx, ++next_msg_id);
	test_proto_atomic_assert_cq_empty(&ctx);

	assert_int_equal(test_proto_atomic_fetch_compare_post(
				 &ctx, FI_SUM, sizeof(old), compare, result,
				 FI_COMPLETION),
			 0);
	req_pke = test_proto_atomic_posted_pke(&ctx);
	efa_rdm_pke_handle_atomrsp_recv(test_proto_atomic_atomrsp(
		&ctx, req_pke->ope->tx_id, old, sizeof(old)));
	test_proto_atomic_read_success(&ctx, sizeof(old),
				       FI_ATOMIC | FI_READ);
	efa_rdm_pke_handle_tx_error(
		req_pke, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);
	test_proto_atomic_assert_clean(&ctx, ++next_msg_id);
	test_proto_atomic_assert_cq_empty(&ctx);

	memset(result, 0, sizeof(old));
	assert_int_equal(test_proto_atomic_fetch_compare_post(
				 &ctx, FI_CSWAP, sizeof(old), compare, result,
				 FI_COMPLETION),
			 0);
	req_pke = test_proto_atomic_posted_pke(&ctx);
	tx_id = req_pke->ope->tx_id;
	efa_rdm_pke_handle_tx_error(
		req_pke, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);
	test_proto_atomic_read_error(
		&ctx, FI_EHOSTUNREACH,
		EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE,
		FI_ATOMIC | FI_READ);
	efa_rdm_pke_handle_atomrsp_recv(test_proto_atomic_atomrsp(
		&ctx, tx_id, old, sizeof(old)));
	assert_memory_equal(result, (uint8_t[sizeof(old)]) {0}, sizeof(old));
	assert_int_equal(fi_eq_readerr(resource->eq, &eq_err_entry, 0),
			 -FI_EAGAIN);
	test_proto_atomic_assert_clean(&ctx, ++next_msg_id);
	test_proto_atomic_assert_cq_empty(&ctx);

	atomrsp_pke = test_proto_atomic_atomrsp(
		&ctx, UINT32_MAX, old, 0);
	atomrsp_pke->pkt_size =
		sizeof(struct efa_rdm_atomrsp_hdr) - 1;
	efa_rdm_pke_handle_atomrsp_recv(atomrsp_pke);
	test_proto_atomic_read_eq_error(
		&ctx, FI_ETRUNC,
		EFA_IO_COMP_STATUS_LOCAL_ERROR_BAD_LENGTH);
	test_proto_atomic_assert_clean(&ctx, next_msg_id);
	test_proto_atomic_assert_cq_empty(&ctx);
	efa_unit_test_buff_destruct(&ctx.send_buff);
}

void test_proto_atomic_responder_post_error(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_atomic_ctx ctx;
	enum {
		TEST_PROTO_ATOMIC_DIRECT_FAILURE,
		TEST_PROTO_ATOMIC_QUEUED_FAILURE,
		TEST_PROTO_ATOMIC_ASYNC_FAILURE,
	};
	const struct {
		enum fi_op op;
		int failure;
	} cases[] = {
		{FI_SUM, TEST_PROTO_ATOMIC_DIRECT_FAILURE},
		{FI_CSWAP, TEST_PROTO_ATOMIC_QUEUED_FAILURE},
		{FI_SUM, TEST_PROTO_ATOMIC_ASYNC_FAILURE},
	};
	struct efa_rdm_pke *req_pke, *req_rx, *resp_pke;
	struct efa_rdm_ope *rxe;
	uint8_t *target, *source, *compare, *result;
	uint32_t next_msg_id;
	size_t i;
	int ret;

	test_proto_atomic_setup(resource, &ctx);
	target = ctx.send_buff.buff;
	source = target + 16;
	compare = source + 16;
	result = compare + 16;
	ctx.ioc.addr = source;
	ctx.rma_ioc.addr = (uint64_t) target;
	ctx.rma_ioc.key = fi_mr_key(ctx.send_buff.mr);
	g_efa_unit_test_mocks.efa_qp_post_send =
		efa_mock_efa_qp_post_send_return_mock;

	for (i = 0; i < ARRAY_SIZE(cases); i++) {
		target[0] = compare[0] = 1;
		source[0] = 2;
		result[0] = 0;
		next_msg_id = ctx.peer->next_msg_id;
		will_return_int(efa_mock_efa_qp_post_send_return_mock, 0);
		switch (cases[i].failure) {
		case TEST_PROTO_ATOMIC_DIRECT_FAILURE:
			will_return_int(
				efa_mock_efa_qp_post_send_return_mock, EIO);
			break;
		case TEST_PROTO_ATOMIC_QUEUED_FAILURE:
			will_return_int(
				efa_mock_efa_qp_post_send_return_mock, ENOMEM);
			will_return_int(
				efa_mock_efa_qp_post_send_return_mock, EIO);
			break;
		case TEST_PROTO_ATOMIC_ASYNC_FAILURE:
			will_return_int(
				efa_mock_efa_qp_post_send_return_mock, 0);
			break;
		}

		assert_int_equal(test_proto_atomic_fetch_compare_post(
					 &ctx, cases[i].op, 1, compare, result,
					 FI_COMPLETION),
				 0);
		req_pke = test_proto_atomic_posted_pke(&ctx);
		req_rx = test_proto_atomic_request_rx(&ctx, req_pke);
		ret = test_proto_atomic_process_fetch_compare(
			req_rx, cases[i].op);

		switch (cases[i].failure) {
		case TEST_PROTO_ATOMIC_DIRECT_FAILURE:
			assert_int_equal(ret, -FI_EIO);
			test_proto_atomic_read_eq_error(
				&ctx, FI_EIO, FI_EFA_ERR_PKT_POST);
			break;
		case TEST_PROTO_ATOMIC_QUEUED_FAILURE:
			assert_int_equal(ret, 0);
			assert_false(dlist_empty(&ctx.ep->ope_queued_list));
			rxe = container_of(ctx.ep->ope_queued_list.next,
					   struct efa_rdm_ope, queued_entry);
			assert_true(rxe->internal_flags &
				    EFA_RDM_OPE_QUEUED_CTRL);
			assert_non_null(rxe->atomrsp_data);
			assert_int_equal(((uint8_t *) rxe->atomrsp_data)[0], 1);
			assert_int_equal(
				efa_rdm_ope_process_queued_ope(rxe), -FI_EIO);
			test_proto_atomic_read_eq_error(
				&ctx, FI_EIO, FI_EFA_ERR_PKT_POST);
			break;
		case TEST_PROTO_ATOMIC_ASYNC_FAILURE:
			assert_int_equal(ret, 0);
			resp_pke = test_proto_atomic_posted_pke(&ctx);
			assert_int_equal(
				efa_rdm_pkt_type_of(resp_pke),
				EFA_RDM_ATOMRSP_PKT);
			efa_rdm_pke_handle_tx_error(
				resp_pke,
				EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);
			test_proto_atomic_read_eq_error(
				&ctx, FI_EHOSTUNREACH,
				EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);
			break;
		}

		assert_int_equal(
			efa_unit_test_get_ope_list_length(
				ctx.ep, EFA_RDM_RXE),
			0);
		assert_int_equal(
			test_proto_atomic_bufpool_free_count(
				ctx.ep->rx_atomrsp_pool),
			ctx.atomrsp_free);
		assert_true(dlist_empty(&ctx.ep->ope_queued_list));
		test_proto_atomic_fail_request(
			&ctx, req_pke, FI_ATOMIC | FI_READ,
			next_msg_id + 1);
	}

	efa_unit_test_buff_destruct(&ctx.send_buff);
}

static ssize_t test_proto_atomic_unexpected_source_copy(void **desc, char *buff,
							size_t buff_size,
							const struct iovec *iov,
							size_t iov_count) {
	(void) desc;
	(void) buff;
	(void) buff_size;
	(void) iov;
	(void) iov_count;
	fail_msg("FI_ATOMIC_READ copied its source");
	return -FI_EIO;
}

void test_proto_atomic_read_null_source(void **state) {
	struct efa_resource *resource = *state;
	struct efa_unit_test_atomic_ctx ctx;
	struct efa_rdm_pke *pke;
	struct efa_rdm_ope *txe;
	struct fi_ioc result_ioc;
	char *result;
	uint8_t old[EFA_UNIT_TEST_ATOMIC_LEN];
	void *result_desc;
	uint32_t next_msg_id;
	size_t hdr_size;
	int api;

	test_proto_atomic_setup(resource, &ctx);
	result = (char *) ctx.send_buff.buff + 2 * ctx.ep->mtu_size;
	result_ioc.addr = result;
	result_ioc.count = sizeof(old);
	result_desc = ctx.desc;
	memset(old, 0x5a, sizeof(old));
	ctx.rma_ioc.count = sizeof(old);
	next_msg_id = ctx.next_msg_id;
	g_efa_unit_test_mocks.efa_copy_from_hmem_iov =
		test_proto_atomic_unexpected_source_copy;
	g_efa_unit_test_mocks.efa_qp_post_send =
		efa_mock_efa_qp_post_send_return_mock;
	will_return_int_always(efa_mock_efa_qp_post_send_return_mock, 0);

	for (api = 0; api < 2; api++) {
		memset(result, 0, sizeof(old));
		if (api == 0) {
			assert_int_equal(
				fi_fetch_atomic(
					resource->ep, NULL, sizeof(old), NULL,
					result, result_desc, ctx.peer_addr,
					EFA_UNIT_TEST_ATOMIC_REMOTE_ADDR,
					EFA_UNIT_TEST_ATOMIC_REMOTE_KEY,
					FI_UINT8, FI_ATOMIC_READ, &ctx),
				0);
		} else {
			assert_int_equal(
				fi_fetch_atomicv(
					resource->ep, NULL, NULL, 0,
					&result_ioc, &result_desc, 1,
					ctx.peer_addr,
					EFA_UNIT_TEST_ATOMIC_REMOTE_ADDR,
					EFA_UNIT_TEST_ATOMIC_REMOTE_KEY,
					FI_UINT8, FI_ATOMIC_READ, &ctx),
				0);
		}
		pke = test_proto_atomic_posted_pke(&ctx);
		txe = pke->ope;
		hdr_size = efa_rdm_pke_get_req_hdr_size(pke);
		assert_ptr_equal(txe->atomic_proto,
				 &efa_rdm_atomic_fetch_proto);
		assert_int_equal(txe->total_len, sizeof(old));
		assert_int_equal(txe->cq_entry.len, sizeof(old));
		assert_memory_equal(pke->wiredata + hdr_size,
				    (uint8_t[sizeof(old)]) {0}, sizeof(old));
		efa_rdm_pke_handle_send_completion(pke);
		efa_rdm_pke_handle_atomrsp_recv(test_proto_atomic_atomrsp(
			&ctx, txe->tx_id, old, sizeof(old)));
		assert_memory_equal(result, old, sizeof(old));
		test_proto_atomic_read_success(&ctx, sizeof(old),
					       FI_ATOMIC | FI_READ);
		test_proto_atomic_assert_clean(&ctx, ++next_msg_id);
	}
	efa_unit_test_buff_destruct(&ctx.send_buff);
}

static int test_proto_atomic_hmem_copy_from_calls;
static int test_proto_atomic_hmem_copy_to_calls;

static int test_proto_atomic_hmem_copy_from(uint64_t device, void *dest,
					    const void *src, size_t size)
{
	(void) device;
	test_proto_atomic_hmem_copy_from_calls++;
	memcpy(dest, src, size);
	return FI_SUCCESS;
}

static int test_proto_atomic_hmem_copy_to(uint64_t device, void *dest,
					  const void *src, size_t size)
{
	(void) device;
	(void) dest;
	(void) src;
	(void) size;
	test_proto_atomic_hmem_copy_to_calls++;
	return -FI_EIO;
}

void test_proto_atomic_compare_responder_iov(void **state) {
	struct efa_resource *resource = *state;
	struct efa_unit_test_atomic_ctx ctx;
	struct fi_ioc source_iov[2], compare_iov[2], result_iov[2];
	struct fi_rma_ioc rma_iov[2];
	struct fi_msg_atomic msg;
	struct efa_rdm_atomrsp_pkt *atomrsp;
	struct efa_rdm_pke *req_pke, *req_rx, *resp_pke, *resp_rx;
	void *desc[2], *compare_desc[2], *result_desc[2];
	uint8_t old[] = {1, 2}, updated[] = {11, 22};
	uint8_t *target, *source, *compare, *result;
	uint64_t key;
	int submitted_before;

	test_proto_atomic_setup(resource, &ctx);
	ctx.ep->base_ep.info->tx_attr->rma_iov_limit = 2;
	target = ctx.send_buff.buff;
	source = target + 16;
	compare = source + 16;
	result = compare + 16;
	target[0] = compare[0] = 1;
	target[1] = compare[1] = 2;
	source[0] = 11;
	source[1] = 22;
	result[0] = result[1] = 0;
	key = fi_mr_key(ctx.send_buff.mr);
	for (int i = 0; i < 2; i++) {
		source_iov[i] =
			(struct fi_ioc) {.addr = source + i, .count = 1};
		compare_iov[i] =
			(struct fi_ioc) {.addr = compare + i, .count = 1};
		result_iov[i] =
			(struct fi_ioc) {.addr = result + i, .count = 1};
		rma_iov[i] =
			(struct fi_rma_ioc) {.addr = (uint64_t) (target + i),
					     .count = 1,
					     .key = key};
		desc[i] = compare_desc[i] = result_desc[i] = ctx.desc;
	}
	msg = ctx.msg;
	msg.msg_iov = source_iov;
	msg.desc = desc;
	msg.iov_count = 2;
	msg.rma_iov = rma_iov;
	msg.rma_iov_count = 2;
	msg.op = FI_CSWAP;
	g_efa_unit_test_mocks.efa_qp_post_send =
		efa_mock_efa_qp_post_send_return_mock;
	will_return_int_always(efa_mock_efa_qp_post_send_return_mock, 0);

	assert_int_equal(fi_compare_atomicmsg(resource->ep, &msg, compare_iov,
					      compare_desc, 2, result_iov,
					      result_desc, 2, FI_COMPLETION),
			 0);
	req_pke = test_proto_atomic_posted_pke(&ctx);
	req_rx = test_proto_atomic_request_rx(&ctx, req_pke);
	submitted_before = g_ibv_submitted_wr_id_cnt;
	efa_rdm_pke_proc_received(req_rx);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, submitted_before + 1);
	resp_pke = test_proto_atomic_posted_pke(&ctx);
	assert_int_equal(efa_rdm_pkt_type_of(resp_pke), EFA_RDM_ATOMRSP_PKT);
	atomrsp = (struct efa_rdm_atomrsp_pkt *) resp_pke->wiredata;
	assert_memory_equal(atomrsp->data, old, sizeof(old));
	assert_memory_equal(target, updated, sizeof(updated));
	resp_rx = test_proto_atomic_atomrsp(&ctx, req_pke->ope->tx_id,
					    atomrsp->data,
					    atomrsp->hdr.seg_length);
	efa_rdm_pke_handle_send_completion(resp_pke);
	efa_rdm_pke_handle_send_completion(req_pke);
	efa_rdm_pke_handle_atomrsp_recv(resp_rx);
	assert_memory_equal(result, old, sizeof(old));
	test_proto_atomic_read_success(&ctx, 2, FI_ATOMIC | FI_READ);
	test_proto_atomic_assert_clean(&ctx, ctx.next_msg_id + 1);
	efa_unit_test_buff_destruct(&ctx.send_buff);
}

static void test_proto_atomic_responder_rejects_rma(
	struct efa_unit_test_atomic_ctx *ctx, enum fi_op op, int expected_err)
{
	struct efa_rdm_pke *req_pke, *req_rx;
	uint8_t *target, *source, *compare, *result;
	uint32_t next_msg_id;
	int submitted_before;

	target = ctx->send_buff.buff;
	source = target + 16;
	compare = source + 16;
	result = compare + 16;
	target[0] = compare[0] = 1;
	source[0] = 2;
	result[0] = 0;
	ctx->ioc.addr = source;
	ctx->rma_ioc.addr = (uint64_t) target;
	next_msg_id = ctx->peer->next_msg_id;

	assert_int_equal(test_proto_atomic_fetch_compare_post(
				 ctx, op, 1, compare, result, FI_COMPLETION),
			 0);
	req_pke = test_proto_atomic_posted_pke(ctx);
	submitted_before = g_ibv_submitted_wr_id_cnt;
	req_rx = test_proto_atomic_request_rx(ctx, req_pke);

	assert_int_equal(
		test_proto_atomic_process_fetch_compare(req_rx, op),
		-expected_err);
	assert_int_equal(target[0], 1);
	assert_int_equal(result[0], 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, submitted_before);
	test_proto_atomic_read_eq_error(
		ctx, expected_err, FI_EFA_ERR_RMA_ADDR);
	test_proto_atomic_fail_request(
		ctx, req_pke, FI_ATOMIC | FI_READ, next_msg_id + 1);
}

struct test_proto_atomic_invalid_hdr {
	uint32_t op;
	uint32_t datatype;
};

static void test_proto_atomic_responder_rejects_invalid_hdr(
	struct efa_unit_test_atomic_ctx *ctx, enum fi_op request_op,
	const struct test_proto_atomic_invalid_hdr *invalid_hdrs,
	size_t invalid_hdr_count)
{
	struct efa_rdm_rta_hdr *rta_hdr;
	struct efa_rdm_pke *req_pke, *req_rx;
	uint8_t *target, *source, *compare, *result;
	uint32_t next_msg_id;
	int submitted_before;
	size_t i;

	target = ctx->send_buff.buff;
	source = target + 16;
	compare = source + 16;
	result = compare + 16;
	ctx->ioc.addr = source;
	ctx->rma_ioc.addr = (uint64_t) target;

	for (i = 0; i < invalid_hdr_count; i++) {
		target[0] = compare[0] = 1;
		source[0] = 2;
		result[0] = 0;
		next_msg_id = ctx->peer->next_msg_id;

		assert_int_equal(test_proto_atomic_fetch_compare_post(
					 ctx, request_op, 1, compare, result,
					 FI_COMPLETION),
				 0);
		req_pke = test_proto_atomic_posted_pke(ctx);
		req_rx = test_proto_atomic_request_rx(ctx, req_pke);
		rta_hdr = efa_rdm_pke_get_rta_hdr(req_rx);
		rta_hdr->atomic_op = invalid_hdrs[i].op;
		rta_hdr->atomic_datatype = invalid_hdrs[i].datatype;
		submitted_before = g_ibv_submitted_wr_id_cnt;

		assert_int_equal(
			test_proto_atomic_process_fetch_compare(
				req_rx, request_op),
			-FI_EINVAL);
		assert_int_equal(target[0], 1);
		assert_int_equal(result[0], 0);
		assert_int_equal(g_ibv_submitted_wr_id_cnt, submitted_before);
		test_proto_atomic_read_eq_error(ctx, FI_EINVAL,
						FI_EFA_ERR_INVALID_DATATYPE);
		test_proto_atomic_fail_request(
			ctx, req_pke, FI_ATOMIC | FI_READ,
			next_msg_id + 1);
	}
}

void test_proto_atomic_fetch_compare_responder_rejects_invalid_op_datatype(
	void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_atomic_ctx ctx;
	const struct test_proto_atomic_invalid_hdr fetch_invalid_hdrs[] = {
		{FI_CSWAP, FI_UINT8},
		{FI_BOR, FI_FLOAT},
		{OFI_SWAP_OP_LAST, FI_UINT8},
		{FI_SUM, OFI_DATATYPE_CNT},
	};
	const struct test_proto_atomic_invalid_hdr compare_invalid_hdrs[] = {
		{FI_SUM, FI_UINT8},
		{FI_MSWAP, FI_FLOAT},
	};

	test_proto_atomic_setup(resource, &ctx);
	g_efa_unit_test_mocks.efa_qp_post_send =
		efa_mock_efa_qp_post_send_return_mock;
	will_return_int_always(efa_mock_efa_qp_post_send_return_mock, 0);

	test_proto_atomic_responder_rejects_invalid_hdr(
		&ctx, FI_SUM, fetch_invalid_hdrs,
		ARRAY_SIZE(fetch_invalid_hdrs));
	test_proto_atomic_responder_rejects_invalid_hdr(
		&ctx, FI_CSWAP, compare_invalid_hdrs,
		ARRAY_SIZE(compare_invalid_hdrs));
	efa_unit_test_buff_destruct(&ctx.send_buff);
}

void test_proto_atomic_read_responder_accepts_read_only_mr(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_atomic_ctx ctx;
	struct efa_rdm_atomrsp_pkt *atomrsp;
	struct efa_rdm_pke *req_pke, *req_rx, *resp_pke, *resp_rx;
	struct efa_mr *efa_mr;
	enum fi_hmem_iface saved_iface;
	int (*saved_copy_from)(uint64_t, void *, const void *, size_t);
	int (*saved_copy_to)(uint64_t, void *, const void *, size_t);
	uint8_t *target, *result;
	uint32_t next_msg_id;
	int ret;

	test_proto_atomic_setup_with_mr_access(
		resource, &ctx, FI_REMOTE_READ);
	target = ctx.send_buff.buff;
	result = target + 16;
	target[0] = 0x5a;
	result[0] = 0;
	ctx.rma_ioc.addr = (uint64_t) target;
	ctx.rma_ioc.key = fi_mr_key(ctx.send_buff.mr);
	g_efa_unit_test_mocks.efa_qp_post_send =
		efa_mock_efa_qp_post_send_return_mock;
	will_return_int_always(efa_mock_efa_qp_post_send_return_mock, 0);

	test_proto_atomic_responder_rejects_rma(&ctx, FI_SUM, FI_EACCES);
	test_proto_atomic_responder_rejects_rma(&ctx, FI_CSWAP, FI_EACCES);
	next_msg_id = ctx.peer->next_msg_id;
	target[0] = 0x5a;
	result[0] = 0;

	assert_int_equal(test_proto_atomic_fetch_compare_post(
				 &ctx, FI_ATOMIC_READ, 1, NULL, result,
				 FI_COMPLETION),
			 0);
	req_pke = test_proto_atomic_posted_pke(&ctx);
	req_rx = test_proto_atomic_request_rx(&ctx, req_pke);
	efa_mr = container_of(ctx.send_buff.mr, struct efa_mr, mr_fid);
	saved_iface = efa_mr->iface;
	saved_copy_from = hmem_ops[FI_HMEM_CUDA].copy_from_hmem;
	saved_copy_to = hmem_ops[FI_HMEM_CUDA].copy_to_hmem;
	efa_mr->iface = FI_HMEM_CUDA;
	hmem_ops[FI_HMEM_CUDA].copy_from_hmem =
		test_proto_atomic_hmem_copy_from;
	hmem_ops[FI_HMEM_CUDA].copy_to_hmem =
		test_proto_atomic_hmem_copy_to;
	test_proto_atomic_hmem_copy_from_calls = 0;
	test_proto_atomic_hmem_copy_to_calls = 0;

	ret = efa_rdm_pke_proc_fetch_rta(req_rx);

	efa_mr->iface = saved_iface;
	hmem_ops[FI_HMEM_CUDA].copy_from_hmem = saved_copy_from;
	hmem_ops[FI_HMEM_CUDA].copy_to_hmem = saved_copy_to;
	assert_int_equal(ret, 0);
	assert_int_equal(test_proto_atomic_hmem_copy_from_calls, 1);
	assert_int_equal(test_proto_atomic_hmem_copy_to_calls, 0);

	resp_pke = test_proto_atomic_posted_pke(&ctx);
	assert_int_equal(efa_rdm_pkt_type_of(resp_pke),
			 EFA_RDM_ATOMRSP_PKT);
	atomrsp = (struct efa_rdm_atomrsp_pkt *) resp_pke->wiredata;
	assert_int_equal(atomrsp->data[0], target[0]);
	assert_int_equal(target[0], 0x5a);

	resp_rx = test_proto_atomic_atomrsp(
		&ctx, req_pke->ope->tx_id, atomrsp->data,
		atomrsp->hdr.seg_length);
	efa_rdm_pke_handle_send_completion(resp_pke);
	efa_rdm_pke_handle_send_completion(req_pke);
	efa_rdm_pke_handle_atomrsp_recv(resp_rx);
	assert_int_equal(result[0], 0x5a);
	test_proto_atomic_read_success(&ctx, 1, FI_ATOMIC | FI_READ);
	test_proto_atomic_assert_clean(&ctx, next_msg_id + 1);
	test_proto_atomic_assert_cq_empty(&ctx);
	efa_unit_test_buff_destruct(&ctx.send_buff);
}

/**
 * @brief Set up an endpoint and a peer that the runt read protocol can be used
 *        with, and register a source buffer larger than the runt.
 *
 * Runting has only been qualified for the HMEM interfaces, so host memory has a
 * runt size of 0 by default and would never select the protocol. Overriding the
 * FI_HMEM_SYSTEM thresholds keeps this coverage runnable on an instance with no
 * GPU; efa_unit_test_mocks_teardown() restores g_efa_hmem_info from its backup
 * after every test, so the overrides do not leak.
 *
 * The spoofed device generation does have to be put back by hand, via
 * restore_proto_runtread_device_version(), which the caller must call as soon as
 * protocol selection is done -- cmocka longjmps out of a failing assertion, so
 * restoring at the end of a test would leak into every later test.
 *
 * Skips the test on a platform whose device cannot do RDMA read, since the whole
 * protocol is predicated on the receiver reading the tail.
 */
static struct efa_rdm_ep *
setup_proto_runtread_test(struct efa_resource *resource,
			  struct efa_unit_test_buff *send_buff,
			  fi_addr_t *peer_addr, struct efa_rdm_peer **peer,
			  uint32_t *saved_vendor_part_id)
{
	struct efa_hmem_info *info = &g_efa_hmem_info[FI_HMEM_SYSTEM];
	struct efa_rdm_ep *ep;

	if (!efa_device_support_rdma_read()) {
		skip();
		return NULL;
	}

	ep = setup_proto_select_test(resource, peer_addr);
	efa_unit_test_buff_construct(send_buff, resource,
				     EFA_UNIT_TEST_PROTO_RUNTREAD_LEN);

	*peer = efa_rdm_ep_get_peer_explicit(ep, *peer_addr);
	(*peer)->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	(*peer)->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_RDMA_READ;

	/*
	 * efa_rdm_interop_rdma_read() also compares the two sides' device
	 * generations, so make them agree.
	 */
	*saved_vendor_part_id = g_efa_selected_device_list[0].ibv_attr.vendor_part_id;
	g_efa_selected_device_list[0].ibv_attr.vendor_part_id = 0xEFA1;
	(*peer)->device_version = 0xEFA1;
	ep->use_device_rdma = true;

	info->runt_size = EFA_UNIT_TEST_PROTO_RUNT_SIZE;
	info->min_read_msg_size = EFA_UNIT_TEST_PROTO_RUNTREAD_LEN;

	/* The medium protocol is tried first and must decline this size. */
	assert_true(EFA_UNIT_TEST_PROTO_RUNTREAD_LEN > info->max_medium_msg_size);

	return ep;
}

static void restore_proto_runtread_device_version(uint32_t saved_vendor_part_id)
{
	g_efa_selected_device_list[0].ibv_attr.vendor_part_id =
		saved_vendor_part_id;
}

/**
 * @brief Test that the runt read protocol is selected for a message past the
 *        interface's minimum read size when the source buffer is registered.
 */
void test_proto_select_runtread_for_large_msg(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	fi_addr_t peer_addr;
	uint32_t saved_vendor_part_id;
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	ep = setup_proto_runtread_test(resource, &send_buff, &peer_addr, &peer,
				       &saved_vendor_part_id);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.txe_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	restore_proto_runtread_device_version(saved_vendor_part_id);

	assert_int_equal(err, 0);
	assert_non_null(proto);
	assert_ptr_equal(proto, &efa_rdm_proto_runtread);

	ofi_buf_free(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief A delivery complete send never selects the runt read protocol.
 *
 * The runt read REQ has no delivery complete variant -- there is nowhere to
 * report the receipt against -- so a FI_DELIVERY_COMPLETE send has to use a
 * different protocol even when every other condition for runting holds.
 * Mainline makes the same call in efa_rdm_peer_select_readbase_rtm().
 *
 * Long read, which is what mainline falls back to, has not been migrated yet, so
 * today no protocol matches and the send takes the legacy path. Once long read
 * is registered this must select it instead of the runt read protocol.
 */
void test_proto_select_declines_runtread_for_delivery_complete(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	fi_addr_t peer_addr;
	uint32_t saved_vendor_part_id;
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	ep = setup_proto_runtread_test(resource, &send_buff, &peer_addr, &peer,
				       &saved_vendor_part_id);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.txe_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg,
						 FI_DELIVERY_COMPLETE, txe,
						 &proto);
	restore_proto_runtread_device_version(saved_vendor_part_id);

	assert_int_equal(err, 0);
	assert_ptr_not_equal(proto, &efa_rdm_proto_runtread);
	assert_null(proto);

	ofi_buf_free(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief The runt read protocol sends the runt over several REQ packets, each
 *        carrying the read iov array the receiver needs for the tail.
 *
 * The receiver reassembles the runt from msg_length and seg_offset and learns
 * where the rest of the message lives from runt_length plus the read iov array
 * that follows the header, so a gap in the segments or a missing iov silently
 * corrupts its copy.
 */
void test_proto_runtread_construct_pkes_carries_read_iov(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	struct efa_rdm_runtread_rtm_base_hdr *rtm_hdr;
	fi_addr_t peer_addr;
	uint32_t saved_vendor_part_id;
	struct fi_msg msg = {0};
	struct iovec iov;
	size_t i, expected_offset = 0;
	uint64_t pke_send_flags;
	int err;

	ep = setup_proto_runtread_test(resource, &send_buff, &peer_addr, &peer,
				       &saved_vendor_part_id);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.txe_pool);
	assert_non_null(txe);

	/* Drive the same sequence efa_rdm_msg_generic_send() drives. */
	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	assert_int_equal(err, 0);
	assert_ptr_equal(proto, &efa_rdm_proto_runtread);

	efa_rdm_proto_txe_fill(txe, ep, peer, &msg, ofi_op_msg, 0, 0, 0, proto);
	txe->msg_id = peer->next_msg_id++;

	err = proto->construct_tx_pkes(ep, txe, &pke_send_flags);
	restore_proto_runtread_device_version(saved_vendor_part_id);
	assert_int_equal(err, 0);

	/*
	 * The runt is only the head of the message; if it covered all of it the
	 * medium protocol should have won the selection.
	 */
	assert_int_equal(txe->bytes_runt, EFA_UNIT_TEST_PROTO_RUNT_SIZE);
	assert_true(txe->bytes_runt < txe->total_len);
	assert_true(ep->send_pkt_entry_vec_size > 1);
	assert_true(ep->send_pkt_entry_vec_size <= EFA_UNIT_TEST_PROTO_MAX_PKES);

	/*
	 * The peer-abort protocol reads txe->req_pkt_type to tell a two-sided RTM
	 * from an operation it does not handle.
	 */
	assert_int_equal(txe->req_pkt_type, EFA_RDM_RUNTREAD_MSGRTM_PKT);

	for (i = 0; i < ep->send_pkt_entry_vec_size; ++i) {
		struct efa_rdm_pke *pke = ep->send_pkt_entry_vec[i];
		struct fi_rma_iov *read_iov;

		assert_non_null(pke);
		assert_ptr_equal(
			pke->handle_pke,
			&efa_rdm_proto_runtread_handle_rtm_send_completion);
		assert_ptr_equal(pke->ope, txe);
		assert_true(pke->payload_size > 0);

		rtm_hdr = efa_rdm_pke_get_runtread_rtm_base_hdr(pke);
		assert_int_equal(rtm_hdr->msg_length, txe->total_len);
		assert_int_equal(rtm_hdr->runt_length, txe->bytes_runt);
		assert_int_equal(rtm_hdr->send_id, txe->tx_id);
		assert_int_equal(rtm_hdr->read_iov_count, txe->iov_count);
		assert_int_equal(rtm_hdr->seg_offset, expected_offset);

		/* The read iov array sits immediately after the REQ header. */
		read_iov = (struct fi_rma_iov *) (pke->wiredata +
						  efa_rdm_pke_get_req_hdr_size(pke));
		assert_int_equal(read_iov[0].addr, (uint64_t) send_buff.buff);
		assert_int_equal(read_iov[0].len, send_buff.size);
		assert_int_equal(read_iov[0].key, fi_mr_key(send_buff.mr));

		expected_offset += pke->payload_size;
	}

	/* The REQ packets carry exactly the runt, no more and no less. */
	assert_int_equal(expected_offset, txe->bytes_runt);

	for (i = 0; i < ep->send_pkt_entry_vec_size; ++i)
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[i]);
	efa_rdm_txe_release(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief The runt read protocol's construct_tx_pkes() and post-send hook are
 *        idempotent, so a txe queued before the handshake can be reposted.
 *
 * efa_rdm_msg_repost_rtm_proto() re-enters construct_tx_pkes() on a txe the
 * first attempt already set up, and efa_rdm_ope_process_queued_ope() retries a
 * txe that returned -FI_EAGAIN without clearing its queued state, so the second
 * call must produce exactly the same packets. This is a sharper hazard than for
 * the medium protocol: the runt size is computed from the peer's remaining runt
 * allowance, which the first attempt itself consumes, so recomputing it would
 * shrink the runt while the segment offsets already on the wire assume the old
 * one.
 *
 * The post-send hook must not double count either. bytes_sent is an assignment,
 * and the domain's read slot is guarded by EFA_RDM_TXE_READ_MSG_COUNTED so the
 * single release site cannot leave the counter stuck above zero -- which would
 * make every later message on the domain skip runting.
 *
 * peer->num_runt_bytes_in_flight is deliberately left as an accumulator: it is
 * balanced by the per-packet send completions, and in production the hook only
 * runs once per message because a repost only happens for an attempt whose
 * packets never reached the device.
 */
void test_proto_runtread_construct_pkes_is_idempotent(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL, *second_proto = NULL;
	struct efa_rdm_ope *second_txe;
	fi_addr_t peer_addr;
	uint32_t saved_vendor_part_id;
	struct fi_msg msg = {0};
	struct iovec iov;
	size_t i, first_pke_cnt, first_seg_offsets[EFA_UNIT_TEST_PROTO_MAX_PKES];
	size_t first_bytes_runt;
	uint64_t first_bytes_sent;
	uint32_t first_protocol;
	uint64_t pke_send_flags;
	int err;

	ep = setup_proto_runtread_test(resource, &send_buff, &peer_addr, &peer,
				       &saved_vendor_part_id);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.txe_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	assert_int_equal(err, 0);
	assert_ptr_equal(proto, &efa_rdm_proto_runtread);

	efa_rdm_proto_txe_fill(txe, ep, peer, &msg, ofi_op_msg, 0, 0, 0, proto);
	txe->msg_id = peer->next_msg_id++;

	err = proto->construct_tx_pkes(ep, txe, &pke_send_flags);
	assert_int_equal(err, 0);

	first_pke_cnt = ep->send_pkt_entry_vec_size;
	assert_true(first_pke_cnt > 1);
	assert_true(first_pke_cnt <= EFA_UNIT_TEST_PROTO_MAX_PKES);
	for (i = 0; i < first_pke_cnt; ++i) {
		first_seg_offsets[i] =
			efa_rdm_pke_get_runtread_rtm_base_hdr(
				ep->send_pkt_entry_vec[i])->seg_offset;
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[i]);
	}
	first_protocol = txe->req_pkt_type;
	first_bytes_runt = txe->bytes_runt;

	/*
	 * The first attempt never reached the device, but the post-send hook
	 * runs on the repost, so mimic a first attempt that did publish its
	 * accounting -- that is the state a non-idempotent write would corrupt.
	 */
	proto->handle_tx_pkes_posted(ep, txe);
	first_bytes_sent = txe->bytes_sent;
	assert_int_equal(first_bytes_sent, txe->bytes_runt);
	assert_true(txe->internal_flags & EFA_RDM_TXE_READ_MSG_COUNTED);
	assert_int_equal(ofi_atomic_get64(
				 &efa_rdm_ep_rdm_domain(ep)->num_read_msg_in_flight),
			 1);

	/* The repost, exactly as efa_rdm_msg_repost_rtm_proto does it. */
	err = proto->construct_tx_pkes(ep, txe, &pke_send_flags);
	assert_int_equal(err, 0);

	assert_int_equal(ep->send_pkt_entry_vec_size, first_pke_cnt);
	assert_int_equal(txe->req_pkt_type, first_protocol);
	assert_int_equal(txe->bytes_runt, first_bytes_runt);
	for (i = 0; i < first_pke_cnt; ++i)
		assert_int_equal(efa_rdm_pke_get_runtread_rtm_base_hdr(
					 ep->send_pkt_entry_vec[i])->seg_offset,
				 first_seg_offsets[i]);

	proto->handle_tx_pkes_posted(ep, txe);
	assert_int_equal(txe->bytes_sent, first_bytes_sent);
	assert_int_equal(ofi_atomic_get64(
				 &efa_rdm_ep_rdm_domain(ep)->num_read_msg_in_flight),
			 1);

	/*
	 * With a read message in flight on the domain, the next send must not
	 * pick a read based protocol at all.
	 */
	second_txe = ofi_buf_alloc(ep->base_ep.txe_pool);
	assert_non_null(second_txe);
	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 second_txe, &second_proto);
	restore_proto_runtread_device_version(saved_vendor_part_id);
	assert_int_equal(err, 0);
	assert_ptr_not_equal(second_proto, &efa_rdm_proto_runtread);
	ofi_buf_free(second_txe);

	for (i = 0; i < ep->send_pkt_entry_vec_size; ++i)
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[i]);
	efa_rdm_txe_release_read_msg_slot(txe);
	efa_rdm_txe_release(txe);
	efa_unit_test_buff_destruct(&send_buff);
}
