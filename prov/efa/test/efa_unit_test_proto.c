/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_unit_tests.h"
#include "rdm/efa_rdm_pke_nonreq.h"
#include "rdm/efa_rdm_pke_rtm.h"
#include "rdm/efa_rdm_proto.h"
#include "rdm/efa_rdm_proto_eager.h"
#include "rdm/efa_rdm_proto_longcts.h"
#include "rdm/efa_rdm_proto_longread.h"
#include "rdm/efa_rdm_proto_medium.h"
#include "rdm/efa_rdm_proto_runtread.h"

/*
 * A size that is too large for a single eager packet (the device MTU is a few
 * KB) but well within the medium threshold (64KB for system memory), so the
 * medium protocol is selected and has to split the message into several
 * packets.
 */
#define EFA_UNIT_TEST_PROTO_MEDIUM_LEN 16384

/*
 * Sizes for the runt read tests: past the medium threshold so medium declines,
 * and comfortably larger than the runt below so the receiver still has a tail to
 * read -- which is what makes this the runt read protocol rather than a plain
 * multi-packet send.
 */
#define EFA_UNIT_TEST_PROTO_RUNTREAD_LEN 131072
#define EFA_UNIT_TEST_PROTO_RUNT_SIZE	 32768

/*
 * Size for the long CTS tests: past the medium threshold so medium declines.
 * The read based protocols decline for a different reason -- the peer does not
 * advertise RDMA read support -- so long CTS, the last entry in the registry, is
 * what the send lands on.
 */
#define EFA_UNIT_TEST_PROTO_LONGCTS_LEN 131072

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
 * @brief A whole fi_send() to a peer that expects zero-copy data transfer runs
 *        on the refactored protocol path, from selection to send completion.
 *
 * The legacy send path used to own this case: mainline forces the handshake in
 * efa_rdm_msg_post_rtm(), then efa_rdm_pke_fill_data() notices the peer wants
 * headerless data and stamps the flags on the packet. Both are gone, so the
 * eager protocol has to carry the case end to end -- protocol selection, the
 * headerless packet its construct_tx_pkes() builds, the user_recv_qp routing
 * efa_rdm_pke_sendv() derives from the packet flags, and a send completion that
 * must reach the protocol callback rather than the headerless arm of the
 * pkt_type switch.
 */
void test_proto_eager_send_zero_copy_end_to_end(void **state)
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

	/*
	 * An old peer that unilaterally enabled zero-copy receive: the endpoint
	 * knows a peer might, and this one advertised its dedicated receive QP
	 * in its handshake. With the handshake in hand the send goes out
	 * immediately instead of being queued.
	 */
	ep->peer_may_have_zcpy_rx = true;
	peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_REQUEST_USER_RECV_QP;
	peer->user_recv_qp.qpn = 99;
	peer->user_recv_qp.qkey = 0xABCD;

	g_efa_unit_test_mocks.efa_qp_post_send =
		&efa_mock_efa_qp_post_send_return_mock;
	will_return_int_maybe(efa_mock_efa_qp_post_send_return_mock, 0);

	err = fi_send(resource->ep, send_buff.buff, send_buff.size,
		      fi_mr_desc(send_buff.mr), peer_addr, NULL);
	assert_int_equal(err, 0);
	assert_int_equal(efa_unit_test_get_ope_list_length(ep, EFA_RDM_TXE), 1);

	/* One headerless packet: every byte on the wire is user data. */
	assert_int_equal(ep->send_pkt_entry_vec_size, 1);
	pkt_entry = ep->send_pkt_entry_vec[0];
	assert_true(pkt_entry->flags & EFA_RDM_PKE_SEND_TO_USER_RECV_QP);
	assert_true(pkt_entry->flags & EFA_RDM_PKE_HAS_NO_BASE_HDR);
	assert_int_equal(pkt_entry->pkt_size, send_buff.size);
	assert_int_equal(pkt_entry->payload_size, send_buff.size);
	assert_non_null(pkt_entry->handle_pke);

	/*
	 * txe->protocol must still record the wire protocol even though no
	 * header carries it, or the peer-abort protocol would not recognize this
	 * send as a two-sided RTM.
	 */
	txe = pkt_entry->ope;
	assert_non_null(txe);
	assert_int_equal(txe->protocol, EFA_RDM_EAGER_MSGRTM_PKT);

	/*
	 * Go through efa_rdm_pke_handle_send_completion() rather than calling
	 * the callback directly: a headerless packet has no base header to
	 * switch on, so the protocol callback has to be dispatched before the
	 * packet type is ever consulted. It does its own
	 * efa_rdm_ep_record_tx_op_completed().
	 */
	efa_rdm_pke_handle_send_completion(pkt_entry);

	assert_int_equal(efa_unit_test_get_ope_list_length(ep, EFA_RDM_TXE), 0);
	memset(&err_entry, 0, sizeof(err_entry));
	assert_int_equal(fi_cq_readerr(resource->cq, &err_entry, 0), -FI_EAGAIN);

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

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
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
 * @brief Before a handshake, a send that would otherwise runt or read falls back
 *        to long CTS instead of waiting for the handshake.
 *
 * Protocol selection never triggers or waits for a handshake. Both read based
 * predicates go through efa_rdm_interop_rdma_read(), which reports no support
 * until the peer's handshake has advertised
 * EFA_RDM_EXTRA_FEATURE_RDMA_READ, so they decline and the send lands on long
 * CTS, which needs nothing from the peer beyond the baseline protocol.
 *
 * This matches the legacy path: its read based arm was gated on the very same
 * efa_rdm_interop_rdma_read() call, so its enforce-handshake step could not be
 * reached for a non-self peer either, and a large first send to a cold peer went
 * out over long CTS there too.
 */
void test_proto_select_longcts_before_handshake(void **state)
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

	/*
	 * Everything else about this send is the runt read case above; only the
	 * handshake is missing. A peer we have not heard from advertises nothing.
	 */
	peer->flags &= ~EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] = 0;
	assert_false(ep->homogeneous_peers);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	restore_proto_runtread_device_version(saved_vendor_part_id);

	assert_int_equal(err, 0);
	assert_non_null(proto);
	assert_ptr_equal(proto, &efa_rdm_proto_longcts);

	ofi_buf_free(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief A homogeneous-peers endpoint selects the runt read protocol before the
 *        handshake.
 *
 * efa_rdm_interop_rdma_read() reports this endpoint's own RDMA read support,
 * without consulting peer->extra_info, when ep->homogeneous_peers is set. So the
 * read based predicates hold from the very first send and no handshake step is
 * needed to reach them -- which is why the fi_mr_abort tests use
 * FI_OPT_EFA_HOMOGENEOUS_PEERS to pin a read protocol.
 *
 * The legacy path behaved identically: its enforce-handshake step for an
 * extra-feature packet type was guarded by !ep->homogeneous_peers.
 */
void test_proto_select_runtread_before_handshake_with_homogeneous_peers(
	void **state)
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

	peer->flags &= ~EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] = 0;
	ep->homogeneous_peers = true;

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
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
 * The legacy read based selector made the same call, falling back to long read
 * -- which is delivery complete by nature -- and so does the refactored
 * registry, since long read is the next entry after runt read.
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

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg,
						 FI_DELIVERY_COMPLETE, txe,
						 &proto);
	restore_proto_runtread_device_version(saved_vendor_part_id);

	assert_int_equal(err, 0);
	assert_ptr_equal(proto, &efa_rdm_proto_longread);

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
	int err;

	ep = setup_proto_runtread_test(resource, &send_buff, &peer_addr, &peer,
				       &saved_vendor_part_id);

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
	assert_ptr_equal(proto, &efa_rdm_proto_runtread);

	efa_rdm_proto_txe_fill(txe, ep, peer, &msg, ofi_op_msg, 0, 0, 0, proto);
	txe->msg_id = peer->next_msg_id++;

	err = proto->construct_tx_pkes(ep, peer, NULL, txe->op, txe->tag,
				       txe->fi_flags, txe->internal_flags, txe);
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
	 * The peer-abort protocol reads txe->protocol to tell a two-sided RTM
	 * from an operation it does not handle.
	 */
	assert_int_equal(txe->protocol, EFA_RDM_RUNTREAD_MSGRTM_PKT);

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
	int err;

	ep = setup_proto_runtread_test(resource, &send_buff, &peer_addr, &peer,
				       &saved_vendor_part_id);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	assert_int_equal(err, 0);
	assert_ptr_equal(proto, &efa_rdm_proto_runtread);

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
			efa_rdm_pke_get_runtread_rtm_base_hdr(
				ep->send_pkt_entry_vec[i])->seg_offset;
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[i]);
	}
	first_protocol = txe->protocol;
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

	/* The repost, with no fi_msg, exactly as efa_rdm_msg_repost_rtm_proto. */
	err = proto->construct_tx_pkes(ep, peer, NULL, txe->op, txe->tag,
				       txe->fi_flags, txe->internal_flags, txe);
	assert_int_equal(err, 0);

	assert_int_equal(ep->send_pkt_entry_vec_size, first_pke_cnt);
	assert_int_equal(txe->protocol, first_protocol);
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
	 * runt. It still reads: long read is the next entry in the registry and
	 * places no limit on how many messages are in flight.
	 */
	second_txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(second_txe);
	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 second_txe, &second_proto);
	restore_proto_runtread_device_version(saved_vendor_part_id);
	assert_int_equal(err, 0);
	assert_ptr_equal(second_proto, &efa_rdm_proto_longread);
	ofi_buf_free(second_txe);

	for (i = 0; i < ep->send_pkt_entry_vec_size; ++i)
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[i]);
	efa_rdm_txe_release_read_msg_slot(txe);
	efa_rdm_txe_release(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Set up an endpoint and a peer that the long read protocol can be used
 *        with, and register a source buffer.
 *
 * Same environment as the runt read tests, minus the peer's runt allowance: with
 * runt_size 0 efa_rdm_peer_get_runt_size() returns 0, so the runt read protocol
 * declines and long read -- the next entry in the registry -- is what a large
 * registered send must land on. That is also the shape of a real
 * FI_EFA_RUNT_SIZE=0 run, and of every host memory send by default.
 *
 * The caller must call restore_proto_runtread_device_version() as soon as
 * protocol selection is done, for the reason given on setup_proto_runtread_test.
 */
static struct efa_rdm_ep *
setup_proto_longread_test(struct efa_resource *resource,
			  struct efa_unit_test_buff *send_buff,
			  fi_addr_t *peer_addr, struct efa_rdm_peer **peer,
			  uint32_t *saved_vendor_part_id)
{
	struct efa_rdm_ep *ep;

	ep = setup_proto_runtread_test(resource, send_buff, peer_addr, peer,
				       saved_vendor_part_id);
	if (!ep)
		return NULL;

	g_efa_hmem_info[FI_HMEM_SYSTEM].runt_size = 0;

	return ep;
}

/**
 * @brief Test that the long read protocol is selected for a message past the
 *        interface's minimum read size when there is no runt allowance left.
 */
void test_proto_select_longread_for_large_msg(void **state)
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

	ep = setup_proto_longread_test(resource, &send_buff, &peer_addr, &peer,
				       &saved_vendor_part_id);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	restore_proto_runtread_device_version(saved_vendor_part_id);

	assert_int_equal(err, 0);
	assert_ptr_equal(proto, &efa_rdm_proto_longread);

	ofi_buf_free(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief An unregistered source iov rules the long read protocol out.
 *
 * The receiver reads every source iov out of the read iov array the REQ carries,
 * so an iov the device cannot read from has no representation on the wire.
 * construct_tx_pkes() would have to abandon the send with -FI_ENOMR, and the
 * refactored path has no equivalent of efa_rdm_ope_post_send_fallback(), so the
 * predicate has to decline up front and let the send fall through to long CTS.
 */
void test_proto_select_declines_longread_without_mr(void **state)
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

	ep = setup_proto_longread_test(resource, &send_buff, &peer_addr, &peer,
				       &saved_vendor_part_id);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	/* No desc, and no MR cache in the unit test domain to build one from. */
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0, NULL);

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	restore_proto_runtread_device_version(saved_vendor_part_id);

	assert_int_equal(err, 0);
	assert_ptr_not_equal(proto, &efa_rdm_proto_longread);
	assert_ptr_not_equal(proto, &efa_rdm_proto_runtread);

	ofi_buf_free(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief The long read protocol sends exactly one payload-free REQ packet whose
 *        body is the read iov array.
 *
 * The receiver learns the message length from the header and where the data
 * lives from the read iov array that follows it, then RDMA reads the whole
 * message; there is nothing else on the wire, so a wrong pkt_size or a missing
 * iov silently corrupts the transfer.
 */
void test_proto_longread_construct_pkes_single_pke(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	struct efa_rdm_longread_rtm_base_hdr *rtm_hdr;
	struct efa_rdm_pke *pke;
	struct fi_rma_iov *read_iov;
	fi_addr_t peer_addr;
	uint32_t saved_vendor_part_id;
	struct fi_msg msg = {0};
	struct iovec iov;
	size_t hdr_size;
	int err;

	ep = setup_proto_longread_test(resource, &send_buff, &peer_addr, &peer,
				       &saved_vendor_part_id);

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
	assert_ptr_equal(proto, &efa_rdm_proto_longread);

	efa_rdm_proto_txe_fill(txe, ep, peer, &msg, ofi_op_msg, 0, 0, 0, proto);
	txe->msg_id = peer->next_msg_id++;

	err = proto->construct_tx_pkes(ep, peer, NULL, txe->op, txe->tag,
				       txe->fi_flags, txe->internal_flags, txe);
	restore_proto_runtread_device_version(saved_vendor_part_id);
	assert_int_equal(err, 0);

	assert_int_equal(ep->send_pkt_entry_vec_size, 1);

	/*
	 * The peer-abort protocol reads txe->protocol to tell a two-sided RTM
	 * from an operation it does not handle.
	 */
	assert_int_equal(txe->protocol, EFA_RDM_LONGREAD_MSGRTM_PKT);

	pke = ep->send_pkt_entry_vec[0];
	assert_non_null(pke);
	assert_ptr_equal(pke->handle_pke,
			 &efa_rdm_proto_longread_handle_rtm_send_completion);
	assert_ptr_equal(pke->ope, txe);

	/* The RTM carries no message data at all. */
	assert_int_equal(pke->payload_size, 0);

	rtm_hdr = efa_rdm_pke_get_longread_rtm_base_hdr(pke);
	assert_int_equal(rtm_hdr->msg_length, txe->total_len);
	assert_int_equal(rtm_hdr->send_id, txe->tx_id);
	assert_int_equal(rtm_hdr->read_iov_count, txe->iov_count);

	/* The read iov array sits immediately after the REQ header. */
	hdr_size = efa_rdm_pke_get_req_hdr_size(pke);
	read_iov = (struct fi_rma_iov *) (pke->wiredata + hdr_size);
	assert_int_equal(read_iov[0].addr, (uint64_t) send_buff.buff);
	assert_int_equal(read_iov[0].len, send_buff.size);
	assert_int_equal(read_iov[0].key, fi_mr_key(send_buff.mr));

	/* The array is the whole body of the packet. */
	assert_int_equal(pke->pkt_size,
			 hdr_size +
				 txe->iov_count * sizeof(struct fi_rma_iov));

	efa_rdm_pke_release_tx(pke);
	efa_rdm_txe_release(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief The long read protocol's construct_tx_pkes() and post-send hook are
 *        idempotent, so a txe queued before the handshake can be reposted.
 *
 * efa_rdm_msg_repost_rtm_proto() re-enters construct_tx_pkes() on a txe the
 * first attempt already set up, and efa_rdm_ope_process_queued_ope() retries a
 * txe that returned -FI_EAGAIN without clearing its queued state.
 *
 * The header writes are all assignments, so the hazard is the post-send hook:
 * the domain's read message slot must be taken once per message, not once per
 * attempt. EFA_RDM_TXE_READ_MSG_COUNTED is the test-and-clear token the single
 * release site consumes, so a second bump would leave the counter stuck above
 * zero and make every later message on the domain skip runting.
 */
void test_proto_longread_construct_pkes_is_idempotent(void **state)
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
	uint32_t first_protocol;
	int err;

	ep = setup_proto_longread_test(resource, &send_buff, &peer_addr, &peer,
				       &saved_vendor_part_id);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	restore_proto_runtread_device_version(saved_vendor_part_id);
	assert_int_equal(err, 0);
	assert_ptr_equal(proto, &efa_rdm_proto_longread);

	efa_rdm_proto_txe_fill(txe, ep, peer, &msg, ofi_op_msg, 0, 0, 0, proto);
	txe->msg_id = peer->next_msg_id++;

	err = proto->construct_tx_pkes(ep, peer, NULL, txe->op, txe->tag,
				       txe->fi_flags, txe->internal_flags, txe);
	assert_int_equal(err, 0);
	assert_int_equal(ep->send_pkt_entry_vec_size, 1);
	first_protocol = txe->protocol;
	efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[0]);

	/*
	 * The first attempt never reached the device, but the post-send hook
	 * runs on the repost, so mimic a first attempt that did publish its
	 * accounting -- that is the state a non-idempotent write would corrupt.
	 */
	proto->handle_tx_pkes_posted(ep, txe);
	assert_true(txe->internal_flags & EFA_RDM_TXE_READ_MSG_COUNTED);
	assert_int_equal(ofi_atomic_get64(
				 &efa_rdm_ep_rdm_domain(ep)->num_read_msg_in_flight),
			 1);

	/* The repost, with no fi_msg, exactly as efa_rdm_msg_repost_rtm_proto. */
	err = proto->construct_tx_pkes(ep, peer, NULL, txe->op, txe->tag,
				       txe->fi_flags, txe->internal_flags, txe);
	assert_int_equal(err, 0);
	assert_int_equal(ep->send_pkt_entry_vec_size, 1);
	assert_int_equal(txe->protocol, first_protocol);
	assert_int_equal(efa_rdm_pke_get_longread_rtm_base_hdr(
				 ep->send_pkt_entry_vec[0])->msg_length,
			 txe->total_len);

	proto->handle_tx_pkes_posted(ep, txe);
	assert_int_equal(ofi_atomic_get64(
				 &efa_rdm_ep_rdm_domain(ep)->num_read_msg_in_flight),
			 1);

	efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[0]);
	efa_rdm_txe_release_read_msg_slot(txe);
	efa_rdm_txe_release(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Set up an endpoint and a handshake-completed peer that no read based
 *        protocol can be used with, and register a source buffer past the medium
 *        threshold.
 *
 * The peer deliberately does not advertise EFA_RDM_EXTRA_FEATURE_RDMA_READ, so
 * efa_rdm_interop_rdma_read() is false and both read protocols decline no matter
 * what the interface thresholds are. That is the real shape of a send to a peer
 * without device RDMA read, and it leaves long CTS -- which needs nothing from
 * the peer beyond the baseline protocol -- as the only match.
 */
static struct efa_rdm_ep *
setup_proto_longcts_test(struct efa_resource *resource,
			 struct efa_unit_test_buff *send_buff,
			 fi_addr_t *peer_addr, struct efa_rdm_peer **peer)
{
	struct efa_rdm_ep *ep;

	ep = setup_proto_select_test(resource, peer_addr);
	efa_unit_test_buff_construct(send_buff, resource,
				     EFA_UNIT_TEST_PROTO_LONGCTS_LEN);

	*peer = efa_rdm_ep_get_peer_explicit(ep, *peer_addr);
	(*peer)->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	/* Eager and medium are tried first and must both decline this size. */
	assert_true(EFA_UNIT_TEST_PROTO_LONGCTS_LEN > ep->mtu_size);
	assert_true(EFA_UNIT_TEST_PROTO_LONGCTS_LEN >
		    g_efa_hmem_info[FI_HMEM_SYSTEM].max_medium_msg_size);

	return ep;
}

/**
 * @brief Test that the long CTS protocol is selected for a large message when no
 *        other protocol can be used.
 */
void test_proto_select_longcts_for_large_msg(void **state)
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

	ep = setup_proto_longcts_test(resource, &send_buff, &peer_addr, &peer);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	assert_int_equal(err, 0);
	assert_ptr_equal(proto, &efa_rdm_proto_longcts);

	ofi_buf_free(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief The long CTS protocol sends exactly one REQ carrying the head of the
 *        message plus a credit request.
 *
 * The receiver learns the message length from the header, copies the REQ's
 * payload as the first segment, and answers with a CTS sized from
 * credit_request; the rest of the message follows as CTSDATA packets keyed off
 * send_id. A wrong msg_length, send_id or payload size silently corrupts the
 * transfer or stalls it.
 */
void test_proto_longcts_construct_pkes_single_pke(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	struct efa_rdm_longcts_rtm_base_hdr *rtm_hdr;
	struct efa_rdm_pke *pke;
	fi_addr_t peer_addr;
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	ep = setup_proto_longcts_test(resource, &send_buff, &peer_addr, &peer);

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
	assert_ptr_equal(proto, &efa_rdm_proto_longcts);

	efa_rdm_proto_txe_fill(txe, ep, peer, &msg, ofi_op_msg, 0, 0, 0, proto);
	txe->msg_id = peer->next_msg_id++;

	err = proto->construct_tx_pkes(ep, peer, NULL, txe->op, txe->tag,
				       txe->fi_flags, txe->internal_flags, txe);
	assert_int_equal(err, 0);

	/* Long CTS sends one unsolicited REQ; the CTS paces everything after. */
	assert_int_equal(ep->send_pkt_entry_vec_size, 1);

	/*
	 * The peer-abort protocol reads txe->protocol to tell a two-sided RTM
	 * from an operation it does not handle.
	 */
	assert_int_equal(txe->protocol, EFA_RDM_LONGCTS_MSGRTM_PKT);

	pke = ep->send_pkt_entry_vec[0];
	assert_non_null(pke);
	assert_ptr_equal(pke->handle_pke,
			 &efa_rdm_proto_longcts_handle_rtm_send_completion);
	assert_ptr_equal(pke->ope, txe);

	/* The REQ carries the head of the message, never all of it. */
	assert_true(pke->payload_size > 0);
	assert_true(pke->payload_size < txe->total_len);

	rtm_hdr = efa_rdm_pke_get_longcts_rtm_base_hdr(pke);
	assert_int_equal(rtm_hdr->msg_length, txe->total_len);
	assert_int_equal(rtm_hdr->send_id, txe->tx_id);
	assert_int_equal(rtm_hdr->credit_request, efa_env.tx_min_credits);

	efa_rdm_pke_release_tx(pke);
	efa_rdm_txe_release(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief The long CTS protocol's construct_tx_pkes() and post-send hook are
 *        idempotent, so a txe queued before the handshake can be reposted.
 *
 * efa_rdm_msg_repost_rtm_proto() re-enters construct_tx_pkes() on a txe the
 * first attempt already set up, and efa_rdm_ope_process_queued_ope() retries a
 * txe that returned -FI_EAGAIN without clearing its queued state.
 *
 * The header writes are all assignments, so the hazard is bytes_sent: the
 * mainline long CTS path accumulated it per packet
 * (efa_rdm_pke_handle_longcts_rtm_sent), and a second accumulation would make
 * the CTSDATA stream resume past the end of the REQ's segment and skip that much
 * of the message.
 */
void test_proto_longcts_construct_pkes_is_idempotent(void **state)
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
	size_t first_payload_size;
	uint32_t first_protocol;
	int err;

	ep = setup_proto_longcts_test(resource, &send_buff, &peer_addr, &peer);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	assert_int_equal(err, 0);
	assert_ptr_equal(proto, &efa_rdm_proto_longcts);

	efa_rdm_proto_txe_fill(txe, ep, peer, &msg, ofi_op_msg, 0, 0, 0, proto);
	txe->msg_id = peer->next_msg_id++;

	err = proto->construct_tx_pkes(ep, peer, NULL, txe->op, txe->tag,
				       txe->fi_flags, txe->internal_flags, txe);
	assert_int_equal(err, 0);
	assert_int_equal(ep->send_pkt_entry_vec_size, 1);
	first_payload_size = ep->send_pkt_entry_vec[0]->payload_size;
	first_protocol = txe->protocol;

	/*
	 * The first attempt never reached the device, but the post-send hook runs
	 * on the repost, so mimic a first attempt that did publish its
	 * accounting -- that is the state an accumulating write would corrupt.
	 */
	proto->handle_tx_pkes_posted(ep, txe);
	assert_int_equal(txe->bytes_sent, first_payload_size);
	efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[0]);

	/* The repost, with no fi_msg, exactly as efa_rdm_msg_repost_rtm_proto. */
	err = proto->construct_tx_pkes(ep, peer, NULL, txe->op, txe->tag,
				       txe->fi_flags, txe->internal_flags, txe);
	assert_int_equal(err, 0);
	assert_int_equal(ep->send_pkt_entry_vec_size, 1);
	assert_int_equal(txe->protocol, first_protocol);
	assert_int_equal(ep->send_pkt_entry_vec[0]->payload_size,
			 first_payload_size);
	assert_int_equal(efa_rdm_pke_get_longcts_rtm_base_hdr(
				 ep->send_pkt_entry_vec[0])->msg_length,
			 txe->total_len);

	proto->handle_tx_pkes_posted(ep, txe);
	assert_int_equal(txe->bytes_sent, first_payload_size);

	efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[0]);
	efa_rdm_txe_release(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief A long CTS txe whose source MR was closed while its REQ was in flight
 *        is completed exactly once by the peer-abort drain helper, even though
 *        the REQ completes successfully.
 *
 * This is what the EFA_RDM_OPE_PEER_ABORT_PENDING branch in the long CTS send
 * completion callback exists for, and the reason it is not dead code the way it
 * would be for the single packet eager protocol: one long CTS message is the REQ
 * plus a stream of CTSDATA packets sharing one txe, so bytes_acked is still
 * short of total_len when the REQ completes. Without the branch that success
 * matches neither arm, so nothing drains the txe and the operation never
 * completes at all.
 *
 * Verifies instead that the success routes to the drain helper, which -- this
 * being the txe's last outstanding WR -- emits one PEER_ERROR_PKT so the peer
 * can unblock its reorder window, and then writes exactly one FI_ECANCELED /
 * FI_EFA_ERR_PEER_ABORTED CQ error entry when that packet's own send completes.
 */
void test_proto_longcts_send_completion_peer_abort(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_pke *req_pke;
	struct fi_cq_err_entry err_entry;
	fi_addr_t peer_addr;
	int err, ret;

	ep = setup_proto_longcts_test(resource, &send_buff, &peer_addr, &peer);
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
	 * Snapshot the REQ: emitting the PEER_ERROR_PKT reuses
	 * ep->send_pkt_entry_vec, so it is only safe to read before the drain.
	 */
	assert_int_equal(ep->send_pkt_entry_vec_size, 1);
	req_pke = ep->send_pkt_entry_vec[0];
	assert_non_null(req_pke->handle_pke);

	txe = req_pke->ope;
	assert_non_null(txe);
	assert_int_equal(txe->protocol, EFA_RDM_LONGCTS_MSGRTM_PKT);
	/* The message is only partly on the wire, which is the whole point. */
	assert_true(req_pke->payload_size < txe->total_len);

	/*
	 * The application closes the source MR while the send is in flight. The
	 * error path marks the txe peer-aborting; because the REQ's WR is still
	 * outstanding, the drain helper defers the emit.
	 */
	efa_unit_test_proto_simulate_source_mr_canceled(txe);
	efa_rdm_txe_handle_error(txe, FI_ECANCELED, FI_EFA_ERR_PKT_POST);
	assert_true(txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_false(txe->internal_flags & EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);

	memset(&err_entry, 0, sizeof(err_entry));
	assert_int_equal(fi_cq_readerr(resource->cq, &err_entry, 0), -FI_EAGAIN);

	/* The REQ completes successfully, draining the txe's last WR. */
	efa_rdm_ep_record_tx_op_completed(ep, req_pke);
	req_pke->handle_pke(req_pke);

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

/**
 * @brief A read NACK continues the message as long CTS on the refactored path.
 *
 * The receiver of a read based protocol that cannot register its receive buffer
 * answers with a EFA_RDM_READ_NACK_PKT, and the sender finishes the message with
 * the long CTS protocol instead. That continuation is not a fresh send: the
 * msg_id is spent, the runt packets already delivered txe->bytes_sent bytes and
 * the receiver already has an rxe for this msg_id. So the REQ must carry no
 * payload, must not move bytes_sent, and must set EFA_RDM_REQ_READ_NACK on the
 * wire -- without it the receiver allocates a second rxe for the msg_id and
 * slides its receive window a second time.
 */
void test_proto_longcts_read_nack_continues_on_refactored_path(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *runt_pkes[EFA_UNIT_TEST_PROTO_MAX_PKES];
	struct efa_rdm_pke *nack_pke, *req_pke;
	struct efa_rdm_read_nack_hdr *nack_hdr;
	struct efa_rdm_rtm_base_hdr *rtm_hdr;
	struct efa_rdm_longcts_rtm_base_hdr *longcts_hdr;
	fi_addr_t peer_addr;
	uint32_t saved_vendor_part_id;
	size_t i, runt_pke_cnt, bytes_sent_before, rx_pkts_to_post_before;
	int err;

	ep = setup_proto_runtread_test(resource, &send_buff, &peer_addr, &peer,
				       &saved_vendor_part_id);

	g_efa_unit_test_mocks.efa_qp_post_send =
		&efa_mock_efa_qp_post_send_return_mock;
	will_return_int_maybe(efa_mock_efa_qp_post_send_return_mock, 0);

	err = fi_send(resource->ep, send_buff.buff, send_buff.size,
		      fi_mr_desc(send_buff.mr), peer_addr, NULL);
	restore_proto_runtread_device_version(saved_vendor_part_id);
	assert_int_equal(err, 0);

	/*
	 * Snapshot the runt packets before anything else: the continuation REQ
	 * reuses ep->send_pkt_entry_vec.
	 */
	runt_pke_cnt = ep->send_pkt_entry_vec_size;
	assert_true(runt_pke_cnt > 1);
	assert_true(runt_pke_cnt <= EFA_UNIT_TEST_PROTO_MAX_PKES);
	for (i = 0; i < runt_pke_cnt; ++i)
		runt_pkes[i] = ep->send_pkt_entry_vec[i];

	txe = runt_pkes[0]->ope;
	assert_non_null(txe);
	assert_int_equal(txe->protocol, EFA_RDM_RUNTREAD_MSGRTM_PKT);
	assert_true(txe->internal_flags & EFA_RDM_TXE_READ_MSG_COUNTED);
	bytes_sent_before = txe->bytes_sent;
	assert_int_equal(bytes_sent_before, txe->bytes_runt);
	assert_true(bytes_sent_before < txe->total_len);

	/* The receiver failed to register its buffer and sent a READ NACK. */
	nack_pke = efa_rdm_pke_alloc(ep, ep->efa_rx_pkt_pool,
				     EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(nack_pke);
	nack_pke->peer = peer;
	nack_hdr = (struct efa_rdm_read_nack_hdr *) nack_pke->wiredata;
	nack_hdr->type = EFA_RDM_READ_NACK_PKT;
	nack_hdr->send_id = txe->tx_id;

	/*
	 * An rx packet entry is normally only allocated by the progress engine,
	 * which accounts for it. Allocating one by hand does not, but releasing it
	 * does, so snapshot the counter and put it back afterwards or
	 * efa_rdm_ep_post_internal_rx_pkts() trips its accounting assertion on
	 * the next progress.
	 */
	rx_pkts_to_post_before = ep->efa_rx_pkts_to_post;
	efa_rdm_pke_handle_read_nack_recv(nack_pke);
	ep->efa_rx_pkts_to_post = rx_pkts_to_post_before;

	/* The fallback goes through the refactored long CTS protocol. */
	assert_true(txe->internal_flags & EFA_RDM_OPE_READ_NACK);
	assert_ptr_equal(txe->proto, &efa_rdm_proto_longcts);
	assert_int_equal(txe->protocol, EFA_RDM_LONGCTS_MSGRTM_PKT);

	/*
	 * The domain read slot is handed back, or every later message on this
	 * domain would decline to runt.
	 */
	assert_false(txe->internal_flags & EFA_RDM_TXE_READ_MSG_COUNTED);
	assert_int_equal(ofi_atomic_get64(
				 &efa_rdm_ep_rdm_domain(ep)->num_read_msg_in_flight),
			 0);

	assert_int_equal(ep->send_pkt_entry_vec_size, 1);
	req_pke = ep->send_pkt_entry_vec[0];
	assert_non_null(req_pke);
	assert_ptr_equal(req_pke->handle_pke,
			 &efa_rdm_proto_longcts_handle_rtm_send_completion);
	assert_ptr_equal(req_pke->ope, txe);

	/*
	 * No payload, and bytes_sent still marks the end of the runt, so the
	 * CTSDATA stream resumes exactly where the runt packets stopped.
	 */
	assert_int_equal(req_pke->payload_size, 0);
	assert_int_equal(txe->bytes_sent, bytes_sent_before);

	rtm_hdr = efa_rdm_pke_get_rtm_base_hdr(req_pke);
	assert_int_equal(rtm_hdr->type, EFA_RDM_LONGCTS_MSGRTM_PKT);
	assert_true(rtm_hdr->flags & EFA_RDM_REQ_READ_NACK);
	assert_int_equal(rtm_hdr->msg_id, txe->msg_id);

	longcts_hdr = efa_rdm_pke_get_longcts_rtm_base_hdr(req_pke);
	assert_int_equal(longcts_hdr->msg_length, txe->total_len);
	assert_int_equal(longcts_hdr->send_id, txe->tx_id);

	/* Drain every WR: the runt packets first, then the payload-free REQ. */
	for (i = 0; i < runt_pke_cnt; ++i) {
		efa_rdm_ep_record_tx_op_completed(ep, runt_pkes[i]);
		runt_pkes[i]->handle_pke(runt_pkes[i]);
	}
	efa_rdm_ep_record_tx_op_completed(ep, req_pke);
	req_pke->handle_pke(req_pke);

	/*
	 * The message is not finished -- the CTSDATA packets that carry the rest
	 * are still owed -- so no completion may have been written and the txe
	 * must still be alive.
	 */
	assert_int_equal(efa_unit_test_get_ope_list_length(ep, EFA_RDM_TXE), 1);
	assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);

	efa_rdm_txe_release(txe);
	efa_unit_test_buff_destruct(&send_buff);
}
