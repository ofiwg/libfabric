/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pke_cmd.h"
#include "rdm/efa_rdm_peer.h"

/**
 * @brief Test efa_rdm_peer_reorder_msg
 *
 * @param resource efa_resource
 * @param exp_msg_id expected message id of receive window
 * @param msg_id received message id
 * @param expected_ret expected return value of efa_rdm_peer_reorder_msg
 */
void test_efa_rdm_peer_reorder_msg_impl(struct efa_resource *resource,
					uint32_t exp_msg_id, uint32_t msg_id,
					int expected_ret)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr;
	fi_addr_t addr;
	struct efa_unit_test_eager_rtm_pkt_attr pkt_attr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_rdm_peer_overflow_pke_list_entry *overflow_pke_list_entry;
	struct efa_rdm_pke *pkt_entry, *overflow_pkt_entry;
	struct dlist_entry *tmp;
	uint32_t overflow_msg_id;
	int ret;

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	/* Create and register a fake peer */
	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0 /* flags */, NULL /* context */);
	assert_int_equal(ret, 1);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	assert_non_null(peer);

	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	efa_rdm_ep->efa_rx_pkts_posted = efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep);

	pkt_attr.msg_id = msg_id;
	pkt_attr.connid = raw_addr.qkey;
	/* Packet type must be >= EFA_RDM_REQ_PKT_BEGIN */
	efa_unit_test_eager_msgrtm_pkt_construct(pkt_entry, &pkt_attr);

	(&peer->robuf)->exp_msg_id = exp_msg_id;
	ret = efa_rdm_peer_reorder_msg(peer, efa_rdm_ep, pkt_entry);
	assert_int_equal(ret, expected_ret);

	if (!ofi_recvwin_id_valid((&peer->robuf), msg_id) &&
	    !ofi_recvwin_id_processed((&peer->robuf), msg_id)) {
		/* Check the overflow_pke_list contains the overflow msg_id */
		assert_int_equal(efa_unit_test_get_dlist_length(&peer->overflow_pke_list), 1);
		dlist_foreach_container_safe (
			&peer->overflow_pke_list,
			struct efa_rdm_peer_overflow_pke_list_entry,
			overflow_pke_list_entry, entry, tmp) {
			overflow_pkt_entry = overflow_pke_list_entry->pkt_entry;
			overflow_msg_id = ((struct efa_rdm_rtm_base_hdr *) overflow_pkt_entry->wiredata)->msg_id;
			assert_int_equal(overflow_msg_id, msg_id);
			/* Clean up */
			dlist_remove(&overflow_pke_list_entry->entry);
			efa_rdm_pke_release_rx(overflow_pke_list_entry->pkt_entry);
			ofi_buf_free(overflow_pke_list_entry);
		}
	} else {
		efa_rdm_pke_release_rx(pkt_entry);
	}
}

void test_efa_rdm_peer_reorder_expected_msg_id(struct efa_resource **state) {
	struct efa_resource *resource = *state;
	uint32_t msg_id, exp_msg_id;
	int expected_ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	msg_id = 0;
	exp_msg_id = 0;
	expected_ret = 0;
	/* Receiving expected message id should return 0 */
	test_efa_rdm_peer_reorder_msg_impl(resource, exp_msg_id, msg_id, expected_ret);

	msg_id = EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE * 3 + 4;
	exp_msg_id = msg_id;
	expected_ret = 0;
	/* Receiving expected message id should return 0 */
	test_efa_rdm_peer_reorder_msg_impl(resource, exp_msg_id, msg_id, expected_ret);
}


void test_efa_rdm_peer_reorder_smaller_msg_id(struct efa_resource **state) {
	struct efa_resource *resource = *state;
	uint32_t msg_id, exp_msg_id;
	int expected_ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	/* Test with exp_msg_id less than robuf size */
	msg_id = 1;
	exp_msg_id = EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE - 1;
	expected_ret = -FI_EALREADY;
	/* Receiving message id smaller than expected should return -FI_EALREADY */
	test_efa_rdm_peer_reorder_msg_impl(resource, exp_msg_id, msg_id, expected_ret);

	/* Test with exp_msg_id greater than robuf size */
	msg_id = EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE - 1;
	exp_msg_id = EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE * 2 + 3;
	expected_ret = -FI_EALREADY;
	/* Receiving message id smaller than expected should return -FI_EALREADY */
	test_efa_rdm_peer_reorder_msg_impl(resource, exp_msg_id, msg_id, expected_ret);
}

void test_efa_rdm_peer_reorder_larger_msg_id(struct efa_resource **state) {
	struct efa_resource *resource = *state;
	uint32_t msg_id, exp_msg_id;
	int expected_ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	msg_id = EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE - 2;
	exp_msg_id = 0;
	expected_ret = 1;
	efa_env.rx_copy_ooo = 0; /* Do not copy this pkt entry */
	/* Receiving message id larger than expected should return 1 */
	test_efa_rdm_peer_reorder_msg_impl(resource, exp_msg_id, msg_id, expected_ret);
}

void test_efa_rdm_peer_reorder_overflow_msg_id(struct efa_resource **state) {
	struct efa_resource *resource = *state;
	uint32_t msg_id, exp_msg_id;
	int expected_ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	/* Test with msg_id equal to robuf size + 1 */
	msg_id = EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE;
	exp_msg_id = 0;
	expected_ret = 1;
	/* Message id that overflows the receive window should be put in the
	 * overflow_pke_list and return 1 */
	test_efa_rdm_peer_reorder_msg_impl(resource, exp_msg_id, msg_id, expected_ret);

	/* Test with larger msg_id */
	msg_id = EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE * 2;
	exp_msg_id = EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE - 2;
	expected_ret = 1;
	/* Message id that overflows the receive window should be put in the
	 * overflow_pke_list and return 1 */
	test_efa_rdm_peer_reorder_msg_impl(resource, exp_msg_id, msg_id, expected_ret);
}

/**
 * @brief Test efa_rdm_peer_move_overflow_pke_to_recvwin
 *
 * @param resource   efa_resource
 * @param msg_id     received message id
 * @param peer 	     efa_rdm_peer
 * @param pkt_entry  packet entry in the overflow list
 */
void test_efa_rdm_peer_move_overflow_pke_to_recvwin_impl(
	struct efa_resource *resource, uint32_t msg_id,
	struct efa_rdm_peer **peer, struct efa_rdm_pke **pkt_entry)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_ep_addr raw_addr;
	fi_addr_t addr;
	struct efa_unit_test_eager_rtm_pkt_attr pkt_attr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_rdm_peer_overflow_pke_list_entry *overflow_pke_list_entry;
	int ret;

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	/* Create and register a fake peer */
	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0 /* flags */, NULL /* context */);
	assert_int_equal(ret, 1);
	*peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	assert_non_null(*peer);

	*pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(*pkt_entry);
	efa_rdm_ep->efa_rx_pkts_posted = efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep);

	pkt_attr.msg_id = msg_id;
	pkt_attr.connid = raw_addr.qkey;
	/* Packet type must be >= EFA_RDM_REQ_PKT_BEGIN */
	efa_unit_test_eager_msgrtm_pkt_construct(*pkt_entry, &pkt_attr);

	overflow_pke_list_entry = ofi_buf_alloc(efa_rdm_ep->overflow_pke_pool);
	overflow_pke_list_entry->pkt_entry = *pkt_entry;
	dlist_insert_head(&overflow_pke_list_entry->entry, &(*peer)->overflow_pke_list);

	(&(*peer)->robuf)->exp_msg_id = efa_env.recvwin_size;
	efa_env.rx_copy_ooo = 0;
	efa_rdm_peer_move_overflow_pke_to_recvwin(*peer);
}

void test_efa_rdm_peer_move_overflow_pke_to_recvwin(struct efa_resource **state) {
	struct efa_resource *resource = *state;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *pkt_entry;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	/* overflow_pke_list has a pkt entry with msg_id EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE / 8.
	 * After calling efa_rdm_peer_move_overflow_pke_to_recvwin when exp_msg_id = EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE,
	 * EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE / 8 will be moved to recvwin and overflow_pke_list will be empty. */
	test_efa_rdm_peer_move_overflow_pke_to_recvwin_impl(
		resource, EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE / 8, &peer, &pkt_entry);

	assert_non_null(*ofi_recvwin_get_msg((&peer->robuf), EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE / 8));
	assert_int_equal(efa_unit_test_get_dlist_length(&peer->overflow_pke_list), 0);

	efa_rdm_pke_release_rx(pkt_entry);
}

void test_efa_rdm_peer_keep_pke_in_overflow_list(struct efa_resource **state) {
	struct efa_resource *resource = *state;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_peer_overflow_pke_list_entry *overflow_pke_list_entry;
	struct dlist_entry *tmp;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	/* overflow_pke_list has a pkt entry with msg_id (EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE * 2) + 1000.
	 * After calling efa_rdm_peer_move_overflow_pke_to_recvwin when exp_msg_id = EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE,
	 * (EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE * 2) + 1000 will stay in overflow_pke_list. */
	test_efa_rdm_peer_move_overflow_pke_to_recvwin_impl(
		resource, (EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE * 2) + 1000, &peer, &pkt_entry);

	assert_int_equal(efa_unit_test_get_dlist_length(&peer->overflow_pke_list), 1);

	dlist_foreach_container_safe (
		&peer->overflow_pke_list,
		struct efa_rdm_peer_overflow_pke_list_entry,
		overflow_pke_list_entry, entry, tmp) {
		dlist_remove(&overflow_pke_list_entry->entry);
		efa_rdm_pke_release_rx(overflow_pke_list_entry->pkt_entry);
		ofi_buf_free(overflow_pke_list_entry);
	}
}

void alloc_pke_in_overflow_list(struct efa_rdm_ep *efa_rdm_ep,
		     struct efa_rdm_pke **pkt_entry, struct efa_rdm_peer *peer,
		     struct efa_ep_addr raw_addr, uint32_t msg_id)
{
	struct efa_unit_test_eager_rtm_pkt_attr pkt_attr = {0};
	struct efa_rdm_peer_overflow_pke_list_entry *overflow_pke_list_entry;
	struct efa_rdm_rtm_base_hdr *rtm_hdr;

	*pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				       EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(*pkt_entry);
	efa_rdm_ep->efa_rx_pkts_posted = efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep);

	pkt_attr.msg_id = msg_id;
	pkt_attr.connid = raw_addr.qkey;
	/* Packet type must be >= EFA_RDM_REQ_PKT_BEGIN */
	efa_unit_test_eager_msgrtm_pkt_construct(*pkt_entry, &pkt_attr);
	rtm_hdr = (struct efa_rdm_rtm_base_hdr *) (*pkt_entry)->wiredata;
	rtm_hdr->type = EFA_RDM_MEDIUM_TAGRTM_PKT;
	rtm_hdr->flags |= EFA_RDM_REQ_MSG;

	overflow_pke_list_entry = ofi_buf_alloc(efa_rdm_ep->overflow_pke_pool);
	overflow_pke_list_entry->pkt_entry = *pkt_entry;
	dlist_insert_head(&overflow_pke_list_entry->entry, &peer->overflow_pke_list);
}

void test_efa_rdm_peer_append_overflow_pke_to_recvwin(struct efa_resource **state) {
	struct efa_resource *resource = *state;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *pkt_entry1, *pkt_entry2;
	struct efa_ep_addr raw_addr;
	fi_addr_t addr;
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_rdm_ep *efa_rdm_ep;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	/* Create and register a fake peer */
	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0 /* flags */, NULL /* context */);
	assert_int_equal(ret, 1);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	assert_non_null(peer);

	alloc_pke_in_overflow_list(efa_rdm_ep, &pkt_entry2, peer, raw_addr, EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE / 2);
	alloc_pke_in_overflow_list(efa_rdm_ep, &pkt_entry1, peer, raw_addr, EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE / 2);
	assert_int_equal(efa_unit_test_get_dlist_length(&peer->overflow_pke_list), 2);

	/* overflow_pke_list has two pkt entries with msg_id EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE / 2.
	* After calling efa_rdm_peer_move_overflow_pke_to_recvwin when exp_msg_id = EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE,
	* two pkt entries of same msg id will be appended to the same entry in recvwin,
	* and overflow_pke_list will be empty. */
	(&peer->robuf)->exp_msg_id = efa_env.recvwin_size;
	efa_env.rx_copy_ooo = 0;
	efa_rdm_peer_move_overflow_pke_to_recvwin(peer);

	pkt_entry1 = *ofi_recvwin_get_msg((&peer->robuf), EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE / 2);
	assert_non_null(pkt_entry1);
	assert_non_null(pkt_entry1->next);
	assert_int_equal(efa_unit_test_get_dlist_length(&peer->overflow_pke_list), 0);
	assert(pkt_entry1->next == pkt_entry2);

	efa_rdm_pke_release_rx(pkt_entry1->next);
	pkt_entry1->next = NULL;
	efa_rdm_pke_release_rx(pkt_entry1);
}

void test_efa_rdm_peer_recvwin_queue_or_append_pke(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *pkt_entry, *ooo_entry;
	struct efa_ep_addr raw_addr;
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_rdm_ep *efa_rdm_ep;
	int ret;
	fi_addr_t addr;
	struct efa_unit_test_eager_rtm_pkt_attr pkt_attr = {0};
	uint32_t msg_id;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Create and register a fake peer */
	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);

	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0 /* flags */, NULL /* context */);
	assert_int_equal(ret, 1);

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	assert_non_null(peer);

	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	efa_rdm_ep->efa_rx_pkts_posted = efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep);

	/* Not the expected msg id is 0, use 3 as a valid ooo msg id */
	msg_id = 3;
	pkt_attr.msg_id = msg_id;
	pkt_attr.connid = raw_addr.qkey;
	efa_unit_test_eager_msgrtm_pkt_construct(pkt_entry, &pkt_attr);

	ooo_entry = efa_rdm_pke_get_ooo_pke(pkt_entry);

	ret = efa_rdm_peer_recvwin_queue_or_append_pke(ooo_entry, msg_id, (&peer->robuf));
	assert_int_equal(ret, 1);

#if ENABLE_DEBUG
	/* The ooo pkt entry should be inserted to the rx_pkt_list */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->rx_pkt_list), 1);
#endif
}

/**
 * @brief Verify peer_destruct clears EFA_RDM_PKE_RNR_RETRANSMIT and decrements counters
 *
 * When a peer is destructed while RNR retransmit packets are outstanding,
 * the destruct must clear the RNR flag and decrement the ep/peer counters
 * before nulling pkt_entry->peer.
 */
void test_efa_rdm_peer_destruct_clears_rnr_flag(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_cq *efa_rdm_cq;
	struct efa_ibv_cq *ibv_cq;
	struct efa_ep_addr raw_addr;
	fi_addr_t peer_addr;
	size_t raw_addr_len = sizeof(raw_addr);
	uint64_t wr_id;
	int ret;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_unit_test_buff_construct(&send_buff, resource, 4096);

	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 0;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL);
	assert_int_equal(ret, 1);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Send a packet so we get a real pkt_entry on outstanding_tx_pkts */
	g_efa_unit_test_mocks.efa_qp_post_send = &efa_mock_efa_qp_post_send_return_mock;
	will_return(efa_mock_efa_qp_post_send_return_mock, 0);

	ret = fi_send(resource->ep, send_buff.buff, send_buff.size, fi_mr_desc(send_buff.mr), peer_addr, NULL);
	assert_int_equal(ret, 0);

	efa_rdm_cq = container_of(resource->cq, struct efa_rdm_cq, efa_cq.util_cq.cq_fid.fid);
	ibv_cq = &efa_rdm_cq->efa_cq.ibv_cq;
	wr_id = (uint64_t) g_ibv_submitted_wr_id_vec[0];
	pkt_entry = efa_rdm_cq_get_pke_from_wr_id(ibv_cq, wr_id);

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	assert_non_null(peer);

	/* Simulate RNR: record completion and queue for retransmit */
	pkt_entry->ope = container_of(efa_rdm_ep->txe_list.next, struct efa_rdm_ope, ep_entry);
	efa_rdm_ep_record_tx_op_completed(efa_rdm_ep, pkt_entry);
	efa_rdm_ep_queue_rnr_pkt(efa_rdm_ep, pkt_entry);

	assert_int_equal(pkt_entry->flags & EFA_RDM_PKE_RNR_RETRANSMIT, EFA_RDM_PKE_RNR_RETRANSMIT);
	assert_int_equal(efa_rdm_ep->efa_rnr_queued_pkt_cnt, 1);
	assert_int_equal(peer->rnr_queued_pkt_cnt, 1);

	/* Now simulate the pkt being successfully retransmitted and on outstanding_tx_pkts.
	 * Remove from queued_pkts and add to outstanding_tx_pkts. */
	dlist_remove(&pkt_entry->entry);
	pkt_entry->flags &= ~EFA_RDM_PKE_IN_OPE_QUEUED_PKTS;
	dlist_insert_tail(&pkt_entry->entry, &peer->outstanding_tx_pkts);
	pkt_entry->flags |= EFA_RDM_PKE_IN_PEER_OUTSTANDING_TX_PKTS;

	/* Remove the txe from the ope_queued_list so ep close doesn't trip over it */
	pkt_entry->ope->internal_flags &= ~EFA_RDM_OPE_QUEUED_RNR;
	dlist_remove(&pkt_entry->ope->queued_entry);

	/* Remove the peer via fi_av_remove, which calls peer_destruct.
	 * This should clear the RNR flag and decrement counters. */
	ret = fi_av_remove(resource->av, &peer_addr, 1, 0);
	assert_int_equal(ret, 0);

	/* Verify the RNR flag was cleared and counters decremented */
	assert_int_equal(pkt_entry->flags & EFA_RDM_PKE_RNR_RETRANSMIT, 0);
	assert_null(pkt_entry->peer);
	assert_int_equal(efa_rdm_ep->efa_rnr_queued_pkt_cnt, 0);

	/* Release the pkt so the pool can be destroyed cleanly during ep close.
	 * Since peer is NULL and RNR flag is cleared, release_tx won't dereference peer. */
	dlist_remove(&pkt_entry->entry);
	pkt_entry->flags &= ~EFA_RDM_PKE_IN_PEER_OUTSTANDING_TX_PKTS;
	efa_rdm_pke_release_tx(pkt_entry);

	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief PEER_ERROR for a msg_id buffered in the overflow list
 *        removes and frees the entry.
 */
void test_efa_rdm_peer_abort_ooo_in_overflow(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *pkt_entry;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t addr;
	uint32_t msg_id;
	bool ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL), 1);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	assert_non_null(peer);

	/* Place a pke in the overflow list with msg_id beyond the window. */
	msg_id = EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + 5;
	alloc_pke_in_overflow_list(efa_rdm_ep, &pkt_entry, peer, raw_addr, msg_id);
	assert_int_equal(efa_unit_test_get_dlist_length(&peer->overflow_pke_list), 1);

	ret = efa_rdm_peer_abort_ooo_msg(peer, msg_id);
	assert_true(ret);
	assert_int_equal(efa_unit_test_get_dlist_length(&peer->overflow_pke_list), 0);
}

/**
 * @brief PEER_ERROR for a msg_id buffered in the recvwin
 *        tombstones the entry (marks EFA_RDM_PKE_ABORTED).
 */
void test_efa_rdm_peer_abort_ooo_in_recvwin(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *pkt_entry;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_unit_test_eager_rtm_pkt_attr pkt_attr = {0};
	fi_addr_t addr;
	uint32_t msg_id;
	bool ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL), 1);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	assert_non_null(peer);

	/* Queue an OOO pke into the recvwin at msg_id = 3 (exp is 0). */
	msg_id = 3;
	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	efa_rdm_ep->efa_rx_pkts_posted = efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep);

	pkt_attr.msg_id = msg_id;
	pkt_attr.connid = raw_addr.qkey;
	efa_unit_test_eager_msgrtm_pkt_construct(pkt_entry, &pkt_attr);

	efa_env.rx_copy_ooo = 0;
	assert_int_equal(efa_rdm_peer_reorder_msg(peer, efa_rdm_ep, pkt_entry), 1);

	/* The pke should be in the recvwin slot now. */
	assert_non_null(*ofi_recvwin_get_msg(&peer->robuf, msg_id));

	/* Abort it. */
	ret = efa_rdm_peer_abort_ooo_msg(peer, msg_id);
	assert_true(ret);

	/* The pke should still be in the slot (tombstone) but marked ABORTED. */
	struct efa_rdm_pke *slot_pke = *ofi_recvwin_get_msg(&peer->robuf, msg_id);
	assert_non_null(slot_pke);
	assert_true(slot_pke->flags & EFA_RDM_PKE_ABORTED);

	/* Clean up: release it manually since proc_pending won't run. */
	efa_rdm_pke_release_rx(slot_pke);
}

/**
 * @brief PEER_ERROR for a msg_id not in overflow or recvwin
 *        returns false (clean miss, no crash).
 */
void test_efa_rdm_peer_abort_ooo_miss(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t addr;
	bool ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL), 1);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	assert_non_null(peer);

	/* msg_id 42 is nowhere in the reorder state. */
	ret = efa_rdm_peer_abort_ooo_msg(peer, 42);
	assert_false(ret);
}
