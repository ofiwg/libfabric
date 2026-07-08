/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pke_cmd.h"
#include "efa_rdm_pke_nonreq.h"
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

void test_efa_rdm_peer_reorder_expected_msg_id(void **state) {
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


void test_efa_rdm_peer_reorder_smaller_msg_id(void **state) {
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

void test_efa_rdm_peer_reorder_larger_msg_id(void **state) {
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

void test_efa_rdm_peer_reorder_overflow_msg_id(void **state) {
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

void test_efa_rdm_peer_move_overflow_pke_to_recvwin(void **state) {
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

void test_efa_rdm_peer_keep_pke_in_overflow_list(void **state) {
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

void test_efa_rdm_peer_append_overflow_pke_to_recvwin(void **state) {
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

void test_efa_rdm_peer_recvwin_queue_or_append_pke(void **state)
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
void test_efa_rdm_peer_destruct_clears_rnr_flag(void **state)
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
	efa_rdm_pke_set_ope(pkt_entry, efa_unit_test_get_first_ope(efa_rdm_ep, EFA_RDM_TXE));
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
void test_efa_rdm_peer_abort_ooo_in_overflow(void **state)
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
 *        marks the entry as aborted (marks EFA_RDM_PKE_ABORTED).
 */
void test_efa_rdm_peer_abort_ooo_in_recvwin(void **state)
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

	/* The pke should still be in the slot (abort marker) but marked ABORTED. */
	struct efa_rdm_pke *slot_pke = *ofi_recvwin_get_msg(&peer->robuf, msg_id);
	assert_non_null(slot_pke);
	assert_true(slot_pke->flags & EFA_RDM_PKE_ABORTED);

	/* Clean up: release it manually since proc_pending won't run. */
	efa_rdm_pke_release_rx(slot_pke);
}

/**
 * @brief A aborted recvwin entry must not wedge the drain loop: when
 *        the window head reaches an abort marker, proc_pending_items_in_robuf
 *        skips it (frees the pke chain, builds no rxe) and slides past it
 *        to the next buffered slot.
 *
 * Buffer two OOO messages (msg_id 1 and 2), mark both aborted, then simulate
 * the in-order head (msg_id 0) having been processed by sliding the window
 * once so the head lands on the first abort marker. Driving the drain must
 * advance the window past BOTH abort markers (exp_msg_id -> 3), free both
 * slots, build no rxe, and write no user CQ entry.
 */
void test_efa_rdm_peer_abort_ooo_recvwin_drain_progresses(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *pke1, *pke2;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_unit_test_eager_rtm_pkt_attr pkt_attr = {0};
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

	/* This test borrows two rx-pool pkes (pke1, pke2) that the drain
	 * later releases, each incrementing efa_rx_pkts_to_post. Account for
	 * that here so the rx-pkt invariant
	 * (to_post + posted + held == rx_pool_size) still holds when the
	 * final fi_cq_read() drives the progress engine's rx refill. */
	efa_rdm_ep->efa_rx_pkts_posted =
		efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep) - 2;
	efa_env.rx_copy_ooo = 0;	/* keep our pkes in the slots */

	/* Buffer OOO msg_id 1 (exp is 0) into recvwin slot 1. */
	pke1 = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				 EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pke1);
	pkt_attr.msg_id = 1;
	pkt_attr.connid = raw_addr.qkey;
	efa_unit_test_eager_msgrtm_pkt_construct(pke1, &pkt_attr);
	assert_int_equal(efa_rdm_peer_reorder_msg(peer, efa_rdm_ep, pke1), 1);

	/* Buffer OOO msg_id 2 into recvwin slot 2. */
	pke2 = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				 EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pke2);
	pkt_attr.msg_id = 2;
	pkt_attr.connid = raw_addr.qkey;
	efa_unit_test_eager_msgrtm_pkt_construct(pke2, &pkt_attr);
	assert_int_equal(efa_rdm_peer_reorder_msg(peer, efa_rdm_ep, pke2), 1);

	/* Mark both buffered messages aborted. */
	ret = efa_rdm_peer_abort_ooo_msg(peer, 1);
	assert_true(ret);
	ret = efa_rdm_peer_abort_ooo_msg(peer, 2);
	assert_true(ret);
	assert_true((*ofi_recvwin_get_msg(&peer->robuf, 1))->flags & EFA_RDM_PKE_ABORTED);
	assert_true((*ofi_recvwin_get_msg(&peer->robuf, 2))->flags & EFA_RDM_PKE_ABORTED);

	/* Simulate the in-order head (msg_id 0) having been processed:
	 * slide the window once so the head lands on the first abort marker. */
	ofi_recvwin_slide(&peer->robuf);

	/* Drain: the loop must skip both abort markers and slide past them. */
	efa_rdm_peer_proc_pending_items_in_robuf(peer, efa_rdm_ep);

	/* The window advanced past both abort markers. msg_id 1 and 2 are now
	 * processed (behind exp_msg_id), so they must NOT be queried via
	 * ofi_recvwin_get_msg() -- it asserts the id is still in-window.
	 * Verify they were consumed with ofi_recvwin_id_processed() instead. */
	assert_true(ofi_recvwin_id_processed(&peer->robuf, 1));
	assert_true(ofi_recvwin_id_processed(&peer->robuf, 2));
	assert_int_equal((&peer->robuf)->exp_msg_id, 3);

	/* No rxe was built and no user CQ entry was written for the
	 * aborted messages. */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->base_ep.ope_list), 0);
	assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);
}

/**
 * @brief PEER_ERROR for a msg_id not in overflow or recvwin
 *        returns false (clean miss, no crash).
 */
void test_efa_rdm_peer_abort_ooo_miss(void **state)
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

/*
 * Build an inbound PEER_ERROR (skip) packet for msg_id and drive it
 * through efa_rdm_peer_queue_aborted_msg_marker(). Mirrors how the
 * real handler (efa_rdm_pke_handle_peer_error_recv) consumes the packet.
 * Returns the helper's return value.
 */
static int deliver_peer_error_skip(struct efa_rdm_ep *efa_rdm_ep,
				   struct efa_rdm_peer *peer, uint32_t msg_id)
{
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_peer_error_hdr *err_hdr;

	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	pkt_entry->peer = peer;

	err_hdr = (struct efa_rdm_peer_error_hdr *) pkt_entry->wiredata;
	err_hdr->type = EFA_RDM_PEER_ERROR_PKT;
	err_hdr->direction = EFA_RDM_PEER_ERROR_TX_TO_RX;
	err_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	/* msg_id-only packet (no op_id hint): the receiver decides -- with no
	 * matched rxe it advances the reorder window past msg_id. */
	err_hdr->flags = EFA_RDM_PKT_CONNID_HDR;
	err_hdr->op_id_valid = 0;
	err_hdr->msg_id = msg_id;
	err_hdr->op_id = 0;
	err_hdr->prov_errno = EFA_IO_COMP_STATUS_FLUSHED;
	err_hdr->connid = 0xbeef;
	pkt_entry->pkt_size = sizeof(struct efa_rdm_peer_error_hdr);

	return efa_rdm_peer_queue_aborted_msg_marker(peer, efa_rdm_ep,
							pkt_entry, msg_id);
}

/**
 * @brief The core fix: a source-aborted msg_id that NEVER arrived must
 *        not head-of-line block the messages buffered behind it.
 *
 * The sender allocated msg_id 0, posted it, then the device flushed it
 * at the source (MR closed) before it ever reached the receiver. Later
 * messages msg_id 1 and 2 DID arrive and are buffered out-of-order,
 * waiting for msg_id 0; the window must slide past 0 or they are
 * stranded forever.
 *
 * An inbound PEER_ERROR (skip) packet for msg_id 0 is queued into the
 * recvwin as an abort marker; the drain then slides past 0 and continues
 * past the buffered 1 and 2 (aborted here so the test releases them
 * cleanly without the full recv-match/CQ machinery), leaving
 * exp_msg_id == 3.
 */
void test_efa_rdm_peer_skip_aborted_msg_id_never_arrived_unblocks_window(
	void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *pke1, *pke2;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_unit_test_eager_rtm_pkt_attr pkt_attr = {0};
	fi_addr_t addr;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL), 1);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	assert_non_null(peer);

	/* pke1, pke2 and the PEER_ERROR abort marker packet are all rx-pool
	 * pkes the drain releases; keep the rx-pkt accounting invariant
	 * intact (see the abort_ooo drain test). */
	efa_rdm_ep->efa_rx_pkts_posted =
		efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep) - 3;
	efa_env.rx_copy_ooo = 0;	/* keep our pkes in the slots */

	/* exp_msg_id starts at 0. msg_id 0 will never arrive. */

	/* msg_id 1 arrives OOO -> buffered in recvwin slot 1. */
	pke1 = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				 EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pke1);
	pkt_attr.msg_id = 1;
	pkt_attr.connid = raw_addr.qkey;
	efa_unit_test_eager_msgrtm_pkt_construct(pke1, &pkt_attr);
	assert_int_equal(efa_rdm_peer_reorder_msg(peer, efa_rdm_ep, pke1), 1);

	/* msg_id 2 arrives OOO -> buffered in recvwin slot 2. */
	pke2 = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				 EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pke2);
	pkt_attr.msg_id = 2;
	pkt_attr.connid = raw_addr.qkey;
	efa_unit_test_eager_msgrtm_pkt_construct(pke2, &pkt_attr);
	assert_int_equal(efa_rdm_peer_reorder_msg(peer, efa_rdm_ep, pke2), 1);

	/* Mark the buffered 1 and 2 aborted so the drain releases them without
	 * the full recv-match path (their delivery is exercised end-to-end
	 * by the fabtest; here we focus on window advancement). */
	assert_true(efa_rdm_peer_abort_ooo_msg(peer, 1));
	assert_true(efa_rdm_peer_abort_ooo_msg(peer, 2));

	/* Window is parked on msg_id 0 (head slot empty). */
	assert_int_equal((&peer->robuf)->exp_msg_id, 0);
	assert_null(*ofi_recvwin_peek((&peer->robuf)));

	/* Inbound PEER_ERROR (skip) for the never-arrived msg_id 0: queued
	 * as an abort marker, then the drain advances the window past 0 and the
	 * (aborted) 1 and 2. */
	ret = deliver_peer_error_skip(efa_rdm_ep, peer, 0);
	assert_int_equal(ret, 1);

	assert_true(ofi_recvwin_id_processed(&peer->robuf, 0));
	assert_true(ofi_recvwin_id_processed(&peer->robuf, 1));
	assert_true(ofi_recvwin_id_processed(&peer->robuf, 2));
	assert_int_equal((&peer->robuf)->exp_msg_id, 3);

	/* No rxe built, no user CQ entry for any skipped id. */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->base_ep.ope_list), 0);
	assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);
}

/**
 * @brief A never-arrived msg_id that is the exact head of the window
 *        (exp_msg_id) with nothing buffered behind it: the abort marker is
 *        queued and the window advances by one on the same call.
 */
void test_efa_rdm_peer_skip_aborted_msg_id_head_advances(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t addr;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL), 1);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	assert_non_null(peer);

	/* One rx-pool pke (the abort marker) is released by the drain. */
	efa_rdm_ep->efa_rx_pkts_posted =
		efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep) - 1;
	efa_env.rx_copy_ooo = 0;

	(&peer->robuf)->exp_msg_id = 5;

	ret = deliver_peer_error_skip(efa_rdm_ep, peer, 5);
	assert_int_equal(ret, 1);

	/* Window advanced exactly one past the head. */
	assert_true(ofi_recvwin_id_processed(&peer->robuf, 5));
	assert_int_equal((&peer->robuf)->exp_msg_id, 6);
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->base_ep.ope_list), 0);
	assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);
}

/**
 * @brief PEER_ERROR (skip) for an already-processed msg_id (window slid
 *        past it) is a clean no-op: the packet is released, return 0,
 *        window unchanged.
 */
void test_efa_rdm_peer_skip_aborted_msg_id_already_processed_noop(
	void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t addr;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL), 1);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	assert_non_null(peer);

	/* The abort marker packet is released immediately (case 1), counting
	 * toward efa_rx_pkts_to_post. */
	efa_rdm_ep->efa_rx_pkts_posted =
		efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep) - 1;

	/* Advance the window so msg_id 0..4 are already processed. */
	(&peer->robuf)->exp_msg_id = 5;

	/* A late/duplicate skip for already-processed msg_id 2: no-op. */
	ret = deliver_peer_error_skip(efa_rdm_ep, peer, 2);
	assert_int_equal(ret, 0);
	assert_int_equal((&peer->robuf)->exp_msg_id, 5);
}

/**
 * @brief PEER_ERROR (skip) for a msg_id whose first segment WAS buffered
 *        out-of-order marks the buffered segment aborted in place (case 2)
 *        and releases the control packet, rather than queueing a second
 *        abort marker. The window then advances when the head is skipped.
 */
void test_efa_rdm_peer_skip_aborted_msg_id_buffered_abort_markers(
	void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *pke1;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_unit_test_eager_rtm_pkt_attr pkt_attr = {0};
	fi_addr_t addr;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL), 1);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	assert_non_null(peer);

	/* Three rx-pool pkes are allocated and returned to the pool by the
	 * release/drain paths: pke1 (buffered, drained), the control packet
	 * for msg_id 1 (released in case 2), and the control packet for
	 * msg_id 0 (queued as abort marker, drained). */
	efa_rdm_ep->efa_rx_pkts_posted =
		efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep) - 3;
	efa_env.rx_copy_ooo = 0;

	/* msg_id 1 arrives OOO (exp is 0) -> buffered in recvwin slot 1. */
	pke1 = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				 EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pke1);
	pkt_attr.msg_id = 1;
	pkt_attr.connid = raw_addr.qkey;
	efa_unit_test_eager_msgrtm_pkt_construct(pke1, &pkt_attr);
	assert_int_equal(efa_rdm_peer_reorder_msg(peer, efa_rdm_ep, pke1), 1);

	/* Skip the BUFFERED msg_id 1: case 2 marks it aborted in place. */
	ret = deliver_peer_error_skip(efa_rdm_ep, peer, 1);
	assert_int_equal(ret, 1);

	/* The buffered pke at slot 1 is now aborted. */
	assert_true((*ofi_recvwin_get_msg(&peer->robuf, 1))->flags &
		    EFA_RDM_PKE_ABORTED);

	/* Window still parked on the never-arrived head msg_id 0. */
	assert_int_equal((&peer->robuf)->exp_msg_id, 0);

	/* Skip the never-arrived head msg_id 0: window advances past 0 and
	 * the aborted 1. */
	ret = deliver_peer_error_skip(efa_rdm_ep, peer, 0);
	assert_int_equal(ret, 1);
	assert_true(ofi_recvwin_id_processed(&peer->robuf, 0));
	assert_true(ofi_recvwin_id_processed(&peer->robuf, 1));
	assert_int_equal((&peer->robuf)->exp_msg_id, 2);

	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->base_ep.ope_list), 0);
	assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);
}

/**
 * @brief An abort marker queued behind the window head (not the most recent
 *        slot) must wait, then be swept by the cascade once the head is
 *        filled.
 *
 * Order of arrival is deliberately back-to-front: the PEER_ERROR (skip)
 * for the highest msg_id arrives first and is queued behind the still-
 * empty head, so it must NOT advance the window. Only when the head slot
 * itself is finally skipped does the drain cascade forward and sweep all
 * the queued abort markers in one pass.
 *
 * This exercises the case where the abort marker is not at the head: it sits
 * in the recvwin until an earlier slot is resolved, exactly like a
 * buffered OOO RTM.
 */
void test_efa_rdm_peer_skip_aborted_msg_id_abort_marker_behind_head(
	void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t addr;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL), 1);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	assert_non_null(peer);

	/* Three abort marker packets are queued and later drained, each
	 * returning to the rx pool. */
	efa_rdm_ep->efa_rx_pkts_posted =
		efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep) - 3;
	efa_env.rx_copy_ooo = 0;

	/* exp_msg_id == 0. Skip for msg_id 2 arrives first: it is queued at
	 * slot 2, behind the empty head 0, so the window must NOT move. */
	ret = deliver_peer_error_skip(efa_rdm_ep, peer, 2);
	assert_int_equal(ret, 1);
	assert_int_equal((&peer->robuf)->exp_msg_id, 0);
	assert_true((*ofi_recvwin_get_msg(&peer->robuf, 2))->flags &
		    EFA_RDM_PKE_ABORTED);

	/* Skip for msg_id 1 arrives next: still behind the head, no move. */
	ret = deliver_peer_error_skip(efa_rdm_ep, peer, 1);
	assert_int_equal(ret, 1);
	assert_int_equal((&peer->robuf)->exp_msg_id, 0);
	assert_true((*ofi_recvwin_get_msg(&peer->robuf, 1))->flags &
		    EFA_RDM_PKE_ABORTED);

	/* Skip for the head msg_id 0 arrives last: now the drain cascades
	 * over 0, then the already-queued abort markers 1 and 2, in one pass. */
	ret = deliver_peer_error_skip(efa_rdm_ep, peer, 0);
	assert_int_equal(ret, 1);

	assert_true(ofi_recvwin_id_processed(&peer->robuf, 0));
	assert_true(ofi_recvwin_id_processed(&peer->robuf, 1));
	assert_true(ofi_recvwin_id_processed(&peer->robuf, 2));
	assert_int_equal((&peer->robuf)->exp_msg_id, 3);

	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->base_ep.ope_list), 0);
	assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);
}

/**
 * @brief RX side of the LONGCTS pre-CTS abort fix: an inbound msg_id-only
 *        PEER_ERROR_PKT for a LONGCTS msg_id that never arrived unblocks
 *        the reorder window WITHOUT producing a completion -- driven
 *        through the full inbound dispatcher
 *        efa_rdm_pke_handle_peer_error_recv().
 *
 * Counterpart to the sender-side tests
 * (test_efa_rdm_txe_handle_error_longcts_prepost_cancel_emits_skip /
 * test_efa_rdm_pke_init_peer_error_for_ope_longcts_pre_cts_skip): a
 * LONGCTS sender aborted before its first CTS has no receiver rxe index,
 * so it emits a msg_id-only packet (no op_id hint). On the wire that
 * packet is indistinguishable from an EAGER/medium msg_id-only abort, so
 * the receiver path is shared; with no matched rxe the receiver-decides
 * dispatcher routes to efa_rdm_peer_queue_aborted_msg_marker(). The other
 * skip tests drive that helper directly, whereas this one exercises the
 * actual receive entry point (efa_rdm_pke_handle_peer_error_recv) with
 * the packet a sender emits for a LONGCTS RTM that never arrived.
 *
 * msg_id 0 (the aborted LONGCTS RTM) never arrives; msg_id 1 arrives OOO
 * and is buffered (then aborted so the drain releases it without the
 * full recv-match/CQ path). Dispatching the packet for 0 abort markers
 * it and slides the window past the buffered 1, leaving exp_msg_id == 2,
 * with no rxe and no CQ entry.
 */
void test_efa_rdm_pke_handle_peer_error_recv_longcts_skip_unblocks_window(
	void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *pke1, *skip_pkt;
	struct efa_rdm_peer_error_hdr *err_hdr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_unit_test_eager_rtm_pkt_attr pkt_attr = {0};
	fi_addr_t addr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL), 1);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	assert_non_null(peer);

	/* pke1 (buffered) and the PEER_ERROR packet are rx-pool pkes the drain
	 * releases; keep the rx-pkt accounting invariant intact. */
	efa_rdm_ep->efa_rx_pkts_posted =
		efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep) - 2;
	efa_env.rx_copy_ooo = 0;	/* keep our pkes in the slots */

	/* exp_msg_id starts at 0; msg_id 0 (the LONGCTS RTM) never arrives. */

	/* msg_id 1 arrives OOO -> buffered in recvwin slot 1, then aborted
	 * so the drain releases it without the full recv-match path. */
	pke1 = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				 EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pke1);
	pkt_attr.msg_id = 1;
	pkt_attr.connid = raw_addr.qkey;
	efa_unit_test_eager_msgrtm_pkt_construct(pke1, &pkt_attr);
	assert_int_equal(efa_rdm_peer_reorder_msg(peer, efa_rdm_ep, pke1), 1);
	assert_true(efa_rdm_peer_abort_ooo_msg(peer, 1));

	/* Window is parked on msg_id 0 (head slot empty) -- the hang state. */
	assert_int_equal((&peer->robuf)->exp_msg_id, 0);
	assert_null(*ofi_recvwin_peek((&peer->robuf)));

	/* Build the inbound PEER_ERROR (skip) packet for the never-arrived
	 * LONGCTS msg_id 0 and deliver it through the FULL dispatcher (the
	 * actual receive entry point), not the abort marker helper. The
	 * dispatcher finds no matched rxe and routes to the abort marker path,
	 * consuming the packet. */
	skip_pkt = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				     EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(skip_pkt);
	skip_pkt->peer = peer;
	err_hdr = (struct efa_rdm_peer_error_hdr *) skip_pkt->wiredata;
	err_hdr->type = EFA_RDM_PEER_ERROR_PKT;
	err_hdr->direction = EFA_RDM_PEER_ERROR_TX_TO_RX;
	err_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	/* msg_id-only (no op_id hint): the dispatcher finds no matched rxe and
	 * routes to the abort-marker path, consuming the packet. */
	err_hdr->flags = EFA_RDM_PKT_CONNID_HDR;
	err_hdr->op_id_valid = 0;
	err_hdr->msg_id = 0;
	err_hdr->op_id = 0;
	err_hdr->prov_errno = EFA_IO_COMP_STATUS_FLUSHED;
	err_hdr->connid = raw_addr.qkey;
	skip_pkt->pkt_size = sizeof(struct efa_rdm_peer_error_hdr);

	efa_rdm_pke_handle_peer_error_recv(skip_pkt);

	/* Window advanced past the aborted 0 and the aborted 1. */
	assert_true(ofi_recvwin_id_processed(&peer->robuf, 0));
	assert_true(ofi_recvwin_id_processed(&peer->robuf, 1));
	assert_int_equal((&peer->robuf)->exp_msg_id, 2);

	/* No rxe built and no user CQ entry: a LONGCTS pre-CTS abort owes no
	 * completion (no recv was ever matched, since no CTS was exchanged). */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->base_ep.ope_list), 0);
	assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);
}

/**
 * @brief Regression test for the MEDIUM overflow-list abort leak.
 *
 * A multi-segment MEDIUM message that arrives out-of-order while the
 * reorder window is full lands in the peer's overflow_pke_list. Unlike
 * the recvwin (which chains same-msg_id segments into a single slot via
 * recvwin_queue_or_append_pke), the overflow path stores EACH segment as
 * a SEPARATE overflow_pke_list_entry. So one msg_id can have multiple
 * overflow entries.
 *
 * efa_rdm_peer_abort_ooo_msg() must remove ALL overflow entries for the
 * aborted msg_id. The bug: it removes only the FIRST match and returns,
 * leaking the remaining segments (never released until ep close) and
 * potentially wedging overflow promotion for a msg_id the window has
 * already slid past.
 *
 * This test stages two segments of the same OOO medium msg_id in the
 * overflow list, aborts it, and asserts the overflow list is fully
 * drained. It FAILS against the buggy single-match implementation.
 */
void test_efa_rdm_peer_abort_ooo_msg_overflow_multi_segment(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *seg0, *seg1, *other;
	struct efa_ep_addr raw_addr;
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t addr;
	const uint32_t aborted_msg_id =
		EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + 5;
	const uint32_t other_msg_id =
		EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + 6;
	bool ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr,
				    &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &addr, 0,
				      NULL), 1);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	assert_non_null(peer);

	/* Window expects msg_id 0; these ids are out of window -> overflow. */

	/* Two segments of the SAME medium msg_id -> two overflow entries. */
	alloc_pke_in_overflow_list(efa_rdm_ep, &seg0, peer, raw_addr,
				   aborted_msg_id);
	alloc_pke_in_overflow_list(efa_rdm_ep, &seg1, peer, raw_addr,
				   aborted_msg_id);
	/* A segment of a DIFFERENT msg_id, to confirm the abort only
	 * removes the targeted id and leaves others intact. */
	alloc_pke_in_overflow_list(efa_rdm_ep, &other, peer, raw_addr,
				   other_msg_id);

	assert_int_equal(efa_unit_test_get_dlist_length(&peer->overflow_pke_list),
			 3);

	/* Abort the multi-segment msg_id. */
	ret = efa_rdm_peer_abort_ooo_msg(peer, aborted_msg_id);
	assert_true(ret);

	/*
	 * BUG ASSERTION: every overflow entry for aborted_msg_id must be
	 * gone. Only the unrelated other_msg_id entry should remain.
	 * The buggy single-match implementation leaves 2 entries (one
	 * leaked aborted segment + the other msg_id).
	 */
	assert_int_equal(efa_unit_test_get_dlist_length(&peer->overflow_pke_list),
			 1);

	/* The surviving entry is the unrelated msg_id. */
	{
		struct efa_rdm_peer_overflow_pke_list_entry *e;
		struct dlist_entry *tmp;
		uint32_t found = 0;

		dlist_foreach_container_safe(&peer->overflow_pke_list,
			struct efa_rdm_peer_overflow_pke_list_entry,
			e, entry, tmp) {
			found = efa_rdm_pke_get_rtm_msg_id(e->pkt_entry);
			assert_int_equal(found, other_msg_id);
			/* clean up the survivor */
			dlist_remove(&e->entry);
			efa_rdm_pke_release_rx_list(e->pkt_entry);
			ofi_buf_free(e);
		}
	}
}
