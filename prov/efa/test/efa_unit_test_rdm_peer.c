/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"

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
	efa_rdm_ep->efa_rx_pkts_posted = efa_rdm_ep_get_rx_pool_size(efa_rdm_ep);

	pkt_attr.msg_id = msg_id;
	pkt_attr.connid = raw_addr.qkey;
	/* Packet type must be >= EFA_RDM_REQ_PKT_BEGIN */
	efa_unit_test_eager_msgrtm_pkt_construct(pkt_entry, &pkt_attr);

	(&peer->robuf)->exp_msg_id = exp_msg_id;
	ret = efa_rdm_peer_reorder_msg(peer, efa_rdm_ep, pkt_entry);
	assert_int_equal(ret, expected_ret);
	(&peer->robuf)->exp_msg_id = 0;

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
}


void test_efa_rdm_peer_reorder_smaller_msg_id(struct efa_resource **state) {
	struct efa_resource *resource = *state;
	uint32_t msg_id, exp_msg_id;
	int expected_ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	msg_id = 1;
	exp_msg_id = 10;
	expected_ret = -FI_EALREADY;
	/* Receiving message id smaller than expected should return -FI_EALREADY */
	test_efa_rdm_peer_reorder_msg_impl(resource, exp_msg_id, msg_id, expected_ret);
}

void test_efa_rdm_peer_reorder_larger_msg_id(struct efa_resource **state) {
	struct efa_resource *resource = *state;
	uint32_t msg_id, exp_msg_id;
	int expected_ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	msg_id = 10;
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

	msg_id = EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE;
	exp_msg_id = 0;
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
	efa_rdm_ep->efa_rx_pkts_posted = efa_rdm_ep_get_rx_pool_size(efa_rdm_ep);

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

	/* overflow_pke_list has a pkt entry with msg_id EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + 1000.
	 * After calling efa_rdm_peer_move_overflow_pke_to_recvwin when exp_msg_id = EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE,
	 * EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + 1000 will be moved to recvwin and overflow_pke_list will be empty. */
	test_efa_rdm_peer_move_overflow_pke_to_recvwin_impl(
		resource, EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + 1000, &peer, &pkt_entry);

	assert_non_null(*ofi_recvwin_get_msg((&peer->robuf), EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + 1000));
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
	efa_rdm_ep->efa_rx_pkts_posted = efa_rdm_ep_get_rx_pool_size(efa_rdm_ep);

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

	alloc_pke_in_overflow_list(efa_rdm_ep, &pkt_entry2, peer, raw_addr, EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + 500);
	alloc_pke_in_overflow_list(efa_rdm_ep, &pkt_entry1, peer, raw_addr, EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + 500);
	assert_int_equal(efa_unit_test_get_dlist_length(&peer->overflow_pke_list), 2);

	/* overflow_pke_list has two pkt entries with msg_id EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + 500.
	* After calling efa_rdm_peer_move_overflow_pke_to_recvwin when exp_msg_id = EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE,
	* two pkt entries of same msg id will be appended to the same entry in recvwin,
	* and overflow_pke_list will be empty. */
	(&peer->robuf)->exp_msg_id = efa_env.recvwin_size;
	efa_env.rx_copy_ooo = 0;
	efa_rdm_peer_move_overflow_pke_to_recvwin(peer);

	pkt_entry1 = *ofi_recvwin_get_msg((&peer->robuf), EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE + 500);
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
	struct efa_rdm_pke *pkt_entry;
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
	efa_rdm_ep->efa_rx_pkts_posted = efa_rdm_ep_get_rx_pool_size(efa_rdm_ep);

	/* Not the expected msg id is 0, use 3 as a valid ooo msg id */
	msg_id = 3;
	pkt_attr.msg_id = msg_id;
	pkt_attr.connid = raw_addr.qkey;
	efa_unit_test_eager_msgrtm_pkt_construct(pkt_entry, &pkt_attr);

	ret = efa_rdm_peer_recvwin_queue_or_append_pke(pkt_entry, msg_id, (&peer->robuf));
	assert_int_equal(ret, 1);

#if ENABLE_DEBUG
	/* The ooo pkt entry should be inserted to the rx_pkt_list */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->rx_pkt_list), 1);
#endif
}
