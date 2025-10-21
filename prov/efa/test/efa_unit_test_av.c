/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"
#include "efa_rdm_cq.h"
#include "efa_rdm_pke_req.h"
#include "efa_av.h"

/**
 * @brief Only works on nodes with EFA devices
 * This test calls fi_av_insert() twice with the same raw address,
 * and verifies that returned fi_addr is the same.
 * Since the addresses to be inserted have the same GID with the ep's self ah,
 * there should be only 1 ibv_create_ah call in the whole test.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_av_insert_duplicate_raw_addr(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;

	g_efa_unit_test_mocks.ibv_create_ah = &efa_mock_ibv_create_ah_check_mock;
	/* the following will_return ensures ibv_create_ah is called exactly once */
	will_return(efa_mock_ibv_create_ah_check_mock, 0);

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr1, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);

	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr2, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);
	assert_int_equal(addr1, addr2);
}

/**
 * @brief Only works on nodes with EFA devices
 * This test calls fi_av_insert() twice with two difference raw address with same GID,
 * and verifies that returned fi_addr is different.
 * Since the addresses to be inserted have the same GID with the ep's self ah,
 * there should be only 1 ibv_create_ah call in the whole test.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_av_insert_duplicate_gid(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;

	g_efa_unit_test_mocks.ibv_create_ah = &efa_mock_ibv_create_ah_check_mock;
	/* the following will_return ensures ibv_create_ah is called exactly once */
	will_return(efa_mock_ibv_create_ah_check_mock, 0);

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr1, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);

	raw_addr.qpn = 2;
	raw_addr.qkey = 0x5678;
	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr2, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);
	assert_int_not_equal(addr1, addr2);
}

void test_efa_ah_cnt_one_av(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;
	struct efa_domain *efa_domain;
	struct efa_ah *efa_ah = NULL;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);

	/* So far we should only have 1 ah from ep self ah, and its refcnt is 1 */
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	HASH_FIND(hh, efa_domain->ah_map, raw_addr.raw, EFA_GID_LEN, efa_ah);
	assert_non_null(efa_ah);
	assert_int_equal(efa_ah->explicit_refcnt, 1);
	assert_int_equal(efa_ah->implicit_refcnt, 0);

	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr1, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);

	raw_addr.qpn = 2;
	raw_addr.qkey = 0x5678;
	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr2, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);
	assert_int_not_equal(addr1, addr2);

	/* So far we should still have 1 ah, and its refcnt is 3 (plus the 2 av entries) */
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	assert_int_equal(efa_ah->explicit_refcnt, 3);
	assert_int_equal(efa_ah->implicit_refcnt, 0);

	/* ah refcnt should be decremented to 1 after av entry removals */
	assert_int_equal(fi_av_remove(resource->av, &addr1, 1, 0), 0);
	assert_int_equal(fi_av_remove(resource->av, &addr2, 1, 0), 0);

	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	assert_int_equal(efa_ah->explicit_refcnt, 1);
	assert_int_equal(efa_ah->implicit_refcnt, 0);

	/* ah map should be empty now after closing ep which destroys the self ah */
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 0);
	/* Reset to NULL to avoid test reaper closing again */
	resource->ep = NULL;
}

void test_efa_ah_cnt_multi_av(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;
	struct efa_domain *efa_domain;
	struct efa_ah *efa_ah = NULL;
	struct fi_av_attr av_attr = {0};
	struct fid_av *av1, *av2;
	struct fid_ep *ep1, *ep2;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);

	/* So far we should only have 1 ah from ep self ah, and its refcnt is 1 */
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	HASH_FIND(hh, efa_domain->ah_map, raw_addr.raw, EFA_GID_LEN, efa_ah);
	assert_non_null(efa_ah);
	assert_int_equal(efa_ah->explicit_refcnt, 1);
	assert_int_equal(efa_ah->implicit_refcnt, 0);


	/* We open 2 avs with the same domain (PD) so they should share same AH given the same GID */
	assert_int_equal(fi_av_open(resource->domain, &av_attr, &av1, NULL), 0);
	assert_int_equal(fi_av_open(resource->domain, &av_attr, &av2, NULL), 0);

	/* Due to the current restriction in efa provider, we have to bind av to ep before inserting av entry */
	/* These eps will not create self ah as they are not enabled */
	assert_int_equal(fi_endpoint(resource->domain, resource->info, &ep1, NULL), 0);
	assert_int_equal(fi_endpoint(resource->domain, resource->info, &ep2, NULL), 0);

	assert_int_equal(fi_ep_bind(ep1, &av1->fid, 0), 0);
	assert_int_equal(fi_ep_bind(ep2, &av2->fid, 0), 0);

	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	num_addr = fi_av_insert(av1, &raw_addr, 1, &addr1, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);

	raw_addr.qpn = 2;
	raw_addr.qkey = 0x5678;
	num_addr = fi_av_insert(av2, &raw_addr, 1, &addr2, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);
	/* They should be as equal as 0 they are in different avs */
	assert_int_equal(addr1, addr2);

	/* So far we should still have 1 ah, and its refcnt is 3 (plus the 2 av entries) */
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	assert_int_equal(efa_ah->explicit_refcnt, 3);
	assert_int_equal(efa_ah->implicit_refcnt, 0);

	/* ah refcnt should be decremented to 1 after av close */
	assert_int_equal(fi_close(&ep1->fid), 0);
	assert_int_equal(fi_close(&ep2->fid), 0);
	assert_int_equal(fi_close(&av1->fid), 0);
	assert_int_equal(fi_close(&av2->fid), 0);

	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	assert_int_equal(efa_ah->explicit_refcnt, 1);
	assert_int_equal(efa_ah->implicit_refcnt, 0);

	/* ah map should be empty now after closing ep which destroys the self ah */
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 0);
	/* Reset to NULL to avoid test reaper closing again */
	resource->ep = NULL;
}

/**
 * @brief This test verifies that multiple endpoints can bind to the same AV
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_av_multiple_ep_impl(struct efa_resource **state, char *fabric_name)
{
	struct efa_resource *resource = *state;
	struct fid_ep *ep1, *ep2;
	int ret;

	/* Resource construct function creates and binds 1 EP to the AV */
	efa_unit_test_resource_construct(resource, FI_EP_RDM, fabric_name);

	/* Create and bind two new endpoints to the same AV */
	fi_endpoint(resource->domain, resource->info, &ep1, NULL);
	ret = fi_ep_bind(ep1, &resource->av->fid, 0);
	assert_int_equal(ret, 0);

	fi_endpoint(resource->domain, resource->info, &ep2, NULL);
	ret = fi_ep_bind(ep2, &resource->av->fid, 0);
	assert_int_equal(ret, 0);

	/* Bind the two new endpoints to the same CQ and enable them */
	fi_ep_bind(ep1, &resource->cq->fid, FI_SEND | FI_RECV);
	ret = fi_enable(ep1);
	assert_int_equal(ret, 0);

	fi_ep_bind(ep2, &resource->cq->fid, FI_SEND | FI_RECV);
	ret = fi_enable(ep2);
	assert_int_equal(ret, 0);

	fi_close(&ep1->fid);
	fi_close(&ep2->fid);
}


/**
 * @brief This test verifies that multiple endpoints can bind to the same AV
 * for the efa fabric
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_av_multiple_ep_efa(struct efa_resource **state)
{
	return test_av_multiple_ep_impl(state, EFA_FABRIC_NAME);
}

/**
 * @brief This test verifies that multiple endpoints can bind to the same AV
 * for the efa-direct fabric
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_av_multiple_ep_efa_direct(struct efa_resource **state)
{
	return test_av_multiple_ep_impl(state, EFA_DIRECT_FABRIC_NAME);
}

static void test_av_verify_av_hash_cnt(struct efa_av *av,
				       int explicit_cur_av_count,
				       int explicit_prv_av_count,
				       int implicit_cur_av_count,
				       int implicit_prv_av_count)
{
	assert_int_equal(HASH_CNT(hh, av->util_av.hash),
			 explicit_cur_av_count + explicit_prv_av_count);
	assert_int_equal(HASH_CNT(hh, av->cur_reverse_av),
			 explicit_cur_av_count);
	assert_int_equal(HASH_CNT(hh, av->prv_reverse_av),
			 explicit_prv_av_count);

	assert_int_equal(HASH_CNT(hh, av->util_av_implicit.hash),
			 implicit_cur_av_count + implicit_prv_av_count);
	assert_int_equal(HASH_CNT(hh, av->cur_reverse_av_implicit),
			 implicit_cur_av_count);
	assert_int_equal(HASH_CNT(hh, av->prv_reverse_av_implicit),
			 implicit_prv_av_count);
}

/**
 * @brief This test removes a peer and inserts it again
 *
 * @param[in]	state	struct efa_resource that is managed by the framework
 */
void test_av_reinsertion(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr, raw_addr_2;
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t fi_addr;
	struct efa_av *av;
	struct efa_rdm_ep *efa_rdm_ep;
	int err;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 174;
	raw_addr.qkey = 0x1234;

	av = container_of(resource->av, struct efa_av, util_av.av_fid);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	err = fi_av_insert(resource->av, &raw_addr, 1, &fi_addr, 0, NULL);
	assert_int_equal(err, 1);
	assert_int_equal(fi_addr, 0);
	test_av_verify_av_hash_cnt(av, 1, 0, 0, 0);

	err = fi_av_lookup(resource->av, fi_addr, &raw_addr_2, &raw_addr_len);
	assert_int_equal(err, 0);
	assert_int_equal(efa_is_same_addr(&raw_addr, &raw_addr_2), 1);

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, fi_addr);
	assert_int_equal(peer->conn->fi_addr, fi_addr);
	assert_int_equal(efa_is_same_addr(&raw_addr, peer->conn->ep_addr), 1);

	err = fi_av_remove(resource->av, &fi_addr, 1, 0);
	assert_int_equal(err, 0);
	test_av_verify_av_hash_cnt(av, 0, 0, 0, 0);

	err = fi_av_insert(resource->av, &raw_addr, 1, &fi_addr, 0, NULL);
	assert_int_equal(err, 1);
	assert_int_equal(fi_addr, 0);
	test_av_verify_av_hash_cnt(av, 1, 0, 0, 0);

	err = fi_av_lookup(resource->av, fi_addr, &raw_addr_2, &raw_addr_len);
	assert_int_equal(err, 0);
	assert_int_equal(efa_is_same_addr(&raw_addr, &raw_addr_2), 1);

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, fi_addr);
	assert_int_equal(peer->conn->fi_addr, fi_addr);
	assert_int_equal(efa_is_same_addr(&raw_addr, peer->conn->ep_addr), 1);

	err = fi_av_remove(resource->av, &fi_addr, 1, 0);
	assert_int_equal(err, 0);
	test_av_verify_av_hash_cnt(av, 0, 0, 0, 0);
}

/**
 * @brief Generate a peer with random QPN and QKEY and insert it into the implicit AV
 *
 * @param[in]	state	struct efa_resource that is managed by the framework
 */
static struct efa_rdm_peer *test_av_get_peer_from_implicit_av(struct efa_resource *resource)
{
	struct efa_ep_addr raw_addr;
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	fi_addr_t implicit_fi_addr, test_addr;
	struct efa_av *av;
	uint32_t ahn;
	int err;

	av = container_of(resource->av, struct efa_av, util_av.av_fid);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);

	raw_addr.qpn = rand();
	raw_addr.qkey = rand();
	ahn = efa_rdm_ep->base_ep.self_ah->ahn;

	/* Manually insert into implicit AV */
	ofi_genlock_lock(&efa_rdm_ep->base_ep.domain->srx_lock);

	err = efa_av_insert_one(av, &raw_addr, &implicit_fi_addr, 0, NULL, true, true);

	peer = efa_rdm_ep_get_peer_implicit(efa_rdm_ep, implicit_fi_addr);

	assert_int_equal(peer->conn->implicit_fi_addr, implicit_fi_addr);
	assert_int_equal(peer->conn->fi_addr, FI_ADDR_NOTAVAIL);
	assert_int_equal(efa_is_same_addr(&raw_addr, peer->conn->ep_addr), 1);

	test_addr = efa_av_reverse_lookup_rdm_implicit(av, ahn, raw_addr.qpn, NULL);
	assert_int_equal(test_addr, implicit_fi_addr);

	ofi_genlock_unlock(&efa_rdm_ep->base_ep.domain->srx_lock);

	return peer;
}

/**
 * @brief This test fakes a peer in the implicit AV and closes the AV with an
 * implicit peer in it
 *
 * @param[in]	state	struct efa_resource that is managed by the framework
 */
void test_av_implicit(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	test_av_get_peer_from_implicit_av(resource);
}

/**
 * @brief This test fakes a peer in the implicit AV and verifies that the peer
 * is moved to the explicit AV when fi_av_insert is called
 *
 * @param[in]	state	struct efa_resource that is managed by the framework
 */
void test_av_implicit_to_explicit(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr, raw_addr_2;
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	fi_addr_t explicit_fi_addr, test_addr;
	struct efa_av *av;
	uint32_t ahn;
	int err;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	av = container_of(resource->av, struct efa_av, util_av.av_fid);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Generate a peer with random QPN and QKEY and insert it into the implicit AV */
	peer = test_av_get_peer_from_implicit_av(resource);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);

	/* Modify the peer and verify that the peer is moved as-is */
	peer->next_msg_id = 355;
	peer->flags |= EFA_RDM_PEER_IN_BACKOFF;

	/* Insert explicitly */
	raw_addr.qpn = peer->conn->ep_addr->qpn;
	raw_addr.qkey = peer->conn->ep_addr->qkey;
	err = fi_av_insert(resource->av, &raw_addr, 1, &explicit_fi_addr, 0, NULL);
	test_av_verify_av_hash_cnt(av, 1, 0, 0, 0);

	err = fi_av_lookup(resource->av, explicit_fi_addr, &raw_addr_2, &raw_addr_len);
	assert_int_equal(err, 0);
	assert_int_equal(efa_is_same_addr(&raw_addr, &raw_addr_2), 1);

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, explicit_fi_addr);
	assert_int_equal(peer->conn->fi_addr, explicit_fi_addr);
	assert_int_equal(peer->conn->implicit_fi_addr, FI_ADDR_NOTAVAIL);
	assert_int_equal(efa_is_same_addr(&raw_addr, peer->conn->ep_addr), 1);

	ahn = efa_rdm_ep->base_ep.self_ah->ahn;
	test_addr = efa_av_reverse_lookup_rdm(av, ahn, raw_addr.qpn, NULL);
	assert_int_equal(test_addr, explicit_fi_addr);

	/* Verify the manually set peer properties above */
	assert_int_equal(peer->next_msg_id, 355);
	assert_true(peer->flags & EFA_RDM_PEER_IN_BACKOFF);

	/* Unset the flag to make fi_av_remove easier */
	peer->flags &= ~EFA_RDM_PEER_IN_BACKOFF;

	err = fi_av_remove(resource->av, &explicit_fi_addr, 1, 0);
	assert_int_equal(err, 0);
	test_av_verify_av_hash_cnt(av, 0, 0, 0, 0);
}

static void test_av_implicit_av_verify_lru_list_first_last_elements(
	struct efa_av *av, struct efa_conn *first_conn_expected,
	struct efa_conn *last_conn_expected)
{
	struct dlist_entry *first_entry, *last_entry;
	struct efa_conn *first_conn_actual, *last_conn_actual;

	first_entry = av->implicit_av_lru_list.next;
	last_entry = av->implicit_av_lru_list.prev;

	first_conn_actual = container_of(first_entry, struct efa_conn,
					 implicit_av_lru_entry);
	last_conn_actual = container_of(last_entry, struct efa_conn,
					implicit_av_lru_entry);

	assert_ptr_equal(first_conn_actual, first_conn_expected);
	assert_ptr_equal(last_conn_actual, last_conn_expected);
}

/**
 * @brief This test inserts three implicit peers and verifies that the last
 * inserted and/or accessed peer is at the tail of the LRU list
 *
 * @param[in]	state	struct efa_resource that is managed by the framework
 */
void test_av_implicit_av_lru_insertion(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer0, *peer1, *peer2;
	struct efa_av *av;
	fi_addr_t implicit_fi_addr;
	uint32_t ahn;
	int err;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	av = container_of(resource->av, struct efa_av, util_av.av_fid);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Manually insert first address into implicit AV */
	peer0 = test_av_get_peer_from_implicit_av(resource);
	test_av_verify_av_hash_cnt(av, 0, 0, 1, 0);

	/* Expected LRU list: HEAD->peer0 */
	test_av_implicit_av_verify_lru_list_first_last_elements(av, peer0->conn, peer0->conn);

	/* Manually insert second address into implicit AV */
	peer1 = test_av_get_peer_from_implicit_av(resource);
	test_av_verify_av_hash_cnt(av, 0, 0, 2, 0);

	/* Expected LRU list: HEAD->peer0->peer1 */
	test_av_implicit_av_verify_lru_list_first_last_elements(av, peer0->conn, peer1->conn);

	/* Manually insert third address into implicit AV */
	peer2 = test_av_get_peer_from_implicit_av(resource);
	test_av_verify_av_hash_cnt(av, 0, 0, 3, 0);

	/* Expected LRU list: HEAD->peer0->peer1->peer2 */
	test_av_implicit_av_verify_lru_list_first_last_elements(av, peer0->conn, peer2->conn);


	/* Access peer0 through the CQ read path */
	ahn = efa_rdm_ep->base_ep.self_ah->ahn;
	ofi_genlock_lock(&efa_rdm_ep->base_ep.domain->srx_lock);
	implicit_fi_addr = efa_av_reverse_lookup_rdm_implicit(
		av, ahn, peer0->conn->ep_addr->qpn, NULL);
	ofi_genlock_unlock(&efa_rdm_ep->base_ep.domain->srx_lock);
	assert_int_equal(implicit_fi_addr, 0);

	/* Expected LRU list: HEAD->peer1->peer2->peer0 */
	test_av_implicit_av_verify_lru_list_first_last_elements(av, peer1->conn, peer0->conn);

	/* Access peer2 through the CQ read path */
	ahn = efa_rdm_ep->base_ep.self_ah->ahn;
	ofi_genlock_lock(&efa_rdm_ep->base_ep.domain->srx_lock);
	implicit_fi_addr = efa_av_reverse_lookup_rdm_implicit(
		av, ahn, peer2->conn->ep_addr->qpn, NULL);
	ofi_genlock_unlock(&efa_rdm_ep->base_ep.domain->srx_lock);
	assert_int_equal(implicit_fi_addr, 2);

	/* Expected LRU list: HEAD->peer1->peer0->peer2 */
	test_av_implicit_av_verify_lru_list_first_last_elements(av, peer1->conn, peer2->conn);


	/* Access peer1 through repeated AV insertion path */
	ofi_genlock_lock(&efa_rdm_ep->base_ep.domain->srx_lock);
	err = efa_av_insert_one(av, peer1->conn->ep_addr, &implicit_fi_addr, 0, NULL, true, true);
	ofi_genlock_unlock(&efa_rdm_ep->base_ep.domain->srx_lock);
	assert_int_equal(err, 0);
	assert_int_equal(implicit_fi_addr, 1);
	test_av_verify_av_hash_cnt(av, 0, 0, 3, 0);

	/* Expected LRU list: HEAD->peer0->peer2->peer1 */
	test_av_implicit_av_verify_lru_list_first_last_elements(av, peer0->conn, peer1->conn);

	/* Access peer2 through repeated AV insertion path */
	ofi_genlock_lock(&efa_rdm_ep->base_ep.domain->srx_lock);
	err = efa_av_insert_one(av, peer2->conn->ep_addr, &implicit_fi_addr, 0, NULL, true, true);
	ofi_genlock_unlock(&efa_rdm_ep->base_ep.domain->srx_lock);
	assert_int_equal(err, 0);
	assert_int_equal(implicit_fi_addr, 2);
	test_av_verify_av_hash_cnt(av, 0, 0, 3, 0);

	/* Expected LRU list: HEAD->peer0->peer1->peer2 */
	test_av_implicit_av_verify_lru_list_first_last_elements(av, peer0->conn, peer2->conn);
}

/**
 * @brief This test sets the implicit AV size to 2 and inserts four implicit
 * peers. It verifies that the least recently used peer is evicted.
 *
 * @param[in]	state	struct efa_resource that is managed by the framework
 */
void test_av_implicit_av_lru_eviction(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer0, *peer1, *peer2, *peer3;
	struct efa_ep_addr_hashable *efa_ep_addr_hashable;
	struct efa_av *av;
	fi_addr_t implicit_fi_addr;
	uint32_t ahn;
	int err;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	av = container_of(resource->av, struct efa_av, util_av.av_fid);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Modify implicit AV size */
	av->implicit_av_size = 2;

	/* Manually insert first address into implicit AV */
	peer0 = test_av_get_peer_from_implicit_av(resource);
	test_av_verify_av_hash_cnt(av, 0, 0, 1, 0);

	/* Expected LRU list: HEAD->peer0 */
	test_av_implicit_av_verify_lru_list_first_last_elements(av, peer0->conn, peer0->conn);

	/* Manually insert second address into implicit AV */
	peer1 = test_av_get_peer_from_implicit_av(resource);
	test_av_verify_av_hash_cnt(av, 0, 0, 2, 0);

	/* Expected LRU list: HEAD->peer0->peer1 */
	test_av_implicit_av_verify_lru_list_first_last_elements(av, peer0->conn, peer1->conn);

	/* Access peer0 through the CQ read path */
	ahn = efa_rdm_ep->base_ep.self_ah->ahn;
	ofi_genlock_lock(&efa_rdm_ep->base_ep.domain->srx_lock);
	implicit_fi_addr = efa_av_reverse_lookup_rdm_implicit(
		av, ahn, peer0->conn->ep_addr->qpn, NULL);
	ofi_genlock_unlock(&efa_rdm_ep->base_ep.domain->srx_lock);
	assert_int_equal(implicit_fi_addr, 0);

	/* Expected LRU list: HEAD->peer1->peer0 */
	test_av_implicit_av_verify_lru_list_first_last_elements(av, peer1->conn, peer0->conn);

	/* Manually insert third address into implicit AV */
	peer2 = test_av_get_peer_from_implicit_av(resource);
	test_av_verify_av_hash_cnt(av, 0, 0, 2, 0);

	/* Expected LRU list: HEAD->peer0->peer2 */
	test_av_implicit_av_verify_lru_list_first_last_elements(av, peer0->conn, peer2->conn);

	/* Verify that peer1 is evicted and added to the evicted hashmap */
	assert_int_equal(HASH_CNT(hh, av->evicted_peers_hashset), 1);
	HASH_FIND(hh, av->evicted_peers_hashset, peer1->conn->ep_addr,
		  sizeof(struct efa_ep_addr), efa_ep_addr_hashable);
	assert_non_null(efa_ep_addr_hashable);
	assert_int_equal(efa_is_same_addr(peer1->conn->ep_addr,
					  &efa_ep_addr_hashable->addr),
			 1);

	/* Access peer0 through repeated AV insertion path */
	ofi_genlock_lock(&efa_rdm_ep->base_ep.domain->srx_lock);
	err = efa_av_insert_one(av, peer0->conn->ep_addr, &implicit_fi_addr, 0, NULL, true, true);
	ofi_genlock_unlock(&efa_rdm_ep->base_ep.domain->srx_lock);
	assert_int_equal(err, 0);
	assert_int_equal(implicit_fi_addr, 0);
	test_av_verify_av_hash_cnt(av, 0, 0, 2, 0);

	/* Expected LRU list: HEAD->peer2->peer0 */
	test_av_implicit_av_verify_lru_list_first_last_elements(av, peer2->conn, peer0->conn);

	/* Manually insert fourth address into implicit AV */
	peer3 = test_av_get_peer_from_implicit_av(resource);
	test_av_verify_av_hash_cnt(av, 0, 0, 2, 0);

	/* Verify that peer2 is evicted and added to the evicted hashmap */
	assert_int_equal(HASH_CNT(hh, av->evicted_peers_hashset), 2);
	HASH_FIND(hh, av->evicted_peers_hashset, peer2->conn->ep_addr,
		  sizeof(struct efa_ep_addr), efa_ep_addr_hashable);
	assert_non_null(efa_ep_addr_hashable);
	assert_int_equal(efa_is_same_addr(peer2->conn->ep_addr,
					  &efa_ep_addr_hashable->addr),
			 1);

	/* Expected LRU list: HEAD->peer0->peer3 */
	test_av_implicit_av_verify_lru_list_first_last_elements(av, peer0->conn, peer3->conn);
}
