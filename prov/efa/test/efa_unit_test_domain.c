/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"

/**
 * @brief Verify the info type in struct efa_domain for efa RDM path
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_info_type_efa_rdm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);
	assert_true(efa_domain->info_type == EFA_INFO_RDM);
}

/**
 * @brief Verify the info type in struct efa_domain for efa direct path
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_info_type_efa_direct(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);
	assert_true(efa_domain->info_type == EFA_INFO_DIRECT);
}

/* test fi_open_ops with a wrong name */
void test_efa_domain_open_ops_wrong_name(struct efa_resource **state)
{
    struct efa_resource *resource = *state;
    int ret;
    struct fi_efa_ops_domain *efa_domain_ops;

    efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

    ret = fi_open_ops(&resource->domain->fid, "arbitrary name", 0, (void **)&efa_domain_ops, NULL);
    assert_int_equal(ret, -FI_EINVAL);
}

static
void test_efa_domain_open_ops_mr_query_common(
                            struct efa_resource *resource,
                            int expected_ret,
                            uint16_t expected_ic_id_validity,
                            uint16_t expected_recv_ic_id,
                            uint16_t expected_rdma_read_ic_id,
                            uint16_t expected_rdma_recv_ic_id)
{
    int ret;
    struct fi_efa_ops_domain *efa_domain_ops;
    struct fi_efa_mr_attr efa_mr_attr = {0};
    struct efa_mr mr = {0};
    struct fid_mr mr_fid = {0};

    mr.mr_fid = mr_fid;
    mr.ibv_mr = NULL;

    ret = fi_open_ops(&resource->domain->fid, FI_EFA_DOMAIN_OPS, 0, (void **)&efa_domain_ops, NULL);
    assert_int_equal(ret, 0);

    ret = efa_domain_ops->query_mr(&mr.mr_fid, &efa_mr_attr);
    assert_int_equal(ret, expected_ret);

    if (expected_ret == -FI_ENOSYS)
        return;

    assert_true(efa_mr_attr.ic_id_validity == expected_ic_id_validity);

    if (efa_mr_attr.ic_id_validity & FI_EFA_MR_ATTR_RECV_IC_ID)
        assert_true(efa_mr_attr.recv_ic_id == expected_recv_ic_id);

    if (efa_mr_attr.ic_id_validity & FI_EFA_MR_ATTR_RDMA_READ_IC_ID)
        assert_true(efa_mr_attr.rdma_read_ic_id == expected_rdma_read_ic_id);

    if (efa_mr_attr.ic_id_validity & FI_EFA_MR_ATTR_RDMA_RECV_IC_ID)
        assert_true(efa_mr_attr.rdma_recv_ic_id == expected_rdma_recv_ic_id);
}

#if HAVE_EFADV_QUERY_MR

void test_efa_domain_open_ops_mr_query(struct efa_resource **state)
{
    struct efa_resource *resource = *state;

    efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

    /* set recv_ic_id as 0 */
    g_efa_unit_test_mocks.efadv_query_mr = &efa_mock_efadv_query_mr_recv_ic_id_0;

    test_efa_domain_open_ops_mr_query_common(
                                resource,
                                0,
                                FI_EFA_MR_ATTR_RECV_IC_ID,
                                0,
                                0 /* ignored */,
                                0 /* ignored */);

    /* set rdma_read_ic_id as 1 */
    g_efa_unit_test_mocks.efadv_query_mr = &efa_mock_efadv_query_mr_rdma_read_ic_id_1;

    test_efa_domain_open_ops_mr_query_common(
                                resource,
                                0,
                                FI_EFA_MR_ATTR_RDMA_READ_IC_ID,
                                0 /* ignored */,
                                1,
                                0 /* ignored */);

    /* set rdma_recv_ic_id as 2 */
    g_efa_unit_test_mocks.efadv_query_mr = &efa_mock_efadv_query_mr_rdma_recv_ic_id_2;

    test_efa_domain_open_ops_mr_query_common(
                                resource,
                                0,
                                FI_EFA_MR_ATTR_RDMA_RECV_IC_ID,
                                0 /* ignored */,
                                0 /* ignored */,
                                2);

    /* set recv_ic_id as 0, rdma_read_ic_id as 1 */
    g_efa_unit_test_mocks.efadv_query_mr = &efa_mock_efadv_query_mr_recv_and_rdma_read_ic_id_0_1;

    test_efa_domain_open_ops_mr_query_common(
                                resource,
                                0,
                                FI_EFA_MR_ATTR_RECV_IC_ID | FI_EFA_MR_ATTR_RDMA_READ_IC_ID,
                                0,
                                1,
                                0 /* ignored */);
}

#else

void test_efa_domain_open_ops_mr_query(struct efa_resource **state)
{
    struct efa_resource *resource = *state;

    efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

    test_efa_domain_open_ops_mr_query_common(
                                resource,
                                -FI_ENOSYS,
                                0, /* ignored */
                                0, /* ignored */
                                1, /* ignored */
                                0  /* ignored */);
}

#endif /* HAVE_EFADV_QUERY_MR */

/**
 * @brief Verify FI_MR_ALLOCATED is set for efa rdm path
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_rdm_attr_mr_allocated(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);
	assert_true(efa_domain->device->rdm_info->domain_attr->mr_mode & FI_MR_ALLOCATED);
}

/**
 * @brief Verify FI_MR_ALLOCATED is set for efa dgram path
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_dgram_attr_mr_allocated(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_DGRAM, EFA_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);
	assert_true(efa_domain->device->dgram_info->domain_attr->mr_mode & FI_MR_ALLOCATED);
}

/**
 * @brief Verify FI_MR_ALLOCATED is set for efa direct path
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_direct_attr_mr_allocated(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);
	assert_true(efa_domain->device->rdm_info->domain_attr->mr_mode & FI_MR_ALLOCATED);
}

/**
 * @brief Verify that the domain level peer lists get cleared when an endpoint is closed
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_peer_list_cleared(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	struct fid_ep *ep1, *ep2;
	struct efa_rdm_ep *efa_rdm_ep1, *efa_rdm_ep2;
	struct efa_rdm_peer peer1 = {0}, peer2 = {0}, peer3 = {0}, peer4 = {0},
			    *peer;
	int i = 0, err;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	// Create two endpoints
	err = fi_endpoint(resource->domain, resource->info, &ep1, NULL);
	assert_int_equal(err, 0);
	err = fi_endpoint(resource->domain, resource->info, &ep2, NULL);
	assert_int_equal(err, 0);

	efa_rdm_ep1 =
		container_of(ep1, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_ep2 =
		container_of(ep2, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	// Associate the peers with the endpoints
	peer1.ep = efa_rdm_ep1;
	peer2.ep = efa_rdm_ep1;
	peer3.ep = efa_rdm_ep2;
	peer4.ep = efa_rdm_ep2;

	dlist_insert_tail(&peer1.handshake_queued_entry,
			  &efa_domain->handshake_queued_peer_list);
	dlist_insert_tail(&peer2.rnr_backoff_entry,
			  &efa_domain->peer_backoff_list);
	dlist_insert_tail(&peer3.handshake_queued_entry,
			  &efa_domain->handshake_queued_peer_list);
	dlist_insert_tail(&peer4.rnr_backoff_entry,
			  &efa_domain->peer_backoff_list);

	fi_close(&ep1->fid);
	dlist_foreach_container (&efa_domain->peer_backoff_list,
				 struct efa_rdm_peer, peer, rnr_backoff_entry) {
		assert_true(peer->ep == efa_rdm_ep2);
		i++;
	}
	assert_int_equal(i, 1);

	dlist_foreach_container (&efa_domain->handshake_queued_peer_list,
				 struct efa_rdm_peer, peer,
				 handshake_queued_entry) {
		assert_true(peer->ep == efa_rdm_ep2);
		i++;
	}
	assert_int_equal(i, 2);

	fi_close(&ep2->fid);
	assert_true(dlist_empty(&efa_domain->peer_backoff_list));
	assert_true(dlist_empty(&efa_domain->handshake_queued_peer_list));
}

void test_efa_domain_open_ops_query_addr(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	int ret;
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	struct efa_ep_addr raw_addr;
	fi_addr_t addr;
	struct fi_efa_ops_domain *efa_domain_ops;
	uint16_t ahn;
	uint16_t remote_qpn;
	uint32_t remote_qkey;

	efa_unit_test_resource_construct(resource, FI_EP_RDM,
					 EFA_DIRECT_FABRIC_NAME);
	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL);
	assert_int_equal(ret, 1);

	ret = fi_open_ops(&resource->domain->fid, FI_EFA_DOMAIN_OPS, 0,
			  (void **) &efa_domain_ops, NULL);
	assert_int_equal(ret, 0);

	ret = efa_domain_ops->query_addr(resource->ep, addr, &ahn,
					    &remote_qpn, &remote_qkey);

	assert_int_equal(ret, FI_SUCCESS);
	assert_int_equal(remote_qpn, 1);
	assert_int_equal(remote_qkey, 0x1234);
}
