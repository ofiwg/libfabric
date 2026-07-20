/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"
#include "efa_cq.h"
#include "efa_cntr.h"
#include "rdm/efa_rdm_cntr.h"
#include "rdm/efa_rdm_atomic.h"


/**
 * @brief Verify the info type in struct efa_domain for efa direct path
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_info_type_efa_direct(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);
	assert_true(efa_domain->info_type == EFA_INFO_DIRECT);
}

/**
 * @brief Verify bounce buffer is allocated for efa-direct domain
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_direct_has_bounce_buffer(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);

	if (efa_domain->info->caps & FI_RMA) {
		assert_non_null(efa_domain->zero_byte_bounce_buf);
		assert_non_null(efa_domain->zero_byte_bounce_buf_mr);
		assert_non_null(efa_domain->zero_byte_bounce_buf_mr->ibv_mr);
	} else {
		assert_null(efa_domain->zero_byte_bounce_buf);
		assert_null(efa_domain->zero_byte_bounce_buf_mr);
	}
}


/* Internal structures from libibverbs/driver.h - needed to access MR access flags */
enum ibv_mr_type {
	IBV_MR_TYPE_MR,
	IBV_MR_TYPE_NULL_MR,
	IBV_MR_TYPE_IMPORTED_MR,
	IBV_MR_TYPE_DMABUF_MR,
};

struct verbs_mr {
	struct ibv_mr ibv_mr;
	enum ibv_mr_type mr_type;
	int access;
};

/**
 * @brief Verify bounce buffer is NOT allocated when FI_RMA is not requested
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_no_bounce_buffer_without_fi_rma_cap_requested(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	struct fi_info *hints, *info;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	hints->caps &= ~FI_RMA;
	assert_int_equal(fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, hints, &info), 0);

	/* Remove FI_RMA from the returned info as well */
	info->caps &= ~FI_RMA;

	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM, FI_VERSION(1, 14), info, true, true);

	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);
	assert_null(efa_domain->zero_byte_bounce_buf);
	assert_null(efa_domain->zero_byte_bounce_buf_mr);

	fi_freeinfo(hints);
	fi_freeinfo(info);
}

/**
 * @brief Verify bounce buffer registered successfully with RDMA read and write capability
 *        assuming platform support of rdma-read/rdma-write
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_bounce_buffer_with_rdma(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	struct verbs_mr *vmr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);

	if (!(efa_domain->info->caps & FI_RMA)) {
		assert_null(efa_domain->zero_byte_bounce_buf);
		assert_null(efa_domain->zero_byte_bounce_buf_mr);
		return;
	}

	assert_non_null(efa_domain->zero_byte_bounce_buf);
	assert_non_null(efa_domain->zero_byte_bounce_buf_mr);
	assert_non_null(efa_domain->zero_byte_bounce_buf_mr->ibv_mr);

	/* Verify MR access flags - with RDMA read and write */
	vmr = container_of(efa_domain->zero_byte_bounce_buf_mr->ibv_mr, struct verbs_mr, ibv_mr);

	/* IBV_ACCESS_LOCAL_WRITE is always set for FI_RECV */
	assert_true(vmr->access & IBV_ACCESS_LOCAL_WRITE);
	/* FI_SEND with RDMA_READ support sets REMOTE_READ for both efa/efa-direct */
	/* Remote Write never requested as this isn't a target buffer*/
	assert_false(vmr->access & IBV_ACCESS_REMOTE_WRITE);
}

/* test fi_open_ops with a wrong name */
void test_efa_domain_open_ops_wrong_name(void **state)
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

void test_efa_domain_open_ops_mr_query(void **state)
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

void test_efa_domain_open_ops_mr_query(void **state)
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

static struct fi_efa_ops_gda *efa_unit_test_construct_gda_ops(
	struct efa_resource *resource)
{
	struct fi_efa_ops_gda *efa_gda_ops;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	ret = fi_open_ops(&resource->domain->fid, FI_EFA_GDA_OPS, 0,
			  (void **)&efa_gda_ops, NULL);
	assert_int_equal(ret, 0);

	return efa_gda_ops;
}


void test_efa_domain_open_ops_query_qp_wqs(void **state)
{
    struct efa_resource *resource = *state;
    int ret;
    struct fi_efa_ops_gda *efa_gda_ops;
    struct fi_efa_wq_attr sq_attr = {0};
    struct fi_efa_wq_attr rq_attr = {0};

    efa_gda_ops = efa_unit_test_construct_gda_ops(resource);

#if HAVE_EFADV_QUERY_QP_WQS
    g_efa_unit_test_mocks.efadv_query_qp_wqs = &efa_mock_efadv_query_qp_wqs;
#endif
    ret = efa_gda_ops->query_qp_wqs(resource->ep, &sq_attr, &rq_attr);

#if HAVE_EFADV_QUERY_QP_WQS
    assert_int_equal(ret, FI_SUCCESS);

    assert_non_null(sq_attr.buffer);
    assert_non_null(sq_attr.doorbell);
    assert_true(sq_attr.entry_size > 0);
    assert_true(sq_attr.num_entries > 0);
    assert_true(sq_attr.max_batch > 0);

    assert_non_null(rq_attr.buffer);
    assert_non_null(rq_attr.doorbell);
    assert_true(rq_attr.entry_size > 0);
    assert_true(rq_attr.num_entries > 0);
    assert_true(rq_attr.max_batch > 0);
#else
    assert_int_equal(ret, -FI_ENOSYS);
#endif /* HAVE_EFADV_QUERY_QP_WQS */
}


void test_efa_domain_open_ops_query_cq(void **state)
{
    struct efa_resource *resource = *state;
    int ret;
    struct fi_efa_ops_gda *efa_gda_ops;
    struct fi_efa_cq_attr cq_attr = {0};

    efa_gda_ops = efa_unit_test_construct_gda_ops(resource);

#if HAVE_EFADV_QUERY_CQ
    g_efa_unit_test_mocks.efadv_query_cq = &efa_mock_efadv_query_cq;
#endif
    ret = efa_gda_ops->query_cq(resource->cq, &cq_attr);

#if HAVE_EFADV_QUERY_CQ
    assert_int_equal(ret, FI_SUCCESS);
    assert_non_null(cq_attr.buffer);
    assert_true(cq_attr.entry_size > 0);
    assert_true(cq_attr.num_entries > 0);
#else
    assert_int_equal(ret, -FI_ENOSYS);
#endif /* HAVE_EFADV_QUERY_CQ */
}



/**
 * @brief Verify FI_MR_ALLOCATED is set for efa dgram path
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_dgram_attr_mr_allocated(void **state)
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
void test_efa_domain_direct_attr_mr_allocated(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);
	assert_true(efa_domain->device->rdm_info->domain_attr->mr_mode & FI_MR_ALLOCATED);
}


void test_efa_domain_open_ops_query_addr(void **state)
{
	struct efa_resource *resource = *state;
	int ret;
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	struct efa_ep_addr raw_addr;
	fi_addr_t addr;
	struct fi_efa_ops_gda *efa_gda_ops;
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

	ret = fi_open_ops(&resource->domain->fid, FI_EFA_GDA_OPS, 0,
			  (void **) &efa_gda_ops, NULL);
	assert_int_equal(ret, 0);

	ret = efa_gda_ops->query_addr(resource->ep, addr, &ahn,
					    &remote_qpn, &remote_qkey);

	assert_int_equal(ret, FI_SUCCESS);
	assert_int_equal(remote_qpn, 1);
	assert_int_equal(remote_qkey, 0x1234);
}

void test_efa_domain_open_ops_cq_open_ext(void **state)
{
    struct efa_resource *resource = *state;
    struct fi_efa_ops_gda *efa_gda_ops;
    struct fi_cq_attr attr = {0};
    struct fi_efa_cq_init_attr efa_cq_init_attr = {
	    .flags = FI_EFA_CQ_INIT_FLAGS_EXT_MEM_DMABUF,
	    .ext_mem_dmabuf =
		    {
			    .buffer = NULL,
			    .length = 64,
			    .offset = 0,
			    .fd = 1,
		    },
    };
    struct fid_cq *cq_fid;
    int ret;

    efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

    ret = fi_open_ops(&resource->domain->fid, FI_EFA_GDA_OPS, 0,
		      (void **) &efa_gda_ops, NULL);
    assert_int_equal(ret, 0);

#if HAVE_CAPS_CQ_WITH_EXT_MEM_DMABUF && HAVE_EFADV_CQ_EX
    if (efa_device_support_cq_with_ext_mem_dmabuf()) {
        g_efa_unit_test_mocks.efadv_create_cq = &efa_mock_efadv_create_cq_with_ibv_create_cq_ex;
        expect_function_call(efa_mock_efadv_create_cq_with_ibv_create_cq_ex);
    }
    ret = efa_gda_ops->cq_open_ext(resource->domain, &attr,
				      &efa_cq_init_attr, &cq_fid, NULL);
#else
    ret = efa_gda_ops->cq_open_ext(resource->domain, &attr,
				      &efa_cq_init_attr, &cq_fid, NULL);
#endif

#if HAVE_CAPS_CQ_WITH_EXT_MEM_DMABUF && HAVE_EFADV_CQ_EX
    if (!efa_device_support_cq_with_ext_mem_dmabuf()) {
	    assert_int_equal(ret, -FI_EOPNOTSUPP);
	    return;
    }
    assert_int_equal(ret, FI_SUCCESS);
    struct efa_cq *efa_cq = container_of(cq_fid, struct efa_cq, util_cq.cq_fid);
    assert_non_null(efa_cq->ibv_cq.ibv_cq_ex);
    assert_int_equal(efa_cq->ibv_cq.ibv_cq_ex_type, EFADV_CQ);
    if (cq_fid)
	    fi_close(&cq_fid->fid);
#else
    assert_int_equal(ret, -FI_ENOSYS);
#endif
}


/**
 * @brief Verify that EFA direct domains use the correct MR operations
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_direct_mr_ops(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);

	/* EFA-direct domains should use efa_domain_mr_ops */
	assert_ptr_equal(efa_domain->util_domain.domain_fid.mr, &efa_domain_mr_ops);
	assert_int_equal(efa_domain->info_type, EFA_INFO_DIRECT);
}

/**
 * @brief Verify that DGRAM domains use the correct MR operations
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_dgram_mr_ops(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_DGRAM, EFA_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);

	/* DGRAM domains should use efa_domain_mr_ops (core EFA MR operations) */
	assert_ptr_equal(efa_domain->util_domain.domain_fid.mr, &efa_domain_mr_ops);
	assert_int_equal(efa_domain->info_type, EFA_INFO_DGRAM);
}




void test_efa_domain_open_ops_get_mr_lkey(void **state)
{
    struct efa_resource *resource = *state;
    struct fi_efa_ops_gda *efa_gda_ops;
    struct efa_mr mr = {0};
    struct fid_mr mr_fid = {0};
    struct ibv_mr ibv_mr = {0};
    uint64_t lkey;

    mr.mr_fid = mr_fid;
    ibv_mr.lkey = 1234567;
    mr.ibv_mr = &ibv_mr;

    efa_gda_ops = efa_unit_test_construct_gda_ops(resource);

    lkey = efa_gda_ops->get_mr_lkey(&mr.mr_fid);
    assert_true(lkey == mr.ibv_mr->lkey);
}

/**
 * @brief Test cntr_open_ext returns -FI_ENOSYS when efadv_create_comp_cntr is unavailable,
 * or succeeds with mocked efadv_create_comp_cntr.
 */
void test_efa_domain_open_ops_cntr_open_ext(void **state)
{
	struct efa_resource *resource = *state;
	struct fi_efa_ops_gda *efa_gda_ops;
	struct fi_cntr_attr attr = {0};
	struct fi_efa_comp_cntr_init_attr efa_attr = {0};
	struct fid_cntr *cntr_fid = NULL;
	int ret;

	efa_env.use_hw_cntr = 1;
	efa_gda_ops = efa_unit_test_construct_gda_ops(resource);

#if HAVE_EFADV_CREATE_COMP_CNTR
	{
	struct efa_domain *efa_domain;
	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);
	efa_unit_test_set_hw_cntr_max_values(efa_domain);
	g_efa_unit_test_mocks.efadv_create_comp_cntr = efa_mock_efadv_create_comp_cntr_return_mock;
	g_efa_unit_test_mocks.ibv_destroy_comp_cntr = efa_mock_ibv_destroy_comp_cntr_return_mock;
	}
#endif

	attr.events = FI_CNTR_EVENTS_COMP;
	ret = efa_gda_ops->cntr_open_ext(resource->domain, &attr, &cntr_fid,
					  NULL, &efa_attr);
#if HAVE_EFADV_CREATE_COMP_CNTR
	assert_int_equal(ret, FI_SUCCESS);
	assert_non_null(cntr_fid);
	fi_close(&cntr_fid->fid);
#else
	assert_int_equal(ret, -FI_ENOSYS);
#endif
}

/**
 * @brief Verify EFA-direct domains install base domain ops
 *
 * @param[in]	state	struct efa_resource managed by the framework
 */
void test_efa_domain_open_installs_base_domain_ops_efa_direct(void **state)
{
	struct efa_resource *resource = *state;
	struct fi_ops_domain *ops;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	ops = resource->domain->ops;

	assert_ptr_equal(ops->av_open, efa_av_open);
	assert_ptr_equal(ops->cq_open, efa_cq_open);
	assert_ptr_equal(ops->endpoint, efa_ep_open);
	assert_ptr_equal(ops->cntr_open, efa_cntr_open);
	assert_ptr_equal(ops->query_atomic, fi_no_query_atomic);
}

/**
 * @brief Verify DGRAM domains install base domain ops
 *
 * DGRAM is served by the "efa" fabric whose .domain slot points to
 * efa_rdm_domain_open. This test exercises the DGRAM redirect inside
 * efa_rdm_domain_open which forwards to efa_domain_open and ends up
 * with the base ops table.
 *
 * @param[in]	state	struct efa_resource managed by the framework
 */
void test_efa_domain_open_installs_base_domain_ops_dgram(void **state)
{
	struct efa_resource *resource = *state;
	struct fi_ops_domain *ops;

	efa_unit_test_resource_construct(resource, FI_EP_DGRAM, EFA_FABRIC_NAME);
	ops = resource->domain->ops;

	assert_ptr_equal(ops->av_open, efa_av_open);
	assert_ptr_equal(ops->cq_open, efa_cq_open);
	assert_ptr_equal(ops->endpoint, efa_ep_open);
	assert_ptr_equal(ops->cntr_open, efa_cntr_open);
	assert_ptr_equal(ops->query_atomic, fi_no_query_atomic);
}

/**
 * @brief Verify FI_EFA_GDA_OPS is rejected on the DGRAM path
 *
 * @param[in]	state	struct efa_resource managed by the framework
 */
void test_efa_domain_gda_ops_rejected_for_dgram(void **state)
{
	struct efa_resource *resource = *state;
	struct fi_efa_ops_gda *efa_gda_ops;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_DGRAM, EFA_FABRIC_NAME);

	ret = fi_open_ops(&resource->domain->fid, FI_EFA_GDA_OPS, 0,
			  (void **)&efa_gda_ops, NULL);
	assert_int_equal(ret, -FI_EOPNOTSUPP);
}
