/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
#include "efa_unit_tests.h"
#include "efa_cntr.h"
#include "efa_hw_cntr.h"
#include "rdm/efa_rdm_cntr.h"

/**
 * @brief get the length of the ibv_cq_poll_list for a given efa_cntr (efa-direct)
 *
 * @param cntr_fid cntr fid
 * @return int the length of the ibv_cq_poll_list
 */
static
int test_efa_cntr_get_ibv_cq_poll_list_length(struct fid_cntr *cntr_fid)
{
	int i = 0;
	struct dlist_entry *item;
	struct efa_cntr *cntr;

	cntr = container_of(cntr_fid, struct efa_cntr, util_cntr.cntr_fid.fid);
	dlist_foreach(&cntr->ibv_cq_poll_list, item) {
		i++;
	}

	return i;
}



/**
 * @brief Check the length of ibv_cq_poll_list in cntr when 1 cq is bind to 1 ep
 * as both tx/rx cq. (efa-direct)
 *
 * @param state struct efa_resource that is managed by the framework
 */
void test_efa_cntr_ibv_cq_poll_list_same_tx_rx_cq_single_ep(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_cntr *cntr;
	struct fi_cntr_attr cntr_attr = {0};

	efa_unit_test_resource_construct_ep_not_enabled(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	assert_int_equal(fi_cntr_open(resource->domain, &cntr_attr, &cntr, NULL), 0);

	/* TODO: expand this test to all flags */
	assert_int_equal(fi_ep_bind(resource->ep, &cntr->fid, FI_TRANSMIT), 0);

	assert_int_equal(fi_enable(resource->ep), 0);

	/* efa_unit_test_resource_construct binds single OFI CQ as both tx/rx cq of ep */
	assert_int_equal(test_efa_cntr_get_ibv_cq_poll_list_length(cntr), 1);

	/* ep must be closed before cq/av/eq... */
	fi_close(&resource->ep->fid);
	resource->ep = NULL;

	fi_close(&cntr->fid);
}

void test_efa_rdm_cntr_ibv_cq_poll_list_same_tx_rx_cq_single_ep(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_cntr *cntr;
	struct fi_cntr_attr cntr_attr = {0};

	efa_unit_test_resource_construct_ep_not_enabled(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	assert_int_equal(fi_cntr_open(resource->domain, &cntr_attr, &cntr, NULL), 0);

	/* TODO: expand this test to all flags */
	assert_int_equal(fi_ep_bind(resource->ep, &cntr->fid, FI_TRANSMIT), 0);

	assert_int_equal(fi_enable(resource->ep), 0);

	/* efa_unit_test_resource_construct binds single OFI CQ as both tx/rx cq of ep */
	assert_int_equal(efa_unit_test_get_dlist_length(
		&container_of(cntr, struct efa_rdm_cntr, efa_cntr.util_cntr.cntr_fid)->efa_cntr.ibv_cq_poll_list), 1);

	/* ep must be closed before cq/av/eq... */
	fi_close(&resource->ep->fid);
	resource->ep = NULL;

	fi_close(&cntr->fid);
}

/**
 * @brief Check the length of ibv_cq_poll_list in cntr when separate tx/rx cq is bind to 1 ep.
 * (efa-direct)
 *
 * @param state struct efa_resource that is managed by the framework
 */
void test_efa_cntr_ibv_cq_poll_list_separate_tx_rx_cq_single_ep(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_cq *txcq, *rxcq;
	struct fi_cq_attr cq_attr = {0};
	struct fid_cntr *cntr;
	struct fi_cntr_attr cntr_attr = {0};

	efa_unit_test_resource_construct_no_cq_and_ep_not_enabled(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	assert_int_equal(fi_cq_open(resource->domain, &cq_attr, &txcq, NULL), 0);

	assert_int_equal(fi_ep_bind(resource->ep, &txcq->fid, FI_SEND), 0);

	assert_int_equal(fi_cq_open(resource->domain, &cq_attr, &rxcq, NULL), 0);

	assert_int_equal(fi_ep_bind(resource->ep, &rxcq->fid, FI_RECV), 0);

	assert_int_equal(fi_cntr_open(resource->domain, &cntr_attr, &cntr, NULL), 0);

	/* TODO: expand this test to all flags */
	assert_int_equal(fi_ep_bind(resource->ep, &cntr->fid, FI_TRANSMIT), 0);

	assert_int_equal(fi_enable(resource->ep), 0);

	assert_int_equal(test_efa_cntr_get_ibv_cq_poll_list_length(cntr), 2);

	/* ep must be closed before cq/av/eq... */
	fi_close(&resource->ep->fid);
	resource->ep = NULL;
	fi_close(&txcq->fid);
	fi_close(&rxcq->fid);
	fi_close(&cntr->fid);
}

void test_efa_rdm_cntr_ibv_cq_poll_list_separate_tx_rx_cq_single_ep(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_cq *txcq, *rxcq;
	struct fi_cq_attr cq_attr = {0};
	struct fid_cntr *cntr;
	struct fi_cntr_attr cntr_attr = {0};

	efa_unit_test_resource_construct_no_cq_and_ep_not_enabled(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	assert_int_equal(fi_cq_open(resource->domain, &cq_attr, &txcq, NULL), 0);

	assert_int_equal(fi_ep_bind(resource->ep, &txcq->fid, FI_SEND), 0);

	assert_int_equal(fi_cq_open(resource->domain, &cq_attr, &rxcq, NULL), 0);

	assert_int_equal(fi_ep_bind(resource->ep, &rxcq->fid, FI_RECV), 0);

	assert_int_equal(fi_cntr_open(resource->domain, &cntr_attr, &cntr, NULL), 0);

	/* TODO: expand this test to all flags */
	assert_int_equal(fi_ep_bind(resource->ep, &cntr->fid, FI_TRANSMIT), 0);

	assert_int_equal(fi_enable(resource->ep), 0);

	assert_int_equal(efa_unit_test_get_dlist_length(
		&container_of(cntr, struct efa_rdm_cntr, efa_cntr.util_cntr.cntr_fid)->efa_cntr.ibv_cq_poll_list), 2);

	/* ep must be closed before cq/av/eq... */
	fi_close(&resource->ep->fid);
	resource->ep = NULL;
	fi_close(&txcq->fid);
	fi_close(&rxcq->fid);
	fi_close(&cntr->fid);
}

void test_efa_rdm_cntr_post_initial_rx_pkts(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct fid_cntr *cntr;
	struct fi_cntr_attr cntr_attr = {0};
	struct efa_rdm_cntr *efa_rdm_cntr;
	uint64_t cnt;

	efa_unit_test_resource_construct_ep_not_enabled(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* At this time, rx pkts are not growed and posted */
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_to_post, 0);
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_posted, 0);
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_held, 0);

	assert_int_equal(fi_cntr_open(resource->domain, &cntr_attr, &cntr, NULL), 0);

	/* TODO: expand this test to all flags */
	assert_int_equal(fi_ep_bind(resource->ep, &cntr->fid, FI_TRANSMIT), 0);

	assert_int_equal(fi_enable(resource->ep), 0);

	efa_rdm_cntr = container_of(cntr, struct efa_rdm_cntr, efa_cntr.util_cntr.cntr_fid);

	/* cntr read need to scan the ep list since a ep is bind */
	assert_true(efa_rdm_cntr->need_to_scan_ep_list);

	cnt = fi_cntr_read(cntr);
	/* No completion should be read */
	assert_int_equal(cnt, 0);

	/* At this time, rx pool size number of rx pkts are posted */
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_posted, efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep));
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_to_post, 0);
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_held, 0);

	/* scan is done */
	assert_false(efa_rdm_cntr->need_to_scan_ep_list);
	/* ep must be closed before cq/av/eq... */
	fi_close(&resource->ep->fid);
	resource->ep = NULL;

	fi_close(&cntr->fid);
}

void test_efa_rdm_cntr_read_before_ep_enable(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_ep *ep;
	struct fid_cntr *cntr;
	struct fi_cntr_attr cntr_attr = {0};
	uint64_t cnt;

	/* TODO: allow shm when shm fixed its bug that
	 cq read cannot be called before ep enable */

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);

	assert_int_equal(fi_endpoint(resource->domain, resource->info, &ep, NULL), 0);

	assert_int_equal(fi_cntr_open(resource->domain, &cntr_attr, &cntr, NULL), 0);

	assert_int_equal(fi_ep_bind(resource->ep, &cntr->fid, FI_TRANSMIT), 0);

	cnt = fi_cntr_read(cntr);
	/* No completion should be read */
	assert_int_equal(cnt, 0);

	/* eps must be closed before cq/av/eq... */
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;

	assert_int_equal(fi_close(&ep->fid), 0);

	assert_int_equal(fi_close(&cntr->fid), 0);
}


#if HAVE_EFADV_CREATE_COMP_CNTR
/**
 * @brief Test efa_hw_cntr_open returns -FI_EOPNOTSUPP for FI_CNTR_EVENTS_BYTES
 * when efadv_create_comp_cntr does not support IBV_COMP_CNTR_TYPE_BYTES.
 */
void test_efa_hw_cntr_open_unsupported_type_bytes(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fi_cntr_attr attr = {0};
	struct efadv_comp_cntr_init_attr cc_attr = {0};
	struct efa_cntr *cntr;
	struct fid_cntr *cntr_fid = NULL;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);
	efa_domain->device->max_comp_cntr = (1ULL << 31) - 1;
	efa_domain->info->domain_attr->max_cntr_value = (1ULL << 31) - 1;
	efa_domain->info->domain_attr->max_err_cntr_value = (1ULL << 31) - 1;
	g_efa_unit_test_mocks.efadv_create_comp_cntr =
		efa_mock_efadv_create_comp_cntr_return_null_enotsup;

	cntr = calloc(1, sizeof(*cntr));
	assert_non_null(cntr);

	attr.events = FI_CNTR_EVENTS_BYTES;
	assert_int_equal(efa_hw_cntr_open(resource->domain, &attr, cntr, &cntr_fid, NULL, &cc_attr),
			 -ENOTSUP);
	assert_null(cntr_fid);
	free(cntr);
}

/**
 * @brief Test efa_hw_cntr_open rejects when max_cntr_value exceeds hw limit
 */
void test_efa_hw_cntr_open_max_cntr_value_exceeded(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fi_cntr_attr attr = {0};
	struct efadv_comp_cntr_init_attr cc_attr = {0};
	struct efa_cntr *cntr;
	struct fid_cntr *cntr_fid = NULL;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);
	efa_domain->device->max_comp_cntr = (1ULL << 31) - 1;
	efa_domain->info->domain_attr->max_cntr_value = UINT64_MAX;

	cntr = calloc(1, sizeof(*cntr));
	assert_non_null(cntr);

	attr.events = FI_CNTR_EVENTS_COMP;
	assert_int_equal(efa_hw_cntr_open(resource->domain, &attr, cntr, &cntr_fid, NULL, &cc_attr),
			 -FI_EOPNOTSUPP);
	free(cntr);
}

/**
 * @brief Test efa_hw_cntr_open returns error when efadv_create_comp_cntr fails
 */
void test_efa_hw_cntr_open_ibv_fail(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fi_cntr_attr attr = {0};
	struct efa_cntr *cntr;
	struct fid_cntr *cntr_fid = NULL;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);
	efa_domain->device->max_comp_cntr = (1ULL << 31) - 1;
	efa_domain->info->domain_attr->max_cntr_value = (1ULL << 31) - 1;
	efa_domain->info->domain_attr->max_err_cntr_value = (1ULL << 31) - 1;
	g_efa_unit_test_mocks.efadv_create_comp_cntr =
		efa_mock_efadv_create_comp_cntr_return_null_enotsup;

	cntr = calloc(1, sizeof(*cntr));
	assert_non_null(cntr);

	attr.events = FI_CNTR_EVENTS_COMP;
	assert_int_equal(efa_hw_cntr_open(resource->domain, &attr, cntr, &cntr_fid, NULL, NULL),
			 -ENOTSUP);
	assert_null(cntr_fid);
	free(cntr);
}

/**
 * @brief Helper to open a hardware counter
 *
 * @param[in]  resource     Test resource
 * @param[in]  enable_ep    If true, construct resource with EP enabled;
 *                          if false, leave EP not enabled.
 * @return opened counter fid
 */
static struct fid_cntr *test_efa_hw_cntr_open(struct efa_resource *resource,
					      bool enable_ep)
{
	struct fi_cntr_attr attr = {0};
	struct efadv_comp_cntr_init_attr cc_attr = {0};
	struct efa_domain *efa_domain;
	struct efa_cntr *efa_cntr;
	struct fid_cntr *cntr_fid = NULL;
	int ret;

	if (enable_ep)
		efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	else
		efa_unit_test_resource_construct_ep_not_enabled(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);
	efa_domain->device->max_comp_cntr = (1ULL << 31) - 1;
	efa_domain->info->domain_attr->max_cntr_value = (1ULL << 31) - 1;
	efa_domain->info->domain_attr->max_err_cntr_value = (1ULL << 31) - 1;
	g_efa_unit_test_mocks.efadv_create_comp_cntr = efa_mock_efadv_create_comp_cntr_return_mock;
	g_efa_unit_test_mocks.ibv_destroy_comp_cntr = efa_mock_ibv_destroy_comp_cntr_return_mock;

	efa_cntr = calloc(1, sizeof(*efa_cntr));
	assert_non_null(efa_cntr);

	attr.events = FI_CNTR_EVENTS_COMP;
	ret = efa_hw_cntr_open(resource->domain, &attr, efa_cntr, &cntr_fid, NULL, &cc_attr);
	assert_int_equal(ret, FI_SUCCESS);
	assert_non_null(cntr_fid);
	assert_non_null(efa_cntr->ibv_comp_cntr);

	return cntr_fid;
}

/**
 * @brief Test efa_hw_cntr_add calls ibv_inc_comp_cntr and returns success
 */
void test_efa_hw_cntr_add(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_cntr *cntr_fid;

	cntr_fid = test_efa_hw_cntr_open(resource, true);
	g_efa_unit_test_mocks.ibv_inc_comp_cntr = efa_mock_ibv_inc_comp_cntr_return_mock;

	assert_int_equal(fi_cntr_add(cntr_fid, 1), FI_SUCCESS);

	fi_close(&cntr_fid->fid);
}

/**
 * @brief Test efa_hw_cntr_adderr calls ibv_inc_err_comp_cntr and returns success
 */
void test_efa_hw_cntr_adderr(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_cntr *cntr_fid;

	cntr_fid = test_efa_hw_cntr_open(resource, true);
	g_efa_unit_test_mocks.ibv_inc_err_comp_cntr = efa_mock_ibv_inc_err_comp_cntr_return_mock;

	assert_int_equal(fi_cntr_adderr(cntr_fid, 1), FI_SUCCESS);

	fi_close(&cntr_fid->fid);
}

/**
 * @brief Test efa_hw_cntr_set calls ibv_set_comp_cntr and returns success
 */
void test_efa_hw_cntr_set(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_cntr *cntr_fid;

	cntr_fid = test_efa_hw_cntr_open(resource, true);
	g_efa_unit_test_mocks.ibv_set_comp_cntr = efa_mock_ibv_set_comp_cntr_return_mock;

	assert_int_equal(fi_cntr_set(cntr_fid, 1), FI_SUCCESS);

	fi_close(&cntr_fid->fid);
}

/**
 * @brief Test efa_hw_cntr_seterr calls ibv_set_err_comp_cntr and returns success
 */
void test_efa_hw_cntr_seterr(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_cntr *cntr_fid;

	cntr_fid = test_efa_hw_cntr_open(resource, true);
	g_efa_unit_test_mocks.ibv_set_err_comp_cntr = efa_mock_ibv_set_err_comp_cntr_return_mock;

	assert_int_equal(fi_cntr_seterr(cntr_fid, 1), FI_SUCCESS);

	fi_close(&cntr_fid->fid);
}

/**
 * @brief Test efa_hw_cntr_read calls ibv_read_comp_cntr and returns the value
 */
void test_efa_hw_cntr_read(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_cntr *cntr_fid;

	cntr_fid = test_efa_hw_cntr_open(resource, true);
	g_efa_unit_test_mocks.ibv_read_comp_cntr = efa_mock_ibv_read_comp_cntr_return_mock;

	will_return(efa_mock_ibv_read_comp_cntr_return_mock, 100);
	assert_int_equal(fi_cntr_read(cntr_fid), 100);

	fi_close(&cntr_fid->fid);
}

/**
 * @brief Test efa_hw_cntr_readerr calls ibv_read_err_comp_cntr and returns the value
 */
void test_efa_hw_cntr_readerr(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_cntr *cntr_fid;

	cntr_fid = test_efa_hw_cntr_open(resource, true);
	g_efa_unit_test_mocks.ibv_read_err_comp_cntr = efa_mock_ibv_read_err_comp_cntr_return_mock;

	will_return(efa_mock_ibv_read_err_comp_cntr_return_mock, 100);
	assert_int_equal(fi_cntr_readerr(cntr_fid), 100);

	fi_close(&cntr_fid->fid);
}

/**
 * @brief Verify fi_enable succeeds when ibv_qp_attach_comp_cntr returns 0
 */
void test_efa_hw_cntr_bind_ep(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_cntr *cntr_fid = NULL;
	int ret;

	cntr_fid = test_efa_hw_cntr_open(resource, false);
	g_efa_unit_test_mocks.ibv_qp_attach_comp_cntr = efa_mock_ibv_qp_attach_comp_cntr_return_mock;
	/* ibv_qp_attach_comp_cntr is deferred to fi_enable when qp is created */
	ret = fi_ep_bind(resource->ep, &cntr_fid->fid, FI_SEND);
	assert_int_equal(ret, 0);

	ret = fi_enable(resource->ep);
	assert_int_equal(ret, 0);

	struct efa_base_ep *base_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);
	assert_true(base_ep->efa_qp_enabled);

	fi_close(&resource->ep->fid);
	resource->ep = NULL;

	fi_close(&cntr_fid->fid);
}

/**
 * @brief Verify fi_enable fails when ibv_qp_attach_comp_cntr returns
 *        -ENOTSUP for a counter bound with FI_REMOTE_READ.
 */
void test_efa_hw_cntr_bind_ep_attach_fail(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_cntr *cntr_fid = NULL;
	int ret;

	cntr_fid = test_efa_hw_cntr_open(resource, false);
	g_efa_unit_test_mocks.ibv_qp_attach_comp_cntr = efa_mock_ibv_qp_attach_comp_cntr_return_enotsup;

	ret = fi_ep_bind(resource->ep, &cntr_fid->fid, FI_REMOTE_READ);
	assert_int_equal(ret, 0);

	ret = fi_enable(resource->ep);
	assert_int_equal(ret, -FI_EOPNOTSUPP);

	fi_close(&resource->ep->fid);
	resource->ep = NULL;

	fi_close(&cntr_fid->fid);
}
#else
void test_efa_hw_cntr_open_unsupported_type_bytes(struct efa_resource **state) { skip(); }
void test_efa_hw_cntr_open_max_cntr_value_exceeded(struct efa_resource **state) { skip(); }
void test_efa_hw_cntr_open_ibv_fail(struct efa_resource **state) { skip(); }
void test_efa_hw_cntr_add(struct efa_resource **state) { skip(); }
void test_efa_hw_cntr_adderr(struct efa_resource **state) { skip(); }
void test_efa_hw_cntr_set(struct efa_resource **state) { skip(); }
void test_efa_hw_cntr_seterr(struct efa_resource **state) { skip(); }
void test_efa_hw_cntr_read(struct efa_resource **state) { skip(); }
void test_efa_hw_cntr_readerr(struct efa_resource **state) { skip(); }
void test_efa_hw_cntr_bind_ep(struct efa_resource **state) { skip(); }
void test_efa_hw_cntr_bind_ep_attach_fail(struct efa_resource **state) { skip(); }
#endif /* HAVE_EFADV_CREATE_COMP_CNTR */
