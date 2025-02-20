/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"
#include "efa_prov_info.h"

/**
 * @brief test that when a wrong fi_info was used to open resource, the error is handled
 * gracefully
 */
void test_info_open_ep_with_wrong_info()
{
	struct fi_info *hints, *info;
	struct fid_fabric *fabric = NULL;
	struct fid_domain *domain = NULL;
	struct fid_ep *ep = NULL;
	int err;

	hints = efa_unit_test_alloc_hints(FI_EP_DGRAM, EFA_FABRIC_NAME);

	err = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, hints, &info);
	assert_int_equal(err, 0);

	/* dgram endpoint require FI_MSG_PREFIX */
	assert_int_equal(info->mode, FI_MSG_PREFIX | FI_CONTEXT2);

	/* make the info wrong by setting the mode to 0 */
	info->mode = 0;

	err = fi_fabric(info->fabric_attr, &fabric, NULL);
	assert_int_equal(err, 0);

	err = fi_domain(fabric, info, &domain, NULL);
	assert_int_equal(err, 0);

	/* because of the error in the info object, fi_endpoint() should fail with -FI_ENODATA */
	err = fi_endpoint(domain, info, &ep, NULL);
	assert_int_equal(err, -FI_ENODATA);
	assert_null(ep);

	err = fi_close(&domain->fid);
	assert_int_equal(err, 0);

	err = fi_close(&fabric->fid);
	assert_int_equal(err, 0);
}

/**
 * @brief Verify that efa rdm path fi_info objects have some expected values
 */
void test_info_rdm_attributes()
{
	struct fi_info *hints, *info = NULL, *info_head = NULL;
	int err;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	assert_non_null(hints);

	err = fi_getinfo(FI_VERSION(1,6), NULL, NULL, 0, hints, &info_head);
	assert_int_equal(err, 0);
	assert_non_null(info_head);

	for (info = info_head; info; info = info->next) {
		assert_true(!strcmp(info->fabric_attr->name, EFA_FABRIC_NAME));
		assert_true(strstr(info->domain_attr->name, "rdm"));
		assert_int_equal(info->ep_attr->max_msg_size, UINT64_MAX);
#if HAVE_CUDA || HAVE_NEURON || HAVE_SYNAPSEAI
		assert_true(info->caps | FI_HMEM);
#endif
	}
}

/**
 * @brief Verify that efa dgram path fi_info objects have some expected values
 */
void test_info_dgram_attributes()
{
	struct fi_info *hints, *info = NULL, *info_head = NULL;
	int err;

	hints = efa_unit_test_alloc_hints(FI_EP_DGRAM, EFA_FABRIC_NAME);
	assert_non_null(hints);

	err = fi_getinfo(FI_VERSION(1,6), NULL, NULL, 0, hints, &info_head);
	assert_int_equal(err, 0);
	assert_non_null(info_head);

	for (info = info_head; info; info = info->next) {
		assert_true(!strcmp(info->fabric_attr->name, EFA_FABRIC_NAME));
		assert_true(strstr(info->domain_attr->name, "dgrm"));
	}
}

/**
 * @brief Verify that efa direct path fi_info objects have some expected values
 */
static void test_info_direct_attributes_impl(struct fi_info *hints,
					     int expected_return)
{
	struct fi_info *info = NULL, *info_head = NULL;
	int err;

	err = fi_getinfo(FI_VERSION(1, 6), NULL, NULL, 0, hints, &info_head);
	assert_int_equal(err, expected_return);
	if (expected_return)
		return;

	assert_non_null(info_head);
	for (info = info_head; info; info = info->next) {
		assert_true(!strcmp(info->fabric_attr->name,
				    EFA_DIRECT_FABRIC_NAME));
		assert_true(strstr(info->domain_attr->name, "rdm"));
		assert_false(info->caps & (FI_ATOMIC | FI_TAGGED));
		assert_false(info->tx_attr->msg_order & FI_ORDER_SAS);
		assert_int_equal(
			info->ep_attr->max_msg_size,
			(hints->caps & FI_RMA) ?
				g_device_list[0].max_rdma_size :
				g_device_list[0].ibv_port_attr.max_msg_sz);
	}

	fi_freeinfo(info_head);
}

void test_info_direct_attributes_no_rma()
{
	struct fi_info *hints;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	assert_non_null(hints);

	/* The default hints for efa-direct only has FI_MSG cap, so it should
	 * succeed anyway */
	test_info_direct_attributes_impl(hints, FI_SUCCESS);
	fi_freeinfo(hints);
}

void test_info_direct_attributes_rma()
{
	struct fi_info *hints;
	bool support_fi_rma = (efa_device_support_rdma_read() &&
			       efa_device_support_rdma_write() &&
			       efa_device_support_unsolicited_write_recv());

	int expected_return = support_fi_rma ? FI_SUCCESS : -FI_ENODATA;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	assert_non_null(hints);

	hints->caps |= FI_RMA;
	test_info_direct_attributes_impl(hints, expected_return);
	fi_freeinfo(hints);
}

/**
 * @brief Verify that efa direct only supports HMEM with p2p
 */
#if HAVE_CUDA || HAVE_NEURON || HAVE_SYNAPSEAI
void test_info_direct_hmem_support_p2p()
{
	struct fi_info *info;
	bool hmem_ops_cuda_init;

	info = fi_allocinfo();

	memset(g_efa_hmem_info, 0, OFI_HMEM_MAX * sizeof(struct efa_hmem_info));

	/* Save current value of hmem_ops[FI_HMEM_CUDA].initialized to reset later
	 * hmem_ops is populated in ofi_hmem_init and only runs once
	 *
	 * CUDA iface will be initialized on Nvidia GPU platforms but not on others
	 * Force setting hmem_ops[FI_HMEM_CUDA].initialized allows this test to
	 * run on all instance types
	*/
	hmem_ops_cuda_init = hmem_ops[FI_HMEM_CUDA].initialized;
	hmem_ops[FI_HMEM_CUDA].initialized = true;

	/* g_efa_hmem_info is backed up and reset after every test in
	 * efa_unit_test_mocks_teardown. So no need to save and reset these fields
	 */
	g_efa_hmem_info[FI_HMEM_CUDA].initialized = true;
	g_efa_hmem_info[FI_HMEM_CUDA].p2p_supported_by_device = true;

	efa_prov_info_set_hmem_flags(info, FI_EP_RDM);
	assert_true(info->caps & FI_HMEM);
	assert_true(info->tx_attr->caps & FI_HMEM);
	assert_true(info->rx_attr->caps & FI_HMEM);
	fi_freeinfo(info);

	info = fi_allocinfo();
	g_efa_hmem_info[FI_HMEM_CUDA].initialized = true;
	g_efa_hmem_info[FI_HMEM_CUDA].p2p_supported_by_device = false;

	efa_prov_info_set_hmem_flags(info, FI_EP_RDM);
	assert_false(info->caps & FI_HMEM);
	assert_false(info->tx_attr->caps & FI_HMEM);
	assert_false(info->rx_attr->caps & FI_HMEM);
	fi_freeinfo(info);

	/* Reset hmem_ops[FI_HMEM_CUDA].initialized */
	hmem_ops[FI_HMEM_CUDA].initialized = hmem_ops_cuda_init;
}
#else
void test_info_direct_hmem_support_p2p()
{
}
#endif

/**
 * @brief Verify info->tx/rx_attr->msg_order is set according to hints.
 *
 */
static void
test_info_tx_rx_msg_order_from_hints(struct fi_info *hints, int expected_ret)
{
	struct fi_info *info;
	int err;

	err = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), NULL, NULL, 0ULL, hints, &info);

	assert_int_equal(err, expected_ret);

	if (expected_ret == FI_SUCCESS) {
		assert_true(hints->tx_attr->msg_order == info->tx_attr->msg_order);
		assert_true(hints->rx_attr->msg_order == info->rx_attr->msg_order);
	}

	fi_freeinfo(info);
}

/**
 * @brief Verify info->tx/rx_attr->op_flags is set according to hints.
 *
 */
static void
test_info_tx_rx_op_flags_from_hints(struct fi_info *hints, int expected_ret)
{
	struct fi_info *info;
	int err;

	err = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), NULL, NULL, 0ULL, hints, &info);

	assert_int_equal(err, expected_ret);

	if (expected_ret == FI_SUCCESS) {
		assert_true(hints->tx_attr->op_flags == info->tx_attr->op_flags);
		assert_true(hints->rx_attr->op_flags == info->rx_attr->op_flags);
	}

	fi_freeinfo(info);
}

/**
 * @brief Verify info->tx/rx_attr->size is set according to hints.
 *
 */
static void test_info_tx_rx_size_from_hints(struct fi_info *hints, int expected_ret)
{
	struct fi_info *info;
	int err;

	err = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), NULL, NULL, 0ULL, hints, &info);

	assert_int_equal(err, expected_ret);

	if (expected_ret == FI_SUCCESS) {
		assert_true(hints->tx_attr->size == info->tx_attr->size);
		assert_true(hints->rx_attr->size == info->rx_attr->size);
	}

	fi_freeinfo(info);
}

void test_info_tx_rx_msg_order_rdm_order_none(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);

	test_info_tx_rx_msg_order_from_hints(resource->hints, 0);
}

void test_info_tx_rx_msg_order_rdm_order_sas(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);

	resource->hints->tx_attr->msg_order = FI_ORDER_SAS;
	resource->hints->rx_attr->msg_order = FI_ORDER_SAS;
	test_info_tx_rx_msg_order_from_hints(resource->hints, 0);
}

void test_info_tx_rx_msg_order_dgram_order_none(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_DGRAM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);

	test_info_tx_rx_msg_order_from_hints(resource->hints, 0);
}

/**
 * @brief dgram endpoint doesn't support any ordering, so fi_getinfo
 * should return -FI_ENODATA if hints requests sas
 */
void test_info_tx_rx_msg_order_dgram_order_sas(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_DGRAM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);

	resource->hints->tx_attr->msg_order = FI_ORDER_SAS;
	resource->hints->rx_attr->msg_order = FI_ORDER_SAS;
	test_info_tx_rx_msg_order_from_hints(resource->hints, -FI_ENODATA);
}

/**
 * @brief Verify max order size is set correctly according to hints
 * 
 * @param hints hints
 * @param expected_ret expected rc of fi_getinfo
 * @param expected_size expected value of max_order_*_size. Ignored when expected_ret is non-zero.
 */
static void
test_info_max_order_size_from_hints(struct fi_info *hints, int expected_ret, size_t expected_size)
{
	struct fi_info *info;
	int err;

	err = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), NULL, NULL, 0ULL, hints, &info);

	assert_int_equal(err, expected_ret);

	if (expected_ret == FI_SUCCESS) {
		assert_true(info->ep_attr->max_order_raw_size == expected_size);
		assert_true(info->ep_attr->max_order_war_size == expected_size);
		assert_true(info->ep_attr->max_order_waw_size == expected_size);
	}

	fi_freeinfo(info);
}

/**
 * DGRAM ep type doesn't support FI_ATOMIC, fi_getinfo should return
 * ENODATA when FI_ATOMIC is requested in hints.
 */
void test_info_max_order_size_dgram_with_atomic(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_DGRAM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);

	resource->hints->caps = FI_ATOMIC;

	test_info_max_order_size_from_hints(resource->hints, -FI_ENODATA, 0);
}

/**
 * RDM ep type supports FI_ATOMIC. When FI_ORDER_ATOMIC_* is NOT requested,
 * max_order_*_size should be 0
 */
void test_info_max_order_size_rdm_with_atomic_no_order(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);


	resource->hints->caps = FI_ATOMIC;
	resource->hints->domain_attr->mr_mode |= FI_MR_VIRT_ADDR | FI_MR_PROV_KEY;

	test_info_max_order_size_from_hints(resource->hints, FI_SUCCESS, 0);
}

/**
 * RDM ep type supports FI_ATOMIC. When FI_ORDER_ATOMIC_* is requested,
 * max_order_*_size should be the max atomic size derived from mtu and headers
 */
void test_info_max_order_size_rdm_with_atomic_order(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	size_t max_atomic_size = g_device_list[0].ibv_port_attr.max_msg_sz
					- sizeof(struct efa_rdm_rta_hdr)
					- g_device_list[0].rdm_info->src_addrlen
					- EFA_RDM_IOV_LIMIT * sizeof(struct fi_rma_iov);

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);

	resource->hints->caps = FI_ATOMIC;
	resource->hints->domain_attr->mr_mode |= FI_MR_VIRT_ADDR | FI_MR_PROV_KEY;
	resource->hints->tx_attr->msg_order |= FI_ORDER_ATOMIC_RAR | FI_ORDER_ATOMIC_RAW | FI_ORDER_ATOMIC_WAR | FI_ORDER_ATOMIC_WAW;
	resource->hints->rx_attr->msg_order = resource->hints->tx_attr->msg_order;

	test_info_max_order_size_from_hints(resource->hints, FI_SUCCESS, max_atomic_size);
}

void test_info_tx_rx_op_flags_rdm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);

	resource->hints->tx_attr->op_flags = FI_DELIVERY_COMPLETE;
	resource->hints->rx_attr->op_flags = FI_COMPLETION;
	test_info_tx_rx_op_flags_from_hints(resource->hints, 0);
}

void test_info_tx_rx_size_rdm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);

	resource->hints->tx_attr->size = 16;
	resource->hints->rx_attr->size = 16;
	test_info_tx_rx_size_from_hints(resource->hints, 0);
}

static void test_info_check_shm_info_from_hints(struct fi_info *hints)
{
	struct fi_info *info;
	struct fid_fabric *fabric = NULL;
	struct fid_domain *domain = NULL;
	int err;
	struct efa_domain *efa_domain;

	err = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, hints, &info);
	/* Do nothing if the current setup does not support FI_HMEM */
	if (err && hints->caps & FI_HMEM) {
		return;
	}
	assert_int_equal(err, 0);

	err = fi_fabric(info->fabric_attr, &fabric, NULL);
	assert_int_equal(err, 0);

	err = fi_domain(fabric, info, &domain, NULL);
	assert_int_equal(err, 0);

	efa_domain = container_of(domain, struct efa_domain, util_domain.domain_fid);
	if (efa_domain->shm_info) {
		if (hints->caps & FI_HMEM)
			assert_true(efa_domain->shm_info->caps & FI_HMEM);
		else
			assert_false(efa_domain->shm_info->caps & FI_HMEM);

		assert_true(efa_domain->shm_info->tx_attr->op_flags == info->tx_attr->op_flags);

		assert_true(efa_domain->shm_info->rx_attr->op_flags == info->rx_attr->op_flags);

		if (hints->domain_attr->threading) {
			assert_true(hints->domain_attr->threading == info->domain_attr->threading);
			assert_true(hints->domain_attr->threading == efa_domain->shm_info->domain_attr->threading);
		}
	}

	fi_close(&domain->fid);

	fi_close(&fabric->fid);

	fi_freeinfo(info);
}

/**
 * @brief Check shm info created by efa_domain() has correct caps.
 *
 */
void test_info_check_shm_info_hmem()
{
	struct fi_info *hints;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);

	hints->caps |= FI_HMEM;
	test_info_check_shm_info_from_hints(hints);

	hints->caps &= ~FI_HMEM;
	test_info_check_shm_info_from_hints(hints);
}

void test_info_check_shm_info_op_flags()
{
	struct fi_info *hints;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);

	hints->tx_attr->op_flags |= FI_COMPLETION;
	hints->rx_attr->op_flags |= FI_COMPLETION;
	test_info_check_shm_info_from_hints(hints);

	hints->tx_attr->op_flags |= FI_DELIVERY_COMPLETE;
	hints->rx_attr->op_flags |= FI_MULTI_RECV;
	test_info_check_shm_info_from_hints(hints);
}

void test_info_check_shm_info_threading()
{
	struct fi_info *hints;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);

	hints->domain_attr->threading = FI_THREAD_DOMAIN;
	test_info_check_shm_info_from_hints(hints);
}

/**
 * @brief Check that case when a user requested FI_HMEM support
 *        using libfabric API < 1.18,
 */
void test_info_check_hmem_cuda_support_on_api_lt_1_18()
{
	struct fi_info *hints, *info = NULL;
	int err;

	if (!hmem_ops[FI_HMEM_CUDA].initialized)
		skip();

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);

	hints->caps |= FI_HMEM;
	hints->domain_attr->mr_mode |= FI_MR_HMEM;

	/* For libfabric API < 1.18,
	 * on a system that support GPUDirect RDMA read,
	 * HMEM cuda is available when GPUDirect RDMA is available,
	 * and environment variable FI_EFA_USE_DEVICE_RDMA set to 1/on/true
	 * otherwise it is not available.
	 */
	setenv("FI_EFA_USE_DEVICE_RDMA", "1", true /* overwrite */);
	err = fi_getinfo(FI_VERSION(1, 6), NULL, NULL, 0, hints, &info);
	if (efa_device_support_rdma_read()) {
		assert_int_equal(err, 0);
		fi_freeinfo(info);
	} else {
		assert_int_equal(err, -FI_ENODATA);
	}

	setenv("FI_EFA_USE_DEVICE_RDMA", "0", true /* overwrite */);
	err = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0, hints, &info);
	assert_int_equal(err, -FI_ENODATA);

	unsetenv("FI_EFA_USE_DEVICE_RDMA");
}

/**
 * @brief Check that case when a user requested FI_HMEM support
 *        using libfabric API >= 1.18,
 */
void test_info_check_hmem_cuda_support_on_api_ge_1_18()
{
	struct fi_info *hints, *info = NULL;
	int err;

	if (!hmem_ops[FI_HMEM_CUDA].initialized)
		skip();

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);

	hints->caps |= FI_HMEM;
	hints->domain_attr->mr_mode |= FI_MR_HMEM;

	/* Piror to version 1.18, libfabric EFA provider support CUDA
	 * memory only when GPUDirect RDMA is available.
	 * In version 1.18, libfabric EFA provider implemented
	 * universal CUDA support through CUDA library.
	 * However, this feature (universal CUDA support) can cause some
	 * middleware to deadlock, thus it is only available
	 * when a user is using 1.18 API.
	 */
	err = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info);
	assert_int_equal(err, 0);
	fi_freeinfo(info);
}

void check_no_hmem_support_when_not_requested(char *fabric_name)
{
	struct fi_info *hints, *info = NULL;
	int err;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, fabric_name);

	err = fi_getinfo(FI_VERSION(1,6), NULL, NULL, 0, hints, &info);
	assert_int_equal(err, 0);
	assert_non_null(info);
	assert_false(info->caps & FI_HMEM);
	fi_freeinfo(info);
}

/**
 * @brief Check that EFA does not claim support of FI_HMEM when
 *        it is not requested
 */
void test_info_check_no_hmem_support_when_not_requested() {
	check_no_hmem_support_when_not_requested(EFA_FABRIC_NAME);
	check_no_hmem_support_when_not_requested(EFA_DIRECT_FABRIC_NAME);
}

/**
 * @brief Check that EFA direct info object is not returned when atomic
 *        or ordering capabilities are requested
 */
void test_info_direct_unsupported()
{
	struct fi_info *hints, *info = NULL;
	int err;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	assert_non_null(hints);

	err = fi_getinfo(FI_VERSION(1,6), NULL, NULL, 0, hints, &info);
	assert_int_equal(err, 0);
	assert_non_null(info);

	hints->caps |= FI_ATOMIC;
	err = fi_getinfo(FI_VERSION(1,6), NULL, NULL, 0, hints, &info);
	assert_int_equal(err, -FI_ENODATA);
	assert_null(info);

	hints->caps &= ~FI_ATOMIC;
	hints->tx_attr->msg_order = FI_ORDER_SAS;
	hints->rx_attr->msg_order = FI_ORDER_SAS;
	err = fi_getinfo(FI_VERSION(1,6), NULL, NULL, 0, hints, &info);
	assert_int_equal(err, -FI_ENODATA);
	assert_null(info);
}

/**
 * @brief Verify that efa-direct fi_info objects are returned before efa info objects
 */
void test_info_direct_ordering()
{
	struct fi_info *hints, *info = NULL, *info_head = NULL;
	bool efa_direct_returned = false, efa_returned = false;
	bool efa_direct_returned_after_efa = false, efa_returned_after_efa_direct = false;
	int err;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, NULL);
	assert_non_null(hints);

	err = fi_getinfo(FI_VERSION(1,6), NULL, NULL, 0, hints, &info_head);
	assert_int_equal(err, 0);
	assert_non_null(info_head);

	for (info = info_head; info; info = info->next) {
		if (!strcmp(info->fabric_attr->name, EFA_DIRECT_FABRIC_NAME)) {
			efa_direct_returned = true;
			if (efa_returned)
				efa_direct_returned_after_efa = true;
		}
		if (!strcmp(info->fabric_attr->name, EFA_FABRIC_NAME)) {
			efa_returned = true;
			if (efa_direct_returned)
				efa_returned_after_efa_direct = true;
		}
	}

	assert_true(efa_direct_returned);
	assert_true(efa_returned);
	assert_true(efa_returned_after_efa_direct);
	assert_false(efa_direct_returned_after_efa);
}

/**
 * @brief core test function for use_device_rdma
 *
 * @param env_val 0/1/-1: set FI_EFA_USE_DEVICE_RDMA to 0 or 1, or leave it unset (-1)
 * @param setopt_val 0/1/-1: set use_device_rdma using fi_setopt to 0 or 1, or leave it unset (-1)
 * @param expected_val expected result of ep->use_device_rdma
 * @param api_version API version to use.
 */
void test_use_device_rdma( const int env_val,
			   const int setopt_val,
			   const int expected_val,
			   const uint32_t api_version )
{
	int ret = 0;
	struct fi_info *hints, *info;
	struct fid_fabric *fabric = NULL;
	struct fid_domain *domain = NULL;
	struct fid_ep *ep = NULL;
	struct efa_rdm_ep *efa_rdm_ep;
	bool rdma_capable_hw;
	char env_str[16];

	if (env_val >= 0) {
		snprintf(env_str, 15, "%d", env_val);
		setenv("FI_EFA_USE_DEVICE_RDMA", env_str, 1);
	} else {
		unsetenv("FI_EFA_USE_DEVICE_RDMA");
	}

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);

	ret = fi_getinfo(api_version, NULL, NULL, 0ULL, hints, &info);
	assert_int_equal(ret, 0);

	rdma_capable_hw = efa_device_support_rdma_read();

	if (expected_val && !rdma_capable_hw) {
		/* cannot test USE_DEVICE_RDMA=1, hardware
		   doesn't support it, and will abort() */
		fi_freeinfo(info);
		skip();
		return;
	}

	ret = fi_fabric(info->fabric_attr, &fabric, NULL);
	assert_int_equal(ret, 0);

	ret = fi_domain(fabric, info, &domain, NULL);
	assert_int_equal(ret, 0);

	fi_endpoint(domain, info, &ep, NULL);
	assert_int_equal(ret, 0);

	if (setopt_val >= 0) {
		bool b_val = (bool) setopt_val;
		int ret_setopt;
		ret_setopt = fi_setopt(&ep->fid, FI_OPT_ENDPOINT,
			FI_OPT_EFA_USE_DEVICE_RDMA, &b_val, sizeof(bool));
		if (FI_VERSION_LT(api_version, FI_VERSION(1,18))) {
			assert_int_not_equal(ret_setopt, 0);
		}
		else if (expected_val != setopt_val) {
			assert_int_not_equal(ret_setopt, 0);
		}
		else {
			assert_int_equal(ret_setopt, 0);
		}
	}

	efa_rdm_ep = container_of(ep, struct efa_rdm_ep,
			base_ep.util_ep.ep_fid.fid);
	assert_int_equal( efa_rdm_ep->use_device_rdma, expected_val );

	assert_int_equal(fi_close(&ep->fid), 0);
	assert_int_equal(fi_close(&domain->fid), 0);
	assert_int_equal(fi_close(&fabric->fid), 0);
	fi_freeinfo(info);

	return;
}

/**
 * Get the name of the "first"(random order) NIC
 *
 * @param[out]	name	The returned name string.
 * 			It should be free'd after use.
 * @returns	FI_SUCCESS on success or a non-zero error code
 */
static int get_first_nic_name(char **name) {
	int ret;
	char *nic_name = NULL;
	struct fi_info *hints, *info;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	ret = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, hints, &info);
	fi_freeinfo(hints);
	if (ret)
		return ret;

	nic_name = info->nic->device_attr->name;
	assert_non_null(nic_name);
	assert_int_not_equal(strlen(nic_name), 0);

	*name = malloc(strlen(nic_name) + 1);
	if (!name)
		return FI_ENOMEM;

	strcpy(*name, nic_name);

	fi_freeinfo(info);

	return FI_SUCCESS;
}

/**
 * Verify the returned NIC from fi_getinfo.
 * Ideally we want to test multi-NIC selection logic, but this test is most likely
 * run on single-NIC platforms. Therefore we make a compromise and only verify the
 * "first" NIC.
 *
 * @param[in]	filter			The value that would be set for FI_EFA_IFACE
 * @param[in]	expect_first_name	The expected name of the "first" NIC	
 */
static void test_efa_nic_selection(const char *filter, const char *expect_first_name, char *fabric_name) {
	int ret;
	struct fi_info *hints, *info;

	efa_env.iface = (char *) filter;
	hints = efa_unit_test_alloc_hints(FI_EP_RDM, fabric_name);
	ret = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, hints, &info);
	fi_freeinfo(hints);
	if (expect_first_name) {
		assert_int_equal(FI_SUCCESS, ret);
		assert_string_equal(expect_first_name, info->nic->device_attr->name);
		fi_freeinfo(info);
	} else {
		assert_int_not_equal(FI_SUCCESS, ret);
	}
}

/**
 * Verify NICs are returned if FI_EFA_IFACE=all
 */
void test_efa_nic_select_all_devices_matches() {
	int ret;
	char *nic_name;

	ret = get_first_nic_name(&nic_name);
	assert_int_equal(ret, FI_SUCCESS);

	test_efa_nic_selection("all", nic_name, EFA_FABRIC_NAME);
	test_efa_nic_selection("all", nic_name, EFA_DIRECT_FABRIC_NAME);

	free(nic_name);
}

/**
 * Verify the "first" NIC can be selected by name
 */
void test_efa_nic_select_first_device_matches() {
	int ret;
	char *nic_name;

	ret = get_first_nic_name(&nic_name);
	assert_int_equal(ret, FI_SUCCESS);

	test_efa_nic_selection(nic_name, nic_name, EFA_FABRIC_NAME);
	test_efa_nic_selection(nic_name, nic_name, EFA_DIRECT_FABRIC_NAME);

	free(nic_name);
}

/**
 * Verify that surrounding commas are handled correctly,
 * i.e. ignored, to match the NIC name.
 */
void test_efa_nic_select_first_device_with_surrounding_comma_matches() {
	int ret;
	char *nic_name, *filter;

	ret = get_first_nic_name(&nic_name);
	assert_int_equal(ret, FI_SUCCESS);

	filter = malloc(strlen(nic_name) + 3);
	assert_non_null(filter);

	strcpy(filter, ",");
	strcat(filter, nic_name);
	strcat(filter, ",");

	test_efa_nic_selection(filter, nic_name, EFA_FABRIC_NAME);
	test_efa_nic_selection(filter, nic_name, EFA_DIRECT_FABRIC_NAME);

	free(filter);
	free(nic_name);
}

/**
 * Verify that only full NIC names are matched, and prefixes,
 * e.g. the first letter, will not accidentally select the wrong NIC.
 */
void test_efa_nic_select_first_device_first_letter_no_match() {
	int ret;
	char *nic_name, filter[2];

	ret = get_first_nic_name(&nic_name);
	assert_int_equal(ret, FI_SUCCESS);

	filter[0] = nic_name[0];
	filter[1] = '\0';

	test_efa_nic_selection(filter, NULL, EFA_FABRIC_NAME);
	test_efa_nic_selection(filter, NULL, EFA_DIRECT_FABRIC_NAME);

	free(nic_name);
}

/**
 * Verify that empty NIC names will not select any NIC
 */
void test_efa_nic_select_empty_device_no_match() {
	test_efa_nic_selection(",", NULL, EFA_FABRIC_NAME);
	test_efa_nic_selection(",", NULL, EFA_DIRECT_FABRIC_NAME);
}

/* indicates the test shouldn't set the setopt or environment
   variable during setup. */
const int VALUE_NOT_SET = -1;

/* settings agree: on */
void test_efa_use_device_rdma_env1_opt1() {
	test_use_device_rdma(1, 1, 1, FI_VERSION(1,18));
}
/* settings agree: off */
void test_efa_use_device_rdma_env0_opt0() {
	test_use_device_rdma(0, 0, 0, FI_VERSION(1,18));
}
/* settings conflict, env on */
void test_efa_use_device_rdma_env1_opt0() {
	test_use_device_rdma(1, 0, 1, FI_VERSION(1,18));
}
/* settings conflict, env off */
void test_efa_use_device_rdma_env0_opt1() {
	test_use_device_rdma(0, 1, 0, FI_VERSION(1,18));
}
/* setopt only on */
void test_efa_use_device_rdma_opt1() {
	test_use_device_rdma(VALUE_NOT_SET, 1, 1, FI_VERSION(1,18));
}
/* setopt only off */
void test_efa_use_device_rdma_opt0() {
	test_use_device_rdma(VALUE_NOT_SET, 0, 0, FI_VERSION(1,18));
}
/* environment only on */
void test_efa_use_device_rdma_env1() {
	test_use_device_rdma(1, VALUE_NOT_SET, 1, FI_VERSION(1,18));
}
/* environment only off */
void test_efa_use_device_rdma_env0() {
	test_use_device_rdma(0, VALUE_NOT_SET, 0, FI_VERSION(1,18));
}
/* setopt rejected in 1,17 */
void test_efa_use_device_rdma_opt_old() {
	test_use_device_rdma(1, 1, 1, FI_VERSION(1,17));
	test_use_device_rdma(0, 0, 0, FI_VERSION(1,17));
}
