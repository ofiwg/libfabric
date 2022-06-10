#include "efa_unit_tests.h"

static void check_ep_pkt_pool_flags(struct efa_resource resource, int expected_flags)
{
       struct fid_ep *ep;
       struct rxr_ep *rxr_ep;
       int ret;
       ret = fi_endpoint(resource.domain, resource.info, &ep, NULL);
       assert_int_equal(ret, 0);
       rxr_ep = container_of(ep, struct rxr_ep, util_ep.ep_fid);
       assert_int_equal(rxr_ep->efa_tx_pkt_pool->attr.flags, expected_flags);
       assert_int_equal(rxr_ep->efa_rx_pkt_pool->attr.flags, expected_flags);
       fi_close(&ep->fid);
}

/**
 * @brief Test the pkt pool flags in rxr_ep_init()
 */
void test_rxr_ep_pkt_pool_flags()
{
	int ret;
	struct efa_resource resource = {0};
	ret = efa_unit_test_resource_construct(&resource);
	assert_int_equal(ret, 0);

	g_efa_fork_status = EFA_FORK_SUPPORT_ON;
	check_ep_pkt_pool_flags(resource, OFI_BUFPOOL_NONSHARED);

	g_efa_fork_status = EFA_FORK_SUPPORT_OFF;
	check_ep_pkt_pool_flags(resource, OFI_BUFPOOL_HUGEPAGES);

	g_efa_fork_status = EFA_FORK_SUPPORT_UNNEEDED;
	check_ep_pkt_pool_flags(resource, OFI_BUFPOOL_HUGEPAGES);

	efa_unit_test_resource_destroy(&resource);
}

/**
 * @brief When the buf pool is created with OFI_BUFPOOL_NONSHARED,
 * test if the allocated memory is page aligned.
 */
void test_rxr_ep_pkt_pool_page_alignment()
{
	int ret;
	struct ofi_bufpool_region *buf;
	struct fid_ep *ep;
	struct rxr_ep *rxr_ep;
	struct efa_resource resource = {0};
	ret = efa_unit_test_resource_construct(&resource);
	assert_int_equal(ret, 0);

	g_efa_fork_status = EFA_FORK_SUPPORT_ON;
	ret = fi_endpoint(resource.domain, resource.info, &ep, NULL);
	assert_int_equal(ret, 0);
	rxr_ep = container_of(ep, struct rxr_ep, util_ep.ep_fid);
	assert_int_equal(rxr_ep->efa_rx_pkt_pool->attr.flags, OFI_BUFPOOL_NONSHARED);

	buf = ofi_buf_alloc(rxr_ep->efa_rx_pkt_pool);
	assert_non_null(buf);
	assert_true(((uintptr_t)ofi_buf_region(buf)->alloc_region % ofi_get_page_size()) == 0);
	ofi_buf_free(buf);

	fi_close(&ep->fid);
	efa_unit_test_resource_destroy(&resource);
}