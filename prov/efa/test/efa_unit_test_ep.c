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

	g_efa_fork_status = EFA_FORK_SUPPORT_OFF;
}

/**
 * @brief when delivery complete atomic was used and handshake packet has not been received
 * verify there is no tx entry leak
 */
void test_rxr_ep_dc_atomic_error_handling()
{
	struct fid_ep *ep;
	struct rxr_ep *rxr_ep;
	struct rdm_peer *peer;
	struct fi_ioc ioc = {0};
	struct fi_rma_ioc rma_ioc = {0};
	struct fi_msg_atomic msg = {0};
	struct efa_resource resource = {0};
	struct efa_ep_addr raw_addr = {0};
	struct ibv_ah ibv_ah = {0};
	fi_addr_t peer_addr;
	int buf[1] = {0}, err, numaddr;

	err = efa_unit_test_resource_construct(&resource);
	assert_int_equal(err, 0);

	err = fi_endpoint(resource.domain, resource.info, &ep, NULL);
	assert_int_equal(err, 0);

	err = fi_ep_bind(ep, &resource.av->fid, 0);
	assert_int_equal(err, 0);

	/* create a fake peer */
	raw_addr.raw[0] = 0xfe;
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	expect_any(__wrap_ibv_create_ah, pd);
	expect_any(__wrap_ibv_create_ah, attr);
	will_return(__wrap_ibv_create_ah, &ibv_ah);
	expect_any(__wrap_efadv_query_ah, ibvah);
	expect_any(__wrap_efadv_query_ah, attr);
	expect_any(__wrap_efadv_query_ah, inlen);
	will_return(__wrap_efadv_query_ah, 0);
	numaddr = fi_av_insert(resource.av, &raw_addr, 1, &peer_addr, 0, NULL);
	assert_int_equal(numaddr, 1);

	msg.addr = peer_addr;

	ioc.addr = buf;
	ioc.count = 1;
	msg.msg_iov = &ioc;
	msg.iov_count = 1;

	msg.rma_iov = &rma_ioc;
	msg.rma_iov_count = 1;
	msg.datatype = FI_INT32;
	msg.op = FI_SUM;

	rxr_ep = container_of(ep, struct rxr_ep, util_ep.ep_fid);
	rxr_ep->use_shm_for_tx = false;
	/* set peer->flag to RXR_PEER_REQ_SENT will make rxr_atomic() think
	 * a REQ packet has been sent to the peer (so no need to send again)
	 * handshake has not been received, so we do not know whether the peer support DC
	 */
	peer = rxr_ep_get_peer(rxr_ep, peer_addr);
	peer->flags = RXR_PEER_REQ_SENT;

	assert_true(dlist_empty(&rxr_ep->tx_entry_list));
	err = fi_atomicmsg(ep, &msg, FI_DELIVERY_COMPLETE);
	/* DC has been reuquested, but ep do not know whether peer supports it, therefore
	 * -FI_EAGAIN should be returned
	 */
	assert_int_equal(err, -FI_EAGAIN);
	/* make sure there is no leaking of tx_entry */
	assert_true(dlist_empty(&rxr_ep->tx_entry_list));

	fi_close(&ep->fid);
	efa_unit_test_resource_destroy(&resource);
}
