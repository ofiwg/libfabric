#include "efa_unit_tests.h"

static void check_ep_pkt_pool_flags(struct efa_resource *resource, int expected_flags)
{
       struct fid_ep *ep;
       struct rxr_ep *rxr_ep;
       int ret;
       ret = fi_endpoint(resource->domain, resource->info, &ep, NULL);
       assert_int_equal(ret, 0);
       rxr_ep = container_of(ep, struct rxr_ep, util_ep.ep_fid);
       assert_int_equal(rxr_ep->efa_tx_pkt_pool->flags, expected_flags);
       assert_int_equal(rxr_ep->efa_rx_pkt_pool->flags, expected_flags);
       fi_close(&ep->fid);
}

/**
 * @brief Test the pkt pool flags in rxr_ep_init()
 * 
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_rxr_ep_pkt_pool_flags(struct efa_resource **state)
{
	int ret;
	struct efa_resource *resource = *state;

	ret = efa_unit_test_resource_construct(resource, FI_EP_RDM);
	assert_int_equal(ret, 0);

	g_efa_fork_status = EFA_FORK_SUPPORT_ON;
	check_ep_pkt_pool_flags(resource, OFI_BUFPOOL_NONSHARED);

	g_efa_fork_status = EFA_FORK_SUPPORT_OFF;
	check_ep_pkt_pool_flags(resource, OFI_BUFPOOL_HUGEPAGES);

	g_efa_fork_status = EFA_FORK_SUPPORT_UNNEEDED;
	check_ep_pkt_pool_flags(resource, OFI_BUFPOOL_HUGEPAGES);
}

/**
 * @brief When the buf pool is created with OFI_BUFPOOL_NONSHARED,
 * test if the allocated memory is page aligned.
 * 
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_rxr_ep_pkt_pool_page_alignment(struct efa_resource **state)
{
	int ret;
	struct ofi_bufpool_region *buf;
	struct rxr_pkt_entry *pkt_entry;
	struct fid_ep *ep;
	struct rxr_ep *rxr_ep;
	struct efa_resource *resource = *state;

	ret = efa_unit_test_resource_construct(resource, FI_EP_RDM);
	assert_int_equal(ret, 0);

	/* Turn on g_efa_fork_status and open a new rxr endpoint */
	g_efa_fork_status = EFA_FORK_SUPPORT_ON;
	ret = fi_endpoint(resource->domain, resource->info, &ep, NULL);
	assert_int_equal(ret, 0);
	rxr_ep = container_of(ep, struct rxr_ep, util_ep.ep_fid);
	assert_int_equal(rxr_ep->efa_rx_pkt_pool->flags, OFI_BUFPOOL_NONSHARED);

	pkt_entry = rxr_pkt_entry_alloc(rxr_ep, rxr_ep->efa_rx_pkt_pool, RXR_PKT_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	assert_true(((uintptr_t)ofi_buf_region(pkt_entry->wiredata)->alloc_region % ofi_get_page_size()) == 0);
	rxr_pkt_entry_release(pkt_entry);

	fi_close(&ep->fid);

	g_efa_fork_status = EFA_FORK_SUPPORT_OFF;
}

/**
 * @brief when delivery complete atomic was used and handshake packet has not been received
 * verify there is no tx entry leak
 * 
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_rxr_ep_dc_atomic_error_handling(struct efa_resource **state)
{
	struct rxr_ep *rxr_ep;
	struct rdm_peer *peer;
	struct fi_ioc ioc = {0};
	struct fi_rma_ioc rma_ioc = {0};
	struct fi_msg_atomic msg = {0};
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t peer_addr;
	int buf[1] = {0}, err, numaddr;

	err = efa_unit_test_resource_construct(resource, FI_EP_RDM);
	assert_int_equal(err, 0);

	/* create a fake peer */
	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	numaddr = fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL);
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

	rxr_ep = container_of(resource->ep, struct rxr_ep, util_ep.ep_fid);
	rxr_ep->use_shm_for_tx = false;
	/* set peer->flag to RXR_PEER_REQ_SENT will make rxr_atomic() think
	 * a REQ packet has been sent to the peer (so no need to send again)
	 * handshake has not been received, so we do not know whether the peer support DC
	 */
	peer = rxr_ep_get_peer(rxr_ep, peer_addr);
	peer->flags = RXR_PEER_REQ_SENT;
	peer->is_local = false;

	assert_true(dlist_empty(&rxr_ep->tx_entry_list));
	err = fi_atomicmsg(resource->ep, &msg, FI_DELIVERY_COMPLETE);
	/* DC has been reuquested, but ep do not know whether peer supports it, therefore
	 * -FI_EAGAIN should be returned
	 */
	assert_int_equal(err, -FI_EAGAIN);
	/* make sure there is no leaking of tx_entry */
	assert_true(dlist_empty(&rxr_ep->tx_entry_list));
}
