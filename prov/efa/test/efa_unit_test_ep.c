#include "efa_unit_tests.h"
#include "rdma_core_mocks.h"

static
void efa_unit_test_mock_efa_cq_read_entry(struct ibv_cq_ex *ibv_cqx, int index, void *buf)
{
}

static
ssize_t efa_unit_test_mock_eq_write_successful(struct fid_eq *eq, uint32_t event,
					       const void *buf, size_t len, uint64_t flags)
{
	check_expected(eq);
	return len;
};

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

	ret = efa_unit_test_resource_construct(&resource, FI_EP_RDM);
	assert_int_equal(ret, 0);

	g_efa_fork_status = EFA_FORK_SUPPORT_ON;
	check_ep_pkt_pool_flags(resource, OFI_BUFPOOL_NONSHARED);

	g_efa_fork_status = EFA_FORK_SUPPORT_OFF;
	check_ep_pkt_pool_flags(resource, OFI_BUFPOOL_HUGEPAGES);

	g_efa_fork_status = EFA_FORK_SUPPORT_UNNEEDED;
	check_ep_pkt_pool_flags(resource, OFI_BUFPOOL_HUGEPAGES);

	efa_unit_test_resource_destruct(&resource);
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

	ret = efa_unit_test_resource_construct(&resource, FI_EP_RDM);
	assert_int_equal(ret, 0);

	/* Turn on g_efa_fork_status and open a new rxr endpoint */
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
	efa_unit_test_resource_destruct(&resource);

	g_efa_fork_status = EFA_FORK_SUPPORT_OFF;
}

/**
 * @brief when delivery complete atomic was used and handshake packet has not been received
 * verify there is no tx entry leak
 */
void test_rxr_ep_dc_atomic_error_handling()
{
	struct rxr_ep *rxr_ep;
	struct rdm_peer *peer;
	struct fi_ioc ioc = {0};
	struct fi_rma_ioc rma_ioc = {0};
	struct fi_msg_atomic msg = {0};
	struct efa_resource resource = {0};
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t peer_addr;
	int buf[1] = {0}, err, numaddr;

	err = efa_unit_test_resource_construct(&resource, FI_EP_RDM);
	assert_int_equal(err, 0);

	/* create a fake peer */
	err = fi_getname(&resource.ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
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

	rxr_ep = container_of(resource.ep, struct rxr_ep, util_ep.ep_fid);
	rxr_ep->use_shm_for_tx = false;
	/* set peer->flag to RXR_PEER_REQ_SENT will make rxr_atomic() think
	 * a REQ packet has been sent to the peer (so no need to send again)
	 * handshake has not been received, so we do not know whether the peer support DC
	 */
	peer = rxr_ep_get_peer(rxr_ep, peer_addr);
	peer->flags = RXR_PEER_REQ_SENT;
	peer->is_local = false;

	assert_true(dlist_empty(&rxr_ep->tx_entry_list));
	err = fi_atomicmsg(resource.ep, &msg, FI_DELIVERY_COMPLETE);
	/* DC has been reuquested, but ep do not know whether peer supports it, therefore
	 * -FI_EAGAIN should be returned
	 */
	assert_int_equal(err, -FI_EAGAIN);
	/* make sure there is no leaking of tx_entry */
	assert_true(dlist_empty(&rxr_ep->tx_entry_list));

	efa_unit_test_resource_destruct(&resource);
}

/* Verify that DGRAM endpoint progress works with normal work completions */
void test_dgram_ep_progress_happy()
{
	int err;
	struct efa_ep *efa_ep;
	struct efa_resource resource = {0};

	efa_unit_test_resource_construct(&resource, FI_EP_DGRAM);

	efa_ep = container_of(resource.ep, struct efa_ep, util_ep.ep_fid);
	efa_unit_test_ibv_cq_ex_use_mock(efa_ep->rcq->ibv_cq_ex);
	efa_ep->rcq->read_entry = &efa_unit_test_mock_efa_cq_read_entry;

	/* Read 5 entries from CQ and then ENOENT */
	will_return(efa_unit_test_mock_ibv_start_poll, 0);
	will_return_count(efa_unit_test_mock_ibv_next_poll, 0, 4);
	will_return(efa_unit_test_mock_ibv_next_poll, ENOENT);
	will_return(efa_unit_test_mock_ibv_end_poll, NULL);

	efa_ep->util_ep.progress(&efa_ep->util_ep);

	/* Error entry will be written to efa_ep->rcq->util_cq.aux_queue */
	err = !slist_empty(&efa_ep->rcq->util_cq.aux_queue);

	/* Cleanup resource before assertion */
	efa_unit_test_resource_destruct(&resource);

	assert_int_equal(err, 0);
}

/* Verify that DGRAM endpoint progress works when CQ is empty */
void test_dgram_ep_progress_with_empty_cq()
{
	int err;
	struct efa_ep *efa_ep;
	struct efa_resource resource = {0};

	efa_unit_test_resource_construct(&resource, FI_EP_DGRAM);

	efa_ep = container_of(resource.ep, struct efa_ep, util_ep.ep_fid);
	efa_unit_test_ibv_cq_ex_use_mock(efa_ep->rcq->ibv_cq_ex);
	efa_ep->rcq->read_entry = &efa_unit_test_mock_efa_cq_read_entry;


	/* Read 5 entries from CQ and then ENOENT */
	will_return(efa_unit_test_mock_ibv_start_poll, ENOENT);

	efa_ep->util_ep.progress(&efa_ep->util_ep);

	/* Error entry will be written to efa_ep->rcq->util_cq.aux_queue */
	err = !slist_empty(&efa_ep->rcq->util_cq.aux_queue);

	/* Cleanup resource before assertion */
	efa_unit_test_resource_destruct(&resource);

	assert_int_equal(err, 0);
}

/* Verify that DGRAM endpoint progress works with bad work completion status */
void test_dgram_ep_progress_encounter_bad_wc_status()
{
	int err;
	struct efa_ep *efa_ep;
	struct efa_resource resource = {0};
	struct fi_cq_err_entry err_entry = {0};

	efa_unit_test_resource_construct(&resource, FI_EP_DGRAM);

	/* Read 1 entry from CQ and encounter a bad wc status */
	will_return(efa_unit_test_mock_ibv_start_poll, 0);
	will_return(efa_unit_test_mock_ibv_end_poll, NULL);
	will_return(efa_unit_test_mock_ibv_read_opcode, IBV_WC_SEND);

	efa_ep = container_of(resource.ep, struct efa_ep, util_ep.ep_fid);
	efa_unit_test_ibv_cq_ex_use_mock(efa_ep->rcq->ibv_cq_ex);
	efa_ep->rcq->read_entry = &efa_unit_test_mock_efa_cq_read_entry;


	efa_ep->rcq->ibv_cq_ex->status = 1;

	efa_ep->util_ep.progress(&efa_ep->util_ep);

	err = ofi_cq_readerr(&efa_ep->rcq->util_cq.cq_fid, &err_entry, 0) != 1;

	/* Cleanup resource before assertion */
	efa_unit_test_resource_destruct(&resource);

	assert_int_equal(err, 0);
	assert_int_equal(err_entry.err, EIO);
	assert_int_equal(err_entry.prov_errno, 1);
}

/* Verify that RDM endpoint progress works with normal send work completions */
void test_rdm_ep_progress_send_completion_happy()
{
	struct efa_cq *efa_cq;
	struct rxr_ep *rxr_ep;
	struct efa_resource resource = {0};
	struct rxr_pkt_entry entry = {0};

	efa_unit_test_resource_construct(&resource, FI_EP_RDM);

	rxr_ep = container_of(resource.ep, struct rxr_ep, util_ep.ep_fid);
	efa_cq = container_of(rxr_ep->rdm_cq, struct efa_cq, util_cq.cq_fid);
	efa_unit_test_ibv_cq_ex_use_mock(efa_cq->ibv_cq_ex);
	efa_cq->read_entry = &efa_unit_test_mock_efa_cq_read_entry;

	efa_cq->ibv_cq_ex->wr_id = (uint64_t)&entry;

	/* Read 5 entries from CQ and then ENOENT */
	will_return(efa_unit_test_mock_ibv_start_poll, 0);
	will_return_count(efa_unit_test_mock_ibv_next_poll, 0, 4);
	will_return(efa_unit_test_mock_ibv_next_poll, ENOENT);
	will_return(efa_unit_test_mock_ibv_end_poll, NULL);
	will_return_maybe(efa_unit_test_mock_ibv_read_opcode, IBV_WC_SEND);
	will_return_count(__wrap_rxr_pkt_handle_send_completion, NULL, 5);
	expect_value_count(__wrap_rxr_pkt_handle_send_completion, ep, rxr_ep, 5);
	expect_value_count(__wrap_rxr_pkt_handle_send_completion, pkt_entry, &entry, 5);

	rxr_ep->util_ep.progress(&rxr_ep->util_ep);

	efa_unit_test_resource_destruct(&resource);
}

/* Verify that RDM endpoint progress works with normal receive completions */
void test_rdm_ep_progress_recv_completion_happy()
{
	struct efa_cq *efa_cq;
	struct rxr_ep *rxr_ep;
	struct efa_resource resource = {0};
	struct rxr_pkt_entry entry = {0};

	efa_unit_test_resource_construct(&resource, FI_EP_RDM);

	rxr_ep = container_of(resource.ep, struct rxr_ep, util_ep.ep_fid);
	efa_cq = container_of(rxr_ep->rdm_cq, struct efa_cq, util_cq.cq_fid);
	efa_unit_test_ibv_cq_ex_use_mock(efa_cq->ibv_cq_ex);
	efa_cq->read_entry = &efa_unit_test_mock_efa_cq_read_entry;

	efa_cq->ibv_cq_ex->wr_id = (uint64_t)&entry;

	/* Read 5 entries from CQ and then ENOENT */
	will_return(efa_unit_test_mock_ibv_start_poll, 0);
	will_return_count(efa_unit_test_mock_ibv_next_poll, 0, 4);
	will_return(efa_unit_test_mock_ibv_next_poll, ENOENT);
	will_return(efa_unit_test_mock_ibv_end_poll, NULL);
	will_return_maybe(efa_unit_test_mock_ibv_read_opcode, IBV_WC_RECV);
	will_return_count(__wrap_rxr_pkt_handle_recv_completion, NULL, 5);
	expect_value_count(__wrap_rxr_pkt_handle_recv_completion, ep, rxr_ep, 5);
	expect_value_count(__wrap_rxr_pkt_handle_recv_completion, pkt_entry, &entry, 5);
	expect_value_count(__wrap_rxr_pkt_handle_recv_completion, lower_ep_type, EFA_EP, 5);

	rxr_ep->util_ep.progress(&rxr_ep->util_ep);

	efa_unit_test_resource_destruct(&resource);
}

/* Verify that RDM endpoint progress works with empty CQ */
void test_rdm_ep_progress_send_empty_cq()
{
	struct efa_cq *efa_cq;
	struct rxr_ep *rxr_ep;
	struct efa_resource resource = {0};
	struct rxr_pkt_entry entry = {0};

	efa_unit_test_resource_construct(&resource, FI_EP_RDM);

	rxr_ep = container_of(resource.ep, struct rxr_ep, util_ep.ep_fid);
	efa_cq = container_of(rxr_ep->rdm_cq, struct efa_cq, util_cq.cq_fid);
	efa_unit_test_ibv_cq_ex_use_mock(efa_cq->ibv_cq_ex);
	efa_cq->read_entry = &efa_unit_test_mock_efa_cq_read_entry;

	efa_cq->ibv_cq_ex->wr_id = (uint64_t)&entry;

	/* Empty CQ */
	will_return(efa_unit_test_mock_ibv_start_poll, ENOENT);

	rxr_ep->util_ep.progress(&rxr_ep->util_ep);

	efa_unit_test_resource_destruct(&resource);
}

/* Verify that RDM endpoint progress works when cq poll returns an unexpected error */
void test_rdm_ep_progress_failed_poll()
{
	int err;
	struct efa_cq *efa_cq;
	struct rxr_ep *rxr_ep;
	struct efa_resource resource = {0};
	struct fi_eq_attr eq_attr = {0};
	struct rxr_pkt_entry entry = {0};
	struct fi_ops_eq fi_ops_eq = {0};

	resource.eq_attr = &eq_attr;
	efa_unit_test_resource_construct(&resource, FI_EP_RDM);

	rxr_ep = container_of(resource.ep, struct rxr_ep, util_ep.ep_fid);
	efa_cq = container_of(rxr_ep->rdm_cq, struct efa_cq, util_cq.cq_fid);

	efa_cq->ibv_cq_ex->wr_id = (uint64_t)&entry;
	efa_unit_test_ibv_cq_ex_use_mock(efa_cq->ibv_cq_ex);
	efa_cq->read_entry = &efa_unit_test_mock_efa_cq_read_entry;

	efa_cq->ibv_cq_ex->status = 0;

	/* Bind EQ to the RDM endpoint to write error to */
	fi_ops_eq.write = &efa_unit_test_mock_eq_write_successful;
	resource.eq->ops = &fi_ops_eq;

	err = fi_ep_bind(&rxr_ep->util_ep.ep_fid, &resource.eq->fid, 0);
	assert_int_equal(err, 0);

	/* Read 5 entries from CQ and then EFAULT */
	will_return(efa_unit_test_mock_ibv_start_poll, 0);
	will_return_count(efa_unit_test_mock_ibv_next_poll, 0, 4);
	will_return(efa_unit_test_mock_ibv_next_poll, EFAULT);
	will_return(efa_unit_test_mock_ibv_end_poll, NULL);
	will_return_maybe(efa_unit_test_mock_ibv_read_opcode, IBV_WC_SEND);
	will_return_count(__wrap_rxr_pkt_handle_send_completion, NULL, 5);
	expect_value_count(__wrap_rxr_pkt_handle_send_completion, ep, rxr_ep, 5);
	expect_value_count(__wrap_rxr_pkt_handle_send_completion, pkt_entry, &entry, 5);
	expect_value(efa_unit_test_mock_eq_write_successful, eq, resource.eq);

	rxr_ep->util_ep.progress(&rxr_ep->util_ep);

	efa_unit_test_resource_destruct(&resource);
}

/* Verify that RDM endpoint progress works with with bad send wc status */
void test_rdm_ep_progress_bad_send_wc_status()
{
	struct efa_cq *efa_cq;
	struct rxr_ep *rxr_ep;
	struct efa_resource resource = {0};
	struct rxr_pkt_entry entry = {0};

	efa_unit_test_resource_construct(&resource, FI_EP_RDM);

	rxr_ep = container_of(resource.ep, struct rxr_ep, util_ep.ep_fid);
	efa_cq = container_of(rxr_ep->rdm_cq, struct efa_cq, util_cq.cq_fid);
	efa_unit_test_ibv_cq_ex_use_mock(efa_cq->ibv_cq_ex);
	efa_cq->read_entry = &efa_unit_test_mock_efa_cq_read_entry;

	efa_cq->ibv_cq_ex->wr_id = (uint64_t)&entry;
	efa_cq->ibv_cq_ex->status = 1;

	/* Read 1 entries with bad wc status */
	will_return(efa_unit_test_mock_ibv_start_poll, 0);
	will_return(efa_unit_test_mock_ibv_end_poll, NULL);
	will_return_maybe(efa_unit_test_mock_ibv_read_opcode, IBV_WC_SEND);
	will_return(__wrap_rxr_pkt_handle_send_error, NULL);
	expect_value(__wrap_rxr_pkt_handle_send_error, ep, rxr_ep);
	expect_value(__wrap_rxr_pkt_handle_send_error, pkt_entry, &entry);
	expect_any(__wrap_rxr_pkt_handle_send_error, err);
	expect_any(__wrap_rxr_pkt_handle_send_error, prov_errno);

	rxr_ep->util_ep.progress(&rxr_ep->util_ep);

	efa_unit_test_resource_destruct(&resource);
}

/* Verify that RDM endpoint progress works with with bad receive wc status */
void test_rdm_ep_progress_bad_recv_wc_status()
{
	struct efa_cq *efa_cq;
	struct rxr_ep *rxr_ep;
	struct efa_resource resource = {0};
	struct rxr_pkt_entry entry = {0};

	efa_unit_test_resource_construct(&resource, FI_EP_RDM);

	rxr_ep = container_of(resource.ep, struct rxr_ep, util_ep.ep_fid);
	efa_cq = container_of(rxr_ep->rdm_cq, struct efa_cq, util_cq.cq_fid);
	efa_unit_test_ibv_cq_ex_use_mock(efa_cq->ibv_cq_ex);
	efa_cq->read_entry = &efa_unit_test_mock_efa_cq_read_entry;

	efa_cq->ibv_cq_ex->wr_id = (uint64_t)&entry;
	efa_cq->ibv_cq_ex->status = 1;

	/* Read 1 entries with bad wc status */
	will_return(efa_unit_test_mock_ibv_start_poll, 0);
	will_return(efa_unit_test_mock_ibv_end_poll, NULL);
	will_return_maybe(efa_unit_test_mock_ibv_read_opcode, IBV_WC_RECV);
	will_return(__wrap_rxr_pkt_handle_recv_error, NULL);
	expect_value(__wrap_rxr_pkt_handle_recv_error, ep, rxr_ep);
	expect_value(__wrap_rxr_pkt_handle_recv_error, pkt_entry, &entry);
	expect_any(__wrap_rxr_pkt_handle_recv_error, err);
	expect_any(__wrap_rxr_pkt_handle_recv_error, prov_errno);

	rxr_ep->util_ep.progress(&rxr_ep->util_ep);

	efa_unit_test_resource_destruct(&resource);
}
