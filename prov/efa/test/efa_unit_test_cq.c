#include "efa_unit_tests.h"

/**
 * @brief implementation of test cases for fi_cq_read() works with empty device CQ for given endpoint type
 *
 * When CQ is empty, fi_cq_read() should return -FI_EAGAIN.
 *
 * @param[in]		ep_type		endpoint type, can be FI_EP_DGRAM or FI_EP_RDM
 */
static
void test_impl_cq_read_empty_cq(enum fi_ep_type ep_type)
{
	struct ibv_cq_ex *ibv_cqx;
	struct fi_cq_data_entry cq_entry;
	struct efa_resource resource = {0};
	int ret;

	efa_unit_test_resource_construct(&resource, ep_type);

	if (ep_type == FI_EP_DGRAM) {
		struct efa_ep *efa_ep;

		efa_ep = container_of(resource.ep, struct efa_ep, util_ep.ep_fid);
		ibv_cqx = efa_ep->rcq->ibv_cq_ex;
	} else {
		struct rxr_ep *rxr_ep;
		struct efa_cq *efa_cq;

		rxr_ep = container_of(resource.ep, struct rxr_ep, util_ep.ep_fid);
		efa_cq = container_of(rxr_ep->rdm_cq, struct efa_cq, util_cq.cq_fid);
		ibv_cqx = efa_cq->ibv_cq_ex;
	}

	ibv_cqx->start_poll = &efa_mock_ibv_start_poll_return_mock;

	/* ibv_start_poll to return ENOENT means device CQ is empty */
	will_return(efa_mock_ibv_start_poll_return_mock, ENOENT);

	ret = fi_cq_read(resource.cq, &cq_entry, 1);

	assert_int_equal(ret, -FI_EAGAIN);

	efa_unit_test_resource_destruct(&resource);
}

/**
 * @brief verify DGRAM CQ's fi_cq_read() works with empty CQ
 *
 * When CQ is empty, fi_cq_read() should return -FI_EAGAIN.
 */
void test_dgram_cq_read_empty_cq()
{
	test_impl_cq_read_empty_cq(FI_EP_DGRAM);
}

/**
 * @brief verify RDM CQ's fi_cq_read() works with empty CQ
 *
 * When CQ is empty, fi_cq_read() should return -FI_EAGAIN.
 */
void test_rdm_cq_read_empty_cq()
{
	test_impl_cq_read_empty_cq(FI_EP_RDM);
}

/**
 * @brief test RDM CQ's fi_cq_read()/fi_cq_readerr() works properly when rdma-core return bad status for send.
 *
 * When ibv_post_send() operation failed, fi_cq_read() should return -FI_EAVAIL, which means error available.
 * then user should call fi_cq_readerr() to get an error CQ entry that contain error code.
 */
static
void test_cq_read_bad_send_status(enum fi_ep_type ep_type)
{
	struct ibv_qp *ibv_qp;
	struct ibv_cq_ex *ibv_cqx;
	struct efa_ep_addr raw_addr;
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	struct fi_cq_err_entry cq_err_entry;
	struct fi_cq_data_entry cq_entry;
	struct efa_resource resource = {0};
	struct efa_unit_test_buff send_buff;
	fi_addr_t addr;
	int ret, err;
	const char *strerror;
	char err_buf;

	efa_unit_test_resource_construct(&resource, ep_type);
	efa_unit_test_buff_construct(&send_buff, &resource, 4096 /* buff_size */);

	if (ep_type == FI_EP_RDM) {
		struct rxr_ep *rxr_ep;
		struct efa_cq *efa_cq;
		struct efa_ep *efa_ep;

		rxr_ep = container_of(resource.ep, struct rxr_ep, util_ep.ep_fid);

		efa_ep = container_of(rxr_ep->rdm_ep, struct efa_ep, util_ep.ep_fid);
		ibv_qp =  efa_ep->qp->ibv_qp;

		efa_cq = container_of(rxr_ep->rdm_cq, struct efa_cq, util_cq.cq_fid);
		ibv_cqx = efa_cq->ibv_cq_ex;

		/* set use_shm_for_tx to false to force rxr_ep to use efa device to send,
		 * which means call ibv_post_send
		 */
		rxr_ep->use_shm_for_tx = false;
	} else {
		struct efa_ep *efa_ep;

		efa_ep = container_of(resource.ep, struct efa_ep, util_ep.ep_fid);
		ibv_qp =  efa_ep->qp->ibv_qp;
		ibv_cqx = efa_ep->rcq->ibv_cq_ex;
	}

	/* this mock will save the send work request (wr) in a global linked list */
	ibv_qp->context->ops.post_send = &efa_mock_ibv_post_send_save_send_wr;

	/* this mock will set ibv_cq_ex->wr_id to the wr_id of the head of global send_wr,
	 * and set ibv_cq_ex->status to mock value */
	ibv_cqx->start_poll = &efa_mock_ibv_start_poll_use_saved_send_wr_with_mock_status;

	ibv_cqx->end_poll = &efa_mock_ibv_end_poll_check_mock;
	ibv_cqx->read_opcode = &efa_mock_ibv_read_opcode_return_mock;
	ibv_cqx->read_vendor_err = &efa_mock_ibv_read_vendor_err_return_mock;

	will_return(efa_mock_ibv_start_poll_use_saved_send_wr_with_mock_status, IBV_WC_GENERAL_ERR);
	will_return(efa_mock_ibv_end_poll_check_mock, NULL);
	will_return(efa_mock_ibv_read_opcode_return_mock, IBV_WC_SEND);
	will_return(efa_mock_ibv_read_vendor_err_return_mock, FI_EFA_LOCAL_ERROR_UNRESP_REMOTE);

	ret = fi_getname(&resource.ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource.av, &raw_addr, 1, &addr, 0 /* flags */, NULL /* context */);
	assert_int_equal(ret, 1);

	assert_null(g_ibv_send_wr_list.head);
	assert_null(g_ibv_send_wr_list.tail);
	err = fi_send(resource.ep, send_buff.buff, send_buff.size, fi_mr_desc(send_buff.mr), addr, NULL /* context */);
	assert_int_equal(err, 0);
	/* fi_send() called efa_mock_ibv_post_send_save_send_wr(), which saved one send_wr in g_ibv_send_wr_list */
	assert_non_null(g_ibv_send_wr_list.head);
	assert_non_null(g_ibv_send_wr_list.tail);

	ret = fi_cq_read(resource.cq, &cq_entry, 1);
	/* fi_cq_read() called efa_mock_ibv_start_poll_use_saved_send_wr(), which pulled one send_wr from g_ibv_send_wr_list */
	assert_null(g_ibv_send_wr_list.head);
	assert_null(g_ibv_send_wr_list.tail);
	assert_int_equal(ret, -FI_EAVAIL);

	ret = fi_cq_readerr(resource.cq, &cq_err_entry, 0);
	strerror = fi_cq_strerror(resource.cq, cq_err_entry.prov_errno, NULL, &err_buf, 0);
	assert_int_equal(ret, 1);
	assert_int_equal(cq_err_entry.err, FI_EIO);
	assert_int_equal(cq_err_entry.prov_errno, FI_EFA_LOCAL_ERROR_UNRESP_REMOTE);
	assert_string_equal(strerror, "Unresponsive receiver");

	efa_unit_test_buff_destruct(&send_buff);
	efa_unit_test_resource_destruct(&resource);
}

/**
 * @brief test that RDM CQ's fi_cq_read()/fi_cq_readerr() works properly when rdma-core return bad status for send.
 *
 * When ibv_post_send() operation failed, fi_cq_read() should return -FI_EAVAIL, which means error available.
 * then user should call fi_cq_readerr() to get an error CQ entry that contain error code.
 */
void test_rdm_cq_read_bad_send_status()
{
	test_cq_read_bad_send_status(FI_EP_RDM);
}

/**
 * @brief test that DGRAM CQ's fi_cq_read()/fi_cq_readerr() works properly when rdma-core return bad status for send.
 *
 * When ibv_post_send() operation failed, fi_cq_read() should return -FI_EAVAIL, which means error available.
 * then user should call fi_cq_readerr() to get an error CQ entry that contain error code.
 */
void test_dgram_cq_read_bad_wc_status()
{
	test_cq_read_bad_send_status(FI_EP_DGRAM);
}

/**
 * @brief verify that fi_cq_read/fi_cq_readerr works properly when rdma-core return bad status for recv.
 *
 * When an ibv_post_recv() operation failed, no data was received. Therefore libfabric cannot
 * find the corresponding RX operation to write a CQ error. It will write an EQ error instead.
 */
void test_rdm_cq_read_bad_recv_status()
{
	struct efa_cq *efa_cq;
	struct rxr_ep *rxr_ep;
	struct efa_resource resource = {0};
	struct rxr_pkt_entry *pkt_entry;
	struct fi_cq_data_entry cq_entry;
	struct fi_eq_err_entry eq_err_entry;
	int ret;

	efa_unit_test_resource_construct(&resource, FI_EP_RDM);
	rxr_ep = container_of(resource.ep, struct rxr_ep, util_ep.ep_fid);

	pkt_entry = rxr_pkt_entry_alloc(rxr_ep, rxr_ep->efa_rx_pkt_pool, RXR_PKT_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);

	efa_cq = container_of(rxr_ep->rdm_cq, struct efa_cq, util_cq.cq_fid);
	efa_cq->ibv_cq_ex->start_poll = &efa_mock_ibv_start_poll_return_mock;
	efa_cq->ibv_cq_ex->end_poll = &efa_mock_ibv_end_poll_check_mock;
	efa_cq->ibv_cq_ex->read_opcode = &efa_mock_ibv_read_opcode_return_mock;
	efa_cq->ibv_cq_ex->read_vendor_err = &efa_mock_ibv_read_vendor_err_return_mock;

	will_return(efa_mock_ibv_start_poll_return_mock, 0);
	will_return(efa_mock_ibv_end_poll_check_mock, NULL);
	/* efa_mock_ibv_read_opcode_return_mock() will be called once in release mode,
	 * but will be called twice in debug mode. because there is an assertion that called ibv_read_opcode(),
	 * therefore use will_return_always()
	 */
	will_return_always(efa_mock_ibv_read_opcode_return_mock, IBV_WC_RECV);
	will_return(efa_mock_ibv_read_vendor_err_return_mock, FI_EFA_LOCAL_ERROR_UNRESP_REMOTE);
	efa_cq->ibv_cq_ex->wr_id = (uintptr_t)pkt_entry;
	efa_cq->ibv_cq_ex->status = IBV_WC_GENERAL_ERR;
	ret = fi_cq_read(resource.cq, &cq_entry, 1);
	/* TODO:
	 *
	 * Our current behavior is to return -FI_EAGAIN, but it is not right.
	 * We need to fix the behaivor in the provider and update the assertion.
	 */
	assert_int_equal(ret, -FI_EAGAIN);

	ret = fi_eq_readerr(resource.eq, &eq_err_entry, 0);
	assert_int_equal(ret, sizeof(eq_err_entry));
	assert_int_equal(eq_err_entry.err, FI_EIO);
	assert_int_equal(eq_err_entry.prov_errno, FI_EFA_LOCAL_ERROR_UNRESP_REMOTE);

	efa_unit_test_resource_destruct(&resource);
}

/**
 * @brief verify that fi_cq_read/fi_cq_readerr works properly when ibv_start_poll failed.
 *
 * When an ibv_start_poll() failed. Libfabric should write an EQ error.
 */
void test_rdm_cq_read_failed_poll()
{
	struct efa_cq *efa_cq;
	struct rxr_ep *rxr_ep;
	struct efa_resource resource = {0};
	struct fi_cq_data_entry cq_entry;
	struct fi_eq_err_entry eq_err_entry;
	int ret;

	efa_unit_test_resource_construct(&resource, FI_EP_RDM);
	rxr_ep = container_of(resource.ep, struct rxr_ep, util_ep.ep_fid);

	efa_cq = container_of(rxr_ep->rdm_cq, struct efa_cq, util_cq.cq_fid);
	efa_cq->ibv_cq_ex->start_poll = &efa_mock_ibv_start_poll_return_mock;
	efa_cq->ibv_cq_ex->end_poll = &efa_mock_ibv_end_poll_check_mock;
	efa_cq->ibv_cq_ex->read_vendor_err = &efa_mock_ibv_read_vendor_err_return_mock;

	will_return(efa_mock_ibv_start_poll_return_mock, EFAULT);
	will_return(efa_mock_ibv_read_vendor_err_return_mock, FI_EFA_LOCAL_ERROR_UNRESP_REMOTE);

	ret = fi_cq_read(resource.cq, &cq_entry, 1);
	/* TODO:
	 * Our current behavior is to return -FI_EAGAIN, but it is not right.
	 * We need to fix the behaivor in the provider and update the test case.
	 */
	assert_int_equal(ret, -FI_EAGAIN);

	ret = fi_eq_readerr(resource.eq, &eq_err_entry, 0);
	assert_int_equal(ret, sizeof(eq_err_entry));
	assert_int_not_equal(eq_err_entry.err, FI_ENOENT);
	assert_int_equal(eq_err_entry.prov_errno, FI_EFA_LOCAL_ERROR_UNRESP_REMOTE);

	efa_unit_test_resource_destruct(&resource);
}

#if HAVE_EFADV_CQ_EX
/**
 * @brief Construct an RDM endpoint and receive an eager MSG RTM packet.
 * Simulate EFA device by setting peer AH to unknown and make sure the
 * endpoint recovers the peer address iff(if and only if) the peer is
 * inserted to AV.
 *
 * @param remove_peer
 */
static void test_impl_rdm_cq_read_unknow_peer_ah(bool remove_peer)
{
	struct efa_cq *efa_cq;
	struct rxr_ep *rxr_ep;
	struct efa_resource resource = {0};
	struct rxr_pkt_entry *pkt_entry;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct fi_cq_data_entry cq_entry;
	struct efa_unit_test_eager_rtm_pkt_attr pkt_attr = {0};
	struct efadv_cq *efadv_cq;
	struct efa_unit_test_buff recv_buff;
	int ret;

	efa_unit_test_resource_construct(&resource, FI_EP_RDM);

	rxr_ep = container_of(resource.ep, struct rxr_ep, util_ep.ep_fid);
	efa_cq = container_of(rxr_ep->rdm_cq, struct efa_cq, util_cq.cq_fid);

	/* Construct a minimal recv buffer */
	efa_unit_test_buff_construct(&recv_buff, &resource, rxr_ep->min_multi_recv_size);

	/* Create and register a fake peer */
	ret = fi_getname(&resource.ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 0;
	raw_addr.qkey = 0x1234;

	struct rdm_peer *peer;
	ret = fi_av_insert(resource.av, &raw_addr, 1, &peer_addr, 0, NULL);
	assert_int_equal(ret, 1);

	/* Skip handshake */
	peer = rxr_ep_get_peer(rxr_ep, peer_addr);
	assert_non_null(peer);
	peer->flags |= RXR_PEER_HANDSHAKE_SENT;

	/* Setup packet entry */
	pkt_entry = rxr_pkt_entry_alloc(rxr_ep, rxr_ep->efa_rx_pkt_pool, RXR_PKT_FROM_EFA_RX_POOL);
	pkt_attr.msg_id = 0;
	pkt_attr.connid = raw_addr.qkey;
	/* Packet type must be in [RXR_REQ_PKT_BEGIN, RXR_EXTRA_REQ_PKT_END) */
	efa_unit_test_eager_msgrtm_pkt_construct(pkt_entry, &pkt_attr);

	/* Setup CQ */
	efa_cq->ibv_cq_ex->wr_id = (uintptr_t)pkt_entry;
	efa_cq->ibv_cq_ex->start_poll = &efa_mock_ibv_start_poll_return_mock;
	efa_cq->ibv_cq_ex->next_poll = &efa_mock_ibv_next_poll_check_function_called_and_return_mock;
	efa_cq->ibv_cq_ex->end_poll = &efa_mock_ibv_end_poll_check_mock;
	efa_cq->ibv_cq_ex->read_slid = &efa_mock_ibv_read_slid_return_mock;
	efa_cq->ibv_cq_ex->read_byte_len = &efa_mock_ibv_read_byte_len_return_mock;
	efa_cq->ibv_cq_ex->read_opcode = &efa_mock_ibv_read_opcode_return_mock;
	efa_cq->ibv_cq_ex->read_src_qp = &efa_mock_ibv_read_src_qp_return_mock;

	efadv_cq = efadv_cq_from_ibv_cq_ex(efa_cq->ibv_cq_ex);
	assert_non_null(efadv_cq);
	efadv_cq->wc_read_sgid = &efa_mock_efadv_wc_read_sgid_return_zero_code_and_expect_next_poll_and_set_gid;

	/* Return unknown AH from efadv */
	will_return(efa_mock_efadv_wc_read_sgid_return_zero_code_and_expect_next_poll_and_set_gid, raw_addr.raw);

	/* Read 1 entry with unknown AH */
	will_return(efa_mock_ibv_start_poll_return_mock, 0);
	will_return(efa_mock_ibv_next_poll_check_function_called_and_return_mock, ENOENT);
	will_return(efa_mock_ibv_end_poll_check_mock, NULL);
	will_return(efa_mock_ibv_read_slid_return_mock, 0xffff); // slid=0xffff(-1) indicates an unknown AH
	will_return(efa_mock_ibv_read_byte_len_return_mock, pkt_entry->pkt_size);
	will_return_maybe(efa_mock_ibv_read_opcode_return_mock, IBV_WC_RECV);
	will_return_maybe(efa_mock_ibv_read_src_qp_return_mock, raw_addr.qpn);

	/* Post receive buffer */
	ret = fi_recv(resource.ep, recv_buff.buff, recv_buff.size, fi_mr_desc(recv_buff.mr), peer_addr, NULL /* context */);
	assert_int_equal(ret, 0);

	if (remove_peer) {
		ret = fi_av_remove(resource.av, &peer_addr, 1, 0);
		assert_int_equal(ret, 0);
	}

	ret = fi_cq_read(resource.cq, &cq_entry, 1);

	if (remove_peer) {
		/* Ignored WC because the peer is removed */
		assert_int_equal(ret, -FI_EAGAIN);
	}
	else {
		/* Found 1 matching rx entry */
		assert_int_equal(ret, 1);
	}

	efa_unit_test_buff_destruct(&recv_buff);
	efa_unit_test_resource_destruct(&resource);
}

/**
 * @brief Verify that RDM endpoint fi_cq_read recovers unknown peer AH
 * by querying efadv to get raw address.
 * A fake peer is registered in AV. The endpoint receives a packet from it,
 * for which the EFA device returns an unknown AH. The endpoint will retrieve
 * the peer's raw address using efadv verbs, and recover it's AH using
 * Raw:QPN:QKey.
 */
void test_rdm_cq_read_recover_forgotten_peer_ah()
{
	test_impl_rdm_cq_read_unknow_peer_ah(false);
}

/**
 * @brief Verify that RDM endpoint progress engine ignores unknown peer AH
 * if the peer is not registered in AV, e.g. removed.
 * The endpoint receives a packet from an alien peer, which corresponds to
 * an unknown AH. The endpoint attempts to look up the AH for the peer but
 * was rightly unable to, thus ignoring the packet.
 */
void test_rdm_cq_read_ignore_removed_peer()
{
	test_impl_rdm_cq_read_unknow_peer_ah(true);
}
#else
void test_rdm_cq_read_recover_forgotten_peer_ah()
{
	skip();
}
void test_rdm_cq_read_ignore_removed_peer()
{
	skip();
}
#endif
