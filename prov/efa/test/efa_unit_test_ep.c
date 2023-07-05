#include "efa_unit_tests.h"

/**
 * @brief Verify the EFA RDM endpoint correctly parses the host id string
 * @param[in]	state		cmocka state variable
 * @param[in]	file_exists	Toggle whether the host id file exists
 * @param[in]	raw_id		The host id string that is written in the host id file.
 * @param[in]	expect_id	Expected parsed host id integer
 */
void test_efa_rdm_ep_host_id(struct efa_resource **state, bool file_exists, char *raw_id, uint64_t expect_id)
{
	int fd = -1;
	ssize_t written_len;
	char host_id_file[] = "XXXXXXXXXX";
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;

	efa_env.host_id_file = NULL;

	if (file_exists) {
		fd = mkstemp(host_id_file);
		if (fd < 0) {
			fail();
		}

		written_len = write(fd, raw_id, strlen(raw_id));
		if (written_len != strlen(raw_id)) {
			close(fd);
			fail();
		}

		efa_env.host_id_file = host_id_file;
	}

	efa_unit_test_resource_construct(resource, FI_EP_RDM);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Remove the temporary file */
	if (efa_env.host_id_file) {
		unlink(efa_env.host_id_file);
		close(fd);
	}

	assert_int_equal(efa_rdm_ep->host_id, expect_id);
}

/**
 * @brief Verify the EFA RDM endpoint ignores non-existent host id file
 */
void test_efa_rdm_ep_ignore_missing_host_id_file(struct efa_resource **state)
{
	test_efa_rdm_ep_host_id(state, false, NULL, 0);
}

/**
 * @brief Verify the EFA RDM endpoint correctly parses a valid host id string
 */
void test_efa_rdm_ep_has_valid_host_id(struct efa_resource **state)
{
	test_efa_rdm_ep_host_id(state, true, "i-01234567812345678", 0x1234567812345678);
}

/**
 * @brief Verify the EFA RDM endpoint ignores a short (<16 char) host id string
 */
void test_efa_rdm_ep_ignore_short_host_id(struct efa_resource **state)
{
	test_efa_rdm_ep_host_id(state, true, "i-012345678", 0);
}

/**
 * @brief Verify the EFA RDM endpoint ignores a malformatted host id string
 */
void test_efa_rdm_ep_ignore_non_hex_host_id(struct efa_resource **state)
{
	test_efa_rdm_ep_host_id(state, true, "i-0abcdefghabcdefgh", 0);
}

#if HAVE_EFADV_CQ_EX
/**
 * @brief Verify the EFA RDM endpoint correctly processes and responds to a handshake packet
 *	Upon receiving a handshake packet from a new remote peer, the endpoint should inspect
 *	the packet header and set the peer host id if HOST_ID_HDR is turned on.
 *	Then the endpoint should respond with a handshake packet, and include the local host id
 *	if and only if it is non-zero.
 * 
 * @param[in]	state		cmocka state variable
 * @param[in]	local_host_id	The local host id
 * @param[in]	peer_host_id	The remote peer host id
 * @param[in]	include_connid	Toggle whether connid should be included in handshake packet
 */
void test_efa_rdm_ep_handshake_exchange_host_id(struct efa_resource **state, uint64_t local_host_id, uint64_t peer_host_id, bool include_connid)
{
	fi_addr_t peer_addr = 0;
	int cq_read_recv_ret, cq_read_send_ret;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_rdm_peer *peer;
	struct efa_resource *resource = *state;
	struct efa_unit_test_handshake_pkt_attr pkt_attr = {0};
	struct fi_cq_data_entry cq_entry;
	struct ibv_qp_ex *ibv_qp;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pkt_entry;
	uint64_t actual_peer_host_id = UINT64_MAX;

	g_efa_unit_test_mocks.local_host_id = local_host_id;
	g_efa_unit_test_mocks.peer_host_id = peer_host_id;

	assert_false(actual_peer_host_id == g_efa_unit_test_mocks.peer_host_id);

	efa_unit_test_resource_construct(resource, FI_EP_RDM);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_ep->host_id = g_efa_unit_test_mocks.local_host_id;
	efa_rdm_ep->use_shm_for_tx = false;

	/* Create and register a fake peer */
	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 0;
	raw_addr.qkey = 0x1234;

	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL), 1);

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	assert_non_null(peer);
	/* Peer host id is uninitialized before handshake */
	assert_int_equal(peer->host_id, 0);
	assert_false(peer->flags && EFA_RDM_PEER_HANDSHAKE_SENT);

	/* Setup rx packet entry. Manually increase counter to avoid underflow */
	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool, EFA_RDM_PKE_FROM_EFA_RX_POOL);
	efa_rdm_ep->efa_rx_pkts_posted++;

	pkt_attr.connid = include_connid ? raw_addr.qkey : 0;
	pkt_attr.host_id = g_efa_unit_test_mocks.peer_host_id;
	efa_unit_test_handshake_pkt_construct(pkt_entry, &pkt_attr);

	ibv_qp = efa_rdm_ep->base_ep.qp->ibv_qp_ex;
	ibv_qp->wr_start = &efa_mock_ibv_wr_start_no_op;
	/* this mock will save the send work request (wr) in a global array */
	ibv_qp->wr_send = &efa_mock_ibv_wr_send_verify_handshake_pkt_local_host_id_and_save_wr;
	ibv_qp->wr_set_inline_data_list = &efa_mock_ibv_wr_set_inline_data_list_no_op;
	ibv_qp->wr_set_ud_addr = &efa_mock_ibv_wr_set_ud_addr_no_op;
	ibv_qp->wr_complete = &efa_mock_ibv_wr_complete_no_op;
	expect_function_call(efa_mock_ibv_wr_send_verify_handshake_pkt_local_host_id_and_save_wr);

	/* Setup CQ */
	efa_rdm_ep->ibv_cq_ex->end_poll = &efa_mock_ibv_end_poll_check_mock;
	efa_rdm_ep->ibv_cq_ex->next_poll = &efa_mock_ibv_next_poll_check_function_called_and_return_mock;
	efa_rdm_ep->ibv_cq_ex->read_byte_len = &efa_mock_ibv_read_byte_len_return_mock;
	efa_rdm_ep->ibv_cq_ex->read_opcode = &efa_mock_ibv_read_opcode_return_mock;
	efa_rdm_ep->ibv_cq_ex->read_slid = &efa_mock_ibv_read_slid_return_mock;
	efa_rdm_ep->ibv_cq_ex->read_src_qp = &efa_mock_ibv_read_src_qp_return_mock;
	efa_rdm_ep->ibv_cq_ex->read_vendor_err = &efa_mock_ibv_read_vendor_err_return_mock;
	efa_rdm_ep->ibv_cq_ex->start_poll = &efa_mock_ibv_start_poll_return_mock;
	efa_rdm_ep->ibv_cq_ex->status = IBV_WC_SUCCESS;
	efa_rdm_ep->ibv_cq_ex->wr_id = (uintptr_t)pkt_entry;
	expect_function_call(efa_mock_ibv_next_poll_check_function_called_and_return_mock);

	/* Receive handshake packet */
	will_return(efa_mock_ibv_end_poll_check_mock, NULL);
	will_return(efa_mock_ibv_next_poll_check_function_called_and_return_mock, ENOENT);
	will_return(efa_mock_ibv_read_byte_len_return_mock, pkt_entry->pkt_size);
	will_return(efa_mock_ibv_read_opcode_return_mock, IBV_WC_RECV);
	will_return(efa_mock_ibv_read_slid_return_mock, efa_rdm_ep_get_peer_ahn(efa_rdm_ep, peer_addr));
	will_return(efa_mock_ibv_read_src_qp_return_mock, raw_addr.qpn);
	will_return(efa_mock_ibv_start_poll_return_mock, IBV_WC_SUCCESS);

	/**
	 * Fire away handshake packet.
	 * Because we don't care if it fails(there is no receiver!), mark it as failed to make mocking simpler.
	 */
	will_return(efa_mock_ibv_end_poll_check_mock, NULL);
	will_return(efa_mock_ibv_read_opcode_return_mock, IBV_WC_SEND);
	will_return(efa_mock_ibv_read_vendor_err_return_mock, FI_EFA_ERR_OTHER);
	will_return(efa_mock_ibv_start_poll_return_mock, IBV_WC_SUCCESS);

	/* Progress the recv wr first to process the received handshake packet. */
	cq_read_recv_ret = fi_cq_read(resource->cq, &cq_entry, 1);

	actual_peer_host_id = peer->host_id;

	/**
	 * We need to poll the CQ twice explicitly to point the CQE
	 * to the saved send wr in handshake
	 */
	efa_rdm_ep->ibv_cq_ex->status = IBV_WC_GENERAL_ERR;
	efa_rdm_ep->ibv_cq_ex->wr_id = (uintptr_t)g_ibv_submitted_wr_id_vec[0];

	/* Progress the send wr to clean up outstanding tx ops */
	cq_read_send_ret = fi_cq_read(resource->cq, &cq_entry, 1);

	/* HANDSHAKE packet does not generate completion entry */
	assert_int_equal(cq_read_recv_ret, -FI_EAGAIN);
	assert_int_equal(cq_read_send_ret, -FI_EAGAIN);

	/* Peer host id is set after handshake */
	assert_true(actual_peer_host_id == g_efa_unit_test_mocks.peer_host_id);
}
#else
void test_efa_rdm_ep_handshake_exchange_host_id() {
	skip();
}
#endif

void test_efa_rdm_ep_handshake_receive_and_send_valid_host_ids_with_connid(struct efa_resource **state)
{
	test_efa_rdm_ep_handshake_exchange_host_id(state, 0x1234567812345678, 0x8765432187654321, true);
}

void test_efa_rdm_ep_handshake_receive_and_send_valid_host_ids_without_connid(struct efa_resource **state)
{
	test_efa_rdm_ep_handshake_exchange_host_id(state, 0x1234567812345678, 0x8765432187654321, false);
}

void test_efa_rdm_ep_handshake_receive_valid_peer_host_id_and_do_not_send_local_host_id(struct efa_resource **state)
{
	test_efa_rdm_ep_handshake_exchange_host_id(state, 0x0, 0x8765432187654321, true);
}

void test_efa_rdm_ep_handshake_receive_without_peer_host_id_and_do_not_send_local_host_id(struct efa_resource **state)
{
	test_efa_rdm_ep_handshake_exchange_host_id(state, 0x0, 0x0, true);
}

/**
 * @brief Test efa_rdm_ep_open() handles rdma-core CQ creation failure gracefully
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_rdm_ep_cq_create_error_handling(struct efa_resource **state)
{

	struct efa_resource *resource = *state;
	struct ibv_device **ibv_device_list;
	struct efa_device efa_device = {0};
	struct efa_domain *efa_domain = NULL;
	struct verbs_context *vctx = NULL;

	ibv_device_list = ibv_get_device_list(&g_device_cnt);
	if (ibv_device_list == NULL) {
		skip();
		return;
	}
	efa_device_construct(&efa_device, 0, ibv_device_list[0]);

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM);
	assert_non_null(resource->hints);
	assert_int_equal(fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, resource->hints, &resource->info), 0);
	assert_int_equal(fi_fabric(resource->info->fabric_attr, &resource->fabric, NULL), 0);
	assert_int_equal(fi_domain(resource->fabric, resource->info, &resource->domain, NULL), 0);

	vctx = verbs_get_ctx_op(efa_device.ibv_ctx, create_cq_ex);
#if HAVE_EFADV_CQ_EX
	g_efa_unit_test_mocks.efadv_create_cq = &efa_mock_efadv_create_cq_set_eopnotsupp_and_return_null;
	expect_function_call(efa_mock_efadv_create_cq_set_eopnotsupp_and_return_null);
#endif
	/* Mock out the create_cq_ex function pointer which is called by ibv_create_cq_ex */
	vctx->create_cq_ex = &efa_mock_create_cq_ex_return_null;
	expect_function_call(efa_mock_create_cq_ex_return_null);

	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);
	efa_domain->device = &efa_device;

	assert_int_not_equal(fi_endpoint(resource->domain, resource->info, &resource->ep, NULL), 0);
}

static void check_ep_pkt_pool_flags(struct fid_ep *ep, int expected_flags)
{
       struct efa_rdm_ep *efa_rdm_ep;

       efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
       assert_int_equal(efa_rdm_ep->efa_tx_pkt_pool->attr.flags, expected_flags);
       assert_int_equal(efa_rdm_ep->efa_rx_pkt_pool->attr.flags, expected_flags);
}

/**
 * @brief Test the pkt pool flags in efa_rdm_ep
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_rdm_ep_pkt_pool_flags(struct efa_resource **state) {
	struct efa_resource *resource = *state;

	efa_env.huge_page_setting = EFA_ENV_HUGE_PAGE_DISABLED;
	efa_unit_test_resource_construct(resource, FI_EP_RDM);
	check_ep_pkt_pool_flags(resource->ep, OFI_BUFPOOL_NONSHARED);
}

/**
 * @brief When the buf pool is created with OFI_BUFPOOL_NONSHARED,
 * test if the allocated memory is page aligned.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_rdm_ep_pkt_pool_page_alignment(struct efa_resource **state)
{
	int ret;
	struct efa_rdm_pke *pkt_entry;
	struct fid_ep *ep;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_resource *resource = *state;

	efa_unit_test_resource_construct(resource, FI_EP_RDM);

	efa_env.huge_page_setting = EFA_ENV_HUGE_PAGE_DISABLED;
	ret = fi_endpoint(resource->domain, resource->info, &ep, NULL);
	assert_int_equal(ret, 0);
	efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	assert_int_equal(efa_rdm_ep->efa_rx_pkt_pool->attr.flags, OFI_BUFPOOL_NONSHARED);

	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool, EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	assert_true(((uintptr_t)ofi_buf_region(pkt_entry)->alloc_region % ofi_get_page_size()) == 0);
	efa_rdm_pke_release_rx(pkt_entry);

	fi_close(&ep->fid);
}



/**
 * @brief when delivery complete atomic was used and handshake packet has not been received
 * verify there is no txe leak
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_rdm_ep_dc_atomic_error_handling(struct efa_resource **state)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct fi_ioc ioc = {0};
	struct fi_rma_ioc rma_ioc = {0};
	struct fi_msg_atomic msg = {0};
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t peer_addr;
	int buf[1] = {0}, err, numaddr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM);

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

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_ep->use_shm_for_tx = false;
	/* set peer->flag to EFA_RDM_PEER_REQ_SENT will make efa_rdm_atomic() think
	 * a REQ packet has been sent to the peer (so no need to send again)
	 * handshake has not been received, so we do not know whether the peer support DC
	 */
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	peer->flags = EFA_RDM_PEER_REQ_SENT;
	peer->is_local = false;

	assert_true(dlist_empty(&efa_rdm_ep->txe_list));
	err = fi_atomicmsg(resource->ep, &msg, FI_DELIVERY_COMPLETE);
	/* DC has been reuquested, but ep do not know whether peer supports it, therefore
	 * -FI_EAGAIN should be returned
	 */
	assert_int_equal(err, -FI_EAGAIN);
	/* make sure there is no leaking of txe */
	assert_true(dlist_empty(&efa_rdm_ep->txe_list));
}

/**
 * @brief verify that when shm was used to send a small message (<4k), no copy was performed.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_rdm_ep_send_with_shm_no_copy(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t peer_addr;
	int num_addr;
	int buff_len = 8;
	char buff[8] = {0};
	int err;

	efa_unit_test_resource_construct(resource, FI_EP_RDM);

	/* create a fake peer */
	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL);
	assert_int_equal(num_addr, 1);

	g_ofi_copy_from_hmem_iov_call_counter = 0;
	g_efa_unit_test_mocks.ofi_copy_from_hmem_iov = efa_mock_ofi_copy_from_hmem_iov_inc_counter;

	err = fi_send(resource->ep, buff, buff_len,
		      NULL /* desc, which is not required by shm */,
		      peer_addr,
		      NULL /* context */);

	assert_int_equal(g_ofi_copy_from_hmem_iov_call_counter, 0);
}

/**
 * @brief verify error is generated for RMA on non-RMA-enabled EP.
 *
 * @param[in] state	struct efa_resource that is managed by the framework
 */
void test_efa_rdm_ep_rma_without_caps(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t peer_addr;
	int num_addr;
	const int buf_len = 8;
	char buf[8] = {0};
	int err;
	uint64_t rma_addr, rma_key;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM);
	resource->hints->caps |= FI_MSG | FI_TAGGED;
	resource->hints->caps &= ~FI_RMA;
	resource->hints->domain_attr->mr_mode = FI_MR_BASIC;
	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM, resource->hints);

	/* ensure we don't have RMA capability. */
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	assert_int_equal( efa_rdm_ep->user_info->caps & FI_RMA, 0);

	/* create a fake peer */
	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL);
	assert_int_equal(num_addr, 1);

	/* create a fake rma_key and address.  fi_read should return before
	 * they are needed. */
	rma_key = 0x1234;
	rma_addr = (uint64_t) &buf;
	err = fi_read(resource->ep, buf, buf_len,
		      NULL, /* desc, not required */
		      peer_addr,
		      rma_addr,
		      rma_key,
		      NULL); /* context */

	assert_int_equal(err, -FI_EOPNOTSUPP);
}

/**
 * @brief verify error is generated for Atomic operations on non-Atomic-enabled EP.
 *
 * @param[in] state	struct efa_resource that is managed by the framework
 */
void test_efa_rdm_ep_atomic_without_caps(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t peer_addr;
	int num_addr;
	const int buf_len = 8;
	char buf[8] = {0};
	int err;
	uint64_t rma_addr, rma_key;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM);
	resource->hints->caps |= FI_MSG | FI_TAGGED;
	resource->hints->caps &= ~FI_ATOMIC;
	resource->hints->domain_attr->mr_mode = FI_MR_BASIC;
	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM, resource->hints);

	/* ensure we don't have ATOMIC capability. */
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	assert_int_equal( efa_rdm_ep->user_info->caps & FI_ATOMIC, 0);

	/* create a fake peer */
	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL);
	assert_int_equal(num_addr, 1);

	/* create a fake rma_key and address.  fi_atomic should return before
	 * they are needed. */
	rma_key = 0x1234;
	rma_addr = (uint64_t) &buf;
	err = fi_atomic(resource->ep, buf, buf_len,
			NULL, /* desc, not required */
			peer_addr,
			rma_addr,
			rma_key,
			FI_INT32,
			FI_SUM,
			NULL); /* context */

	assert_int_equal(err, -FI_EOPNOTSUPP);
}
