/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"
#include "rdm/efa_rdm_cq.h"
#include "efa_rdm_pke_utils.h"
#include "efa_data_path_direct_entry.h"

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

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

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
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pkt_entry;
	uint64_t actual_peer_host_id = UINT64_MAX;
	struct efa_rdm_cq *efa_rdm_cq;
	struct efa_ibv_cq *ibv_cq;

	g_efa_unit_test_mocks.local_host_id = local_host_id;
	g_efa_unit_test_mocks.peer_host_id = peer_host_id;

	assert_false(actual_peer_host_id == g_efa_unit_test_mocks.peer_host_id);

	/* disable shm to force using efa device to send */
	efa_unit_test_resource_construct_rdm_shm_disabled(resource);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_cq = container_of(resource->cq, struct efa_rdm_cq, efa_cq.util_cq.cq_fid.fid);
	ibv_cq = &(efa_rdm_cq->efa_cq.ibv_cq);

	efa_rdm_ep->host_id = g_efa_unit_test_mocks.local_host_id;

	/* Create and register a fake peer */
	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 0;
	raw_addr.qkey = 0x1234;

	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL), 1);

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	assert_non_null(peer);
	/* Peer host id is uninitialized before handshake */
	assert_int_equal(peer->host_id, 0);
	assert_int_not_equal(peer->flags & EFA_RDM_PEER_HANDSHAKE_SENT, EFA_RDM_PEER_HANDSHAKE_SENT);

	/*
	 * The rx pkt entry should only be allocated and posted by the progress engine.
	 * However, to mock a receive completion, we have to allocate an rx entry
	 * and modify it out of band. The proess engine grow the rx pool in the first
	 * call and set efa_rdm_ep->efa_rx_pkts_posted as the rx pool size. Here we
	 * follow the progress engine to set the efa_rx_pkts_posted counter manually
	 * TODO: modify the rx pkt as part of the ibv cq poll mock so we don't have to
	 * allocate pkt entry and hack the pkt counters.
	 */
	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool, EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	efa_rdm_ep->efa_rx_pkts_posted = efa_rdm_ep_get_rx_pool_size(efa_rdm_ep);

	pkt_attr.connid = include_connid ? raw_addr.qkey : 0;
	pkt_attr.host_id = g_efa_unit_test_mocks.peer_host_id;
	pkt_attr.device_version = 0xefa0;
	efa_unit_test_handshake_pkt_construct(pkt_entry, &pkt_attr);

	/* Setup QP mocks */
	g_efa_unit_test_mocks.efa_qp_wr_start = &efa_mock_efa_qp_wr_start_no_op;
	/* this mock will save the send work request (wr) in a global array */
	g_efa_unit_test_mocks.efa_qp_wr_send = &efa_mock_efa_qp_wr_send_verify_handshake_pkt_local_host_id_and_save_wr;
	g_efa_unit_test_mocks.efa_qp_wr_set_inline_data_list = &efa_mock_efa_qp_wr_set_inline_data_list_no_op;
	g_efa_unit_test_mocks.efa_qp_wr_set_sge_list = &efa_mock_efa_qp_wr_set_sge_list_no_op;
	g_efa_unit_test_mocks.efa_qp_wr_set_ud_addr = &efa_mock_efa_qp_wr_set_ud_addr_no_op;
	g_efa_unit_test_mocks.efa_qp_wr_complete = &efa_mock_efa_qp_wr_complete_no_op;
	expect_function_call(efa_mock_efa_qp_wr_send_verify_handshake_pkt_local_host_id_and_save_wr);

	/* Setup CQ mocks */
	g_efa_unit_test_mocks.efa_ibv_cq_end_poll = &efa_mock_efa_ibv_cq_end_poll_check_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_next_poll = &efa_mock_efa_ibv_cq_next_poll_check_function_called_and_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_byte_len = &efa_mock_efa_ibv_cq_wc_read_byte_len_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_opcode = &efa_mock_efa_ibv_cq_wc_read_opcode_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_slid = &efa_mock_efa_ibv_cq_wc_read_slid_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_src_qp = &efa_mock_efa_ibv_cq_wc_read_src_qp_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_qp_num = &efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_wc_flags = &efa_mock_efa_ibv_cq_wc_read_wc_flags_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_vendor_err = &efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_return_mock;
	ibv_cq->ibv_cq_ex->status = IBV_WC_SUCCESS;
	ibv_cq->ibv_cq_ex->wr_id = (uintptr_t)pkt_entry;
	expect_function_call(efa_mock_efa_ibv_cq_next_poll_check_function_called_and_return_mock);

	/* Receive handshake packet */
	will_return(efa_mock_efa_ibv_cq_end_poll_check_mock, NULL);
	will_return(efa_mock_efa_ibv_cq_next_poll_check_function_called_and_return_mock, ENOENT);
	will_return(efa_mock_efa_ibv_cq_wc_read_byte_len_return_mock, pkt_entry->pkt_size);
	will_return(efa_mock_efa_ibv_cq_wc_read_opcode_return_mock, IBV_WC_RECV);
	will_return(efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock, efa_rdm_ep->base_ep.qp->qp_num);
	will_return(efa_mock_efa_ibv_cq_wc_read_wc_flags_return_mock, 0);
	will_return(efa_mock_efa_ibv_cq_wc_read_slid_return_mock, efa_rdm_ep_get_peer_ahn(efa_rdm_ep, peer_addr));
	will_return(efa_mock_efa_ibv_cq_wc_read_src_qp_return_mock, raw_addr.qpn);
	will_return(efa_mock_efa_ibv_cq_start_poll_return_mock, IBV_WC_SUCCESS);

	/**
	 * Fire away handshake packet.
	 * Because we don't care if it fails(there is no receiver!), mark it as failed to make mocking simpler.
	 */
	will_return(efa_mock_efa_ibv_cq_end_poll_check_mock, NULL);
	will_return(efa_mock_efa_ibv_cq_wc_read_opcode_return_mock, IBV_WC_SEND);
	will_return(efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock, efa_rdm_ep->base_ep.qp->qp_num);
	will_return(efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock, FI_EFA_ERR_OTHER);
	will_return(efa_mock_efa_ibv_cq_start_poll_return_mock, IBV_WC_SUCCESS);

	/* Progress the recv wr first to process the received handshake packet. */
	cq_read_recv_ret = fi_cq_read(resource->cq, &cq_entry, 1);

	actual_peer_host_id = peer->host_id;

	/**
	 * We need to poll the CQ twice explicitly to point the CQE
	 * to the saved send wr in handshake
	 */
	ibv_cq->ibv_cq_ex->status = IBV_WC_GENERAL_ERR;
	ibv_cq->ibv_cq_ex->wr_id = (uintptr_t)g_ibv_submitted_wr_id_vec[0];

	/* Progress the send wr to clean up outstanding tx ops */
	cq_read_send_ret = fi_cq_read(resource->cq, &cq_entry, 1);

	/* HANDSHAKE packet does not generate completion entry */
	assert_int_equal(cq_read_recv_ret, -FI_EAGAIN);
	assert_int_equal(cq_read_send_ret, -FI_EAGAIN);

	/* Peer host id is set after handshake */
	assert_true(actual_peer_host_id == g_efa_unit_test_mocks.peer_host_id);

	/* Device version should be stored after handshake */
	assert_int_equal(peer->device_version, 0xefa0);

	/* reset the mocked cq before it's polled by ep close */
	will_return_always(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
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
	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
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

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

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
 * @brief When using LL128 protocol, test the packet allocated from read_copy_pkt_pool
 *  is 128 byte aligned.
 * 
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_rdm_read_copy_pkt_pool_128_alignment(struct efa_resource **state)
{
	int ret;
	struct efa_rdm_pke *pkt_entry;
	struct fid_ep *ep;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain = NULL;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	/* rx_readcopy_pkt_pool is only created when application requested FI_HMEM */
	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);
	efa_domain->util_domain.mr_mode |= FI_MR_HMEM;

	ret = fi_endpoint(resource->domain, resource->info, &ep, NULL);
	assert_int_equal(ret, 0);
	efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_ep->sendrecv_in_order_aligned_128_bytes = 1;

	pkt_entry =
		efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->rx_readcopy_pkt_pool,
				  EFA_RDM_PKE_FROM_READ_COPY_POOL);
	assert_non_null(pkt_entry);
	efa_rdm_ep->rx_readcopy_pkt_pool_used++;
	assert(ofi_is_addr_aligned((void *) pkt_entry->wiredata,
				   EFA_RDM_IN_ORDER_ALIGNMENT));
	efa_rdm_pke_release_rx(pkt_entry);

	fi_close(&ep->fid);
}

/**
 * @brief When using LL128 protocol, the copy method is local read. 
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_rdm_pke_get_available_copy_methods_align128(struct efa_resource **state)
{
	int ret;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_mr efa_mr;
	struct efa_resource *resource = *state;
	bool local_read_available, gdrcopy_available, cuda_memcpy_available;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_mr.peer.iface = FI_HMEM_CUDA;

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_ep->sendrecv_in_order_aligned_128_bytes = 1;
	
	/* p2p is available */
	g_efa_hmem_info[FI_HMEM_CUDA].p2p_supported_by_device = true;
	efa_rdm_ep->hmem_p2p_opt = FI_HMEM_P2P_ENABLED;

	/* RDMA read is supported */
	efa_rdm_ep->use_device_rdma = true;
	uint64_t caps = efa_rdm_ep_domain(efa_rdm_ep)->device->device_caps;
	efa_rdm_ep_domain(efa_rdm_ep)->device->device_caps |=
		EFADV_DEVICE_ATTR_CAPS_RDMA_READ;

	ret = efa_rdm_pke_get_available_copy_methods(
		efa_rdm_ep, &efa_mr, &local_read_available,
		&cuda_memcpy_available, &gdrcopy_available);

	efa_rdm_ep_domain(efa_rdm_ep)->device->device_caps = caps;

	assert_int_equal(ret, 0);
	assert_true(local_read_available);
	assert_false(cuda_memcpy_available);
	assert_false(gdrcopy_available);
}

/**
 * @brief when delivery complete atomic was used and handshake packet has not been received
 * verify the txe is queued
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_rdm_ep_dc_atomic_queue_before_handshake(struct efa_resource **state)
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
	struct efa_rdm_ope *txe;

	/* disable shm to force using efa device to send */
	efa_unit_test_resource_construct_rdm_shm_disabled(resource);

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

	/* set peer->flag to EFA_RDM_PEER_REQ_SENT will make efa_rdm_atomic() think
	 * a REQ packet has been sent to the peer (so no need to send again)
	 * handshake has not been received, so we do not know whether the peer support DC
	 */
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	peer->flags = EFA_RDM_PEER_REQ_SENT;
	peer->is_local = false;

	assert_false(efa_rdm_ep->homogeneous_peers);
	assert_true(dlist_empty(&efa_rdm_ep->txe_list));
	err = fi_atomicmsg(resource->ep, &msg, FI_DELIVERY_COMPLETE);
	/* DC has been reuquested, but ep do not know whether peer supports it, therefore
	 * the ope has been queued to domain->ope_queued_list
	 */
	assert_int_equal(err, 0);
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list),  1);
	assert_int_equal(efa_unit_test_get_dlist_length(&(efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list)), 1);
	txe = container_of(efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list.next, struct efa_rdm_ope, queued_entry);
	assert_true((txe->op == ofi_op_atomic));
	assert_true(txe->internal_flags & EFA_RDM_OPE_QUEUED_BEFORE_HANDSHAKE);
}

/**
 * @brief when delivery complete send was used and handshake packet has not been received
 * verify the txe is queued
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_rdm_ep_dc_send_queue_before_handshake(struct efa_resource **state)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct fi_msg msg = {0};
	struct iovec iov;
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t peer_addr;
	int err, numaddr;
	struct efa_rdm_ope *txe;

	/* disable shm to force using efa device to send */
	efa_unit_test_resource_construct_rdm_shm_disabled(resource);

	/* create a fake peer */
	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	numaddr = fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL);
	assert_int_equal(numaddr, 1);

	msg.addr = peer_addr;
	msg.iov_count = 1;
	iov.iov_base = NULL;
	iov.iov_len = 0;
	msg.msg_iov = &iov;
	msg.desc = NULL;

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* set peer->flag to EFA_RDM_PEER_REQ_SENT will make efa_rdm_atomic() think
	 * a REQ packet has been sent to the peer (so no need to send again)
	 * handshake has not been received, so we do not know whether the peer support DC
	 */
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	peer->flags = EFA_RDM_PEER_REQ_SENT;
	peer->is_local = false;

	assert_false(efa_rdm_ep->homogeneous_peers);
	assert_true(dlist_empty(&efa_rdm_ep->txe_list));
	err = fi_sendmsg(resource->ep, &msg, FI_DELIVERY_COMPLETE);
	/* DC has been reuquested, but ep do not know whether peer supports it, therefore
	 * the ope has been queued to domain->ope_queued_list
	 */
	assert_int_equal(err, 0);
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list),  1);
	assert_int_equal(efa_unit_test_get_dlist_length(&(efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list)), 1);
	txe = container_of(efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list.next, struct efa_rdm_ope, queued_entry);
	assert_true((txe->op == ofi_op_msg));
	assert_true(txe->internal_flags & EFA_RDM_OPE_QUEUED_BEFORE_HANDSHAKE);
}

/**
 * @brief when delivery complete send was used and handshake packet has not been received
 * verify the txes are queued before the number of requests reach EFA_RDM_MAX_QUEUED_OPE_BEFORE_HANDSHAKE.
 * After reaching the limit, fi_send should return -FI_EAGAIN
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_rdm_ep_dc_send_queue_limit_before_handshake(struct efa_resource **state)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct fi_msg msg = {0};
	struct iovec iov;
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t peer_addr;
	int err, numaddr;
	int i;

	/* disable shm to force using efa device to send */
	efa_unit_test_resource_construct_rdm_shm_disabled(resource);

	/* create a fake peer */
	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	numaddr = fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL);
	assert_int_equal(numaddr, 1);

	msg.addr = peer_addr;
	msg.iov_count = 1;
	iov.iov_base = NULL;
	iov.iov_len = 0;
	msg.msg_iov = &iov;
	msg.desc = NULL;

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* set peer->flag to EFA_RDM_PEER_REQ_SENT will make efa_rdm_atomic() think
	 * a REQ packet has been sent to the peer (so no need to send again)
	 * handshake has not been received, so we do not know whether the peer support DC
	 */
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	peer->flags = EFA_RDM_PEER_REQ_SENT;
	peer->is_local = false;

	assert_false(efa_rdm_ep->homogeneous_peers);
	assert_true(dlist_empty(&efa_rdm_ep->txe_list));

	for (i = 0; i < EFA_RDM_MAX_QUEUED_OPE_BEFORE_HANDSHAKE; i++) {
		err = fi_sendmsg(resource->ep, &msg, FI_DELIVERY_COMPLETE);
		assert_int_equal(err, 0);
	}

	assert_true(efa_rdm_ep->ope_queued_before_handshake_cnt == EFA_RDM_MAX_QUEUED_OPE_BEFORE_HANDSHAKE);
	err = fi_sendmsg(resource->ep, &msg, FI_DELIVERY_COMPLETE);
	assert_int_equal(err, -FI_EAGAIN);
}

/**
 * @brief verify tx entry is queued for rma (read or write) request before handshake is made.
 *
 * @param[in] state	struct efa_resource that is managed by the framework
 * @param[in] op op code
 */
void test_efa_rdm_ep_rma_queue_before_handshake(struct efa_resource **state, int op)
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
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	resource->hints->caps |= FI_MSG | FI_TAGGED | FI_RMA;
	resource->hints->domain_attr->mr_mode |= MR_MODE_BITS;
	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM, FI_VERSION(1, 14),
	                                            resource->hints, true, true);

	/* ensure we don't have RMA capability. */
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

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

	/* set peer->flag to EFA_RDM_PEER_REQ_SENT will make efa_rdm_atomic() think
	 * a REQ packet has been sent to the peer (so no need to send again)
	 * handshake has not been received, so we do not know whether the peer support DC
	 */
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	peer->flags = EFA_RDM_PEER_REQ_SENT;
	peer->is_local = false;
	assert_false(efa_rdm_ep->homogeneous_peers);
	assert_true(dlist_empty(&efa_rdm_ep->txe_list));

	if (op == ofi_op_read_req) {
		err = fi_read(resource->ep, buf, buf_len,
				NULL, /* desc, not required */
				peer_addr,
				rma_addr,
				rma_key,
				NULL); /* context */
	} else if (op == ofi_op_write) {
		err = fi_write(resource->ep, buf, buf_len,
				NULL, /* desc, not required */
				peer_addr,
				rma_addr,
				rma_key,
				NULL); /* context */
	} else {
		fprintf(stderr, "Unknown op code %d\n", op);
		fail();
	}
	assert_int_equal(err, 0);
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list),  1);
	assert_int_equal(efa_unit_test_get_dlist_length(&(efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list)), 1);
	txe = container_of(efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list.next, struct efa_rdm_ope, queued_entry);
	assert_true((txe->op == op));
	assert_true(txe->internal_flags & EFA_RDM_OPE_QUEUED_BEFORE_HANDSHAKE);
}

void test_efa_rdm_ep_write_queue_before_handshake(struct efa_resource **state)
{
	test_efa_rdm_ep_rma_queue_before_handshake(state, ofi_op_write);
}

void test_efa_rdm_ep_read_queue_before_handshake(struct efa_resource **state)
{
	test_efa_rdm_ep_rma_queue_before_handshake(state, ofi_op_read_req);
}

/**
 * @brief Test the efa_rdm_ep_trigger_handshake function
 * with different peer setup and check the txe flags
 *
 * @param state efa_resource
 */
void test_efa_rdm_ep_trigger_handshake(struct efa_resource **state)
{
	struct efa_rdm_ope *txe;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_resource *resource = *state;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;

	g_efa_unit_test_mocks.efa_rdm_ope_post_send = &efa_mock_efa_rdm_ope_post_send_return_mock;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	will_return_always(efa_mock_efa_rdm_ope_post_send_return_mock, FI_SUCCESS);

	/* Create and register a fake peer */
	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 0;
	raw_addr.qkey = 0x1234;

	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL), 1);

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	assert_non_null(peer);

	/* No txe should have been allocated yet */
	assert_true(dlist_empty(&efa_rdm_ep->txe_list));

	/*
	 * When the peer already has made , the function should be a no-op
	 * and no txe is allocated
	 */
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED | EFA_RDM_PEER_REQ_SENT;
	assert_int_equal(efa_rdm_ep_trigger_handshake(efa_rdm_ep, peer), FI_SUCCESS);
	assert_true(dlist_empty(&efa_rdm_ep->txe_list));

	/*
	 * Reset the peer flags to 0, now we should expect a txe allocated
	 */
	peer->flags = 0;
	assert_int_equal(efa_rdm_ep_trigger_handshake(efa_rdm_ep, peer), FI_SUCCESS);
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list),  1);

	txe = container_of(efa_rdm_ep->txe_list.next, struct efa_rdm_ope, ep_entry);

	assert_true(txe->fi_flags & EFA_RDM_TXE_NO_COMPLETION);
	assert_true(txe->fi_flags & EFA_RDM_TXE_NO_COUNTER);
	assert_true(txe->internal_flags & EFA_RDM_OPE_INTERNAL);

	efa_rdm_txe_release(txe);
}

/**
 * @brief When local support unsolicited write, but the peer doesn't, fi_writedata
 * (use rdma-write with imm) should fail as FI_EINVAL
 *
 * @param state struct efa_resource that is managed by the framework
 */
void test_efa_rdm_ep_rma_inconsistent_unsolicited_write_recv(struct efa_resource **state)
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
	struct efa_rdm_peer *peer;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	resource->hints->caps |= FI_MSG | FI_TAGGED | FI_RMA;
	resource->hints->domain_attr->mr_mode |= MR_MODE_BITS;
	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM, FI_VERSION(1, 22),
	                                            resource->hints, true, true);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/**
	 * TODO: It's better to mock this function
	 * so we can test on platform that doesn't
	 * support rdma-write.
	 */
	if (!(efa_rdm_ep_support_rdma_write(efa_rdm_ep)))
		skip();

	/* Make local ep support unsolicited write recv */
	efa_rdm_ep->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_UNSOLICITED_WRITE_RECV;

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

	/*
	 * Fake a peer that has made handshake and
	 * does not support unsolicited write recv
	 */
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_RDMA_WRITE;
	peer->extra_info[0] &= ~EFA_RDM_EXTRA_FEATURE_UNSOLICITED_WRITE_RECV;
	/* make sure shm is not used */
	peer->is_local = false;

	assert_false(efa_rdm_ep->homogeneous_peers);
	err = fi_writedata(resource->ep, buf, buf_len,
			    NULL, /* desc, not required */
			    0x1234,
			    peer_addr,
			    rma_addr,
			    rma_key,
			    NULL); /* context */
	assert_int_equal(err, -FI_EOPNOTSUPP);
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

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

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

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	resource->hints->caps |= FI_MSG | FI_TAGGED;
	resource->hints->caps &= ~FI_RMA;
	resource->hints->domain_attr->mr_mode |= MR_MODE_BITS;
	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM, FI_VERSION(1, 14),
	                                            resource->hints, true, true);

	/* ensure we don't have RMA capability. */
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	assert_int_equal( efa_rdm_ep->base_ep.info->caps & FI_RMA, 0);

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

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	resource->hints->caps |= FI_MSG | FI_TAGGED;
	resource->hints->caps &= ~FI_ATOMIC;
	resource->hints->domain_attr->mr_mode |= MR_MODE_BITS;
	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM, FI_VERSION(1, 14),
	                                            resource->hints, true, true);

	/* ensure we don't have ATOMIC capability. */
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	assert_int_equal( efa_rdm_ep->base_ep.info->caps & FI_ATOMIC, 0);

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

/*
 * Check fi_getopt return with different input opt_len
 */
void test_efa_rdm_ep_getopt(struct efa_resource **state, size_t opt_len, int expected_return)
{
	struct efa_resource *resource = *state;
	size_t opt_val;
	size_t opt_len_temp;
	size_t i;
	int ret;
	int opt_names[] = {
		FI_OPT_MIN_MULTI_RECV,
		FI_OPT_EFA_RNR_RETRY,
		FI_OPT_FI_HMEM_P2P,
		FI_OPT_EFA_EMULATED_READ,
		FI_OPT_EFA_EMULATED_WRITE,
		FI_OPT_EFA_EMULATED_ATOMICS,
	};
	size_t num_opt_names = sizeof(opt_names) / sizeof(int);

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	for (i = 0; i < num_opt_names; i++) {
		opt_len_temp = opt_len;
		ret = fi_getopt(&resource->ep->fid, FI_OPT_ENDPOINT, opt_names[i], &opt_val, &opt_len_temp);
		assert_int_equal(ret, expected_return);
	}
}

/* undersized optlen should return -FI_ETOOSMALL */
void test_efa_rdm_ep_getopt_undersized_optlen(struct efa_resource **state)
{
	test_efa_rdm_ep_getopt(state, 0, -FI_ETOOSMALL);
}

/* oversized optlen should return FI_SUCCESS */
void test_efa_rdm_ep_getopt_oversized_optlen(struct efa_resource **state)
{
	test_efa_rdm_ep_getopt(state, 16, FI_SUCCESS);
}

void test_efa_rdm_ep_setopt_shared_memory_permitted(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;

	/* disable shm to force using efa device to send */
	efa_unit_test_resource_construct_rdm_shm_disabled(resource);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_null(ep->shm_ep);
}

void test_efa_rdm_ep_setopt_homogeneous_peers(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	bool optval = true;
	
	efa_unit_test_resource_construct_ep_not_enabled(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	ep = container_of(resource->ep, struct efa_rdm_ep,
		base_ep.util_ep.ep_fid);
	assert_false(ep->homogeneous_peers);
	assert_int_equal(fi_setopt(&resource->ep->fid, FI_OPT_ENDPOINT,
				   FI_OPT_EFA_HOMOGENEOUS_PEERS, &optval,
				   sizeof(optval)),
			 FI_SUCCESS);
	assert_true(ep->homogeneous_peers);
}

/**
 * @brief Test fi_enable with different optval of fi_setopt for
 * FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES optname.
 * @param state struct efa_resource that is managed by the framework
 * @param expected_status expected return status of fi_enable
 * @param optval the optval passed to fi_setopt
 */
void test_efa_rdm_ep_enable_qp_in_order_aligned_128_bytes_common(struct efa_resource **state, int expected_status, bool optval)
{
	struct efa_resource *resource = *state;

	efa_unit_test_resource_construct_ep_not_enabled(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	/* fi_setopt should always succeed */
	assert_int_equal(fi_setopt(&resource->ep->fid, FI_OPT_ENDPOINT,
				   FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES, &optval,
				   sizeof(optval)), expected_status);
}

#if HAVE_EFA_DATA_IN_ORDER_ALIGNED_128_BYTES
/**
 * @brief Test the case where fi_enable should return success
 *
 * @param state struct efa_resource that is managed by the framework
 */
void test_efa_rdm_ep_enable_qp_in_order_aligned_128_bytes_good(struct efa_resource **state)
{
	/* mock ibv_query_qp_data_in_order to return required capability */
	g_efa_unit_test_mocks.ibv_query_qp_data_in_order = &efa_mock_ibv_query_qp_data_in_order_return_in_order_aligned_128_bytes;
	test_efa_rdm_ep_enable_qp_in_order_aligned_128_bytes_common(state, FI_SUCCESS, true);
}

/**
 * @brief Test the case where fi_enable should return -FI_EOPNOTSUPP
 *
 * @param state struct efa_resource that is managed by the framework
 */
void test_efa_rdm_ep_enable_qp_in_order_aligned_128_bytes_bad(struct efa_resource **state)
{
	/* mock ibv_query_qp_data_in_order to return zero capability */
	g_efa_unit_test_mocks.ibv_query_qp_data_in_order = &efa_mock_ibv_query_qp_data_in_order_return_0;
	test_efa_rdm_ep_enable_qp_in_order_aligned_128_bytes_common(state, -FI_EOPNOTSUPP, true);
}

#else

void test_efa_rdm_ep_enable_qp_in_order_aligned_128_bytes_good(struct efa_resource **state)
{
	test_efa_rdm_ep_enable_qp_in_order_aligned_128_bytes_common(state, FI_SUCCESS, false);
}

void test_efa_rdm_ep_enable_qp_in_order_aligned_128_bytes_bad(struct efa_resource **state)
{
	test_efa_rdm_ep_enable_qp_in_order_aligned_128_bytes_common(state, -FI_EOPNOTSUPP, true);
}

#endif

static void test_efa_rdm_ep_use_zcpy_rx_impl(struct efa_resource *resource,
                                             bool cuda_p2p_disabled,
                                             bool cuda_p2p_supported,
                                             bool expected_use_zcpy_rx)
{
	struct efa_rdm_ep *ep;
	size_t max_msg_size = 1000;
	size_t inject_msg_size = 0;
	size_t inject_rma_size = 0;
	bool shm_permitted = false;
	ofi_hmem_disable_p2p = cuda_p2p_disabled;

	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM, FI_VERSION(1, 14),
	                                            resource->hints, false, true);

	/* System memory P2P should always be enabled */
	assert_true(g_efa_hmem_info[FI_HMEM_SYSTEM].initialized);
	assert_true(g_efa_hmem_info[FI_HMEM_SYSTEM].p2p_supported_by_device);

	/**
	 * We want to be able to run this test on any platform:
	 * 1. Fake CUDA support.
	 * 2. Disable all other hmem ifaces.
	 */
	g_efa_hmem_info[FI_HMEM_CUDA].initialized = true;
	g_efa_hmem_info[FI_HMEM_CUDA].p2p_supported_by_device = cuda_p2p_supported;

	g_efa_hmem_info[FI_HMEM_NEURON].initialized = false;
	g_efa_hmem_info[FI_HMEM_SYNAPSEAI].initialized = false;

	ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	if (cuda_p2p_supported)
		ep->hmem_p2p_opt = FI_HMEM_P2P_ENABLED;

	/* Set sufficiently small max_msg_size */
	assert_int_equal(fi_setopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_MAX_MSG_SIZE,
			&max_msg_size, sizeof max_msg_size), 0);

	/* Disable shm */
	assert_int_equal(fi_setopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_SHARED_MEMORY_PERMITTED,
			&shm_permitted, sizeof shm_permitted), 0);

	assert_true(ep->base_ep.max_msg_size == max_msg_size);

	/* Enable EP */
	assert_int_equal(fi_enable(resource->ep), 0);

	assert_true(ep->use_zcpy_rx == expected_use_zcpy_rx);

	assert_int_equal(fi_getopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_INJECT_MSG_SIZE,
			&inject_msg_size, &(size_t){sizeof inject_msg_size}), 0);
	assert_int_equal(ep->base_ep.inject_msg_size, inject_msg_size);

	assert_int_equal(fi_getopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_INJECT_RMA_SIZE,
			&inject_rma_size, &(size_t){sizeof inject_rma_size}), 0);
	assert_int_equal(ep->base_ep.inject_rma_size, inject_rma_size);

	if (expected_use_zcpy_rx) {
		assert_int_equal(inject_msg_size, efa_rdm_ep_domain(ep)->device->efa_attr.inline_buf_size);
		assert_int_equal(inject_rma_size, efa_rdm_ep_domain(ep)->device->efa_attr.inline_buf_size);
	} else {
		assert_int_equal(inject_msg_size, resource->info->tx_attr->inject_size);
		assert_int_equal(inject_rma_size, resource->info->tx_attr->inject_size);
	}
	/* restore global variable */
	ofi_hmem_disable_p2p = 0;
}

/**
 * @brief Verify zcpy_rx is enabled when the following requirements are met:
 * 1. app doesn't require FI_ORDER_SAS in tx or rx's msg_order
 * 2. app uses FI_MSG_PREFIX mode
 * 3. app's max msg size is smaller than mtu_size - prefix_size
 * 4. app doesn't use FI_DIRECTED_RECV, FI_TAGGED, FI_ATOMIC capability
 */
void test_efa_rdm_ep_user_zcpy_rx_disabled(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);

	resource->hints->mode = FI_MSG_PREFIX;
	resource->hints->caps = FI_MSG;

	test_efa_rdm_ep_use_zcpy_rx_impl(resource, false, true, true);
}

/**
 * @brief Verify zcpy_rx is disabled if CUDA P2P is explictly disabled
 */
void test_efa_rdm_ep_user_disable_p2p_zcpy_rx_disabled(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);

	resource->hints->mode = FI_MSG_PREFIX;
	resource->hints->caps = FI_MSG;

	test_efa_rdm_ep_use_zcpy_rx_impl(resource, true, false, false);
}

/**
 * @brief When sas is requested for either tx or rx. zcpy will be disabled
 */
void test_efa_rdm_ep_user_zcpy_rx_unhappy_due_to_sas(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);

	resource->hints->tx_attr->msg_order = FI_ORDER_SAS;
	resource->hints->rx_attr->msg_order = FI_ORDER_SAS;
	resource->hints->mode = FI_MSG_PREFIX;
	resource->hints->caps = FI_MSG;

	test_efa_rdm_ep_use_zcpy_rx_impl(resource, false, true, false);
}

/**
 * @brief Verify zcpy_rx is disabled if CUDA P2P is enabled but not supported
 */
void test_efa_rdm_ep_user_p2p_not_supported_zcpy_rx_happy(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);

	resource->hints->mode = FI_MSG_PREFIX;
	resource->hints->caps = FI_MSG;

	test_efa_rdm_ep_use_zcpy_rx_impl(resource, false, false, false);
}

/**
 * @brief Verify zcpy_rx is disabled if FI_MR_LOCAL is not set
 */
void test_efa_rdm_ep_user_zcpy_rx_unhappy_due_to_no_mr_local(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);

	resource->hints->caps = FI_MSG;
	resource->hints->domain_attr->mr_mode &= ~FI_MR_LOCAL;

	test_efa_rdm_ep_use_zcpy_rx_impl(resource, false, true, false);
}

void test_efa_rdm_ep_close_discard_posted_recv(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	char buf[16];

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	/* Post recv and then close ep */
	assert_int_equal(fi_recv(resource->ep, (void *) buf, 16, NULL, FI_ADDR_UNSPEC, NULL), 0);

	assert_int_equal(fi_close(&resource->ep->fid), 0);

	/* CQ should be empty and no err entry */
	assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);

	/* Reset to NULL to avoid test reaper closing again */
	resource->ep = NULL;
}

void test_efa_rdm_ep_zcpy_recv_cancel(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fi_context cancel_context = {0};
	struct efa_unit_test_buff recv_buff;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);

	resource->hints->caps = FI_MSG;

	/* enable zero-copy recv mode in ep */
	test_efa_rdm_ep_use_zcpy_rx_impl(resource, false, true, true);

	/* Construct a recv buffer with mr */
	efa_unit_test_buff_construct(&recv_buff, resource, 16);

	assert_int_equal(fi_recv(resource->ep, recv_buff.buff, recv_buff.size, fi_mr_desc(recv_buff.mr), FI_ADDR_UNSPEC, &cancel_context), 0);

	assert_int_equal(fi_cancel((struct fid *)resource->ep, &cancel_context), -FI_EOPNOTSUPP);

	/**
	 * the buf is still posted to rdma-core, so unregistering mr can
	 * return non-zero. Currently ignore this failure.
	 */
	(void) fi_close(&recv_buff.mr->fid);
	free(recv_buff.buff);
}

/**
 * @brief When user posts more than rx size fi_recv, we should return eagain and make sure
 * there is no rx entry leaked
 */
void test_efa_rdm_ep_zcpy_recv_eagain(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff recv_buff;
	int i;
	struct efa_rdm_ep *efa_rdm_ep;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);

	resource->hints->caps = FI_MSG;

	/* enable zero-copy recv mode in ep */
	test_efa_rdm_ep_use_zcpy_rx_impl(resource, false, true, true);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Construct a recv buffer with mr */
	efa_unit_test_buff_construct(&recv_buff, resource, 16);

	for (i = 0; i < efa_rdm_ep->base_ep.info->rx_attr->size; i++)
		assert_int_equal(fi_recv(resource->ep, recv_buff.buff, recv_buff.size, fi_mr_desc(recv_buff.mr), FI_ADDR_UNSPEC, NULL), 0);

	/* we should have rx number of rx entry before and after the extra recv post */
	assert_true(efa_unit_test_get_dlist_length(&efa_rdm_ep->rxe_list) == efa_rdm_ep->base_ep.info->rx_attr->size);
	assert_int_equal(fi_recv(resource->ep, recv_buff.buff, recv_buff.size, fi_mr_desc(recv_buff.mr), FI_ADDR_UNSPEC, NULL), -FI_EAGAIN);
	assert_true(efa_unit_test_get_dlist_length(&efa_rdm_ep->rxe_list) == efa_rdm_ep->base_ep.info->rx_attr->size);

	/**
	 * the buf is still posted to rdma-core, so unregistering mr can
	 * return non-zero. Currently ignore this failure.
	 */
	(void) fi_close(&recv_buff.mr->fid);
	free(recv_buff.buff);
}

/**
 * @brief when efa_rdm_ep_post_handshake_error failed due to pkt pool exhaustion, 
 * make sure both txe is cleaned
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_rdm_ep_post_handshake_error_handling_pke_exhaustion(struct efa_resource **state)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t peer_addr;
	int err, numaddr;
	struct efa_rdm_pke **pkt_entry_vec;
	int i;
	size_t tx_size;

	/* disable shm to force using efa device to send */
	efa_unit_test_resource_construct_rdm_shm_disabled(resource);

	/* create a fake peer */
	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	numaddr = fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL);
	assert_int_equal(numaddr, 1);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	tx_size = efa_rdm_ep->base_ep.info->tx_attr->size;

	/* set peer->flag to EFA_RDM_PEER_REQ_SENT will make efa_rdm_atomic() think
	 * a REQ packet has been sent to the peer (so no need to send again)
	 * handshake has not been received, so we do not know whether the peer support DC
	 */
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	peer->flags = EFA_RDM_PEER_REQ_SENT;
	peer->is_local = false;

	pkt_entry_vec = calloc(tx_size, sizeof(struct efa_rdm_pke *));
	assert_non_null(pkt_entry_vec);

	/* Exhaust the tx pkt pool */
	for (i = 0; i < tx_size; i++) {
		pkt_entry_vec[i] = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_tx_pkt_pool, EFA_RDM_PKE_FROM_EFA_TX_POOL);
		assert_non_null(pkt_entry_vec[i]);
	}

	/* txe list should be empty before and after the failed handshake post call */
	assert_true(dlist_empty(&efa_rdm_ep->txe_list));
	assert_int_equal(efa_rdm_ep_post_handshake(efa_rdm_ep, peer), -FI_EAGAIN);
	assert_true(dlist_empty(&efa_rdm_ep->txe_list));

	for (i = 0; i < tx_size; i++)
		efa_rdm_pke_release_tx(pkt_entry_vec[i]);

	free(pkt_entry_vec);
}

static
void test_efa_rdm_ep_rx_refill_impl(struct efa_resource **state, int threshold, int rx_size)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pkt_entry;
	int i;
	size_t threshold_orig;

	if (threshold < 4 || rx_size < 4) {
		fprintf(stderr, "Too small threshold or rx_size for this test\n");
		fail();
	}

	threshold_orig = efa_env.internal_rx_refill_threshold;

	efa_env.internal_rx_refill_threshold = threshold;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	assert_non_null(resource->hints);
	resource->hints->rx_attr->size = rx_size;
	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM, FI_VERSION(1, 14),
	                                            resource->hints, true, true);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	assert_int_equal(efa_rdm_ep_get_rx_pool_size(efa_rdm_ep), rx_size);

	/* Grow the rx pool and post rx pkts */
	efa_rdm_ep_post_internal_rx_pkts(efa_rdm_ep);
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_posted, efa_rdm_ep_get_rx_pool_size(efa_rdm_ep));

	assert_int_equal(efa_rdm_ep->efa_rx_pkts_to_post, 0);
	for (i = 0; i < 4; i++) {
		pkt_entry = ofi_bufpool_get_ibuf(efa_rdm_ep->efa_rx_pkt_pool, i);
		assert_non_null(pkt_entry);
		efa_rdm_pke_release_rx(pkt_entry);
	}
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_to_post, 4);

	efa_rdm_ep_bulk_post_internal_rx_pkts(efa_rdm_ep);

	/**
	 * efa_rx_pkts_to_post < FI_EFA_RX_REFILL_THRESHOLD
	 * pkts should NOT be refilled
	 */
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_to_post, 4);
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_posted, rx_size);

	/* releasing more pkts to reach the threshold or rx_size*/
	for (i = 4; i < MIN(rx_size, threshold); i++) {
		pkt_entry = ofi_bufpool_get_ibuf(efa_rdm_ep->efa_rx_pkt_pool, i);
		assert_non_null(pkt_entry);
		efa_rdm_pke_release_rx(pkt_entry);
	}

	assert_int_equal(efa_rdm_ep->efa_rx_pkts_to_post, i);

	efa_rdm_ep_bulk_post_internal_rx_pkts(efa_rdm_ep);

	/**
	 * efa_rx_pkts_to_post == min(FI_EFA_RX_REFILL_THRESHOLD, FI_EFA_RX_SIZE)
	 * pkts should be refilled
	 */
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_to_post, 0);
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_posted, rx_size + i);

	/* recover the original value */
	efa_env.internal_rx_refill_threshold = threshold_orig;
}

void test_efa_rdm_ep_rx_refill_threshold_smaller_than_rx_size(struct efa_resource **state)
{
	test_efa_rdm_ep_rx_refill_impl(state, 8, 64);
}

void test_efa_rdm_ep_rx_refill_threshold_larger_than_rx_size(struct efa_resource **state)
{
	test_efa_rdm_ep_rx_refill_impl(state, 128, 64);
}

/**
 * @brief when unsolicited write recv is supported (checked by cq),
 * efa_rdm_ep_support_unsolicited_write_recv
 * should return true, otherwise it should return false
 *
 * @param[in]	state			struct efa_resource that is managed by the framework
 * @param[in]	is_supported	support status
 */
void test_efa_rdm_ep_support_unsolicited_write_recv(struct efa_resource **state)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_cq *efa_cq;
	struct efa_resource *resource = *state;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid);

	assert_int_equal(efa_cq->ibv_cq.unsolicited_write_recv_enabled,
			 efa_rdm_ep_support_unsolicited_write_recv(efa_rdm_ep));
}

/**
 * @brief Test the default operational sizes for efa_rdm_ep
 *
 * @param state
 */
void test_efa_rdm_ep_default_sizes(struct efa_resource **state)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_resource *resource = *state;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* sizes shared with base_ep */
	assert_int_equal(efa_rdm_ep->base_ep.max_msg_size, resource->info->ep_attr->max_msg_size);
	assert_int_equal(efa_rdm_ep->base_ep.max_rma_size, resource->info->ep_attr->max_msg_size);
	assert_int_equal(efa_rdm_ep->base_ep.inject_msg_size, resource->info->tx_attr->inject_size);
	assert_int_equal(efa_rdm_ep->base_ep.inject_rma_size, resource->info->tx_attr->inject_size);
	assert_int_equal(efa_rdm_ep->base_ep.rnr_retry, EFA_RDM_DEFAULT_RNR_RETRY);

	/* efa_rdm_ep's own fields */
	assert_int_equal(efa_rdm_ep->max_tagged_size, resource->info->ep_attr->max_msg_size);
	assert_int_equal(efa_rdm_ep->max_atomic_size, resource->info->ep_attr->max_msg_size);
	assert_int_equal(efa_rdm_ep->inject_tagged_size, resource->info->tx_attr->inject_size);
	assert_int_equal(efa_rdm_ep->inject_atomic_size, resource->info->tx_attr->inject_size);
}

/**
 * @brief Test the fi_endpoint API for efa_ep
 * for rdm ep type (because the dgram ep type should
 * have the same logic)
 * @param state 
 */
void test_efa_ep_open(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_base_ep *efa_ep;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct_ep_not_enabled(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	efa_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);
	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	/* Check various size limits defaults */
	assert_true(efa_ep->max_msg_size == efa_domain->device->ibv_port_attr.max_msg_sz);
	assert_true(efa_ep->max_rma_size == efa_domain->device->max_rdma_size);
	assert_true(efa_ep->inject_msg_size == efa_domain->device->efa_attr.inline_buf_size);
	/* TODO: update inject_rma_size to inline size after firmware
	 * supports inline rdma write */
	assert_true(efa_ep->inject_rma_size == 0);
	assert_int_equal(efa_ep->rnr_retry, EFA_RNR_INFINITE_RETRY);
}

/**
 * @brief Test the fi_cancel API for efa_ep
 * (for rdm ep type because dgram logic should be the same)
 * It should return -FI_ENOSYS as device doesn't support it;
 * @param state 
 */
void test_efa_ep_cancel(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	int ret;

	efa_unit_test_resource_construct_ep_not_enabled(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	ret = fi_cancel((struct fid *)resource->ep, NULL);
	assert_int_equal(ret, -FI_ENOSYS);
}

/**
 * @brief Test the fi_getopt API fo efa_ep
 *
 * @param state
 */
void test_efa_ep_getopt(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	int optval_int;
	bool optval_bool;
	size_t optval_size_t;
	size_t optlen;
	struct efa_base_ep *efa_ep;

	efa_unit_test_resource_construct_ep_not_enabled(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	efa_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);

	optlen = sizeof(optval_int);
	assert_int_equal(fi_getopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_FI_HMEM_P2P, &optval_int, &optlen), 0);
	assert_int_equal(optval_int, FI_HMEM_P2P_REQUIRED);

	optlen = sizeof(optval_bool);

	assert_int_equal(fi_getopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_EFA_EMULATED_READ, &optval_bool, &optlen), 0);
	assert_false(optval_bool);

	assert_int_equal(fi_getopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_EFA_EMULATED_WRITE, &optval_bool, &optlen), 0);
	assert_false(optval_bool);

	optlen = sizeof(optval_size_t);
	assert_int_equal(fi_getopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_EFA_RNR_RETRY, &optval_size_t, &optlen), 0);
	assert_int_equal(optval_size_t, efa_ep->rnr_retry);

	assert_int_equal(fi_getopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_MAX_MSG_SIZE, &optval_size_t, &optlen), 0);
	assert_int_equal(optval_size_t, efa_ep->max_msg_size);

	assert_int_equal(fi_getopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_MAX_RMA_SIZE, &optval_size_t, &optlen), 0);
	assert_int_equal(optval_size_t, efa_ep->max_rma_size);

	assert_int_equal(fi_getopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_INJECT_MSG_SIZE, &optval_size_t, &optlen), 0);
	assert_int_equal(optval_size_t, efa_ep->inject_msg_size);

	assert_int_equal(fi_getopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_INJECT_RMA_SIZE, &optval_size_t, &optlen), 0);
	assert_int_equal(optval_size_t, efa_ep->inject_rma_size);
}

/**
 * @brief Test the fi_setopt API for efa_ep
 * When RMA is requested, FI_OPT_EFA_USE_DEVICE_RDMA
 * cannot be set as false
 * @param state
 */
void test_efa_ep_setopt_use_device_rdma(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	bool optval;
	struct efa_base_ep *efa_ep;

	efa_unit_test_resource_construct_ep_not_enabled(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	efa_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);

	/* Hard code RMA caps in ep->info for local testing purpose */
	efa_ep->info->caps |= FI_RMA;

	/* Disable rdma is not allowed when user requests FI_RMA */
	optval = false;
	assert_int_equal(fi_setopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_EFA_USE_DEVICE_RDMA, &optval, sizeof(optval)), -FI_EOPNOTSUPP);
}

/**
 * @brief Test the fi_setopt API for efa_ep
 * FI_OPT_FI_HMEM_P2P cannot be set as FI_HMEM_P2P_DISABLED
 * @param state
 */
void test_efa_ep_setopt_hmem_p2p(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	int optval;
	int optvals[] = {
		FI_HMEM_P2P_DISABLED,
		FI_HMEM_P2P_ENABLED,
		FI_HMEM_P2P_PREFERRED,
		FI_HMEM_P2P_REQUIRED,
	};
	size_t num_optvals = sizeof(optvals) / sizeof(int);
	int i, expected_return;

	efa_unit_test_resource_construct_ep_not_enabled(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	/* FI_HMEM_P2P_DISABLED is not allowed */
	for (i = 0; i < num_optvals; i++) {
		optval = optvals[i];
		expected_return = (optval == FI_HMEM_P2P_DISABLED) ? -FI_EOPNOTSUPP : FI_SUCCESS;
		assert_int_equal(fi_setopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_FI_HMEM_P2P, &optval, sizeof(optval)), expected_return);
	}
}

/**
 * @brief Test the fi_setopt API for efa_ep with FI_OPT_EFA_RNR_RETRY
 * @param state
 */
void test_efa_ep_setopt_rnr_retry(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	size_t optval;
	struct efa_base_ep *efa_ep;

	efa_unit_test_resource_construct_ep_not_enabled(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	efa_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);
	assert_false(efa_ep->efa_qp_enabled);

	optval = 7;
	assert_int_equal(fi_setopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_EFA_RNR_RETRY, &optval, sizeof(optval)), FI_SUCCESS);
	assert_int_equal(efa_ep->rnr_retry, optval);

	/* hack qp enabled status to allow local test */
	efa_ep->efa_qp_enabled = true;
	/* fi_setopt should fail when it's called after ep enable */
	assert_int_equal(fi_setopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_EFA_RNR_RETRY, &optval, sizeof(optval)), -FI_EINVAL);
	/* recover */
	efa_ep->efa_qp_enabled = false;
}

/**
 * @brief Test the fi_setopt API for efa_ep with FI_OPT_*_SIZE
 * @param state
 */
void test_efa_ep_setopt_sizes(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	size_t optval;
	struct efa_base_ep *efa_ep;

	efa_unit_test_resource_construct_ep_not_enabled(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	efa_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);

	size_t size_thresholds[] = {
		[FI_OPT_MAX_MSG_SIZE] = (size_t) efa_ep->domain->device->ibv_port_attr.max_msg_sz,
		[FI_OPT_MAX_RMA_SIZE] = (size_t) efa_ep->domain->device->max_rdma_size,
		[FI_OPT_INJECT_MSG_SIZE] = (size_t) efa_ep->domain->device->efa_attr.inline_buf_size,
		[FI_OPT_INJECT_RMA_SIZE] = (size_t) 0,
	};
	int optnames[] = {
		FI_OPT_MAX_MSG_SIZE,
		FI_OPT_MAX_RMA_SIZE,
		FI_OPT_INJECT_MSG_SIZE,
		FI_OPT_INJECT_RMA_SIZE,
	};
	size_t num_optnames = sizeof(optnames) / sizeof(int);
	int i, optname;

	for (i = 0; i < num_optnames; i++) {
		optname = optnames[i];

		/* set optval <= threshold is allowed */
		optval = 0.5 * size_thresholds[optname];
		assert_int_equal(fi_setopt(&resource->ep->fid, FI_OPT_ENDPOINT, optname, &optval, sizeof(optval)), FI_SUCCESS);

		/* set optval > threshold is NOT allowed */
		optval = size_thresholds[optname] + 10;
		assert_int_equal(fi_setopt(&resource->ep->fid, FI_OPT_ENDPOINT, optname, &optval, sizeof(optval)), -FI_EINVAL);
	}
}

/**
 * @brief Test fi_ep_bind and fi_enable API for efa_ep
 *
 * @param state
 */
void test_efa_ep_bind_and_enable(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_base_ep *efa_ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	efa_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);

	assert_true(efa_ep->efa_qp_enabled);
	/* we shouldn't have user recv qp for efa-direct */
	assert_true(efa_ep->user_recv_qp == NULL);
}

#if HAVE_EFA_DATA_PATH_DIRECT

/**
 * @brief qp's data_path_direct status should be consistent
 * with cq's status
 *
 * This test is against efa-direct fabric
 *
 * @param state unit test resources
 */
static
void test_efa_ep_data_path_direct_equal_to_cq_data_path_direct_impl(struct efa_resource **state, bool data_path_direct_enabled)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	bool data_path_direct_enabled_orig;
	struct fid_ep *ep;
	struct efa_base_ep *efa_ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid);

	/* recover the cq boolean */
	data_path_direct_enabled_orig = efa_cq->ibv_cq.data_path_direct_enabled;
	efa_cq->ibv_cq.data_path_direct_enabled = data_path_direct_enabled;

	/* open a test ep */
	assert_int_equal(fi_endpoint(resource->domain, resource->info, &ep, NULL), 0);
	efa_ep = container_of(ep, struct efa_base_ep, util_ep.ep_fid);
	assert_int_equal(fi_ep_bind(ep, &resource->cq->fid, FI_SEND | FI_RECV), 0);
	assert_int_equal(fi_enable(ep), 0);

	assert_true(efa_ep->qp->data_path_direct_enabled == data_path_direct_enabled);

	assert_int_equal(fi_close(&ep->fid), 0);

	/* recover the mocked boolean */
	efa_cq->ibv_cq.data_path_direct_enabled = data_path_direct_enabled_orig;
}

void test_efa_ep_data_path_direct_equal_to_cq_data_path_direct_happy(struct efa_resource **state)
{
	test_efa_ep_data_path_direct_equal_to_cq_data_path_direct_impl(state, true);
}

void test_efa_ep_data_path_direct_equal_to_cq_data_path_direct_unhappy(struct efa_resource **state)
{
	test_efa_ep_data_path_direct_equal_to_cq_data_path_direct_impl(state, false);
}
#else

/* No value to test this, already covered by test_efa_rdm_ep_data_path_direct_ops */
void test_efa_ep_data_path_direct_equal_to_cq_data_path_direct_happy(struct efa_resource **state)
{
	skip();
}

/* No value to test this, already covered by test_efa_rdm_ep_data_path_direct_ops */
void test_efa_ep_data_path_direct_equal_to_cq_data_path_direct_unhappy(struct efa_resource **state)
{
	skip();
}

#endif /* HAVE_EFA_DIRECT_CQ */


/**
 * @brief Test qp's data_path_direct status for efa-rdm ep
 * Currently, data_path_direct should always be disabled by efa-rdm.
 *
 * @param state pointer of efa_resource
 */
void test_efa_rdm_ep_data_path_direct_disabled(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_base_ep *efa_ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);

	assert_false(efa_ep->qp->data_path_direct_enabled);
}
