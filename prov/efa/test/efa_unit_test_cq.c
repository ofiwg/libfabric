/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"
#include "rdm/efa_rdm_cq.h"
#include "efa_av.h"
#include "efa_data_path_direct_entry.h"

/**
 * @brief implementation of test cases for fi_cq_read() works with empty device CQ for given endpoint type
 *
 * When CQ is empty, fi_cq_read() should return -FI_EAGAIN.
 *
 * @param[in]		resource	struct efa_resource that is managed by the framework
 * @param[in]		ep_type		endpoint type, can be FI_EP_DGRAM or FI_EP_RDM
 */
static
void test_impl_cq_read_empty_cq(struct efa_resource *resource, enum fi_ep_type ep_type)
{
	struct fi_cq_data_entry cq_entry;
	int ret;

	efa_unit_test_resource_construct(resource, ep_type, EFA_FABRIC_NAME);
	g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_return_mock;

	/* ibv_start_poll to return ENOENT means device CQ is empty */
	will_return(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);

	ret = fi_cq_read(resource->cq, &cq_entry, 1);

	assert_int_equal(ret, -FI_EAGAIN);

	/* reset the mocked cq before it's polled by ep close */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief verify DGRAM CQ's fi_cq_read() works with empty CQ
 *
 * When CQ is empty, fi_cq_read() should return -FI_EAGAIN.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_dgram_cq_read_empty_cq(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	test_impl_cq_read_empty_cq(resource, FI_EP_DGRAM);
}

/**
 * @brief verify RDM CQ's fi_cq_read() works with empty CQ
 *
 * When CQ is empty, fi_cq_read() should return -FI_EAGAIN.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_ibv_cq_ex_read_empty_cq(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	test_impl_cq_read_empty_cq(resource, FI_EP_RDM);
}

/**
 * @brief test RDM CQ's fi_cq_read()/fi_cq_readerr() works properly when rdma-core return bad status for send.
 *
 * When the send operation failed, fi_cq_read() should return -FI_EAVAIL, which means error available.
 * then user should call fi_cq_readerr() to get an error CQ entry that contain error code.
 *
 * @param[in]  state            struct efa_resource that is managed by the framework
 * @param[in]  local_host_id    Local(sender) host id
 * @param[in]  peer_host_id     Peer(receiver) host id
 * @param[in]  vendor_error     Vendor error returned by ibv_read_vendor_err
 */
static void test_rdm_cq_read_bad_send_status(struct efa_resource *resource,
                                             uint64_t local_host_id, uint64_t peer_host_id,
                                             int vendor_error, int efa_error)
{
	const char *strerror;
	fi_addr_t addr;
	int ret, err;
	char host_id_str[] = "xxxxx host id: i-01234567812345678";
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	struct efa_ep_addr raw_addr;
	struct efa_unit_test_buff send_buff;
	struct fi_cq_data_entry cq_entry;
	struct fi_cq_err_entry cq_err_entry = {0};
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;

	/* disable shm to force using efa device to send */
	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_unit_test_buff_construct(&send_buff, resource, 4096 /* buff_size */);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_ep->host_id = local_host_id;

	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0 /* flags */, NULL /* context */);
	assert_int_equal(ret, 1);

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	assert_non_null(peer);
	peer->host_id = peer_host_id;

	g_efa_unit_test_mocks.efa_qp_wr_start = &efa_mock_efa_qp_wr_start_no_op;
	/* this mock will save the send work request (wr) in a global list */
	g_efa_unit_test_mocks.efa_qp_wr_send = &efa_mock_efa_qp_wr_send_save_wr;
	g_efa_unit_test_mocks.efa_qp_wr_set_sge_list = &efa_mock_efa_qp_wr_set_sge_list_no_op;
	g_efa_unit_test_mocks.efa_qp_wr_set_ud_addr = &efa_mock_efa_qp_wr_set_ud_addr_no_op;
	g_efa_unit_test_mocks.efa_qp_wr_complete = &efa_mock_efa_qp_wr_complete_no_op;
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);

	err = fi_send(resource->ep, send_buff.buff, send_buff.size, fi_mr_desc(send_buff.mr), addr, NULL /* context */);
	assert_int_equal(err, 0);
	/* fi_send() called efa_mock_efa_qp_wr_send_save_wr(), which saved one send_wr in g_ibv_submitted_wr_id_vec */
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	/* this mock will set ibv_cq_ex->wr_id to the wr_id f the head of global send_wr,
	 * and set ibv_cq_ex->status to mock value */
	g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_use_saved_send_wr_with_mock_status;
	g_efa_unit_test_mocks.efa_ibv_cq_end_poll = &efa_mock_efa_ibv_cq_end_poll_check_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_opcode = &efa_mock_efa_ibv_cq_wc_read_opcode_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_vendor_err = &efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_qp_num = &efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock;
	will_return(efa_mock_efa_ibv_cq_start_poll_use_saved_send_wr_with_mock_status, IBV_WC_GENERAL_ERR);
	will_return(efa_mock_efa_ibv_cq_end_poll_check_mock, NULL);
	will_return(efa_mock_efa_ibv_cq_wc_read_opcode_return_mock, IBV_WC_SEND);
	will_return(efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock, vendor_error);
	will_return(efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock, efa_rdm_ep->base_ep.qp->qp_num);
	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	/* fi_cq_read() called efa_mock_efa_ibv_cq_start_poll_use_saved_send_wr(), which pulled one send_wr from g_ibv_submitted_wr_idv=_vec */
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	assert_int_equal(ret, -FI_EAVAIL);

	/* Allocate memory to read CQ error */
	cq_err_entry.err_data_size = EFA_ERROR_MSG_BUFFER_LENGTH;
	cq_err_entry.err_data = malloc(cq_err_entry.err_data_size);
	assert_non_null(cq_err_entry.err_data);

	ret = fi_cq_readerr(resource->cq, &cq_err_entry, 0);
	assert_true(cq_err_entry.err_data_size > 0);
	strerror = fi_cq_strerror(resource->cq, cq_err_entry.prov_errno, cq_err_entry.err_data, NULL, 0);

	assert_int_equal(ret, 1);
	assert_int_not_equal(cq_err_entry.err, FI_SUCCESS);
	assert_int_equal(cq_err_entry.prov_errno, efa_error);

	/* Reset value */
	memset(host_id_str, 0, sizeof(host_id_str));

	/* Set expected host id */
	if (local_host_id) {
		snprintf(host_id_str, sizeof(host_id_str), "My host id: i-%017lx", local_host_id);
	} else {
		strcpy(host_id_str, "My host id: N/A");
	}
	/* Look for My host id */
	assert_non_null(strstr(strerror, host_id_str));

	/* Reset value */
	memset(host_id_str, 0, sizeof(host_id_str));

	/* Set expected host id */
	if (peer_host_id) {
		snprintf(host_id_str, sizeof(host_id_str), "Peer host id: i-%017lx", peer_host_id);
	} else {
		strcpy(host_id_str, "Peer host id: N/A");
	}
	/* Look for peer host id */
	assert_non_null(strstr(strerror, host_id_str));
	efa_unit_test_buff_destruct(&send_buff);

	/* reset the mocked cq before it's polled by ep close */
	will_return_always(efa_mock_efa_ibv_cq_start_poll_use_saved_send_wr_with_mock_status, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief test that RDM CQ's fi_cq_read()/fi_cq_readerr() works properly when rdma-core returns
 * unresponsive receiver error for send.
 *
 * When send operation failed, fi_cq_read() should return -FI_EAVAIL, which means error available.
 * then user should call fi_cq_readerr() to get an error CQ entry that contain error code.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_rdm_cq_read_bad_send_status_unresponsive_receiver(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	test_rdm_cq_read_bad_send_status(resource,
					 0x1234567812345678, 0x8765432187654321,
					 EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE, FI_EFA_ERR_UNESTABLISHED_RECV_UNRESP);
}

/**
 * @brief test that RDM CQ's fi_cq_read()/fi_cq_readerr() works properly when rdma-core returns
 * unresponsive receiver error for send. This test verifies peer host id is printed correctly if it is unknown.
 *
 * When send operation failed, fi_cq_read() should return -FI_EAVAIL, which means error available.
 * then user should call fi_cq_readerr() to get an error CQ entry that contain error code.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_rdm_cq_read_bad_send_status_unresponsive_receiver_missing_peer_host_id(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	test_rdm_cq_read_bad_send_status(resource,
					 0x1234567812345678, 0,
					 EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE, FI_EFA_ERR_UNESTABLISHED_RECV_UNRESP);
}

/**
 * @brief test that RDM CQ's fi_cq_read()/fi_cq_readerr() works properly when rdma-core returns
 * unreachable remote error for send.
 *
 * When send operation failed, fi_cq_read() should return -FI_EAVAIL, which means error available.
 * then user should call fi_cq_readerr() to get an error CQ entry that contain error code.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_rdm_cq_read_bad_send_status_unreachable_receiver(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	test_rdm_cq_read_bad_send_status(resource,
					 0x1234567812345678, 0x8765432187654321,
					 EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);
}

/**
 * @brief test that RDM CQ's fi_cq_read()/fi_cq_readerr() works properly when rdma-core returns
 * invalid qpn error for send.
 *
 * When send operation failed, fi_cq_read() should return -FI_EAVAIL, which means error available.
 * then user should call fi_cq_readerr() to get an error CQ entry that contain error code.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_rdm_cq_read_bad_send_status_invalid_qpn(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	test_rdm_cq_read_bad_send_status(resource,
					 0x1234567812345678, 0x8765432187654321,
					 EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_DEST_QPN, EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_DEST_QPN);
}

/**
 * @brief test that RDM CQ's fi_cq_read()/fi_cq_readerr() works properly when rdma-core returns
 * message too long error for send.
 *
 * When send operation failed, fi_cq_read() should return -FI_EAVAIL, which means error available.
 * then user should call fi_cq_readerr() to get an error CQ entry that contain error code.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_rdm_cq_read_bad_send_status_message_too_long(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	test_rdm_cq_read_bad_send_status(resource,
					 0x1234567812345678, 0x8765432187654321,
					 EFA_IO_COMP_STATUS_LOCAL_ERROR_BAD_LENGTH, EFA_IO_COMP_STATUS_LOCAL_ERROR_BAD_LENGTH);
}

/**
 * @brief Test the error handling for a handshake tx completion err
 * TODO: Cover the RNR case test where the handshake packet should be queued
 * @param state test resource
 * @param prov_errno rdma core vendor error
 * @param expect_eq_err whether an eq error is expected
 */
static
void test_rdm_cq_handshake_bad_send_status_impl(struct efa_resource **state, int prov_errno, bool expect_eq_err)
{
	fi_addr_t peer_addr = 0;
	int ret;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_rdm_peer *peer;
	struct efa_resource *resource = *state;
	struct efa_unit_test_handshake_pkt_attr pkt_attr = {0};
	struct fi_cq_data_entry cq_entry;
	struct fi_eq_err_entry eq_err_entry;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_cq *efa_rdm_cq;
	struct efa_ibv_cq *ibv_cq;
	struct efa_rdm_ope *txe;

	/* disable shm to force using efa device to send */
	efa_unit_test_resource_construct_rdm_shm_disabled(resource);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_cq = container_of(resource->cq, struct efa_rdm_cq, efa_cq.util_cq.cq_fid.fid);
	ibv_cq = &efa_rdm_cq->efa_cq.ibv_cq;

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

	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_tx_pkt_pool, EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pkt_entry);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->internal_flags |= EFA_RDM_OPE_INTERNAL;
	pkt_entry->ope = txe;
	pkt_entry->peer = peer;

	pkt_attr.connid = raw_addr.qkey;
	pkt_attr.host_id = 0x8765432187654321;
	pkt_attr.device_version = 0xefa0;
	efa_unit_test_handshake_pkt_construct(pkt_entry, &pkt_attr);

	/* Setup CQ */
	g_efa_unit_test_mocks.efa_ibv_cq_end_poll = &efa_mock_efa_ibv_cq_end_poll_check_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_opcode = &efa_mock_efa_ibv_cq_wc_read_opcode_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_qp_num = &efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_vendor_err = &efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_return_mock;
	ibv_cq->ibv_cq_ex->wr_id = (uintptr_t)pkt_entry;

	/* Mock cq to simulate the send comp error */
	will_return(efa_mock_efa_ibv_cq_end_poll_check_mock, NULL);
	will_return(efa_mock_efa_ibv_cq_wc_read_opcode_return_mock, IBV_WC_SEND);
	will_return(efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock, efa_rdm_ep->base_ep.qp->qp_num);
	will_return(efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock, prov_errno);
	will_return(efa_mock_efa_ibv_cq_start_poll_return_mock, IBV_WC_SUCCESS);

	efa_rdm_ep->efa_outstanding_tx_ops = 1;
	ibv_cq->ibv_cq_ex->status = IBV_WC_GENERAL_ERR;
	ibv_cq->ibv_cq_ex->wr_id = (uintptr_t)pkt_entry;

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	/* HANDSHAKE packet does not generate completion entry or error*/
	assert_int_equal(ret, -FI_EAGAIN);

	ret = fi_eq_readerr(resource->eq, &eq_err_entry, 0);
	if (expect_eq_err) {
		assert_int_equal(ret, sizeof(eq_err_entry));
		assert_int_equal(eq_err_entry.prov_errno, prov_errno);
	} else {
		assert_int_equal(ret, -FI_EAGAIN);
	}

	/* reset the mocked cq before it's polled by ep close */
	will_return_always(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

void test_rdm_cq_handshake_bad_send_status_bad_qpn(struct efa_resource **state)
{
	test_rdm_cq_handshake_bad_send_status_impl(state, EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_DEST_QPN, false);
}

void test_rdm_cq_handshake_bad_send_status_unresp_remote(struct efa_resource **state)
{
	test_rdm_cq_handshake_bad_send_status_impl(state, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE, false);
}

void test_rdm_cq_handshake_bad_send_status_unreach_remote(struct efa_resource **state)
{
	test_rdm_cq_handshake_bad_send_status_impl(state, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE, false);
}

void test_rdm_cq_handshake_bad_send_status_remote_abort(struct efa_resource **state)
{
	test_rdm_cq_handshake_bad_send_status_impl(state, EFA_IO_COMP_STATUS_REMOTE_ERROR_ABORT, false);
}

void test_rdm_cq_handshake_bad_send_status_unsupported_op(struct efa_resource **state)
{
	test_rdm_cq_handshake_bad_send_status_impl(state, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNSUPPORTED_OP, true);
}


/**
 * @brief Verify that unsolicited write recv status is tracked in efa_ibv_cq correctly
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_ibv_cq_unsolicited_write_recv_status(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;


	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid);

	assert_true(efa_use_unsolicited_write_recv() == efa_cq->ibv_cq.unsolicited_write_recv_enabled);
}

/**
 * @brief verify that fi_cq_read/fi_cq_readerr works properly when rdma-core return bad status for recv.
 *
 * When an ibv_post_recv() operation failed, no data was received. Therefore libfabric cannot
 * find the corresponding RX operation to write a CQ error. It will write an EQ error instead.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_ibv_cq_ex_read_bad_recv_status(struct efa_resource **state)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_resource *resource = *state;
	struct efa_rdm_pke *pkt_entry;
	struct fi_cq_data_entry cq_entry;
	struct fi_eq_err_entry eq_err_entry;
	int ret;
	struct efa_rdm_cq *efa_rdm_cq;
	struct efa_ibv_cq *ibv_cq;
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	int err, numaddr;


	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

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

	efa_rdm_cq = container_of(resource->cq, struct efa_rdm_cq, efa_cq.util_cq.cq_fid.fid);
	ibv_cq = &efa_rdm_cq->efa_cq.ibv_cq;

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	numaddr = fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL);
	assert_int_equal(numaddr, 1);

	g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_end_poll = &efa_mock_efa_ibv_cq_end_poll_check_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_opcode = &efa_mock_efa_ibv_cq_wc_read_opcode_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_vendor_err = &efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_qp_num = &efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock;

	will_return(efa_mock_efa_ibv_cq_start_poll_return_mock, 0);
	will_return(efa_mock_efa_ibv_cq_end_poll_check_mock, NULL);
	/* efa_mock_efa_ibv_cq_wc_read_opcode_return_mock() will be called once in release mode,
	 * but will be called twice in debug mode. because there is an assertion that called ibv_read_opcode(),
	 * therefore use will_return_always()
	 */
	will_return_always(efa_mock_efa_ibv_cq_wc_read_opcode_return_mock, IBV_WC_RECV);
	will_return_always(efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock, efa_rdm_ep->base_ep.qp->qp_num);
	will_return(efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE);
	/* the recv error will not populate to application cq because it's an EFA internal error and
	 * and not related to any application recv. Currently we can only read the error from eq.
	 */
	ibv_cq->ibv_cq_ex->wr_id = (uintptr_t)pkt_entry;
	ibv_cq->ibv_cq_ex->status = IBV_WC_GENERAL_ERR;

#if HAVE_CAPS_UNSOLICITED_WRITE_RECV
	if (ibv_cq->unsolicited_write_recv_enabled) {
		g_efa_unit_test_mocks.efa_ibv_cq_wc_is_unsolicited = &efa_mock_efa_ibv_cq_wc_is_unsolicited_return_mock;
		will_return(efa_mock_efa_ibv_cq_wc_is_unsolicited_return_mock, false);
	}
#endif

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, -FI_EAGAIN);

	ret = fi_eq_readerr(resource->eq, &eq_err_entry, 0);
	assert_int_equal(ret, sizeof(eq_err_entry));
	assert_int_not_equal(eq_err_entry.err, FI_SUCCESS);

	/* TODO - Fix pkt recv error path */
	assert_int_equal(eq_err_entry.prov_errno, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE);

	/* reset the mocked cq before it's polled by ep close */
	will_return_always(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief verify that fi_cq_read/fi_eq_read works properly when rdma-core return bad status for
 * recv rdma with imm.
 *
 * When getting a wc error of op code IBV_WC_RECV_RDMA_WITH_IMM, libfabric cannot find the
 * corresponding application operation to write a cq error.
 * It will write an EQ error instead.
 *
 * @param[in]	state					struct efa_resource that is managed by the framework
 * @param[in]	use_unsolicited_recv	whether to use unsolicited write recv
 */
void test_ibv_cq_ex_read_bad_recv_rdma_with_imm_status_impl(struct efa_resource **state, bool use_unsolicited_recv)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_resource *resource = *state;
	struct fi_cq_data_entry cq_entry;
	struct fi_eq_err_entry eq_err_entry;
	int ret;
	struct efa_rdm_cq *efa_rdm_cq;
	struct efa_ibv_cq *ibv_cq;
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	int err, numaddr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	efa_rdm_cq = container_of(resource->cq, struct efa_rdm_cq, efa_cq.util_cq.cq_fid.fid);
	ibv_cq = &efa_rdm_cq->efa_cq.ibv_cq;

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	numaddr = fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL);
	assert_int_equal(numaddr, 1);

	g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_end_poll = &efa_mock_efa_ibv_cq_end_poll_check_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_opcode = &efa_mock_efa_ibv_cq_wc_read_opcode_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_vendor_err = &efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_qp_num = &efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock;

	will_return(efa_mock_efa_ibv_cq_start_poll_return_mock, 0);
	will_return(efa_mock_efa_ibv_cq_end_poll_check_mock, NULL);
	/* efa_mock_efa_ibv_cq_wc_read_opcode_return_mock() will be called once in release mode,
	 * but will be called twice in debug mode. because there is an assertion that called ibv_read_opcode(),
	 * therefore use will_return_always()
	 */
	will_return_always(efa_mock_efa_ibv_cq_wc_read_opcode_return_mock, IBV_WC_RECV_RDMA_WITH_IMM);
	will_return_always(efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock, efa_rdm_ep->base_ep.qp->qp_num);
	will_return(efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock, EFA_IO_COMP_STATUS_FLUSHED);


#if HAVE_CAPS_UNSOLICITED_WRITE_RECV
	if (use_unsolicited_recv) {
		g_efa_unit_test_mocks.efa_ibv_cq_wc_is_unsolicited = &efa_mock_efa_ibv_cq_wc_is_unsolicited_return_mock;
		ibv_cq->unsolicited_write_recv_enabled = true;
		will_return(efa_mock_efa_ibv_cq_wc_is_unsolicited_return_mock, true);
		ibv_cq->ibv_cq_ex->wr_id = 0;
	} else {
		/*
		 * For solicited write recv, it will consume an internal rx pkt
		 */
		ibv_cq->unsolicited_write_recv_enabled = false;
		struct efa_rdm_pke *pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool, EFA_RDM_PKE_FROM_EFA_RX_POOL);
		assert_non_null(pkt_entry);
		efa_rdm_ep->efa_rx_pkts_posted = efa_rdm_ep_get_rx_pool_size(efa_rdm_ep);
		ibv_cq->ibv_cq_ex->wr_id = (uintptr_t)pkt_entry;
	}
#else
	/*
	 * Always test with solicited recv
	 */
	ibv_cq->unsolicited_write_recv_enabled = false;
	struct efa_rdm_pke *pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool, EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	efa_rdm_ep->efa_rx_pkts_posted = efa_rdm_ep_get_rx_pool_size(efa_rdm_ep);
	ibv_cq->ibv_cq_ex->wr_id = (uintptr_t)pkt_entry;
#endif
	/* the recv rdma with imm will not populate to application cq because it's an EFA internal error and
	 * and not related to any application operations. Currently we can only read the error from eq.
	 */
	ibv_cq->ibv_cq_ex->status = IBV_WC_GENERAL_ERR;
	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, -FI_EAGAIN);

	ret = fi_eq_readerr(resource->eq, &eq_err_entry, 0);
	assert_int_equal(ret, sizeof(eq_err_entry));
	assert_int_not_equal(eq_err_entry.err, FI_SUCCESS);
	assert_int_equal(eq_err_entry.prov_errno, EFA_IO_COMP_STATUS_FLUSHED);

	/* reset the mocked cq before it's polled by ep close */
	will_return_always(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

void test_ibv_cq_ex_read_bad_recv_rdma_with_imm_status_use_unsolicited_recv(struct efa_resource **state)
{
	test_ibv_cq_ex_read_bad_recv_rdma_with_imm_status_impl(state, true);
}

void test_ibv_cq_ex_read_bad_recv_rdma_with_imm_status_use_solicited_recv(struct efa_resource **state)
{
	test_ibv_cq_ex_read_bad_recv_rdma_with_imm_status_impl(state, false);
}

/**
 * @brief verify that fi_cq_read/fi_cq_readerr works properly when ibv_start_poll failed.
 *
 * When an ibv_start_poll() failed, it currently means the QP associated with the CQE is
 * destroyed. According to libfabric man page, such cqe should be ignored and not reported
 * to application cqs.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_ibv_cq_ex_read_failed_poll(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fi_cq_data_entry cq_entry;
	struct fi_cq_err_entry cq_err_entry;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_end_poll = &efa_mock_efa_ibv_cq_end_poll_check_mock;

	will_return(efa_mock_efa_ibv_cq_start_poll_return_mock, EINVAL);

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, -FI_EAGAIN);

	ret = fi_cq_readerr(resource->cq, &cq_err_entry, 0);
	assert_int_equal(ret, -FI_EAGAIN);

	/* reset the mocked cq before it's polled by ep close */
	will_return_always(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief Test efa_rdm_cq_open() handles rdma-core CQ creation failure gracefully
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_rdm_cq_create_error_handling(struct efa_resource **state)
{

	struct efa_resource *resource = *state;
	struct ibv_device **ibv_device_list;
	struct efa_device efa_device = {0};
	struct efa_domain *efa_domain = NULL;
	struct verbs_context *vctx = NULL;
	struct fi_cq_attr cq_attr = {0};
	int ret, total_device_cnt, i;

	ibv_device_list = ibv_get_device_list(&total_device_cnt);
	if (ibv_device_list == NULL) {
		skip();
		return;
	}

	for (i = 0; i < total_device_cnt; i++) {
		ret = efa_device_construct_gid(&efa_device, ibv_device_list[i]);
		if (ret)
			continue;
		ret = efa_device_construct_data(&efa_device, ibv_device_list[i]);
		if (!ret)
			break;
	}

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
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

	assert_int_not_equal(fi_cq_open(resource->domain, &cq_attr, &resource->cq, NULL), 0);
	/* set cq as NULL to avoid double free by fi_close in cleanup stage */
	resource->cq = NULL;
	ibv_close_device(efa_device.ibv_ctx);
	ibv_free_device_list(ibv_device_list);
}

/**
 * @brief get the length of the ibv_cq_poll_list for a given efa_rdm_cq
 *
 * @param cq_fid cq fid
 * @return int the length of the ibv_cq_poll_list
 */
static
int test_efa_rdm_cq_get_ibv_cq_poll_list_length(struct fid_cq *cq_fid)
{
	struct efa_rdm_cq *cq;

	cq = container_of(cq_fid, struct efa_rdm_cq, efa_cq.util_cq.cq_fid.fid);
	return efa_unit_test_get_dlist_length(&cq->ibv_cq_poll_list);
}

/**
 * @brief Check the length of ibv_cq_poll_list when 1 cq is bind to 1 ep
 * as both tx/rx cq.
 *
 * @param state struct efa_resource that is managed by the framework
 */
void test_efa_rdm_cq_ibv_cq_poll_list_same_tx_rx_cq_single_ep(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	/* efa_unit_test_resource_construct binds single OFI CQ as both tx/rx cq of ep */
	assert_int_equal(test_efa_rdm_cq_get_ibv_cq_poll_list_length(resource->cq), 1);
}

/**
 * @brief Check the length of ibv_cq_poll_list when separate tx/rx cq is bind to 1 ep.
 *
 * @param state struct efa_resource that is managed by the framework
 */
void test_efa_rdm_cq_ibv_cq_poll_list_separate_tx_rx_cq_single_ep(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_cq *txcq, *rxcq;
	struct fi_cq_attr cq_attr = {0};

	efa_unit_test_resource_construct_no_cq_and_ep_not_enabled(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	assert_int_equal(fi_cq_open(resource->domain, &cq_attr, &txcq, NULL), 0);

	assert_int_equal(fi_ep_bind(resource->ep, &txcq->fid, FI_SEND), 0);

	assert_int_equal(fi_cq_open(resource->domain, &cq_attr, &rxcq, NULL), 0);

	assert_int_equal(fi_ep_bind(resource->ep, &rxcq->fid, FI_RECV), 0);

	assert_int_equal(fi_enable(resource->ep), 0);

	assert_int_equal(test_efa_rdm_cq_get_ibv_cq_poll_list_length(txcq), 2);

	assert_int_equal(test_efa_rdm_cq_get_ibv_cq_poll_list_length(rxcq), 2);

	/* ep must be closed before cq/av/eq... */
	fi_close(&resource->ep->fid);
	resource->ep = NULL;
	fi_close(&txcq->fid);
	fi_close(&rxcq->fid);
}

void test_efa_rdm_cq_post_initial_rx_pkts(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_cq *efa_rdm_cq;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_cq = container_of(resource->cq, struct efa_rdm_cq, efa_cq.util_cq.cq_fid.fid);

	/* At this time, rx pkts are not growed and posted */
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_to_post, 0);
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_posted, 0);
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_held, 0);

	/* cq read need to scan the ep list since a ep is bind */
	assert_true(efa_rdm_cq->need_to_scan_ep_list);
	fi_cq_read(resource->cq, NULL, 0);

	/* At this time, rx pool size number of rx pkts are posted */
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_posted, efa_rdm_ep_get_rx_pool_size(efa_rdm_ep));
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_to_post, 0);
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_held, 0);

	/* scan is done */
	assert_false(efa_rdm_cq->need_to_scan_ep_list);
}

void test_efa_rdm_cq_before_ep_enable(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_ep *ep;

	/* TODO: allow shm when shm fixed its bug that
	 cq read cannot be called before ep enable */
	efa_unit_test_resource_construct_rdm_shm_disabled(resource);

	assert_int_equal(fi_endpoint(resource->domain, resource->info, &ep, NULL), 0);

	assert_int_equal(fi_ep_bind(ep, &resource->cq->fid, FI_SEND | FI_RECV), 0);

	/* cq read should return EAGAIN as its empty */
	assert_int_equal(fi_cq_read(resource->cq, NULL, 0), -FI_EAGAIN);

	assert_int_equal(fi_close(&ep->fid), 0);
}

#if HAVE_EFADV_CQ_EX
/**
 * @brief Construct an RDM endpoint and receive an eager MSG RTM packet.
 * Simulate EFA device by setting peer AH to unknown and make sure the
 * endpoint recovers the peer address iff(if and only if) the peer is
 * inserted to AV.
 *
 * @param resource		struct efa_resource that is managed by the framework
 * @param remove_peer	Boolean value that indicates if the peer was removed explicitly
 * @param support_efadv_cq	Boolean value that indicates if EFA device supports EFA DV CQ
 */
static void test_impl_ibv_cq_ex_read_unknow_peer_ah(struct efa_resource *resource, bool remove_peer, bool support_efadv_cq)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct fi_cq_data_entry cq_entry;
	struct efa_unit_test_eager_rtm_pkt_attr pkt_attr = {0};
	struct efa_unit_test_buff recv_buff;
	int ret;
	struct efa_rdm_cq *efa_rdm_cq;
	struct efa_ibv_cq *ibv_cq;
	struct ibv_cq_ex *ibv_cqx;

	/*
	 * Always use mocked efadv_create_cq instead of the real one.
	 * Otherwise the test is undeterministic depending on the host kernel:
	 * - If the kernel supports EFA DV CQ and we set support_efadv_cq = true, then the test will pass
	 * - If the kernel does NOT support EFA DV CQ and we set support_efadv_cq = true, then the test will fail
	 */
	if (support_efadv_cq) {
		g_efa_unit_test_mocks.efadv_create_cq = &efa_mock_efadv_create_cq_with_ibv_create_cq_ex;
		expect_function_call(efa_mock_efadv_create_cq_with_ibv_create_cq_ex);
	} else {
		g_efa_unit_test_mocks.efadv_create_cq = &efa_mock_efadv_create_cq_set_eopnotsupp_and_return_null;
		expect_function_call(efa_mock_efadv_create_cq_set_eopnotsupp_and_return_null);
	}

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_cq = container_of(resource->cq, struct efa_rdm_cq, efa_cq.util_cq.cq_fid.fid);
	ibv_cq = &efa_rdm_cq->efa_cq.ibv_cq;
	ibv_cqx = ibv_cq->ibv_cq_ex;

	/* Construct a minimal recv buffer */
	efa_unit_test_buff_construct(&recv_buff, resource, efa_rdm_ep->min_multi_recv_size);

	/* Create and register a fake peer */
	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 0;
	raw_addr.qkey = 0x1234;

	struct efa_rdm_peer *peer;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL);
	assert_int_equal(ret, 1);

	/* Skip handshake */
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	assert_non_null(peer);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_SENT;

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

	pkt_attr.msg_id = 0;
	pkt_attr.connid = raw_addr.qkey;
	/* Packet type must be in [EFA_RDM_REQ_PKT_BEGIN, EFA_RDM_EXTRA_REQ_PKT_END) */
	efa_unit_test_eager_msgrtm_pkt_construct(pkt_entry, &pkt_attr);

	/* Setup CQ */
	ibv_cqx->wr_id = (uintptr_t)pkt_entry;
	g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_next_poll = &efa_mock_efa_ibv_cq_next_poll_check_function_called_and_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_end_poll = &efa_mock_efa_ibv_cq_end_poll_check_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_slid = &efa_mock_efa_ibv_cq_wc_read_slid_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_byte_len = &efa_mock_efa_ibv_cq_wc_read_byte_len_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_opcode = &efa_mock_efa_ibv_cq_wc_read_opcode_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_qp_num = &efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_wc_flags = &efa_mock_efa_ibv_cq_wc_read_wc_flags_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_src_qp = &efa_mock_efa_ibv_cq_wc_read_src_qp_return_mock;

	if (support_efadv_cq) {
		g_efa_unit_test_mocks.efa_ibv_cq_wc_read_sgid = &efa_mock_efa_ibv_cq_wc_read_sgid_return_zero_code_and_expect_next_poll_and_set_gid;

		/* Return unknown AH from efadv */
		will_return(efa_mock_efa_ibv_cq_wc_read_sgid_return_zero_code_and_expect_next_poll_and_set_gid, raw_addr.raw);
	} else {
		expect_function_call(efa_mock_efa_ibv_cq_next_poll_check_function_called_and_return_mock);
	}

	/* Read 1 entry with unknown AH */
	will_return(efa_mock_efa_ibv_cq_start_poll_return_mock, 0);
	will_return(efa_mock_efa_ibv_cq_next_poll_check_function_called_and_return_mock, ENOENT);
	will_return(efa_mock_efa_ibv_cq_end_poll_check_mock, NULL);
	will_return(efa_mock_efa_ibv_cq_wc_read_slid_return_mock, 0xffff); // slid=0xffff(-1) indicates an unknown AH
	will_return(efa_mock_efa_ibv_cq_wc_read_byte_len_return_mock, pkt_entry->pkt_size);
	will_return_maybe(efa_mock_efa_ibv_cq_wc_read_opcode_return_mock, IBV_WC_RECV);
	will_return_maybe(efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock, efa_rdm_ep->base_ep.qp->qp_num);
	will_return_maybe(efa_mock_efa_ibv_cq_wc_read_wc_flags_return_mock, 0);
	will_return_maybe(efa_mock_efa_ibv_cq_wc_read_src_qp_return_mock, raw_addr.qpn);

	/* Post receive buffer */
	ret = fi_recv(resource->ep, recv_buff.buff, recv_buff.size, fi_mr_desc(recv_buff.mr), peer_addr, NULL /* context */);
	assert_int_equal(ret, 0);

	if (remove_peer) {
		ret = fi_av_remove(resource->av, &peer_addr, 1, 0);
		assert_int_equal(ret, 0);
	}

	ret = fi_cq_read(resource->cq, &cq_entry, 1);

	if (remove_peer || !support_efadv_cq) {
		/* Ignored WC because the peer is removed, or EFA device does not support extended CQ */
		assert_int_equal(ret, -FI_EAGAIN);
	}
	else {
		/* Found 1 matching rxe */
		assert_int_equal(ret, 1);
	}

	efa_unit_test_buff_destruct(&recv_buff);

	/* reset the mocked cq before it's polled by ep close */
	will_return_always(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);

	/* When the peer is removed, we add it implicitly and try to send a
	 * handshake packet. So we need to reset efa_outstanding_tx_ops before
	 * closing the endpoint. */
	if (remove_peer)
		efa_rdm_ep->efa_outstanding_tx_ops = 0;
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief Verify that RDM endpoint fi_cq_read recovers unknown peer AH
 * by querying efadv to get raw address.
 * A fake peer is registered in AV. The endpoint receives a packet from it,
 * for which the EFA device returns an unknown AH. The endpoint will retrieve
 * the peer's raw address using efadv verbs, and recover it's AH using
 * Raw:QPN:QKey.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_ibv_cq_ex_read_recover_forgotten_peer_ah(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	test_impl_ibv_cq_ex_read_unknow_peer_ah(resource, false, true);
}

/**
 * @brief Verify that RDM endpoint falls back to ibv_create_cq_ex if rdma-core
 * provides efadv_create_cq verb but EFA device does not support EFA DV CQ.
 * In this case the endpoint will not attempt to recover a forgotten peer's address.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_rdm_fallback_to_ibv_create_cq_ex_cq_read_ignore_forgotton_peer(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	test_impl_ibv_cq_ex_read_unknow_peer_ah(resource, false, false);
}

/**
 * @brief Verify that RDM endpoint progress engine ignores unknown peer AH
 * if the peer is not registered in AV, e.g. removed.
 * The endpoint receives a packet from an alien peer, which corresponds to
 * an unknown AH. The endpoint attempts to look up the AH for the peer but
 * was rightly unable to, thus ignoring the packet.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_ibv_cq_ex_read_ignore_removed_peer(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	test_impl_ibv_cq_ex_read_unknow_peer_ah(resource, true, true);
}
#else
void test_ibv_cq_ex_read_recover_forgotten_peer_ah()
{
	skip();
}
void test_rdm_fallback_to_ibv_create_cq_ex_cq_read_ignore_forgotton_peer()
{
	skip();
}
void test_ibv_cq_ex_read_ignore_removed_peer()
{
	skip();
}
#endif

static void test_efa_cq_read_prep(struct efa_resource *resource,
			     int ibv_wc_opcode, int status, int vendor_error,
			     struct efa_context *ctx, int wc_flags,
			     bool is_unsolicited_write_recv)
{
	int ret;
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	struct efa_ep_addr raw_addr;
	struct efa_ibv_cq *ibv_cq;
	struct ibv_cq_ex *ibv_cqx;
	struct efa_cq *efa_cq;
	struct efa_base_ep *base_ep;
	fi_addr_t addr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0 /* flags */, NULL /* context */);
	assert_int_equal(ret, 1);
	if (ctx)
		ctx->addr = addr;

	base_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);
	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid.fid);
	ibv_cq = &efa_cq->ibv_cq;
	ibv_cqx = ibv_cq->ibv_cq_ex;

	/* Make wr_id as 0 for unsolicited write recv as a stress test */
	ibv_cqx->wr_id = is_unsolicited_write_recv ? 0 : (uintptr_t) ctx;
	ibv_cq->unsolicited_write_recv_enabled = is_unsolicited_write_recv;
	ibv_cqx->status = status;

    /* Set up the mock operations */
    g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_return_mock;
    g_efa_unit_test_mocks.efa_ibv_cq_next_poll = &efa_mock_efa_ibv_cq_next_poll_return_mock;
    g_efa_unit_test_mocks.efa_ibv_cq_end_poll = &efa_mock_efa_ibv_cq_end_poll_check_mock;
    g_efa_unit_test_mocks.efa_ibv_cq_wc_read_opcode = &efa_mock_efa_ibv_cq_wc_read_opcode_return_mock;
    g_efa_unit_test_mocks.efa_ibv_cq_wc_read_vendor_err = &efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock;
    g_efa_unit_test_mocks.efa_ibv_cq_wc_read_wc_flags = &efa_mock_efa_ibv_cq_wc_read_wc_flags_return_mock;
    g_efa_unit_test_mocks.efa_ibv_cq_wc_read_imm_data = &efa_mock_efa_ibv_cq_wc_read_imm_data_return_mock;
    g_efa_unit_test_mocks.efa_ibv_cq_wc_read_qp_num = &efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock;
    g_efa_unit_test_mocks.efa_ibv_cq_wc_read_src_qp = &efa_mock_efa_ibv_cq_wc_read_src_qp_return_mock;
    g_efa_unit_test_mocks.efa_ibv_cq_wc_read_slid = &efa_mock_efa_ibv_cq_wc_read_slid_return_mock;
    g_efa_unit_test_mocks.efa_ibv_cq_wc_read_byte_len = &efa_mock_efa_ibv_cq_wc_read_byte_len_return_mock;

    will_return(efa_mock_efa_ibv_cq_start_poll_return_mock, 0);
    will_return_maybe(efa_mock_efa_ibv_cq_next_poll_return_mock, ENOENT);
    will_return_maybe(efa_mock_efa_ibv_cq_end_poll_check_mock, NULL);
    will_return_maybe(efa_mock_efa_ibv_cq_wc_read_opcode_return_mock, ibv_wc_opcode);
    will_return_maybe(efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock, vendor_error);
    will_return_maybe(efa_mock_efa_ibv_cq_wc_read_wc_flags_return_mock, wc_flags);
    will_return_maybe(efa_mock_efa_ibv_cq_wc_read_imm_data_return_mock, 0x1);
    will_return_maybe(efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock, base_ep->qp->qp_num);
    will_return_maybe(efa_mock_efa_ibv_cq_wc_read_byte_len_return_mock, 4096);
    will_return_maybe(efa_mock_efa_ibv_cq_wc_read_slid_return_mock, efa_av_addr_to_conn(base_ep->av, addr)->ah->ahn);
    will_return_maybe(efa_mock_efa_ibv_cq_wc_read_src_qp_return_mock, raw_addr.qpn);


#if HAVE_CAPS_UNSOLICITED_WRITE_RECV
    if (ibv_cq->unsolicited_write_recv_enabled) {
        g_efa_unit_test_mocks.efa_ibv_cq_wc_is_unsolicited = &efa_mock_efa_ibv_cq_wc_is_unsolicited_return_mock;
        will_return_maybe(efa_mock_efa_ibv_cq_wc_is_unsolicited_return_mock, is_unsolicited_write_recv);
    }
#endif
}

/**
 * @brief test EFA CQ's fi_cq_read() works properly when rdma-core return
 * success status for send operation without wr_id (inject). In this case
 * no completion should be generated.
 */
void test_efa_cq_read_no_completion(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fi_cq_data_entry cq_entry;
	int ret;

	test_efa_cq_read_prep(resource, IBV_WC_SEND, IBV_WC_SUCCESS, 0,
			 NULL, 0, false);

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, -FI_EAGAIN);

	/* reset the mocked cq before it's polled by ep close */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief test EFA CQ's fi_cq_read() works properly when rdma-core return
 * success status for send operation.
 */
void test_efa_cq_read_send_success(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_context *efa_context;
	struct fi_context2 ctx;
	struct fi_cq_data_entry cq_entry = {0};
	int ret;

	efa_context = (struct efa_context *) &ctx;
	efa_context->completion_flags = FI_SEND | FI_MSG;

	test_efa_cq_read_prep(resource, IBV_WC_SEND, IBV_WC_SUCCESS, 0,
			 efa_context, 0, false);

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, 1);

	assert_true(efa_context == cq_entry.op_context);
	assert_true(cq_entry.flags == efa_context->completion_flags);

	/* reset the mocked cq before it's polled by ep close */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief test EFA CQ's fi_cq_read() works properly when rdma-core return
 * success status for senddata operation.
 */
void test_efa_cq_read_senddata_success(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_context *efa_context;
	struct fi_context2 ctx;
	struct fi_cq_data_entry cq_entry = {0};
	int ret;

	efa_context = (struct efa_context *) &ctx;
	efa_context->completion_flags = FI_SEND | FI_MSG | FI_REMOTE_CQ_DATA;

	test_efa_cq_read_prep(resource, IBV_WC_SEND, IBV_WC_SUCCESS, 0,
			 efa_context, IBV_WC_WITH_IMM, false);

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, 1);

	assert_true(efa_context == cq_entry.op_context);
	assert_true(cq_entry.flags == efa_context->completion_flags);

	/* reset the mocked cq before it's polled by ep close */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief test EFA CQ's fi_cq_read() works properly when rdma-core return
 * success status for recv operation.
 */
void test_efa_cq_read_recv_success(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_context *efa_context;
	struct fi_cq_data_entry cq_entry;
	struct fi_context2 ctx;
	int ret;

	efa_context = (struct efa_context *) &ctx;
	efa_context->completion_flags = FI_RECV | FI_MSG;

	test_efa_cq_read_prep(resource, IBV_WC_RECV, IBV_WC_SUCCESS, 0,
			 efa_context, 0, false);

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, 1);

	assert_true(efa_context == cq_entry.op_context);
	assert_true(efa_context->completion_flags == cq_entry.flags);

	/* reset the mocked cq before it's polled by ep close */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief test EFA CQ's fi_cq_read() works properly when rdma-core return
 * success status for senddata operation.
 */
void test_efa_cq_read_write_success(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_context *efa_context;
	struct fi_context2 ctx;
	struct fi_cq_data_entry cq_entry = {0};
	int ret;

	efa_context = (struct efa_context *) &ctx;
	efa_context->completion_flags = FI_WRITE | FI_RMA;

	test_efa_cq_read_prep(resource, IBV_WC_SEND, IBV_WC_SUCCESS, 0,
			 efa_context, 0, false);

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, 1);

	assert_true(efa_context == cq_entry.op_context);
	assert_true(cq_entry.flags == efa_context->completion_flags);

	/* reset the mocked cq before it's polled by ep close */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief test EFA CQ's fi_cq_read() works properly when rdma-core return
 * success status for writedata operation.
 */
void test_efa_cq_read_writedata_success(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_context *efa_context;
	struct fi_context2 ctx;
	struct fi_cq_data_entry cq_entry = {0};
	int ret;

	efa_context = (struct efa_context *) &ctx;
	efa_context->completion_flags = FI_WRITE | FI_RMA | FI_REMOTE_CQ_DATA;

	test_efa_cq_read_prep(resource, IBV_WC_RDMA_WRITE, IBV_WC_SUCCESS, 0,
			 efa_context, IBV_WC_WITH_IMM, false);

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, 1);

	assert_true(efa_context == cq_entry.op_context);
	assert_true(cq_entry.flags == efa_context->completion_flags);

	/* reset the mocked cq before it's polled by ep close */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief test EFA CQ's fi_cq_read() works properly when rdma-core return
 * success status for read operation.
 */
void test_efa_cq_read_read_success(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_context *efa_context;
	struct fi_context2 ctx;
	struct fi_cq_data_entry cq_entry = {0};
	int ret;

	efa_context = (struct efa_context *) &ctx;
	efa_context->completion_flags = FI_READ | FI_RMA;

	test_efa_cq_read_prep(resource, IBV_WC_RDMA_READ, IBV_WC_SUCCESS, 0,
			 efa_context, 0, false);

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, 1);

	assert_true(efa_context == cq_entry.op_context);
	assert_true(cq_entry.flags == efa_context->completion_flags);

	/* reset the mocked cq before it's polled by ep close */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief test EFA CQ's fi_cq_read() works properly when rdma-core return
 * success status for recv rdma with imm operation.
 */
void test_efa_cq_read_recv_rdma_with_imm_success(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fi_cq_data_entry cq_entry;
	int ret;

	test_efa_cq_read_prep(resource, IBV_WC_RECV_RDMA_WITH_IMM, IBV_WC_SUCCESS, 0,
			 NULL, IBV_WC_WITH_IMM, true);

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, 1);

	assert_true(cq_entry.op_context == NULL);
	assert_true(cq_entry.flags == (FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA | FI_RMA));

	/* reset the mocked cq before it's polled by ep close */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

static void efa_cq_check_cq_err_entry(struct efa_resource *resource, int vendor_error) {
	struct fi_cq_err_entry cq_err_entry = {0};
	const char *strerror;
	int ret;

	/* Allocate memory to read CQ error */
	cq_err_entry.err_data_size = EFA_ERROR_MSG_BUFFER_LENGTH;
	cq_err_entry.err_data = malloc(cq_err_entry.err_data_size);
	assert_non_null(cq_err_entry.err_data);

	ret = fi_cq_readerr(resource->cq, &cq_err_entry, 0);
	assert_true(cq_err_entry.err_data_size > 0);
	strerror = fi_cq_strerror(resource->cq, cq_err_entry.prov_errno,
				  cq_err_entry.err_data, NULL, 0);

	assert_int_equal(ret, 1);
	assert_int_not_equal(cq_err_entry.err, FI_SUCCESS);
	assert_int_equal(cq_err_entry.prov_errno, vendor_error);
	assert_true(strlen(strerror) > 0);
}

/**
 * @brief test EFA CQ's fi_cq_read()/fi_cq_readerr() works properly when rdma-core return bad status for send.
 *
 * When the send operation failed, fi_cq_read() should return -FI_EAVAIL, which means error available.
 * then user should call fi_cq_readerr() to get an error CQ entry that contain error code.
 *
 * @param[in]  state            struct efa_resource that is managed by the framework
 */
void test_efa_cq_read_send_failure(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_context *efa_context;
	struct fi_cq_data_entry cq_entry;
	struct fi_context2 ctx;
	int ret;

	efa_context = (struct efa_context *) &ctx;
	efa_context->completion_flags = FI_SEND | FI_MSG;

	test_efa_cq_read_prep(resource, IBV_WC_SEND, IBV_WC_GENERAL_ERR,
			 EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE, (struct efa_context *) &ctx, 0, false);

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, -FI_EAVAIL);

	efa_cq_check_cq_err_entry(resource,
				  EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE);

	/* reset the mocked cq before it's polled by ep close */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief test EFA CQ's fi_cq_read()/fi_cq_readerr() works properly when rdma-core return bad status for recv.
 *
 * When the recv operation failed, fi_cq_read() should return -FI_EAVAIL, which means error available.
 * then user should call fi_cq_readerr() to get an error CQ entry that contain error code.
 *
 * @param[in]  state            struct efa_resource that is managed by the framework
 */
void test_efa_cq_read_recv_failure(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_context *efa_context;
	struct fi_cq_data_entry cq_entry;
	struct fi_context2 ctx;
	int ret;

	efa_context = (struct efa_context *) &ctx;
	efa_context->completion_flags = FI_RECV | FI_MSG;

	test_efa_cq_read_prep(resource, IBV_WC_RECV, IBV_WC_GENERAL_ERR,
			 EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE, (struct efa_context *) &ctx, 0, false);

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, -FI_EAVAIL);

	efa_cq_check_cq_err_entry(resource,
				  EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE);

	/* reset the mocked cq before it's polled by ep close */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief test EFA CQ's fi_cq_read()/fi_cq_readerr() works properly when rdma-core return bad status for recv.
 *
 * When the recv_rdma_with_imm operation failed, fi_cq_read() should return -FI_EAVAIL, which means error available.
 * then user should call fi_cq_readerr() to get an error CQ entry that contain error code.
 *
 * @param[in]  state            struct efa_resource that is managed by the framework
 */
void test_efa_cq_recv_rdma_with_imm_failure(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fi_cq_data_entry cq_entry;
	int ret;

	test_efa_cq_read_prep(resource, IBV_WC_RECV_RDMA_WITH_IMM, IBV_WC_GENERAL_ERR,
			 EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE, NULL, IBV_WC_WITH_IMM, true);

	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, -FI_EAVAIL);

	efa_cq_check_cq_err_entry(resource,
				  EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE);

	/* reset the mocked cq before it's polled by ep close */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief check efa cq's data_path_direct status for different device generation and wait obj
 * This test is against efa-direct fabric
 *
 * @param state unit test resources
 */
static void test_efa_cq_data_path_direct_status(
	struct efa_resource **state, uint32_t vendor_part_id,
	bool data_path_direct_enabled, enum fi_wait_obj wait_obj)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	uint32_t vendor_id_orig = g_efa_selected_device_list[0].ibv_attr.vendor_part_id;
	struct fid_cq *cq;
	struct fi_cq_attr cq_attr = {
		.format = FI_CQ_FORMAT_DATA,
		.wait_obj = wait_obj,
	};
	bool use_data_path_direct_orig = efa_env.use_data_path_direct;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	/* mock the vendor part id */
	g_efa_selected_device_list[0].ibv_attr.vendor_part_id = vendor_part_id;
	efa_env.use_data_path_direct = 1;

	ret = fi_cq_open(resource->domain, &cq_attr, &cq, NULL);
	if (ret && wait_obj != FI_WAIT_NONE)
		/* EFA device doesn't support cq notification. */
		return;
		
	assert_int_equal(ret, 0);
	efa_cq = container_of(cq, struct efa_cq, util_cq.cq_fid);

	assert_true(efa_cq->ibv_cq.data_path_direct_enabled == data_path_direct_enabled);
	assert_int_equal(fi_close(&cq->fid), 0);

	/* Recover the mocked vendor_id */
	g_efa_selected_device_list[0].ibv_attr.vendor_part_id = vendor_id_orig;
	efa_env.use_data_path_direct = use_data_path_direct_orig;
}

#if HAVE_EFA_DATA_PATH_DIRECT
/**
 * @brief Make sure data_path_direct is disabled when user specifies FI_EFA_USE_DATA_PATH_DIRECT=0;
 * This test is against efa-direct fabric
 *
 * @param state unit test resources
 */
void test_efa_cq_data_path_direct_disabled_by_env(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	bool use_data_path_direct_orig = efa_env.use_data_path_direct;

	efa_env.use_data_path_direct = 0;
	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid);

	/* cq direct should be disabled when env disabled it */
	assert_false(efa_cq->ibv_cq.data_path_direct_enabled);

	/* recover the env */
	efa_env.use_data_path_direct = use_data_path_direct_orig;
}

/**
 * @brief Make sure data_path_direct is disabled when device is old
 * This test is against efa-direct fabric
 *
 * @param state unit test resources
 */
void test_efa_cq_data_path_direct_disabled_with_old_device(struct efa_resource **state)
{
	test_efa_cq_data_path_direct_status(state, 0xefa0, false, FI_WAIT_NONE);
}

/**
 * @brief Make sure data_path_direct is enabled when device is new enough
 * This test is against efa-direct fabric
 *
 * @param state unit test resources
 */
void test_efa_cq_data_path_direct_enabled_with_new_device(struct efa_resource **state)
{
	test_efa_cq_data_path_direct_status(state, 0xefa1, true, FI_WAIT_NONE);
}

/**
 * @brief Make sure data_path_direct is enabled with db,
 * and disabled without db and with wait obj.
 * This test is against efa-direct fabric
 *
 * @param state unit test resources
 */
void test_efa_cq_data_path_direct_with_wait_obj(struct efa_resource **state)
{
#if HAVE_EFADV_CQ_ATTR_DB
	test_efa_cq_data_path_direct_status(state, 0xefa1, true, FI_WAIT_UNSPEC);
#else
	test_efa_cq_data_path_direct_status(state, 0xefa1, false, FI_WAIT_UNSPEC);
#endif
}

#else

void test_efa_cq_data_path_direct_disabled_by_env(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid);

	/* cq direct should always be disabled */
	assert_false(efa_cq->ibv_cq.data_path_direct_enabled);
}

void test_efa_cq_data_path_direct_disabled_with_old_device(struct efa_resource **state)
{
	/* cq direct should always be disabled */
	test_efa_cq_data_path_direct_status(state, 0xefa0, false, FI_WAIT_NONE);
}

void test_efa_cq_data_path_direct_enabled_with_new_device(struct efa_resource **state)
{
	/* cq direct should always be disabled */
	test_efa_cq_data_path_direct_status(state, 0xefa1, false, FI_WAIT_NONE);
}

void test_efa_cq_data_path_direct_with_wait_obj(struct efa_resource **state)
{
	/* cq direct should always be disabled */
#if HAVE_EFADV_CQ_ATTR_DB
	test_efa_cq_data_path_direct_status(state, 0xefa1, false, FI_WAIT_UNSPEC);
#else
	test_efa_cq_data_path_direct_status(state, 0xefa1, false, FI_WAIT_UNSPEC);
#endif
}

#endif /* HAVE_EFA_DIRECT_CQ */

/**
 * @brief Test cq data_path_direct status for efa-rdm
 * This test is against efa fabric
 * Currently, data_path_direct should always be disabled by efa-rdm.
 * @param state pointer of efa_resource
 */
void test_efa_rdm_cq_data_path_direct_disabled(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid);

	assert_false(efa_cq->ibv_cq.data_path_direct_enabled);
}

/**
 * @brief test efa_cq_trywait() returns -FI_EINVAL when no completion channel is present
 *
 * When there is no ibv_comp_channel associated with the CQ, efa_cq_trywait() should return -FI_EINVAL.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_cq_trywait_no_channel(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	int ret;

	/* Construct CQ without wait object (no completion channel) */
	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid.fid);

	assert_null(efa_cq->ibv_cq.channel);

	ret = efa_cq_trywait(efa_cq);
	assert_int_equal(ret, -FI_EINVAL);
}

static int test_efa_cq_sread_prep(struct efa_resource *resource)
{
	struct fid_cq *cq;
	struct fi_cq_attr cq_attr = {
		.format = FI_CQ_FORMAT_DATA,
		.wait_obj = FI_WAIT_UNSPEC,
	};
	int ret;

	g_efa_unit_test_mocks.efa_ibv_req_notify_cq = efa_mock_ibv_req_notify_cq_return_mock;
	g_efa_unit_test_mocks.efa_ibv_get_cq_event = efa_mock_ibv_get_cq_event_return_mock;
	will_return_maybe(efa_mock_ibv_req_notify_cq_return_mock, 0);
	will_return_maybe(efa_mock_ibv_get_cq_event_return_mock, -1);

	efa_unit_test_resource_construct_no_cq_and_ep_not_enabled(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	ret = fi_cq_open(resource->domain, &cq_attr, &cq, NULL);
	if (ret)
		/* EFA device doesn't support cq notification. Don't test sread. */
		return ret;

	assert_int_equal(ret, 0);
	assert_int_equal(fi_ep_bind(resource->ep, &cq->fid, FI_SEND | FI_RECV), 0);
	assert_int_equal(fi_enable(resource->ep), 0);

	resource->cq = cq;
	return ret;
}

/**
 * @brief test efa_cq_trywait() returns -FI_EAGAIN when completions are available
 *
 * When there are completions available in the util_cq circular queue, efa_cq_trywait() should return -FI_EAGAIN.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_cq_trywait_completions_available(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	struct fi_cq_tagged_entry cq_entry = {
		.flags = FI_SEND | FI_MSG,
		.len = 1024,
		.data = 0x12345678,
		.tag = 0,
	};
	int ret;

	ret = test_efa_cq_sread_prep(resource);
	if (ret)
		return;

	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid.fid);
	assert_non_null(efa_cq->ibv_cq.channel);

	/* Add a completion to the util_cq circular queue */
	ret = ofi_cq_write(&efa_cq->util_cq, cq_entry.op_context, cq_entry.flags,
			   cq_entry.len, cq_entry.buf, cq_entry.data, cq_entry.tag);
	assert_int_equal(ret, 0);

	/* trywait should return -FI_EAGAIN since completions are available */
	ret = efa_cq_trywait(efa_cq);
	assert_int_equal(ret, -FI_EAGAIN);
}

/**
 * @brief test efa_cq_trywait() succeeds when CQ is empty and ready to wait
 *
 * When the CQ is empty and there are no pending events, efa_cq_trywait() should return FI_SUCCESS.
 *
 * @param[in]	state	struct efa_resource that is managed by the framework
 */
void test_efa_cq_trywait_success(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	int ret;

	ret = test_efa_cq_sread_prep(resource);
	if (ret)
		return;

	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid.fid);
	assert_non_null(efa_cq->ibv_cq.channel);

	assert_int_equal(efa_cq_trywait(efa_cq), FI_SUCCESS);
}

/**
 * @brief test fi_cq_sread() returns -FI_EINVAL when no completion channel is present
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_cq_sread_einval(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	struct fi_cq_data_entry cq_entry = {0};
	int ret;

	/* Construct CQ without wait object */
	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid.fid);

	assert_null(efa_cq->wait_obj);
	assert_null(efa_cq->ibv_cq.channel);

	ret = fi_cq_sread(resource->cq, &cq_entry, 1, NULL, 1);
	assert_int_equal(ret, -FI_EINVAL);
}

/**
 * @brief test EFA CQ's fi_cq_sread() returns -FI_EAGAIN when CQ is empty
 *
 * @param[in]  state	struct efa_resource that is managed by the framework
 */
void test_efa_cq_sread_eagain(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fi_cq_data_entry cq_entry = {0};
	int ret;

	ret = test_efa_cq_sread_prep(resource);
	if (ret)
		return;

	/* poll timeout because there is no cq */
	ret = fi_cq_sread(resource->cq, &cq_entry, 1, NULL, 1);
	assert_int_equal(ret, -FI_EAGAIN);
}

/**
 * @brief test efa_cq_control() with FI_GETWAIT command when completion channel is available
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_cq_control_getwait_with_channel(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	int fd = -1;
	int ret;

	ret = test_efa_cq_sread_prep(resource);
	if (ret)
		return;

	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid.fid);
	assert_non_null(efa_cq->ibv_cq.channel);

	ret = fi_control(&resource->cq->fid, FI_GETWAIT, &fd);
	assert_int_equal(ret, 0);
	assert_true(fd >= 0);
	assert_int_equal(fd, efa_cq->ibv_cq.channel->fd);
}

/**
 * @brief test efa_cq_control() with FI_GETWAIT command when no completion channel is present
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_cq_control_getwait_no_channel(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	int fd = -1;
	int ret;

	/* Construct CQ without wait object (no completion channel) */
	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid.fid);

	assert_null(efa_cq->ibv_cq.channel);

	ret = fi_control(&resource->cq->fid, FI_GETWAIT, &fd);
	assert_int_equal(ret, -FI_ENODATA);
}

/**
 * @brief test efa_cq_control() with FI_GETWAITOBJ command
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_cq_control_getwaitobj(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	enum fi_wait_obj wait_obj;
	int ret;

	ret = test_efa_cq_sread_prep(resource);
	if (ret)
		return;

	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid.fid);

	ret = fi_control(&resource->cq->fid, FI_GETWAITOBJ, &wait_obj);
	assert_int_equal(ret, 0);
	assert_int_equal(wait_obj, efa_cq->wait_obj);
	assert_int_equal(wait_obj, FI_WAIT_FD);
}

/**
 * @brief test efa_cq_control() with invalid command
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_cq_control_invalid_command(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	int dummy_arg = 0;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	ret = fi_control(&resource->cq->fid, 999 /* invalid command */, &dummy_arg);
	assert_int_equal(ret, -FI_ENOSYS);
}

/**
 * @brief Verify CQ ep_list_lock uses no-op locking with FI_THREAD_COMPLETION and FI_PROGRESS_CONTROL_UNIFIED
 *
 * @param[in] state struct efa_resource managed by the framework
 */
void test_efa_cq_ep_list_lock_type_no_op(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	assert_non_null(resource->hints);
	resource->hints->domain_attr->progress = FI_PROGRESS_CONTROL_UNIFIED;
	resource->hints->domain_attr->threading = FI_THREAD_COMPLETION;
	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM, FI_VERSION(2, 0), resource->hints, false, true);

	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid.fid);
	assert_int_equal(efa_cq->util_cq.ep_list_lock.lock_type, OFI_LOCK_NOOP);
}

/**
 * @brief Verify CQ ep_list_lock uses mutex locking with FI_THREAD_SAFE
 *
 * @param[in] state struct efa_resource managed by the framework
 */
void test_efa_cq_ep_list_lock_type_mutex(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	assert_non_null(resource->hints);
	resource->hints->domain_attr->threading = FI_THREAD_SAFE;
	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM, FI_VERSION(2, 0), resource->hints, false, true);

	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid.fid);
	assert_int_equal(efa_cq->util_cq.ep_list_lock.lock_type, OFI_LOCK_MUTEX);
}

/**
 * @brief Test CQ ops override when counter is bound to endpoint
 *
 * This test verifies the fix from commit 643af57a4 that properly sets CQ ops
 * when counters are bound to endpoints.
 */
void test_efa_cq_ops_override_with_counter_binding(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_cntr *cntr;
	struct fi_cntr_attr cntr_attr = {0};
	int ret;

	efa_unit_test_resource_construct_ep_not_enabled(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	/* Initially CQ should use bypass ops */
	assert_ptr_equal(resource->cq->ops, &efa_cq_bypass_util_cq_ops);

	/* Create counter */
	ret = fi_cntr_open(resource->domain, &cntr_attr, &cntr, NULL);
	assert_int_equal(ret, 0);

	/* Bind counter to endpoint - this should trigger CQ ops override */
	ret = fi_ep_bind(resource->ep, &cntr->fid, FI_SEND);
	assert_int_equal(ret, 0);

	ret = fi_enable(resource->ep);
	assert_int_equal(ret, 0);

	/* CQ ops should be set to util ops when counter is bound */
	assert_ptr_equal(resource->cq->ops, &efa_cq_ops);

	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
	fi_close(&cntr->fid);
}

/**
 * @brief Test CQ readfrom input validation
 *
 * This test verifies the fix from commit 14f8cd478 that adds input validation
 * to efa_cq_readfrom to return -FI_EAGAIN for invalid parameters.
 */
void test_efa_cq_readfrom_input_validation(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fi_cq_data_entry cq_entry;
	fi_addr_t src_addr;
	ssize_t ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	/* Test with NULL buffer */
	ret = fi_cq_readfrom(resource->cq, NULL, 1, &src_addr);
	assert_int_equal(ret, -FI_EAGAIN);

	/* Test with zero count */
	ret = fi_cq_readfrom(resource->cq, &cq_entry, 0, &src_addr);
	assert_int_equal(ret, -FI_EAGAIN);

	/* Test with valid parameters but empty CQ */
	g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_return_mock;
	will_return(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);

	ret = fi_cq_readfrom(resource->cq, &cq_entry, 1, &src_addr);
	assert_int_equal(ret, -FI_EAGAIN);

	/* Reset mocks */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

static void test_efa_cq_readerr_common(struct efa_resource *resource, bool user_owned_buffer)
{
	struct efa_context *efa_context;
	struct fi_context2 ctx;
	struct fi_cq_data_entry cq_entry;
	struct fi_cq_err_entry cq_err_entry = {0};
	struct efa_cq *efa_cq;
	struct efa_ibv_cq *ibv_cq;
	ssize_t ret;

	efa_context = (struct efa_context *) &ctx;
	efa_context->completion_flags = FI_SEND | FI_MSG;

	test_efa_cq_read_prep(resource, IBV_WC_SEND, IBV_WC_GENERAL_ERR,
				 EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE, efa_context, 0, false);

	/* Trigger error condition */
	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, -FI_EAVAIL);

	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid);
	ibv_cq = &efa_cq->ibv_cq;

	/* Check poll state before readerr */
	assert_true(ibv_cq->poll_active);
	assert_int_equal(ibv_cq->poll_err, 0);

	if (user_owned_buffer) {
		cq_err_entry.err_data_size = EFA_ERROR_MSG_BUFFER_LENGTH;
		cq_err_entry.err_data = malloc(cq_err_entry.err_data_size);
		assert_non_null(cq_err_entry.err_data);
	}

	ret = fi_cq_readerr(resource->cq, &cq_err_entry, 0);
	assert_int_equal(ret, 1);
	assert_int_not_equal(cq_err_entry.err, FI_SUCCESS);
	assert_int_equal(cq_err_entry.prov_errno, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE);
	assert_non_null(cq_err_entry.err_data);
	assert_true(cq_err_entry.err_data_size > 0);

	/* Check poll state after readerr - should be reset */
	assert_false(ibv_cq->poll_active);
	assert_int_equal(ibv_cq->poll_err, 0);

	if (user_owned_buffer)
		free(cq_err_entry.err_data);

	/* Reset mocks */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief Test CQ readerr return value with user-owned buffer
 */
void test_efa_cq_readerr_return_value_user_buffer(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	test_efa_cq_readerr_common(resource, true);
}

/**
 * @brief Test CQ readerr return value with provider-owned buffer
 */
void test_efa_cq_readerr_return_value_provider_buffer(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	test_efa_cq_readerr_common(resource, false);
}

/**
 * @brief verify that fi_cq_read/fi_cq_readerr works properly when ibv_start_poll failed.
 *
 * When an ibv_start_poll() failed, it currently means the QP associated with the CQE is
 * destroyed. According to libfabric man page, such cqe should be ignored and not reported
 * to application cqs.
 *
 */
void test_efa_cq_readfrom_start_poll_error(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fi_cq_data_entry cq_entry;
	struct fi_cq_err_entry cq_err_entry = {0};
	ssize_t ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_return_mock;
	will_return(efa_mock_efa_ibv_cq_start_poll_return_mock, EINVAL);

	ret = fi_cq_readfrom(resource->cq, &cq_entry, 1, NULL);
	assert_int_equal(ret, -FI_EAGAIN);

	struct efa_cq *efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid);
	struct efa_ibv_cq *ibv_cq = &efa_cq->ibv_cq;

	/* Check poll state before readerr */
	assert_false(ibv_cq->poll_active);
	assert_int_equal(ibv_cq->poll_err, 0);

	/* Test fi_cq_readerr after start_poll error */
	ret = fi_cq_readerr(resource->cq, &cq_err_entry, 0);
	assert_int_equal(ret, -FI_EAGAIN);

	/* Check poll states after readerr - should be not changed */
	assert_false(ibv_cq->poll_active);
	assert_int_equal(ibv_cq->poll_err, 0);

	/* Reset mocks */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}
/**
 * @brief Test that efa_cq_readfrom reads from util_cq when entries are available
 *
 * This test verifies
 * efa_cq_readfrom reads from util_cq when there are CQEs available.
 */
void test_efa_cq_readfrom_util_cq_entries(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	struct fi_cq_data_entry cq_entry = {0};
	struct fi_cq_tagged_entry util_entry = {
		.op_context = (void *)0x12345678,
		.flags = FI_SEND | FI_MSG,
		.len = 1024,
		.data = 0xdeadbeef,
		.tag = 0,
	};
	ssize_t ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid);

	/* Add entry to util_cq */
	ret = ofi_cq_write(&efa_cq->util_cq, util_entry.op_context, util_entry.flags,
			   util_entry.len, util_entry.buf, util_entry.data, util_entry.tag);
	assert_int_equal(ret, 0);

	/* Mock empty device CQ */
	g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_return_mock;
	will_return(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);

	/* Read should get entry from util_cq */
	ret = fi_cq_readfrom(resource->cq, &cq_entry, 1, NULL);
	assert_int_equal(ret, 1);
	assert_ptr_equal(cq_entry.op_context, util_entry.op_context);
	assert_int_equal(cq_entry.flags, util_entry.flags);

	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}

/**
 * @brief Test that efa_cq_readerr reads from util_cq when error entries are available
 *
 * This test verifies
 * efa_cq_readerr reads from util_cq when there are error entries available.
 */
void test_efa_cq_readerr_util_cq_error(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	struct fi_cq_data_entry cq_entry = {0};
	struct fi_cq_err_entry cq_err_entry = {0};
	struct fi_cq_err_entry util_err = {
		.op_context = (void *)0x87654321,
		.flags = FI_RECV | FI_MSG,
		.err = FI_ETRUNC,
		.prov_errno = -FI_ETRUNC,
	};
	ssize_t ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid);

	/* Add error to util_cq */
	ret = ofi_cq_write_error(&efa_cq->util_cq, &util_err);
	assert_int_equal(ret, 0);

	/* Mock empty device CQ */
	g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_return_mock;
	will_return(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);

	/* Read should return error available */
	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, -FI_EAVAIL);

	/* Read error should get entry from util_cq */
	ret = fi_cq_readerr(resource->cq, &cq_err_entry, 0);
	assert_int_equal(ret, 1);
	assert_ptr_equal(cq_err_entry.op_context, util_err.op_context);
	assert_int_equal(cq_err_entry.flags, util_err.flags);
	assert_int_equal(cq_err_entry.err, util_err.err);

	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}
/**
 * @brief Test that efa_cq_start_poll doesn't restart polling when poll is already active
 *
 * This test verifies that efa_cq_start_poll prevents CQE index shifting
 * when efa_cq_start_poll is called while poll is already active.
 *
 * Scenario: fi_cq_read hits completion error -> ep close calls efa_cq_poll_ibv_cq
 * -> efa_cq_start_poll should return early -> error written to util_cq -> fi_cq_readerr retrieves it
 */
void test_efa_cq_poll_active_no_restart(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	struct efa_context *efa_context;
	struct fi_context2 ctx;
	struct fi_cq_data_entry cq_entry;
	struct fi_cq_err_entry cq_err_entry = {0};
	ssize_t ret;

	efa_context = (struct efa_context *) &ctx;
	efa_context->completion_flags = FI_SEND | FI_MSG;

	test_efa_cq_read_prep(resource, IBV_WC_SEND, IBV_WC_GENERAL_ERR,
			      EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE, efa_context, 0, false);

	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid);

	/* First fi_cq_read hits error, leaves poll_active = true */
	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, -FI_EAVAIL);
	assert_true(efa_cq->ibv_cq.poll_active);

	/* Simulate ep close calling efa_cq_poll_ibv_cq which calls efa_cq_start_poll again */
	/* This should NOT restart polling due to poll_active being true */
	efa_cq_progress(&efa_cq->util_cq);

	/* Error should now be in util_cq, retrievable via fi_cq_readerr */
	ret = fi_cq_readerr(resource->cq, &cq_err_entry, 0);
	assert_int_equal(ret, 1);
	assert_int_equal(cq_err_entry.prov_errno, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE);

	/* Reset mocks */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}
/**
 * @brief Test mixed successful and error CQEs handling
 *
 * This test simulates 2 successful CQEs followed by 1 error CQE from device CQ.
 * First fi_cq_read(..., buf, 3) should return 2 (successful entries).
 * Second fi_cq_read should return -FI_EAVAIL (error available).
 */
void test_efa_cq_read_mixed_success_error(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_cq *efa_cq;
	struct efa_context *efa_context1, *efa_context2, *efa_context3;
	struct fi_context2 ctx1, ctx2, ctx3;
	struct fi_cq_data_entry cq_entries[3];
	struct fi_cq_err_entry cq_err_entry = {0};
	struct efa_base_ep *base_ep;
	struct ibv_cq_ex *ibv_cqx;
	ssize_t ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid);
	base_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);
	ibv_cqx = efa_cq->ibv_cq.ibv_cq_ex;
	efa_cq->ibv_cq.unsolicited_write_recv_enabled = false;

	/* Create fake peer */
	struct efa_ep_addr raw_addr;
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t addr;
	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL);
	assert_int_equal(ret, 1);

	/* Setup contexts for 3 operations */
	efa_context1 = (struct efa_context *) &ctx1;
	efa_context1->completion_flags = FI_SEND | FI_MSG;
	efa_context1->addr = addr;
	efa_context2 = (struct efa_context *) &ctx2;
	efa_context2->completion_flags = FI_SEND | FI_MSG;
	efa_context2->addr = addr;
	efa_context3 = (struct efa_context *) &ctx3;
	efa_context3->completion_flags = FI_SEND | FI_MSG;
	efa_context3->addr = addr;

	/* Setup mocks - need custom mock to simulate status changes */
	g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_next_poll = &efa_mock_efa_ibv_cq_next_poll_simulate_status_change;
	g_efa_unit_test_mocks.efa_ibv_cq_end_poll = &efa_mock_efa_ibv_cq_end_poll_check_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_opcode = &efa_mock_efa_ibv_cq_wc_read_opcode_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_vendor_err = &efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_wc_flags = &efa_mock_efa_ibv_cq_wc_read_wc_flags_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_qp_num = &efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock;
	g_efa_unit_test_mocks.efa_ibv_cq_wc_read_byte_len = &efa_mock_efa_ibv_cq_wc_read_byte_len_return_mock;

	/* Setup initial state: CQE1 (success) */
	ibv_cqx->wr_id = (uintptr_t)efa_context1;
	ibv_cqx->status = IBV_WC_SUCCESS;

	/* Mock sequence for first fi_cq_read call */
	will_return(efa_mock_efa_ibv_cq_start_poll_return_mock, 0);
	/* Common CQE reads mocks shared by all 3 CQEs */
	will_return_maybe(efa_mock_efa_ibv_cq_wc_read_opcode_return_mock, IBV_WC_SEND);
	will_return_maybe(efa_mock_efa_ibv_cq_wc_read_wc_flags_return_mock, 0);
	will_return_maybe(efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock, base_ep->qp->qp_num);
	will_return_maybe(efa_mock_efa_ibv_cq_wc_read_byte_len_return_mock, 1024);

	/* CQE1 reads using common mocks */
	/* Move to CQE2, set status to success, set wr_id to context2 */
	will_return(efa_mock_efa_ibv_cq_next_poll_simulate_status_change, IBV_WC_SUCCESS);
	will_return(efa_mock_efa_ibv_cq_next_poll_simulate_status_change, efa_context2);
	will_return(efa_mock_efa_ibv_cq_next_poll_simulate_status_change, 0);
	/* CQE2 reads using common mocks */
	/* Move to CQE3, set status to error, set wr_id to context3 */
	will_return(efa_mock_efa_ibv_cq_next_poll_simulate_status_change, IBV_WC_GENERAL_ERR);
	will_return(efa_mock_efa_ibv_cq_next_poll_simulate_status_change, efa_context3);
	will_return(efa_mock_efa_ibv_cq_next_poll_simulate_status_change, 0);
	will_return(efa_mock_efa_ibv_cq_end_poll_check_mock, NULL);

	/* First fi_cq_read should return 2 successful entries, stop at error */
	ret = fi_cq_read(resource->cq, cq_entries, 3);
	assert_int_equal(ret, 2);
	assert_ptr_equal(cq_entries[0].op_context, efa_context1);
	assert_ptr_equal(cq_entries[1].op_context, efa_context2);

	/* Second fi_cq_read should see error status and return -FI_EAVAIL */
	ret = fi_cq_read(resource->cq, cq_entries, 3);
	assert_int_equal(ret, -FI_EAVAIL);


	/* CQE3 read err using common mocks + extra following */
	will_return(efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE);

	/* Verify error can be retrieved */
	cq_err_entry.err_data_size = EFA_ERROR_MSG_BUFFER_LENGTH;
	cq_err_entry.err_data = malloc(cq_err_entry.err_data_size);
	assert_non_null(cq_err_entry.err_data);

	ret = fi_cq_readerr(resource->cq, &cq_err_entry, 0);
	assert_int_equal(ret, 1);
	assert_int_equal(cq_err_entry.prov_errno, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE);

	free(cq_err_entry.err_data);

	/* Reset mocks */
	will_return_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;
}