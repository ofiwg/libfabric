/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"

typedef void (*efa_rdm_ope_handle_error_func_t)(struct efa_rdm_ope *ope, int err, int prov_errno);

void test_efa_rdm_ope_prepare_to_post_send_impl(struct efa_resource *resource,
						enum fi_hmem_iface iface,
						size_t total_len,
						int expected_ret,
						int expected_pkt_entry_cnt,
						int *expected_pkt_entry_data_size_vec)
{
	struct efa_ep_addr raw_addr;
	struct efa_mr mock_mr;
	struct efa_rdm_ope mock_txe;
	struct efa_rdm_peer mock_peer;
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t addr;
	int pkt_entry_cnt, pkt_entry_data_size_vec[1024];
	int i, err, ret;

	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0 /* flags */, NULL /* context */);
	assert_int_equal(ret, 1);

	mock_mr.peer.iface = iface;

	struct efa_rdm_ep *efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct efa_rdm_peer *peer = efa_rdm_ep_get_peer(efa_rdm_ep, 0);

	memset(&mock_txe, 0, sizeof(mock_txe));
	mock_txe.total_len = total_len;
	mock_txe.peer = peer;
	mock_txe.iov_count = 1;
	mock_txe.iov[0].iov_base = NULL;
	mock_txe.iov[0].iov_len = 9000;
	mock_txe.desc[0] = &mock_mr;
	mock_txe.ep = efa_rdm_ep;
	mock_txe.peer = &mock_peer;

	err = efa_rdm_ope_prepare_to_post_send(&mock_txe,
					       EFA_RDM_MEDIUM_MSGRTM_PKT,
					       &pkt_entry_cnt,
					       pkt_entry_data_size_vec);

	assert_int_equal(err, expected_ret);

	if (err)
		return;

	assert_int_equal(pkt_entry_cnt, expected_pkt_entry_cnt);

	for (i = 0; i < pkt_entry_cnt; ++i)
		assert_int_equal(pkt_entry_data_size_vec[i], expected_pkt_entry_data_size_vec[i]);
}

/**
 * @brief verify efa_rdm_ope_prepare_to_post_send()'s return code
 *
 * Verify that efa_rdm_ope_prepare_to_post_send() will return
 * -FI_EAGAIN, when there is not enough TX packet available,
 */
void test_efa_rdm_ope_prepare_to_post_send_with_no_enough_tx_pkts(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_ep->efa_outstanding_tx_ops = efa_rdm_ep->efa_max_outstanding_tx_ops - 1;
	/* we need at least 2 packets to send this message, but only 1 is available,
	 * therefore efa_rdm_ope_prepare_to_post_send() should return
	 * -FI_EAGAIN.
	 */
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_SYSTEM,
						   9000, -FI_EAGAIN, -1, NULL);
	efa_rdm_ep->efa_outstanding_tx_ops = 0;
}

/**
 * @brief verify the pkt_entry_cnt and data size for host memory
 */
void test_efa_rdm_ope_prepare_to_post_send_host_memory(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	size_t msg_length;
	int expected_pkt_entry_cnt;
	int expected_pkt_entry_data_size_vec[1024];

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	/* data size should be aligned and evenly distributed.
	 * alignment for host memory is 8 byte by default.
	 */
	msg_length = 9000;
	expected_pkt_entry_cnt = 2;
	expected_pkt_entry_data_size_vec[0] = 4496;
	expected_pkt_entry_data_size_vec[1] = 4504;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_SYSTEM,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);

	msg_length = 12000;
	expected_pkt_entry_cnt = 2;
	expected_pkt_entry_data_size_vec[0] = 6000;
	expected_pkt_entry_data_size_vec[1] = 6000;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_SYSTEM,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);

	msg_length = 18004;
	expected_pkt_entry_cnt = 3;
	expected_pkt_entry_data_size_vec[0] = 6000;
	expected_pkt_entry_data_size_vec[1] = 6000;
	expected_pkt_entry_data_size_vec[2] = 6004;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_SYSTEM,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);

}

/**
 * @brief verify the pkt_entry_cnt and data size for host memory when align128 was requested
 */
void test_efa_rdm_ope_prepare_to_post_send_host_memory_align128(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	size_t msg_length;
	int expected_pkt_entry_cnt;
	int expected_pkt_entry_data_size_vec[1024];

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_ep->sendrecv_in_order_aligned_128_bytes = true;

	/* if user requested 128 byte alignment, then all but the last
	 * last packet's data size should be 128 aligned
	 */
	msg_length = 9000;
	expected_pkt_entry_cnt = 2;
	expected_pkt_entry_data_size_vec[0] = 4480;
	expected_pkt_entry_data_size_vec[1] = 4520;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_SYSTEM,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);

	msg_length = 12000;
	expected_pkt_entry_cnt = 2;
	expected_pkt_entry_data_size_vec[0] = 5888;
	expected_pkt_entry_data_size_vec[1] = 6112;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_SYSTEM,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);

	msg_length = 18004;
	expected_pkt_entry_cnt = 3;
	expected_pkt_entry_data_size_vec[0] = 5888;
	expected_pkt_entry_data_size_vec[1] = 5888;
	expected_pkt_entry_data_size_vec[2] = 6228;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_SYSTEM,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);
}

/**
 * @brief verify the pkt_entry_cnt and data size for cuda memory
 */
void test_efa_rdm_ope_prepare_to_post_send_cuda_memory(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	size_t msg_length;
	int expected_pkt_entry_cnt;
	int expected_pkt_entry_data_size_vec[1024];

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	/* default alignment of cuda memory is 64 bytes */
	msg_length = 12000;
	expected_pkt_entry_cnt = 2;
	expected_pkt_entry_data_size_vec[0] = 5952;
	expected_pkt_entry_data_size_vec[1] = 6048;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_CUDA,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);
}

/**
 * @brief verify the pkt_entry_cnt and data size for cuda memory when align128 was requested
 */
void test_efa_rdm_ope_prepare_to_post_send_cuda_memory_align128(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	size_t msg_length;
	int expected_pkt_entry_cnt;
	int expected_pkt_entry_data_size_vec[1024];

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_ep->sendrecv_in_order_aligned_128_bytes = true;

	msg_length = 12000;
	expected_pkt_entry_cnt = 2;
	/* if user requested 128 byte alignment, then all but the last
	 * last packet's data size should be 128 aligned
	 */
	expected_pkt_entry_data_size_vec[0] = 5888;
	expected_pkt_entry_data_size_vec[1] = 6112;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_CUDA,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);
}

/**
 * @brief verify that 0 byte write can be submitted successfully
 */
void test_efa_rdm_ope_post_write_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff local_buff;
	struct efa_ep_addr raw_addr;
	struct efa_rdm_ope mock_txe;
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t addr;
	int ret, err;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0 /* flags */, NULL /* context */);
	assert_int_equal(ret, 1);

	struct efa_rdm_ep *efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct efa_rdm_peer *peer = efa_rdm_ep_get_peer(efa_rdm_ep, 0);

	efa_unit_test_buff_construct(&local_buff, resource, 4096 /* buff_size */);
	memset(&mock_txe, 0, sizeof(mock_txe));
	mock_txe.total_len = 0;
	mock_txe.peer = peer;
	mock_txe.iov_count = 1;
	mock_txe.iov[0].iov_base = local_buff.buff;
	mock_txe.iov[0].iov_len = 0;
	mock_txe.desc[0] = fi_mr_desc(local_buff.mr);
	mock_txe.rma_iov_count = 1;
	mock_txe.rma_iov[0].addr = 0x87654321;
	mock_txe.rma_iov[0].key = 123456;
	mock_txe.rma_iov[0].len = 0;

	mock_txe.ep = efa_rdm_ep;

	g_efa_unit_test_mocks.efa_qp_wr_start = &efa_mock_efa_qp_wr_start_no_op;
	g_efa_unit_test_mocks.efa_qp_wr_rdma_write = &efa_mock_efa_qp_wr_rdma_write_save_wr;
	g_efa_unit_test_mocks.efa_qp_wr_set_sge_list = &efa_mock_efa_qp_wr_set_sge_list_no_op;
	g_efa_unit_test_mocks.efa_qp_wr_set_ud_addr = &efa_mock_efa_qp_wr_set_ud_addr_no_op;
	g_efa_unit_test_mocks.efa_qp_wr_complete = &efa_mock_efa_qp_wr_complete_no_op;

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	err = efa_rdm_ope_post_remote_write(&mock_txe);
	assert_int_equal(err, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_rdm_pke_release_tx((struct efa_rdm_pke *)g_ibv_submitted_wr_id_vec[0]);
	mock_txe.ep->efa_outstanding_tx_ops = 0;
	efa_unit_test_buff_destruct(&local_buff);
}

/**
 * @brief efa_rdm_rxe_post_local_read_or_queue should call
 * efa_rdm_pke_read.
 * When efa_rdm_pke_read failed,
 * make sure there is no txe leak in efa_rdm_rxe_post_local_read_or_queue
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 * @param[in]	efa_rdm_pke_read_return	return code of efa_rdm_pke_read
 */
static
void test_efa_rdm_rxe_post_local_read_or_queue_impl(struct efa_resource *resource, int efa_rdm_pke_read_return)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_ope *rxe;
	struct efa_mr cuda_mr = {0};
	char buf[16];
	struct iovec iov = {
		.iov_base = buf,
		.iov_len = sizeof buf
	};

	/**
	 * TODO: Ideally we should mock efa_rdm_ope_post_remote_read_or_queue here,
	 * but this function is currently cannot be mocked as it is at the same file
	 * with efa_rdm_rxe_post_local_read_or_queue, see this restriction in
	 * prov/efa/test/README.md's mocking session
	 */
	g_efa_unit_test_mocks.efa_rdm_pke_read = &efa_mock_efa_rdm_pke_read_return_mock;

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Fake a rdma read enabled device */
	efa_rdm_ep_domain(efa_rdm_ep)->device->max_rdma_size = efa_env.efa_read_segment_size;

	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool, EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	pkt_entry->payload = pkt_entry->wiredata;

	rxe = efa_rdm_ep_alloc_rxe(efa_rdm_ep, NULL, ofi_op_tagged);
	cuda_mr.peer.iface = FI_HMEM_CUDA;

	rxe->desc[0] = &cuda_mr;
	rxe->iov_count = 1;
	rxe->iov[0] = iov;
	pkt_entry->ope = rxe;

	assert_true(dlist_empty(&efa_rdm_ep->txe_list));

	will_return(efa_mock_efa_rdm_pke_read_return_mock, efa_rdm_pke_read_return);

	assert_int_equal(efa_rdm_rxe_post_local_read_or_queue(rxe, 0, pkt_entry, pkt_entry->payload, 16), efa_rdm_pke_read_return);

	/* Clean up the rx entry no matter what returns */
	efa_rdm_pke_release_rx(pkt_entry);
}

void test_efa_rdm_rxe_post_local_read_or_queue_unhappy(struct efa_resource **state)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_resource *resource = *state;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	test_efa_rdm_rxe_post_local_read_or_queue_impl(resource, -FI_ENOMR);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Make sure txe is cleaned for a failed read */
	assert_true(dlist_empty(&efa_rdm_ep->txe_list));
}

void test_efa_rdm_rxe_post_local_read_or_queue_happy(struct efa_resource **state)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_resource *resource = *state;
	struct efa_rdm_pke *tx_pkt_entry;
	struct efa_rdm_ope *txe;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	test_efa_rdm_rxe_post_local_read_or_queue_impl(resource, FI_SUCCESS);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	/* Now we should have a txe allocated */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list),  1);
	txe = container_of(efa_rdm_ep->txe_list.next, struct efa_rdm_ope, ep_entry);
	assert_true(txe->internal_flags & EFA_RDM_OPE_INTERNAL);

	/* We also have a tx pkt allocated inside efa_rdm_ope_read
	 * and we need to clean it */
	tx_pkt_entry = ofi_bufpool_get_ibuf(efa_rdm_ep->efa_tx_pkt_pool, 0);
	efa_rdm_pke_release(tx_pkt_entry);
}

static
void test_efa_rdm_ope_handle_error_impl(
	struct efa_resource *resource,
	efa_rdm_ope_handle_error_func_t efa_rdm_ope_handle_error,
	struct efa_rdm_ope *ope, bool expect_cq_error)
{
	struct fi_cq_data_entry cq_entry;
	struct fi_cq_err_entry cq_err_entry = {0};
	struct fi_eq_err_entry eq_err_entry;

	efa_rdm_ope_handle_error(ope, FI_ENOTCONN,
				 EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);

	if (expect_cq_error) {
		assert_int_equal(fi_cq_read(resource->cq, &cq_entry, 1),
				 -FI_EAVAIL);
		assert_int_equal(fi_cq_readerr(resource->cq, &cq_err_entry, 0),
				 1);
		assert_int_equal(cq_err_entry.err, FI_ENOTCONN);
		assert_int_equal(cq_err_entry.prov_errno,
				 EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);
	} else {
		/* We should expect an empty cq and an eq error */
		assert_int_equal(fi_cq_read(resource->cq, &cq_entry, 1),
				 -FI_EAGAIN);
		assert_int_equal(fi_eq_readerr(resource->eq, &eq_err_entry, 0),
				 sizeof(eq_err_entry));
		assert_int_equal(eq_err_entry.err, FI_ENOTCONN);
		assert_int_equal(eq_err_entry.prov_errno,
				 EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);
	}
}

void test_efa_rdm_txe_handle_error_write_cq(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *txe;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_write);
	assert_non_null(txe);

	test_efa_rdm_ope_handle_error_impl(resource, efa_rdm_txe_handle_error, txe, true);

	efa_rdm_txe_release(txe);
}

void test_efa_rdm_txe_handle_error_not_write_cq(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *txe;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_write);
	assert_non_null(txe);

	txe->internal_flags |= EFA_RDM_OPE_INTERNAL;

	test_efa_rdm_ope_handle_error_impl(resource, efa_rdm_txe_handle_error, txe, false);

	efa_rdm_txe_release(txe);
}

void test_efa_rdm_rxe_handle_error_write_cq(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *rxe;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	rxe = efa_unit_test_alloc_rxe(resource, ofi_op_tagged);
	assert_non_null(rxe);

	test_efa_rdm_ope_handle_error_impl(resource, efa_rdm_rxe_handle_error, rxe, true);

	efa_rdm_rxe_release(rxe);
}

void test_efa_rdm_rxe_handle_error_not_write_cq(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *rxe;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	rxe = efa_unit_test_alloc_rxe(resource, ofi_op_tagged);
	assert_non_null(rxe);

	rxe->internal_flags |= EFA_RDM_OPE_INTERNAL;

	test_efa_rdm_ope_handle_error_impl(resource, efa_rdm_rxe_handle_error, rxe, false);

	efa_rdm_rxe_release(rxe);
}

void test_efa_rdm_rxe_map(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ep *efa_rdm_ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	rxe = efa_unit_test_alloc_rxe(resource, ofi_op_tagged);
	rxe->msg_id = 1;
	assert_non_null(rxe);

	/* rxe has not been inserted to any rxe_map yet */
	assert_null(rxe->rxe_map);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	/* efa_unit_test_alloc_rxe only inserts one address.
	 * So fi_addr is guaranteed to be 0
	 */
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, 0);

	efa_rdm_rxe_map_insert(&peer->rxe_map, rxe->msg_id, rxe);
	assert_true(rxe->rxe_map == &peer->rxe_map);
	assert_true(rxe == efa_rdm_rxe_map_lookup(rxe->rxe_map, rxe->msg_id));

	efa_rdm_rxe_release(rxe);

	/**
	 * Now the map_entry_pool should be empty so we can destroy it
	 * Otherwise there will be an assertion error on the use cnt is
	 * is non-zero
	 */
	ofi_bufpool_destroy(efa_rdm_ep->map_entry_pool);
	efa_rdm_ep->map_entry_pool = NULL;
}

void test_efa_rdm_rxe_list_removal(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_ep *efa_rdm_ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	rxe = efa_unit_test_alloc_rxe(resource, ofi_op_tagged);
	assert_non_null(rxe);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	/* insert to lists */
	rxe->state = EFA_RDM_OPE_SEND;
	dlist_insert_tail(&rxe->entry, &efa_rdm_ep_domain(efa_rdm_ep)->ope_longcts_send_list);
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep_domain(efa_rdm_ep)->ope_longcts_send_list), 1);

	/* Lists should be empty after releasing the ope */
	efa_rdm_rxe_release(rxe);
	dlist_empty(&efa_rdm_ep_domain(efa_rdm_ep)->ope_longcts_send_list);
}

void test_efa_rdm_txe_list_removal(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *txe;
	struct efa_rdm_ep *efa_rdm_ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_tagged);
	assert_non_null(txe);

	/* insert to lists */
	txe->state = EFA_RDM_OPE_SEND;
	dlist_insert_tail(&txe->entry, &efa_rdm_ep_domain(efa_rdm_ep)->ope_longcts_send_list);
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep_domain(efa_rdm_ep)->ope_longcts_send_list), 1);

	txe->internal_flags |= EFA_RDM_OPE_QUEUED_CTRL;
	dlist_insert_tail(&txe->queued_entry, &efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list);
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list), 1);

	/* Lists should be empty after releasing the ope */
	efa_rdm_txe_release(txe);
	assert_true(dlist_empty(&efa_rdm_ep_domain(efa_rdm_ep)->ope_longcts_send_list));
	assert_true(dlist_empty(&efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list));
}

void test_efa_rdm_txe_prepare_local_read_pkt_entry(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *txe;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pkt_entry;
	struct fid_ep *ep;
	struct efa_domain *efa_domain;
	struct fi_msg msg = {0};

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	/* rx_readcopy_pkt_pool is only created when application requested FI_HMEM */
	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);
	efa_domain->util_domain.mr_mode |= FI_MR_HMEM;

	assert_int_equal(fi_endpoint(resource->domain, resource->info, &ep, NULL), 0);
	efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	txe = efa_rdm_ep_alloc_txe(efa_rdm_ep, NULL, &msg, ofi_op_msg, 0, 0);
	assert_non_null(txe);

	/* Use ooo rx pkt because it doesn't have mr so a read_copy pkt clone is enforced. */
	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->rx_ooo_pkt_pool, EFA_RDM_PKE_FROM_OOO_POOL);
	pkt_entry->payload_size = 4;
	pkt_entry->payload = pkt_entry->wiredata + 16;
	pkt_entry->pkt_size = 32;
	assert_non_null(pkt_entry);
	txe->local_read_pkt_entry = pkt_entry;
	txe->rma_iov_count = 1;

	assert_int_equal(efa_rdm_txe_prepare_local_read_pkt_entry(txe), 0);

#if ENABLE_DEBUG
	/* The read copy pkt entry should be inserted to the rx_pkt_list */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->rx_pkt_list), 1);
#endif

	/**
	 * When we close the ep, the read copy pkt entry should be
	 * released. The buffer pool destroy should succeed without
	 * assertion errors.
	 */
	assert_int_equal(fi_close(&ep->fid), 0);
}
/**
 * @brief Test that queue flags are properly cleaned up after error handling
 *
 * This test verifies that queue flags are properly cleaned up after
 * dlist_remove in error handling functions.
 */
void test_efa_rdm_txe_handle_error_queue_flags_cleanup(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *txe;
	struct efa_rdm_ep *efa_rdm_ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_write);
	assert_non_null(txe);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Set up txe with queued flags */
	txe->internal_flags |= EFA_RDM_OPE_QUEUED_CTRL;
	dlist_insert_tail(&txe->queued_entry, &efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list);

	/* Verify txe is in queued list */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list), 1);
	assert_true(txe->internal_flags & EFA_RDM_OPE_QUEUED_CTRL);

	/* Handle error - this should clean up queue flags */
	efa_rdm_txe_handle_error(txe, FI_ENOTCONN, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);

	/* Verify queue flags are cleaned up */
	assert_false(txe->internal_flags & EFA_RDM_OPE_QUEUED_CTRL);
	assert_true(dlist_empty(&efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list));

	/* Release should not cause duplicate dlist_remove */
	efa_rdm_txe_release(txe);
}

/**
 * @brief Test that queue flags are properly cleaned up for rxe error handling
 */
void test_efa_rdm_rxe_handle_error_queue_flags_cleanup(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_ep *efa_rdm_ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	rxe = efa_unit_test_alloc_rxe(resource, ofi_op_tagged);
	assert_non_null(rxe);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Set up rxe with queued flags */
	rxe->internal_flags |= EFA_RDM_OPE_QUEUED_READ;
	dlist_insert_tail(&rxe->queued_entry, &efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list);

	/* Verify rxe is in queued list */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list), 1);
	assert_true(rxe->internal_flags & EFA_RDM_OPE_QUEUED_READ);

	/* Handle error - this should clean up queue flags */
	efa_rdm_rxe_handle_error(rxe, FI_ENOTCONN, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);

	/* Verify queue flags are cleaned up */
	assert_false(rxe->internal_flags & EFA_RDM_OPE_QUEUED_READ);
	assert_true(dlist_empty(&efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list));

	/* Release should not cause duplicate dlist_remove */
	efa_rdm_rxe_release(rxe);
}


