/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"
#include "rdm/efa_rdm_pke_cmd.h"
#include "rdm/efa_rdm_pke_nonreq.h"

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
	uint64_t wr_id;
	struct efa_rdm_pke *pkt_entry;
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

	/* Mock general QP post write function to save work request IDs */
	g_efa_unit_test_mocks.efa_qp_post_write = &efa_mock_efa_qp_post_write_return_mock;
	will_return(efa_mock_efa_qp_post_write_return_mock, 0);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	err = efa_rdm_ope_post_remote_write(&mock_txe);
	assert_int_equal(err, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	wr_id = (uint64_t) g_ibv_submitted_wr_id_vec[0];
	pkt_entry = efa_rdm_cq_get_pke_from_wr_id_solicited(wr_id);

	efa_rdm_pke_release_tx(pkt_entry);
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

/**
 * @brief Test error state prevents duplicate error handling for txe
 *
 * This test verifies that a new error state prevents duplicate error handling.
 */
void test_efa_rdm_txe_handle_error_duplicate_prevention(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *txe;
	struct efa_rdm_ep *efa_rdm_ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_write);
	assert_non_null(txe);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Set txe to EFA_RDM_OPE_SEND state and add to longcts_send_list */
	txe->state = EFA_RDM_OPE_SEND;
	dlist_insert_tail(&txe->entry, &efa_rdm_ep_domain(efa_rdm_ep)->ope_longcts_send_list);

	/* Verify txe is in the list */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep_domain(efa_rdm_ep)->ope_longcts_send_list), 1);

	/* First error handling call */
	efa_rdm_txe_handle_error(txe, FI_ENOTCONN, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);

	/* Verify txe is removed from list and in error state */
	assert_true(dlist_empty(&efa_rdm_ep_domain(efa_rdm_ep)->ope_longcts_send_list));
	assert_int_equal(txe->state, EFA_RDM_OPE_ERR);

	/* Second error handling call should be a no-op */
	efa_rdm_txe_handle_error(txe, FI_EAGAIN, EFA_IO_COMP_STATUS_LOCAL_ERROR_BAD_LENGTH);

	/* State should remain EFA_RDM_OPE_ERR */
	assert_int_equal(txe->state, EFA_RDM_OPE_ERR);

	efa_rdm_txe_release(txe);
}

/**
 * @brief Test error state prevents duplicate error handling for rxe
 */
void test_efa_rdm_rxe_handle_error_duplicate_prevention(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_ep *efa_rdm_ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	rxe = efa_unit_test_alloc_rxe(resource, ofi_op_tagged);
	assert_non_null(rxe);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Set rxe to EFA_RDM_OPE_SEND state and add to longcts_send_list */
	rxe->state = EFA_RDM_OPE_SEND;
	dlist_insert_tail(&rxe->entry, &efa_rdm_ep_domain(efa_rdm_ep)->ope_longcts_send_list);

	/* Verify rxe is in the list */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep_domain(efa_rdm_ep)->ope_longcts_send_list), 1);

	/* First error handling call */
	efa_rdm_rxe_handle_error(rxe, FI_ENOTCONN, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);

	/* Verify rxe is removed from list and in error state */
	assert_true(dlist_empty(&efa_rdm_ep_domain(efa_rdm_ep)->ope_longcts_send_list));
	assert_int_equal(rxe->state, EFA_RDM_OPE_ERR);

	/* Second error handling call should be a no-op */
	efa_rdm_rxe_handle_error(rxe, FI_EAGAIN, EFA_IO_COMP_STATUS_LOCAL_ERROR_BAD_LENGTH);

	/* State should remain EFA_RDM_OPE_ERR */
	assert_int_equal(rxe->state, EFA_RDM_OPE_ERR);

	efa_rdm_rxe_release(rxe);
}


/**
 * @brief Common helper for testing RECEIPT/EOR packet tracking functionality
 *
 * This helper function sets up the test environment and mocks for testing
 * RECEIPT or EOR packet posting, tracking in ope_posted_ack_list, and
 * completion handling with various return codes and error conditions.
 *
 * @param[in] resource Test resource structure
 * @param[in] pkt_type Packet type (EFA_RDM_RECEIPT_PKT or EFA_RDM_EOR_PKT)
 * @param[in] post_return_code Return code for packet posting
 * @param[in] ibv_cq_status IBV completion queue status
 * @param[in] vendor_err Vendor-specific error code
 * @param[out] rxe_allocated Pointer to allocated RX operation entry
 */
static void test_efa_rdm_ope_ack_packet_tracking_common(
	struct efa_resource *resource,
	int pkt_type,
	int post_return_code,
	int ibv_cq_status,
	int vendor_err,
	struct efa_rdm_ope **rxe_allocated)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_ope *rxe;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* List should be initially empty */
	assert_true(dlist_empty(&efa_rdm_ep->ope_posted_ack_list));

	/* Allocate rx entry */
	rxe = efa_unit_test_alloc_rxe(resource, ofi_op_tagged);
	assert_non_null(rxe);

	/* Mock efa_qp_post_send to return success */
	g_efa_unit_test_mocks.efa_qp_post_send = &efa_mock_efa_qp_post_send_return_mock;
	will_return(efa_mock_efa_qp_post_send_return_mock, post_return_code);

	assert_int_equal(efa_rdm_ope_post_send_or_queue(rxe, pkt_type), -post_return_code);;
	if (!post_return_code) {
		/* Mock cq ops to simulate the send completion of the posted wr */
		g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_use_saved_send_wr_with_mock_status;
		g_efa_unit_test_mocks.efa_ibv_cq_next_poll = &efa_mock_efa_ibv_cq_next_poll_return_mock;
		g_efa_unit_test_mocks.efa_ibv_cq_end_poll = &efa_mock_efa_ibv_cq_end_poll_check_mock;
		g_efa_unit_test_mocks.efa_ibv_cq_wc_read_opcode = &efa_mock_efa_ibv_cq_wc_read_opcode_return_mock;
		g_efa_unit_test_mocks.efa_ibv_cq_wc_read_qp_num = &efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock;
		g_efa_unit_test_mocks.efa_ibv_cq_wc_read_vendor_err = &efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock;

		will_return_int(efa_mock_efa_ibv_cq_start_poll_use_saved_send_wr_with_mock_status, ibv_cq_status);
		expect_function_call(efa_mock_efa_ibv_cq_end_poll_check_mock);
		will_return_int(efa_mock_efa_ibv_cq_wc_read_opcode_return_mock, IBV_WC_SEND);
		will_return_uint(efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock, efa_rdm_ep->base_ep.qp->qp_num);
		will_return_int_maybe(efa_mock_efa_ibv_cq_next_poll_return_mock, ENOENT);
		will_return_uint_maybe(efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock, vendor_err);
	}

	*rxe_allocated = rxe;
}

/**
 * @brief Test RECEIPT/EOR packet tracking via CQ read
 *
 * This test verifies that RECEIPT or EOR packets are correctly added to
 * ope_posted_ack_list when posted and properly removed when completed
 * via fi_cq_read with successful completion status.
 *
 * @param[in] state Test resource state
 * @param[in] pkt_type Packet type (EFA_RDM_RECEIPT_PKT or EFA_RDM_EOR_PKT)
 */
static
void test_efa_rdm_ope_receit_eor_packet_tracking_cq_read_common(struct efa_resource **state, int pkt_type)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_ep *efa_rdm_ep;

	test_efa_rdm_ope_ack_packet_tracking_common(resource, pkt_type, 0, IBV_WC_SUCCESS, 0, &rxe);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* It should post a RECEIPT packet and add the rxe to the list */
	assert_false(dlist_empty(&efa_rdm_ep->ope_posted_ack_list));

	/* Poll the cq via cq read */
	(void) fi_cq_read(resource->cq, NULL, 0);

	/* The cq poll should remove the rxe from the list */
	assert_true(dlist_empty(&efa_rdm_ep->ope_posted_ack_list));
}

/**
 * @brief Test packet tracking via wait_send with successful completion
 *
 * This test verifies that RECEIPT or EOR packets are correctly added to
 * ope_posted_ack_list when posted and properly removed when completed
 * via efa_rdm_ep_wait_send with successful completion status.
 */
static
void test_efa_rdm_ope_ack_packet_tracking_wait_send_common(struct efa_resource **state, int pkt_type)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_ep *efa_rdm_ep;

	test_efa_rdm_ope_ack_packet_tracking_common(resource, pkt_type, 0, IBV_WC_SUCCESS, 0, &rxe);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* It should post a RECEIPT packet and add the rxe to the list */
	assert_false(dlist_empty(&efa_rdm_ep->ope_posted_ack_list));

	/* Poll the cq via wait_send */
	efa_rdm_ep_wait_send(efa_rdm_ep);

	/* The cq poll should remove the rxe from the list */
	assert_true(dlist_empty(&efa_rdm_ep->ope_posted_ack_list));
}

/**
 * @brief Test that failed packet posting does not add to ope_posted_ack_list
 *
 * This test verifies that when RECEIPT or EOR packet posting fails,
 * the operation is not added to the ope_posted_ack_list, ensuring
 * proper list management during error conditions.
 */
static
void test_efa_rdm_ope_ack_packet_failed_posting_common(struct efa_resource **state, int pkt_type)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_ope *rxe;

	test_efa_rdm_ope_ack_packet_tracking_common(resource, pkt_type, -FI_EINVAL, IBV_WC_SUCCESS, 0, &rxe);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* should NOT add to list due to failure */
	assert_true(dlist_empty(&efa_rdm_ep->ope_posted_ack_list));
}

/**
 * @brief Test packet tracking with unresponsive peer during wait_send
 *
 * This test verifies that wait_send does not wait for operations from
 * unresponsive peers. It simulates a peer becoming unresponsive and
 * verifies that subsequent wait_send calls skip operations from that peer.
 */
static
void test_efa_rdm_ope_ack_packet_tracking_unresponsive_wait_send_common(struct efa_resource **state, int pkt_type)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_ope *rxe, *rxe2;

	test_efa_rdm_ope_ack_packet_tracking_common(resource, pkt_type, 0, IBV_WC_GENERAL_ERR, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE, &rxe);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/*
	 * Call the 1st efa_rdm_ep_wait_send.
	 * It will first poll the cq and mark the peer as EFA_RDM_PEER_UNRESP
	 * Then it will not try to poll the cq again because of the peer becomes
	 * unresponsive. See logic in efa_rdm_ep_close_should_wait_send()
	 */
	efa_rdm_ep_wait_send(efa_rdm_ep);
	assert_true(!!(rxe->peer->flags & EFA_RDM_PEER_UNRESP));

	/* Now posting another ctrl packet against the same unresponsive peer */
	rxe2 = efa_rdm_ep_alloc_rxe(efa_rdm_ep, rxe->peer, ofi_op_tagged);
	assert_non_null(rxe2);
	will_return(efa_mock_efa_qp_post_send_return_mock, 0);
	assert_int_equal(efa_rdm_ope_post_send_or_queue(rxe2, pkt_type), 0);

	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->ope_posted_ack_list), 1);

	/* Simulate the case where we are NOT getting more unresponsive error for this peer */
	g_efa_unit_test_mocks.efa_ibv_cq_start_poll = &efa_mock_efa_ibv_cq_start_poll_return_mock;
	will_return_int_maybe(efa_mock_efa_ibv_cq_start_poll_return_mock, ENOENT);

	/* Kick off the second wait_send, which should NOT try to progress more because of the unresp peer */
	efa_rdm_ep_wait_send(efa_rdm_ep);
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->ope_posted_ack_list), 1);
}


/**
 * @brief Test RECEIPT packet tracking via CQ read
 *
 * Verifies that RECEIPT packets are correctly added to and removed from
 * ope_posted_ack_list when posted and completed via fi_cq_read.
 *
 * @param[in] state cmocka state variable
 */
void test_efa_rdm_ope_receipt_packet_tracking_cq_read(struct efa_resource **state)
{
	test_efa_rdm_ope_receit_eor_packet_tracking_cq_read_common(state, EFA_RDM_RECEIPT_PKT);
}

/**
 * @brief Test RECEIPT packet tracking via wait_send
 *
 * Verifies that RECEIPT packets are correctly added to and removed from
 * ope_posted_ack_list when posted and completed via efa_rdm_ep_wait_send.
 *
 * @param[in] state cmocka state variable
 */
void test_efa_rdm_ope_receipt_packet_tracking_wait_send(struct efa_resource **state)
{
	test_efa_rdm_ope_ack_packet_tracking_wait_send_common(state, EFA_RDM_RECEIPT_PKT);
}

/**
 * @brief Test failed RECEIPT packet posting
 *
 * Verifies that failed RECEIPT packet posting does not add operations
 * to the ope_posted_ack_list.
 *
 * @param[in] state cmocka state variable
 */
void test_efa_rdm_ope_receipt_packet_failed_posting(struct efa_resource **state)
{
	test_efa_rdm_ope_ack_packet_failed_posting_common(state, EFA_RDM_RECEIPT_PKT);
}

/**
 * @brief Test RECEIPT packet tracking with unresponsive peer
 *
 * Verifies that wait_send skips operations from unresponsive peers,
 * preventing indefinite blocking during endpoint closure.
 *
 * @param[in] state cmocka state variable
 */
void test_efa_rdm_ope_receipt_packet_tracking_unresponsive_wait_send(struct efa_resource **state)
{
	test_efa_rdm_ope_ack_packet_tracking_unresponsive_wait_send_common(state, EFA_RDM_RECEIPT_PKT);
}

/**
 * @brief Test EOR packet tracking via CQ read
 *
 * Verifies that EOR packets are correctly added to and removed from
 * ope_posted_ack_list when posted and completed via fi_cq_read.
 *
 * @param[in] state cmocka state variable
 */
void test_efa_rdm_ope_eor_packet_tracking_cq_read(struct efa_resource **state)
{
	test_efa_rdm_ope_receit_eor_packet_tracking_cq_read_common(state, EFA_RDM_EOR_PKT);
}

/**
 * @brief Test EOR packet tracking via wait_send
 *
 * Verifies that EOR packets are correctly added to and removed from
 * ope_posted_ack_list when posted and completed via efa_rdm_ep_wait_send.
 *
 * @param[in] state cmocka state variable
 */
void test_efa_rdm_ope_eor_packet_tracking_wait_send(struct efa_resource **state)
{
	test_efa_rdm_ope_ack_packet_tracking_wait_send_common(state, EFA_RDM_EOR_PKT);
}

/**
 * @brief Test failed EOR packet posting
 *
 * Verifies that failed EOR packet posting does not add operations
 * to the ope_posted_ack_list.
 *
 * @param[in] state cmocka state variable
 */
void test_efa_rdm_ope_eor_packet_failed_posting(struct efa_resource **state)
{
	test_efa_rdm_ope_ack_packet_failed_posting_common(state, EFA_RDM_EOR_PKT);
}

/**
 * @brief Test EOR packet tracking with unresponsive peer
 *
 * Verifies that wait_send skips operations from unresponsive peers,
 * preventing indefinite blocking during endpoint closure.
 *
 * @param[in] state cmocka state variable
 */
void test_efa_rdm_ope_eor_packet_tracking_unresponsive_wait_send(struct efa_resource **state)
{
	test_efa_rdm_ope_ack_packet_tracking_unresponsive_wait_send_common(state, EFA_RDM_EOR_PKT);
}

/**
 * @brief Test that atomic_ex.compare_desc array is properly copied
 * and persists after the caller's stack frame is destroyed
 *
 * This test verifies the compare_desc fix where compare_desc was a
 * pointer to stack memory that became dangling.
 *
 * @param[in]	state	struct efa_resource that is managed by the framework
 */
void test_efa_rdm_atomic_compare_desc_persistence(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff, result_buff, compare_buff;
	struct efa_ep_addr raw_addr;
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t addr;
	uint64_t operand = 0x1234567890ABCDEF;
	uint64_t compare = 0;
	void *desc_array[1];
	void *result_desc_array[1];
	int ret;
	struct fi_ioc ioc = {0};
	struct fi_ioc compare_ioc = {0};
	struct fi_ioc result_ioc = {0};
	struct fi_rma_ioc rma_ioc = {0};
	struct fi_msg_atomic msg = {0};
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;

	/* disable shm to force using efa device to send */
	efa_unit_test_resource_construct_rdm_shm_disabled(resource);

	/* Setup peer address */
	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL);
	assert_int_equal(ret, 1);

	/* Set peer flags to simulate handshake state */
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	peer->flags = EFA_RDM_PEER_REQ_SENT;
	peer->is_local = false;

	/* Setup buffers */
	efa_unit_test_buff_construct(&send_buff, resource, sizeof(uint64_t));
	efa_unit_test_buff_construct(&result_buff, resource, sizeof(uint64_t));
	efa_unit_test_buff_construct(&compare_buff, resource, sizeof(uint64_t));

	memcpy(send_buff.buff, &operand, sizeof(uint64_t));
	memcpy(compare_buff.buff, &compare, sizeof(uint64_t));

	/* Create desc array on stack */
	desc_array[0] = fi_mr_desc(send_buff.mr);
	result_desc_array[0] = fi_mr_desc(result_buff.mr);
	void *compare_desc_array[1];
	compare_desc_array[0] = fi_mr_desc(compare_buff.mr);
	void *original_desc_value = compare_desc_array[0];

	/* Setup atomic message with FI_DELIVERY_COMPLETE to force queuing */
	ioc.addr = send_buff.buff;
	ioc.count = 1;
	compare_ioc.addr = compare_buff.buff;
	compare_ioc.count = 1;
	result_ioc.addr = result_buff.buff;
	result_ioc.count = 1;

	msg.msg_iov = &ioc;
	msg.desc = desc_array;
	msg.iov_count = 1;
	msg.addr = addr;
	msg.rma_iov = &rma_ioc;
	msg.rma_iov_count = 1;
	msg.datatype = FI_UINT64;
	msg.op = FI_CSWAP;

	/*
	 * Call fi_compare_atomicmsg with FI_DELIVERY_COMPLETE.
	 * This forces the operation to be queued when handshake is not complete.
	 * The old buggy code would store a pointer to compare_desc_array,
	 * which becomes dangling when this function returns.
	 * The fix copies the array contents into txe->atomic_ex.compare_desc[].
	 */
	ret = fi_compare_atomicmsg(resource->ep, &msg, &compare_ioc, compare_desc_array, 1,
				   &result_ioc, result_desc_array, 1, FI_DELIVERY_COMPLETE);

	/* Operation should succeed (queued) */
	assert_int_equal(ret, 0);

	/* Destroy stack array to simulate function return */
	compare_desc_array[0] = (void *)(uintptr_t)0xDEADBEEF;
	
	/* Retrieve queued txe from ope_queued_list */
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	assert_false(dlist_empty(&efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list));
	txe = container_of(efa_rdm_ep_domain(efa_rdm_ep)->ope_queued_list.next,
			   struct efa_rdm_ope, queued_entry);
	
	/* Verify compare_desc was copied, not just pointer stored */
	assert_ptr_equal(txe->atomic_ex.compare_desc[0], original_desc_value);

	efa_unit_test_buff_destruct(&send_buff);
	efa_unit_test_buff_destruct(&result_buff);
	efa_unit_test_buff_destruct(&compare_buff);
}
/**
 * @brief Common helper for DC packet TXE release testing
 *
 * Sets up test environment and packet entries for DC packet testing.
 * Tests the specified completion order and verifies TXE release behavior.
 *
 * @param[in] resource Test resource structure
 * @param[in] send_first If true, send completion happens first; if false, receipt first
 */
static void test_efa_rdm_txe_dc_release_common(struct efa_resource *resource, bool send_first)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *dc_pkt_entry, *receipt_pkt_entry;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Allocate TXE and set up for DC operation */
	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;
	txe->efa_outstanding_tx_ops = 1;

	/* Create fake DC packet entry */
	dc_pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_tx_pkt_pool, EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(dc_pkt_entry);
	dc_pkt_entry->ope = txe;
	dc_pkt_entry->ep = efa_rdm_ep;
	dc_pkt_entry->peer = txe->peer;
	/* Set DC packet type in wiredata */
	struct efa_rdm_base_hdr *base_hdr = (struct efa_rdm_base_hdr *)dc_pkt_entry->wiredata;
	base_hdr->type = EFA_RDM_DC_EAGER_MSGRTM_PKT;

	/* Create fake receipt packet entry */
	receipt_pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool, EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(receipt_pkt_entry);
	receipt_pkt_entry->ope = txe;
	receipt_pkt_entry->ep = efa_rdm_ep;

	/* Verify TXE is not ready for release initially */
	assert_false(efa_rdm_txe_dc_ready_for_release(txe));
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 1);

	if (send_first) {
		/* Send completion first - should not release TXE yet */
		efa_rdm_pke_handle_send_completion(dc_pkt_entry);
		assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 1);
		assert_false(efa_rdm_txe_dc_ready_for_release(txe));

		/* Receipt handling - should now release TXE */
		efa_rdm_pke_handle_receipt_recv(receipt_pkt_entry);
	} else {
		/* Receipt handling first - should not release TXE yet */
		efa_rdm_pke_handle_receipt_recv(receipt_pkt_entry);
		assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 1);
		assert_true(txe->internal_flags & EFA_RDM_TXE_RECEIPT_RECEIVED);
		assert_false(efa_rdm_txe_dc_ready_for_release(txe));

		/* Send completion - should now release TXE */
		efa_rdm_pke_handle_send_completion(dc_pkt_entry);
	}

	/* Verify TXE is released */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 0);
}

/**
 * @brief Test DC packet TXE release with send completion first
 *
 * This test verifies the DC (Delivery Complete) TXE release logic when
 * send completion arrives before receipt acknowledgment.
 *
 * @param[in] state cmocka state variable
 */
void test_efa_rdm_txe_dc_send_first(struct efa_resource **state)
{
	test_efa_rdm_txe_dc_release_common(*state, true);
}

/**
 * @brief Test DC packet TXE release with receipt completion first
 *
 * This test verifies the race condition fix where receipt acknowledgment
 * arrives before send completion. The TXE should only be released when
 * both conditions are met, regardless of order.
 *
 * @param[in] state cmocka state variable
 */
void test_efa_rdm_txe_dc_receipt_first(struct efa_resource **state)
{
	test_efa_rdm_txe_dc_release_common(*state, false);
}