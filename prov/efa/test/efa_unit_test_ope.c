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
						size_t *expected_pkt_entry_data_size_vec)
{
	struct efa_ep_addr raw_addr;
	struct efa_rdm_mr mock_mr;
	struct efa_rdm_ope mock_txe;
	struct efa_rdm_peer mock_peer;
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t addr;
	size_t pkt_entry_cnt;
	size_t pkt_entry_data_size_vec[1024];
	int i, err, ret;

	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0 /* flags */, NULL /* context */);
	assert_int_equal(ret, 1);

	mock_mr.efa_mr.iface = iface;

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
	size_t expected_pkt_entry_data_size_vec[1024];

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
	size_t expected_pkt_entry_data_size_vec[1024];

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
	size_t expected_pkt_entry_data_size_vec[1024];

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
	size_t expected_pkt_entry_data_size_vec[1024];

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
	cuda_mr.iface = FI_HMEM_CUDA;

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

/*
 * When a multi-segment fi_send/fi_write/fi_read posts some segments
 * and then fails on a later segment, the synchronous error return to
 * the app is the sole error report for that op. The partial-post fix
 * sets EFA_RDM_TXE_NO_COMPLETION on the txe so the success-completion
 * path does not report a spurious CQ entry or counter when the
 * in-flight segment(s) eventually drain. If one of those in-flight
 * segments fails instead of succeeds, efa_rdm_txe_handle_error must
 * also honor EFA_RDM_TXE_NO_COMPLETION, or the app sees a duplicate
 * error report (one synchronous, one via the error CQ).
 *
 * These three tests cover each op type and assert: no error CQ entry
 * when NO_COMPLETION is set. The counter bump is guarded by the same
 * conditional, so suppressing the CQ write suppresses both.
 */
static
void test_efa_rdm_txe_handle_error_suppressed_impl(struct efa_resource *resource,
						   uint32_t op)
{
	struct efa_rdm_ope *txe;
	struct fi_cq_err_entry cq_err = {0};
	struct fi_cq_data_entry cq_entry;

	txe = efa_unit_test_alloc_txe(resource, op);
	assert_non_null(txe);
	txe->internal_flags |= EFA_RDM_TXE_NO_COMPLETION;

	efa_rdm_txe_handle_error(txe, FI_ENOTCONN,
				 EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);

	/*
	 * No error CQ entry and no regular CQ entry: NO_COMPLETION
	 * short-circuits before ofi_cq_write_error() and
	 * efa_cntr_report_error().
	 */
	assert_int_equal(fi_cq_read(resource->cq, &cq_entry, 1), -FI_EAGAIN);
	assert_int_equal(fi_cq_readerr(resource->cq, &cq_err, 0), -FI_EAGAIN);

	efa_rdm_txe_release(txe);
}

void test_efa_rdm_txe_handle_error_suppressed_write(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	test_efa_rdm_txe_handle_error_suppressed_impl(resource, ofi_op_write);
}

void test_efa_rdm_txe_handle_error_suppressed_read(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	test_efa_rdm_txe_handle_error_suppressed_impl(resource, ofi_op_read_req);
}

void test_efa_rdm_txe_handle_error_suppressed_send(struct efa_resource **state)
{
	struct efa_resource *resource = *state;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	test_efa_rdm_txe_handle_error_suppressed_impl(resource, ofi_op_msg);
}

/*
 * All fi_inject* entry points (fi_inject, fi_injectdata, fi_tinject,
 * fi_tinjectdata, fi_inject_write, fi_inject_writedata,
 * fi_inject_atomic) set EFA_RDM_TXE_NO_COMPLETION on the txe to
 * suppress the success CQ per inject semantics, but also set
 * FI_INJECT in fi_flags. Per the libfabric fi_msg/fi_rma man pages,
 * an inject op that succeeds synchronously but later fails
 * asynchronously MUST still report an error CQ entry. Verify that
 * efa_rdm_txe_handle_error does NOT suppress the error CQ when
 * FI_INJECT is set, even though NO_COMPLETION is also set.
 */
void test_efa_rdm_txe_handle_error_inject_still_reports_cq_error(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *txe;
	struct fi_cq_err_entry cq_err = {0};

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_write);
	assert_non_null(txe);
	txe->fi_flags |= FI_INJECT;
	txe->internal_flags |= EFA_RDM_TXE_NO_COMPLETION;

	efa_rdm_txe_handle_error(txe, FI_ENOTCONN,
				 EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);

	/* Error CQ entry must be present despite NO_COMPLETION. */
	assert_int_equal(fi_cq_readerr(resource->cq, &cq_err, 0), 1);
	assert_int_equal(cq_err.err, FI_ENOTCONN);
	assert_int_equal(cq_err.prov_errno,
			 EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);

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

	txe = ofi_buf_alloc(efa_rdm_ep->ope_pool);
	assert_non_null(txe);
	efa_rdm_txe_construct(txe, efa_rdm_ep, NULL, &msg, ofi_op_msg, 0, 0);

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
	 * Mock efa_qp_post_send to succeed so the compare atomic completes.
	 * The test verifies that compare_desc is properly copied into the txe
	 * (not just a pointer to the caller's stack array).
	 */
	g_efa_unit_test_mocks.efa_qp_post_send = &efa_mock_efa_qp_post_send_return_mock;
	will_return_int(efa_mock_efa_qp_post_send_return_mock, 0);

	ret = fi_compare_atomicmsg(resource->ep, &msg, &compare_ioc, compare_desc_array, 1,
				   &result_ioc, result_desc_array, 1, 0);
	assert_int_equal(ret, 0);

	/* Destroy stack array to simulate function return */
	compare_desc_array[0] = (void *)(uintptr_t)0xDEADBEEF;

	/* Retrieve the txe from the txe_list */
	assert_false(dlist_empty(&efa_rdm_ep->txe_list));
	txe = container_of(efa_rdm_ep->txe_list.next, struct efa_rdm_ope, ep_entry);

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
/**
 * @brief Common helper for DC packet TXE release testing
 *
 * Sets up test environment and packet entries for DC packet testing.
 * Tests the specified completion order and verifies TXE release behavior.
 *
 * @param[in] resource Test resource structure
 * @param[in] send_first If true, send completion happens first; if false, receipt first
 * @param[in] txe_in_send_state If true, TXE is in EFA_RDM_OPE_SEND state; if false, different state
 */
/**
 * @brief Common test for txe release ordering when response/ack arrives
 *
 * This tests that a txe is only released when both:
 * 1. Response/ack received (EFA_RDM_TXE_REMOTE_ACK_RECEIVED set)
 * 2. All TX ops completed (efa_outstanding_tx_ops == 0)
 *
 * @param[in] resource		test resource
 * @param[in] send_first	if true, send completion arrives before response
 * @param[in] pkt_type		request packet type to test
 */
static void test_efa_rdm_txe_with_resp_release_common(struct efa_resource *resource,
					    bool send_first, int pkt_type)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *req_pkt_entry, *resp_pkt_entry;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Allocate TXE based on protocol */
	if (pkt_type == EFA_RDM_SHORT_RTR_PKT || pkt_type == EFA_RDM_LONGCTS_RTR_PKT) {
		txe = efa_unit_test_alloc_txe(resource, ofi_op_read_req);
		txe->cq_entry.flags = FI_READ;
		/* Set len >= total_len to avoid truncation error path */
		txe->cq_entry.len = 1000;
		/* Non-zero total_len so bytes_copied != total_len initially */
		txe->total_len = 1000;
		txe->bytes_copied = 0;
		/* Ensure CQ entry is written by efa_rdm_txe_report_completion */
		txe->fi_flags |= FI_COMPLETION;
	} else if (pkt_type == EFA_RDM_FETCH_RTA_PKT || pkt_type == EFA_RDM_COMPARE_RTA_PKT) {
		txe = efa_unit_test_alloc_txe(resource, ofi_op_atomic);
		txe->cq_entry.flags = FI_ATOMIC | FI_READ;
	} else {
		/* DC protocols */
		txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
		txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;
	}
	assert_non_null(txe);
	txe->efa_outstanding_tx_ops = 1;

	/* Set txe state based on packet type */
	if (pkt_type == EFA_RDM_CTSDATA_PKT) {
		txe->state = EFA_RDM_OPE_SEND;
		dlist_insert_tail(&txe->entry, &efa_rdm_ep_domain(efa_rdm_ep)->ope_longcts_send_list);
	} else {
		txe->state = EFA_RDM_TXE_REQ;
	}

	/* Create request packet entry */
	req_pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_tx_pkt_pool, EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(req_pkt_entry);
	req_pkt_entry->ope = txe;
	req_pkt_entry->ep = efa_rdm_ep;
	req_pkt_entry->peer = txe->peer;
	struct efa_rdm_base_hdr *req_hdr = (struct efa_rdm_base_hdr *)req_pkt_entry->wiredata;
	req_hdr->type = pkt_type;
	if (pkt_type == EFA_RDM_CTSDATA_PKT) {
		req_pkt_entry->flags |= EFA_RDM_PKE_DC_LONGCTS_DATA;
		struct efa_rdm_ctsdata_hdr *ctsdata_hdr = efa_rdm_pke_get_ctsdata_hdr(req_pkt_entry);
		ctsdata_hdr->seg_length = 0;
	}

	/* Create response packet entry (not needed for RTR which uses efa_rdm_ope_handle_recv_completed) */
	if (pkt_type != EFA_RDM_SHORT_RTR_PKT && pkt_type != EFA_RDM_LONGCTS_RTR_PKT) {
		resp_pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool, EFA_RDM_PKE_FROM_EFA_RX_POOL);
		assert_non_null(resp_pkt_entry);
		resp_pkt_entry->ope = txe;
		resp_pkt_entry->ep = efa_rdm_ep;
		if (pkt_type == EFA_RDM_FETCH_RTA_PKT || pkt_type == EFA_RDM_COMPARE_RTA_PKT) {
			struct efa_rdm_atomrsp_pkt *atomrsp_pkt = (struct efa_rdm_atomrsp_pkt *)resp_pkt_entry->wiredata;
			atomrsp_pkt->hdr.type = EFA_RDM_ATOMRSP_PKT;
			atomrsp_pkt->hdr.recv_id = txe->tx_id;
			atomrsp_pkt->hdr.seg_length = 0;
			txe->atomic_ex.resp_iov_count = 0;
		} else {
			/* DC protocols use RECEIPT as response */
			struct efa_rdm_receipt_hdr *receipt_hdr = efa_rdm_pke_get_receipt_hdr(resp_pkt_entry);
			receipt_hdr->tx_id = txe->tx_id;
		}
	}

	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 1);

	if (send_first) {
		/* Send completion first - should not release TXE yet */
		efa_rdm_pke_handle_send_completion(req_pkt_entry);
		assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 1);

		/* Response arrives - should release TXE now */
		if (pkt_type == EFA_RDM_FETCH_RTA_PKT || pkt_type == EFA_RDM_COMPARE_RTA_PKT) {
			efa_rdm_pke_handle_atomrsp_recv(resp_pkt_entry);
		} else if (pkt_type == EFA_RDM_SHORT_RTR_PKT || pkt_type == EFA_RDM_LONGCTS_RTR_PKT) {
			/* Simulate all read data received and copied */
			txe->bytes_received = txe->total_len;
			txe->bytes_copied = txe->total_len;
			efa_rdm_ope_handle_recv_completed(txe);
		} else {
			efa_rdm_pke_handle_receipt_recv(resp_pkt_entry);
		}
	} else {
		/* Response arrives first - should not release TXE yet */
		if (pkt_type == EFA_RDM_FETCH_RTA_PKT || pkt_type == EFA_RDM_COMPARE_RTA_PKT) {
			efa_rdm_pke_handle_atomrsp_recv(resp_pkt_entry);
			assert_true(txe->internal_flags & EFA_RDM_TXE_REMOTE_ACK_RECEIVED);
		} else if (pkt_type == EFA_RDM_SHORT_RTR_PKT || pkt_type == EFA_RDM_LONGCTS_RTR_PKT) {
			/* Simulate all read data received and copied */
			txe->bytes_received = txe->total_len;
			txe->bytes_copied = txe->total_len;
			efa_rdm_ope_handle_recv_completed(txe);
		} else {
			efa_rdm_pke_handle_receipt_recv(resp_pkt_entry);
			assert_true(txe->internal_flags & EFA_RDM_TXE_REMOTE_ACK_RECEIVED);
		}
		assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 1);

		/* Send completion - should release TXE now */
		efa_rdm_pke_handle_send_completion(req_pkt_entry);
	}

	/* Verify TXE is released */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 0);
}

/**
 * @brief Test DC packet TXE release with send completion first (TXE in SEND state)
 *
 * This test verifies the DC (Delivery Complete) TXE release logic when
 * send completion arrives before receipt acknowledgment for long-cts TXEs.
 *
 * @param[in] state cmocka state variable
 */
void test_efa_rdm_txe_dc_ctsdata_send_first(struct efa_resource **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, true, EFA_RDM_CTSDATA_PKT);
}

/**
 * @brief Test DC packet TXE release with receipt completion first (TXE in SEND state)
 *
 * This test verifies the race condition fix where receipt acknowledgment
 * arrives before send completion. The TXE should only be released when
 * both conditions are met, regardless of order.
 *
 * @param[in] state cmocka state variable
 */
void test_efa_rdm_txe_dc_ctsdata_resp_first(struct efa_resource **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, false, EFA_RDM_CTSDATA_PKT);
}

/**
 * @brief Test DC packet TXE release with send completion first (TXE not in SEND state)
 *
 * This test verifies the DC TXE release logic for non-long-cts TXEs when
 * send completion arrives before receipt acknowledgment.
 *
 * @param[in] state cmocka state variable
 */
void test_efa_rdm_txe_dc_eager_rtm_send_first(struct efa_resource **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, true, EFA_RDM_DC_EAGER_MSGRTM_PKT);
}

/**
 * @brief Test DC packet TXE release with receipt completion first (TXE not in SEND state)
 *
 * This test verifies the bug fix where non-long-cts TXEs get the
 * EFA_RDM_TXE_REMOTE_ACK_RECEIVED flag set, allowing proper release.
 *
 * @param[in] state cmocka state variable
 */
void test_efa_rdm_txe_dc_eager_rtm_resp_first(struct efa_resource **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, false, EFA_RDM_DC_EAGER_MSGRTM_PKT);
}

/**
 * @brief Test SHORT_RTR txe release: send completion before recv completed
 */
void test_efa_rdm_txe_short_rtr_send_first(struct efa_resource **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, true, EFA_RDM_SHORT_RTR_PKT);
}

/**
 * @brief Test SHORT_RTR txe release: recv completed before send completion
 */
void test_efa_rdm_txe_short_rtr_resp_first(struct efa_resource **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, false, EFA_RDM_SHORT_RTR_PKT);
}

/**
 * @brief Test FETCH_RTA txe release: send completion before ATOMRSP
 */
void test_efa_rdm_txe_fetch_rta_send_first(struct efa_resource **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, true, EFA_RDM_FETCH_RTA_PKT);
}

/**
 * @brief Test FETCH_RTA txe release: ATOMRSP before send completion
 */
void test_efa_rdm_txe_fetch_rta_resp_first(struct efa_resource **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, false, EFA_RDM_FETCH_RTA_PKT);
}

/**
 * @brief Test COMPARE_RTA txe release: send completion before ATOMRSP
 */
void test_efa_rdm_txe_compare_rta_send_first(struct efa_resource **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, true, EFA_RDM_COMPARE_RTA_PKT);
}

/**
 * @brief Test COMPARE_RTA txe release: ATOMRSP before send completion
 */
void test_efa_rdm_txe_compare_rta_resp_first(struct efa_resource **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, false, EFA_RDM_COMPARE_RTA_PKT);
}

/**
 * @brief Common test for longcts ope release ordering with CTS send completion
 *
 * In the longcts protocol, the ope sends a CTS packet and then receives
 * CTSDATA. The ope can only be released when both:
 * 1. All data has been received (bytes_received == total_len)
 * 2. All TX ops have completed (efa_outstanding_tx_ops == 0), i.e.
 *    the CTS send completion has arrived.
 *
 * CTS can be sent by rxe (longcts msg/write) or txe (emulated longcts read).
 *
 * @param[in] resource		test resource
 * @param[in] send_first	if true, CTS send completion arrives before recv completed
 * @param[in] op		operation type (ofi_op_msg, ofi_op_write, ofi_op_read_req)
 */
static void test_efa_rdm_ope_longcts_cts_release_common(struct efa_resource *resource,
							bool send_first, uint32_t op)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_ope *ope;
	struct efa_rdm_pke *cts_pkt_entry;
	bool is_txe;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Emulated longcts read uses txe, msg/write uses rxe */
	is_txe = (op == ofi_op_read_req);

	if (is_txe) {
		ope = efa_unit_test_alloc_txe(resource, op);
		ope->cq_entry.flags = FI_READ;
	} else {
		ope = efa_unit_test_alloc_rxe(resource, op);
	}
	assert_non_null(ope);
	ope->efa_outstanding_tx_ops = 1; /* CTS packet in flight */
	ope->total_len = 1000;
	ope->bytes_received = 0;
	ope->bytes_copied = 0;
	if (is_txe)
		ope->state = EFA_RDM_TXE_REQ;
	else
		ope->state = EFA_RDM_RXE_RECV;

	/* Create fake CTS packet entry */
	cts_pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_tx_pkt_pool, EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(cts_pkt_entry);
	cts_pkt_entry->ope = ope;
	cts_pkt_entry->ep = efa_rdm_ep;
	cts_pkt_entry->peer = ope->peer;
	struct efa_rdm_base_hdr *cts_hdr = (struct efa_rdm_base_hdr *)cts_pkt_entry->wiredata;
	cts_hdr->type = EFA_RDM_CTS_PKT;

	if (is_txe)
		assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 1);
	else
		assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->rxe_list), 1);

	if (send_first) {
		/* CTS send completion first - recv not done, should not release */
		efa_rdm_pke_handle_send_completion(cts_pkt_entry);
		assert_int_equal(ope->efa_outstanding_tx_ops, 0);
		if (is_txe)
			assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 1);
		else
			assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->rxe_list), 1);

		/* Simulate recv completed */
		ope->bytes_received = ope->total_len;
		ope->bytes_copied = ope->total_len;
		efa_rdm_ope_handle_recv_completed(ope);
	} else {
		/* Simulate recv completed first - CTS still outstanding */
		ope->bytes_received = ope->total_len;
		ope->bytes_copied = ope->total_len;
		efa_rdm_ope_handle_recv_completed(ope);
		/* ope should NOT be released yet */
		if (is_txe)
			assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 1);
		else
			assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->rxe_list), 1);

		/* CTS send completion - should release ope now */
		efa_rdm_pke_handle_send_completion(cts_pkt_entry);
	}

	/* Verify ope is released */
	if (is_txe)
		assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 0);
	else
		assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->rxe_list), 0);
}

/**
 * @brief Test longcts msg rxe release: CTS send completion before recv completed
 */
void test_efa_rdm_rxe_longcts_msg_cts_send_first(struct efa_resource **state)
{
	test_efa_rdm_ope_longcts_cts_release_common(*state, true, ofi_op_msg);
}

/**
 * @brief Test longcts msg rxe release: recv completed before CTS send completion
 */
void test_efa_rdm_rxe_longcts_msg_cts_recv_first(struct efa_resource **state)
{
	test_efa_rdm_ope_longcts_cts_release_common(*state, false, ofi_op_msg);
}

/**
 * @brief Test longcts write rxe release: CTS send completion before recv completed
 */
void test_efa_rdm_rxe_longcts_write_cts_send_first(struct efa_resource **state)
{
	test_efa_rdm_ope_longcts_cts_release_common(*state, true, ofi_op_write);
}

/**
 * @brief Test longcts write rxe release: recv completed before CTS send completion
 */
void test_efa_rdm_rxe_longcts_write_cts_recv_first(struct efa_resource **state)
{
	test_efa_rdm_ope_longcts_cts_release_common(*state, false, ofi_op_write);
}

/**
 * @brief Test emulated longcts read txe release: CTS send completion before recv completed
 */
void test_efa_rdm_txe_longcts_read_cts_send_first(struct efa_resource **state)
{
	test_efa_rdm_ope_longcts_cts_release_common(*state, true, ofi_op_read_req);
}

/**
 * @brief Test emulated longcts read txe release: recv completed before CTS send completion
 */
void test_efa_rdm_txe_longcts_read_cts_recv_first(struct efa_resource **state)
{
	test_efa_rdm_ope_longcts_cts_release_common(*state, false, ofi_op_read_req);
}

/**
 * @brief Test DC longcts write rxe release: RECEIPT send completion before CTS
 *
 * In DC longcts write, the receiver rxe sends CTS, receives CTSDATA,
 * then posts a RECEIPT. If the RECEIPT send completion arrives before
 * the CTS send completion, the rxe must not be released until all
 * outstanding TX ops complete.
 */
/**
 * @brief Common test for DC longcts write rxe release with CTS and RECEIPT
 *
 * In DC longcts write, the receiver rxe sends CTS, receives CTSDATA,
 * then posts a RECEIPT via efa_rdm_ope_handle_recv_completed. The rxe
 * can only be released when all outstanding TX ops (CTS + RECEIPT)
 * have completed.
 *
 * @param[in] resource		test resource
 * @param[in] cts_first		if true, CTS send completion arrives before RECEIPT
 */
static void test_efa_rdm_rxe_dc_longcts_write_cts_receipt_order_common(
	struct efa_resource *resource, bool cts_first)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_pke *cts_pkt_entry;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Allocate RXE for DC longcts write receive */
	rxe = efa_unit_test_alloc_rxe(resource, ofi_op_write);
	assert_non_null(rxe);
	rxe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;
	/* 1 outstanding TX op: CTS in flight */
	rxe->efa_outstanding_tx_ops = 1;
	rxe->total_len = 1000;
	rxe->cq_entry.len = rxe->total_len;
	rxe->bytes_received = rxe->total_len;
	rxe->bytes_copied = rxe->total_len;
	rxe->state = EFA_RDM_RXE_RECV;

	/* Create fake CTS packet entry */
	cts_pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_tx_pkt_pool, EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(cts_pkt_entry);
	cts_pkt_entry->ope = rxe;
	cts_pkt_entry->ep = efa_rdm_ep;
	cts_pkt_entry->peer = rxe->peer;
	struct efa_rdm_base_hdr *cts_hdr = (struct efa_rdm_base_hdr *)cts_pkt_entry->wiredata;
	cts_hdr->type = EFA_RDM_CTS_PKT;

	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->rxe_list), 1);

	/*
	 * Simulate recv completed: efa_rdm_ope_handle_recv_completed will
	 * post a RECEIPT packet (because DC is requested) and set
	 * EFA_RDM_OPE_RECV_COMPLETED. Mock efa_qp_post_send so the
	 * RECEIPT posting succeeds.
	 */
	g_efa_unit_test_mocks.efa_qp_post_send = &efa_mock_efa_qp_post_send_return_mock;
	will_return(efa_mock_efa_qp_post_send_return_mock, 0);
	efa_rdm_ope_handle_recv_completed(rxe);

	/* rxe should NOT be released: CTS + RECEIPT outstanding */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->rxe_list), 1);
	assert_true(rxe->internal_flags & EFA_RDM_OPE_RECV_COMPLETED);
	assert_true(rxe->internal_flags & EFA_RDM_RXE_ACK_IN_FLIGHT);
	assert_int_equal(rxe->efa_outstanding_tx_ops, 2);

	/* Get the RECEIPT pkt entry from the mocked post */
	struct efa_rdm_pke *receipt_pkt_entry = efa_rdm_ep->send_pkt_entry_vec[0];

	if (cts_first) {
		/* CTS send completion first - RECEIPT still outstanding */
		efa_rdm_pke_handle_send_completion(cts_pkt_entry);
		assert_int_equal(rxe->efa_outstanding_tx_ops, 1);
		assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->rxe_list), 1);

		/* RECEIPT send completion - should release rxe now */
		efa_rdm_pke_handle_send_completion(receipt_pkt_entry);
	} else {
		/* RECEIPT send completion first - CTS still outstanding */
		efa_rdm_pke_handle_send_completion(receipt_pkt_entry);
		/* Now we shouldn't have such flag */
		assert_false(rxe->internal_flags & EFA_RDM_RXE_ACK_IN_FLIGHT);
		assert_int_equal(rxe->efa_outstanding_tx_ops, 1);
		assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->rxe_list), 1);

		/* CTS send completion - should release rxe now */
		efa_rdm_pke_handle_send_completion(cts_pkt_entry);
	}

	/* Verify rxe is released */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->rxe_list), 0);
}

/**
 * @brief Test DC longcts write: CTS send completion before RECEIPT
 */
void test_efa_rdm_rxe_dc_longcts_write_cts_before_receipt(struct efa_resource **state)
{
	test_efa_rdm_rxe_dc_longcts_write_cts_receipt_order_common(*state, true);
}

/**
 * @brief Test DC longcts write: RECEIPT send completion before CTS
 */
void test_efa_rdm_rxe_dc_longcts_write_receipt_before_cts(struct efa_resource **state)
{
	test_efa_rdm_rxe_dc_longcts_write_cts_receipt_order_common(*state, false);
}

/* RDM MSG 0-byte tests */
void test_efa_rdm_msg_send_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	assert_int_equal(efa_rdm_ep->efa_outstanding_tx_ops, 0);
	ret = fi_send(resource->ep, NULL, 0, NULL, addr, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(efa_rdm_ep->efa_outstanding_tx_ops, 1);
}

void test_efa_rdm_msg_sendv_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	struct iovec iov = {0};
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_sendv(resource->ep, &iov, NULL, 0, addr, NULL);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_msg_sendmsg_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	struct iovec iov = {0};
	struct fi_msg msg = {0};
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	efa_unit_test_construct_msg(&msg, &iov, 0, addr, NULL, 0, NULL);

	ret = fi_sendmsg(resource->ep, &msg, 0);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_msg_senddata_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_senddata(resource->ep, NULL, 0, NULL, 0, addr, NULL);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_msg_inject_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_inject(resource->ep, NULL, 0, addr);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_msg_injectdata_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_injectdata(resource->ep, NULL, 0, 0, addr);
	assert_int_equal(ret, 0);
}

/* RDM Tagged 0-byte tests */
void test_efa_rdm_tagged_send_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_tsend(resource->ep, NULL, 0, NULL, addr, 0, NULL);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_tagged_sendv_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	struct iovec iov = {0};
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_tsendv(resource->ep, &iov, NULL, 0, addr, 0, NULL);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_tagged_sendmsg_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	struct iovec iov = {0};
	struct fi_msg_tagged tmsg = {0};
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	efa_unit_test_construct_tmsg(&tmsg, &iov, 0, addr, NULL, 0, NULL, 0, 0);

	ret = fi_tsendmsg(resource->ep, &tmsg, 0);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_tagged_senddata_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_tsenddata(resource->ep, NULL, 0, NULL, 0, addr, 0, NULL);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_tagged_inject_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_tinject(resource->ep, NULL, 0, addr, 0);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_tagged_injectdata_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_tinjectdata(resource->ep, NULL, 0, 0, addr, 0);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_msg_send_0_byte_with_inject_flag(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct iovec iov = {0};
	struct fi_msg msg = {0};
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	efa_unit_test_construct_msg(&msg, &iov, 0, addr, NULL, 0, NULL);

	ret = fi_sendmsg(resource->ep, &msg, FI_INJECT);
	assert_int_equal(ret, 0);
}
