/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"
#include "rdm/efa_rdm_pke_cmd.h"
#include "rdm/efa_rdm_pke_nonreq.h"
#include "rdm/efa_rdm_mr.h"
#include "rdm/efa_rdm_srx.h"
#include "ofi_util.h"

typedef void (*efa_rdm_ope_handle_error_func_t)(struct efa_rdm_ope *ope, int err, int prov_errno);

/* Compose the self-detected (RDMA READ) call-site behavior: mark the
 * rxe peer-aborted, notify the sender, then drain-release the rxe.
 * Mirrors the efa_rdm_pke_handle_tx_error dispatch so the emission
 * tests exercise the real mark+emit+drain flow. */
static void mark_then_emit_peer_error(struct efa_rdm_ope *rxe, int prov_errno)
{
	/* Mirror efa_rdm_pke_handle_tx_error: if the rxe is already in
	 * peer-abort recovery, a sibling failure is a pure drain -- do
	 * not re-run mark/emit (emit asserts !EFA_RDM_OPE_PEER_ABORT_PENDING). */
	if (rxe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING) {
		efa_rdm_rxe_release_peer_abort_if_drained(rxe);
		return;
	}
	efa_rdm_rxe_mark_peer_aborted(rxe, prov_errno);
	efa_rdm_rxe_emit_peer_error(rxe, prov_errno);
	efa_rdm_rxe_release_peer_abort_if_drained(rxe);
}

/* Tear down an rxe left in the post-emit, PEER_ERROR_PKT-in-flight
 * state by a test that does not drive the real send-completion path
 * (that path is covered by a dedicated test). Drains the queued
 * PEER_ERROR_PKT and then runs the drain helper, which writes the
 * deferred RX error completion, returns the peer_rxe to the SRX, and
 * frees the rxe so ASan sees no leak. Acquires the SRX lock itself;
 * the caller must NOT hold it. */
static void peer_abort_test_release_inflight_rxe(struct efa_rdm_ep *ep,
						 struct util_srx_ctx *srx_ctx,
						 struct efa_rdm_ope *rxe)
{
	struct efa_rdm_pke *p;
	struct dlist_entry *tmp;

	ofi_genlock_lock(srx_ctx->lock);
	dlist_foreach_container_safe(&rxe->queued_pkts, struct efa_rdm_pke,
				     p, entry, tmp) {
		efa_rdm_pke_release_tx(p);
		ep->efa_outstanding_tx_ops--;
	}
	if (rxe->internal_flags & EFA_RDM_OPE_QUEUED_FLAGS) {
		dlist_remove(&rxe->queued_entry);
		rxe->internal_flags &= ~EFA_RDM_OPE_QUEUED_FLAGS;
	}
	/* Simulate the PEER_ERROR_PKT send completion draining: the
	 * drain helper writes the deferred RX error completion, returns
	 * the peer_rxe to the SRX, and releases the rxe. */
	rxe->efa_outstanding_tx_ops = 0;
	efa_rdm_rxe_release_peer_abort_if_drained(rxe);
	ofi_genlock_unlock(srx_ctx->lock);
}

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
	struct efa_rdm_peer mock_peer = {0};
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
void test_efa_rdm_ope_prepare_to_post_send_with_no_enough_tx_pkts(void **state)
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
void test_efa_rdm_ope_prepare_to_post_send_host_memory(void **state)
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
void test_efa_rdm_ope_prepare_to_post_send_host_memory_align128(void **state)
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
void test_efa_rdm_ope_prepare_to_post_send_cuda_memory(void **state)
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
void test_efa_rdm_ope_prepare_to_post_send_cuda_memory_align128(void **state)
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
void test_efa_rdm_ope_post_write_0_byte(void **state)
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
void test_efa_rdm_rxe_post_local_read_or_queue_impl(struct efa_resource *resource, int efa_rdm_pke_read_return, bool force_clone)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_ope *rxe;
	struct efa_domain *efa_domain;
	struct fid_ep *ep = NULL;
	struct ofi_bufpool *src_pool;
	enum efa_rdm_pke_alloc_type src_alloc_type;
	struct efa_rdm_mr cuda_mr = {0};
	char buf[16];
	size_t held_before;
	size_t to_post_before;
	size_t readcopy_before;
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

	if (force_clone) {
		/* The clone-swap path needs rx_readcopy_pkt_pool, which is only created when the application requests FI_HMEM.*/
		efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);
		efa_domain->util_domain.mr_mode |= FI_MR_HMEM;
		assert_int_equal(fi_endpoint(resource->domain, resource->info, &ep, NULL), 0);
		efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	} else {
		efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	}

	/* Fake a rdma read enabled device */
	efa_rdm_ep_domain(efa_rdm_ep)->device->max_rdma_size = efa_env.efa_read_segment_size;

	/* ooo pool forces efa_rdm_txe_prepare_local_read_pkt_entry() to clone
	 * into the read-copy pool and free the original */
	if (force_clone) {
		src_pool = efa_rdm_ep->rx_ooo_pkt_pool;
		src_alloc_type = EFA_RDM_PKE_FROM_OOO_POOL;
	} else {
		src_pool = efa_rdm_ep->efa_rx_pkt_pool;
		src_alloc_type = EFA_RDM_PKE_FROM_EFA_RX_POOL;
	}
	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, src_pool, src_alloc_type);
	assert_non_null(pkt_entry);
	pkt_entry->payload = pkt_entry->wiredata;
	pkt_entry->payload_size = sizeof buf;

	rxe = efa_rdm_ep_alloc_rxe(efa_rdm_ep, NULL, ofi_op_tagged);
	cuda_mr.efa_mr.iface = FI_HMEM_CUDA;

	rxe->desc[0] = &cuda_mr.efa_mr;
	rxe->iov_count = 1;
	rxe->iov[0] = iov;
	efa_rdm_pke_set_ope(pkt_entry, rxe);

	assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE), 0);

	held_before = efa_rdm_ep->efa_rx_pkts_held;
	to_post_before = efa_rdm_ep->efa_rx_pkts_to_post;
	readcopy_before = efa_rdm_ep->rx_readcopy_pkt_pool_used;

	will_return(efa_mock_efa_rdm_pke_read_return_mock, efa_rdm_pke_read_return);

	assert_int_equal(efa_rdm_rxe_post_local_read_or_queue(rxe, 0, pkt_entry, pkt_entry->payload, 16), efa_rdm_pke_read_return);

	if (efa_rdm_pke_read_return == FI_SUCCESS) {
		/* mark_held fired: held++, flag set */
		struct efa_rdm_pke *context_pkt;
		struct efa_rdm_ope *txe;
		assert_int_equal(efa_rdm_ep->efa_rx_pkts_held, held_before + 1);
		assert_int_equal(efa_rdm_ep->efa_rx_pkts_to_post, to_post_before);
		assert_true(pkt_entry->flags & EFA_RDM_PKE_HELD_BY_PROGRESS);

		/* The internal txe must be live with local_read_pkt_entry set. */
		assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE), 1);
		txe = efa_unit_test_get_first_ope(efa_rdm_ep, EFA_RDM_TXE);
		assert_true(txe->internal_flags & EFA_RDM_OPE_INTERNAL);
		assert_non_null(txe->local_read_pkt_entry);

		txe->local_read_pkt_entry->payload_size = 16;
		context_pkt = ofi_bufpool_get_ibuf(efa_rdm_ep->efa_tx_pkt_pool, 0);
		context_pkt->flags |= EFA_RDM_PKE_LOCAL_READ;
		efa_rdm_pke_handle_rma_completion(context_pkt);

		/* handle_data_copied released the held pkt: held back to baseline, to_post ++ */
		assert_int_equal(efa_rdm_ep->efa_rx_pkts_held, held_before);
		assert_int_equal(efa_rdm_ep->efa_rx_pkts_to_post, to_post_before + 1);
		assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE), 0);
	} else {
		/* On error, efa_rdm_rxe_post_local_read_or_queue() now owns and
		 * releases the rx pkt itself.
		 * The caller must NOT release it again.
		 */
		assert_int_equal(efa_rdm_ep->efa_rx_pkts_held, held_before);
		if (force_clone) {
			assert_int_equal(efa_rdm_ep->rx_readcopy_pkt_pool_used, readcopy_before);
		} else {
			assert_int_equal(efa_rdm_ep->efa_rx_pkts_to_post, to_post_before + 1);
		}
		/* The internal txe must be cleaned up for a failed read. */
		assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE), 0);
	}

	/* The clone variant uses a dedicated endpoint*/
	if (force_clone)
		assert_int_equal(fi_close(&ep->fid), 0);
}

void test_efa_rdm_rxe_post_local_read_or_queue_unhappy(void **state)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_resource *resource = *state;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	test_efa_rdm_rxe_post_local_read_or_queue_impl(resource, -FI_ENOMR, false);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Make sure txe is cleaned for a failed read */
	assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE), 0);
}

void test_efa_rdm_rxe_post_local_read_or_queue_happy(void **state)
{
	struct efa_resource *resource = *state;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	/*
	 * The impl drives the read-completion handler on success, which
	 * releases both the held rx pkt and the internal txe. No additional
	 * cleanup is required here.
	 */
	test_efa_rdm_rxe_post_local_read_or_queue_impl(resource, FI_SUCCESS, false);
}

/**
 * @brief Verify the clone-swap error path of efa_rdm_rxe_post_local_read_or_queue.
 */
void test_efa_rdm_rxe_post_local_read_or_queue_clone_error(void **state)
{
	struct efa_resource *resource = *state;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	test_efa_rdm_rxe_post_local_read_or_queue_impl(resource, -FI_ENOMR, true);
}

static
void test_efa_rdm_ope_handle_error_impl(
	struct efa_resource *resource,
	efa_rdm_ope_handle_error_func_t efa_rdm_ope_handle_error,
	struct efa_rdm_ope *ope, bool expect_cq_error)
{
	struct fi_cq_data_entry cq_entry;
	struct fi_cq_err_entry cq_err_entry = {0};
	struct fi_eq_err_entry eq_err_entry = {0};

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

void test_efa_rdm_txe_handle_error_write_cq(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *txe;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_write);
	assert_non_null(txe);

	test_efa_rdm_ope_handle_error_impl(resource, efa_rdm_txe_handle_error, txe, true);

	efa_rdm_txe_release(txe);
}

void test_efa_rdm_txe_handle_error_not_write_cq(void **state)
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

void test_efa_rdm_txe_handle_error_suppressed_write(void **state)
{
	struct efa_resource *resource = *state;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	test_efa_rdm_txe_handle_error_suppressed_impl(resource, ofi_op_write);
}

void test_efa_rdm_txe_handle_error_suppressed_read(void **state)
{
	struct efa_resource *resource = *state;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	test_efa_rdm_txe_handle_error_suppressed_impl(resource, ofi_op_read_req);
}

void test_efa_rdm_txe_handle_error_suppressed_send(void **state)
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
void test_efa_rdm_txe_handle_error_inject_still_reports_cq_error(void **state)
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

void test_efa_rdm_rxe_handle_error_write_cq(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *rxe;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	rxe = efa_unit_test_alloc_rxe(resource, ofi_op_tagged);
	assert_non_null(rxe);

	test_efa_rdm_ope_handle_error_impl(resource, efa_rdm_rxe_handle_error, rxe, true);

	efa_rdm_rxe_release(rxe);
}

void test_efa_rdm_rxe_handle_error_not_write_cq(void **state)
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

void test_efa_rdm_rxe_map(void **state)
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

void test_efa_rdm_rxe_list_removal(void **state)
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
	dlist_insert_tail(&rxe->entry, &efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_longcts_send_list);
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_longcts_send_list), 1);

	/* Lists should be empty after releasing the ope */
	efa_rdm_rxe_release(rxe);
	dlist_empty(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_longcts_send_list);
}

void test_efa_rdm_txe_list_removal(void **state)
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
	dlist_insert_tail(&txe->entry, &efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_longcts_send_list);
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_longcts_send_list), 1);

	txe->internal_flags |= EFA_RDM_OPE_QUEUED_CTRL;
	dlist_insert_tail(&txe->queued_entry, &efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_queued_list);
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_queued_list), 1);

	/* Lists should be empty after releasing the ope */
	efa_rdm_txe_release(txe);
	assert_true(dlist_empty(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_longcts_send_list));
	assert_true(dlist_empty(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_queued_list));
}

void test_efa_rdm_txe_prepare_local_read_pkt_entry(void **state)
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

	txe = ofi_buf_alloc(efa_rdm_ep->base_ep.ope_pool);
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
void test_efa_rdm_txe_handle_error_queue_flags_cleanup(void **state)
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
	dlist_insert_tail(&txe->queued_entry, &efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_queued_list);

	/* Verify txe is in queued list */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_queued_list), 1);
	assert_true(txe->internal_flags & EFA_RDM_OPE_QUEUED_CTRL);

	/* Handle error - this should clean up queue flags */
	efa_rdm_txe_handle_error(txe, FI_ENOTCONN, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);

	/* Verify queue flags are cleaned up */
	assert_false(txe->internal_flags & EFA_RDM_OPE_QUEUED_CTRL);
	assert_true(dlist_empty(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_queued_list));

	/* Release should not cause duplicate dlist_remove */
	efa_rdm_txe_release(txe);
}

/**
 * @brief Test that queue flags are properly cleaned up for rxe error handling
 */
void test_efa_rdm_rxe_handle_error_queue_flags_cleanup(void **state)
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
	dlist_insert_tail(&rxe->queued_entry, &efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_queued_list);

	/* Verify rxe is in queued list */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_queued_list), 1);
	assert_true(rxe->internal_flags & EFA_RDM_OPE_QUEUED_READ);

	/* Handle error - this should clean up queue flags */
	efa_rdm_rxe_handle_error(rxe, FI_ENOTCONN, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);

	/* Verify queue flags are cleaned up */
	assert_false(rxe->internal_flags & EFA_RDM_OPE_QUEUED_READ);
	assert_true(dlist_empty(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_queued_list));

	/* Release should not cause duplicate dlist_remove */
	efa_rdm_rxe_release(rxe);
}

/**
 * @brief Test error state prevents duplicate error handling for txe
 *
 * This test verifies that a new error state prevents duplicate error handling.
 */
void test_efa_rdm_txe_handle_error_duplicate_prevention(void **state)
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
	dlist_insert_tail(&txe->entry, &efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_longcts_send_list);

	/* Verify txe is in the list */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_longcts_send_list), 1);

	/* First error handling call */
	efa_rdm_txe_handle_error(txe, FI_ENOTCONN, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);

	/* Verify txe is removed from list and in error state */
	assert_true(dlist_empty(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_longcts_send_list));
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
void test_efa_rdm_rxe_handle_error_duplicate_prevention(void **state)
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
	dlist_insert_tail(&rxe->entry, &efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_longcts_send_list);

	/* Verify rxe is in the list */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_longcts_send_list), 1);

	/* First error handling call */
	efa_rdm_rxe_handle_error(rxe, FI_ENOTCONN, EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE);

	/* Verify rxe is removed from list and in error state */
	assert_true(dlist_empty(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_longcts_send_list));
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
void test_efa_rdm_ope_receit_eor_packet_tracking_cq_read_common(void **state, int pkt_type)
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
void test_efa_rdm_ope_ack_packet_tracking_wait_send_common(void **state, int pkt_type)
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
void test_efa_rdm_ope_ack_packet_failed_posting_common(void **state, int pkt_type)
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
void test_efa_rdm_ope_ack_packet_tracking_unresponsive_wait_send_common(void **state, int pkt_type)
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
void test_efa_rdm_ope_receipt_packet_tracking_cq_read(void **state)
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
void test_efa_rdm_ope_receipt_packet_tracking_wait_send(void **state)
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
void test_efa_rdm_ope_receipt_packet_failed_posting(void **state)
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
void test_efa_rdm_ope_receipt_packet_tracking_unresponsive_wait_send(void **state)
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
void test_efa_rdm_ope_eor_packet_tracking_cq_read(void **state)
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
void test_efa_rdm_ope_eor_packet_tracking_wait_send(void **state)
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
void test_efa_rdm_ope_eor_packet_failed_posting(void **state)
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
void test_efa_rdm_ope_eor_packet_tracking_unresponsive_wait_send(void **state)
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
void test_efa_rdm_atomic_compare_desc_persistence(void **state)
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
	assert_true(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE) > 0);
	txe = efa_unit_test_get_first_ope(efa_rdm_ep, EFA_RDM_TXE);

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
	} else if (pkt_type == EFA_RDM_LONGREAD_MSGRTM_PKT || pkt_type == EFA_RDM_LONGREAD_TAGRTM_PKT) {
		txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
		txe->cq_entry.flags = FI_SEND | FI_MSG;
		txe->total_len = 1000;
		txe->bytes_runt = 0;
		txe->bytes_acked = 0;
		txe->fi_flags |= FI_COMPLETION;
	} else if (pkt_type == EFA_RDM_LONGREAD_RTW_PKT) {
		txe = efa_unit_test_alloc_txe(resource, ofi_op_write);
		txe->cq_entry.flags = FI_WRITE | FI_RMA;
		txe->total_len = 1000;
		txe->bytes_runt = 0;
		txe->bytes_acked = 0;
		txe->fi_flags |= FI_COMPLETION;
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
		dlist_insert_tail(&txe->entry, &efa_rdm_ep_rdm_domain(efa_rdm_ep)->ope_longcts_send_list);
	} else {
		txe->state = EFA_RDM_TXE_REQ;
	}

	/* Create request packet entry */
	req_pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_tx_pkt_pool, EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(req_pkt_entry);
	efa_rdm_pke_set_ope(req_pkt_entry, txe);
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
		efa_rdm_pke_set_ope(resp_pkt_entry, txe);
		resp_pkt_entry->ep = efa_rdm_ep;
		if (pkt_type == EFA_RDM_FETCH_RTA_PKT || pkt_type == EFA_RDM_COMPARE_RTA_PKT) {
			struct efa_rdm_atomrsp_pkt *atomrsp_pkt = (struct efa_rdm_atomrsp_pkt *)resp_pkt_entry->wiredata;
			atomrsp_pkt->hdr.type = EFA_RDM_ATOMRSP_PKT;
			atomrsp_pkt->hdr.recv_id = txe->tx_id;
			atomrsp_pkt->hdr.seg_length = 0;
			txe->atomic_ex.resp_iov_count = 0;
		} else if (pkt_type == EFA_RDM_LONGREAD_MSGRTM_PKT ||
			   pkt_type == EFA_RDM_LONGREAD_TAGRTM_PKT ||
			   pkt_type == EFA_RDM_LONGREAD_RTW_PKT) {
			/* EOR as response for longread protocols */
			struct efa_rdm_eor_hdr *eor_hdr = (struct efa_rdm_eor_hdr *)resp_pkt_entry->wiredata;
			eor_hdr->type = EFA_RDM_EOR_PKT;
			eor_hdr->send_id = txe->tx_id;
		} else {
			/* DC protocols use RECEIPT as response */
			struct efa_rdm_receipt_hdr *receipt_hdr = efa_rdm_pke_get_receipt_hdr(resp_pkt_entry);
			receipt_hdr->tx_id = txe->tx_id;
		}
	}

	assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE), 1);

	if (send_first) {
		/* Send completion first - should not release TXE yet */
		efa_rdm_pke_handle_send_completion(req_pkt_entry);
		assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE), 1);

		/* Response arrives - should release TXE now */
		if (pkt_type == EFA_RDM_FETCH_RTA_PKT || pkt_type == EFA_RDM_COMPARE_RTA_PKT) {
			efa_rdm_pke_handle_atomrsp_recv(resp_pkt_entry);
		} else if (pkt_type == EFA_RDM_SHORT_RTR_PKT || pkt_type == EFA_RDM_LONGCTS_RTR_PKT) {
			/* Simulate all read data received and copied */
			txe->bytes_received = txe->total_len;
			txe->bytes_copied = txe->total_len;
			efa_rdm_ope_handle_recv_completed(txe);
		} else if (pkt_type == EFA_RDM_LONGREAD_MSGRTM_PKT ||
			   pkt_type == EFA_RDM_LONGREAD_TAGRTM_PKT ||
			   pkt_type == EFA_RDM_LONGREAD_RTW_PKT) {
			efa_rdm_pke_handle_eor_recv(resp_pkt_entry);
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
		} else if (pkt_type == EFA_RDM_LONGREAD_MSGRTM_PKT ||
			   pkt_type == EFA_RDM_LONGREAD_TAGRTM_PKT ||
			   pkt_type == EFA_RDM_LONGREAD_RTW_PKT) {
			efa_rdm_pke_handle_eor_recv(resp_pkt_entry);
			assert_true(txe->internal_flags & EFA_RDM_TXE_REMOTE_ACK_RECEIVED);
		} else {
			efa_rdm_pke_handle_receipt_recv(resp_pkt_entry);
			assert_true(txe->internal_flags & EFA_RDM_TXE_REMOTE_ACK_RECEIVED);
		}
		assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE), 1);

		/* Send completion - should release TXE now */
		efa_rdm_pke_handle_send_completion(req_pkt_entry);
	}

	/* Verify TXE is released */
	assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE), 0);
}

/**
 * @brief Test DC packet TXE release with send completion first (TXE in SEND state)
 *
 * This test verifies the DC (Delivery Complete) TXE release logic when
 * send completion arrives before receipt acknowledgment for long-cts TXEs.
 *
 * @param[in] state cmocka state variable
 */
void test_efa_rdm_txe_dc_ctsdata_send_first(void **state)
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
void test_efa_rdm_txe_dc_ctsdata_resp_first(void **state)
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
void test_efa_rdm_txe_dc_eager_rtm_send_first(void **state)
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
void test_efa_rdm_txe_dc_eager_rtm_resp_first(void **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, false, EFA_RDM_DC_EAGER_MSGRTM_PKT);
}

/**
 * @brief Test SHORT_RTR txe release: send completion before recv completed
 */
void test_efa_rdm_txe_short_rtr_send_first(void **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, true, EFA_RDM_SHORT_RTR_PKT);
}

/**
 * @brief Test SHORT_RTR txe release: recv completed before send completion
 */
void test_efa_rdm_txe_short_rtr_resp_first(void **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, false, EFA_RDM_SHORT_RTR_PKT);
}

/**
 * @brief Test FETCH_RTA txe release: send completion before ATOMRSP
 */
void test_efa_rdm_txe_fetch_rta_send_first(void **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, true, EFA_RDM_FETCH_RTA_PKT);
}

/**
 * @brief Test FETCH_RTA txe release: ATOMRSP before send completion
 */
void test_efa_rdm_txe_fetch_rta_resp_first(void **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, false, EFA_RDM_FETCH_RTA_PKT);
}

/**
 * @brief Test COMPARE_RTA txe release: send completion before ATOMRSP
 */
void test_efa_rdm_txe_compare_rta_send_first(void **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, true, EFA_RDM_COMPARE_RTA_PKT);
}

/**
 * @brief Test COMPARE_RTA txe release: ATOMRSP before send completion
 */
void test_efa_rdm_txe_compare_rta_resp_first(void **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, false, EFA_RDM_COMPARE_RTA_PKT);
}

/**
 * @brief Test LONGREAD_TAGRTM txe release: send completion before EOR
 */
void test_efa_rdm_txe_longread_tagrtm_send_first(void **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, true, EFA_RDM_LONGREAD_TAGRTM_PKT);
}

/**
 * @brief Test LONGREAD_TAGRTM txe release: EOR before send completion
 */
void test_efa_rdm_txe_longread_tagrtm_resp_first(void **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, false, EFA_RDM_LONGREAD_TAGRTM_PKT);
}

/**
 * @brief Test LONGREAD_MSGRTM txe release: send completion before EOR
 */
void test_efa_rdm_txe_longread_msgrtm_send_first(void **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, true, EFA_RDM_LONGREAD_MSGRTM_PKT);
}

/**
 * @brief Test LONGREAD_MSGRTM txe release: EOR before send completion
 */
void test_efa_rdm_txe_longread_msgrtm_resp_first(void **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, false, EFA_RDM_LONGREAD_MSGRTM_PKT);
}

/**
 * @brief Test LONGREAD_RTW txe release: send completion before EOR
 */
void test_efa_rdm_txe_longread_rtw_send_first(void **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, true, EFA_RDM_LONGREAD_RTW_PKT);
}

/**
 * @brief Test LONGREAD_RTW txe release: EOR before send completion
 */
void test_efa_rdm_txe_longread_rtw_resp_first(void **state)
{
	test_efa_rdm_txe_with_resp_release_common(*state, false, EFA_RDM_LONGREAD_RTW_PKT);
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
	efa_rdm_pke_set_ope(cts_pkt_entry, ope);
	cts_pkt_entry->ep = efa_rdm_ep;
	cts_pkt_entry->peer = ope->peer;
	struct efa_rdm_base_hdr *cts_hdr = (struct efa_rdm_base_hdr *)cts_pkt_entry->wiredata;
	cts_hdr->type = EFA_RDM_CTS_PKT;

	if (is_txe)
		assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE), 1);
	else
		assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_RXE), 1);

	if (send_first) {
		/* CTS send completion first - recv not done, should not release */
		efa_rdm_pke_handle_send_completion(cts_pkt_entry);
		assert_int_equal(ope->efa_outstanding_tx_ops, 0);
		if (is_txe)
			assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE), 1);
		else
			assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_RXE), 1);

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
			assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE), 1);
		else
			assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_RXE), 1);

		/* CTS send completion - should release ope now */
		efa_rdm_pke_handle_send_completion(cts_pkt_entry);
	}

	/* Verify ope is released */
	if (is_txe)
		assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE), 0);
	else
		assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_RXE), 0);
}

/**
 * @brief Test longcts msg rxe release: CTS send completion before recv completed
 */
void test_efa_rdm_rxe_longcts_msg_cts_send_first(void **state)
{
	test_efa_rdm_ope_longcts_cts_release_common(*state, true, ofi_op_msg);
}

/**
 * @brief Test longcts msg rxe release: recv completed before CTS send completion
 */
void test_efa_rdm_rxe_longcts_msg_cts_recv_first(void **state)
{
	test_efa_rdm_ope_longcts_cts_release_common(*state, false, ofi_op_msg);
}

/**
 * @brief Test longcts write rxe release: CTS send completion before recv completed
 */
void test_efa_rdm_rxe_longcts_write_cts_send_first(void **state)
{
	test_efa_rdm_ope_longcts_cts_release_common(*state, true, ofi_op_write);
}

/**
 * @brief Test longcts write rxe release: recv completed before CTS send completion
 */
void test_efa_rdm_rxe_longcts_write_cts_recv_first(void **state)
{
	test_efa_rdm_ope_longcts_cts_release_common(*state, false, ofi_op_write);
}

/**
 * @brief Test emulated longcts read txe release: CTS send completion before recv completed
 */
void test_efa_rdm_txe_longcts_read_cts_send_first(void **state)
{
	test_efa_rdm_ope_longcts_cts_release_common(*state, true, ofi_op_read_req);
}

/**
 * @brief Test emulated longcts read txe release: recv completed before CTS send completion
 */
void test_efa_rdm_txe_longcts_read_cts_recv_first(void **state)
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
	efa_rdm_pke_set_ope(cts_pkt_entry, rxe);
	cts_pkt_entry->ep = efa_rdm_ep;
	cts_pkt_entry->peer = rxe->peer;
	struct efa_rdm_base_hdr *cts_hdr = (struct efa_rdm_base_hdr *)cts_pkt_entry->wiredata;
	cts_hdr->type = EFA_RDM_CTS_PKT;

	assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_RXE), 1);

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
	assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_RXE), 1);
	assert_true(rxe->internal_flags & EFA_RDM_OPE_RECV_COMPLETED);
	assert_true(rxe->internal_flags & EFA_RDM_RXE_ACK_IN_FLIGHT);
	assert_int_equal(rxe->efa_outstanding_tx_ops, 2);

	/* Get the RECEIPT pkt entry from the mocked post */
	struct efa_rdm_pke *receipt_pkt_entry = efa_rdm_ep->send_pkt_entry_vec[0];

	if (cts_first) {
		/* CTS send completion first - RECEIPT still outstanding */
		efa_rdm_pke_handle_send_completion(cts_pkt_entry);
		assert_int_equal(rxe->efa_outstanding_tx_ops, 1);
		assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_RXE), 1);

		/* RECEIPT send completion - should release rxe now */
		efa_rdm_pke_handle_send_completion(receipt_pkt_entry);
	} else {
		/* RECEIPT send completion first - CTS still outstanding */
		efa_rdm_pke_handle_send_completion(receipt_pkt_entry);
		/* Now we shouldn't have such flag */
		assert_false(rxe->internal_flags & EFA_RDM_RXE_ACK_IN_FLIGHT);
		assert_int_equal(rxe->efa_outstanding_tx_ops, 1);
		assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_RXE), 1);

		/* CTS send completion - should release rxe now */
		efa_rdm_pke_handle_send_completion(cts_pkt_entry);
	}

	/* Verify rxe is released */
	assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_RXE), 0);
}

/**
 * @brief Test DC longcts write: CTS send completion before RECEIPT
 */
void test_efa_rdm_rxe_dc_longcts_write_cts_before_receipt(void **state)
{
	test_efa_rdm_rxe_dc_longcts_write_cts_receipt_order_common(*state, true);
}

/**
 * @brief Test DC longcts write: RECEIPT send completion before CTS
 */
void test_efa_rdm_rxe_dc_longcts_write_receipt_before_cts(void **state)
{
	test_efa_rdm_rxe_dc_longcts_write_cts_receipt_order_common(*state, false);
}

/* RDM MSG 0-byte tests */
void test_efa_rdm_msg_send_0_byte_no_shm(void **state)
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

void test_efa_rdm_msg_sendv_0_byte_no_shm(void **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	struct iovec iov = {0};
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_sendv(resource->ep, &iov, NULL, 0, addr, NULL);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_msg_sendmsg_0_byte_no_shm(void **state)
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

void test_efa_rdm_msg_senddata_0_byte_no_shm(void **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_senddata(resource->ep, NULL, 0, NULL, 0, addr, NULL);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_msg_inject_0_byte_no_shm(void **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_inject(resource->ep, NULL, 0, addr);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_msg_injectdata_0_byte_no_shm(void **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_injectdata(resource->ep, NULL, 0, 0, addr);
	assert_int_equal(ret, 0);
}

/* RDM Tagged 0-byte tests */
void test_efa_rdm_tagged_send_0_byte_no_shm(void **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_tsend(resource->ep, NULL, 0, NULL, addr, 0, NULL);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_tagged_sendv_0_byte_no_shm(void **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	struct iovec iov = {0};
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_tsendv(resource->ep, &iov, NULL, 0, addr, 0, NULL);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_tagged_sendmsg_0_byte_no_shm(void **state)
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

void test_efa_rdm_tagged_senddata_0_byte_no_shm(void **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_tsenddata(resource->ep, NULL, 0, NULL, 0, addr, 0, NULL);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_tagged_inject_0_byte_no_shm(void **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_tinject(resource->ep, NULL, 0, addr, 0);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_tagged_injectdata_0_byte_no_shm(void **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_tinjectdata(resource->ep, NULL, 0, 0, addr, 0);
	assert_int_equal(ret, 0);
}

void test_efa_rdm_msg_send_0_byte_with_inject_flag(void **state)
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


/**
 * @brief Matched recv, peer-clean abort: mark defers all user-visible
 *        work until WR drain, then writes a clean RX error completion
 *        (FI_ECANCELED / FI_EFA_ERR_PEER_ABORTED) and reaps the rxe.
 *
 * Asserts the two-stage contract: efa_rdm_rxe_mark_peer_aborted() does
 * nothing application-visible (no CQ entry while a WR is outstanding),
 * and efa_rdm_rxe_release_peer_abort_if_drained() is a no-op until
 * efa_outstanding_tx_ops reaches 0, at which point it writes the error
 * completion and frees the rxe + peer_rxe.
 */
void test_efa_rdm_rxe_peer_abort_writes_error_completion_at_drain(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct fi_cq_err_entry err_entry;
	fi_addr_t peer_addr = 0;
	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(efa_rdm_ep);
	peer_srx = util_get_peer_srx(efa_rdm_ep->peer_srx_ep);

	/* Create a fake peer so the error path can reach peer info. */
	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	assert_non_null(peer);

	/* Post a recv and match it. */
	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_recv(efa_rdm_ep->peer_srx_ep, &iov, &desc, 1,
				    FI_ADDR_UNSPEC, /*context=*/(void *) 0xa1, 0);
	assert_int_equal(ret, FI_SUCCESS);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);
	assert_non_null(peer_rxe);

	/* Build an rxe that owns the matched peer_rxe. */
	rxe = efa_rdm_ep_alloc_rxe(efa_rdm_ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->peer_rxe = peer_rxe;
	rxe->cq_entry.op_context = peer_rxe->context;
	rxe->cq_entry.flags = FI_RECV | FI_MSG;
	rxe->bytes_received = 0;
	rxe->bytes_copied = 0;

	/* Simulate one in-flight device WR (e.g. a LONGREAD RDMA READ)
	 * still using the rxe as wr_id. */
	rxe->efa_outstanding_tx_ops = 1;

	/* First failure: mark only. No CQ entry, peer_rxe untouched,
	 * rxe still alive. */
	efa_rdm_rxe_mark_peer_aborted(rxe,
		EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);
	assert_true(rxe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_non_null(rxe->peer_rxe);

	/* Not yet drained: the helper is a no-op. */
	efa_rdm_rxe_release_peer_abort_if_drained(rxe);
	assert_non_null(rxe->peer_rxe);
	ofi_genlock_unlock(srx_ctx->lock);
	assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);

	/* Last WR drains. Now the helper writes the error completion,
	 * frees peer_rxe, and releases the rxe. */
	ofi_genlock_lock(srx_ctx->lock);
	rxe->efa_outstanding_tx_ops = 0;
	efa_rdm_rxe_release_peer_abort_if_drained(rxe);
	ofi_genlock_unlock(srx_ctx->lock);

	/* The peer_rxe was returned to the SRX (freed), so it is no
	 * longer posted in msg_queue. */
	assert_true(slist_empty(&srx_ctx->msg_queue));

	/* User sees a clean, dedicated peer-abort error completion. */
	memset(&err_entry, 0, sizeof(err_entry));
	ret = fi_cq_readerr(resource->cq, &err_entry, 0);
	assert_int_equal(ret, 1);
	assert_int_equal(err_entry.err, FI_ECANCELED);
	assert_int_equal(err_entry.prov_errno, FI_EFA_ERR_PEER_ABORTED);
}

/**
 * @brief Spec-compliance for multi-recv on peer-abort: 1 success + 1 failure
 *        on a buffer big enough for 3 messages.
 *
 * Posts a single multi-recv buffer sized for three 32-byte messages
 * (96 bytes, min_multi_recv_size = 32), then:
 *   1. Carves a child for message #1 and writes its success CQ entry
 *      via the standard SRX free_entry path (simulating what happens
 *      when the receive completes normally).
 *   2. Carves a child for message #2 and runs it through
 *      efa_rdm_rxe_mark_peer_aborted + drain as if its in-protocol
 *      device op failed with a peer-clean abort
 *      (REMOTE_ERROR_BAD_ADDRESS).
 *
 * Spec rules being verified (fi_cq(3) / fi_msg(3) / util_srx):
 *
 *   - Message #1 produces a successful FI_RECV completion
 *     (fi_cq(3): "each operation gets a completion").
 *   - Message #2 produces an error CQ entry readable via
 *     fi_cq_readerr (fi_cq(3): "operations which fail are reported
 *     out of band"). The handler also frees peer_rxe2 via the SRX
 *     so multi_recv_ref bookkeeping stays correct.
 *   - The owner buffer is NOT yet released. fi_msg(3) generates
 *     the FI_MULTI_RECV release entry only when the buffer is
 *     consumed; util_free_entry's strict `<` check on
 *     min_multi_recv_size means 96 - 32 - 32 = 32 bytes
 *     remaining (NOT < 32) keeps the owner alive in the SRX
 *     queue, ready to match a third message.
 *   - No FI_MULTI_RECV CQ entry is produced; the user CQ contains
 *     exactly the message #1 success and the message #2 error.
 *   - The owner stays in srx_ctx->msg_queue with multi_recv_ref
 *     back to zero (no in-flight) and 32 bytes of remaining
 *     capacity.
 *
 * This test deliberately verifies the negative case: a peer-abort
 * error on a multi-recv child does NOT cause the buffer to be
 * "returned" to the user early. The buffer is only released when
 * its remaining size drops strictly below min_multi_recv_size, per
 * the standard FI_MULTI_RECV consumption rule.
 */
void test_efa_rdm_rxe_mark_peer_aborted_multi_recv_writes_err(
	void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe1 = NULL, *peer_rxe2 = NULL;
	struct util_rx_entry *owner_entry;
	struct efa_rdm_ope *rxe2;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct fi_cq_err_entry err_entry;
	struct fi_cq_data_entry cq_entry;
	fi_addr_t peer_addr = 0;
	struct iovec iov;
	char buf[96];
	void *desc = NULL;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(efa_rdm_ep);
	peer_srx = util_get_peer_srx(efa_rdm_ep->peer_srx_ep);

	/* 96-byte buffer, 32-byte messages, min_multi_recv_size = 32:
	 *   - After carve #1 (32B): 64 left, 64 >= 32, owner stays.
	 *   - After carve #2 (32B): 32 left, NOT < 32, owner stays.
	 *
	 * util_free_entry releases the owner only when both
	 * multi_recv_ref hits 0 AND the remaining size is strictly less
	 * than min_multi_recv_size. After 1 success + 1 fail, the
	 * remaining 32 bytes (room for one more 32-byte message) keep
	 * the owner alive in the SRX queue. No FI_MULTI_RECV release
	 * entry fires until a future match consumes that last slot. */
	srx_ctx->min_multi_recv_size = 32;

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len),
			 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr,
				      0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	assert_non_null(peer);

	/* 1. Post a multi-recv. */
	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_recv(efa_rdm_ep->peer_srx_ep, &iov, &desc, 1,
				    FI_ADDR_UNSPEC,
				    /*context=*/(void *) 0x8AB071EDul,
				    FI_MULTI_RECV);
	assert_int_equal(ret, FI_SUCCESS);
	assert_false(slist_empty(&srx_ctx->msg_queue));
	owner_entry = container_of(srx_ctx->msg_queue.head,
				   struct util_rx_entry, s_entry);
	assert_int_equal(owner_entry->peer_entry.msg_size, 96);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0;
	match_attr.msg_size = 32;

	ofi_genlock_lock(srx_ctx->lock);

	/* 2. Match message #1, simulate success. We don't go through
	 *    the full provider receive path here (no real packet); we
	 *    just call free_entry the way the success path does after
	 *    delivering data, and we write the FI_RECV CQ entry the
	 *    way efa_rdm_rxe_report_completion would. */
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe1);
	assert_int_equal(ret, FI_SUCCESS);
	assert_non_null(peer_rxe1);
	assert_ptr_equal(peer_rxe1->owner_context, owner_entry);
	assert_int_equal(owner_entry->multi_recv_ref, 1);
	assert_int_equal(owner_entry->peer_entry.msg_size, 64);

	/* Simulate success completion for message #1 by writing
	 * directly to the user CQ — this is the contract the standard
	 * receive path satisfies on success. */
	ret = ofi_peer_cq_write(efa_rdm_ep->base_ep.util_ep.rx_cq,
				peer_rxe1->context, FI_RECV | FI_MSG,
				match_attr.msg_size, peer_rxe1->iov[0].iov_base,
				0, 0, FI_ADDR_NOTAVAIL);
	assert_int_equal(ret, 0);

	peer_srx->owner_ops->free_entry(peer_rxe1);
	assert_int_equal(owner_entry->multi_recv_ref, 0);

	/* Owner remains queued with 64 bytes left for two more
	 * messages — confirm the standard success path did not release
	 * the buffer. */
	assert_false(slist_empty(&srx_ctx->msg_queue));

	/* 3. Match message #2 — the one that's about to fail. */
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe2);
	assert_int_equal(ret, FI_SUCCESS);
	assert_non_null(peer_rxe2);
	assert_ptr_equal(peer_rxe2->owner_context, owner_entry);
	assert_int_equal(owner_entry->multi_recv_ref, 1);
	assert_int_equal(owner_entry->peer_entry.msg_size, 32);

	/* After carve #2 the remaining 32 bytes are NOT strictly less
	 * than min=32, so util_process_multi_recv leaves the owner in
	 * msg_queue. */
	assert_false(slist_empty(&srx_ctx->msg_queue));

	rxe2 = efa_rdm_ep_alloc_rxe(efa_rdm_ep, peer, ofi_op_msg);
	assert_non_null(rxe2);
	rxe2->state = EFA_RDM_RXE_MATCHED;
	rxe2->peer_rxe = peer_rxe2;
	rxe2->cq_entry.op_context = peer_rxe2->context;
	rxe2->cq_entry.flags = FI_RECV | FI_MSG;
	/* A medium-style matched rxe has no outstanding device WR, so the
	 * drain fires immediately. */
	rxe2->efa_outstanding_tx_ops = 0;

	/* 4. Trigger the abort path. mark defers, then the drain helper
	 *    writes a clean FI_ECANCELED / FI_EFA_ERR_PEER_ABORTED error
	 *    CQ entry for message #2 and releases peer_rxe2 through
	 *    util_free_entry so multi_recv_ref decrements correctly.
	 *    Because remaining (32) is NOT < min (32), no FI_MULTI_RECV
	 *    release entry is produced -- the buffer stays alive for a
	 *    potential third message. (Releasing the rxe frees it, so it
	 *    must not be dereferenced afterward.) */
	efa_rdm_rxe_mark_peer_aborted(
		rxe2, EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);
	efa_rdm_rxe_release_peer_abort_if_drained(rxe2);

	/* multi_recv_ref returned to 0; owner remains queued with
	 * 32 bytes of remaining capacity. */
	assert_int_equal(owner_entry->multi_recv_ref, 0);
	assert_int_equal(owner_entry->peer_entry.msg_size, 32);
	assert_false(slist_empty(&srx_ctx->msg_queue));
	assert_ptr_equal(srx_ctx->msg_queue.head, &owner_entry->s_entry);

	ofi_genlock_unlock(srx_ctx->lock);

	/* 5. Spec assertions on the user CQ.
	 *
	 *    Message #1: a success FI_RECV entry. */
	memset(&cq_entry, 0, sizeof(cq_entry));
	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, 1);
	assert_true(cq_entry.flags & FI_RECV);
	assert_false(cq_entry.flags & FI_MULTI_RECV);
	assert_ptr_equal(cq_entry.op_context, (void *) 0x8AB071EDul);

	/*    Message #2: a clean peer-abort error CQ entry. */
	memset(&err_entry, 0, sizeof(err_entry));
	ret = fi_cq_readerr(resource->cq, &err_entry, 0);
	assert_int_equal(ret, 1);
	assert_int_equal(err_entry.err, FI_ECANCELED);
	assert_int_equal(err_entry.prov_errno, FI_EFA_ERR_PEER_ABORTED);
	assert_ptr_equal(err_entry.op_context, (void *) 0x8AB071EDul);

	/*    No FI_MULTI_RECV release entry yet — the buffer is not
	 *    consumed. fi_cq_read returns -FI_EAGAIN. */
	memset(&cq_entry, 0, sizeof(cq_entry));
	ret = fi_cq_read(resource->cq, &cq_entry, 1);
	assert_int_equal(ret, -FI_EAGAIN);

	/* rxe2 was already reaped by the drain helper above (it wrote the
	 * error completion, returned peer_rxe2 to the SRX, and released
	 * the rxe); it must NOT be released again here. */
}

/**
 * @brief Build a TX pkt_entry configured as a receiver-initiated RDMA
 *        READ context (i.e. of type EFA_RDM_RMA_CONTEXT_PKT with
 *        context_type EFA_RDM_RDMA_READ_CONTEXT) bound to an rxe,
 *        then call the TX error dispatcher with the supplied
 *        prov_errno.
 *
 * The helper takes ownership of the pkt_entry: the dispatcher releases
 * it on every code path. On a peer-abort prov_errno the dispatcher
 * marks the rxe and (once drained) reaps it internally; this helper
 * does not release the rxe itself.
 *
 * Pre-bumps rxe->efa_outstanding_tx_ops (and ep/peer counters) to
 * mirror a submitted RDMA READ WR; handle_tx_error's
 * record_tx_op_completed at the top will return the counters to
 * their baseline so the drain-gated rxe release fires correctly.
 */
static void run_longread_read_error(struct efa_resource *resource,
				    struct efa_rdm_ope *rxe, int prov_errno)
{
	struct efa_rdm_ep *ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_rma_context_pkt *ctx_pkt;

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pkt_entry);
	efa_rdm_pke_set_ope(pkt_entry, rxe);
	pkt_entry->peer = rxe->peer;

	/* Forge the receiver-initiated RDMA READ context layout that
	 * efa_rdm_pke_init_read_context would have produced. */
	ctx_pkt = (struct efa_rdm_rma_context_pkt *) pkt_entry->wiredata;
	ctx_pkt->type = EFA_RDM_RMA_CONTEXT_PKT;
	ctx_pkt->version = EFA_RDM_PROTOCOL_VERSION;
	ctx_pkt->flags = 0;
	ctx_pkt->context_type = EFA_RDM_RDMA_READ_CONTEXT;
	ctx_pkt->seg_size = 0;

	/* Simulate a submitted RDMA READ WR on this rxe so the
	 * record_tx_op_completed decrement at the top of handle_tx_error
	 * leaves the per-rxe counter at its accurate post-drain value
	 * (otherwise it would underflow and defeat the drain-gated
	 * release). */
	ep->efa_outstanding_tx_ops++;
	if (rxe->peer)
		rxe->peer->efa_outstanding_tx_ops++;
	rxe->efa_outstanding_tx_ops++;

	efa_rdm_pke_handle_tx_error(pkt_entry, prov_errno);
}

/**
 * @brief Assert the user CQ holds exactly one clean peer-abort error
 *        completion (FI_ECANCELED / FI_EFA_ERR_PEER_ABORTED).
 */
static void assert_cq_peer_aborted_error(struct efa_resource *resource)
{
	struct fi_cq_err_entry err_entry;
	int ret;

	memset(&err_entry, 0, sizeof(err_entry));
	ret = fi_cq_readerr(resource->cq, &err_entry, 0);
	assert_int_equal(ret, 1);
	assert_int_equal(err_entry.err, FI_ECANCELED);
	assert_int_equal(err_entry.prov_errno, FI_EFA_ERR_PEER_ABORTED);
}

/**
 * @brief Helper: build a matched-from-SRX user-posted recv rxe and
 *        the underlying SRX peer_rxe. The caller must hold no SRX
 *        lock; this function takes/releases it internally.
 *
 * Returns the rxe (owning a peer_rxe). On out-of-band cleanup the
 * caller is responsible for releasing the rxe (if the handler did not
 * already reap it) and for freeing the peer_rxe via the SRX owner_ops
 * if needed.
 */
static struct efa_rdm_ope *build_matched_rxe(struct efa_resource *resource)
{
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct iovec *iov;
	void *desc = NULL;
	int ret;

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	peer_srx = util_get_peer_srx(ep->peer_srx_ep);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	iov = calloc(1, sizeof(*iov));
	assert_non_null(iov);
	iov->iov_base = calloc(1, 16);
	iov->iov_len = 16;

	ret = util_srx_generic_recv(ep->peer_srx_ep, iov, &desc, 1,
				    FI_ADDR_UNSPEC, /*context=*/(void *) 0xa1, 0);
	assert_int_equal(ret, FI_SUCCESS);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);
	assert_non_null(peer_rxe);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->peer_rxe = peer_rxe;
	rxe->cq_entry.flags = FI_RECV | FI_MSG;
	rxe->cq_entry.op_context = (void *) 0xa1;
	ofi_genlock_unlock(srx_ctx->lock);

	free(iov->iov_base);
	free(iov);
	return rxe;
}

/**
 * @brief LONGREAD READ failure with REMOTE_ERROR_BAD_ADDRESS (7) →
 *        mark + drain writes a clean RX error completion, reaps rxe.
 */
void test_efa_rdm_pke_handle_tx_error_longread_bad_address_peer_aborts(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct efa_rdm_ope *rxe;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);

	rxe = build_matched_rxe(resource);

	ofi_genlock_lock(srx_ctx->lock);
	run_longread_read_error(resource, rxe,
				EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);
	/* The failing READ was the last WR: the drain helper wrote the
	 * RX error completion, returned the peer_rxe to the SRX (freed),
	 * and reaped the rxe. */
	assert_true(slist_empty(&srx_ctx->msg_queue));
	ofi_genlock_unlock(srx_ctx->lock);

	/* User sees a clean, dedicated peer-abort error completion. */
	assert_cq_peer_aborted_error(resource);
}

/**
 * @brief LONGREAD READ failure with REMOTE_ERROR_ABORT (8) →
 *        mark + drain writes a clean RX error completion, reaps rxe.
 */
void test_efa_rdm_pke_handle_tx_error_longread_abort_peer_aborts(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct efa_rdm_ope *rxe;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);

	rxe = build_matched_rxe(resource);

	ofi_genlock_lock(srx_ctx->lock);
	run_longread_read_error(resource, rxe,
				EFA_IO_COMP_STATUS_REMOTE_ERROR_ABORT);
	assert_true(slist_empty(&srx_ctx->msg_queue));
	ofi_genlock_unlock(srx_ctx->lock);

	assert_cq_peer_aborted_error(resource);
}

/**
 * @brief LONGREAD READ failure with REMOTE_ERROR_BAD_LENGTH (11) →
 *        existing behavior preserved (CQ error written, NOT re-queued).
 *
 * BAD_LENGTH is a real protocol violation, not a peer-clean abort,
 * and the user must continue to see it.
 */
void test_efa_rdm_pke_handle_tx_error_longread_bad_length_writes_cq_err(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct efa_rdm_ope *rxe;
	struct fi_peer_rx_entry *peer_rxe_held;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);

	rxe = build_matched_rxe(resource);
	peer_rxe_held = rxe->peer_rxe;

	ofi_genlock_lock(srx_ctx->lock);
	run_longread_read_error(resource, rxe,
				EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_LENGTH);

	/* The existing path runs efa_rdm_rxe_handle_error → state ERR. */
	assert_int_equal(rxe->state, EFA_RDM_OPE_ERR);
	/* peer_rxe was NOT returned to the SRX. */
	assert_true(slist_empty(&srx_ctx->msg_queue));

	/* Cleanup. */
	efa_rdm_ep_get_peer_srx(ep)->owner_ops->free_entry(peer_rxe_held);
	rxe->peer_rxe = NULL;
	ofi_genlock_unlock(srx_ctx->lock);
	efa_rdm_rxe_release(rxe);
}


/**
 * @brief LONGREAD tagged-recv READ failure with peer-abort prov_errno
 *        is routed through the new handler: the matched recv completes
 *        with a clean RX error (FI_ECANCELED / FI_EFA_ERR_PEER_ABORTED)
 *        and the rxe is reaped at WR drain.
 *
 * This mirrors test_efa_rdm_pke_handle_tx_error_longread_bad_address_peer_aborts
 * but exercises the tagged path (ofi_op_tagged + tag_queue), confirming
 * the dispatcher and abort logic do not have a hidden msg-vs-tag
 * dependency.
 */
void test_efa_rdm_pke_handle_tx_error_longread_tagged_peer_aborts(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	const uint64_t tag_value = 0x42;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	peer_srx = util_get_peer_srx(ep->peer_srx_ep);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	/* Post a tagged recv. */
	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_trecv(ep->peer_srx_ep, &iov, &desc, 1,
				     FI_ADDR_UNSPEC, /*context=*/(void *) 0xa1,
				     tag_value, /*ignore=*/0, 0);
	assert_int_equal(ret, FI_SUCCESS);

	/* Match by tag. */
	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = tag_value;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_tag(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	/* Build a tagged rxe owning the matched peer_rxe. */
	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_tagged);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->peer_rxe = peer_rxe;
	rxe->cq_entry.flags = FI_RECV | FI_TAGGED;
	rxe->cq_entry.op_context = (void *) 0xa1;
	rxe->tag = tag_value;

	run_longread_read_error(resource, rxe,
				EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);

	/* Drain wrote the RX error completion and returned the peer_rxe
	 * to the SRX (freed): neither queue holds it. */
	assert_true(slist_empty(&srx_ctx->tag_queue));
	assert_true(slist_empty(&srx_ctx->msg_queue));
	ofi_genlock_unlock(srx_ctx->lock);

	/* User sees a clean, dedicated peer-abort error completion. */
	assert_cq_peer_aborted_error(resource);
}

/**
 * @brief When the peer advertises EFA_RDM_EXTRA_FEATURE_PEER_ERROR, the
 *        receiver-side abort handler must:
 *          1. Keep the matched peer_rxe owned by the rxe (it is
 *             returned to the SRX only at drain, not re-queued).
 *          2. Set EFA_RDM_OPE_PEER_ABORT_PENDING on the rxe.
 *          3. Stash the prov_errno on the rxe.
 *          4. Post a PEER_ERROR_PKT (queues an outbound TX pkt on the ep).
 *          5. NOT release the rxe yet — release is deferred to the
 *             PEER_ERROR_PKT send completion.
 */
void test_efa_rdm_rxe_emit_peer_error_emits_pkt(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	size_t outstanding_before;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	peer_srx = util_get_peer_srx(ep->peer_srx_ep);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	/* Mark the peer as supporting PEER_ERROR_PKT. */
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	/* Post + match a recv to build a peer_rxe. */
	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_recv(ep->peer_srx_ep, &iov, &desc, 1,
				    FI_ADDR_UNSPEC, /*context=*/(void *) 0xa1, 0);
	assert_int_equal(ret, FI_SUCCESS);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->peer_rxe = peer_rxe;
	rxe->tx_id = 0xdead;  /* sender's send op id, propagated from RTM */

	outstanding_before = ep->efa_outstanding_tx_ops;

	mark_then_emit_peer_error(rxe,
		EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);

	/* The peer_rxe is NOT re-queued: it stays owned by the rxe and
	 * is returned to the SRX only at drain. So msg_queue is empty. */
	assert_true(slist_empty(&srx_ctx->msg_queue));
	assert_non_null(rxe->peer_rxe);

	/* rxe is still alive with the in-flight flag set; no user CQ
	 * entry yet (deferred to drain). */
	assert_true(rxe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_int_equal(rxe->peer_error_prov_errno,
			 EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);

	/* A TX pkt was posted (efa_outstanding_tx_ops increments). */
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before + 1);
	ofi_genlock_unlock(srx_ctx->lock);
	assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);

	/* The real send-completion path is exercised by a separate
	 * test; here just reap the in-flight rxe. */
	peer_abort_test_release_inflight_rxe(ep, srx_ctx, rxe);
}

/**
 * @brief Multi-recv child + peer supports PEER_ERROR_PKT: the
 *        receiver-side abort handler must:
 *          1. Write a user CQ error for the failed slice
 *             (multi-recv children are unsafe to re-queue, per
 *             test_..._multi_recv_writes_err_for_failed_msg).
 *          2. Free the matched peer_rxe via the SRX free_entry op
 *             so multi_recv_ref decrements (fi_msg(3) bookkeeping)
 *             and clear rxe->peer_rxe to avoid a double-free.
 *          3. Stash prov_errno on the rxe and set
 *             EFA_RDM_OPE_PEER_ABORT_PENDING.
 *          4. Post a PEER_ERROR_PKT (efa_outstanding_tx_ops++).
 *          5. NOT release the rxe yet — release is deferred to the
 *             PEER_ERROR_PKT send completion (or to the TX-error
 *             path; covered by separate tests).
 *
 * This test fills the (multi-recv child) x (peer supports) cell of
 * the receiver-side abort handler matrix. The other three cells are
 * covered by:
 *   - (single-buffer) x (peer supports):
 *     test_efa_rdm_rxe_emit_peer_error_emits_pkt
 *   - (single-buffer) x (no peer support):
 *     test_efa_rdm_rxe_peer_abort_writes_error_completion_at_drain
 *   - (multi-recv) x (no peer support):
 *     test_efa_rdm_rxe_mark_peer_aborted_multi_recv_writes_err
 */
void test_efa_rdm_rxe_emit_peer_error_multi_recv_emits_pkt(
	void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct util_rx_entry *owner_entry;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct fi_cq_err_entry err_entry;
	fi_addr_t peer_addr = 0;
	struct iovec iov;
	char buf[96];
	void *desc = NULL;
	size_t outstanding_before;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	peer_srx = util_get_peer_srx(ep->peer_srx_ep);

	/* Same buffer arithmetic as
	 * test_..._multi_recv_writes_err_for_failed_msg: 96 bytes,
	 * three 32-byte messages, min=32 keeps the owner queued
	 * after a single carve. */
	srx_ctx->min_multi_recv_size = 32;

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len),
			 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr,
				      0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	/* Mark peer as supporting PEER_ERROR_PKT. */
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	/* Post a multi-recv. */
	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_recv(ep->peer_srx_ep, &iov, &desc, 1,
				    FI_ADDR_UNSPEC,
				    /*context=*/(void *) 0x8AB071EDul,
				    FI_MULTI_RECV);
	assert_int_equal(ret, FI_SUCCESS);
	assert_false(slist_empty(&srx_ctx->msg_queue));
	owner_entry = container_of(srx_ctx->msg_queue.head,
				   struct util_rx_entry, s_entry);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0;
	match_attr.msg_size = 32;

	ofi_genlock_lock(srx_ctx->lock);

	/* Carve message #1 — the one that fails. */
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);
	assert_non_null(peer_rxe);
	assert_ptr_equal(peer_rxe->owner_context, owner_entry);
	assert_int_equal(owner_entry->multi_recv_ref, 1);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->peer_rxe = peer_rxe;
	rxe->cq_entry.op_context = peer_rxe->context;
	rxe->cq_entry.flags = FI_RECV | FI_MSG;
	rxe->tx_id = 0xdead;  /* sender's send op id, propagated from RTM */

	outstanding_before = ep->efa_outstanding_tx_ops;

	/* Trigger the abort path. mark + emit defer all user-visible
	 * work: the matched peer_rxe stays owned by the rxe, and because
	 * the peer supports the feature a PEER_ERROR_PKT is posted. */
	mark_then_emit_peer_error(rxe,
		EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);

	/* Deferred: no state change, peer_rxe still attached,
	 * multi_recv_ref unchanged, no CQ entry yet. */
	assert_non_null(rxe->peer_rxe);
	assert_int_equal(owner_entry->multi_recv_ref, 1);

	/* Emit-PEER_ERROR_PKT: in-flight flag set, prov_errno stashed,
	 * and a TX op posted. */
	assert_true(rxe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_int_equal(rxe->peer_error_prov_errno,
			 EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before + 1);

	ofi_genlock_unlock(srx_ctx->lock);
	assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);

	/* Drain (PEER_ERROR_PKT send completion) writes the deferred
	 * error completion, returns peer_rxe to the SRX (multi_recv_ref
	 * -> 0), and reaps the rxe. */
	peer_abort_test_release_inflight_rxe(ep, srx_ctx, rxe);

	/* User CQ now contains exactly the message #1 peer-abort error
	 * entry. No FI_MULTI_RECV release entry yet — the buffer still
	 * has 64 bytes left (NOT < min=32). */
	memset(&err_entry, 0, sizeof(err_entry));
	ret = fi_cq_readerr(resource->cq, &err_entry, 0);
	assert_int_equal(ret, 1);
	assert_int_equal(err_entry.err, FI_ECANCELED);
	assert_int_equal(err_entry.prov_errno, FI_EFA_ERR_PEER_ABORTED);
	assert_ptr_equal(err_entry.op_context, (void *) 0x8AB071EDul);
	assert_int_equal(owner_entry->multi_recv_ref, 0);
	assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);
}

/**
 * @brief When the peer-abort prov_errno is REMOTE_ERROR_ABORT (peer
 *        endpoint torn down), the receiver-side abort handler must
 *        skip PEER_ERROR_PKT emission entirely, even if the peer had
 *        previously advertised the feature.
 *
 * Rationale: posting to a torn-down peer endpoint generates a TX
 * failure (typically LOCAL_ERROR_UNRESP_REMOTE) and the peer is
 * gone with its txe anyway, so there is nothing on the remote side
 * to reap. The local user-visible remedy (a clean FI_ECANCELED /
 * FI_EFA_ERR_PEER_ABORTED error completion) still runs at drain; with
 * no PEER_ERROR_PKT in flight the rxe drains and is reaped immediately.
 *
 * Steps:
 *   1. Mark peer as supporting PEER_ERROR_PKT.
 *   2. Build a matched rxe.
 *   3. Trigger abort with prov_errno=REMOTE_ERROR_ABORT.
 *   4. Assert no TX pkt posted, the rxe was reaped, and the user got
 *      the peer-abort error completion (no PEER_ERROR_PKT emitted).
 */
void test_efa_rdm_rxe_emit_peer_error_skips_on_peer_ep_closed(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	size_t outstanding_before;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	peer_srx = util_get_peer_srx(ep->peer_srx_ep);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	/* Even though the peer advertises support, an ABORT prov_errno
	 * means the peer EP is gone — emission must still be skipped. */
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_recv(ep->peer_srx_ep, &iov, &desc, 1,
				    FI_ADDR_UNSPEC, /*context=*/(void *) 0xa1, 0);
	assert_int_equal(ret, FI_SUCCESS);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->peer_rxe = peer_rxe;
	rxe->tx_id = 0xdead;

	outstanding_before = ep->efa_outstanding_tx_ops;

	mark_then_emit_peer_error(rxe,
		EFA_IO_COMP_STATUS_REMOTE_ERROR_ABORT);

	/* No PEER_ERROR_PKT posted (peer EP gone): efa_outstanding_tx_ops
	 * unchanged, so the drain fired immediately. */
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before);
	/* The rxe was reaped at drain: peer_rxe returned to the SRX
	 * (freed, not re-queued), so msg_queue is empty. */
	assert_true(slist_empty(&srx_ctx->msg_queue));
	ofi_genlock_unlock(srx_ctx->lock);

	/* The local user still gets a clean peer-abort error completion
	 * (only the cross-peer notification was skipped). */
	assert_cq_peer_aborted_error(resource);
}

/**
 * @brief When ep->homogeneous_peers is set, the receiver-side abort
 *        handler must emit a PEER_ERROR_PKT for a BAD_ADDRESS abort
 *        even if the peer's handshake has not been received and the
 *        feature bit therefore appears unset.
 *
 * Rationale: FI_OPT_EFA_HOMOGENEOUS_PEERS is the user's contract that
 * all peers run the same software with identical capabilities, so
 * the handshake check is implicitly satisfied. This mirrors the
 * efa_rdm_interop_rdma_read pattern used elsewhere in the codebase.
 */
void test_efa_rdm_rxe_emit_peer_error_with_homogeneous_peers(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	size_t outstanding_before;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	peer_srx = util_get_peer_srx(ep->peer_srx_ep);

	/* User asserts homogeneous peers; deliberately do NOT mark the
	 * peer's handshake as received, to confirm the override
	 * bypasses the handshake check. */
	ep->homogeneous_peers = true;

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);
	/* No HANDSHAKE_RECEIVED, no extra_info bit set. */
	assert_false(peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED);
	assert_false(efa_rdm_peer_support_peer_error(peer));

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_recv(ep->peer_srx_ep, &iov, &desc, 1,
				    FI_ADDR_UNSPEC, /*context=*/(void *) 0xa1, 0);
	assert_int_equal(ret, FI_SUCCESS);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->peer_rxe = peer_rxe;
	rxe->tx_id = 0xdead;

	outstanding_before = ep->efa_outstanding_tx_ops;

	mark_then_emit_peer_error(rxe,
		EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);

	/* Despite no handshake, a PEER_ERROR_PKT was posted. */
	assert_true(rxe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before + 1);
	ofi_genlock_unlock(srx_ctx->lock);

	peer_abort_test_release_inflight_rxe(ep, srx_ctx, rxe);
}

/**
 * @brief When the peer's handshake has not been received and
 *        homogeneous_peers is off, the receiver-side abort handler
 *        must skip PEER_ERROR_PKT emission and reap the rxe
 *        cleanly. The local user-visible remedy (a clean
 *        FI_ECANCELED / FI_EFA_ERR_PEER_ABORTED error completion)
 *        still runs at drain.
 *
 * This documents the no-handshake-yet race: PEER_ERROR is an
 * extra-feature gated by the handshake bitmap, and we conservatively
 * skip when the bit is unknown.
 */
void test_efa_rdm_rxe_emit_peer_error_skips_when_no_handshake(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	size_t outstanding_before;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	peer_srx = util_get_peer_srx(ep->peer_srx_ep);

	/* Default: homogeneous_peers is false. No handshake from peer. */
	assert_false(ep->homogeneous_peers);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);
	assert_false(peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED);

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_recv(ep->peer_srx_ep, &iov, &desc, 1,
				    FI_ADDR_UNSPEC, /*context=*/(void *) 0xa1, 0);
	assert_int_equal(ret, FI_SUCCESS);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->peer_rxe = peer_rxe;
	rxe->tx_id = 0xdead;

	outstanding_before = ep->efa_outstanding_tx_ops;

	mark_then_emit_peer_error(rxe,
		EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);

	/* No PEER_ERROR_PKT posted (handshake unknown): no extra TX op,
	 * so the drain fired immediately. */
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before);
	/* The rxe was reaped at drain: peer_rxe returned to the SRX
	 * (freed, not re-queued), so msg_queue is empty. */
	assert_true(slist_empty(&srx_ctx->msg_queue));
	ofi_genlock_unlock(srx_ctx->lock);

	/* The local user still gets a clean peer-abort error completion
	 * (only the cross-peer notification was skipped). */
	assert_cq_peer_aborted_error(resource);
}

/**
 * @brief Verify that the PEER_ERROR_PKT send-completion handler
 *        releases the rxe via the drain helper, writing the deferred
 *        peer-abort error completion.
 *
 * We exercise the handler directly by hand-building a TX pkt_entry
 * that points at an rxe in the post-emit (EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED) state,
 * then calling efa_rdm_pke_handle_send_completion.
 */
void test_efa_rdm_pke_handle_send_completion_peer_error_releases_rxe(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_peer *peer;
	struct efa_rdm_peer_error_hdr *err_hdr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	size_t outstanding_before;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->internal_flags |= EFA_RDM_OPE_PEER_ABORT_PENDING;

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pkt_entry);
	efa_rdm_pke_set_ope(pkt_entry, rxe);
	pkt_entry->peer = peer;

	err_hdr = (struct efa_rdm_peer_error_hdr *) pkt_entry->wiredata;
	err_hdr->type = EFA_RDM_PEER_ERROR_PKT;
	err_hdr->direction = EFA_RDM_PEER_ERROR_RX_TO_TX;
	err_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	err_hdr->flags = EFA_RDM_PKT_CONNID_HDR;
	err_hdr->op_id_valid = 0;
	err_hdr->op_id = 0xdead;
	err_hdr->prov_errno = EFA_IO_COMP_STATUS_REMOTE_ERROR_ABORT;
	err_hdr->connid = 0xc0ffee;
	pkt_entry->pkt_size = sizeof(struct efa_rdm_peer_error_hdr);
	ep->efa_outstanding_tx_ops++;
	rxe->efa_outstanding_tx_ops++;
	outstanding_before = ep->efa_outstanding_tx_ops;

	/* Drive the send-completion path. */
	efa_rdm_pke_handle_send_completion(pkt_entry);

	/* The handler must have decremented outstanding_tx_ops and
	 * released both the pkt_entry and the rxe. The drain wrote the
	 * deferred peer-abort error completion. We cannot dereference
	 * rxe after this. */
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before - 1);
	assert_cq_peer_aborted_error(resource);
}

/**
 * @brief Verify that an asynchronous TX failure of a PEER_ERROR_PKT
 *        is handled by efa_rdm_pke_handle_tx_error: the in-flight
 *        flag is cleared, the rxe is released, the pkt is released,
 *        and no second user-visible CQ error is written.
 *
 * This is the failure-mode counterpart to
 * test_efa_rdm_pke_handle_send_completion_peer_error_releases_rxe.
 * It simulates the case where the PEER_ERROR_PKT was posted but the
 * peer endpoint then went away before the WR was processed by the
 * NIC (LOCAL_ERROR_UNRESP_REMOTE), and confirms the rxe is reaped
 * cleanly rather than being leaked or generating a duplicate CQ
 * error. In the error-completion design the rxe's single terminal
 * completion (FI_ECANCELED / FI_EFA_ERR_PEER_ABORTED) is written when
 * this last WR drains; the PEER_ERROR_PKT's own async failure adds no
 * separate error.
 */
void test_efa_rdm_pke_handle_tx_error_peer_error_pkt_releases_rxe(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	size_t outstanding_before;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	/* Build an rxe in the post-mark, awaiting-PEER_ERROR_PKT-completion
	 * state: matched and EFA_RDM_OPE_PEER_ABORT_PENDING set. The user
	 * error completion is still owed and is written when this last WR
	 * drains. */
	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->cq_entry.flags = FI_RECV | FI_MSG;
	rxe->internal_flags |= EFA_RDM_OPE_PEER_ABORT_PENDING;

	/* Build the TX pkt_entry that owns the rxe. */
	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pkt_entry);
	efa_rdm_pke_set_ope(pkt_entry, rxe);
	pkt_entry->peer = peer;
	pkt_entry->pkt_size = sizeof(struct efa_rdm_peer_error_hdr);

	/* Forge the on-wire PEER_ERROR_PKT header so the dispatcher's
	 * pkt-type discriminator (efa_rdm_pkt_type_of) recognizes this
	 * as the PEER_ERROR_PKT itself failing async (vs a sibling
	 * READ-context WR on the same rxe). */
	{
		struct efa_rdm_peer_error_hdr *err_hdr =
			(struct efa_rdm_peer_error_hdr *) pkt_entry->wiredata;
		err_hdr->type = EFA_RDM_PEER_ERROR_PKT;
		err_hdr->direction = EFA_RDM_PEER_ERROR_RX_TO_TX;
		err_hdr->version = EFA_RDM_PROTOCOL_VERSION;
		err_hdr->flags = EFA_RDM_PKT_CONNID_HDR;
		err_hdr->op_id_valid = 0;
	}

	ep->efa_outstanding_tx_ops++;
	rxe->efa_outstanding_tx_ops++;
	outstanding_before = ep->efa_outstanding_tx_ops;

	/* Simulate an async TX failure (e.g., peer EP went away). */
	efa_rdm_pke_handle_tx_error(pkt_entry,
		EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE);

	/* The handler released the pkt and, as the last WR to drain,
	 * wrote the deferred peer-abort error completion and reaped the
	 * rxe. efa_rdm_pke_release_tx decrements efa_outstanding_tx_ops. */
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before - 1);

	/* The PEER_ERROR_PKT's own async failure does not surface as a
	 * separate recv error: the single terminal completion is the
	 * clean FI_ECANCELED / FI_EFA_ERR_PEER_ABORTED written at drain.
	 * Read that error first (a pending error makes fi_cq_read return
	 * -FI_EAVAIL, not -FI_EAGAIN), then confirm nothing else remains. */
	assert_cq_peer_aborted_error(resource);
	assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);
}

/**
 * @brief Re-entry safety on the receiver-side abort handler.
 *
 * After mark+emit (with peer-support, so the rxe stays alive
 * waiting on the PEER_ERROR_PKT send), a second failure on the same
 * rxe (which happens in practice when a single long-read transfer
 * posts multiple RDMA READ WRs and more than one fails) must not:
 *  - re-run the abort handling,
 *  - post a duplicate PEER_ERROR_PKT,
 *  - or trip an assertion.
 *
 * The first invocation sets EFA_RDM_OPE_PEER_ABORT_PENDING; the
 * second invocation must early-return on that guard.
 */
void test_efa_rdm_rxe_emit_peer_error_reentry_safe(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	size_t outstanding_after_first;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	peer_srx = util_get_peer_srx(ep->peer_srx_ep);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_recv(ep->peer_srx_ep, &iov, &desc, 1,
				    FI_ADDR_UNSPEC, /*context=*/(void *) 0xa1, 0);
	assert_int_equal(ret, FI_SUCCESS);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->peer_rxe = peer_rxe;
	rxe->tx_id = 0xdead;

	/* First failure. */
	mark_then_emit_peer_error(rxe,
		EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);

	assert_true(rxe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	outstanding_after_first = ep->efa_outstanding_tx_ops;

	/* Second failure on the same rxe -- must early-return (EFA_RDM_OPE_PEER_ABORT_PENDING
	 * is the re-entry guard). No new PEER_ERROR_PKT posted, no
	 * double-free. */
	mark_then_emit_peer_error(rxe,
		EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);

	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_after_first);
	assert_true(rxe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	ofi_genlock_unlock(srx_ctx->lock);

	peer_abort_test_release_inflight_rxe(ep, srx_ctx, rxe);
}


/**
 * @brief Regression test for the multi-WR long-read use-after-free.
 *
 * A single LONGREAD/RUNTREAD transfer posts multiple RDMA READ WRs
 * on one rxe (each its own pkt_entry as wr_id, all sharing
 * pkt_entry->ope = rxe and bumping rxe->efa_outstanding_tx_ops).
 * When the sender cancels its source MR, every in-flight READ WR
 * fails with REMOTE_ERROR_BAD_ADDRESS. The CQ poll processes them
 * one at a time through efa_rdm_pke_handle_tx_error.
 *
 * The first failure runs mark+emit and sets
 * EFA_RDM_OPE_PEER_ABORT_PENDING; every subsequent sibling failure is
 * recognized by the dispatcher (the EFA_RDM_OPE_PEER_ABORT_PENDING branch) and treated as a pure
 * drain -- no re-recover, no re-emit, no premature release. The rxe is
 * freed only by efa_rdm_rxe_release_peer_abort_if_drained() once every
 * WR that uses it as wr_id has drained. Releasing on a sibling failure
 * (as a naive implementation might) would be a use-after-free while the
 * PEER_ERROR_PKT and other READ WRs are still in flight.
 */
void test_efa_rdm_pke_handle_tx_error_sibling_read_wr_does_not_release_rxe(
	void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *pkt1, *pkt2;
	struct efa_rdm_rma_context_pkt *ctx;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr,
				    &raw_addr_len),
			 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr,
				      0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->cq_entry.flags = FI_RECV | FI_MSG;
	rxe->internal_flags |= EFA_RDM_OPE_PEER_ABORT_PENDING;

	/* Three outstanding TX ops on this rxe: two sibling RDMA READ
	 * WRs (about to fail) plus one PEER_ERROR_PKT we synthetically
	 * drain at the end. */
	ep->efa_outstanding_tx_ops += 3;
	peer->efa_outstanding_tx_ops += 3;
	rxe->efa_outstanding_tx_ops += 3;

	pkt1 = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				 EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pkt1);
	efa_rdm_pke_set_ope(pkt1, rxe);
	pkt1->peer = peer;
	ctx = (struct efa_rdm_rma_context_pkt *) pkt1->wiredata;
	ctx->type = EFA_RDM_RMA_CONTEXT_PKT;
	ctx->version = EFA_RDM_PROTOCOL_VERSION;
	ctx->flags = 0;
	ctx->context_type = EFA_RDM_RDMA_READ_CONTEXT;

	pkt2 = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				 EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pkt2);
	efa_rdm_pke_set_ope(pkt2, rxe);
	pkt2->peer = peer;
	ctx = (struct efa_rdm_rma_context_pkt *) pkt2->wiredata;
	ctx->type = EFA_RDM_RMA_CONTEXT_PKT;
	ctx->version = EFA_RDM_PROTOCOL_VERSION;
	ctx->flags = 0;
	ctx->context_type = EFA_RDM_RDMA_READ_CONTEXT;

	/* First sibling RDMA READ WR fails. The EFA_RDM_OPE_PEER_ABORT_PENDING branch must
	 * NOT clear IN_FLIGHT (this packet is not the PEER_ERROR_PKT)
	 * and must NOT release the rxe -- the PEER_ERROR_PKT and another
	 * sibling READ WR are still in flight. */
	efa_rdm_pke_handle_tx_error(pkt1,
		EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);

	assert_int_equal(rxe->state, EFA_RDM_RXE_MATCHED);
	assert_true(rxe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_int_equal(rxe->efa_outstanding_tx_ops, 2);

	/* Second sibling RDMA READ WR fails. Same expectations: the rxe
	 * stays alive (this is the regression check; the original code
	 * released it here, leaving the in-flight PEER_ERROR_PKT and the
	 * untouched sibling pointing at freed memory). */
	efa_rdm_pke_handle_tx_error(pkt2,
		EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);

	assert_int_equal(rxe->state, EFA_RDM_RXE_MATCHED);
	assert_true(rxe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_int_equal(rxe->efa_outstanding_tx_ops, 1);

	/* Drain the synthetic PEER_ERROR_PKT. The drain helper now sees
	 * efa_outstanding_tx_ops == 0 and frees the rxe. */
	rxe->efa_outstanding_tx_ops--;
	ep->efa_outstanding_tx_ops--;
	peer->efa_outstanding_tx_ops--;
	efa_rdm_rxe_release_peer_abort_if_drained(rxe);

	/* The final drain wrote the single deferred peer-abort error
	 * completion and reaped the rxe. Read that error first (a pending
	 * error makes fi_cq_read return -FI_EAVAIL, not -FI_EAGAIN), then
	 * confirm there is no spurious success entry. */
	assert_cq_peer_aborted_error(resource);
	assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);
}
/**
 * @brief LONGREAD direction: PEER_ERROR_PKT recv with send_id set ->
 *        the sender's txe is failed and released; num_read_msg_in_flight
 *        is decremented.
 */
void test_efa_rdm_pke_handle_peer_error_recv_longread_fails_txe(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_peer_error_hdr *err_hdr;
	struct efa_rdm_ope *txe;
	struct fi_cq_err_entry err_entry;
	uint64_t in_flight_before;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	/* Build a real txe (uses the common helper which sets up a peer). */
	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_TXE_REQ;
	txe->internal_flags = 0;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->cq_entry.op_context = (void *) 0xa1;

	/* The sender would have incremented num_read_msg_in_flight when
	 * it sent the LONGREAD/RUNTREAD RTM. Simulate that. */
	efa_rdm_ep_rdm_domain(ep)->num_read_msg_in_flight = 1;
	in_flight_before = efa_rdm_ep_rdm_domain(ep)->num_read_msg_in_flight;

	/* Build the inbound PEER_ERROR_PKT pointing at our txe via send_id. */
	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_rx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	ep->efa_rx_pkts_posted = efa_base_ep_get_rx_pool_size(&ep->base_ep);

	err_hdr = (struct efa_rdm_peer_error_hdr *) pkt_entry->wiredata;
	err_hdr->type = EFA_RDM_PEER_ERROR_PKT;
	err_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	err_hdr->flags = EFA_RDM_PKT_CONNID_HDR;
	err_hdr->op_id_valid = 1;
	err_hdr->msg_id = txe->msg_id;
	err_hdr->op_id = txe->tx_id;
	err_hdr->direction = EFA_RDM_PEER_ERROR_RX_TO_TX;
	err_hdr->prov_errno = EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS;
	err_hdr->connid = 0xc0ffee;
	pkt_entry->pkt_size = sizeof(struct efa_rdm_peer_error_hdr);

	efa_rdm_pke_handle_peer_error_recv(pkt_entry);

	/* num_read_msg_in_flight decremented. */
	assert_int_equal(efa_rdm_ep_rdm_domain(ep)->num_read_msg_in_flight,
			 in_flight_before - 1);

	/* User must see a TX CQ error with the clean, dedicated
	 * peer-abort code (not the raw wire prov_errno). */
	memset(&err_entry, 0, sizeof(err_entry));
	ret = fi_cq_readerr(resource->cq, &err_entry, 0);
	assert_int_equal(ret, 1);
	assert_int_equal(err_entry.err, FI_ECANCELED);
	assert_int_equal(err_entry.prov_errno, FI_EFA_ERR_PEER_ABORTED);

	/* The aborted op owes exactly one completion: a second error read
	 * must find the CQ empty. A duplicate terminal completion (e.g.
	 * handle_error writing an error and the drain helper writing another)
	 * surfaces here as a second readerr. Use fi_cq_readerr, not fi_cq_read,
	 * so the check does not drive the progress engine -- whose internal RX
	 * repost would trip the rx-pkt accounting this test hand-sets up. */
	memset(&err_entry, 0, sizeof(err_entry));
	assert_int_equal(fi_cq_readerr(resource->cq, &err_entry, 0), -FI_EAGAIN);
}

/**
 * @brief LONGCTS direction: PEER_ERROR_PKT recv with recv_id set ->
 *        receiver's matched rxe is marked peer-aborted and, once
 *        drained, completes with a clean FI_ECANCELED /
 *        FI_EFA_ERR_PEER_ABORTED error and is reaped.
 */
void test_efa_rdm_pke_handle_peer_error_recv_longcts_reaps_rxe(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_peer_error_hdr *err_hdr;
	struct efa_rdm_ope *rxe;
	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	peer_srx = util_get_peer_srx(ep->peer_srx_ep);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	/* Post + match a recv to build the rxe. */
	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_recv(ep->peer_srx_ep, &iov, &desc, 1,
				    FI_ADDR_UNSPEC, /*context=*/(void *) 0xa1, 0);
	assert_int_equal(ret, FI_SUCCESS);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->peer_rxe = peer_rxe;

	/* Build the inbound PEER_ERROR_PKT pointing at our rxe via recv_id. */
	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_rx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	ep->efa_rx_pkts_posted += 1;

	err_hdr = (struct efa_rdm_peer_error_hdr *) pkt_entry->wiredata;
	err_hdr->type = EFA_RDM_PEER_ERROR_PKT;
	err_hdr->direction = EFA_RDM_PEER_ERROR_TX_TO_RX;
	err_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	err_hdr->flags = EFA_RDM_PKT_CONNID_HDR;
	err_hdr->op_id_valid = 1;
	err_hdr->msg_id = rxe->msg_id;
	err_hdr->op_id = rxe->rx_id;
	err_hdr->prov_errno = EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY;
	err_hdr->connid = 0xbeef;
	pkt_entry->pkt_size = sizeof(struct efa_rdm_peer_error_hdr);

	efa_rdm_pke_handle_peer_error_recv(pkt_entry);

	/* The matched rxe has no outstanding WR, so the drain fires
	 * immediately: peer_rxe returned to the SRX (freed, not
	 * re-queued), so the queue is empty, and the rxe is reaped. */
	assert_true(slist_empty(&srx_ctx->msg_queue));
	ofi_genlock_unlock(srx_ctx->lock);

	/* The receiver's matched recv gets a clean peer-abort error. */
	assert_cq_peer_aborted_error(resource);
}



/**
 * @brief LONGCTS direction with a tagged rxe: PEER_ERROR_PKT recv
 *        with recv_id set must mark the matched tagged rxe
 *        peer-aborted and, once drained, complete it with a clean
 *        peer-abort error and reap it (from the tag path, not msg).
 *
 * Sister test to test_efa_rdm_pke_handle_peer_error_recv_longcts_reaps_rxe
 * (which uses ofi_op_msg) — this confirms the dispatcher's LONGCTS
 * branch and the abort handler's tagged path don't have a hidden
 * msg-vs-tag dependency.
 */
void test_efa_rdm_pke_handle_peer_error_recv_longcts_tagged(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_peer_error_hdr *err_hdr;
	struct efa_rdm_ope *rxe;
	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	const uint64_t tag_value = 0x1234;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	peer_srx = util_get_peer_srx(ep->peer_srx_ep);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	/* Post + match a tagged recv. */
	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_trecv(ep->peer_srx_ep, &iov, &desc, 1,
				     FI_ADDR_UNSPEC, /*context=*/(void *) 0xa1,
				     tag_value, /*ignore=*/0, 0);
	assert_int_equal(ret, FI_SUCCESS);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = tag_value;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_tag(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_tagged);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->peer_rxe = peer_rxe;
	rxe->cq_entry.flags = FI_RECV | FI_TAGGED;
	rxe->tag = tag_value;

	/* Build the inbound PEER_ERROR_PKT pointing at our tagged rxe via recv_id. */
	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_rx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	ep->efa_rx_pkts_posted += 1;

	err_hdr = (struct efa_rdm_peer_error_hdr *) pkt_entry->wiredata;
	err_hdr->type = EFA_RDM_PEER_ERROR_PKT;
	err_hdr->direction = EFA_RDM_PEER_ERROR_TX_TO_RX;
	err_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	err_hdr->flags = EFA_RDM_PKT_CONNID_HDR;
	err_hdr->op_id_valid = 1;
	err_hdr->msg_id = rxe->msg_id;
	err_hdr->op_id = rxe->rx_id;
	err_hdr->prov_errno = EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY;
	err_hdr->connid = 0xbeef;
	pkt_entry->pkt_size = sizeof(struct efa_rdm_peer_error_hdr);

	efa_rdm_pke_handle_peer_error_recv(pkt_entry);

	/* The matched tagged rxe has no outstanding WR, so the drain
	 * fires immediately: peer_rxe returned to the SRX (freed), so
	 * neither queue holds it, and the rxe is reaped. */
	assert_true(slist_empty(&srx_ctx->tag_queue));
	assert_true(slist_empty(&srx_ctx->msg_queue));
	ofi_genlock_unlock(srx_ctx->lock);

	/* The receiver's matched recv gets a clean peer-abort error. */
	assert_cq_peer_aborted_error(resource);
}

/**
 * @brief LONGCTS pre-CTS: a msg_id-only PEER_ERROR resolves
 *        the matched rxe via the peer's rxe_list fallback scan (LONGCTS
 *        rxes are not in rxe_map), marks it peer-aborted, and completes
 *        with FI_ECANCELED.
 */
void test_efa_rdm_pke_handle_peer_error_recv_longcts_msg_id_only(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_peer_error_hdr *err_hdr;
	struct efa_rdm_ope *rxe;
	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	uint32_t msg_id = 42;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	peer_srx = util_get_peer_srx(ep->peer_srx_ep);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_recv(ep->peer_srx_ep, &iov, &desc, 1,
				    FI_ADDR_UNSPEC, /*context=*/(void *) 0xa1, 0);
	assert_int_equal(ret, FI_SUCCESS);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->peer_rxe = peer_rxe;
	rxe->msg_id = msg_id;

	/*
	 * LONGCTS rxes are NOT in rxe_map — they are only on
	 * peer->rxe_list (via peer_entry, set by efa_rdm_ep_alloc_rxe).
	 * The fallback scan in efa_rdm_pke_handle_peer_error_recv must
	 * find this rxe by msg_id.
	 */
	assert_null(rxe->rxe_map);

	/*
	 * Build a msg_id-only PEER_ERROR (op_id_valid=0). This is what a
	 * sender emits for a pre-CTS abort: the CTS was never processed
	 * (or was dropped by the PEER_ABORT_PENDING check), so rx_id is
	 * unknown and op_id cannot be supplied.
	 */
	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_rx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	ep->efa_rx_pkts_posted += 1;
	pkt_entry->peer = peer;

	err_hdr = (struct efa_rdm_peer_error_hdr *) pkt_entry->wiredata;
	err_hdr->type = EFA_RDM_PEER_ERROR_PKT;
	err_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	err_hdr->flags = EFA_RDM_PKT_CONNID_HDR;
	err_hdr->op_id_valid = 0;
	err_hdr->msg_id = msg_id;
	err_hdr->op_id = 0;
	err_hdr->direction = EFA_RDM_PEER_ERROR_TX_TO_RX;
	err_hdr->prov_errno = EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY;
	err_hdr->connid = 0xbeef;
	pkt_entry->pkt_size = sizeof(struct efa_rdm_peer_error_hdr);

	efa_rdm_pke_handle_peer_error_recv(pkt_entry);

	/*
	 * The msg_id resolves via peer->rxe_list scan → handle_matched_rxe.
	 * No outstanding WR so drain fires immediately: peer_rxe freed, rxe
	 * reaped, queue empty.
	 */
	assert_true(slist_empty(&srx_ctx->msg_queue));
	ofi_genlock_unlock(srx_ctx->lock);

	assert_cq_peer_aborted_error(resource);
}

/**
 * @brief Inbound dispatcher drops a PEER_ERROR_PKT whose op_id is
 *        out of range, without touching domain state.
 *
 * op_id is wire-supplied. ofi_bufpool_get_ibuf() does not
 * bounds-check, so an out-of-range id would otherwise resolve to an
 * arbitrary slot. The dispatcher validates the index with
 * ofi_bufpool_ibuf_is_valid() first and drops the packet. Verify a
 * huge op_id neither underflows num_read_msg_in_flight nor writes a
 * user CQ error.
 */
void test_efa_rdm_pke_handle_peer_error_recv_invalid_op_id_dropped(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_peer_error_hdr *err_hdr;
	struct fi_cq_err_entry err_entry;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	/* No txe/rxe allocated; num_read_msg_in_flight stays 0 so an
	 * unguarded decrement would wrap it. */
	assert_int_equal(efa_rdm_ep_rdm_domain(ep)->num_read_msg_in_flight, 0);

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_rx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	ep->efa_rx_pkts_posted = efa_base_ep_get_rx_pool_size(&ep->base_ep);

	err_hdr = (struct efa_rdm_peer_error_hdr *) pkt_entry->wiredata;
	err_hdr->type = EFA_RDM_PEER_ERROR_PKT;
	err_hdr->direction = EFA_RDM_PEER_ERROR_RX_TO_TX;
	err_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	err_hdr->flags = EFA_RDM_PKT_CONNID_HDR;
	err_hdr->op_id_valid = 1;
	err_hdr->msg_id = 0xffffffff;
	err_hdr->op_id = 0xffffffff;	/* far out of range */
	err_hdr->prov_errno = EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS;
	err_hdr->connid = 0xc0ffee;
	pkt_entry->pkt_size = sizeof(struct efa_rdm_peer_error_hdr);

	efa_rdm_pke_handle_peer_error_recv(pkt_entry);

	/* Counter not touched (no underflow). */
	assert_int_equal(efa_rdm_ep_rdm_domain(ep)->num_read_msg_in_flight, 0);

	/* No user CQ error written. */
	ret = fi_cq_readerr(resource->cq, &err_entry, 0);
	assert_int_equal(ret, -FI_EAGAIN);
}
/*
 * Simulate the source MR being closed mid-transfer (its generation bumped)
 * so efa_rdm_mr_gen_check_ope() reports the MR was canceled -- the
 * precondition the sender peer-abort path (efa_rdm_txe_mark_peer_abort_if_needed)
 * requires before it will mark the txe and drive the PEER_ERROR emit/drain.
 * The caller must also set txe->protocol to the aborting RTM type.
 */
static void efa_unit_test_txe_simulate_source_mr_canceled(struct efa_rdm_ope *txe)
{
	static struct efa_rdm_mr stale_source_mr;

	stale_source_mr.gen = 1;	/* current MR generation (valid sentinel-free) */
	txe->iov_count = 1;
	txe->desc[0] = &stale_source_mr;
	txe->desc_gen[0] = 2;		/* dispatch-time snapshot, now stale (!= mr gen) */
}

/**
 * @brief LONGCTS sender-side: txe in mid-CTSDATA gets
 *        LOCAL_ERROR_INVALID_LKEY (post-WR-submit race) →
 *        TX CQ error written AND a PEER_ERROR_PKT posted.
 */
void test_efa_rdm_txe_handle_error_emits_peer_error_on_invalid_lkey(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	struct fi_cq_err_entry err_entry;
	size_t outstanding_before;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_OPE_SEND;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->cq_entry.op_context = (void *) 0xa1;
	txe->total_len = 1024;
	txe->bytes_sent = 256;	/* mid-CTSDATA */

	/* Mark the peer as supporting PEER_ERROR_PKT. */
	peer = txe->peer;
	assert_non_null(peer);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	/* Simulate the LONGCTS_*RTM has already been linked into
	 * the longcts send list. The error handler removes it. */
	dlist_insert_tail(&txe->entry,
			  &efa_rdm_ep_rdm_domain(ep)->ope_longcts_send_list);

	txe->protocol = EFA_RDM_LONGCTS_MSGRTM_PKT;
	efa_unit_test_txe_simulate_source_mr_canceled(txe);

	/* Simulate a CTSDATA WR still in flight so handle_error's drain is a
	 * no-op until that last WR drains. */
	txe->efa_outstanding_tx_ops = 1;

	outstanding_before = ep->efa_outstanding_tx_ops;

	efa_rdm_txe_handle_error(txe, FI_EINVAL,
		EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* handle_error marks the txe and drives the drain itself, but with a
	 * WR still in flight nothing is emitted yet and the user completion
	 * is withheld. */
	assert_true(txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	memset(&err_entry, 0, sizeof(err_entry));
	ret = fi_cq_readerr(resource->cq, &err_entry, 0);
	assert_int_equal(ret, -FI_EAGAIN);
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before);

	/* The last WR drains; the drain helper now emits the PEER_ERROR_PKT. */
	txe->efa_outstanding_tx_ops--;
	efa_rdm_txe_progress_peer_abort_if_drained(txe);

	/* PEER_ERROR_PKT was posted (efa_outstanding_tx_ops bumped). */
	assert_int_equal(ep->efa_outstanding_tx_ops,
			 outstanding_before + 1);

	/* The txe's peer_error_prov_errno was set so the wire packet
	 * carries the right cause. */
	assert_int_equal(txe->peer_error_prov_errno,
			 EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);
}

/**
 * @brief LONGCTS sender-side: txe in mid-CTSDATA gets
 *        FI_ECANCELED (gen check before-post detection) →
 *        TX CQ error written AND a PEER_ERROR_PKT posted.
 */
void test_efa_rdm_txe_handle_error_emits_peer_error_on_canceled(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	size_t outstanding_before;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_tagged);
	assert_non_null(txe);
	txe->state = EFA_RDM_OPE_SEND;
	txe->cq_entry.flags = FI_SEND | FI_TAGGED;
	txe->cq_entry.op_context = (void *) 0xb2;
	txe->total_len = 1024;
	txe->bytes_sent = 512;

	peer = txe->peer;
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	dlist_insert_tail(&txe->entry,
			  &efa_rdm_ep_rdm_domain(ep)->ope_longcts_send_list);

	outstanding_before = ep->efa_outstanding_tx_ops;

	txe->protocol = EFA_RDM_LONGCTS_MSGRTM_PKT;
	efa_unit_test_txe_simulate_source_mr_canceled(txe);

	/* Simulate the call site in efa_domain.c: after the gen check
	 * returns -FI_ECANCELED, the caller passes err = FI_ECANCELED
	 * and prov_errno = FI_EFA_ERR_PKT_POST. Our hook examines the
	 * err field. */
	efa_rdm_txe_handle_error(txe, FI_ECANCELED, FI_EFA_ERR_PKT_POST);
	/* handle_error marks the txe and, with no WR in flight, drives the
	 * single emit itself. */

	assert_int_equal(ep->efa_outstanding_tx_ops,
			 outstanding_before + 1);
	assert_int_equal(txe->peer_error_prov_errno, FI_EFA_ERR_PKT_POST);
}

/**
 * @brief Before-post (gen-check) cancellation of a queued EAGER two-sided
 *        RTM emits a msg_id-only PEER_ERROR_PKT.
 *
 * Regression test for the reorder-window-stranding gap: an EAGER /
 * medium / runt-only RTM canceled BEFORE its WR reached the device is
 * reported through efa_rdm_txe_handle_error() directly (from
 * efa_rdm_ope_process_queued_ope() for a queued before-handshake / RNR /
 * EAGAIN op, or the LONGCTS drip loop), NOT through
 * efa_rdm_pke_handle_tx_error(). The skip-emit decision now lives in
 * efa_rdm_txe_handle_error(), so this path must also emit. Mirrors
 * test_efa_rdm_pke_handle_tx_error_eager_emits_skip (device-WR path) but
 * calls handle_error directly with the FI_ECANCELED / FI_EFA_ERR_PKT_POST
 * the gen-check cancellation site passes.
 */
void test_efa_rdm_txe_handle_error_eager_prepost_cancel_emits_skip(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	size_t outstanding_before;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	/* Queued EAGER two-sided RTM that never reached the device: still
	 * in TXE_REQ (no CTS, no OPE_SEND), protocol records the selected
	 * EAGER type. */
	txe->state = EFA_RDM_TXE_REQ;
	txe->protocol = EFA_RDM_EAGER_MSGRTM_PKT;
	efa_unit_test_txe_simulate_source_mr_canceled(txe);
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->cq_entry.op_context = (void *) 0xa1;
	txe->msg_id = 0x99;

	peer = txe->peer;
	assert_non_null(peer);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	outstanding_before = ep->efa_outstanding_tx_ops;

	/* Before-post gen-check cancellation call site: err = FI_ECANCELED,
	 * prov_errno = FI_EFA_ERR_PKT_POST. */
	efa_rdm_txe_handle_error(txe, FI_ECANCELED, FI_EFA_ERR_PKT_POST);
	/* handle_error marks the txe and, with no WR in flight, drives the
	 * single emit itself. */

	/* The consolidated emit decision marked the txe PENDING and (single
	 * WR, already drained) emitted the PEER_ERROR_PKT, keeping the txe
	 * alive for that packet's completion. The emit carries msg_id only
	 * for this eager txe (see efa_rdm_pke_init_peer_error_for_ope, covered
	 * by test_efa_rdm_pke_init_peer_error_for_ope_eager_skip). */
	assert_true(txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_true(txe->internal_flags & EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before + 1);
}

/**
 * @brief Pre-CTS (TXE_REQ) cancellation of a LONGCTS two-sided RTM emits
 *        a msg_id-only PEER_ERROR_PKT.
 *
 * Regression test for the LONGCTS reorder-window-stall hang: a LONGCTS
 * RTM canceled while still in EFA_RDM_TXE_REQ (its first CTS never
 * processed, so txe->rx_id is unknown) used to be suppressed by the
 * prev_state == OPE_SEND gate, leaving the receiver's reorder window
 * parked forever on the never-delivered msg_id. The fix marks the txe
 * PENDING (rx_id stays the unset sentinel -- no rx_id known pre-CTS) and, with its
 * single WR already drained, emits a msg_id-only PEER_ERROR_PKT; the
 * receiver decides the outcome from its reorder-window state (see
 * test_efa_rdm_pke_init_peer_error_for_ope_longcts_pre_cts_skip).
 *
 * Mirrors test_efa_rdm_txe_handle_error_eager_prepost_cancel_emits_skip
 * but for the LONGCTS protocol in TXE_REQ.
 */
void test_efa_rdm_txe_handle_error_longcts_prepost_cancel_emits_skip(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	size_t outstanding_before;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	/* LONGCTS RTM aborted before its first CTS: still in TXE_REQ (no
	 * OPE_SEND), no CTSDATA acked, protocol records the LONGCTS type. */
	txe->state = EFA_RDM_TXE_REQ;
	txe->protocol = EFA_RDM_LONGCTS_MSGRTM_PKT;
	efa_unit_test_txe_simulate_source_mr_canceled(txe);
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->cq_entry.op_context = (void *) 0xc3;
	txe->msg_id = 0x77;
	txe->bytes_acked = 0;

	peer = txe->peer;
	assert_non_null(peer);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	outstanding_before = ep->efa_outstanding_tx_ops;

	/* Before-post / pre-CTS cancellation call site (drip loop or
	 * process_queued_ope): err = FI_ECANCELED,
	 * prov_errno = FI_EFA_ERR_PKT_POST. */
	efa_rdm_txe_handle_error(txe, FI_ECANCELED, FI_EFA_ERR_PKT_POST);
	/* handle_error marks the txe and, with no WR in flight, drives the
	 * single emit itself. */

	/* The consolidated emit decision marked the txe PENDING, recorded
	 * that no rx_id is known (pre-CTS), so it is signalled msg_id-only
	 * (rx_id stays the unset sentinel), and emitted the PEER_ERROR_PKT, keeping
	 * the txe alive for that packet's completion. */
	assert_true(txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_int_equal(txe->rx_id, EFA_RDM_OPE_INVALID_ID);
	assert_true(txe->internal_flags & EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before + 1);
}

/**
 * @brief Pre-match (TXE_REQ) cancellation of a RUNTREAD RTM that has a
 *        tail READ remainder emits a msg_id-only PEER_ERROR_PKT.
 *
 * Regression test for the read-base reorder-window-stall hang. A RUNTREAD
 * RTM carries user data (the runt segments) backed by the user MR, so the
 * source-MR close can flush/cancel it before the receiver matches the
 * recv. A runtread WITH a tail READ (total_len > bytes_runt) used to be
 * excluded from the msg_id notification path (only runt-only was covered)
 * and is not LONGCTS, so the sender-abort gate did not fire at all,
 * leaving the receiver's reorder window parked forever on the
 * never-delivered msg_id. The fix makes all runtread msg_id-keyed, so
 * efa_rdm_txe_handle_error() marks the txe PENDING and (single WR, already
 * drained) emits the PEER_ERROR_PKT. With no rx_id (runtread exchanges no
 * CTS) the emit is msg_id-only and the receiver decides the outcome (see
 * test_efa_rdm_pke_init_peer_error_for_ope_runtread).
 *
 * Mirrors test_efa_rdm_txe_handle_error_longcts_prepost_cancel_emits_skip
 * but for a RUNTREAD RTM with a READ remainder. Like every runtread abort
 * it is keyed by msg_id (it has no rx_id), so rx_id stays the unset sentinel
 * and no op_id hint is emitted.
 */
void test_efa_rdm_txe_handle_error_runtread_prepost_cancel_emits_skip(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	size_t outstanding_before;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	/* Runt-only RUNTREAD RTM (total_len == bytes_runt, no tail READ)
	 * aborted before any runt segment was acked: still in TXE_REQ, no
	 * rx_id (runtread exchanges no CTS), protocol records the RUNTREAD
	 * type. Only runt-only runtread is sender-signalled; runtread WITH a
	 * tail READ is detected receiver-side and is not emitted here. */
	txe->state = EFA_RDM_TXE_REQ;
	txe->protocol = EFA_RDM_RUNTREAD_MSGRTM_PKT;
	efa_unit_test_txe_simulate_source_mr_canceled(txe);
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->cq_entry.op_context = (void *) 0xb4;
	txe->msg_id = 0x55;
	txe->total_len = 4096;
	txe->bytes_runt = 4096;
	txe->bytes_acked = 0;

	peer = txe->peer;
	assert_non_null(peer);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	outstanding_before = ep->efa_outstanding_tx_ops;

	/* Before-post / pre-match cancellation call site: err = FI_ECANCELED,
	 * prov_errno = FI_EFA_ERR_PKT_POST. */
	efa_rdm_txe_handle_error(txe, FI_ECANCELED, FI_EFA_ERR_PKT_POST);
	/* handle_error marks the txe and, with no WR in flight, drives the
	 * single emit itself. */

	/* PENDING + emitted, but rx_id unset: runtread exchanges no CTS,
	 * so no rx_id is ever known and the emit is msg_id-only. */
	assert_true(txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_int_equal(txe->rx_id, EFA_RDM_OPE_INVALID_ID);
	assert_true(txe->internal_flags & EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before + 1);
}

/**
 * @brief LONGCTS sender-side fallback: peer doesn't advertise
 *        EFA_RDM_EXTRA_FEATURE_PEER_ERROR → no PEER_ERROR_PKT.
 *        Sender still sees TX CQ error.
 */
void test_efa_rdm_txe_handle_error_no_emit_when_peer_unsupported(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	struct fi_cq_err_entry err_entry;
	size_t outstanding_before;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_OPE_SEND;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->total_len = 1024;
	txe->bytes_sent = 256;

	peer = txe->peer;
	/* Handshake received but feature bit NOT advertised — old peer. */
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] = 0;

	dlist_insert_tail(&txe->entry,
			  &efa_rdm_ep_rdm_domain(ep)->ope_longcts_send_list);

	outstanding_before = ep->efa_outstanding_tx_ops;

	efa_rdm_txe_handle_error(txe, FI_EINVAL,
		EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* User still sees a TX CQ error. */
	memset(&err_entry, 0, sizeof(err_entry));
	ret = fi_cq_readerr(resource->cq, &err_entry, 0);
	assert_int_equal(ret, 1);
	assert_int_equal(err_entry.prov_errno,
			 EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* No PEER_ERROR_PKT was posted. */
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before);
}

/**
 * @brief LONGCTS sender-side: when ep->homogeneous_peers is set,
 *        emit PEER_ERROR_PKT for an MR-cancel abort even if the
 *        receiver's handshake has not been received and the feature
 *        bit therefore appears unset.
 *
 * Mirrors test_efa_rdm_rxe_emit_peer_error_with_homogeneous_peers
 * in the rxe direction.
 */
void test_efa_rdm_txe_handle_error_emits_peer_error_with_homogeneous_peers(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	size_t outstanding_before;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	ep->homogeneous_peers = true;

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_OPE_SEND;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->total_len = 1024;
	txe->bytes_sent = 256;

	peer = txe->peer;
	/* Deliberately leave handshake-received unset and feature bit
	 * cleared, to confirm homogeneous_peers bypasses the gate. */
	assert_false(peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED);
	assert_false(efa_rdm_peer_support_peer_error(peer));

	dlist_insert_tail(&txe->entry,
			  &efa_rdm_ep_rdm_domain(ep)->ope_longcts_send_list);

	txe->protocol = EFA_RDM_LONGCTS_MSGRTM_PKT;
	efa_unit_test_txe_simulate_source_mr_canceled(txe);

	outstanding_before = ep->efa_outstanding_tx_ops;

	efa_rdm_txe_handle_error(txe, FI_EINVAL,
		EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);
	/* handle_error marks the txe and, with no WR in flight, drives the
	 * single emit itself. */

	/* PEER_ERROR_PKT was posted despite no handshake. */
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before + 1);
	assert_int_equal(txe->peer_error_prov_errno,
			 EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);
}

/**
 * @brief LONGCTS sender-side: when neither homogeneous_peers nor
 *        is_self applies and the receiver's handshake has not yet
 *        been received, skip PEER_ERROR_PKT emission. The user
 *        still sees a TX CQ error; the receiver's rxe leaks
 *        (status quo).
 */
void test_efa_rdm_txe_handle_error_skips_peer_error_when_no_handshake(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	struct fi_cq_err_entry err_entry;
	size_t outstanding_before;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	assert_false(ep->homogeneous_peers);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_OPE_SEND;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->total_len = 1024;
	txe->bytes_sent = 256;

	peer = txe->peer;
	/* No handshake received from receiver. */
	assert_false(peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED);

	dlist_insert_tail(&txe->entry,
			  &efa_rdm_ep_rdm_domain(ep)->ope_longcts_send_list);

	outstanding_before = ep->efa_outstanding_tx_ops;

	efa_rdm_txe_handle_error(txe, FI_EINVAL,
		EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* User still sees a TX CQ error. */
	memset(&err_entry, 0, sizeof(err_entry));
	ret = fi_cq_readerr(resource->cq, &err_entry, 0);
	assert_int_equal(ret, 1);
	assert_int_equal(err_entry.prov_errno,
			 EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* No PEER_ERROR_PKT was posted. */
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before);
}

/**
 * @brief LONGCTS emit gate is keyed on state, not bytes_sent.
 *
 * A medium (or runting-read) RTM txe sets bytes_sent but never
 * receives a CTS, so it stays in EFA_RDM_TXE_REQ and its rx_id is
 * never populated. If such a txe hits LOCAL_ERROR_INVALID_LKEY
 * (e.g. user closed the source MR), the sender must NOT emit a
 * PEER_ERROR_PKT: doing so would put a stale/garbage rx_id on the
 * wire (the op_id field is txe->rx_id). The user still gets a TX CQ
 * error. This pins the state-based gate that replaced the looser
 * bytes_sent > 0 condition.
 */
void test_efa_rdm_txe_handle_error_no_emit_when_not_longcts(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	struct fi_cq_err_entry err_entry;
	size_t outstanding_before;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	/* Medium-RTM-like txe: data sent, but no CTS received, so it is
	 * still in TXE_REQ and rx_id is unset. */
	txe->state = EFA_RDM_TXE_REQ;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->total_len = 1024;
	txe->bytes_sent = 256;

	peer = txe->peer;
	assert_non_null(peer);
	/* Peer fully supports the feature; only the state gate should
	 * stop emission. */
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	outstanding_before = ep->efa_outstanding_tx_ops;

	efa_rdm_txe_handle_error(txe, FI_EINVAL,
		EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* User still sees a TX CQ error. */
	memset(&err_entry, 0, sizeof(err_entry));
	ret = fi_cq_readerr(resource->cq, &err_entry, 0);
	assert_int_equal(ret, 1);
	assert_int_equal(err_entry.prov_errno,
			 EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* No PEER_ERROR_PKT was posted despite bytes_sent > 0 and full
	 * peer support, because the txe was not in LONGCTS send state. */
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before);
}

/**
 * @brief Regression test for the RDMA-READ success-completion leak.
 *
 * A long-read receive posts multiple RDMA READ WRs on one rxe. When
 * the source MR is canceled mid-transfer the device delivers a mix
 * of failed and successful read completions for that rxe. The failed
 * WR marks the rxe peer-aborted (EFA_RDM_OPE_PEER_ABORT_PENDING; the user error
 * completion and rxe release are deferred to drain); a sibling that
 * still completes successfully lands in
 * efa_rdm_pke_handle_rma_completion()'s READ path.
 *
 * That success path must call the drain helper under the
 * EFA_RDM_OPE_PEER_ABORT_PENDING guard: when the successful sibling is
 * the last WR to drain, the rxe must be released there, and no spurious
 * EOR / success completion posted. This test fails one read WR,
 * then drives a successful completion for the sibling and asserts
 * the rxe leaves ep->base_ep.ope_list (freed exactly once), the user gets a
 * single clean peer-abort error completion, and no spurious success
 * completion is written.
 */
void test_efa_rdm_pke_handle_rma_read_completion_drains_recovered_rxe(
	void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *fail_pkt, *ok_pkt;
	struct efa_rdm_rma_context_pkt *ctx;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	peer_srx = util_get_peer_srx(ep->peer_srx_ep);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr,
				    &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr,
				      0, NULL), 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_recv(ep->peer_srx_ep, &iov, &desc, 1,
				    FI_ADDR_UNSPEC, /*context=*/(void *) 0xa1, 0);
	assert_int_equal(ret, FI_SUCCESS);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->peer_rxe = peer_rxe;
	rxe->bytes_read_total_len = 32;	/* two 16-byte READ WRs */

	ep->efa_outstanding_tx_ops += 2;
	peer->efa_outstanding_tx_ops += 2;
	rxe->efa_outstanding_tx_ops += 2;

	/* WR #1 fails (peer-abort) -> mark sets EFA_RDM_OPE_PEER_ABORT_PENDING. No peer feature
	 * support so no PEER_ERROR_PKT; drain is a no-op (sibling still
	 * outstanding), and the user error completion is deferred. */
	fail_pkt = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				     EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(fail_pkt);
	efa_rdm_pke_set_ope(fail_pkt, rxe);
	fail_pkt->peer = peer;
	ctx = (struct efa_rdm_rma_context_pkt *) fail_pkt->wiredata;
	ctx->type = EFA_RDM_RMA_CONTEXT_PKT;
	ctx->version = EFA_RDM_PROTOCOL_VERSION;
	ctx->flags = 0;
	ctx->context_type = EFA_RDM_RDMA_READ_CONTEXT;
	efa_rdm_pke_handle_tx_error(fail_pkt,
		EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);

	assert_true(rxe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_int_equal(rxe->efa_outstanding_tx_ops, 1);
	assert_int_equal(efa_unit_test_get_dlist_length(&ep->base_ep.ope_list), 1);

	/* WR #2 completes SUCCESSFULLY (the leak trigger). The forged
	 * context releases the pkt itself, so do not touch ok_pkt
	 * after the call. */
	ok_pkt = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				   EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(ok_pkt);
	efa_rdm_pke_set_ope(ok_pkt, rxe);
	ok_pkt->peer = peer;
	ctx = (struct efa_rdm_rma_context_pkt *) ok_pkt->wiredata;
	ctx->type = EFA_RDM_RMA_CONTEXT_PKT;
	ctx->version = EFA_RDM_PROTOCOL_VERSION;
	ctx->flags = 0;
	ctx->context_type = EFA_RDM_RDMA_READ_CONTEXT;
	ctx->seg_size = 16;

	efa_rdm_ep_record_tx_op_completed(ep, ok_pkt);
	efa_rdm_pke_handle_rma_completion(ok_pkt);
	ofi_genlock_unlock(srx_ctx->lock);

	/* Fix: drained on the success path. rxe is freed (off rxe_list),
	 * no spurious EOR, no success completion. The deferred peer-abort
	 * error completion was written at drain -- read it first (a pending
	 * error makes fi_cq_read return -FI_EAVAIL, not -FI_EAGAIN), then
	 * confirm the success queue is empty (no spurious EOR/success). */
	assert_int_equal(efa_unit_test_get_dlist_length(&ep->base_ep.ope_list), 0);
	assert_int_equal(ep->efa_outstanding_tx_ops, 0);
	assert_cq_peer_aborted_error(resource);
	assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);
}

/**
 * @brief Regression test for the CTS-outstanding inbound LONGCTS leak.
 *
 * On the LONGCTS recv path the receiver refills its window by posting
 * additional CTS packets mid-transfer, so a CTS can be in flight when
 * the sender's MR-cancel PEER_ERROR_PKT arrives. The inbound handler
 * recovers the rxe (marks EFA_RDM_OPE_PEER_ABORT_PENDING) and attempts a drain,
 * but the drain is a no-op while the CTS is outstanding
 * (efa_outstanding_tx_ops > 0).
 *
 * The CTS send-completion must call the drain helper. This test asserts
 * the rxe survives the inbound PEER_ERROR_PKT (CTS still in flight)
 * and is freed exactly when the CTS completes.
 */
void test_efa_rdm_pke_handle_peer_error_recv_longcts_cts_outstanding(
	void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct efa_rdm_pke *err_pkt, *cts_pkt;
	struct efa_rdm_peer_error_hdr *err_hdr;
	struct efa_rdm_ope *rxe;
	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	peer_srx = util_get_peer_srx(ep->peer_srx_ep);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
			 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_recv(ep->peer_srx_ep, &iov, &desc, 1,
				    FI_ADDR_UNSPEC, /*context=*/(void *) 0xa1, 0);
	assert_int_equal(ret, FI_SUCCESS);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_OPE_SEND;	/* LONGCTS recv: sending CTS/CTSDATA */
	/*
	 * A real rxe transitions to OPE_SEND together with being inserted
	 * onto ope_longcts_send_list (efa_rdm_pke_handle_cts_recv). Mirror
	 * that here so the drain's efa_rdm_rxe_handle_error() can validly
	 * dlist_remove(&rxe->entry) for the OPE_SEND state.
	 */
	dlist_insert_tail(&rxe->entry,
			  &efa_rdm_ep_rdm_domain(ep)->ope_longcts_send_list);
	rxe->peer_rxe = peer_rxe;

	/* Simulate a CTS in flight on this rxe (window refill). */
	cts_pkt = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				    EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(cts_pkt);
	efa_rdm_pke_set_ope(cts_pkt, rxe);
	cts_pkt->peer = peer;
	((struct efa_rdm_base_hdr *) cts_pkt->wiredata)->type = EFA_RDM_CTS_PKT;
	ep->efa_outstanding_tx_ops += 1;
	peer->efa_outstanding_tx_ops += 1;
	rxe->efa_outstanding_tx_ops += 1;

	/* Inbound PEER_ERROR_PKT (LONGCTS direction) targeting our rxe. */
	err_pkt = efa_rdm_pke_alloc(ep, ep->efa_rx_pkt_pool,
				    EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(err_pkt);
	ep->efa_rx_pkts_posted += 1;
	err_hdr = (struct efa_rdm_peer_error_hdr *) err_pkt->wiredata;
	err_hdr->type = EFA_RDM_PEER_ERROR_PKT;
	err_hdr->direction = EFA_RDM_PEER_ERROR_TX_TO_RX;
	err_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	err_hdr->flags = EFA_RDM_PKT_CONNID_HDR;
	err_hdr->op_id_valid = 1;
	err_hdr->msg_id = rxe->msg_id;
	err_hdr->op_id = rxe->rx_id;
	err_hdr->prov_errno = EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY;
	err_hdr->connid = 0xbeef;
	err_pkt->pkt_size = sizeof(struct efa_rdm_peer_error_hdr);

	efa_rdm_pke_handle_peer_error_recv(err_pkt);

	/* Recovery ran (EFA_RDM_OPE_PEER_ABORT_PENDING set) but the rxe must NOT be freed yet:
	 * the CTS is still outstanding, so the drain was a no-op. */
	assert_true(rxe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_int_equal(efa_unit_test_get_dlist_length(&ep->base_ep.ope_list), 1);

	/* The CTS send completes. Its send-completion handler must now
	 * drain and free the rxe (the regression: a bare break left it
	 * leaked forever). */
	efa_rdm_pke_handle_send_completion(cts_pkt);
	ofi_genlock_unlock(srx_ctx->lock);

	assert_int_equal(efa_unit_test_get_dlist_length(&ep->base_ep.ope_list), 0);
	assert_int_equal(ep->efa_outstanding_tx_ops, 0);
}

/* Forge an RTM TX pkt_entry of the given base-header type bound to txe
 * and drive the TX error dispatcher. Pre-bumps the outstanding-tx
 * counters to mirror a submitted RTM WR so handle_tx_error's
 * record_tx_op_completed returns them to baseline. */
static void run_rtm_tx_error_with_type(struct efa_resource *resource,
				       struct efa_rdm_ope *txe,
				       int pkt_type, int prov_errno)
{
	struct efa_rdm_ep *ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_base_hdr *base_hdr;

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pkt_entry);
	efa_rdm_pke_set_ope(pkt_entry, txe);
	pkt_entry->peer = txe->peer;

	/* Mirror efa_rdm_msg_post_rtm: the selected protocol is recorded
	 * on the txe. The forged txe bypasses post_rtm, so set it here so
	 * the PEER_ERROR ref_kind derivation sees the right protocol. */
	txe->protocol = pkt_type;

	base_hdr = (struct efa_rdm_base_hdr *) pkt_entry->wiredata;
	base_hdr->type = pkt_type;
	base_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	base_hdr->flags = 0;

	ep->efa_outstanding_tx_ops++;
	txe->peer->efa_outstanding_tx_ops++;
	txe->efa_outstanding_tx_ops++;

	efa_rdm_pke_handle_tx_error(pkt_entry, prov_errno);
}

/* Build a matched rxe, insert it into the peer's rxe_map under msg_id,
 * and drive an inbound medium PEER_ERROR_PKT (ref_kind=MSG_ID,
 * op_id=msg_id). Asserts the matched peer_rxe is returned to the SRX
 * (freed, not re-queued) so the given queue is empty, and a clean
 * peer-abort error completion is written to the user CQ. op selects
 * msg vs tagged. */
static void run_medium_inbound_peer_abort(struct efa_resource *resource,
				       uint32_t op, uint64_t tag,
				       struct slist *queue,
				       struct slist *other_queue)
{
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_peer_error_hdr *err_hdr;
	struct efa_rdm_ope *rxe;
	const uint64_t msg_id = 0x77;
	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	int ret;

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	peer_srx = util_get_peer_srx(ep->peer_srx_ep);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr,
				    &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr,
				      0, NULL), 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	if (op == ofi_op_tagged)
		ret = util_srx_generic_trecv(ep->peer_srx_ep, &iov, &desc, 1,
					     FI_ADDR_UNSPEC, (void *) 0xa1,
					     tag, 0, 0);
	else
		ret = util_srx_generic_recv(ep->peer_srx_ep, &iov, &desc, 1,
					    FI_ADDR_UNSPEC, (void *) 0xa1, 0);
	assert_int_equal(ret, FI_SUCCESS);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = tag;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	if (op == ofi_op_tagged)
		ret = peer_srx->owner_ops->get_tag(peer_srx, &match_attr,
						   &peer_rxe);
	else
		ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr,
						   &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, op);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->peer_rxe = peer_rxe;
	rxe->msg_id = msg_id;
	rxe->tag = tag;
	rxe->cq_entry.flags = (op == ofi_op_tagged) ? (FI_RECV | FI_TAGGED)
						    : (FI_RECV | FI_MSG);
	/* A medium message inserts its rxe into the peer's rxe_map
	 * keyed by msg_id; the inbound MSG_ID path resolves it there. */
	efa_rdm_rxe_map_insert(&peer->rxe_map, msg_id, rxe);

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_rx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	ep->efa_rx_pkts_posted += 1;
	pkt_entry->peer = peer;

	err_hdr = (struct efa_rdm_peer_error_hdr *) pkt_entry->wiredata;
	err_hdr->type = EFA_RDM_PEER_ERROR_PKT;
	err_hdr->direction = EFA_RDM_PEER_ERROR_TX_TO_RX;
	err_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	err_hdr->flags = EFA_RDM_PKT_CONNID_HDR;
	err_hdr->op_id_valid = 0;
	err_hdr->msg_id = msg_id;
	err_hdr->op_id = 0;
	err_hdr->prov_errno = EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY;
	err_hdr->connid = 0xbeef;
	pkt_entry->pkt_size = sizeof(struct efa_rdm_peer_error_hdr);

	efa_rdm_pke_handle_peer_error_recv(pkt_entry);

	/* The medium rxe has no outstanding WR, so the drain fires
	 * immediately: the matched peer_rxe is returned to the SRX
	 * (freed, not re-queued) and the rxe is reaped, so neither queue
	 * holds it. */
	assert_true(slist_empty(queue));
	if (other_queue)
		assert_true(slist_empty(other_queue));
	ofi_genlock_unlock(srx_ctx->lock);

	/* The matched recv gets a clean peer-abort error completion. */
	assert_cq_peer_aborted_error(resource);
}

/**
 * @brief Medium inbound (msg): a PEER_ERROR_PKT with ref_kind=MSG_ID
 *        resolves the rxe via the peer's rxe_map, marks it
 *        peer-aborted, and (no outstanding WR) completes it with a
 *        clean peer-abort error at drain.
 */
void test_efa_rdm_pke_handle_peer_error_recv_medium_reaps_rxe(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);

	run_medium_inbound_peer_abort(resource, ofi_op_msg, 0,
				   &srx_ctx->msg_queue, &srx_ctx->tag_queue);
}

/**
 * @brief Medium inbound (tagged): same as above but confirms tag_queue
 *        routing for a tagged rxe.
 */
void test_efa_rdm_pke_handle_peer_error_recv_medium_tagged(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);

	run_medium_inbound_peer_abort(resource, ofi_op_tagged, 0x42,
				   &srx_ctx->tag_queue, &srx_ctx->msg_queue);
}

/**
 * @brief Medium inbound miss: a msg_id-only PEER_ERROR_PKT whose msg_id is
 *        not in the peer's rxe_map (e.g. all medium WRs failed so the
 *        receiver never built an rxe) does not crash and writes no user CQ
 *        error. With no matched rxe the receiver-decides dispatcher routes
 *        it to the abort-marker path (unblocking the reorder window); an
 *        in-window msg_id is used so this stays on the recvwin path.
 */
void test_efa_rdm_pke_handle_peer_error_recv_medium_msg_id_not_found_dropped(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_peer_error_hdr *err_hdr;
	struct fi_cq_err_entry err_entry;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr,
				    &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr,
				      0, NULL), 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_rx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	ep->efa_rx_pkts_posted = efa_base_ep_get_rx_pool_size(&ep->base_ep);
	pkt_entry->peer = peer;

	err_hdr = (struct efa_rdm_peer_error_hdr *) pkt_entry->wiredata;
	err_hdr->type = EFA_RDM_PEER_ERROR_PKT;
	err_hdr->direction = EFA_RDM_PEER_ERROR_TX_TO_RX;
	err_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	err_hdr->flags = EFA_RDM_PKT_CONNID_HDR;
	err_hdr->op_id_valid = 0;
	err_hdr->msg_id = 0x7;	/* never inserted, but in-window */
	err_hdr->op_id = 0;
	err_hdr->prov_errno = EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY;
	err_hdr->connid = 0xbeef;
	pkt_entry->pkt_size = sizeof(struct efa_rdm_peer_error_hdr);

	efa_rdm_pke_handle_peer_error_recv(pkt_entry);

	ret = fi_cq_readerr(resource->cq, &err_entry, 0);
	assert_int_equal(ret, -FI_EAGAIN);
}

/**
 * @brief PEER_ERROR_PKT (MSG_ID) arrives for an unexpected rxe
 *        (RXE_UNEXP). The rxe has no user op bound, so the handler must:
 *          1. Release unexp_pkt (buffered segments).
 *          2. Release the rxe via efa_rdm_rxe_release (frees peer_rxe
 *             through the SRX, removes from rxe_map).
 *          3. NOT call recover (which would corrupt the SRX).
 *          4. NOT write a user CQ error (no user op).
 */
void test_efa_rdm_pke_handle_peer_error_recv_medium_unexpected_tears_down(
	void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct util_srx_ctx *srx_ctx;
	struct efa_rdm_peer *peer;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct efa_rdm_pke *pkt_entry, *unexp_pkt;
	struct efa_rdm_peer_error_hdr *err_hdr;
	struct efa_rdm_ope *rxe;
	struct fi_peer_rx_entry *peer_rxe;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	const uint32_t msg_id = 0x55;
	size_t rxe_list_len_before;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	peer_srx = util_get_peer_srx(ep->peer_srx_ep);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr,
				    &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &peer_addr,
				      0, NULL), 1);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	assert_non_null(peer);

	/* Allocate the inbound PEER_ERROR pke early (efa_rx_pkt_pool has
	 * limited entries). */
	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_rx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	ep->efa_rx_pkts_posted = efa_base_ep_get_rx_pool_size(&ep->base_ep);
	pkt_entry->peer = peer;

	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_recv(ep->peer_srx_ep, &iov, &desc, 1,
				    FI_ADDR_UNSPEC, (void *) 0xa1, 0);
	assert_int_equal(ret, FI_SUCCESS);

	ofi_genlock_lock(srx_ctx->lock);

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0;
	match_attr.msg_size = 16;
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_UNEXP;
	rxe->peer_rxe = peer_rxe;
	rxe->msg_id = msg_id;

	unexp_pkt = efa_rdm_pke_alloc(ep, ep->rx_unexp_pkt_pool,
				      EFA_RDM_PKE_FROM_UNEXP_POOL);
	assert_non_null(unexp_pkt);
	rxe->unexp_pkt = unexp_pkt;

	efa_rdm_rxe_map_insert(&peer->rxe_map, msg_id, rxe);

	rxe_list_len_before = efa_unit_test_get_dlist_length(&ep->base_ep.ope_list);

	/* Fill in the PEER_ERROR_PKT wiredata. */
	err_hdr = (struct efa_rdm_peer_error_hdr *) pkt_entry->wiredata;
	err_hdr->type = EFA_RDM_PEER_ERROR_PKT;
	err_hdr->direction = EFA_RDM_PEER_ERROR_TX_TO_RX;
	err_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	err_hdr->flags = EFA_RDM_PKT_CONNID_HDR;
	err_hdr->op_id_valid = 0;
	err_hdr->msg_id = msg_id;
	err_hdr->op_id = 0;
	err_hdr->prov_errno = EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY;
	err_hdr->connid = 0xbeef;
	pkt_entry->pkt_size = sizeof(struct efa_rdm_peer_error_hdr);

	efa_rdm_pke_handle_peer_error_recv(pkt_entry);
	ofi_genlock_unlock(srx_ctx->lock);

	/* rxe was freed (removed from rxe_list). */
	assert_int_equal(efa_unit_test_get_dlist_length(&ep->base_ep.ope_list),
			 rxe_list_len_before - 1);

	/* SRX msg_queue must be empty (peer_rxe freed, not re-queued). */
	assert_true(slist_empty(&srx_ctx->msg_queue));
}

/**
 * @brief Medium sender-side: a medium RTM WR fails with
 *        LOCAL_ERROR_INVALID_LKEY (user closed the source MR) and the
 *        peer supports the feature. A single failing WR with
 *        bytes_acked == 0 (zero-delivery) emits a msg_id-only
 *        PEER_ERROR_PKT at drain: the receiver matched no rxe (owes no
 *        completion) but its reorder window reserved this msg_id, so the
 *        abort unblocks it. The txe stays alive until the PEER_ERROR
 *        packet's own completion drains it.
 */
void test_efa_rdm_pke_handle_tx_error_medium_emits_skip_on_zero_delivery(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	struct fi_cq_err_entry err_entry;
	size_t outstanding_before;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	/* Medium txe never receives a CTS: it stays in TXE_REQ. */
	txe->state = EFA_RDM_TXE_REQ;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->cq_entry.op_context = (void *) 0xa1;
	txe->msg_id = 0x99;

	peer = txe->peer;
	assert_non_null(peer);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	outstanding_before = ep->efa_outstanding_tx_ops;

	efa_unit_test_txe_simulate_source_mr_canceled(txe);
	run_rtm_tx_error_with_type(resource, txe, EFA_RDM_MEDIUM_MSGRTM_PKT,
				   EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* The completion is withheld -- deferred to the PEER_ERROR packet's
	 * drain (not written eagerly here). */
	memset(&err_entry, 0, sizeof(err_entry));
	ret = fi_cq_readerr(resource->cq, &err_entry, 0);
	assert_int_equal(ret, -FI_EAGAIN);

	/* Zero-delivery (bytes_acked == 0): the single data WR drained, so
	 * the drain helper emitted a msg_id-only PEER_ERROR_PKT (window
	 * unblock). That packet is itself a WR, so outstanding rose by one
	 * and the txe is still alive (off no list yet -- kept for the skip
	 * packet's completion). */
	assert_true(txe->internal_flags & EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before + 1);
}

/**
 * @brief Medium sender-side: a medium message posts multiple RTM WRs.
 *        With bytes_acked > 0 (partial delivery), the deferred drain
 *        emits exactly one PEER_ERROR_PKT. A second failing sibling WR
 *        drains the txe and triggers the emit. No duplicate is possible
 *        because EMITTED is set on the first emit.
 */
void test_efa_rdm_pke_handle_tx_error_medium_emits_peer_error_once(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_TXE_REQ;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->cq_entry.op_context = (void *) 0xa1;
	txe->msg_id = 0x99;
	txe->total_len = 1024;
	/* Simulate partial delivery: one segment landed before MR cancel. */
	txe->bytes_acked = 256;

	peer = txe->peer;
	assert_non_null(peer);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	/* Simulate a second WR still in flight so the first failure's
	 * drain is a no-op. */
	txe->efa_outstanding_tx_ops = 1;

	/* First failing medium WR: sets PENDING, drain sees outstanding=1
	 * → no-op (sibling still in flight). */
	efa_unit_test_txe_simulate_source_mr_canceled(txe);
	run_rtm_tx_error_with_type(resource, txe, EFA_RDM_MEDIUM_MSGRTM_PKT,
				   EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);
	assert_true(txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_false(txe->internal_flags & EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);
	assert_int_equal(txe->efa_outstanding_tx_ops, 1);

	/* Simulate the sibling WR draining (its completion arrived). */
	txe->efa_outstanding_tx_ops--;
	/* Now txe->efa_outstanding_tx_ops == 0 with PENDING set: the drain
	 * helper emits (bytes_acked > 0). */
	efa_rdm_txe_progress_peer_abort_if_drained(txe);

	/* Exactly one PEER_ERROR_PKT posted at the drain point. */
	assert_true(txe->internal_flags & EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);
	/* The emitted PEER_ERROR_PKT bumps outstanding by 1. */
	assert_int_equal(txe->efa_outstanding_tx_ops, 1);
}

/**
 * @brief Medium sender-side, DC variant: a DC_MEDIUM RTM WR failing
 *        with INVALID_LKEY and bytes_acked == 0 (zero-delivery) emits a
 *        msg_id-only PEER_ERROR_PKT at drain to unblock the
 *        receiver's reorder window. Pins efa_rdm_pkt_type_is_medium()
 *        coverage of DC medium types.
 */
void test_efa_rdm_pke_handle_tx_error_dc_medium_emits_skip_on_zero_delivery(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	size_t outstanding_before;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_TXE_REQ;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->msg_id = 0x99;

	peer = txe->peer;
	assert_non_null(peer);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	outstanding_before = ep->efa_outstanding_tx_ops;

	efa_unit_test_txe_simulate_source_mr_canceled(txe);
	run_rtm_tx_error_with_type(resource, txe, EFA_RDM_DC_MEDIUM_MSGRTM_PKT,
				   EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* Zero-delivery: drain emits the PEER_ERROR_PKT (one WR), txe alive. */
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before + 1);
}

/**
 * @brief Medium sender-side: peer does not advertise the feature ->
 *        no PEER_ERROR_PKT. The user still sees a TX CQ error.
 */
void test_efa_rdm_pke_handle_tx_error_medium_no_emit_when_peer_unsupported(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	struct fi_cq_err_entry err_entry;
	size_t outstanding_before;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	assert_false(ep->homogeneous_peers);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_TXE_REQ;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->msg_id = 0x99;

	peer = txe->peer;
	/* Handshake received but feature bit NOT advertised -- old peer. */
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] = 0;

	outstanding_before = ep->efa_outstanding_tx_ops;

	run_rtm_tx_error_with_type(resource, txe, EFA_RDM_MEDIUM_MSGRTM_PKT,
				   EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	memset(&err_entry, 0, sizeof(err_entry));
	ret = fi_cq_readerr(resource->cq, &err_entry, 0);
	assert_int_equal(ret, 1);
	assert_int_equal(err_entry.prov_errno,
			 EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* No PEER_ERROR_PKT posted. */
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before);
}

/**
 * @brief Medium sender-side: ep->homogeneous_peers bypasses the
 *        handshake check. A single WR with bytes_acked == 0
 *        (zero-delivery) emits a msg_id-only PEER_ERROR_PKT at drain to unblock the
 *        window. The PENDING flag confirms the handshake bypass.
 */
void test_efa_rdm_pke_handle_tx_error_medium_emits_skip_with_homogeneous_peers(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	size_t outstanding_before;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	ep->homogeneous_peers = true;

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_TXE_REQ;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->msg_id = 0x99;

	peer = txe->peer;
	assert_false(peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED);
	assert_false(efa_rdm_peer_support_peer_error(peer));

	outstanding_before = ep->efa_outstanding_tx_ops;

	efa_unit_test_txe_simulate_source_mr_canceled(txe);
	run_rtm_tx_error_with_type(resource, txe, EFA_RDM_MEDIUM_MSGRTM_PKT,
				   EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* Zero-delivery with homogeneous_peers: handshake bypassed, drain
	 * emits the msg_id-only PEER_ERROR_PKT (one WR), txe kept alive. */
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before + 1);
}

/**
 * @brief An eager two-sided RTM failing with INVALID_LKEY (source MR
 *        closed) emits a msg_id-only PEER_ERROR_PKT: eager owes the
 *        receiver no completion, but its single REQ consumed a msg_id the
 *        receiver's reorder window reserved, so the abort unblocks it.
 */
void test_efa_rdm_pke_handle_tx_error_eager_emits_skip(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	struct fi_cq_err_entry err_entry;
	size_t outstanding_before;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_TXE_REQ;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->cq_entry.op_context = (void *) 0xa1;
	txe->msg_id = 0x99;

	peer = txe->peer;
	assert_non_null(peer);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	outstanding_before = ep->efa_outstanding_tx_ops;

	efa_unit_test_txe_simulate_source_mr_canceled(txe);
	run_rtm_tx_error_with_type(resource, txe, EFA_RDM_EAGER_MSGRTM_PKT,
				   EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* The completion is withheld -- deferred to the PEER_ERROR packet's
	 * drain (not written eagerly here). */
	memset(&err_entry, 0, sizeof(err_entry));
	ret = fi_cq_readerr(resource->cq, &err_entry, 0);
	assert_int_equal(ret, -FI_EAGAIN);

	/* The eager classifier marked the txe PENDING; the single WR drained,
	 * so the drain helper emitted the PEER_ERROR_PKT (one new WR), keeping
	 * the txe alive for its completion. The wire identifiers (msg_id only
	 * for eager) are derived at emit time -- see
	 * test_efa_rdm_pke_init_peer_error_for_ope_eager_skip. */
	assert_true(txe->internal_flags & EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before + 1);
}

/**
 * @brief An eager two-sided RTM whose peer does NOT advertise the
 *        feature must not emit -- the user still gets a TX CQ error and
 *        the reorder window unblock is skipped (status quo for old
 *        peers).
 */
void test_efa_rdm_pke_handle_tx_error_eager_no_emit_when_peer_unsupported(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	size_t outstanding_before;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);
	assert_false(ep->homogeneous_peers);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_TXE_REQ;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->msg_id = 0x99;

	peer = txe->peer;
	/* Handshake received but feature bit NOT advertised -- old peer. */
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] = 0;

	outstanding_before = ep->efa_outstanding_tx_ops;

	run_rtm_tx_error_with_type(resource, txe, EFA_RDM_EAGER_MSGRTM_PKT,
				   EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* No PEER_ERROR_PKT posted, not flagged for abort. */
	assert_false(txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before);
}

/**
 * @brief A one-sided eager RTW (write) failing with INVALID_LKEY must
 *        NOT take the MSG_ID_SKIP path: it is not a two-sided RTM, owes
 *        the target no completion, and consumes no reorder-window slot.
 *        Guards efa_rdm_pkt_type_is_eager_rtm() excluding RTW.
 */
void test_efa_rdm_pke_handle_tx_error_eager_rtw_no_emit(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	size_t outstanding_before;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_write);
	assert_non_null(txe);
	txe->state = EFA_RDM_TXE_REQ;
	txe->cq_entry.flags = FI_RMA | FI_WRITE;
	txe->msg_id = 0x99;

	peer = txe->peer;
	assert_non_null(peer);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	outstanding_before = ep->efa_outstanding_tx_ops;

	run_rtm_tx_error_with_type(resource, txe, EFA_RDM_EAGER_RTW_PKT,
				   EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* RTW is excluded from the eager_rtm predicate: no abort emit. */
	assert_false(txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before);
}

/**
 * @brief A medium RTM failing with a non-LKEY prov_errno must NOT emit
 *        a PEER_ERROR_PKT -- the medium emit gate is INVALID_LKEY-only.
 */
void test_efa_rdm_pke_handle_tx_error_medium_no_emit_on_non_lkey_errno(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	size_t outstanding_before;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_TXE_REQ;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->msg_id = 0x99;

	peer = txe->peer;
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	outstanding_before = ep->efa_outstanding_tx_ops;

	run_rtm_tx_error_with_type(resource, txe, EFA_RDM_MEDIUM_MSGRTM_PKT,
				   EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_LENGTH);

	/* No PEER_ERROR_PKT posted: gate is INVALID_LKEY-only. */
	assert_int_equal(ep->efa_outstanding_tx_ops, outstanding_before);
}

/**
 * @brief Runt-only runting-read sender-side: a RUNTREAD RTM WR whose
 *        transfer has no READ remainder (total_len == bytes_runt) fails
 *        with INVALID_LKEY and the peer supports the feature -> the txe
 *        is flagged PEER_ABORT_PENDING and the abort is signalled by
 *        msg_id (REF_MSG_ID), exactly like a medium transfer.
 *
 * This is the runt-only analog of the medium MR-cancel: the whole
 * message rode the REQ packets, so there is no READ to fail and the
 * medium-style msg_id PEER_ERROR_PKT is the only abort signal.
 */
void test_efa_rdm_pke_handle_tx_error_runtread_only_emits_peer_error(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_TXE_REQ;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->cq_entry.op_context = (void *) 0xa1;
	txe->msg_id = 0x99;
	/* Runt-only: the whole message is the runt, no READ remainder. */
	txe->total_len = 4096;
	txe->bytes_runt = 4096;

	peer = txe->peer;
	assert_non_null(peer);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	/* Keep the txe alive past the single WR drain so we can inspect
	 * the flags the gate set (a sibling WR still in flight). */
	txe->efa_outstanding_tx_ops = 1;

	efa_unit_test_txe_simulate_source_mr_canceled(txe);
	run_rtm_tx_error_with_type(resource, txe, EFA_RDM_RUNTREAD_MSGRTM_PKT,
				   EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* The gate accepted the runt-only runtread: PENDING set, errno
	 * stashed. Runtread exchanges no CTS, so the emit is msg_id-only
	 * (rx_id stays the unset sentinel). */
	assert_true(txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_int_equal(txe->peer_error_prov_errno,
			 EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);
	assert_int_equal(txe->rx_id, EFA_RDM_OPE_INVALID_ID);
}

/**
 * @brief Runting-read WITH a READ remainder (bytes_runt < total_len)
 *        now ALSO emits a msg_id-only PEER_ERROR_PKT from the sender.
 *
 * Receiver-decides model (RuntRead bidirectional): the runt RTM(s) carry
 * user data from the source MR, so a source-MR cancel can flush them before
 * the receiver matches. The sender can no longer infer whether the receiver
 * matched, so it always marks the txe PEER_ABORT_PENDING and emits a
 * msg_id-only PEER_ERROR_PKT. This may race the receiver's own tail-READ
 * failure notification (RX->TX); both are made idempotent on the receiver.
 * (This inverts the pre-rework behavior, where the sender suppressed the
 * emit for the with-READ case and relied solely on the receiver's READ
 * failure.)
 */
void test_efa_rdm_pke_handle_tx_error_runtread_with_read_emits(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_TXE_REQ;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->cq_entry.op_context = (void *) 0xa1;
	txe->msg_id = 0x99;
	/* Has a READ remainder: runt portion is only part of the message. */
	txe->total_len = 1048576;
	txe->bytes_runt = 4096;

	peer = txe->peer;
	assert_non_null(peer);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	/* Keep the txe alive past the single WR drain so we can inspect the
	 * flags the gate set (a sibling WR still in flight). */
	txe->efa_outstanding_tx_ops = 1;

	/* Simulate the source-MR cancel so the MR gen check fails and the
	 * abort path is taken. */
	efa_unit_test_txe_simulate_source_mr_canceled(txe);
	run_rtm_tx_error_with_type(resource, txe, EFA_RDM_RUNTREAD_MSGRTM_PKT,
				   EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	/* The gate now accepts runtread WITH a READ remainder (previously
	 * excluded): PENDING set, errno stashed. Runtread exchanges no CTS,
	 * so the emit is msg_id-only (rx_id unset). The receiver reconciles
	 * this msg_id-only notification with any tail-READ-failure notification
	 * idempotently. */
	assert_true(txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	assert_int_equal(txe->peer_error_prov_errno,
			 EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);
	assert_int_equal(txe->rx_id, EFA_RDM_OPE_INVALID_ID);
}

/**
 * @brief Regression test for the LONGCTS sender-side abort txe leak.
 *
 * When a LONGCTS transfer's source MR is canceled mid-CTSDATA, the
 * failing WR runs the abort path: it writes the TX CQ error, emits a
 * PEER_ERROR_PKT (its own outstanding WR on the txe), and marks the
 * txe PEER_ABORT_PENDING|PEER_ERROR_EMITTED -- it does NOT release the
 * txe, because the PEER_ERROR_PKT (and any sibling CTSDATA WRs) still
 * use it as wr_id.
 *
 * The PEER_ERROR_PKT completion must free the errored txe: the
 * success-completion path only releases on bytes_acked == total_len
 * (never true post-abort) and sibling failures early-return on
 * OPE_ERR. This test drives the abort, then completes the
 * emitted PEER_ERROR_PKT and asserts the drain-gated release frees the
 * txe exactly once.
 */
void test_efa_rdm_pke_handle_tx_error_longcts_abort_drains_txe(
	void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *err_pkt;
	struct fi_cq_data_entry cq_entry;
	struct fi_cq_err_entry cq_err_entry = {0};
	size_t txe_base;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_OPE_SEND;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->total_len = 4096;
	txe->bytes_acked = 256;        /* partial; never reaches total_len */
	txe->rx_id = 0x7;
	peer = txe->peer;
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;
	dlist_insert_tail(&txe->entry,
			  &efa_rdm_ep_rdm_domain(ep)->ope_longcts_send_list);

	txe->protocol = EFA_RDM_LONGCTS_MSGRTM_PKT;
	efa_unit_test_txe_simulate_source_mr_canceled(txe);

	/* efa_rdm_txe_construct (via the helper) already links the txe on
	 * ep->base_ep.ope_list; track it as the freed observable. */
	txe_base = efa_unit_test_get_dlist_length(&ep->base_ep.ope_list);

	/* Simulate the failing CTSDATA WR still being counted as in flight so
	 * handle_error's drain is a no-op until it drains. */
	txe->efa_outstanding_tx_ops = 1;

	/* A CTSDATA WR fails with INVALID_LKEY (source MR canceled).
	 * handle_error marks the txe and drives the drain itself, but with the
	 * WR still in flight it withholds the completion and emits nothing. */
	efa_rdm_txe_handle_error(txe, FI_EINVAL,
		EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);
	assert_int_equal(txe->state, EFA_RDM_OPE_ERR);
	assert_true(txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	/* Not yet emitted, nothing on the CQ, txe untouched. */
	assert_false(txe->internal_flags & EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);
	assert_int_equal(fi_cq_read(resource->cq, &cq_entry, 1), -FI_EAGAIN);
	assert_int_equal(efa_unit_test_get_dlist_length(&ep->base_ep.ope_list),
			 txe_base);

	/* The CTSDATA WR drains; the drain helper now emits the
	 * PEER_ERROR_PKT and keeps the txe alive until that packet's own
	 * send completion. */
	txe->efa_outstanding_tx_ops--;
	efa_rdm_txe_progress_peer_abort_if_drained(txe);
	assert_true(txe->internal_flags & EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);
	assert_int_equal(fi_cq_read(resource->cq, &cq_entry, 1), -FI_EAGAIN);
	assert_int_equal(efa_unit_test_get_dlist_length(&ep->base_ep.ope_list),
			 txe_base);
	assert_int_equal(txe->efa_outstanding_tx_ops, 1);

	/* The emitted PEER_ERROR_PKT completes. Complete the real packet
	 * the abort posted (on peer->outstanding_tx_pkts) so no live pke
	 * is left referencing the txe. The drain helper now surfaces the
	 * single completion and frees the txe. */
	err_pkt = NULL;
	dlist_foreach_container(&peer->outstanding_tx_pkts, struct efa_rdm_pke,
				err_pkt, entry) {
		if (efa_rdm_pke_get_base_hdr(err_pkt)->type ==
		    EFA_RDM_PEER_ERROR_PKT)
			break;
	}
	assert_non_null(err_pkt);
	assert_int_equal(efa_rdm_pke_get_base_hdr(err_pkt)->type,
			 EFA_RDM_PEER_ERROR_PKT);
	efa_rdm_pke_handle_send_completion(err_pkt);

	/* Exactly one TX completion, surfaced as the dedicated
	 * FI_ECANCELED / FI_EFA_ERR_PEER_ABORTED pair, only after the
	 * notification went out. */
	assert_int_equal(fi_cq_read(resource->cq, &cq_entry, 1), -FI_EAVAIL);
	assert_int_equal(fi_cq_readerr(resource->cq, &cq_err_entry, 0), 1);
	assert_int_equal(cq_err_entry.err, FI_ECANCELED);
	assert_int_equal(cq_err_entry.prov_errno, FI_EFA_ERR_PEER_ABORTED);
	/* No second completion. */
	assert_int_equal(fi_cq_read(resource->cq, &cq_entry, 1), -FI_EAGAIN);

	/* txe freed exactly once (off txe_list), outstanding == 0. */
	assert_int_equal(ep->efa_outstanding_tx_ops, 0);
	assert_int_equal(efa_unit_test_get_dlist_length(&ep->base_ep.ope_list),
			 txe_base - 1);
}

/*
 * A peer that does not advertise PEER_ERROR has no notification path, so the
 * abort completion must be written eagerly (with the raw errno) rather than
 * withheld -- withholding it would hang the op forever.
 */
void test_efa_rdm_txe_handle_error_no_defer_when_peer_unsupported(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	struct fi_cq_data_entry cq_entry;
	struct fi_cq_err_entry cq_err_entry = {0};

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_OPE_SEND;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->total_len = 4096;
	txe->bytes_acked = 256;
	peer = txe->peer;
	/* Deny every route to feature support. */
	peer->is_self = false;
	ep->homogeneous_peers = false;

	efa_rdm_txe_handle_error(txe, FI_EINVAL,
		EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);

	assert_false(txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	/* Completion written now, carrying the raw errno (not deferred). */
	assert_int_equal(fi_cq_read(resource->cq, &cq_entry, 1), -FI_EAVAIL);
	assert_int_equal(fi_cq_readerr(resource->cq, &cq_err_entry, 0), 1);
	assert_int_equal(cq_err_entry.err, FI_EINVAL);
	assert_int_equal(cq_err_entry.prov_errno,
			 EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);
}

/*
 * A CTSDATA send completion that lands on an already-aborting txe must not
 * write a (second) completion or release the txe; the completion + release
 * belong to the drain helper (run by the caller, not exercised here).
 */
void test_efa_rdm_ctsdata_send_completion_aborting_txe_no_completion(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *pkt_entry;
	struct fi_cq_data_entry cq_entry;
	size_t txe_base;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);
	txe->state = EFA_RDM_OPE_ERR;
	txe->cq_entry.flags = FI_SEND | FI_MSG;
	txe->total_len = 4096;
	txe->bytes_acked = 4096; /* would reach total_len if not guarded */
	txe->internal_flags |= EFA_RDM_OPE_PEER_ABORT_PENDING;
	/* Simulate a sibling WR still in flight so the drain helper is a no-op
	 * here: the aborting txe must be left untouched (no completion, not
	 * freed, nothing emitted) until its last WR drains. */
	txe->efa_outstanding_tx_ops = 1;
	txe_base = efa_unit_test_get_dlist_length(&ep->base_ep.ope_list);

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pkt_entry);
	pkt_entry->ope = txe;

	efa_rdm_pke_handle_ctsdata_send_completion(pkt_entry);

	/* No completion written, txe untouched (drain owns it). */
	assert_int_equal(fi_cq_read(resource->cq, &cq_entry, 1), -FI_EAGAIN);
	assert_int_equal(efa_unit_test_get_dlist_length(&ep->base_ep.ope_list),
			 txe_base);

	efa_rdm_pke_release_tx(pkt_entry);
}
