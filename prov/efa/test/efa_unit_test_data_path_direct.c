/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_unit_tests.h"
#include "ofi_util.h"
#include "efa_io_defs.h"
#include "efa_data_path_direct_entry.h"

static void test_efa_data_path_direct_multiple_sge_fail_impl(struct efa_resource **state, uint32_t fi_opcode)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff local_buff1, local_buff2;
	struct ibv_sge sge_list[2];
	struct efa_qp *qp;
	struct efa_cq *efa_cq;
	bool data_path_direct_enabled_orig;
	struct fid_ep *ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_cq = container_of(resource->cq, struct efa_cq, util_cq.cq_fid);
	data_path_direct_enabled_orig = efa_cq->ibv_cq.data_path_direct_enabled;
	efa_cq->ibv_cq.data_path_direct_enabled = true;

	/* Open a test ep with data path direct enabled */
	assert_int_equal(fi_endpoint(resource->domain, resource->info, &ep, NULL), 0);
	assert_int_equal(fi_ep_bind(ep, &resource->cq->fid, FI_SEND | FI_RECV), 0);
	assert_int_equal(fi_enable(ep), 0);

	qp = container_of(ep, struct efa_base_ep, util_ep.ep_fid)->qp;
	efa_unit_test_buff_construct(&local_buff1, resource, 2048);
	efa_unit_test_buff_construct(&local_buff2, resource, 2048);

	/* Setup SGE list with 2 elements (exceeds EFA_IO_TX_DESC_NUM_RDMA_BUFS=1) */
	sge_list[0].addr = (uintptr_t)local_buff1.buff;
	sge_list[0].length = local_buff1.size;
	sge_list[0].lkey = ((struct efa_mr *)fi_mr_desc(local_buff1.mr))->ibv_mr->lkey;
	sge_list[1].addr = (uintptr_t)local_buff2.buff;
	sge_list[1].length = local_buff2.size;
	sge_list[1].lkey = ((struct efa_mr *)fi_mr_desc(local_buff2.mr))->ibv_mr->lkey;

	/* Test multiple SGE failure with consolidated post functions */
	int ret;
	if (fi_opcode == FI_READ) {
		ret = efa_data_path_direct_post_read(qp, sge_list, 2, 123456, 0x87654321, 0, 0, NULL, 0, 0);
	} else {
		ret = efa_data_path_direct_post_write(qp, sge_list, 2, 123456, 0x87654321, 0, 0, 0, NULL, 0, 0);
	}
	assert_int_equal(ret, EINVAL);

	assert_int_equal(fi_close(&ep->fid), 0);
	efa_cq->ibv_cq.data_path_direct_enabled = data_path_direct_enabled_orig;
	efa_unit_test_buff_destruct(&local_buff1);
	efa_unit_test_buff_destruct(&local_buff2);
#else
	skip();
#endif
}

void test_efa_data_path_direct_rdma_read_multiple_sge_fail(struct efa_resource **state)
{
	test_efa_data_path_direct_multiple_sge_fail_impl(state, FI_READ);
}

void test_efa_data_path_direct_rdma_write_multiple_sge_fail(struct efa_resource **state)
{
	test_efa_data_path_direct_multiple_sge_fail_impl(state, FI_WRITE);
}


/**
 * @brief Verify that QP generation counter increments on each QP creation
 * for the same QPN slot, and that gen_mask/shifted_gen are set correctly
 * on the data path direct work queues.
 */
void test_efa_data_path_direct_qp_gen_initialization(struct efa_resource **state)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	struct efa_resource *resource = *state;
	struct efa_base_ep *base_ep;
	struct efa_qp *qp;
	struct efa_data_path_direct_wq *sq_wq, *rq_wq;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	base_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);
	qp = base_ep->qp;

	/* gen should be non-zero after first QP creation */
	assert_int_not_equal(qp->data_path_direct_qp.gen, 0);

	sq_wq = &qp->data_path_direct_qp.sq.wq;
	rq_wq = &qp->data_path_direct_qp.rq.wq;

	/* gen_mask should be the inverse of desc_mask, within 16 bits */
	assert_int_equal(sq_wq->gen_mask, (uint16_t)~sq_wq->desc_mask);
	assert_int_equal(rq_wq->gen_mask, (uint16_t)~rq_wq->desc_mask);

	/* shifted_gen should be gen shifted into the upper bits */
	assert_int_equal(sq_wq->shifted_gen,
			 (uint16_t)(qp->data_path_direct_qp.gen << __bf_shf(sq_wq->desc_mask + 1)));
	assert_int_equal(rq_wq->shifted_gen,
			 (uint16_t)(qp->data_path_direct_qp.gen << __bf_shf(rq_wq->desc_mask + 1)));

	/* gen_mask and desc_mask together should cover all 16 bits (no gap) */
	assert_int_equal((uint16_t)(sq_wq->gen_mask | sq_wq->desc_mask), 0xFFFF);
	assert_int_equal((uint16_t)(rq_wq->gen_mask | rq_wq->desc_mask), 0xFFFF);

	/* gen_mask and desc_mask should not overlap */
	assert_int_equal((uint16_t)(sq_wq->gen_mask & sq_wq->desc_mask), 0);
	assert_int_equal((uint16_t)(rq_wq->gen_mask & rq_wq->desc_mask), 0);
#else
	skip();
#endif
}

/**
 * @brief Verify that efa_wq_get_dev_req_id returns a request ID with
 * generation bits OR'd in, and that efa_wq_put_dev_req_id correctly
 * strips them before returning the index to the pool.
 */
void test_efa_data_path_direct_dev_req_id_roundtrip(struct efa_resource **state)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	struct efa_resource *resource = *state;
	struct efa_base_ep *base_ep;
	struct efa_qp *qp;
	struct efa_data_path_direct_wq *sq_wq;
	uint32_t dev_req_id;
	uint32_t wrid_idx;
	uint64_t test_wr_id = 0xDEADBEEF;
	uint16_t pool_next_before;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	base_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);
	qp = base_ep->qp;
	sq_wq = &qp->data_path_direct_qp.sq.wq;

	pool_next_before = sq_wq->wrid_idx_pool_next;

	/* Get a dev_req_id — should have gen bits set */
	dev_req_id = efa_wq_get_dev_req_id(sq_wq, test_wr_id);

	/* The generation bits should be present */
	assert_int_equal(dev_req_id & sq_wq->gen_mask, sq_wq->shifted_gen);

	/* The wrid index portion should be valid (within desc_mask range) */
	wrid_idx = dev_req_id & ~sq_wq->gen_mask;
	assert_true(wrid_idx <= sq_wq->desc_mask);

	/* The wr_id should be stored in the wrid array */
	assert_int_equal(sq_wq->wrid[wrid_idx], test_wr_id);

	/* Pool next should have advanced by 1 */
	assert_int_equal(sq_wq->wrid_idx_pool_next, pool_next_before + 1);

	/* Put it back using dev_req_id (with gen bits) */
	efa_wq_put_dev_req_id(sq_wq, dev_req_id);

	/* Pool next should be back to original */
	assert_int_equal(sq_wq->wrid_idx_pool_next, pool_next_before);
#else
	skip();
#endif
}

/**
 * @brief Verify that efa_data_path_direct_is_valid_wrid_qp_gen detects
 * a stale completion whose generation bits don't match the current QP.
 */
void test_efa_data_path_direct_stale_completion_detected(struct efa_resource **state)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	struct efa_resource *resource = *state;
	struct efa_base_ep *base_ep;
	struct efa_qp *qp;
	struct efa_data_path_direct_wq *sq_wq;
	struct efa_io_cdesc_common fake_cqe = {0};

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	base_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);
	qp = base_ep->qp;
	sq_wq = &qp->data_path_direct_qp.sq.wq;

	/* Build a fake CQE for a send completion with correct generation */
	fake_cqe.req_id = sq_wq->shifted_gen;
	EFA_SET(&fake_cqe.flags, EFA_IO_CDESC_COMMON_Q_TYPE, EFA_IO_SEND_QUEUE);

	/* Current gen should be valid */
	assert_true(efa_data_path_direct_is_valid_wrid_qp_gen(&fake_cqe, qp));

	/* Now simulate a stale completion with a different generation.
	 * req_id is uint16_t, so use a gen value offset by 1 within the
	 * 16-bit range to ensure it differs from the current shifted_gen.
	 */
	fake_cqe.req_id = (uint16_t)(sq_wq->shifted_gen + (1 << __bf_shf(sq_wq->desc_mask + 1)));

	/* Stale gen should be detected as invalid */
	assert_false(efa_data_path_direct_is_valid_wrid_qp_gen(&fake_cqe, qp));

	/* Unsolicited recv completions should always be valid (no wrid to check) */
	EFA_SET(&fake_cqe.flags, EFA_IO_CDESC_COMMON_Q_TYPE, EFA_IO_RECV_QUEUE);
	EFA_SET(&fake_cqe.flags, EFA_IO_CDESC_COMMON_UNSOLICITED, 1);
	assert_true(efa_data_path_direct_is_valid_wrid_qp_gen(&fake_cqe, qp));
#else
	skip();
#endif
}

/**
 * @brief Verify that the QP generation counter increments across
 * QP destroy/create cycles on the same QPN slot.
 */
void test_efa_data_path_direct_qp_gen_increments_across_qps(struct efa_resource **state)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	struct efa_resource *resource = *state;
	struct efa_base_ep *base_ep;
	uint8_t first_gen;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	base_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);
	first_gen = base_ep->qp->data_path_direct_qp.gen;

	efa_unit_test_resource_destruct(resource);
	memset(resource, 0, sizeof(*resource));

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	base_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);

	assert_int_equal(base_ep->qp->data_path_direct_qp.gen, first_gen + 1);
#else
	skip();
#endif
}
