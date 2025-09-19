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

	/* Setup SGE list with 2 elements (exceeds EFA_DEVICE_MAX_RDMA_SGE=1) */
	sge_list[0].addr = (uintptr_t)local_buff1.buff;
	sge_list[0].length = local_buff1.size;
	sge_list[0].lkey = ((struct efa_mr *)fi_mr_desc(local_buff1.mr))->ibv_mr->lkey;
	sge_list[1].addr = (uintptr_t)local_buff2.buff;
	sge_list[1].length = local_buff2.size;
	sge_list[1].lkey = ((struct efa_mr *)fi_mr_desc(local_buff2.mr))->ibv_mr->lkey;

	/* Start RDMA operation and test multiple SGE failure */
	if (fi_opcode == FI_READ)
		efa_data_path_direct_wr_rdma_read(qp, 123456, 0x87654321);
	else
		efa_data_path_direct_wr_rdma_write(qp, 123456, 0x87654321);
	efa_data_path_direct_wr_set_sge_list(qp, 2, sge_list);
	assert_int_equal(qp->data_path_direct_qp.wr_session_err, EINVAL);

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