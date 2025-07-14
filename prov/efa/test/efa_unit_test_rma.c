/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_unit_tests.h"
#include "ofi_util.h"

extern struct fi_ops_rma efa_rma_ops;

static void test_efa_rma_prep(struct efa_resource *resource, fi_addr_t *addr)
{
	struct efa_ep_addr raw_addr;
	struct efa_base_ep *base_ep;
	size_t raw_addr_len = sizeof(raw_addr);
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	base_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);
	/* Add rma caps explicitly to ep->info to allow local test */
	base_ep->info->caps |= FI_RMA;
	/* Set up the mock operations */
	g_efa_unit_test_mocks.efa_qp_post_recv = &efa_mock_efa_qp_post_recv_return_mock;
	g_efa_unit_test_mocks.efa_qp_wr_complete = &efa_mock_efa_qp_wr_complete_no_op;
	g_efa_unit_test_mocks.efa_qp_wr_rdma_read = &efa_mock_efa_qp_wr_rdma_read_save_wr;
	g_efa_unit_test_mocks.efa_qp_wr_rdma_write = &efa_mock_efa_qp_wr_rdma_write_save_wr;
	g_efa_unit_test_mocks.efa_qp_wr_rdma_write_imm =
		&efa_mock_efa_qp_wr_rdma_write_imm_save_wr;
	g_efa_unit_test_mocks.efa_qp_wr_send = &efa_mock_efa_qp_wr_send_save_wr;
	g_efa_unit_test_mocks.efa_qp_wr_send_imm = &efa_mock_efa_qp_wr_send_imm_save_wr;
	g_efa_unit_test_mocks.efa_qp_wr_set_inline_data_list =
		&efa_mock_efa_qp_wr_set_inline_data_list_no_op;
	g_efa_unit_test_mocks.efa_qp_wr_set_sge_list = &efa_mock_efa_qp_wr_set_sge_list_no_op;
	g_efa_unit_test_mocks.efa_qp_wr_set_ud_addr = &efa_mock_efa_qp_wr_set_ud_addr_no_op;
	g_efa_unit_test_mocks.efa_qp_wr_start = &efa_mock_efa_qp_wr_start_no_op;

	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, addr, 0 /* flags */,
			   NULL /* context */);
	assert_int_equal(ret, 1);
}

void test_efa_rma_read(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff local_buff;
	fi_addr_t src_addr;
	void *desc;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	test_efa_rma_prep(resource, &src_addr);
	efa_unit_test_buff_construct(&local_buff, resource, 4096 /* buff_size */);

	desc = fi_mr_desc(local_buff.mr);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_read(resource->ep, local_buff.buff, local_buff.size, desc,
		      src_addr, remote_addr, remote_key, NULL /* context */);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&local_buff);
}

void test_efa_rma_readv(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff local_buff;
	struct iovec iov;
	fi_addr_t src_addr;
	void *desc;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	test_efa_rma_prep(resource, &src_addr);
	efa_unit_test_buff_construct(&local_buff, resource, 4096 /* buff_size */);

	iov.iov_base = local_buff.buff;
	iov.iov_len = local_buff.size;
	desc = fi_mr_desc(local_buff.mr);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_readv(resource->ep, &iov, &desc, 1, src_addr, remote_addr,
		       remote_key, NULL /* context */);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&local_buff);
}

void test_efa_rma_readmsg(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff local_buff;
	struct iovec iov;
	struct fi_msg_rma msg = {0};
	struct fi_rma_iov rma_iov;
	fi_addr_t src_addr;
	void *desc;
	int ret;

	test_efa_rma_prep(resource, &src_addr);
	efa_unit_test_buff_construct(&local_buff, resource, 4096 /* buff_size */);

	iov.iov_base = local_buff.buff;
	iov.iov_len = local_buff.size;
	desc = fi_mr_desc(local_buff.mr);
	rma_iov.len = local_buff.size;
	rma_iov.addr = 0x87654321;
	rma_iov.key = 123456;
	efa_unit_test_construct_msg_rma(&msg, &iov, &desc, 1, src_addr,
					&rma_iov, 1, NULL, 0);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_readmsg(resource->ep, &msg, 0);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&local_buff);
}

void test_efa_rma_write(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff local_buff;
	fi_addr_t dest_addr;
	void *desc;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	test_efa_rma_prep(resource, &dest_addr);
	efa_unit_test_buff_construct(&local_buff, resource, 4096 /* buff_size */);

	desc = fi_mr_desc(local_buff.mr);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_write(resource->ep, local_buff.buff, local_buff.size, desc,
		       dest_addr, remote_addr, remote_key, NULL /* context */);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&local_buff);
}

void test_efa_rma_writev(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff local_buff;
	struct iovec iov;
	fi_addr_t dest_addr;
	void *desc;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	test_efa_rma_prep(resource, &dest_addr);
	efa_unit_test_buff_construct(&local_buff, resource, 4096 /* buff_size */);

	iov.iov_base = local_buff.buff;
	iov.iov_len = local_buff.size;
	desc = fi_mr_desc(local_buff.mr);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_writev(resource->ep, &iov, &desc, 1, dest_addr, remote_addr,
			remote_key, NULL /* context */);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&local_buff);
}

void test_efa_rma_writemsg(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff local_buff;
	struct iovec iov;
	struct fi_msg_rma msg = {0};
	struct fi_rma_iov rma_iov;
	fi_addr_t dest_addr;
	void *desc;
	int ret;

	test_efa_rma_prep(resource, &dest_addr);
	efa_unit_test_buff_construct(&local_buff, resource, 4096 /* buff_size */);

	iov.iov_base = local_buff.buff;
	iov.iov_len = local_buff.size;
	desc = fi_mr_desc(local_buff.mr);
	rma_iov.len = local_buff.size;
	rma_iov.addr = 0x87654321;
	rma_iov.key = 123456;
	efa_unit_test_construct_msg_rma(&msg, &iov, &desc, 1, dest_addr, &rma_iov,
					1, NULL, 0);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_writemsg(resource->ep, &msg, 0);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&local_buff);
}

void test_efa_rma_writedata(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff local_buff;
	fi_addr_t dest_addr;
	void *desc;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	test_efa_rma_prep(resource, &dest_addr);
	efa_unit_test_buff_construct(&local_buff, resource, 4096 /* buff_size */);

	desc = fi_mr_desc(local_buff.mr);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_writedata(resource->ep, local_buff.buff, local_buff.size, desc,
			   0, dest_addr, remote_addr, remote_key,
			   NULL /* context */);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&local_buff);
}

void test_efa_rma_inject_write(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff local_buff;
	fi_addr_t dest_addr;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	test_efa_rma_prep(resource, &dest_addr);
	efa_unit_test_buff_construct(&local_buff, resource, 32 /* buff_size */);

	ret = fi_inject_write(resource->ep, local_buff.buff, local_buff.size,
			      dest_addr, remote_addr, remote_key);
	assert_int_equal(ret, -FI_ENOSYS);

	efa_unit_test_buff_destruct(&local_buff);
}

void test_efa_rma_inject_writedata(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff local_buff;
	fi_addr_t dest_addr;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	test_efa_rma_prep(resource, &dest_addr);
	efa_unit_test_buff_construct(&local_buff, resource, 32 /* buff_size */);

	ret = fi_inject_writedata(resource->ep, local_buff.buff,
				  local_buff.size, 0, dest_addr, remote_addr,
				  remote_key);
	assert_int_equal(ret, -FI_ENOSYS);

	efa_unit_test_buff_destruct(&local_buff);
}

void test_efa_rma_writemsg_with_inject(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff local_buff;
	struct iovec iov;
	struct fi_msg_rma msg = {0};
	struct fi_rma_iov rma_iov;
	fi_addr_t dest_addr;
	void *desc;
	int ret;

	test_efa_rma_prep(resource, &dest_addr);
	efa_unit_test_buff_construct(&local_buff, resource, 4096 /* buff_size */);

	iov.iov_base = local_buff.buff;
	iov.iov_len = local_buff.size;
	desc = fi_mr_desc(local_buff.mr);
	rma_iov.len = local_buff.size;
	rma_iov.addr = 0x87654321;
	rma_iov.key = 123456;
	efa_unit_test_construct_msg_rma(&msg, &iov, &desc, 1, dest_addr, &rma_iov,
					1, NULL, 0);

	ret = fi_writemsg(resource->ep, &msg, FI_INJECT);
	assert_int_equal(ret, -FI_ENOSYS);

	efa_unit_test_buff_destruct(&local_buff);
}
