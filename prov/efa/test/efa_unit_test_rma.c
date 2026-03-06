/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_unit_tests.h"
#include "ofi_util.h"

extern struct fi_ops_rma efa_rma_ops;

/**
 * @return true if device supports FI_RMA, false otherwise.
 *
 * When FI_RMA is not supported, this function verifies that fi_getinfo
 * fails with FI_RMA hints, then constructs the resource without FI_RMA
 * so callers can test the -FI_EOPNOTSUPP error path.
 */
static bool test_efa_rma_prep_with_inject_size(struct efa_resource *resource,
					       fi_addr_t *addr,
					       size_t inject_size)
{
	struct efa_ep_addr raw_addr;
	size_t raw_addr_len = sizeof(raw_addr);
	bool fi_rma_supported;
	int ret;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	assert_non_null(resource->hints);

	if (inject_size)
		resource->hints->tx_attr->inject_size = inject_size;

	fi_rma_supported = efa_device_support_rdma_read() && efa_device_support_rdma_write();

	if (fi_rma_supported) {
		resource->hints->caps |= FI_RMA;
		resource->hints->mode |= FI_RX_CQ_DATA;
	} else {
		/* Verify fi_getinfo rejects FI_RMA on non-RDMA platforms */
		struct fi_info *hints_rma, *info_rma = NULL;

		hints_rma = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
		assert_non_null(hints_rma);
		hints_rma->caps |= FI_RMA;
		hints_rma->mode |= FI_RX_CQ_DATA;
		ret = fi_getinfo(FI_VERSION(2, 0), NULL, NULL, 0ULL, hints_rma, &info_rma);
		assert_int_not_equal(ret, 0);
		fi_freeinfo(hints_rma);
	}

	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
						    FI_VERSION(2, 0),
						    resource->hints, true, true);

	/* Set up the mock operations */
	g_efa_unit_test_mocks.efa_qp_post_recv = &efa_mock_efa_qp_post_recv_return_mock;
	/* Mock general QP post functions to save work request IDs */
	g_efa_unit_test_mocks.efa_qp_post_read = &efa_mock_efa_qp_post_read_return_mock;
	g_efa_unit_test_mocks.efa_qp_post_write = &efa_mock_efa_qp_post_write_return_mock;
	will_return_int_maybe(efa_mock_efa_qp_post_read_return_mock, 0);
	will_return_int_maybe(efa_mock_efa_qp_post_write_return_mock, 0);

	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, addr, 0 /* flags */,
			   NULL /* context */);
	assert_int_equal(ret, 1);

	return fi_rma_supported;
}

static bool test_efa_rma_prep(struct efa_resource *resource, fi_addr_t *addr)
{
	return test_efa_rma_prep_with_inject_size(resource, addr, 0);
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

	bool fi_rma_supported = test_efa_rma_prep(resource, &src_addr);

	if (!fi_rma_supported) {
		ret = fi_read(resource->ep, NULL, 0, NULL, src_addr, remote_addr, remote_key, NULL);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

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

	bool fi_rma_supported = test_efa_rma_prep(resource, &src_addr);

	if (!fi_rma_supported) {
		ret = fi_readv(resource->ep, NULL, NULL, 0, src_addr, remote_addr, remote_key, NULL);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

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

	bool fi_rma_supported = test_efa_rma_prep(resource, &src_addr);

	if (!fi_rma_supported) {
		efa_unit_test_construct_msg_rma(&msg, &iov, NULL, 0, src_addr, &rma_iov, 1, NULL, 0);
		ret = fi_readmsg(resource->ep, &msg, 0);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

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

	bool fi_rma_supported = test_efa_rma_prep(resource, &dest_addr);

	if (!fi_rma_supported) {
		ret = fi_write(resource->ep, NULL, 0, NULL, dest_addr, remote_addr, remote_key, NULL);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

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

	bool fi_rma_supported = test_efa_rma_prep(resource, &dest_addr);

	if (!fi_rma_supported) {
		ret = fi_writev(resource->ep, NULL, NULL, 0, dest_addr, remote_addr, remote_key, NULL);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

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

	bool fi_rma_supported = test_efa_rma_prep(resource, &dest_addr);

	if (!fi_rma_supported) {
		efa_unit_test_construct_msg_rma(&msg, &iov, NULL, 0, dest_addr, &rma_iov, 1, NULL, 0);
		ret = fi_writemsg(resource->ep, &msg, 0);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

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

void test_efa_rma_writemsg_with_wide_wqe_inject(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff local_buff;
	struct iovec iov;
	struct fi_msg_rma msg = {0};
	struct fi_rma_iov rma_iov;
	fi_addr_t dest_addr;
	size_t inject_rma_size, sz;
	int ret;

	bool fi_rma_supported = test_efa_rma_prep_with_inject_size(resource, &dest_addr, 42);

	if (!fi_rma_supported)
		skip();

	sz = sizeof inject_rma_size;
	ret = fi_getopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_INJECT_RMA_SIZE,
			&inject_rma_size, &sz);
	if (ret || !inject_rma_size) {
		/* Firmware does not support inline RDMA write */
		skip();
	}
	efa_unit_test_buff_construct(&local_buff, resource, 32 /* buff_size */);

	iov.iov_base = local_buff.buff;
	iov.iov_len = local_buff.size;
	rma_iov.len = local_buff.size;
	rma_iov.addr = 0x87654321;
	rma_iov.key = 123456;
	efa_unit_test_construct_msg_rma(&msg, &iov, NULL, 1, dest_addr, &rma_iov,
					1, NULL, 0);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_writemsg(resource->ep, &msg, FI_INJECT);
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

	bool fi_rma_supported = test_efa_rma_prep(resource, &dest_addr);

	if (!fi_rma_supported) {
		ret = fi_writedata(resource->ep, NULL, 0, NULL, 0, dest_addr, remote_addr, remote_key, NULL);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

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
	size_t inject_rma_size, sz;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	bool fi_rma_supported = test_efa_rma_prep_with_inject_size(resource, &dest_addr, 42);

	if (!fi_rma_supported) {
		ret = fi_inject_write(resource->ep, NULL, 0, dest_addr, remote_addr, remote_key);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

	sz = sizeof inject_rma_size;
	ret = fi_getopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_INJECT_RMA_SIZE,
			&inject_rma_size, &sz);
	if (ret || !inject_rma_size) {
		/* Firmware does not support inline RDMA write */
		skip();
	}

	efa_unit_test_buff_construct(&local_buff, resource, 32 /* buff_size */);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_inject_write(resource->ep, local_buff.buff, local_buff.size,
			      dest_addr, remote_addr, remote_key);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&local_buff);
}

void test_efa_rma_inject_writedata(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff local_buff;
	fi_addr_t dest_addr;
	size_t inject_rma_size, sz;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	bool fi_rma_supported = test_efa_rma_prep_with_inject_size(resource, &dest_addr, 42);

	if (!fi_rma_supported) {
		ret = fi_inject_writedata(resource->ep, NULL, 0, 0, dest_addr, remote_addr, remote_key);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

	sz = sizeof inject_rma_size;
	ret = fi_getopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_INJECT_RMA_SIZE,
			&inject_rma_size, &sz);
	if (ret || !inject_rma_size) {
		/* Firmware does not support inline RDMA write */
		skip();
	}

	efa_unit_test_buff_construct(&local_buff, resource, 32 /* buff_size */);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_inject_writedata(resource->ep, local_buff.buff,
				  local_buff.size, 0, dest_addr, remote_addr,
				  remote_key);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

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
	size_t inject_rma_size, sz;
	int ret;

	bool fi_rma_supported = test_efa_rma_prep_with_inject_size(resource, &dest_addr, 42);

	if (!fi_rma_supported) {
		efa_unit_test_construct_msg_rma(&msg, &iov, NULL, 0, dest_addr, &rma_iov, 1, NULL, 0);
		ret = fi_writemsg(resource->ep, &msg, FI_INJECT);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

	sz = sizeof inject_rma_size;
	ret = fi_getopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_INJECT_RMA_SIZE,
			&inject_rma_size, &sz);
	if (ret || !inject_rma_size) {
		/* Firmware does not support inline RDMA write */
		skip();
	}

	efa_unit_test_buff_construct(&local_buff, resource, 32 /* buff_size */);

	iov.iov_base = local_buff.buff;
	iov.iov_len = local_buff.size;
	rma_iov.len = local_buff.size;
	rma_iov.addr = 0x87654321;
	rma_iov.key = 123456;
	efa_unit_test_construct_msg_rma(&msg, &iov, NULL, 1, dest_addr, &rma_iov,
					1, NULL, 0);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_writemsg(resource->ep, &msg, FI_INJECT);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&local_buff);
}

/* 0-byte RMA tests - require device FI_RMA support (RDMA read + write) for bounce buffer */
void test_efa_rma_read_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t src_addr;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	bool fi_rma_supported = test_efa_rma_prep(resource, &src_addr);

	if (!fi_rma_supported) {
		ret = fi_read(resource->ep, NULL, 0, NULL, src_addr, remote_addr, remote_key, NULL);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_read(resource->ep, NULL, 0, NULL, src_addr, remote_addr, remote_key, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

void test_efa_rma_readv_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct iovec iov = {0};
	fi_addr_t src_addr;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	bool fi_rma_supported = test_efa_rma_prep(resource, &src_addr);

	if (!fi_rma_supported) {
		ret = fi_readv(resource->ep, &iov, NULL, 0, src_addr, remote_addr, remote_key, NULL);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_readv(resource->ep, &iov, NULL, 0, src_addr, remote_addr, remote_key, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

void test_efa_rma_readmsg_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct iovec iov = {0};
	struct fi_msg_rma msg = {0};
	struct fi_rma_iov rma_iov = {0};
	fi_addr_t src_addr;
	int ret;

	bool fi_rma_supported = test_efa_rma_prep(resource, &src_addr);

	if (!fi_rma_supported) {
		efa_unit_test_construct_msg_rma(&msg, &iov, NULL, 0, src_addr, &rma_iov, 1, NULL, 0);
		ret = fi_readmsg(resource->ep, &msg, 0);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

	rma_iov.addr = 0x87654321;
	rma_iov.key = 123456;
	efa_unit_test_construct_msg_rma(&msg, &iov, NULL, 0, src_addr, &rma_iov, 1, NULL, 0);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_readmsg(resource->ep, &msg, 0);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

void test_efa_rma_write_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t dest_addr;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	bool fi_rma_supported = test_efa_rma_prep(resource, &dest_addr);

	if (!fi_rma_supported) {
		ret = fi_write(resource->ep, NULL, 0, NULL, dest_addr, remote_addr, remote_key, NULL);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_write(resource->ep, NULL, 0, NULL, dest_addr, remote_addr, remote_key, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

void test_efa_rma_writev_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct iovec iov = {0};
	fi_addr_t dest_addr;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	bool fi_rma_supported = test_efa_rma_prep(resource, &dest_addr);

	if (!fi_rma_supported) {
		ret = fi_writev(resource->ep, &iov, NULL, 0, dest_addr, remote_addr, remote_key, NULL);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_writev(resource->ep, &iov, NULL, 0, dest_addr, remote_addr, remote_key, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

void test_efa_rma_writemsg_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct iovec iov = {0};
	struct fi_msg_rma msg = {0};
	struct fi_rma_iov rma_iov = {0};
	fi_addr_t dest_addr;
	int ret;

	bool fi_rma_supported = test_efa_rma_prep(resource, &dest_addr);

	if (!fi_rma_supported) {
		efa_unit_test_construct_msg_rma(&msg, &iov, NULL, 0, dest_addr, &rma_iov, 1, NULL, 0);
		ret = fi_writemsg(resource->ep, &msg, 0);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

	rma_iov.addr = 0x87654321;
	rma_iov.key = 123456;
	efa_unit_test_construct_msg_rma(&msg, &iov, NULL, 0, dest_addr, &rma_iov, 1, NULL, 0);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_writemsg(resource->ep, &msg, 0);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

void test_efa_rma_writedata_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t dest_addr;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	bool fi_rma_supported = test_efa_rma_prep(resource, &dest_addr);

	if (!fi_rma_supported) {
		ret = fi_writedata(resource->ep, NULL, 0, NULL, 0, dest_addr, remote_addr, remote_key, NULL);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_writedata(resource->ep, NULL, 0, NULL, 0, dest_addr, remote_addr, remote_key, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

void test_efa_rma_inject_write_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t dest_addr;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	bool fi_rma_supported = test_efa_rma_prep(resource, &dest_addr);

	if (!fi_rma_supported) {
		ret = fi_inject_write(resource->ep, NULL, 0, dest_addr, remote_addr, remote_key);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_inject_write(resource->ep, NULL, 0, dest_addr, remote_addr, remote_key);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

void test_efa_rma_inject_writedata_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t dest_addr;
	int ret;
	uint64_t remote_addr = 0x87654321;
	uint64_t remote_key = 123456;

	bool fi_rma_supported = test_efa_rma_prep(resource, &dest_addr);

	if (!fi_rma_supported) {
		ret = fi_inject_writedata(resource->ep, NULL, 0, 0, dest_addr, remote_addr, remote_key);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_inject_writedata(resource->ep, NULL, 0, 0, dest_addr, remote_addr, remote_key);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

void test_efa_rma_write_0_byte_with_inject_flag(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct iovec iov = {0};
	struct fi_msg_rma msg = {0};
	struct fi_rma_iov rma_iov = {0};
	fi_addr_t dest_addr;
	int ret;

	bool fi_rma_supported = test_efa_rma_prep(resource, &dest_addr);

	if (!fi_rma_supported) {
		efa_unit_test_construct_msg_rma(&msg, &iov, NULL, 0, dest_addr, &rma_iov, 1, NULL, 0);
		ret = fi_writemsg(resource->ep, &msg, FI_INJECT);
		assert_int_equal(ret, -FI_EOPNOTSUPP);
		return;
	}

	rma_iov.addr = 0x87654321;
	rma_iov.key = 123456;
	efa_unit_test_construct_msg_rma(&msg, &iov, NULL, 0, dest_addr, &rma_iov, 1, NULL, 0);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_writemsg(resource->ep, &msg, FI_INJECT);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

