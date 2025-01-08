/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_unit_tests.h"
#include "ofi_util.h"


static void test_efa_msg_recv_prep(struct efa_resource *resource,
				   fi_addr_t *addr)
{
	struct ibv_qp *ibv_qp;
	struct efa_ep_addr raw_addr;
	struct efa_base_ep *base_ep;
	size_t raw_addr_len = sizeof(raw_addr);
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_PROV_NAME);

	base_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);
	ibv_qp = base_ep->qp->ibv_qp;
	ibv_qp->context->ops.post_recv = &efa_mock_ibv_post_recv;
	will_return(efa_mock_ibv_post_recv, 0);

	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, addr, 0 /* flags */,
			   NULL /* context */);
	assert_int_equal(ret, 1);
}

void test_efa_msg_fi_recv(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	fi_addr_t addr;
	int ret;
	void *desc;

	test_efa_msg_recv_prep(resource, &addr);
	efa_unit_test_buff_construct(&send_buff, resource, 4096 /* buff_size */);

	desc = fi_mr_desc(send_buff.mr);

	ret = fi_recv(resource->ep, send_buff.buff, send_buff.size, desc, addr,
		      NULL /* context */);
	assert_int_equal(ret, 0);

	efa_unit_test_buff_destruct(&send_buff);
}

void test_efa_msg_fi_recvv(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct iovec iov;
	fi_addr_t addr;
	int ret;
	void *desc;

	test_efa_msg_recv_prep(resource, &addr);
	efa_unit_test_buff_construct(&send_buff, resource, 4096 /* buff_size */);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	desc = fi_mr_desc(send_buff.mr);

	ret = fi_recvv(resource->ep, &iov, &desc, 1, addr, NULL /* context */);
	assert_int_equal(ret, 0);

	efa_unit_test_buff_destruct(&send_buff);
}

void test_efa_msg_fi_recvmsg(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	fi_addr_t addr;
	int ret;
	void *desc;
	struct iovec iov;
	struct fi_msg msg = {0};

	test_efa_msg_recv_prep(resource, &addr);
	efa_unit_test_buff_construct(&send_buff, resource, 4096 /* buff_size */);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	desc = fi_mr_desc(send_buff.mr);
	efa_unit_test_construct_msg(&msg, &iov, 1, addr, NULL, 0, &desc);

	ret = fi_recvmsg(resource->ep, &msg, 0);
	assert_int_equal(ret, 0);

	efa_unit_test_buff_destruct(&send_buff);
}

static void test_efa_msg_send_prep(struct efa_resource *resource,
				   fi_addr_t *addr)
{
	struct ibv_qp_ex *ibv_qpx;
	struct efa_ep_addr raw_addr;
	struct efa_base_ep *base_ep;
	size_t raw_addr_len = sizeof(raw_addr);
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_PROV_NAME);

	base_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);
	ibv_qpx = base_ep->qp->ibv_qp_ex;

	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, addr, 0 /* flags */,
			   NULL /* context */);
	assert_int_equal(ret, 1);

	ibv_qpx->wr_start = &efa_mock_ibv_wr_start_no_op;
	/* this mock will save the send work request (wr) in a global list */
	ibv_qpx->wr_send = &efa_mock_ibv_wr_send_save_wr;
	ibv_qpx->wr_send_imm = &efa_mock_ibv_wr_send_imm_save_wr;
	ibv_qpx->wr_set_inline_data_list = &efa_mock_ibv_wr_set_inline_data_list_no_op;
	ibv_qpx->wr_set_sge_list = &efa_mock_ibv_wr_set_sge_list_no_op;
	ibv_qpx->wr_set_ud_addr = &efa_mock_ibv_wr_set_ud_addr_no_op;
	ibv_qpx->wr_complete = &efa_mock_ibv_wr_complete_no_op;
}

void test_efa_msg_fi_send(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	fi_addr_t addr;
	void *desc;
	int ret;

	test_efa_msg_send_prep(resource, &addr);
	efa_unit_test_buff_construct(&send_buff, resource, 4096 /* buff_size */);

	desc = fi_mr_desc(send_buff.mr);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_send(resource->ep, send_buff.buff, send_buff.size, desc, addr,
		      NULL /* context */);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&send_buff);
}

void test_efa_msg_fi_sendv(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	fi_addr_t addr;
	struct iovec iov;
	void *desc;
	int ret;

	test_efa_msg_send_prep(resource, &addr);
	efa_unit_test_buff_construct(&send_buff, resource, 4096 /* buff_size */);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	desc = fi_mr_desc(send_buff.mr);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_sendv(resource->ep, &iov, &desc, 1, addr, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&send_buff);
}

void test_efa_msg_fi_sendmsg(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	fi_addr_t addr;
	struct iovec iov;
	void *desc;
	int ret;
	struct fi_msg msg = {0};

	test_efa_msg_send_prep(resource, &addr);
	efa_unit_test_buff_construct(&send_buff, resource, 4096 /* buff_size */);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	desc = fi_mr_desc(send_buff.mr);

	efa_unit_test_construct_msg(&msg, &iov, 1, addr, NULL, 0, &desc);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_sendmsg(resource->ep, &msg, 0);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&send_buff);
}

void test_efa_msg_fi_senddata(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	fi_addr_t addr;
	void *desc;
	int ret;
	uint64_t data = 0x1234567890ABCDEF;

	test_efa_msg_send_prep(resource, &addr);
	efa_unit_test_buff_construct(&send_buff, resource, 4096 /* buff_size */);

	desc = fi_mr_desc(send_buff.mr);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_senddata(resource->ep, send_buff.buff, send_buff.size, desc,
			  data, addr, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&send_buff);
}

void test_efa_msg_fi_inject(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	fi_addr_t addr;
	int ret;

	test_efa_msg_send_prep(resource, &addr);
	efa_unit_test_buff_construct(&send_buff, resource, 32);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_inject(resource->ep, send_buff.buff, send_buff.size, addr);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&send_buff);
}

void test_efa_msg_fi_injectdata(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	fi_addr_t addr;
	int ret;
	uint64_t data = 0x1234567890ABCDEF;

	test_efa_msg_send_prep(resource, &addr);
	efa_unit_test_buff_construct(&send_buff, resource, 32);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_injectdata(resource->ep, send_buff.buff, send_buff.size, data,
			    addr);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&send_buff);
}
