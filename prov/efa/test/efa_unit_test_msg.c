/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_unit_tests.h"
#include "ofi_util.h"
#include "efa_rdm_msg.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_protocol.h"


static void test_efa_msg_recv_prep(struct efa_resource *resource,
				   fi_addr_t *addr)
{
	struct efa_ep_addr raw_addr;
	size_t raw_addr_len = sizeof(raw_addr);
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	g_efa_unit_test_mocks.efa_qp_post_recv = &efa_mock_efa_qp_post_recv_return_mock;
	will_return(efa_mock_efa_qp_post_recv_return_mock, 0);

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
	struct efa_ep_addr raw_addr;
	size_t raw_addr_len = sizeof(raw_addr);
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, addr, 0 /* flags */,
			   NULL /* context */);
	assert_int_equal(ret, 1);

	g_efa_unit_test_mocks.efa_qp_post_recv = &efa_mock_efa_qp_post_recv_return_mock;
	/* Mock general QP post send function to save work request IDs */
	g_efa_unit_test_mocks.efa_qp_post_send = &efa_mock_efa_qp_post_send_return_mock;
	will_return_int_maybe(efa_mock_efa_qp_post_send_return_mock, 0);
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

/* 0-byte MSG tests */
void test_efa_msg_send_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	test_efa_msg_send_prep(resource, &addr);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_send(resource->ep, NULL, 0, NULL, addr, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

void test_efa_msg_sendv_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	struct iovec iov = {0};
	int ret;

	test_efa_msg_send_prep(resource, &addr);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_sendv(resource->ep, &iov, NULL, 0, addr, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

void test_efa_msg_sendmsg_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	struct iovec iov = {0};
	struct fi_msg msg = {0};
	int ret;

	test_efa_msg_send_prep(resource, &addr);

	efa_unit_test_construct_msg(&msg, &iov, 0, addr, NULL, 0, NULL);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_sendmsg(resource->ep, &msg, 0);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

void test_efa_msg_senddata_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;
	uint64_t data = 0x1234567890ABCDEF;

	test_efa_msg_send_prep(resource, &addr);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_senddata(resource->ep, NULL, 0, NULL, data, addr, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

void test_efa_msg_inject_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	test_efa_msg_send_prep(resource, &addr);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_inject(resource->ep, NULL, 0, addr);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

void test_efa_msg_injectdata_0_byte(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;
	uint64_t data = 0x1234567890ABCDEF;

	test_efa_msg_send_prep(resource, &addr);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_injectdata(resource->ep, NULL, 0, data, addr);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

void test_efa_msg_send_0_byte_with_inject_flag(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct iovec iov = {0};
	struct fi_msg msg = {0};
	fi_addr_t addr;
	int ret;

	test_efa_msg_send_prep(resource, &addr);

	efa_unit_test_construct_msg(&msg, &iov, 0, addr, NULL, 0, NULL);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_sendmsg(resource->ep, &msg, FI_INJECT);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
}

/**
 * @brief Verify that fi_sendmsg with FI_INJECT and HMEM desc returns -FI_EOPNOTSUPP
 */
void test_efa_msg_sendmsg_inject_with_hmem_fails(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_mr *efa_mr;
	struct iovec iov;
	void *desc;
	int ret;
	fi_addr_t addr;
	struct fi_msg msg = {0};

	test_efa_msg_send_prep(resource, &addr);
	efa_unit_test_buff_construct(&send_buff, resource, 32);

	desc = fi_mr_desc(send_buff.mr);
	efa_mr = (struct efa_mr *)desc;
	efa_mr->iface = FI_HMEM_CUDA;

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;

	efa_unit_test_construct_msg(&msg, &iov, 1, addr, NULL, 0, &desc);

	ret = fi_sendmsg(resource->ep, &msg, FI_INJECT);
	assert_int_equal(ret, -FI_EOPNOTSUPP);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);

	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Verify that a multi-iov send where only a non-first desc is HMEM
 * does not use the inline path. This catches the bug where only desc[0]
 * was checked for HMEM.
 */
void test_efa_msg_sendmsg_multi_iov_second_desc_hmem_fails(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff buff0, buff1;
	struct efa_mr *efa_mr;
	struct iovec iov[2];
	void *desc[2];
	int ret;
	fi_addr_t addr;
	struct fi_msg msg = {0};

	test_efa_msg_send_prep(resource, &addr);

	/* Override the post_send mock to verify use_inline is false */
	g_efa_unit_test_mocks.efa_qp_post_send = &efa_mock_efa_qp_post_send_verify_not_inline;
	will_return(efa_mock_efa_qp_post_send_verify_not_inline, 0);

	efa_unit_test_buff_construct(&buff0, resource, 16);
	efa_unit_test_buff_construct(&buff1, resource, 16);

	desc[0] = fi_mr_desc(buff0.mr);
	desc[1] = fi_mr_desc(buff1.mr);
	/* Only mark the second desc as HMEM */
	efa_mr = (struct efa_mr *)desc[1];
	efa_mr->iface = FI_HMEM_CUDA;

	iov[0].iov_base = buff0.buff;
	iov[0].iov_len = buff0.size;
	iov[1].iov_base = buff1.buff;
	iov[1].iov_len = buff1.size;

	efa_unit_test_construct_msg(&msg, iov, 2, addr, NULL, 0, desc);

	ret = fi_sendmsg(resource->ep, &msg, 0);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	efa_unit_test_buff_destruct(&buff1);
	efa_unit_test_buff_destruct(&buff0);
}

/**
 * @brief Verify that fi_sendmsg with FI_INJECT and message larger than
 * inline_buf_size returns -FI_EOPNOTSUPP
 */
void test_efa_msg_sendmsg_inject_with_large_msg_fails(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_base_ep *base_ep;
	struct iovec iov;
	void *desc;
	int ret;
	fi_addr_t addr;
	struct fi_msg msg = {0};
	size_t buf_size;

	test_efa_msg_send_prep(resource, &addr);

	base_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);
	buf_size = base_ep->domain->device->efa_attr.inline_buf_size +
		   base_ep->info->ep_attr->msg_prefix_size + 1;

	efa_unit_test_buff_construct(&send_buff, resource, buf_size);

	desc = fi_mr_desc(send_buff.mr);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;

	efa_unit_test_construct_msg(&msg, &iov, 1, addr, NULL, 0, &desc);

	ret = fi_sendmsg(resource->ep, &msg, FI_INJECT);
	assert_int_equal(ret, -FI_EOPNOTSUPP);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);

	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Verify that fi_inject with message larger than
 * inline_buf_size returns -FI_EOPNOTSUPP
 */
void test_efa_msg_inject_with_large_msg_fails(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_base_ep *base_ep;
	struct efa_unit_test_buff send_buff;
	fi_addr_t addr;
	size_t buf_size;
	int ret;

	test_efa_msg_send_prep(resource, &addr);
	efa_unit_test_buff_construct(&send_buff, resource, 32);
	base_ep = container_of(resource->ep, struct efa_base_ep, util_ep.ep_fid);
	buf_size = base_ep->domain->device->efa_attr.inline_buf_size +
		   base_ep->info->ep_attr->msg_prefix_size + 1;

	ret = fi_inject(resource->ep, send_buff.buff, buf_size, addr);
	assert_int_equal(ret, -FI_EOPNOTSUPP);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);

	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Test efa_rdm_msg_get_tx_flags correctly merges per-op flags with
 * endpoint tx_op_flags and respects tx_msg_flags for FI_COMPLETION.
 */
void test_efa_rdm_msg_get_tx_flags(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	uint64_t result;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	/* DC with completion: both DC and COMPLETION should be present */
	ep->base_ep.util_ep.tx_op_flags = FI_COMPLETION | FI_DELIVERY_COMPLETE;
	ep->base_ep.util_ep.tx_msg_flags = FI_COMPLETION;
	result = efa_rdm_msg_get_tx_flags(ep, 0);
	assert_true(result & FI_DELIVERY_COMPLETE);
	assert_true(result & FI_COMPLETION);

	/* Selective completion (tx_msg_flags=0): DC present, COMPLETION stripped */
	ep->base_ep.util_ep.tx_msg_flags = 0;
	result = efa_rdm_msg_get_tx_flags(ep, 0);
	assert_true(result & FI_DELIVERY_COMPLETE);
	assert_false(result & FI_COMPLETION);

	/* Per-op flags merged with ep flags */
	ep->base_ep.util_ep.tx_op_flags = FI_DELIVERY_COMPLETE;
	ep->base_ep.util_ep.tx_msg_flags = FI_COMPLETION;
	result = efa_rdm_msg_get_tx_flags(ep, FI_INJECT);
	assert_true(result & FI_INJECT);
	assert_true(result & FI_DELIVERY_COMPLETE);
}

/**
 * @brief Test that sending with FI_DELIVERY_COMPLETE in tx_op_flags produces
 * a DC eager packet type.
 */
void test_efa_rdm_msg_send_dc_eager_pkt_type(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_base_hdr *base_hdr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t addr;
	int ret;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_unit_test_buff_construct(&send_buff, resource, 64);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL), 1);

	ep->base_ep.util_ep.tx_op_flags = FI_COMPLETION | FI_DELIVERY_COMPLETE;
	ep->base_ep.util_ep.tx_msg_flags = FI_COMPLETION;

	/* Mark peer as handshake received so DC send doesn't queue */
	peer = efa_rdm_ep_get_peer(ep, addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_DELIVERY_COMPLETE;

	g_efa_unit_test_mocks.efa_qp_post_send = &efa_mock_efa_qp_post_send_return_mock;
	will_return(efa_mock_efa_qp_post_send_return_mock, 0);

	ret = fi_send(resource->ep, send_buff.buff, send_buff.size,
		      fi_mr_desc(send_buff.mr), addr, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	pkt_entry = efa_rdm_cq_get_pke_from_wr_id_solicited(
		(uint64_t)g_ibv_submitted_wr_id_vec[0]);
	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	assert_int_equal(base_hdr->type, EFA_RDM_DC_EAGER_MSGRTM_PKT);
	assert_true(pkt_entry->ope->fi_flags & FI_DELIVERY_COMPLETE);

	efa_rdm_txe_release(pkt_entry->ope);
	efa_rdm_pke_release_tx(pkt_entry);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Test that binding CQ with FI_SELECTIVE_COMPLETION sets tx_msg_flags=0
 * and that a send without FI_COMPLETION flag does not set FI_COMPLETION in the
 * ope
 */
void test_efa_rdm_msg_send_selective_completion(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_ep_addr raw_addr;
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t addr;
	struct fi_cq_attr cq_attr = { .format = FI_CQ_FORMAT_DATA };
	int ret;

	/* Construct resource without CQ and EP not enabled */
	efa_unit_test_resource_construct_no_cq_and_ep_not_enabled(
		resource, FI_EP_RDM, EFA_FABRIC_NAME);

	/* Open CQ and bind with FI_SELECTIVE_COMPLETION */
	ret = fi_cq_open(resource->domain, &cq_attr, &resource->cq, NULL);
	assert_int_equal(ret, 0);
	ret = fi_ep_bind(resource->ep, &resource->cq->fid,
			 FI_SEND | FI_RECV | FI_SELECTIVE_COMPLETION);
	assert_int_equal(ret, 0);

	g_efa_unit_test_mocks.efa_qp_post_recv = &efa_mock_efa_qp_post_recv_return_mock;

	/* Disable SHM so send goes through EFA path */
	bool shm_permitted = false;
	ret = fi_setopt(&resource->ep->fid, FI_OPT_ENDPOINT,
			FI_OPT_SHARED_MEMORY_PERMITTED, &shm_permitted,
			sizeof(shm_permitted));
	assert_int_equal(ret, 0);

	ret = fi_enable(resource->ep);
	assert_int_equal(ret, 0);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	/* Verify tx_msg_flags is 0 (selective completion) */
	assert_int_equal(ep->base_ep.util_ep.tx_msg_flags, 0);
	/* tx_op_flags should not have FI_COMPLETION */
	assert_false(ep->base_ep.util_ep.tx_op_flags & FI_COMPLETION);

	/* Set up peer address */
	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL);
	assert_int_equal(ret, 1);

	g_efa_unit_test_mocks.efa_qp_post_send = &efa_mock_efa_qp_post_send_return_mock;
	will_return(efa_mock_efa_qp_post_send_return_mock, 0);

	efa_unit_test_buff_construct(&send_buff, resource, 64);

	/* Send without FI_COMPLETION flag */
	ret = fi_send(resource->ep, send_buff.buff, send_buff.size,
		      fi_mr_desc(send_buff.mr), addr, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	pkt_entry = efa_rdm_cq_get_pke_from_wr_id_solicited(
		(uint64_t)g_ibv_submitted_wr_id_vec[0]);
	assert_false(pkt_entry->ope->fi_flags & FI_COMPLETION);

	efa_rdm_txe_release(pkt_entry->ope);
	efa_rdm_pke_release_tx(pkt_entry);
	efa_unit_test_buff_destruct(&send_buff);
}
