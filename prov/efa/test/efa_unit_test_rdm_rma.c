/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_unit_tests.h"
#include "efa_rdm_rma.h"
#include "efa_rdm_pke_nonreq.h"
#include "efa_rdm_cq.h"

static bool test_efa_rdm_rma_should_write_using_rdma_helper(
		struct efa_rdm_ep *ep, struct efa_rdm_peer *peer,
		struct efa_rdm_ope *txe, bool use_device_rdma,
		bool homogeneous_peers, bool use_p2p,
		bool use_unsolicited_write_recv)
{
	uint32_t efa_device_caps_orig;
	bool result;

	efa_device_caps_orig = g_efa_selected_device_list[0].device_caps;
	g_efa_selected_device_list[0].device_caps |= EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE;

	ep->use_device_rdma = use_device_rdma;
	ep->homogeneous_peers = homogeneous_peers;
	if (use_device_rdma)
		ep->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_RDMA_WRITE;
	if (use_unsolicited_write_recv)
		ep->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_UNSOLICITED_WRITE_RECV;

	result = efa_rdm_rma_should_write_using_rdma(ep, txe, peer, use_p2p);

	g_efa_selected_device_list[0].device_caps = efa_device_caps_orig;
	return result;
}

/**
 * @brief Test that FI_REMOTE_CQ_DATA with multiple IOVs returns false
 *
 * When FI_REMOTE_CQ_DATA is set and iov_count > 1, the function should
 * return false to use send emulation instead of RDMA write.
 */
void test_efa_rdm_rma_should_write_using_rdma_remote_cq_data_multiple_iovs_returns_false(
		void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_peer peer = {0};
	struct efa_rdm_ope txe = {0};
	bool result;
	struct efa_rdm_ep *ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid.fid);

	peer.flags = EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer.extra_info[0] = EFA_RDM_EXTRA_FEATURE_RDMA_WRITE;
	txe.fi_flags = FI_REMOTE_CQ_DATA;
	txe.iov_count = 2;
	txe.rma_iov_count = 1;

	result = test_efa_rdm_rma_should_write_using_rdma_helper(
			ep, &peer, &txe, true, true, true, true);
	assert_false(result);
}

/**
 * @brief Test that FI_REMOTE_CQ_DATA with multiple RMA IOVs returns false
 *
 * When FI_REMOTE_CQ_DATA is set and rma_iov_count > 1, the function should
 * return false to use send emulation instead of RDMA write.
 */
void test_efa_rdm_rma_should_write_using_rdma_remote_cq_data_multiple_rma_iovs_returns_false(
		void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_peer peer = {0};
	struct efa_rdm_ope txe = {0};
	bool result;
	struct efa_rdm_ep *ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid.fid);

	peer.flags = EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer.extra_info[0] = EFA_RDM_EXTRA_FEATURE_RDMA_WRITE;
	txe.fi_flags = FI_REMOTE_CQ_DATA;
	txe.iov_count = 1;
	txe.rma_iov_count = 2;

	result = test_efa_rdm_rma_should_write_using_rdma_helper(
			ep, &peer, &txe, true, true, true, true);
	assert_false(result);
}

/**
 * @brief Test that use_device_rdma false returns false
 *
 * When use_device_rdma is false, the function should return false
 * regardless of other conditions.
 */
void test_efa_rdm_rma_should_write_using_rdma_use_device_rdma_false_returns_false(
		void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_peer peer = {0};
	struct efa_rdm_ope txe = {0};
	bool result;
	struct efa_rdm_ep *ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid.fid);

	peer.flags = EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer.extra_info[0] = EFA_RDM_EXTRA_FEATURE_RDMA_WRITE;
	txe.fi_flags = 0;
	txe.iov_count = 1;
	txe.rma_iov_count = 1;

	result = test_efa_rdm_rma_should_write_using_rdma_helper(
			ep, &peer, &txe, false, true, true, true);
	assert_false(result);
}

/**
 * @brief Test that peer without RDMA write support returns false
 *
 * When peer doesn't support RDMA write, the function should return false
 * even when other conditions are favorable.
 */
void test_efa_rdm_rma_should_write_using_rdma_peer_no_rdma_write_support_returns_false(
		void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_peer peer = {0};
	struct efa_rdm_ope txe = {0};
	bool result;
	struct efa_rdm_ep *ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid.fid);

	peer.flags = EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer.extra_info[0] = 0; // No RDMA write support
	txe.fi_flags = 0;
	txe.iov_count = 1;
	txe.rma_iov_count = 1;

	result = test_efa_rdm_rma_should_write_using_rdma_helper(
			ep, &peer, &txe, true, false, true, true);
	assert_false(result);
}

/**
 * @brief Test that lack of P2P support returns false
 *
 * When use_p2p is false, the function should return false regardless
 * of RDMA write support status.
 */
void test_efa_rdm_rma_should_write_using_rdma_no_p2p_support_returns_false(
		void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_peer peer = {0};
	struct efa_rdm_ope txe = {0};
	bool result;
	struct efa_rdm_ep *ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid.fid);

	peer.flags = EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer.extra_info[0] = EFA_RDM_EXTRA_FEATURE_RDMA_WRITE;
	txe.fi_flags = 0;
	txe.iov_count = 1;
	txe.rma_iov_count = 1;

	result = test_efa_rdm_rma_should_write_using_rdma_helper(
			ep, &peer, &txe, true, true, false, true);
	assert_false(result);
}

/**
 * @brief Test that P2P and RDMA write support returns true
 *
 * When use_p2p is true and both endpoint and peer support RDMA write,
 * the function should return true to use RDMA write.
 */
void test_efa_rdm_rma_should_write_using_rdma_p2p_and_rdma_write_support_returns_true(
		void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_peer peer = {0};
	struct efa_rdm_ope txe = {0};
	bool result;
	struct efa_rdm_ep *ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid.fid);

	peer.flags = EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer.extra_info[0] = EFA_RDM_EXTRA_FEATURE_RDMA_WRITE;
	txe.fi_flags = 0;
	txe.iov_count = 1;
	txe.rma_iov_count = 1;

	result = test_efa_rdm_rma_should_write_using_rdma_helper(
			ep, &peer, &txe, true, true, true, true);
	assert_true(result);
}

/**
 * @brief Test that FI_REMOTE_CQ_DATA with single IOVs and RDMA support returns true
 *
 * When FI_REMOTE_CQ_DATA is set with single IOV/RMA IOV and both P2P and
 * RDMA write are supported, the function should return true.
 */
void test_efa_rdm_rma_should_write_using_rdma_remote_cq_data_single_iovs_with_rdma_support(
		void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_peer peer = {0};
	struct efa_rdm_ope txe = {0};
	bool result;
	struct efa_rdm_ep *ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid.fid);

	peer.flags = EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer.extra_info[0] = EFA_RDM_EXTRA_FEATURE_RDMA_WRITE | EFA_RDM_EXTRA_FEATURE_UNSOLICITED_WRITE_RECV;
	txe.fi_flags = FI_REMOTE_CQ_DATA;
	txe.iov_count = 1;
	txe.rma_iov_count = 1;

	result = test_efa_rdm_rma_should_write_using_rdma_helper(
			ep, &peer, &txe, true, true, true, true);
	assert_true(result);
}

/**
 * @brief Test that FI_REMOTE_CQ_DATA with peer w/o unsolicited write recv return false
 *
 * When FI_REMOTE_CQ_DATA is set with single IOV/RMA IOV and both P2P and
 * RDMA write are supported, but peer doesn't support unsolicited write recv,
 * the function should return false.
 */
void test_efa_rdm_rma_should_write_using_rdma_unsolicited_write_recv_not_match(
		void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope txe = {0};
	bool result;
	struct efa_rdm_ep *ep;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	int err, num_addr;
	fi_addr_t peer_addr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid.fid);
	/* create a fake peer */
	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL);
	assert_int_equal(num_addr, 1);

	/*
	 * Fake a peer that has made handshake and
	 * does not support unsolicited write recv
	 */
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_RDMA_WRITE;
	peer->extra_info[0] &= ~EFA_RDM_EXTRA_FEATURE_UNSOLICITED_WRITE_RECV;
	/* make sure shm is not used */
	peer->is_local = false;

	txe.fi_flags = FI_REMOTE_CQ_DATA;
	txe.iov_count = 1;
	txe.rma_iov_count = 1;

	result = test_efa_rdm_rma_should_write_using_rdma_helper(
			ep, peer, &txe, true, false, true, true);
	assert_false(result);
}

/* RDM 0-byte operation tests */
void test_efa_rdm_rma_read_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	assert_int_equal(efa_rdm_ep->efa_outstanding_tx_ops, 0);
	ret = fi_read(resource->ep, NULL, 0, NULL, addr, 0x87654321, 123456, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(efa_rdm_ep->efa_outstanding_tx_ops, 1);
	
}

void test_efa_rdm_rma_readv_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	struct iovec iov = {0};
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_readv(resource->ep, &iov, NULL, 0, addr, 0x87654321, 123456, NULL);
	assert_int_equal(ret, 0);
	
}

void test_efa_rdm_rma_readmsg_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	struct iovec iov = {0};
	struct fi_msg_rma msg = {0};
	struct fi_rma_iov rma_iov = {.addr = 0x87654321, .len = 0, .key = 123456};
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	efa_unit_test_construct_msg_rma(&msg, &iov, NULL, 0, addr, &rma_iov, 1, NULL, 0);

	ret = fi_readmsg(resource->ep, &msg, 0);
	assert_int_equal(ret, 0);
	
}

void test_efa_rdm_rma_write_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	assert_int_equal(efa_rdm_ep->efa_outstanding_tx_ops, 0);
	ret = fi_write(resource->ep, NULL, 0, NULL, addr, 0x87654321, 123456, NULL);
	assert_int_equal(ret, 0);
	assert_int_equal(efa_rdm_ep->efa_outstanding_tx_ops, 1);
}

void test_efa_rdm_rma_writev_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	struct iovec iov = {0};
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_writev(resource->ep, &iov, NULL, 0, addr, 0x87654321, 123456, NULL);
	assert_int_equal(ret, 0);
	
}

void test_efa_rdm_rma_writemsg_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	struct iovec iov = {0};
	struct fi_msg_rma msg = {0};
	struct fi_rma_iov rma_iov = {.addr = 0x87654321, .len = 0, .key = 123456};
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	efa_unit_test_construct_msg_rma(&msg, &iov, NULL, 0, addr, &rma_iov, 1, NULL, 0);

	ret = fi_writemsg(resource->ep, &msg, 0);
	assert_int_equal(ret, 0);
	
}

void test_efa_rdm_rma_writedata_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_writedata(resource->ep, NULL, 0, NULL, 0, addr, 0x87654321, 123456, NULL);
	assert_int_equal(ret, 0);
	
}

void test_efa_rdm_rma_inject_write_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_inject_write(resource->ep, NULL, 0, addr, 0x87654321, 123456);
	assert_int_equal(ret, 0);
	
}

void test_efa_rdm_rma_inject_writedata_0_byte_no_shm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	ret = fi_inject_writedata(resource->ep, NULL, 0, 0, addr, 0x87654321, 123456);
	assert_int_equal(ret, 0);
	
}

void test_efa_rdm_rma_write_0_byte_with_inject_flag(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct iovec iov = {0};
	struct fi_msg_rma msg = {0};
	struct fi_rma_iov rma_iov = {.addr = 0x87654321, .len = 0, .key = 123456};
	fi_addr_t addr;
	int ret;

	efa_unit_test_rdm_0byte_prep(resource, &addr);

	efa_unit_test_construct_msg_rma(&msg, &iov, NULL, 0, addr, &rma_iov, 1, NULL, 0);

	ret = fi_writemsg(resource->ep, &msg, FI_INJECT);
	assert_int_equal(ret, 0);
}
/**
 * @brief Test that partial RDMA write post failure doesn't release the txe
 *
 * When efa_rdm_ope_post_remote_write() posts the first segment of a
 * multi-segment write successfully but fails on a subsequent segment,
 * the caller efa_rdm_rma_generic_writemsg() must not release the txe
 * because the in-flight segment still references it. Releasing it causes
 * a double-free when the completion arrives.
 *
 * This test forces a 2-segment write by shrinking max_rdma_size, then
 * saturates the tx queue so the second segment fails. It verifies that
 * the txe is NOT released before the in-flight completion arrives, and
 * IS released cleanly by the completion path (no leak).
 */
void test_efa_rdm_rma_post_remote_write_partial_fail_no_txe_release(
		struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_unit_test_buff send_buff;
	struct iovec iov;
	struct fi_msg_rma msg = {0};
	struct fi_rma_iov rma_iov;
	struct efa_rdm_pke *inflight_pke;
	fi_addr_t addr;
	struct efa_ep_addr raw_addr;
	size_t raw_addr_len = sizeof(raw_addr);
	size_t max_rdma_size_orig;
	uint64_t wr_id;
	void *desc;
	int ret;

	/* Skip test on platforms that don't support RDMA write. */
	if (!efa_device_support_rdma_write()) {
		skip();
		return;
	}

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	/* Set up peer with RDMA write support */
	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL);
	assert_int_equal(ret, 1);

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_RDMA_WRITE |
				EFA_RDM_EXTRA_FEATURE_RDMA_READ |
				EFA_RDM_EXTRA_FEATURE_UNSOLICITED_WRITE_RECV;

	/* Enable RDMA write and set device caps */
	efa_rdm_ep->use_device_rdma = true;

	/* Create a 128-byte registered buffer */
	efa_unit_test_buff_construct(&send_buff, resource, 128);
	desc = fi_mr_desc(send_buff.mr);

	/* Force 2-segment write: 128 bytes with max_rdma_size=64 */
	max_rdma_size_orig = efa_rdm_ep_domain(efa_rdm_ep)->device->max_rdma_size;
	efa_rdm_ep_domain(efa_rdm_ep)->device->max_rdma_size = 64;

	/* Mock: first segment's efa_qp_post_write succeeds */
	g_efa_unit_test_mocks.efa_qp_post_write = &efa_mock_efa_qp_post_write_return_mock;
	will_return(efa_mock_efa_qp_post_write_return_mock, 0);

	/*
	 * Leave room for exactly 1 tx op. The first segment will post and
	 * fill the queue, then the second segment hits -FI_EAGAIN.
	 */
	efa_rdm_ep->efa_outstanding_tx_ops = efa_rdm_ep->efa_max_outstanding_tx_ops - 1;

	iov.iov_base = send_buff.buff;
	iov.iov_len = 128;
	rma_iov.addr = 0x87654321;
	rma_iov.len = 128;
	rma_iov.key = 123456;
	efa_unit_test_construct_msg_rma(&msg, &iov, &desc, 1, addr,
					&rma_iov, 1, NULL, 0);

	assert_int_equal(g_ibv_submitted_wr_id_cnt, 0);
	ret = fi_writemsg(resource->ep, &msg, 0);
	assert_int_equal(ret, -FI_EAGAIN);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);

	/*
	 * With the fix, the txe must still be live because the first
	 * segment is in-flight (efa_outstanding_tx_ops > 0).
	 */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 1);
	assert_int_equal(efa_rdm_ep->efa_outstanding_tx_ops,
			 efa_rdm_ep->efa_max_outstanding_tx_ops);
	assert_int_equal(peer->efa_outstanding_tx_ops, 1);

	/*
	 * Drive the in-flight segment's completion through the real
	 * completion handler. Pre-fix, this dereferences the freed txe
	 * (SEGV/UAF). Post-fix, it releases the txe cleanly because the
	 * generic writemsg clamped bytes_write_total_len to match what
	 * was submitted.
	 */
	wr_id = (uint64_t) g_ibv_submitted_wr_id_vec[0];
	inflight_pke = efa_rdm_cq_get_pke_from_wr_id_solicited(wr_id);
	efa_rdm_ep_record_tx_op_completed(efa_rdm_ep, inflight_pke);
	efa_rdm_pke_handle_rma_completion(inflight_pke);

	/* The txe was released by the completion path. No leak. */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 0);
	assert_int_equal(efa_rdm_ep->efa_outstanding_tx_ops,
			 efa_rdm_ep->efa_max_outstanding_tx_ops - 1);
	assert_int_equal(peer->efa_outstanding_tx_ops, 0);
	assert_true(dlist_empty(&peer->outstanding_tx_pkts));

	/* Restore */
	efa_rdm_ep_domain(efa_rdm_ep)->device->max_rdma_size = max_rdma_size_orig;
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Reproduce the rma_pingpong double-free under app retry
 *
 * Mirrors the pingpong application pattern: after fi_writemsg returns
 * -FI_EAGAIN on a multi-segment write, the application retries the same
 * operation. A second fi_writemsg allocates a second txe while the first
 * txe's segment-1 packet is still in-flight. Pre-fix, the first txe is
 * already freed by the unconditional release in efa_rdm_rma_generic_writemsg,
 * so when segment 1 completes, efa_rdm_pke_handle_rma_completion dereferences
 * freed memory and eventually calls free() twice (double-free / UAF).
 *
 * This test is expected to SEGV or trip ASan / glibc tcache pre-fix, and
 * pass cleanly post-fix.
 */
void test_efa_rdm_rma_partial_post_retry_no_double_free(
		struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_unit_test_buff send_buff;
	struct iovec iov;
	struct fi_msg_rma msg = {0};
	struct fi_rma_iov rma_iov;
	struct efa_rdm_pke *inflight_pke;
	fi_addr_t addr;
	struct efa_ep_addr raw_addr;
	size_t raw_addr_len = sizeof(raw_addr);
	size_t max_rdma_size_orig;
	uint64_t first_wr_id;
	void *desc;
	int ret;

	/* Skip test on platforms that don't support RDMA write. */
	if (!efa_device_support_rdma_write()) {
		skip();
		return;
	}

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0, NULL);
	assert_int_equal(ret, 1);

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_RDMA_WRITE |
				EFA_RDM_EXTRA_FEATURE_RDMA_READ |
				EFA_RDM_EXTRA_FEATURE_UNSOLICITED_WRITE_RECV;

	efa_rdm_ep->use_device_rdma = true;

	efa_unit_test_buff_construct(&send_buff, resource, 128);
	desc = fi_mr_desc(send_buff.mr);

	/* Force 2-segment write: 128 bytes with max_rdma_size=64 */
	max_rdma_size_orig = efa_rdm_ep_domain(efa_rdm_ep)->device->max_rdma_size;
	efa_rdm_ep_domain(efa_rdm_ep)->device->max_rdma_size = 64;

	/*
	 * Saturate tx queue so every call's first segment posts and second
	 * segment hits -FI_EAGAIN. Each call leaves exactly 1 in-flight pkt
	 * but fails overall.
	 */
	efa_rdm_ep->efa_outstanding_tx_ops = efa_rdm_ep->efa_max_outstanding_tx_ops - 1;

	/* Only the first call posts — it will fill the queue to max. */
	g_efa_unit_test_mocks.efa_qp_post_write = &efa_mock_efa_qp_post_write_return_mock;
	will_return(efa_mock_efa_qp_post_write_return_mock, 0);

	iov.iov_base = send_buff.buff;
	iov.iov_len = 128;
	rma_iov.addr = 0x87654321;
	rma_iov.len = 128;
	rma_iov.key = 123456;
	efa_unit_test_construct_msg_rma(&msg, &iov, &desc, 1, addr,
					&rma_iov, 1, NULL, 0);

	/* First attempt: posts segment 1, fails on segment 2. */
	ret = fi_writemsg(resource->ep, &msg, 0);
	assert_int_equal(ret, -FI_EAGAIN);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
	first_wr_id = (uint64_t) g_ibv_submitted_wr_id_vec[0];

	/*
	 * Simulate app retry BEFORE the in-flight completion arrives. The
	 * queue is still saturated so even segment 1 of the retry gets
	 * -FI_EAGAIN immediately (nothing posted). The retry must not
	 * disturb the first in-flight txe.
	 */
	ret = fi_writemsg(resource->ep, &msg, 0);
	assert_int_equal(ret, -FI_EAGAIN);
	assert_int_equal(g_ibv_submitted_wr_id_cnt, 1);
	/* txe1 must still be live; retry's txe2 was released inline. */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 1);
	assert_int_equal(peer->efa_outstanding_tx_ops, 1);

	/*
	 * Now drive the in-flight completion. Pre-fix, the first txe was
	 * freed and context_pkt_entry->ope points into freed memory — the
	 * handler's write to txe->bytes_write_completed and subsequent
	 * efa_rdm_txe_release trigger UAF / double-free.
	 */
	inflight_pke = efa_rdm_cq_get_pke_from_wr_id_solicited(first_wr_id);
	efa_rdm_ep_record_tx_op_completed(efa_rdm_ep, inflight_pke);
	efa_rdm_pke_handle_rma_completion(inflight_pke);

	/* Post-fix: both txes have been released; peer/ep state is clean. */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->txe_list), 0);
	assert_int_equal(peer->efa_outstanding_tx_ops, 0);
	assert_true(dlist_empty(&peer->outstanding_tx_pkts));

	efa_rdm_ep_domain(efa_rdm_ep)->device->max_rdma_size = max_rdma_size_orig;
	efa_unit_test_buff_destruct(&send_buff);
}
