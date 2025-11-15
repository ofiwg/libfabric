/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_unit_tests.h"
#include "efa_rdm_rma.h"

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