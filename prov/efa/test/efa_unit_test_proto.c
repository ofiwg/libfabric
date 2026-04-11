/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_unit_tests.h"
#include "rdm/efa_rdm_proto.h"
#include "rdm/efa_rdm_proto_eager.h"
#include "rdm/efa_rdm_proto_longcts.h"
#include "rdm/efa_rdm_proto_longread.h"
#include "rdm/efa_rdm_proto_medium.h"
#include "rdm/efa_rdm_proto_runtread.h"

/* Tests from efa_unit_test_proto_select.c */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */


/**
 * @brief Helper to set up an endpoint, peer, and TXE for protocol selection
 * tests.
 *
 * Returns the efa_rdm_ep. Caller must provide a peer_addr output and a
 * pre-allocated txe pointer output.
 */
static struct efa_rdm_ep *setup_proto_select_test(struct efa_resource *resource,
						  fi_addr_t *peer_addr)
{
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_rdm_ep *ep;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, peer_addr, 0, NULL),
		1);

	return ep;
}

/**
 * @brief Test that eager protocol is selected for small messages.
 */
void test_proto_select_eager_for_small_msg(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	fi_addr_t peer_addr;
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	ep = setup_proto_select_test(resource, &peer_addr);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	iov.iov_base = NULL;
	iov.iov_len = 64; /* Small message, fits in eager */
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0, NULL);

	txe = ofi_buf_alloc(ep->ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	assert_int_equal(err, 0);
	assert_non_null(proto);
	assert_ptr_equal(proto, &efa_rdm_proto_eager);

	ofi_buf_free(txe);
}

/**
 * @brief Test that medium protocol is selected for messages between eager
 * capacity and 64KB.
 */
void test_proto_select_medium_for_mid_msg(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	fi_addr_t peer_addr;
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	ep = setup_proto_select_test(resource, &peer_addr);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	/* 16KB - too large for eager, fits in medium */
	iov.iov_base = NULL;
	iov.iov_len = 16384;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0, NULL);

	txe = ofi_buf_alloc(ep->ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	assert_int_equal(err, 0);
	assert_non_null(proto);
	assert_ptr_equal(proto, &efa_rdm_proto_medium);

	ofi_buf_free(txe);
}

/**
 * @brief Test that long CTS is selected for large messages when no p2p
 * or no registered memory is available.
 */
void test_proto_select_longcts_for_large_msg_no_p2p(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	fi_addr_t peer_addr;
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	ep = setup_proto_select_test(resource, &peer_addr);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	/* 128KB - too large for medium, no desc so no read-based protocols */
	iov.iov_base = NULL;
	iov.iov_len = 131072;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0, NULL);

	txe = ofi_buf_alloc(ep->ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	assert_int_equal(err, 0);
	assert_non_null(proto);
	assert_ptr_equal(proto, &efa_rdm_proto_longcts);

	ofi_buf_free(txe);
}

/**
 * @brief Test that eager is selected before medium for messages that fit
 * in eager (protocol priority ordering).
 */
void test_proto_select_eager_has_priority_over_medium(
	struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	fi_addr_t peer_addr;
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	ep = setup_proto_select_test(resource, &peer_addr);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	/* 1 byte - fits in both eager and medium, eager should win */
	iov.iov_base = NULL;
	iov.iov_len = 1;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0, NULL);

	txe = ofi_buf_alloc(ep->ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	assert_int_equal(err, 0);
	assert_ptr_equal(proto, &efa_rdm_proto_eager);

	ofi_buf_free(txe);
}

/**
 * @brief Test that zero-length messages select eager protocol.
 */
void test_proto_select_eager_for_zero_len_msg(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	fi_addr_t peer_addr;
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	ep = setup_proto_select_test(resource, &peer_addr);
	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	iov.iov_base = NULL;
	iov.iov_len = 0;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0, NULL);

	txe = ofi_buf_alloc(ep->ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg, 0,
						 txe, &proto);
	assert_int_equal(err, 0);
	assert_ptr_equal(proto, &efa_rdm_proto_eager);

	ofi_buf_free(txe);
}

/**
 * @brief Test that long read protocol is selected over long CTS when
 * p2p is available, memory is registered, and peer supports RDMA read.
 */
void test_proto_select_longread_over_longcts_with_p2p(
	struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	struct efa_unit_test_buff send_buff;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct fi_msg msg = {0};
	struct iovec iov;
	void *desc;
	int err;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	/* 2MB - above min_read_msg_size (1MB default) */
	efa_unit_test_buff_construct(&send_buff, resource, 2 * 1024 * 1024);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr,
				    &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1,
				      &peer_addr, 0, NULL), 1);

	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	/* Enable RDMA read support on peer */
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_RDMA_READ;
	peer->device_version =
		efa_rdm_ep_domain(ep)->device->ibv_attr.vendor_part_id;

	/* Enable RDMA read on the endpoint and device */
	ep->use_device_rdma = true;
	efa_rdm_ep_domain(ep)->device->device_caps |=
		EFADV_DEVICE_ATTR_CAPS_RDMA_READ;

	desc = fi_mr_desc(send_buff.mr);
	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0, &desc);

	txe = ofi_buf_alloc(ep->ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg,
						 0, txe, &proto);
	assert_int_equal(err, 0);
	assert_non_null(proto);
	assert_ptr_equal(proto, &efa_rdm_proto_longread);

	/* Clean up MRs that select_send_protocol may have registered */
	for (int i = 0; i < txe->iov_count; i++) {
		if (txe->mr[i])
			fi_close(&txe->mr[i]->fid);
	}
	ofi_buf_free(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Test that runt read protocol is selected over long read and
 * long CTS when conditions are met: p2p available, memory registered,
 * peer supports RDMA read, no reads in flight, and runt is allowed.
 *
 * Runt read has higher priority than long read in the protocol list.
 */
void test_proto_select_runtread_over_longread(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_proto *proto = NULL;
	struct efa_unit_test_buff send_buff;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct fi_msg msg = {0};
	struct iovec iov;
	void *desc;
	int err;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	/* 2MB - above min_read_msg_size */
	efa_unit_test_buff_construct(&send_buff, resource, 2 * 1024 * 1024);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr,
				    &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1,
				      &peer_addr, 0, NULL), 1);

	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_RDMA_READ;
	peer->device_version =
		efa_rdm_ep_domain(ep)->device->ibv_attr.vendor_part_id;
	/* Runt read requires no reads in flight and runt allowed */
	efa_rdm_ep_domain(ep)->num_read_msg_in_flight = 0;
	/* Set runt_size > num_runt_bytes_in_flight for system memory */
	g_efa_hmem_info[FI_HMEM_SYSTEM].runt_size = 1000;
	peer->num_runt_bytes_in_flight = 2000;

	/* Enable RDMA read on the endpoint and device */
	ep->use_device_rdma = true;
	efa_rdm_ep_domain(ep)->device->device_caps |=
		EFADV_DEVICE_ATTR_CAPS_RDMA_READ;

	desc = fi_mr_desc(send_buff.mr);
	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0, &desc);

	txe = ofi_buf_alloc(ep->ope_pool);
	assert_non_null(txe);

	err = efa_rdm_proto_select_send_protocol(ep, peer, &msg, ofi_op_msg,
						 0, txe, &proto);
	assert_int_equal(err, 0);
	assert_non_null(proto);
	assert_ptr_equal(proto, &efa_rdm_proto_runtread);

	for (int i = 0; i < txe->iov_count; i++) {
		if (txe->mr[i])
			fi_close(&txe->mr[i]->fid);
	}
	ofi_buf_free(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Test that eager construct_tx_pkes produces exactly 1 PKE with
 * the correct callback set.
 */
void test_proto_eager_construct_pkes_single_pke(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_unit_test_buff_construct(&send_buff, resource, 64);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->ope_pool);
	assert_non_null(txe);

	/* Initialize fields that select_send_protocol would set */
	txe->ep = ep;
	txe->total_len = send_buff.size;
	txe->iov_count = 1;
	memcpy(txe->iov, &iov, sizeof(iov));
	txe->desc[0] = fi_mr_desc(send_buff.mr);
	memset(txe->mr, 0, sizeof(*txe->mr));

	err = efa_rdm_proto_eager.construct_tx_pkes(ep, peer, &msg, ofi_op_msg,
						    0, 0, txe);
	assert_int_equal(err, 0);
	assert_int_equal(ep->send_pkt_entry_vec_size, 1);
	assert_non_null(ep->send_pkt_entry_vec[0]);
	assert_non_null(ep->send_pkt_entry_vec[0]->callback);
	assert_ptr_equal(ep->send_pkt_entry_vec[0]->ope, txe);

	/* Clean up */
	efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[0]);
	efa_rdm_txe_release(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Test that eager send completion callback releases TXE and PKE
 * for non-DC messages.
 */
void test_proto_eager_send_completion_releases_txe(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *pkt_entry;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_unit_test_buff_construct(&send_buff, resource, 64);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	/* Mock efa_qp_post_send to succeed */
	g_efa_unit_test_mocks.efa_qp_post_send =
		&efa_mock_efa_qp_post_send_return_mock;
	will_return_int_maybe(efa_mock_efa_qp_post_send_return_mock, 0);

	/* Send a message via fi_send which goes through the new code path */
	err = fi_send(resource->ep, send_buff.buff, send_buff.size,
		      fi_mr_desc(send_buff.mr), peer_addr, NULL);
	assert_int_equal(err, 0);
	assert_int_equal(efa_unit_test_get_dlist_length(&ep->txe_list), 1);

	/* Get the TXE and PKE */
	txe = container_of(ep->txe_list.next, struct efa_rdm_ope, ep_entry);
	pkt_entry = ep->send_pkt_entry_vec[0];
	assert_non_null(pkt_entry->callback);

	/* Simulate send completion: record_tx_op_completed + callback */
	efa_rdm_ep_record_tx_op_completed(ep, pkt_entry);
	pkt_entry->callback(pkt_entry);

	/* TXE should be released */
	assert_int_equal(efa_unit_test_get_dlist_length(&ep->txe_list), 0);

	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Test that eager assigns msg_id from peer->next_msg_id.
 */
void test_proto_eager_assigns_msg_id(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	uint32_t initial_msg_id;
	int err;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_unit_test_buff_construct(&send_buff, resource, 64);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	initial_msg_id = peer->next_msg_id;

	g_efa_unit_test_mocks.efa_qp_post_send =
		&efa_mock_efa_qp_post_send_return_mock;
	will_return_int_maybe(efa_mock_efa_qp_post_send_return_mock, 0);

	err = fi_send(resource->ep, send_buff.buff, send_buff.size,
		      fi_mr_desc(send_buff.mr), peer_addr, NULL);
	assert_int_equal(err, 0);

	/* msg_id should have been assigned and next_msg_id incremented */
	struct efa_rdm_ope *txe =
		container_of(ep->txe_list.next, struct efa_rdm_ope, ep_entry);
	assert_int_equal(txe->msg_id, initial_msg_id);
	assert_int_equal(peer->next_msg_id, initial_msg_id + 1);

	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Test that medium construct_tx_pkes produces multiple PKEs for a
 * message that requires segmentation (16KB = 2 packets at ~8KB MTU).
 */
void test_proto_medium_construct_pkes_multi_pke(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct fi_msg msg = {0};
	struct iovec iov;
	int err, i;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_unit_test_buff_construct(&send_buff, resource, 16384);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	void *desc = fi_mr_desc(send_buff.mr);

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    &desc);

	txe = ofi_buf_alloc(ep->ope_pool);
	assert_non_null(txe);

	txe->ep = ep;
	txe->total_len = send_buff.size;
	txe->iov_count = 1;
	memcpy(txe->iov, &iov, sizeof(iov));
	txe->desc[0] = fi_mr_desc(send_buff.mr);
	memset(txe->mr, 0, sizeof(*txe->mr));

	err = efa_rdm_proto_medium.construct_tx_pkes(ep, peer, &msg, ofi_op_msg,
						     0, 0, txe);
	assert_int_equal(err, 0);

	/* 16KB should require at least 2 packets */
	assert_true(ep->send_pkt_entry_vec_size >= 2);

	/* Each PKE should have a callback and correct TXE */
	for (i = 0; i < ep->send_pkt_entry_vec_size; i++) {
		assert_non_null(ep->send_pkt_entry_vec[i]);
		assert_non_null(ep->send_pkt_entry_vec[i]->callback);
		assert_ptr_equal(ep->send_pkt_entry_vec[i]->ope, txe);
	}

	/* Clean up */
	for (i = 0; i < ep->send_pkt_entry_vec_size; i++)
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[i]);
	efa_rdm_txe_release(txe);
	efa_unit_test_buff_destruct(&send_buff);
}

/**
 * @brief Test that medium send completion tracks bytes_acked and only
 * releases TXE when all bytes are acked.
 */
void test_proto_medium_send_completion_tracks_bytes_acked(
	struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct fi_msg msg = {0};
	struct iovec iov;
	int err, i;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	efa_unit_test_buff_construct(&send_buff, resource, 16384);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	g_efa_unit_test_mocks.efa_qp_post_send =
		&efa_mock_efa_qp_post_send_return_mock;
	will_return_int_maybe(efa_mock_efa_qp_post_send_return_mock, 0);

	err = fi_send(resource->ep, send_buff.buff, send_buff.size,
		      fi_mr_desc(send_buff.mr), peer_addr, NULL);
	assert_int_equal(err, 0);
	assert_int_equal(efa_unit_test_get_dlist_length(&ep->txe_list), 1);

	txe = container_of(ep->txe_list.next, struct efa_rdm_ope, ep_entry);
	assert_true(ep->send_pkt_entry_vec_size >= 2);

	/* Complete first PKE - TXE should NOT be released yet */
	struct efa_rdm_pke *first_pke = ep->send_pkt_entry_vec[0];
	efa_rdm_ep_record_tx_op_completed(ep, first_pke);
	first_pke->callback(first_pke);
	assert_int_equal(efa_unit_test_get_dlist_length(&ep->txe_list), 1);

	/* Complete remaining PKEs */
	for (i = 1; i < ep->send_pkt_entry_vec_size; i++) {
		struct efa_rdm_pke *pke = ep->send_pkt_entry_vec[i];
		efa_rdm_ep_record_tx_op_completed(ep, pke);
		pke->callback(pke);
	}

	/* Now TXE should be released */
	assert_int_equal(efa_unit_test_get_dlist_length(&ep->txe_list), 0);

	efa_unit_test_buff_destruct(&send_buff);
}

/* Long CTS protocol TX tests */

/**
 * @brief Test that long read construct_tx_pkes produces 1 PKE with
 * correct pkt_size (header + read IOVs).
 */
void test_proto_longcts_construct_pkes_single_rtm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_unit_test_buff send_buff;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct fi_msg msg = {0};
	struct iovec iov;
	int err;

	efa_unit_test_resource_construct_rdm_shm_disabled(resource);
	/* 128KB - large enough to require long CTS */
	efa_unit_test_buff_construct(&send_buff, resource, 131072);

	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	peer = efa_rdm_ep_get_peer(ep, peer_addr);
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	iov.iov_base = send_buff.buff;
	iov.iov_len = send_buff.size;
	efa_unit_test_construct_msg(&msg, &iov, 1, peer_addr, NULL, 0,
				    (void **) &send_buff.mr);

	txe = ofi_buf_alloc(ep->ope_pool);
	assert_non_null(txe);

	txe->ep = ep;
	txe->total_len = send_buff.size;
	txe->iov_count = 1;
	memcpy(txe->iov, &iov, sizeof(iov));
	txe->desc[0] = fi_mr_desc(send_buff.mr);
	memset(txe->mr, 0, sizeof(*txe->mr));

	err = efa_rdm_proto_longcts.construct_tx_pkes(ep, peer, &msg,
						      ofi_op_msg, 0, 0, txe);
	assert_int_equal(err, 0);

	/* Long CTS sends exactly 1 RTM packet initially */
	assert_int_equal(ep->send_pkt_entry_vec_size, 1);
	assert_non_null(ep->send_pkt_entry_vec[0]);
	assert_non_null(ep->send_pkt_entry_vec[0]->callback);
	assert_ptr_equal(ep->send_pkt_entry_vec[0]->ope, txe);

	/* Clean up */
	efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[0]);
	efa_rdm_txe_release(txe);
	efa_unit_test_buff_destruct(&send_buff);
}
