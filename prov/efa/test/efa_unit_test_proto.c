/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_unit_tests.h"
#include "rdm/efa_rdm_proto.h"
#include "rdm/efa_rdm_proto_eager.h"

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
	assert_non_null(ep->send_pkt_entry_vec[0]->handle_pke);
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
	struct efa_rdm_pke *pkt_entry;
	fi_addr_t peer_addr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
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
	pkt_entry = ep->send_pkt_entry_vec[0];
	assert_non_null(pkt_entry->handle_pke);

	/* Simulate send completion: record_tx_op_completed + callback */
	efa_rdm_ep_record_tx_op_completed(ep, pkt_entry);
	pkt_entry->handle_pke(pkt_entry);

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
