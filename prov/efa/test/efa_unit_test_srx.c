/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"
#include "ofi_util.h"
#include "efa_rdm_ep.h"
#include "efa_rdm_msg.h"
#include "efa_av.h"
#include "efa_rdm_pke_rtm.h"

/**
 * @brief This test validates whether the default min_multi_recv size is correctly
 * passed from ep to srx, and whether is correctly modified when application
 * change it via fi_setopt
 *
 */
void test_efa_srx_min_multi_recv_size(struct efa_resource **state)
{
        struct efa_resource *resource = *state;
        struct efa_rdm_ep *efa_rdm_ep;
        struct util_srx_ctx *srx_ctx;
        size_t min_multi_recv_size_new;

        efa_unit_test_resource_construct_ep_not_enabled(resource, FI_EP_RDM, EFA_FABRIC_NAME);

        efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
        /* Set a new min_multi_recv_size via setopt*/
        min_multi_recv_size_new = 1024;
        assert_int_equal(fi_setopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
			&min_multi_recv_size_new, sizeof(min_multi_recv_size_new)), 0);

        /* Enable EP */
        assert_int_equal(fi_enable(resource->ep), FI_SUCCESS);

        /* Check whether srx->min_multi_recv_size is set correctly */
        srx_ctx = efa_rdm_ep_get_peer_srx_ctx(efa_rdm_ep);
        assert_true(srx_ctx->min_multi_recv_size == min_multi_recv_size_new);
}


/* This test verified that cq is correctly bound to srx when it's bound to ep */
void test_efa_srx_cq(struct efa_resource **state)
{
        struct efa_resource *resource = *state;
        struct efa_rdm_ep *efa_rdm_ep;
        struct util_srx_ctx *srx_ctx;

        efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

        efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
        srx_ctx = efa_rdm_ep_get_peer_srx_ctx(efa_rdm_ep);
        assert_true((void *) &srx_ctx->cq->cq_fid == (void *) resource->cq);
}

/* This test verified that srx_lock created in efa_domain is correctly passed to srx */
void test_efa_srx_lock(struct efa_resource **state)
{
        struct efa_resource *resource = *state;
        struct efa_rdm_ep *efa_rdm_ep;
        struct util_srx_ctx *srx_ctx;
        struct efa_domain *efa_domain;

        efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

        efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
        srx_ctx = efa_rdm_ep_get_peer_srx_ctx(efa_rdm_ep);
        efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid.fid);
        assert_true(((void *) srx_ctx->lock == (void *) &efa_domain->srx_lock));
}


/**
 * @brief Test srx's start ops is updating the rxe status correctly
 *
 * @param state
 */
void test_efa_srx_unexp_pkt(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct util_srx_ctx *srx_ctx;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_pke *pke;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_conn conn = {0};
	struct efa_rdm_peer peer;
	struct efa_unit_test_eager_rtm_pkt_attr pke_attr = {.msg_id = 0,
							    .connid = 0x1234};
	void *desc;
	struct iovec iov;

	g_efa_unit_test_mocks.efa_rdm_pke_proc_matched_rtm =
		&efa_mock_efa_rdm_pke_proc_matched_rtm_no_op;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(efa_rdm_ep);

	/* Fake a rx pkt entry */
	pke = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pke);
	efa_rdm_ep->efa_rx_pkts_posted =
		efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep);

	/* Create a fake peer */
	/* TODO: peer must be constructed by CQ read path */
	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 0;
	raw_addr.qkey = 0x1234;
	conn.ep_addr = &raw_addr;
	efa_rdm_peer_construct(&peer, efa_rdm_ep, &conn);
	pke->peer = &peer;

	efa_unit_test_eager_msgrtm_pkt_construct(pke, &pke_attr);
	/**
	 * Allocate an rxe with the rx pkt.
	 * Since there is no recv posted, the rxe must be unexpected
	 */
	ofi_genlock_lock(srx_ctx->lock);
	rxe = efa_rdm_msg_alloc_rxe_for_msgrtm(efa_rdm_ep, &pke);
	assert_true(rxe->state == EFA_RDM_RXE_UNEXP);
	assert_true(rxe->unexp_pkt == pke);
	srx_ctx->peer_srx.owner_ops->queue_msg(rxe->peer_rxe);
	ofi_genlock_unlock(srx_ctx->lock);

	/* Fake an application posted receive */
	util_srx_generic_recv(efa_rdm_ep->peer_srx_ep, &iov, &desc, 1, 0, NULL,
			      0);

	/* Make sure rxe is updated as mateched and unexp_pkt is NULL */
	assert_true(rxe->state == EFA_RDM_RXE_MATCHED);
	assert_true(rxe->unexp_pkt == NULL);

	ofi_genlock_lock(srx_ctx->lock);
	efa_rdm_pke_release_rx(pke);
	efa_rdm_rxe_release(rxe);
	ofi_genlock_unlock(srx_ctx->lock);

	/* Destroy the fake peer constructed above */
	efa_rdm_peer_destruct(&peer, efa_rdm_ep);
}

/**
 * @brief Test that util_foreach_unspec only processes entries belonging to the
 * calling provider's fid_peer_srx, and skips entries from other providers.
 *
 * Both owner and peer providers queue messages from unknown peers into the same
 * unspec queue, with peer_entry.srx assigned to its own fid_peer_srx. During
 * fi_av_insert that calls util_foreach_unspec to scan this queue, each provider
 * should only process entries from its own provider and ignore entries from the
 * other one.
 */
static fi_addr_t test_foreach_unspec_get_addr(struct fi_peer_rx_entry *entry)
{
	/* Return a valid address so the entry gets moved out of unspec queue */
	return 0;
}

static int test_foreach_unspec_discard_no_op(struct fi_peer_rx_entry *entry)
{
	return FI_SUCCESS;
}

void test_efa_srx_foreach_unspec_skips_other_provider(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct util_srx_ctx *srx_ctx;
	struct util_rx_entry *msg_entry_efa, *msg_entry_shm;
	struct util_rx_entry *tag_entry_efa, *tag_entry_shm;
	struct fid_peer_srx *efa_srx;
	struct fid_peer_srx *shm_srx;
	struct fi_ops_srx_peer *efa_peer_ops;
	struct fi_ops_srx_peer *shm_peer_ops;
	int (*saved_efa_discard_msg)(struct fi_peer_rx_entry *);
	int (*saved_efa_discard_tag)(struct fi_peer_rx_entry *);
	int (*saved_shm_discard_msg)(struct fi_peer_rx_entry *);
	int (*saved_shm_discard_tag)(struct fi_peer_rx_entry *);
	struct util_rx_entry *remaining_msg;
	struct util_rx_entry *remaining_tag;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(efa_rdm_ep);
	efa_srx = &srx_ctx->peer_srx;
	shm_srx = efa_rdm_ep->shm_peer_srx;

	/* Enable directed receive so foreach_unspec will move resolved entries */
	srx_ctx->dir_recv = true;

	ofi_genlock_lock(srx_ctx->lock);

	/* Allocate entries for the unspec msg queue */
	msg_entry_efa = (struct util_rx_entry *) ofi_buf_alloc(srx_ctx->rx_pool);
	assert_non_null(msg_entry_efa);
	msg_entry_efa->peer_entry.srx = efa_srx;
	msg_entry_efa->peer_entry.addr = FI_ADDR_UNSPEC;
	msg_entry_efa->peer_entry.flags = FI_MSG | FI_RECV;
	msg_entry_efa->status = RX_ENTRY_UNEXP;

	msg_entry_shm = (struct util_rx_entry *) ofi_buf_alloc(srx_ctx->rx_pool);
	assert_non_null(msg_entry_shm);
	msg_entry_shm->peer_entry.srx = shm_srx;
	msg_entry_shm->peer_entry.addr = FI_ADDR_UNSPEC;
	msg_entry_shm->peer_entry.flags = FI_MSG | FI_RECV;
	msg_entry_shm->status = RX_ENTRY_UNEXP;

	/* Allocate entries for the unspec tag queue */
	tag_entry_efa = (struct util_rx_entry *) ofi_buf_alloc(srx_ctx->rx_pool);
	assert_non_null(tag_entry_efa);
	tag_entry_efa->peer_entry.srx = efa_srx;
	tag_entry_efa->peer_entry.addr = FI_ADDR_UNSPEC;
	tag_entry_efa->peer_entry.flags = FI_TAGGED | FI_RECV;
	tag_entry_efa->status = RX_ENTRY_UNEXP;

	tag_entry_shm = (struct util_rx_entry *) ofi_buf_alloc(srx_ctx->rx_pool);
	assert_non_null(tag_entry_shm);
	tag_entry_shm->peer_entry.srx = shm_srx;
	tag_entry_shm->peer_entry.addr = FI_ADDR_UNSPEC;
	tag_entry_shm->peer_entry.flags = FI_TAGGED | FI_RECV;
	tag_entry_shm->status = RX_ENTRY_UNEXP;

	/* Insert all entries into the unspec queues */
	dlist_insert_tail(&msg_entry_efa->d_entry,
			  &srx_ctx->unspec_unexp_msg_queue);
	dlist_insert_tail(&msg_entry_shm->d_entry,
			  &srx_ctx->unspec_unexp_msg_queue);
	dlist_insert_tail(&tag_entry_efa->d_entry,
			  &srx_ctx->unspec_unexp_tag_queue);
	dlist_insert_tail(&tag_entry_shm->d_entry,
			  &srx_ctx->unspec_unexp_tag_queue);

	assert_int_equal(efa_unit_test_get_dlist_length(
				 &srx_ctx->unspec_unexp_msg_queue), 2);
	assert_int_equal(efa_unit_test_get_dlist_length(
				 &srx_ctx->unspec_unexp_tag_queue), 2);

	/* Call foreach_unspec with efa_srx - should only process efa entries */
	efa_srx->owner_ops->foreach_unspec_addr(efa_srx,
						&test_foreach_unspec_get_addr);

	/*
	 * After foreach_unspec, the efa entries should have been moved out
	 * (address resolved), while the shm provider's entries should remain
	 * in the unspec queues.
	 */
	assert_int_equal(efa_unit_test_get_dlist_length(
				 &srx_ctx->unspec_unexp_msg_queue), 1);
	assert_int_equal(efa_unit_test_get_dlist_length(
				 &srx_ctx->unspec_unexp_tag_queue), 1);

	/* Verify the remaining entries belong to the shm provider */
	remaining_msg = container_of(srx_ctx->unspec_unexp_msg_queue.next,
				     struct util_rx_entry, d_entry);
	assert_ptr_equal(remaining_msg->peer_entry.srx, shm_srx);

	remaining_tag = container_of(srx_ctx->unspec_unexp_tag_queue.next,
				     struct util_rx_entry, d_entry);
	assert_ptr_equal(remaining_tag->peer_entry.srx, shm_srx);

	/* Clean up: use shm_srx foreach_unspec to move shm entries out */
	shm_srx->owner_ops->foreach_unspec_addr(shm_srx,
						&test_foreach_unspec_get_addr);

	assert_int_equal(efa_unit_test_get_dlist_length(
				 &srx_ctx->unspec_unexp_msg_queue), 0);
	assert_int_equal(efa_unit_test_get_dlist_length(
				 &srx_ctx->unspec_unexp_tag_queue), 0);

	/*
	 * Replace discard_msg/discard_tag with no-ops on both efa and shm
	 * peer_ops before ep close, since our test entries don't have valid
	 * pkt_entry pointers. Save the original function pointers so we
	 * can restore them before closing.
	 */
	efa_peer_ops = efa_srx->peer_ops;
	shm_peer_ops = shm_srx->peer_ops;
	saved_efa_discard_msg = efa_peer_ops->discard_msg;
	saved_efa_discard_tag = efa_peer_ops->discard_tag;
	saved_shm_discard_msg = shm_peer_ops->discard_msg;
	saved_shm_discard_tag = shm_peer_ops->discard_tag;

	efa_peer_ops->discard_msg = test_foreach_unspec_discard_no_op;
	efa_peer_ops->discard_tag = test_foreach_unspec_discard_no_op;
	shm_peer_ops->discard_msg = test_foreach_unspec_discard_no_op;
	shm_peer_ops->discard_tag = test_foreach_unspec_discard_no_op;

	ofi_genlock_unlock(srx_ctx->lock);

	/* ep close will call util_srx_close which drains remaining entries */
	fi_close(&resource->ep->fid);
	resource->ep = NULL;

	/* Restore original discard ops using saved function pointers */
	efa_peer_ops->discard_msg = saved_efa_discard_msg;
	efa_peer_ops->discard_tag = saved_efa_discard_tag;
	shm_peer_ops->discard_msg = saved_shm_discard_msg;
	shm_peer_ops->discard_tag = saved_shm_discard_tag;
}
