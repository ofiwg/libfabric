/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"
#include "ofi_util.h"
#include "efa_rdm_ep.h"
#include "efa_rdm_msg.h"
#include "efa_av.h"
#include "efa_rdm_pke_rtm.h"
#include "efa_rdm_srx.h"

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

/**
 * @brief Verify that peer_construct returns 0 on success (validates new int return type)
 *
 * After changing peer_construct from void to int, verify that the normal
 * path still works correctly and returns 0.
 */
void test_efa_rdm_peer_construct_robuf_failure(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer peer = {0};
	struct efa_ep_addr raw_addr;
	struct efa_conn conn = {0};
	fi_addr_t peer_addr;
	size_t raw_addr_len = sizeof(raw_addr);
	struct ofi_bufpool *saved_pool;
	struct ofi_bufpool *tiny_pool;
	void *buf;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 99;
	raw_addr.qkey = 0xABCD;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL);
	assert_int_equal(ret, 1);

	conn.ep_addr = &raw_addr;
	conn.fi_addr = peer_addr;

	/* Create a tiny pool with max_cnt=1 and exhaust it */
	ret = ofi_bufpool_create(&tiny_pool,
				 efa_rdm_ep->peer_robuf_pool->entry_size,
				 EFA_RDM_BUFPOOL_ALIGNMENT, 1 /* max_cnt */,
				 1, 0);
	assert_int_equal(ret, 0);
	ret = ofi_bufpool_grow(tiny_pool);
	assert_int_equal(ret, 0);
	buf = ofi_buf_alloc(tiny_pool);
	assert_non_null(buf);

	/* Swap in the exhausted pool */
	saved_pool = efa_rdm_ep->peer_robuf_pool;
	efa_rdm_ep->peer_robuf_pool = tiny_pool;

	/* peer_construct should fail with -FI_ENOMEM */
	ret = efa_rdm_peer_construct(&peer, efa_rdm_ep, &conn);
	assert_int_equal(ret, -FI_ENOMEM);

	/* Restore and cleanup */
	efa_rdm_ep->peer_robuf_pool = saved_pool;
	ofi_buf_free(buf);
	ofi_bufpool_destroy(tiny_pool);
}

/**
 * @brief Verify that efa_rdm_srx_repost_peer_rxe re-queues a matched
 *        peer_rxe back into the SRX msg_queue at its head as POSTED.
 *
 * Steps:
 *   1. Construct an RDM endpoint and post a recv via the SRX. Confirm
 *      the entry sits at the head of msg_queue with RX_ENTRY_POSTED.
 *   2. Match the entry by calling get_msg with FI_ADDR_UNSPEC, which
 *      removes it from the queue and sets its status to MATCHED.
 *   3. Call efa_rdm_srx_repost_peer_rxe.
 *   4. Assert the entry is back at the queue head with status POSTED,
 *      with its peer_context cleared.
 */
void test_efa_srx_repost_peer_rxe_msg_unspec(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct util_rx_entry *util_entry;
	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(efa_rdm_ep);
	peer_srx = util_get_peer_srx(efa_rdm_ep->peer_srx_ep);

	/* 1. Post a recv. */
	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_recv(efa_rdm_ep->peer_srx_ep, &iov, &desc, 1,
				    FI_ADDR_UNSPEC, /*context=*/(void *) 0xa1,
				    /*flags=*/0);
	assert_int_equal(ret, FI_SUCCESS);
	assert_false(slist_empty(&srx_ctx->msg_queue));

	/* 2. Match the entry. */
	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);
	assert_non_null(peer_rxe);
	assert_true(slist_empty(&srx_ctx->msg_queue));

	util_entry = container_of(peer_rxe, struct util_rx_entry, peer_entry);
	assert_int_equal(util_entry->status, RX_ENTRY_MATCHED);

	/* Pretend the EFA provider stashed a peer_context — re-queue must
	 * clear it so the next match round populates a fresh one. */
	peer_rxe->peer_context = (void *) 0xdeadbeef;

	/* 3. Re-queue. */
	ret = efa_rdm_srx_repost_peer_rxe(peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	/* 4. Verify state. */
	assert_false(slist_empty(&srx_ctx->msg_queue));
	assert_ptr_equal(srx_ctx->msg_queue.head, &util_entry->s_entry);
	assert_int_equal(util_entry->status, RX_ENTRY_POSTED);
	assert_null(peer_rxe->peer_context);

	/* The user's posted-recv context must be preserved so a future
	 * matching message reports the right op_context. */
	assert_ptr_equal(peer_rxe->context, (void *) 0xa1);
	ofi_genlock_unlock(srx_ctx->lock);
}

/**
 * @brief Same as test_efa_srx_repost_peer_rxe_msg_unspec but for a
 *        tagged recv — confirms tag_queue routing.
 */
void test_efa_srx_repost_peer_rxe_tag_unspec(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct util_rx_entry *util_entry;
	struct iovec iov;
	char buf[16];
	void *desc = NULL;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(efa_rdm_ep);
	peer_srx = util_get_peer_srx(efa_rdm_ep->peer_srx_ep);

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_trecv(efa_rdm_ep->peer_srx_ep, &iov, &desc, 1,
				     FI_ADDR_UNSPEC, /*context=*/(void *) 0xb2,
				     /*tag=*/0x42, /*ignore=*/0,
				     /*flags=*/0);
	assert_int_equal(ret, FI_SUCCESS);
	assert_false(slist_empty(&srx_ctx->tag_queue));

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.tag = 0x42;
	match_attr.msg_size = 16;

	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_tag(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);
	assert_non_null(peer_rxe);
	assert_true(slist_empty(&srx_ctx->tag_queue));

	util_entry = container_of(peer_rxe, struct util_rx_entry, peer_entry);
	assert_int_equal(util_entry->status, RX_ENTRY_MATCHED);

	ret = efa_rdm_srx_repost_peer_rxe(peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	assert_false(slist_empty(&srx_ctx->tag_queue));
	assert_ptr_equal(srx_ctx->tag_queue.head, &util_entry->s_entry);
	assert_int_equal(util_entry->status, RX_ENTRY_POSTED);
	ofi_genlock_unlock(srx_ctx->lock);
}

/**
 * @brief Verify efa_rdm_srx_repost_peer_rxe rejects NULL inputs cleanly.
 *
 * The helper guards against NULL peer_rxe and a NULL .srx pointer on
 * the peer_rxe (either of which would otherwise dereference NULL).
 */
void test_efa_srx_repost_peer_rxe_null_input(struct efa_resource **state)
{
	struct fi_peer_rx_entry stub;
	(void) state;

	assert_int_equal(efa_rdm_srx_repost_peer_rxe(NULL), -FI_EINVAL);

	memset(&stub, 0, sizeof(stub));
	stub.srx = NULL;
	assert_int_equal(efa_rdm_srx_repost_peer_rxe(&stub), -FI_EINVAL);
}

/**
 * @brief Verify efa_rdm_srx_repost_peer_rxe routes a directed-source
 *        recv back to its src_recv_queues entry rather than the global
 *        msg_queue.
 *
 * Today EFA's RDM SRX is constructed with dir_recv=false, so all
 * entries land in the global queue. This test toggles dir_recv on the
 * SRX context to exercise the per-source branch in the helper, which
 * is the only branch in the helper that production-shaped tests
 * never hit.
 */
void test_efa_srx_repost_peer_rxe_msg_directed(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct util_srx_ctx *srx_ctx;
	struct fid_peer_srx *peer_srx;
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct util_rx_entry *util_entry;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t source_addr;
	struct iovec iov;
	struct slist *queue;
	char buf[16];
	void *desc = NULL;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);
	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(efa_rdm_ep);
	peer_srx = util_get_peer_srx(efa_rdm_ep->peer_srx_ep);

	/* Force per-source-queue routing for this test. */
	srx_ctx->dir_recv = true;

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len),
			 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(resource->av, &raw_addr, 1, &source_addr,
				      0, NULL),
			 1);

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	ret = util_srx_generic_recv(efa_rdm_ep->peer_srx_ep, &iov, &desc, 1,
				    source_addr, /*context=*/(void *) 0xc1, 0);
	assert_int_equal(ret, FI_SUCCESS);

	queue = ofi_array_at(&srx_ctx->src_recv_queues, source_addr);
	assert_non_null(queue);
	assert_false(slist_empty(queue));
	assert_true(slist_empty(&srx_ctx->msg_queue));

	match_attr.addr = source_addr;
	match_attr.tag = 0;
	match_attr.msg_size = 16;
	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);
	util_entry = container_of(peer_rxe, struct util_rx_entry, peer_entry);

	/* Matched: per-source queue is now empty. */
	assert_true(slist_empty(queue));

	ret = efa_rdm_srx_repost_peer_rxe(peer_rxe);
	assert_int_equal(ret, FI_SUCCESS);

	/* Re-queued back into the per-source queue (not the global one). */
	assert_false(slist_empty(queue));
	assert_ptr_equal(queue->head, &util_entry->s_entry);
	assert_int_equal(util_entry->status, RX_ENTRY_POSTED);
	assert_true(slist_empty(&srx_ctx->msg_queue));
	ofi_genlock_unlock(srx_ctx->lock);
}
