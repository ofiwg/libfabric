/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"
#include "ofi_util.h"
#include "efa_rdm_ep.h"
#include "efa_rdm_msg.h"
#include "efa_av.h"

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
        struct efa_unit_test_eager_rtm_pkt_attr pke_attr = {
                .msg_id = 0,
                .connid = 0x1234
        };

        g_efa_unit_test_mocks.efa_rdm_pke_proc_matched_rtm = &efa_mock_efa_rdm_pke_proc_matched_rtm_no_op;

        efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

        efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
        srx_ctx = efa_rdm_ep_get_peer_srx_ctx(efa_rdm_ep);

        /* Fake a rx pkt entry */
        pke = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool, EFA_RDM_PKE_FROM_EFA_RX_POOL);
        assert_non_null(pke);
        efa_rdm_ep->efa_rx_pkts_posted = efa_rdm_ep_get_rx_pool_size(efa_rdm_ep);
        pke->addr = FI_ADDR_UNSPEC;

        /* Create a fake peer */
        /* TODO: peer must be constructed by CQ read path */
        assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
        raw_addr.qpn = 0;
        raw_addr.qkey = 0x1234;
        conn.ep_addr = &raw_addr;
        efa_rdm_peer_construct(&peer, efa_rdm_ep, &conn);

        efa_unit_test_eager_msgrtm_pkt_construct(pke, &pke_attr);
        /**
         * Allocate an rxe with the rx pkt.
         * Since there is no recv posted, the rxe must be unexpected
         */
        ofi_genlock_lock(srx_ctx->lock);
        rxe = efa_rdm_msg_alloc_rxe_for_msgrtm(efa_rdm_ep, &peer, &pke);
        assert_true(rxe->state == EFA_RDM_RXE_UNEXP);
        assert_true(rxe->unexp_pkt == pke);

        /* Start progressing the unexpected rxe */
        srx_ctx->peer_srx.peer_ops->start_msg(rxe->peer_rxe);

        /* Make sure rxe is updated as mateched and unexp_pkt is NULL */
        assert_true(rxe->state == EFA_RDM_RXE_MATCHED);
        assert_true(rxe->unexp_pkt == NULL);

        efa_rdm_pke_release_rx(pke);
        efa_rdm_rxe_release(rxe);
        ofi_genlock_unlock(srx_ctx->lock);
}
