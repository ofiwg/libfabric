/*
 * Copyright (c) Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "efa_unit_tests.h"
#include "ofi_util.h"
#include "efa_rdm_ep.h"

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

        efa_unit_test_resource_construct(resource, FI_EP_RDM);

        efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
        srx_ctx = efa_rdm_ep_get_peer_srx_ctx(efa_rdm_ep);
        /*
         * After ep is enabled, the srx->min_multi_recv_size should be
         * exactly the same with ep->min_multi_recv_size
         */
        assert_true(efa_rdm_ep->min_multi_recv_size == srx_ctx->min_multi_recv_size);
        /* Set a new min_multi_recv_size via setopt*/
        min_multi_recv_size_new = 1024;
        assert_int_equal(fi_setopt(&resource->ep->fid, FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
			&min_multi_recv_size_new, sizeof(min_multi_recv_size_new)), 0);

        /* Check whether srx->min_multi_recv_size is set correctly */
        assert_true(srx_ctx->min_multi_recv_size == min_multi_recv_size_new);
}


/* This test verified that cq is correctly bound to srx when it's bound to ep */
void test_efa_srx_cq(struct efa_resource **state)
{
        struct efa_resource *resource = *state;
        struct efa_rdm_ep *efa_rdm_ep;
        struct util_srx_ctx *srx_ctx;

        efa_unit_test_resource_construct(resource, FI_EP_RDM);

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

        efa_unit_test_resource_construct(resource, FI_EP_RDM);

        efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
        srx_ctx = efa_rdm_ep_get_peer_srx_ctx(efa_rdm_ep);
        efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid.fid);
        assert_true(((void *) srx_ctx->lock == (void *) &efa_domain->srx_lock));
}
