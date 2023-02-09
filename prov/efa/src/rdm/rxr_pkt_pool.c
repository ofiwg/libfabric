/*
 * Copyright (c) 2019-2022 Amazon.com, Inc. or its affiliates.
 * All rights reserved.
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

#include "efa.h"

static int rxr_pkt_pool_mr_reg_hndlr(struct ofi_bufpool_region *region)
{
	size_t ret;
	struct fid_mr *mr;
	struct efa_domain *domain = region->pool->attr.context;

	ret = fi_mr_reg(&domain->util_domain.domain_fid, region->alloc_region,
			region->pool->alloc_size, FI_SEND | FI_RECV, 0, 0, 0,
			&mr, NULL);

	region->context = mr;
	return ret;
}

static void rxr_pkt_pool_mr_dereg_hndlr(struct ofi_bufpool_region *region)
{
	ssize_t ret;

	ret = fi_close((struct fid *)region->context);
	if (ret)
		EFA_WARN(FI_LOG_EP_CTRL,
			"Unable to deregister memory in a buf pool: %s\n",
			fi_strerror(-ret));
}

size_t rxr_pkt_pool_mr_flags()
{
	if (g_efa_fork_status == EFA_FORK_SUPPORT_ON) {
		/*
		 * Make sure that no data structures can share the memory pages used
		 * for this buffer pool.
		 * When fork support is on, registering a buffer with ibv_reg_mr will
		 * set MADV_DONTFORK on the underlying pages.  After fork() the child
		 * process will not have a page mapping at that address.
		 */
		return OFI_BUFPOOL_NONSHARED;
	}

	/* use huge page reduce the number of registered pages for the same amount of memory.
	 */
	return OFI_BUFPOOL_HUGEPAGES;
}

/*
 * rxr_pkt_pool_create creates a packet pool. The pool is allowed to grow if
 * max_cnt is 0 and is fixed size otherwise.
 *
 * Important arguments:
 * 	    mr: whether memory registration for the wiredata pool is required
 */
int rxr_pkt_pool_create(struct rxr_ep *ep,
			size_t chunk_cnt, size_t max_cnt,
			bool mr, bool with_sendv_pool,
			struct rxr_pkt_pool **pkt_pool)
{
	int ret;
	struct rxr_pkt_pool *pool;

	pool = calloc(1, sizeof(**pkt_pool));
	if (!pool)
		return -FI_ENOMEM;

	struct ofi_bufpool_attr wiredata_attr = {
		.size = sizeof(struct rxr_pkt_entry) + ep->mtu_size,
		.alignment = RXR_BUF_POOL_ALIGNMENT,
		.max_cnt = max_cnt,
		.chunk_cnt = chunk_cnt,
		.alloc_fn = mr ? rxr_pkt_pool_mr_reg_hndlr : NULL,
		.free_fn = mr ? rxr_pkt_pool_mr_dereg_hndlr : NULL,
		.init_fn = NULL,
		.context = rxr_ep_domain(ep),
		.flags = mr ? rxr_pkt_pool_mr_flags() : 0,
	};

	ret = ofi_bufpool_create_attr(&wiredata_attr, &pool->entry_pool);
	if (ret) {
		free(pool);
		return ret;
	}

	if (with_sendv_pool) {
		ret = ofi_bufpool_create(&pool->sendv_pool, sizeof(struct rxr_pkt_sendv),
					 RXR_BUF_POOL_ALIGNMENT, max_cnt, chunk_cnt, 0);
		if (ret) {
			ofi_bufpool_destroy(pool->entry_pool);
			free(pool);
			return ret;
		}
	}

	*pkt_pool = pool;
	return 0;
}

int rxr_pkt_pool_grow(struct rxr_pkt_pool *rxr_pkt_pool)
{
	int err;

	err = ofi_bufpool_grow(rxr_pkt_pool->entry_pool);
	if (err)
		return err;

	if (rxr_pkt_pool->sendv_pool) {
		err = ofi_bufpool_grow(rxr_pkt_pool->sendv_pool);
		if (err)
			return err;
	}

	return 0;
}

/*
 * rxr_pkt_pool_destroy frees the packet pool
 */
void rxr_pkt_pool_destroy(struct rxr_pkt_pool *pkt_pool)
{
	ofi_bufpool_destroy(pkt_pool->entry_pool);
	if (pkt_pool->sendv_pool)
		ofi_bufpool_destroy(pkt_pool->sendv_pool);
	free(pkt_pool);
}
