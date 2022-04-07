/*
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 * (C) Copyright 2021-2022 Hewlett Packard Enterprise Development LP
 *
 * * This software is available to you under a choice of one of two
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

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <ofi_list.h>
#include <ofi.h>

#include "cxip.h"

#define CXIP_INFO(...) _CXIP_INFO(FI_LOG_EP_CTRL, __VA_ARGS__)

static const char*
cxip_ptelist_to_str(struct cxip_ptelist_bufpool *pool)
{
	return cxi_ptl_list_to_str(pool->attr.list_type);
}

static int cxip_ptelist_unlink_buf(struct cxip_ptelist_buf *buf)
{
	struct cxip_rxc *rxc = buf->rxc;
	int ret;

	ret = cxip_pte_unlink(rxc->rx_pte, buf->pool->attr.list_type,
			      buf->req->req_id, rxc->rx_cmdq);
	if (ret)
		RXC_DBG(rxc, "PtlTE %d failed to write command %d %s\n",
			rxc->rx_pte->pte->ptn, ret, fi_strerror(-ret));

	return ret;
}

static int cxip_ptelist_link_buf(struct cxip_ptelist_buf *buf,
				 bool seq_restart)
{
	struct cxip_rxc *rxc = buf->rxc;
	uint32_t le_flags = C_LE_MANAGE_LOCAL | C_LE_NO_TRUNCATE |
		C_LE_UNRESTRICTED_BODY_RO | C_LE_OP_PUT |
		C_LE_UNRESTRICTED_END_RO | C_LE_EVENT_UNLINK_DISABLE;
	int ret;

	/* Match all eager, long sends */
	union cxip_match_bits mb = {
		.le_type = CXIP_LE_TYPE_RX
	};
	union cxip_match_bits ib = {
		.tag = ~0,
		.tx_id = ~0,
		.cq_data = 1,
		.tagged = 1,
		.match_comp = 1,
	};

	if (!(buf->pool->attr.list_type == C_PTL_LIST_OVERFLOW &&
	      cxip_env.hybrid_preemptive))
		le_flags |= C_LE_EVENT_LINK_DISABLE;

	if (seq_restart)
		le_flags |= C_LE_RESTART_SEQ;

	RXC_DBG(rxc, "PtlTE %d %s link buf %p num linked %u\n",
		rxc->rx_pte->pte->ptn, cxip_ptelist_to_str(buf->pool),
		buf, ofi_atomic_get32(&buf->pool->bufs_linked));

	/* Reset request buffer stats used to know when the buffer is consumed.
	 */
	assert(dlist_empty(&buf->request.pending_ux_list));
	buf->unlink_length = 0;
	buf->cur_offset = 0;

	/* Take a request buffer reference for the link. */
	ret = cxip_pte_append(rxc->rx_pte,
			      CXI_VA_TO_IOVA(buf->md->md, buf->data),
			      buf->pool->attr.buf_size, buf->md->md->lac,
			      buf->pool->attr.list_type,
			      buf->req->req_id, mb.raw,
			      ib.raw, CXI_MATCH_ID_ANY,
			      buf->pool->attr.min_space_avail,
			      le_flags, NULL, rxc->rx_cmdq, true);
	if (ret) {
		RXC_WARN(rxc, "PtlTE %d failed to write %s append %d %s\n",
			 rxc->rx_pte->pte->ptn, cxip_ptelist_to_str(buf->pool),
			 ret, fi_strerror(-ret));
		return ret;
	}

	dlist_remove(&buf->buf_entry);
	dlist_insert_tail(&buf->buf_entry,
			  &buf->pool->active_bufs);
	ofi_atomic_inc32(&buf->pool->bufs_linked);

	/* Reference taken until buffer is consumed or manually
	 * unlinked.
	 */
	cxip_ptelist_buf_get(buf);

	RXC_DBG(rxc, "PtlTE %d APPEND %s buf %p num linked %u\n",
		rxc->rx_pte->pte->ptn, cxip_ptelist_to_str(buf->pool),
		buf, ofi_atomic_get32(&buf->pool->bufs_linked));

	return ret;
}

/*
 * cxip_ptelist_buf_alloc() - Allocate a buffer for the Ptl buffer pool.
 */
static struct cxip_ptelist_buf*
cxip_ptelist_buf_alloc(struct cxip_ptelist_bufpool *pool)
{
	struct cxip_rxc *rxc = pool->rxc;
	struct cxip_ptelist_buf *buf;
	size_t buf_size = sizeof(*buf) + pool->attr.buf_size;
	int ret;

	buf = calloc(1, buf_size);
	if (!buf)
		goto err;

	if (rxc->hmem) {
		ret = ofi_hmem_host_register(buf, buf_size);
		if (ret)
			goto err_free_buf;
	}

	ret = cxip_map(rxc->domain, buf->data, pool->attr.buf_size, &buf->md);
	if (ret)
		goto err_unreg_buf;

	buf->req = cxip_cq_req_alloc(rxc->recv_cq, true, buf);
	if (!buf->req) {
		ret = -FI_ENOMEM;
		goto err_unmap_buf;
	}

	buf->pool = pool;
	buf->req->cb = pool->attr.ptelist_cb;
	buf->rxc = rxc;
	buf->le_type = CXIP_LE_TYPE_RX;

	if (pool->attr.list_type == C_PTL_LIST_REQUEST)
		buf->req->type = CXIP_REQ_RBUF;
	else
		buf->req->type = CXIP_REQ_OFLOW;

	ofi_atomic_initialize32(&buf->refcount, 0);
	dlist_init(&buf->request.pending_ux_list);
	dlist_init(&buf->buf_entry);
	ofi_atomic_inc32(&pool->bufs_allocated);

	RXC_DBG(rxc, "PtlTE %d allocated %s buf %p num alloc %u\n",
		rxc->rx_pte->pte->ptn,
		cxip_ptelist_to_str(buf->pool), buf,
		ofi_atomic_get32(&pool->bufs_allocated));

	return buf;

err_unmap_buf:
	cxip_unmap(buf->md);
err_unreg_buf:
	if (rxc->hmem)
		ofi_hmem_host_unregister(buf);
err_free_buf:
	free(buf);
err:
	return NULL;
}

static void cxip_ptelist_buf_free(struct cxip_ptelist_buf *buf)
{
	struct cxip_ux_send *ux;
	struct dlist_entry *tmp;
	struct cxip_rxc *rxc = buf->rxc;

	/* Sanity check making sure the buffer was properly removed before
	 * freeing.
	 */
	assert(dlist_empty(&buf->buf_entry));

	if (buf->pool->attr.list_type == C_PTL_LIST_REQUEST) {
		dlist_foreach_container_safe(&buf->request.pending_ux_list,
					     struct cxip_ux_send,
					     ux, rxc_entry, tmp) {
			dlist_remove(&ux->rxc_entry);
			_cxip_req_buf_ux_free(ux, false);
		}
	}

	if (ofi_atomic_get32(&buf->refcount) != 0)
		RXC_FATAL(rxc, "PtlTE %d %s buf %p non-zero refcount %u\n",
			  rxc->rx_pte->pte->ptn,
			  cxip_ptelist_to_str(buf->pool), buf,
			  ofi_atomic_get32(&buf->refcount));
	cxip_cq_req_free(buf->req);
	cxip_unmap(buf->md);
	if (rxc->hmem)
		ofi_hmem_host_unregister(buf);

	ofi_atomic_dec32(&buf->pool->bufs_allocated);

	RXC_DBG(rxc, "PtlTE %d freeing %s buf %p num_alloc %u\n",
		rxc->rx_pte->pte->ptn, cxip_ptelist_to_str(buf->pool), buf,
		ofi_atomic_get32(&buf->pool->bufs_allocated));
	free(buf);
}

static void cxip_ptelist_buf_dlist_free(struct dlist_entry *head)
{
	struct cxip_ptelist_buf *buf;
	struct dlist_entry *tmp;

	dlist_foreach_container_safe(head, struct cxip_ptelist_buf, buf,
				     buf_entry, tmp) {
		dlist_remove_init(&buf->buf_entry);
		cxip_ptelist_buf_free(buf);
	}
}

void cxip_ptelist_buf_link_err(struct cxip_ptelist_buf *buf,
			       int rc_link_error)
{
	struct cxip_rxc *rxc = buf->pool->rxc;

	RXC_WARN(rxc, "PtlTE %d %s buffer %p link error %s\n",
		 rxc->rx_pte->pte->ptn, cxip_ptelist_to_str(buf->pool),
		 buf, cxi_rc_to_str(rc_link_error));

	assert(rc_link_error == C_RC_NO_SPACE);

	cxip_ptelist_buf_put(buf, false);
	ofi_atomic_dec32(&buf->pool->bufs_linked);

	/* We are running out of LE resources, do not repost
	 * immediately.
	 */
	assert(ofi_atomic_get32(&buf->refcount) == 0);
	dlist_remove(&buf->buf_entry);
	dlist_insert_tail(&buf->buf_entry, &buf->pool->free_bufs);
	ofi_atomic_inc32(&buf->pool->bufs_free);
}

void cxip_ptelist_buf_unlink(struct cxip_ptelist_buf *buf)
{
	struct cxip_ptelist_bufpool *pool = buf->pool;

	cxip_ptelist_buf_put(buf, false);
	ofi_atomic_dec32(&pool->bufs_linked);

	RXC_DBG(pool->rxc, "PtlTE: %d %s buffer unlink\n",
		pool->rxc->rx_pte->pte->ptn, cxip_ptelist_to_str(pool));
}

int cxip_ptelist_bufpool_init(struct cxip_rxc *rxc,
			      struct cxip_ptelist_bufpool **pool,
			      struct cxip_ptelist_bufpool_attr *attr)
{
	int i;
	struct cxip_ptelist_buf *buf;
	struct dlist_entry tmp_buf_list;
	struct dlist_entry *tmp;
	struct cxip_ptelist_bufpool *_pool;
	int ret;

	if (attr->list_type != C_PTL_LIST_REQUEST &&
	    attr->list_type != C_PTL_LIST_OVERFLOW)
		return -FI_EINVAL;

	_pool = calloc(1, sizeof(*_pool));
	if (!_pool)
		return -FI_ENOMEM;

	_pool->attr = *attr;
	_pool->rxc = rxc;
	dlist_init(&_pool->active_bufs);
	dlist_init(&_pool->consumed_bufs);
	dlist_init(&_pool->free_bufs);
	ofi_atomic_initialize32(&_pool->bufs_linked, 0);
	ofi_atomic_initialize32(&_pool->bufs_allocated, 0);
	ofi_atomic_initialize32(&_pool->bufs_free, 0);

	dlist_init(&tmp_buf_list);

	for (i = 0; i < _pool->attr.min_posted; i++) {
		buf = cxip_ptelist_buf_alloc(_pool);
		if (!buf) {
			ret = -FI_ENOMEM;
			goto err_free_bufs;
		}

		dlist_insert_tail(&buf->buf_entry, &tmp_buf_list);
	}

	/* Since this is called during RXC initialization, RXQ CMDQ should be
	 * empty. Thus, linking a request buffer should not fail.
	 */
	dlist_foreach_container_safe(&tmp_buf_list, struct cxip_ptelist_buf,
				     buf, buf_entry, tmp) {
		ret = cxip_ptelist_link_buf(buf, false);
		if (ret != FI_SUCCESS)
			CXIP_FATAL("Failed to link request buffer: %d %s\n",
				   ret, fi_strerror(-ret));
	}

	*pool = _pool;
	return FI_SUCCESS;

err_free_bufs:
	cxip_ptelist_buf_dlist_free(&tmp_buf_list);

	return ret;
}

void cxip_ptelist_bufpool_fini(struct cxip_ptelist_bufpool *pool)
{
	struct cxip_rxc *rxc = pool->rxc;
	struct cxip_ptelist_buf *buf;
	int ret;

	assert(rxc->rx_pte->state == C_PTLTE_DISABLED);

	CXIP_INFO("PtlTE %d number of %s buffers allocated %d\n",
		  rxc->rx_pte->pte->ptn, cxip_ptelist_to_str(pool),
		  ofi_atomic_get32(&pool->bufs_allocated));

	/* All request buffers are split between the active and consumed list.
	 * Only active buffers need to be unlinked.
	 */
	dlist_foreach_container(&pool->active_bufs, struct cxip_ptelist_buf,
				buf, buf_entry) {
		ret = cxip_ptelist_unlink_buf(buf);
		if (ret != FI_SUCCESS)
			CXIP_FATAL("PtlTE %d failed to unlink %s buf %d %s\n",
				   rxc->rx_pte->pte->ptn,
				   cxip_ptelist_to_str(pool), ret,
				   fi_strerror(-ret));
	}

	do {
		cxip_cq_progress(rxc->recv_cq);
	} while (ofi_atomic_get32(&pool->bufs_linked));

	cxip_ptelist_buf_dlist_free(&pool->active_bufs);
	cxip_ptelist_buf_dlist_free(&pool->consumed_bufs);
	cxip_ptelist_buf_dlist_free(&pool->free_bufs);

	assert(ofi_atomic_get32(&pool->bufs_allocated) == 0);

	assert(pool);
	free(pool);
}

/*
 * cxip_ptelist_buf_replenish() - Replenish PtlTE overflow or request list
 * buffers.
 *
 * Caller must hold rxc->rx_lock.
 */
int cxip_ptelist_buf_replenish(struct cxip_ptelist_bufpool *pool,
			       bool seq_restart)
{
	struct cxip_rxc *rxc = pool->rxc;
	struct cxip_ptelist_buf *buf;
	int bufs_added = 0;
	int ret = FI_SUCCESS;

	if (rxc->msg_offload && pool->attr.list_type == C_PTL_LIST_REQUEST)
		return FI_SUCCESS;

	while (ofi_atomic_get32(&pool->bufs_linked) < pool->attr.min_posted) {

		/* Always prefer to use a free buffer for which
		 * reposting was deferred or an append failed.
		 */
		if (!dlist_empty(&pool->free_bufs)) {
			dlist_pop_front(&pool->free_bufs,
					struct cxip_ptelist_buf, buf,
					buf_entry);
			ofi_atomic_dec32(&buf->pool->bufs_free);

			RXC_DBG(rxc, "PtlTE %d %s LINK REPOST buf %p\n",
				rxc->rx_pte->pte->ptn,
				cxip_ptelist_to_str(pool), buf);
		} else {
			buf = cxip_ptelist_buf_alloc(pool);

			RXC_DBG(rxc, "PtlTE %d %s LINK NEW buf %p\n",
				rxc->rx_pte->pte->ptn,
				cxip_ptelist_to_str(pool), buf);

		}

		if (!buf) {
			RXC_WARN(rxc, "PtlTE %d %s buffer allocation err\n",
				 rxc->rx_pte->pte->ptn,
				 cxip_ptelist_to_str(pool));
			break;
		}

		RXC_DBG(rxc, "PtlTE %d link %s buf entry %p\n",
			rxc->rx_pte->pte->ptn, cxip_ptelist_to_str(pool),
			buf);

		ret = cxip_ptelist_link_buf(buf, !bufs_added);
		if (ret) {
			RXC_WARN(rxc, "PtlTE %d %s append failure %d %s\n",
				 rxc->rx_pte->pte->ptn,
				 cxip_ptelist_to_str(pool), ret,
				 fi_strerror(-ret));

			dlist_insert_tail(&buf->buf_entry,
					  &pool->free_bufs);
			ofi_atomic_inc32(&pool->bufs_free);
			break;
		}
		bufs_added++;
	}

	/* If no buffer appended, check for fatal conditions. */
	if (!bufs_added) {
		if (ofi_atomic_get32(&pool->bufs_linked) < 1)
			RXC_FATAL(rxc, "PtlTE %d %s buffer list exhausted\n",
				  rxc->rx_pte->pte->ptn,
				  cxip_ptelist_to_str(pool));
	}

	RXC_DBG(rxc, "PtlTE %d %s current bufs alloc %u, num linked %u\n",
		rxc->rx_pte->pte->ptn, cxip_ptelist_to_str(pool),
		ofi_atomic_get32(&pool->bufs_allocated),
		ofi_atomic_get32(&pool->bufs_linked));

	return ret;
}

void cxip_ptelist_buf_get(struct cxip_ptelist_buf *buf)
{
	ofi_atomic_inc32(&buf->refcount);

	RXC_DBG(buf->rxc, "PtlTE %d %s GET buf %p refcnt %u\n",
		buf->rxc->rx_pte->pte->ptn,
		cxip_ptelist_to_str(buf->pool),
		buf, ofi_atomic_get32(&buf->refcount));
}

void cxip_ptelist_buf_put(struct cxip_ptelist_buf *buf, bool repost)
{
	int ret;
	int refcount = ofi_atomic_dec32(&buf->refcount);

	RXC_DBG(buf->rxc, "PtlTE %d %s PUT buf %p refcnt %u repost %d\n",
		buf->rxc->rx_pte->pte->ptn,
		cxip_ptelist_to_str(buf->pool), buf, refcount, repost);

	if (refcount < 0) {
		RXC_FATAL(buf->rxc,
			  "PtlTE %d %s buffer refcount underflow %d\n",
			  buf->rxc->rx_pte->pte->ptn,
			  cxip_ptelist_to_str(buf->pool), refcount);
		/* not needed */
		return;
	}

	if (refcount == 0 && repost) {

		/* Overflow buffers should just be freed if no longer
		 * in hardware RX match mode.
		 */
		if (buf->pool->attr.list_type == C_PTL_LIST_OVERFLOW &&
		    (!buf->rxc->msg_offload || buf->rxc->state != RXC_ENABLED))
			goto free_buf;

		/* Limit immediate repost if already sufficient */
		if (ofi_atomic_get32(&buf->pool->bufs_linked) <
		    buf->pool->attr.max_posted) {

			do {
				ret = cxip_ptelist_link_buf(buf, false);
			} while (ret == -FI_EAGAIN);

			if (ret != FI_SUCCESS)
				RXC_FATAL(buf->rxc,
					"PtlTE %d fatal %s buf link err %d %s",
					buf->rxc->rx_pte->pte->ptn,
					cxip_ptelist_to_str(buf->pool),
					ret, fi_strerror(-ret));

			return;
		}

		/* To avoid thrashing on buffer allocation, cache
		 * free buffers until a sufficient number are kept
		 * for reuse. This will help bursty traffic from
		 * holding on to unnecessary buffers.
		 */

		/* TODO: To allow for the number of REQUEST list buffers to
		 * shrink we need to address cxip_req_buf_progress_pending_ux()
		 * to recognize the buffer can be unreferenced within the
		 * loop by the function cxip_req_buf_ux_free().
		 */
		if (buf->pool->attr.list_type == C_PTL_LIST_REQUEST ||
		    !buf->pool->attr.max_cached ||
		    ofi_atomic_get32(&buf->pool->bufs_allocated) <
		    buf->pool->attr.max_cached) {

			dlist_remove(&buf->buf_entry);
			dlist_insert_tail(&buf->buf_entry,
					&buf->pool->free_bufs);
			ofi_atomic_inc32(&buf->pool->bufs_free);

			return;
		}

free_buf:
		dlist_remove_init(&buf->buf_entry);
		cxip_ptelist_buf_free(buf);
	}
}

void cxip_ptelist_buf_consumed(struct cxip_ptelist_buf *buf)
{
	RXC_DBG(buf->rxc, "PtlTE %d %s CONSUMED off %ld len %ld buf %p\n",
		buf->rxc->rx_pte->pte->ptn,
		cxip_ptelist_to_str(buf->pool), buf->cur_offset,
		buf->unlink_length, buf);

	dlist_remove(&buf->buf_entry);
	dlist_insert_tail(&buf->buf_entry,
			  &buf->pool->consumed_bufs);

	/* Since buffer is consumed, return reference
	 * taken during the initial linking.
	 */
	cxip_ptelist_buf_put(buf, true);
}

void _cxip_req_buf_ux_free(struct cxip_ux_send *ux, bool repost)
{
	struct cxip_ptelist_buf *buf = ux->req->req_ctx;

	assert(ux->req->type == CXIP_REQ_RBUF);

	cxip_ptelist_buf_put(buf, repost);
	free(ux);

	RXC_DBG(buf->rxc, "PtlTE %d %s buf %p ux %p\n",
		buf->rxc->rx_pte->pte->ptn,
		cxip_ptelist_to_str(buf->pool), buf, ux);
}
