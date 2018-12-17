/*
 * Copyright (c) 2013-2016 Intel Corporation. All rights reserved.
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

#include <inttypes.h>
#include <math.h>

#include "ofi.h"
#include <ofi_util.h>

#include "rxm.h"

static int rxm_match_noop(struct dlist_entry *item, const void *arg)
{
	OFI_UNUSED(item);
	OFI_UNUSED(arg);
	return 1;
}

static int rxm_match_recv_entry(struct dlist_entry *item, const void *arg)
{
	struct rxm_recv_match_attr *attr = (struct rxm_recv_match_attr *) arg;
	struct rxm_recv_entry *recv_entry =
		container_of(item, struct rxm_recv_entry, entry);
	return ofi_match_addr(recv_entry->addr, attr->addr);
}

static int rxm_match_recv_entry_tag(struct dlist_entry *item, const void *arg)
{
	struct rxm_recv_match_attr *attr = (struct rxm_recv_match_attr *)arg;
	struct rxm_recv_entry *recv_entry =
		container_of(item, struct rxm_recv_entry, entry);
	return ofi_match_tag(recv_entry->tag, recv_entry->ignore, attr->tag);
}

static int rxm_match_recv_entry_tag_addr(struct dlist_entry *item, const void *arg)
{
	struct rxm_recv_match_attr *attr = (struct rxm_recv_match_attr *)arg;
	struct rxm_recv_entry *recv_entry =
		container_of(item, struct rxm_recv_entry, entry);
	return ofi_match_addr(recv_entry->addr, attr->addr) &&
		ofi_match_tag(recv_entry->tag, recv_entry->ignore, attr->tag);
}

static int rxm_match_recv_entry_context(struct dlist_entry *item, const void *context)
{
	struct rxm_recv_entry *recv_entry =
		container_of(item, struct rxm_recv_entry, entry);
	return recv_entry->context == context;
}

static int rxm_match_unexp_msg(struct dlist_entry *item, const void *arg)
{
	struct rxm_recv_match_attr *attr = (struct rxm_recv_match_attr *)arg;
	struct rxm_unexp_msg *unexp_msg =
		container_of(item, struct rxm_unexp_msg, entry);
	return ofi_match_addr(attr->addr, unexp_msg->addr);
}

static int rxm_match_unexp_msg_tag(struct dlist_entry *item, const void *arg)
{
	struct rxm_recv_match_attr *attr = (struct rxm_recv_match_attr *)arg;
	struct rxm_unexp_msg *unexp_msg =
		container_of(item, struct rxm_unexp_msg, entry);
	return ofi_match_tag(attr->tag, attr->ignore, unexp_msg->tag);
}

static int rxm_match_unexp_msg_tag_addr(struct dlist_entry *item, const void *arg)
{
	struct rxm_recv_match_attr *attr = (struct rxm_recv_match_attr *)arg;
	struct rxm_unexp_msg *unexp_msg =
		container_of(item, struct rxm_unexp_msg, entry);
	return ofi_match_addr(attr->addr, unexp_msg->addr) &&
		ofi_match_tag(attr->tag, attr->ignore, unexp_msg->tag);
}

static inline int
rxm_mr_buf_reg(struct rxm_ep *rxm_ep, void *addr, size_t len, void **context)
{
	int ret = FI_SUCCESS;
	struct fid_mr *mr;
	struct rxm_domain *rxm_domain = container_of(rxm_ep->util_ep.domain,
						     struct rxm_domain, util_domain);

	*context = NULL;
	if (rxm_ep->msg_mr_local) {
		struct fid_domain *msg_domain =
			(struct fid_domain *)rxm_domain->msg_domain;

		ret = fi_mr_reg(msg_domain, addr, len,
				FI_SEND | FI_RECV | FI_READ | FI_WRITE,
				0, 0, 0, &mr, NULL);
		*context = mr;
	}

	return ret;
}

static void rxm_buf_reg_set_common(struct rxm_buf *hdr, struct rxm_pkt *pkt,
				   uint8_t type, void *mr_desc)
{
	if (pkt) {
		pkt->ctrl_hdr.version = RXM_CTRL_VERSION;
		pkt->hdr.version = OFI_OP_VERSION;
		pkt->ctrl_hdr.type = type;
	}
	if (hdr) {
		hdr->desc = mr_desc;
	}
}

static int rxm_buf_reg(void *pool_ctx, void *addr, size_t len, void **context)
{
	struct rxm_buf_pool *pool = (struct rxm_buf_pool *)pool_ctx;
	size_t i, entry_sz = pool->pool->entry_sz;
	int ret;
	void *mr_desc;
	uint8_t type;
	struct rxm_buf *hdr;
	struct rxm_pkt *pkt;
	struct rxm_rx_buf *rx_buf;
	struct rxm_tx_base_buf *tx_base_buf;
	struct rxm_tx_eager_buf *tx_eager_buf;
	struct rxm_tx_sar_buf *tx_sar_buf;
	struct rxm_tx_rndv_buf *tx_rndv_buf;
	struct rxm_tx_atomic_buf *tx_atomic_buf;
	struct rxm_rma_buf *rma_buf;

	if ((pool->type != RXM_BUF_POOL_TX_INJECT) && pool->rxm_ep->msg_mr_local) {
		ret = rxm_mr_buf_reg(pool->rxm_ep, addr, len, context);
		if (ret)
			return ret;
		mr_desc = fi_mr_desc((struct fid_mr *)*context);
	} else {
		*context = mr_desc = NULL;
	}

	for (i = 0; i < pool->pool->attr.chunk_cnt; i++) {
		switch (pool->type) {
		case RXM_BUF_POOL_RX:
			rx_buf = (struct rxm_rx_buf *)
				((char *)addr + i * entry_sz);
			rx_buf->ep = pool->rxm_ep;

			hdr = &rx_buf->hdr;
			pkt = NULL;
			type = ofi_ctrl_data; /* This can be any value */
			break;
		case RXM_BUF_POOL_TX:
			tx_eager_buf = (struct rxm_tx_eager_buf *)
				((char *)addr + i * entry_sz);
			tx_eager_buf->hdr.state = RXM_TX;

			hdr = &tx_eager_buf->hdr;
			pkt = &tx_eager_buf->pkt;
			type = ofi_ctrl_data;
			break;
		case RXM_BUF_POOL_TX_INJECT:
			tx_base_buf = (struct rxm_tx_base_buf *)
				((char *)addr + i * entry_sz);
			tx_base_buf->hdr.state = RXM_INJECT_TX;

			hdr = NULL;
			pkt = &tx_base_buf->pkt;
			type = ofi_ctrl_data;
			break;
		case RXM_BUF_POOL_TX_SAR:
			tx_sar_buf = (struct rxm_tx_sar_buf *)
				((char *)addr + i * entry_sz);
			tx_sar_buf->hdr.state = RXM_SAR_TX;

			hdr = &tx_sar_buf->hdr;
			pkt = &tx_sar_buf->pkt;
			type = ofi_ctrl_seg_data;
			break;
		case RXM_BUF_POOL_TX_RNDV:
			tx_rndv_buf = (struct rxm_tx_rndv_buf *)
				((char *)addr + i * entry_sz);

			hdr = &tx_rndv_buf->hdr;
			pkt = &tx_rndv_buf->pkt;
			type = ofi_ctrl_large_data;
			break;
		case RXM_BUF_POOL_TX_ATOMIC:
			tx_atomic_buf = (struct rxm_tx_atomic_buf *)
				((char *)addr + i * entry_sz);

			hdr = &tx_atomic_buf->hdr;
			pkt = &tx_atomic_buf->pkt;
			type = ofi_ctrl_atomic;
			break;
		case RXM_BUF_POOL_TX_ACK:
			tx_base_buf = (struct rxm_tx_base_buf *)
				((char *)addr + i * entry_sz);
			tx_base_buf->pkt.hdr.op = ofi_op_msg;

			hdr = &tx_base_buf->hdr;
			pkt = &tx_base_buf->pkt;
			type = ofi_ctrl_ack;
			break;
		case RXM_BUF_POOL_RMA:
			rma_buf = (struct rxm_rma_buf *)
				((char *)addr + i * entry_sz);
			rma_buf->pkt.hdr.op = ofi_op_msg;
			rma_buf->hdr.state = RXM_RMA;

			hdr = &rma_buf->hdr;
			pkt = &rma_buf->pkt;
			type = ofi_ctrl_data;
			break;
		default:
			assert(0);
			hdr = NULL;
			pkt = NULL;
			mr_desc = NULL;
			type = ofi_ctrl_data;
			break;
		}
		rxm_buf_reg_set_common(hdr, pkt, type, mr_desc);
	}

	return FI_SUCCESS;
}

static inline void rxm_buf_close(void *pool_ctx, void *context)
{
	struct rxm_buf_pool *pool = (struct rxm_buf_pool *)pool_ctx;
	struct rxm_ep *rxm_ep = pool->rxm_ep;

	if ((rxm_ep->msg_mr_local) && (pool->type != RXM_BUF_POOL_TX_INJECT)) {
		/* We would get a (fid_mr *) in context but
		 * it is safe to cast it into (fid *) */
		fi_close((struct fid *)context);
	}
}

static void rxm_buf_pool_destroy(struct rxm_buf_pool *pool)
{
	/* This indicates whether the pool is allocated or not */
	if (pool->rxm_ep) {
		util_buf_pool_destroy(pool->pool);
	}
}

static int rxm_buf_pool_create(struct rxm_ep *rxm_ep,
			       size_t chunk_count, size_t size,
			       struct rxm_buf_pool *pool,
			       enum rxm_buf_pool_type type)
{
	struct util_buf_attr attr = {
		.size		= size,
		.alignment	= 16,
		.max_cnt	= 0,
		.chunk_cnt	= chunk_count,
		.alloc_hndlr	= rxm_buf_reg,
		.free_hndlr	= rxm_buf_close,
		.ctx		= pool,
		.track_used	= 0,
	};
	int ret;

	switch (type) {
	case RXM_BUF_POOL_TX_RNDV:
	case RXM_BUF_POOL_TX_ATOMIC:
	case RXM_BUF_POOL_TX_SAR:
		attr.indexing.used = 1;
		break;
	default:
		attr.indexing.used = 0;
		break;
	}

	pool->rxm_ep = rxm_ep;
	pool->type = type;
	ret = util_buf_pool_create_attr(&attr, &pool->pool);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to create buf pool\n");
		return -FI_ENOMEM;
	}

	return 0;
}

static void rxm_recv_entry_init(struct rxm_recv_entry *entry, void *arg)
{
	struct rxm_recv_queue *recv_queue = arg;

	assert(recv_queue->type != RXM_RECV_QUEUE_UNSPEC);

	entry->recv_queue = recv_queue;
	entry->sar.msg_id = RXM_SAR_RX_INIT;
	entry->sar.total_recv_len = 0;
	entry->comp_flags = FI_RECV;

	if (recv_queue->type == RXM_RECV_QUEUE_MSG)
		entry->comp_flags |= FI_MSG;
	else
		entry->comp_flags |= FI_TAGGED;
}

static int rxm_recv_queue_init(struct rxm_ep *rxm_ep,  struct rxm_recv_queue *recv_queue,
			       size_t size, enum rxm_recv_queue_type type)
{
	recv_queue->rxm_ep = rxm_ep;
	recv_queue->type = type;
	recv_queue->fs = rxm_recv_fs_create(size, rxm_recv_entry_init, recv_queue);
	if (!recv_queue->fs)
		return -FI_ENOMEM;

	dlist_init(&recv_queue->recv_list);
	dlist_init(&recv_queue->unexp_msg_list);
	if (type == RXM_RECV_QUEUE_MSG) {
		if (rxm_ep->rxm_info->caps & FI_DIRECTED_RECV) {
			recv_queue->match_recv = rxm_match_recv_entry;
			recv_queue->match_unexp = rxm_match_unexp_msg;
		} else {
			recv_queue->match_recv = rxm_match_noop;
			recv_queue->match_unexp = rxm_match_noop;
		}
	} else {
		if (rxm_ep->rxm_info->caps & FI_DIRECTED_RECV) {
			recv_queue->match_recv = rxm_match_recv_entry_tag_addr;
			recv_queue->match_unexp = rxm_match_unexp_msg_tag_addr;
		} else {
			recv_queue->match_recv = rxm_match_recv_entry_tag;
			recv_queue->match_unexp = rxm_match_unexp_msg_tag;
		}
	}

	return 0;
}

static void rxm_recv_queue_close(struct rxm_recv_queue *recv_queue)
{
	/* It indicates that the recv_queue were allocated */
	if (recv_queue->fs) {
		rxm_recv_fs_free(recv_queue->fs);
	}
	// TODO cleanup recv_list and unexp msg list
}

static int rxm_ep_txrx_pool_create(struct rxm_ep *rxm_ep)
{
	int ret, i;
	size_t queue_sizes[] = {
		[RXM_BUF_POOL_RX] = rxm_ep->msg_info->rx_attr->size,
		[RXM_BUF_POOL_TX] = rxm_ep->msg_info->tx_attr->size,
		[RXM_BUF_POOL_TX_INJECT] = rxm_ep->msg_info->tx_attr->size,
		[RXM_BUF_POOL_TX_ACK] = rxm_ep->msg_info->tx_attr->size,
		[RXM_BUF_POOL_TX_RNDV] = rxm_ep->msg_info->tx_attr->size,
		[RXM_BUF_POOL_TX_ATOMIC] = rxm_ep->msg_info->tx_attr->size,
		[RXM_BUF_POOL_TX_SAR] = rxm_ep->msg_info->tx_attr->size,
		[RXM_BUF_POOL_RMA] = rxm_ep->msg_info->tx_attr->size,
	};
	size_t entry_sizes[] = {		
		[RXM_BUF_POOL_RX] = rxm_ep->eager_limit +
				    sizeof(struct rxm_rx_buf),
		[RXM_BUF_POOL_TX] = rxm_ep->eager_limit +
				    sizeof(struct rxm_tx_eager_buf),
		[RXM_BUF_POOL_TX_INJECT] = rxm_ep->inject_limit +
					   sizeof(struct rxm_tx_base_buf),
		[RXM_BUF_POOL_TX_ACK] = sizeof(struct rxm_tx_base_buf),
		[RXM_BUF_POOL_TX_RNDV] = sizeof(struct rxm_rndv_hdr) +
					 rxm_ep->buffered_min +
					 sizeof(struct rxm_tx_rndv_buf),
		[RXM_BUF_POOL_TX_ATOMIC] = rxm_ep->eager_limit +
					 sizeof(struct rxm_tx_atomic_buf),
		[RXM_BUF_POOL_TX_SAR] = rxm_ep->eager_limit +
					sizeof(struct rxm_tx_sar_buf),
		[RXM_BUF_POOL_RMA] = rxm_ep->eager_limit +
				     sizeof(struct rxm_rma_buf),
	};

	dlist_init(&rxm_ep->repost_ready_list);

	rxm_ep->buf_pools = calloc(1, RXM_BUF_POOL_MAX * sizeof(*rxm_ep->buf_pools));
	if (!rxm_ep->buf_pools)
		return -FI_ENOMEM;

	for (i = RXM_BUF_POOL_START; i < RXM_BUF_POOL_MAX; i++) {
		if ((i == RXM_BUF_POOL_TX_INJECT) &&
		    (rxm_ep->util_ep.domain->threading != FI_THREAD_SAFE))
			continue;

		ret = rxm_buf_pool_create(rxm_ep, queue_sizes[i], entry_sizes[i],
					  &rxm_ep->buf_pools[i], i);
		if (ret)
			goto err;
	}

	return FI_SUCCESS;
err:
	while (--i >= RXM_BUF_POOL_START)
		rxm_buf_pool_destroy(&rxm_ep->buf_pools[i]);
	free(rxm_ep->buf_pools);
	return ret;
}

static void rxm_ep_txrx_pool_destroy(struct rxm_ep *rxm_ep)
{
	size_t i;

	for (i = RXM_BUF_POOL_START; i < RXM_BUF_POOL_MAX; i++)
		rxm_buf_pool_destroy(&rxm_ep->buf_pools[i]);
	free(rxm_ep->buf_pools);
}

static int rxm_ep_rx_queue_init(struct rxm_ep *rxm_ep)
{
	int ret;

	ret = rxm_recv_queue_init(rxm_ep, &rxm_ep->recv_queue,
				  rxm_ep->rxm_info->rx_attr->size,
				  RXM_RECV_QUEUE_MSG);
	if (ret)
		return ret;

	ret = rxm_recv_queue_init(rxm_ep, &rxm_ep->trecv_queue,
				  rxm_ep->rxm_info->rx_attr->size,
				  RXM_RECV_QUEUE_TAGGED);
	if (ret)
		goto err_recv_tag;

	return FI_SUCCESS;
err_recv_tag:
	rxm_recv_queue_close(&rxm_ep->recv_queue);
	return ret;
}

static void rxm_ep_rx_queue_close(struct rxm_ep *rxm_ep)
{
	rxm_recv_queue_close(&rxm_ep->trecv_queue);
	rxm_recv_queue_close(&rxm_ep->recv_queue);
}

/* It is safe to call this function, even if `rxm_ep_txrx_res_open`
 * has not yet been called */
static void rxm_ep_txrx_res_close(struct rxm_ep *rxm_ep)
{
	rxm_ep_rx_queue_close(rxm_ep);
	if (rxm_ep->buf_pools)
		rxm_ep_txrx_pool_destroy(rxm_ep);
}

static int rxm_setname(fid_t fid, void *addr, size_t addrlen)
{
	struct rxm_ep *rxm_ep;

	rxm_ep = container_of(fid, struct rxm_ep, util_ep.ep_fid.fid);
	return fi_setname(&rxm_ep->msg_pep->fid, addr, addrlen);
}

static int rxm_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct rxm_ep *rxm_ep;

	rxm_ep = container_of(fid, struct rxm_ep, util_ep.ep_fid.fid);
	return fi_getname(&rxm_ep->msg_pep->fid, addr, addrlen);
}

static struct fi_ops_cm rxm_ops_cm = {
	.size = sizeof(struct fi_ops_cm),
	.setname = rxm_setname,
	.getname = rxm_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

static int rxm_ep_cancel_recv(struct rxm_ep *rxm_ep,
			      struct rxm_recv_queue *recv_queue, void *context)
{
	struct fi_cq_err_entry err_entry;
	struct rxm_recv_entry *recv_entry;
	struct dlist_entry *entry;
	int ret;

	ofi_ep_lock_acquire(&rxm_ep->util_ep);
	entry = dlist_remove_first_match(&recv_queue->recv_list,
					 rxm_match_recv_entry_context,
					 context);
	if (entry) {
		recv_entry = container_of(entry, struct rxm_recv_entry, entry);
		memset(&err_entry, 0, sizeof(err_entry));
		err_entry.op_context = recv_entry->context;
		err_entry.flags |= recv_entry->comp_flags;
		err_entry.tag = recv_entry->tag;
		err_entry.err = FI_ECANCELED;
		err_entry.prov_errno = -FI_ECANCELED;
		rxm_recv_entry_release(recv_queue, recv_entry);
		ret = ofi_cq_write_error(rxm_ep->util_ep.rx_cq, &err_entry);
	} else {
		ret = 0;
	}
	ofi_ep_lock_release(&rxm_ep->util_ep);
	return ret;
}

static ssize_t rxm_ep_cancel(fid_t fid_ep, void *context)
{
	struct rxm_ep *rxm_ep = container_of(fid_ep, struct rxm_ep, util_ep.ep_fid);
	int ret;

	ret = rxm_ep_cancel_recv(rxm_ep, &rxm_ep->recv_queue, context);
	if (ret)
		return ret;

	ret = rxm_ep_cancel_recv(rxm_ep, &rxm_ep->trecv_queue, context);
	if (ret)
		return ret;

	return 0;
}

static int rxm_ep_getopt(fid_t fid, int level, int optname, void *optval,
			 size_t *optlen)
{
	struct rxm_ep *rxm_ep =
		container_of(fid, struct rxm_ep, util_ep.ep_fid);

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_MIN_MULTI_RECV:
		assert(sizeof(rxm_ep->min_multi_recv_size) == sizeof(size_t));
		*(size_t *)optval = rxm_ep->min_multi_recv_size;
		*optlen = sizeof(size_t);
		break;
	case FI_OPT_BUFFERED_MIN:
		assert(sizeof(rxm_ep->buffered_min) == sizeof(size_t));
		*(size_t *)optval = rxm_ep->buffered_min;
		*optlen = sizeof(size_t);
		break;
	case FI_OPT_BUFFERED_LIMIT:
		assert(sizeof(rxm_ep->buffered_limit) == sizeof(size_t));
		*(size_t *)optval = rxm_ep->buffered_limit;
		*optlen = sizeof(size_t);
		break;
	default:
		return -FI_ENOPROTOOPT;
	}
	return FI_SUCCESS;
}

static int rxm_ep_setopt(fid_t fid, int level, int optname,
			 const void *optval, size_t optlen)
{
	struct rxm_ep *rxm_ep =
		container_of(fid, struct rxm_ep, util_ep.ep_fid);
	int ret = FI_SUCCESS;

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_MIN_MULTI_RECV:
		rxm_ep->min_multi_recv_size = *(size_t *)optval;
		break;
	case FI_OPT_BUFFERED_MIN:
		if (rxm_ep->buf_pools) {
			FI_WARN(&rxm_prov, FI_LOG_EP_DATA,
				"Endpoint already enabled. Can't set opt now!\n");
			ret = -FI_EOPBADSTATE;
		} else if (*(size_t *)optval > rxm_ep->buffered_limit) {
			FI_WARN(&rxm_prov, FI_LOG_EP_DATA,
			"Invalid value for FI_OPT_BUFFERED_MIN: %zu "
			"( > FI_OPT_BUFFERED_LIMIT: %zu)\n",
			*(size_t *)optval, rxm_ep->buffered_limit);
			ret = -FI_EINVAL;
		} else {
			rxm_ep->buffered_min = *(size_t *)optval;
		}
		break;
	case FI_OPT_BUFFERED_LIMIT:
		if (rxm_ep->buf_pools) {
			FI_WARN(&rxm_prov, FI_LOG_EP_DATA,
				"Endpoint already enabled. Can't set opt now!\n");
			ret = -FI_EOPBADSTATE;
		/* We do not check for maximum as we allow sizes up to SIZE_MAX */
		} else if (*(size_t *)optval < rxm_ep->buffered_min) {
			FI_WARN(&rxm_prov, FI_LOG_EP_DATA,
			"Invalid value for FI_OPT_BUFFERED_LIMIT: %zu"
			" ( < FI_OPT_BUFFERED_MIN: %zu)\n",
			*(size_t *)optval, rxm_ep->buffered_min);
			ret = -FI_EINVAL;
		} else {
			rxm_ep->buffered_limit = *(size_t *)optval;
		}
		break;
	default:
		ret = -FI_ENOPROTOOPT;
	}
	return ret;
}

static struct fi_ops_ep rxm_ops_ep = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = rxm_ep_cancel,
	.getopt = rxm_ep_getopt,
	.setopt = rxm_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static int rxm_ep_discard_recv(struct rxm_ep *rxm_ep, struct rxm_rx_buf *rx_buf,
			       void *context)
{
	RXM_DBG_ADDR_TAG(FI_LOG_EP_DATA, "Discarding message",
			 rx_buf->unexp_msg.addr, rx_buf->unexp_msg.tag);

	dlist_insert_tail(&rx_buf->repost_entry,
			  &rx_buf->ep->repost_ready_list);
	return ofi_cq_write(rxm_ep->util_ep.rx_cq, context, FI_TAGGED | FI_RECV,
			    0, NULL, rx_buf->pkt.hdr.data, rx_buf->pkt.hdr.tag);
}

static int rxm_ep_peek_recv(struct rxm_ep *rxm_ep, fi_addr_t addr, uint64_t tag,
			    uint64_t ignore, void *context, uint64_t flags,
			    struct rxm_recv_queue *recv_queue)
{
	struct rxm_rx_buf *rx_buf;

	RXM_DBG_ADDR_TAG(FI_LOG_EP_DATA, "Peeking message", addr, tag);

	rxm_ep_do_progress(&rxm_ep->util_ep);

	rx_buf = rxm_check_unexp_msg_list(recv_queue, addr, tag, ignore);
	if (!rx_buf) {
		FI_DBG(&rxm_prov, FI_LOG_EP_DATA, "Message not found\n");
		return ofi_cq_write_error_peek(rxm_ep->util_ep.rx_cq, tag,
					       context);
	}

	FI_DBG(&rxm_prov, FI_LOG_EP_DATA, "Message found\n");

	if (flags & FI_DISCARD) {
		dlist_remove(&rx_buf->unexp_msg.entry);
		return rxm_ep_discard_recv(rxm_ep, rx_buf, context);
	}

	if (flags & FI_CLAIM) {
		FI_DBG(&rxm_prov, FI_LOG_EP_DATA, "Marking message for Claim\n");
		((struct fi_context *)context)->internal[0] = rx_buf;
		dlist_remove(&rx_buf->unexp_msg.entry);
	}

	return ofi_cq_write(rxm_ep->util_ep.rx_cq, context, FI_TAGGED | FI_RECV,
			    rx_buf->pkt.hdr.size, NULL,
			    rx_buf->pkt.hdr.data, rx_buf->pkt.hdr.tag);
}

static inline ssize_t
rxm_ep_format_rx_res(struct rxm_ep *rxm_ep, const struct iovec *iov,
		     void **desc, size_t count, fi_addr_t src_addr,
		     uint64_t tag, uint64_t ignore, void *context,
		     uint64_t flags, struct rxm_recv_queue *recv_queue,
		     struct rxm_recv_entry **recv_entry)
{
	size_t i;

	*recv_entry = rxm_recv_entry_get(recv_queue);
	if (OFI_UNLIKELY(!*recv_entry))
		return -FI_EAGAIN;

	(*recv_entry)->rxm_iov.count 	= (uint8_t)count;
	(*recv_entry)->addr 		= src_addr;
	(*recv_entry)->context 		= context;
	(*recv_entry)->flags 		= flags;
	(*recv_entry)->ignore 		= ignore;
	(*recv_entry)->tag		= tag;

	for (i = 0; i < count; i++) {
		(*recv_entry)->rxm_iov.iov[i].iov_base = iov[i].iov_base;
		(*recv_entry)->total_len +=
			(*recv_entry)->rxm_iov.iov[i].iov_len = iov[i].iov_len;
		if (desc)
			(*recv_entry)->rxm_iov.desc[i] = desc[i];
	}

	(*recv_entry)->multi_recv.len	= (*recv_entry)->total_len;
	(*recv_entry)->multi_recv.buf	= iov[0].iov_base;

	return FI_SUCCESS;
}

static inline ssize_t
rxm_ep_post_recv(struct rxm_ep *rxm_ep, const struct iovec *iov,
		 void **desc, size_t count, fi_addr_t src_addr,
		 uint64_t tag, uint64_t ignore, void *context,
		 uint64_t op_flags, struct rxm_recv_queue *recv_queue)
{
	struct rxm_recv_entry *recv_entry;
	ssize_t ret;

	assert(count <= rxm_ep->rxm_info->rx_attr->iov_limit);

	ret = rxm_ep_format_rx_res(rxm_ep, iov, desc, count, src_addr,
				   tag, ignore, context, op_flags,
				   recv_queue, &recv_entry);
	if (OFI_UNLIKELY(ret))
		return ret;

	FI_DBG(&rxm_prov, FI_LOG_EP_DATA, "Posting recv with length: %zu "
	       "tag: 0x%" PRIx64 " ignore: 0x%" PRIx64 "\n",
	       recv_entry->total_len, recv_entry->tag, recv_entry->ignore);
	ret = rxm_process_recv_entry(recv_queue, recv_entry);

	return ret;
}

static inline ssize_t
rxm_ep_recv_common(struct rxm_ep *rxm_ep, const struct iovec *iov,
		   void **desc, size_t count, fi_addr_t src_addr,
		   uint64_t tag, uint64_t ignore, void *context,
		   uint64_t op_flags, struct rxm_recv_queue *recv_queue)
{
	ssize_t ret;

	ofi_ep_lock_acquire(&rxm_ep->util_ep);
	ret = rxm_ep_post_recv(rxm_ep, iov, desc, count, src_addr,
			       tag, ignore, context, op_flags, recv_queue);
	ofi_ep_lock_release(&rxm_ep->util_ep);
	return ret;
}

static ssize_t rxm_ep_recv_common_flags(struct rxm_ep *rxm_ep, const struct iovec *iov,
					void **desc, size_t count, fi_addr_t src_addr,
					uint64_t tag, uint64_t ignore, void *context,
					uint64_t flags, uint64_t op_flags,
					struct rxm_recv_queue *recv_queue)
{
	struct rxm_recv_entry *recv_entry;
	struct fi_recv_context *recv_ctx;
	struct rxm_rx_buf *rx_buf;
	ssize_t ret = 0;

	assert(count <= rxm_ep->rxm_info->rx_attr->iov_limit);
	assert(!(flags & FI_PEEK) ||
		(recv_queue->type == RXM_RECV_QUEUE_TAGGED));
	assert(!(flags & (FI_MULTI_RECV)) ||
		(recv_queue->type == RXM_RECV_QUEUE_MSG));

	ofi_ep_lock_acquire(&rxm_ep->util_ep);
	if (rxm_ep->rxm_info->mode & FI_BUFFERED_RECV) {
		assert(!(flags & FI_PEEK));
		recv_ctx = context;
		context = recv_ctx->context;
		rx_buf = container_of(recv_ctx, struct rxm_rx_buf, recv_context);

		if (flags & FI_CLAIM) {
			FI_DBG(&rxm_prov, FI_LOG_EP_DATA,
			       "Claiming buffered receive\n");
			goto claim;
		}

		assert(flags & FI_DISCARD);
		FI_DBG(&rxm_prov, FI_LOG_EP_DATA, "Discarding buffered receive\n");
		dlist_insert_tail(&rx_buf->repost_entry,
				  &rx_buf->ep->repost_ready_list);
		goto unlock;
	}

	if (flags & FI_PEEK) {
		ret = rxm_ep_peek_recv(rxm_ep, src_addr, tag, ignore,
					context, flags, recv_queue);
		goto unlock;
	}

	if (!(flags & FI_CLAIM)) {
		ret = rxm_ep_post_recv(rxm_ep, iov, desc, count, src_addr,
				       tag, ignore, context, flags | op_flags,
				       recv_queue);
		goto unlock;
	}

	rx_buf = ((struct fi_context *)context)->internal[0];
	assert(rx_buf);
	FI_DBG(&rxm_prov, FI_LOG_EP_DATA, "Claim message\n");

	if (flags & FI_DISCARD) {
		ret = rxm_ep_discard_recv(rxm_ep, rx_buf, context);
		goto unlock;
	}

claim:
	ret = rxm_ep_format_rx_res(rxm_ep, iov, desc, count, src_addr,
				   tag, ignore, context, flags | op_flags,
				   recv_queue, &recv_entry);
	if (OFI_UNLIKELY(ret))
		goto unlock;

	if (rxm_ep->rxm_info->mode & FI_BUFFERED_RECV)
		recv_entry->comp_flags |= FI_CLAIM;

	rx_buf->recv_entry = recv_entry;
	ret = rxm_cq_handle_rx_buf(rx_buf);

unlock:
	ofi_ep_lock_release(&rxm_ep->util_ep);
	return ret;
}

static ssize_t rxm_ep_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			       uint64_t flags)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_recv_common_flags(rxm_ep, msg->msg_iov, msg->desc, msg->iov_count,
					msg->addr, 0, 0, msg->context,
					flags, (rxm_ep_rx_flags(rxm_ep) & FI_COMPLETION),
					&rxm_ep->recv_queue);
}

static ssize_t rxm_ep_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
			    fi_addr_t src_addr, void *context)
{
	struct rxm_ep *rxm_ep =
		container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	struct iovec iov = {
		.iov_base	= buf,
		.iov_len	= len,
	};

	return rxm_ep_recv_common(rxm_ep, &iov, &desc, 1, src_addr, 0, 0,
				  context, rxm_ep_rx_flags(rxm_ep),
				  &rxm_ep->recv_queue);
}

static ssize_t rxm_ep_recvv(struct fid_ep *ep_fid, const struct iovec *iov,
		void **desc, size_t count, fi_addr_t src_addr, void *context)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_recv_common(rxm_ep, iov, desc, count, src_addr, 0, 0,
				  context, rxm_ep_rx_flags(rxm_ep),
				  &rxm_ep->recv_queue);
}

static void rxm_rndv_hdr_init(struct rxm_ep *rxm_ep, void *buf,
			      const struct iovec *iov, size_t count,
			      struct fid_mr **mr)
{
	struct rxm_rndv_hdr *rndv_hdr = (struct rxm_rndv_hdr *)buf;
	size_t i;

	for (i = 0; i < count; i++) {
		rndv_hdr->iov[i].addr = RXM_MR_VIRT_ADDR(rxm_ep->msg_info) ?
			(uintptr_t)iov[i].iov_base : 0;
		rndv_hdr->iov[i].len = (uint64_t)iov[i].iov_len;
		rndv_hdr->iov[i].key = fi_mr_key(mr[i]);
	}
	rndv_hdr->count = (uint8_t)count;
}

static inline ssize_t
rxm_ep_msg_inject_send(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
		       struct rxm_pkt *tx_pkt, size_t pkt_size,
		       ofi_cntr_inc_func cntr_inc_func)
{
	FI_DBG(&rxm_prov, FI_LOG_EP_DATA, "Posting inject with length: %" PRIu64
	       " tag: 0x%" PRIx64 "\n", pkt_size, tx_pkt->hdr.tag);

	assert((tx_pkt->hdr.flags & FI_REMOTE_CQ_DATA) || !tx_pkt->hdr.flags);
	assert(pkt_size <= rxm_ep->inject_limit);

	ssize_t ret = fi_inject(rxm_conn->msg_ep, tx_pkt, pkt_size, 0);
	if (OFI_LIKELY(!ret)) {
		cntr_inc_func(rxm_ep->util_ep.tx_cntr);
	} else {
		FI_DBG(&rxm_prov, FI_LOG_EP_DATA,
		       "fi_inject for MSG provider failed with ret - %" PRId64"\n",
		       ret);
		if (OFI_LIKELY(ret == -FI_EAGAIN))
			rxm_ep_do_progress(&rxm_ep->util_ep);
	}
	return ret;
}

static inline ssize_t
rxm_ep_msg_normal_send(struct rxm_conn *rxm_conn, struct rxm_pkt *tx_pkt,
		       size_t pkt_size, void *desc, void *context)
{
	FI_DBG(&rxm_prov, FI_LOG_EP_DATA, "Posting send with length: %" PRIu64
	       " tag: 0x%" PRIx64 "\n", pkt_size, tx_pkt->hdr.tag);

	assert((tx_pkt->hdr.flags & FI_REMOTE_CQ_DATA) || !tx_pkt->hdr.flags);

	return fi_send(rxm_conn->msg_ep, tx_pkt, pkt_size, desc, 0, context);
}

static inline ssize_t
rxm_ep_alloc_rndv_tx_res(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn, void *context,
			uint8_t count, const struct iovec *iov, void **desc, size_t data_len,
			uint64_t data, uint64_t flags, uint64_t tag, uint8_t op,
			struct rxm_tx_rndv_buf **tx_rndv_buf)
{
	struct fid_mr **mr_iov;
	ssize_t ret;
	struct rxm_tx_rndv_buf *tx_buf = (struct rxm_tx_rndv_buf *)
			rxm_tx_buf_alloc(rxm_ep, RXM_BUF_POOL_TX_RNDV);

	if (OFI_UNLIKELY(!tx_buf)) {
		FI_WARN(&rxm_prov, FI_LOG_EP_DATA,
			"Ran out of buffers from RNDV buffer pool\n");
		return -FI_EAGAIN;
	}

	rxm_ep_format_tx_buf_pkt(rxm_conn, data_len, op, data, tag, flags, &(tx_buf)->pkt);
	tx_buf->pkt.ctrl_hdr.msg_id = rxm_tx_buf_2_msg_id(rxm_ep, RXM_BUF_POOL_TX_RNDV, tx_buf);
	tx_buf->app_context = context;
	tx_buf->flags = flags;
	tx_buf->count = count;

	if (!rxm_ep->rxm_mr_local) {
		ret = rxm_ep_msg_mr_regv(rxm_ep, iov, tx_buf->count,
					 FI_REMOTE_READ, tx_buf->mr);
		if (ret)
			goto err;
		mr_iov = tx_buf->mr;
	} else {
		/* desc is msg fid_mr * array */
		mr_iov = (struct fid_mr **)desc;
	}

	rxm_rndv_hdr_init(rxm_ep, &tx_buf->pkt.data, iov, tx_buf->count, mr_iov);

	ret = sizeof(struct rxm_pkt) + sizeof(struct rxm_rndv_hdr);

	if (rxm_ep->rxm_info->mode & FI_BUFFERED_RECV) {
		ofi_copy_from_iov(rxm_pkt_rndv_data(&tx_buf->pkt),
				  rxm_ep->buffered_min, iov, count, 0);
		ret += rxm_ep->buffered_min;
	}

	*tx_rndv_buf = tx_buf;
	return ret;
err:
	*tx_rndv_buf = NULL;
	rxm_tx_buf_release(rxm_ep, RXM_BUF_POOL_TX_RNDV, tx_buf);
	return ret;
}

static inline ssize_t
rxm_ep_rndv_tx_send(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
		   struct rxm_tx_rndv_buf *tx_buf, size_t pkt_size)
{
	ssize_t ret;

	RXM_LOG_STATE(FI_LOG_EP_DATA, tx_buf->pkt, RXM_TX, RXM_RNDV_TX);
	if (pkt_size <= rxm_ep->inject_limit) {
		RXM_LOG_STATE(FI_LOG_CQ, tx_buf->pkt, RXM_RNDV_TX, RXM_RNDV_ACK_WAIT);
		tx_buf->hdr.state = RXM_RNDV_ACK_WAIT;

		ret = rxm_ep_msg_inject_send(rxm_ep, rxm_conn, &tx_buf->pkt,
					     pkt_size, ofi_cntr_inc_noop);
	} else {
		tx_buf->hdr.state = RXM_RNDV_TX;

		ret = rxm_ep_msg_normal_send(rxm_conn, &tx_buf->pkt, pkt_size,
					     tx_buf->hdr.desc, tx_buf);
	}
	if (OFI_UNLIKELY(ret))
		goto err;
	return FI_SUCCESS;
err:
	FI_DBG(&rxm_prov, FI_LOG_EP_DATA,
	       "Transmit for MSG provider failed\n");
	if (!rxm_ep->rxm_mr_local)
		rxm_ep_msg_mr_closev(tx_buf->mr, tx_buf->count);
	rxm_tx_buf_release(rxm_ep, RXM_BUF_POOL_TX_RNDV, tx_buf);
	return ret;
}

static inline size_t
rxm_ep_sar_calc_segs_cnt(struct rxm_ep *rxm_ep, size_t data_len)
{
	return (data_len + rxm_ep->eager_limit - 1) /
	       rxm_ep->eager_limit;
}

static inline struct rxm_tx_sar_buf *
rxm_ep_sar_tx_prepare_segment(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
			      void *app_context, size_t total_len, size_t seg_len,
			      size_t seg_no, uint64_t data, uint64_t flags, uint64_t tag,
			      uint8_t op, enum rxm_sar_seg_type seg_type, uint64_t *msg_id)
{
	struct rxm_tx_sar_buf *tx_buf = (struct rxm_tx_sar_buf *)
		rxm_tx_buf_alloc(rxm_ep, RXM_BUF_POOL_TX_SAR);

	if (OFI_UNLIKELY(!tx_buf)) {
		FI_WARN(&rxm_prov, FI_LOG_EP_DATA,
			"Ran out of buffers from SAR buffer pool\n");
		return NULL;
	};

	rxm_ep_format_tx_buf_pkt(rxm_conn, total_len, op, data, tag, flags, &tx_buf->pkt);
	if (seg_type == RXM_SAR_SEG_FIRST) {
		*msg_id = tx_buf->pkt.ctrl_hdr.msg_id =
			rxm_tx_buf_2_msg_id(rxm_ep, RXM_BUF_POOL_TX_SAR, tx_buf);
	} else {
		tx_buf->pkt.ctrl_hdr.msg_id = *msg_id;
	}
	tx_buf->pkt.ctrl_hdr.seg_size = seg_len;
	tx_buf->pkt.ctrl_hdr.seg_no = seg_no;
	tx_buf->app_context = app_context;
	tx_buf->flags = flags;
	rxm_sar_set_seg_type(&tx_buf->pkt.ctrl_hdr, seg_type);

	return tx_buf;
}

static void
rxm_ep_sar_tx_cleanup(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
		      struct rxm_tx_sar_buf *tx_buf)
{
	struct rxm_tx_sar_buf *first_tx_buf =
		rxm_msg_id_2_tx_buf(rxm_ep, RXM_BUF_POOL_TX_SAR,
				    tx_buf->pkt.ctrl_hdr.msg_id);
	rxm_tx_buf_release(rxm_ep, RXM_BUF_POOL_TX_SAR, first_tx_buf);
	rxm_tx_buf_release(rxm_ep, RXM_BUF_POOL_TX_SAR, tx_buf);
}

static inline ssize_t
rxm_ep_sar_tx_prepare_and_send_segment(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
				       void *app_context, size_t data_len, size_t remain_len,
				       uint64_t msg_id, size_t seg_len, size_t seg_no, size_t segs_cnt,
				       uint64_t data, uint64_t flags, uint64_t tag, uint8_t op,
				       const struct iovec *iov, uint8_t count, size_t *iov_offset,
				       struct rxm_tx_sar_buf **out_tx_buf)
{
	struct rxm_tx_sar_buf *tx_buf;
	enum rxm_sar_seg_type seg_type = RXM_SAR_SEG_MIDDLE;

	if (seg_no == (segs_cnt - 1)) {
		seg_type = RXM_SAR_SEG_LAST;
		assert(remain_len <= seg_len);
		seg_len = remain_len;
	}

	tx_buf = rxm_ep_sar_tx_prepare_segment(rxm_ep, rxm_conn, app_context, data_len, seg_len,
					       seg_no, data, flags, tag, op, seg_type, &msg_id);
	if (OFI_UNLIKELY(!tx_buf)) {
		*out_tx_buf = NULL;
		return -FI_EAGAIN;
	}

	ofi_copy_from_iov(tx_buf->pkt.data, seg_len, iov, count, *iov_offset);
	*iov_offset += seg_len;

	*out_tx_buf = tx_buf;

	return fi_send(rxm_conn->msg_ep, &tx_buf->pkt, sizeof(struct rxm_pkt) +
		       tx_buf->pkt.ctrl_hdr.seg_size, tx_buf->hdr.desc, 0, tx_buf);
}

static inline ssize_t
rxm_ep_sar_tx_send(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
		   void *context, uint8_t count, const struct iovec *iov,
		   size_t data_len, size_t segs_cnt, uint64_t data,
		   uint64_t flags, uint64_t tag, uint8_t op)
{
	struct rxm_tx_sar_buf *tx_buf, *first_tx_buf;
	size_t i, iov_offset = 0, remain_len = data_len;
	ssize_t ret;
	struct rxm_deferred_tx_entry *def_tx_entry;
	uint64_t msg_id = 0;

	assert(segs_cnt >= 2);	

	first_tx_buf = rxm_ep_sar_tx_prepare_segment(rxm_ep, rxm_conn, context, data_len,
						     rxm_ep->eager_limit, 0, data, flags,
						     tag, op, RXM_SAR_SEG_FIRST, &msg_id);
	if (OFI_UNLIKELY(!first_tx_buf))
		return -FI_EAGAIN;

	ofi_copy_from_iov(first_tx_buf->pkt.data, rxm_ep->eager_limit,
			  iov, count, iov_offset);
	iov_offset += rxm_ep->eager_limit;

	ret = fi_send(rxm_conn->msg_ep, &first_tx_buf->pkt, sizeof(struct rxm_pkt) +
		      first_tx_buf->pkt.ctrl_hdr.seg_size, first_tx_buf->hdr.desc, 0, first_tx_buf);
	if (OFI_UNLIKELY(ret)) {
		if (OFI_LIKELY(ret == -FI_EAGAIN))
			rxm_ep_do_progress(&rxm_ep->util_ep);
		rxm_tx_buf_release(rxm_ep, RXM_BUF_POOL_TX_SAR, first_tx_buf);
		return ret;
	}

	remain_len -= rxm_ep->eager_limit;

	for (i = 1; i < segs_cnt; i++) {
		ret = rxm_ep_sar_tx_prepare_and_send_segment(
					rxm_ep, rxm_conn, context, data_len, remain_len,
					msg_id, rxm_ep->eager_limit, i, segs_cnt, data,
					flags, tag, op, iov, count, &iov_offset, &tx_buf);
		if (OFI_UNLIKELY(ret)) {
			if (OFI_LIKELY(ret == -FI_EAGAIN)) {
				def_tx_entry = rxm_ep_alloc_deferred_tx_entry(rxm_ep, rxm_conn,
									      RXM_DEFERRED_TX_SAR_SEG);
				if (OFI_UNLIKELY(!def_tx_entry)) {
					if (tx_buf)
						rxm_tx_buf_release(rxm_ep, RXM_BUF_POOL_TX_SAR,
								   tx_buf);
					return -FI_ENOMEM;
				}
				memcpy(def_tx_entry->sar_seg.payload.iov, iov, sizeof(*iov) * count);
				def_tx_entry->sar_seg.payload.count = count;
				def_tx_entry->sar_seg.payload.cur_iov_offset = iov_offset;
				def_tx_entry->sar_seg.payload.tag = tag;
				def_tx_entry->sar_seg.payload.data = data;
				def_tx_entry->sar_seg.cur_seg_tx_buf = tx_buf;
				def_tx_entry->sar_seg.app_context = context;
				def_tx_entry->sar_seg.flags = flags;
				def_tx_entry->sar_seg.op = op;
				def_tx_entry->sar_seg.next_seg_no = i;
				def_tx_entry->sar_seg.segs_cnt = segs_cnt;
				def_tx_entry->sar_seg.total_len = data_len;
				def_tx_entry->sar_seg.remain_len = remain_len;
				def_tx_entry->sar_seg.msg_id = msg_id;
				rxm_ep_enqueue_deferred_tx_queue(def_tx_entry);
				return 0;
			}

			rxm_tx_buf_release(rxm_ep, RXM_BUF_POOL_TX_SAR, first_tx_buf);
			return ret;
		}
		remain_len -= rxm_ep->eager_limit;
	}

	return 0;
}

static inline ssize_t
rxm_ep_emulate_inject(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
		      const void *buf, size_t len, size_t pkt_size,
		      uint64_t data, uint64_t flags, uint64_t tag,
		      uint8_t op)
{
	struct rxm_tx_eager_buf *tx_buf;
	ssize_t ret;

	FI_DBG(&rxm_prov, FI_LOG_EP_DATA, "passed data (size = %zu) "
	       "is too big for MSG provider (max inject size = %zd)\n",
	       pkt_size, rxm_ep->inject_limit);

	tx_buf = (struct rxm_tx_eager_buf *)
		  rxm_tx_buf_alloc(rxm_ep, RXM_BUF_POOL_TX);
	if (OFI_UNLIKELY(!tx_buf)) {
		FI_WARN(&rxm_prov, FI_LOG_EP_DATA,
			"Ran out of buffers from Eager buffer pool\n");
		return -FI_EAGAIN;
	}
	/* This is needed so that we don't report bogus context in fi_cq_err_entry */
	tx_buf->app_context = NULL;

	rxm_ep_format_tx_buf_pkt(rxm_conn, len, op, data, tag, flags, &tx_buf->pkt);
	memcpy(tx_buf->pkt.data, buf, len);
	tx_buf->flags = flags;

	ret = rxm_ep_msg_normal_send(rxm_conn, &tx_buf->pkt, pkt_size,
				     tx_buf->hdr.desc, tx_buf);
	if (OFI_UNLIKELY(ret)) {
		if (OFI_LIKELY(ret == -FI_EAGAIN))
			rxm_ep_do_progress(&rxm_ep->util_ep);
		rxm_tx_buf_release(rxm_ep, RXM_BUF_POOL_TX, tx_buf);
	}
	return ret;
}

static inline ssize_t
rxm_ep_inject_send_fast(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
			const void *buf, size_t len, struct rxm_pkt *inject_pkt)
{
	size_t pkt_size = sizeof(struct rxm_pkt) + len;
	ssize_t ret;

	assert(len <= rxm_ep->eager_limit);

	if (pkt_size <= rxm_ep->inject_limit) {
		inject_pkt->hdr.size = len;
		memcpy(inject_pkt->data, buf, len);
		ret = rxm_ep_msg_inject_send(rxm_ep, rxm_conn, inject_pkt,
					      pkt_size, rxm_ep->util_ep.tx_cntr_inc);
	} else {
		ret = rxm_ep_emulate_inject(rxm_ep, rxm_conn, buf, len, pkt_size,
					    inject_pkt->hdr.data, inject_pkt->hdr.flags,
					    inject_pkt->hdr.tag, inject_pkt->hdr.op);
	}
	return ret;
}

static inline ssize_t
rxm_ep_inject_send(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
		   const void *buf, size_t len, uint64_t data,
		   uint64_t flags, uint64_t tag, uint8_t op)
{
	size_t pkt_size = sizeof(struct rxm_pkt) + len;
	ssize_t ret;

	assert(len <= rxm_ep->eager_limit);

	ofi_ep_lock_acquire(&rxm_ep->util_ep);
	if (pkt_size <= rxm_ep->inject_limit) {
		struct rxm_tx_base_buf *tx_buf = (struct rxm_tx_base_buf *)
			rxm_tx_buf_alloc(rxm_ep, RXM_BUF_POOL_TX_INJECT);
		if (OFI_UNLIKELY(!tx_buf)) {
			FI_WARN(&rxm_prov, FI_LOG_EP_DATA,
				"Ran out of buffers from Eager Inject buffer pool\n");
			ret = -FI_EAGAIN;
			goto unlock;
		}
		rxm_ep_format_tx_buf_pkt(rxm_conn, len, op, data, tag,
					 flags, &tx_buf->pkt);
		memcpy(tx_buf->pkt.data, buf, len);

		ret = rxm_ep_msg_inject_send(rxm_ep, rxm_conn, &tx_buf->pkt,
					     pkt_size, rxm_ep->util_ep.tx_cntr_inc);
		rxm_tx_buf_release(rxm_ep, RXM_BUF_POOL_TX_INJECT, tx_buf);
	} else {
		ret = rxm_ep_emulate_inject(rxm_ep, rxm_conn, buf, len,
					    pkt_size, data, flags, tag, op);
	}
unlock:
	ofi_ep_lock_release(&rxm_ep->util_ep);
	return ret;

}

static inline ssize_t
rxm_ep_inject_send_common(struct rxm_ep *rxm_ep, const struct iovec *iov, size_t count,
			  struct rxm_conn *rxm_conn, void *context, uint64_t data,
			  uint64_t flags, uint64_t tag, uint8_t op, size_t data_len,
			  size_t total_len, struct rxm_pkt *inject_pkt)
{
	int ret;

	if (rxm_ep->util_ep.domain->threading != FI_THREAD_SAFE) {
		assert((op == inject_pkt->hdr.op) &&
		       ((flags & FI_REMOTE_CQ_DATA) == inject_pkt->hdr.flags));

		inject_pkt->hdr.data = data;
		inject_pkt->hdr.tag = tag;
		inject_pkt->hdr.size = data_len;
		ofi_copy_from_iov(inject_pkt->data, inject_pkt->hdr.size,
				  iov, count, 0);

		ret = rxm_ep_msg_inject_send(rxm_ep, rxm_conn, inject_pkt,
					     total_len, rxm_ep->util_ep.tx_cntr_inc);
	} else {
		struct rxm_tx_base_buf *tx_buf = (struct rxm_tx_base_buf *)
			rxm_tx_buf_alloc(rxm_ep, RXM_BUF_POOL_TX_INJECT);
		if (OFI_UNLIKELY(!tx_buf)) {
			FI_WARN(&rxm_prov, FI_LOG_EP_DATA,
				"Ran out of buffers from Eager Inject buffer pool\n");
			return -FI_EAGAIN;
		}
		rxm_ep_format_tx_buf_pkt(rxm_conn, data_len, op, data, tag,
				         flags, &tx_buf->pkt);
		ofi_copy_from_iov(tx_buf->pkt.data, tx_buf->pkt.hdr.size,
				  iov, count, 0);

		ret = rxm_ep_msg_inject_send(rxm_ep, rxm_conn, &tx_buf->pkt,
					     total_len, rxm_ep->util_ep.tx_cntr_inc);
		rxm_tx_buf_release(rxm_ep, RXM_BUF_POOL_TX_INJECT, tx_buf);
	}
	if (OFI_UNLIKELY(ret))
		return ret;

	if (flags & FI_COMPLETION) {
		ret = ofi_cq_write(rxm_ep->util_ep.tx_cq, context,
				   ofi_tx_flags[op], 0, NULL, 0, 0);
		if (OFI_UNLIKELY(ret)) {
			FI_WARN(&rxm_prov, FI_LOG_CQ,
				"Unable to report completion\n");
			return ret;
		}
		rxm_cq_log_comp(ofi_tx_flags[op]);
	}
	return FI_SUCCESS;
}

static ssize_t
rxm_ep_send_common(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
		   const struct iovec *iov, void **desc, size_t count,
		   void *context, uint64_t data, uint64_t flags, uint64_t tag,
		   uint8_t op, struct rxm_pkt *inject_pkt)
{
	size_t data_len = ofi_total_iov_len(iov, count);
	size_t total_len = sizeof(struct rxm_pkt) + data_len;
	ssize_t ret;

	assert(count <= rxm_ep->rxm_info->tx_attr->iov_limit);
	assert((!(flags & FI_INJECT) && (data_len > rxm_ep->eager_limit)) ||
	       (data_len <= rxm_ep->eager_limit));

	ofi_ep_lock_acquire(&rxm_ep->util_ep);
	if (total_len <= rxm_ep->inject_limit) {
		ret = rxm_ep_inject_send_common(rxm_ep, iov, count, rxm_conn,
						context, data, flags, tag, op,
						data_len, total_len, inject_pkt);
	} else if (data_len <= rxm_ep->eager_limit) {
		struct rxm_tx_eager_buf *tx_buf = (struct rxm_tx_eager_buf *)
			rxm_tx_buf_alloc(rxm_ep, RXM_BUF_POOL_TX);

		if (OFI_UNLIKELY(!tx_buf)) {
			FI_WARN(&rxm_prov, FI_LOG_EP_DATA,
				"Ran out of buffers from Eager buffer pool\n");
			ret = -FI_EAGAIN;
			goto unlock;
		}

		rxm_ep_format_tx_buf_pkt(rxm_conn, data_len, op, data, tag,
					 flags, &tx_buf->pkt);
		ofi_copy_from_iov(tx_buf->pkt.data, tx_buf->pkt.hdr.size,
				  iov, count, 0);
		tx_buf->app_context = context;
		tx_buf->flags = flags;

		ret = rxm_ep_msg_normal_send(rxm_conn, &tx_buf->pkt, total_len,
					     tx_buf->hdr.desc, tx_buf);
		if (OFI_UNLIKELY(ret)) {
			if (ret == -FI_EAGAIN)
				rxm_ep_do_progress(&rxm_ep->util_ep);
			rxm_tx_buf_release(rxm_ep, RXM_BUF_POOL_TX, tx_buf);
		}
	} else if (data_len <= rxm_ep->sar_limit) {
		ret = rxm_ep_sar_tx_send(rxm_ep, rxm_conn, context,
					 count, iov, data_len,
					 rxm_ep_sar_calc_segs_cnt(rxm_ep, data_len),
					 data, flags, tag, op);
	} else {
		struct rxm_tx_rndv_buf *tx_buf;

		ret = rxm_ep_alloc_rndv_tx_res(rxm_ep, rxm_conn, context, (uint8_t)count,
					      iov, desc, data_len, data, flags, tag, op,
					      &tx_buf);
		if (OFI_LIKELY(ret >= 0))
			ret = rxm_ep_rndv_tx_send(rxm_ep, rxm_conn, tx_buf, ret);
	}
unlock:
	ofi_ep_lock_release(&rxm_ep->util_ep);
	return ret;
}

struct rxm_deferred_tx_entry *
rxm_ep_alloc_deferred_tx_entry(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
			       enum rxm_deferred_tx_entry_type type)
{
	struct rxm_deferred_tx_entry *def_tx_entry =
			calloc(1, sizeof(*def_tx_entry));
	if (OFI_UNLIKELY(!def_tx_entry))
		return NULL;

	def_tx_entry->rxm_ep = rxm_ep;
	def_tx_entry->rxm_conn = rxm_conn;
	def_tx_entry->type = type;
	dlist_init(&def_tx_entry->entry);

	return def_tx_entry;
}

static inline void
rxm_ep_sar_handle_segment_failure(struct rxm_deferred_tx_entry *def_tx_entry, ssize_t ret)
{
	rxm_ep_sar_tx_cleanup(def_tx_entry->rxm_ep, def_tx_entry->rxm_conn,
			      def_tx_entry->sar_seg.cur_seg_tx_buf);
	rxm_cq_write_error(def_tx_entry->rxm_ep->util_ep.tx_cq,
			   def_tx_entry->rxm_ep->util_ep.tx_cntr,
			   def_tx_entry->sar_seg.app_context, ret);
}

/* Returns FI_SUCCESS if the SAR deferred TX queue is empty,
 * otherwise, it returns -FI_EAGAIN or error from MSG provider */
static ssize_t
rxm_ep_progress_sar_deferred_segments(struct rxm_deferred_tx_entry *def_tx_entry)
{
	ssize_t ret = 0;
	struct rxm_tx_sar_buf *tx_buf = def_tx_entry->sar_seg.cur_seg_tx_buf;

	if (tx_buf) {
		ret = fi_send(def_tx_entry->rxm_conn->msg_ep, &tx_buf->pkt, sizeof(tx_buf->pkt) +
			      tx_buf->pkt.ctrl_hdr.seg_size, tx_buf->hdr.desc, 0, tx_buf);
		if (OFI_UNLIKELY(ret)) {
			if (OFI_LIKELY(ret != -FI_EAGAIN)) {
				rxm_ep_sar_handle_segment_failure(def_tx_entry, ret);
				goto sar_finish;
			}
			return ret;
		}

		def_tx_entry->sar_seg.next_seg_no++;
		def_tx_entry->sar_seg.remain_len -= def_tx_entry->rxm_ep->eager_limit;

		if (def_tx_entry->sar_seg.next_seg_no == def_tx_entry->sar_seg.segs_cnt) {
			assert(rxm_sar_get_seg_type(&tx_buf->pkt.ctrl_hdr) == RXM_SAR_SEG_LAST);
			goto sar_finish;
		}
	}

	while (def_tx_entry->sar_seg.next_seg_no != def_tx_entry->sar_seg.segs_cnt) {
		ret = rxm_ep_sar_tx_prepare_and_send_segment(
				def_tx_entry->rxm_ep, def_tx_entry->rxm_conn,
				def_tx_entry->sar_seg.app_context,
				def_tx_entry->sar_seg.total_len, def_tx_entry->sar_seg.remain_len,
				def_tx_entry->sar_seg.msg_id, def_tx_entry->rxm_ep->eager_limit,
				def_tx_entry->sar_seg.next_seg_no, def_tx_entry->sar_seg.segs_cnt,
				def_tx_entry->sar_seg.payload.data, def_tx_entry->sar_seg.flags,
				def_tx_entry->sar_seg.payload.tag, def_tx_entry->sar_seg.op,
				def_tx_entry->sar_seg.payload.iov,
				def_tx_entry->sar_seg.payload.count,
				&def_tx_entry->sar_seg.payload.cur_iov_offset,
				&def_tx_entry->sar_seg.cur_seg_tx_buf);
		if (OFI_UNLIKELY(ret)) {
			if (OFI_LIKELY(ret != -FI_EAGAIN)) {
				rxm_ep_sar_handle_segment_failure(def_tx_entry, ret);
				goto sar_finish;
			}

			return ret;
		}
		def_tx_entry->sar_seg.next_seg_no++;
		def_tx_entry->sar_seg.remain_len -= def_tx_entry->rxm_ep->eager_limit;
	}

sar_finish:
	rxm_ep_dequeue_deferred_tx_queue(def_tx_entry);
	free(def_tx_entry);

	return ret;
}

void rxm_ep_progress_deferred_queue(struct rxm_ep *rxm_ep,
				    struct rxm_conn *rxm_conn)
{
	struct rxm_deferred_tx_entry *def_tx_entry;
	ssize_t ret = 0;

	while (!dlist_empty(&rxm_conn->deferred_tx_queue) && !ret) {
		def_tx_entry = container_of(rxm_conn->deferred_tx_queue.next,
					    struct rxm_deferred_tx_entry, entry);
		switch (def_tx_entry->type) {
		case RXM_DEFERRED_TX_RNDV_ACK:
			ret = fi_send(def_tx_entry->rxm_conn->msg_ep,
				      &def_tx_entry->rndv_ack.rx_buf->
					recv_entry->rndv.tx_buf->pkt,
				      sizeof(def_tx_entry->rndv_ack.rx_buf->
					recv_entry->rndv.tx_buf->pkt),
				      def_tx_entry->rndv_ack.rx_buf->recv_entry->
					rndv.tx_buf->hdr.desc,
				      0, def_tx_entry->rndv_ack.rx_buf);
			if (OFI_UNLIKELY(ret)) {
				if (OFI_LIKELY(ret == -FI_EAGAIN))
					break;
				rxm_cq_write_error(def_tx_entry->rxm_ep->util_ep.rx_cq,
						   def_tx_entry->rxm_ep->util_ep.rx_cntr,
						   def_tx_entry->rndv_read.rx_buf->
							recv_entry->context, ret);
			}
			rxm_ep_dequeue_deferred_tx_queue(def_tx_entry);
			free(def_tx_entry);
			break;
		case RXM_DEFERRED_TX_RNDV_READ:
			ret = fi_readv(def_tx_entry->rxm_conn->msg_ep,
				       def_tx_entry->rndv_read.rxm_iov.iov,
				       def_tx_entry->rndv_read.rxm_iov.desc,
				       def_tx_entry->rndv_read.rxm_iov.count, 0,
				       def_tx_entry->rndv_read.rma_iov.addr,
				       def_tx_entry->rndv_read.rma_iov.key,
				       def_tx_entry->rndv_read.rx_buf);
			if (OFI_UNLIKELY(ret)) {
				if (OFI_LIKELY(ret == -FI_EAGAIN))
					break;
				rxm_cq_write_error(def_tx_entry->rxm_ep->util_ep.rx_cq,
						   def_tx_entry->rxm_ep->util_ep.rx_cntr,
						   def_tx_entry->rndv_read.rx_buf->
							recv_entry->context, ret);
				break;
			}
			rxm_ep_dequeue_deferred_tx_queue(def_tx_entry);
			free(def_tx_entry);
			break;
		case RXM_DEFERRED_TX_SAR_SEG:
			ret = rxm_ep_progress_sar_deferred_segments(def_tx_entry);
			break;
		case RXM_DEFERRED_TX_ATOMIC_RESP:
			ret = rxm_atomic_send_respmsg(rxm_ep,
					def_tx_entry->rxm_conn,
					def_tx_entry->atomic_resp.tx_buf,
					def_tx_entry->atomic_resp.len);
			if (OFI_UNLIKELY(ret))
				if (OFI_LIKELY(ret == -FI_EAGAIN))
					break;
			rxm_ep_dequeue_deferred_tx_queue(def_tx_entry);
			free(def_tx_entry);
			break;
		}
	}
}

static ssize_t rxm_ep_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			      uint64_t flags)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, msg->addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	return rxm_ep_send_common(rxm_ep, rxm_conn, msg->msg_iov, msg->desc,
				  msg->iov_count, msg->context, msg->data,
				  flags | (rxm_ep_tx_flags(rxm_ep) & FI_COMPLETION),
				  0, ofi_op_msg,
				  ((flags & FI_REMOTE_CQ_DATA) ?
				   rxm_conn->inject_data_pkt : rxm_conn->inject_pkt));
}

static ssize_t rxm_ep_send(struct fid_ep *ep_fid, const void *buf, size_t len,
			   void *desc, fi_addr_t dest_addr, void *context)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct iovec iov = {
		.iov_base = (void *)buf,
		.iov_len = len,
	};
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, dest_addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	return rxm_ep_send_common(rxm_ep, rxm_conn, &iov, &desc, 1, context,
				  0, rxm_ep_tx_flags(rxm_ep), 0, ofi_op_msg,
				  rxm_conn->inject_pkt);
}

static ssize_t rxm_ep_sendv(struct fid_ep *ep_fid, const struct iovec *iov,
			    void **desc, size_t count, fi_addr_t dest_addr,
			    void *context)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, dest_addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	return rxm_ep_send_common(rxm_ep, rxm_conn, iov, desc, count, context,
				  0, rxm_ep_tx_flags(rxm_ep), 0, ofi_op_msg,
				  rxm_conn->inject_pkt);
}

static ssize_t rxm_ep_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
			     fi_addr_t dest_addr)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, dest_addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	return rxm_ep_inject_send(rxm_ep, rxm_conn, buf, len, 0,
				  rxm_ep->util_ep.inject_op_flags,
				  0, ofi_op_msg);
}

static ssize_t rxm_ep_inject_fast(struct fid_ep *ep_fid, const void *buf, size_t len,
				  fi_addr_t dest_addr)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, dest_addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	return rxm_ep_inject_send_fast(rxm_ep, rxm_conn, buf, len,
				       rxm_conn->inject_pkt);
}

static ssize_t rxm_ep_senddata(struct fid_ep *ep_fid, const void *buf, size_t len,
			       void *desc, uint64_t data, fi_addr_t dest_addr,
			       void *context)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct iovec iov = {
		.iov_base = (void *)buf,
		.iov_len = len,
	};
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, dest_addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	return rxm_ep_send_common(rxm_ep, rxm_conn, &iov, desc, 1, context, data,
				  rxm_ep_tx_flags(rxm_ep) | FI_REMOTE_CQ_DATA,
				  0, ofi_op_msg, rxm_conn->inject_data_pkt);
}

static ssize_t rxm_ep_injectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
				 uint64_t data, fi_addr_t dest_addr)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, dest_addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	return rxm_ep_inject_send(rxm_ep, rxm_conn, buf, len, data,
				  rxm_ep->util_ep.inject_op_flags |
				  FI_REMOTE_CQ_DATA, 0, ofi_op_msg);
}

static ssize_t rxm_ep_injectdata_fast(struct fid_ep *ep_fid, const void *buf, size_t len,
				      uint64_t data, fi_addr_t dest_addr)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, dest_addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	rxm_conn->inject_data_pkt->hdr.data = data;

	return rxm_ep_inject_send_fast(rxm_ep, rxm_conn, buf, len,
				       rxm_conn->inject_data_pkt);
}

static struct fi_ops_msg rxm_ops_msg = {
	.size = sizeof(struct fi_ops_msg),
	.recv = rxm_ep_recv,
	.recvv = rxm_ep_recvv,
	.recvmsg = rxm_ep_recvmsg,
	.send = rxm_ep_send,
	.sendv = rxm_ep_sendv,
	.sendmsg = rxm_ep_sendmsg,
	.inject = rxm_ep_inject,
	.senddata = rxm_ep_senddata,
	.injectdata = rxm_ep_injectdata,
};

static struct fi_ops_msg rxm_ops_msg_thread_unsafe = {
	.size = sizeof(struct fi_ops_msg),
	.recv = rxm_ep_recv,
	.recvv = rxm_ep_recvv,
	.recvmsg = rxm_ep_recvmsg,
	.send = rxm_ep_send,
	.sendv = rxm_ep_sendv,
	.sendmsg = rxm_ep_sendmsg,
	.inject = rxm_ep_inject_fast,
	.senddata = rxm_ep_senddata,
	.injectdata = rxm_ep_injectdata_fast,
};

static ssize_t rxm_ep_trecvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
			       uint64_t flags)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_recv_common_flags(rxm_ep, msg->msg_iov, msg->desc, msg->iov_count,
					msg->addr, msg->tag, msg->ignore, msg->context,
					flags, (rxm_ep_rx_flags(rxm_ep) & FI_COMPLETION),
					&rxm_ep->trecv_queue);
}

static ssize_t rxm_ep_trecv(struct fid_ep *ep_fid, void *buf, size_t len,
			    void *desc, fi_addr_t src_addr, uint64_t tag,
			    uint64_t ignore, void *context)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);
	struct iovec iov = {
		.iov_base	= buf,
		.iov_len	= len,
	};

	return rxm_ep_recv_common(rxm_ep, &iov, &desc, 1, src_addr, tag, ignore,
				  context, rxm_ep_rx_flags(rxm_ep),
				  &rxm_ep->trecv_queue);
}

static ssize_t rxm_ep_trecvv(struct fid_ep *ep_fid, const struct iovec *iov,
			     void **desc, size_t count, fi_addr_t src_addr,
			     uint64_t tag, uint64_t ignore, void *context)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_recv_common(rxm_ep, iov, desc, count, src_addr, tag, ignore,
				  context, rxm_ep_rx_flags(rxm_ep),
				  &rxm_ep->trecv_queue);
}

static ssize_t rxm_ep_tsendmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
			       uint64_t flags)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, msg->addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	return rxm_ep_send_common(rxm_ep, rxm_conn, msg->msg_iov, msg->desc,
				  msg->iov_count, msg->context, msg->data,
				  flags | (rxm_ep_tx_flags(rxm_ep) & FI_COMPLETION),
				  msg->tag, ofi_op_tagged,
				  ((flags & FI_REMOTE_CQ_DATA) ?
				   rxm_conn->tinject_data_pkt : rxm_conn->tinject_pkt));
}

static ssize_t rxm_ep_tsend(struct fid_ep *ep_fid, const void *buf, size_t len,
			    void *desc, fi_addr_t dest_addr, uint64_t tag,
			    void *context)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct iovec iov = {
		.iov_base = (void *)buf,
		.iov_len = len,
	};
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, dest_addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	return rxm_ep_send_common(rxm_ep, rxm_conn, &iov, &desc, 1, context, 0,
				  rxm_ep_tx_flags(rxm_ep), tag, ofi_op_tagged,
				  rxm_conn->tinject_pkt);
}

static ssize_t rxm_ep_tsendv(struct fid_ep *ep_fid, const struct iovec *iov,
			     void **desc, size_t count, fi_addr_t dest_addr,
			     uint64_t tag, void *context)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, dest_addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	return rxm_ep_send_common(rxm_ep, rxm_conn, iov, desc, count, context, 0,
				  rxm_ep_tx_flags(rxm_ep), tag, ofi_op_tagged,
				  rxm_conn->tinject_pkt);
}

static ssize_t rxm_ep_tinject(struct fid_ep *ep_fid, const void *buf, size_t len,
			      fi_addr_t dest_addr, uint64_t tag)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, dest_addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	return rxm_ep_inject_send(rxm_ep, rxm_conn, buf, len, 0,
				  rxm_ep->util_ep.inject_op_flags, tag,
				  ofi_op_tagged);
}

static ssize_t rxm_ep_tinject_fast(struct fid_ep *ep_fid, const void *buf, size_t len,
				   fi_addr_t dest_addr, uint64_t tag)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, dest_addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	rxm_conn->tinject_pkt->hdr.tag = tag;

	return rxm_ep_inject_send_fast(rxm_ep, rxm_conn, buf, len,
				       rxm_conn->tinject_pkt);
}

static ssize_t rxm_ep_tsenddata(struct fid_ep *ep_fid, const void *buf, size_t len,
				void *desc, uint64_t data, fi_addr_t dest_addr,
				uint64_t tag, void *context)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct iovec iov = {
		.iov_base = (void *)buf,
		.iov_len = len,
	};
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, dest_addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	return rxm_ep_send_common(rxm_ep, rxm_conn, &iov, desc, 1, context, data,
				  rxm_ep_tx_flags(rxm_ep) | FI_REMOTE_CQ_DATA,
				  tag, ofi_op_tagged, rxm_conn->tinject_data_pkt);
}

static ssize_t rxm_ep_tinjectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
				  uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, dest_addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	return rxm_ep_inject_send(rxm_ep, rxm_conn, buf, len, data,
				  rxm_ep->util_ep.inject_op_flags |
				  FI_REMOTE_CQ_DATA, tag, ofi_op_tagged);
}

static ssize_t rxm_ep_tinjectdata_fast(struct fid_ep *ep_fid, const void *buf, size_t len,
				       uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	int ret;
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	ret = rxm_ep_prepare_tx(rxm_ep, dest_addr, &rxm_conn);
	if (OFI_UNLIKELY(ret))
		return ret;

	rxm_conn->tinject_data_pkt->hdr.tag = tag;
	rxm_conn->tinject_data_pkt->hdr.data = data;

	return rxm_ep_inject_send_fast(rxm_ep, rxm_conn, buf, len,
				       rxm_conn->tinject_data_pkt);
}

static struct fi_ops_tagged rxm_ops_tagged = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = rxm_ep_trecv,
	.recvv = rxm_ep_trecvv,
	.recvmsg = rxm_ep_trecvmsg,
	.send = rxm_ep_tsend,
	.sendv = rxm_ep_tsendv,
	.sendmsg = rxm_ep_tsendmsg,
	.inject = rxm_ep_tinject,
	.senddata = rxm_ep_tsenddata,
	.injectdata = rxm_ep_tinjectdata,
};

static struct fi_ops_tagged rxm_ops_tagged_thread_unsafe = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = rxm_ep_trecv,
	.recvv = rxm_ep_trecvv,
	.recvmsg = rxm_ep_trecvmsg,
	.send = rxm_ep_tsend,
	.sendv = rxm_ep_tsendv,
	.sendmsg = rxm_ep_tsendmsg,
	.inject = rxm_ep_tinject_fast,
	.senddata = rxm_ep_tsenddata,
	.injectdata = rxm_ep_tinjectdata_fast,
};

static int rxm_ep_msg_res_close(struct rxm_ep *rxm_ep)
{
	int ret, retv = 0;

	if (rxm_ep->srx_ctx) {
		ret = fi_close(&rxm_ep->srx_ctx->fid);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, \
				"Unable to close msg shared ctx\n");
			retv = ret;
		}
	}

	fi_freeinfo(rxm_ep->msg_info);
	return retv;
}

static int rxm_listener_close(struct rxm_ep *rxm_ep)
{
	int ret, retv = 0;

	if (rxm_ep->msg_pep) {
		ret = fi_close(&rxm_ep->msg_pep->fid);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
				"Unable to close msg pep\n");
			retv = ret;
		}
	}
	if (rxm_ep->msg_eq) {
		ret = fi_close(&rxm_ep->msg_eq->fid);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
				"Unable to close msg EQ\n");
			retv = ret;
		}
	}
	return retv;
}

static int rxm_ep_close(struct fid *fid)
{
	int ret, retv = 0;
	struct rxm_ep *rxm_ep =
		container_of(fid, struct rxm_ep, util_ep.ep_fid.fid);

	if (rxm_ep->cmap)
		rxm_cmap_free(rxm_ep->cmap);

	// TODO move this to cmap_free and encapsulate eq progress fns
	// these vars shouldn't be accessed outside rxm_conn file
	fastlock_destroy(&rxm_ep->msg_eq_entry_list_lock);
	slistfd_free(&rxm_ep->msg_eq_entry_list);

	ret = rxm_listener_close(rxm_ep);
	if (ret)
		retv = ret;

	rxm_ep_txrx_res_close(rxm_ep);

	if (rxm_ep->msg_cq) {
		ret = fi_close(&rxm_ep->msg_cq->fid);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to close msg CQ\n");
			retv = ret;
		}
	}

	ret = rxm_ep_msg_res_close(rxm_ep);
	if (ret)
		retv = ret;

	ofi_endpoint_close(&rxm_ep->util_ep);
	fi_freeinfo(rxm_ep->rxm_info);
	free(rxm_ep);
	return retv;
}

static int rxm_ep_msg_get_wait_cq_fd(struct rxm_ep *rxm_ep,
				     enum fi_wait_obj wait_obj)
{
	int ret = FI_SUCCESS;

	if ((wait_obj != FI_WAIT_NONE) && (!rxm_ep->msg_cq_fd)) {
		ret = fi_control(&rxm_ep->msg_cq->fid, FI_GETWAIT, &rxm_ep->msg_cq_fd);
		if (ret)
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
				"Unable to get MSG CQ fd\n");
	}
	return ret;
}

static int rxm_ep_msg_cq_open(struct rxm_ep *rxm_ep, enum fi_wait_obj wait_obj)
{
	struct rxm_domain *rxm_domain;
	struct fi_cq_attr cq_attr = { 0 };
	int ret;

	assert((wait_obj == FI_WAIT_NONE) || (wait_obj == FI_WAIT_FD));

	cq_attr.size = (rxm_ep->msg_info->tx_attr->size +
			rxm_ep->msg_info->rx_attr->size) * rxm_def_univ_size;
	cq_attr.format = FI_CQ_FORMAT_DATA;
	cq_attr.wait_obj = wait_obj;

	rxm_domain = container_of(rxm_ep->util_ep.domain, struct rxm_domain, util_domain);

	ret = fi_cq_open(rxm_domain->msg_domain, &cq_attr, &rxm_ep->msg_cq, NULL);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to open MSG CQ\n");
		return ret;
	}

	ret = rxm_ep_msg_get_wait_cq_fd(rxm_ep, wait_obj);
	if (ret)
		goto err;

	return 0;
err:
	fi_close(&rxm_ep->msg_cq->fid);
	return ret;
}

static int rxm_ep_eq_entry_list_trywait(void *arg)
{
	struct rxm_ep *rxm_ep = (struct rxm_ep *)arg;
	int ret;

	fastlock_acquire(&rxm_ep->msg_eq_entry_list_lock);
	ret = slistfd_empty(&rxm_ep->msg_eq_entry_list) ? 0 : -FI_EAGAIN;
	fastlock_release(&rxm_ep->msg_eq_entry_list_lock);
	return ret;
}

static int rxm_ep_trywait(void *arg)
{
	struct rxm_fabric *rxm_fabric;
	struct rxm_ep *rxm_ep = (struct rxm_ep *)arg;
	struct fid *fids[1] = {&rxm_ep->msg_cq->fid};

	rxm_fabric = container_of(rxm_ep->util_ep.domain->fabric,
				  struct rxm_fabric, util_fabric);
	return fi_trywait(rxm_fabric->msg_fabric, fids, 1);
}

static int rxm_ep_wait_fd_add(struct rxm_ep *rxm_ep, struct util_wait *wait)
{
	int ret;

	ret = ofi_wait_fd_add(wait, rxm_ep->msg_cq_fd, FI_EPOLL_IN,
			      rxm_ep_trywait, rxm_ep,
			      &rxm_ep->util_ep.ep_fid.fid);
	if (ret)
		return ret;

	if (rxm_ep->util_ep.domain->data_progress == FI_PROGRESS_MANUAL) {
		ret = ofi_wait_fd_add(
				wait, slistfd_get_fd(&rxm_ep->msg_eq_entry_list),
				FI_EPOLL_IN, rxm_ep_eq_entry_list_trywait,
				rxm_ep, &rxm_ep->util_ep.ep_fid.fid);
		if (ret) {
			ofi_wait_fd_del(wait, rxm_ep->msg_cq_fd);
			return ret;
		}
	}
	return 0;
}

static int rxm_ep_bind(struct fid *ep_fid, struct fid *bfid, uint64_t flags)
{
	struct rxm_ep *rxm_ep =
		container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	struct util_cq *cq;
	struct util_av *av;
	struct util_cntr *cntr;
	int ret = 0;

	switch (bfid->fclass) {
	case FI_CLASS_AV:
		av = container_of(bfid, struct util_av, av_fid.fid);
		ret = ofi_ep_bind_av(&rxm_ep->util_ep, av);
		if (ret)
			return ret;

		ret = fi_listen(rxm_ep->msg_pep);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
				"Unable to set msg PEP to listen state\n");
			return ret;
		}

		ret = rxm_conn_cmap_alloc(rxm_ep);
		if (ret)
			return ret;

		break;
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct util_cq, cq_fid.fid);

		ret = ofi_ep_bind_cq(&rxm_ep->util_ep, cq, flags);
		if (ret)
			return ret;

		if (!rxm_ep->msg_cq) {
			ret = rxm_ep_msg_cq_open(rxm_ep, cq->wait ?
						 FI_WAIT_FD : FI_WAIT_NONE);
			if (ret)
				return ret;
		}

		if (cq->wait) {
			ret = rxm_ep_wait_fd_add(rxm_ep, cq->wait);
			if (ret)
				goto err;
		}
		break;
	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct util_cntr, cntr_fid.fid);

		ret = ofi_ep_bind_cntr(&rxm_ep->util_ep, cntr, flags);
		if (ret)
			return ret;

		if (!rxm_ep->msg_cq) {
			ret = rxm_ep_msg_cq_open(rxm_ep, cntr->wait ?
						 FI_WAIT_FD : FI_WAIT_NONE);
			if (ret)
				return ret;
		} else if (!rxm_ep->msg_cq_fd && cntr->wait) {
			/* Reopen CQ with WAIT fd set */
			ret = fi_close(&rxm_ep->msg_cq->fid);
			if (ret)
				FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
					"Unable to close msg CQ\n");
			ret = rxm_ep_msg_cq_open(rxm_ep, FI_WAIT_FD);
			if (ret)
				return ret;
		}

		if (cntr->wait) {
			ret = rxm_ep_wait_fd_add(rxm_ep, cntr->wait);
			if (ret)
				goto err;
		}
		break;
	case FI_CLASS_EQ:
		break;
	default:
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "invalid fid class\n");
		ret = -FI_EINVAL;
		break;
	}
	return ret;
err:
	if (fi_close(&rxm_ep->msg_cq->fid))
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to close msg CQ\n");
	return ret;
}

static void rxm_ep_sar_init(struct rxm_ep *rxm_ep)
{
	size_t param;

	/* The SAR initialization must be done after Eager is initialized */
	assert(rxm_ep->eager_limit > 0);

	if (!fi_param_get_size_t(&rxm_prov, "sar_limit", &param)) {
		if (param <= rxm_ep->eager_limit) {
			FI_WARN(&rxm_prov, FI_LOG_CORE,
				"Requsted SAR limit (%zd) less or equal "
				"Eager limit (%zd). SAR limit won't be used. "
				"Messages of size <= SAR limit would be "
				"transmitted via Inject/Eager protocol. "
				"Messages of size > SAR limit would be "
				"transmitted via Rendezvous protocol\n",
				param, rxm_ep->eager_limit);
			param = rxm_ep->eager_limit;
		}

		rxm_ep->sar_limit = param;
	} else {
		size_t sar_limit = rxm_ep->msg_info->tx_attr->size *
				   rxm_ep->eager_limit;

		rxm_ep->sar_limit = (sar_limit > RXM_SAR_LIMIT) ?
				    RXM_SAR_LIMIT : sar_limit;
	}
}

static void rxm_ep_settings_init(struct rxm_ep *rxm_ep)
{
	size_t max_prog_val;

	assert(rxm_ep->msg_info);

	max_prog_val = MIN(rxm_ep->msg_info->tx_attr->size,
			   rxm_ep->msg_info->rx_attr->size) / 2;
	rxm_ep->comp_per_progress = (rxm_ep->comp_per_progress > max_prog_val) ?
				    max_prog_val : rxm_ep->comp_per_progress;

	rxm_ep->msg_mr_local = ofi_mr_local(rxm_ep->msg_info);
	rxm_ep->rxm_mr_local = ofi_mr_local(rxm_ep->rxm_info);

	rxm_ep->inject_limit = rxm_ep->msg_info->tx_attr->inject_size;

	if (!rxm_ep->buffered_min) {
		if (rxm_ep->inject_limit >
		    (sizeof(struct rxm_pkt) + sizeof(struct rxm_rndv_hdr)))
			rxm_ep->buffered_min = (rxm_ep->inject_limit -
						(sizeof(struct rxm_pkt) +
						 sizeof(struct rxm_rndv_hdr)));
		else
			assert(!rxm_ep->buffered_min);
	}

	rxm_ep->eager_limit = rxm_ep->rxm_info->tx_attr->inject_size;

	rxm_ep->min_multi_recv_size = rxm_ep->min_multi_recv_size ?
				      rxm_ep->min_multi_recv_size :
				      rxm_ep->eager_limit;
	rxm_ep->buffered_limit = rxm_ep->buffered_limit ?
				 rxm_ep->buffered_limit :
				 rxm_ep->eager_limit;

	rxm_ep_sar_init(rxm_ep);

 	FI_INFO(&rxm_prov, FI_LOG_CORE,
		"Settings:\n"
		"\t\t MR local: MSG - %d, RxM - %d\n"
		"\t\t Completions per progress: MSG - %zu\n"
		"\t\t Protocol limits: MSG Inject - %zu, "
				      "Eager - %zu, "
				      "SAR - %zu\n",
		rxm_ep->msg_mr_local, rxm_ep->rxm_mr_local,
		rxm_ep->comp_per_progress,
		rxm_ep->inject_limit, rxm_ep->eager_limit, rxm_ep->sar_limit);
}

static int rxm_ep_txrx_res_open(struct rxm_ep *rxm_ep)
{
	int ret;

	rxm_ep_settings_init(rxm_ep);

	ret = rxm_ep_txrx_pool_create(rxm_ep);
	if (ret)
		return ret;

	dlist_init(&rxm_ep->deferred_tx_conn_queue);

	ret = rxm_ep_rx_queue_init(rxm_ep);
	if (ret)
		goto err;

	return FI_SUCCESS;
err:
	rxm_ep_txrx_pool_destroy(rxm_ep);
	return ret;
}

static int rxm_ep_ctrl(struct fid *fid, int command, void *arg)
{
	int ret;
	struct rxm_ep *rxm_ep
		= container_of(fid, struct rxm_ep, util_ep.ep_fid.fid);

	switch (command) {
	case FI_ENABLE:
		if (!rxm_ep->util_ep.rx_cq || !rxm_ep->util_ep.tx_cq)
			return -FI_ENOCQ;
		if (!rxm_ep->util_ep.av || !rxm_ep->cmap)
			return -FI_EOPBADSTATE;
		/* At the time of enabling endpoint, FI_OPT_BUFFERED_MIN,
		 * FI_OPT_BUFFERED_LIMIT should have been frozen so we can
		 * create the rendezvous protocol message pool with the right
		 * size */
		ret = rxm_ep_txrx_res_open(rxm_ep);
		if (ret)
			return ret;

		if (rxm_ep->srx_ctx) {
			ret = rxm_ep_prepost_buf(rxm_ep, rxm_ep->srx_ctx);
			if (ret) {
				rxm_cmap_free(rxm_ep->cmap);
				FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
					"Unable to prepost recv bufs\n");
				goto err;
			}
		}
		break;
	default:
		return -FI_ENOSYS;
	}
	return 0;
err:
	rxm_ep_txrx_res_close(rxm_ep);
	return ret;
}

static struct fi_ops rxm_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxm_ep_close,
	.bind = rxm_ep_bind,
	.control = rxm_ep_ctrl,
	.ops_open = fi_no_ops_open,
};

static int rxm_listener_open(struct rxm_ep *rxm_ep)
{
	struct fi_eq_attr eq_attr = {
		.wait_obj = FI_WAIT_UNSPEC,
		.flags = FI_WRITE,
	};
	int ret;
	struct rxm_fabric *rxm_fabric =
		container_of(rxm_ep->util_ep.domain->fabric,
			     struct rxm_fabric, util_fabric);

	ret = fi_eq_open(rxm_fabric->msg_fabric, &eq_attr, &rxm_ep->msg_eq, NULL);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to open msg EQ\n");
		return ret;
	}

	ret = fi_passive_ep(rxm_fabric->msg_fabric, rxm_ep->msg_info,
			    &rxm_ep->msg_pep, rxm_ep);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to open msg PEP\n");
		goto err;
	}

	ret = fi_pep_bind(rxm_ep->msg_pep, &rxm_ep->msg_eq->fid, 0);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
			"Unable to bind msg PEP to msg EQ\n");
		goto err;
	}

	return 0;
err:
	rxm_listener_close(rxm_ep);
	return ret;
}

static int rxm_ep_msg_res_open(struct rxm_ep *rxm_ep)
{
	int ret;
	struct rxm_domain *rxm_domain =
		container_of(rxm_ep->util_ep.domain, struct rxm_domain, util_domain);

 	ret = ofi_get_core_info(rxm_ep->util_ep.domain->fabric->fabric_fid.api_version,
				NULL, NULL, 0, &rxm_util_prov, rxm_ep->rxm_info,
				rxm_info_to_core, &rxm_ep->msg_info);
	if (ret)
		return ret;

 	if (rxm_ep->msg_info->ep_attr->rx_ctx_cnt == FI_SHARED_CONTEXT) {
		ret = fi_srx_context(rxm_domain->msg_domain, rxm_ep->msg_info->rx_attr,
				     &rxm_ep->srx_ctx, NULL);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
				"Unable to open shared receive context\n");
			goto err1;
		}
	}

 	ret = rxm_listener_open(rxm_ep);
	if (ret)
		goto err2;

 	/* Zero out the port as we would be creating multiple MSG EPs for a single
	 * RXM EP and we don't want address conflicts. */
	if (rxm_ep->msg_info->src_addr) {
		if (((struct sockaddr *)rxm_ep->msg_info->src_addr)->sa_family == AF_INET)
			((struct sockaddr_in *)(rxm_ep->msg_info->src_addr))->sin_port = 0;
		else
			((struct sockaddr_in6 *)(rxm_ep->msg_info->src_addr))->sin6_port = 0;
	}

	return 0;
err2:
	if (rxm_ep->srx_ctx)
		fi_close(&rxm_ep->srx_ctx->fid);
err1:
	fi_freeinfo(rxm_ep->msg_info);
	return ret;
}

int rxm_endpoint(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **ep_fid, void *context)
{
	struct rxm_ep *rxm_ep;
	int ret;

	rxm_ep = calloc(1, sizeof(*rxm_ep));
	if (!rxm_ep)
		return -FI_ENOMEM;

	rxm_ep->rxm_info = fi_dupinfo(info);
	if (!rxm_ep->rxm_info) {
		ret = -FI_ENOMEM;
		goto err1;
	}

	if (fi_param_get_int(&rxm_prov, "comp_per_progress",
			     (int *)&rxm_ep->comp_per_progress))
		rxm_ep->comp_per_progress = 1;

	ret = ofi_endpoint_init(domain, &rxm_util_prov, info, &rxm_ep->util_ep,
				context, &rxm_ep_progress);
	if (ret)
		goto err1;

	ret = rxm_ep_msg_res_open(rxm_ep);
	if (ret)
		goto err2;

	slistfd_init(&rxm_ep->msg_eq_entry_list);
	fastlock_init(&rxm_ep->msg_eq_entry_list_lock);

	*ep_fid = &rxm_ep->util_ep.ep_fid;
	(*ep_fid)->fid.ops = &rxm_ep_fi_ops;
	(*ep_fid)->ops = &rxm_ops_ep;
	(*ep_fid)->cm = &rxm_ops_cm;
	if (rxm_ep->util_ep.domain->threading != FI_THREAD_SAFE) {
		(*ep_fid)->msg = &rxm_ops_msg_thread_unsafe;
		(*ep_fid)->tagged = &rxm_ops_tagged_thread_unsafe;
	} else {
		(*ep_fid)->msg = &rxm_ops_msg;
		(*ep_fid)->tagged = &rxm_ops_tagged;
	}
	(*ep_fid)->rma = &rxm_ops_rma;

	if (rxm_ep->rxm_info->caps & FI_ATOMIC)
		(*ep_fid)->atomic = &rxm_ops_atomic;

	return 0;
err2:
	ofi_endpoint_close(&rxm_ep->util_ep);
err1:
	if (rxm_ep->rxm_info)
		fi_freeinfo(rxm_ep->rxm_info);
	free(rxm_ep);
	return ret;
}
