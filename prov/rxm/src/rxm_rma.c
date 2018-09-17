/*
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
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

#include "rxm.h"

typedef ssize_t rxm_rma_msg_fn(struct fid_ep *ep_fid,
			       const struct fi_msg_rma *msg, uint64_t flags);

static inline ssize_t
rxm_ep_rma_reg_iov(struct rxm_ep *rxm_ep, const struct iovec *msg_iov,
		   void **desc, void **desc_storage,
		   size_t iov_count, uint64_t comp_flags,
		   struct rxm_tx_entry *tx_entry)
{
	size_t i;

	if (rxm_ep->msg_mr_local) {
		if (!rxm_ep->rxm_mr_local) {
			ssize_t ret =
				rxm_ep_msg_mr_regv(rxm_ep, msg_iov, iov_count,
						   comp_flags & (FI_WRITE | FI_READ),
						   tx_entry->mr);
			if (OFI_UNLIKELY(ret))
				return ret;

			for (i = 0; i < iov_count; i++)
				desc_storage[i] = fi_mr_desc(tx_entry->mr[i]);
		} else {
			for (i = 0; i < iov_count; i++)
				desc_storage[i] = fi_mr_desc(desc[i]);
		}
	}
	return FI_SUCCESS;
}

static inline void
rxm_ep_rma_fill_msg(struct fi_msg_rma *msg_rma, struct iovec *iov,
		    size_t iov_count, void **desc,
		    struct rxm_rma_iov_storage *rma_iov,
		    struct rxm_tx_entry *tx_entry,
		    const struct fi_msg_rma *orig_msg)
{
	msg_rma->msg_iov = iov;
	msg_rma->desc = desc;
	msg_rma->iov_count = iov_count;
	msg_rma->addr = orig_msg->addr;
	msg_rma->rma_iov = rma_iov->iov;
	msg_rma->rma_iov_count = rma_iov->count;
	msg_rma->context = tx_entry;
	msg_rma->data = orig_msg->data;
}

static inline void
rxm_ep_rma_fill_msg_no_buf(struct rxm_rma_buf *rma_buf,
			   struct rxm_tx_entry *tx_entry,
			   const struct fi_msg_rma *orig_msg)
{
	rma_buf->rxm_iov.count = (uint8_t)orig_msg->iov_count;

	rxm_ep_rma_fill_msg(&rma_buf->msg, rma_buf->rxm_iov.iov,
			    rma_buf->rxm_iov.count, rma_buf->rxm_iov.desc,
			    &rma_buf->rxm_rma_iov, tx_entry, orig_msg);
}

static inline void
rxm_ep_rma_fill_msg_buf(struct rxm_rma_buf *rma_buf,
			struct rxm_tx_entry *tx_entry,
			const struct fi_msg_rma *orig_msg)
{
	ofi_copy_from_iov(rma_buf->pkt.data, rma_buf->pkt.hdr.size,
			  orig_msg->msg_iov, orig_msg->iov_count, 0);

	rma_buf->rxm_iov.iov[0].iov_base = &rma_buf->pkt.data;
	rma_buf->rxm_iov.iov[0].iov_len = rma_buf->pkt.hdr.size;

	rxm_ep_rma_fill_msg(&rma_buf->msg, rma_buf->rxm_iov.iov,
			    1, &rma_buf->hdr.desc, &rma_buf->rxm_rma_iov,
			    tx_entry, orig_msg);
}

static inline ssize_t
rxm_ep_format_rma_res_lightweight(struct rxm_ep *rxm_ep, uint64_t flags,
				  uint64_t comp_flags, const struct fi_msg_rma *orig_msg,
				  struct rxm_tx_entry **tx_entry)
{
	*tx_entry = rxm_tx_entry_get(&rxm_ep->send_queue);
	if (OFI_UNLIKELY(!*tx_entry)) {
		FI_WARN(&rxm_prov, FI_LOG_CQ,
			"Unable to allocate TX entry for RMA!\n");
		rxm_ep_progress_multi(&rxm_ep->util_ep);
		return -FI_EAGAIN;
	}

	(*tx_entry)->state = RXM_TX_NOBUF;
	(*tx_entry)->context = orig_msg->context;
	(*tx_entry)->flags = flags;
	(*tx_entry)->comp_flags = FI_RMA | comp_flags;
	(*tx_entry)->count = orig_msg->iov_count;

	return FI_SUCCESS;
}

static inline ssize_t
rxm_ep_format_rma_buf(struct rxm_ep *rxm_ep, size_t total_size,
		      const struct fi_msg_rma *orig_msg,
		      struct rxm_rma_buf **rma_buf, struct rxm_tx_entry *tx_entry)
{
	size_t i;

	*rma_buf = rxm_rma_buf_get(rxm_ep);
	if (OFI_UNLIKELY(!*rma_buf))
		return -FI_EAGAIN;

	tx_entry->state = RXM_TX_RMA;
	tx_entry->rma_buf = *rma_buf;
	(*rma_buf)->pkt.hdr.size = total_size;
	(*rma_buf)->rxm_iov.count = orig_msg->iov_count;
	(*rma_buf)->rxm_rma_iov.count = orig_msg->rma_iov_count;
	for (i = 0; i < orig_msg->iov_count; i++)
		(*rma_buf)->rxm_iov.iov[i] = orig_msg->msg_iov[i];
	for (i = 0; i < orig_msg->rma_iov_count; i++)
		(*rma_buf)->rxm_rma_iov.iov[i] = orig_msg->rma_iov[i];

	return FI_SUCCESS;
}

static inline ssize_t
rxm_ep_format_rma_res(struct rxm_ep *rxm_ep, size_t total_size,
		      uint64_t flags, uint64_t comp_flags,
		      const struct fi_msg_rma *orig_msg,
		      struct rxm_rma_buf **rma_buf,
		      struct rxm_tx_entry **tx_entry)
{
	ssize_t ret;

	ret = rxm_ep_format_rma_res_lightweight(rxm_ep, flags, comp_flags,
						orig_msg, tx_entry);
	if (OFI_UNLIKELY(ret))
		return ret;

	ret = rxm_ep_format_rma_buf(rxm_ep, total_size, orig_msg,
				    rma_buf, *tx_entry);
	if (OFI_UNLIKELY(ret))
		goto err;

	return FI_SUCCESS;
err:
	FI_WARN(&rxm_prov, FI_LOG_CQ, "Unable to allocate RMA resources!\n");
	rxm_tx_entry_release(&rxm_ep->send_queue, *tx_entry);
	return ret;
}

void rxm_ep_handle_postponed_rma_op(struct rxm_ep *rxm_ep,
				    struct rxm_conn *rxm_conn,
				    struct rxm_tx_entry *tx_entry)
{
	ssize_t ret;
	struct util_cntr *cntr;
	struct util_cq *cq;
	struct fi_cq_err_entry err_entry;

	FI_DBG(&rxm_prov, FI_LOG_EP_DATA,
	       "Perform deffered RMA operation (len - %"PRIu64") for %p conn\n",
	       tx_entry->rma_buf->pkt.hdr.size, rxm_conn);

	if (tx_entry->comp_flags & FI_WRITE) {
		uint64_t flags = ((tx_entry->flags & FI_INJECT) ?
				  ((tx_entry->flags & ~FI_INJECT) |
				   FI_COMPLETION) : tx_entry->flags);
		ret = fi_writemsg(rxm_conn->msg_ep,
				  &tx_entry->rma_buf->msg,
				  flags);
		if (OFI_UNLIKELY(ret)) {
			cntr = rxm_ep->util_ep.wr_cntr;
			cq = rxm_ep->util_ep.tx_cq;
			goto err;
		}
	} else if (tx_entry->comp_flags & FI_READ) {
		ret = fi_readmsg(rxm_conn->msg_ep,
				 &tx_entry->rma_buf->msg,
				 tx_entry->flags);
		if (OFI_UNLIKELY(ret)) {
			cntr = rxm_ep->util_ep.rd_cntr;
			cq = rxm_ep->util_ep.tx_cq;
			goto err;
		}
	} else {
		assert(0);
	}

	return;
err:
	FI_WARN(&rxm_prov, FI_LOG_EP_DATA,
		"Unable to perform deffered RMA operation\n");

	memset(&err_entry, 0, sizeof(err_entry));
	err_entry.op_context = tx_entry->context;
	err_entry.prov_errno = (int)ret;

	rxm_cntr_incerr(cntr);
	if (ofi_cq_write_error(cq, &err_entry))
		assert(0);
}

static inline ssize_t
rxm_ep_format_rma_inject_res(struct rxm_ep *rxm_ep, size_t total_size,
			     uint64_t flags, uint64_t comp_flags,
			     const struct fi_msg_rma *orig_msg,
			     struct rxm_rma_buf **rma_buf,
			     struct rxm_tx_entry **tx_entry)
{
	ssize_t ret = rxm_ep_format_rma_res(rxm_ep, total_size, flags, comp_flags,
					    orig_msg, rma_buf, tx_entry);
	if (OFI_UNLIKELY(ret))
		return ret;

	rxm_ep_rma_fill_msg_buf(*rma_buf, *tx_entry, orig_msg);

	return ret;
}

static inline ssize_t
rxm_ep_format_rma_non_inject_res(struct rxm_ep *rxm_ep, size_t total_size,
				 uint64_t flags, uint64_t comp_flags,
				 const struct fi_msg_rma *orig_msg,
				 struct rxm_rma_buf **rma_buf,
				 struct rxm_tx_entry **tx_entry)
{
	ssize_t ret = rxm_ep_format_rma_res(rxm_ep, total_size, flags, comp_flags,
					    orig_msg, rma_buf, tx_entry);
	if (OFI_UNLIKELY(ret))
		return ret;

	ret = rxm_ep_rma_reg_iov(rxm_ep, (*rma_buf)->rxm_iov.iov,
				 /* addr of desc from rma_buf will be assign to itself */
				 orig_msg->desc,
				 (*rma_buf)->rxm_iov.desc,
				 orig_msg->iov_count,
				 comp_flags & (FI_WRITE | FI_READ), *tx_entry);
	if (OFI_UNLIKELY(ret))
		goto err;

	rxm_ep_rma_fill_msg_no_buf(*rma_buf, *tx_entry, orig_msg);

	return ret;
err:
	rxm_rma_buf_release(rxm_ep, (*tx_entry)->rma_buf);
	rxm_tx_entry_release(&rxm_ep->send_queue, *tx_entry);
	return ret;
}

static inline int
rxm_ep_postpone_rma(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
		    size_t total_size, uint64_t flags,
		    uint64_t comp_flags, const struct fi_msg_rma *orig_msg)
{
	struct rxm_tx_entry *tx_entry;
	struct rxm_rma_buf *rma_buf;
	int ret;

	if (flags & FI_INJECT) {
		assert(comp_flags & FI_WRITE);
		ret = rxm_ep_format_rma_inject_res(rxm_ep, total_size,
						   flags, comp_flags, orig_msg,
						   &rma_buf, &tx_entry);
	} else {
		ret = rxm_ep_format_rma_non_inject_res(rxm_ep, total_size,
						       flags, comp_flags, orig_msg,
						       &rma_buf, &tx_entry);
	}
	if (OFI_UNLIKELY(ret))
		return ret;

	dlist_insert_tail(&tx_entry->postponed_entry,
			  &rxm_conn->postponed_tx_list);

	return ret;
}

static ssize_t
rxm_ep_rma_common(struct rxm_ep *rxm_ep, const struct fi_msg_rma *msg, uint64_t flags,
		  rxm_rma_msg_fn rma_msg, uint64_t comp_flags)
{
	struct rxm_tx_entry *tx_entry;
	struct fi_msg_rma msg_rma = *msg;
	struct util_cmap_handle *handle;
	struct rxm_conn *rxm_conn;
	void *mr_desc[RXM_IOV_LIMIT] = { 0 };
	int ret;

	assert(msg->rma_iov_count <= rxm_ep->rxm_info->tx_attr->rma_iov_limit);

	fastlock_acquire(&rxm_ep->util_ep.cmap->lock);
	handle = ofi_cmap_acquire_handle(rxm_ep->util_ep.cmap, msg->addr);
	if (OFI_UNLIKELY(!handle)) {
		fastlock_release(&rxm_ep->util_ep.cmap->lock);
		return -FI_EAGAIN;
	} else if (OFI_UNLIKELY(handle->state != CMAP_CONNECTED)) {
		ret = ofi_cmap_handle_connect(rxm_ep->util_ep.cmap,
					      msg->addr, handle);
		if (OFI_UNLIKELY(ret != -FI_EAGAIN))
			goto cmap_err;
		rxm_conn = container_of(handle, struct rxm_conn, handle);
		ret = rxm_ep_postpone_rma(rxm_ep, rxm_conn,
					  ofi_total_iov_len(msg->msg_iov,
							    msg->iov_count),
					  flags, comp_flags, msg);
cmap_err:
		fastlock_release(&rxm_ep->util_ep.cmap->lock);
		return ret;
	}
	fastlock_release(&rxm_ep->util_ep.cmap->lock);
	rxm_conn = container_of(handle, struct rxm_conn, handle);

	ret = rxm_ep_format_rma_res_lightweight(rxm_ep, flags, comp_flags,
						msg, &tx_entry);
	if (OFI_UNLIKELY(ret))
		return -FI_EAGAIN;

	msg_rma.context = tx_entry;

	ret = rxm_ep_rma_reg_iov(rxm_ep, msg->msg_iov, msg_rma.desc, mr_desc,
				 msg->iov_count, comp_flags & (FI_WRITE | FI_READ),
				 tx_entry);
	if (OFI_UNLIKELY(ret))
		goto err;
	msg_rma.desc = mr_desc;

	ret = rma_msg(rxm_conn->msg_ep, &msg_rma, flags);
	if (OFI_LIKELY(!ret))
		return ret;

	if ((rxm_ep->msg_mr_local) && (!rxm_ep->rxm_mr_local))
		rxm_ep_msg_mr_closev(tx_entry->mr, tx_entry->count);
err:
	rxm_tx_entry_release(&rxm_ep->send_queue, tx_entry);
	return ret;
}

static ssize_t rxm_ep_readmsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
			      uint64_t flags)
{
	struct rxm_ep *rxm_ep =
		container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);

	return rxm_ep_rma_common(rxm_ep, msg, flags, fi_readmsg, FI_READ);
}

static ssize_t rxm_ep_readv(struct fid_ep *ep_fid, const struct iovec *iov,
			    void **desc, size_t count, fi_addr_t src_addr,
			    uint64_t addr, uint64_t key, void *context)
{
	struct fi_rma_iov rma_iov = {
		.addr = addr,
		.len = ofi_total_iov_len(iov, count),
		.key = key,
	};
	struct fi_msg_rma msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = src_addr,
		.rma_iov = &rma_iov,
		.rma_iov_count = 1,
		.context = context,
		.data = 0,
	};
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_readmsg(ep_fid, &msg, rxm_ep_tx_flags(rxm_ep));
}

static ssize_t rxm_ep_read(struct fid_ep *ep_fid, void *buf, size_t len,
			   void *desc, fi_addr_t src_addr, uint64_t addr,
			   uint64_t key, void *context)
{
	struct iovec iov = {
		.iov_base = (void*)buf,
		.iov_len = len,
	};
	struct fi_rma_iov rma_iov = {
		.addr = addr,
		.len = len,
		.key = key,
	};
	struct fi_msg_rma msg = {
		.msg_iov = &iov,
		.desc = &desc,
		.iov_count = 1,
		.addr = src_addr,
		.rma_iov = &rma_iov,
		.rma_iov_count = 1,
		.context = context,
		.data = 0,
	};
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_readmsg(ep_fid, &msg, rxm_ep_tx_flags(rxm_ep));
}

static ssize_t
rxm_ep_rma_inject(struct rxm_ep *rxm_ep, const struct fi_msg_rma *msg, uint64_t flags)
{
	struct rxm_tx_entry *tx_entry;
	struct rxm_rma_buf *rma_buf;
	struct util_cmap_handle *handle;
	struct rxm_conn *rxm_conn;
	size_t total_size = ofi_total_iov_len(msg->msg_iov, msg->iov_count);
	ssize_t ret;

	assert(msg->rma_iov_count <= rxm_ep->rxm_info->tx_attr->rma_iov_limit);

	fastlock_acquire(&rxm_ep->util_ep.cmap->lock);
	handle = ofi_cmap_acquire_handle(rxm_ep->util_ep.cmap, msg->addr);
	if (OFI_UNLIKELY(!handle)) {
		fastlock_release(&rxm_ep->util_ep.cmap->lock);
		return -FI_EAGAIN;
	} else if (OFI_UNLIKELY(handle->state != CMAP_CONNECTED)) {
		ret = ofi_cmap_handle_connect(rxm_ep->util_ep.cmap,
					      msg->addr, handle);
		if (OFI_UNLIKELY(ret != -FI_EAGAIN))
			goto cmap_err;
		rxm_conn = container_of(handle, struct rxm_conn, handle);
		ret = rxm_ep_postpone_rma(rxm_ep, rxm_conn, total_size,
					  flags, FI_WRITE, msg);
cmap_err:
		fastlock_release(&rxm_ep->util_ep.cmap->lock);
		return ret;
	}
	fastlock_release(&rxm_ep->util_ep.cmap->lock);
	rxm_conn = container_of(handle, struct rxm_conn, handle);

	if (OFI_UNLIKELY(total_size > rxm_ep->rxm_info->tx_attr->inject_size))
		return -FI_EMSGSIZE;

	/* Use fi_inject_write instead of fi_writemsg since the latter generates
	 * completion by default */
	if ((total_size <= rxm_ep->msg_info->tx_attr->inject_size) &&
	    !(flags & FI_COMPLETION)) {
		if (flags & FI_REMOTE_CQ_DATA)
			ret = fi_inject_writedata(rxm_conn->msg_ep,
						  msg->msg_iov->iov_base,
						  msg->msg_iov->iov_len, msg->data,
						  msg->addr, msg->rma_iov->addr,
						  msg->rma_iov->key);
		else
			ret = fi_inject_write(rxm_conn->msg_ep,
					      msg->msg_iov->iov_base,
					      msg->msg_iov->iov_len, msg->addr,
					      msg->rma_iov->addr,
					      msg->rma_iov->key);
		if (OFI_LIKELY(!ret))
			rxm_cntr_inc(rxm_ep->util_ep.wr_cntr);
		return ret;
	}

	ret = rxm_ep_format_rma_inject_res(rxm_ep, total_size, flags, FI_WRITE,
					   msg, &rma_buf, &tx_entry);
	if (OFI_UNLIKELY(ret))
		return ret;
	flags = (flags & ~FI_INJECT) | FI_COMPLETION;
	ret = fi_writemsg(rxm_conn->msg_ep, &rma_buf->msg, flags);
	if (OFI_UNLIKELY(ret)) {
		if (ret == -FI_EAGAIN)
			rxm_ep_progress_multi(&rxm_ep->util_ep);
		goto err;
	}
	return 0;
err:
	rxm_rma_buf_release(rxm_ep, tx_entry->rma_buf);
	rxm_tx_entry_release(&rxm_ep->send_queue, tx_entry);
	return ret;
}

static ssize_t rxm_ep_writemsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
			       uint64_t flags)
{
	struct rxm_ep *rxm_ep =
		container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);

	if (flags & FI_INJECT)
		return rxm_ep_rma_inject(rxm_ep, msg, flags);
	else
		return rxm_ep_rma_common(rxm_ep, msg, flags,
					 fi_writemsg, FI_WRITE);
}

static ssize_t rxm_ep_writev(struct fid_ep *ep_fid, const struct iovec *iov,
			     void **desc, size_t count, fi_addr_t dest_addr,
			     uint64_t addr, uint64_t key, void *context)
{
	struct fi_rma_iov rma_iov = {
		.addr = addr,
		.len = ofi_total_iov_len(iov, count),
		.key = key,
	};
	struct fi_msg_rma msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = dest_addr,
		.rma_iov = &rma_iov,
		.rma_iov_count = 1,
		.context = context,
		.data = 0,
	};
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_writemsg(ep_fid, &msg, rxm_ep_tx_flags(rxm_ep));
}

static ssize_t rxm_ep_writedata(struct fid_ep *ep_fid, const void *buf,
				size_t len, void *desc, uint64_t data,
				fi_addr_t dest_addr, uint64_t addr,
				uint64_t key, void *context)
{
	struct fi_rma_iov rma_iov = {
		.addr = addr,
		.len = len,
		.key = key,
	};
	struct iovec iov = {
		.iov_base = (void*)buf,
		.iov_len = len,
	};
	struct fi_msg_rma msg = {
		.msg_iov = &iov,
		.desc = &desc,
		.iov_count = 1,
		.addr = dest_addr,
		.rma_iov = &rma_iov,
		.rma_iov_count = 1,
		.context = context,
		.data = data,
	};
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_writemsg(ep_fid, &msg, rxm_ep_tx_flags(rxm_ep) |
			       FI_REMOTE_CQ_DATA);
}

static ssize_t rxm_ep_write(struct fid_ep *ep_fid, const void *buf,
			    size_t len, void *desc, fi_addr_t dest_addr,
			    uint64_t addr, uint64_t key, void *context)
{
	struct fi_rma_iov rma_iov = {
		.addr = addr,
		.len = len,
		.key = key,
	};
	struct iovec iov = {
		.iov_base = (void*)buf,
		.iov_len = len,
	};
	struct fi_msg_rma msg = {
		.msg_iov = &iov,
		.desc = &desc,
		.iov_count = 1,
		.addr = dest_addr,
		.rma_iov = &rma_iov,
		.rma_iov_count = 1,
		.context = context,
		.data = 0,
	};
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_writemsg(ep_fid, &msg, rxm_ep_tx_flags(rxm_ep));
}

static ssize_t rxm_ep_inject_write(struct fid_ep *ep_fid, const void *buf,
			     size_t len, fi_addr_t dest_addr, uint64_t addr,
			     uint64_t key)
{
	struct fi_rma_iov rma_iov = {
		.addr = addr,
		.len = len,
		.key = key,
	};
	struct iovec iov = {
		.iov_base = (void*)buf,
		.iov_len = len,
	};
	struct fi_msg_rma msg = {
		.msg_iov = &iov,
		.desc = NULL,
		.iov_count = 1,
		.addr = dest_addr,
		.rma_iov = &rma_iov,
		.rma_iov_count = 1,
		.context = NULL,
		.data = 0,
	};
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_writemsg(ep_fid, &msg,
			       (rxm_ep_tx_flags(rxm_ep) & ~FI_COMPLETION) |
			       FI_INJECT);
}

static ssize_t rxm_ep_inject_writedata(struct fid_ep *ep_fid, const void *buf,
				       size_t len, uint64_t data,
				       fi_addr_t dest_addr, uint64_t addr,
				       uint64_t key)
{
	struct fi_rma_iov rma_iov = {
		.addr = addr,
		.len = len,
		.key = key,
	};
	struct iovec iov = {
		.iov_base = (void*)buf,
		.iov_len = len,
	};
	struct fi_msg_rma msg = {
		.msg_iov = &iov,
		.desc = NULL,
		.iov_count = 1,
		.addr = dest_addr,
		.rma_iov = &rma_iov,
		.rma_iov_count = 1,
		.context = NULL,
		.data = data,
	};
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_writemsg(ep_fid, &msg,
			       (rxm_ep_tx_flags(rxm_ep) & ~FI_COMPLETION) |
			       FI_INJECT | FI_REMOTE_CQ_DATA);
}

struct fi_ops_rma rxm_ops_rma = {
	.size = sizeof (struct fi_ops_rma),
	.read = rxm_ep_read,
	.readv = rxm_ep_readv,
	.readmsg = rxm_ep_readmsg,
	.write = rxm_ep_write,
	.writev = rxm_ep_writev,
	.writemsg = rxm_ep_writemsg,
	.inject = rxm_ep_inject_write,
	.writedata = rxm_ep_writedata,
	.injectdata = rxm_ep_inject_writedata,
};
