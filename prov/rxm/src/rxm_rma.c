/*
 * Copyright (c) 2017-2020 Intel Corporation. All rights reserved.
 * (C) Copyright 2020 Hewlett Packard Enterprise Development LP
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


static ssize_t
rxm_ep_rma_reg_iov(struct rxm_ep *rxm_ep, const struct iovec *msg_iov,
		   void **desc, void **desc_storage, size_t iov_count,
		   uint64_t access, struct rxm_tx_buf *rma_buf)
{
	size_t i, ret, page_size;
	struct iovec reg_iov[RXM_IOV_LIMIT];

	assert(iov_count <= RXM_IOV_LIMIT);

	if (!rxm_ep->msg_mr_local)
		return FI_SUCCESS;

	if (!rxm_ep->rdm_mr_local) {
		page_size = page_sizes[OFI_PAGE_SIZE];
		for (i = 0; i < iov_count; i++) {
			if (!msg_iov[i].iov_len) {
				reg_iov[i] = msg_iov[i];
				continue;
			}
			reg_iov[i].iov_base = ofi_get_page_start(
						msg_iov[i].iov_base, page_size);
			reg_iov[i].iov_len = ofi_get_page_bytes(
						msg_iov[i].iov_base,
						msg_iov[i].iov_len, page_size);
		}

		ret = rxm_msg_mr_regv(rxm_ep, reg_iov, iov_count, SIZE_MAX,
				      access, rma_buf->rma.mr);

		if (ret)
			ret = rxm_msg_mr_regv(rxm_ep, msg_iov, iov_count,
					      SIZE_MAX, access,
					      rma_buf->rma.mr);

		if (OFI_UNLIKELY(ret))
			return ret;

		for (i = 0; i < iov_count; i++)
			desc_storage[i] = fi_mr_desc(rma_buf->rma.mr[i]);
		rma_buf->rma.count = (uint8_t) iov_count;
	} else {
		for (i = 0; i < iov_count; i++)
			desc_storage[i] =
				fi_mr_desc(((struct rxm_mr *) desc[i])->msg_mr);
	}
	return FI_SUCCESS;
}

/* An inline write does not need its local buffer registered. */
static bool
rxm_ep_rma_write_is_inline(struct rxm_ep *rxm_ep, const struct fi_msg_rma *msg)
{
	uint64_t device;
	size_t len;

	if (!rxm_ep->msg_mr_local || rxm_ep->rdm_mr_local)
		return false;

	len = ofi_total_iov_len(msg->msg_iov, msg->iov_count);
	return len <= rxm_ep->inject_limit &&
	       rxm_iov_desc_to_hmem_iface_dev(msg->msg_iov, msg->desc,
					      msg->iov_count, &device) == FI_HMEM_SYSTEM;
}

static bool
rxm_ep_rma_can_bounce(struct rxm_ep *rxm_ep, const struct fi_msg_rma *msg,
		      size_t len, size_t bounce_size)
{
	uint64_t device;

	return len && len <= bounce_size &&
	       rxm_ep->msg_mr_local && !rxm_ep->rdm_mr_local &&
	       rxm_iov_desc_to_hmem_iface_dev(msg->msg_iov, msg->desc,
					      msg->iov_count, &device) == FI_HMEM_SYSTEM;
}

/* Point the read at pkt.data, which is already registered, and keep the user
 * iov for rxm_finish_rma() to copy into. */
static void
rxm_ep_format_bounce_read(struct rxm_tx_buf *rma_buf, struct fi_msg_rma *msg,
			  struct iovec *bounce_iov, size_t len)
{
	assert(msg->iov_count <= RXM_IOV_LIMIT);
	memcpy(rma_buf->rma.iov, msg->msg_iov,
	       sizeof(*msg->msg_iov) * msg->iov_count);
	rma_buf->rma.iov_count = (uint8_t) msg->iov_count;
	rma_buf->pkt.hdr.size = len;

	bounce_iov->iov_base = &rma_buf->pkt.data;
	bounce_iov->iov_len = len;
	msg->msg_iov = bounce_iov;
	msg->desc = &rma_buf->hdr.desc;
	msg->iov_count = 1;
}

/* Copy the write into pkt.data, which is already registered, and send it from
 * there.  rma.iov_count stays 0, so rxm_finish_rma() copies nothing back. */
static void
rxm_ep_format_bounce_write(struct rxm_tx_buf *rma_buf, struct fi_msg_rma *msg,
			   struct iovec *bounce_iov, size_t len)
{
	size_t copied __attribute__((unused));

	copied = ofi_copy_from_iov(&rma_buf->pkt.data, len, msg->msg_iov,
				   msg->iov_count, 0);
	assert(copied == len);
	rma_buf->pkt.hdr.size = len;

	bounce_iov->iov_base = &rma_buf->pkt.data;
	bounce_iov->iov_len = len;
	msg->msg_iov = bounce_iov;
	msg->desc = &rma_buf->hdr.desc;
	msg->iov_count = 1;
}

static ssize_t
rxm_ep_rma_common(struct rxm_ep *rxm_ep, const struct fi_msg_rma *msg,
		  uint64_t flags, ssize_t (*rma_msg)(struct fid_ep *ep_fid,
		  const struct fi_msg_rma *msg, uint64_t flags),
		  uint64_t comp_flags)
{
	struct rxm_tx_buf *rma_buf;
	struct fi_msg_rma msg_rma = *msg;
	struct rxm_conn *rxm_conn;
	void *mr_desc[RXM_IOV_LIMIT] = { 0 };
	struct iovec bounce_iov;
	uint64_t msg_flags;
	size_t len;
	ssize_t ret;

	assert(msg->rma_iov_count <= rxm_ep->rxm_info->tx_attr->rma_iov_limit);

	ofi_genlock_lock(&rxm_ep->util_ep.lock);

	ret = rxm_get_conn(rxm_ep, msg->addr, &rxm_conn);
	if (ret)
		goto unlock;

	rma_buf = rxm_get_tx_buf(rxm_ep);
	if (!rma_buf) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	rma_buf->hdr.state = RXM_RMA;
	rma_buf->pkt.ctrl_hdr.type = rxm_ctrl_eager;
	rma_buf->app_context = msg->context;
	rma_buf->flags = flags;
	rma_buf->rma.count = 0;
	rma_buf->rma.iov_count = 0;
	msg_flags = flags;

	len = ofi_total_iov_len(msg_rma.msg_iov, msg_rma.iov_count);
	if ((comp_flags & FI_WRITE) &&
	    rxm_ep_rma_write_is_inline(rxm_ep, &msg_rma)) {
		/* rma_buf->flags keeps the caller's flags, so the write still
		 * completes as a normal write in rxm_finish_rma() */
		msg_rma.desc = NULL;
		msg_flags |= FI_INJECT;
	} else if ((comp_flags & FI_WRITE) &&
		   rxm_ep_rma_can_bounce(rxm_ep, &msg_rma, len,
					 rxm_rma_write_bounce_size)) {
		/* the caller's flags are kept, so this completes as the write
		 * it is, but the buffer is reusable on return */
		rxm_ep_format_bounce_write(rma_buf, &msg_rma, &bounce_iov, len);
	} else if ((comp_flags & FI_READ) &&
		   rxm_ep_rma_can_bounce(rxm_ep, &msg_rma, len,
					 rxm_rma_read_bounce_size)) {
		rxm_ep_format_bounce_read(rma_buf, &msg_rma, &bounce_iov, len);
	} else {
		ret = rxm_ep_rma_reg_iov(rxm_ep, msg_rma.msg_iov, msg_rma.desc,
					 mr_desc, msg_rma.iov_count,
					 comp_flags & (FI_WRITE | FI_READ),
					 rma_buf);
		if (OFI_UNLIKELY(ret))
			goto release;

		msg_rma.desc = mr_desc;
	}
	msg_rma.context = rma_buf;

	ret = rma_msg(rxm_conn_msg_ep(rxm_conn, NULL), &msg_rma, msg_flags);
	if (!ret)
		goto unlock;

	if ((rxm_ep->msg_mr_local) && (!rxm_ep->rdm_mr_local))
		rxm_msg_mr_closev(rma_buf->rma.mr, rma_buf->rma.count);
release:
	rxm_free_tx_buf(rxm_ep, rma_buf);
unlock:
	ofi_genlock_unlock(&rxm_ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_ep_readmsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
	       uint64_t flags)
{
	struct rxm_ep *rxm_ep;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	return rxm_ep_rma_common(rxm_ep, msg, flags | rxm_ep->util_ep.tx_msg_flags,
				 fi_readmsg, FI_READ);
}

static ssize_t rxm_ep_readv(struct fid_ep *ep_fid, const struct iovec *iov,
			    void **desc, size_t count, fi_addr_t src_addr,
			    uint64_t addr, uint64_t key, void *context)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);
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

	return rxm_ep_rma_common(rxm_ep, &msg, rxm_ep->util_ep.tx_op_flags,
				 fi_readmsg, FI_READ);
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
	struct rxm_ep *rxm_ep;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	return rxm_ep_rma_common(rxm_ep, &msg, rxm_ep->util_ep.tx_op_flags,
				 fi_readmsg, FI_READ);
}

static void
rxm_ep_format_rma_msg(struct rxm_tx_buf *rma_buf,
		      const struct fi_msg_rma *orig_msg,
		      struct iovec *rxm_iov, struct fi_msg_rma *rxm_msg)
{
	ssize_t ret __attribute__((unused));
	rxm_msg->context = rma_buf;
	rxm_msg->addr = orig_msg->addr;
	rxm_msg->data = orig_msg->data;

	ret = rxm_copy_from_hmem_iov(orig_msg->desc, rma_buf->pkt.data,
				     rma_buf->pkt.hdr.size, orig_msg->msg_iov,
				     orig_msg->iov_count, 0);
	assert((size_t) ret == rma_buf->pkt.hdr.size);

	rxm_iov->iov_base = &rma_buf->pkt.data;
	rxm_iov->iov_len = rma_buf->pkt.hdr.size;
	rxm_msg->msg_iov = rxm_iov;
	rxm_msg->desc = &rma_buf->hdr.desc;
	rxm_msg->iov_count = 1;

	rxm_msg->rma_iov = orig_msg->rma_iov;
	rxm_msg->rma_iov_count = orig_msg->rma_iov_count;
}

static ssize_t
rxm_ep_rma_emulate_inject_msg(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
			      size_t total_size, const struct fi_msg_rma *msg,
			      uint64_t flags)
{
	struct rxm_tx_buf *rma_buf;
	ssize_t ret;
	struct iovec rxm_msg_iov = { 0 };
	struct fi_msg_rma rxm_rma_msg = { 0 };

	assert(msg->rma_iov_count <= rxm_ep->rxm_info->tx_attr->rma_iov_limit);

	rma_buf = rxm_get_tx_buf(rxm_ep);
	if (!rma_buf)
		return -FI_EAGAIN;

	rma_buf->hdr.state = RXM_RMA;
	rma_buf->pkt.ctrl_hdr.type = rxm_ctrl_eager;
	rma_buf->pkt.hdr.size = total_size;
	rma_buf->app_context = msg->context;
	rma_buf->flags = flags;
	rxm_ep_format_rma_msg(rma_buf, msg, &rxm_msg_iov, &rxm_rma_msg);

	flags = (flags & ~FI_INJECT) | FI_COMPLETION;

	ret = fi_writemsg(rxm_conn_msg_ep(rxm_conn, NULL), &rxm_rma_msg, flags);
	if (ret) {
		if (ret == -FI_EAGAIN)
			rxm_ep_do_progress(&rxm_ep->util_ep);
		rxm_free_tx_buf(rxm_ep, rma_buf);
	}
	return ret;
}

static ssize_t
rxm_ep_rma_emulate_inject(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
			  const void *buf, size_t len, uint64_t data,
			  fi_addr_t dest_addr, uint64_t addr, uint64_t key,
			  uint64_t flags)
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

	return rxm_ep_rma_emulate_inject_msg(rxm_ep, rxm_conn, len, &msg, flags);
}

static ssize_t
rxm_ep_rma_inject_common(struct rxm_ep *rxm_ep, const struct fi_msg_rma *msg,
			 uint64_t flags)
{
	struct rxm_conn *rxm_conn;
	size_t total_size = ofi_total_iov_len(msg->msg_iov, msg->iov_count);
	ssize_t ret;

	assert(total_size <= rxm_ep->rxm_info->tx_attr->inject_size);

	ofi_genlock_lock(&rxm_ep->util_ep.lock);

	ret = rxm_get_conn(rxm_ep, msg->addr, &rxm_conn);
	if (ret)
		goto unlock;

	if ((total_size > rxm_ep->rxm_info->tx_attr->inject_size) ||
	    rxm_ep->util_ep.cntrs[CNTR_WR] ||
	    (flags & FI_COMPLETION) || (msg->iov_count > 1) ||
	    (msg->rma_iov_count > 1)) {
		ret = rxm_ep_rma_emulate_inject_msg(rxm_ep, rxm_conn,
						    total_size, msg, flags);
		goto unlock;
	}

	if (flags & FI_REMOTE_CQ_DATA) {
		ret = fi_inject_writedata(rxm_conn_msg_ep(rxm_conn, NULL),
					  msg->msg_iov->iov_base,
					  msg->msg_iov->iov_len, msg->data,
					  msg->addr, msg->rma_iov->addr,
					  msg->rma_iov->key);
	} else {
		ret = fi_inject_write(rxm_conn_msg_ep(rxm_conn, NULL),
				      msg->msg_iov->iov_base,
				      msg->msg_iov->iov_len, msg->addr,
				      msg->rma_iov->addr,
				      msg->rma_iov->key);
	}
	if (ret == -FI_EAGAIN)
		rxm_ep_do_progress(&rxm_ep->util_ep);
	else if (ret)
		FI_WARN(&rxm_prov, FI_LOG_EP_DATA, "fi_inject_write* for"
			"MSG provider failed: %zd\n", ret);
unlock:
	ofi_genlock_unlock(&rxm_ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_ep_generic_writemsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
			uint64_t flags)
{
	struct rxm_ep *rxm_ep;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	if (flags & FI_INJECT)
		return rxm_ep_rma_inject_common(rxm_ep, msg, flags);
	else
		return rxm_ep_rma_common(rxm_ep, msg, flags,
					 fi_writemsg, FI_WRITE);
}

static ssize_t
rxm_ep_writemsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
		uint64_t flags)
{
	struct rxm_ep *rxm_ep;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	return rxm_ep_generic_writemsg(ep_fid, msg, flags |
				       rxm_ep->util_ep.tx_msg_flags);
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
	struct rxm_ep *rxm_ep;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	return rxm_ep_generic_writemsg(ep_fid, &msg,
				       rxm_ep->util_ep.tx_op_flags);
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
	struct rxm_ep *rxm_ep;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	return rxm_ep_generic_writemsg(ep_fid, &msg, rxm_ep->util_ep.tx_op_flags |
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
	struct rxm_ep *rxm_ep;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	return rxm_ep_generic_writemsg(ep_fid, &msg, rxm_ep->util_ep.tx_op_flags);
}

static ssize_t rxm_ep_inject_write(struct fid_ep *ep_fid, const void *buf,
				   size_t len, fi_addr_t dest_addr,
				   uint64_t addr, uint64_t key)
{
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep;
	ssize_t ret;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&rxm_ep->util_ep.lock);

	ret = rxm_get_conn(rxm_ep, dest_addr, &rxm_conn);
	if (ret)
		goto unlock;

	if (len > rxm_ep->inject_limit || rxm_ep->util_ep.cntrs[CNTR_WR]) {
		ret = rxm_ep_rma_emulate_inject(rxm_ep, rxm_conn, buf, len, 0,
						dest_addr, addr, key,
						FI_INJECT);
		goto unlock;
	}

	ret = fi_inject_write(rxm_conn_msg_ep(rxm_conn, NULL),
			      buf, len, dest_addr, addr, key);
	if (ret == -FI_EAGAIN)
		rxm_ep_do_progress(&rxm_ep->util_ep);
	else if (ret)
		FI_WARN(&rxm_prov, FI_LOG_EP_DATA, "fi_inject_write for"
			" MSG provider failed: %zd\n", ret);
unlock:
	ofi_genlock_unlock(&rxm_ep->util_ep.lock);
	return ret;
}

static ssize_t rxm_ep_inject_writedata(struct fid_ep *ep_fid, const void *buf,
				       size_t len, uint64_t data,
				       fi_addr_t dest_addr, uint64_t addr,
				       uint64_t key)
{
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep;
	ssize_t ret;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&rxm_ep->util_ep.lock);

	ret = rxm_get_conn(rxm_ep, dest_addr, &rxm_conn);
	if (ret)
		goto unlock;

	if (len > rxm_ep->inject_limit || rxm_ep->util_ep.cntrs[CNTR_WR]) {
		ret = rxm_ep_rma_emulate_inject(
			rxm_ep, rxm_conn, buf, len, data, dest_addr,
			addr, key, FI_REMOTE_CQ_DATA | FI_INJECT);
		goto unlock;
	}

	ret = fi_inject_writedata(rxm_conn_msg_ep(rxm_conn, NULL),
				  buf, len, data, dest_addr, addr, key);
	if (ret == -FI_EAGAIN)
		rxm_ep_do_progress(&rxm_ep->util_ep);
	else if (ret)
		FI_WARN(&rxm_prov, FI_LOG_EP_DATA, "fi_inject_writedata"
			" for MSG provider failed: %zd\n", ret);
unlock:
	ofi_genlock_unlock(&rxm_ep->util_ep.lock);
	return ret;
}

struct fi_ops_rma rxm_rma_ops = {
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


static ssize_t
rxm_read_thru(struct fid_ep *ep_fid, void *buf, size_t len,
	      void *desc, fi_addr_t src_addr, uint64_t addr,
	      uint64_t key, void *context)
{
	struct rxm_ep *ep;
	struct rxm_conn *conn;
	ssize_t ret;

	ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&ep->util_ep.lock);

	ret = rxm_get_conn(ep, src_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_read(rxm_conn_msg_ep(conn, NULL), buf, len, desc, src_addr, addr,
		      key, context);
unlock:
	ofi_genlock_unlock(&ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_readv_thru(struct fid_ep *ep_fid, const struct iovec *iov,
	       void **desc, size_t count, fi_addr_t src_addr,
	       uint64_t addr, uint64_t key, void *context)
{
	struct rxm_ep *ep;
	struct rxm_conn *conn;
	ssize_t ret;

	ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&ep->util_ep.lock);

	ret = rxm_get_conn(ep, src_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_readv(rxm_conn_msg_ep(conn, NULL),
		       iov, desc, count, src_addr, addr, key, context);
unlock:
	ofi_genlock_unlock(&ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_readmsg_thru(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
		 uint64_t flags)
{
	struct rxm_ep *ep;
	struct rxm_conn *conn;
	ssize_t ret;

	ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&ep->util_ep.lock);

	ret = rxm_get_conn(ep, msg->addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_readmsg(rxm_conn_msg_ep(conn, NULL), msg, flags);
unlock:
	ofi_genlock_unlock(&ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_write_thru(struct fid_ep *ep_fid, const void *buf,
	       size_t len, void *desc, fi_addr_t dest_addr,
	       uint64_t addr, uint64_t key, void *context)
{
	struct rxm_ep *ep;
	struct rxm_conn *conn;
	ssize_t ret;

	ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&ep->util_ep.lock);

	ret = rxm_get_conn(ep, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_write(rxm_conn_msg_ep(conn, NULL), buf, len, desc, dest_addr, 
		       addr, key, context);
unlock:
	ofi_genlock_unlock(&ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_writev_thru(struct fid_ep *ep_fid, const struct iovec *iov,
		void **desc, size_t count, fi_addr_t dest_addr,
		uint64_t addr, uint64_t key, void *context)
{
	struct rxm_ep *ep;
	struct rxm_conn *conn;
	ssize_t ret;

	ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&ep->util_ep.lock);

	ret = rxm_get_conn(ep, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_writev(rxm_conn_msg_ep(conn, NULL),
			iov, desc, count, dest_addr, addr, key, context);
unlock:
	ofi_genlock_unlock(&ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_writemsg_thru(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
		  uint64_t flags)
{
	struct rxm_ep *ep;
	struct rxm_conn *conn;
	ssize_t ret;

	ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&ep->util_ep.lock);

	ret = rxm_get_conn(ep, msg->addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_writemsg(rxm_conn_msg_ep(conn, NULL), msg, flags);
unlock:
	ofi_genlock_unlock(&ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_inject_write_thru(struct fid_ep *ep_fid, const void *buf,
		      size_t len, fi_addr_t dest_addr,
		      uint64_t addr, uint64_t key)
{
	struct rxm_ep *ep;
	struct rxm_conn *conn;
	ssize_t ret;

	ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&ep->util_ep.lock);

	ret = rxm_get_conn(ep, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_inject_write(rxm_conn_msg_ep(conn, NULL),
			      buf, len, dest_addr, addr, key);
unlock:
	ofi_genlock_unlock(&ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_writedata_thru(struct fid_ep *ep_fid, const void *buf,
		   size_t len, void *desc, uint64_t data,
		   fi_addr_t dest_addr, uint64_t addr,
		   uint64_t key, void *context)
{
	struct rxm_ep *ep;
	struct rxm_conn *conn;
	ssize_t ret;

	ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&ep->util_ep.lock);

	ret = rxm_get_conn(ep, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_writedata(rxm_conn_msg_ep(conn, NULL),
			   buf, len, desc, data, dest_addr, addr, key, context);
unlock:
	ofi_genlock_unlock(&ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_inject_writedata_thru(struct fid_ep *ep_fid, const void *buf,
			  size_t len, uint64_t data, fi_addr_t dest_addr,
			  uint64_t addr, uint64_t key)
{
	struct rxm_ep *ep;
	struct rxm_conn *conn;
	ssize_t ret;

	ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&ep->util_ep.lock);

	ret = rxm_get_conn(ep, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_inject_writedata(rxm_conn_msg_ep(conn, NULL),
				  buf, len, data, dest_addr, addr, key);
unlock:
	ofi_genlock_unlock(&ep->util_ep.lock);
	return ret;
}

struct fi_ops_rma rxm_rma_thru_ops = {
	.size = sizeof (struct fi_ops_rma),
	.read = rxm_read_thru,
	.readv = rxm_readv_thru,
	.readmsg = rxm_readmsg_thru,
	.write = rxm_write_thru,
	.writev = rxm_writev_thru,
	.writemsg = rxm_writemsg_thru,
	.inject = rxm_inject_write_thru,
	.writedata = rxm_writedata_thru,
	.injectdata = rxm_inject_writedata_thru,
};
