/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2017-2019 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include "config.h"

#include "efa_verbs/efa_ib.h"
#include "efa_verbs/efa_io_defs.h"

#include "ofi.h"
#include "ofi_enosys.h"
#include "ofi_iov.h"

#include "efa.h"

#define EFA_SETUP_IOV(iov, buf, len)           \
	do {                                   \
		iov.iov_base = (void *)buf;    \
		iov.iov_len = (size_t)len;     \
	} while (0)

#define EFA_SETUP_MSG(msg, iov, _desc, count, _addr, _context, _data)    \
	do {                                                             \
		msg.msg_iov = (const struct iovec *)iov;                 \
		msg.desc = (void **)_desc;                               \
		msg.iov_count = (size_t)count;                           \
		msg.addr = (fi_addr_t)_addr;                             \
		msg.context = (void *)_context;                          \
		msg.data = (uint32_t)_data;                              \
	} while (0)

#ifndef EFA_MSG_DUMP
static inline void dump_msg(const struct fi_msg *msg, const char *context) {}
#else
#define DUMP_IOV(i, iov, desc) \
	EFA_DBG(FI_LOG_EP_DATA, \
		"\t{ iov[%d] = { base = %p, buff = \"%s\", len = %zu }, desc = %p },\n", \
		i, iov.iov_base, (char *)iov.iov_base, iov.iov_len, (desc ? desc[i] : NULL))

static inline void dump_msg(const struct fi_msg *msg, const char *context)
{
	int i;

	EFA_DBG(FI_LOG_EP_DATA, "%s: { data = %u, addr = %" PRIu64 ", iov_count = %zu, [\n",
		context, (unsigned)msg->data, msg->addr, msg->iov_count);
	for (i = 0; i < msg->iov_count; ++i)
		DUMP_IOV(i, msg->msg_iov[i], msg->desc);
	EFA_DBG(FI_LOG_EP_DATA, " ] }\n");
}
#endif /* EFA_MSG_DUMP */

static ssize_t efa_post_recv_validate(struct efa_ep *ep, const struct fi_msg *msg)
{
	struct efa_qp *qp = ep->qp;
	//size_t len;

	if (OFI_UNLIKELY(!ep->rcq)) {
		EFA_WARN(FI_LOG_EP_DATA, "No receive cq was bound to ep.\n");
		return -FI_EINVAL;
	}

	if (OFI_UNLIKELY(msg->iov_count > qp->rq.wq.max_sge)) {
		EFA_WARN(FI_LOG_EP_DATA, "requested sge[%zu] is greater than max supported[%d]!\n",
			 msg->iov_count, qp->rq.wq.max_sge);
		return -FI_EINVAL;
	}

	if (OFI_UNLIKELY(msg->msg_iov[0].iov_len <
			 ep->info->ep_attr->msg_prefix_size)) {
		EFA_WARN(FI_LOG_EP_DATA, "prefix not present on first iov, iov_len[%zu]\n",
			 msg->msg_iov[0].iov_len);
		return -EINVAL;
	}

/* XXX: tests pass the prefix twice for some reason and break this check (will be removed when we move to libibverbs)
	len = ofi_total_iov_len(msg->msg_iov, msg->iov_count);

	if (OFI_UNLIKELY(len > ep->info->ep_attr->max_msg_size +
			       ep->msg_prefix_size)) {
		EFA_WARN(FI_LOG_EP_DATA, "requested size[%zu] is greater than max[%zu]!\n",
			 len, ep->info->ep_attr->max_msg_size + ep->msg_prefix_size);
		return -FI_EINVAL;
	}
*/

	if (OFI_UNLIKELY((qp->rq.wq.wqe_posted - qp->rq.wq.wqe_completed) == qp->rq.wq.wqe_cnt)) {
		EFA_DBG(FI_LOG_EP_DATA, "rq is full! posted[%u] completed[%u] wqe_cnt[%u]\n",
			qp->rq.wq.wqe_posted, qp->rq.wq.wqe_completed, qp->rq.wq.wqe_cnt);
		return -FI_EAGAIN;
	}

	return 0;
}

static ssize_t efa_post_recv(struct efa_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	struct efa_qp *qp = ep->qp;
	struct efa_io_rx_desc rx_buf = {};
	uint32_t wqe_index, rq_desc_offset;
	size_t i;
	ssize_t err;
	uintptr_t addr;

	dump_msg(msg, "recv");

	err = efa_post_recv_validate(ep, msg);
	if (OFI_UNLIKELY(err))
		return err;

	/* Save wrid */
	/* Get the next wrid to be used from the index pool. */
	wqe_index = qp->rq.wq.wrid_idx_pool[qp->rq.wq.wrid_idx_pool_next];
	qp->rq.wq.wrid[wqe_index] = (uintptr_t)msg->context;
	rx_buf.req_id = wqe_index;
	qp->rq.wq.wqe_posted++;

	/* Will never overlap, as efa_post_recv_validate() succeeded. */
	qp->rq.wq.wrid_idx_pool_next++;
	assert(qp->rq.wq.wrid_idx_pool_next <= qp->rq.wq.wqe_cnt);

	/* Default init of the rx buffer */
	set_efa_io_rx_desc_first(&rx_buf, 1);
	set_efa_io_rx_desc_last(&rx_buf, 0);

	for (i = 0; i < msg->iov_count; i++) {
		/* Set last indication if need) */
		if (i == (msg->iov_count - 1))
			set_efa_io_rx_desc_last(&rx_buf, 1);

		addr = (uintptr_t)msg->msg_iov[i].iov_base;

		/* Set RX buffer desc from SGE */
		rx_buf.length = msg->msg_iov[i].iov_len;
		set_efa_io_rx_desc_lkey(&rx_buf, (uint32_t)(uintptr_t)msg->desc[i]);
		rx_buf.buf_addr_lo = addr;
		rx_buf.buf_addr_hi = addr >> 32;

		/* Copy descriptor to RX ring  */
		rq_desc_offset = (qp->rq.wq.desc_idx & qp->rq.wq.desc_mask) * sizeof(rx_buf);
		memcpy(qp->rq.buf + rq_desc_offset, &rx_buf, sizeof(rx_buf));

		/* Wrap rx descriptor index */
		qp->rq.wq.desc_idx++;
		if ((qp->rq.wq.desc_idx & qp->rq.wq.desc_mask) == 0)
			qp->rq.wq.phase++;

		/* reset descriptor for next iov */
		memset(&rx_buf, 0, sizeof(rx_buf));
	}

	if (flags & FI_MORE)
		return 0;

	wmb();
	*qp->rq.db = qp->rq.wq.desc_idx;

	return 0;
}

static ssize_t efa_ep_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg, uint64_t flags)
{
	struct efa_ep *ep;

	ep = container_of(ep_fid, struct efa_ep, ep_fid);

	return efa_post_recv(ep, msg, flags);
}

static ssize_t efa_ep_recv(struct fid_ep *ep_fid, void *buf, size_t len,
			   void *desc, fi_addr_t src_addr, void *context)
{
	struct efa_ep *ep;
	struct iovec iov;
	struct fi_msg msg;

	ep = container_of(ep_fid, struct efa_ep, ep_fid);

	EFA_SETUP_IOV(iov, buf, len);
	EFA_SETUP_MSG(msg, &iov, &desc, 1, src_addr, context, 0);

	return efa_post_recv(ep, &msg, 0);
}

static ssize_t efa_ep_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
			    size_t count, fi_addr_t src_addr, void *context)
{
	struct efa_ep *ep;
	struct fi_msg msg;

	ep = container_of(ep_fid, struct efa_ep, ep_fid);

	EFA_SETUP_MSG(msg, iov, desc, count, src_addr, context, 0);

	return efa_post_recv(ep, &msg, 0);
}

static ssize_t efa_post_send_validate(struct efa_ep *ep, const struct fi_msg *msg,
				      struct efa_conn *conn, uint64_t flags, size_t *len)
{
	struct efa_qp *qp = ep->qp;

	if (OFI_UNLIKELY(!ep->scq)) {
		EFA_WARN(FI_LOG_EP_DATA, "No send cq was bound to ep.\n");
		return -FI_EINVAL;
	}

	if (OFI_UNLIKELY(msg->iov_count > qp->sq.wq.max_sge)) {
		EFA_WARN(FI_LOG_EP_DATA, "requested sge[%zu] is greater than max supported[%d]!\n",
			 msg->iov_count, qp->sq.wq.max_sge);
		return -FI_EINVAL;
	}

	if (OFI_UNLIKELY(!conn->ah)) {
		EFA_WARN(FI_LOG_EP_DATA, "Invalid fi_addr\n");
		return -FI_EINVAL;
	}

	if (OFI_UNLIKELY(msg->msg_iov[0].iov_len <
			 ep->info->ep_attr->msg_prefix_size)) {
		EFA_WARN(FI_LOG_EP_DATA, "prefix not present on first iov, iov_len[%zu]\n",
			 msg->msg_iov[0].iov_len);
		return -EINVAL;
	}

	*len = ofi_total_iov_len(msg->msg_iov, msg->iov_count) - ep->info->ep_attr->msg_prefix_size;
	if (OFI_UNLIKELY(*len > ep->info->ep_attr->max_msg_size)) {
		EFA_WARN(FI_LOG_EP_DATA, "requested size[%zu] is greater than max[%zu]!\n",
			 *len, ep->info->ep_attr->max_msg_size);
		return -FI_EINVAL;
	}

	if (OFI_UNLIKELY((qp->sq.wq.wqe_posted - qp->sq.wq.wqe_completed) == qp->sq.wq.wqe_cnt)) {
		EFA_DBG(FI_LOG_EP_DATA, "sq is full! posted[%u] completed[%u] wqe_cnt[%u]\n",
			qp->sq.wq.wqe_posted, qp->sq.wq.wqe_completed, qp->sq.wq.wqe_cnt);
		return -FI_EAGAIN;
	}

	return 0;
}

static void efa_post_send_inline_data(struct efa_ep *ep,
				      const struct fi_msg *msg,
				      struct efa_io_tx_wqe *tx_wqe,
				      int *desc_size)
{
	const struct iovec *iov = msg->msg_iov;
	uint32_t total_length = 0;
	uint32_t length;
	uintptr_t addr;
	size_t i;

	for (i = 0; i < msg->iov_count; i++) {
		length = iov[i].iov_len;
		addr = (uintptr_t)iov[i].iov_base;

		/* Whole prefix must be on the first sgl */
		if (!i) {
			/* Check if payload exists */
			if (length <= ep->info->ep_attr->msg_prefix_size)
				continue;

			addr += ep->info->ep_attr->msg_prefix_size;
			length -= ep->info->ep_attr->msg_prefix_size;
		}

		memcpy(tx_wqe->data.inline_data + total_length, (void *)addr, length);
		total_length += length;
	}
	*desc_size += total_length;

	set_efa_io_tx_meta_desc_inline_msg(&tx_wqe->common, 1);
	tx_wqe->common.length = total_length;
}

static void efa_post_send_immediate_data(const struct fi_msg *msg,
					 struct efa_io_tx_meta_desc *meta_desc)
{
	uint32_t imm_data;

	imm_data = htonl((uint32_t)msg->data);
	meta_desc->immediate_data = imm_data;
	set_efa_io_tx_meta_desc_has_imm(meta_desc, 1);
}

static void efa_post_send_sgl(struct efa_ep *ep, const struct fi_msg *msg,
			      struct efa_io_tx_wqe *tx_wqe, int *desc_size,
			      uint16_t *num_descs)
{
	struct efa_io_tx_buf_desc *tx_buf;
	size_t sgl_idx = 0;
	uint32_t length;
	uintptr_t addr;
	size_t i;

	for (i = 0; i < msg->iov_count; i++) {
		tx_buf = &tx_wqe->data.sgl[sgl_idx];
		addr = (uintptr_t)msg->msg_iov[i].iov_base;
		length = msg->msg_iov[i].iov_len;

		/* Whole prefix must be on the first sgl */
		if (!i) {
			/* Check if payload exists */
			if (length <= ep->info->ep_attr->msg_prefix_size)
				continue;

			addr += ep->info->ep_attr->msg_prefix_size;
			length -= ep->info->ep_attr->msg_prefix_size;
		}

		/* Set TX buffer desc from SGE */
		tx_buf->length = length;
		tx_buf->lkey = (msg->desc ? ((uint32_t)(uintptr_t)msg->desc[i]) : 0);
		tx_buf->buf_addr_lo = addr & 0xFFFFFFFF;
		tx_buf->buf_addr_hi = addr >> 32;
		sgl_idx++;
	}

	*num_descs = sgl_idx;
	*desc_size += (sizeof(struct efa_io_tx_buf_desc) * sgl_idx);
}

static ssize_t efa_post_send(struct efa_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	struct efa_qp *qp = ep->qp;
	struct efa_io_tx_meta_desc *meta_desc;
	struct efa_io_tx_wqe tx_wqe = {};
	uint32_t sq_desc_offset, wrid_idx;
	int desc_size = sizeof(tx_wqe.common) + sizeof(tx_wqe.u);
	struct efa_conn *conn;
	size_t len;
	int ret;

	dump_msg(msg, "send");

	conn = ep->av->addr_to_conn(ep->av, msg->addr);

	ret = efa_post_send_validate(ep, msg, conn, flags, &len);
	if (OFI_UNLIKELY(ret))
		return ret;

	meta_desc = &tx_wqe.common;

	if (flags & FI_REMOTE_CQ_DATA)
		efa_post_send_immediate_data(msg, meta_desc);

	if (len <= qp->sq.max_inline_data)
		efa_post_send_inline_data(ep, msg, &tx_wqe, &desc_size);
	else
		efa_post_send_sgl(ep, msg, &tx_wqe, &desc_size, &meta_desc->length);

	/* Get the next wrid to be used from the index pool. */
	wrid_idx = qp->sq.wq.wrid_idx_pool[qp->sq.wq.wrid_idx_pool_next];
	qp->sq.wq.wrid[wrid_idx] = (uintptr_t)msg->context;
	meta_desc->req_id = wrid_idx;
	qp->sq.wq.wqe_posted++;

	/* Will never overlap, as efa_post_send_validate() succeeded. */
	qp->sq.wq.wrid_idx_pool_next++;
	assert(qp->sq.wq.wrid_idx_pool_next <= qp->sq.wq.wqe_cnt);

	/* Set rest of the descriptor fields. */
	set_efa_io_tx_meta_desc_meta_desc(meta_desc, 1);
	set_efa_io_tx_meta_desc_phase(meta_desc, qp->sq.wq.phase);
	set_efa_io_tx_meta_desc_first(meta_desc, 1);
	set_efa_io_tx_meta_desc_last(meta_desc, 1);
	meta_desc->dest_qp_num = conn->ep_addr.qpn;
	set_efa_io_tx_meta_desc_comp_req(meta_desc, 1);
	meta_desc->ah = conn->ah->efa_address_handle;

	/* Copy descriptor */
	sq_desc_offset = (qp->sq.wq.desc_idx & qp->sq.wq.desc_mask) * sizeof(tx_wqe);
	memcpy(qp->sq.desc + sq_desc_offset, &tx_wqe, desc_size);

	/* advance index and change phase */
	qp->sq.wq.desc_idx++;
	if ((qp->sq.wq.desc_idx & qp->sq.wq.desc_mask) == 0)
		qp->sq.wq.phase++;

	if (flags & FI_MORE)
		return 0;

	wmb();
	*qp->sq.db = qp->sq.wq.desc_idx;

	return 0;
}

static ssize_t efa_ep_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg, uint64_t flags)
{
	struct efa_ep *ep;

	ep = container_of(ep_fid, struct efa_ep, ep_fid);

	return efa_post_send(ep, msg, flags);
}

static ssize_t efa_ep_send(struct fid_ep *ep_fid, const void *buf, size_t len,
			   void *desc, fi_addr_t dest_addr, void *context)
{
	struct efa_ep *ep;
	struct fi_msg msg;
	struct iovec iov;
	uint64_t flags;

	ep = container_of(ep_fid, struct efa_ep, ep_fid);

	EFA_SETUP_IOV(iov, buf, len);
	EFA_SETUP_MSG(msg, &iov, &desc, 1, dest_addr, context, 0);
	flags = ep->info->tx_attr->op_flags;

	return efa_post_send(ep, &msg, flags);
}

static ssize_t efa_ep_senddata(struct fid_ep *ep_fid, const void *buf, size_t len,
			       void *desc, uint64_t data, fi_addr_t dest_addr, void *context)
{
	struct efa_ep *ep;
	struct fi_msg msg;
	struct iovec iov;
	uint64_t flags;

	ep = container_of(ep_fid, struct efa_ep, ep_fid);

	EFA_SETUP_IOV(iov, buf, len);
	EFA_SETUP_MSG(msg, &iov, &desc, 1, dest_addr, context, data);

	flags = ep->info->tx_attr->op_flags | FI_REMOTE_CQ_DATA;

	return efa_post_send(ep, &msg, flags);
}

static ssize_t efa_ep_sendv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
			    size_t count, fi_addr_t dest_addr, void *context)
{
	struct efa_ep *ep;
	struct fi_msg msg;
	uint64_t flags;

	ep = container_of(ep_fid, struct efa_ep, ep_fid);

	EFA_SETUP_MSG(msg, iov, desc, count, dest_addr, context, 0);

	flags = ep->info->tx_attr->op_flags;

	return efa_post_send(ep, &msg, flags);
}

struct fi_ops_msg efa_ep_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = efa_ep_recv,
	.recvv = efa_ep_recvv,
	.recvmsg = efa_ep_recvmsg,
	.send = efa_ep_send,
	.sendv = efa_ep_sendv,
	.sendmsg = efa_ep_sendmsg,
	.inject = fi_no_msg_inject,
	.senddata = efa_ep_senddata,
	.injectdata = fi_no_msg_injectdata,
};
