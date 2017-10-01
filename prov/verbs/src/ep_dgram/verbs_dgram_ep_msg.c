/*
 * Copyright (c) 2017 Intel Corporation, Inc.  All rights reserved.
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

#include "verbs_dgram.h"

static inline
void fi_ibv_dgram_recv_setup(struct fi_ibv_dgram_wr_entry *wr_entry,
			     struct ibv_recv_wr *wr)
{
	struct fi_ibv_dgram_ep *ep = wr_entry->hdr.ep;

	if (fi_ibv_dgram_is_completion(ep->ep_flags,
				       wr_entry->hdr.flags)) {
		wr_entry->hdr.suc_cb = fi_ibv_dgram_rx_cq_comp;
		wr_entry->hdr.err_cb = fi_ibv_dgram_rx_cq_report_error;
	} else {
		wr_entry->hdr.suc_cb = fi_ibv_dgram_rx_cq_no_action;
		wr_entry->hdr.err_cb = fi_ibv_dgram_rx_cq_no_action;

		fi_ibv_dgram_wr_entry_release(
			&ep->grh_pool,
			(struct fi_ibv_dgram_wr_entry_hdr *)wr_entry
		);
	}
}

static inline ssize_t
fi_ibv_dgram_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		     uint64_t flags)
{
	struct fi_ibv_dgram_ep *ep;
	struct ibv_recv_wr wr = { 0 };
	struct ibv_sge *sge;
	struct fi_ibv_dgram_wr_entry *wr_entry;
	ssize_t i;

	ep = container_of(ep_fid, struct fi_ibv_dgram_ep,
			  util_ep.ep_fid.fid);
	assert(ep && ep->util_ep.rx_cq);

	wr_entry = (struct fi_ibv_dgram_wr_entry *)
		fi_ibv_dgram_wr_entry_get(&ep->grh_pool);
	if (OFI_UNLIKELY(!wr_entry))
		return -FI_ENOMEM;

	wr_entry->hdr.ep = ep;
	wr_entry->hdr.context = msg->context;

	wr.wr_id = (uintptr_t)wr_entry;
	wr.next = NULL;

	sge = alloca(sizeof(*sge) * msg->iov_count + 1);
	sge[0].addr = (uintptr_t)(wr_entry->grh_buf);
	sge[0].length = VERBS_DGRAM_GRH_LENGTH;
	sge[0].lkey = (uint32_t)(uintptr_t)(wr_entry->hdr.desc);
	for (i = 0; i < msg->iov_count; i++) {
		sge[i + 1].addr = (uintptr_t)(msg->msg_iov[i].iov_base);
		sge[i + 1].length = (uint32_t)(msg->msg_iov[i].iov_len);
		sge[i + 1].lkey = (uint32_t)(uintptr_t)(msg->desc[i]);
	}
	wr.sg_list = sge;
	wr.num_sge = msg->iov_count + 1;

	fi_ibv_dgram_recv_setup(wr_entry, &wr);

	return FI_IBV_INVOKE_POST(
		recv, recv, ep->ibv_qp, &wr,
		(fi_ibv_dgram_wr_entry_release(
				&ep->grh_pool,
				(struct fi_ibv_dgram_wr_entry_hdr *)
				wr_entry)));
}

static inline ssize_t
fi_ibv_dgram_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		   size_t count, fi_addr_t src_addr, void *context)
{
	struct fi_msg msg = {
		.msg_iov	= iov,
		.desc		= desc,
		.iov_count	= count,
		.addr		= src_addr,
		.context	= context,
	};

	return fi_ibv_dgram_recvmsg(ep_fid, &msg, 0);
}

static inline ssize_t
fi_ibv_dgram_recv(struct fid_ep *ep_fid, void *buf, size_t len,
		  void *desc, fi_addr_t src_addr, void *context)
{
	struct iovec iov = {
		.iov_base	= buf,
		.iov_len	= len,
	};

	return fi_ibv_dgram_recvv(ep_fid, &iov, &desc,
				  1, src_addr, context);
}

static inline
void fi_ibv_dgram_send_setup(struct fi_ibv_dgram_wr_entry *wr_entry,
			     struct ibv_send_wr *wr,
			     size_t total_len)
{
	struct fi_ibv_dgram_ep *ep = wr_entry->hdr.ep;
	int32_t unsignaled_cnt = ofi_atomic_inc32(&ep->unsignaled_send_cnt) + 1;

	if (fi_ibv_dgram_is_completion(ep->ep_flags,
				       wr_entry->hdr.flags)) {
		wr->send_flags |= IBV_SEND_SIGNALED;
		wr_entry->hdr.suc_cb = fi_ibv_dgram_tx_cq_comp;
		wr_entry->hdr.err_cb = fi_ibv_dgram_tx_cq_report_error;
		wr_entry->hdr.comp_unsignaled_cnt = unsignaled_cnt;
	} else if (unsignaled_cnt == ep->max_unsignaled_send_cnt) {
		wr->send_flags |= IBV_SEND_SIGNALED;
		wr_entry->hdr.suc_cb = fi_ibv_dgram_tx_cq_no_action;
		wr_entry->hdr.err_cb = fi_ibv_dgram_tx_cq_no_action;
		wr_entry->hdr.comp_unsignaled_cnt = unsignaled_cnt;
	} else {
		/* No need other actions */
		fi_ibv_dgram_wr_entry_release(
			&ep->grh_pool,
			(struct fi_ibv_dgram_wr_entry_hdr *)wr_entry
		);
	}

	if ((wr_entry->hdr.flags & FI_INJECT) &&
	    (total_len <= ep->info->tx_attr->inject_size))
		wr->send_flags |= IBV_SEND_INLINE;
}

static inline ssize_t
fi_ibv_dgram_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		     uint64_t flags)
{
	struct fi_ibv_dgram_ep *ep;
	struct ibv_send_wr wr = { 0 };
	struct ibv_sge *sge;
	struct fi_ibv_dgram_wr_entry *wr_entry = NULL;
	struct fi_ibv_dgram_av_entry *av_entry;
	struct fi_ibv_dgram_av *av;
	size_t total_len = 0, i;

	assert(!(flags & FI_FENCE));

	ep = container_of(ep_fid, struct fi_ibv_dgram_ep,
			  util_ep.ep_fid.fid);
	assert(ep && ep->util_ep.tx_cq);

	wr_entry = (struct fi_ibv_dgram_wr_entry *)
		fi_ibv_dgram_wr_entry_get(&ep->grh_pool);
	if (OFI_UNLIKELY(!wr_entry))
		return -FI_ENOMEM;

	wr_entry->hdr.ep = ep;
	wr_entry->hdr.context = msg->context;
	wr_entry->hdr.flags = flags;

	sge = alloca(sizeof(*sge) * msg->iov_count + 1);
	sge[0].addr = (uintptr_t)(wr_entry->grh_buf);
	sge[0].length = 0;
	sge[0].lkey = (uint32_t)(uintptr_t)(wr_entry->hdr.desc);
	if (msg->desc) {
		for (i = 0; i < msg->iov_count; i++) {
			sge[i + 1].addr = (uintptr_t)(msg->msg_iov[i].iov_base);
			sge[i + 1].length = (uint32_t)(msg->msg_iov[i].iov_len);
			sge[i + 1].lkey = (uint32_t)(uintptr_t)(msg->desc[i]);
			total_len += msg->msg_iov[i].iov_len;
		}
	} else {
		for (i = 0; i < msg->iov_count; i++) {
			sge[i + 1].addr = (uintptr_t)(msg->msg_iov[i].iov_base);
			sge[i + 1].length = (uint32_t)(msg->msg_iov[i].iov_len);
			sge[i + 1].lkey = 0;
			total_len += msg->msg_iov[i].iov_len;
		}
	}

	wr.wr_id = (uintptr_t)wr_entry;
	wr.next = NULL;
	wr.sg_list = sge;
	wr.num_sge = msg->iov_count + 1;

	if (flags & FI_REMOTE_CQ_DATA) {
		wr.opcode = IBV_WR_SEND_WITH_IMM;
		wr.imm_data = htonl((uint32_t)msg->data);
	} else {
		wr.opcode = IBV_WR_SEND;
	}

	av = container_of(&ep->util_ep.av->av_fid, struct fi_ibv_dgram_av,
			  util_av.av_fid);
	av_entry = fi_ibv_dgram_av_lookup_av_entry(av, (int)msg->addr);
	if (OFI_UNLIKELY(!av_entry)) {
		fi_ibv_dgram_wr_entry_release(
			&ep->grh_pool,
			(struct fi_ibv_dgram_wr_entry_hdr *)wr_entry);
		return -FI_ENOENT;
	}

	wr.wr.ud.ah = av_entry->ah;
	wr.wr.ud.remote_qpn = av_entry->addr->qpn;
	wr.wr.ud.remote_qkey = 0x11111111;

	fi_ibv_dgram_send_setup(wr_entry, &wr, total_len);

	return FI_IBV_INVOKE_POST(
		send, send, ep->ibv_qp, &wr,
			((wr.send_flags & IBV_SEND_SIGNALED)	?
				fi_ibv_dgram_wr_entry_release(
					&ep->grh_pool,
					(struct fi_ibv_dgram_wr_entry_hdr *)
						wr_entry)	:
				NULL));
}

static inline ssize_t
fi_ibv_dgram_sendv(struct fid_ep *ep_fid, const struct iovec *iov,
		   void **desc, size_t count, fi_addr_t dest_addr,
		   void *context)
{
	struct fi_msg msg = {
		.msg_iov	= iov,
		.desc		= desc,
		.iov_count	= count,
		.addr		= dest_addr,
		.context	= context,
	};

	return fi_ibv_dgram_sendmsg(ep_fid, &msg, 0);
}

static inline ssize_t
fi_ibv_dgram_senddata(struct fid_ep *ep_fid, const void *buf,
		      size_t len, void *desc, uint64_t data,
		      fi_addr_t dest_addr, void *context)
{
	struct iovec iov = {
		.iov_base	= (void *)buf,
		.iov_len	= len,
	};

	struct fi_msg msg = {
		.msg_iov	= &iov,
		.desc		= &desc,
		.iov_count	= 1,
		.addr		= dest_addr,
		.context	= context,
		.data		= data,
	};

	return fi_ibv_dgram_sendmsg(ep_fid, &msg, FI_REMOTE_CQ_DATA);
}

static inline ssize_t
fi_ibv_dgram_send(struct fid_ep *ep_fid, const void *buf, size_t len,
		  void *desc, fi_addr_t dest_addr, void *context)
{
	struct iovec iov = {
		.iov_base	= (void *)buf,
		.iov_len	= len,
	};

	return fi_ibv_dgram_sendv(ep_fid, &iov, &desc,
				  1, dest_addr, context);
}

static inline ssize_t
fi_ibv_dgram_injectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
			uint64_t data, fi_addr_t dest_addr)
{
    struct iovec iov = {
		.iov_base	= (void *)buf,
		.iov_len	= len,
	};

	struct fi_msg msg = {
		.msg_iov	= &iov,
		.iov_count	= 1,
		.addr		= dest_addr,
		.data		= data,
	};

	return fi_ibv_dgram_sendmsg(ep_fid, &msg, FI_INJECT |
						  FI_REMOTE_CQ_DATA);
}

static inline ssize_t
fi_ibv_dgram_inject(struct fid_ep *ep_fid, const void *buf,
		    size_t len, fi_addr_t dest_addr)
{
	return fi_ibv_dgram_injectdata(ep_fid, buf, len, 0, dest_addr);
}

struct fi_ops_msg fi_ibv_dgram_msg_ops = {
	.size		= sizeof(fi_ibv_dgram_msg_ops),
	.recv		= fi_ibv_dgram_recv,
	.recvv		= fi_ibv_dgram_recvv,
	.recvmsg	= fi_ibv_dgram_recvmsg,
	.send		= fi_ibv_dgram_send,
	.sendv		= fi_ibv_dgram_sendv,
	.sendmsg	= fi_ibv_dgram_sendmsg,
	.inject		= fi_ibv_dgram_inject,
	.senddata	= fi_ibv_dgram_senddata,
	.injectdata	= fi_ibv_dgram_injectdata,
};
