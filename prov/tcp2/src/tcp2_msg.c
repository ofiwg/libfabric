/*
 * Copyright (c) 2018-2022 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *	   Redistribution and use in source and binary forms, with or
 *	   without modification, are permitted provided that the following
 *	   conditions are met:
 *
 *		- Redistributions of source code must retain the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer.
 *
 *		- Redistributions in binary form must reproduce the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer in the documentation and/or other materials
 *		  provided with the distribution.
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
#include <rdma/fi_errno.h>
#include "rdma/fi_eq.h"
#include "ofi_iov.h"
#include <ofi_prov.h>
#include "tcp2.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <ofi_util.h>
#include <unistd.h>
#include <string.h>
#include <poll.h>
#include <arpa/inet.h>
#include <netdb.h>


static inline struct tcp2_xfer_entry *
tcp2_alloc_send(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *send_entry;

	send_entry = tcp2_alloc_tx(ep);
	if (send_entry)
		send_entry->hdr.base_hdr.op = ofi_op_msg;

	return send_entry;
}

static inline struct tcp2_xfer_entry *
tcp2_alloc_tsend(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *send_entry;

	send_entry = tcp2_alloc_tx(ep);
	if (send_entry) {
		assert(ep->srx);
		send_entry->hdr.base_hdr.op = ofi_op_tagged;
	}

	return send_entry;
}

static inline void
tcp2_init_tx_sizes(struct tcp2_xfer_entry *tx_entry, size_t hdr_len,
		   size_t data_len)
{
	tx_entry->hdr.base_hdr.size = hdr_len + data_len;
	tx_entry->hdr.base_hdr.hdr_size = (uint8_t) hdr_len;
}

static inline void
tcp2_init_tx_inject(struct tcp2_xfer_entry *tx_entry, size_t hdr_len,
		    const void *buf, size_t data_len)
{
	assert(data_len <= TCP2_MAX_INJECT);
	tcp2_init_tx_sizes(tx_entry, hdr_len, data_len);

	tx_entry->iov[0].iov_base = (void *) &tx_entry->hdr;
	memcpy((uint8_t *) &tx_entry->hdr + hdr_len, (uint8_t *) buf,
		data_len);
	tx_entry->iov[0].iov_len = hdr_len + data_len;
	tx_entry->iov_cnt = 1;
}

static inline void
tcp2_init_tx_buf(struct tcp2_xfer_entry *tx_entry, size_t hdr_len,
		 const void *buf, size_t data_len)
{
	if (data_len <= TCP2_MAX_INJECT) {
		tcp2_init_tx_inject(tx_entry, hdr_len, buf, data_len);
		return;
	}

	tcp2_init_tx_sizes(tx_entry, hdr_len, data_len);
	tx_entry->iov[0].iov_base = (void *) &tx_entry->hdr;
	tx_entry->iov[0].iov_len = hdr_len;
	tx_entry->iov[1].iov_base = (void *) buf;
	tx_entry->iov[1].iov_len = data_len;
	tx_entry->iov_cnt = 2;
}

static inline void
tcp2_init_tx_iov(struct tcp2_xfer_entry *tx_entry, size_t hdr_len,
		 const struct iovec *iov, size_t count)
{
	size_t data_len;

	assert(count <= TCP2_IOV_LIMIT);
	data_len = ofi_total_iov_len(iov, count);
	tcp2_init_tx_sizes(tx_entry, hdr_len, data_len);

	tx_entry->iov[0].iov_base = (void *) &tx_entry->hdr;
	if (data_len <= TCP2_MAX_INJECT) {
		ofi_copy_iov_buf(iov, count, 0, (uint8_t *) &tx_entry->hdr +
				 hdr_len, TCP2_MAX_INJECT, OFI_COPY_IOV_TO_BUF);
		tx_entry->iov[0].iov_len = hdr_len + data_len;
		tx_entry->iov_cnt = 1;
	} else {
		tx_entry->iov[0].iov_len = hdr_len;
		tx_entry->iov_cnt = count + 1;
		memcpy(&tx_entry->iov[1], &iov[0], count * sizeof(struct iovec));
	}
}

static inline bool
tcp2_queue_recv(struct tcp2_ep *ep, struct tcp2_xfer_entry *recv_entry)
{
	bool ret;

	ofi_genlock_lock(&tcp2_ep2_progress(ep)->lock);
	ret = ep->rx_avail;
	if (ret) {
		slist_insert_tail(&recv_entry->entry, &ep->rx_queue);
		ep->rx_avail--;
	}
	ofi_genlock_unlock(&tcp2_ep2_progress(ep)->lock);
	return ret;
}

static ssize_t
tcp2_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg, uint64_t flags)
{
	struct tcp2_xfer_entry *recv_entry;
	struct tcp2_ep *ep;

	ep = container_of(ep_fid, struct tcp2_ep, util_ep.ep_fid);

	assert(msg->iov_count <= TCP2_IOV_LIMIT);

	recv_entry = tcp2_alloc_rx(ep);
	if (!recv_entry)
		return -FI_EAGAIN;

	recv_entry->iov_cnt = msg->iov_count;
	memcpy(&recv_entry->iov[0], &msg->msg_iov[0],
	       msg->iov_count * sizeof(struct iovec));

	recv_entry->cq_flags = tcp2_rx_completion_flag(ep, flags) |
			       FI_MSG | FI_RECV;
	recv_entry->context = msg->context;

	if (!tcp2_queue_recv(ep, recv_entry)) {
		ofi_genlock_lock(&tcp2_ep2_progress(ep)->lock);
		tcp2_free_rx(recv_entry);
		ofi_genlock_unlock(&tcp2_ep2_progress(ep)->lock);
		return -FI_EAGAIN;
	}
	return FI_SUCCESS;
}

static ssize_t
tcp2_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	  fi_addr_t src_addr, void *context)
{
	struct tcp2_xfer_entry *recv_entry;
	struct tcp2_ep *ep;

	ep = container_of(ep_fid, struct tcp2_ep, util_ep.ep_fid);

	recv_entry = tcp2_alloc_rx(ep);
	if (!recv_entry)
		return -FI_EAGAIN;

	recv_entry->iov_cnt = 1;
	recv_entry->iov[0].iov_base = buf;
	recv_entry->iov[0].iov_len = len;

	recv_entry->cq_flags = tcp2_rx_completion_flag(ep, 0) |
			       FI_MSG | FI_RECV;
	recv_entry->context = context;

	if (!tcp2_queue_recv(ep, recv_entry)) {
		ofi_genlock_lock(&tcp2_ep2_progress(ep)->lock);
		tcp2_free_rx(recv_entry);
		ofi_genlock_unlock(&tcp2_ep2_progress(ep)->lock);
		return -FI_EAGAIN;
	}
	return FI_SUCCESS;
}

static ssize_t
tcp2_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
	   size_t count, fi_addr_t src_addr, void *context)
{
	struct tcp2_xfer_entry *recv_entry;
	struct tcp2_ep *ep;

	ep = container_of(ep_fid, struct tcp2_ep, util_ep.ep_fid);

	assert(count <= TCP2_IOV_LIMIT);

	recv_entry = tcp2_alloc_rx(ep);
	if (!recv_entry)
		return -FI_EAGAIN;

	recv_entry->iov_cnt = count;
	memcpy(recv_entry->iov, iov, count * sizeof(*iov));

	recv_entry->cq_flags = tcp2_rx_completion_flag(ep, 0) |
			       FI_MSG | FI_RECV;
	recv_entry->context = context;

	if (!tcp2_queue_recv(ep, recv_entry)) {
		ofi_genlock_lock(&tcp2_ep2_progress(ep)->lock);
		tcp2_free_rx(recv_entry);
		ofi_genlock_unlock(&tcp2_ep2_progress(ep)->lock);
		return -FI_EAGAIN;
	}
	return FI_SUCCESS;
}

static ssize_t
tcp2_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg, uint64_t flags)
{
	struct tcp2_ep *ep;
	struct tcp2_xfer_entry *tx_entry;
	size_t hdr_len;

	ep = container_of(ep_fid, struct tcp2_ep, util_ep.ep_fid);
	tx_entry = tcp2_alloc_send(ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	if (flags & FI_REMOTE_CQ_DATA) {
		tx_entry->hdr.base_hdr.flags = TCP2_REMOTE_CQ_DATA;
		tx_entry->hdr.cq_data_hdr.cq_data = msg->data;
		hdr_len = sizeof(tx_entry->hdr.cq_data_hdr);
	} else {
		hdr_len = sizeof(tx_entry->hdr.base_hdr);
	}

	tcp2_init_tx_iov(tx_entry, hdr_len, msg->msg_iov, msg->iov_count);
	tx_entry->cq_flags = tcp2_tx_completion_flag(ep, flags) |
			     FI_MSG | FI_SEND;
	tcp2_set_ack_flags(tx_entry, flags);
	tx_entry->context = msg->context;

	tcp2_queue_send(ep, tx_entry);
	return FI_SUCCESS;
}

static ssize_t
tcp2_send(struct fid_ep *ep_fid, const void *buf, size_t len,
	  void *desc, fi_addr_t dest_addr, void *context)
{
	struct tcp2_ep *ep;
	struct tcp2_xfer_entry *tx_entry;

	ep = container_of(ep_fid, struct tcp2_ep, util_ep.ep_fid);

	tx_entry = tcp2_alloc_send(ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	tcp2_init_tx_buf(tx_entry, sizeof(tx_entry->hdr.base_hdr), buf, len);
	tx_entry->context = context;
	tx_entry->cq_flags = tcp2_tx_completion_flag(ep, 0) |
			     FI_MSG | FI_SEND;
	tcp2_set_ack_flags(tx_entry, ep->util_ep.tx_op_flags);

	tcp2_queue_send(ep, tx_entry);
	return FI_SUCCESS;
}

static ssize_t
tcp2_sendv(struct fid_ep *ep_fid, const struct iovec *iov,
	   void **desc, size_t count, fi_addr_t dest_addr, void *context)
{
	struct tcp2_ep *ep;
	struct tcp2_xfer_entry *tx_entry;

	ep = container_of(ep_fid, struct tcp2_ep, util_ep.ep_fid);

	tx_entry = tcp2_alloc_send(ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	tcp2_init_tx_iov(tx_entry, sizeof(tx_entry->hdr.base_hdr), iov, count);
	tx_entry->context = context;
	tx_entry->cq_flags = tcp2_tx_completion_flag(ep, 0) |
			     FI_MSG | FI_SEND;
	tcp2_set_ack_flags(tx_entry, ep->util_ep.tx_op_flags);

	tcp2_queue_send(ep, tx_entry);
	return FI_SUCCESS;
}


static ssize_t
tcp2_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
	    fi_addr_t dest_addr)
{
	struct tcp2_ep *ep;
	struct tcp2_xfer_entry *tx_entry;

	ep = container_of(ep_fid, struct tcp2_ep, util_ep.ep_fid);

	tx_entry = tcp2_alloc_send(ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	tcp2_init_tx_inject(tx_entry, sizeof(tx_entry->hdr.base_hdr), buf, len);
	tx_entry->ctrl_flags = TCP2_INJECT_OP;

	tcp2_queue_send(ep, tx_entry);
	return FI_SUCCESS;
}

static ssize_t
tcp2_senddata(struct fid_ep *ep_fid, const void *buf, size_t len,
	      void *desc, uint64_t data, fi_addr_t dest_addr, void *context)
{
	struct tcp2_ep *ep;
	struct tcp2_xfer_entry *tx_entry;

	ep = container_of(ep_fid, struct tcp2_ep, util_ep.ep_fid);

	tx_entry = tcp2_alloc_send(ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	tx_entry->hdr.cq_data_hdr.base_hdr.size =
		len + sizeof(tx_entry->hdr.cq_data_hdr);
	tx_entry->hdr.cq_data_hdr.base_hdr.flags = TCP2_REMOTE_CQ_DATA;
	tx_entry->hdr.cq_data_hdr.cq_data = data;

	tcp2_init_tx_buf(tx_entry, sizeof(tx_entry->hdr.cq_data_hdr),
			 buf, len);
	tx_entry->context = context;
	tx_entry->cq_flags = tcp2_tx_completion_flag(ep, 0) |
			     FI_MSG | FI_SEND;
	tcp2_set_ack_flags(tx_entry, ep->util_ep.tx_op_flags);

	tcp2_queue_send(ep, tx_entry);
	return FI_SUCCESS;
}

static ssize_t
tcp2_injectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
		uint64_t data, fi_addr_t dest_addr)
{
	struct tcp2_ep *ep;
	struct tcp2_xfer_entry *tx_entry;

	ep = container_of(ep_fid, struct tcp2_ep, util_ep.ep_fid);

	tx_entry = tcp2_alloc_send(ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	tx_entry->hdr.cq_data_hdr.base_hdr.flags = TCP2_REMOTE_CQ_DATA;
	tx_entry->hdr.cq_data_hdr.cq_data = data;

	tcp2_init_tx_inject(tx_entry, sizeof(tx_entry->hdr.cq_data_hdr),
			    buf, len);
	tx_entry->ctrl_flags = TCP2_INJECT_OP;

	tcp2_queue_send(ep, tx_entry);
	return FI_SUCCESS;
}

struct fi_ops_msg tcp2_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = tcp2_recv,
	.recvv = tcp2_recvv,
	.recvmsg = tcp2_recvmsg,
	.send = tcp2_send,
	.sendv = tcp2_sendv,
	.sendmsg = tcp2_sendmsg,
	.inject = tcp2_inject,
	.senddata = tcp2_senddata,
	.injectdata = tcp2_injectdata,
};

static ssize_t
tcp2_tsendmsg(struct fid_ep *fid_ep, const struct fi_msg_tagged *msg,
	      uint64_t flags)
{
	struct tcp2_ep *ep;
	struct tcp2_xfer_entry *tx_entry;
	size_t hdr_len;

	ep = container_of(fid_ep, struct tcp2_ep, util_ep.ep_fid);
	tx_entry = tcp2_alloc_tsend(ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	if (flags & FI_REMOTE_CQ_DATA) {
		tx_entry->hdr.base_hdr.flags |= TCP2_REMOTE_CQ_DATA;
		tx_entry->hdr.tag_data_hdr.cq_data_hdr.cq_data = msg->data;
		tx_entry->hdr.tag_data_hdr.tag = msg->tag;
		hdr_len = sizeof(tx_entry->hdr.tag_data_hdr);
	} else {
		tx_entry->hdr.tag_hdr.tag = msg->tag;
		hdr_len = sizeof(tx_entry->hdr.tag_hdr);
	}

	tcp2_init_tx_iov(tx_entry, hdr_len, msg->msg_iov, msg->iov_count);
	tx_entry->cq_flags = tcp2_tx_completion_flag(ep, flags) |
			     FI_TAGGED | FI_SEND;
	tcp2_set_ack_flags(tx_entry, flags);
	tx_entry->context = msg->context;

	tcp2_queue_send(ep, tx_entry);
	return FI_SUCCESS;
}

static ssize_t
tcp2_tsend(struct fid_ep *fid_ep, const void *buf, size_t len,
	   void *desc, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	struct tcp2_ep *ep;
	struct tcp2_xfer_entry *tx_entry;

	ep = container_of(fid_ep, struct tcp2_ep, util_ep.ep_fid);
	tx_entry = tcp2_alloc_tsend(ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	tx_entry->hdr.tag_hdr.tag = tag;

	tcp2_init_tx_buf(tx_entry, sizeof(tx_entry->hdr.tag_hdr), buf, len);
	tx_entry->context = context;
	tx_entry->cq_flags = tcp2_tx_completion_flag(ep, 0) |
			     FI_TAGGED | FI_SEND;
	tcp2_set_ack_flags(tx_entry, ep->util_ep.tx_op_flags);

	tcp2_queue_send(ep, tx_entry);
	return FI_SUCCESS;
}

static ssize_t
tcp2_tsendv(struct fid_ep *fid_ep, const struct iovec *iov, void **desc,
	    size_t count, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	struct tcp2_ep *ep;
	struct tcp2_xfer_entry *tx_entry;

	ep = container_of(fid_ep, struct tcp2_ep, util_ep.ep_fid);
	tx_entry = tcp2_alloc_tsend(ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	tx_entry->hdr.tag_hdr.tag = tag;

	tcp2_init_tx_iov(tx_entry, sizeof(tx_entry->hdr.tag_hdr), iov, count);
	tx_entry->context = context;
	tx_entry->cq_flags = tcp2_tx_completion_flag(ep, 0) |
			     FI_TAGGED | FI_SEND;
	tcp2_set_ack_flags(tx_entry, ep->util_ep.tx_op_flags);

	tcp2_queue_send(ep, tx_entry);
	return FI_SUCCESS;
}


static ssize_t
tcp2_tinject(struct fid_ep *fid_ep, const void *buf, size_t len,
	     fi_addr_t dest_addr, uint64_t tag)
{
	struct tcp2_ep *ep;
	struct tcp2_xfer_entry *tx_entry;

	ep = container_of(fid_ep, struct tcp2_ep, util_ep.ep_fid);
	tx_entry = tcp2_alloc_tsend(ep);
	if (!tx_entry)
		return -FI_EAGAIN;
	tx_entry->hdr.tag_hdr.tag = tag;

	tcp2_init_tx_inject(tx_entry, sizeof(tx_entry->hdr.tag_hdr), buf, len);
	tx_entry->ctrl_flags = TCP2_INJECT_OP;

	tcp2_queue_send(ep, tx_entry);
	return FI_SUCCESS;
}

static ssize_t
tcp2_tsenddata(struct fid_ep *fid_ep, const void *buf, size_t len, void *desc,
	       uint64_t data, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	struct tcp2_ep *ep;
	struct tcp2_xfer_entry *tx_entry;

	ep = container_of(fid_ep, struct tcp2_ep, util_ep.ep_fid);
	tx_entry = tcp2_alloc_tsend(ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	tx_entry->hdr.base_hdr.flags |= TCP2_REMOTE_CQ_DATA;
	tx_entry->hdr.tag_data_hdr.tag = tag;
	tx_entry->hdr.tag_data_hdr.cq_data_hdr.cq_data = data;

	tcp2_init_tx_buf(tx_entry, sizeof(tx_entry->hdr.tag_data_hdr),
			 buf, len);
	tx_entry->context = context;
	tx_entry->cq_flags = tcp2_tx_completion_flag(ep, 0) |
			     FI_TAGGED | FI_SEND;
	tcp2_set_ack_flags(tx_entry, ep->util_ep.tx_op_flags);

	tcp2_queue_send(ep, tx_entry);
	return FI_SUCCESS;
}

static ssize_t
tcp2_tinjectdata(struct fid_ep *fid_ep, const void *buf, size_t len,
		 uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	struct tcp2_ep *ep;
	struct tcp2_xfer_entry *tx_entry;

	ep = container_of(fid_ep, struct tcp2_ep, util_ep.ep_fid);

	tx_entry = tcp2_alloc_tsend(ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	tx_entry->hdr.base_hdr.flags |= TCP2_REMOTE_CQ_DATA;
	tx_entry->hdr.tag_data_hdr.tag = tag;
	tx_entry->hdr.tag_data_hdr.cq_data_hdr.cq_data = data;

	tcp2_init_tx_inject(tx_entry, sizeof(tx_entry->hdr.tag_data_hdr),
			    buf, len);
	tx_entry->ctrl_flags = TCP2_INJECT_OP;

	tcp2_queue_send(ep, tx_entry);
	return FI_SUCCESS;
}

struct fi_ops_tagged tcp2_tagged_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = fi_no_tagged_recv,
	.recvv = fi_no_tagged_recvv,
	.recvmsg = fi_no_tagged_recvmsg,
	.send = tcp2_tsend,
	.sendv = tcp2_tsendv,
	.sendmsg = tcp2_tsendmsg,
	.inject = tcp2_tinject,
	.senddata = tcp2_tsenddata,
	.injectdata = tcp2_tinjectdata,
};
