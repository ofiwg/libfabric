/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#include <rdma/fi_errno.h>

#include "config.h"

#include "fi_verbs.h"
#include "ep_rdm/verbs_rdm.h"
#include "ep_rdm/verbs_utils.h"

#define VERBS_COMP_READ_FLAGS(ep, flags) \
	((!VERBS_SELECTIVE_COMP(ep) || (flags & \
	  (FI_COMPLETION | FI_TRANSMIT_COMPLETE | FI_DELIVERY_COMPLETE))) ? \
	   IBV_SEND_SIGNALED : 0)
#define VERBS_COMP_READ(ep) \
	VERBS_COMP_READ_FLAGS(ep, ep->info->tx_attr->op_flags)

extern struct util_buf_pool* fi_ibv_rdm_request_pool;

static ssize_t
fi_ibv_msg_ep_rma_write(struct fid_ep *ep_fid, const void *buf, size_t len,
		     void *desc, fi_addr_t dest_addr,
		     uint64_t addr, uint64_t key, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_RDMA_WRITE;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	wr.send_flags = VERBS_INJECT(ep, len) | VERBS_COMP(ep);

	return fi_ibv_send_buf(ep, &wr, buf, len, desc, context);
}

static ssize_t
fi_ibv_msg_ep_rma_writev(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		      size_t count, fi_addr_t dest_addr,
		      uint64_t addr, uint64_t key, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;


	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_RDMA_WRITE;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	return fi_ibv_send_iov(ep, &wr, iov, desc, count, context);
}

static ssize_t
fi_ibv_msg_ep_rma_writemsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
			uint64_t flags)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	if (flags & FI_REMOTE_CQ_DATA) {
		wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
		wr.imm_data = htonl((uint32_t)msg->data);
	} else {
		wr.opcode = IBV_WR_RDMA_WRITE;
	}

	wr.wr.rdma.remote_addr = msg->rma_iov->addr;
	wr.wr.rdma.rkey = (uint32_t) msg->rma_iov->key;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	return fi_ibv_send_msg(ep, &wr, msg, flags);
}

static ssize_t
fi_ibv_msg_ep_rma_read(struct fid_ep *ep_fid, void *buf, size_t len,
		    void *desc, fi_addr_t src_addr,
		    uint64_t addr, uint64_t key, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_RDMA_READ;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	wr.send_flags = VERBS_COMP_READ(ep);

	return fi_ibv_send_buf(ep, &wr, buf, len, desc, context);
}

static ssize_t
fi_ibv_msg_ep_rma_readv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		     size_t count, fi_addr_t src_addr,
		     uint64_t addr, uint64_t key, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;
	size_t len = 0;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_RDMA_READ;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	wr.send_flags = VERBS_COMP_READ(ep);

	fi_ibv_set_sge_iov(wr.sg_list, iov, count, desc, len);

	return fi_ibv_send(ep, &wr, len, count, context);
}

static ssize_t
fi_ibv_msg_ep_rma_readmsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
			uint64_t flags)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;
	size_t len = 0;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_RDMA_READ;
	wr.wr.rdma.remote_addr = msg->rma_iov->addr;
	wr.wr.rdma.rkey = (uint32_t) msg->rma_iov->key;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	wr.send_flags = VERBS_COMP_READ_FLAGS(ep, flags);

	fi_ibv_set_sge_iov(wr.sg_list, msg->msg_iov, msg->iov_count, msg->desc,	len);

	return fi_ibv_send(ep, &wr, len, msg->iov_count, msg->context);
}

static ssize_t
fi_ibv_msg_ep_rma_writedata(struct fid_ep *ep_fid, const void *buf, size_t len,
			void *desc, uint64_t data, fi_addr_t dest_addr,
			uint64_t addr, uint64_t key, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	wr.imm_data = htonl((uint32_t)data);
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	wr.send_flags = VERBS_INJECT(ep, len) | VERBS_COMP(ep);

	return fi_ibv_send_buf(ep, &wr, buf, len, desc, context);
}

static ssize_t
fi_ibv_msg_ep_rma_inject_write(struct fid_ep *ep_fid, const void *buf, size_t len,
		     fi_addr_t dest_addr, uint64_t addr, uint64_t key)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_RDMA_WRITE;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;
	wr.send_flags = IBV_SEND_INLINE;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	return fi_ibv_send_buf_inline(ep, &wr, buf, len);
}

static ssize_t
fi_ibv_msg_ep_rma_inject_writedata(struct fid_ep *ep_fid, const void *buf, size_t len,
			uint64_t data, fi_addr_t dest_addr, uint64_t addr,
			uint64_t key)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	wr.imm_data = htonl((uint32_t)data);
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;
	wr.send_flags = IBV_SEND_INLINE;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	return fi_ibv_send_buf_inline(ep, &wr, buf, len);
}

static struct fi_ops_rma fi_ibv_msg_ep_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = fi_ibv_msg_ep_rma_read,
	.readv = fi_ibv_msg_ep_rma_readv,
	.readmsg = fi_ibv_msg_ep_rma_readmsg,
	.write = fi_ibv_msg_ep_rma_write,
	.writev = fi_ibv_msg_ep_rma_writev,
	.writemsg = fi_ibv_msg_ep_rma_writemsg,
	.inject = fi_ibv_msg_ep_rma_inject_write,
	.writedata = fi_ibv_msg_ep_rma_writedata,
	.injectdata = fi_ibv_msg_ep_rma_inject_writedata,
};

struct fi_ops_rma *fi_ibv_msg_ep_ops_rma(struct fi_ibv_msg_ep *ep)
{
	return &fi_ibv_msg_ep_rma_ops;
}

static inline ssize_t
fi_ibv_rdm_ep_rma_preinit(void **desc, void **raw_buf, size_t len,
			  struct fi_ibv_rdm_conn *conn,
			  struct fi_ibv_rdm_ep *ep)
{
	int again = 0;

	if ((desc == NULL && len >= ep->rndv_threshold) ||
	    (desc && *desc == NULL && len >= ep->rndv_threshold))
	{
		return -FI_EINVAL;
	}

	again = (!fi_ibv_rdm_check_connection(conn, ep) ||
		RMA_RESOURCES_IS_BUSY(conn, ep) || conn->postponed_entry);

	if ((!again) && raw_buf && desc && (*desc == NULL)) {
		*raw_buf = fi_ibv_rdm_rma_prepare_resources(conn, ep);
		if (*raw_buf) {
			*desc = (void*)(uintptr_t)conn->rma_mr->lkey;
		}
		again = !(*raw_buf);
	}

	if (again) {
		fi_ibv_rdm_tagged_poll(ep);
		return -FI_EAGAIN;
	}

	return FI_SUCCESS;
}

static ssize_t
fi_ibv_rdm_ep_rma_readmsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
		uint64_t flags)
{
	struct fi_ibv_rdm_ep *ep =
		container_of(ep_fid, struct fi_ibv_rdm_ep, ep_fid);

	struct fi_ibv_rdm_conn *conn = (struct fi_ibv_rdm_conn *)msg->addr;

	struct fi_ibv_rdm_rma_start_data start_data = {
		.ep_rdm = ep,
		.conn = conn,
		.context = msg->context,
		.flags = FI_RMA | FI_READ | (ep->tx_selective_completion ?
			(flags & FI_COMPLETION) : FI_COMPLETION),
		.data_len = (uint64_t)msg->msg_iov[0].iov_len,
		.rbuf = msg->rma_iov[0].addr,
		.lbuf = (uintptr_t)msg->msg_iov[0].iov_base,
		.rkey = (uint32_t)msg->rma_iov[0].key,
		.lkey = (uint32_t)(uintptr_t)msg->desc[0],
		.op_code = IBV_WR_RDMA_READ
	};

	struct fi_ibv_rma_post_ready_data post_ready_data = { .ep_rdm = ep };

	void *raw_buf = NULL;
	ssize_t ret = FI_SUCCESS;

	if(msg->iov_count != 1 || msg->rma_iov_count != 1) {
		assert(0);
		return -FI_EMSGSIZE;
	}

	ret = fi_ibv_rdm_ep_rma_preinit(&msg->desc[0], &raw_buf,
					msg->msg_iov[0].iov_len,
					conn, ep);
	if (ret) {
		return ret;
	}

	struct fi_ibv_rdm_request *request =
		util_buf_alloc(fi_ibv_rdm_request_pool);
	FI_IBV_RDM_DBG_REQUEST("get_from_pool: ", request, FI_LOG_DEBUG);

	/* Initial state */
	request->state.eager = FI_IBV_STATE_EAGER_BEGIN;
	request->state.rndv  = FI_IBV_STATE_RNDV_NOT_USED;
	request->state.err   = FI_SUCCESS;

	request->rmabuf = raw_buf;

	fi_ibv_rdm_req_hndl(request, FI_IBV_EVENT_RMA_START, &start_data);

	return fi_ibv_rdm_req_hndl(request, FI_IBV_EVENT_POST_READY,
				   &post_ready_data);
}

static ssize_t
fi_ibv_rdm_ep_rma_readv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t src_addr, uint64_t addr, uint64_t key,
		void *context)
{
	struct fi_ibv_rdm_ep *ep_rdm = 
		container_of(ep, struct fi_ibv_rdm_ep, ep_fid);

	struct fi_rma_iov rma_iov = {
		.addr = addr,
		.len = 0,
		.key = key
	};

	struct fi_msg_rma msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = src_addr,
		.rma_iov = &rma_iov,
		.rma_iov_count = 1,
		.context = context,
		.data = 0
	};

	size_t i;
	for (i = 0; i < count; i++) {
		rma_iov.len += iov[i].iov_len;
	}

	return fi_ibv_rdm_ep_rma_readmsg(ep, &msg,
		(ep_rdm->tx_selective_completion ? 0ULL : FI_COMPLETION));
}

static ssize_t
fi_ibv_rdm_ep_rma_read(struct fid_ep *ep_fid, void *buf, size_t len,
		    void *desc, fi_addr_t src_addr,
		    uint64_t addr, uint64_t key, void *context)
{
	const struct iovec iov = {
		.iov_base = buf,
		.iov_len = len
	};

	return fi_ibv_rdm_ep_rma_readv(ep_fid, &iov, &desc, 1, src_addr, addr,
					key, context);
}

static ssize_t
fi_ibv_rdm_ep_rma_writemsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
		uint64_t flags)
{
	struct fi_ibv_rdm_ep *ep = container_of(ep_fid, struct fi_ibv_rdm_ep,
						ep_fid);
	struct fi_ibv_rdm_conn *conn = (struct fi_ibv_rdm_conn *) msg->addr;
	struct fi_ibv_rdm_request *request = NULL;
	void *raw_buf = NULL;
	ssize_t ret = FI_SUCCESS;

	struct fi_ibv_rdm_rma_start_data start_data = {
		.conn = conn,
		.ep_rdm = ep,
		.context = msg->context,
		.flags = FI_RMA | FI_WRITE | (ep->tx_selective_completion ?
			(flags & FI_COMPLETION) : FI_COMPLETION),
		.data_len = (uint64_t)msg->msg_iov[0].iov_len,
		.rbuf = msg->rma_iov[0].addr,
		.lbuf = (uintptr_t)msg->msg_iov[0].iov_base,
		.rkey = (uint32_t)msg->rma_iov[0].key,
		.lkey = (uint32_t)(uintptr_t)msg->desc[0],
		.op_code = IBV_WR_RDMA_WRITE
	};

	if(msg->iov_count != 1 && msg->rma_iov_count != 1) {
		assert(0);
		return -FI_EMSGSIZE;
	}

	ret = fi_ibv_rdm_ep_rma_preinit(&msg->desc[0], &raw_buf,
					msg->msg_iov[0].iov_len,
					conn, ep);
	if (ret) {
		return ret;
	}

	request = util_buf_alloc(fi_ibv_rdm_request_pool);
	FI_IBV_RDM_DBG_REQUEST("get_from_pool: ", request, FI_LOG_DEBUG);

	/* Initial state */
	request->state.eager = FI_IBV_STATE_EAGER_BEGIN;
	request->state.rndv  = FI_IBV_STATE_RNDV_NOT_USED;
	request->state.err   = FI_SUCCESS;

	request->rmabuf = raw_buf;

	fi_ibv_rdm_req_hndl(request, FI_IBV_EVENT_RMA_START, &start_data);

	struct fi_ibv_rma_post_ready_data post_ready_data = { .ep_rdm = ep };

	return fi_ibv_rdm_req_hndl(request, FI_IBV_EVENT_POST_READY,
				   &post_ready_data);
}

static ssize_t
fi_ibv_rdm_ep_rma_writev(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t dest_addr, uint64_t addr, uint64_t key,
		void *context)
{
	struct fi_rma_iov rma_iov = {
		.addr = addr,
		.len = 0,
		.key = key
	};

	struct fi_msg_rma msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = dest_addr,
		.rma_iov = &rma_iov,
		.rma_iov_count = 1,
		.context = context,
		.data = 0
	};

	size_t i;
	for (i = 0; i < count; i++) {
		rma_iov.len += iov[i].iov_len;
	}

	struct fi_ibv_rdm_ep *ep_rdm =
		container_of(ep_fid, struct fi_ibv_rdm_ep, ep_fid);

	return fi_ibv_rdm_ep_rma_writemsg(ep_fid, &msg,
		(ep_rdm->tx_selective_completion ? 0ULL : FI_COMPLETION));
}

static ssize_t
fi_ibv_rdm_ep_rma_write(struct fid_ep *ep_fid, const void *buf, size_t len,
			void *desc, fi_addr_t dest_addr, uint64_t addr,
			uint64_t key, void *context)
{
	const struct iovec iov = {
		.iov_base = (void *)buf,
		.iov_len = len
	};

	return fi_ibv_rdm_ep_rma_writev(ep_fid, &iov, &desc, 1, dest_addr, addr,
					key, context);
}

static ssize_t fi_ibv_rdm_ep_rma_inject_write(struct fid_ep *ep,
					      const void *buf, size_t len,
					      fi_addr_t dest_addr,
					      uint64_t addr, uint64_t key)
{
	struct fi_ibv_rdm_ep *ep_rdm = container_of(ep, struct fi_ibv_rdm_ep,
						    ep_fid);
	struct fi_ibv_rdm_conn *conn = (struct fi_ibv_rdm_conn *) dest_addr;
	struct fi_ibv_rdm_request *request = NULL;

	struct fi_ibv_rdm_rma_start_data start_data = {
		.conn = conn,
		.ep_rdm = ep_rdm,
		.flags = 0, /* inject does not generate completion */
		.data_len = (uint64_t)len,
		.rbuf = addr,
		.lbuf = (uintptr_t)buf,
		.rkey = (uint32_t)key,
		.lkey = 0
	};

	ssize_t ret = fi_ibv_rdm_ep_rma_preinit(NULL, NULL, len, conn, ep_rdm);
	if (ret) {
		return ret;
	}

	request = util_buf_alloc(fi_ibv_rdm_request_pool);

	FI_IBV_RDM_DBG_REQUEST("get_from_pool: ", request, FI_LOG_DEBUG);

	/* Initial state */
	request->state.eager = FI_IBV_STATE_EAGER_RMA_INJECT;
	request->state.rndv  = FI_IBV_STATE_RNDV_NOT_USED;
	request->state.err   = FI_SUCCESS;

	ret = fi_ibv_rdm_req_hndl(request, FI_IBV_EVENT_RMA_START, &start_data);

	switch (ret)
	{
	case FI_SUCCESS:
		return ret;
	case -FI_EAGAIN:
		break;
	default:
		ret = -errno;
		break;
	}

	FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
	util_buf_release(fi_ibv_rdm_request_pool, request);

	fi_ibv_rdm_tagged_poll(ep_rdm);

	return ret;
}

static struct fi_ops_rma fi_ibv_rdm_ep_rma_ops = {
	.size		= sizeof(struct fi_ops_rma),
	.read		= fi_ibv_rdm_ep_rma_read,
	.readv		= fi_ibv_rdm_ep_rma_readv,
	.readmsg	= fi_ibv_rdm_ep_rma_readmsg,
	.write		= fi_ibv_rdm_ep_rma_write,
	.writev		= fi_ibv_rdm_ep_rma_writev,
	.writemsg	= fi_ibv_rdm_ep_rma_writemsg,
	.inject		= fi_ibv_rdm_ep_rma_inject_write,
	.writedata	= fi_no_rma_writedata,
	.injectdata	= fi_no_rma_injectdata,
};

struct fi_ops_rma *fi_ibv_rdm_ep_ops_rma(struct fi_ibv_rdm_ep *ep)
{
	return &fi_ibv_rdm_ep_rma_ops;
}
