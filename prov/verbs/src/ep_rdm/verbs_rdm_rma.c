/*
 * Copyright (c) 2018 Intel Corporation, Inc.  All rights reserved.
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

#include "verbs_rdm.h"

static inline ssize_t
fi_ibv_rdm_ep_rma_preinit(void **desc, struct fi_ibv_rdm_buf **rdm_buf,
			  size_t len, struct fi_ibv_rdm_conn *conn,
			  struct fi_ibv_rdm_ep *ep)
{
	assert(desc && rdm_buf);

	if (*desc == NULL && len < ep->rndv_threshold) {
		*rdm_buf = fi_ibv_rdm_rma_prepare_resources(conn);
		if (*rdm_buf)
			*desc = (void *)(uintptr_t)
				fi_ibv_mr_internal_lkey(&conn->rma_md);
		else
			goto again;
	} else if (!fi_ibv_rdm_check_connection(conn) ||
		   RMA_RESOURCES_IS_BUSY(conn, ep) ||
		   conn->postponed_entry) {
		goto again;
	}

	return FI_SUCCESS;
again:
	fi_ibv_rdm_tagged_poll(ep);
	return -FI_EAGAIN;
}

static ssize_t
fi_ibv_rdm_ep_rma_readmsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
		uint64_t flags)
{
	struct fi_ibv_rdm_ep *ep =
		container_of(ep_fid, struct fi_ibv_rdm_ep, ep_fid);
	struct fi_ibv_rdm_conn *conn = ep->av->addr_to_conn(ep, msg->addr);
	struct fi_ibv_rdm_rma_start_data start_data = {
		.ep_rdm = ep,
		.conn = conn,
		.context = msg->context,
		.flags = FI_RMA | FI_READ | GET_TX_COMP_FLAG(ep, flags),
		.data_len = (uint64_t)msg->msg_iov[0].iov_len,
		.rbuf = (uintptr_t)msg->rma_iov[0].addr,
		.lbuf = (uintptr_t)msg->msg_iov[0].iov_base,
		.mr_rkey = (uint64_t)(uintptr_t)(msg->rma_iov[0].key),
		.mr_lkey = (uint64_t)(uintptr_t)(msg->desc ? msg->desc[0] : NULL),
		.op_code = IBV_WR_RDMA_READ
	};
	struct fi_ibv_rma_post_ready_data post_ready_data = { .ep_rdm = ep };

	struct fi_ibv_rdm_buf *rdm_buf = NULL;
	ssize_t ret = FI_SUCCESS;
	struct fi_ibv_rdm_request *request;

	if(msg->iov_count != 1 || msg->rma_iov_count != 1) {
		assert(0);
		return -FI_EMSGSIZE;
	}

	ret = fi_ibv_rdm_ep_rma_preinit((void**)&start_data.mr_lkey, &rdm_buf,
					msg->msg_iov[0].iov_len,
					conn, ep);
	if (ret) {
		return ret;
	}

	request = util_buf_alloc(ep->fi_ibv_rdm_request_pool);
	if (OFI_UNLIKELY(!request))
		return -FI_EAGAIN;

	fi_ibv_rdm_zero_request(request);
	request->ep = ep;
	FI_IBV_RDM_DBG_REQUEST("get_from_pool: ", request, FI_LOG_DEBUG);

	/* Initial state */
	request->state.eager = FI_IBV_STATE_EAGER_BEGIN;
	request->state.rndv  = FI_IBV_STATE_RNDV_NOT_USED;
	request->state.err   = FI_SUCCESS;

	request->minfo.is_tagged = 0;
	request->rmabuf = rdm_buf;

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

	return fi_ibv_rdm_ep_rma_readmsg(ep, &msg, GET_TX_COMP(ep_rdm));
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
	struct fi_ibv_rdm_conn *conn = ep->av->addr_to_conn(ep, msg->addr);
	struct fi_ibv_rdm_request *request = NULL;
	struct fi_ibv_rdm_buf *rdm_buf = NULL;
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
		.mr_rkey = msg->rma_iov[0].key,
		.mr_lkey = (uint64_t)(uintptr_t)(msg->desc ? msg->desc[0] : NULL),
		.op_code = IBV_WR_RDMA_WRITE
	};

	if(msg->iov_count != 1 && msg->rma_iov_count != 1) {
		assert(0);
		return -FI_EMSGSIZE;
	}

	ret = fi_ibv_rdm_ep_rma_preinit((void**)&start_data.mr_lkey, &rdm_buf,
					msg->msg_iov[0].iov_len,
					conn, ep);
	if (ret) {
		return ret;
	}

	request = util_buf_alloc(ep->fi_ibv_rdm_request_pool);
	if (OFI_UNLIKELY(!request))
		return -FI_EAGAIN;

	fi_ibv_rdm_zero_request(request);
	request->ep = ep;
	/* Initial state */
	request->state.eager = FI_IBV_STATE_EAGER_BEGIN;
	request->state.rndv  = FI_IBV_STATE_RNDV_NOT_USED;
	request->state.err   = FI_SUCCESS;
	request->minfo.is_tagged = 0;
	request->rmabuf = rdm_buf;

	FI_IBV_RDM_DBG_REQUEST("get_from_pool: ", request, FI_LOG_DEBUG);

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

	return fi_ibv_rdm_ep_rma_writemsg(ep_fid, &msg, GET_TX_COMP(ep_rdm));
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
	struct fi_ibv_rdm_conn *conn = ep_rdm->av->addr_to_conn(ep_rdm, dest_addr);
	struct fi_ibv_rdm_rma_start_data start_data = {
		.conn = conn,
		.ep_rdm = ep_rdm,
		.flags = 0, /* inject does not generate completion */
		.data_len = (uint64_t)len,
		.rbuf = addr,
		.lbuf = (uintptr_t)buf,
		.mr_rkey = (uint64_t)key,
		.mr_lkey = 0
	};
	ssize_t ret;
	struct fi_ibv_rdm_request *request =
		util_buf_alloc(ep_rdm->fi_ibv_rdm_request_pool);

	if (OFI_UNLIKELY(!request))
		return -FI_EAGAIN;

	fi_ibv_rdm_zero_request(request);
	request->ep = ep_rdm;
	FI_IBV_RDM_DBG_REQUEST("get_from_pool: ", request, FI_LOG_DEBUG);

	/* Initial state */
	request->state.eager = FI_IBV_STATE_EAGER_RMA_INJECT;
	request->state.rndv  = FI_IBV_STATE_RNDV_NOT_USED;
	request->state.err   = FI_SUCCESS;

	request->minfo.is_tagged = 0;
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
	util_buf_release(ep_rdm->fi_ibv_rdm_request_pool, request);

	fi_ibv_rdm_tagged_poll(ep_rdm);

	return ret;
}

struct fi_ops_rma fi_ibv_rdm_ep_rma_ops = {
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

