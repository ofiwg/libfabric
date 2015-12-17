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

extern struct fi_ibv_mem_pool fi_ibv_rdm_tagged_request_pool;

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

static ssize_t
fi_ibv_rdm_ep_rma_read(struct fid_ep *ep_fid, void *buf, size_t len,
		    void *desc, fi_addr_t src_addr,
		    uint64_t addr, uint64_t key, void *context)
{
	ssize_t ret = FI_SUCCESS;
	struct fi_ibv_rdm_ep *ep = container_of(ep_fid, struct fi_ibv_rdm_ep,
						ep_fid);

	if (desc == NULL && len >= ep->rndv_threshold) {
		goto out_errinput;
	}

	struct fi_ibv_rdm_tagged_conn *conn =
		(struct fi_ibv_rdm_tagged_conn *) src_addr;

	if (desc == NULL) {
		int again = 1;

		if (!conn->postponed_entry) {
			void *raw_sbuf =
				fi_ibv_rdm_tagged_prepare_send_resources(conn,
									 ep);

			if (raw_sbuf) {
				memcpy (raw_sbuf, buf, len);
				buf = raw_sbuf;
				desc = (void*)(uintptr_t)conn->s_mr->lkey;
				again = 0;
			}
		}

		if (again) {
			goto out_again;
		}
	}

	struct fi_ibv_rdm_tagged_request *request =
	    (struct fi_ibv_rdm_tagged_request *)
	    fi_verbs_mem_pool_get(&fi_ibv_rdm_tagged_request_pool);
	FI_IBV_RDM_TAGGED_DBG_REQUEST("get_from_pool: ", request, FI_LOG_DEBUG);

	/* Initial state */
	request->state.eager = FI_IBV_STATE_EAGER_BEGIN;
	request->state.rndv  = FI_IBV_STATE_RNDV_NOT_USED;

	struct fi_ibv_rdm_rma_start_data data = {
		.ep_rdm = container_of(ep_fid, struct fi_ibv_rdm_ep, ep_fid),
		.conn = (struct fi_ibv_rdm_tagged_conn *) src_addr,
		.context = context,
		.data_len = (uint32_t)len,
		.rbuf = addr,
		.lbuf = (uintptr_t)buf,
		.rkey = (uint32_t)key,
		.lkey = (uint32_t)(uintptr_t)desc,
		.op_code = IBV_WR_RDMA_READ
	};

	ret = fi_ibv_rdm_tagged_req_hndl(request,
		FI_IBV_EVENT_RMA_START, &data);
	ret = (ret == FI_EP_RDM_HNDL_SUCCESS) ? FI_SUCCESS : -FI_EOTHER;

out:
	return ret;

out_again:
	fi_ibv_rdm_tagged_poll(ep);
	ret = -FI_EAGAIN;
	goto out;

out_errinput:
	ret = -FI_EINVAL;
	goto out;
}

static ssize_t
fi_ibv_rdm_ep_rma_write(struct fid_ep *ep_fid, const void *buf, size_t len,
		     void *desc, fi_addr_t dest_addr,
		     uint64_t addr, uint64_t key, void *context)
{
	ssize_t ret = FI_SUCCESS;
	struct fi_ibv_rdm_ep *ep = container_of(ep_fid, struct fi_ibv_rdm_ep,
						ep_fid);

	if (desc == NULL && len >= ep->rndv_threshold) {
		goto out_errinput;
	}

	struct fi_ibv_rdm_tagged_conn *conn =
		(struct fi_ibv_rdm_tagged_conn *) dest_addr;

	if (desc == NULL) {
		int again = 1;

		if (!conn->postponed_entry) {
			void *raw_sbuf =
				fi_ibv_rdm_tagged_prepare_send_resources(conn,
									 ep);

			if (raw_sbuf) {
				memcpy (raw_sbuf, buf, len);
				buf = raw_sbuf;
				desc = (void*)(uintptr_t)conn->s_mr->lkey;
				again = 0;
			}
		}

		if (again) {
			goto out_again;
		}
	}

	struct fi_ibv_rdm_tagged_request *request =
	    (struct fi_ibv_rdm_tagged_request *)
	    fi_verbs_mem_pool_get(&fi_ibv_rdm_tagged_request_pool);
	FI_IBV_RDM_TAGGED_DBG_REQUEST("get_from_pool: ", request, FI_LOG_DEBUG);

	/* Initial state */
	request->state.eager = FI_IBV_STATE_EAGER_BEGIN;
	request->state.rndv  = FI_IBV_STATE_RNDV_NOT_USED;

	struct fi_ibv_rdm_rma_start_data data = {
		.conn = conn,
		.ep_rdm = ep,
		.context = context,
		.data_len = (uint32_t)len,
		.rbuf = addr,
		.lbuf = (uintptr_t)buf,
		.rkey = (uint32_t)key,
		.lkey = (uint32_t)(uintptr_t)desc,
		.op_code = IBV_WR_RDMA_WRITE
	};

	ret = fi_ibv_rdm_tagged_req_hndl(request,
		FI_IBV_EVENT_RMA_START, &data);
	ret = (ret == FI_EP_RDM_HNDL_SUCCESS) ? FI_SUCCESS : -FI_EOTHER;
out:
	return ret;

out_again:
	fi_ibv_rdm_tagged_poll(ep);
	ret = -FI_EAGAIN;
	goto out;

out_errinput:
	ret = -FI_EINVAL;
	goto out;
}

static ssize_t fi_ibv_rdm_ep_rma_inject_write(struct fid_ep *ep,
					      const void *buf, size_t len,
					      fi_addr_t dest_addr,
					      uint64_t addr, uint64_t key)
{
	struct fi_ibv_rdm_ep *ep_rdm = container_of(ep, struct fi_ibv_rdm_ep,
						    ep_fid);

	if (len >= ep_rdm->rndv_threshold) {
		return -FI_EMSGSIZE;
	}

	struct fi_ibv_rdm_tagged_conn *conn =
		(struct fi_ibv_rdm_tagged_conn *) dest_addr;

	if (!conn->postponed_entry) {
		void *raw_sbuf =
			fi_ibv_rdm_tagged_prepare_send_resources(conn, ep_rdm);

		if (raw_sbuf) {
			memcpy(raw_sbuf, buf, len);

			struct ibv_sge sge = { 0 };
			struct ibv_send_wr wr = { 0 };
			struct ibv_send_wr *bad_wr = NULL;
			wr.wr_id = FI_IBV_RDM_PACK_SERVICE_WR(conn);
			wr.sg_list = &sge;
			wr.num_sge = 1;
			wr.wr.rdma.remote_addr = addr;
			wr.wr.rdma.rkey = (uint32_t)key;
			wr.send_flags = (len < ep_rdm->max_inline_rc)
					? IBV_SEND_INLINE : 0;
			wr.opcode = IBV_WR_RDMA_WRITE;
			sge.addr = (uint64_t)raw_sbuf;
			sge.length = len;
			sge.lkey = conn->s_mr->lkey;

			FI_IBV_RDM_TAGGED_INC_SEND_COUNTERS(conn, ep_rdm,
							    wr.send_flags);
			int ret = ibv_post_send(conn->qp, &wr, &bad_wr);
			return (ret == 0) ? -FI_SUCCESS : -errno;
		}
	}

	return -FI_EAGAIN;
}

static struct fi_ops_rma fi_ibv_rdm_ep_rma_ops = {
	.size		= sizeof(struct fi_ops_rma),
	.read		= fi_ibv_rdm_ep_rma_read,
	.readv		= fi_no_rma_readv,
	.readmsg	= fi_no_rma_readmsg,
	.write		= fi_ibv_rdm_ep_rma_write,
	.writev		= fi_no_rma_writev,
	.writemsg	= fi_no_rma_writemsg,
	.inject		= fi_ibv_rdm_ep_rma_inject_write,
	.writedata	= fi_no_rma_writedata,
	.injectdata	= fi_no_rma_injectdata,
};

struct fi_ops_rma *fi_ibv_rdm_ep_ops_rma(struct fi_ibv_rdm_ep *ep)
{
	return &fi_ibv_rdm_ep_rma_ops;
}
