/*
 * Copyright (c) 2013-2016 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
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

#include "ofi_iov.h"
#include "ofi_enosys.h"

#include "verbs_rdm.h"

static ssize_t fi_ibv_rdm_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
				  uint64_t flags)
{
	ssize_t ret = FI_SUCCESS;
	struct fi_ibv_rdm_ep *ep_rdm =
		container_of(ep, struct fi_ibv_rdm_ep, ep_fid);

	if (msg->iov_count > 1) {
		assert(0);
		return -FI_EMSGSIZE;
	}

	struct fi_ibv_rdm_conn *conn = ep_rdm->av->addr_to_conn(ep_rdm, msg->addr);

	struct fi_ibv_rdm_tagged_recv_start_data recv_data = {
		.peek_data = {
			.minfo = {
				.conn = conn,
				.tag = 0,
				.tagmask = 0,
				.is_tagged = 0
			},
			.context = msg->context,
			.flags = ep_rdm->rx_op_flags |
				(ep_rdm->rx_selective_completion ? flags :
				(flags | FI_COMPLETION))
		},
		.dest_addr =
			(msg->iov_count) ? msg->msg_iov[0].iov_base : NULL,
		.data_len = (msg->iov_count) ? msg->msg_iov[0].iov_len : 0,
		.ep = ep_rdm
	};
	struct fi_ibv_rdm_request *request =
		util_buf_alloc(ep_rdm->fi_ibv_rdm_request_pool);
	if (OFI_UNLIKELY(!request))
		return -FI_EAGAIN;

	fi_ibv_rdm_zero_request(request);
	request->ep = ep_rdm;
	FI_IBV_RDM_DBG_REQUEST("get_from_pool: ", request, FI_LOG_DEBUG);
	ret = fi_ibv_rdm_req_hndl(request, FI_IBV_EVENT_RECV_START,
				  &recv_data);

	VERBS_DBG(FI_LOG_EP_DATA,
		  "conn %p, len %zu, rbuf %p, fi_ctx %p, posted_recv %"PRIu32"\n",
		  conn, recv_data.data_len, recv_data.dest_addr,
		  msg->context, ep_rdm->posted_recvs);

	if (!ret && !request->state.err)
		ret = rdm_trecv_second_event(request, ep_rdm);

	return ret;
}

static ssize_t
fi_ibv_rdm_recvv(struct fid_ep *ep, const struct iovec *iov,
		 void **desc, size_t count, fi_addr_t src_addr,
		 void *context)
{
	const struct fi_msg msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = src_addr,
		.context = context,
		.data = 0
	};

	return fi_ibv_rdm_recvmsg(ep, &msg, 0ULL);
}

static ssize_t
fi_ibv_rdm_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
		fi_addr_t src_addr, void *context)
{
	const struct iovec iov = {
		.iov_base = buf,
		.iov_len = len
	};
	return fi_ibv_rdm_recvv(ep, &iov, &desc, 1, src_addr, context);
}

static ssize_t fi_ibv_rdm_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
				  uint64_t flags)
{
	struct fi_ibv_rdm_ep *ep_rdm = 
		container_of(ep, struct fi_ibv_rdm_ep, ep_fid);
	size_t i;
	struct fi_ibv_rdm_send_start_data sdata = {
		.ep_rdm = ep_rdm,
		.conn = ep_rdm->av->addr_to_conn(ep_rdm, msg->addr),
		.data_len = ofi_total_iov_len(msg->msg_iov, msg->iov_count),
		.context = msg->context,
		.flags = FI_MSG | FI_SEND | GET_TX_COMP_FLAG(ep_rdm, flags),
		.tag = 0,
		.is_tagged = 0,
		.buf.src_addr = NULL,
		.iov_count = 0,
		.imm = (uint32_t) 0,
		.stype = IBV_RDM_SEND_TYPE_GEN,
	};

	switch (msg->iov_count) {
	case 1:
		sdata.buf.src_addr = msg->msg_iov[0].iov_base;
		/* FALL THROUGH */
	case 0:
		break;
	default:
		/* TODO: 
		 * extra allocation & memcpy can be optimized if it's possible
		 * to send immediately
		 */
		if ((msg->iov_count > sdata.ep_rdm->iov_per_rndv_thr) ||
		    (sdata.data_len > sdata.ep_rdm->rndv_threshold))
			return -FI_EMSGSIZE;
		sdata.buf.iovec_arr =
			util_buf_alloc(ep_rdm->fi_ibv_rdm_extra_buffers_pool);
		for (i = 0; i < msg->iov_count; i++) {
			sdata.buf.iovec_arr[i].iov_base = msg->msg_iov[i].iov_base;
			sdata.buf.iovec_arr[i].iov_len = msg->msg_iov[i].iov_len;
		}
		sdata.iov_count = msg->iov_count;
		sdata.stype = IBV_RDM_SEND_TYPE_VEC;
		break;
	}

	return fi_ibv_rdm_send_common(&sdata);
}

static ssize_t fi_ibv_rdm_sendv(struct fid_ep *ep, const struct iovec *iov,
				void **desc, size_t count, fi_addr_t dest_addr,
				void *context)
{
	struct fi_ibv_rdm_ep *ep_rdm =
		container_of(ep, struct fi_ibv_rdm_ep, ep_fid);

	const struct fi_msg msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = dest_addr,
		.context = context,
		.data = 0
	};

	return fi_ibv_rdm_sendmsg(ep, &msg, GET_TX_COMP(ep_rdm));
}

static ssize_t fi_ibv_rdm_send(struct fid_ep *ep, const void *buf, size_t len,
			       void *desc, fi_addr_t dest_addr, void *context)
{
	const struct iovec iov = {
		.iov_base = (void *)buf,
		.iov_len = len
	};
	return fi_ibv_rdm_sendv(ep, &iov, &desc, 1, dest_addr, context);
}

static ssize_t fi_ibv_rdm_inject(struct fid_ep *ep_fid, const void *buf,
				 size_t len, fi_addr_t dest_addr)
{
	struct fi_ibv_rdm_ep *ep =
		container_of(ep_fid, struct fi_ibv_rdm_ep, ep_fid);
	struct fi_ibv_rdm_conn *conn = ep->av->addr_to_conn(ep, dest_addr);
	const size_t size = len + sizeof(struct fi_ibv_rdm_header);

	if (OFI_UNLIKELY(len > ep->rndv_threshold)) {
		assert(0);
		return -FI_EMSGSIZE;
	}

	if (!conn->postponed_entry) {
		struct fi_ibv_rdm_buf *sbuf = 
			fi_ibv_rdm_prepare_send_resources(conn);
		if (sbuf) {
			struct ibv_send_wr *bad_wr = NULL;
			struct ibv_sge sge = {
				.addr = (uintptr_t)(void *)sbuf,
				.length = size + FI_IBV_RDM_BUFF_SERVICE_DATA_SIZE,
				.lkey = fi_ibv_mr_internal_lkey(&conn->s_md),
			};
			struct ibv_send_wr wr = {
				.wr_id = FI_IBV_RDM_PACK_SERVICE_WR(conn),
				.sg_list = &sge,
				.num_sge = 1,
				.wr.rdma.remote_addr = (uintptr_t)
					fi_ibv_rdm_get_remote_addr(conn, sbuf),
				.wr.rdma.rkey = conn->remote_rbuf_rkey,
				.send_flags = (sge.length < ep->max_inline_rc)
					       ? IBV_SEND_INLINE : 0,
				.opcode = ep->eopcode,
			};

			sbuf->service_data.pkt_len = size;
			sbuf->header.tag = 0;
			sbuf->header.service_tag = 0;

			FI_IBV_RDM_SET_PKTTYPE(sbuf->header.service_tag,
					       FI_IBV_RDM_MSG_PKT);
			memcpy(&sbuf->payload, buf, len);

			FI_IBV_RDM_INC_SIG_POST_COUNTERS(conn, ep);
			if (OFI_UNLIKELY(ibv_post_send(conn->qp[0], &wr, &bad_wr))) {
				assert(0);
				return -errno;
			} else {
				VERBS_DBG(FI_LOG_EP_DATA,
					  "posted %d bytes, conn %p, len %zu\n",
					  sge.length, conn, len);
				return FI_SUCCESS;
			}
		}
	}

	fi_ibv_rdm_tagged_poll(ep);

	return -FI_EAGAIN;
}

static ssize_t fi_ibv_rdm_senddata(struct fid_ep *ep, const void *buf,
				   size_t len, void *desc, uint64_t data,
				   fi_addr_t dest_addr, void *context)
{
	assert(0);
	return -FI_ENOSYS;
}

static ssize_t fi_ibv_rdm_injectdata(struct fid_ep *ep, const void *buf,
				     size_t len, uint64_t data,
				     fi_addr_t dest_addr)
{
	assert(0);
	return -FI_ENOSYS;
}

struct fi_ops_msg fi_ibv_rdm_ep_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = fi_ibv_rdm_recv,
	.recvv = fi_ibv_rdm_recvv,
	.recvmsg = fi_ibv_rdm_recvmsg,
	.send = fi_ibv_rdm_send,
	.sendv = fi_ibv_rdm_sendv,
	.sendmsg = fi_ibv_rdm_sendmsg,
	.inject = fi_ibv_rdm_inject,
	.senddata = fi_ibv_rdm_senddata,
	.injectdata = fi_ibv_rdm_injectdata
};
