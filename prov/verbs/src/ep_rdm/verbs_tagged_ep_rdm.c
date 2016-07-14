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

#include <prov/verbs/src/fi_verbs.h>

#include <config.h>

#include <fi_list.h>
#include <fi_enosys.h>
#include <rdma/fi_tagged.h>

#include "verbs_queuing.h"
#include "verbs_tagged_ep_rdm_states.h"

struct fi_ibv_rdm_cq fi_ibv_rdm_comp_queue;

extern struct util_buf_pool *fi_ibv_rdm_tagged_request_pool;
extern struct util_buf_pool *fi_ibv_rdm_tagged_extra_buffers_pool;

DEFINE_LIST(fi_ibv_rdm_tagged_recv_posted_queue);
DEFINE_LIST(fi_ibv_rdm_tagged_recv_unexp_queue);
DEFINE_LIST(fi_ibv_rdm_postponed_queue);

static inline int fi_ibv_rdm_tagged_poll_send(struct fi_ibv_rdm_ep *ep);
static inline int fi_ibv_rdm_tagged_poll_recv(struct fi_ibv_rdm_ep *ep);

int fi_ibv_rdm_tagged_prepare_send_request(
	struct fi_ibv_rdm_tagged_request *request, struct fi_ibv_rdm_ep *ep)
{
#if ENABLE_DEBUG
	int res =
		FI_IBV_RDM_TAGGED_SENDS_OUTGOING_ARE_LIMITED(request->minfo.conn, ep);
	if (res) {
		FI_IBV_RDM_DBG_REQUEST
			("failed because SENDS_OUTGOING_ARE_LIMITED", request,
			FI_LOG_DEBUG);
		return !res;
	}
	res = PEND_SEND_IS_LIMITED(ep);
	if (res) {
		FI_IBV_RDM_DBG_REQUEST
			("failed because PEND_SEND_IS_LIMITED", request,
			FI_LOG_DEBUG);
		return !res;
	}
#endif // ENABLE_DEBUG
	request->sbuf = fi_ibv_rdm_prepare_send_resources(request->minfo.conn, ep);
	return !!request->sbuf;
}

int fi_ibv_rdm_prepare_rma_request(struct fi_ibv_rdm_tagged_request *request,
	struct fi_ibv_rdm_ep *ep)
{
	request->rmabuf =
		fi_ibv_rdm_rma_prepare_resources(request->minfo.conn, ep);
	return !!request->rmabuf;
}

static int fi_ibv_rdm_tagged_getname(fid_t fid, void *addr, size_t * addrlen)
{
	struct fi_ibv_rdm_ep *ep =
		container_of(fid, struct fi_ibv_rdm_ep, ep_fid);

	if (FI_IBV_RDM_DFLT_ADDRLEN > *addrlen) {
		*addrlen = FI_IBV_RDM_DFLT_ADDRLEN;
		return -FI_ETOOSMALL;
	}
	memset(addr, 0, *addrlen);
	memcpy(addr, &ep->cm.my_addr, FI_IBV_RDM_DFLT_ADDRLEN);
	ep->addrlen = *addrlen;

	return 0;
}

static inline ssize_t
rdm_trecv_second_event(struct fi_ibv_rdm_tagged_request *request,
			struct fi_ibv_rdm_ep *ep)
{
	ssize_t ret = FI_SUCCESS;

	switch (request->state.rndv)
	{
	case FI_IBV_STATE_RNDV_NOT_USED:
		if (request->state.eager != FI_IBV_STATE_EAGER_RECV_WAIT4PKT) {
			struct fi_ibv_recv_got_pkt_process_data data = {
				.ep = ep
			};
			ret = fi_ibv_rdm_tagged_req_hndl(request,
							 FI_IBV_EVENT_RECV_START,
							 &data);
		}
		break;
	case FI_IBV_STATE_RNDV_RECV_WAIT4RES:
		if (fi_ibv_rdm_tagged_prepare_send_request(request, ep)) {
			struct fi_ibv_rdm_tagged_send_ready_data data = {
				.ep = ep
			};
			ret = fi_ibv_rdm_tagged_req_hndl(request,
							FI_IBV_EVENT_POST_READY,
							&data);
		}
		break;
	default:
		break;
	}

	return ret;
}

static ssize_t
fi_ibv_rdm_tagged_recvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
			  uint64_t flags)
{
	ssize_t ret = FI_SUCCESS;
	struct fi_ibv_rdm_ep *ep_rdm =
		container_of(ep_fid, struct fi_ibv_rdm_ep, ep_fid);

	if (msg->iov_count > 1) {
		assert(0);
		return -FI_EMSGSIZE;
	}

	struct fi_ibv_rdm_tagged_conn *conn =
		(msg->addr == FI_ADDR_UNSPEC) ? NULL :
		(struct fi_ibv_rdm_tagged_conn *) msg->addr;

	struct fi_ibv_rdm_tagged_recv_start_data recv_data = {
		.peek_data = {
			.minfo = {
				.conn = conn,
				.tag = msg->tag,
				.tagmask = ~(msg->ignore)
			},
			.context = msg->context,
			.flags = (ep_rdm->rx_selective_completion ?
				flags : (flags | FI_COMPLETION))
		},
		.dest_addr =
			(msg->iov_count) ? msg->msg_iov[0].iov_base : NULL,
		.data_len = (msg->iov_count) ? msg->msg_iov[0].iov_len : 0,
		.ep = ep_rdm
	};

	struct fi_ibv_rdm_tagged_request *request =
		util_buf_alloc(fi_ibv_rdm_tagged_request_pool);

	fi_ibv_rdm_tagged_zero_request(request);
	FI_IBV_RDM_DBG_REQUEST("get_from_pool: ", request, FI_LOG_DEBUG);

	if (flags & FI_PEEK) {
		recv_data.peek_data.flags |= FI_COMPLETION;
		ret = fi_ibv_rdm_tagged_req_hndl(request,
						FI_IBV_EVENT_RECV_PEEK,
						&recv_data.peek_data);
		if (ret == -FI_ENOMSG) {
			fi_ibv_rdm_tagged_poll(ep_rdm);
		}
	} else if (flags & FI_CLAIM) {
		recv_data.peek_data.flags |= FI_COMPLETION;
		ret = fi_ibv_rdm_tagged_req_hndl(request,
						 FI_IBV_EVENT_RECV_START,
						 &recv_data);
		if (!ret) {
			ret = rdm_trecv_second_event(request, ep_rdm);
		}
	} else {
		ret = fi_ibv_rdm_tagged_req_hndl(request,
						 FI_IBV_EVENT_RECV_START,
						 &recv_data);

		VERBS_DBG(FI_LOG_EP_DATA,
			"fi_recvfrom: conn %p, tag 0x%llx, len %llu, rbuf %p, fi_ctx %p, posted_recv %d\n",
			conn, msg->tag, recv_data.data_len, recv_data.dest_addr,
			msg->context, ep_rdm->posted_recvs);

		if (!ret && !request->state.err) {
			ret = rdm_trecv_second_event(request, ep_rdm);
		}
	}

	return ret;
}

static ssize_t
fi_ibv_rdm_tagged_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
			size_t count, fi_addr_t src_addr, uint64_t tag,
			uint64_t ignore, void *context)
{
	struct fi_ibv_rdm_ep *ep_rdm =
		container_of(ep_fid, struct fi_ibv_rdm_ep, ep_fid);

	const struct fi_msg_tagged msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = src_addr,
		.tag = tag,
		.ignore = ignore,
		.context = context,
		.data = 0
	};

	return fi_ibv_rdm_tagged_recvmsg(ep_fid, &msg,
		(ep_rdm->rx_selective_completion ? 0ULL : FI_COMPLETION));

}

static ssize_t fi_ibv_rdm_tagged_recvfrom(struct fid_ep *ep_fid, void *buf,
					  size_t len, void *desc,
					  fi_addr_t src_addr, uint64_t tag,
					  uint64_t ignore, void *context)
{
	const struct iovec iov = {
		.iov_base = buf,
		.iov_len = len
	};

	return fi_ibv_rdm_tagged_recvv(ep_fid, &iov, &desc, 1, src_addr, tag,
					ignore, context);
}

static inline ssize_t 
fi_ibv_rdm_tagged_inject(struct fid_ep *fid, const void *buf, size_t len, 
			 fi_addr_t dest_addr, uint64_t tag)
{
	struct fi_ibv_rdm_tagged_conn *conn =
		(struct fi_ibv_rdm_tagged_conn *) dest_addr;
	struct fi_ibv_rdm_ep *ep =
		container_of(fid, struct fi_ibv_rdm_ep, ep_fid);

	const size_t size = len + sizeof(struct fi_ibv_rdm_header);

	if (len > ep->rndv_threshold) {
		return -FI_EMSGSIZE;
	}

	const int in_order = (conn->postponed_entry) ? 0 : 1;

	if (in_order) {
		struct fi_ibv_rdm_buf *sbuf = 
			fi_ibv_rdm_prepare_send_resources(conn, ep);
		if (sbuf) {
			struct ibv_sge sge = {0};
			struct ibv_send_wr wr = {0};
			struct ibv_send_wr *bad_wr = NULL;

			sge.addr = (uintptr_t)(void*)sbuf;
			sge.length = size + FI_IBV_RDM_BUFF_SERVICE_DATA_SIZE;
			sge.lkey = conn->s_mr->lkey;

			wr.wr_id = FI_IBV_RDM_PACK_SERVICE_WR(conn);
			wr.sg_list = &sge;
			wr.num_sge = 1;
			wr.wr.rdma.remote_addr = (uintptr_t)
				fi_ibv_rdm_get_remote_addr(conn, sbuf);
			wr.wr.rdma.rkey = conn->remote_rbuf_rkey;
			wr.send_flags = (sge.length < ep->max_inline_rc)
				? IBV_SEND_INLINE : 0;
			wr.imm_data = 0;
			wr.opcode = ep->topcode;

			sbuf->service_data.pkt_len = size;
			sbuf->header.tag = tag;
			sbuf->header.service_tag = 0;

			FI_IBV_RDM_SET_PKTTYPE(sbuf->header.service_tag,
					       FI_IBV_RDM_EAGER_PKT);
			if ((len > 0) && (buf)) {
				memcpy(&sbuf->payload, buf, len);
			}

			FI_IBV_RDM_INC_SIG_POST_COUNTERS(conn, ep,
							 wr.send_flags);
			if (ibv_post_send(conn->qp[0], &wr, &bad_wr)) {
				assert(0);
				return -errno;
			} else {
				VERBS_DBG(FI_LOG_EP_DATA,
					"posted %d bytes, conn %p, len %d, tag 0x%llx\n",
					sge.length, conn, len, tag);
				return FI_SUCCESS;
			}
		}
	}

	fi_ibv_rdm_tagged_poll(ep);

	return -FI_EAGAIN;
}

static ssize_t
fi_ibv_rdm_tagged_send_common(struct fi_ibv_rdm_tsend_start_data* sdata)
{
	struct fi_ibv_rdm_tagged_request *request =
		util_buf_alloc(fi_ibv_rdm_tagged_request_pool);
	FI_IBV_RDM_DBG_REQUEST("get_from_pool: ", request, FI_LOG_DEBUG);

	/* Initial state */
	request->state.eager = FI_IBV_STATE_EAGER_BEGIN;
	request->state.rndv  = FI_IBV_STATE_RNDV_NOT_USED;
	request->state.err   = FI_SUCCESS;

	const int in_order = (sdata->conn->postponed_entry) ? 0 : 1;
	int ret = fi_ibv_rdm_tagged_req_hndl(request, FI_IBV_EVENT_SEND_START,
		sdata);

	if (!ret && in_order &&
		fi_ibv_rdm_tagged_prepare_send_request(request, sdata->ep_rdm))
	{
		struct fi_ibv_rdm_tagged_send_ready_data req_data = 
			{ .ep = sdata->ep_rdm };
		ret = fi_ibv_rdm_tagged_req_hndl(request, 
			FI_IBV_EVENT_POST_READY, &req_data);
	}

	return ret;
}

static ssize_t fi_ibv_rdm_tagged_senddatato(struct fid_ep *fid, const void *buf,
					    size_t len, void *desc,
					    uint64_t data, fi_addr_t dest_addr,
					    uint64_t tag, void *context)
{
	struct fi_ibv_rdm_ep *ep_rdm = 
		container_of(fid, struct fi_ibv_rdm_ep, ep_fid);

	struct fi_ibv_rdm_tsend_start_data sdata = {
		.ep_rdm = container_of(fid, struct fi_ibv_rdm_ep, ep_fid),
		.conn = (struct fi_ibv_rdm_tagged_conn *) dest_addr,
		.data_len = len,
		.context = context,
		.flags = FI_TAGGED | FI_SEND |
			(ep_rdm->tx_selective_completion ? 0ULL : FI_COMPLETION),
		.tag = tag,
		.buf.src_addr = (void*)buf,
		.iov_count = 0,
		.imm = (uint32_t) data,
		.stype = IBV_RDM_SEND_TYPE_GEN
	};

	return fi_ibv_rdm_tagged_send_common(&sdata);
}

static ssize_t fi_ibv_rdm_tagged_sendto(struct fid_ep *fid, const void *buf,
					size_t len, void *desc,
					fi_addr_t dest_addr, uint64_t tag,
					void *context)
{
	return fi_ibv_rdm_tagged_senddatato(fid, buf, len, desc, 0, dest_addr,
					    tag, context);
}

static ssize_t fi_ibv_rdm_tagged_sendmsg(struct fid_ep *ep,
	const struct fi_msg_tagged *msg, uint64_t flags)
{
	struct fi_ibv_rdm_ep *ep_rdm = 
		container_of(ep, struct fi_ibv_rdm_ep, ep_fid);

	struct fi_ibv_rdm_tsend_start_data sdata = {
		.ep_rdm = container_of(ep, struct fi_ibv_rdm_ep, ep_fid),
		.conn = (struct fi_ibv_rdm_tagged_conn *) msg->addr,
		.data_len = 0,
		.context = msg->context,
		.flags = FI_TAGGED | FI_SEND | (ep_rdm->tx_selective_completion ?
			(flags & FI_COMPLETION) : FI_COMPLETION),
		.tag = msg->tag,
		.buf.src_addr = NULL,
		.iov_count = 0,
		.imm = (uint32_t) 0,
		.stype = IBV_RDM_SEND_TYPE_UND
	};

	size_t i;
	for (i = 0; i < msg->iov_count; i++) {
		sdata.data_len += msg->msg_iov[i].iov_len;
	}

	if ((msg->iov_count > (sdata.ep_rdm->rndv_threshold / sizeof(struct iovec))) ||
	    (msg->iov_count > 1 && (sdata.data_len > sdata.ep_rdm->rndv_threshold)))
	{
		return -FI_EMSGSIZE;
	}

	switch (msg->iov_count)
	{
	case 1:
		sdata.buf.src_addr = msg->msg_iov[0].iov_base;
	case 0:
		sdata.stype = IBV_RDM_SEND_TYPE_GEN;
		break;
	default:
		/* TODO: 
		 * extra allocation & memcpy can be optimized if it's possible
		 * to send immediately
		 */
		sdata.buf.iovec_arr =
			util_buf_alloc(fi_ibv_rdm_tagged_extra_buffers_pool);
		for (i = 0; i < msg->iov_count; i++) {
			sdata.buf.iovec_arr[i].iov_base = msg->msg_iov[i].iov_base;
			sdata.buf.iovec_arr[i].iov_len = msg->msg_iov[i].iov_len;
		}
		sdata.iov_count = msg->iov_count;
		sdata.stype = IBV_RDM_SEND_TYPE_VEC;
		break;
	}

	return fi_ibv_rdm_tagged_send_common(&sdata);
}

static ssize_t fi_ibv_rdm_tagged_sendv(struct fid_ep *ep,
				       const struct iovec *iov, void **desc,
				       size_t count, fi_addr_t dest_addr,
				       uint64_t tag, void *context)
{
	struct fi_ibv_rdm_ep *ep_rdm = 
		container_of(ep, struct fi_ibv_rdm_ep, ep_fid);

	const struct fi_msg_tagged msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = dest_addr,
		.tag = tag,
		.ignore = 0,
		.context = context,
		.data = 0
	};

	return fi_ibv_rdm_tagged_sendmsg(ep, &msg,
		(ep_rdm->tx_selective_completion ? 0ULL : FI_COMPLETION));
}

struct fi_ops_tagged fi_ibv_rdm_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = fi_ibv_rdm_tagged_recvfrom,
	.recvv = fi_ibv_rdm_tagged_recvv,
	.recvmsg = fi_ibv_rdm_tagged_recvmsg,
	.send = fi_ibv_rdm_tagged_sendto,
	.sendv = fi_ibv_rdm_tagged_sendv,
	.sendmsg = fi_ibv_rdm_tagged_sendmsg,
	.inject = fi_ibv_rdm_tagged_inject,
	.senddata = fi_ibv_rdm_tagged_senddatato,
	.injectdata = fi_no_tagged_injectdata
};

struct fi_ops_cm fi_ibv_rdm_tagged_ep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.getname = fi_ibv_rdm_tagged_getname,	/* TODO */
};

static inline void
fi_ibv_rdm_tagged_release_remote_sbuff(struct fi_ibv_rdm_tagged_conn *conn,
	struct fi_ibv_rdm_ep *ep)
{
	struct ibv_sge sge;
	sge.addr = (uint64_t) & conn->sbuf_ack_status;
	sge.length = sizeof(conn->sbuf_ack_status);
	sge.lkey = conn->ack_mr->lkey;

	struct ibv_send_wr *bad_wr = NULL;
	struct ibv_send_wr wr = { 0 };

	wr.wr_id = FI_IBV_RDM_PACK_SERVICE_WR(conn);
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.wr.rdma.remote_addr = (uint64_t)
		&conn->remote_sbuf_head->service_data.status;
	wr.wr.rdma.rkey = conn->remote_sbuf_rkey;
	wr.send_flags = (sge.length < ep->max_inline_rc) ? IBV_SEND_INLINE : 0;
	/* w/o imm - do not put it into recv completion queue */
	wr.opcode = IBV_WR_RDMA_WRITE;

	FI_IBV_RDM_INC_SIG_POST_COUNTERS(conn, ep, wr.send_flags);
	VERBS_DBG(FI_LOG_EP_DATA,
		"posted %d bytes, remote sbuff released\n", sge.length);
	int ret = ibv_post_send(conn->qp[0], &wr, &bad_wr);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_EP_DATA, "ibv_post_send", errno);
		assert(0);
	};

	if (conn->sends_outgoing > ep->n_buffs) {
		fi_ibv_rdm_tagged_poll_send(ep);
	}
}

static inline void
fi_ibv_rdm_process_recv(struct fi_ibv_rdm_ep *ep,
	struct fi_ibv_rdm_tagged_conn *conn, int arrived_len,
	struct fi_ibv_rdm_buf *rbuf)
{
	struct fi_ibv_rdm_tagged_request *request = NULL;

	int pkt_type = FI_IBV_RDM_GET_PKTTYPE(rbuf->header.service_tag);

	if (pkt_type == FI_IBV_RDM_RNDV_ACK_PKT) {
		memcpy(&request, &rbuf->payload, sizeof(request));
		assert(request);
		VERBS_DBG(FI_LOG_EP_DATA,
			"GOT RNDV ACK from conn %p, id %p\n", conn, request);
	} else if (pkt_type != FI_IBV_RDM_RMA_PKT) {
		struct fi_verbs_rdm_tagged_minfo minfo = {
			.conn = conn,
			.tag = rbuf->header.tag,
			.tagmask = 0
		};

		struct dlist_entry *found_entry =
			dlist_find_first_match(&fi_ibv_rdm_tagged_recv_posted_queue,
			fi_ibv_rdm_tagged_req_match_by_info,
			&minfo);

		if (found_entry) {
			struct fi_ibv_rdm_tagged_request *found_request =
				container_of(found_entry,
				struct fi_ibv_rdm_tagged_request,
				queue_entry);

			fi_ibv_rdm_tagged_remove_from_posted_queue
				(found_request, ep);

			request = found_request;
		} else {
			request = util_buf_alloc(fi_ibv_rdm_tagged_request_pool);
			fi_ibv_rdm_tagged_zero_request(request);

			FI_IBV_RDM_DBG_REQUEST("get_from_pool: ", request,
						FI_LOG_DEBUG);
		}
	}

	/* RMA packets are not handled yet (without IMM) */
	if (pkt_type != FI_IBV_RDM_RMA_PKT) {

		struct fi_ibv_recv_got_pkt_preprocess_data p = {
			.conn = conn,
			.ep = ep,
			.rbuf = rbuf,
			.arrived_len = arrived_len,
			.pkt_type = pkt_type,
			.imm_data = 0 // TODO:
		};

		fi_ibv_rdm_tagged_req_hndl(request,
			FI_IBV_EVENT_RECV_GOT_PKT_PROCESS, &p);
	}
}

static inline
void check_and_repost_receives(struct fi_ibv_rdm_ep *ep,
	struct fi_ibv_rdm_tagged_conn *conn)
{
	conn->recv_preposted--;
	VERBS_DBG(FI_LOG_EP_DATA, "conn %p remain prepost recvs %d\n", conn, conn->recv_preposted);
	if (conn->recv_preposted < ep->recv_preposted_threshold) {
		int to_post = ep->rq_wr_depth - conn->recv_preposted;
		ssize_t res = fi_ibv_rdm_repost_receives(conn, ep, to_post);
		if (res < 0) {
			VERBS_INFO(FI_LOG_EP_DATA, "repost recv failed %d\n", res);
			/* TODO: err code propagation */
			abort();
		}
		VERBS_DBG(FI_LOG_EP_DATA,
			"reposted_recvs, posted %d, local_credits %d\n",
			to_post, conn->recv_preposted);
	}
}

static inline int 
fi_ibv_rdm_process_recv_wc(struct fi_ibv_rdm_ep *ep, struct ibv_wc *wc)
{
	struct fi_ibv_rdm_tagged_conn *conn = (void *) wc->wr_id;

	struct fi_ibv_rdm_buf *rbuf = 
		fi_ibv_rdm_get_rbuf(conn, ep, conn->recv_processed);

	FI_IBV_PREFETCH_ADDR(rbuf);

	FI_IBV_DBG_OPCODE(wc->opcode, "RECV");

	if (!FI_IBV_RDM_CHECK_RECV_WC(wc)) {
		assert(0 && "Error recv wc\n");
		return 1;
	}

	conn->recv_completions++;
	if (conn->recv_completions & ep->n_buffs) {
		conn->recv_completions = 0;
	}

	VERBS_DBG(FI_LOG_EP_DATA, "conn %p recv_completions %d\n",
		conn, conn->recv_completions);

	check_and_repost_receives(ep, conn);

	if (rbuf->service_data.status == BUF_STATUS_RECVED &&
	    fi_ibv_rdm_buffer_check_seq_num(rbuf, conn->recv_processed))
	{
		do {
			assert(rbuf->service_data.pkt_len > 0);

			fi_ibv_rdm_process_recv(ep, conn, 
				rbuf->service_data.pkt_len, rbuf);

			VERBS_DBG(FI_LOG_EP_DATA, "processed: conn %p, pkt # %d\n",
				conn, rbuf->service_data.seq_num);

			fi_ibv_rdm_set_buffer_status(rbuf, BUF_STATUS_FREE);
			rbuf->service_data.seq_num = (uint16_t)(-1);

			conn->recv_processed++;
			if (conn->recv_processed & ep->n_buffs) {
				conn->recv_processed = 0;
				fi_ibv_rdm_tagged_release_remote_sbuff(conn, ep);
			}
			VERBS_DBG(FI_LOG_EP_DATA, "conn %p recv_processed %d\n",
				conn, conn->recv_processed);

			rbuf = fi_ibv_rdm_get_rbuf(conn, ep, 
				conn->recv_processed);

		/* Do not process w/o completion! */
		} while (conn->recv_processed != conn->recv_completions &&
			 rbuf->service_data.status == BUF_STATUS_RECVED);
	} else {
		VERBS_DBG(FI_LOG_EP_DATA, "not processed: conn %p, status: %d\n",
			conn, rbuf->service_data.status);
	}

	return 0;
}

static inline int fi_ibv_rdm_tagged_poll_recv(struct fi_ibv_rdm_ep *ep)
{
	const int wc_count = ep->n_buffs; /* TODO: to set from upper level */
	struct ibv_wc wc[wc_count];
	int ret = 0;
	int err = 0;
	int i = 0;

	do {
		ret = ibv_poll_cq(ep->rcq, wc_count, wc);
		for (i = 0; i < ret && !err; ++i) {
			err = fi_ibv_rdm_process_recv_wc(ep, &wc[i]);
		}
	} while (!err && ret == wc_count);

	if (!err && ret >= 0) {
		return FI_SUCCESS;
	}

	/* error handling */

	VERBS_INFO(FI_LOG_EP_DATA, "ibv_poll_cq returned %d\n", ret);

	for(i = 0; i < wc_count; i++) {

		if (wc[i].status != IBV_WC_SUCCESS) {
			struct fi_ibv_rdm_tagged_conn *conn = 
				(void *)wc[i].wr_id;

			if (wc[i].status == IBV_WC_WR_FLUSH_ERR && conn &&
				conn->state != FI_VERBS_CONN_ESTABLISHED)
			{
				return FI_SUCCESS;
			}

			VERBS_INFO(FI_LOG_EP_DATA, "got ibv_wc[%d].status = %d:%s\n",
				i, wc[i].status, ibv_wc_status_str(wc[i].status));
			assert(0);
			return -FI_EOTHER;
		}

		if (wc[i].opcode != IBV_WC_RECV_RDMA_WITH_IMM &&
		    wc[i].opcode != IBV_WC_RECV)
		{
			VERBS_INFO(FI_LOG_EP_DATA, "got ibv_wc[%d].opcode = %d\n",
				i, wc[i].opcode);
		}
	}

	return -FI_EOTHER;
}

static inline int
fi_ibv_rdm_process_send_wc(struct fi_ibv_rdm_ep *ep, struct ibv_wc *wc)
{
	if (wc->status != IBV_WC_SUCCESS) {
		return 1;
	}

	if (FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(wc->wr_id)) {
		VERBS_DBG(FI_LOG_EP_DATA, "CQ COMPL: SEND -> 0x1\n");
		struct fi_ibv_rdm_tagged_conn *conn =
			(struct fi_ibv_rdm_tagged_conn *)
			FI_IBV_RDM_UNPACK_SERVICE_WR(wc->wr_id);
		FI_IBV_RDM_TAGGED_DEC_SEND_COUNTERS(conn, ep);

		return 0;
	} else {
		FI_IBV_DBG_OPCODE(wc->opcode, "SEND");
		struct fi_ibv_rdm_tagged_request *request =
			(struct fi_ibv_rdm_tagged_request *)
			FI_IBV_RDM_UNPACK_WR(wc->wr_id);

		struct fi_ibv_rdm_tagged_send_completed_data data = 
			{ .ep = ep };

		return fi_ibv_rdm_tagged_req_hndl(request, 
			FI_IBV_EVENT_POST_LC, &data);
	}
}

static inline int fi_ibv_rdm_tagged_poll_send(struct fi_ibv_rdm_ep *ep)
{
	const int wc_count = ep->n_buffs;
	struct ibv_wc wc[wc_count];
	int ret = 0;
	int err = 0;
	int i = 0;

	if (ep->posted_sends > 0) {
		do {
			ret = ibv_poll_cq(ep->scq, wc_count, wc);
			for (i = 0; i < ret && !err; ++i) {
				err = fi_ibv_rdm_process_send_wc(ep, &wc[i]);
			}
		} while (!err && ret == wc_count);
	}

	if (err || ret < 0) {
		goto wc_error;
	}

	struct fi_ibv_rdm_tagged_send_ready_data data = { .ep = ep };
	struct dlist_entry *item;
	dlist_foreach((&fi_ibv_rdm_postponed_queue), item) {
		if (fi_ibv_rdm_tagged_send_postponed_process(item, &data)) {
			/* we can't process all postponed items till foreach */
			/* implementation is not safety for removing during  */
			/* iterating                                         */
			break;
		}
	}

	return FI_SUCCESS;

wc_error:
	if (ret < 0) {
		VERBS_INFO(FI_LOG_EP_DATA, "ibv_poll_cq returned %d\n", ret);
		assert(0);
	}

	for (i = 0; i < wc_count; i++) {
		if (wc[i].status != IBV_WC_SUCCESS) {
			struct fi_ibv_rdm_tagged_conn *conn;
			if (FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(wc[i].wr_id)) {
				conn = FI_IBV_RDM_UNPACK_SERVICE_WR(wc[i].wr_id);
			} else {
				struct fi_ibv_rdm_tagged_request *req =
					(void *)wc[i].wr_id;
				conn = req->minfo.conn;
			}

			VERBS_INFO(FI_LOG_EP_DATA,
				"got ibv_wc.status = %d:%s, pend_send: %d, connection: %p\n",
				wc[i].status,
				ibv_wc_status_str(wc[i].status),
				ep->posted_sends, conn);
			assert(0);
		}
	}

	return -FI_EOTHER;
}

int fi_ibv_rdm_tagged_poll(struct fi_ibv_rdm_ep *ep)
{
	int ret = fi_ibv_rdm_tagged_poll_send(ep);
	if (ret) {
		return ret;
	}

	return fi_ibv_rdm_tagged_poll_recv(ep);
}
