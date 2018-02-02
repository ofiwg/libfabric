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

#include <inttypes.h>
#include <stdlib.h>

#include <ofi_list.h>
#include "../fi_verbs.h"
#include "verbs_rdm.h"
#include "verbs_queuing.h"
#include "verbs_tagged_ep_rdm_states.h"

typedef ssize_t (*fi_ep_rdm_request_handler_t)
	(struct fi_ibv_rdm_request *request, void *data);

static fi_ep_rdm_request_handler_t
	fi_ibv_rdm_req_hndl_arr [FI_IBV_STATE_EAGER_COUNT]
				[FI_IBV_STATE_RNDV_COUNT]
				[FI_IBV_EVENT_COUNT];

#if ENABLE_DEBUG

enum fi_ibv_rdm_hndl_req_log_state {
	hndl_req_log_state_in = 0,
	hndl_req_log_state_out = 10000
};

#define FI_IBV_RDM_HNDL_REQ_LOG_IN()						\
enum fi_ibv_rdm_hndl_req_log_state state = hndl_req_log_state_in;		\
do {										\
	FI_IBV_RDM_DBG_REQUEST("\t> IN\t< ", request, FI_LOG_DEBUG);		\
} while(0)

#define FI_IBV_RDM_HNDL_REQ_LOG() do {						\
	state++;								\
	char prefix[128];							\
	snprintf(prefix, 128, "\t> %d\t< ", state);				\
	FI_IBV_RDM_DBG_REQUEST(prefix, request, FI_LOG_DEBUG);			\
} while(0)

#define FI_IBV_RDM_HNDL_REQ_LOG_OUT() do {					\
	assert(state < hndl_req_log_state_out);					\
	FI_IBV_RDM_DBG_REQUEST("\t> OUT\t< ", request, FI_LOG_DEBUG);		\
} while(0)

#else // ENABLE_DEBUG
#define FI_IBV_RDM_HNDL_REQ_LOG_IN()
#define FI_IBV_RDM_HNDL_REQ_LOG()
#define FI_IBV_RDM_HNDL_REQ_LOG_OUT()
#endif // ENABLE_DEBUG

static ssize_t
fi_ibv_rdm_init_send_request(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();
	ssize_t ret;
	struct fi_ibv_rdm_send_start_data *p = data;
	request->minfo.conn = p->conn;
	request->minfo.tag = p->tag;
	request->minfo.is_tagged = p->is_tagged;
	request->iov_count = p->iov_count;

	/* Indeed, both branches are the same, just for readability */
	if (request->iov_count) {
		request->iovec_arr = p->buf.iovec_arr;
	} else {
		request->src_addr = p->buf.src_addr;
	}

	request->sbuf = NULL;
	request->len = p->data_len;
	request->comp_flags = p->flags;
	request->imm = p->imm;
	request->context = p->context;
	request->state.eager = FI_IBV_STATE_EAGER_BEGIN;
	request->state.rndv =
	    (p->data_len + sizeof(struct fi_ibv_rdm_header)
	     <= p->ep_rdm->rndv_threshold)
	    ? FI_IBV_STATE_RNDV_NOT_USED : FI_IBV_STATE_RNDV_SEND_BEGIN;

	FI_IBV_RDM_HNDL_REQ_LOG();

	ret = fi_ibv_rdm_move_to_postponed_queue(request);
	if (ret)
		return ret;
	request->state.eager = FI_IBV_STATE_EAGER_SEND_POSTPONED;
	if (request->state.rndv == FI_IBV_STATE_RNDV_SEND_BEGIN) {
		request->state.rndv = FI_IBV_STATE_RNDV_SEND_WAIT4SEND;
	}
		
	FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	return FI_SUCCESS;
}

static ssize_t
fi_ibv_rdm_eager_send_ready(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();

	assert(request->state.eager == FI_IBV_STATE_EAGER_SEND_POSTPONED);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED);

	fi_ibv_rdm_remove_from_postponed_queue(request);
	struct fi_ibv_rdm_tagged_send_ready_data *p = data;

	ssize_t ret = FI_SUCCESS;
	struct ibv_sge sge;

	struct fi_ibv_rdm_conn *conn = request->minfo.conn;
	const int size = request->len + sizeof(struct fi_ibv_rdm_header);

	assert(request->sbuf);

	struct ibv_send_wr wr = { 0 };
	struct ibv_send_wr *bad_wr = NULL;

	wr.wr_id = FI_IBV_RDM_PACK_WR(request);
	assert(FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(wr.wr_id) == 0);
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.wr.rdma.remote_addr = fi_ibv_rdm_get_remote_addr(conn, request->sbuf);
	wr.wr.rdma.rkey = conn->remote_rbuf_rkey;
	wr.send_flags = 0;

	sge.addr = (uintptr_t)request->sbuf;
	sge.length = size + FI_IBV_RDM_BUFF_SERVICE_DATA_SIZE;
	request->sbuf->service_data.pkt_len = size;

	if (sge.length <= p->ep->max_inline_rc) {
		wr.send_flags |= IBV_SEND_INLINE;
	}

	sge.lkey = fi_ibv_mr_internal_lkey(&conn->s_md);

	wr.imm_data = 0;
	wr.opcode = p->ep->eopcode;
	struct fi_ibv_rdm_buf *sbuf = (struct fi_ibv_rdm_buf *)request->sbuf;
	uint64_t *payload = &sbuf->payload;

	sbuf->header.service_tag = 0;
	if (request->minfo.is_tagged) {
		sbuf->header.tag = request->minfo.tag;
		FI_IBV_RDM_SET_PKTTYPE(sbuf->header.service_tag, FI_IBV_RDM_EAGER_PKT);
	} else {
		FI_IBV_RDM_SET_PKTTYPE(sbuf->header.service_tag, FI_IBV_RDM_MSG_PKT);
	}

	if (request->len > 0) {
		if (request->iov_count == 0) {
			memcpy(payload, request->src_addr, request->len);
		} else {
			size_t i;
			for (i = 0; i < request->iov_count; i++) {
				memcpy(payload, request->iovec_arr[i].iov_base,
					request->iovec_arr[i].iov_len);
				payload += request->iovec_arr[i].iov_len;
			}
		}
	}

	FI_IBV_RDM_INC_SIG_POST_COUNTERS(request->minfo.conn, p->ep);
	VERBS_DBG(FI_LOG_EP_DATA, "posted %d bytes, conn %p, tag 0x%" PRIx64 "\n",
		  sge.length, request->minfo.conn, request->minfo.tag);

	ret = ibv_post_send(conn->qp[0], &wr, &bad_wr);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_EP_DATA, "ibv_post_send", errno);
		ret = -errno;
		assert(0);
	};

	fi_ibv_rdm_cntr_inc(p->ep->send_cntr);

	if (request->comp_flags & FI_COMPLETION) {
		fi_ibv_rdm_move_to_cq(p->ep->fi_scq, request);
		request->state.eager = FI_IBV_STATE_EAGER_SEND_WAIT4LC;
	} else {
		request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;
	}

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();

	return ret;
}

static ssize_t
fi_ibv_rdm_eager_send_lc(struct fi_ibv_rdm_request *request,
				void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();

	assert(request->state.eager == FI_IBV_STATE_EAGER_SEND_WAIT4LC ||
	       request->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED);

	VERBS_DBG(FI_LOG_EP_DATA, "conn %p, tag 0x%" PRIx64 ", len %" PRIu64 "\n",
		  request->minfo.conn, request->minfo.tag, request->len);

	struct fi_ibv_rdm_tagged_send_completed_data *p = data;
	FI_IBV_RDM_DEC_SIG_POST_COUNTERS(request->minfo.conn, p->ep);

	if (request->iov_count) {
		util_buf_release(
			request->ep->fi_ibv_rdm_extra_buffers_pool,
			request->iovec_arr);
	}

	if (request->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE) {
		FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
		util_buf_release(request->ep->fi_ibv_rdm_request_pool,
				 request);
	} else {
		request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;
	}

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	return FI_SUCCESS;
}

static ssize_t
fi_ibv_rdm_rndv_rts_send_ready(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();

	assert(request->state.eager == FI_IBV_STATE_EAGER_SEND_POSTPONED);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_SEND_WAIT4SEND);
	assert(request->sbuf);

	VERBS_DBG(FI_LOG_EP_DATA, "conn %p, tag 0x%" PRIx64 ", len %" PRIu64 "\n",
		  request->minfo.conn, request->minfo.tag, request->len);

	fi_ibv_rdm_remove_from_postponed_queue(request);
	struct fi_ibv_rdm_tagged_send_ready_data *p = data;
	struct ibv_sge sge;
	struct fi_ibv_rdm_conn *conn = request->minfo.conn;
	struct fi_ibv_rdm_rndv_header *header = (void *)&request->sbuf->header;
	struct ibv_send_wr wr = { 0 }, *bad_wr = NULL;
	int ret;

	wr.wr_id = FI_IBV_RDM_PACK_WR(request);
	assert(FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(wr.wr_id) == 0);
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.wr.rdma.remote_addr = (uintptr_t)
		fi_ibv_rdm_get_remote_addr(conn, request->sbuf);
	wr.wr.rdma.rkey = conn->remote_rbuf_rkey;
	wr.send_flags = 0;
	wr.opcode = p->ep->eopcode;
	wr.imm_data = 0;

	sge.addr = (uintptr_t)request->sbuf;
	sge.length = FI_IBV_RDM_BUFF_SERVICE_DATA_SIZE + sizeof(*header);
	sge.lkey = fi_ibv_mr_internal_lkey(&conn->s_md);
	request->sbuf->service_data.pkt_len = sizeof(*header);

	if (request->minfo.is_tagged) {
		header->base.tag = request->minfo.tag;
		header->is_tagged = 1;
	} else {
		header->is_tagged = 0;
	}
	header->base.service_tag = 0;
	header->total_len = request->len;
	header->src_addr = (uintptr_t)request->src_addr;

	header->id = (uintptr_t)request;
	request->rndv.id = (uintptr_t)request;

	ret = p->ep->domain->internal_mr_reg(p->ep->domain,
					     (void *)request->src_addr,
					     request->len,
					     FI_REMOTE_READ,
					     &request->rndv.md);
	if (ret) {
		VERBS_WARN(FI_LOG_EP_DATA,
			   "Unable to register MR, ret = %d", ret);
		assert(0);
		return ret;
	}
	header->mem_rkey = fi_ibv_mr_internal_rkey(&request->rndv.md);

	FI_IBV_RDM_SET_PKTTYPE(header->base.service_tag,
			       FI_IBV_RDM_RNDV_RTS_PKT);

	VERBS_DBG(FI_LOG_EP_DATA,
		  "fi_senddatato: RNDV conn %p, tag 0x%" PRIx64 ", len %"PRIu64", "
		  "src_addr %p, rkey 0x%"PRIx64", fi_ctx %p, imm %d, post_sends %"PRIu32"\n",
		  conn, request->minfo.tag, request->len, request->src_addr,
		  header->mem_rkey, request->context, (int)wr.imm_data,
		  p->ep->posted_sends);

	FI_IBV_RDM_INC_SIG_POST_COUNTERS(request->minfo.conn, p->ep);
	VERBS_DBG(FI_LOG_EP_DATA, "posted %d bytes, conn %p, tag 0x%" PRIx64 "\n",
		  sge.length, request->minfo.conn,
		  request->minfo.tag);
	ret = ibv_post_send(conn->qp[0], &wr, &bad_wr);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_EP_DATA, "ibv_post_send", errno);
		assert(0);
		return -errno;
	};

	request->state.eager = FI_IBV_STATE_EAGER_SEND_WAIT4LC;
	request->state.rndv = FI_IBV_STATE_RNDV_SEND_WAIT4ACK;

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	return FI_SUCCESS;
}

static ssize_t
fi_ibv_rdm_rndv_rts_lc(struct fi_ibv_rdm_request *request,
			      void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();

	assert(((request->state.eager == FI_IBV_STATE_EAGER_SEND_WAIT4LC) &&
		(request->state.rndv == FI_IBV_STATE_RNDV_SEND_WAIT4ACK)) ||
	       ((request->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE) &&
		(request->state.rndv == FI_IBV_STATE_RNDV_SEND_END)) ||
	       ((request->state.eager == FI_IBV_STATE_EAGER_SEND_WAIT4LC) &&
		(request->state.rndv == FI_IBV_STATE_RNDV_SEND_END)));
	assert(request->minfo.conn);

	VERBS_DBG(FI_LOG_EP_DATA, "conn %p, tag 0x%" PRIx64 ", len %" PRIu64 "\n",
		  request->minfo.conn, request->minfo.tag,
		  request->len);

	struct fi_ibv_rdm_tagged_send_completed_data *p = data;

	FI_IBV_RDM_DEC_SIG_POST_COUNTERS(request->minfo.conn, p->ep);

	if (request->state.eager == FI_IBV_STATE_EAGER_SEND_WAIT4LC) {
		request->state.eager = FI_IBV_STATE_EAGER_SEND_END;
	} else { /* (request->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE) */
		FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
		util_buf_release(request->ep->fi_ibv_rdm_request_pool,
				 request);
	}

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();

	return FI_SUCCESS;
}

static ssize_t
fi_ibv_rdm_rndv_end(struct fi_ibv_rdm_request *request,
			   void *data)
{
	struct fi_ibv_recv_got_pkt_preprocess_data *p = data;
	int ret;

	FI_IBV_RDM_HNDL_REQ_LOG_IN();

	assert(request->state.eager == FI_IBV_STATE_EAGER_SEND_END ||
	       (request->state.eager == FI_IBV_STATE_EAGER_SEND_WAIT4LC));
	assert(request->state.rndv == FI_IBV_STATE_RNDV_SEND_WAIT4ACK);

	assert((sizeof(struct fi_ibv_rdm_request *) +
		sizeof(struct fi_ibv_rdm_header)) == p->arrived_len);
	assert(request->rndv.md.mr);
	assert(p->rbuf);

	ret = p->ep->domain->internal_mr_dereg(&request->rndv.md);
	if (ret)
		VERBS_INFO(FI_LOG_EP_DATA,
			   "Unable to deregister MR, ret = %d", ret);

	if (request->state.eager == FI_IBV_STATE_EAGER_SEND_END)
		request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;

	request->state.rndv = FI_IBV_STATE_RNDV_SEND_END;
	FI_IBV_RDM_HNDL_REQ_LOG();

	fi_ibv_rdm_cntr_inc(p->ep->send_cntr);

	if (request->comp_flags & FI_COMPLETION) {
		fi_ibv_rdm_move_to_cq(p->ep->fi_scq, request);
	} else if (request->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE) {
		FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
		util_buf_release(
			request->ep->fi_ibv_rdm_request_pool,
			request);
	}

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();

	return FI_SUCCESS;
}


static inline ssize_t
fi_ibv_rdm_copy_unexp_request(struct fi_ibv_rdm_request *request,
			      struct fi_ibv_rdm_request *unexp)
{
	ssize_t ret = FI_SUCCESS;
	if (request->len && (request->len < unexp->len)) {
		VERBS_INFO(FI_LOG_EP_DATA,
			   "RECV TRUNCATE, unexp len %" PRIu64 ", "
			   "req->len=%" PRIu64 ", conn %p, tag 0x%" PRIx64 ", "
			   "tagmask %" PRIx64 "\n",
			   unexp->len, request->len, request->minfo.conn,
			   request->minfo.tag, request->minfo.tagmask);

		util_buf_release(
			unexp->ep->fi_ibv_rdm_extra_buffers_pool,
			unexp->unexp_rbuf);
		ret = -FI_ETRUNC;
		return ret;
	}

	request->minfo.conn = unexp->minfo.conn;
	request->minfo.tag = unexp->minfo.tag;
	request->minfo.is_tagged = unexp->minfo.is_tagged;
	request->len = unexp->len;
	request->rest_len = unexp->rest_len;
	request->unexp_rbuf = unexp->unexp_rbuf;
	request->state = unexp->state;

	assert((request->state.eager == FI_IBV_STATE_EAGER_RECV_WAIT4RECV) ||
	       (request->state.eager == FI_IBV_STATE_EAGER_RECV_CLAIMED));

	VERBS_DBG(FI_LOG_EP_DATA, "found req: len = %" PRIu64 ", eager_state = %s, rndv_state = %s \n",
		  unexp->len,
		  fi_ibv_rdm_req_eager_state_to_str(unexp->state.eager),
		  fi_ibv_rdm_req_rndv_state_to_str(unexp->state.rndv));

	if (request->state.rndv != FI_IBV_STATE_RNDV_NOT_USED) {
		assert(request->state.rndv == FI_IBV_STATE_RNDV_RECV_WAIT4RES);

		request->rndv.mr_rkey = unexp->rndv.mr_rkey;
		request->rndv.id = unexp->rndv.id;
		request->rndv.remote_addr = unexp->rndv.remote_addr;
	}

	return ret;
}

/*
 * FI_MULTI_RECV path is implemented through a parent multi_request.
 * This is a kind of supervisor which keep another preposted request while multi
 * recv buffer has enough space for incoming data. The last prepost releases
 * the parent and sets FI_MULTI_RECV completion flag.
 */
static struct fi_ibv_rdm_request *
fi_ibv_rdm_repost_multi_recv(struct fi_ibv_rdm_request *request,
			     size_t offset, struct fi_ibv_rdm_ep *ep)
{
	struct fi_ibv_rdm_multi_request *parent;
	struct fi_ibv_rdm_request *prepost;

	if (!(prepost = util_buf_alloc(ep->fi_ibv_rdm_request_pool))) {
		VERBS_WARN(FI_LOG_EP_DATA, "Unable to allocate memory for "
			   "multi recv prepost request\n");
		return NULL;
	}

	fi_ibv_rdm_zero_request(prepost);
	prepost->ep = ep;
	FI_IBV_RDM_DBG_REQUEST("get_from_pool: ", prepost, FI_LOG_DEBUG);
	FI_IBV_RDM_DBG_REQUEST("repost from: ", request, FI_LOG_DEBUG);

	parent = request->parent;
	request->parent = NULL;
	parent->prepost = prepost;
	parent->offset += offset;

	VERBS_DBG(FI_LOG_EP_DATA,
		  "multi_recv parent: prepost %p, buf %p, len %" PRIu64
		  ", offset %" PRIu64 " min_size %" PRIu64 "\n",
		  parent->prepost, parent->buf, parent->len,
		  parent->offset, parent->min_size);

	prepost->parent = parent;
	prepost->minfo = request->minfo;
	prepost->dest_buf = parent->buf + parent->offset;

	prepost->comp_flags = request->comp_flags;
	prepost->len = parent->len - parent->offset;
	if (prepost->len < parent->min_size) {
		/* This is the last one, parent can be released */
		prepost->comp_flags |= FI_MULTI_RECV;
		util_buf_release(ep->fi_ibv_rdm_multi_request_pool, prepost->parent);
		fi_ibv_rdm_remove_from_multi_recv_list(prepost->parent, ep);
		prepost->parent = NULL;
		FI_IBV_RDM_DBG_REQUEST("get_from_pool: ", prepost, FI_LOG_DEBUG);
	}
	prepost->context = request->context;
	prepost->context->internal[0] = (void *)prepost;

	/* TODO: way for (RNDV) optimization is do registration only once */
	//prepost->rndv = request->rndv;

	prepost->state.eager = FI_IBV_STATE_EAGER_RECV_WAIT4PKT;
	prepost->state.rndv = FI_IBV_STATE_RNDV_NOT_USED;
	prepost->state.err = FI_SUCCESS;
	fi_ibv_rdm_move_to_posted_queue(prepost, ep);
	return prepost;
}

static inline ssize_t
fi_ibv_rdm_try_unexp_recv(struct fi_ibv_rdm_request *request,
			  struct fi_ibv_rdm_tagged_recv_start_data *rdata)
{
	struct dlist_entry *found_entry = NULL;
	struct fi_ibv_rdm_request *found_request = NULL;
	struct fi_ibv_rdm_request *repost = NULL;
	ssize_t ret = FI_ENOMSG;

	do {
		found_entry =
			dlist_find_first_match(&rdata->ep->fi_ibv_rdm_unexp_queue,
					       fi_ibv_rdm_req_match_by_info3,
					       &rdata->peek_data);
		if (found_entry) {
			ret = FI_SUCCESS;
			found_request =
				container_of(found_entry,
					     struct fi_ibv_rdm_request,
					     queue_entry);

			fi_ibv_rdm_remove_from_unexp_queue(found_request);

			if (request->parent) {
				repost = fi_ibv_rdm_repost_multi_recv(request,
					 found_request->len, rdata->ep);
				if (!repost) {
					ret = -FI_ENOMEM;
					break;
				}
			}

			ret = fi_ibv_rdm_copy_unexp_request(request, found_request);

			assert((ret != FI_SUCCESS) ||
				((rdata->peek_data.flags & FI_CLAIM) &&
					(request->state.eager == 
						FI_IBV_STATE_EAGER_RECV_CLAIMED) &&
					(request->context == found_request->context)) ||
			       (!(rdata->peek_data.flags & FI_CLAIM) && 
					(request->state.eager == 
						FI_IBV_STATE_EAGER_RECV_WAIT4RECV)));

			FI_IBV_RDM_DBG_REQUEST("to_pool: ", found_request, FI_LOG_DEBUG);
			util_buf_release(
				found_request->ep->fi_ibv_rdm_request_pool,
				found_request);

			if (ret == FI_SUCCESS &&
			    request->state.rndv == FI_IBV_STATE_RNDV_RECV_WAIT4RES) {
				request->state.eager = FI_IBV_STATE_EAGER_RECV_END;
				ret = fi_ibv_rdm_move_to_postponed_queue(request);
				/* Will fail `while` check and return the result */
			}
		}
	/* 
	* Unexpected queue may contain several entries
	* in case of multi recv, we need to handle them all
	*/
	} while (repost && repost->parent && found_entry && !ret);

	return ret;
}

static ssize_t
fi_ibv_rdm_init_recv_request(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();

	ssize_t ret = FI_SUCCESS;
	struct fi_ibv_rdm_tagged_recv_start_data *p = data;

	if (p->peek_data.flags & FI_MULTI_RECV) {
		request->parent =
			util_buf_alloc(request->ep->fi_ibv_rdm_multi_request_pool);
		if (!request->parent) {
			VERBS_WARN(FI_LOG_EP_DATA, "Unable to allocate memory "
				   "for parent \n");
			return -FI_ENOMEM;
		}
		fi_ibv_rdm_add_to_multi_recv_list(request->parent, request->ep);
		request->parent->prepost = request;
		request->parent->buf = p->dest_addr;
		request->parent->len = p->data_len;
		request->parent->offset = 0;
		request->parent->min_size = p->ep->min_multi_recv_size;
		VERBS_DBG(FI_LOG_EP_DATA, "conn %p, multi_recv %p, parent %p\n",
			request->minfo.conn, request, request->parent);
	}

	request->minfo = p->peek_data.minfo;
	request->context = p->peek_data.context;
	request->context->internal[0] = (void *)request;
	request->dest_buf = p->dest_addr;
	request->len = p->data_len;
	request->comp_flags =
		(p->peek_data.minfo.is_tagged ? FI_TAGGED : FI_MSG ) | FI_RECV |
		(p->peek_data.flags & FI_COMPLETION);
	request->state.eager = FI_IBV_STATE_EAGER_RECV_WAIT4PKT;
	request->state.rndv = FI_IBV_STATE_RNDV_NOT_USED;
	request->state.err = FI_SUCCESS;

	VERBS_DBG(FI_LOG_EP_DATA, "conn %p, tag 0x%" PRIx64 ", len %" PRIu64 "\n",
		  request->minfo.conn, request->minfo.tag, request->len);

	ret = fi_ibv_rdm_try_unexp_recv(request, p);
	if (ret == FI_ENOMSG) {
		fi_ibv_rdm_move_to_posted_queue(request, p->ep);
		ret = FI_SUCCESS;
	} else if (ret != FI_SUCCESS) {
		request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;

		fi_ibv_rdm_cntr_inc_err(p->ep->recv_cntr);

		if (request->comp_flags & FI_COMPLETION) {
			fi_ibv_rdm_move_to_errcq(p->ep->fi_rcq, request, ret);
		}
		ret = FI_SUCCESS;
	}

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	return ret;
}

static ssize_t
fi_ibv_rdm_tagged_peek_request(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();
	assert(request->state.eager == FI_IBV_STATE_EAGER_BEGIN);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED);

	ssize_t ret = FI_SUCCESS;

	struct fi_ibv_rdm_tagged_recv_start_data *p = data;
	struct fi_ibv_rdm_tagged_peek_data *peek_data = &p->peek_data;
	struct dlist_entry *found_entry =
		dlist_find_first_match(&p->ep->fi_ibv_rdm_unexp_queue,
				       fi_ibv_rdm_req_match_by_info2,
				       &peek_data->minfo);

	/* TODO: to check behaviour for multi recv */
	assert(!(peek_data->flags & FI_MULTI_RECV));

	request->context = peek_data->context;
	request->comp_flags = peek_data->flags;

	if (found_entry) {
		struct fi_ibv_rdm_request *found_request =
			container_of(found_entry, struct fi_ibv_rdm_request,
				     queue_entry);
		assert(found_request);

		ret = fi_ibv_rdm_copy_unexp_request(request, found_request);

		if (ret) {
			goto err;
		}

		if (peek_data->flags & FI_CLAIM) {
			ret = fi_ibv_rdm_req_hndl(found_request,
						  FI_IBV_EVENT_RECV_CLAIM,
						  peek_data);

			if (ret) {
				goto err;
			}
		}

		if (peek_data->flags & FI_DISCARD) {
			ret = fi_ibv_rdm_req_hndl(found_request,
						  FI_IBV_EVENT_RECV_DISCARD,
						  NULL);

			if (ret) {
				goto err;
			}
		}

		fi_ibv_rdm_cntr_inc(p->ep->recv_cntr);

		if (request->comp_flags & FI_COMPLETION) {
			request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;
			fi_ibv_rdm_move_to_cq(p->ep->fi_rcq, request);
		} else {
			FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
			util_buf_release(
				request->ep->fi_ibv_rdm_request_pool,
				request);
		}
		
		FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	} else {
		ret = -FI_ENOMSG;
		goto err;
	}

out:
	FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	return ret;
err:
	request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;

	fi_ibv_rdm_cntr_inc_err(p->ep->recv_cntr);

	if (request->comp_flags & FI_COMPLETION) {
		fi_ibv_rdm_move_to_errcq(p->ep->fi_rcq, request, ret);
	}

	ret = FI_SUCCESS;
	goto out;
}

static ssize_t
fi_ibv_rdm_init_unexp_recv_request(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();

	assert(request->state.eager == FI_IBV_STATE_EAGER_BEGIN);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED);

	struct fi_ibv_recv_got_pkt_preprocess_data *p = data;
	struct fi_ibv_rdm_buf *rbuf = p->rbuf;
	ssize_t ret = FI_SUCCESS;

	FI_IBV_RDM_HNDL_REQ_LOG();

	switch (p->pkt_type) {
	case FI_IBV_RDM_EAGER_PKT:
	case FI_IBV_RDM_MSG_PKT:
		FI_IBV_RDM_HNDL_REQ_LOG();

		request->minfo.conn = p->conn;
		request->minfo.tag = rbuf->header.tag;
		request->minfo.is_tagged =
			((p->pkt_type == FI_IBV_RDM_EAGER_PKT) ? 1 : 0);
		request->len = 
			p->arrived_len - sizeof(struct fi_ibv_rdm_header);
		request->comp_flags =
			(request->minfo.is_tagged ? FI_TAGGED :
						    FI_MSG) | FI_RECV;
		
		assert(request->len <= p->ep->rndv_threshold);

		if (request->len > 0) {
			request->unexp_rbuf =
				util_buf_alloc(request->ep->fi_ibv_rdm_extra_buffers_pool);
			if (!request->unexp_rbuf) {
				ret = -FI_ENOMEM;
				VERBS_WARN(FI_LOG_EP_DATA,
					   "Unable allocate memory from the pool "
					   "for uenxpected buffer");
				goto fn;
			}
			memcpy(request->unexp_rbuf, &rbuf->payload,
			       request->len);
		} else {
			request->unexp_rbuf = NULL;
		}
		request->imm = p->imm_data;
		request->state.eager = FI_IBV_STATE_EAGER_RECV_WAIT4RECV;
		break;
	case FI_IBV_RDM_RNDV_RTS_PKT:
		FI_IBV_RDM_HNDL_REQ_LOG();
		assert(p->arrived_len == sizeof(struct fi_ibv_rdm_rndv_header));
		struct fi_ibv_rdm_rndv_header *h = (void *)&rbuf->header;

		request->minfo.conn = p->conn;
		request->minfo.tag = h->base.tag;
		request->minfo.is_tagged = h->is_tagged;
		request->rndv.id = (uintptr_t)h->id;
		request->rndv.remote_addr = (void *)h->src_addr;
		request->rndv.mr_rkey = h->mem_rkey;
		request->len = h->total_len;
		request->rest_len = h->total_len;
		request->comp_flags = (h->is_tagged ? FI_TAGGED :
						      FI_MSG) | FI_RECV;
		request->imm = p->imm_data;
		request->state.eager = FI_IBV_STATE_EAGER_RECV_WAIT4RECV;
		request->state.rndv = FI_IBV_STATE_RNDV_RECV_WAIT4RES;
		break;
	default:
		if (p->pkt_type == FI_IBV_RDM_RNDV_ACK_PKT) {
			FI_IBV_RDM_DBG_REQUEST("Unexpected RNDV ack!!!",
					       request, FI_LOG_INFO);
		}

		VERBS_INFO(FI_LOG_EP_DATA,
			"Got unknown unexpected pkt: %" PRIu64 "\n",
			p->pkt_type);
		assert(0);
		ret = -FI_EOTHER;
	}

	fi_ibv_rdm_move_to_unexpected_queue(request, p->ep);
fn:
	FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	return ret;
}

static ssize_t
fi_ibv_rdm_eager_recv_got_pkt(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();
	struct fi_ibv_recv_got_pkt_preprocess_data *p = data;
	struct fi_ibv_rdm_buf *rbuf = p->rbuf;
	ssize_t ret;
	assert(request->state.eager == FI_IBV_STATE_EAGER_RECV_WAIT4PKT);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED);

	switch (p->pkt_type) {
	case FI_IBV_RDM_EAGER_PKT:
	case FI_IBV_RDM_MSG_PKT:
	{
		const size_t data_len = p->arrived_len - sizeof(rbuf->header);
		assert(data_len <= p->ep->rndv_threshold);

		if (request->parent) {
			if (!fi_ibv_rdm_repost_multi_recv(request, data_len, p->ep))
				return -FI_ENOMEM;
			
		}

		if (request->len >= data_len) {
			request->minfo.conn = p->conn;
			request->minfo.tag = rbuf->header.tag;
			request->minfo.is_tagged =
				((p->pkt_type == FI_IBV_RDM_EAGER_PKT) ? 1 : 0);

			request->len = data_len;
			request->exp_rbuf = &rbuf->payload;
			request->imm = p->imm_data;

			if (request->dest_buf) {
				assert(request->exp_rbuf);
				memcpy(request->dest_buf,
					request->exp_rbuf, request->len);
			}

			if (request->parent) {
				if (!fi_ibv_rdm_repost_multi_recv(request, data_len,
							     p->ep))
					return -FI_ENOMEM;
			}

			fi_ibv_rdm_cntr_inc(p->ep->recv_cntr);

			if (request->comp_flags & FI_COMPLETION) {
				request->state.eager = 
					FI_IBV_STATE_EAGER_READY_TO_FREE;
				fi_ibv_rdm_move_to_cq(p->ep->fi_rcq, request);
			} else {
				FI_IBV_RDM_DBG_REQUEST("to_pool: ", request,
							FI_LOG_DEBUG);
				util_buf_release(
					request->ep->fi_ibv_rdm_request_pool,
					request);
			}
		} else {
			VERBS_INFO(FI_LOG_EP_DATA,
				   "%s: %d RECV TRUNCATE, data_len=%zu, "
				   "posted_len=%" PRIu64 ", conn %p, tag 0x%" PRIx64 ", "
				   "tagmask %" PRIx64 "\n",
				   __FUNCTION__, __LINE__, data_len,
				   request->len, request->minfo.conn,
				   request->minfo.tag, request->minfo.tagmask);

			if (request->parent) {
				if (!fi_ibv_rdm_repost_multi_recv(request, data_len,
							     p->ep))
					return -FI_ENOMEM;
			}

			request->state.eager =
				FI_IBV_STATE_EAGER_READY_TO_FREE;

			fi_ibv_rdm_cntr_inc_err(p->ep->recv_cntr);

			if (request->comp_flags & FI_COMPLETION) {
				fi_ibv_rdm_move_to_errcq(p->ep->fi_rcq, request,
							 FI_ETRUNC);
			}
		}

		FI_IBV_RDM_HNDL_REQ_LOG();
		break;
	}
	case FI_IBV_RDM_RNDV_RTS_PKT:
	{
		struct fi_ibv_rdm_rndv_header *rndv_header = 
			(void *)&rbuf->header;

		assert(p->arrived_len == sizeof(*rndv_header));

		if (request->len < rndv_header->total_len) {
			/* rndv protocol finalization requires memory
			 * deregistration, entry in errcq will be generated
			 * after acknowledgement in normal flow */
			request->state.err = FI_ETRUNC;
		}

		request->minfo.conn = p->conn;
		request->minfo.tag = rndv_header->base.tag;
		request->minfo.is_tagged = rndv_header->is_tagged;
		request->rndv.remote_addr = (void *)rndv_header->src_addr;
		request->rndv.mr_rkey = rndv_header->mem_rkey;
		request->len = rndv_header->total_len;
		request->rest_len = rndv_header->total_len;
		request->imm = p->imm_data;
		request->rndv.id = rndv_header->id;

		if (request->parent) {
			if (!fi_ibv_rdm_repost_multi_recv(request,
						     rndv_header->total_len,
						     p->ep))
				return -FI_ENOMEM;
		}

		ret = fi_ibv_rdm_move_to_postponed_queue(request);
		if (ret)
			return ret;

		request->state.eager = FI_IBV_STATE_EAGER_RECV_END;
		request->state.rndv = FI_IBV_STATE_RNDV_RECV_WAIT4RES;
		FI_IBV_RDM_HNDL_REQ_LOG();
		break;
	}
	default:
		VERBS_INFO(FI_LOG_EP_DATA,
			"Got unknown unexpected pkt: %" PRIu64 "\n",
			p->pkt_type);
		assert(0);
		FI_IBV_RDM_HNDL_REQ_LOG_OUT();
		return -FI_EOTHER;
	}

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	return FI_SUCCESS;
}

static ssize_t
fi_ibv_rdm_eager_recv_process_unexp_pkt(struct fi_ibv_rdm_request *request,
					void *data)
{
	struct fi_ibv_recv_got_pkt_process_data *p = data;

	FI_IBV_RDM_HNDL_REQ_LOG_IN();

	assert((request->state.eager == FI_IBV_STATE_EAGER_RECV_WAIT4RECV) ||
	       (request->state.eager == FI_IBV_STATE_EAGER_RECV_CLAIMED));
	assert(request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED);

	if (request->dest_buf && request->len != 0) {
		memcpy(request->dest_buf, request->unexp_rbuf, request->len);
	}

	if (request->unexp_rbuf) {
		util_buf_release(
			request->ep->fi_ibv_rdm_extra_buffers_pool,
			request->unexp_rbuf);
		request->unexp_rbuf = NULL;
	}

	fi_ibv_rdm_cntr_inc(p->ep->recv_cntr);

	if (request->comp_flags & FI_COMPLETION) {
		request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;
		fi_ibv_rdm_move_to_cq(p->ep->fi_rcq, request);
	} else {
		FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
		util_buf_release(request->ep->fi_ibv_rdm_request_pool,
				 request);
	}

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	return FI_SUCCESS;
}

static ssize_t
fi_ibv_rdm_tagged_recv_claim(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();
	assert(request->state.eager == FI_IBV_STATE_EAGER_RECV_WAIT4RECV);
	assert((request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED) ||
	       (request->state.rndv == FI_IBV_STATE_RNDV_RECV_WAIT4RES));

	ssize_t ret = FI_SUCCESS;
	struct fi_ibv_rdm_tagged_peek_data *peek_data = data;
	assert(peek_data->context);

	request->state.eager = FI_IBV_STATE_EAGER_RECV_CLAIMED;
	request->context = peek_data->context;

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	return ret;
}

static ssize_t
fi_ibv_rdm_eager_recv_discard(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();
	assert(request->state.eager == FI_IBV_STATE_EAGER_RECV_WAIT4RECV);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED);
	assert(data == NULL);

	fi_ibv_rdm_remove_from_unexp_queue(request);

	if (request->unexp_rbuf) {
		util_buf_release(
			request->ep->fi_ibv_rdm_extra_buffers_pool,
			request->unexp_rbuf);
		request->unexp_rbuf = NULL;
	}

	FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
	util_buf_release(request->ep->fi_ibv_rdm_request_pool,
			 request);

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	return FI_SUCCESS;
}

static inline ssize_t
fi_ibv_rdm_rndv_read_reg_mr(struct fi_ibv_rdm_ep *ep,
			    struct fi_ibv_rdm_request *request)
{
	return ep->domain->internal_mr_reg(ep->domain,
					   (void *)request->src_addr,
					   request->len,
					   FI_REMOTE_READ,
					   &request->rndv.md);
}

static ssize_t
fi_ibv_rdm_rndv_recv_post_read(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();
	assert(request->state.eager == FI_IBV_STATE_EAGER_RECV_END);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_RECV_WAIT4RES);

	struct fi_ibv_rdm_tagged_send_ready_data *p = data;
	const size_t offset = request->len - request->rest_len;
	const size_t seg_cursize =
		MIN(p->ep->rndv_seg_size, request->rest_len);
	struct ibv_send_wr wr = { 0 };
	struct ibv_send_wr *bad_wr = NULL;
	struct ibv_sge sge;
	ssize_t ret = FI_SUCCESS;

	fi_ibv_rdm_remove_from_postponed_queue(request);
	VERBS_DBG(FI_LOG_EP_DATA,
		  "\t REQUEST: conn %p, tag 0x%" PRIx64
		  ", len %" PRIu64 ", rest %" PRIu64
		  ", dest_buf %p, src_addr %p, rkey 0x%"PRIx64"\n",
		  request->minfo.conn, request->minfo.tag, request->len,
		  request->rest_len, request->dest_buf,
		  request->rndv.remote_addr, request->rndv.mr_rkey);

	assert((request->minfo.conn->cm_role != FI_VERBS_CM_SELF) ||
	       (request->rndv.remote_addr != request->dest_buf));

	/* First segment */
	if (offset == 0) {
		ret = fi_ibv_rdm_rndv_read_reg_mr(p->ep, request);
		if (ret) {
			return ret;
		}
		request->post_counter = 0;
	}

	wr.wr_id = FI_IBV_RDM_PACK_WR(request);
	assert(FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(wr.wr_id) == 0);
	wr.opcode = IBV_WR_RDMA_READ;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = 0;
	wr.wr.rdma.remote_addr = (uintptr_t)
		((char *)request->rndv.remote_addr + offset);
	wr.wr.rdma.rkey = request->rndv.mr_rkey;

	sge.addr = (uintptr_t)((char *)request->dest_buf + offset);
	sge.length = (request->state.err == FI_SUCCESS ? seg_cursize : 0);
	sge.lkey = fi_ibv_mr_internal_lkey(&request->rndv.md);

	request->rest_len -= seg_cursize;
	request->post_counter++;
	FI_IBV_RDM_INC_SIG_POST_COUNTERS(request->minfo.conn, p->ep);
	VERBS_DBG(FI_LOG_EP_DATA, "posted %d bytes, conn %p, tag 0x%" PRIx64 "\n",
		  sge.length, request->minfo.conn, request->minfo.tag);
	ret = ibv_post_send(request->minfo.conn->qp[0], &wr, &bad_wr);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_EP_DATA, "ibv_post_send", errno);
		assert(0);
		return -errno;
	};

	if (request->rest_len && request->state.err == FI_SUCCESS) {
		/* Move to postponed queue for the next iteration */
		ret = fi_ibv_rdm_move_to_postponed_queue(request);
	} else {
		request->state.eager = FI_IBV_STATE_EAGER_RECV_END;
		request->state.rndv = FI_IBV_STATE_RNDV_RECV_WAIT4LC;
	}

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	return ret;
}

static ssize_t
fi_ibv_rdm_rndv_recv_read_lc(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();

	struct fi_ibv_rdm_tagged_send_completed_data *p = data;
	struct fi_ibv_rdm_conn *conn = request->minfo.conn;
	struct fi_ibv_rdm_buf *sbuf = request->sbuf;
	const int ack_size = 
		sizeof(struct fi_ibv_rdm_header) + sizeof(request->rndv.id);
	struct ibv_sge sge = {
		.addr = (uintptr_t)sbuf,
		.length = ack_size + FI_IBV_RDM_BUFF_SERVICE_DATA_SIZE,
		.lkey = fi_ibv_mr_internal_lkey(&conn->s_md),
	};
	struct ibv_send_wr wr = {
		.wr_id = FI_IBV_RDM_PACK_WR(request),
		.opcode = p->ep->eopcode,
		.sg_list = &sge,
		.num_sge = 1,
		.wr.rdma.remote_addr = 
			(uintptr_t)fi_ibv_rdm_get_remote_addr(conn,
							      request->sbuf),
		.wr.rdma.rkey = conn->remote_rbuf_rkey,
		.send_flags = (sge.length < p->ep->max_inline_rc) ?
			       IBV_SEND_INLINE : 0,
	};
	struct ibv_send_wr *bad_wr = NULL;
	ssize_t ret = FI_SUCCESS;

	assert(request->len > (p->ep->rndv_threshold
			       - sizeof(struct fi_ibv_rdm_header)));
	assert(request->state.eager == FI_IBV_STATE_EAGER_RECV_END);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_RECV_WAIT4LC ||
	       request->state.rndv == FI_IBV_STATE_RNDV_RECV_WAIT4RES);
	assert(FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(wr.wr_id) == 0);

	FI_IBV_RDM_DEC_SIG_POST_COUNTERS(conn, p->ep);
	request->post_counter--;

	if (request->rest_len || request->post_counter) {
		FI_IBV_RDM_HNDL_REQ_LOG_OUT();
		return ret;
	}

	assert(request->sbuf);

	sbuf->header.tag = 0;
	sbuf->header.service_tag = 0;
	FI_IBV_RDM_SET_PKTTYPE(sbuf->header.service_tag,
			       FI_IBV_RDM_RNDV_ACK_PKT);
	sbuf->service_data.pkt_len = ack_size;
	memcpy(&sbuf->payload, &request->rndv.id, sizeof(request->rndv.id));

	FI_IBV_RDM_INC_SIG_POST_COUNTERS(request->minfo.conn, p->ep);
	VERBS_DBG(FI_LOG_EP_DATA,
		  "posted %d bytes, conn %p, tag 0x%" PRIx64 ", request %p\n",
		  sge.length, request->minfo.conn, request->minfo.tag, request);
	ret = ibv_post_send(conn->qp[0], &wr, &bad_wr);
	if (ret == 0) {
		assert(request->rndv.md.mr);
		p->ep->domain->internal_mr_dereg(&request->rndv.md);
		VERBS_DBG(FI_LOG_EP_DATA,
			  "SENDING RNDV ACK: conn %p, sends_outgoing = %"PRIu32", "
			  "post_sends = %"PRIu32"\n",
			  conn, conn->av_entry->sends_outgoing,
			  p->ep->posted_sends);
	} else {
		VERBS_INFO_ERRNO(FI_LOG_EP_DATA, "ibv_post_send", errno);
		assert(0);
		ret = -errno;
		request->state.err = ret;
	}
	request->state.eager = FI_IBV_STATE_EAGER_SEND_WAIT4LC;
	request->state.rndv = FI_IBV_STATE_RNDV_RECV_END;

	if (request->state.err == FI_SUCCESS) {
		fi_ibv_rdm_cntr_inc(p->ep->recv_cntr);

		if (request->comp_flags & FI_COMPLETION) {
			fi_ibv_rdm_move_to_cq(p->ep->fi_rcq, request);
		}
	} else {
		fi_ibv_rdm_cntr_inc_err(p->ep->recv_cntr);

		if (request->comp_flags & FI_COMPLETION) {
			fi_ibv_rdm_move_to_errcq(p->ep->fi_rcq, request,
						 request->state.err);
		}
	}

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	return ret;
}

static ssize_t
fi_ibv_rdm_rndv_recv_ack_lc(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();
	assert(request->state.eager == FI_IBV_STATE_EAGER_SEND_WAIT4LC ||
	       request->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_RECV_END);

	struct fi_ibv_rdm_tagged_send_completed_data *p = data;
	FI_IBV_RDM_DEC_SIG_POST_COUNTERS(request->minfo.conn, p->ep);

	if (request->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE) {
		FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
		util_buf_release(
			request->ep->fi_ibv_rdm_request_pool,
			request);
	} else {
		request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;
		request->state.rndv = FI_IBV_STATE_RNDV_RECV_END;
	}

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	return FI_SUCCESS;
}

static ssize_t
fi_ibv_rdm_rma_init_request(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();
	assert(request->state.eager == FI_IBV_STATE_EAGER_BEGIN);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED);

	struct fi_ibv_rdm_rma_start_data *p =
		(struct fi_ibv_rdm_rma_start_data *)data;
	ssize_t ret = FI_SUCCESS;
	int lmr_access = 0;

	request->context = p->context;
	request->minfo.conn = p->conn;
	request->len = p->data_len;
	request->rest_len = p->data_len;
	request->post_counter = 0;

	request->rma.remote_addr = p->rbuf;
	request->rma.mr_rkey = p->mr_rkey;
	request->rma.mr_lkey = p->mr_lkey;
	request->rma.opcode = p->op_code;
	assert(!request->rma.md.mr);

	request->comp_flags = p->flags;
	if (p->op_code == IBV_WR_RDMA_READ) {
		request->dest_buf = (void*)p->lbuf;
		lmr_access |= FI_READ;
	} else {
		assert(p->op_code == IBV_WR_RDMA_WRITE);
		lmr_access |= FI_WRITE;
		request->src_addr = (void*)p->lbuf;
	}

	if (request->rmabuf && request->len >= p->ep_rdm->max_inline_rc) {
		memcpy(&request->rmabuf->payload, request->src_addr,
			request->len);
	} else if (!request->rmabuf && !p->mr_lkey) {
		ret = p->ep_rdm->domain->internal_mr_reg(p->ep_rdm->domain,
							 (void *)p->lbuf, p->data_len,
							 lmr_access,
							 &request->rma.md);
		if (!ret)
			request->rma.mr_lkey =
				fi_ibv_mr_internal_lkey(&request->rma.md);
	}

	request->state.eager = FI_IBV_STATE_EAGER_RMA_INITIALIZED;

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();

	return ret;
}

static ssize_t
fi_ibv_rdm_rma_inject_request(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();
	assert(request->state.eager == FI_IBV_STATE_EAGER_RMA_INJECT);
	assert(request->state.rndv  == FI_IBV_STATE_RNDV_NOT_USED);


	struct ibv_sge sge = { 0 };
	struct ibv_send_wr wr = { 0 };
	struct ibv_send_wr *bad_wr = NULL;
	struct fi_ibv_rdm_rma_start_data *p = data;
	int ret;

	request->minfo.conn = p->conn;
	request->len = p->data_len;
	request->comp_flags = p->flags;
	request->rmabuf = NULL;

	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.wr.rdma.remote_addr = p->rbuf;
	wr.wr.rdma.rkey = p->mr_rkey;
	wr.send_flags = 0;
	wr.wr_id = FI_IBV_RDM_PACK_WR(request);
	assert(FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(wr.wr_id) == 0);
	wr.opcode = IBV_WR_RDMA_WRITE;
	sge.length = request->len;
	sge.addr = p->lbuf;

	if ((request->len < p->ep_rdm->max_inline_rc) && 
	    (!RMA_RESOURCES_IS_BUSY(request->minfo.conn, p->ep_rdm)) &&
	    fi_ibv_rdm_check_connection(request->minfo.conn)) {
		wr.send_flags |= IBV_SEND_INLINE;
	} else if (fi_ibv_rdm_prepare_rma_request(request, p->ep_rdm)) {
		memcpy(&request->rmabuf->payload, (void*)p->lbuf, p->data_len);
		sge.addr = (uintptr_t)&request->rmabuf->payload;
		sge.lkey = fi_ibv_mr_internal_rkey(
				&request->minfo.conn->rma_md);
	} else {
		FI_IBV_RDM_HNDL_REQ_LOG_OUT();
		return -FI_EAGAIN;
	}

	FI_IBV_RDM_INC_SIG_POST_COUNTERS(request->minfo.conn, p->ep_rdm);

	ret = ibv_post_send(request->minfo.conn->qp[0], &wr, &bad_wr);
	request->state.eager = FI_IBV_STATE_EAGER_RMA_INJECT_WAIT4LC;
	FI_IBV_RDM_HNDL_REQ_LOG_OUT();

	return (ret == 0) ? FI_SUCCESS : -errno;
}

static ssize_t
fi_ibv_rdm_rma_post_ready(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();
	assert((request->state.eager == FI_IBV_STATE_EAGER_RMA_INITIALIZED &&
		request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED) ||
	       (request->state.eager == FI_IBV_STATE_EAGER_RMA_POSTPONED &&
		request->state.rndv == FI_IBV_STATE_ZEROCOPY_RMA_WAIT4LC));

	int ret;
	struct fi_ibv_rma_post_ready_data *p = data;
	
	const size_t offset = request->len - request->rest_len;
	const size_t seg_cursize =
		MIN(p->ep_rdm->rndv_seg_size, request->rest_len);

	struct ibv_sge sge = { 0 };
	struct ibv_send_wr wr = { 0 };
	struct ibv_send_wr *bad_wr = NULL;

	wr.wr_id = FI_IBV_RDM_PACK_WR(request);
	assert(FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(wr.wr_id) == 0);
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.wr.rdma.remote_addr = request->rma.remote_addr;
	wr.wr.rdma.rkey = request->rma.mr_rkey;
	wr.send_flags = 0;
	wr.opcode = request->rma.opcode;

	if (request->state.eager == FI_IBV_STATE_EAGER_RMA_POSTPONED) {
		fi_ibv_rdm_remove_from_postponed_queue(request);
		request->state.eager = FI_IBV_STATE_EAGER_RMA_INITIALIZED;
	}

	/* buffered operation */
	if (request->rmabuf) {
		if (request->rma.opcode == IBV_WR_RDMA_WRITE && 
		    request->len < p->ep_rdm->max_inline_rc) {
			wr.send_flags |= IBV_SEND_INLINE;
			sge.addr = (uintptr_t)request->src_addr;
		} else {
			sge.addr = (uintptr_t)&request->rmabuf->payload;
			sge.lkey = fi_ibv_mr_internal_lkey(
					&request->minfo.conn->rma_md);
		}
		request->state.eager = FI_IBV_STATE_EAGER_RMA_WAIT4LC;
	} else {
		/* src_addr or dest_buf from an union
		 *  for write or read properly */
		sge.addr = ((uintptr_t)request->src_addr) + offset;
		sge.lkey = request->rma.mr_lkey;
		request->state.rndv = FI_IBV_STATE_ZEROCOPY_RMA_WAIT4LC;
	}

	sge.length = seg_cursize;

	request->rest_len -= seg_cursize;
	request->post_counter++;
	FI_IBV_RDM_INC_SIG_POST_COUNTERS(request->minfo.conn, p->ep_rdm);

	ret = ibv_post_send(request->minfo.conn->qp[0], &wr, &bad_wr);
	if (request->rest_len && !ret) {
		ret = fi_ibv_rdm_move_to_postponed_queue(request);
		if (ret)
			return ret;
		request->state.eager = FI_IBV_STATE_EAGER_RMA_POSTPONED;
	}

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();

	return (!ret) ? FI_SUCCESS : -errno;
}

static ssize_t
fi_ibv_rdm_rma_inject_lc(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();
	assert(request->state.eager == FI_IBV_STATE_EAGER_RMA_INJECT_WAIT4LC);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED);

	struct fi_ibv_rdm_tagged_send_completed_data *p = data;

	FI_IBV_RDM_DEC_SIG_POST_COUNTERS(request->minfo.conn, p->ep);
	if (request->rmabuf) {
		fi_ibv_rdm_set_buffer_status(request->rmabuf, BUF_STATUS_FREE);
	} /* else inline flag was set */

	FI_IBV_RDM_HNDL_REQ_LOG();

	FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
	util_buf_release(request->ep->fi_ibv_rdm_request_pool,
			 request);

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();

	return FI_SUCCESS;
}

static ssize_t
fi_ibv_rdm_rma_buffered_lc(struct fi_ibv_rdm_request *request, void *data)
{
	FI_IBV_RDM_HNDL_REQ_LOG_IN();

	assert(request->state.eager == FI_IBV_STATE_EAGER_RMA_WAIT4LC);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED);

	struct fi_ibv_rdm_tagged_send_completed_data *p = data;
	FI_IBV_RDM_DEC_SIG_POST_COUNTERS(request->minfo.conn, p->ep);

	assert(request->rmabuf);
	if (request->rma.opcode == IBV_WR_RDMA_READ) {
		memcpy(request->dest_buf, &request->rmabuf->payload, request->len);
	}
	fi_ibv_rdm_set_buffer_status(request->rmabuf, BUF_STATUS_FREE);

	if (request->rma.opcode == IBV_WR_RDMA_READ) {
		fi_ibv_rdm_cntr_inc(p->ep->read_cntr);
	} else if (request->rma.opcode == IBV_WR_RDMA_WRITE) {
		fi_ibv_rdm_cntr_inc(p->ep->write_cntr);
	}

	if (request->comp_flags & FI_COMPLETION) {
		fi_ibv_rdm_move_to_cq(p->ep->fi_scq, request);
	} else {
		request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;
		FI_IBV_RDM_HNDL_REQ_LOG();
	}

	if (request->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE) {
		FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
		util_buf_release(
			request->ep->fi_ibv_rdm_request_pool,
			request);
	} else {
		request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;
	}

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	return FI_SUCCESS;
}


static ssize_t
fi_ibv_rdm_rma_zerocopy_lc(struct fi_ibv_rdm_request *request, void *data)
{
	ssize_t ret = FI_SUCCESS;
	FI_IBV_RDM_HNDL_REQ_LOG_IN();

	assert(request->state.eager == FI_IBV_STATE_EAGER_RMA_INITIALIZED ||
		(request->state.eager == FI_IBV_STATE_EAGER_RMA_POSTPONED));
	assert(request->state.rndv == FI_IBV_STATE_ZEROCOPY_RMA_WAIT4LC);
	assert(!request->rmabuf);

	VERBS_DBG(FI_LOG_EP_DATA, "conn %p, tag 0x%" PRIx64 ", len %" PRIu64 "\n",
		  request->minfo.conn, request->minfo.tag, request->len);

	struct fi_ibv_rdm_tagged_send_completed_data *p = data;
	FI_IBV_RDM_DEC_SIG_POST_COUNTERS(request->minfo.conn, p->ep);
	request->post_counter--;

	if (request->rest_len == 0 && request->post_counter == 0) {
		if (request->rma.md.mr)
			ret = p->ep->domain->internal_mr_dereg(&request->rma.md);

		if (request->rma.opcode == IBV_WR_RDMA_READ)
			fi_ibv_rdm_cntr_inc(p->ep->read_cntr);
		else if (request->rma.opcode == IBV_WR_RDMA_WRITE)
			fi_ibv_rdm_cntr_inc(p->ep->write_cntr);

		if (request->comp_flags & FI_COMPLETION) {
			if (ret)
				fi_ibv_rdm_move_to_errcq(p->ep->fi_scq, request, ret);
			else
				fi_ibv_rdm_move_to_cq(p->ep->fi_scq, request);
			request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;
			request->state.rndv = FI_IBV_STATE_ZEROCOPY_RMA_END;
		} else {
			FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
			util_buf_release(
				request->ep->fi_ibv_rdm_request_pool,
				request);
		}
	}

	FI_IBV_RDM_HNDL_REQ_LOG_OUT();
	return ret;
}

static ssize_t
fi_ibv_rdm_req_hndl_err(struct fi_ibv_rdm_request *request, void *data)
{
	VERBS_INFO(FI_LOG_EP_DATA,
		"\t> IN\t< eager_state = %s, rndv_state = %s, len = %lu\n",
		fi_ibv_rdm_req_eager_state_to_str(request->state.eager),
		fi_ibv_rdm_req_rndv_state_to_str(request->state.rndv),
		request->len);

	assert(0);
	return -FI_EOTHER;
}

ssize_t fi_ibv_rdm_req_hndls_init(void)
{
	size_t i, j, k;

	for (i = 0; i < FI_IBV_STATE_EAGER_COUNT; ++i) {
		for (j = 0; j < FI_IBV_STATE_RNDV_COUNT; ++j) {
			for (k = 0; k < FI_IBV_EVENT_COUNT; ++k) {
				fi_ibv_rdm_req_hndl_arr[i][j][k] =
					fi_ibv_rdm_req_hndl_err;
			}
		}
	}

	// EAGER_SEND stuff
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_BEGIN]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_SEND_START] =
			fi_ibv_rdm_init_send_request;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_SEND_POSTPONED]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_POST_READY] =
			fi_ibv_rdm_eager_send_ready;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_SEND_WAIT4LC]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_POST_LC] =
			fi_ibv_rdm_eager_send_lc;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_READY_TO_FREE]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_POST_LC] =
			fi_ibv_rdm_eager_send_lc;

	// EAGER_RECV stuff
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_BEGIN]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_RECV_START] =
			fi_ibv_rdm_init_recv_request;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_BEGIN]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_RECV_PEEK] =
			fi_ibv_rdm_tagged_peek_request;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_BEGIN]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_RECV_CLAIM] =
			fi_ibv_rdm_tagged_recv_claim;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_BEGIN]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_RECV_GOT_PKT_PROCESS] =
			fi_ibv_rdm_init_unexp_recv_request;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RECV_WAIT4PKT]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_RECV_GOT_PKT_PROCESS] =
			fi_ibv_rdm_eager_recv_got_pkt;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RECV_WAIT4RECV]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_RECV_START] =
			fi_ibv_rdm_eager_recv_process_unexp_pkt;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RECV_WAIT4RECV]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_RECV_CLAIM] =
			fi_ibv_rdm_tagged_recv_claim;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RECV_CLAIMED]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_RECV_START] =
			fi_ibv_rdm_eager_recv_process_unexp_pkt;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RECV_WAIT4RECV]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_RECV_DISCARD] =
			fi_ibv_rdm_eager_recv_discard;

	// RNDV_SEND stuff
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_BEGIN]
		[FI_IBV_STATE_RNDV_SEND_BEGIN][FI_IBV_EVENT_SEND_START] =
			fi_ibv_rdm_init_send_request;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_SEND_POSTPONED]
		[FI_IBV_STATE_RNDV_SEND_WAIT4SEND][FI_IBV_EVENT_POST_READY] =
			fi_ibv_rdm_rndv_rts_send_ready;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_SEND_WAIT4LC]
		[FI_IBV_STATE_RNDV_SEND_WAIT4ACK][FI_IBV_EVENT_POST_LC] =
			fi_ibv_rdm_rndv_rts_lc;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_READY_TO_FREE]
		[FI_IBV_STATE_RNDV_SEND_END][FI_IBV_EVENT_POST_LC] =
			fi_ibv_rdm_rndv_rts_lc;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_SEND_WAIT4LC]
		[FI_IBV_STATE_RNDV_SEND_END][FI_IBV_EVENT_POST_LC] =
			fi_ibv_rdm_rndv_rts_lc;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_SEND_WAIT4LC]
		[FI_IBV_STATE_RNDV_SEND_WAIT4ACK][FI_IBV_EVENT_RECV_GOT_PKT_PROCESS]
			= fi_ibv_rdm_rndv_end;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_SEND_END]
		[FI_IBV_STATE_RNDV_SEND_WAIT4ACK][FI_IBV_EVENT_RECV_GOT_PKT_PROCESS]
			= fi_ibv_rdm_rndv_end;
	
	// RNDV_RECV stuff
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_BEGIN]
		[FI_IBV_STATE_RNDV_RECV_BEGIN][FI_IBV_EVENT_RECV_START] =
			fi_ibv_rdm_init_recv_request;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RECV_WAIT4RECV]
		[FI_IBV_STATE_RNDV_RECV_WAIT4RES][FI_IBV_EVENT_RECV_CLAIM] =
			fi_ibv_rdm_tagged_recv_claim;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RECV_END]
		[FI_IBV_STATE_RNDV_RECV_WAIT4RES][FI_IBV_EVENT_POST_READY] =
			fi_ibv_rdm_rndv_recv_post_read;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RECV_END]
		[FI_IBV_STATE_RNDV_RECV_WAIT4RES][FI_IBV_EVENT_POST_LC] =
			fi_ibv_rdm_rndv_recv_read_lc;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RECV_END]
		[FI_IBV_STATE_RNDV_RECV_WAIT4LC][FI_IBV_EVENT_POST_LC] =
			fi_ibv_rdm_rndv_recv_read_lc;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RECV_END]
		[FI_IBV_STATE_RNDV_RECV_WAIT4LC][FI_IBV_EVENT_POST_READY] =
			fi_ibv_rdm_rndv_recv_read_lc;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_SEND_WAIT4LC]
		[FI_IBV_STATE_RNDV_RECV_END][FI_IBV_EVENT_POST_LC] =
			fi_ibv_rdm_rndv_recv_ack_lc;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_READY_TO_FREE]
		[FI_IBV_STATE_RNDV_RECV_END][FI_IBV_EVENT_POST_LC] =
			fi_ibv_rdm_rndv_recv_ack_lc;

	// RMA read/write stuff
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_BEGIN]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_RMA_START] =
			fi_ibv_rdm_rma_init_request;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RMA_INJECT]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_RMA_START] =
			fi_ibv_rdm_rma_inject_request;	
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RMA_INITIALIZED]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_POST_READY] =
			fi_ibv_rdm_rma_post_ready;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RMA_POSTPONED]
		[FI_IBV_STATE_ZEROCOPY_RMA_WAIT4LC][FI_IBV_EVENT_POST_READY] =
			fi_ibv_rdm_rma_post_ready;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RMA_WAIT4LC]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_POST_LC] =
			fi_ibv_rdm_rma_buffered_lc;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RMA_INJECT_WAIT4LC]
		[FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_POST_LC] =
			fi_ibv_rdm_rma_inject_lc;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RMA_INITIALIZED]
		[FI_IBV_STATE_ZEROCOPY_RMA_WAIT4LC][FI_IBV_EVENT_POST_LC] =
			fi_ibv_rdm_rma_zerocopy_lc;
	fi_ibv_rdm_req_hndl_arr[FI_IBV_STATE_EAGER_RMA_POSTPONED]
		[FI_IBV_STATE_ZEROCOPY_RMA_WAIT4LC][FI_IBV_EVENT_POST_LC] =
			fi_ibv_rdm_rma_zerocopy_lc;

	return FI_SUCCESS;
}

ssize_t fi_ibv_rdm_req_hndls_clean(void)
{
	size_t i, j, k;
	for (i = 0; i < FI_IBV_STATE_EAGER_COUNT; ++i) {
		for (j = 0; j < FI_IBV_STATE_RNDV_COUNT; ++j) {
			for (k = 0; k < FI_IBV_EVENT_COUNT; ++k) {
				fi_ibv_rdm_req_hndl_arr[i][j][k] = NULL;
			}
		}
	}
	return FI_SUCCESS;
}

ssize_t
fi_ibv_rdm_req_hndl(struct fi_ibv_rdm_request * request,
		    enum fi_ibv_rdm_request_event event, void *data)
{
	VERBS_DBG(FI_LOG_EP_DATA, "\t%p, eager_state = %s, rndv_state = %s, event = %s\n",
		  request,
		  fi_ibv_rdm_req_eager_state_to_str(request->state.eager),
		  fi_ibv_rdm_req_rndv_state_to_str(request->state.rndv),
		  fi_ibv_rdm_event_to_str(event));
	assert(fi_ibv_rdm_req_hndl_arr[request->state.eager]
				      [request->state.rndv][event]);

	return fi_ibv_rdm_req_hndl_arr[request->state.eager]
				      [request->state.rndv]
				      [event] (request, data);
}

char *
fi_ibv_rdm_req_eager_state_to_str(enum fi_ibv_rdm_request_eager_state state)
{
	switch (state) {
	case FI_IBV_STATE_EAGER_BEGIN:
		return "STATE_EAGER_BEGIN";

	case FI_IBV_STATE_EAGER_SEND_POSTPONED:
		return "STATE_EAGER_SEND_POSTPONED";
	case FI_IBV_STATE_EAGER_SEND_WAIT4LC:
		return "STATE_EAGER_SEND_WAIT4LC";
	case FI_IBV_STATE_EAGER_SEND_END:
		return "STATE_EAGER_SEND_END";

	case FI_IBV_STATE_EAGER_RECV_BEGIN:
		return "STATE_EAGER_RECV_BEGIN";
	case FI_IBV_STATE_EAGER_RECV_WAIT4PKT:
		return "STATE_EAGER_RECV_WAIT4PKT";
	case FI_IBV_STATE_EAGER_RECV_WAIT4RECV:
		return "STATE_EAGER_RECV_WAIT4RECV";
	case FI_IBV_STATE_EAGER_RECV_CLAIMED:
		return "FI_IBV_STATE_EAGER_RECV_CLAIMED";
	case FI_IBV_STATE_EAGER_RECV_END:
		return "STATE_EAGER_RECV_END";

	case FI_IBV_STATE_EAGER_RMA_INJECT:
		return "FI_IBV_STATE_EAGER_RMA_INJECT";
	case FI_IBV_STATE_EAGER_RMA_INITIALIZED:
		return "FI_IBV_STATE_EAGER_RMA_INITIALIZED";
	case FI_IBV_STATE_EAGER_RMA_POSTPONED:
		return "FI_IBV_STATE_EAGER_RMA_POSTPONED";
	case FI_IBV_STATE_EAGER_RMA_WAIT4LC:
		return "FI_IBV_STATE_EAGER_RMA_WAIT4LC";
	case FI_IBV_STATE_EAGER_RMA_INJECT_WAIT4LC:
		return "FI_IBV_STATE_EAGER_RMA_INJECT_WAIT4LC";
	case FI_IBV_STATE_EAGER_RMA_END:
		return "FI_IBV_STATE_EAGER_RMA_END";

	case FI_IBV_STATE_EAGER_READY_TO_FREE:
		return "STATE_EAGER_READY_TO_FREE";

	case FI_IBV_STATE_EAGER_COUNT:
		return "STATE_EAGER_COUNT";
	default:
		return "STATE_EAGER_UNKNOWN!!!";
	}
}

char *fi_ibv_rdm_req_rndv_state_to_str(enum fi_ibv_rdm_request_rndv_state state)
{
	switch (state) {
	case FI_IBV_STATE_RNDV_NOT_USED:
		return "STATE_RNDV_NOT_USED";
	case FI_IBV_STATE_RNDV_SEND_BEGIN:
		return "STATE_RNDV_SEND_BEGIN";
	case FI_IBV_STATE_RNDV_SEND_WAIT4SEND:
		return "STATE_RNDV_SEND_WAIT4SEND";
	case FI_IBV_STATE_RNDV_SEND_WAIT4ACK:
		return "STATE_RNDV_SEND_WAIT4ACK";
	case FI_IBV_STATE_RNDV_SEND_END:
		return "STATE_RNDV_SEND_END";

	case FI_IBV_STATE_RNDV_RECV_BEGIN:
		return "STATE_RNDV_RECV_BEGIN";
	case FI_IBV_STATE_RNDV_RECV_WAIT4RES:
		return "STATE_RNDV_RECV_WAIT4RES";
	case FI_IBV_STATE_RNDV_RECV_WAIT4RECV:
		return "STATE_RNDV_RECV_WAIT4RECV";
	case FI_IBV_STATE_RNDV_RECV_WAIT4LC:
		return "STATE_RNDV_RECV_WAIT4LC";
	case FI_IBV_STATE_RNDV_RECV_END:
		return "STATE_RNDV_RECV_END";

	case FI_IBV_STATE_ZEROCOPY_RMA_WAIT4LC:
		return "FI_IBV_STATE_ZEROCOPY_RMA_WAIT4LC";
	case FI_IBV_STATE_ZEROCOPY_RMA_END:
		return "FI_IBV_STATE_ZEROCOPY_RMA_END";

	case FI_IBV_STATE_RNDV_COUNT:
		return "STATE_RNDV_COUNT";
	default:
		return "STATE_RNDV_UNKNOWN!!!";
	}
}

char *fi_ibv_rdm_event_to_str(enum fi_ibv_rdm_request_event event)
{
	switch (event) {
	case FI_IBV_EVENT_SEND_START:
		return "EVENT_SEND_START";
	case FI_IBV_EVENT_POST_READY:
		return "FI_IBV_EVENT_POST_READY";
	case FI_IBV_EVENT_POST_LC:
		return "FI_IBV_EVENT_POST_LC";

	case FI_IBV_EVENT_RECV_START:
		return "EVENT_RECV_START";
	case FI_IBV_EVENT_RECV_GOT_PKT_PREPROCESS:
		return "EVENT_RECV_GOT_PKT_PREPROCESS";
	case FI_IBV_EVENT_RECV_GOT_PKT_PROCESS:
		return "EVENT_RECV_GOT_PKT_PROCESS";
	case FI_IBV_EVENT_RECV_GOT_ACK:
		return "EVENT_RECV_GOT_ACK";
	case FI_IBV_EVENT_RECV_PEEK:
		return "FI_IBV_EVENT_RECV_PEEK";
	case FI_IBV_EVENT_RECV_CLAIM:
		return "FI_IBV_EVENT_RECV_CLAIM";
	case FI_IBV_EVENT_RECV_DISCARD:
		return "FI_IBV_EVENT_RECV_DISCARD";

	case FI_IBV_EVENT_RMA_START:
		return "FI_IBV_EVENT_RMA_START";

	case FI_IBV_EVENT_COUNT:
		return "EVENT_COUNT";
	default:
		return "EVENT_UNKNOWN!!!";
	}
}
