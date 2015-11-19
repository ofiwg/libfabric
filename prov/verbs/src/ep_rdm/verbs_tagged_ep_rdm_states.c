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

#include <fi_list.h>
#include "../fi_verbs.h"
#include "verbs_rdm.h"
#include "verbs_queuing.h"
#include "verbs_tagged_ep_rdm_states.h"

extern struct dlist_entry fi_ibv_rdm_tagged_send_postponed_queue;
extern struct fi_ibv_mem_pool fi_ibv_rdm_tagged_request_pool;
extern struct fi_ibv_mem_pool fi_ibv_rdm_tagged_unexp_buffers_pool;

typedef enum fi_rdm_tagged_req_hndl_ret (*fi_ep_rdm_request_handler_t)
	(struct fi_ibv_rdm_tagged_request * request, void *data);

static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_err_hndl(struct fi_ibv_rdm_tagged_request *request,
		void *data);
static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_init_send_request(struct fi_ibv_rdm_tagged_request *request,
		void *data);
static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_eager_send_ready(struct fi_ibv_rdm_tagged_request *request,
		void *data);
static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_eager_send_lc(struct fi_ibv_rdm_tagged_request *request,
		void *data);
static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_rndv_rts_send_ready(struct fi_ibv_rdm_tagged_request *request,
		void *data);
static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_rndv_rts_lc(struct fi_ibv_rdm_tagged_request *request,
		void *data);
static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_rndv_end(struct fi_ibv_rdm_tagged_request *request,
		void *data);
static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_init_recv_request(struct fi_ibv_rdm_tagged_request *request,
		    void *data);
static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_init_unexp_recv_request(struct fi_ibv_rdm_tagged_request *request,
		void *data);
static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_eager_recv_got_pkt(struct fi_ibv_rdm_tagged_request *request,
		void *data);
static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_eager_recv_got_unexp_pkt(struct fi_ibv_rdm_tagged_request *request,
		void *data);
static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_rndv_recv_post_read(struct fi_ibv_rdm_tagged_request *request,
		void *data);
static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_rndv_recv_read_lc(struct fi_ibv_rdm_tagged_request *request,
		void *data);
static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_rndv_recv_ack_lc(struct fi_ibv_rdm_tagged_request *request,
		void *data);

static fi_ep_rdm_request_handler_t
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_COUNT]
	                          [FI_IBV_STATE_RNDV_COUNT]
	                          [FI_IBV_EVENT_COUNT];

enum fi_rdm_tagged_req_hndl_ret fi_ibv_rdm_tagged_req_hndls_init(void)
{
	size_t i, j, k;

	for (i = 0; i < FI_IBV_STATE_EAGER_COUNT; ++i) {
		for (j = 0; j < FI_IBV_STATE_RNDV_COUNT; ++j) {
			for (k = 0; k < FI_IBV_EVENT_COUNT; ++k) {
				fi_ibv_rdm_tagged_hndl_arr[i][j][k] =
				    fi_ibv_rdm_tagged_err_hndl;
			}
		}
	}

	// EAGER_SEND stuff
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_BEGIN]
	    [FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_SEND_START] =
	    fi_ibv_rdm_tagged_init_send_request;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_SEND_POSTPONED]
	    [FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_SEND_READY] =
	    fi_ibv_rdm_tagged_eager_send_ready;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_SEND_WAIT4LC]
	    [FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_SEND_GOT_LC] =
	    fi_ibv_rdm_tagged_eager_send_lc;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_READY_TO_FREE]
	    [FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_SEND_GOT_LC] =
	    fi_ibv_rdm_tagged_eager_send_lc;

	// EAGER_RECV stuff
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_BEGIN]
	    [FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_RECV_START] =
	    fi_ibv_rdm_tagged_init_recv_request;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_BEGIN]
	    [FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_RECV_GOT_PKT_PROCESS] =
	    fi_ibv_rdm_tagged_init_unexp_recv_request;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_RECV_WAIT4PKT]
	    [FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_RECV_GOT_PKT_PROCESS] =
	    fi_ibv_rdm_tagged_eager_recv_got_pkt;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_RECV_WAIT4RECV]
	    [FI_IBV_STATE_RNDV_NOT_USED][FI_IBV_EVENT_RECV_START] =
	    fi_ibv_rdm_tagged_eager_recv_got_unexp_pkt;

	// RNDV_SEND stuff
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_BEGIN]
	    [FI_IBV_STATE_RNDV_SEND_BEGIN][FI_IBV_EVENT_SEND_START] =
	    fi_ibv_rdm_tagged_init_send_request;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_SEND_POSTPONED]
	    [FI_IBV_STATE_RNDV_SEND_WAIT4SEND][FI_IBV_EVENT_SEND_READY] =
	    fi_ibv_rdm_tagged_rndv_rts_send_ready;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_SEND_WAIT4LC]
	    [FI_IBV_STATE_RNDV_SEND_WAIT4ACK][FI_IBV_EVENT_SEND_GOT_LC] =
	    fi_ibv_rdm_tagged_rndv_rts_lc;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_READY_TO_FREE]
	    [FI_IBV_STATE_RNDV_SEND_END][FI_IBV_EVENT_SEND_GOT_LC] =
	    fi_ibv_rdm_tagged_rndv_rts_lc;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_SEND_WAIT4LC]
	    [FI_IBV_STATE_RNDV_SEND_WAIT4ACK][FI_IBV_EVENT_RECV_GOT_PKT_PROCESS]
	    = fi_ibv_rdm_tagged_rndv_end;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_SEND_END]
	    [FI_IBV_STATE_RNDV_SEND_WAIT4ACK][FI_IBV_EVENT_RECV_GOT_PKT_PROCESS]
	    = fi_ibv_rdm_tagged_rndv_end;

	// RNDV_RECV stuff
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_BEGIN]
	    [FI_IBV_STATE_RNDV_RECV_BEGIN][FI_IBV_EVENT_RECV_START] =
	    fi_ibv_rdm_tagged_init_recv_request;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_RECV_END]
	    [FI_IBV_STATE_RNDV_RECV_WAIT4RES][FI_IBV_EVENT_SEND_READY] =
	    fi_ibv_rdm_tagged_rndv_recv_post_read;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_RECV_END]
	    [FI_IBV_STATE_RNDV_RECV_WAIT4LC][FI_IBV_EVENT_SEND_GOT_LC] =
	    fi_ibv_rdm_tagged_rndv_recv_read_lc;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_RECV_END]
	    [FI_IBV_STATE_RNDV_RECV_WAIT4LC][FI_IBV_EVENT_SEND_READY] =
	    fi_ibv_rdm_tagged_rndv_recv_read_lc;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_SEND_WAIT4LC]
	    [FI_IBV_STATE_RNDV_RECV_END][FI_IBV_EVENT_SEND_GOT_LC] =
	    fi_ibv_rdm_tagged_rndv_recv_ack_lc;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_READY_TO_FREE]
	    [FI_IBV_STATE_RNDV_RECV_END][FI_IBV_EVENT_SEND_GOT_LC] =
	    fi_ibv_rdm_tagged_rndv_recv_ack_lc;
	fi_ibv_rdm_tagged_hndl_arr[FI_IBV_STATE_EAGER_RECV_END]
	    [FI_IBV_STATE_RNDV_RECV_WAIT4LC][FI_IBV_EVENT_SEND_GOT_LC] =
	    fi_ibv_rdm_tagged_rndv_recv_read_lc;

	return FI_EP_RDM_HNDL_SUCCESS;
}

enum fi_rdm_tagged_req_hndl_ret fi_ibv_rdm_tagged_req_hndls_clean(void)
{
	size_t i, j, k;
	for (i = 0; i < FI_IBV_STATE_EAGER_COUNT; ++i) {
		for (j = 0; j < FI_IBV_STATE_RNDV_COUNT; ++j) {
			for (k = 0; k < FI_IBV_EVENT_COUNT; ++k) {
				fi_ibv_rdm_tagged_hndl_arr[i][j][k] = NULL;
			}
		}
	}
	return FI_EP_RDM_HNDL_SUCCESS;
}

enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_req_hndl(struct fi_ibv_rdm_tagged_request * request,
			   enum fi_ibv_rdm_tagged_request_event event, void *data)
{
	VERBS_DBG(FI_LOG_EP_DATA, "\t%p, eager_state = %s, rndv_state = %s, event = %s\n",
		     request,
		     fi_ibv_rdm_tagged_req_eager_state_to_str
		     (request->state.eager),
		     fi_ibv_rdm_tagged_req_rndv_state_to_str(request->state.
							     rndv),
		     fi_ibv_rdm_tagged_req_event_to_str(event));
	assert(fi_ibv_rdm_tagged_hndl_arr[request->state.eager]
	       [request->state.rndv]
	       [event]);

	return fi_ibv_rdm_tagged_hndl_arr[request->state.eager]
	    [request->state.rndv]
	    [event] (request, data);
}

static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_err_hndl(struct fi_ibv_rdm_tagged_request *request,
			   void *data)
{
	VERBS_INFO(FI_LOG_EP_DATA,
		"\t> IN\t< eager_state = %s, rndv_state = %s, len = %d\n",
		fi_ibv_rdm_tagged_req_eager_state_to_str(request->state.eager),
		fi_ibv_rdm_tagged_req_rndv_state_to_str(request->state.rndv),
		request->len);

	assert(0);
	return FI_EP_RDM_HNDL_NOT_INIT;
}

#if ENABLE_DEBUG

enum fi_ibv_rdm_tagged_hndl_req_log_state {
	hndl_req_log_state_in = 0,
	hndl_req_log_state_out = 10000
};

#define FI_IBV_RDM_TAGGED_HANDLER_LOG_IN()                                  \
enum fi_ibv_rdm_tagged_hndl_req_log_state state = hndl_req_log_state_in;    \
do {                                                                        \
	FI_IBV_RDM_TAGGED_DBG_REQUEST("\t> IN\t< ", request, FI_LOG_DEBUG); \
} while(0)

#define FI_IBV_RDM_TAGGED_HANDLER_LOG() do {                            \
	state++;                                                        \
	char prefix[128];                                               \
	snprintf(prefix, 128, "\t> %d\t< ", state);                     \
	FI_IBV_RDM_TAGGED_DBG_REQUEST(prefix, request, FI_LOG_DEBUG);   \
} while(0)

#define FI_IBV_RDM_TAGGED_HANDLER_LOG_OUT() do {                              \
	assert(state < hndl_req_log_state_out);                               \
	FI_IBV_RDM_TAGGED_DBG_REQUEST("\t> OUT\t< ", request, FI_LOG_DEBUG);  \
} while(0)

#else // ENABLE_DEBUG
#define FI_IBV_RDM_TAGGED_HANDLER_LOG_IN()
#define FI_IBV_RDM_TAGGED_HANDLER_LOG()
#define FI_IBV_RDM_TAGGED_HANDLER_LOG_OUT()
#endif // ENABLE_DEBUG

static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_init_send_request(struct fi_ibv_rdm_tagged_request *request,
				    void *data)
{
	FI_IBV_RDM_TAGGED_HANDLER_LOG_IN();

	struct fi_ibv_rdm_tagged_send_start_data *p = data;
	request->tag = p->tag;
	request->src_addr = p->src_addr;
	request->len = p->data_len;
	request->imm = p->imm;
	request->context = p->context;
	request->conn = p->conn;
	request->state.eager = FI_IBV_STATE_EAGER_BEGIN;
	request->state.rndv =
	    (p->data_len + sizeof(struct fi_ibv_rdm_tagged_header)
	     <= p->rndv_threshold)
	    ? FI_IBV_STATE_RNDV_NOT_USED : FI_IBV_STATE_RNDV_SEND_BEGIN;

	FI_IBV_RDM_TAGGED_HANDLER_LOG();

	fi_ibv_rdm_tagged_move_to_postponed_queue(request);
	request->state.eager = FI_IBV_STATE_EAGER_SEND_POSTPONED;
	if (request->state.rndv == FI_IBV_STATE_RNDV_SEND_BEGIN) {
		request->state.rndv = FI_IBV_STATE_RNDV_SEND_WAIT4SEND;
	}

	FI_IBV_RDM_TAGGED_HANDLER_LOG_OUT();
	return FI_EP_RDM_HNDL_SUCCESS;
}

static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_eager_send_ready(struct fi_ibv_rdm_tagged_request *request,
				   void *data)
{
	FI_IBV_RDM_TAGGED_HANDLER_LOG_IN();

	assert(request->state.eager == FI_IBV_STATE_EAGER_SEND_POSTPONED);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED);

	fi_ibv_rdm_tagged_remove_from_postponed_queue(request);
	struct fi_ibv_rdm_tagged_send_ready_data *p = data;

	int ret = 0;
	struct ibv_sge sge;

	struct fi_ibv_rdm_tagged_conn *conn = request->conn;
	int size = request->len + sizeof(struct fi_ibv_rdm_tagged_header);

	assert(request->sbuf);

	struct ibv_send_wr wr = { 0 };
	struct ibv_send_wr *bad_wr = NULL;
	wr.wr_id = (uintptr_t) request;

	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.wr.rdma.remote_addr =
	    fi_ibv_rdm_tagged_get_remote_addr(conn, request->sbuf);
	wr.wr.rdma.rkey = conn->remote_rbuf_rkey;
	wr.send_flags = 0;

	sge.addr = (uintptr_t) request->sbuf;
	sge.length = size;

	if (sge.length <= p->ep->max_inline_rc) {
		wr.send_flags |= IBV_SEND_INLINE;
	}

	sge.lkey = conn->s_mr->lkey;

	wr.imm_data =
	    fi_ibv_rdm_tagged_get_buff_service_data(request->sbuf)->seq_number;
	wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	struct fi_ibv_rdm_tagged_buf *sbuf =
	    (struct fi_ibv_rdm_tagged_buf *)request->sbuf;
	void *payload = (void *)&(sbuf->payload[0]);

	sbuf->header.tag = request->tag;
	sbuf->header.service_tag = 0;
	FI_IBV_RDM_SET_PKTTYPE(sbuf->header.service_tag, FI_IBV_RDM_EAGER_PKT);
	if (request->src_addr) {
		memcpy(payload, request->src_addr, request->len);
	}

	FI_IBV_RDM_TAGGED_INC_SEND_COUNTERS_REQ(request, p->ep, wr.send_flags);
	VERBS_DBG(FI_LOG_EP_DATA, "posted %d bytes, conn %p, tag 0x%llx\n", sge.length,
		     request->conn, request->tag);

	ret = ibv_post_send(conn->qp, &wr, &bad_wr);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_EP_DATA, "ibv_post_send", errno);

		assert(0);
	};

	fi_ibv_rdm_tagged_move_to_ready_queue(request);
	request->state.eager = FI_IBV_STATE_EAGER_SEND_WAIT4LC;

	FI_IBV_RDM_TAGGED_HANDLER_LOG_OUT();

	return ret;
}

static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_eager_send_lc(struct fi_ibv_rdm_tagged_request *request,
				void *data)
{
	FI_IBV_RDM_TAGGED_HANDLER_LOG_IN();

	assert(request->state.eager == FI_IBV_STATE_EAGER_SEND_WAIT4LC ||
	       request->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED);

	VERBS_DBG(FI_LOG_EP_DATA, "conn %p, tag 0x%llx, len %d\n", request->conn,
		     request->tag, request->len);

	struct fi_ibv_rdm_tagged_send_completed_data *p = data;
	FI_IBV_RDM_TAGGED_DEC_SEND_COUNTERS(request->conn, p->ep);

	request->send_completions_wait--;

	if (request->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE) {
		FI_IBV_RDM_TAGGED_DBG_REQUEST("to_pool: ", request,
					      FI_LOG_DEBUG);
		fi_ibv_mem_pool_return(&request->mpe,
				       &fi_ibv_rdm_tagged_request_pool);
	} else {
		fi_ibv_rdm_tagged_move_to_ready_queue(request);
		request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;
	}

	FI_IBV_RDM_TAGGED_HANDLER_LOG_OUT();
	return FI_EP_RDM_HNDL_SUCCESS;
}

static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_rndv_rts_send_ready(struct fi_ibv_rdm_tagged_request *request,
				      void *data)
{
	FI_IBV_RDM_TAGGED_HANDLER_LOG_IN();

	assert(request->state.eager == FI_IBV_STATE_EAGER_SEND_POSTPONED);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_SEND_WAIT4SEND);
	assert(request->sbuf);

	VERBS_DBG(FI_LOG_EP_DATA, "conn %p, tag 0x%llx, len %d\n", request->conn,
		     request->tag, request->len);

	fi_ibv_rdm_tagged_remove_from_postponed_queue(request);
	struct fi_ibv_rdm_tagged_send_ready_data *p = data;

	struct ibv_sge sge;

	struct fi_ibv_rdm_tagged_conn *conn = request->conn;
	struct fi_ibv_rdm_tagged_rndv_header *header =
	    (struct fi_ibv_rdm_tagged_rndv_header *)request->sbuf;
	struct ibv_mr *mr = NULL;

	struct ibv_send_wr wr, *bad_wr = NULL;
	memset(&wr, 0, sizeof(wr));
	wr.wr_id = (uintptr_t) request;
	assert(FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(wr.wr_id) == 0);

	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.wr.rdma.remote_addr =
	    fi_ibv_rdm_tagged_get_remote_addr(conn, request->sbuf);
	wr.wr.rdma.rkey = conn->remote_rbuf_rkey;
	wr.send_flags = 0;
	wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	wr.imm_data =
	    fi_ibv_rdm_tagged_get_buff_service_data(request->sbuf)->seq_number;

	sge.addr = (uintptr_t) request->sbuf;
	sge.length = sizeof(struct fi_ibv_rdm_tagged_rndv_header);
	sge.lkey = conn->s_mr->lkey;

	header->base.tag = request->tag;
	header->base.service_tag = 0;
	header->len = request->len;
	header->src_addr = (uint64_t) (uintptr_t) request->src_addr;

	header->id = request;
	request->rndv_id = request;

	mr = ibv_reg_mr(p->ep->domain->pd, (void *)request->src_addr,
			request->len, IBV_ACCESS_REMOTE_READ);
	if (NULL == mr) {
		VERBS_INFO_ERRNO(FI_LOG_EP_DATA, "ibv_reg_mr", errno);
		assert(0);
		return FI_EOTHER;
	}

	header->mem_key = mr->rkey;
	request->rndv_mr = mr;

	struct fi_ibv_rdm_tagged_buf *sbuf = request->sbuf;

	FI_IBV_RDM_SET_PKTTYPE(sbuf->header.service_tag,
			       FI_IBV_RDM_RNDV_RTS_PKT);

	VERBS_DBG(FI_LOG_EP_DATA,
	     "fi_senddatato: RNDV conn %p, tag 0x%llx, len %d, src_addr %p,"
	     "rkey 0x%lx, fi_ctx %p, imm %d, pend_send %d\n", conn,
	     (long long unsigned int)request->tag, (int)request->len,
	     request->src_addr, (long unsigned int)mr->rkey, request->context,
	     (int)wr.imm_data, p->ep->pend_send);

	FI_IBV_RDM_TAGGED_INC_SEND_COUNTERS_REQ(request, p->ep, wr.send_flags);
	VERBS_DBG(FI_LOG_EP_DATA, "posted %d bytes, conn %p, tag 0x%llx\n",
		sge.length, request->conn, request->tag);
	int ret = ibv_post_send(conn->qp, &wr, &bad_wr);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_EP_DATA, "ibv_post_send", errno);
		assert(0);
	};

	request->state.eager = FI_IBV_STATE_EAGER_SEND_WAIT4LC;
	request->state.rndv = FI_IBV_STATE_RNDV_SEND_WAIT4ACK;

	FI_IBV_RDM_TAGGED_HANDLER_LOG_OUT();
	return FI_EP_RDM_HNDL_SUCCESS;
}

static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_rndv_rts_lc(struct fi_ibv_rdm_tagged_request *request,
			      void *data)
{
	FI_IBV_RDM_TAGGED_HANDLER_LOG_IN();
	enum fi_rdm_tagged_req_hndl_ret ret = FI_EP_RDM_HNDL_SUCCESS;

	assert(((request->state.eager == FI_IBV_STATE_EAGER_SEND_WAIT4LC) &&
		(request->state.rndv == FI_IBV_STATE_RNDV_SEND_WAIT4ACK)) ||
	       ((request->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE) &&
		(request->state.rndv == FI_IBV_STATE_RNDV_SEND_END)));
	assert(request->conn);

	VERBS_DBG(FI_LOG_EP_DATA, "conn %p, tag 0x%llx, len %d\n", request->conn,
		     request->tag, request->len);

	struct fi_ibv_rdm_tagged_send_completed_data *p = data;

	FI_IBV_RDM_TAGGED_DEC_SEND_COUNTERS(request->conn, p->ep);
	request->send_completions_wait--;

	if (request->state.eager == FI_IBV_STATE_EAGER_SEND_WAIT4LC) {
		request->state.eager = FI_IBV_STATE_EAGER_SEND_END;
	}

	FI_IBV_RDM_TAGGED_HANDLER_LOG_OUT();

	return ret;
}

static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_rndv_end(struct fi_ibv_rdm_tagged_request *request,
			   void *data)
{
	FI_IBV_RDM_TAGGED_HANDLER_LOG_IN();
	assert(request->state.eager == FI_IBV_STATE_EAGER_SEND_END ||
	       (request->state.eager == FI_IBV_STATE_EAGER_SEND_WAIT4LC));
	assert(request->state.rndv == FI_IBV_STATE_RNDV_SEND_WAIT4ACK);

#ifndef NDEBUG
	struct fi_ibv_recv_got_pkt_preprocess_data *p = data;
#endif /* NDEBUG */

	assert((sizeof(struct fi_ibv_rdm_tagged_request *) +
		sizeof(struct fi_ibv_rdm_tagged_header)) == p->arrived_len);
	assert(request->rndv_mr);
	assert(p->rbuf);

	int ret = ibv_dereg_mr(request->rndv_mr);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_EP_DATA, "ibv_dereg_mr", errno);
	}

	request->state.rndv = FI_IBV_STATE_RNDV_SEND_END;

	fi_ibv_rdm_tagged_move_to_ready_queue(request);

	FI_IBV_RDM_TAGGED_HANDLER_LOG_OUT();

	return FI_EP_RDM_HNDL_SUCCESS;
}

static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_init_recv_request(struct fi_ibv_rdm_tagged_request *request,
				    void *data)
{
	FI_IBV_RDM_TAGGED_HANDLER_LOG_IN();

	struct fi_ibv_rdm_tagged_recv_start_data *p = data;

	request->tag = p->tag;
	request->tagmask = p->tagmask;
	request->context = p->context;
	request->context->internal[0] = (void *)request;
	request->dest_buf = p->dest_addr;
	request->conn = p->conn;
	request->len = p->data_len;
	request->state.eager = FI_IBV_STATE_EAGER_RECV_WAIT4PKT;
	request->state.rndv = FI_IBV_STATE_RNDV_NOT_USED;

	VERBS_DBG(FI_LOG_EP_DATA, "conn %p, tag 0x%llx, len %d\n", request->conn,
		     request->tag, request->len);

	struct fi_verbs_rdm_tagged_request_minfo minfo = {
		.conn = p->conn,
		.tag = p->tag,
		.tagmask = p->tagmask
	};

	struct dlist_entry *found_entry =
	    dlist_find_first_match(&fi_ibv_rdm_tagged_recv_unexp_queue,
				   fi_ibv_rdm_tagged_req_match_by_info2,
				   &minfo);

	if (found_entry) {

		struct fi_ibv_rdm_tagged_request *found_request =
		    container_of(found_entry, struct fi_ibv_rdm_tagged_request,
				 queue_entry);

		if (request->len < found_request->len) {
			VERBS_INFO(FI_LOG_EP_DATA,
				"RECV TRUNCATE, unexp len %d, "
				"req->len=%d, conn %p, tag 0x%llx, "
				"tagmask %llx\n",
				found_request->len, request->len, request->conn,
				request->tag, request->tagmask);
			abort();
		}

		request->tag = found_request->tag;
		request->len = found_request->len;
		request->conn = found_request->conn;
		request->unexp_rbuf = found_request->unexp_rbuf;
		request->state.eager = found_request->state.eager;
		request->state.rndv = found_request->state.rndv;

		assert(request->state.eager ==
		       FI_IBV_STATE_EAGER_RECV_WAIT4RECV);
		if (request->state.rndv != FI_IBV_STATE_RNDV_NOT_USED) {
			assert(request->state.rndv ==
			       FI_IBV_STATE_RNDV_RECV_WAIT4RES);

			request->rndv_remote_key =
			    found_request->rndv_remote_key;
			request->rndv_id = found_request->rndv_id;
			request->src_addr = found_request->src_addr;
		}
		VERBS_DBG(FI_LOG_EP_DATA, "found req: len = %d, eager_state = %s, "
			     "rndv_state = %s \n",
			     found_request->len,
			     fi_ibv_rdm_tagged_req_eager_state_to_str
			     (found_request->state.eager),
			     fi_ibv_rdm_tagged_req_rndv_state_to_str
			     (found_request->state.rndv));

		FI_IBV_RDM_TAGGED_HANDLER_LOG();

		fi_ibv_rdm_tagged_remove_from_unexp_queue(found_request);

		assert(found_request->send_completions_wait == 0);
		FI_IBV_RDM_TAGGED_DBG_REQUEST("to_pool: ", request,
					      FI_LOG_DEBUG);
		fi_ibv_mem_pool_return(&found_request->mpe,
				       &fi_ibv_rdm_tagged_request_pool);

		if (request->state.rndv == FI_IBV_STATE_RNDV_RECV_WAIT4RES) {
			request->state.eager = FI_IBV_STATE_EAGER_RECV_END;
			fi_ibv_rdm_tagged_move_to_postponed_queue(request);
		}
#if ENABLE_DEBUG
		request->conn->unexp_counter++;
#endif // ENABLE_DEBUG

	} else {

#if ENABLE_DEBUG
		if (request->conn) {
			request->conn->exp_counter++;
		}
#endif // ENABLE_DEBUG

		fi_ibv_rdm_tagged_move_to_posted_queue(request, p->ep);
	}

	FI_IBV_RDM_TAGGED_HANDLER_LOG_OUT();
	return FI_EP_RDM_HNDL_SUCCESS;
}

static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_init_unexp_recv_request(
		struct fi_ibv_rdm_tagged_request *request, void *data)
{
	FI_IBV_RDM_TAGGED_HANDLER_LOG_IN();

	assert(request->state.eager == FI_IBV_STATE_EAGER_BEGIN);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED);

	struct fi_ibv_recv_got_pkt_preprocess_data *p = data;
	struct fi_ibv_rdm_tagged_buf *rbuf = p->rbuf;

	FI_IBV_RDM_TAGGED_HANDLER_LOG();

	switch (p->pkt_type) {
	case FI_IBV_RDM_EAGER_PKT:
		FI_IBV_RDM_TAGGED_HANDLER_LOG();
		assert(p->arrived_len <= p->ep->rndv_threshold);

		request->tag = rbuf->header.tag;
		request->conn = p->conn;
		request->len = p->arrived_len -
		    sizeof(struct fi_ibv_rdm_tagged_header);
		if (request->len > 0) {
			request->unexp_rbuf =
			    (struct fi_ibv_rdm_tagged_unexp_rbuff *)
			    fi_verbs_mem_pool_get
			    (&fi_ibv_rdm_tagged_unexp_buffers_pool);
			memcpy(request->unexp_rbuf->payload, rbuf->payload,
				request->len);
		} else {
			request->unexp_rbuf = NULL;
		}
		request->imm = p->imm_data;
		request->state.eager = FI_IBV_STATE_EAGER_RECV_WAIT4RECV;
		break;
	case FI_IBV_RDM_RNDV_RTS_PKT:
		FI_IBV_RDM_TAGGED_HANDLER_LOG();
		assert(p->arrived_len ==
		       sizeof(struct fi_ibv_rdm_tagged_rndv_header));
		struct fi_ibv_rdm_tagged_rndv_header *h =
		    (struct fi_ibv_rdm_tagged_rndv_header *)rbuf;

		request->tag = h->base.tag;
		request->src_addr = (void *)h->src_addr;
		request->rndv_remote_key = h->mem_key;
		request->len = h->len;
		request->imm = p->imm_data;
		request->conn = p->conn;
		request->rndv_id = h->id;
		request->state.eager = FI_IBV_STATE_EAGER_RECV_WAIT4RECV;
		request->state.rndv = FI_IBV_STATE_RNDV_RECV_WAIT4RES;
		break;
	case FI_IBV_RDM_RNDV_ACK_PKT:
		FI_IBV_RDM_TAGGED_DBG_REQUEST("Unexpected RNDV ack!!!",
					      request, FI_LOG_INFO);
		assert(0);
		break;
	default:
		VERBS_INFO(FI_LOG_EP_DATA,
			"Got unknown unexpected pkt: %" PRIu64 "\n",
			p->pkt_type);
		assert(0);
	}

	fi_ibv_rdm_tagged_move_to_unexpected_queue(request);

	FI_IBV_RDM_TAGGED_HANDLER_LOG_OUT();
	return FI_EP_RDM_HNDL_SUCCESS;
}

static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_eager_recv_got_pkt(struct fi_ibv_rdm_tagged_request *request,
				     void *data)
{
	FI_IBV_RDM_TAGGED_HANDLER_LOG_IN();
	struct fi_ibv_recv_got_pkt_preprocess_data *p = data;
	struct fi_ibv_rdm_tagged_buf *rbuf = p->rbuf;
	assert(request->state.eager == FI_IBV_STATE_EAGER_RECV_WAIT4PKT);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED);

	switch (p->pkt_type) {
	case FI_IBV_RDM_EAGER_PKT:
		assert(p->pkt_type == FI_IBV_RDM_EAGER_PKT);
		assert(p->arrived_len <= p->ep->rndv_threshold);
		assert(p->arrived_len - sizeof(rbuf->header) <= request->len);

		request->tag = rbuf->header.tag;
		request->conn = p->conn;
		request->len = p->arrived_len - sizeof(rbuf->header);
		request->expected_recv_buf = rbuf->payload;
		request->imm = p->imm_data;

		if (request->dest_buf) {
			assert(request->expected_recv_buf);
			memcpy(request->dest_buf,
				request->expected_recv_buf, request->len);
		}

		request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;
		fi_ibv_rdm_tagged_move_to_ready_queue(request);
		FI_IBV_RDM_TAGGED_HANDLER_LOG();
		break;
	case FI_IBV_RDM_RNDV_RTS_PKT:
		assert(p->arrived_len ==
		       sizeof(struct fi_ibv_rdm_tagged_rndv_header));

		struct fi_ibv_rdm_tagged_rndv_header *rndv_header =
		    (struct fi_ibv_rdm_tagged_rndv_header *)rbuf;

		request->tag = rndv_header->base.tag;
		request->src_addr = (void *)rndv_header->src_addr;
		request->rndv_remote_key = rndv_header->mem_key;
		request->len = rndv_header->len;
		request->imm = p->imm_data;
		request->conn = p->conn;
		request->rndv_id = rndv_header->id;

		fi_ibv_rdm_tagged_move_to_postponed_queue(request);

		request->state.eager = FI_IBV_STATE_EAGER_RECV_END;
		request->state.rndv = FI_IBV_STATE_RNDV_RECV_WAIT4RES;
		FI_IBV_RDM_TAGGED_HANDLER_LOG();
		break;
	}

	FI_IBV_RDM_TAGGED_HANDLER_LOG_OUT();
	return FI_EP_RDM_HNDL_SUCCESS;
}

static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_eager_recv_got_unexp_pkt(
		struct fi_ibv_rdm_tagged_request *request, void *data)
{
	FI_IBV_RDM_TAGGED_HANDLER_LOG_IN();
	assert(request->state.eager == FI_IBV_STATE_EAGER_RECV_WAIT4RECV);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_NOT_USED);

	if (request->dest_buf && request->len != 0) {
		memcpy(request->dest_buf, request->unexp_rbuf->payload,
			request->len);
	}

	if (request->unexp_rbuf) {
		fi_ibv_mem_pool_return(&request->unexp_rbuf->mpe,
				       &fi_ibv_rdm_tagged_unexp_buffers_pool);
		request->unexp_rbuf = NULL;
	}

	fi_ibv_rdm_tagged_move_to_ready_queue(request);
	request->state.eager = FI_IBV_STATE_EAGER_RECV_END;

	FI_IBV_RDM_TAGGED_HANDLER_LOG_OUT();
	return FI_EP_RDM_HNDL_SUCCESS;
}

static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_rndv_recv_post_read(struct fi_ibv_rdm_tagged_request *request,
				      void *data)
{
	FI_IBV_RDM_TAGGED_HANDLER_LOG_IN();
	assert(request->state.eager == FI_IBV_STATE_EAGER_RECV_END);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_RECV_WAIT4RES);

	struct fi_ibv_rdm_tagged_send_ready_data *p = data;

	fi_ibv_rdm_tagged_remove_from_postponed_queue(request);
	VERBS_DBG(FI_LOG_EP_DATA, "\t REQUEST: conn %p, tag 0x%llx, len %d, dest_buf %p, "
		     "src_addr %p, rkey 0x%lx\n",
		     request->conn, (long long unsigned int)request->tag,
		     request->len, request->dest_buf,
		     request->src_addr,
		     (long unsigned int)request->rndv_remote_key);

	struct ibv_send_wr wr = { 0 };
	struct ibv_send_wr *bad_wr = NULL;
	struct ibv_sge sge;
	int ret;

	wr.wr_id = (uintptr_t) request;
	assert(FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(wr.wr_id) == 0);
	wr.opcode = IBV_WR_RDMA_READ;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = 0;
	wr.wr.rdma.remote_addr = (uintptr_t) request->src_addr;
	wr.wr.rdma.rkey = request->rndv_remote_key;

	request->rndv_mr = ibv_reg_mr(p->ep->domain->pd,
				      request->dest_buf,
				      request->len, IBV_ACCESS_LOCAL_WRITE);
	assert(request->rndv_mr);

	sge.addr = (uintptr_t) request->dest_buf;
	sge.length = request->len;
	sge.lkey = request->rndv_mr->lkey;

	FI_IBV_RDM_TAGGED_INC_SEND_COUNTERS_REQ(request, p->ep, wr.send_flags);
	VERBS_DBG(FI_LOG_EP_DATA, "posted %d bytes, conn %p, tag 0x%llx\n", sge.length,
		     request->conn, request->tag);
	ret = ibv_post_send(request->conn->qp, &wr, &bad_wr);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_EP_DATA, "ibv_post_send", errno);
		assert(0);
		return FI_EP_RDM_HNDL_ERROR;
	};

	request->state.eager = FI_IBV_STATE_EAGER_RECV_END;
	request->state.rndv = FI_IBV_STATE_RNDV_RECV_WAIT4LC;

	FI_IBV_RDM_TAGGED_HANDLER_LOG_OUT();
	return FI_EP_RDM_HNDL_SUCCESS;
}

static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_rndv_recv_read_lc(struct fi_ibv_rdm_tagged_request *request,
				    void *data)
{
	FI_IBV_RDM_TAGGED_HANDLER_LOG_IN();

	struct fi_ibv_rdm_tagged_send_completed_data *p = data;

	assert(request->len > (p->ep->rndv_threshold
			       - sizeof(struct fi_ibv_rdm_tagged_header)));
	assert(request->state.eager == FI_IBV_STATE_EAGER_RECV_END);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_RECV_WAIT4LC);

	FI_IBV_RDM_TAGGED_DEC_SEND_COUNTERS(request->conn, p->ep);

	struct fi_ibv_rdm_tagged_conn *conn = request->conn;
	assert(request->sbuf);

	int ret = 0;
	struct ibv_send_wr wr = { 0 };
	struct ibv_sge sge = { 0 };
	struct ibv_send_wr *bad_wr = NULL;

	struct fi_ibv_rdm_tagged_buf *sbuf = request->sbuf;
	sbuf->header.tag = 0;
	sbuf->header.service_tag = 0;
	FI_IBV_RDM_SET_PKTTYPE(sbuf->header.service_tag,
			       FI_IBV_RDM_RNDV_ACK_PKT);

	memcpy(sbuf->payload, &(request->rndv_id), sizeof(request->rndv_id));

	wr.wr_id = ((uint64_t) (uintptr_t) (void *) request);
	assert(FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(wr.wr_id) == 0);

	wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	wr.send_flags = IBV_SEND_INLINE;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.wr.rdma.remote_addr =
	    fi_ibv_rdm_tagged_get_remote_addr(conn, request->sbuf);
	wr.wr.rdma.rkey = conn->remote_rbuf_rkey;
	wr.imm_data =
	    fi_ibv_rdm_tagged_get_buff_service_data(request->sbuf)->seq_number;

	sge.addr = (uintptr_t) sbuf;
	sge.length = sizeof(struct fi_ibv_rdm_tagged_header) +
	    sizeof(request->rndv_id);
	sge.lkey = conn->s_mr->lkey;

	FI_IBV_RDM_TAGGED_INC_SEND_COUNTERS_REQ(request, p->ep, wr.send_flags);
	VERBS_DBG(FI_LOG_EP_DATA,
		"posted %d bytes, conn %p, tag 0x%llx, request %p\n",
		sge.length, request->conn, request->tag, request);
	ret = ibv_post_send(conn->qp, &wr, &bad_wr);
	if (ret == 0) {
		assert(request->rndv_mr);
		ibv_dereg_mr(request->rndv_mr);
		VERBS_DBG(FI_LOG_EP_DATA,
			"SENDING RNDV ACK: conn %p, sends_outgoing = %d, "
			"pend_send = %d\n", conn, conn->sends_outgoing,
			p->ep->pend_send);
	} else {
		VERBS_INFO_ERRNO(FI_LOG_EP_DATA, "ibv_post_send", errno);
		assert(0);
	}
	request->state.eager = FI_IBV_STATE_EAGER_SEND_WAIT4LC;
	request->state.rndv = FI_IBV_STATE_RNDV_RECV_END;
	fi_ibv_rdm_tagged_move_to_ready_queue(request);

	FI_IBV_RDM_TAGGED_HANDLER_LOG_OUT();
	return FI_EP_RDM_HNDL_SUCCESS;
}

static enum fi_rdm_tagged_req_hndl_ret
fi_ibv_rdm_tagged_rndv_recv_ack_lc(struct fi_ibv_rdm_tagged_request *request,
				   void *data)
{
	FI_IBV_RDM_TAGGED_HANDLER_LOG_IN();
	assert(request->state.eager == FI_IBV_STATE_EAGER_SEND_WAIT4LC ||
	       request->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE);
	assert(request->state.rndv == FI_IBV_STATE_RNDV_RECV_END);

	struct fi_ibv_rdm_tagged_send_completed_data *p = data;
	FI_IBV_RDM_TAGGED_DEC_SEND_COUNTERS(request->conn, p->ep);

	if (request->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE) {
		FI_IBV_RDM_TAGGED_DBG_REQUEST("to_pool: ", request,
					      FI_LOG_DEBUG);
		fi_ibv_mem_pool_return(&request->mpe,
				       &fi_ibv_rdm_tagged_request_pool);
	} else {
		request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;
		request->state.rndv = FI_IBV_STATE_RNDV_RECV_END;
	}

	FI_IBV_RDM_TAGGED_HANDLER_LOG_OUT();
	return FI_EP_RDM_HNDL_SUCCESS;
}

char *fi_ibv_rdm_tagged_req_eager_state_to_str(
		enum fi_ibv_rdm_tagged_request_eager_state state)
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
	case FI_IBV_STATE_EAGER_RECV_END:
		return "STATE_EAGER_RECV_END";
	case FI_IBV_STATE_EAGER_READY_TO_FREE:
		return "STATE_EAGER_READY_TO_FREE";

	case FI_IBV_STATE_EAGER_COUNT:
		return "STATE_EAGER_COUNT";
	default:
		return "STATE_EAGER_UNKNOWN!!!";
	}
}

char *fi_ibv_rdm_tagged_req_rndv_state_to_str(
		enum fi_ibv_rdm_tagged_request_rndv_state state)
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
	case FI_IBV_STATE_RNDV_COUNT:
		return "STATE_RNDV_COUNT";
	default:
		return "STATE_RNDV_UNKNOWN!!!";
	}
}

char *fi_ibv_rdm_tagged_req_event_to_str(
		enum fi_ibv_rdm_tagged_request_event event)
{
	switch (event) {
	case FI_IBV_EVENT_SEND_START:
		return "EVENT_SEND_START";
	case FI_IBV_EVENT_SEND_READY:
		return "EVENT_SEND_READY";
	case FI_IBV_EVENT_SEND_COMPLETED:
		return "EVENT_SEND_COMPLETED";
	case FI_IBV_EVENT_SEND_GOT_CTS:
		return "EVENT_SEND_GOT_CTS";
	case FI_IBV_EVENT_SEND_GOT_LC:
		return "EVENT_SEND_GOT_LC";

	case FI_IBV_EVENT_RECV_START:
		return "EVENT_RECV_START";
	case FI_IBV_EVENT_RECV_GOT_PKT_PREPROCESS:
		return "EVENT_RECV_GOT_PKT_PREPROCESS";
	case FI_IBV_EVENT_RECV_GOT_PKT_PROCESS:
		return "EVENT_RECV_GOT_PKT_PROCESS";
	case FI_IBV_EVENT_RECV_GOT_ACK:
		return "EVENT_RECV_GOT_ACK";

	case FI_IBV_EVENT_COUNT:
		return "EVENT_COUNT";
	default:
		return "EVENT_UNKNOWN!!!";
	}
}
