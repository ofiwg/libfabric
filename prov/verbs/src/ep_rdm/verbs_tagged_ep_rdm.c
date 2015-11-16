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


#if defined(__ICC) || defined(__INTEL_COMPILER) || \
 defined(__GNUC__) || defined(__GNUG__)
#include "xmmintrin.h"
#endif /* ICC || GCC */


DEFINE_LIST(fi_ibv_rdm_tagged_request_ready_queue);
DEFINE_LIST(fi_ibv_rdm_tagged_recv_posted_queue);
DEFINE_LIST(fi_ibv_rdm_tagged_recv_unexp_queue);
DEFINE_LIST(fi_ibv_rdm_tagged_send_postponed_queue);

struct fi_ibv_mem_pool fi_ibv_rdm_tagged_request_pool;
struct fi_ibv_mem_pool fi_ibv_rdm_tagged_postponed_pool;
struct fi_ibv_mem_pool fi_ibv_rdm_tagged_unexp_buffers_pool;

static inline int fi_ibv_rdm_tagged_poll_send(struct fi_ibv_rdm_ep *ep);
static inline int fi_ibv_rdm_tagged_poll_recv(struct fi_ibv_rdm_ep *ep);

static inline int
fi_ibv_rdm_tagged_init_request_sbuf(struct fi_ibv_rdm_tagged_request *request,
				    struct fi_ibv_rdm_ep *ep);

static inline void *fi_ibv_rdm_tagged_get_sbuf(struct fi_ibv_rdm_tagged_conn
					       *conn, struct fi_ibv_rdm_ep *ep);

int fi_ibv_rdm_tagged_prepare_send_resources(
	struct fi_ibv_rdm_tagged_request *request, struct fi_ibv_rdm_ep *ep)
{
	if (request->conn->state != FI_VERBS_CONN_ESTABLISHED) {
		pthread_mutex_lock(&ep->cm_lock);
		if (request->conn->state == FI_VERBS_CONN_ALLOCATED) {
			fi_ibv_rdm_start_connection(ep, request->conn);
		}
		pthread_mutex_unlock(&ep->cm_lock);
		usleep(1000);
		return 0;
	}
#if !ENABLE_DEBUG
	return (!SEND_RESOURCES_IS_BUSY(request->conn, ep)) ?
		fi_ibv_rdm_tagged_init_request_sbuf(request, ep) : 0;
#else // ENABLE_DEBUG
	int res = FI_IBV_RDM_TAGGED_SENDS_OUTGOING_ARE_LIMITED(request->conn, ep);
	if (res) {
		FI_IBV_RDM_TAGGED_DBG_REQUEST
		    ("failed because SENDS_OUTGOING_ARE_LIMITED", request,
		     FI_LOG_DEBUG);
		return !res;
	}
	res = PEND_SEND_IS_LIMITED(ep);
	if (res) {
		FI_IBV_RDM_TAGGED_DBG_REQUEST
		    ("failed because PEND_SEND_IS_LIMITED", request,
		     FI_LOG_DEBUG);
		return !res;
	}
	res = fi_ibv_rdm_tagged_init_request_sbuf(request, ep);
	if (!res) {
		FI_IBV_RDM_TAGGED_DBG_REQUEST("failed because sbuf error",
					      request, FI_LOG_DEBUG);
		return res;
	}
	return res;
#endif // ENABLE_DEBUG
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
	memcpy(addr, ep->my_rdm_addr, FI_IBV_RDM_DFLT_ADDRLEN);
	ep->fi_ibv_rdm_addrlen = *addrlen;

	return 0;
}

static ssize_t fi_ibv_rdm_tagged_recvfrom(struct fid_ep *ep_fid, void *buf,
					  size_t len, void *desc,
					  fi_addr_t src_addr, uint64_t tag,
					  uint64_t ignore, void *context)
{
	int ret = 0;

	struct fi_ibv_rdm_tagged_request *request =
	    (struct fi_ibv_rdm_tagged_request *)
	    fi_verbs_mem_pool_get(&fi_ibv_rdm_tagged_request_pool);
	fi_ibv_rdm_tagged_zero_request(request);
	FI_IBV_RDM_TAGGED_DBG_REQUEST("get_from_pool: ", request, FI_LOG_DEBUG);

	struct fi_ibv_rdm_tagged_conn *conn = (src_addr == FI_ADDR_UNSPEC)
	    ? NULL : (struct fi_ibv_rdm_tagged_conn *)src_addr;
	struct fi_ibv_rdm_ep *ep = container_of(ep_fid,
						struct fi_ibv_rdm_ep, ep_fid);

	{
		struct fi_ibv_rdm_tagged_recv_start_data req_data = {
			.tag = tag,
			.tagmask = ~ignore,
			.context = context,
			.dest_addr = buf,
			.conn = conn,
			.ep = ep,
			.data_len = len
		};

		ret = fi_ibv_rdm_tagged_req_hndl(request,
				FI_IBV_EVENT_RECV_START, &req_data);
		if (ret || request->state.eager ==
		    FI_IBV_STATE_EAGER_RECV_WAIT4PKT) {
			goto out;
		}
	}

	VERBS_DBG(FI_LOG_EP_DATA,
	    "fi_recvfrom: conn %p, tag 0x%llx, len %d, rbuf %p, fi_ctx %p, "
	     "pend_recv %d\n",
	     conn, (long long unsigned int)tag, (int)len, buf, context,
	     ep->pend_recv);

	struct fi_ibv_recv_got_pkt_process_data data = {
		.ep = ep
	};

	if (request->state.rndv == FI_IBV_STATE_RNDV_RECV_WAIT4RES) {
		if (fi_ibv_rdm_tagged_prepare_send_resources(request, ep)) {
			ret =
			    fi_ibv_rdm_tagged_req_hndl(request,
						       FI_IBV_EVENT_SEND_READY,
						       &data);
		}
	} else {
		ret =
		    fi_ibv_rdm_tagged_req_hndl(request, FI_IBV_EVENT_RECV_START,
					       &data);
	}

out:
	return ret;
}

static ssize_t fi_ibv_rdm_tagged_recvmsg(struct fid_ep *ep,
					 const struct fi_msg_tagged *msg,
					 uint64_t flags)
{
	void *buf;
	size_t len;

	if (!msg || msg->iov_count > 1)
		return -FI_EINVAL;

	if (msg->iov_count) {
		buf = msg->msg_iov[0].iov_base;
		len = msg->msg_iov[0].iov_len;
	} else {
		buf = NULL;
		len = 0;
	}

	return fi_ibv_rdm_tagged_recvfrom(ep, buf, len,
					  msg->desc ? msg->desc[0] : NULL,
					  msg->addr, msg->tag, msg->ignore,
					  msg->context);
}

static ssize_t fi_ibv_rdm_tagged_sendto(struct fid_ep *fid, const void *buf,
					size_t len, void *desc,
					fi_addr_t dest_addr,
					uint64_t tag, void *context);

static inline ssize_t fi_ibv_rdm_tagged_inject(struct fid_ep *fid,
					       const void *buf,
					       size_t len,
					       fi_addr_t dest_addr,
					       uint64_t tag)
{
	struct fi_ibv_rdm_tagged_conn *conn =
	    (struct fi_ibv_rdm_tagged_conn *) dest_addr;
	struct fi_ibv_rdm_ep *ep =
	    container_of(fid, struct fi_ibv_rdm_ep, ep_fid);

	const size_t size = len + sizeof(struct fi_ibv_rdm_tagged_header);

	if (size > ep->max_inline_rc) {
		goto out_again;
	}

	if (conn->state != FI_VERBS_CONN_ESTABLISHED) {
		goto out_connect;
	}

	const int in_order = (conn->postponed_entry) ? 0 : 1;
	if (in_order) {
		void *raw_sbuf = fi_ibv_rdm_tagged_get_sbuf(conn, ep);
		if (raw_sbuf) {
			struct ibv_sge sge = { 0 };
			struct ibv_send_wr wr = { 0 };
			struct ibv_send_wr *bad_wr = NULL;
			wr.wr_id = FI_IBV_RDM_PACK_SERVICE_WR(conn);
			wr.sg_list = &sge;
			wr.num_sge = 1;
			wr.wr.rdma.remote_addr =
			    fi_ibv_rdm_tagged_get_remote_addr(conn, raw_sbuf);
			wr.wr.rdma.rkey = conn->remote_rbuf_rkey;
			wr.send_flags = IBV_SEND_INLINE;
			wr.imm_data =
			    fi_ibv_rdm_tagged_get_buff_service_data(raw_sbuf)->
			    seq_number;
			wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;

			sge.addr = (uintptr_t) (raw_sbuf);
			sge.length = size;
			sge.lkey = conn->s_mr->lkey;

			struct fi_ibv_rdm_tagged_buf *sbuf =
			    (struct fi_ibv_rdm_tagged_buf *)raw_sbuf;
			void *payload = (void *)&(sbuf->payload[0]);

			sbuf->header.tag = tag;
			sbuf->header.service_tag = 0;
			FI_IBV_RDM_SET_PKTTYPE(sbuf->header.service_tag,
					       FI_IBV_RDM_EAGER_PKT);
			if (buf && (len > 0)) {
				memcpy(payload, buf, len);
			}
			//INCREASE_SEND_COUNTERS(conn, ep, wr.send_flags);
			VERBS_DBG(FI_LOG_EP_DATA,
				"posted %d bytes, conn %p, tag 0x%llx\n",
				sge.length, conn, tag);
			if (!ibv_post_send(conn->qp, &wr, &bad_wr)) {
				return FI_SUCCESS;
			}
		}
	}

out_again:
	fi_ibv_rdm_tagged_poll(ep);
	return FI_EAGAIN;

out_connect:
	pthread_mutex_lock(&ep->cm_lock);
	if (conn->state == FI_VERBS_CONN_ALLOCATED) {
		fi_ibv_rdm_start_connection(ep, conn);
	}
	pthread_mutex_unlock(&ep->cm_lock);
	goto out_again;
}

static ssize_t fi_ibv_rdm_tagged_senddatato(struct fid_ep *fid, const void *buf,
					    size_t len, void *desc,
					    uint64_t data,
					    fi_addr_t dest_addr, uint64_t tag,
					    void *context)
{
	struct fi_ibv_rdm_tagged_conn *conn =
	    (struct fi_ibv_rdm_tagged_conn *) dest_addr;

#if defined(__ICC) || defined(__INTEL_COMPILER) || \
    defined(__GNUC__) || defined(__GNUG__)
	_mm_prefetch((const char *)
		     fi_ibv_rdm_tagged_get_buff_service_data(conn->sbuf_head),
		     _MM_HINT_T0);
#endif /* ICC || GCC */

	struct fi_ibv_rdm_ep *ep =
	    container_of(fid, struct fi_ibv_rdm_ep, ep_fid);
	struct fi_ibv_rdm_tagged_request *request =
	    (struct fi_ibv_rdm_tagged_request *)
	    fi_verbs_mem_pool_get(&fi_ibv_rdm_tagged_request_pool);
	fi_ibv_rdm_tagged_zero_request(request);
	FI_IBV_RDM_TAGGED_DBG_REQUEST("get_from_pool: ", request, FI_LOG_DEBUG);

	int ret = 0;

	int in_order = (conn->postponed_entry) ? 0 : 1;

	{
		struct fi_ibv_rdm_tagged_send_start_data req_data = {
			.src_addr = buf,
			.context = context,
			.conn = conn,
			.tag = tag,
			.data_len = len,
			.rndv_threshold = ep->rndv_threshold,
			.imm = (uint32_t) data
		};
		ret =
		    fi_ibv_rdm_tagged_req_hndl(request, FI_IBV_EVENT_SEND_START,
					       &req_data);
		if (ret) {
			goto out;
		}
	}

	if (in_order && fi_ibv_rdm_tagged_prepare_send_resources(request, ep)) {
		struct fi_ibv_rdm_tagged_send_ready_data req_data = {.ep = ep };
		ret =
		    fi_ibv_rdm_tagged_req_hndl(request, FI_IBV_EVENT_SEND_READY,
					       &req_data);
	}

out:
	return ret;
}

static ssize_t fi_ibv_rdm_tagged_sendto(struct fid_ep *fid, const void *buf,
					size_t len, void *desc,
					fi_addr_t dest_addr,
					uint64_t tag, void *context)
{
	return fi_ibv_rdm_tagged_senddatato(fid, buf, len, desc, 0, dest_addr,
					    tag, context);
}

struct fi_ops_tagged fi_ibv_rdm_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = fi_ibv_rdm_tagged_recvfrom,
	.senddata = fi_ibv_rdm_tagged_senddatato,
	.send = fi_ibv_rdm_tagged_sendto,
	.recvmsg = fi_ibv_rdm_tagged_recvmsg,
	.inject = fi_ibv_rdm_tagged_inject,
	.injectdata = fi_no_tagged_injectdata,
};

struct fi_ops_cm fi_ibv_rdm_tagged_ep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.getname = fi_ibv_rdm_tagged_getname,	/* TODO */
};

static inline void
fi_ibv_rdm_tagged_process_recv(struct fi_ibv_rdm_ep *ep,
			       struct fi_ibv_rdm_tagged_conn *conn,
			       int arrived_len,
			       uint32_t imm_data,
			       struct fi_ibv_rdm_tagged_buf *rbuf)
{
	struct fi_ibv_rdm_tagged_request *request = NULL;

	int pkt_type = FI_IBV_RDM_GET_PKTTYPE(rbuf->header.service_tag);

	if (pkt_type == FI_IBV_RDM_RNDV_ACK_PKT) {
		memcpy(&request, rbuf->payload,
		      sizeof(struct fi_ibv_rdm_tagged_request *));
		assert(request);
		VERBS_DBG(FI_LOG_EP_DATA,
			"GOT RNDV ACK from conn %p, id %d\n", conn, request);
	} else {
		struct fi_verbs_rdm_tagged_request_minfo minfo = {
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

			const int data_len = arrived_len -
				sizeof(struct fi_ibv_rdm_tagged_header);

			if (found_request->len < data_len) {
				FI_IBV_ERROR
				    ("%s: %d RECV TRUNCATE, data_len=%d, posted_len=%d, "
				     "conn %p, tag 0x%llx, tagmask %llx\n",
				     __FUNCTION__, __LINE__, data_len,
				     found_request->len, found_request->conn,
				     (long long unsigned int)found_request->tag,
				     (long long unsigned int)found_request->
				     tagmask);
				abort();
			}

			fi_ibv_rdm_tagged_remove_from_posted_queue
			    (found_request, ep);
			request = found_request;
		} else {
			request = (struct fi_ibv_rdm_tagged_request *)
			    fi_verbs_mem_pool_get
			    (&fi_ibv_rdm_tagged_request_pool);
			fi_ibv_rdm_tagged_zero_request(request);

			FI_IBV_RDM_TAGGED_DBG_REQUEST("get_from_pool: ",
						      request, FI_LOG_DEBUG);
		}
	}

	struct fi_ibv_recv_got_pkt_preprocess_data p = {
		.conn = conn,
		.ep = ep,
		.rbuf = rbuf,
		.arrived_len = arrived_len,
		.pkt_type = pkt_type,
		.imm_data = imm_data
	};

	fi_ibv_rdm_tagged_req_hndl(request, FI_IBV_EVENT_RECV_GOT_PKT_PROCESS,
				   &p);
}

static inline void
fi_ibv_rdm_tagged_release_remote_sbuff(struct fi_ibv_rdm_tagged_conn *conn,
				       struct fi_ibv_rdm_ep *ep, int seq_num)
{
	// not needed till nobody use service data of recv buffers
	char *buff = conn->sbuf_ack_head + seq_num * ep->buff_len;
	fi_ibv_rdm_tagged_set_buffer_status(buff, BUF_STATUS_FREE);
#if ENABLE_DEBUG
	char *rbuf = (char *)fi_ibv_rdm_tagged_get_rbuf(conn, ep, seq_num);
	memset(rbuf, 0,
	      ep->buff_len - FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE);
#endif // ENABLE_DEBUG
	struct ibv_sge sge;
	sge.addr =
	    (uintptr_t) &fi_ibv_rdm_tagged_get_buff_service_data(buff)->status;
	sge.length = sizeof(enum fi_ibv_rdm_tagged_buffer_status);
	sge.lkey = conn->r_mr->lkey;

	struct ibv_send_wr *bad_wr = NULL;
	struct ibv_send_wr wr = { 0 };

	wr.wr_id = FI_IBV_RDM_PACK_SERVICE_WR(conn);
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.wr.rdma.remote_addr =
	    (uintptr_t) &fi_ibv_rdm_tagged_get_buff_service_data(
			    conn->remote_sbuf_head + seq_num * ep->buff_len)->status;
	wr.wr.rdma.rkey = conn->remote_sbuf_rkey;
	wr.send_flags = IBV_SEND_INLINE;
	wr.opcode = IBV_WR_RDMA_WRITE;	// w/o imm - do not put it into recv
	// completion queue

	FI_IBV_RDM_TAGGED_INC_SEND_COUNTERS(conn, ep, wr.send_flags);
	VERBS_DBG(FI_LOG_EP_DATA,
		"posted %d bytes, remote sbuff released %d\n", sge.length,
		seq_num);
	int ret = ibv_post_send(conn->qp, &wr, &bad_wr);
	if (ret) {
		FI_IBV_ERROR
		    ("ibv_post_send fail, ret %d, errno %d, strerror %s", ret,
		     errno, strerror(errno));
		assert(0);
	};

	if (conn->sends_outgoing > ep->n_buffs) {
		fi_ibv_rdm_tagged_poll_send(ep);
	}
}

static inline
void check_and_repost_receives(struct fi_ibv_rdm_ep *ep,
				struct fi_ibv_rdm_tagged_conn *conn)
{
	if (conn->recv_preposted < ep->recv_preposted_threshold) {
		int to_post = ep->rq_wr_depth - conn->recv_preposted;
		if (fi_ibv_rdm_tagged_repost_receives(conn, ep, to_post) !=
		    to_post) {
			FI_IBV_ERROR("repost recv failed\n");
			abort();
		}
		VERBS_DBG(FI_LOG_EP_DATA,
			"reposted_recvs, posted %d, local_credits %d\n",
			to_post, conn->recv_preposted);
	}
}

static inline void
fi_ibv_rdm_tagged_got_recv_completion(struct fi_ibv_rdm_ep *ep,
				      struct fi_ibv_rdm_tagged_conn *conn,
				      int arrived_len, uint32_t imm_data)
{
	assert(arrived_len > 0);
	conn->recv_preposted--;
	check_and_repost_receives(ep, conn);
	struct fi_ibv_rdm_tagged_buf *rbuf =
	    fi_ibv_rdm_tagged_get_rbuf(conn, ep, imm_data);
	fi_ibv_rdm_tagged_process_recv(ep, conn, arrived_len, imm_data, rbuf);
	fi_ibv_rdm_tagged_release_remote_sbuff(conn, ep, imm_data);
}

static inline int fi_ibv_rdm_tagged_poll_recv(struct fi_ibv_rdm_ep *ep)
{
	const int wc_count = ep->n_buffs;	// TODO: to set from upper level
	struct ibv_wc wc[wc_count];
	int i;
	int ret;

	do {
		ret = ibv_poll_cq(ep->rcq, wc_count, wc);
		for (i = 0; i < ret; ++i) {
			if (wc[i].status == IBV_WC_SUCCESS
			    && wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
				FI_IBV_DBG_OPCODE(wc[i].opcode, "RECV");

				struct fi_ibv_rdm_tagged_conn *conn =
				    (struct fi_ibv_rdm_tagged_conn *)(uintptr_t)
				    wc[i].wr_id;

				fi_ibv_rdm_tagged_got_recv_completion(ep, conn,
								      wc[i].
								      byte_len,
								      wc[i].
								      imm_data);
			} else {
				goto wc_error;
			}
		}
	} while (ret == wc_count);

	if (ret >= 0)		// Success
	{
		return 0;
	}

wc_error:
	FI_IBV_ERROR("%s: ibv_poll_cq returned %d\n", __FUNCTION__, ret);

	if (wc[i].status != IBV_WC_SUCCESS) {
		FI_IBV_ERROR("%s: got ibv_wc.status = %d:%s\n",
			     __FUNCTION__, wc[i].status,
			     ibv_wc_status_str(wc[i].status));
		assert(0);
		return FI_EOTHER;
	}

	if (wc[i].opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
		FI_IBV_ERROR("%s: got ibv_wc[i].opcode = %d\n",
			     __FUNCTION__, wc[i].opcode);
	}

	assert(0);

	return FI_EOTHER;
}

static inline int fi_ibv_rdm_tagged_poll_send(struct fi_ibv_rdm_ep *ep)
{
	const int wc_count = ep->n_buffs;
	struct ibv_wc wc[wc_count];
	int i = 0;
	int ret = 0;

	if (ep->total_outgoing_send > 0) {
		do {
			ret = ibv_poll_cq(ep->scq, wc_count, wc);
			for (i = 0; i < ret; ++i) {
				if (wc[i].status == IBV_WC_SUCCESS) {
					if (FI_IBV_RDM_CHECK_SERVICE_WR_FLAG
					    (wc[i].wr_id)) {
						VERBS_DBG(FI_LOG_EP_DATA,
							"CQ COMPL: SEND -> 0x1\n");
						struct fi_ibv_rdm_tagged_conn
						    *conn =
						    (struct
						     fi_ibv_rdm_tagged_conn *)
						    FI_IBV_RDM_UNPACK_SERVICE_WR
						    (wc[i].wr_id);
						FI_IBV_RDM_TAGGED_DEC_SEND_COUNTERS
						    (conn, ep);
					} else {
						FI_IBV_DBG_OPCODE(wc[i].opcode,
								  "SEND");
						struct fi_ibv_rdm_tagged_request
						    *request =
						    (struct
						     fi_ibv_rdm_tagged_request
						     *)
						    FI_IBV_RDM_UNPACK_WR(wc[i].
									 wr_id);

						struct fi_ibv_rdm_tagged_send_completed_data
						    data = {
							.ep = ep
						};

						fi_ibv_rdm_tagged_req_hndl
						    (request,
						     FI_IBV_EVENT_SEND_GOT_LC,
						     &data);
					}
				} else {
					goto wc_error;
				}
			}
		} while (ret == wc_count);
	}

	struct fi_ibv_rdm_tagged_send_ready_data data = {.ep = ep };
	struct dlist_entry *item;
	dlist_foreach((&fi_ibv_rdm_tagged_send_postponed_queue), item)
		fi_ibv_rdm_tagged_send_postponed_process(item, &data);

	if (ret >= 0)		// Success
	{
		return 0;
	}

wc_error:
	if (ret < 0 || wc[i].status != IBV_WC_SUCCESS) {
		if (ret < 0) {
			FI_IBV_ERROR("%s: ibv_poll_cq returned %d\n",
				     __FUNCTION__, ret);
			assert(0);
		}

		if (wc[i].status != IBV_WC_SUCCESS && ret != 0) {
			struct fi_ibv_rdm_tagged_conn *conn =
			    (FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(wc[i].wr_id))
			    ? (struct fi_ibv_rdm_tagged_conn *)
			    FI_IBV_RDM_UNPACK_SERVICE_WR(wc[i].wr_id)
			    : ((struct fi_ibv_rdm_tagged_request *)wc[i].
			       wr_id)->conn;

			FI_IBV_ERROR
			    ("%s: got ibv_wc.status = %d:%s, pend_send: %d, "
			     "connection: %p\n", __FUNCTION__, wc[i].status,
			     ibv_wc_status_str(wc[i].status), ep->pend_send,
			     (conn));
			assert(0);
		}
	}
	return FI_EOTHER;
}

int fi_ibv_rdm_tagged_poll(struct fi_ibv_rdm_ep *ep)
{
	int ret = fi_ibv_rdm_tagged_poll_send(ep);
	if (ret) {
		return ret;
	}

	return fi_ibv_rdm_tagged_poll_recv(ep);
}

static inline int
fi_ibv_rdm_tagged_init_request_sbuf(struct fi_ibv_rdm_tagged_request *request,
				    struct fi_ibv_rdm_ep *ep)
{
	assert(request->sbuf == NULL);
	request->sbuf = fi_ibv_rdm_tagged_get_sbuf(request->conn, ep);
	return !!request->sbuf;
}

static inline void *fi_ibv_rdm_tagged_get_sbuf(struct fi_ibv_rdm_tagged_conn
					       *conn, struct fi_ibv_rdm_ep *ep)
{
	assert(conn);

#if ENABLE_DEBUG
	{
		int i;
		char s[1024];
		char *p = s;
		sprintf(p, "N:%1d ", ep->n_buffs);
		p += 4;
		for (i = 0; i < ep->n_buffs; ++i, p += 4) {
			sprintf(p, "%1d:%1d ",
				fi_ibv_rdm_tagged_get_buff_service_data(
					conn->sbuf_mem_reg +
					FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE +
					i * ep->buff_len)->seq_number,
				fi_ibv_rdm_tagged_get_buffer_status(
					conn->sbuf_mem_reg +
					FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE +
					i * ep->buff_len));
		}
		VERBS_DBG(FI_LOG_EP_DATA,
			"conn %p sbufs status before: %s\n", conn, s);
	}
#endif // ENABLE_DEBUG

	void *sbuf = NULL;
	if (fi_ibv_rdm_tagged_get_buffer_status(conn->sbuf_head) ==
	    BUF_STATUS_FREE) {
		fi_ibv_rdm_tagged_set_buffer_status(conn->sbuf_head,
						    BUF_STATUS_BUSY);
		sbuf = conn->sbuf_head;
		fi_ibv_rdm_tagged_push_sbuff_head(conn, ep);
	}
#if ENABLE_DEBUG
	assert(sbuf
	       ? (fi_ibv_rdm_tagged_get_buffer_status(sbuf) == BUF_STATUS_BUSY)
	       : 1);
	{
		int i;
		char s[1024];
		char *p = s;
		sprintf(p, "N:%1d ", ep->n_buffs);
		p += 4;
		for (i = 0; i < ep->n_buffs; ++i, p += 4) {
			sprintf(p, "%1d:%1d ",
				fi_ibv_rdm_tagged_get_buff_service_data(
					conn->sbuf_mem_reg +
					FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE +
					i * ep->buff_len)->seq_number,
				fi_ibv_rdm_tagged_get_buffer_status(
					conn->sbuf_mem_reg +
					FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE +
					i * ep->buff_len));
		}
		VERBS_DBG(FI_LOG_EP_DATA,
			"conn %p sbufs status after:  %s\n", conn, s);
	}
#endif // ENABLE_DEBUG

	VERBS_DBG(FI_LOG_EP_DATA,
		     "conn %p sbuf allocated: %d:%p, head: %p, begin: %p\n",
		     conn,
		     (sbuf) ?
		     fi_ibv_rdm_tagged_get_buff_service_data(sbuf)->
		     seq_number : -1, sbuf, conn->sbuf_head,
		     conn->sbuf_mem_reg);

	return sbuf;
}
