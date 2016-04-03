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

#include <malloc.h>
#include <rdma/rdma_cma.h>
#include <fi_list.h>

#include "../fi_verbs.h"
#include "verbs_utils.h"
#include "verbs_rdm.h"


extern struct fi_provider fi_ibv_prov;
extern struct fi_ibv_rdm_tagged_conn *fi_ibv_rdm_tagged_conn_hash;


static struct ibv_mr *
fi_ibv_rdm_alloc_and_reg(struct fi_ibv_rdm_ep *ep, void **buf, size_t size)
{
	*buf = memalign(FI_IBV_RDM_BUF_ALIGNMENT, size);
	if (*buf) {
		memset(*buf, 0, size);
		return ibv_reg_mr(ep->domain->pd, *buf, size,
				  IBV_ACCESS_LOCAL_WRITE |
				  IBV_ACCESS_REMOTE_WRITE);
	}
	return NULL;
}

static int
fi_ibv_rdm_dereg_and_free(struct ibv_mr **mr, char **buff)
{
	int ret = ibv_dereg_mr(*mr);

	if (ret == 0) {
		*mr = NULL;
		free(*buff);
		*buff = NULL;
	}

	return ret;
}

static inline int
fi_ibv_rdm_tagged_batch_repost_receives(struct fi_ibv_rdm_tagged_conn *conn,
					struct fi_ibv_rdm_ep *ep,
					int num_to_post)
{
	const size_t idx = (conn->cm_role == FI_VERBS_CM_SELF) ? 1 : 0;
	struct ibv_recv_wr *bad_wr = NULL;
	struct ibv_recv_wr wr[num_to_post];
	int ret = 0;
	int last = num_to_post - 1;
	int i;

	for (i = 0; i < num_to_post; i++) {
		wr[i].wr_id = (uintptr_t) conn;
		wr[i].next = &wr[i + 1];
		wr[i].sg_list = NULL;
		wr[i].num_sge = 0;
	}
	wr[last].next = NULL;

	if (ibv_post_recv(conn->qp[idx], wr, &bad_wr) == 0) {
		ret = num_to_post;
	} else {
		int found = 0;
		for (i = 0; !found && (i < num_to_post); i++) {
			found = (&wr[i] == bad_wr);
		}

		if (!found) {
			VERBS_INFO(FI_LOG_EP_DATA, "Failed to post recv\n");
		}

		ret = i;
	}
	conn->recv_preposted += ret;
	return ret;
}

int fi_ibv_rdm_tagged_repost_receives(struct fi_ibv_rdm_tagged_conn *conn,
				      struct fi_ibv_rdm_ep *ep, int num_to_post)
{
	assert(num_to_post > 0);
	const int batch_size = 100;

	int rest = num_to_post;
	while (rest) {
		const int batch = MIN(rest, batch_size);
		const int ret =
		    fi_ibv_rdm_tagged_batch_repost_receives(conn, ep, batch);

		rest -= ret;
		if (ret != batch) {
			break;
		}
	}

	return num_to_post - rest;
}

static inline int
fi_ibv_rdm_tagged_prepare_conn_memory(struct fi_ibv_rdm_ep *ep,
				      struct fi_ibv_rdm_tagged_conn *conn)
{
	assert(conn->s_mr == NULL);
	assert(conn->r_mr == NULL);

	const size_t size = ep->buff_len * ep->n_buffs;
	conn->s_mr = fi_ibv_rdm_alloc_and_reg(ep,
				(void **) &conn->sbuf_mem_reg, size);
	assert(conn->s_mr);

	conn->r_mr = fi_ibv_rdm_alloc_and_reg(ep,
				(void **) &conn->rbuf_mem_reg, size);
	assert(conn->r_mr);

	conn->rma_mr = fi_ibv_rdm_alloc_and_reg(ep,
				(void **) &conn->rmabuf_mem_reg, size);
	assert(conn->rma_mr);

	fi_ibv_rdm_tagged_buffer_lists_init(conn, ep);
	return 0;
}

static inline void
fi_ibv_rdm_tagged_init_qp_attributes(struct ibv_qp_init_attr *qp_attr,
				     struct fi_ibv_rdm_ep *ep)
{
	assert(ep->scq && ep->rcq);
	memset(qp_attr, 0, sizeof(*qp_attr));
	qp_attr->send_cq = ep->scq;
	qp_attr->recv_cq = ep->rcq;
	qp_attr->qp_type = IBV_QPT_RC;
	qp_attr->cap.max_send_wr = ep->sq_wr_depth;
	qp_attr->cap.max_recv_wr = ep->rq_wr_depth;
	qp_attr->cap.max_send_sge = 1;
	qp_attr->cap.max_recv_sge = 1;
	qp_attr->cap.max_inline_data = ep->max_inline_rc;

}

static inline void
fi_ibv_rdm_pack_cm_params(struct rdma_conn_param *cm_params,
			  struct fi_ibv_rdm_tagged_conn *conn,
			  struct fi_ibv_rdm_ep *ep)
{
	memset(cm_params, 0, sizeof(struct rdma_conn_param));
	cm_params->responder_resources = 2;
	cm_params->initiator_depth = 2;

	cm_params->private_data_len = FI_IBV_RDM_DFLT_ADDRLEN;

	if ((conn->cm_role != FI_VERBS_CM_SELF) && (conn->r_mr && conn->s_mr)) {
		cm_params->private_data_len += sizeof(conn->r_mr->rkey);
		cm_params->private_data_len += sizeof(conn->remote_rbuf_mem_reg);
		cm_params->private_data_len += sizeof(conn->s_mr->rkey);
		cm_params->private_data_len += sizeof(conn->remote_sbuf_mem_reg);
	}

	cm_params->private_data = malloc(cm_params->private_data_len);

	char *p = (char *) cm_params->private_data;
	memcpy(p, &ep->cm.my_addr, FI_IBV_RDM_DFLT_ADDRLEN);
	p += FI_IBV_RDM_DFLT_ADDRLEN;

	if ((conn->cm_role != FI_VERBS_CM_SELF) && (conn->r_mr && conn->s_mr)) {
		memcpy(p, &conn->r_mr->rkey, sizeof(conn->r_mr->rkey));
		p += sizeof(conn->r_mr->rkey);
		memcpy(p, &conn->rbuf_mem_reg, sizeof(conn->rbuf_mem_reg));
		p += sizeof(conn->rbuf_mem_reg);

		memcpy(p, &conn->s_mr->rkey, sizeof(conn->s_mr->rkey));
		p += sizeof(conn->s_mr->rkey);
		memcpy(p, &conn->sbuf_mem_reg, sizeof(conn->sbuf_mem_reg));
		p += sizeof(conn->sbuf_mem_reg);
	}
}


static inline void
fi_ibv_rdm_unpack_cm_params(struct rdma_conn_param *cm_param,
			  struct fi_ibv_rdm_tagged_conn *conn,
			  struct fi_ibv_rdm_ep *ep)
{
	char *p = (char *)cm_param->private_data;

	if (conn->cm_role == FI_VERBS_CM_SELF) {
		if (conn->r_mr && conn->s_mr) {
			memcpy(&conn->addr, &ep->cm.my_addr,
				FI_IBV_RDM_DFLT_ADDRLEN);
			conn->remote_rbuf_rkey = conn->r_mr->rkey;
			conn->remote_rbuf_mem_reg = conn->r_mr->addr;

			conn->remote_sbuf_rkey = conn->s_mr->rkey;
			conn->remote_sbuf_mem_reg = conn->s_mr->addr;

			conn->remote_sbuf_head = conn->remote_sbuf_mem_reg +
				FI_IBV_RDM_BUFF_SERVICE_DATA_SIZE;
		}
	} else {
		if (conn->state == FI_VERBS_CONN_ALLOCATED) {
			memcpy(&conn->addr, p, FI_IBV_RDM_DFLT_ADDRLEN);
		}
		p += FI_IBV_RDM_DFLT_ADDRLEN;

		conn->remote_rbuf_rkey = *(uint32_t *) (p);
		p += sizeof(conn->r_mr->rkey);
		conn->remote_rbuf_mem_reg = *(char **)(p);
		p += sizeof(conn->remote_rbuf_mem_reg);

		conn->remote_sbuf_rkey = *(uint32_t *) (p);
		p += sizeof(conn->s_mr->rkey);
		conn->remote_sbuf_mem_reg = *(char **)(p);
		p += sizeof(conn->remote_sbuf_mem_reg);

		conn->remote_sbuf_head = conn->remote_sbuf_mem_reg +
			FI_IBV_RDM_BUFF_SERVICE_DATA_SIZE;
	}
}

static inline int
fi_ibv_rdm_tagged_process_addr_resolved(struct rdma_cm_id *id,
					struct fi_ibv_rdm_ep *ep)
{
	struct ibv_qp_init_attr qp_attr;
	struct fi_ibv_rdm_tagged_conn *conn = id->context;

	VERBS_INFO(FI_LOG_AV, "ADDR_RESOLVED conn %p, addr %s:%u\n",
		   conn, inet_ntoa(conn->addr.sin_addr),
		   ntohs(conn->addr.sin_port));

	assert(id->verbs == ep->domain->verbs);

	do {
		fi_ibv_rdm_tagged_init_qp_attributes(&qp_attr, ep);
		if (rdma_create_qp(id, ep->domain->pd, &qp_attr)) {
			VERBS_INFO(FI_LOG_AV, "rdma_create_qp failed\n");
		}

		if (conn->cm_role == FI_VERBS_CM_PASSIVE) {
			break;
		}

		conn->qp[0] = id->qp;
		assert(conn->id[0] == id);
		if (conn->cm_role == FI_VERBS_CM_SELF) {
			break;
		}

		fi_ibv_rdm_tagged_prepare_conn_memory(ep, conn);
		if (ep->rq_wr_depth != fi_ibv_rdm_tagged_repost_receives(conn,
			ep, ep->rq_wr_depth)) {
			VERBS_INFO(FI_LOG_AV, "repost receives failed\n");
			return -FI_ENOMEM;
		}
	} while (0);

	if (rdma_resolve_route(id, FI_IBV_RDM_CM_RESOLVEADDR_TIMEOUT)) {
		fprintf(stderr, " rdma_resolve_route failed\n");
	}
	return 0;
}

static inline int
fi_ibv_rdm_tagged_process_connect_request(struct rdma_cm_event *event,
					  struct fi_ibv_rdm_ep *ep)
{
	struct ibv_qp_init_attr qp_attr;
	struct rdma_conn_param cm_params;
	struct fi_ibv_rdm_tagged_conn *conn = NULL;
	struct rdma_cm_id *id = event->id;
	int ret = 0;

	char *p = (char *) event->param.conn.private_data;

	if (ep->is_closing) {
		int rej_message = 0xdeadbeef;
		if (rdma_reject(id, &rej_message, sizeof(int))) {
			fprintf(stderr, "rdma_reject failed\n");
			rdma_destroy_id(id);
		}
		return ret;
	}

	HASH_FIND(hh, fi_ibv_rdm_tagged_conn_hash, p, FI_IBV_RDM_DFLT_ADDRLEN,
		  conn);

	if (!conn) {
		conn = memalign(FI_IBV_RDM_MEM_ALIGNMENT, sizeof *conn);
		if (!conn)
			return -FI_ENOMEM;

		memset(conn, 0, sizeof(struct fi_ibv_rdm_tagged_conn));

		conn->state = FI_VERBS_CONN_ALLOCATED;
		dlist_init(&conn->postponed_requests_head);
		fi_ibv_rdm_unpack_cm_params(&event->param.conn, conn, ep);
		fi_ibv_rdm_conn_init_cm_role(conn, ep);

		FI_INFO(&fi_ibv_prov, FI_LOG_AV,
			"CONN REQUEST, NOT found in hash, new conn %p %d, addr %s:%u, HASH ADD\n",
			conn, conn->cm_role, inet_ntoa(conn->addr.sin_addr),
			ntohs(conn->addr.sin_port));

		HASH_ADD(hh, fi_ibv_rdm_tagged_conn_hash, addr,
			FI_IBV_RDM_DFLT_ADDRLEN, conn);
	} else {
		if (conn->cm_role != FI_VERBS_CM_ACTIVE) {
			/*
			 * Do it before rdma_create_qp since that call would
			 * modify event->param.conn.private_data buffer
			 */
			fi_ibv_rdm_unpack_cm_params(&event->param.conn, conn,
						    ep);
		}

		FI_INFO(&fi_ibv_prov, FI_LOG_AV,
			"CONN REQUEST,  FOUND in hash, conn %p %d, addr %s:%u\n",
			conn, conn->cm_role, inet_ntoa(conn->addr.sin_addr),
			ntohs(conn->addr.sin_port));
	}

	if (conn->cm_role == FI_VERBS_CM_ACTIVE) {
		int rej_message = 0xdeadbeef;
		if (rdma_reject(id, &rej_message, sizeof(int))) {
			fprintf(stderr, "rdma_reject failed\n");
			rdma_destroy_id(id);
		}
		if (conn->state == FI_VERBS_CONN_ALLOCATED) {
			fi_ibv_rdm_start_connection(ep, conn);
		}
	} else {
		assert(conn->state == FI_VERBS_CONN_ALLOCATED ||
		       conn->state == FI_VERBS_CONN_STARTED);

		const size_t idx = 
			(conn->cm_role == FI_VERBS_CM_PASSIVE) ? 0 : 1;

		conn->state = FI_VERBS_CONN_STARTED;

		assert (conn->id[idx] == NULL);
		conn->id[idx] = id;

		fi_ibv_rdm_tagged_prepare_conn_memory(ep, conn);
		fi_ibv_rdm_tagged_init_qp_attributes(&qp_attr, ep);
		rdma_create_qp(id, ep->domain->pd, &qp_attr);
		conn->qp[idx] = id->qp;
		if (ep->rq_wr_depth !=
		    fi_ibv_rdm_tagged_repost_receives(conn, ep,
						      ep->rq_wr_depth))
		{
			VERBS_INFO(FI_LOG_AV, "repost receives failed\n");
			return -FI_ENOMEM;
		}

		id->context = conn;

		fi_ibv_rdm_pack_cm_params(&cm_params, conn, ep);

		ret = rdma_accept(id, &cm_params);
		assert(ret == 0);
		if (cm_params.private_data) {
			free((void *) cm_params.private_data);
		}
	} 
	return ret;
}

static inline int
fi_ibv_rdm_tagged_process_route_resolved(struct rdma_cm_event *event,
					 struct fi_ibv_rdm_ep *ep)
{
	struct fi_ibv_rdm_tagged_conn *conn = event->id->context;
	int ret = 0;

	struct rdma_conn_param cm_params;
	fi_ibv_rdm_pack_cm_params(&cm_params, conn, ep);

	VERBS_INFO(FI_LOG_AV,
		"ROUTE RESOLVED, conn %p, addr %s:%u\n", conn,
		inet_ntoa(conn->addr.sin_addr), ntohs(conn->addr.sin_port));

	ret = rdma_connect(event->id, &cm_params);
	assert(ret == 0);

	free((void *)cm_params.private_data);

	return ret;
}

static inline int
fi_ibv_rdm_tagged_process_event_established(struct rdma_cm_event *event,
					    struct fi_ibv_rdm_ep *ep)
{
	struct fi_ibv_rdm_tagged_conn *conn =
	    (struct fi_ibv_rdm_tagged_conn *)event->id->context;
#if defined(ENABLE_DEBUG) && ENABLE_DEBUG > 0
	if (conn->state != FI_VERBS_CONN_STARTED &&
	    conn->cm_role != FI_VERBS_CM_SELF)
	{
		fprintf(stderr, "state = %d, conn %p", conn->state, conn);
		assert(0 && "Wrong state");
	}
#endif

	if (conn->cm_role == FI_VERBS_CM_ACTIVE ||
	    conn->cm_role == FI_VERBS_CM_SELF)
	{
		fi_ibv_rdm_unpack_cm_params(&event->param.conn, conn, ep);
	}

	FI_INFO(&fi_ibv_prov, FI_LOG_AV, "CONN ESTABLISHED, conn %p, addr %s:%u\n",
		conn, inet_ntoa(conn->addr.sin_addr),
		ntohs(conn->addr.sin_port));
	
	/* Do not count self twice */
	if (conn->state != FI_VERBS_CONN_ESTABLISHED) {
		ep->num_active_conns++;
		conn->state = FI_VERBS_CONN_ESTABLISHED;
	}
	return 0;
}

int fi_ibv_rdm_tagged_conn_cleanup(struct fi_ibv_rdm_tagged_conn *conn)
{
	int ret = 0;

	VERBS_DBG(FI_LOG_AV, "conn %p, exp = %lld unexp = %lld\n", conn,
		     conn->exp_counter, conn->unexp_counter);

	if (conn->id[0]) {
		rdma_destroy_qp(conn->id[0]);
		if ((ret = rdma_destroy_id(conn->id[0]))) {
			VERBS_INFO(FI_LOG_AV, 
				"rdma_destroy_id failed, ret = %d\n", ret);
		}
	}

	if (conn->id[1]) {
		assert(conn->cm_role == FI_VERBS_CM_SELF);
		rdma_destroy_qp(conn->id[1]);
		if ((ret = rdma_destroy_id(conn->id[1]))) {
			VERBS_INFO(FI_LOG_AV, 
				"rdma_destroy_id failed, ret = %d\n", ret);
		}
	}

	if (conn->s_mr)
		fi_ibv_rdm_dereg_and_free(&conn->s_mr, &conn->sbuf_mem_reg);
	if (conn->r_mr)
		fi_ibv_rdm_dereg_and_free(&conn->r_mr, &conn->rbuf_mem_reg);
	if (conn->rma_mr)
		fi_ibv_rdm_dereg_and_free(&conn->rma_mr, &conn->rmabuf_mem_reg);

	memset(conn, 0, sizeof(*conn));
	free(conn);
	return ret;
}

static inline int
fi_ibv_rdm_tagged_process_event_disconnected(struct fi_ibv_rdm_ep *ep,
					     struct rdma_cm_event *event)
{
	struct fi_ibv_rdm_tagged_conn *conn = event->id->context;

	ep->num_active_conns--;
	
	if (conn->state == FI_VERBS_CONN_ESTABLISHED) {
		conn->state = FI_VERBS_CONN_REMOTE_DISCONNECT;
	} else {
		assert(conn->state == FI_VERBS_CONN_LOCAL_DISCONNECT);
		conn->state = FI_VERBS_CONN_CLOSED;
	}
	VERBS_INFO(FI_LOG_AV,
		   "Disconnected from conn %p, addr %s:%u\n",
		   conn, inet_ntoa(conn->addr.sin_addr),
		   ntohs(conn->addr.sin_port));
	if (conn->state == FI_VERBS_CONN_CLOSED) {
		fi_ibv_rdm_tagged_conn_cleanup(conn);
	}

	return 0;
}

static inline int
fi_ibv_rdm_tagged_process_event_rejected(struct fi_ibv_rdm_ep *ep,
					 struct rdma_cm_event *event)
{
	struct fi_ibv_rdm_tagged_conn *conn = event->id->context;
	int ret = 0;

	if (NULL != event->param.conn.private_data &&
	    *((int *)event->param.conn.private_data) == 0xdeadbeef ) {
		assert(conn->cm_role == FI_VERBS_CM_PASSIVE);
		rdma_destroy_qp(event->id);
		ret = rdma_destroy_id(event->id);
		VERBS_INFO(FI_LOG_AV,
			"Rejected from conn %p, addr %s:%u, cm_role %d, status %d\n",
			conn, inet_ntoa(conn->addr.sin_addr),
			ntohs(conn->addr.sin_port),
			conn->cm_role,
			event->status);
	} else {
		VERBS_INFO(FI_LOG_AV,
			"Unexpected REJECT from conn %p, addr %s:%u, cm_role %d, msg len %d, msg %x, status %d\n",
			conn, inet_ntoa(conn->addr.sin_addr),
			ntohs(conn->addr.sin_port),
			conn->cm_role,
			event->param.conn.private_data_len,
			event->param.conn.private_data ?
			*(int *)event->param.conn.private_data : 0,
			event->status);
		conn->state = FI_VERBS_CONN_REJECTED;

	}
	return ret;
}

static int fi_ibv_rdm_tagged_process_event(struct rdma_cm_event *event,
					   struct fi_ibv_rdm_ep *ep)
{
	int ret;
	switch (event->event) {
	case RDMA_CM_EVENT_ADDR_RESOLVED:
		ret = fi_ibv_rdm_tagged_process_addr_resolved(event->id, ep);
		break;
	case RDMA_CM_EVENT_ROUTE_RESOLVED:
		ret = fi_ibv_rdm_tagged_process_route_resolved(event, ep);
		break;
	case RDMA_CM_EVENT_ESTABLISHED:
		ret = fi_ibv_rdm_tagged_process_event_established(event, ep);
		break;
	case RDMA_CM_EVENT_DISCONNECTED:
		ret = fi_ibv_rdm_tagged_process_event_disconnected(ep, event);
		break;
	case RDMA_CM_EVENT_CONNECT_REQUEST:
		ret = fi_ibv_rdm_tagged_process_connect_request(event, ep);
		break;
	case RDMA_CM_EVENT_REJECTED:
		ret = fi_ibv_rdm_tagged_process_event_rejected(ep, event);
		break;
	case RDMA_CM_EVENT_TIMEWAIT_EXIT:
		ret = FI_SUCCESS;
		break;
	case RDMA_CM_EVENT_ADDR_ERROR:
	case RDMA_CM_EVENT_ROUTE_ERROR:
	case RDMA_CM_EVENT_CONNECT_ERROR:
	case RDMA_CM_EVENT_UNREACHABLE:

	default:
		ret = -FI_EOTHER;
		VERBS_INFO(FI_LOG_AV, "got unexpected rdmacm event, %s\n",
			     rdma_event_str(event->event));
		abort();
		break;
	}
	return ret;

}

int fi_ibv_rdm_tagged_cm_progress(struct fi_ibv_rdm_ep *ep)
{
	struct rdma_cm_event *event = NULL;
	int ret = 0;

	rdma_get_cm_event(ep->cm.ec, &event);

	while (event) {
		pthread_mutex_lock(&ep->cm_lock);

		void *data = NULL;
		struct rdma_cm_event event_copy;
		memcpy(&event_copy, event, sizeof(*event));
		if (event->param.conn.private_data_len) {
			data = malloc(event->param.conn.private_data_len);
			memcpy(data, event->param.conn.private_data,
				      event->param.conn.private_data_len);
			event_copy.param.conn.private_data = data;
			event_copy.param.conn.private_data_len =
			    event->param.conn.private_data_len;
		}
		rdma_ack_cm_event(event);
		ret = fi_ibv_rdm_tagged_process_event(&event_copy, ep);
		free(data);
		event = NULL;
		pthread_mutex_unlock(&ep->cm_lock);
		rdma_get_cm_event(ep->cm.ec, &event);
	}
	return ret;
}
