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

#ifndef _VERBS_RDM_H
#define _VERBS_RDM_H

#include <rdma/rdma_cma.h>
#include <rdma/fi_atomic.h>
#include <uthash.h>

#include "../fi_verbs.h"
#include "verbs_utils.h"
#include "verbs_tagged_ep_rdm_states.h"

#define FI_IBV_EP_TYPE_IS_RDM(_info)	\
	(_info && _info->ep_attr && (_info->ep_attr->type == FI_EP_RDM))

#define FI_IBV_RDM_ST_PKTTYPE_MASK  ((uint32_t) 0xFF)
#define FI_IBV_RDM_EAGER_PKT		0
#define FI_IBV_RDM_RNDV_RTS_PKT		1
#define FI_IBV_RDM_RNDV_ACK_PKT		2
#define FI_IBV_RDM_RMA_PKT		3
#define FI_IBV_RDM_MSG_PKT		4
#define FI_IBV_RDM_SET_PKTTYPE(dest, type) (dest |= type)
#define FI_IBV_RDM_GET_PKTTYPE(value) (value & FI_IBV_RDM_ST_PKTTYPE_MASK)


/* ------- SERVICE_TAG layout ---------
 * [ 24 bits - Unused at the moment ] [ 8 bits - pkt type ]
 *                                    IBV_RDM_ST_PKTYPE_MASK
 */

/* WR - work request */
#define FI_IBV_RDM_SERVICE_WR_MASK              ((uint64_t)0x1)
#define FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(value)	\
        (value & FI_IBV_RDM_SERVICE_WR_MASK)

#define FI_IBV_RDM_PACK_WR(value)               ((uint64_t)(uintptr_t)(void *)value)
#define FI_IBV_RDM_UNPACK_WR(value)             ((void *)(uintptr_t)value)

#define FI_IBV_RDM_PACK_SERVICE_WR(value)					\
        (((uint64_t)(uintptr_t)(void *)value) | FI_IBV_RDM_SERVICE_WR_MASK)

#define FI_IBV_RDM_UNPACK_SERVICE_WR(value)				\
        ((void *)(uintptr_t)(value & (~(FI_IBV_RDM_SERVICE_WR_MASK))))

/* Send/Recv counters control */
#define FI_IBV_RDM_INC_SIG_POST_COUNTERS(_connection, _ep)				\
do {                                                                			\
	(_connection)->av_entry->sends_outgoing++;					\
	(_ep)->posted_sends++;								\
	VERBS_DBG(FI_LOG_CQ, "SEND_COUNTER++, conn %p, sends_outgoing %"PRIu32"\n",	\
		  _connection, (_connection)->av_entry->sends_outgoing);		\
} while (0)

#define FI_IBV_RDM_DEC_SIG_POST_COUNTERS(_connection, _ep)				\
do {											\
	(_connection)->av_entry->sends_outgoing--;					\
	(_ep)->posted_sends--;								\
	VERBS_DBG(FI_LOG_CQ, "SEND_COUNTER--, conn %p, sends_outgoing %"PRIu32"\n",	\
		  _connection, (_connection)->av_entry->sends_outgoing);		\
} while (0)

#define OUTGOING_POST_LIMIT(_connection, _ep)					\
	((_connection)->av_entry->sends_outgoing >= (_ep)->sq_wr_depth - 1)

#define PEND_POST_LIMIT(_ep)						\
	((_ep)->posted_sends > 0.5 * (_ep)->scq_depth)

#define TSEND_RESOURCES_IS_BUSY(_connection, _ep)			\
	(OUTGOING_POST_LIMIT(_connection, _ep) || PEND_POST_LIMIT(_ep))

#define RMA_RESOURCES_IS_BUSY(_connection, _ep)				\
	(OUTGOING_POST_LIMIT(_connection, _ep) || PEND_POST_LIMIT(_ep))

#define GET_TX_COMP(ep_rdm)						\
	(!ep_rdm->rx_selective_completion ||		\
	(ep_rdm->rx_op_flags & FI_COMPLETION) ?		\
	FI_COMPLETION : 0ULL)

#define GET_TX_COMP_FLAG(ep_rdm, flag)			\
	(!ep_rdm->rx_selective_completion ||		\
	(ep_rdm->rx_op_flags & FI_COMPLETION) ?		\
	FI_COMPLETION : (flags & FI_COMPLETION))

struct fi_ibv_rdm_header {
/*	uint64_t imm_data; TODO: not implemented */
	uint64_t tag;
	uint32_t service_tag;
	uint32_t padding;
};

struct fi_ibv_rdm_rndv_header {
	struct fi_ibv_rdm_header base;
	uint64_t src_addr;
	uint64_t id; /* pointer to request on sender side */
	uint64_t total_len;
	uint64_t mem_rkey;
	uint32_t is_tagged;
};

struct fi_ibv_rdm_multi_request {
	/* working request, will be renewed for every data arriving */
	struct dlist_entry		list_entry;
	struct fi_ibv_rdm_request	*prepost;
	struct fi_ibv_rdm_ep		*ep;
	uint8_t				*buf;
	uint64_t			len;
	uint64_t			offset;
	uint64_t			min_size;
};

struct fi_ibv_rdm_request {

	/* Accessors and match info */

	/* Request can be an element of only one queue at the moment */
	struct dlist_entry queue_entry;
	/* multi recv handling */
	struct fi_ibv_rdm_multi_request *parent;
	struct {
		enum fi_ibv_rdm_request_eager_state eager;
		enum fi_ibv_rdm_request_rndv_state rndv;
		ssize_t err; /* filled in case of moving to errcq */
	} state;

	struct fi_ibv_rdm_ep	*ep;
	struct fi_ibv_rdm_minfo	minfo;

	/* User data: buffers, lens, imm, context */

	union {
		void *src_addr;
		void *dest_buf;
		struct iovec *iovec_arr;
	};

	union {
		void *exp_rbuf;
		struct fi_ibv_rdm_buf *unexp_rbuf;
		struct fi_ibv_rdm_buf *sbuf;
		struct fi_ibv_rdm_buf *rmabuf;
		struct iovec *rmaiovec_arr;
	};

	/*
	 * iovec_arr is a mem pool entry if iov_count > 0
	 * and must be freed properly
	 */
	uint64_t iov_count;
	uint64_t len;
	uint64_t rest_len;
	uint64_t comp_flags;
	struct fi_context *context;
	uint32_t post_counter;
	uint32_t imm;

	union {
		/* RNDV info */
		struct {
			/* pointer to request on sender side */
			uint64_t id;
			/* registered buffer on sender side */
			void* remote_addr;
			/* registered mr of local src_addr */
			struct fi_ibv_mem_desc md;
			uint64_t mr_rkey;
		} rndv;
		
		/* RMA info */
		struct {
			struct fi_ibv_mem_desc md;
			uint64_t mr_rkey;
			uint64_t mr_lkey;
			uint64_t remote_addr;
			enum ibv_wr_opcode opcode;
		} rma;
	};
};

static inline void
fi_ibv_rdm_zero_request(struct fi_ibv_rdm_request *request)
{
	memset(request, 0, sizeof(*request));
}

void fi_ibv_rdm_print_request(char *buf, struct fi_ibv_rdm_request *request);

#define BUF_STATUS_FREE 	((uint16_t) 0)
#define BUF_STATUS_BUSY 	((uint16_t) 1)
#define BUF_STATUS_RECEIVED 	((uint16_t) 2)

struct fi_ibv_rdm_buf_service_data {
	volatile uint16_t status;
	uint16_t seq_num;
	int32_t pkt_len;
};

#define FI_IBV_RDM_BUFF_SERVICE_DATA_SIZE	\
	(offsetof(struct fi_ibv_rdm_buf, header))

struct fi_ibv_rdm_buf {
	struct fi_ibv_rdm_buf_service_data service_data;
	struct fi_ibv_rdm_header header;
	uint64_t payload;
};

struct fi_ibv_rdm_cm {
	struct rdma_event_channel*	ec;
	struct rdma_cm_id*		listener;
	int				is_bound;

	/* av_hash has a sockaddr_in -> [ep - conn] associative */
	struct fi_ibv_rdm_av_entry*	av_hash;
	/* Used only for FI_AV_TABLE */
	struct fi_ibv_rdm_av_entry**	av_table;

	struct slist			av_removed_entry_head;
	pthread_mutex_t			cm_lock;
	pthread_t			cm_progress_thread;
	int				cm_progress_timeout;
	int				fi_ibv_rdm_tagged_cm_progress_running;
};

struct fi_ibv_rdm_cntr {
	struct fid_cntr		fid;
	struct fi_ibv_domain	*domain;
	ofi_atomic32_t		ep_ref;
	uint64_t		value;
	struct fi_cntr_attr	attr;
	uint64_t		err_count;
};

struct fi_ibv_rdm_ep {
	struct fid_ep ep_fid;
	struct fi_ibv_domain	*domain;
	struct slist_entry	list_entry;
	struct fi_ibv_rdm_cq	*fi_scq;
	struct fi_ibv_rdm_cq	*fi_rcq;

	struct fi_ibv_rdm_cntr	*send_cntr;
	struct fi_ibv_rdm_cntr	*recv_cntr;
	struct fi_ibv_rdm_cntr	*read_cntr;
	struct fi_ibv_rdm_cntr	*write_cntr;

	struct fi_info	*info;

	struct util_buf_pool	*fi_ibv_rdm_request_pool;
	struct util_buf_pool	*fi_ibv_rdm_multi_request_pool;
	struct util_buf_pool	*fi_ibv_rdm_postponed_pool;
	/*
	 * extra buffer size equal eager buffer size,
	 * it is used for any intermediate needs like
	 * unexpected recv, pack/unpack noncontig messages, etc
	 */
	struct util_buf_pool	*fi_ibv_rdm_extra_buffers_pool;

	struct dlist_entry	fi_ibv_rdm_posted_queue;
	struct dlist_entry	fi_ibv_rdm_postponed_queue;
	struct dlist_entry	fi_ibv_rdm_unexp_queue;
	struct dlist_entry	fi_ibv_rdm_multi_recv_list;

	size_t			addrlen;
	struct rdma_addrinfo	*rai;
	struct sockaddr_in	my_addr;

	struct fi_ibv_av	*av;
	int		tx_selective_completion;
	int 		rx_selective_completion;
	size_t 		min_multi_recv_size;
	uint64_t 	tx_op_flags;
	uint64_t 	rx_op_flags;

	/*
	 * ibv_post_send opcode for eager messaging.
	 * It must generate work completion in receive CQ
	 */
	enum ibv_wr_opcode	eopcode;
	struct ibv_cq		*scq;
	struct ibv_cq		*rcq;

	int	buff_len;
	int	n_buffs;
	int	rq_wr_depth;    // RQ depth
	int	sq_wr_depth;    // SQ depth
	uint32_t	posted_sends;
	uint32_t	posted_recvs;
	int	num_active_conns;
	int	max_inline_rc;
	int	rndv_threshold;
	int	rndv_seg_size;
	size_t	iov_per_rndv_thr;
	int	scq_depth;
	int	rcq_depth;

	int	is_closing;
	int	recv_preposted_threshold;
};

enum fi_rdm_cm_conn_state {
	FI_VERBS_CONN_ALLOCATED,
	FI_VERBS_CONN_STARTED,
	FI_VERBS_CONN_REJECTED,
	FI_VERBS_CONN_ESTABLISHED,
	FI_VERBS_CONN_LOCAL_DISCONNECT,
	FI_VERBS_CONN_CLOSED
};

enum fi_rdm_cm_role {
	FI_VERBS_CM_UNDEFINED,
	FI_VERBS_CM_ACTIVE,
	FI_VERBS_CM_PASSIVE,
	FI_VERBS_CM_SELF,
};

struct fi_ibv_rdm_av_entry {
	/* association of conn and EPs */
	struct fi_ibv_rdm_conn	*conn_hash;
	struct sockaddr_in	addr;
	uint32_t		sends_outgoing;
	uint32_t		recv_preposted;
	struct slist_entry	removed_next;
	UT_hash_handle		hh;
};

struct fi_ibv_rdm_conn {

	/* 
	 * In normal case only qp[0] and id[0] are used.
	 * qp[1] and id[1] are used for establishing connection to self
	 * like passive side
	 */
	struct ibv_qp *qp[2];
	struct rdma_cm_id *id[2];
	struct sockaddr_in addr;
	struct fi_ibv_rdm_ep *ep;
	enum fi_rdm_cm_role cm_role;
	enum fi_rdm_cm_conn_state state;

	char *sbuf_mem_reg;
	struct fi_ibv_rdm_buf *sbuf_head;
	uint16_t sbuf_ack_status;

	char *rbuf_mem_reg;
	struct fi_ibv_rdm_buf *rbuf_head;

	char *rmabuf_mem_reg;
	struct fi_ibv_rdm_buf *rmabuf_head;

	struct dlist_entry postponed_requests_head;
	struct fi_ibv_rdm_postponed_entry *postponed_entry;

	struct fi_ibv_mem_desc s_md;
	struct fi_ibv_mem_desc r_md;
	struct fi_ibv_mem_desc ack_md;
	struct fi_ibv_mem_desc rma_md;

	uint32_t remote_sbuf_rkey;
	uint32_t remote_rbuf_rkey;

	char *remote_sbuf_mem_reg;
	char *remote_rbuf_mem_reg;
	struct fi_ibv_rdm_buf *remote_sbuf_head;

	/* counter for eager buffer releasing */
	uint16_t recv_completions;
	/* counter to control OOO behaviour, works in pair with recv_completions */
	uint16_t recv_processed;

	struct fi_ibv_rdm_av_entry *av_entry;
	UT_hash_handle hh;
#if ENABLE_DEBUG
	size_t unexp_counter;
	size_t exp_counter;
#endif
};

struct fi_ibv_rdm_postponed_entry {
	struct dlist_entry queue_entry;

	struct fi_ibv_rdm_conn *conn;
};

struct fi_conn_priv_params {
	char addr[FI_IBV_RDM_DFLT_ADDRLEN];

	uint32_t rbuf_rkey;
	uint32_t sbuf_rkey;
	char *rbuf_mem_reg;
	char *sbuf_mem_reg;
};

static inline void
fi_ibv_rdm_set_buffer_status(struct fi_ibv_rdm_buf *buff, uint16_t status)
{
	buff->service_data.status = status;
	if (status == BUF_STATUS_FREE) {
		buff->service_data.pkt_len = 0;
	}
}

static inline int
fi_ibv_rdm_buffer_check_seq_num(struct fi_ibv_rdm_buf *buff, uint16_t seq_num)
{
	VERBS_DBG(FI_LOG_EP_DATA, "seq num: %d <-> %d\n",
		buff->service_data.seq_num, seq_num);
	return (seq_num == buff->service_data.seq_num);
}

static inline uintptr_t
fi_ibv_rdm_get_remote_addr(struct fi_ibv_rdm_conn *conn,
			   struct fi_ibv_rdm_buf *local_sbuff)
{
	return (uintptr_t) (conn->remote_rbuf_mem_reg +
			    ((char *)local_sbuff - conn->sbuf_mem_reg));
}

static inline void
fi_ibv_rdm_push_buff_pointer(char *area_start, size_t area_size,
			     struct fi_ibv_rdm_buf **rdm_buff, size_t offset)
{
	char *buff = (char*)(*rdm_buff);
	char *buff_tmp = buff + offset;

	VERBS_DBG(FI_LOG_EP_DATA, "old_pointer: %p\n", buff);

	buff = buff_tmp < (area_start + area_size) ? buff_tmp : area_start;
	
	VERBS_DBG(FI_LOG_EP_DATA, "new_pointer: %p\n", buff);

	*rdm_buff = (struct fi_ibv_rdm_buf *)buff;
}

static inline void
fi_ibv_rdm_push_sbuff_head(struct fi_ibv_rdm_conn *conn, 
			   struct fi_ibv_rdm_ep *ep)
{
	fi_ibv_rdm_push_buff_pointer(conn->sbuf_mem_reg,
				     ep->buff_len * ep->n_buffs,
				     &conn->sbuf_head, ep->buff_len);
}

static inline void
fi_ibv_rdm_push_rmabuff_head(struct fi_ibv_rdm_conn *conn,
			     struct fi_ibv_rdm_ep *ep)
{
	fi_ibv_rdm_push_buff_pointer(conn->rmabuf_mem_reg,
				     ep->buff_len * ep->n_buffs,
				     &conn->rmabuf_head, ep->buff_len);
}

static inline struct fi_ibv_rdm_buf *
fi_ibv_rdm_get_rmabuf(struct fi_ibv_rdm_conn *conn,
		      struct fi_ibv_rdm_ep *ep, uint16_t seq_num)
{
	char *rmabuf = conn->rmabuf_mem_reg + (seq_num * ep->buff_len);
	VERBS_DBG(FI_LOG_EP_DATA, "rma buf %d\n", seq_num);
	return (struct fi_ibv_rdm_buf *) rmabuf;
}

static inline struct fi_ibv_rdm_buf *
fi_ibv_rdm_get_rbuf(struct fi_ibv_rdm_conn *conn,
		    struct fi_ibv_rdm_ep *ep, uint16_t seq_num)
{
	struct fi_ibv_rdm_buf *rbuf = (struct fi_ibv_rdm_buf *)
		(conn->rbuf_mem_reg + (seq_num * ep->buff_len));

	VERBS_DBG(FI_LOG_EP_DATA, "recv buf %d <-> %d\n",
		seq_num, rbuf->service_data.seq_num);

	return  rbuf;
}

static inline struct fi_ibv_rdm_buf *
fi_ibv_rdm_get_sbuf(struct fi_ibv_rdm_conn *conn,
		    struct fi_ibv_rdm_ep *ep, uint16_t seq_num)
{
	char *sbuf = conn->sbuf_mem_reg + (seq_num * ep->buff_len);
	VERBS_DBG(FI_LOG_EP_DATA, "send buf %d\n", seq_num);
	return (struct fi_ibv_rdm_buf *)sbuf;
}

static inline void
fi_ibv_rdm_buffer_lists_init(struct fi_ibv_rdm_conn *conn,
			     struct fi_ibv_rdm_ep *ep)
{
	int i;

	conn->sbuf_head = (struct fi_ibv_rdm_buf *)conn->sbuf_mem_reg;
	conn->rbuf_head = (struct fi_ibv_rdm_buf *)conn->rbuf_mem_reg;
	conn->sbuf_ack_status = BUF_STATUS_FREE;

	conn->rmabuf_head = (struct fi_ibv_rdm_buf *)conn->rmabuf_mem_reg;

	for (i = 0; i < ep->n_buffs; ++i) {
		fi_ibv_rdm_set_buffer_status(fi_ibv_rdm_get_sbuf(conn, ep, i),
			BUF_STATUS_FREE);
		fi_ibv_rdm_get_sbuf(conn, ep, i)->service_data.seq_num = i;

		fi_ibv_rdm_set_buffer_status(fi_ibv_rdm_get_rbuf(conn, ep, i),
			BUF_STATUS_FREE);
		/* should be initialized by sender */
		fi_ibv_rdm_get_rbuf(conn, ep, i)->service_data.seq_num = 
			(uint16_t)(-1);

		fi_ibv_rdm_set_buffer_status(fi_ibv_rdm_get_rmabuf(conn, ep, i),
			BUF_STATUS_FREE);
		fi_ibv_rdm_get_rmabuf(conn, ep, i)->service_data.seq_num = i;
	}
}

static inline void fi_ibv_rdm_cntr_inc(struct fi_ibv_rdm_cntr *cntr)
{
	if (cntr) {
		cntr->fid.ops->add(&cntr->fid, 1);
	}
}

static inline void fi_ibv_rdm_cntr_inc_err(struct fi_ibv_rdm_cntr *cntr)
{
	if (cntr) {
		cntr->err_count++;
	}
}

int fi_ibv_rdm_tagged_poll(struct fi_ibv_rdm_ep *ep);
int fi_ibv_rdm_tagged_poll_recv(struct fi_ibv_rdm_ep *ep);
ssize_t fi_ibv_rdm_cm_progress(struct fi_ibv_rdm_ep *ep);
ssize_t
fi_ibv_rdm_start_overall_disconnection(struct fi_ibv_rdm_av_entry *av_entry);
ssize_t fi_ibv_rdm_start_disconnection(struct fi_ibv_rdm_conn *conn);
int fi_ibv_av_entry_alloc(struct fi_ibv_domain *domain,
			  struct fi_ibv_rdm_av_entry **av_entry,
			  void *addr);
ssize_t fi_ibv_rdm_conn_cleanup(struct fi_ibv_rdm_conn *conn);
ssize_t fi_ibv_rdm_overall_conn_cleanup(struct fi_ibv_rdm_av_entry *av_entry);
ssize_t fi_ibv_rdm_start_connection(struct fi_ibv_rdm_ep *ep,
				    struct fi_ibv_rdm_conn *conn);
ssize_t fi_ibv_rdm_repost_receives(struct fi_ibv_rdm_conn *conn,
				   struct fi_ibv_rdm_ep *ep,
				   int num_to_post);
int fi_ibv_rdm_tagged_open_ep(struct fid_domain *domain, struct fi_info *info,
                              struct fid_ep **ep, void *context);
int fi_ibv_rdm_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		   struct fid_cq **cq, void *context);

int fi_ibv_rdm_tagged_prepare_send_request(struct fi_ibv_rdm_request *request,
					   struct fi_ibv_rdm_ep *ep);
int fi_ibv_rdm_prepare_rma_request(struct fi_ibv_rdm_request *request,
				   struct fi_ibv_rdm_ep *ep);

static inline struct fi_ibv_rdm_buf *
fi_ibv_rdm_get_sbuf_head(struct fi_ibv_rdm_conn *conn, struct fi_ibv_rdm_ep *ep)
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
			struct fi_ibv_rdm_buf *buf = 
				fi_ibv_rdm_get_sbuf(conn, ep, i);
			sprintf(p, "%1d:%1d ", buf->service_data.seq_num,
				buf->service_data.status);
		}
		VERBS_DBG(FI_LOG_EP_DATA,
			"conn %p sbufs status before: %s\n", conn, s);
	}
#endif // ENABLE_DEBUG
	struct fi_ibv_rdm_buf *sbuf = NULL;

	if (conn->sbuf_head->service_data.status == BUF_STATUS_FREE) {

		/* We have made whole circle. Reset buffer states */ 
		if (conn->sbuf_head == fi_ibv_rdm_get_sbuf(conn, ep, 0)) {
			do {
				fi_ibv_rdm_set_buffer_status(conn->sbuf_head,
					BUF_STATUS_FREE);
				fi_ibv_rdm_push_sbuff_head(conn, ep);
			} while (conn->sbuf_head != fi_ibv_rdm_get_sbuf(conn, ep, 0));
		}

		/* notification for receiver */
		fi_ibv_rdm_set_buffer_status(conn->sbuf_head, BUF_STATUS_RECEIVED);

		sbuf = conn->sbuf_head;
		fi_ibv_rdm_push_sbuff_head(conn, ep);
	}
#if ENABLE_DEBUG
	assert(sbuf ? (sbuf->service_data.status == BUF_STATUS_RECEIVED) : 1);
	{
		int i;
		char s[1024];
		char *p = s;
		sprintf(p, "N:%1d ", ep->n_buffs);
		p += 4;
		for (i = 0; i < ep->n_buffs; ++i, p += 4) {
			struct fi_ibv_rdm_buf *buf = 
				fi_ibv_rdm_get_sbuf(conn, ep, i);
			sprintf(p, "%1d:%1d ", buf->service_data.seq_num,
				buf->service_data.status);
		}
		VERBS_DBG(FI_LOG_EP_DATA,
			"conn %p sbufs status after:  %s\n", conn, s);
	}

	if (sbuf) {
		VERBS_DBG(FI_LOG_EP_DATA, "sending pkt # %d\n",
			  sbuf->service_data.seq_num);
	}
#endif // ENABLE_DEBUG

	VERBS_DBG(FI_LOG_EP_DATA,
		"conn %p sbuf allocated: %p, head: %p, begin: %p\n",
		conn, sbuf, conn->sbuf_head, conn->sbuf_mem_reg);

	return sbuf;
}

static inline void *
fi_ibv_rdm_rma_get_buf_head(struct fi_ibv_rdm_conn *conn,
			    struct fi_ibv_rdm_ep *ep)
{
	assert(conn);
	void *buf = NULL;

	if (conn->rmabuf_head->service_data.status == BUF_STATUS_FREE) {
		fi_ibv_rdm_set_buffer_status(conn->rmabuf_head, BUF_STATUS_BUSY);
		buf = conn->rmabuf_head;
		fi_ibv_rdm_push_rmabuff_head(conn, ep);
	}
	return buf;
}

static inline int
fi_ibv_rdm_check_connection(struct fi_ibv_rdm_conn *conn)
{
	return (conn->state == FI_VERBS_CONN_ESTABLISHED);
}

static inline struct fi_ibv_rdm_buf *
fi_ibv_rdm_prepare_send_resources(struct fi_ibv_rdm_conn *conn)
{
	return (fi_ibv_rdm_check_connection(conn) &&
		!TSEND_RESOURCES_IS_BUSY(conn, conn->ep)) ?
		fi_ibv_rdm_get_sbuf_head(conn, conn->ep) : NULL;
}

static inline void *
fi_ibv_rdm_rma_prepare_resources(struct fi_ibv_rdm_conn *conn)
{
	return (fi_ibv_rdm_check_connection(conn) &&
		!TSEND_RESOURCES_IS_BUSY(conn, conn->ep)) ?
		fi_ibv_rdm_rma_get_buf_head(conn, conn->ep) : NULL;
}

static inline int
fi_ibv_rdm_process_send_wc(struct fi_ibv_rdm_ep *ep,
			   struct ibv_wc *wc)
{
	if (wc->status != IBV_WC_SUCCESS) {
		return 1;
	}

	if (FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(wc->wr_id)) {
		VERBS_DBG(FI_LOG_EP_DATA, "CQ COMPL: SEND -> 0x1\n");
		struct fi_ibv_rdm_conn *conn =
			(struct fi_ibv_rdm_conn *)
			FI_IBV_RDM_UNPACK_SERVICE_WR(wc->wr_id);
		FI_IBV_RDM_DEC_SIG_POST_COUNTERS(conn, ep);

		return 0;
	} else {
		FI_IBV_DBG_OPCODE(wc->opcode, "SEND");
		struct fi_ibv_rdm_request *request =
			FI_IBV_RDM_UNPACK_WR(wc->wr_id);

		struct fi_ibv_rdm_tagged_send_completed_data data =
			{ .ep = ep };

		return fi_ibv_rdm_req_hndl(request, FI_IBV_EVENT_POST_LC,
					   &data);
	}
}

static inline void
fi_ibv_rdm_process_err_send_wc(struct fi_ibv_rdm_ep *ep,
			       struct ibv_wc *wc)
{
	if (wc->status != IBV_WC_SUCCESS) {
		struct fi_ibv_rdm_conn *conn;
		if (FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(wc->wr_id)) {
			conn = FI_IBV_RDM_UNPACK_SERVICE_WR(wc->wr_id);
		} else {
			struct fi_ibv_rdm_request *req =
				FI_IBV_RDM_UNPACK_WR(wc->wr_id);
			conn = req->minfo.conn;
			FI_IBV_RDM_DBG_REQUEST("to_pool: ", req,
					       FI_LOG_DEBUG);
			util_buf_release(ep->fi_ibv_rdm_request_pool, req);
		}
		VERBS_INFO(FI_LOG_EP_DATA, "got ibv_wc.status = %d:%s, "
			   "pend_send: %"PRIu32", connection: %p, request = %p (%s)\n",
			   wc->status,
			   ibv_wc_status_str(wc->status),
			   ep->posted_sends, conn,
			   (void *)wc->wr_id,
			   FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(wc->wr_id) ?
				"service" : "not service");
	}
}

#endif /* _VERBS_RDM_H */
