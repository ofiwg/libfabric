/*
 * Copyright (c) 2016 Intel Corporation, Inc.  All rights reserved.
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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>

#include <ofi.h>
#include <ofi_enosys.h>
#include <ofi_util.h>
#include <ofi_list.h>
#include <ofi_proto.h>
#include <ofi_iov.h>

#ifndef _RXM_H_
#define _RXM_H_

#endif

#define RXM_MAJOR_VERSION 1
#define RXM_MINOR_VERSION 0

#define RXM_BUF_SIZE 16384
#define RXM_IOV_LIMIT 4

#define RXM_MR_MODES	(OFI_MR_BASIC_MAP | FI_MR_LOCAL)
#define RXM_MR_VIRT_ADDR(info) ((info->domain_attr->mr_mode == FI_MR_BASIC) ||\
				info->domain_attr->mr_mode & FI_MR_VIRT_ADDR)

#define RXM_MR_PROV_KEY(info) ((info->domain_attr->mr_mode == FI_MR_BASIC) ||\
			       info->domain_attr->mr_mode & FI_MR_PROV_KEY)

#define RXM_LOG_STATE(subsystem, pkt, prev_state, next_state) 			\
	FI_DBG(&rxm_prov, subsystem, "[LMT] msg_id: 0x%" PRIx64 " %s -> %s\n",	\
	       pkt.ctrl_hdr.msg_id, rxm_proto_state_str[prev_state],		\
	       rxm_proto_state_str[next_state])

#define RXM_LOG_STATE_TX(subsystem, tx_entry, next_state)		\
	RXM_LOG_STATE(subsystem, tx_entry->tx_buf->pkt, tx_entry->state,\
		      next_state)

#define RXM_LOG_STATE_RX(subsystem, rx_buf, next_state)			\
	RXM_LOG_STATE(subsystem, rx_buf->pkt, rx_buf->hdr.state,	\
		      next_state)

#define RXM_DBG_ADDR_TAG(subsystem, log_str, addr, tag) 	\
	FI_DBG(&rxm_prov, subsystem, log_str 			\
	       " (fi_addr: 0x%" PRIx64 " tag: 0x%" PRIx64 ")\n",\
	       addr, tag)

#define RXM_GET_PROTO_STATE(comp)			\
	(*(enum rxm_proto_state *)			\
	  ((unsigned char *)(comp)->op_context +	\
		offsetof(struct rxm_buf, state)))

#define RXM_SET_PROTO_STATE(comp, new_state)				\
do {									\
	(*(enum rxm_proto_state *)					\
	  ((unsigned char *)(comp)->op_context +			\
		offsetof(struct rxm_buf, state))) = (new_state);	\
} while (0)

extern struct fi_provider rxm_prov;
extern struct util_prov rxm_util_prov;
extern struct fi_ops_rma rxm_ops_rma;
extern int rxm_defer_requests;

extern size_t rxm_msg_tx_size;
extern size_t rxm_msg_rx_size;
extern size_t rxm_def_univ_size;

struct rxm_fabric {
	struct util_fabric util_fabric;
	struct fid_fabric *msg_fabric;
};

struct rxm_conn {
	struct fid_ep *msg_ep;
	struct dlist_entry postponed_tx_list;
	struct util_cmap_handle handle;
};

struct rxm_domain {
	struct util_domain util_domain;
	struct fid_domain *msg_domain;
	uint8_t mr_local;
};

struct rxm_mr {
	struct fid_mr mr_fid;
	struct fid_mr *msg_mr;
};

struct rxm_cm_data {
	struct sockaddr name;
	uint64_t conn_id;
};

struct rxm_rma_iov {
	uint8_t count;
	struct ofi_rma_iov iov[];
};

/*
 * Macros to generate enums and associated string values
 * e.g.
 * #define RXM_PROTO_STATES(FUNC)	\
 * 	FUNC(STATE1),			\
 * 	FUNC(STATE2),			\
 * 	...				\
 * 	FUNC(STATEn)
 *
 * enum rxm_state {
 * 	RXM_PROTO_STATES(OFI_ENUM_VAL)
 * };
 *
 * char *rxm_state_str[] = {
 * 	RXM_PROTO_STATES(OFI_STR)
 * };
 */

/* RXM protocol states / tx/rx context */
#define RXM_PROTO_STATES(FUNC)	\
	FUNC(RXM_TX_NOBUF),	\
	FUNC(RXM_TX),		\
	FUNC(RXM_TX_RMA),	\
	FUNC(RXM_RX),		\
	FUNC(RXM_LMT_TX),	\
	FUNC(RXM_LMT_ACK_WAIT),	\
	FUNC(RXM_LMT_READ),	\
	FUNC(RXM_LMT_ACK_SENT), \
	FUNC(RXM_LMT_ACK_RECVD),\
	FUNC(RXM_LMT_FINISH),

enum rxm_proto_state {
	RXM_PROTO_STATES(OFI_ENUM_VAL)
};

extern char *rxm_proto_state_str[];

struct rxm_pkt {
	struct ofi_ctrl_hdr ctrl_hdr;
	struct ofi_op_hdr hdr;
	char data[];
};

struct rxm_recv_match_attr {
	fi_addr_t addr;
	uint64_t tag;
	uint64_t ignore;
};

struct rxm_unexp_msg {
	struct dlist_entry entry;
	fi_addr_t addr;
	uint64_t tag;
};

struct rxm_iov {
	struct iovec iov[RXM_IOV_LIMIT];
	void *desc[RXM_IOV_LIMIT];
	uint8_t count;
};

struct rxm_rma_iov_storage {
	struct fi_rma_iov iov[RXM_IOV_LIMIT];
	uint8_t count;
};

enum rxm_buf_pool_type {
	RXM_BUF_POOL_RX		= 0,
	RXM_BUF_POOL_START	= RXM_BUF_POOL_RX,
	RXM_BUF_POOL_TX_MSG,
	RXM_BUF_POOL_TX_START	= RXM_BUF_POOL_TX_MSG,
	RXM_BUF_POOL_TX_TAGGED,
	RXM_BUF_POOL_TX_ACK,
	RXM_BUF_POOL_TX_LMT,
	RXM_BUF_POOL_TX_END	= RXM_BUF_POOL_TX_LMT,
	RXM_BUF_POOL_RMA,
	RXM_BUF_POOL_MAX,
};

struct rxm_buf {
	/* Must stay at top */
	struct fi_context fi_context;

	enum rxm_proto_state state;

	struct dlist_entry entry;
	void *desc;
	/* MSG EP / shared context to which bufs would be posted to */
	struct fid_ep *msg_ep;
};

struct rxm_rx_buf {
	/* Must stay at top */
	struct rxm_buf hdr;
	struct dlist_entry entry;

	struct rxm_ep *ep;
	struct dlist_entry repost_entry;
	struct rxm_conn *conn;
	struct rxm_recv_queue *recv_queue;
	struct rxm_recv_entry *recv_entry;
	struct rxm_unexp_msg unexp_msg;
	uint64_t comp_flags;
	uint8_t repost;

	/* Used for large messages */
	struct rxm_iov match_iov[RXM_IOV_LIMIT];
	struct rxm_rma_iov *rma_iov;
	size_t index;
	struct fid_mr *mr[RXM_IOV_LIMIT];

	/* Must stay at bottom */
	struct rxm_pkt pkt;
};

struct rxm_tx_buf {
	/* Must stay at top */
	struct rxm_buf hdr;

	enum rxm_buf_pool_type type;

	/* Must stay at bottom */
	struct rxm_pkt pkt;
};

struct rxm_rma_buf {
	/* Must stay at top */
	struct rxm_buf hdr;

	struct fi_msg_rma msg;
	struct rxm_iov rxm_iov;
	struct rxm_rma_iov_storage rxm_rma_iov;

	/* Must stay at bottom */
	struct rxm_pkt pkt;
};

struct rxm_tx_entry {
	/* Must stay at top */
	union {
		struct fi_context fi_context;
		struct dlist_entry postponed_entry;
	};

	enum rxm_proto_state state;

	struct rxm_ep *ep;
	uint8_t count;
	void *context;
	uint64_t flags;
	uint64_t comp_flags;
	union {
		struct rxm_tx_buf *tx_buf;
		struct rxm_rma_buf *rma_buf;
	};

	/* Used for large messages and RMA */
	struct fid_mr *mr[RXM_IOV_LIMIT];
	struct rxm_rx_buf *rx_buf;
};
DECLARE_FREESTACK(struct rxm_tx_entry, rxm_txe_fs);

struct rxm_recv_entry {
	struct dlist_entry entry;
	struct rxm_iov rxm_iov;
	fi_addr_t addr;
	void *context;
	uint64_t flags;
	uint64_t tag;
	uint64_t ignore;
	uint64_t comp_flags;
	size_t total_len;
	void *multi_recv_buf;
};
DECLARE_FREESTACK(struct rxm_recv_entry, rxm_recv_fs);

struct rxm_send_queue {
	struct rxm_txe_fs *fs;
	fastlock_t lock;
};

enum rxm_recv_queue_type {
	RXM_RECV_QUEUE_MSG,
	RXM_RECV_QUEUE_TAGGED,
};

struct rxm_recv_queue {
	struct rxm_ep *rxm_ep;
	enum rxm_recv_queue_type type;
	struct rxm_recv_fs *fs;
	struct dlist_entry recv_list;
	struct dlist_entry unexp_msg_list;
	dlist_func_t *match_recv;
	dlist_func_t *match_unexp;
	fastlock_t lock;
};

struct rxm_buf_pool {
	struct util_buf_pool *pool;
	enum rxm_buf_pool_type type;
	struct rxm_ep *ep;
	fastlock_t lock;
};

struct rxm_ep {
	struct util_ep 		util_ep;
	struct fi_info 		*rxm_info;
	struct fi_info 		*msg_info;
	struct fid_pep 		*msg_pep;
	struct fid_eq 		*msg_eq;
	struct fid_cq 		*msg_cq;
	int			msg_cq_fd;
	struct dlist_entry	msg_cq_fd_ref_list;
	struct fid_ep 		*srx_ctx;
	size_t 			comp_per_progress;
	int			msg_mr_local;
	int			rxm_mr_local;
	size_t			min_multi_recv_size;

	struct rxm_buf_pool	buf_pools[RXM_BUF_POOL_MAX];

	struct dlist_entry	post_rx_list;
	struct dlist_entry	repost_ready_list;

	struct rxm_send_queue	send_queue;
	struct rxm_recv_queue	recv_queue;
	struct rxm_recv_queue	trecv_queue;
};

struct rxm_ep_wait_ref {
	struct util_wait	*wait;
	struct dlist_entry	entry;
};

extern struct fi_provider rxm_prov;
extern struct fi_info rxm_info;
extern struct fi_fabric_attr rxm_fabric_attr;
extern struct fi_domain_attr rxm_domain_attr;
extern struct fi_tx_attr rxm_tx_attr;
extern struct fi_rx_attr rxm_rx_attr;

// TODO move to common code?
static inline int rxm_match_addr(fi_addr_t recv_addr, fi_addr_t rx_addr)
{
	return (recv_addr == FI_ADDR_UNSPEC) || (recv_addr == rx_addr);
}

static inline int rxm_match_tag(uint64_t tag, uint64_t ignore, uint64_t match_tag)
{
	return ((tag | ignore) == (match_tag | ignore));
}

#define rxm_ep_rx_flags(rxm_ep)	((rxm_ep)->util_ep.rx_op_flags)
#define rxm_ep_tx_flags(rxm_ep)	((rxm_ep)->util_ep.tx_op_flags)

int rxm_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
			void *context);
int rxm_info_to_core(uint32_t version, const struct fi_info *rxm_info,
		     struct fi_info *core_info);
int rxm_info_to_rxm(uint32_t version, const struct fi_info *core_info,
		    struct fi_info *info);
int rxm_domain_open(struct fid_fabric *fabric, struct fi_info *info,
			     struct fid_domain **dom, void *context);
int rxm_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
			 struct fid_cq **cq_fid, void *context);
ssize_t rxm_cq_handle_data(struct rxm_rx_buf *rx_buf);

int rxm_endpoint(struct fid_domain *domain, struct fi_info *info,
			  struct fid_ep **ep, void *context);

struct util_cmap *rxm_conn_cmap_alloc(struct rxm_ep *rxm_ep);

void rxm_ep_progress_one(struct util_ep *util_ep);
void rxm_ep_progress_multi(struct util_ep *util_ep);

int rxm_ep_prepost_buf(struct rxm_ep *rxm_ep, struct fid_ep *msg_ep);

int rxm_ep_msg_mr_regv(struct rxm_ep *rxm_ep, const struct iovec *iov,
		       size_t count, uint64_t access, struct fid_mr **mr);
void rxm_ep_msg_mr_closev(struct fid_mr **mr, size_t count);

void rxm_ep_handle_postponed_tx_op(struct rxm_ep *rxm_ep,
				   struct rxm_conn *rxm_conn,
				   struct rxm_tx_entry *tx_entry);
void rxm_ep_handle_postponed_rma_op(struct rxm_ep *rxm_ep,
				    struct rxm_conn *rxm_conn,
				    struct rxm_tx_entry *tx_entry);

static inline void rxm_cntr_inc(struct util_cntr *cntr)
{
	if (cntr)
		cntr->cntr_fid.ops->add(&cntr->cntr_fid, 1);
}

static inline void rxm_cntr_incerr(struct util_cntr *cntr)
{
	if (cntr)
		cntr->cntr_fid.ops->adderr(&cntr->cntr_fid, 1);
}

/* Caller must hold recv_queue->lock */
static inline struct rxm_rx_buf *
rxm_check_unexp_msg_list(struct rxm_recv_queue *recv_queue, fi_addr_t addr,
			 uint64_t tag, uint64_t ignore)
{
	struct rxm_recv_match_attr match_attr;
	struct dlist_entry *entry;

	if (dlist_empty(&recv_queue->unexp_msg_list))
		return NULL;

	match_attr.addr 	= addr;
	match_attr.tag 		= tag;
	match_attr.ignore 	= ignore;

	entry = dlist_find_first_match(&recv_queue->unexp_msg_list,
				       recv_queue->match_unexp, &match_attr);
	if (!entry)
		return NULL;

	RXM_DBG_ADDR_TAG(FI_LOG_EP_DATA, "Match for posted recv found in unexp"
			 " msg list\n", match_attr.addr, match_attr.tag);

	return container_of(entry, struct rxm_rx_buf, unexp_msg.entry);
}

static inline int
rxm_process_recv_entry(struct rxm_recv_queue *recv_queue,
		       struct rxm_recv_entry *recv_entry)
{
	struct rxm_rx_buf *rx_buf;

	fastlock_acquire(&recv_queue->lock);
	rx_buf = rxm_check_unexp_msg_list(recv_queue, recv_entry->addr,
					  recv_entry->tag, recv_entry->ignore);
	if (rx_buf) {
		dlist_remove(&rx_buf->unexp_msg.entry);
		rx_buf->recv_entry = recv_entry;
		fastlock_release(&recv_queue->lock);
		return rxm_cq_handle_data(rx_buf);
	}

	RXM_DBG_ADDR_TAG(FI_LOG_EP_DATA, "Enqueuing recv", recv_entry->addr,
			 recv_entry->tag);
	dlist_insert_tail(&recv_entry->entry, &recv_queue->recv_list);
	fastlock_release(&recv_queue->lock);

	return FI_SUCCESS;
}

static inline
struct rxm_buf *rxm_buf_get(struct rxm_buf_pool *pool)
{
	struct rxm_buf *buf;

	fastlock_acquire(&pool->lock);
	buf = util_buf_alloc(pool->pool);
	if (OFI_UNLIKELY(!buf)) {
		fastlock_release(&pool->lock);
		return NULL;
	}
	fastlock_release(&pool->lock);
	return buf;
}

static inline
void rxm_buf_release(struct rxm_buf_pool *pool, struct rxm_buf *buf)
{
	fastlock_acquire(&pool->lock);
	util_buf_release(pool->pool, buf);
	fastlock_release(&pool->lock);
}

static inline struct rxm_tx_buf *
rxm_tx_buf_get(struct rxm_ep *rxm_ep, enum rxm_buf_pool_type type)
{
	assert((type == RXM_BUF_POOL_TX_MSG) ||
	       (type == RXM_BUF_POOL_TX_TAGGED) ||
	       (type == RXM_BUF_POOL_TX_ACK) ||
	       (type == RXM_BUF_POOL_TX_LMT));
	return (struct rxm_tx_buf *)rxm_buf_get(&rxm_ep->buf_pools[type]);
}

static inline void
rxm_tx_buf_release(struct rxm_ep *rxm_ep, struct rxm_tx_buf *tx_buf)
{
	assert((tx_buf->type == RXM_BUF_POOL_TX_MSG) ||
	       (tx_buf->type == RXM_BUF_POOL_TX_TAGGED) ||
	       (tx_buf->type == RXM_BUF_POOL_TX_ACK) ||
	       (tx_buf->type == RXM_BUF_POOL_TX_LMT));
	tx_buf->pkt.hdr.flags &= ~OFI_REMOTE_CQ_DATA;
	rxm_buf_release(&rxm_ep->buf_pools[tx_buf->type],
			(struct rxm_buf *)tx_buf);
}

static inline struct rxm_rx_buf *rxm_rx_buf_get(struct rxm_ep *rxm_ep)
{
	return (struct rxm_rx_buf *)rxm_buf_get(
			&rxm_ep->buf_pools[RXM_BUF_POOL_RX]);
}

static inline void
rxm_rx_buf_release(struct rxm_ep *rxm_ep, struct rxm_rx_buf *rx_buf)
{
	rxm_buf_release(&rxm_ep->buf_pools[RXM_BUF_POOL_RX],
			(struct rxm_buf *)rx_buf);
}

static inline struct rxm_rma_buf *rxm_rma_buf_get(struct rxm_ep *rxm_ep)
{
	return (struct rxm_rma_buf *)rxm_buf_get(
			&rxm_ep->buf_pools[RXM_BUF_POOL_RMA]);
}

static inline void
rxm_rma_buf_release(struct rxm_ep *rxm_ep, struct rxm_rma_buf *rx_buf)
{
	rxm_buf_release(&rxm_ep->buf_pools[RXM_BUF_POOL_RMA],
			(struct rxm_buf *)rx_buf);
}

#define rxm_entry_pop(queue, entry)			\
	do {						\
		fastlock_acquire(&queue->lock);		\
		entry = freestack_isempty(queue->fs) ?	\
			NULL : freestack_pop(queue->fs);\
		fastlock_release(&queue->lock);		\
	} while (0)

#define rxm_entry_push(queue, entry)			\
	do {						\
		fastlock_acquire(&queue->lock);		\
		freestack_push(queue->fs, entry);	\
		fastlock_release(&queue->lock);		\
	} while (0)

#define rxm_tx_entry_cleanup(entry)		(entry)->tx_buf = NULL
#define rxm_recv_entry_cleanup(entry)		(entry)->total_len = 0

#define RXM_DEFINE_QUEUE_ENTRY(type, queue_type)				\
static inline struct rxm_ ## type ## _entry *					\
rxm_ ## type ## _entry_get(struct rxm_ ## queue_type ## _queue *queue)		\
{										\
	struct rxm_ ## type ## _entry *entry;					\
	rxm_entry_pop(queue, entry);						\
	if (!entry) {								\
		FI_WARN(&rxm_prov, FI_LOG_CQ,					\
			"Exhausted " #type "_entry freestack\n");		\
		return NULL;							\
	}									\
	return entry;								\
}										\
										\
static inline void								\
rxm_ ## type ## _entry_release(struct rxm_ ## queue_type ## _queue *queue,	\
			       struct rxm_ ## type ## _entry *entry)		\
{										\
	rxm_ ## type ## _entry_cleanup(entry);					\
	rxm_entry_push(queue, entry);						\
}

RXM_DEFINE_QUEUE_ENTRY(tx, send);
RXM_DEFINE_QUEUE_ENTRY(recv, recv);
