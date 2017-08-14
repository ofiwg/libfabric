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

#include <fi.h>
#include <fi_enosys.h>
#include <fi_util.h>
#include <fi_list.h>
#include <fi_proto.h>

#ifndef _RXM_H_
#define _RXM_H_

#endif

#define RXM_MAJOR_VERSION 1
#define RXM_MINOR_VERSION 0

#define RXM_BUF_SIZE 16384
#define RXM_IOV_LIMIT 4

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

extern struct fi_provider rxm_prov;
extern struct util_prov rxm_util_prov;
extern struct fi_ops_rma rxm_ops_rma;

struct rxm_fabric {
	struct util_fabric util_fabric;
	struct fid_fabric *msg_fabric;
};

struct rxm_conn {
	struct fid_ep *msg_ep;
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
	FUNC(RXM_NONE),		\
	FUNC(RXM_TX_NOBUF),	\
	FUNC(RXM_TX),		\
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

struct rxm_buf {
	/* Must stay at top */
	enum rxm_proto_state state;

	struct dlist_entry entry;
	void *desc;
	/* MSG EP / shared context to which bufs would be posted to */
	struct fid_ep *msg_ep;
};

struct rxm_rx_buf {
	/* Must stay at top */
	struct rxm_buf hdr;

	struct rxm_ep *ep;
	struct rxm_conn *conn;
	struct rxm_recv_queue *recv_queue;
	struct rxm_recv_entry *recv_entry;
	struct rxm_unexp_msg unexp_msg;
	uint64_t comp_flags;

	/* Used for large messages */
	struct rxm_iov match_iov[RXM_IOV_LIMIT];
	struct rxm_rma_iov *rma_iov;
	size_t index;
	struct fid_mr *mr[RXM_IOV_LIMIT];

	struct rxm_pkt pkt;
};

struct rxm_tx_buf {
	/* Must stay at top */
	struct rxm_buf hdr;

	struct rxm_pkt pkt;
};

struct rxm_tx_entry {
	/* Must stay at top */
	enum rxm_proto_state state;

	struct rxm_ep *ep;
	uint8_t count;
	void *context;
	uint64_t flags;
	uint64_t comp_flags;
	struct rxm_tx_buf *tx_buf;

	/* Used for large messages */
	struct fid_mr *mr[RXM_IOV_LIMIT];
	struct rxm_rx_buf *rx_buf;
};
DECLARE_FREESTACK(struct rxm_tx_entry, rxm_txe_fs);

struct rxm_recv_entry {
	struct dlist_entry entry;
	struct iovec iov[RXM_IOV_LIMIT];
	void *desc[RXM_IOV_LIMIT];
	uint8_t count;
	fi_addr_t addr;
	void *context;
	uint64_t flags;
	uint64_t tag;
	uint64_t ignore;
};
DECLARE_FREESTACK(struct rxm_recv_entry, rxm_recv_fs);

struct rxm_send_queue {
	struct rxm_txe_fs *fs;
	struct ofi_key_idx tx_key_idx;
	fastlock_t lock;
};

enum rxm_recv_queue_type {
	RXM_RECV_QUEUE_MSG,
	RXM_RECV_QUEUE_TAGGED,
};

struct rxm_recv_queue {
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
	struct dlist_entry buf_list;
	uint8_t local_mr;
	fastlock_t lock;
};

struct rxm_ep {
	struct util_ep util_ep;
	struct fi_info *rxm_info;
	struct fi_info *msg_info;
	struct fid_pep *msg_pep;
	struct fid_eq *msg_eq;
	struct fid_cq *msg_cq;
	struct fid_ep *srx_ctx;
	size_t comp_per_progress;

	struct rxm_buf_pool tx_pool;
	struct rxm_buf_pool rx_pool;

	struct rxm_send_queue send_queue;
	struct rxm_recv_queue recv_queue;
	struct rxm_recv_queue trecv_queue;
};

extern struct fi_provider rxm_prov;
extern struct fi_info rxm_info;
extern struct fi_fabric_attr rxm_fabric_attr;
extern struct fi_domain_attr rxm_domain_attr;
extern struct fi_tx_attr rxm_tx_attr;
extern struct fi_rx_attr rxm_rx_attr;

// TODO move to common code?
static inline int rxm_match_addr(fi_addr_t addr, fi_addr_t match_addr)
{
	return (addr == FI_ADDR_UNSPEC) || (match_addr == FI_ADDR_UNSPEC) ||
		(addr == match_addr);
}

static inline int rxm_match_tag(uint64_t tag, uint64_t ignore, uint64_t match_tag)
{
	return ((tag | ignore) == (match_tag | ignore));
}

static inline uint64_t rxm_ep_tx_flags(struct fid_ep *ep_fid) {
	struct util_ep *util_ep = container_of(ep_fid, struct util_ep,
					       ep_fid);
	return util_ep->tx_op_flags;
}

static inline uint64_t rxm_ep_rx_flags(struct fid_ep *ep_fid) {
	struct util_ep *util_ep = container_of(ep_fid, struct util_ep,
					       ep_fid);
	return util_ep->rx_op_flags;
}

int rxm_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
			void *context);
int rxm_info_to_core(uint32_t version, struct fi_info *rxm_info,
		     struct fi_info *core_info);
int rxm_info_to_rxm(uint32_t version, struct fi_info *core_info,
		    struct fi_info *info);
int rxm_domain_open(struct fid_fabric *fabric, struct fi_info *info,
			     struct fid_domain **dom, void *context);
int rxm_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
			 struct fid_cq **cq_fid, void *context);
void rxm_cq_progress(struct rxm_ep *rxm_ep);
int rxm_cq_comp(struct util_cq *util_cq, void *context, uint64_t flags, size_t len,
		void *buf, uint64_t data, uint64_t tag);
int rxm_cq_handle_data(struct rxm_rx_buf *rx_buf);

int rxm_endpoint(struct fid_domain *domain, struct fi_info *info,
			  struct fid_ep **ep, void *context);

void *rxm_conn_event_handler(void *arg);
int rxm_conn_connect(struct util_ep *util_ep, struct util_cmap_handle *handle,
		    fi_addr_t fi_addr);
int rxm_conn_process_connreq(struct rxm_ep *rxm_ep, struct fi_info *msg_info,
		void *data);
struct util_cmap_handle *rxm_conn_alloc(void);
void rxm_conn_close(struct util_cmap_handle *handle);
void rxm_conn_free(struct util_cmap_handle *handle);
int rxm_conn_signal(struct util_ep *util_ep, void *context,
		    enum ofi_cmap_signal signal);

int rxm_ep_repost_buf(struct rxm_rx_buf *buf);
int rxm_ep_prepost_buf(struct rxm_ep *rxm_ep, struct fid_ep *msg_ep);

int ofi_match_addr(fi_addr_t addr, fi_addr_t match_addr);
int ofi_match_tag(uint64_t tag, uint64_t ignore, uint64_t match_tag);
void rxm_pkt_init(struct rxm_pkt *pkt);
int rxm_ep_msg_mr_regv(struct rxm_ep *rxm_ep, const struct iovec *iov,
		       size_t count, uint64_t access, struct fid_mr **mr);
void rxm_ep_msg_mr_closev(struct fid_mr **mr, size_t count);
struct rxm_buf *rxm_buf_get(struct rxm_buf_pool *pool);
void rxm_buf_release(struct rxm_buf_pool *pool, struct rxm_buf *buf);

struct rxm_tx_entry *rxm_tx_entry_get(struct rxm_send_queue *queue);
struct rxm_recv_entry *rxm_recv_entry_get(struct rxm_recv_queue *queue);

void rxm_tx_entry_release(struct rxm_send_queue *queue, struct rxm_tx_entry *entry);
void rxm_recv_entry_release(struct rxm_recv_queue *queue, struct rxm_recv_entry *entry);
