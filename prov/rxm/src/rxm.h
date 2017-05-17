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
#include <pthread.h>

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

#define RXM_IOV_LIMIT 4

/*
 * Macros to generate enums and associated string values
 * e.g.
 * #define RXM_STATES(FUNC)	\
 * 	FUNC(STATE1),		\
 * 	FUNC(STATE2),		\
 * 	...			\
 * 	FUNC(STATEn)
 *
 * enum rxm_state {
 * 	RXM_STATES(ENUM_VAL)
 * };
 *
 * char *rxm_state_str[] = {
 * 	RXM_STATES(STR)
 * };
 */
#define ENUM(X) X
#define STR(X) #X

#define RXM_MR_LOCAL(info) \
	((FI_VERSION_LT(info->fabric_attr->api_version, FI_VERSION(1, 5)) && \
	  (info->mode & FI_LOCAL_MR)) || (info->domain_attr->mr_mode & FI_MR_LOCAL))

#define RXM_MR_VIRT_ADDR(info) ((info->domain_attr->mr_mode == FI_MR_BASIC) ||\
				info->domain_attr->mr_mode & FI_MR_VIRT_ADDR)

#define RXM_MR_PROV_KEY(info) ((info->domain_attr->mr_mode == FI_MR_BASIC) ||\
			       info->domain_attr->mr_mode & FI_MR_PROV_KEY)

#define RXM_LOG_STATE(subsystem, prev_state, next_state) \
		FI_DBG(&rxm_prov, subsystem, \
		       "LMT: " #prev_state " -> " #next_state "\n");

extern struct fi_provider rxm_prov;
extern struct util_prov rxm_util_prov;

struct rxm_fabric {
	struct util_fabric util_fabric;
	struct fid_fabric *msg_fabric;
	struct fid_eq *msg_eq;
	pthread_t msg_listener_thread;
};

struct rxm_conn {
	struct fid_ep *msg_ep;
	struct util_cmap_handle handle;
};

struct rxm_domain {
	struct util_domain util_domain;
	struct fid_domain *msg_domain;
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

/* RXM protocol states / tx/rx context */
#define RXM_PROTO_STATES(FUNC)	\
	FUNC(RXM_NONE),		\
	FUNC(RXM_TX),		\
	FUNC(RXM_RX),		\
	FUNC(RXM_LMT_TX),	\
	FUNC(RXM_LMT_ACK_WAIT),	\
	FUNC(RXM_LMT_READ),	\
	FUNC(RXM_LMT_ACK_SENT), \
	FUNC(RXM_LMT_FINISH),

enum rxm_proto_state {
	RXM_PROTO_STATES(ENUM)
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
	/* MSG EP / shared context to which bufs would be posted to */
	struct fid_ep *msg_ep;
};

struct rxm_rx_buf {
	/* Must stay at top */
	struct rxm_buf hdr;

	struct rxm_ep *ep;
	struct rxm_conn *conn;
	struct rxm_recv_fs *recv_fs;
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

#define RXM_BUF_SIZE 16384
#define RXM_TX_DATA_SIZE (RXM_BUF_SIZE - sizeof(struct rxm_pkt))

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
	uint64_t msg_id;
	struct fid_mr *mr[RXM_IOV_LIMIT];
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
};

struct rxm_recv_queue {
	struct rxm_recv_fs *fs;
	struct dlist_entry recv_list;
	struct dlist_entry unexp_msg_list;
	dlist_func_t *match_recv;
	dlist_func_t *match_unexp;
};

struct rxm_buf_pool {
	struct util_buf_pool *pool;
	struct dlist_entry buf_list;
	uint8_t local_mr;
};

struct rxm_ep {
	struct util_ep util_ep;
	struct fi_info *rxm_info;
	struct fi_info *msg_info;
	struct fid_pep *msg_pep;
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

void *rxm_msg_listener(void *arg);
int rxm_msg_connect(struct rxm_ep *rxm_ep, fi_addr_t fi_addr,
		struct fi_info *msg_info);
int rxm_msg_process_connreq(struct rxm_ep *rxm_ep, struct fi_info *msg_info,
		void *data);
void rxm_conn_close(void *arg);
int rxm_get_conn(struct rxm_ep *rxm_ep, fi_addr_t fi_addr, struct rxm_conn **rxm_conn);

int rxm_ep_repost_buf(struct rxm_rx_buf *buf);
int ofi_match_addr(fi_addr_t addr, fi_addr_t match_addr);
int ofi_match_tag(uint64_t tag, uint64_t ignore, uint64_t match_tag);
void rxm_pkt_init(struct rxm_pkt *pkt);
int rxm_ep_msg_mr_regv(struct rxm_ep *rxm_ep, const struct iovec *iov,
		       size_t count, uint64_t access, struct fid_mr **mr);
void rxm_ep_msg_mr_closev(struct fid_mr **mr, size_t count);
struct rxm_buf *rxm_buf_get(struct rxm_buf_pool *pool);
void rxm_buf_release(struct rxm_buf_pool *pool, struct rxm_buf *buf);
void *rxm_buf_get_desc(struct rxm_buf_pool *pool, void *buf);
