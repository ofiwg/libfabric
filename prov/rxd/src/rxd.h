/*
 * Copyright (c) 2015-2017 Intel Corporation, Inc.  All rights reserved.
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

#include <pthread.h>
#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_trigger.h>

#include <ofi.h>
#include <ofi_proto.h>
#include <ofi_enosys.h>
#include <ofi_rbuf.h>
#include <ofi_list.h>
#include <ofi_util.h>

#ifndef _RXD_H_
#define _RXD_H_

#define RXD_MAJOR_VERSION 	(1)
#define RXD_MINOR_VERSION 	(0)
#define RXD_PROTOCOL_VERSION 	(1)
#define RXD_FI_VERSION 		FI_VERSION(1,6)

#define RXD_IOV_LIMIT		4
#define RXD_MAX_DGRAM_ADDR	128

#define RXD_MAX_TX_BITS 	10
#define RXD_MAX_RX_BITS 	10
#define RXD_TX_ID(seq, id)	(((seq) << RXD_MAX_TX_BITS) | id)
#define RXD_RX_ID(seq, id)	(((seq) << RXD_MAX_RX_BITS) | id)
#define RXD_TX_IDX_BITS		((1ULL << RXD_MAX_TX_BITS) - 1)
#define RXD_RX_IDX_BITS		((1ULL << RXD_MAX_RX_BITS) - 1)

#define RXD_BUF_POOL_ALIGNMENT	16
#define RXD_TX_POOL_CHUNK_CNT	1024
#define RXD_RX_POOL_CHUNK_CNT	1024

#define RXD_MAX_RX_CREDITS	16
#define RXD_MAX_PEER_TX		8
#define RXD_MAX_UNACKED		128

#define RXD_EP_MAX_UNEXP_PKT	512
#define RXD_EP_MAX_UNEXP_MSG	128

#define RXD_USE_OP_FLAGS	(1ULL << 61)
#define RXD_NO_COMPLETION	(1ULL << 62)

#define RXD_MAX_PKT_RETRY	50

extern int rxd_progress_spin_count;
extern int rxd_reposted_bufs;

extern struct fi_provider rxd_prov;
extern struct fi_info rxd_info;
extern struct fi_fabric_attr rxd_fabric_attr;
extern struct util_prov rxd_util_prov;
extern struct fi_ops_rma rxd_ops_rma;

enum {
	RXD_TX_CONN = 0,
	RXD_TX_MSG,
	RXD_TX_TAG,
	RXD_TX_WRITE,
	RXD_TX_READ_REQ,
	RXD_TX_READ_RSP,
};

struct rxd_fabric {
	struct util_fabric util_fabric;
	struct fid_fabric *dg_fabric;
};

struct rxd_domain {
	struct util_domain util_domain;
	struct fid_domain *dg_domain;

	ssize_t max_mtu_sz;
	int mr_mode;
	struct ofi_mr_map mr_map;//TODO use util_domain mr_map instead
};

struct rxd_av {
	struct util_av util_av;
	struct fid_av *dg_av;

	int dg_av_used;
	size_t dg_addrlen;
};

struct rxd_cq;
typedef int (*rxd_cq_write_fn)(struct rxd_cq *cq,
			       struct fi_cq_tagged_entry *cq_entry);
struct rxd_cq {
	struct util_cq util_cq;
	rxd_cq_write_fn write_fn;
};

struct rxd_peer {
	uint64_t		nxt_msg_id;
	uint64_t		exp_msg_id;
	uint64_t		conn_data;
	fi_addr_t		fiaddr;

	enum util_cmap_state	state;
	uint16_t		active_tx_cnt;
};

struct rxd_ep {
	struct util_ep util_ep;
	struct fid_ep *dg_ep;
	struct fid_cq *dg_cq;

	struct rxd_peer *peer_info;
	size_t max_peers;

	int conn_data_set;
	uint64_t conn_data;

	size_t rx_size;
	size_t credits;
//	uint64_t num_out;

	int do_local_mr;
	struct dlist_entry wait_rx_list;
	struct dlist_entry unexp_tag_list;
	struct dlist_entry unexp_msg_list;
	uint16_t num_unexp_pkt;
	uint16_t num_unexp_msg;

	struct util_buf_pool *tx_pkt_pool;
	struct util_buf_pool *rx_pkt_pool;
	struct slist rx_pkt_list;

	struct rxd_tx_entry_fs *tx_entry_fs;
	struct dlist_entry tx_entry_list;

	struct rxd_rx_entry_fs *rx_entry_fs;
	struct dlist_entry rx_entry_list;

	struct rxd_recv_fs *recv_fs;
	struct dlist_entry recv_list;

	struct rxd_trecv_fs *trecv_fs;
	struct dlist_entry trecv_list;
	fastlock_t lock;
};

static inline struct rxd_domain *rxd_ep_domain(struct rxd_ep *ep)
{
	return container_of(ep->util_ep.domain, struct rxd_domain, util_domain);
}

static inline struct rxd_av *rxd_ep_av(struct rxd_ep *ep)
{
	return container_of(ep->util_ep.av, struct rxd_av, util_av);
}

static inline struct rxd_cq *rxd_ep_tx_cq(struct rxd_ep *ep)
{
	return container_of(ep->util_ep.tx_cq, struct rxd_cq, util_cq);
}

static inline struct rxd_cq *rxd_ep_rx_cq(struct rxd_ep *ep)
{
	return container_of(ep->util_ep.rx_cq, struct rxd_cq, util_cq);
}

struct rxd_rx_buf {
	struct fi_context context;
	struct slist_entry entry;
	struct rxd_ep *ep;
	struct fid_mr *mr;
	char buf[];
};

struct rxd_rx_entry {
	struct ofi_op_hdr op_hdr;
	uint32_t exp_seg_no;
	uint64_t msg_id;
	uint64_t key;
	uint64_t done;
	uint64_t peer;
	uint16_t credits;
	uint32_t last_win_seg;
	fi_addr_t source;
	struct rxd_peer *peer_info;
	struct rxd_rx_buf *unexp_buf;
	uint64_t nack_stamp;
	struct dlist_entry entry;

	union {
		struct rxd_recv_entry *recv;
		struct rxd_trecv_entry *trecv;

		struct {
			struct iovec iov[RXD_IOV_LIMIT];
		} write;

		struct {
			struct rxd_tx_entry *tx_entry;
		} read_rsp;
	};

	union {
		struct dlist_entry wait_entry;
		struct dlist_entry unexp_entry;
	};
};
DECLARE_FREESTACK(struct rxd_rx_entry, rxd_rx_entry_fs);

struct rxd_tx_entry {
	fi_addr_t peer;
	uint64_t msg_id;
	uint64_t flags;
	uint64_t rx_key;
	uint64_t bytes_sent;
	uint32_t seg_no;
	uint32_t window;
	uint64_t retry_time;
	uint8_t retry_cnt;

	struct dlist_entry entry;
	struct dlist_entry pkt_list;

	uint8_t op_type;
	struct ofi_op_hdr op_hdr;

	union {
		struct {
			struct fi_msg msg;
			struct iovec msg_iov[RXD_IOV_LIMIT];
		} msg;

		struct {
			struct fi_msg_tagged tmsg;
			struct iovec msg_iov[RXD_IOV_LIMIT];
		} tmsg;

		struct {
			struct fi_msg_rma msg;
			struct iovec src_iov[RXD_IOV_LIMIT];
			struct fi_rma_iov dst_iov[RXD_IOV_LIMIT];
		} write;

		struct {
			struct fi_msg_rma msg;
			struct fi_rma_iov src_iov[RXD_IOV_LIMIT];
			struct iovec dst_iov[RXD_IOV_LIMIT];
		} read_req;

		struct {
			uint64_t peer_msg_id;
			uint8_t iov_count;
			struct iovec src_iov[RXD_IOV_LIMIT];
		} read_rsp;
	};
};
DECLARE_FREESTACK(struct rxd_tx_entry, rxd_tx_entry_fs);

struct rxd_recv_entry {
	struct dlist_entry entry;
	struct fi_msg msg;
	uint64_t flags;
	struct iovec iov[RXD_IOV_LIMIT];
	void *desc[RXD_IOV_LIMIT];
};
DECLARE_FREESTACK(struct rxd_recv_entry, rxd_recv_fs);

struct rxd_trecv_entry {
	struct dlist_entry entry;
	struct fi_msg_tagged msg;
	uint64_t flags;
	struct rxd_rx_entry *rx_entry;
	struct iovec iov[RXD_IOV_LIMIT];
	void *desc[RXD_IOV_LIMIT];
};
DECLARE_FREESTACK(struct rxd_trecv_entry, rxd_trecv_fs);

struct rxd_pkt_data_start {
	struct ofi_ctrl_hdr ctrl;
	struct ofi_op_hdr op;
	char data[];
};

struct rxd_pkt_data {
	struct ofi_ctrl_hdr ctrl;
	char data[];
};

#define RXD_PKT_FIRST	(1 << 0)
#define RXD_PKT_LAST	(1 << 1)
#define RXD_LOCAL_COMP	(1 << 2)
#define RXD_REMOTE_ACK	(1 << 3)
#define RXD_NOT_ACKED	RXD_REMOTE_ACK

struct rxd_pkt_meta {
	struct fi_context context;
	struct dlist_entry entry;
	struct rxd_tx_entry *tx_entry;
	struct rxd_ep *ep;
	struct fid_mr *mr;
	int flags;

	/* TODO: use iov and remove data copies */
	char pkt_data[]; /* rxd_pkt_data*, followed by data */
};

int rxd_info_to_core(uint32_t version, const struct fi_info *rxd_info,
		     struct fi_info *core_info);
int rxd_info_to_rxd(uint32_t version, const struct fi_info *core_info,
		    struct fi_info *info);

int rxd_fabric(struct fi_fabric_attr *attr,
	       struct fid_fabric **fabric, void *context);
int rxd_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		    struct fid_domain **dom, void *context);
int rxd_av_create(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		  struct fid_av **av, void *context);
int rxd_endpoint(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **ep, void *context);
int rxd_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context);
int rxd_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		  struct fid_cntr **cntr_fid, void *context);


/* AV sub-functions */
int rxd_av_insert_dg_addr(struct rxd_av *av, uint64_t hint_index,
			  const void *addr, fi_addr_t *dg_fiaddr);
fi_addr_t rxd_av_dg_addr(struct rxd_av *av, fi_addr_t fi_addr);
fi_addr_t rxd_av_fi_addr(struct rxd_av *av, fi_addr_t dg_fiaddr);
int rxd_av_dg_reverse_lookup(struct rxd_av *av, uint64_t start_idx,
			     const void *addr, fi_addr_t *dg_fiaddr);

/* EP sub-functions */
void rxd_handle_send_comp(struct fi_cq_msg_entry *comp);
void rxd_handle_recv_comp(struct rxd_ep *ep, struct fi_cq_msg_entry *comp);
int rxd_ep_repost_buff(struct rxd_rx_buf *rx_buf);
int rxd_ep_reply_ack(struct rxd_ep *ep, struct ofi_ctrl_hdr *in_ctrl,
		     uint8_t type, uint16_t seg_size, uint64_t rx_key,
		     uint64_t source, fi_addr_t dest);
struct rxd_peer *rxd_ep_getpeer_info(struct rxd_ep *rxd_ep, fi_addr_t addr);

void rxd_ep_check_unexp_msg_list(struct rxd_ep *ep,
				 struct rxd_recv_entry *recv_entry);
void rxd_ep_check_unexp_tag_list(struct rxd_ep *ep,
				 struct rxd_trecv_entry *trecv_entry);
void rxd_ep_handle_data_msg(struct rxd_ep *ep, struct rxd_peer *peer,
			    struct rxd_rx_entry *rx_entry,
			    struct iovec *iov, size_t iov_count,
			    struct ofi_ctrl_hdr *ctrl, void *data,
			    struct rxd_rx_buf *rx_buf);
void rxd_ep_free_acked_pkts(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry,
			    uint32_t seg_no);
ssize_t rxd_ep_start_xfer(struct rxd_ep *ep, struct rxd_peer *peer,
			  uint8_t op, struct rxd_tx_entry *tx_entry);
ssize_t rxd_ep_connect(struct rxd_ep *ep, struct rxd_peer *peer, fi_addr_t addr);
int rxd_mr_verify(struct rxd_domain *rxd_domain, ssize_t len,
		  uintptr_t *io_addr, uint64_t key, uint64_t access);


/* Tx/Rx entry sub-functions */
struct rxd_tx_entry *rxd_tx_entry_alloc(struct rxd_ep *ep,
	struct rxd_peer *peer, fi_addr_t addr, uint64_t flags, uint8_t op);
void rxd_tx_entry_progress(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry);
void rxd_tx_entry_discard(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry);
void rxd_tx_entry_free(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry);
void rxd_tx_entry_done(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry);
void rxd_set_timeout(struct rxd_tx_entry *tx_entry);

void rxd_tx_pkt_free(struct rxd_pkt_meta *pkt_meta);
void rxd_rx_entry_free(struct rxd_ep *ep, struct rxd_rx_entry *rx_entry);


/* CQ sub-functions */
void rxd_cq_report_error(struct rxd_cq *cq, struct fi_cq_err_entry *err_entry);
void rxd_cq_report_tx_comp(struct rxd_cq *cq, struct rxd_tx_entry *tx_entry);
void rxd_cntr_report_tx_comp(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry);

#endif
