/*
 * Copyright (c) 2015-2016 Intel Corporation, Inc.  All rights reserved.
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

#include <fi.h>
#include <fi_proto.h>
#include <fi_enosys.h>
#include <fi_rbuf.h>
#include <fi_list.h>
#include <fi_util.h>

#ifndef _RXD_H_
#define _RXD_H_

#define RXD_MAJOR_VERSION 	(1)
#define RXD_MINOR_VERSION 	(0)
#define RXD_PROTOCOL_VERSION 	(1)
#define RXD_FI_VERSION 		FI_VERSION(1,3)

#define RXD_IOV_LIMIT		(4)
#define RXD_DEF_CQ_CNT		(8)
#define RXD_DEF_EP_CNT 		(8)
#define RXD_AV_DEF_COUNT	(128)

#define RXD_MAX_TX_BITS 	(10)
#define RXD_MAX_RX_BITS 	(10)
#define RXD_TX_ID(seq, id)	(((seq) << RXD_MAX_TX_BITS) | id)
#define RXD_RX_ID(seq, id)	(((seq) << RXD_MAX_RX_BITS) | id)
#define RXD_TX_IDX_BITS		((1ULL << RXD_MAX_TX_BITS) - 1)
#define RXD_RX_IDX_BITS		((1ULL << RXD_MAX_RX_BITS) - 1)

#define RXD_BUF_POOL_ALIGNMENT	(16)
#define RXD_TX_POOL_CHUNK_CNT	(1024)
#define RXD_RX_POOL_CHUNK_CNT	(1024)

#define RXD_MAX_RX_WIN		(16)
#define RXD_MAX_OUT_TX_MSG	(8)
#define RXD_MAX_UNACKED		(128)

#define RXD_EP_MAX_UNEXP_PKT	(512)
#define RXD_EP_MAX_UNEXP_MSG	(128)

#define RXD_USE_OP_FLAGS	(1ULL << 61)
#define RXD_NO_COMPLETION	(1ULL << 62)
#define RXD_UNEXP_ENTRY		(1ULL << 63)


#define RXD_RETRY_TIMEOUT	(900)
#define RXD_WAIT_TIMEOUT	(2000)
#define RXD_MAX_PKT_RETRY	(50)

#define RXD_PKT_LOCAL_ACK	(1)
#define RXD_PKT_REMOTE_ACK	(1 << 1)
#define RXD_PKT_DONE (RXD_PKT_LOCAL_ACK | RXD_PKT_REMOTE_ACK)

#define RXD_PKT_MARK_LOCAL_ACK(_pkt)	((_pkt)->ref |= RXD_PKT_LOCAL_ACK)
#define RXD_PKT_MARK_REMOTE_ACK(_pkt)	((_pkt)->ref |= RXD_PKT_REMOTE_ACK)
#define RXD_PKT_IS_COMPLETE(_pkt)	((_pkt)->ref == RXD_PKT_DONE)

#define RXD_COPY_IOV_TO_BUF	(1)
#define RXD_COPY_BUF_TO_IOV	(2)

extern struct fi_provider rxd_prov;
extern struct fi_info rxd_info;
extern struct fi_fabric_attr rxd_fabric_attr;
extern struct util_prov rxd_util_prov;

enum {
	RXD_PKT_ORDR_OK = 0,
	RXD_PKT_ORDR_UNEXP,
	RXD_PKT_ORDR_DUP,
};

enum {
	RXD_TX_CONN = 0,
	RXD_TX_MSG,
	RXD_TX_TAG,
};

enum {
	RXD_PKT_STRT = 0,
	RXD_PKT_DATA,
	RXD_PKT_LAST,
};

struct rxd_fabric {
	struct util_fabric util_fabric;
	struct fid_fabric *dg_fabric;
};

struct rxd_domain {
	struct util_domain util_domain;
	struct fid_domain *dg_domain;

	size_t addrlen;
	ssize_t max_mtu_sz;
	uint64_t dg_mode;
	int do_progress;
	pthread_t progress_thread;
	fastlock_t lock;

	struct dlist_entry ep_list;
	struct dlist_entry cq_list;
};

struct rxd_av {
	struct fid_av *dg_av;
	fastlock_t lock;

	struct util_av util_av;
	int dg_av_used;
	size_t addrlen;
	size_t size;
};

struct rxd_cq;
typedef int (*rxd_cq_write_fn)(struct rxd_cq *cq,
			       struct fi_cq_tagged_entry *cq_entry);
struct rxd_cq {
	struct util_cq util_cq;
	struct fid_cq *dg_cq;
	struct rxd_domain *domain;
	rxd_cq_write_fn write_fn;
	struct dlist_entry dom_entry;
	struct util_buf_pool *unexp_pool;
	struct dlist_entry unexp_list;
	fastlock_t lock;
};

struct rxd_peer {
	uint64_t nxt_msg_id;
	uint64_t exp_msg_id;
	uint64_t conn_data;

	uint8_t addr_published;
	uint8_t conn_initiated;
	uint16_t num_msg_out;
	uint8_t pad[4];
};

struct rxd_ep {
	struct fid_ep ep;
	struct fid_ep *dg_ep;

	struct rxd_domain *domain;
	struct rxd_cq *rx_cq;
	struct rxd_cq *tx_cq;
	struct rxd_av *av;

	struct rxd_peer *peer_info;
	size_t max_peers;

	void *name;
	size_t addrlen;
	int conn_data_set;
	uint64_t conn_data;

	size_t rx_size;
	size_t credits;
	uint64_t num_out;

	int do_local_mr;
	uint64_t caps;

	struct dlist_entry dom_entry;
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

struct rxd_unexp_cq_entry {
	struct dlist_entry entry;
	struct fi_cq_msg_entry cq_entry;
};

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
	uint16_t window;
	uint32_t last_win_seg;
	fi_addr_t source;
	struct rxd_peer *peer_info;
	struct rxd_rx_buf *unexp_buf;
	uint64_t nack_stamp;
	struct dlist_entry entry;

	union {
		struct rxd_recv_entry *recv;
		struct rxd_trecv_entry *trecv;
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
	uint64_t done;
	uint64_t rx_key;
	uint32_t nxt_seg_no;
	uint32_t win_sz;
	int num_unacked;
	int is_waiting;
	uint64_t retry_stamp;

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
#define RXD_START_DATA_PKT_SZ (sizeof(struct rxd_pkt_data_start))
#define RXD_MAX_STRT_DATA_PKT_SZ(ep) (ep->domain->max_mtu_sz - RXD_START_DATA_PKT_SZ)

struct rxd_pkt_data {
	struct ofi_ctrl_hdr ctrl;
	char data[];
};
#define RXD_DATA_PKT_SZ (sizeof(struct rxd_pkt_data))
#define RXD_MAX_DATA_PKT_SZ(ep)	(ep->domain->max_mtu_sz - RXD_DATA_PKT_SZ)

struct rxd_pkt_meta {
	struct fi_context context;
	struct dlist_entry entry;
	struct rxd_tx_entry *tx_entry;
	struct rxd_ep *ep;
	struct fid_mr *mr;
	uint64_t us_stamp;
	uint8_t ref;
	uint8_t type;
	uint8_t retries;
	uint8_t pad[5];

	char pkt_data[]; /* rxd_pkt, followed by data */
};

int rxd_alter_layer_info(struct fi_info *layer_info, struct fi_info *base_info);
int rxd_alter_base_info(struct fi_info *base_info, struct fi_info *layer_info);

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


/* AV sub-functions */
fi_addr_t rxd_av_get_dg_addr(struct rxd_av *av, fi_addr_t fi_addr);
int rxd_av_insert_dg_av(struct rxd_av *av, const void *addr);
fi_addr_t rxd_av_get_fi_addr(struct rxd_av *av, fi_addr_t dg_addr);
int rxd_av_dg_reverse_lookup(struct rxd_av *av, uint64_t start_idx,
			     const void *addr, size_t addrlen, uint64_t *idx);

/* EP sub-functions */
void rxd_ep_lock_if_required(struct rxd_ep *rxd_ep);
void rxd_ep_unlock_if_required(struct rxd_ep *rxd_ep);
int rxd_ep_repost_buff(struct rxd_rx_buf *rx_buf);
void rxd_ep_progress(struct rxd_ep *ep);
int rxd_ep_reply_ack(struct rxd_ep *ep, struct ofi_ctrl_hdr *in_ctrl,
		     uint8_t type, uint16_t seg_size, uint64_t rx_key,
		     uint64_t source, fi_addr_t dest);
int rxd_ep_reply_nack(struct rxd_ep *ep, struct ofi_ctrl_hdr *in_ctrl,
		      uint32_t seg_no, uint64_t rx_key,
		      uint64_t source, fi_addr_t dest);
int rxd_ep_reply_discard(struct rxd_ep *ep, struct ofi_ctrl_hdr *in_ctrl,
			 uint32_t seg_no, uint64_t rx_key,
			 uint64_t source, fi_addr_t dest);
struct rxd_peer *rxd_ep_getpeer_info(struct rxd_ep *rxd_ep, fi_addr_t addr);

void rxd_ep_check_unexp_msg_list(struct rxd_ep *ep, struct rxd_recv_entry *recv_entry);
void rxd_ep_check_unexp_tag_list(struct rxd_ep *ep, struct rxd_trecv_entry *trecv_entry);
void rxd_ep_handle_data_msg(struct rxd_ep *ep, struct rxd_peer *peer,
			    struct rxd_rx_entry *rx_entry,
			    struct iovec *iov, size_t iov_count,
			    struct ofi_ctrl_hdr *ctrl, void *data,
			    struct rxd_rx_buf *rx_buf);
void rxd_ep_free_acked_pkts(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry,
			    uint32_t seg_no);
uint64_t rxd_ep_copy_iov_buf(const struct iovec *iov, size_t iov_count,
			     void *buf, uint64_t data_sz, uint64_t skip, int dir);
int rxd_ep_retry_pkt(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry,
		     struct rxd_pkt_meta *pkt);

/* Tx/Rx entry sub-functions */
struct rxd_tx_entry *rxd_tx_entry_acquire(struct rxd_ep *ep, struct rxd_peer *peer);
int rxd_tx_entry_progress(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry,
			  struct ofi_ctrl_hdr *ack);
void rxd_tx_entry_discard(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry);
void rxd_tx_entry_release(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry);
void rxd_tx_entry_done(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry);

struct rxd_pkt_meta *rxd_tx_pkt_acquire(struct rxd_ep *ep);
void rxd_tx_pkt_release(struct rxd_pkt_meta *pkt_meta);
void rxd_rx_entry_release(struct rxd_ep *ep, struct rxd_rx_entry *rx_entry);


/* CQ sub-functions */
void rxd_cq_progress(struct util_cq *util_cq);
void rxd_cq_report_error(struct rxd_cq *cq, struct fi_cq_err_entry *err_entry);
void rxd_cq_report_tx_comp(struct rxd_cq *cq, struct rxd_tx_entry *tx_entry);
void rxd_cq_report_rx_comp(struct rxd_cq *cq, struct rxd_rx_entry *rx_entry);

#endif
