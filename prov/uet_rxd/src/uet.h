/*
 * Copyright (c) 2015-2018 Intel Corporation, Inc.  All rights reserved.
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
#include <ofi_tree.h>
#include <ofi_atomic.h>
#include <ofi_indexer.h>
#include "uet_proto.h"

#ifndef _RXD_H_
#define _RXD_H_

#define UET_PROTOCOL_VERSION 	(2)

#define UET_MAX_MTU_SIZE	4096

#define UET_MAX_TX_BITS 	10
#define UET_MAX_RX_BITS 	10

#define UET_BUF_POOL_ALIGNMENT	16
#define UET_TX_POOL_CHUNK_CNT	1024
#define UET_RX_POOL_CHUNK_CNT	1024
#define UET_MAX_PENDING		128
#define UET_MAX_PKT_RETRY	50
#define UET_ADDR_INVALID	0

#define UET_PKT_IN_USE		(1 << 0)
#define UET_PKT_ACKED		(1 << 1)

#define UET_REMOTE_CQ_DATA	(1 << 0)
#define UET_NO_TX_COMP		(1 << 1)
#define UET_NO_RX_COMP		(1 << 2)
#define UET_INJECT		(1 << 3)
#define UET_TAG_HDR		(1 << 4)
#define UET_INLINE		(1 << 5)
#define UET_MULTI_RECV		(1 << 6)

#define UET_IDX_OFFSET(x)	(x + 1)

struct uet_env {
	int spin_count;
	int retry;
	int max_peers;
	int max_unacked;
	int rescan;
};

extern struct uet_env uet_env;
extern struct fi_provider uet_prov;
extern struct fi_info uet_info;
extern struct fi_fabric_attr uet_fabric_attr;
extern struct util_prov uet_util_prov;
extern struct fi_ops_msg uet_ops_msg;
extern struct fi_ops_tagged uet_ops_tagged;
extern struct fi_ops_rma uet_ops_rma;
extern struct fi_ops_atomic uet_ops_atomic;

struct uet_fabric {
	struct util_fabric util_fabric;
	struct fid_fabric *dg_fabric;
};

struct uet_domain {
	struct util_domain util_domain;
	struct fid_domain *dg_domain;

	ssize_t max_mtu_sz;
	ssize_t max_inline_msg;
	ssize_t max_inline_rma;
	ssize_t max_inline_atom;
	ssize_t max_seg_sz;
	struct ofi_mr_map mr_map;//TODO use util_domain mr_map instead
};

struct uet_peer {
	struct dlist_entry entry;
	fi_addr_t peer_addr;
	uint64_t tx_seq_no;
	uint64_t rx_seq_no;
	uint64_t last_rx_ack;
	uint64_t last_tx_ack;
	uint16_t rx_window;
	uint16_t tx_window;
	int retry_cnt;

	uint16_t unacked_cnt;
	uint8_t active;

	uint16_t curr_rx_id;
	uint16_t curr_tx_id;

	struct uet_unexp_msg *curr_unexp;
	struct dlist_entry tx_list;
	struct dlist_entry rx_list;
	struct dlist_entry rma_rx_list;
	struct dlist_entry unacked;
	struct dlist_entry buf_pkts;
};

struct uet_addr {
	fi_addr_t fi_addr;
	fi_addr_t dg_addr;
};

struct uet_av {
	struct util_av util_av;
	struct fid_av *dg_av;
	struct ofi_rbmap rbmap;

	int dg_av_used;
	size_t dg_addrlen;
	struct indexer fi_addr_idx;
	struct indexer rxdaddr_dg_idx;
	struct index_map rxdaddr_fi_idm;
};

struct uet_cq;
typedef int (*uet_cq_write_fn)(struct uet_cq *cq,
			       struct fi_cq_tagged_entry *cq_entry);
struct uet_cq {
	struct util_cq util_cq;
	uet_cq_write_fn write_fn;
};

enum uet_pool_type {
	UET_BUF_POOL_RX,
	UET_BUF_POOL_TX,
};

struct uet_buf_pool {
	enum uet_pool_type type;
	struct ofi_bufpool *pool;
	struct uet_ep *uet_ep;
};

struct uet_ep {
	struct util_ep util_ep;
	struct fid_ep *dg_ep;
	struct fid_cq *dg_cq;

	size_t rx_size;
	size_t tx_size;
	size_t tx_prefix_size;
	size_t rx_prefix_size;
	size_t min_multi_recv_size;
	int do_local_mr;
	int next_retry;
	int dg_cq_fd;
	uint32_t tx_flags;
	uint32_t rx_flags;

	size_t tx_msg_avail;
	size_t rx_msg_avail;
	size_t tx_rma_avail;
	size_t rx_rma_avail;

	struct uet_buf_pool tx_pkt_pool;
	struct uet_buf_pool rx_pkt_pool;
	struct slist rx_pkt_list;

	struct uet_buf_pool tx_entry_pool;
	struct uet_buf_pool rx_entry_pool;

	struct dlist_entry unexp_list;
	struct dlist_entry unexp_tag_list;
	struct dlist_entry rx_list;
	struct dlist_entry rx_tag_list;
	struct dlist_entry active_peers;
	struct dlist_entry rts_sent_list;
	struct dlist_entry ctrl_pkts;

	struct index_map peers_idm;
};
/* ensure ep lock is held before this function is called */
static inline struct uet_peer *uet_peer(struct uet_ep *ep, fi_addr_t uet_addr)
{
	return ofi_idm_lookup(&ep->peers_idm, (int) uet_addr);

}
static inline struct uet_domain *uet_ep_domain(struct uet_ep *ep)
{
	return container_of(ep->util_ep.domain, struct uet_domain, util_domain);
}

static inline struct uet_av *uet_ep_av(struct uet_ep *ep)
{
	return container_of(ep->util_ep.av, struct uet_av, util_av);
}

static inline struct uet_cq *uet_ep_tx_cq(struct uet_ep *ep)
{
	return container_of(ep->util_ep.tx_cq, struct uet_cq, util_cq);
}

static inline struct uet_cq *uet_ep_rx_cq(struct uet_ep *ep)
{
	return container_of(ep->util_ep.rx_cq, struct uet_cq, util_cq);
}

struct uet_x_entry {
	fi_addr_t peer;
	uint16_t tx_id;
	uint16_t rx_id;
	uint64_t bytes_done;
	uint64_t next_seg_no;
	uint64_t start_seq;
	uint64_t offset;
	uint64_t num_segs;
	uint32_t op;

	uint32_t flags;
	uint64_t ignore;
	uint8_t iov_count;
	uint8_t res_count;

	struct iovec iov[UET_IOV_LIMIT];
	struct iovec res_iov[UET_IOV_LIMIT];

	struct fi_cq_tagged_entry cq_entry;

	struct uet_pkt_entry *pkt;
	struct dlist_entry entry;
};

static inline uint32_t uet_tx_flags(uint64_t fi_flags)
{
	uint32_t uet_flags = 0;

	if (fi_flags & FI_REMOTE_CQ_DATA)
		uet_flags |= UET_REMOTE_CQ_DATA;
	if (fi_flags & FI_INJECT)
		uet_flags |= UET_INJECT;
	if (fi_flags & FI_COMPLETION)
		return uet_flags;

	return uet_flags | UET_NO_TX_COMP;
}

static inline uint32_t uet_rx_flags(uint64_t fi_flags)
{
	uint32_t uet_flags = 0;

	if (fi_flags & FI_MULTI_RECV)
		uet_flags |= UET_MULTI_RECV;
	if (fi_flags & FI_COMPLETION)
		return uet_flags;

	return uet_flags | UET_NO_RX_COMP;
}

struct uet_pkt_entry {
	struct dlist_entry d_entry;
	struct slist_entry s_entry;//TODO - keep both or make separate tx/rx pkt structs
	uint8_t flags;
	size_t pkt_size;
	uint64_t timestamp;
	struct fi_context context;
	struct fid_mr *mr;
	void *desc;
	fi_addr_t peer;
	void *pkt;
};

struct uet_unexp_msg {
	struct dlist_entry entry;
	struct uet_pkt_entry *pkt_entry;
	struct dlist_entry pkt_list;
	struct uet_base_hdr *base_hdr;
	struct uet_sar_hdr *sar_hdr;
	struct uet_tag_hdr *tag_hdr;
	struct uet_data_hdr *data_hdr;
	size_t msg_size;
	void *msg;
};

static inline int uet_pkt_type(struct uet_pkt_entry *pkt_entry)
{
	return ((struct uet_base_hdr *) (pkt_entry->pkt))->type;
}

static inline struct uet_base_hdr *uet_get_base_hdr(struct uet_pkt_entry *pkt_entry)
{
	return &((struct uet_ack_pkt *) (pkt_entry->pkt))->base_hdr;
}

static inline uint64_t uet_set_pkt_seq(struct uet_peer *peer,
				       struct uet_pkt_entry *pkt_entry)
{
	uet_get_base_hdr(pkt_entry)->seq_no = peer->tx_seq_no++;

	return uet_get_base_hdr(pkt_entry)->seq_no;
}

static inline struct uet_ext_hdr *uet_get_ext_hdr(struct uet_pkt_entry *pkt_entry)
{
	return &((struct uet_ack_pkt *) (pkt_entry->pkt))->ext_hdr;
}

static inline struct uet_sar_hdr *uet_get_sar_hdr(struct uet_pkt_entry *pkt_entry)
{
	return (struct uet_sar_hdr *) ((char *) pkt_entry->pkt +
		sizeof(struct uet_base_hdr));
}

static inline void uet_set_tx_pkt(struct uet_ep *ep, struct uet_pkt_entry *pkt_entry)
{
	pkt_entry->pkt = (void *) ((char *) pkt_entry +
			  sizeof(*pkt_entry) + ep->tx_prefix_size);
}

static inline void uet_set_rx_pkt(struct uet_ep *ep, struct uet_pkt_entry *pkt_entry)
{
	pkt_entry->pkt = (void *) ((char *) pkt_entry +
			  sizeof(*pkt_entry) + ep->rx_prefix_size);
}

static inline void *uet_pkt_start(struct uet_pkt_entry *pkt_entry)
{
	return (void *) ((char *) pkt_entry + sizeof(*pkt_entry));
}

static inline size_t uet_pkt_size(struct uet_ep *ep, struct uet_base_hdr *base_hdr,
				   void *ptr)
{
	return ((char *) ptr - (char *) base_hdr) + ep->tx_prefix_size;
}

static inline void uet_remove_free_pkt_entry(struct uet_pkt_entry *pkt_entry)
{
	dlist_remove(&pkt_entry->d_entry);
	ofi_buf_free(pkt_entry);
}

static inline void uet_free_unexp_msg(struct uet_unexp_msg *unexp_msg)
{
	ofi_buf_free(unexp_msg->pkt_entry);
	dlist_remove(&unexp_msg->entry);
	free(unexp_msg);
}

struct uet_match_attr {
	fi_addr_t	peer;
	uint64_t	tag;
	uint64_t	ignore;
};

static inline int uet_match_addr(fi_addr_t addr, fi_addr_t match_addr)
{
	return (addr == UET_ADDR_INVALID || addr == match_addr);
}

static inline int uet_match_tag(uint64_t tag, uint64_t ignore, uint64_t match_tag)
{
	return ((tag | ignore ) == (match_tag | ignore));
}

int uet_info_to_core(uint32_t version, const struct fi_info *uet_info,
		     const struct fi_info *base_info, struct fi_info *core_info);
int uet_info_to_rxd(uint32_t version, const struct fi_info *core_info,
		    const struct fi_info *base_info, struct fi_info *info);

int uet_fabric(struct fi_fabric_attr *attr,
	       struct fid_fabric **fabric, void *context);
int uet_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		    struct fid_domain **dom, void *context);
int uet_av_create(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		  struct fid_av **av, void *context);
int uet_endpoint(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **ep, void *context);
int uet_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context);
int uet_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		  struct fid_cntr **cntr_fid, void *context);
int uet_query_atomic(struct fid_domain *domain, enum fi_datatype datatype,
		     enum fi_op op, struct fi_atomic_attr *attr, uint64_t flags);

/* AV sub-functions */
int uet_av_insert_dg_addr(struct uet_av *av, const void *addr,
			  fi_addr_t *dg_fiaddr, uint64_t flags,
			  void *context);

/* Pkt resource functions */
ssize_t uet_ep_post_buf(struct uet_ep *ep);
void uet_ep_send_ack(struct uet_ep *uet_ep, fi_addr_t peer);
struct uet_pkt_entry *uet_get_tx_pkt(struct uet_ep *ep);
struct uet_x_entry *uet_get_tx_entry(struct uet_ep *ep, uint32_t op);
struct uet_x_entry *uet_get_rx_entry(struct uet_ep *ep, uint32_t op);
ssize_t uet_ep_send_pkt(struct uet_ep *ep, struct uet_pkt_entry *pkt_entry);
ssize_t uet_ep_post_data_pkts(struct uet_ep *ep, struct uet_x_entry *tx_entry);
void uet_insert_unacked(struct uet_ep *ep, fi_addr_t peer,
			struct uet_pkt_entry *pkt_entry);
ssize_t uet_send_rts_if_needed(struct uet_ep *uet_ep, fi_addr_t uet_addr);
int uet_start_xfer(struct uet_ep *ep, struct uet_x_entry *tx_entry);
void uet_init_data_pkt(struct uet_ep *ep, struct uet_x_entry *tx_entry,
		       struct uet_pkt_entry *pkt_entry);
void uet_init_base_hdr(struct uet_ep *uet_ep, void **ptr,
		       struct uet_x_entry *tx_entry);
void uet_init_sar_hdr(void **ptr, struct uet_x_entry *tx_entry,
		      size_t iov_count);
void uet_init_tag_hdr(void **ptr, struct uet_x_entry *tx_entry);
void uet_init_data_hdr(void **ptr, struct uet_x_entry *tx_entry);
void uet_init_rma_hdr(void **ptr, const struct fi_rma_iov *rma_iov,
		      size_t rma_count);
void uet_init_atom_hdr(void **ptr, enum fi_datatype datatype,
		       enum fi_op atomic_op);
size_t uet_init_msg(void **ptr, const struct iovec *iov, size_t iov_count,
		    size_t total_len, size_t avail_len);
static inline void uet_check_init_cq_data(void **ptr, struct uet_x_entry *tx_entry,
			      		  size_t *max_inline)
{
	if (tx_entry->flags & UET_REMOTE_CQ_DATA) {
		uet_init_data_hdr(ptr, tx_entry);
		*max_inline -= sizeof(tx_entry->cq_entry.data);
	}
}

/* Tx/Rx entry sub-functions */
struct uet_x_entry *uet_tx_entry_init_common(struct uet_ep *ep, fi_addr_t addr,
			uint32_t op, const struct iovec *iov, size_t iov_count,
			uint64_t tag, uint64_t data, uint32_t flags, void *context,
			struct uet_base_hdr **base_hdr, void **ptr);
struct uet_x_entry *uet_rx_entry_init(struct uet_ep *ep,
			const struct iovec *iov, size_t iov_count, uint64_t tag,
			uint64_t ignore, void *context, fi_addr_t addr,
			uint32_t op, uint32_t flags);
void uet_tx_entry_free(struct uet_ep *ep, struct uet_x_entry *tx_entry);
void uet_rx_entry_free(struct uet_ep *ep, struct uet_x_entry *rx_entry);
int uet_get_timeout(int retry_cnt);
uint64_t uet_get_retry_time(uint64_t start, int retry_cnt);

/* Generic message functions */
ssize_t uet_ep_generic_recvmsg(struct uet_ep *uet_ep, const struct iovec *iov,
			       size_t iov_count, fi_addr_t addr, uint64_t tag,
			       uint64_t ignore, void *context, uint32_t op,
			       uint32_t uet_flags, uint64_t flags);
ssize_t uet_ep_generic_sendmsg(struct uet_ep *uet_ep, const struct iovec *iov,
			       size_t iov_count, fi_addr_t addr, uint64_t tag,
			       uint64_t data, void *context, uint32_t op,
			       uint32_t uet_flags);
ssize_t uet_ep_generic_inject(struct uet_ep *uet_ep, const struct iovec *iov,
			      size_t iov_count, fi_addr_t addr, uint64_t tag,
			      uint64_t data, uint32_t op, uint32_t uet_flags);

/* Progress functions */
void uet_tx_entry_progress(struct uet_ep *ep, struct uet_x_entry *tx_entry,
			   int try_send);
void uet_handle_recv_comp(struct uet_ep *ep, struct fi_cq_msg_entry *comp);
void uet_handle_send_comp(struct uet_ep *ep, struct fi_cq_msg_entry *comp);
void uet_handle_error(struct uet_ep *ep);
void uet_progress_op(struct uet_ep *ep, struct uet_x_entry *rx_entry,
		     struct uet_pkt_entry *pkt_entry,
		     struct uet_base_hdr *base_hdr,
		     struct uet_sar_hdr *sar_hdr,
		     struct uet_tag_hdr *tag_hdr,
		     struct uet_data_hdr *data_hdr,
		     struct uet_rma_hdr *rma_hdr,
		     struct uet_atom_hdr *atom_hdr,
		     void **msg, size_t size);
void uet_ep_recv_data(struct uet_ep *ep, struct uet_x_entry *x_entry,
		      struct uet_data_pkt *pkt, size_t size);
void uet_progress_tx_list(struct uet_ep *ep, struct uet_peer *peer);
struct uet_x_entry *uet_progress_multi_recv(struct uet_ep *ep,
					    struct uet_x_entry *rx_entry,
					    size_t total_size);
void uet_ep_progress(struct util_ep *util_ep);
void uet_cleanup_unexp_msg(struct uet_unexp_msg *unexp_msg);

/* CQ sub-functions */
void uet_cq_report_error(struct uet_cq *cq, struct fi_cq_err_entry *err_entry);
void uet_cq_report_tx_comp(struct uet_cq *cq, struct uet_x_entry *tx_entry);


int uet_create_peer(struct uet_ep *ep, uint64_t uet_addr);

#endif
