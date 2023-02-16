/*
 * Copyright (c) 2019-2022 Amazon.com, Inc. or its affiliates.
 * All rights reserved.
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

#ifndef _EFA_RDM_EP_H
#define _EFA_RDM_EP_H

#include "efa_tp.h"

enum ibv_cq_ex_type {
	IBV_CQ,
	EFADV_CQ
};

enum rxr_lower_ep_type {
	EFA_EP = 1,
	SHM_EP,
};

/** @brief Information of a queued copy.
 *
 * This struct is used when receiving buffer is on device.
 * Under such circumstance, batching a series copies to
 * do them at the same time can avoid memory barriers between
 * copies, and improve performance.
 */
struct rxr_queued_copy {
	struct rxr_pkt_entry *pkt_entry;
	char *data;
	size_t data_size;
	size_t data_offset;
};

#define RXR_EP_MAX_QUEUED_COPY (8)

struct rxr_ep {
	struct efa_base_ep base_ep;

	/* per-version extra feature/request flag */
	uint64_t extra_info[RXR_MAX_NUM_EXINFO];

	struct ibv_cq_ex *ibv_cq_ex;

	enum ibv_cq_ex_type ibv_cq_ex_type;

	/* shm provider fid */
	bool use_shm_for_tx;
	struct fid_ep *shm_ep;
	struct fid_cq *shm_cq;

	/*
	 * RxR rx/tx queue sizes. These may be different from the core
	 * provider's rx/tx size and will either limit the number of possible
	 * receives/sends or allow queueing.
	 */
	size_t rx_size;
	size_t tx_size;
	size_t mtu_size;
	size_t rx_iov_limit;
	size_t tx_iov_limit;
	size_t inject_size;

	/* Endpoint's capability to support zero-copy rx */
	bool use_zcpy_rx;

	/* Application requested resource management support */
	int handle_resource_management;

	/* rx/tx queue size of core provider */
	size_t efa_max_outstanding_rx_ops;
	size_t efa_max_outstanding_tx_ops;
	size_t max_data_payload_size;

	/* Resource management flag */
	uint64_t rm_full;

	/* application's ordering requirements */
	uint64_t msg_order;

	/* Application's maximum msg size hint */
	size_t max_msg_size;

	/* Applicaiton's message prefix size. */
	size_t msg_prefix_size;

	/* RxR protocol's max header size */
	size_t max_proto_hdr_size;

	/* tx iov limit of EFA device */
	size_t efa_device_iov_limit;

	/* threshold to release multi_recv buffer */
	size_t min_multi_recv_size;

	/* buffer pool for send & recv */
	struct rxr_pkt_pool *efa_tx_pkt_pool;
	struct rxr_pkt_pool *efa_rx_pkt_pool;

	/*
	 * buffer pool for send & recv for shm as mtu size is different from
	 * the one of efa, and do not require local memory registration
	 */
	struct rxr_pkt_pool *shm_tx_pkt_pool;
	struct rxr_pkt_pool *shm_rx_pkt_pool;

	/* staging area for unexpected and out-of-order packets */
	struct rxr_pkt_pool *rx_unexp_pkt_pool;
	struct rxr_pkt_pool *rx_ooo_pkt_pool;

	/* staging area for read copy */
	struct rxr_pkt_pool *rx_readcopy_pkt_pool;
	int rx_readcopy_pkt_pool_used;
	int rx_readcopy_pkt_pool_max_used;

	/* datastructure to maintain rxr send/recv states */
	struct ofi_bufpool *op_entry_pool;
	/* data structure to maintain read */
	struct ofi_bufpool *read_entry_pool;
	/* data structure to maintain pkt rx map */
	struct ofi_bufpool *map_entry_pool;
	/* rxr medium message pkt_entry to rx_entry map */
	struct rxr_pkt_rx_map *pkt_rx_map;
	/*
	 * buffer pool for atomic response data, used by
	 * emulated fetch and compare atomic.
	 */
	struct ofi_bufpool *rx_atomrsp_pool;
	/* rx_entries with recv buf */
	struct dlist_entry rx_list;
	/* rx_entries without recv buf (unexpected message) */
	struct dlist_entry rx_unexp_list;
	/* rx_entries with tagged recv buf */
	struct dlist_entry rx_tagged_list;
	/* rx_entries without tagged recv buf (unexpected message) */
	struct dlist_entry rx_unexp_tagged_list;
	/* list of pre-posted recv buffers */
	struct dlist_entry rx_posted_buf_list;
	/* list of pre-posted recv buffers for shm */
	struct dlist_entry rx_posted_buf_shm_list;
	/* op entries with queued rnr packets */
	struct dlist_entry op_entry_queued_rnr_list;
	/* op entries with queued ctrl packets */
	struct dlist_entry op_entry_queued_ctrl_list;
	/* op entries with queued read requests */
	struct dlist_entry op_entry_queued_read_list;
	/* tx/rx_entries used by long CTS msg/write/read protocol
         * which have data to be sent */
	struct dlist_entry op_entry_longcts_send_list;
	/* read entries with data to be read */
	struct dlist_entry read_pending_list;
	/* list of #efa_rdm_peer that are in backoff due to RNR */
	struct dlist_entry peer_backoff_list;
	/* list of #efa_rdm_peer that will retry posting handshake pkt */
	struct dlist_entry handshake_queued_peer_list;

#if ENABLE_DEBUG
	/* tx/rx_entries waiting to receive data in
         * long CTS msg/read/write protocols */
	struct dlist_entry op_entry_recv_list;
	/* counter tracking op_entry_recv_list */
	size_t pending_recv_counter;

	/* rx packets being processed or waiting to be processed */
	struct dlist_entry rx_pkt_list;

	/* tx packets waiting for send completion */
	struct dlist_entry tx_pkt_list;

	size_t efa_total_posted_tx_ops;
	size_t shm_total_posted_tx_ops;
	size_t send_comps;
	size_t failed_send_comps;
	size_t recv_comps;
#endif
	/* track allocated rx_entries and tx_entries for endpoint cleanup */
	struct dlist_entry rx_entry_list;
	struct dlist_entry tx_entry_list;

	/*
	 * number of posted RX packets for shm
	 */
	size_t shm_rx_pkts_posted;
	/*
	 * number of RX packets to be posted by progress engine for shm.
	 * It exists because posting RX packets by bulk is more efficient.
	 */
	size_t shm_rx_pkts_to_post;
	/*
	 * number of posted RX packets for EFA device
	 */
	size_t efa_rx_pkts_posted;
	/*
	 * Number of RX packets to be posted by progress engine for EFA device.
	 * It exists because posting RX packets by bulk is more efficient.
	 */
	size_t efa_rx_pkts_to_post;

	/* number of outstanding tx ops on efa device */
	size_t efa_outstanding_tx_ops;
	/* number of outstanding tx ops on shm */
	size_t shm_outstanding_tx_ops;

	struct rxr_queued_copy queued_copy_vec[RXR_EP_MAX_QUEUED_COPY];
	int queued_copy_num;
	int blocking_copy_rx_entry_num; /* number of RX entries that are using gdrcopy/cudaMemcpy */

	int	hmem_p2p_opt; /* what to do for hmem transfers */
};

int rxr_ep_flush_queued_blocking_copy_to_hmem(struct rxr_ep *ep);

#define rxr_rx_flags(rxr_ep) ((rxr_ep)->base_ep.util_ep.rx_op_flags)
#define rxr_tx_flags(rxr_ep) ((rxr_ep)->base_ep.util_ep.tx_op_flags)

struct efa_ep_addr *rxr_ep_raw_addr(struct rxr_ep *ep);

const char *rxr_ep_raw_addr_str(struct rxr_ep *ep, char *buf, size_t *buflen);

struct efa_ep_addr *rxr_ep_get_peer_raw_addr(struct rxr_ep *ep, fi_addr_t addr);

const char *rxr_ep_get_peer_raw_addr_str(struct rxr_ep *ep, fi_addr_t addr, char *buf, size_t *buflen);

struct efa_rdm_peer *rxr_ep_get_peer(struct rxr_ep *ep, fi_addr_t addr);

struct rxr_op_entry *rxr_ep_alloc_tx_entry(struct rxr_ep *rxr_ep,
					   const struct fi_msg *msg,
					   uint32_t op,
					   uint64_t tag,
					   uint64_t flags);

struct rxr_op_entry *rxr_ep_alloc_rx_entry(struct rxr_ep *ep,
					   fi_addr_t addr, uint32_t op);

void rxr_ep_record_tx_op_submitted(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry);

void rxr_ep_record_tx_op_completed(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry);

static inline size_t rxr_get_rx_pool_chunk_cnt(struct rxr_ep *ep)
{
	return MIN(ep->efa_max_outstanding_rx_ops, ep->rx_size);
}

static inline size_t rxr_get_tx_pool_chunk_cnt(struct rxr_ep *ep)
{
	return MIN(ep->efa_max_outstanding_tx_ops, ep->tx_size);
}

static inline int rxr_need_sas_ordering(struct rxr_ep *ep)
{
	return ep->msg_order & FI_ORDER_SAS;
}

static inline int rxr_ep_use_zcpy_rx(struct rxr_ep *ep, struct fi_info *info)
{
	return !(ep->base_ep.util_ep.caps & FI_DIRECTED_RECV) &&
		!(ep->base_ep.util_ep.caps & FI_TAGGED) &&
		!(ep->base_ep.util_ep.caps & FI_ATOMIC) &&
		(ep->max_msg_size <= ep->mtu_size - ep->max_proto_hdr_size) &&
		!rxr_need_sas_ordering(ep) &&
		info->mode & FI_MSG_PREFIX &&
		rxr_env.use_zcpy_rx;
}

/* Initialization functions */
int rxr_endpoint(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **ep, void *context);

/* EP sub-functions */
void rxr_ep_progress(struct util_ep *util_ep);
void rxr_ep_progress_internal(struct rxr_ep *rxr_ep);

int rxr_ep_post_user_recv_buf(struct rxr_ep *ep, struct rxr_op_entry *rx_entry,
			      uint64_t flags);

struct efa_rdm_peer;

int rxr_ep_determine_rdma_read_support(struct rxr_ep *ep, fi_addr_t addr,
				       struct efa_rdm_peer *peer);
int rxr_ep_determine_rdma_write_support(struct rxr_ep *ep, fi_addr_t addr,
					struct efa_rdm_peer *peer);

struct rxr_op_entry *rxr_ep_lookup_mediumrtm_rx_entry(struct rxr_ep *ep,
						      struct rxr_pkt_entry *pkt_entry);

void rxr_ep_record_mediumrtm_rx_entry(struct rxr_ep *ep,
				      struct rxr_pkt_entry *pkt_entry,
				      struct rxr_op_entry *rx_entry);

void rxr_ep_queue_rnr_pkt(struct rxr_ep *ep,
			  struct dlist_entry *list,
			  struct rxr_pkt_entry *pkt_entry);

void rxr_ep_handle_misc_shm_completion(struct rxr_ep *ep,
				       struct fi_cq_data_entry *cq_entry,
				       fi_addr_t src_addr);

static inline
struct efa_domain *rxr_ep_domain(struct rxr_ep *ep)
{
	return container_of(ep->base_ep.util_ep.domain, struct efa_domain, util_domain);
}

/**
 * @brief return whether this endpoint should write error cq entry for RNR.
 *
 * For an endpoint to write RNR completion, two conditions must be met:
 *
 * First, the end point must be able to receive RNR completion from rdma-core,
 * which means rnr_etry must be less then EFA_RNR_INFINITE_RETRY.
 *
 * Second, the app need to request this feature when opening endpoint
 * (by setting info->domain_attr->resource_mgmt to FI_RM_DISABLED).
 * The setting was saved as rxr_ep->handle_resource_management.
 *
 * @param[in]	ep	endpoint
 */
static inline
bool rxr_ep_should_write_rnr_completion(struct rxr_ep *ep)
{
	return (rxr_env.rnr_retry < EFA_RNR_INFINITE_RETRY) &&
		(ep->handle_resource_management == FI_RM_DISABLED);
}

/*
 * RM flags
 */
#define RXR_RM_TX_CQ_FULL	BIT_ULL(0)
#define RXR_RM_RX_CQ_FULL	BIT_ULL(1)

/*
 * today we have only cq res check, in future we will have ctx, and other
 * resource check as well.
 */
static inline
uint64_t is_tx_res_full(struct rxr_ep *ep)
{
	return ep->rm_full & RXR_RM_TX_CQ_FULL;
}

static inline
uint64_t is_rx_res_full(struct rxr_ep *ep)
{
	return ep->rm_full & RXR_RM_RX_CQ_FULL;
}

static inline
void rxr_rm_rx_cq_check(struct rxr_ep *ep, struct util_cq *rx_cq)
{
	ofi_genlock_lock(&rx_cq->cq_lock);
	if (ofi_cirque_isfull(rx_cq->cirq))
		ep->rm_full |= RXR_RM_RX_CQ_FULL;
	else
		ep->rm_full &= ~RXR_RM_RX_CQ_FULL;
	ofi_genlock_unlock(&rx_cq->cq_lock);
}

static inline
void rxr_rm_tx_cq_check(struct rxr_ep *ep, struct util_cq *tx_cq)
{
	ofi_genlock_lock(&tx_cq->cq_lock);
	if (ofi_cirque_isfull(tx_cq->cirq))
		ep->rm_full |= RXR_RM_TX_CQ_FULL;
	else
		ep->rm_full &= ~RXR_RM_TX_CQ_FULL;
	ofi_genlock_unlock(&tx_cq->cq_lock);
}

/*
 * @brief: check whether we should use p2p for this transaction
 *
 * @param[in]	ep	rxr_ep
 * @param[in]	efa_mr	memory registration struct
 *
 * @return: 0 if p2p should not be used, 1 if it should, and negative FI code
 * if the transfer should fail.
 */
static inline
int rxr_ep_use_p2p(struct rxr_ep *rxr_ep, struct efa_mr *efa_mr)
{
	if (!efa_mr)
		return 0;

	/*
	 * always send from host buffers if we have a descriptor
	 */
	if (efa_mr->peer.iface == FI_HMEM_SYSTEM)
		return 1;

	if (rxr_ep_domain(rxr_ep)->hmem_info[efa_mr->peer.iface].p2p_supported_by_device)
		return (rxr_ep->hmem_p2p_opt != FI_HMEM_P2P_DISABLED);

	if (rxr_ep->hmem_p2p_opt == FI_HMEM_P2P_REQUIRED) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Peer to peer support is currently required, but not available.\n");
		return -FI_ENOSYS;
	}

	return 0;
}

static inline
ssize_t efa_rdm_ep_post_flush(struct rxr_ep *ep, struct ibv_send_wr **bad_wr)
{
	ssize_t ret;

#if HAVE_LTTNG
	struct ibv_send_wr *head = ep->base_ep.xmit_more_wr_head.next;

	while (head) {
		efa_tracepoint_wr_id_post_send((void *) head->wr_id);
		head = head->next;
	}
#endif

	ret = ibv_post_send(ep->base_ep.qp->ibv_qp, ep->base_ep.xmit_more_wr_head.next, bad_wr);
	ep->base_ep.xmit_more_wr_head.next = NULL;
	ep->base_ep.xmit_more_wr_tail = &ep->base_ep.xmit_more_wr_head;
	return ret;
}

#endif
