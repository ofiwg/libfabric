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

#ifndef _RXR_PKT_ENTRY_H
#define _RXR_PKT_ENTRY_H

#include <ofi_list.h>

#define RXR_PKT_ENTRY_IN_USE		BIT_ULL(0) /**< this packet entry is being used */
#define RXR_PKT_ENTRY_RNR_RETRANSMIT	BIT_ULL(1) /**< this packet entry encountered RNR and is being retransmitted*/
#define RXR_PKT_ENTRY_LOCAL_READ	BIT_ULL(2) /**< this packet entry is used as context of a local read operation */
#define RXR_PKT_ENTRY_DC_LONGCTS_DATA	BIT_ULL(3) /**< this DATA packet entry is used by a delivery complete LONGCTS send/write protocol*/
#define RXR_PKT_ENTRY_LOCAL_WRITE	BIT_ULL(4) /**< this packet entry is used as context of an RDMA Write to self */


/**
 * @enum for packet entry allocation type
 */
enum rxr_pkt_entry_alloc_type {
	RXR_PKT_FROM_EFA_TX_POOL = 1, /**< packet is allocated from `ep->efa_tx_pkt_pool` */
	RXR_PKT_FROM_EFA_RX_POOL,     /**< packet is allocated from `ep->efa_rx_pkt_pool` */
	RXR_PKT_FROM_SHM_TX_POOL,     /**< packet is allocated from `ep->shm_tx_pkt_pool` */
	RXR_PKT_FROM_SHM_RX_POOL,     /**< packet is allocated from `ep->shm_rx_pkt_pool` */
	RXR_PKT_FROM_UNEXP_POOL,      /**< packet is allocated from `ep->rx_unexp_pkt_pool` */
	RXR_PKT_FROM_OOO_POOL,	      /**< packet is allocated from e`p->rx_ooo_pkt_pool` */
	RXR_PKT_FROM_USER_BUFFER,     /**< packet is from user provided buffer` */
	RXR_PKT_FROM_READ_COPY_POOL,  /**< packet is allocated from `ep->rx_readcopy_pkt_pool` */
};

struct rxr_pkt_sendv {
	/**
	 * @brief number of iovec to be passed to device and sent over wire
	 * 
	 * Because core EP current only support 2 iov,
	 * and for the sake of code simplicity, we use 2 iov.
	 * One for header, and the other for data.
	 * iov_count here is used as an indication
	 * of whether iov is used, it is either 0 or 2.
	 */
	int iov_count;
	struct iovec iov[2];
	void *desc[2];
};

/* The efa_send_wr and efa_recv_wr structs are used by both
 * RDM provider and DGRAM provider
 * TODO: Move to a common file that's imported by both providers
 */
struct efa_send_wr {
	/** @brief Work request struct used by rdma-core */
	struct ibv_send_wr wr;

	/** @brief Scatter gather element array
	 *
	 * @details
	 * EFA device supports a maximum of 2 iov/SGE
	 */
	struct ibv_sge sge[2];
};

struct efa_recv_wr {
	/** @brief Work request struct used by rdma-core */
	struct ibv_recv_wr wr;

	/** @brief Scatter gather element array
	 *
	 * @details
	 * EFA device supports a maximum of 2 iov/SGE
	 * For receive, we only use 1 SGE
	 */
	struct ibv_sge sge[1];
};

/**
 * @brief Packet entry
 * 
 * rxr_pkt_entry is used the following occassions:
 * 
 * First, it is used as the context of the request EFA provider posted to EFA device and SHM:
 * 
 * For each request EFA provider submits to (EFA device or SHM), it will allocate a packet entry.
 * 
 * When the request was submitted, the pointer of the packet entry will be used as context.
 * For EFA device, context is work request ID (`wr_id`), For SHM, context is the `op_context`
 * in a `fi_msg`.
 * 
 * When the request was completed, EFA device or SHM will return a completion entry, with the
 * the pointer to the rxr_pkt_entry in it.
 * 
 * Sometimes, the completion can be a Receiver Not Ready (RNR) error completion.
 * In that case the packet entry will be queued and resubmitted. For the resubmission,
 * the packet entry must contain all the information of the request.
 * 
 * An operation can be either a TX operation or a receive (RX) operation
 * 
 * For EFA device, a TX operation can be send/read.
 * 
 * For SHM, a TX operation can be send/read/write/atomic.
 * 
 * When used as context of request, packet was allocated from endpoint's shm/efa_tx/rx_pool.
 * When the request is to EFA device, the packet's memory must be registered to EFA device,
 * and the memory registration must be stored as the `mr` field of packet entry.
 * 
 * Second, packet entries can be used to store received packet entries that is
 * unexpected or out-of-order. This is because the efa/shm_rx_pkt_pool's size is fixed,
 * therefore it cannot be used to hold unexpected/out-of-order packets. When an unexpected/out-of-order
 * packet is received, a new packet entry will be cloned from unexpected/ooo_pkt_pool.
 * The old packet will be released then reposted to EFA device or SHM. The new packet
 * (allocated from unexpected/ooo_pkt_pool)'s memory is not registered
 * 
 * Finally, packet entries can be used to support local read copy. Local read copy means
 * to copy data from a packet entry to HMEM receive buffer through EFA device's read capability.
 * Local require a packet entry's memory to be registered with device. If the packet entry's memory
 * is not registered (when it is unexpected or out-of-order). A new packet entry will be cloned
 * using endpoint's read_copy_pkt_pool, whose memory was registered.
 */

struct rxr_pkt_entry {
	/**
	 * entry to the linked list of outstanding/queued packet entries
	 *
	 * `entry` is used for sending only.
	 * It is either linked to `peer->outstanding_tx_pkts` (after a packet has been successfully sent, but it get a completion),
	 * or linked to `op_entry->queued_pkts` (after it encountered RNR error completion).
	 */
	struct dlist_entry entry;
#if ENABLE_DEBUG
	/** @brief entry to a linked list of posted buf list */
	struct dlist_entry dbg_entry;
	uint8_t pad[48];
#endif
	/** @brief pointer to #rxr_op_entry or #rxr_read_entry */
	void *x_entry;

	/** @brief number of bytes sent/received over wire */
	size_t pkt_size;

	/**
	 * @brief memory registration
	 *
	 * @details
	 * If this packet is used by EFA device, `mr` the memory registration of wiredata over the EFA device.
	 * If this packet is used by SHM, `mr` is NULL because SHM does not require memory registration
	 *
	 * @todo
	 * Use type `struct ibv_mr` instead of `struct fid_mr` for this field
	 */
	struct fid_mr *mr;
	/**
	 * @brief peer address
	 *
	 * @details
	 * When sending a packet, `addr` will be provided by application and it cannot be FI_ADDR_NOTAVAIL.
	 * However, after a packet is sent, application can remove a peer by calling fi_av_remove().
	 * When removing the peering, `addr` will be set to FI_ADDR_NOTAVAIL. Later, when device report
	 * completion for such a TX packet, the TX completion will be ignored.
	 *
	 * When receiving a packet, lower device will set `addr`. If the sender's address is not in
	 * address vector (AV), `lower device will set `addr` to FI_ADDR_NOTAVAIL. This can happen in
	 * two scenarios:
	 *
	 * 1. There has been no prior communication with the peer. In this case, the packet should have
	 *    peer's raw address in the header, and progress engine will insert the raw address into
	 *    address vector, and update `addr`.
	 *
	 * 2. This packet is from a peer whose address has been removed from AV. In this case, the
	 *    recived packet will be ignored because all resources associated with peer has been released.
	 */
	fi_addr_t addr;

	/** @brief indicate where the memory of this packet entry reside */
	enum rxr_pkt_entry_alloc_type alloc_type;

	/** 
	 * @brief flags indicating the status of the packet entry
	 * 
	 * @details
	 * Possible flags include  #RXR_PKT_ENTRY_IN_USE #RXR_PKT_ENTRY_RNR_RETRANSMIT,
	 * #RXR_PKT_ENTRY_LOCAL_READ, and #RXR_PKT_ENTRY_DC_LONGCTS_DATA
	 */
	uint32_t flags;

	/**
	 * @brief link multiple MEDIUM RTM with same message ID together
	 *
	 * @details
	 * used on receiver side only
	 */
	struct rxr_pkt_entry *next;

	/** @brief information of send buffer
	 *  @todo make this field a union with `next` to reduce memory usage
	 */
	struct rxr_pkt_sendv *send;

	/** @brief Work request struct used by rdma-core.
	 *  @todo move this field out of rxr_pkt_entry, which requires re-implement the buld send.
	 */
	struct efa_send_wr *send_wr;

	/**
	 * @brief Work request struct used by rdma-core for receive.
	 * @todo move this field out of rxr_pkt_entry to a separate pool.
	 */
	struct efa_recv_wr recv_wr;

	/** @brief buffer that contains data that is going over wire */
	char wiredata[0];
};

#if defined(static_assert) && defined(__x86_64__)
#if ENABLE_DEBUG
static_assert(sizeof(struct rxr_pkt_entry) == 192, "rxr_pkt_entry check");
#else
static_assert(sizeof(struct rxr_pkt_entry) == 128, "rxr_pkt_entry check");
#endif
#endif

static inline void *rxr_pkt_start(struct rxr_pkt_entry *pkt_entry)
{
	return pkt_entry->wiredata;
}

struct rxr_ep;

struct rxr_op_entry;

struct rxr_pkt_pool;

struct rxr_pkt_entry *rxr_pkt_entry_init_prefix(struct rxr_ep *ep,
						const struct fi_msg *posted_buf,
						struct ofi_bufpool *pkt_pool);

struct rxr_pkt_entry *rxr_pkt_entry_alloc(struct rxr_ep *ep,
					  struct rxr_pkt_pool *pkt_pool,
					  enum rxr_pkt_entry_alloc_type alloc_type);

void rxr_pkt_entry_release_tx(struct rxr_ep *ep,
			      struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_entry_release_rx(struct rxr_ep *ep,
			      struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_entry_append(struct rxr_pkt_entry *dst,
			  struct rxr_pkt_entry *src);

struct rxr_pkt_entry *rxr_pkt_entry_clone(struct rxr_ep *ep,
					  struct rxr_pkt_pool *pkt_pool,
					  enum rxr_pkt_entry_alloc_type alloc_type,
					  struct rxr_pkt_entry *src);

struct rxr_pkt_entry *rxr_pkt_get_unexp(struct rxr_ep *ep,
					struct rxr_pkt_entry **pkt_entry_ptr);

ssize_t rxr_pkt_entry_send(struct rxr_ep *ep,
			   struct rxr_pkt_entry *pkt_entry, uint64_t flags);

int rxr_pkt_entry_read(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry,
		       void *local_buf, size_t len, void *desc,
		       uint64_t remote_buf, size_t remote_key);

ssize_t rxr_pkt_entry_recv(struct rxr_ep *ep,
			   struct rxr_pkt_entry *pkt_entry, void **desc,
			   uint64_t flags);

ssize_t rxr_pkt_entry_inject(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry,
			     fi_addr_t addr);

int rxr_pkt_entry_write(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry,
		       void *local_buf, size_t len, void *desc,
		       uint64_t remote_buf, size_t remote_key);

struct rxr_pkt_rx_key {
	uint64_t msg_id;
	fi_addr_t addr;
};

struct rxr_op_entry;

struct rxr_pkt_rx_map {
	struct rxr_pkt_rx_key key;
	struct rxr_op_entry *rx_entry;
	UT_hash_handle hh;
};

struct rxr_op_entry *rxr_pkt_rx_map_lookup(struct rxr_ep *ep,
					   struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_rx_map_insert(struct rxr_ep *ep,
			   struct rxr_pkt_entry *pkt_entry,
			   struct rxr_op_entry *rx_entry);

void rxr_pkt_rx_map_remove(struct rxr_ep *pkt_rx_map,
			   struct rxr_pkt_entry *pkt_entry,
			   struct rxr_op_entry *rx_entry);

static inline bool rxr_pkt_entry_has_hmem_mr(struct rxr_pkt_sendv *send)
{
	/* the device only support send up 2 iov, so iov_count cannot be > 2 */
	assert(send->iov_count == 1 || send->iov_count == 2);
	/* first iov is always on host memory, because it must contain packet header */
	assert(!efa_mr_is_hmem(send->desc[0]));
	return (send->iov_count == 2) && efa_mr_is_hmem(send->desc[1]);
}

#endif
