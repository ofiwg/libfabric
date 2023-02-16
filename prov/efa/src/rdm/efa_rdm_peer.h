/*
 * Copyright (c) 2019-2023 Amazon.com, Inc. or its affiliates.
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
#ifndef EFA_RDM_PEER_H
#define EFA_RDM_PEER_H

#include "rxr.h"


#define EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE	(16384)

OFI_DECL_RECVWIN_BUF(struct rxr_pkt_entry*, efa_rdm_robuf, uint32_t);

#define EFA_RDM_PEER_REQ_SENT BIT_ULL(0) /**< A REQ packet has been sent to the peer (peer should send a handshake back) */
#define EFA_RDM_PEER_HANDSHAKE_SENT BIT_ULL(1) /**< a handshake packet has been sent to the peer */
#define EFA_RDM_PEER_HANDSHAKE_RECEIVED BIT_ULL(2) /**< a handshaked packet has been received from this peer */
#define EFA_RDM_PEER_IN_BACKOFF BIT_ULL(3) /**< peer is in backoff mode due to RNR (Endpoint should not send packet to this peer) */
/**
 * @details
 * FI_EAGAIN error was encountered when sending handsahke to this peer,
 * the peer was put in rxr_ep->handshake_queued_peer_list.
 * Progress engine will retry sending handshake.
 */
#define EFA_RDM_PEER_HANDSHAKE_QUEUED      BIT_ULL(5)

struct efa_rdm_peer {
	bool is_self;			/**< flag indicating whether the peer is the endpoint itself */
	bool is_local;			/**< flag indicating wehther the peer is local (on the same instance) */
	fi_addr_t efa_fiaddr;		/**< libfabric addr from efa provider's perspective */
	fi_addr_t shm_fiaddr;		/**< libfabric addr from shm provider's perspective */
	/**
	 * @brief reorder buffer
	 * 
	 * @details temporarily hold packets that are out-of-order, whose msg_id is larger that the one EP is expecting from the peer
	 */
	struct efa_rdm_robuf robuf;
	uint32_t next_msg_id;		/**< msg_id to be assigned to the next packet sent to the peer. */
	uint32_t flags;			/**< flags such as #EFA_RDM_PEER_REQ_SENT #EFA_RDM_PEER_HANDSHAKE_SENT #EFA_RDM_PEER_HANDSHAKE_RECEIVED and #EFA_RDM_PEER_IN_BACKOFF */
	uint32_t nextra_p3;		/**< number of members in extra_info plus 3 (See protocol v4 document section 2.1) */
	uint64_t extra_info[RXR_MAX_NUM_EXINFO]; /**< the feature/request flag for each version (See protocol v4 document section 2.1)*/
	size_t efa_outstanding_tx_ops;	/**< tracks outstanding tx ops (send/read) to this peer on EFA device */
	size_t shm_outstanding_tx_ops;  /**< tracks outstanding tx ops (send/read/write/atomic) to this peer on SHM */
	struct dlist_entry outstanding_tx_pkts; /**< a list of outstanding pkts sent to the peer */
	uint64_t rnr_backoff_begin_ts;	/**< timestamp for RNR backoff period begin */
	uint64_t rnr_backoff_wait_time;	/**< how long the RNR backoff period last */
	int rnr_queued_pkt_cnt;		/**< queued RNR packet count */
	struct dlist_entry rnr_backoff_entry;	/**< linked to rxr_ep peer_backoff_list */
	struct dlist_entry handshake_queued_entry; /**< linked with rxr_ep->handshake_queued_peer_list */
	struct dlist_entry rx_unexp_list; /**< a list of unexpected untagged rx_entry for this peer */
	struct dlist_entry rx_unexp_tagged_list; /**< a list of unexpected tagged rx_entry for this peer */
	struct dlist_entry tx_entry_list; /**< a list of tx_entry related to this peer */
	struct dlist_entry rx_entry_list; /**< a list of rx_entry relased to this peer */

	/**
	 * @brief number of bytes that has been sent as part of runting protocols
	 * @details this value is capped by rxr_env.efa_runt_size
	 */
	int64_t num_runt_bytes_in_flight;

	/**
	 * @brief number of messages that are using read based protocol
	 */
	int64_t num_read_msg_in_flight;
};

/**
 * @brief check for peer's RDMA_READ support, assuming HANDSHAKE has already occurred
 *
 * @param[in] peer	A peer which we have already received a HANDSHAKE from
 * @return bool		The peer's RDMA_READ support
 */
static inline
bool efa_rdm_peer_support_rdma_read(struct efa_rdm_peer *peer)
{
	/* RDMA READ is an extra feature defined in version 4 (the base version).
	 * Because it is an extra feature, an EP will assume the peer does not support
	 * it before a handshake packet was received.
	 */
	return (peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED) &&
	       (peer->extra_info[0] & RXR_EXTRA_FEATURE_RDMA_READ);
}

/**
 * @brief check for peer's RDMA_WRITE support, assuming HANDSHAKE has already occurred
 *
 * @param[in] peer	A peer which we have already received a HANDSHAKE from
 * @return bool		The peer's RDMA_WRITE support
 */
static inline
bool efa_rdm_peer_support_rdma_write(struct efa_rdm_peer *peer)
{
	/* RDMA WRITE is an extra feature defined in version 4 (the base version).
	 * Because it is an extra feature, an EP will assume the peer does not support
	 * it before a handshake packet was received.
	 */
	return (peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED) &&
	       (peer->extra_info[0] & RXR_EXTRA_FEATURE_RDMA_WRITE);
}

static inline
bool efa_rdm_peer_support_delivery_complete(struct efa_rdm_peer *peer)
{
	/* FI_DELIVERY_COMPLETE is an extra feature defined
	 * in version 4 (the base version).
	 * Because it is an extra feature,
	 * an EP will assume the peer does not support
	 * it before a handshake packet was received.
	 */
	return (peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED) &&
	       (peer->extra_info[0] & RXR_EXTRA_FEATURE_DELIVERY_COMPLETE);
}

/**
 * @brief determine if both peers support RDMA read
 *
 * This function can only return true if a handshake packet has already been
 * exchanged, and the peer set the RXR_EXTRA_FEATURE_RDMA_READ flag.
 * @params[in]		ep		Endpoint for communication with peer
 * @params[in]		peer		An EFA peer
 * @return		boolean		both self and peer support RDMA read
 */
static inline
bool efa_both_support_rdma_read(struct rxr_ep *ep, struct efa_rdm_peer *peer)
{
	return efa_domain_support_rdma_read(rxr_ep_domain(ep)) &&
	       (peer->is_self || efa_rdm_peer_support_rdma_read(peer));
}

/**
 * @brief determine if both peers support RDMA write
 *
 * This function can only return true if a handshake packet has already been
 * exchanged, and the peer set the RXR_EXTRA_FEATURE_RDMA_WRITE flag.
 * @params[in]		ep		Endpoint for communication with peer
 * @params[in]		peer		An EFA peer
 * @return		boolean		both self and peer support RDMA write
 */
static inline
bool efa_both_support_rdma_write(struct rxr_ep *ep, struct efa_rdm_peer *peer)
{
	return efa_domain_support_rdma_write(rxr_ep_domain(ep)) &&
	       (peer->is_self || efa_rdm_peer_support_rdma_write(peer));
}

/**
 * @brief determines whether a peer needs the endpoint to include
 * raw address int the req packet header.
 *
 * There are two cases a peer need the raw address in REQ packet header:
 *
 * 1. the initial packets to a peer should include the raw address,
 * because the peer might not have ep's address in its address vector
 * causing the peer to be unable to send packet back. Normally, after
 * an endpoint received a hanshake packet from a peer, it can stop
 * including raw address in packet header.
 *
 * 2. If the peer requested to keep the header length constant through
 * out the communiciton, endpoint will include the raw address in the
 * header even afer received handshake from a header to conform to the
 * request. Usually, peer has this request because they are in zero
 * copy receive mode, which requires the packet header size to remain
 * the same.
 *
 * @params[in]	peer	pointer to rdm_peer
 * @return	a boolean indicating whether the peer needs the raw address header
 */
static inline
bool efa_rdm_peer_need_raw_addr_hdr(struct efa_rdm_peer *peer)
{
	if (OFI_UNLIKELY(!(peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED)))
		return true;

	return peer->extra_info[0] & RXR_EXTRA_REQUEST_CONSTANT_HEADER_LENGTH;
}

/**
 * @brief determines whether a peer needs the endpoint to include
 * connection ID (connid) in packet header.
 *
 * Connection ID is a 4 bytes random integer identifies an endpoint.
 * Including connection ID in a packet's header allows peer to
 * identify sender of the packet. It is necessary because device
 * only report GID+QPN of a received packet, while QPN may be reused
 * accross device endpoint teardown and initialization.
 *
 * EFA uses qkey as connection ID.
 *
 * @params[in]	peer	pointer to rdm_peer
 * @return	a boolean indicating whether the peer needs connection ID
 */
static inline
bool efa_rdm_peer_need_connid(struct efa_rdm_peer *peer)
{
	return (peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED) &&
	       (peer->extra_info[0] & RXR_EXTRA_REQUEST_CONNID_HEADER);
}

struct efa_conn;

void efa_rdm_peer_construct(struct efa_rdm_peer *peer, struct rxr_ep *ep, struct efa_conn *conn);

void efa_rdm_peer_destruct(struct efa_rdm_peer *peer, struct rxr_ep *ep);

int efa_rdm_peer_reorder_msg(struct efa_rdm_peer *peer, struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry);

void efa_rdm_peer_proc_pending_items_in_robuf(struct efa_rdm_peer *peer, struct rxr_ep *ep);

#endif /* EFA_RDM_PEER_H */