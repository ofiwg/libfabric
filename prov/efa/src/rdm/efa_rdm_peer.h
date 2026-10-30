/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#ifndef EFA_RDM_PEER_H
#define EFA_RDM_PEER_H

#include "ofi_recvwin.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_protocol.h"
#include "efa_rdm_rxe_map.h"

#define EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE	(16)

#define EFA_RDM_PEER_REQ_SENT BIT_ULL(0) /**< A REQ packet has been sent to the peer (peer should send a handshake back) */
#define EFA_RDM_PEER_HANDSHAKE_SENT BIT_ULL(1) /**< a handshake packet has been sent to the peer */
#define EFA_RDM_PEER_HANDSHAKE_RECEIVED BIT_ULL(2) /**< a handshaked packet has been received from this peer */
#define EFA_RDM_PEER_IN_BACKOFF BIT_ULL(3) /**< peer is in backoff mode due to RNR (Endpoint should not send packet to this peer) */
/**
 * @details
 * FI_EAGAIN error was encountered when sending handshake to this peer,
 * the peer was put in efa_rdm_ep->handshake_queued_peer_list.
 * Progress engine will retry sending handshake.
 */
#define EFA_RDM_PEER_HANDSHAKE_QUEUED      BIT_ULL(5)
#define EFA_RDM_PEER_UNRESP                BIT_ULL(6) /**< peer is unresponsive and we should not wait for completions */


OFI_DECL_RECVWIN_BUF(struct efa_rdm_pke*, efa_rdm_robuf, uint32_t);

/*
 * recvwindow macro covers only the initilization and
 * the alloc/free are required to be implemented by the provider
 */

/**
 * @details
 *
 * @param[in] recvq			circular buffer object to be created
 * @param[in] size			size of the queue
 * @param[in] alloc_from_bufpool 	true means allocate from the buffer pool provided, false means do calloc
 * @param[in] pool			buffer pool to be used for allocating the queue from when alloc_from_bufpool is true
 * @return 0 on success or error otherwise
 *
 */
static inline
int efa_recvwin_buf_alloc(struct efa_rdm_robuf *recvq,
                unsigned int size,  bool alloc_from_bufpool, struct ofi_bufpool *pool)
{
	assert(size == roundup_power_of_two(size));
	recvq->exp_msg_id = 0;
	recvq->win_size = size;
	if (alloc_from_bufpool) {
		assert(pool != NULL);
		assert(size <= pool->entry_size);
		recvq->pending = (struct recvwin_cirq*) ofi_buf_alloc(pool);
	} else {
		recvq->pending = (struct recvwin_cirq*) calloc(1, sizeof(struct recvwin_cirq) +
					sizeof(struct efa_rdm_pke*) * size);
	}
	if (recvq->pending) {
		recvwin_cirq_init(recvq->pending, size);
		return 0;
	} else {
	       return -FI_ENOMEM;
	}
}

static inline
void efa_recvwin_free(struct efa_rdm_robuf *recvq, bool free_to_bufpool)
{
	if (free_to_bufpool)
		ofi_buf_free(recvq->pending);
	else
		free(recvq->pending);
}

struct efa_rdm_peer_user_recv_qp
{
	uint32_t qpn;
	uint32_t qkey;
};

struct efa_rdm_peer_overflow_pke_list_entry {
	struct dlist_entry entry;
	struct efa_rdm_pke *pkt_entry;
};

struct efa_rdm_peer {
	struct efa_rdm_ep *ep;		/**< local ep */
	bool is_self;			/**< flag indicating whether the peer is the endpoint itself */
	bool is_local;			/**< flag indicating whether the peer is local (on the same instance) */
	/**
	 * @brief Whether this peer's buffers can be accessed by the NIC directly.
	 *
	 * p2p means the NIC can directly access the peer's buffers for all
	 * operations posted to the NIC (RDMA read/write sources and targets).
	 *
	 * Received from the peer's handshake HMEM_P2P_HDR.
	 * true = peer has no accelerator memory, or accelerator memory with p2p support.
	 * false = peer has accelerator memory (FI_HMEM) without p2p support.
	 */
	bool p2p_supported;
	uint32_t device_version;	/**< EFA device version */
	/**
	 * @brief Address handle, QP number and QKey of the peer
	 *
	 * @details Copied from the AV entry when the peer is constructed and not
	 * written afterwards, so they are read without a lock. They are the same
	 * in the implicit and explicit AV entries, so they stay correct when the
	 * peer moves to the explicit AV and the implicit entry is freed. Every AV
	 * entry release path destroys the entry's peers before it releases the
	 * address handle, so ah outlives the peer.
	 */
	struct efa_ah *ah;
	uint16_t qpn;
	uint32_t qkey;
	/**
	 * @brief reorder buffer
	 *
	 * @details temporarily hold packets that are out-of-order, whose msg_id is larger that the one EP is expecting from the peer
	 */
	struct efa_rdm_robuf robuf;
	uint32_t next_msg_id;		/**< msg_id to be assigned to the next packet sent to the peer. */
	uint32_t flags;			/**< flags such as #EFA_RDM_PEER_REQ_SENT #EFA_RDM_PEER_HANDSHAKE_SENT #EFA_RDM_PEER_HANDSHAKE_RECEIVED and #EFA_RDM_PEER_IN_BACKOFF */
	uint32_t nextra_p3;		/**< number of members in extra_info plus 3 (See protocol v4 document section 2.1) */
	uint64_t extra_info[EFA_RDM_MAX_NUM_EXINFO]; /**< the feature/request flag for each version (See protocol v4 document section 2.1)*/
	size_t efa_outstanding_tx_ops;	/**< tracks outstanding tx ops (send/read) to this peer on EFA device */
	struct dlist_entry outstanding_tx_pkts; /**< a list of outstanding pkts sent to the peer */
	uint64_t rnr_backoff_begin_ts;	/**< timestamp for RNR backoff period begin */
	uint64_t rnr_backoff_wait_time;	/**< how long the RNR backoff period last */
	int rnr_queued_pkt_cnt;		/**< queued RNR packet count */
	struct dlist_entry rnr_backoff_entry;	/**< linked to efa_rdm_ep->peer_backoff_list */
	struct dlist_entry handshake_queued_entry; /**< linked with efa_rdm_ep->handshake_queued_peer_list */
	struct dlist_entry txe_list; /**< a list of txe related to this peer */
	struct dlist_entry rxe_list; /**< a list of rxe related to this peer */
	struct dlist_entry overflow_pke_list; /**< a list of out-of-order pke that overflow the current recvwin */
	/**
	 * @brief number of bytes that has been sent as part of runting protocols
	 * @details this value is capped by efa_env.efa_runt_size
	 */
	int64_t num_runt_bytes_in_flight;
	/**
	 * only valid when (extra_info[0] & EFA_RDM_EXTRA_FEATURE_REQUEST_USER_RECV_QP) is non-zero
	 */
	struct efa_rdm_peer_user_recv_qp user_recv_qp;

	struct efa_rdm_rxe_map rxe_map; 	/**< Hashmap used to match received mulreq packets with RX entries */
	/*
	 * The fields below are not read on the per-packet send and receive
	 * paths (fi_addr is read once per completion, as it was from the AV
	 * entry before). They are kept last so that ah, qpn and qkey fit in
	 * front of robuf and every per-packet field stays in the first two
	 * cache lines.
	 */
	uint64_t host_id; 		/* Optional peer host id. Default 0 */
	/**
	 * @brief Explicit and implicit fi_addr of the peer
	 *
	 * @details Same meaning as in efa_rdm_av_entry: a peer in the explicit AV
	 * has fi_addr set and implicit_fi_addr FI_ADDR_NOTAVAIL, and a peer only
	 * in the implicit AV the reverse. Moving the peer to the explicit AV
	 * changes them while lock-free readers can run, so they are only accessed
	 * through efa_rdm_peer_av_index(), efa_rdm_peer_implicit_fi_addr(),
	 * efa_rdm_peer_is_explicit() and efa_rdm_peer_move_to_explicit().
	 */
	fi_addr_t fi_addr;
	fi_addr_t implicit_fi_addr;
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
	       (peer->extra_info[0] & EFA_RDM_EXTRA_FEATURE_RDMA_READ);
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
	       (peer->extra_info[0] & EFA_RDM_EXTRA_FEATURE_RDMA_WRITE);
}

/**
 * @brief check for peer's unsolicited write support, assuming HANDSHAKE has already occurred
 *
 * @param[in] peer	A peer which we have already received a HANDSHAKE from
 * @return bool		The peer's unsolicited write recv support
 */
static inline
bool efa_rdm_peer_support_unsolicited_write_recv(struct efa_rdm_peer *peer)
{
	/* Unsolicited write recv is an extra feature defined in version 4 (the base version).
	 * Because it is an extra feature, an EP will assume the peer does not support
	 * it before a handshake packet was received.
	 */
	return (peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED) &&
	       (peer->extra_info[0] & EFA_RDM_EXTRA_FEATURE_UNSOLICITED_WRITE_RECV);
}

/**
 * @brief whether the peer advertises EFA_RDM_PEER_ERROR_PKT support
 *
 * Negotiated via the handshake. A peer without this bit cannot decode a
 * PEER_ERROR_PKT, so we must not emit one to it (the caller then falls
 * back to the legacy "leak the txe / write a local CQ error" behavior).
 *
 * @param[in] peer	a peer from whom we have received a HANDSHAKE
 */
static inline
bool efa_rdm_peer_support_peer_error(struct efa_rdm_peer *peer)
{
	return (peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED) &&
	       (peer->extra_info[0] & EFA_RDM_EXTRA_FEATURE_PEER_ERROR);
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
 * an endpoint received a handshake packet from a peer, it can stop
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

	return peer->extra_info[0] & EFA_RDM_EXTRA_REQUEST_CONSTANT_HEADER_LENGTH;
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
 * Since libfabric 2.6, connid header is treated as a baseline request.
 * All supported peers (v2.0+) request it, so we always include connid
 * after handshake without checking the peer's extra_info.
 *
 * @params[in]	peer	pointer to rdm_peer
 * @return	a boolean indicating whether the peer needs connection ID
 */
static inline
bool efa_rdm_peer_need_connid(struct efa_rdm_peer *peer)
{
	return (peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED);
}

struct efa_rdm_av_entry;

int efa_rdm_peer_construct(struct efa_rdm_peer *peer, struct efa_rdm_ep *ep, struct efa_rdm_av_entry *av_entry);

void efa_rdm_peer_destruct(struct efa_rdm_peer *peer, struct efa_rdm_ep *ep);

int efa_rdm_peer_reorder_msg(struct efa_rdm_peer *peer, struct efa_rdm_ep *ep, struct efa_rdm_pke *pkt_entry);

int efa_rdm_peer_recvwin_queue_or_append_pke(struct efa_rdm_pke *pkt_entry, uint32_t msg_id, struct efa_rdm_robuf *robuf);

void efa_rdm_peer_move_overflow_pke_to_recvwin(struct efa_rdm_peer *peer);

void efa_rdm_peer_proc_pending_items_in_robuf(struct efa_rdm_peer *peer, struct efa_rdm_ep *ep);

bool efa_rdm_peer_abort_ooo_msg(struct efa_rdm_peer *peer, uint32_t msg_id);

int efa_rdm_peer_queue_aborted_msg_marker(struct efa_rdm_peer *peer, struct efa_rdm_ep *ep, struct efa_rdm_pke *pkt_entry, uint32_t msg_id);

size_t efa_rdm_peer_get_runt_size(struct efa_rdm_peer *peer, struct efa_rdm_ep *ep, struct efa_rdm_ope *ope);

int efa_rdm_peer_select_readbase_rtm(struct efa_rdm_peer *peer, struct efa_rdm_ep *ep, struct efa_rdm_ope *ope);

/**
 * @brief Return the util_av index of a peer in the explicit AV, or
 * FI_ADDR_NOTAVAIL if the peer is only in the implicit AV. Safe without the
 * AV locks.
 */
static inline fi_addr_t efa_rdm_peer_av_index(const struct efa_rdm_peer *peer)
{
	return __atomic_load_n(&peer->fi_addr, __ATOMIC_ACQUIRE);
}

/**
 * @brief Return the implicit fi_addr of a peer, or FI_ADDR_NOTAVAIL if the
 * peer is in the explicit AV. Safe without the AV locks.
 */
static inline fi_addr_t efa_rdm_peer_implicit_fi_addr(const struct efa_rdm_peer *peer)
{
	return __atomic_load_n(&peer->implicit_fi_addr, __ATOMIC_ACQUIRE);
}

/**
 * @brief Return whether the peer is in the explicit AV. Safe without the AV
 * locks; once it returns true it keeps returning true for the peer's lifetime.
 */
static inline bool efa_rdm_peer_is_explicit(const struct efa_rdm_peer *peer)
{
	return efa_rdm_peer_av_index(peer) != FI_ADDR_NOTAVAIL;
}

/**
 * @brief Record that an implicit peer moved to the explicit AV at fi_addr
 *
 * @details Lock-free readers can run concurrently. fi_addr is published
 * before implicit_fi_addr is cleared, so a reader can briefly see both set,
 * but never neither, and efa_rdm_peer_is_explicit() never goes back to false.
 *
 * @param[in,out]	peer	rdm peer, currently only in the implicit AV
 * @param[in]		fi_addr	explicit fi_addr of the peer
 */
static inline void efa_rdm_peer_move_to_explicit(struct efa_rdm_peer *peer,
						 fi_addr_t fi_addr)
{
	assert(fi_addr != FI_ADDR_NOTAVAIL);
	assert(!efa_rdm_peer_is_explicit(peer));
	__atomic_store_n(&peer->fi_addr, fi_addr, __ATOMIC_RELEASE);
	__atomic_store_n(&peer->implicit_fi_addr, FI_ADDR_NOTAVAIL, __ATOMIC_RELEASE);
}

/**
 * @brief the fi_addr the application sees for a peer in the explicit AV
 *
 * efa_rdm_peer_av_index() is the util_av index, which keys the peer maps and
 * util_srx. With FI_AV_MAP the application fi_addr is the AV entry's address
 * instead. FI_ADDR_NOTAVAIL for a peer only in the implicit AV. A macro
 * because struct efa_rdm_ep is not complete here.
 */
#define efa_rdm_peer_fi_addr(peer)                                             \
	efa_av_index_to_fi_addr((peer)->ep->base_ep.av,                        \
				efa_rdm_peer_av_index(peer))

/* Macro for getting peer address string */
#define EFA_RDM_GET_PEER_ADDR_STR(ep, peer, peer_addr_str) \
	char peer_addr_str[OFI_ADDRSTRLEN] = {0}; \
	efa_base_ep_get_peer_raw_addr_str(&ep->base_ep, efa_rdm_peer_fi_addr(peer), peer_addr_str, &(size_t){sizeof peer_addr_str});

#endif /* EFA_RDM_PEER_H */
