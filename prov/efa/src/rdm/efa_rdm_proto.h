/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#ifndef _EFA_RDM_PROTO_H
#define _EFA_RDM_PROTO_H

#include "efa.h"
#include "efa_rdm_pkt_type.h"

/* The EFA provider has five protocols (eager, medium, runt read, long read and
 * long CTS). Size the registry a little larger to hold the NULL sentinel that
 * terminates it plus room for future protocols.
 */
#define EFA_RDM_MAX_PROTO 8

/**
 * @brief Interface for EFA RDM protocols.
 *
 * Each protocol (eager, medium, long CTS, long read, runt read)
 * implements this interface to define how it handles TX and RX operations.
 *
 * The TX send path works as follows:
 *
 * 1. efa_rdm_proto_select_send_protocol() iterates through the protocol
 *    registry (efa_rdm_protocols[]) in priority order, calling
 *    can_use_protocol_for_send() on each. The first match is selected.
 *
 * 2. The selected protocol's construct_tx_pkes() builds the packet
 *    entries and stores them in ep->send_pkt_entry_vec.
 *
 * 3. efa_rdm_msg_generic_send posts all packet entries via efa_rdm_pke_sendv().
 *
 * 4. handle_tx_pkes_posted() is called for post-send bookkeeping.
 *
 * 5. When the device completes a send, the CQ handler invokes the
 *    per-packet callback (set in step 2) which handles completion
 *    logic, CQ reporting, and TXE/PKE release.
 *
 * Protocol priority is determined by position in efa_rdm_protocols[].
 * Protocols are ordered from most restrictive (eager) to least
 * restrictive (long CTS) so the most efficient protocol is always
 * selected.
 */
struct efa_rdm_proto {
	/* Human readable protocol name, e.g. "eager". Used for tracing the
	 * outcome of protocol selection and in unit test assertions.
	 */
	char name[32];

	/* TX path handlers */

	/* This function determines whether the protocol can be used for a given
	 * send operation.
	 *
	 * @param[in] txe		send operation. iov, desc, iov_count and
	 *				total_len are already filled in.
	 * @param[in] peer		peer the operation is addressed to. The read
	 *				based protocols need it to check the peer's
	 *				RDMA read support and, for runt read, the
	 *				peer's in flight runt byte accounting.
	 * @param[in] req_pkt_type	REQ packet type this protocol would use
	 * @param[in] header_flags	optional headers the REQ will carry
	 * @param[in] iface		HMEM interface of the source buffer
	 * @param[in] use_p2p		whether the device can access the source
	 *				buffer directly, as returned by
	 *				efa_rdm_ep_use_p2p_for_mr(). The read
	 *				based protocols require it.
	 */
	bool (*can_use_protocol_for_send)(struct efa_rdm_ope *txe,
					  struct efa_rdm_peer *peer,
					  int req_pkt_type,
					  uint16_t header_flags, int iface,
					  bool use_p2p);

	/* This function will allocate the pkes that need to be sent for a given
	 * TX operation. At the end of this function, ep->send_pkt_entry_vec
	 * will be correctly populated with the all of the pkes that need to be
	 * sent including copying the application data into the pke buffer if
	 * necessary. Each pke will have an appropriate callback function set to
	 * handle the TX completion of that pke. This function also constructs
	 * and returns the txe
	 */
	int (*construct_tx_pkes)(struct efa_rdm_ep *ep,
				 struct efa_rdm_peer *peer,
				 const struct fi_msg *msg, uint32_t op,
				 uint64_t tag, uint64_t flags,
				 uint32_t internal_flags,
				 struct efa_rdm_ope *txe);

	/* This function is called after all pkes are posted to the EFA device.
	 * It is useful for some protocols: e.g. to register the buffer after
	 * posting a Long CTS RTM pke or to update the number of in flight reads
	 * and read bytes
	 */
	void (*handle_tx_pkes_posted)(struct efa_rdm_ep *ep,
				      struct efa_rdm_ope *txe);

	/* TX utitlities */
	int req_pkt_type;
	int req_pkt_type_tagged;
	int req_pkt_type_dc;
	int req_pkt_type_tagged_dc;
};

/* Registry of the supported protocols, in priority order, terminated by a NULL
 * sentinel. Iterate with the sentinel, not with ARRAY_SIZE(): the array is
 * declared here with an explicit size so that ARRAY_SIZE() in a translation
 * unit other than efa_rdm_proto.c would silently return EFA_RDM_MAX_PROTO
 * rather than the number of registered protocols.
 */
extern struct efa_rdm_proto *efa_rdm_protocols[EFA_RDM_MAX_PROTO];

/**
 * @brief Select the appropriate send protocol for a TX operation.
 *
 * Iterates through registered protocols in priority order and selects
 * the first one whose can_use_protocol_for_send() returns true.
 *
 * It will also handle memory registration of user buffers. If read based
 * protocols are appropriate but MR fails, it will automatically switch to a
 * different protocol.
 *
 * Selection never triggers or waits for a handshake. A predicate that needs an
 * extra feature from the peer reports "cannot use" until the handshake has
 * advertised it, so a send issued before the handshake simply lands on a
 * protocol that needs nothing extra. See the comment on the sole remaining
 * enforce-handshake call in efa_rdm_msg_generic_send().
 *
 * @param[in]  ep     Endpoint
 * @param[in]  peer   Peer to send to
 * @param[in]  msg    Message descriptor from application
 * @param[in]  op     Operation type (ofi_op_msg or ofi_op_tagged)
 * @param[in]  flags  Operation flags (FI_INJECT, FI_DELIVERY_COMPLETE, etc.)
 * @param[out] txe    Pre-allocated TXE, partially initialized on return
 * @param[out] proto  Selected protocol. Never NULL on success: the long CTS
 *                    protocol is registered last and can always be used.
 * @return 0 on success, negative errno on failure. -FI_EOPNOTSUPP if no
 *         registered protocol can carry the operation.
 */
int efa_rdm_proto_select_send_protocol(struct efa_rdm_ep *ep,
				       struct efa_rdm_peer *peer,
				       const struct fi_msg *msg, uint32_t op,
				       uint64_t flags, struct efa_rdm_ope *txe,
				       struct efa_rdm_proto **proto);

/* Utility funcions */
static inline void
efa_rdm_proto_handle_tx_pkes_posted_no_op(struct efa_rdm_ep *ep,
					  struct efa_rdm_ope *txe)
{
	return;
};

/**
 * @brief Pick the REQ packet type a protocol uses for an operation.
 *
 * The tagged and delivery-complete variants of a REQ packet type are laid out
 * at fixed offsets from the base type, so a protocol only records the base
 * types and derives the rest.
 *
 * @param[in] proto	protocol that will carry the operation
 * @param[in] op	operation type (ofi_op_msg or ofi_op_tagged)
 * @param[in] flags	effective operation flags, as returned by
 *			efa_rdm_msg_get_tx_flags()
 * @param[in] peer	peer the operation is addressed to
 */
static inline int efa_rdm_proto_req_pkt_type(struct efa_rdm_proto *proto,
					     uint32_t op, uint64_t flags,
					     struct efa_rdm_peer *peer)
{
	bool tagged = (op == ofi_op_tagged);
	bool delivery_complete_requested;

	/*
	 * An injected send completes as soon as the buffer is reusable, and a
	 * headerless send has nowhere to put the DC send_id, so neither can use
	 * the delivery complete variant.
	 */
	if (flags & FI_INJECT ||
	    efa_rdm_peer_expects_zero_hdr_data_transfer(peer))
		delivery_complete_requested = false;
	else
		delivery_complete_requested = flags & FI_DELIVERY_COMPLETE;

	/*
	 * For performance consideration, this code assumes the tagged rtm
	 * packet type id is always the correspondent message rtm packet type
	 * id + 1, thus the assertions here.
	 */
	assert(proto->req_pkt_type_tagged == proto->req_pkt_type + 1);
	assert(proto->req_pkt_type_tagged_dc == proto->req_pkt_type_dc + 1);

	return delivery_complete_requested ? proto->req_pkt_type_dc + tagged :
					     proto->req_pkt_type + tagged;
}

/**
 * @brief Initialize a TXE for use by the new protocol interface.
 *
 * Similar to efa_rdm_txe_construct but does not set mr, desc, or
 * total_len since those are already populated by
 * efa_rdm_proto_select_send_protocol.
 */
void efa_rdm_proto_txe_fill(struct efa_rdm_ope *txe, struct efa_rdm_ep *ep,
			    struct efa_rdm_peer *peer, const struct fi_msg *msg,
			    uint32_t op, uint64_t tag, uint64_t flags,
			    uint32_t internal_flags,
			    struct efa_rdm_proto *proto);

#endif /* _EFA_RDM_PROTO_H */
