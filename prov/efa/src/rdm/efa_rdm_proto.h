/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#ifndef _EFA_RDM_PROTO_H
#define _EFA_RDM_PROTO_H

#include "efa.h"
#include "efa_rdm_pkt_type.h"

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
	/* TX path handlers */

	/* This function determines whether the protocol can be used for a given
	 * send operation.
	 */
	bool (*can_use_protocol_for_send)(struct efa_rdm_ope *txe,
					  int req_pkt_type,
					  uint16_t header_flags, int iface);

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
 * @param[in]  ep     Endpoint
 * @param[in]  peer   Peer to send to
 * @param[in]  msg    Message descriptor from application
 * @param[in]  op     Operation type (ofi_op_msg or ofi_op_tagged)
 * @param[in]  flags  Operation flags (FI_INJECT, FI_DELIVERY_COMPLETE, etc.)
 * @param[out] txe    Pre-allocated TXE, partially initialized on return
 * @param[out] proto  Selected protocol, or NULL if none matched
 * @return 0 on success, negative errno on failure
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

#endif /* _EFA_RDM_PROTO_H */
