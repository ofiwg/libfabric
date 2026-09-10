/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto.h"
#include "efa.h"
#include "efa_rdm_domain.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_proto_eager.h"
#include "efa_rdm_msg.h"

/**
 * @brief Undo the memory registrations the selection loop made.
 *
 * efa_rdm_ope_try_fill_desc() registers the source buffer so a read based
 * protocol can be evaluated. When no protocol is selected the caller falls back
 * to the legacy send path, whose efa_rdm_txe_construct() clears txe->mr without
 * closing it, so nothing would ever release those registrations. Hand them back
 * here instead.
 *
 * TODO: Remove after all protocols are migrated to the new code path
 */
static void efa_rdm_proto_release_selection_mrs(struct efa_rdm_ope *txe)
{
	int i, err;

	for (i = 0; i < txe->iov_count; ++i) {
		if (!txe->mr[i])
			continue;

		err = fi_close((struct fid *) txe->mr[i]);
		if (OFI_UNLIKELY(err))
			EFA_WARN(FI_LOG_EP_DATA,
				 "mr dereg failed during protocol selection. err=%d\n",
				 err);

		txe->mr[i] = NULL;
		txe->desc[i] = NULL;
	}
}

/* List of supported protocols.
 * The protocols listed here will be tried in the order they're listed.
 * The first protocol that can be used for the TX operation will be used.
 *
 * Only efa_rdm_proto_select_send_protocol() below walks this array, so it is
 * internal to this file and its length is whatever the initializer holds.
 */
static struct efa_rdm_proto * const efa_rdm_protocols[] = {
	&efa_rdm_proto_eager,
};

void efa_rdm_proto_txe_init_buffers(struct efa_rdm_ep *ep,
						  const struct fi_msg *msg,
						  struct efa_rdm_ope *txe)
{
	txe->ep = ep;
	txe->iov_count = msg->iov_count;
	memcpy(txe->iov, msg->msg_iov, sizeof(struct iovec) * msg->iov_count);
	memset(txe->mr, 0, sizeof(*txe->mr) * msg->iov_count);

	efa_rdm_mr_gen_init_ope_desc(txe);
	if (msg->desc) {
		memcpy(txe->desc, msg->desc, sizeof(*msg->desc) * msg->iov_count);
		efa_rdm_mr_gen_capture_in_ope_desc(txe);
	} else {
		memset(txe->desc, 0, sizeof(*txe->desc) * msg->iov_count);
	}
	txe->total_len = ofi_total_iov_len(msg->msg_iov, msg->iov_count);
}

int efa_rdm_proto_select_send_protocol(struct efa_rdm_ep *ep,
				       struct efa_rdm_peer *peer,
				       const struct fi_msg *msg, uint32_t op,
				       uint64_t flags, struct efa_rdm_ope *txe,
				       struct efa_rdm_proto **proto)
{
	struct efa_rdm_proto *selected_proto;
	int req_pkt_type, iface, err;
	bool use_p2p, mr_attempted = false;
	uint16_t header_flags = 0;
	uint64_t effective_flags;

	/*
	 * efa_rdm_msg_generic_send() sends a peer that only accepts headerless
	 * packets straight to the zero-copy protocol, so every protocol
	 * considered here writes a REQ header.
	 */
	assert(!efa_rdm_peer_expects_zero_hdr_data_transfer(peer));

	/*
	 * Resolve the endpoint's tx_op_flags now: an endpoint-level
	 * FI_DELIVERY_COMPLETE must steer protocol selection just as a
	 * per-operation flag does.
	 */
	effective_flags = efa_rdm_msg_get_tx_flags(ep, flags);

	efa_rdm_proto_txe_init_buffers(ep, msg, txe);

	iface = (msg->desc && msg->desc[0]) ?
			((struct efa_mr *) msg->desc[0])->iface :
			FI_HMEM_SYSTEM;

	/* Synapse AI only supports long read */
	if (iface == FI_HMEM_SYNAPSEAI) {
		*proto = NULL;
		txe->proto = NULL;
		return 0;
	}

	/*
	 * The read based protocols can only be used when the device can access
	 * the source buffer directly, so resolve p2p availability once here
	 * instead of in each predicate. A negative return means the transfer
	 * cannot be performed at all.
	 */
	err = efa_rdm_ep_use_p2p_for_mr(ep, txe->desc[0]);
	if (err < 0)
		return err;
	use_p2p = err;

	/* Logic copied from efa_rdm_txe_max_req_data_capacity */
	if (efa_rdm_peer_need_raw_addr_hdr(peer))
		header_flags |= EFA_RDM_REQ_OPT_RAW_ADDR_HDR;
	else if (efa_rdm_peer_need_connid(peer))
		header_flags |= EFA_RDM_PKT_CONNID_HDR;

	if (flags & FI_REMOTE_CQ_DATA)
		header_flags |= EFA_RDM_REQ_OPT_CQ_DATA_HDR;

	for (int i = 0; i < ARRAY_SIZE(efa_rdm_protocols); ++i) {
		selected_proto = efa_rdm_protocols[i];

		req_pkt_type = efa_rdm_proto_req_pkt_type(
			selected_proto, op, effective_flags, peer);

		/* All protocols other than the eager protocol can benefit from
		 * registering the application buffers: the read based protocols
		 * cannot be used at all without a registered source buffer, and
		 * the others avoid a bounce copy.
		 *
		 * Ask for FI_REMOTE_READ too whenever p2p is available, because
		 * read based protocols require it.
		 *
		 * TODO: Move efa_rdm_ope_try_fill_desc to efa_rdm_proto.c
		 */
		if (!mr_attempted && selected_proto != &efa_rdm_proto_eager) {
			uint64_t access =
				FI_SEND | (use_p2p ? FI_REMOTE_READ : 0);

			if (efa_is_cache_available(efa_rdm_ep_rdm_domain(ep)))
				efa_rdm_ope_try_fill_desc(txe, 0, access);
			mr_attempted = true;
		}

		if (selected_proto->can_use_protocol_for_send(
			    txe, req_pkt_type, header_flags, iface)) {
			*proto = selected_proto;
			txe->proto = selected_proto;
			txe->req_pkt_type = req_pkt_type;
			EFA_DBG(FI_LOG_EP_DATA,
				"Selected the %s protocol for a %zu byte send\n",
				selected_proto->name, txe->total_len);
			return FI_SUCCESS;
		}
	}

	/*
	 * No protocol matched, so release any MRs that were registered
	 * TODO: Remove after all protocols moved to new code path
	 */
	if (mr_attempted)
		efa_rdm_proto_release_selection_mrs(txe);

	*proto = NULL;
	txe->proto = NULL;
	return FI_SUCCESS;
}

/* Utility funcions */

void efa_rdm_proto_txe_fill(struct efa_rdm_ope *txe, struct efa_rdm_ep *ep,
			    struct efa_rdm_peer *peer, const struct fi_msg *msg,
			    uint32_t op, uint64_t tag, uint64_t flags,
			    uint32_t internal_flags, struct efa_rdm_proto *proto)
{
	/*
	 * txe->mr, txe->desc and the MR generation snapshot were already
	 * populated by efa_rdm_proto_select_send_protocol(), which needs them
	 * to decide whether a protocol can be used, so use the construct
	 * helper that leaves them alone.
	 */
	efa_rdm_txe_construct_common(txe, ep, peer, msg, op, flags,
				     internal_flags);

	if (op == ofi_op_tagged) {
		txe->cq_entry.tag = tag;
		txe->tag = tag;
	}
}
