/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto.h"
#include "efa.h"
#include "efa_rdm_domain.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_proto_eager.h"
#include "efa_rdm_proto_longread.h"
#include "efa_rdm_proto_medium.h"
#include "efa_rdm_proto_runtread.h"
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
 * Only the slots the selection loop registered are touched: txe->mr[] was
 * zeroed before the loop, so a non-NULL entry is one this code owns, and the
 * matching txe->desc[] entry was NULL before try_fill_desc filled it.
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
 * The NULL sentinel terminates the iteration in
 * efa_rdm_proto_select_send_protocol().
 */
struct efa_rdm_proto *efa_rdm_protocols[EFA_RDM_MAX_PROTO] = {
	&efa_rdm_proto_eager,
	&efa_rdm_proto_medium,
	&efa_rdm_proto_runtread,
	&efa_rdm_proto_longread,
	NULL, /* Sentinel used to stop iteration */
};

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
	 * Resolve the endpoint's tx_op_flags now: an endpoint-level
	 * FI_DELIVERY_COMPLETE must steer protocol selection just as a
	 * per-operation flag does.
	 */
	effective_flags = efa_rdm_msg_get_tx_flags(ep, flags);

	txe->ep = ep;
	/*
	 * The predicates need the effective flags: the runt read protocol has no
	 * delivery complete REQ variant, so it has to rule out a
	 * FI_DELIVERY_COMPLETE send. efa_rdm_txe_construct_common() assigns the
	 * same value again later, from the same helper.
	 */
	txe->fi_flags = effective_flags;
	txe->iov_count = msg->iov_count;
	memcpy(txe->iov, msg->msg_iov, sizeof(struct iovec) * msg->iov_count);
	memset(txe->mr, 0, sizeof(*txe->mr) * msg->iov_count);
	/*
	 * Snapshot each source MR's generation, exactly as
	 * efa_rdm_txe_construct() does. The peer-abort (MR abort) protocol
	 * compares this snapshot against the live MR generation to tell an
	 * application-closed MR from an ordinary error; without it,
	 * efa_rdm_mr_gen_check_ope() reads stale values from the recycled
	 * ope pool slot and can cancel a healthy transfer.
	 */
	efa_rdm_mr_gen_init_ope_desc(txe);
	if (msg->desc) {
		memcpy(txe->desc, msg->desc, sizeof(*msg->desc) * msg->iov_count);
		efa_rdm_mr_gen_capture_in_ope_desc(txe);
	} else {
		memset(txe->desc, 0, sizeof(*txe->desc) * msg->iov_count);
	}
	txe->total_len = ofi_total_iov_len(msg->msg_iov, msg->iov_count);

	iface = (msg->desc && msg->desc[0]) ?
			((struct efa_mr *) msg->desc[0])->iface :
			FI_HMEM_SYSTEM;

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

	for (int i = 0; i < EFA_RDM_MAX_PROTO; ++i) {
		selected_proto = efa_rdm_protocols[i];

		if (!selected_proto)
			break;

		req_pkt_type = efa_rdm_proto_req_pkt_type(
			selected_proto, op, effective_flags, peer);

		/* All protocols other than the eager protocol can benefit from
		 * registering the application buffers: the read based protocols
		 * cannot be used at all without a registered source buffer, and
		 * the others avoid a bounce copy. Eager is the first protocol
		 * tried, so by the time this runs eager has already been ruled
		 * out.
		 * TODO: Move efa_rdm_ope_try_fill_desc to efa_rdm_proto.c
		 */
		if (!mr_attempted && selected_proto != &efa_rdm_proto_eager) {
			if (efa_is_cache_available(efa_rdm_ep_rdm_domain(ep)))
				efa_rdm_ope_try_fill_desc(txe, 0, FI_SEND);
			mr_attempted = true;
		}

		if (selected_proto->can_use_protocol_for_send(
			    txe, peer, req_pkt_type, header_flags, iface,
			    use_p2p)) {
			*proto = selected_proto;
			EFA_DBG(FI_LOG_EP_DATA,
				"Selected the %s protocol for a %zu byte send\n",
				selected_proto->name, txe->total_len);
			return FI_SUCCESS;
		}
	}

	/*
	 * No protocol matched, so the message is larger than a single eager
	 * packet and the caller falls back to the old code path. A zero-copy
	 * (headerless) peer reaches here for any message too large for eager,
	 * which is a legal application call.
	 */
	if (mr_attempted)
		efa_rdm_proto_release_selection_mrs(txe);

	*proto = NULL;
	return FI_SUCCESS;
}

/* Utility funcions */

void efa_rdm_proto_txe_fill(struct efa_rdm_ope *txe, struct efa_rdm_ep *ep,
			    struct efa_rdm_peer *peer, const struct fi_msg *msg,
			    uint32_t op, uint64_t tag, uint64_t flags,
			    uint32_t internal_flags,
			    struct efa_rdm_proto *proto)
{
	/*
	 * txe->mr, txe->desc and the MR generation snapshot were already
	 * populated by efa_rdm_proto_select_send_protocol(), which needs them
	 * to decide whether a protocol can be used, so use the construct
	 * helper that leaves them alone.
	 */
	efa_rdm_txe_construct_common(txe, ep, peer, msg, op, flags,
				     internal_flags);

	/*
	 * efa_rdm_txe_construct_common() clears txe->proto, so set it here
	 * after the reset. This must be the only place that assigns
	 * txe->proto during the fresh-send path — the pre-handshake repost
	 * path (efa_rdm_ope_repost_ope_queued_before_handshake) relies on
	 * txe->proto staying valid to choose the refactored send path.
	 */
	txe->proto = proto;

	if (op == ofi_op_tagged) {
		txe->cq_entry.tag = tag;
		txe->tag = tag;
	}
}
