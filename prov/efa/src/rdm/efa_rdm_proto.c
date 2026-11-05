/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto.h"
#include "efa.h"
#include "efa_rdm_domain.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_pke_nonreq.h"
#include "protocols/efa_rdm_proto_eager.h"
#include "protocols/efa_rdm_proto_eager_write.h"
#include "protocols/efa_rdm_proto_longcts.h"
#include "protocols/efa_rdm_proto_longread.h"
#include "protocols/efa_rdm_proto_longread_write.h"
#include "protocols/efa_rdm_proto_longcts_write.h"
#include "protocols/efa_rdm_proto_medium.h"
#include "protocols/efa_rdm_proto_runtread.h"
#include "protocols/efa_rdm_proto_longcts_rtr.h"
#include "protocols/efa_rdm_proto_short_rtr.h"
#include "efa_rdm_msg.h"

/**
 * @brief Undo the memory registrations the selection loop made.
 *
 * efa_rdm_ope_try_fill_desc() registers the source buffer so a read based
 * protocol can be evaluated. Hand those registrations back when no protocol
 * can carry the operation.
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

// TODO double-check for ROCr
#if HAVE_CUDA || HAVE_NEURON || HAVE_ROCR
size_t efa_rdm_proto_max_eager_msg_size_with_largest_header(void){
	static bool computed = false;
	static size_t size = 0;

	if (!computed) {
		assert(g_efa_selected_device_list);
		size = g_efa_selected_device_list[0].ibv_port_attr.max_msg_sz - efa_rdm_pkt_type_get_max_hdr_size();
		computed = true;
	}

	return size;
}
#else
size_t efa_rdm_proto_max_eager_msg_size_with_largest_header(void){
	return 0;
}
#endif


/* List of supported protocols.
 * The protocols listed here will be tried in the order they're listed.
 * The first protocol that can be used for the TX operation will be used.
 */
static struct efa_rdm_proto * const efa_rdm_protocols[] = {
	&efa_rdm_proto_eager,
	&efa_rdm_proto_medium,
	&efa_rdm_proto_runtread,
	&efa_rdm_proto_longread,
	/* Long CTS is the final fallback for messages within its size bounds. */
	&efa_rdm_proto_longcts,
};

/*
 * Emulated write protocols, tried in order during selection.
 */
static struct efa_rdm_proto * const efa_rdm_emulated_write_protocols[] = {
	&efa_rdm_proto_eager_write,
	&efa_rdm_proto_longread_write,
	&efa_rdm_proto_longcts_write,
};

/*
 * Emulated read protocols, tried in order during selection.
 */
static struct efa_rdm_proto * const efa_rdm_emulated_read_protocols[] = {
	&efa_rdm_proto_short_rtr,
	&efa_rdm_proto_longcts_rtr,
};

static void efa_rdm_proto_init_bounds(struct efa_rdm_proto *proto, enum fi_hmem_iface target_iface, enum fi_hmem_iface bounds_profile_iface){
	bool apply_param = true;

	proto->min_size[target_iface] = 0;
	proto->max_size[target_iface] = SIZE_MAX;
	if (proto->set_bounds)
		apply_param = proto->set_bounds(proto, target_iface, bounds_profile_iface);

	/* Keep defaults unless a set env override applies to this bounds profile. */
	if (!apply_param)
		return;

	if (proto->min_param_name)
		fi_param_get_size_t(&efa_prov, proto->min_param_name, &proto->min_size[target_iface]);
	if (proto->max_param_name)
		fi_param_get_size_t(&efa_prov, proto->max_param_name, &proto->max_size[target_iface]);
}

void efa_rdm_proto_init_bounds_for_iface(enum fi_hmem_iface target_iface, enum fi_hmem_iface bounds_profile_iface){
	size_t i;

	for (i = 0; i < ARRAY_SIZE(efa_rdm_protocols); ++i)
		efa_rdm_proto_init_bounds(efa_rdm_protocols[i], target_iface, bounds_profile_iface);
	for (i = 0; i < ARRAY_SIZE(efa_rdm_emulated_write_protocols); ++i)
		efa_rdm_proto_init_bounds(efa_rdm_emulated_write_protocols[i], target_iface, bounds_profile_iface);
	for (i = 0; i < ARRAY_SIZE(efa_rdm_emulated_read_protocols); ++i)
		efa_rdm_proto_init_bounds(efa_rdm_emulated_read_protocols[i], target_iface, bounds_profile_iface);
}

void efa_rdm_proto_handle_receipt_recv(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_receipt_hdr *receipt_hdr =
		efa_rdm_pke_get_receipt_hdr(pkt_entry);
	struct efa_rdm_ope *txe =
		efa_rdm_ep_live_txe_from_id(pkt_entry->ep, receipt_hdr->tx_id);

	if (!txe) {
		EFA_INFO(FI_LOG_CQ,
			 "RECEIPT names a send that is no longer live, dropping it\n");
		efa_rdm_pke_release_rx(pkt_entry);
		return;
	}

	/* Write send completion immediately to preserve DC semantics. */
	efa_rdm_txe_report_completion(txe);

	if (txe->state == EFA_RDM_OPE_SEND)
		dlist_remove(&txe->entry);

	/*
	 * The TXE is released either here or when the request/CTSDATA packet's
	 * send completes, whichever happens last.
	 */
	txe->internal_flags |= EFA_RDM_TXE_REMOTE_ACK_RECEIVED;
	if (efa_rdm_txe_with_remote_ack_ready_for_release(txe))
		efa_rdm_txe_release(txe);

	efa_rdm_pke_release_rx(pkt_entry);
}

void efa_rdm_proto_handle_eor_recv(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_eor_hdr *eor_hdr =
		(struct efa_rdm_eor_hdr *) pkt_entry->wiredata;
	struct efa_rdm_ope *txe =
		efa_rdm_ep_live_txe_from_id(pkt_entry->ep, eor_hdr->send_id);

	if (!txe) {
		EFA_INFO(FI_LOG_CQ,
			 "EOR names a send that is no longer live, dropping it\n");
		efa_rdm_pke_release_rx(pkt_entry);
		return;
	}

	efa_rdm_txe_release_read_msg_slot(txe);

	txe->bytes_acked += txe->total_len - txe->bytes_runt;
	if (txe->bytes_acked == txe->total_len) {
		efa_rdm_txe_report_completion(txe);
		txe->internal_flags |= EFA_RDM_TXE_REMOTE_ACK_RECEIVED;
		if (efa_rdm_txe_with_remote_ack_ready_for_release(txe))
			efa_rdm_txe_release(txe);
	}

	efa_rdm_pke_release_rx(pkt_entry);
}

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
	uint64_t effective_flags, mr_access_flags;

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

	/* can_use_protocol() reads the peer off the txe, before txe_fill runs. */
	txe->peer = peer;

	/*
	 * The predicates need the effective flags: the runt read protocol has no
	 * delivery complete REQ variant, so it has to rule out a
	 * FI_DELIVERY_COMPLETE send. efa_rdm_txe_construct_common() assigns the
	 * same value again later, from the same helper.
	 */
	txe->fi_flags = effective_flags;

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
	mr_access_flags = FI_SEND | (use_p2p ? FI_REMOTE_READ : 0);

	/* Synapse AI only supports long read */
	if (iface == FI_HMEM_SYNAPSEAI) {
		if (efa_is_cache_available(efa_rdm_ep_rdm_domain(ep)))
			efa_rdm_ope_try_fill_desc(txe, 0, mr_access_flags);

		*proto = &efa_rdm_proto_longread;
		txe->proto = &efa_rdm_proto_longread;
		txe->req_pkt_type = efa_rdm_proto_req_pkt_type(
			&efa_rdm_proto_longread, op, effective_flags, peer);
		return FI_SUCCESS;
	}

	header_flags = efa_rdm_proto_req_header_flags(peer, effective_flags);

	/*
	 * FI_EFA_RDM_FORCE_SEND_PROTO pins the choice to one protocol by name.
	 * The name replaces the size decision rather than narrowing it, so the
	 * named protocol is used whether or not this message suits it; that is
	 * the point, since no threshold can steer a message that fits in a
	 * single packet away from eager.
	 */
	if (efa_env.rdm_force_send_proto) {
		for (int i = 0; i < ARRAY_SIZE(efa_rdm_protocols); ++i) {
			selected_proto = efa_rdm_protocols[i];

			if (strcmp(efa_env.rdm_force_send_proto,
				   selected_proto->name))
				continue;

			if (selected_proto->wants_mr &&
			    efa_is_cache_available(efa_rdm_ep_rdm_domain(ep)))
				efa_rdm_ope_try_fill_desc(
					txe, 0, mr_access_flags);

			*proto = selected_proto;
			txe->proto = selected_proto;
			txe->req_pkt_type = efa_rdm_proto_req_pkt_type(
				selected_proto, op, effective_flags, peer);
			EFA_INFO(FI_LOG_EP_DATA,
				 "Forced the %s protocol for a %zu byte send\n",
				 selected_proto->name, txe->total_len);
			return FI_SUCCESS;
		}

		EFA_WARN(FI_LOG_EP_DATA,
			 "FI_EFA_RDM_FORCE_SEND_PROTO=%s does not name a send protocol\n",
			 efa_env.rdm_force_send_proto);
		*proto = NULL;
		txe->proto = NULL;
		return -FI_EOPNOTSUPP;
	}

	for (int i = 0; i < ARRAY_SIZE(efa_rdm_protocols); ++i) {
		selected_proto = efa_rdm_protocols[i];

		req_pkt_type = efa_rdm_proto_req_pkt_type(
			selected_proto, op, effective_flags, peer);

		/* If a protocol can benefit from memory registration, attempt
		 * to register the memory.
		 *
		 * Ask for FI_REMOTE_READ whenever p2p is available, because
		 * read based protocols require it.
		 *
		 * TODO: Move efa_rdm_ope_try_fill_desc to efa_rdm_proto.c
		 */
		if (!mr_attempted && selected_proto->wants_mr) {

			if (efa_is_cache_available(efa_rdm_ep_rdm_domain(ep)))
				efa_rdm_ope_try_fill_desc(txe, 0, mr_access_flags);
			mr_attempted = true;
		}

		if (selected_proto->can_use_protocol(selected_proto, txe,
						     req_pkt_type,
						     header_flags, iface,
						     use_p2p)) {
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
	 */
	if (mr_attempted)
		efa_rdm_proto_release_selection_mrs(txe);

	*proto = NULL;
	txe->proto = NULL;
	return -FI_EOPNOTSUPP;
}

/* Utility funcions */

/**
 * @brief Split a transfer's REQ data into near-equal packets of at most
 *        max_data_size bytes.
 *
 * The data is cut into near-equal pieces rounded down to the memory alignment,
 * because the device handles similarly sized sends best. Leftover bytes go on
 * the last packet, or on one extra packet if they do not fit.
 *
 * @return 0 on success, -FI_EAGAIN when not enough TX packets are free
 */
ssize_t efa_rdm_proto_split_req_data(struct efa_rdm_ep *ep,
				     struct efa_rdm_ope *txe,
				     size_t total_data_size,
				     size_t max_data_size,
				     size_t *pkt_entry_cnt,
				     size_t *pkt_entry_data_size_vec)
{
	size_t i, cnt, max_cnt, data_size, full_cnt, remainder;
	size_t memory_alignment;
	enum fi_hmem_iface iface;

	assert(total_data_size);
	assert(max_data_size);

	max_cnt = MIN(efa_rdm_ep_get_available_tx_pkts(ep),
		      efa_base_ep_get_tx_pool_size(&ep->base_ep));

	iface = txe->desc[0] ? ((struct efa_mr *) txe->desc[0])->iface :
			       FI_HMEM_SYSTEM;
	memory_alignment = efa_rdm_ep_get_memory_alignment(ep, iface);

	cnt = (total_data_size - 1) / max_data_size + 1;
	data_size = (total_data_size - 1) / cnt + 1;
	/* A single packet may be smaller than the memory alignment. */
	if (cnt > 1)
		data_size &= ~(memory_alignment - 1);
	assert(data_size);

	full_cnt = total_data_size / data_size;
	remainder = total_data_size - full_cnt * data_size;
	cnt = full_cnt + (data_size + remainder > max_data_size ? 1 : 0);

	if (cnt > max_cnt)
		return -FI_EAGAIN;

	assert(full_cnt > 0);
	for (i = 0; i < full_cnt; ++i)
		pkt_entry_data_size_vec[i] = data_size;

	if (cnt == full_cnt)
		pkt_entry_data_size_vec[full_cnt - 1] += remainder;
	else
		pkt_entry_data_size_vec[full_cnt] = remainder;

	*pkt_entry_cnt = cnt;
	return FI_SUCCESS;
}

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

void efa_rdm_proto_select_emulated_write_protocol(struct efa_rdm_ep *ep,
						  struct efa_rdm_peer *peer,
						  struct efa_rdm_ope *txe,
						  bool use_p2p,
						  struct efa_rdm_proto **proto)
{
	struct efa_rdm_proto *selected_proto;
	uint16_t header_flags;
	int req_pkt_type, iface, i;
	bool mr_attempted = false;

	iface = txe->desc[0] ?
			((struct efa_mr *) txe->desc[0])->iface :
			FI_HMEM_SYSTEM;

	header_flags = efa_rdm_proto_req_header_flags(peer, txe->fi_flags);

	for (i = 0; i < ARRAY_SIZE(efa_rdm_emulated_write_protocols); ++i) {
		selected_proto = efa_rdm_emulated_write_protocols[i];

		req_pkt_type = efa_rdm_proto_req_pkt_type(
			selected_proto, txe->op, txe->fi_flags, peer);

		/*
		 * Register the source buffer for a protocol that benefits from
		 * it, asking for FI_REMOTE_READ when p2p is available because
		 * the read based protocols require it.
		 */
		if (!mr_attempted && selected_proto->wants_mr) {
			uint64_t access = FI_SEND |
					  (use_p2p ? FI_REMOTE_READ : 0);

			if (efa_is_cache_available(efa_rdm_ep_rdm_domain(ep)))
				efa_rdm_ope_try_fill_desc(txe, 0, access);
			mr_attempted = true;
		}

		if (selected_proto->can_use_protocol(selected_proto, txe,
						     req_pkt_type,
						     header_flags, iface,
						     use_p2p)) {
			*proto = selected_proto;
			txe->proto = selected_proto;
			txe->req_pkt_type = req_pkt_type;
			return;
		}
	}

	/*
	 * No emulated write protocol matched. This is not expected; the caller
	 * reports an error.
	 */
	*proto = NULL;
	txe->proto = NULL;
}

void efa_rdm_proto_select_emulated_read_protocol(struct efa_rdm_ep *ep,
						 struct efa_rdm_peer *peer,
						 struct efa_rdm_ope *txe,
						 struct efa_rdm_proto **proto)
{
	struct efa_rdm_proto *selected_proto;
	uint16_t header_flags;
	int req_pkt_type, iface, i;

	iface = txe->desc[0] ?
			((struct efa_mr *) txe->desc[0])->iface :
			FI_HMEM_SYSTEM;

	header_flags = efa_rdm_proto_req_header_flags(peer, txe->fi_flags);

	for (i = 0; i < ARRAY_SIZE(efa_rdm_emulated_read_protocols); ++i) {
		selected_proto = efa_rdm_emulated_read_protocols[i];

		req_pkt_type = efa_rdm_proto_req_pkt_type(
			selected_proto, txe->op, txe->fi_flags, peer);

		if (selected_proto->can_use_protocol(selected_proto, txe,
						     req_pkt_type,
						     header_flags, iface,
						     false /* use_p2p */)) {
			*proto = selected_proto;
			txe->proto = selected_proto;
			txe->req_pkt_type = req_pkt_type;
			return;
		}
	}

	/*
	 * No emulated read protocol matched. This is not expected; the caller
	 * asserts that a protocol was selected.
	 */
	*proto = NULL;
	txe->proto = NULL;
}
