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

int efa_rdm_proto_set_min_bound(struct efa_rdm_proto *proto, enum fi_hmem_iface iface, size_t value){
	assert(proto && (int) iface >= 0 && iface < OFI_HMEM_MAX);

	if (value > proto->bounds[iface].max)
		return -FI_EINVAL;

	proto->bounds[iface].min = value;
	return FI_SUCCESS;
}

int efa_rdm_proto_set_max_bound(struct efa_rdm_proto *proto, enum fi_hmem_iface iface, size_t value){
	assert(proto && (int) iface >= 0 && iface < OFI_HMEM_MAX);

	if (value < proto->bounds[iface].min)
		return -FI_EINVAL;

	proto->bounds[iface].max = value;
	return FI_SUCCESS;
}

int efa_rdm_proto_set_bound_unsupported(struct efa_rdm_proto *proto, enum fi_hmem_iface iface, size_t value){
	(void) proto;
	(void) iface;
	(void) value;
	return -FI_EOPNOTSUPP;
}

// TODO double-check for ROCr
#if HAVE_CUDA || HAVE_NEURON || HAVE_ROCR
static size_t efa_rdm_proto_max_eager_msg_size_with_largest_header(void){
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
static size_t efa_rdm_proto_max_eager_msg_size_with_largest_header(void){
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
	/* Long CTS takes any send the others above decline, so it is last. */
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

/*
 * Every protocol registry. The bound params are defined, and the bounds are
 * initialized, saved and restored, for the protocols listed here.
 */
static const struct {
	struct efa_rdm_proto * const *protos;
	size_t count;
} efa_rdm_proto_registries[] = {
	{ efa_rdm_protocols, ARRAY_SIZE(efa_rdm_protocols) },
	{ efa_rdm_emulated_write_protocols, ARRAY_SIZE(efa_rdm_emulated_write_protocols) },
	{ efa_rdm_emulated_read_protocols, ARRAY_SIZE(efa_rdm_emulated_read_protocols) },
};

/* Number of protocols in efa_rdm_proto_registries */
#define EFA_RDM_PROTO_COUNT \
	(ARRAY_SIZE(efa_rdm_protocols) + \
	 ARRAY_SIZE(efa_rdm_emulated_write_protocols) + \
	 ARRAY_SIZE(efa_rdm_emulated_read_protocols))

/* proto->min_bound_param & proto->max_bound_param so it is 2 times the number of protocols */
#define EFA_RDM_PROTO_BOUND_PARAM_CAPACITY (2 * EFA_RDM_PROTO_COUNT)

/**
 * @brief Return the protocol at @p idx when the registries are walked in order.
 *
 * @return the protocol, or NULL once @p idx is past the last protocol
 */
static struct efa_rdm_proto *efa_rdm_proto_registry_get(size_t idx){
	size_t i;

	for (i = 0; i < ARRAY_SIZE(efa_rdm_proto_registries); i++) {
		if (idx < efa_rdm_proto_registries[i].count)
			return efa_rdm_proto_registries[i].protos[idx];
		idx -= efa_rdm_proto_registries[i].count;
	}

	return NULL;
}

static size_t efa_rdm_proto_inter_max_medium_message_size_default(enum fi_hmem_iface policy_iface){
	/* Only eager and the read based protocols carry device memory */
	return policy_iface == FI_HMEM_SYSTEM ? EFA_DEFAULT_INTER_MAX_MEDIUM_MESSAGE_SIZE : 0;
}

static size_t efa_rdm_proto_read_min_default(enum fi_hmem_iface policy_iface, size_t system_default){
	switch (policy_iface) {
	case FI_HMEM_SYSTEM:
		return system_default;
	case FI_HMEM_SYNAPSEAI:
		/* Sends always use long read, see efa_rdm_proto_select_send_protocol() */
		return 1;
	default:
		/* Only eager and the read based protocols carry device memory */
		return efa_rdm_proto_max_eager_msg_size_with_largest_header() + 1;
	}
}

static size_t efa_rdm_proto_inter_min_read_message_size_default(enum fi_hmem_iface policy_iface){
	return efa_rdm_proto_read_min_default(policy_iface, EFA_DEFAULT_INTER_MIN_READ_MESSAGE_SIZE);
}

static size_t efa_rdm_proto_inter_min_read_write_size_default(enum fi_hmem_iface policy_iface){
	return efa_rdm_proto_read_min_default(policy_iface, EFA_DEFAULT_INTER_MIN_READ_WRITE_SIZE);
}

const struct efa_rdm_proto_bound_param efa_rdm_proto_inter_max_medium_message_size = {
	.param = "inter_max_medium_message_size",
	.env = "FI_EFA_INTER_MAX_MEDIUM_MESSAGE_SIZE",
	.help = "The maximum message size for inter EFA medium message protocol (Default 65536).",
	.host_policy_only = true,
	.policy_default = &efa_rdm_proto_inter_max_medium_message_size_default,
};

const struct efa_rdm_proto_bound_param efa_rdm_proto_inter_min_read_message_size = {
	.param = "inter_min_read_message_size",
	.env = "FI_EFA_INTER_MIN_READ_MESSAGE_SIZE",
	.help = "The minimum message size in bytes for inter EFA read message "
		"protocol. If instance support RDMA read, messages whose size is "
		"larger than this value will be sent by read message protocol (Default 1048576).",
	.policy_default = &efa_rdm_proto_inter_min_read_message_size_default,
};

const struct efa_rdm_proto_bound_param efa_rdm_proto_inter_min_read_write_size = {
	.param = "inter_min_read_write_size",
	.env = "FI_EFA_INTER_MIN_READ_WRITE_SIZE",
	.help = "The minimum message size for inter EFA write to use read write protocol. "
		"If firmware support RDMA read, and FI_EFA_USE_DEVICE_RDMA is 1, "
		"write requests whose size is larger than this value will use the read write protocol (Default 65536). "
		"If the efa device supports RDMA write, device RDMA write will always be used.",
	.policy_default = &efa_rdm_proto_inter_min_read_write_size_default,
};

static bool efa_rdm_proto_bound_param_seen(
	const struct efa_rdm_proto_bound_param *param,
	const struct efa_rdm_proto_bound_param *const *seen,
	size_t seen_count){
	size_t i;

	for (i = 0; i < seen_count; i++)
		if (seen[i] == param)
			return true;

	return false;
}

void efa_rdm_proto_define_bound_params(void){
	const struct efa_rdm_proto_bound_param* seen[EFA_RDM_PROTO_BOUND_PARAM_CAPACITY] = {0};
	const struct efa_rdm_proto_bound_param *params[2];
	struct efa_rdm_proto *proto;
	size_t seen_count = 0;
	size_t i, j;

	for (i = 0; (proto = efa_rdm_proto_registry_get(i)); i++) {
		/* A bound param needs a setter that can apply it */
		assert(!proto->min_bound_param || (proto->set_min_bound && proto->set_min_bound != &efa_rdm_proto_set_bound_unsupported));
		assert(!proto->max_bound_param || (proto->set_max_bound && proto->set_max_bound != &efa_rdm_proto_set_bound_unsupported));

		params[0] = proto->min_bound_param;
		params[1] = proto->max_bound_param;

		for (j = 0; j < ARRAY_SIZE(params); j++) {
			if (!params[j] || efa_rdm_proto_bound_param_seen(params[j], seen, seen_count))
				continue;

			fi_param_define(&efa_prov, params[j]->param,
					FI_PARAM_SIZE_T, "%s", params[j]->help);
			seen[seen_count++] = params[j];
		}
	}
}

static void efa_rdm_proto_set_bounds(struct efa_rdm_proto *proto, enum fi_hmem_iface iface, size_t min, size_t max){
	proto->bounds[iface].min = min;
	proto->bounds[iface].max = max;
}

/* The bound @p param sets for @p policy_iface before the environment override, or @p unbounded when it sets none */
static size_t efa_rdm_proto_bound_param_default(const struct efa_rdm_proto_bound_param *param, enum fi_hmem_iface policy_iface, size_t unbounded){
	return (param && param->policy_default) ? param->policy_default(policy_iface) : unbounded;
}

static void efa_rdm_proto_apply_bound_param(
	struct efa_rdm_proto *proto,
	const struct efa_rdm_proto_bound_param *param, bool is_min,
	enum fi_hmem_iface target_iface, enum fi_hmem_iface policy_iface,
	const struct efa_rdm_proto_bound_param **ignored,
	size_t *ignored_count){
	size_t value;
	int ret;

	if (!param || fi_param_get_size_t(&efa_prov, param->param, &value))
		return;

	/* SynapseAI always uses long read, and the HMEM policies only use
	 * eager and read based protocols.
	 */
	if (policy_iface == FI_HMEM_SYNAPSEAI ||
	    (param->host_policy_only && policy_iface != FI_HMEM_SYSTEM)) {
		if (!efa_rdm_proto_bound_param_seen(param, ignored, *ignored_count)) {
			EFA_WARN(FI_LOG_CORE, "%s is ignored for %s\n", param->env,
				 fi_tostr(&target_iface, FI_TYPE_HMEM_IFACE));
			ignored[(*ignored_count)++] = param;
		}
		return;
	}

	ret = is_min ?
		proto->set_min_bound(proto, target_iface, value) :
		proto->set_max_bound(proto, target_iface, value);
	if (ret)
		EFA_WARN(FI_LOG_CORE,
			 "Ignoring rejected %s=%zu for %s on %s: %s\n",
			 param->env, value, proto->name,
			 fi_tostr(&target_iface, FI_TYPE_HMEM_IFACE),
			 fi_strerror(-ret));
}

void efa_rdm_proto_init_bounds_for_iface(enum fi_hmem_iface target_iface, enum fi_hmem_iface policy_iface){
	const struct efa_rdm_proto_bound_param
		*ignored[EFA_RDM_PROTO_BOUND_PARAM_CAPACITY] = {0};
	struct efa_rdm_proto *proto;
	size_t ignored_count = 0;
	size_t i;

	assert((int) target_iface >= 0 && target_iface < OFI_HMEM_MAX);

	switch (policy_iface) {
	case FI_HMEM_SYSTEM:
	case FI_HMEM_CUDA:
	case FI_HMEM_ROCR:
	case FI_HMEM_NEURON:
	case FI_HMEM_SYNAPSEAI:
		break;
	default:
		EFA_WARN(FI_LOG_CORE, "No protocol bounds policy for %s\n",
			 fi_tostr(&policy_iface, FI_TYPE_HMEM_IFACE));
		return;
	}

	/* Each protocol starts at [0, SIZE_MAX], takes its params' defaults, then their environment overrides */
	for (i = 0; (proto = efa_rdm_proto_registry_get(i)); i++) {
		efa_rdm_proto_set_bounds(proto, target_iface, efa_rdm_proto_bound_param_default(proto->min_bound_param, policy_iface, 0), efa_rdm_proto_bound_param_default(proto->max_bound_param, policy_iface, SIZE_MAX));
		efa_rdm_proto_apply_bound_param(
			proto, proto->min_bound_param, true,
			target_iface, policy_iface, ignored, &ignored_count);
		efa_rdm_proto_apply_bound_param(
			proto, proto->max_bound_param, false,
			target_iface, policy_iface, ignored, &ignored_count);
	}
}

static struct efa_rdm_proto_bounds efa_rdm_proto_saved_bounds[EFA_RDM_PROTO_COUNT][OFI_HMEM_MAX];

void efa_rdm_proto_bounds_snapshot(void){
	struct efa_rdm_proto *proto;
	size_t i;

	for (i = 0; (proto = efa_rdm_proto_registry_get(i)); i++)
		memcpy(efa_rdm_proto_saved_bounds[i], proto->bounds, sizeof(efa_rdm_proto_saved_bounds[i]));
}

void efa_rdm_proto_bounds_restore(void){
	struct efa_rdm_proto *proto;
	size_t i;

	for (i = 0; (proto = efa_rdm_proto_registry_get(i)); i++)
		memcpy(proto->bounds, efa_rdm_proto_saved_bounds[i], sizeof(efa_rdm_proto_saved_bounds[i]));
}

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
	uint16_t header_flags = 0;
	int req_pkt_type, iface, i;
	bool mr_attempted = false;

	iface = txe->desc[0] ?
			((struct efa_mr *) txe->desc[0])->iface :
			FI_HMEM_SYSTEM;

	if (efa_rdm_peer_need_raw_addr_hdr(peer))
		header_flags |= EFA_RDM_REQ_OPT_RAW_ADDR_HDR;
	else if (efa_rdm_peer_need_connid(peer))
		header_flags |= EFA_RDM_PKT_CONNID_HDR;

	if (txe->fi_flags & FI_REMOTE_CQ_DATA)
		header_flags |= EFA_RDM_REQ_OPT_CQ_DATA_HDR;

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
	 * No emulated write protocol matched, so release any MRs that were
	 * registered. The caller reports an error.
	 */
	if (mr_attempted)
		efa_rdm_proto_release_selection_mrs(txe);

	*proto = NULL;
	txe->proto = NULL;
}

void efa_rdm_proto_select_emulated_read_protocol(struct efa_rdm_ep *ep,
						 struct efa_rdm_peer *peer,
						 struct efa_rdm_ope *txe,
						 struct efa_rdm_proto **proto)
{
	struct efa_rdm_proto *selected_proto;
	uint16_t header_flags = 0;
	int req_pkt_type, iface, i;

	iface = txe->desc[0] ?
			((struct efa_mr *) txe->desc[0])->iface :
			FI_HMEM_SYSTEM;

	if (efa_rdm_peer_need_raw_addr_hdr(peer))
		header_flags |= EFA_RDM_REQ_OPT_RAW_ADDR_HDR;
	else if (efa_rdm_peer_need_connid(peer))
		header_flags |= EFA_RDM_PKT_CONNID_HDR;

	if (txe->fi_flags & FI_REMOTE_CQ_DATA)
		header_flags |= EFA_RDM_REQ_OPT_CQ_DATA_HDR;

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
