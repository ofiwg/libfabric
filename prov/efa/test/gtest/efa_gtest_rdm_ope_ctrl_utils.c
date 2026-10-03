/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_rdm_ope_ctrl_utils.h"
#include "efa_gtest_common_helpers.h"
#include "efa.h"
#include "efa_av.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_ope.h"
#include "rdm/efa_rdm_pke.h"
#include "rdm/efa_rdm_pke_nonreq.h"
#include "rdm/efa_rdm_pke_utils.h"
#include "rdm/efa_rdm_peer.h"
#include "rdm/efa_rdm_protocol.h"
#include "rdm/efa_rdm_proto.h"
#include "rdm/protocols/efa_rdm_proto_eager.h"
#include "ofi_util.h"

#define EFA_TEST_CTRL_SOURCE_LEN 64
#define EFA_TEST_CTRL_PROV_ERRNO 42
/* Must exceed one MTU so the READRSP remainder case can overflow a packet. */
#define EFA_TEST_CTRL_SOURCE_MAX 16384

/*
 * One case's live state. Static because the C++ side drives post, decode and
 * cleanup as three separate calls and must not have to carry EFA internals.
 */
static struct {
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *ope;
	struct efa_rdm_peer *peer;
	size_t saved_outstanding_tx_ops;
	int saved_outstanding_valid;
	char source[EFA_TEST_CTRL_SOURCE_MAX];
} g_ctrl;

static struct efa_rdm_ep *efa_test_ctrl_ep(struct fid_ep *ep)
{
	return container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
}

/*
 * Own GID with a different QPN, not a fabricated one: these tests run against
 * the real device, which rejects an AH for a GID that is not on the fabric.
 */
static int efa_test_ctrl_setup_peer(struct fid_ep *ep, struct fid_av *av)
{
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr;
	int ret;

	ret = fi_getname(&ep->fid, &raw_addr, &raw_addr_len);
	if (ret)
		return ret;
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	if (fi_av_insert(av, &raw_addr, 1, &peer_addr, 0, NULL) != 1)
		return -FI_EINVAL;

	g_ctrl.peer = efa_rdm_ep_get_peer_explicit(g_ctrl.ep, peer_addr);
	if (!g_ctrl.peer)
		return -FI_EINVAL;

	g_ctrl.peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	g_ctrl.peer->av_entry->shm_fi_addr = FI_ADDR_NOTAVAIL;
	return 0;
}

static int efa_test_ctrl_needs_txe(int which)
{
	return which == EFA_TEST_CTRL_CTS_TXE ||
	       which == EFA_TEST_CTRL_PEER_ERROR_TXE_PROTO;
}

static uint32_t efa_test_ctrl_op_for(int which)
{
	switch (which) {
	case EFA_TEST_CTRL_READRSP_FITS:
	case EFA_TEST_CTRL_READRSP_REMAINDER:
		return ofi_op_read_rsp;
	case EFA_TEST_CTRL_ATOMRSP:
		return ofi_op_atomic_fetch;
	default:
		return ofi_op_msg;
	}
}

/* Give the ope the source buffer the data carrying packets read from. */
static void efa_test_ctrl_set_source(struct efa_rdm_ope *ope, size_t len)
{
	size_t i;

	assert(len <= EFA_TEST_CTRL_SOURCE_MAX);
	for (i = 0; i < len; ++i)
		g_ctrl.source[i] = (char) (i + 1);

	ope->iov_count = 1;
	ope->iov[0].iov_base = g_ctrl.source;
	ope->iov[0].iov_len = len;
	/* No descriptor, so the payload init takes its copy path and needs
	 * neither a registration nor p2p. */
	ope->desc[0] = NULL;
	ope->total_len = len;
	ope->cq_entry.len = len;
}

static int efa_test_ctrl_build_ope(int which, struct efa_test_ctrl_result *out)
{
	struct efa_rdm_ep *ep = g_ctrl.ep;
	struct efa_rdm_ope *ope;
	size_t len;

	if (efa_test_ctrl_needs_txe(which)) {
		struct iovec iov = {.iov_base = g_ctrl.source,
				    .iov_len = EFA_TEST_CTRL_SOURCE_LEN};
		struct fi_msg msg = {.msg_iov = &iov, .iov_count = 1};

		ope = ofi_buf_alloc(ep->base_ep.txe_pool);
		if (!ope)
			return -FI_ENOMEM;
		efa_rdm_txe_construct(ope, ep, g_ctrl.peer, &msg, ofi_op_msg, 0,
				      0);
	} else {
		ope = efa_rdm_ep_alloc_rxe(ep, g_ctrl.peer,
					   efa_test_ctrl_op_for(which));
		if (!ope)
			return -FI_ENOMEM;
	}
	g_ctrl.ope = ope;

	/* Distinct values so an assertion cannot pass on the wrong field. */
	ope->tx_id = 0x11;
	ope->rx_id = 0x22;
	ope->msg_id = 0x33;
	out->ids.tx_id = ope->tx_id;
	out->ids.rx_id = ope->rx_id;
	out->ids.msg_id = ope->msg_id;
	out->ids.connid = efa_rdm_ep_raw_addr(ep)->qkey;

	switch (which) {
	case EFA_TEST_CTRL_CTS_RXE:
		ope->total_len = EFA_TEST_CTRL_SOURCE_LEN;
		ope->bytes_received = 0;
		break;
	case EFA_TEST_CTRL_CTS_TXE:
		/* The emulated long CTS read sends CTS from a txe. */
		ope->total_len = EFA_TEST_CTRL_SOURCE_LEN;
		ope->bytes_received = 0;
		ope->cq_entry.flags |= FI_READ;
		break;
	case EFA_TEST_CTRL_READRSP_FITS:
		len = EFA_TEST_CTRL_SOURCE_LEN;
		efa_test_ctrl_set_source(ope, len);
		/* The RTR carries the window, as efa_rdm_pke_alloc_rtr_rxe()
		 * sets it; handle_readrsp_sent() draws the response down. */
		ope->window = len;
		out->source_len = len;
		memcpy(out->source_bytes, g_ctrl.source, len);
		break;
	case EFA_TEST_CTRL_READRSP_REMAINDER:
		/* Longer than one packet can carry, so the response is clamped
		 * and the ope goes on the long CTS send list. */
		len = efa_test_ctrl_readrsp_max_payload(
			      &ep->base_ep.util_ep.ep_fid) +
		      1;
		efa_test_ctrl_set_source(ope, len);
		ope->window = len;
		out->source_len = len;
		break;
	case EFA_TEST_CTRL_ATOMRSP:
		len = EFA_TEST_CTRL_SOURCE_LEN;
		efa_test_ctrl_set_source(ope, len);
		ope->atomrsp_data = ofi_buf_alloc(ep->rx_atomrsp_pool);
		if (!ope->atomrsp_data)
			return -FI_ENOMEM;
		memcpy(ope->atomrsp_data, g_ctrl.source, len);
		out->source_len = len;
		memcpy(out->source_bytes, g_ctrl.source, len);
		break;
	case EFA_TEST_CTRL_PEER_ERROR_RXE:
		ope->peer_error_prov_errno = EFA_TEST_CTRL_PROV_ERRNO;
		out->ids.prov_errno = EFA_TEST_CTRL_PROV_ERRNO;
		break;
	case EFA_TEST_CTRL_PEER_ERROR_TXE_PROTO:
		/* A migrated protocol on the ope must not divert a non-REQ
		 * packet into that protocol's construct_tx_pkes(). */
		ope->proto = &efa_rdm_proto_eager;
		ope->req_pkt_type = EFA_RDM_EAGER_MSGRTM_PKT;
		ope->peer_error_prov_errno = EFA_TEST_CTRL_PROV_ERRNO;
		out->ids.prov_errno = EFA_TEST_CTRL_PROV_ERRNO;
		break;
	default:
		break;
	}

	return 0;
}

static int efa_test_ctrl_pkt_type_for(int which)
{
	switch (which) {
	case EFA_TEST_CTRL_CTS_RXE:
	case EFA_TEST_CTRL_CTS_TXE:
		return EFA_RDM_CTS_PKT;
	case EFA_TEST_CTRL_READRSP_FITS:
	case EFA_TEST_CTRL_READRSP_REMAINDER:
		return EFA_RDM_READRSP_PKT;
	case EFA_TEST_CTRL_EOR:
		return EFA_RDM_EOR_PKT;
	case EFA_TEST_CTRL_RECEIPT:
		return EFA_RDM_RECEIPT_PKT;
	case EFA_TEST_CTRL_READ_NACK:
		return EFA_RDM_READ_NACK_PKT;
	case EFA_TEST_CTRL_ATOMRSP:
		return EFA_RDM_ATOMRSP_PKT;
	case EFA_TEST_CTRL_PEER_ERROR_RXE:
	case EFA_TEST_CTRL_PEER_ERROR_TXE_PROTO:
		return EFA_RDM_PEER_ERROR_PKT;
	default:
		return -1;
	}
}

static int efa_test_ctrl_on_list(struct dlist_entry *head,
				 struct dlist_entry *node)
{
	struct dlist_entry *item;

	dlist_foreach(head, item) {
		if (item == node)
			return 1;
	}
	return 0;
}

int efa_test_ctrl_post(struct fid_ep *ep, struct fid_av *av, int which,
		       int perturb, struct efa_test_ctrl_result *out)
{
	struct efa_rdm_ep *efa_rdm_ep = efa_test_ctrl_ep(ep);
	int pkt_type, ret;

	memset(&g_ctrl, 0, sizeof(g_ctrl));
	g_ctrl.ep = efa_rdm_ep;

	ret = efa_test_ctrl_setup_peer(ep, av);
	if (ret)
		return ret;

	ret = efa_test_ctrl_build_ope(which, out);
	if (ret)
		return ret;

	pkt_type = efa_test_ctrl_pkt_type_for(which);
	if (pkt_type < 0)
		return -FI_EINVAL;

	if (perturb == EFA_TEST_CTRL_PERTURB_FI_MORE)
		g_ctrl.ope->fi_flags |= FI_MORE;

	if (perturb == EFA_TEST_CTRL_PERTURB_TX_FULL) {
		g_ctrl.saved_outstanding_tx_ops =
			efa_rdm_ep->efa_outstanding_tx_ops;
		g_ctrl.saved_outstanding_valid = 1;
		efa_rdm_ep->efa_outstanding_tx_ops =
			efa_rdm_ep->efa_max_outstanding_tx_ops;
	}

	out->ret = efa_rdm_ope_post_send_or_queue(g_ctrl.ope, pkt_type);

	if (g_ctrl.saved_outstanding_valid)
		efa_rdm_ep->efa_outstanding_tx_ops =
			g_ctrl.saved_outstanding_tx_ops;

	out->queued_ctrl_flag_set =
		!!(g_ctrl.ope->internal_flags & EFA_RDM_OPE_QUEUED_CTRL);
	out->queued_ctrl_type = g_ctrl.ope->queued_ctrl_type;
	out->queued_list_empty = dlist_empty(&efa_rdm_ep->ope_queued_list);
	out->on_posted_ack_list =
		efa_test_ctrl_on_list(&efa_rdm_ep->ope_posted_ack_list,
				      &g_ctrl.ope->ack_list_entry);
	out->on_longcts_send_list =
		efa_test_ctrl_on_list(&efa_rdm_ep->ope_longcts_send_list,
				      &g_ctrl.ope->entry);
	out->ope_window = g_ctrl.ope->window;
	out->ope_bytes_sent = g_ctrl.ope->bytes_sent;
	out->ope_state = g_ctrl.ope->state;
	out->outstanding_tx_ops = efa_rdm_ep->efa_outstanding_tx_ops;
	return 0;
}

void efa_test_ctrl_decode_posted(struct fid_ep *ep,
				 struct efa_test_ctrl_wire *out)
{
	struct efa_rdm_ep *efa_rdm_ep = efa_test_ctrl_ep(ep);
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_base_hdr *base_hdr;

	if (efa_rdm_ep->send_pkt_entry_vec_size != 1)
		return;

	pkt_entry = efa_rdm_ep->send_pkt_entry_vec[0];
	if (!pkt_entry)
		return;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	out->decoded = 1;
	out->pkt_type = base_hdr->type;
	out->version = base_hdr->version;
	out->flags = base_hdr->flags;
	out->pkt_size = pkt_entry->pkt_size;
	out->payload_size = pkt_entry->payload_size;

	switch (base_hdr->type) {
	case EFA_RDM_CTS_PKT: {
		struct efa_rdm_cts_hdr *hdr =
			(struct efa_rdm_cts_hdr *) pkt_entry->wiredata;

		out->send_id = hdr->send_id;
		out->recv_id = hdr->recv_id;
		out->recv_length = hdr->recv_length;
		out->connid = hdr->connid;
		break;
	}
	case EFA_RDM_READRSP_PKT: {
		struct efa_rdm_readrsp_hdr *hdr =
			efa_rdm_pke_get_readrsp_hdr(pkt_entry);

		out->send_id = hdr->send_id;
		out->recv_id = hdr->recv_id;
		out->seg_length = hdr->seg_length;
		out->connid = hdr->connid;
		break;
	}
	case EFA_RDM_EOR_PKT: {
		struct efa_rdm_eor_hdr *hdr =
			(struct efa_rdm_eor_hdr *) pkt_entry->wiredata;

		out->send_id = hdr->send_id;
		out->recv_id = hdr->recv_id;
		out->connid = hdr->connid;
		break;
	}
	case EFA_RDM_READ_NACK_PKT: {
		struct efa_rdm_read_nack_hdr *hdr =
			(struct efa_rdm_read_nack_hdr *) pkt_entry->wiredata;

		out->send_id = hdr->send_id;
		out->recv_id = hdr->recv_id;
		out->connid = hdr->connid;
		break;
	}
	case EFA_RDM_RECEIPT_PKT: {
		struct efa_rdm_receipt_hdr *hdr =
			efa_rdm_pke_get_receipt_hdr(pkt_entry);

		out->tx_id = hdr->tx_id;
		out->msg_id = hdr->msg_id;
		out->connid = hdr->connid;
		break;
	}
	case EFA_RDM_ATOMRSP_PKT: {
		struct efa_rdm_atomrsp_pkt *pkt =
			(struct efa_rdm_atomrsp_pkt *) pkt_entry->wiredata;

		out->recv_id = pkt->hdr.recv_id;
		out->seg_length = pkt->hdr.seg_length;
		out->connid = pkt->hdr.connid;
		if (pkt->hdr.seg_length <= EFA_TEST_CTRL_PAYLOAD_MAX)
			memcpy(out->payload_bytes, pkt->data,
			       pkt->hdr.seg_length);
		break;
	}
	case EFA_RDM_PEER_ERROR_PKT: {
		struct efa_rdm_peer_error_hdr *hdr =
			efa_rdm_pke_get_peer_error_hdr(pkt_entry);

		out->msg_id = hdr->msg_id;
		out->op_id = hdr->op_id;
		out->emitter_ope_type = hdr->emitter_ope_type;
		out->prov_errno = hdr->prov_errno;
		out->connid = hdr->connid;
		break;
	}
	default:
		break;
	}

	/* A data carrying packet copies into wiredata after its header. */
	if (pkt_entry->payload && pkt_entry->payload_size &&
	    pkt_entry->payload_size <= EFA_TEST_CTRL_PAYLOAD_MAX &&
	    base_hdr->type != EFA_RDM_ATOMRSP_PKT)
		memcpy(out->payload_bytes, pkt_entry->payload,
		       pkt_entry->payload_size);
}

void efa_test_ctrl_cleanup(void)
{
	struct efa_rdm_ep *ep = g_ctrl.ep;
	struct efa_rdm_ope *ope = g_ctrl.ope;

	if (!ep || !ope)
		return;

	/* A posted packet is counted and linked; undo both before releasing. */
	if (ep->send_pkt_entry_vec_size == 1 && ep->send_pkt_entry_vec[0] &&
	    ope->efa_outstanding_tx_ops) {
		struct efa_rdm_pke *pkt_entry = ep->send_pkt_entry_vec[0];

		efa_rdm_ep_record_tx_op_completed(ep, pkt_entry);
		efa_rdm_pke_release_tx(pkt_entry);
		ep->send_pkt_entry_vec_size = 0;
	}

	if (ope->internal_flags & EFA_RDM_OPE_QUEUED_CTRL) {
		ope->internal_flags &= ~EFA_RDM_OPE_QUEUED_CTRL;
		dlist_remove(&ope->queued_entry);
	}

	if (ope->atomrsp_data) {
		ofi_buf_free(ope->atomrsp_data);
		ope->atomrsp_data = NULL;
	}

	if (ope->type == EFA_RDM_TXE)
		efa_rdm_txe_release(ope);
	else
		efa_rdm_rxe_release(ope);

	memset(&g_ctrl, 0, sizeof(g_ctrl));
}

int efa_test_ctrl_expected_pkt_type(int which)
{
	return efa_test_ctrl_pkt_type_for(which);
}

size_t efa_test_ctrl_expected_hdr_size(int which)
{
	switch (efa_test_ctrl_pkt_type_for(which)) {
	case EFA_RDM_CTS_PKT:
		return sizeof(struct efa_rdm_cts_hdr);
	case EFA_RDM_READRSP_PKT:
		return sizeof(struct efa_rdm_readrsp_hdr);
	case EFA_RDM_EOR_PKT:
		return sizeof(struct efa_rdm_eor_hdr);
	case EFA_RDM_RECEIPT_PKT:
		return sizeof(struct efa_rdm_receipt_hdr);
	case EFA_RDM_READ_NACK_PKT:
		return sizeof(struct efa_rdm_read_nack_hdr);
	case EFA_RDM_ATOMRSP_PKT:
		return sizeof(struct efa_rdm_atomrsp_hdr);
	case EFA_RDM_PEER_ERROR_PKT:
		return sizeof(struct efa_rdm_peer_error_hdr);
	default:
		return 0;
	}
}

int efa_test_ctrl_protocol_version(void)
{
	return EFA_RDM_PROTOCOL_VERSION;
}

uint16_t efa_test_ctrl_connid_hdr_flag(void)
{
	return EFA_RDM_PKT_CONNID_HDR;
}

uint16_t efa_test_ctrl_cts_read_req_flag(void)
{
	return EFA_RDM_CTS_READ_REQ;
}

int efa_test_ctrl_ope_type_txe(void)
{
	return EFA_RDM_TXE;
}

int efa_test_ctrl_ope_type_rxe(void)
{
	return EFA_RDM_RXE;
}

uint32_t efa_test_ctrl_ope_id_invalid(void)
{
	return EFA_RDM_OPE_ID_INVALID;
}

uint64_t efa_test_ctrl_expected_cts_recv_length(struct fid_ep *ep,
					       uint64_t bytes_left)
{
	struct efa_rdm_ep *efa_rdm_ep = efa_test_ctrl_ep(ep);
	uint64_t credits =
		(uint64_t) efa_env.tx_min_credits *
		efa_rdm_ep->max_data_payload_size;

	return MIN(bytes_left, credits);
}

size_t efa_test_ctrl_readrsp_max_payload(struct fid_ep *ep)
{
	struct efa_rdm_ep *efa_rdm_ep = efa_test_ctrl_ep(ep);

	return efa_rdm_ep->mtu_size - sizeof(struct efa_rdm_readrsp_hdr);
}
