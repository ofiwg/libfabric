/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_rdm_mr_abort_utils.h"
#include "efa.h"
#include "efa_av.h"
#include "rdm/efa_rdm_cq.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_mr.h"
#include "rdm/efa_rdm_ope.h"
#include "rdm/efa_rdm_peer.h"
#include "rdm/efa_rdm_pke.h"
#include "rdm/efa_rdm_pke_cmd.h"
#include "rdm/efa_rdm_pke_utils.h"
#include "rdm/efa_rdm_protocol.h"

static struct {
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_rdm_mr stale_source_mr;
} g_mr_abort;

static int efa_test_mr_abort_pkt_type(int pkt)
{
	switch (pkt) {
	case EFA_TEST_MR_ABORT_PKT_MEDIUM_TAGRTM:
		return EFA_RDM_MEDIUM_TAGRTM_PKT;
	case EFA_TEST_MR_ABORT_PKT_LONGCTS_TAGRTM:
		return EFA_RDM_LONGCTS_TAGRTM_PKT;
	case EFA_TEST_MR_ABORT_PKT_PEER_ERROR:
		return EFA_RDM_PEER_ERROR_PKT;
	case EFA_TEST_MR_ABORT_PKT_HANDSHAKE:
		return EFA_RDM_HANDSHAKE_PKT;
	}
	return -1;
}

int efa_test_mr_abort_wire_type(int pkt)
{
	return efa_test_mr_abort_pkt_type(pkt);
}

static bool efa_test_mr_abort_on_list(struct dlist_entry *head,
				      struct dlist_entry *entry)
{
	struct dlist_entry *cur;

	dlist_foreach(head, cur) {
		if (cur == entry)
			return true;
	}
	return false;
}

static size_t efa_test_mr_abort_list_len(struct dlist_entry *head)
{
	struct dlist_entry *cur;
	size_t n = 0;

	dlist_foreach(head, cur)
		n++;
	return n;
}

static bool efa_test_mr_abort_txe_alive(void)
{
	return g_mr_abort.txe &&
	       efa_test_mr_abort_on_list(&g_mr_abort.ep->base_ep.ope_list,
					 &g_mr_abort.txe->ep_entry);
}

/*
 * Own GID with a different QPN, so the real device accepts the AH. The txe is
 * a two-sided tagged send whose source MR generation has moved past the
 * dispatch-time snapshot, i.e. the application closed the MR.
 */
int efa_test_mr_abort_setup(struct fid_ep *ep, struct fid_av *av, int req_pkt,
			    bool handshake_received, void *op_context)
{
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct fi_msg msg = {0};
	fi_addr_t peer_addr;
	struct efa_rdm_ope *txe;
	int ret;

	memset(&g_mr_abort, 0, sizeof(g_mr_abort));
	g_mr_abort.ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	ret = fi_getname(&ep->fid, &raw_addr, &raw_addr_len);
	if (ret)
		return ret;
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	if (fi_av_insert(av, &raw_addr, 1, &peer_addr, 0, NULL) != 1)
		return -FI_EINVAL;

	g_mr_abort.peer = efa_rdm_ep_get_peer_explicit(g_mr_abort.ep, peer_addr);
	if (!g_mr_abort.peer)
		return -FI_EINVAL;
	g_mr_abort.peer->av_entry->shm_fi_addr = FI_ADDR_NOTAVAIL;
	if (handshake_received) {
		g_mr_abort.peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
		g_mr_abort.peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;
	}

	txe = ofi_buf_alloc(g_mr_abort.ep->base_ep.txe_pool);
	if (!txe)
		return -FI_ENOMEM;
	msg.addr = peer_addr;
	efa_rdm_txe_construct(txe, g_mr_abort.ep, g_mr_abort.peer, &msg,
			      ofi_op_tagged, 0, 0);
	txe->state = EFA_RDM_TXE_REQ;
	txe->cq_entry.flags = FI_SEND | FI_TAGGED;
	txe->cq_entry.op_context = op_context;
	txe->total_len = 65536;
	txe->req_pkt_type = efa_test_mr_abort_pkt_type(req_pkt);

	g_mr_abort.stale_source_mr.gen = 1;
	txe->iov_count = 1;
	txe->desc[0] = &g_mr_abort.stale_source_mr;
	txe->desc_gen[0] = 2;

	g_mr_abort.txe = txe;
	return 0;
}

/* The txe state right after the abort path posted its PEER_ERROR_PKT. */
void efa_test_mr_abort_set_peer_error_emitted(void)
{
	struct efa_rdm_ope *txe = g_mr_abort.txe;

	txe->state = EFA_RDM_OPE_ERR;
	txe->internal_flags |= EFA_RDM_OPE_PEER_ABORT_PENDING |
			       EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED;
	txe->iov_count = 0;
}

/* A send WR of the given type that the device accepted, as after sendv. */
void *efa_test_mr_abort_forge_tx_pke(int pkt)
{
	struct efa_rdm_ep *ep = g_mr_abort.ep;
	struct efa_rdm_ope *txe = g_mr_abort.txe;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_base_hdr *base_hdr;

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      ep->efa_tx_bounce_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	if (!pkt_entry)
		return NULL;
	efa_rdm_pke_set_ope(pkt_entry, txe);
	pkt_entry->peer = txe->peer;

	base_hdr = (struct efa_rdm_base_hdr *) pkt_entry->wiredata;
	base_hdr->type = efa_test_mr_abort_pkt_type(pkt);
	base_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	base_hdr->flags = 0;

	ep->efa_outstanding_tx_ops++;
	txe->peer->efa_outstanding_tx_ops++;
	txe->efa_outstanding_tx_ops++;
	return pkt_entry;
}

void efa_test_mr_abort_handle_tx_error(void *pke, int err)
{
	int prov_errno = err == EFA_TEST_MR_ABORT_ERR_RNR ?
		EFA_IO_COMP_STATUS_REMOTE_ERROR_RNR :
		EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY;

	efa_rdm_pke_handle_tx_error((struct efa_rdm_pke *) pke, prov_errno);
}

void efa_test_mr_abort_snapshot(struct efa_test_mr_abort_state *out)
{
	struct efa_rdm_ep *ep = g_mr_abort.ep;
	struct efa_rdm_ope *txe = g_mr_abort.txe;
	struct efa_rdm_peer *peer = g_mr_abort.peer;

	memset(out, 0, sizeof(*out));
	out->ep_outstanding = ep->efa_outstanding_tx_ops;
	out->ep_rnr_queued_pkt_cnt = ep->efa_rnr_queued_pkt_cnt;
	out->peer_handshake_sent = peer->flags & EFA_RDM_PEER_HANDSHAKE_SENT;
	out->peer_handshake_queued = peer->flags & EFA_RDM_PEER_HANDSHAKE_QUEUED;
	out->handshake_queued_peers =
		efa_test_mr_abort_list_len(&ep->handshake_queued_peer_list);

	out->alive = efa_test_mr_abort_txe_alive();
	if (!out->alive)
		return;
	out->state_err = txe->state == EFA_RDM_OPE_ERR;
	out->peer_abort_pending =
		txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING;
	out->peer_error_emitted =
		txe->internal_flags & EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED;
	out->queued_rnr = txe->internal_flags & EFA_RDM_OPE_QUEUED_RNR;
	out->on_ope_queued_list =
		efa_test_mr_abort_on_list(&ep->ope_queued_list,
					  &txe->queued_entry);
	out->queued_pkts = efa_test_mr_abort_list_len(&txe->queued_pkts);
	out->txe_outstanding = txe->efa_outstanding_tx_ops;
}

void efa_test_mr_abort_complete_send(uintptr_t wr_id)
{
	struct efa_rdm_pke *pkt_entry =
		efa_rdm_cq_get_pke_from_wr_id_solicited(wr_id);

	if (pkt_entry)
		efa_rdm_pke_handle_send_completion(pkt_entry);
}

void efa_test_mr_abort_cleanup(void)
{
	struct efa_rdm_ope *txe = g_mr_abort.txe;
	struct efa_rdm_peer *peer = g_mr_abort.peer;
	struct efa_rdm_pke *pkt_entry;
	struct dlist_entry *tmp;

	if (peer && (peer->flags & EFA_RDM_PEER_HANDSHAKE_QUEUED)) {
		dlist_remove(&peer->handshake_queued_entry);
		peer->flags &= ~EFA_RDM_PEER_HANDSHAKE_QUEUED;
	}

	if (efa_test_mr_abort_txe_alive()) {
		if (txe->internal_flags & EFA_RDM_OPE_QUEUED_FLAGS) {
			dlist_remove(&txe->queued_entry);
			txe->internal_flags &= ~EFA_RDM_OPE_QUEUED_FLAGS;
		}
		dlist_foreach_container_safe(&txe->queued_pkts,
					     struct efa_rdm_pke, pkt_entry,
					     entry, tmp)
			efa_rdm_pke_release_tx(pkt_entry);
		txe->internal_flags &= ~EFA_RDM_OPE_PEER_ABORT_PENDING;
		efa_rdm_txe_release(txe);
	}

	memset(&g_mr_abort, 0, sizeof(g_mr_abort));
}
