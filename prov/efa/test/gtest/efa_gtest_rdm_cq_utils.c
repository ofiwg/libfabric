/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_av.h"
#include "efa_base_ep.h"
#include "efa_cq.h"
#include "rdm/efa_rdm_av.h"
#include "rdm/efa_rdm_cq.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_peer.h"
#include "rdm/efa_rdm_pke.h"
#include "rdm/efa_rdm_pke_nonreq.h"
#include "rdm/efa_rdm_protocol.h"
#include "efa_gtest_rdm_cq_utils.h"

static struct efa_rdm_ep *efa_test_rdm_cq_ep(struct fid_ep *ep)
{
	return container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
}

static struct efa_av *efa_test_rdm_cq_av(struct fid_av *av)
{
	return container_of(av, struct efa_av, util_av.av_fid);
}

int efa_test_rdm_cq_race_setup(struct fid_ep *ep, struct fid_av *av,
			       struct fid_cq *cq, int with_connid,
			       struct efa_test_rdm_cq_race_ctx *ctx)
{
	struct efa_rdm_ep *efa_rdm_ep = efa_test_rdm_cq_ep(ep);
	struct efa_rdm_handshake_hdr *handshake_hdr;
	struct efa_rdm_handshake_opt_connid_hdr *connid_hdr;
	struct efa_rdm_handshake_opt_device_version_hdr *device_version_hdr;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_rdm_pke *pke;
	int nex = (EFA_RDM_NUM_EXTRA_FEATURE_OR_REQUEST - 1) / 64 + 1;
	size_t offset;
	int ret;

	memset(ctx, 0, sizeof(*ctx));
	ctx->ep = ep;
	ctx->av = av;
	ctx->cq = cq;
	ctx->racing_addr = FI_ADDR_NOTAVAIL;

	ret = fi_getname(&ep->fid, &raw_addr, &raw_addr_len);
	if (ret)
		return ret;

	memcpy(ctx->src_gid.raw, raw_addr.raw, sizeof(ctx->src_gid.raw));
	ctx->qpn = raw_addr.qpn + 1;
	ctx->qkey = raw_addr.qkey;
	ctx->ahn = efa_rdm_ep->self_ah->ahn;

	pke = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				EFA_RDM_PKE_FROM_EFA_RX_POOL);
	if (!pke)
		return -FI_ENOMEM;

	handshake_hdr = (struct efa_rdm_handshake_hdr *) pke->wiredata;
	memset(handshake_hdr, 0, sizeof(*handshake_hdr));
	handshake_hdr->type = EFA_RDM_HANDSHAKE_PKT;
	handshake_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	/* The handshake handler reads the device version unconditionally */
	handshake_hdr->flags = EFA_RDM_HANDSHAKE_DEVICE_VERSION_HDR;
	if (with_connid)
		handshake_hdr->flags |= EFA_RDM_PKT_CONNID_HDR;
	handshake_hdr->nextra_p3 = nex + 3;
	handshake_hdr->extra_info[0] = EFA_TEST_RDM_CQ_EXTRA_INFO_SENTINEL;
	ctx->nextra_p3 = handshake_hdr->nextra_p3;

	offset = sizeof(*handshake_hdr) + nex * sizeof(uint64_t);
	if (with_connid) {
		connid_hdr = (struct efa_rdm_handshake_opt_connid_hdr *)
				     (pke->wiredata + offset);
		connid_hdr->connid = ctx->qkey;
		offset += sizeof(*connid_hdr);
	}
	device_version_hdr = (struct efa_rdm_handshake_opt_device_version_hdr *)
				     (pke->wiredata + offset);
	device_version_hdr->device_version = EFA_TEST_RDM_CQ_DEVICE_VERSION;
	offset += sizeof(*device_version_hdr);

	pke->pkt_size = offset;
	ctx->pke = pke;
	ctx->pkt_size = offset;
	ctx->wr_id = (uint64_t) pke | (uint64_t) pke->gen;

	/*
	 * RX packet entries are normally allocated and posted by the progress
	 * engine, which grows the pool on its first run and sets
	 * efa_rx_pkts_posted to the pool size. Mimic that for the entry
	 * fabricated above, so the recv completion's accounting stays balanced.
	 */
	efa_rdm_ep->efa_rx_pkts_posted =
		efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep);

	return 0;
}

fi_addr_t efa_test_rdm_cq_race_insert(struct efa_test_rdm_cq_race_ctx *ctx)
{
	struct efa_rdm_ep *efa_rdm_ep = efa_test_rdm_cq_ep(ctx->ep);
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_rdm_peer *peer;
	fi_addr_t fi_addr = FI_ADDR_NOTAVAIL;

	if (fi_getname(&ctx->ep->fid, &raw_addr, &raw_addr_len))
		return FI_ADDR_NOTAVAIL;
	raw_addr.qpn = ctx->qpn;
	raw_addr.qkey = ctx->qkey;

	if (fi_av_insert(ctx->av, &raw_addr, 1, &fi_addr, 0, NULL) != 1)
		return FI_ADDR_NOTAVAIL;

	/*
	 * Suppress the handshake the recv path would post to a peer it has not
	 * handshaken with, which would leave an outstanding TX op behind.
	 */
	peer = efa_rdm_ep_get_peer_explicit(efa_rdm_ep, fi_addr);
	if (!peer)
		return FI_ADDR_NOTAVAIL;
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_SENT;

	ctx->racing_addr = fi_addr;
	return fi_addr;
}

int efa_test_rdm_cq_race_poll(struct efa_test_rdm_cq_race_ctx *ctx)
{
	struct efa_rdm_cq *efa_rdm_cq =
		container_of(ctx->cq, struct efa_rdm_cq, efa_cq.util_cq.cq_fid);
	int ret;

	ofi_genlock_lock(&efa_rdm_cq->efa_cq.util_cq.ep_list_lock);
	ret = efa_rdm_cq_poll_ibv_cq(1, &efa_rdm_cq->efa_cq.ibv_cq);
	ofi_genlock_unlock(&efa_rdm_cq->efa_cq.util_cq.ep_list_lock);

	return ret;
}

int efa_test_rdm_cq_reports_sgid(struct efa_ibv_cq *ibv_cq)
{
#if HAVE_EFADV_CQ_EX
	return ibv_cq->ibv_cq_ex_type == EFADV_CQ;
#else
	return 0;
#endif
}

void efa_test_rdm_cq_peer_state(struct fid_ep *ep, fi_addr_t addr,
				struct efa_test_rdm_cq_peer_state *out)
{
	struct efa_rdm_ep *efa_rdm_ep = efa_test_rdm_cq_ep(ep);
	struct efa_rdm_peer *peer;

	memset(out, 0, sizeof(*out));
	out->explicit_fi_addr = FI_ADDR_NOTAVAIL;
	out->implicit_fi_addr = FI_ADDR_NOTAVAIL;

	peer = efa_rdm_ep_peer_map_lookup(efa_rdm_ep->fi_addr_to_peer_map, addr);
	if (!peer)
		return;

	out->peer_exists = 1;
	out->handshake_received =
		!!(peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED);
	out->nextra_p3 = peer->nextra_p3;
	out->extra_info0 = peer->extra_info[0];
	out->device_version = peer->device_version;
	out->explicit_fi_addr = peer->av_entry->efa_av_entry.fi_addr;
	out->implicit_fi_addr = peer->av_entry->implicit_fi_addr;
}

size_t efa_test_rdm_cq_implicit_av_count(struct fid_av *av)
{
	struct efa_rdm_av *rdm_av =
		(struct efa_rdm_av *) efa_test_rdm_cq_av(av);

	/* implicit_av_size is the configured capacity, not the entry count */
	return HASH_CNT(hh, rdm_av->util_av_implicit.hash);
}
