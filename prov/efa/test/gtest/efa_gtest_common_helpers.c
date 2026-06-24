/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_av.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_ope.h"
#include "rdm/efa_rdm_peer.h"
#include "efa_gtest_common_helpers.h"

void efa_test_fabricate_addr(struct fid_ep *ep, struct efa_ep_addr *addr)
{
	size_t addr_len = sizeof(*addr);
	static uint8_t gid_suffix = 1;

	memset(addr, 0, addr_len);
	// Get ep's actual address
	fi_getname(&ep->fid, addr, &addr_len);
	// Flip the first bit
	addr->raw[0] ^= 0xFF;
	addr->raw[15] = gid_suffix++;
	addr->qpn = 1;
	addr->qkey = 0x5678;
}

int efa_test_explicit_av_insert(struct fid_ep *ep, struct fid_av *av,
				fi_addr_t *addr)
{
	struct efa_ep_addr raw_addr;

	efa_test_fabricate_addr(ep, &raw_addr);
	return fi_av_insert(av, &raw_addr, 1, addr, 0, NULL);
}

fi_addr_t efa_test_insert_peer_new_gid(struct fid_ep *ep, struct fid_av *av)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_av *efa_av;
	struct efa_ep_addr raw_addr;
	fi_addr_t fi_addr = FI_ADDR_NOTAVAIL;
	int err;

	efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_av = container_of(av, struct efa_av, util_av.av_fid);

	efa_test_fabricate_addr(ep, &raw_addr);

	ofi_genlock_lock(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->srx_lock);
	err = efa_av_insert_one(efa_av, &raw_addr, &fi_addr, 0, NULL, true,
				true);
	ofi_genlock_unlock(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->srx_lock);

	if (err)
		return FI_ADDR_NOTAVAIL;

	return fi_addr;
}

/*
 * Allocate a txe bound to a freshly inserted peer, mirroring the cmocka
 * efa_unit_test_alloc_txe helper. Returns NULL on failure.
 */
static struct efa_rdm_ope *efa_test_alloc_txe(struct fid_ep *ep,
					      struct fid_av *av,
					      struct efa_rdm_peer **peer_out)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct efa_ep_addr raw_addr;
	fi_addr_t peer_addr = 0;
	struct fi_msg msg = {0};

	efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	efa_test_fabricate_addr(ep, &raw_addr);
	raw_addr.qpn = 0;
	raw_addr.qkey = 0x1234;
	if (fi_av_insert(av, &raw_addr, 1, &peer_addr, 0, NULL) != 1)
		return NULL;

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);

	txe = ofi_buf_alloc(efa_rdm_ep->ope_pool);
	if (!txe)
		return NULL;

	efa_rdm_txe_construct(txe, efa_rdm_ep, peer, &msg, ofi_op_msg, 0, 0);

	if (peer_out)
		*peer_out = peer;
	return txe;
}

int efa_test_ope_process_queued_no_flag(struct fid_ep *ep, struct fid_av *av)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_domain *domain;
	struct efa_rdm_ope *txe;
	int ret = 0;

	efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	domain = efa_rdm_ep_rdm_domain(efa_rdm_ep);

	txe = efa_test_alloc_txe(ep, av, NULL);
	if (!txe)
		return -1;

	/* No EFA_RDM_OPE_QUEUED_* flag set: must be a no-op returning 0. */
	txe->internal_flags &= ~EFA_RDM_OPE_QUEUED_FLAGS;

	if (efa_rdm_ope_process_queued_ope(txe) != 0)
		ret = -2;
	if (txe->internal_flags & EFA_RDM_OPE_QUEUED_FLAGS)
		ret = -3;
	if (!dlist_empty(&domain->ope_queued_list))
		ret = -4;

	efa_rdm_txe_release(txe);
	return ret;
}

int efa_test_ope_process_queued_before_handshake_eagain(struct fid_ep *ep,
							struct fid_av *av)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_domain *domain;
	struct efa_rdm_peer *peer = NULL;
	struct efa_rdm_ope *txe;
	size_t cnt_before;
	int ret = 0;

	efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	domain = efa_rdm_ep_rdm_domain(efa_rdm_ep);

	txe = efa_test_alloc_txe(ep, av, &peer);
	if (!txe || !peer)
		return -1;

	/*
	 * Peer must NOT have received a handshake, so the before-handshake
	 * repost short-circuits with -FI_EAGAIN before any device call.
	 */
	peer->flags &= ~EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	txe->internal_flags &= ~EFA_RDM_OPE_QUEUED_FLAGS;
	txe->internal_flags |= EFA_RDM_OPE_QUEUED_BEFORE_HANDSHAKE;
	dlist_insert_tail(&txe->queued_entry, &domain->ope_queued_list);
	cnt_before = efa_rdm_ep->ope_queued_before_handshake_cnt++;

	/*
	 * The function must derive BEFORE_HANDSHAKE from internal_flags and
	 * dispatch to the before-handshake repost, which returns -FI_EAGAIN.
	 */
	if (efa_rdm_ope_process_queued_ope(txe) != -FI_EAGAIN)
		ret = -2;
	/* On EAGAIN the flag stays set ... */
	if (!(txe->internal_flags & EFA_RDM_OPE_QUEUED_BEFORE_HANDSHAKE))
		ret = -3;
	/* ... the ope stays queued ... */
	if (dlist_empty(&domain->ope_queued_list))
		ret = -4;
	/* ... and the before-handshake counter is not decremented. */
	if (efa_rdm_ep->ope_queued_before_handshake_cnt != cnt_before + 1)
		ret = -5;

	/* Restore state for a clean teardown. */
	txe->internal_flags &= ~EFA_RDM_OPE_QUEUED_BEFORE_HANDSHAKE;
	dlist_remove(&txe->queued_entry);
	efa_rdm_ep->ope_queued_before_handshake_cnt--;
	efa_rdm_txe_release(txe);
	return ret;
}
