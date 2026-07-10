/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "ofi_mem.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_pke.h"
#include "efa.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_pke_rtm.h"
#include "efa_gtest_rdm_pke_utils.h"

int efa_test_rtm_read_nack_missing_rxe(struct fid_ep *ep, fi_addr_t peer_addr,
				       int tagged, ssize_t *ret)
{
	struct efa_rdm_ep *efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct efa_rdm_peer *peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	struct efa_rdm_pke *pke;
	struct efa_rdm_rtm_base_hdr *rtm_hdr;

	if (!peer)
		return 0;

	pke = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				EFA_RDM_PKE_FROM_EFA_RX_POOL);
	if (!pke)
		return 0;
	pke->peer = peer;
	pke->ep = efa_rdm_ep;

	rtm_hdr = (struct efa_rdm_rtm_base_hdr *) pke->wiredata;
	memset(rtm_hdr, 0, sizeof(*rtm_hdr));
	rtm_hdr->type = tagged ? EFA_RDM_LONGCTS_TAGRTM_PKT
			       : EFA_RDM_LONGCTS_MSGRTM_PKT;
	rtm_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	rtm_hdr->flags = EFA_RDM_REQ_MSG | EFA_RDM_REQ_READ_NACK;
	if (tagged)
		rtm_hdr->flags |= EFA_RDM_REQ_TAGGED;
	/* since peer has empty rxe_map, any msg_id can trigger lookup error */
	rtm_hdr->msg_id = 0x77u;

	*ret = efa_rdm_pke_proc_rtm_rta(pke, peer);
	return 1;
}

struct efa_rdm_pke *efa_test_pke_build_unexp_chain(struct fid_ep *ep, size_t n)
{
	struct efa_rdm_ep *efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct efa_rdm_pke *head = NULL, *prev = NULL, *pke;
	size_t i;

	for (i = 0; i < n; i++) {
		pke = efa_rdm_pke_alloc(efa_rdm_ep,
					efa_rdm_ep->rx_unexp_pkt_pool,
					EFA_RDM_PKE_FROM_UNEXP_POOL);
		if (!pke)
			return NULL;
		if (prev)
			prev->next = pke;
		else
			head = pke;
		prev = pke;
	}

	return head;
}

/* no header prototype */
void efa_rdm_pke_release_cloned(struct efa_rdm_pke *pkt_entry);

void efa_test_pke_release_cloned(struct efa_rdm_pke *head)
{
	efa_rdm_pke_release_cloned(head);
}

size_t efa_test_ep_unexp_pool_outstanding(struct fid_ep *ep)
{
	struct efa_rdm_ep *efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct ofi_bufpool *pool = efa_rdm_ep->rx_unexp_pkt_pool;
	struct slist_entry *item;
	size_t free_cnt = 0;

	for (item = pool->free_list.entries.head; item; item = item->next)
		free_cnt++;

	return pool->entry_cnt - free_cnt;
}
