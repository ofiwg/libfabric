/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_rdm_pke_utils.h"
#include "efa.h"
#include "efa_env.h"
#include "ofi_mem.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_ope.h"
#include "rdm/efa_rdm_pke.h"
#include "rdm/efa_rdm_pke_rtm.h"
#include "rdm/efa_rdm_protocol.h"

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

int efa_test_failed_reorder_msg_releases_rx_pkt(struct fid_ep *ep,
						fi_addr_t peer_addr,
						size_t *to_post_before,
						size_t *to_post_after)
{
	struct efa_rdm_ep *efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct efa_rdm_peer *peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	struct efa_rdm_pke *pke;
	struct efa_rdm_rtm_base_hdr *rtm_hdr;
	uint32_t msg_id;

	if (!peer)
		return -FI_EINVAL;

	pke = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				EFA_RDM_PKE_FROM_EFA_RX_POOL);
	if (!pke)
		return -FI_ENOMEM;
	pke->peer = peer;
	pke->ep = efa_rdm_ep;

	/* a valid (in-window) but unexpected msg_id */
	msg_id = ofi_recvwin_next_exp_id(&peer->robuf) + 1;

	rtm_hdr = (struct efa_rdm_rtm_base_hdr *) pke->wiredata;
	memset(rtm_hdr, 0, sizeof(*rtm_hdr));
	rtm_hdr->type = EFA_RDM_LONGCTS_MSGRTM_PKT;
	rtm_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	rtm_hdr->flags = EFA_RDM_REQ_MSG;
	rtm_hdr->msg_id = msg_id;

	/* efa_rx_pkts_to_post should be incremented by pke_release_rx */
	*to_post_before = efa_rdm_ep->efa_rx_pkts_to_post;
	efa_rdm_pke_handle_rtm_rta_recv(pke);
	*to_post_after = efa_rdm_ep->efa_rx_pkts_to_post;
	return 0;
}

static size_t efa_test_bufpool_free_count(struct ofi_bufpool *pool)
{
	struct slist_entry *item;
	size_t count = 0;

	for (item = pool->free_list.entries.head; item; item = item->next)
		count++;
	return count;
}

int efa_test_failed_reorder_msg_overflow_releases_rx_pkt_and_entry(
	struct fid_ep *ep, fi_addr_t peer_addr, size_t *to_post_before,
	size_t *to_post_after, size_t *overflow_free_before,
	size_t *overflow_free_after)
{
	struct efa_rdm_ep *efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct efa_rdm_peer *peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	struct efa_rdm_pke *pke;
	struct efa_rdm_rtm_base_hdr *rtm_hdr;
	uint32_t msg_id;
	void *warmup;

	if (!peer)
		return -FI_EINVAL;

	/* triggers resize of the empty pool */
	warmup = ofi_buf_alloc(efa_rdm_ep->overflow_pke_pool);
	if (!warmup)
		return -FI_ENOMEM;
	ofi_buf_free(warmup);

	pke = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				EFA_RDM_PKE_FROM_EFA_RX_POOL);
	if (!pke)
		return -FI_ENOMEM;
	pke->peer = peer;
	pke->ep = efa_rdm_ep;

	/* unexpected and out-of-window msg_id -> overflow branch */
	msg_id = ofi_recvwin_next_exp_id(&peer->robuf) + peer->robuf.win_size;

	rtm_hdr = (struct efa_rdm_rtm_base_hdr *) pke->wiredata;
	memset(rtm_hdr, 0, sizeof(*rtm_hdr));
	rtm_hdr->type = EFA_RDM_LONGCTS_MSGRTM_PKT;
	rtm_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	rtm_hdr->flags = EFA_RDM_REQ_MSG;
	rtm_hdr->msg_id = msg_id;

	*to_post_before = efa_rdm_ep->efa_rx_pkts_to_post;
	*overflow_free_before =
		efa_test_bufpool_free_count(efa_rdm_ep->overflow_pke_pool);
	efa_rdm_pke_handle_rtm_rta_recv(pke);
	*to_post_after = efa_rdm_ep->efa_rx_pkts_to_post;
	*overflow_free_after =
		efa_test_bufpool_free_count(efa_rdm_ep->overflow_pke_pool);
	return 0;
}

int efa_test_get_tx_min_credits(void)
{
	return efa_env.tx_min_credits;
}

int efa_test_rtm_pkt_type(enum efa_test_rtm_variant variant)
{
	switch (variant) {
	case EFA_TEST_RTM_EAGER_MSG:
		/* fallthrough */
	case EFA_TEST_RTM_EAGER_MSG_ZERO_HDR:
		return EFA_RDM_EAGER_MSGRTM_PKT;
	case EFA_TEST_RTM_EAGER_TAG:
		return EFA_RDM_EAGER_TAGRTM_PKT;
	case EFA_TEST_RTM_DC_EAGER_MSG:
		return EFA_RDM_DC_EAGER_MSGRTM_PKT;
	case EFA_TEST_RTM_DC_EAGER_TAG:
		return EFA_RDM_DC_EAGER_TAGRTM_PKT;
	case EFA_TEST_RTM_MEDIUM_MSG:
		return EFA_RDM_MEDIUM_MSGRTM_PKT;
	case EFA_TEST_RTM_MEDIUM_TAG:
		return EFA_RDM_MEDIUM_TAGRTM_PKT;
	case EFA_TEST_RTM_DC_MEDIUM_MSG:
		return EFA_RDM_DC_MEDIUM_MSGRTM_PKT;
	case EFA_TEST_RTM_DC_MEDIUM_TAG:
		return EFA_RDM_DC_MEDIUM_TAGRTM_PKT;
	case EFA_TEST_RTM_LONGCTS_MSG:
		return EFA_RDM_LONGCTS_MSGRTM_PKT;
	case EFA_TEST_RTM_LONGCTS_TAG:
		return EFA_RDM_LONGCTS_TAGRTM_PKT;
	case EFA_TEST_RTM_DC_LONGCTS_MSG:
		return EFA_RDM_DC_LONGCTS_MSGRTM_PKT;
	case EFA_TEST_RTM_DC_LONGCTS_TAG:
		return EFA_RDM_DC_LONGCTS_TAGRTM_PKT;
	case EFA_TEST_RTM_LONGREAD_MSG:
		return EFA_RDM_LONGREAD_MSGRTM_PKT;
	case EFA_TEST_RTM_LONGREAD_TAG:
		return EFA_RDM_LONGREAD_TAGRTM_PKT;
	case EFA_TEST_RTM_RUNTREAD_MSG:
		return EFA_RDM_RUNTREAD_MSGRTM_PKT;
	case EFA_TEST_RTM_RUNTREAD_TAG:
		return EFA_RDM_RUNTREAD_TAGRTM_PKT;
	}
	return -1;
}

// Construct a txe populated with sentinels
static struct efa_rdm_ope *efa_test_rtm_alloc_txe(struct efa_rdm_ep *ep,
						  struct fid_av *av,
						  uint32_t op, size_t len,
						  void *buf)
{
	fi_addr_t peer_addr;
	struct efa_rdm_peer *peer;
	struct iovec iov;
	struct fi_msg msg = {0};
	struct efa_rdm_ope *txe;

	if (efa_test_av_insert_self(&ep->base_ep.util_ep.ep_fid, av,
				    &peer_addr) != 1)
		return NULL;
	peer = efa_rdm_ep_get_peer(ep, peer_addr);

	iov.iov_base = buf;
	iov.iov_len = len;
	msg.msg_iov = &iov;
	msg.iov_count = 1;
	msg.addr = peer_addr;

	txe = ofi_buf_alloc(ep->base_ep.ope_pool);
	if (!txe)
		return NULL;
	efa_rdm_txe_construct(txe, ep, peer, &msg, op, 0, 0);

	txe->msg_id = EFA_TEST_RTM_MSG_ID;
	txe->tx_id = EFA_TEST_RTM_SEND_ID;
	txe->tag = EFA_TEST_RTM_TAG;
	return txe;
}

void efa_test_rtm_init_build(struct fid_ep *ep, struct fid_av *av,
			     enum efa_test_rtm_variant variant,
			     struct efa_test_rtm_init_result *out)
{
	struct efa_rdm_ep *efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_base_hdr *base_hdr;
	enum efa_test_rtm_family family = EFA_TEST_RTM_FAMILY(variant);
	uint32_t op =
		EFA_TEST_RTM_IS_TAGGED(variant) ? ofi_op_tagged : ofi_op_msg;

	size_t total_len = (family == EFA_TEST_RTM_FAM_EAGER) ?
				   EFA_TEST_RTM_EAGER_LEN :
				   EFA_TEST_RTM_LONG_LEN;
	// One buffer large enough for the longest message
	static char buf[EFA_TEST_RTM_LONG_LEN];

	memset(out, 0, sizeof(*out));

	txe = efa_test_rtm_alloc_txe(efa_rdm_ep, av, op, total_len, buf);
	if (!txe) {
		out->ret = -FI_ENOMEM;
		return;
	}
	if (family == EFA_TEST_RTM_FAM_RUNTREAD)
		txe->bytes_runt = EFA_TEST_RTM_RUNT_LEN;

	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	if (!pkt_entry) {
		efa_rdm_txe_release(txe);
		out->ret = -FI_ENOMEM;
		return;
	}

	switch (variant) {
	case EFA_TEST_RTM_EAGER_MSG_ZERO_HDR:
		pkt_entry->flags |= EFA_RDM_PKE_HAS_NO_BASE_HDR;
		/* fallthrough */
	case EFA_TEST_RTM_EAGER_MSG:
		out->ret = efa_rdm_pke_init_eager_msgrtm(pkt_entry, txe);
		break;
	case EFA_TEST_RTM_EAGER_TAG:
		out->ret = efa_rdm_pke_init_eager_tagrtm(pkt_entry, txe);
		break;
	case EFA_TEST_RTM_DC_EAGER_MSG:
		out->ret = efa_rdm_pke_init_dc_eager_msgrtm(pkt_entry, txe);
		break;
	case EFA_TEST_RTM_DC_EAGER_TAG:
		out->ret = efa_rdm_pke_init_dc_eager_tagrtm(pkt_entry, txe);
		break;
	case EFA_TEST_RTM_MEDIUM_MSG:
		out->ret = efa_rdm_pke_init_medium_msgrtm(
			pkt_entry, txe, EFA_TEST_RTM_MEDIUM_SEG,
			EFA_TEST_RTM_MEDIUM_DATA);
		break;
	case EFA_TEST_RTM_MEDIUM_TAG:
		out->ret = efa_rdm_pke_init_medium_tagrtm(
			pkt_entry, txe, EFA_TEST_RTM_MEDIUM_SEG,
			EFA_TEST_RTM_MEDIUM_DATA);
		break;
	case EFA_TEST_RTM_DC_MEDIUM_MSG:
		out->ret = efa_rdm_pke_init_dc_medium_msgrtm(
			pkt_entry, txe, EFA_TEST_RTM_MEDIUM_SEG,
			EFA_TEST_RTM_MEDIUM_DATA);
		break;
	case EFA_TEST_RTM_DC_MEDIUM_TAG:
		out->ret = efa_rdm_pke_init_dc_medium_tagrtm(
			pkt_entry, txe, EFA_TEST_RTM_MEDIUM_SEG,
			EFA_TEST_RTM_MEDIUM_DATA);
		break;
	case EFA_TEST_RTM_LONGCTS_MSG:
		out->ret = efa_rdm_pke_init_longcts_msgrtm(pkt_entry, txe);
		break;
	case EFA_TEST_RTM_LONGCTS_TAG:
		out->ret = efa_rdm_pke_init_longcts_tagrtm(pkt_entry, txe);
		break;
	case EFA_TEST_RTM_DC_LONGCTS_MSG:
		out->ret = efa_rdm_pke_init_dc_longcts_msgrtm(pkt_entry, txe);
		break;
	case EFA_TEST_RTM_DC_LONGCTS_TAG:
		out->ret = efa_rdm_pke_init_dc_longcts_tagrtm(pkt_entry, txe);
		break;
	case EFA_TEST_RTM_LONGREAD_MSG:
		out->ret = efa_rdm_pke_init_longread_msgrtm(pkt_entry, txe);
		break;
	case EFA_TEST_RTM_LONGREAD_TAG:
		out->ret = efa_rdm_pke_init_longread_tagrtm(pkt_entry, txe);
		break;
	case EFA_TEST_RTM_RUNTREAD_MSG:
		out->ret = efa_rdm_pke_init_runtread_msgrtm(
			pkt_entry, txe, EFA_TEST_RTM_RUNT_SEG,
			EFA_TEST_RTM_RUNT_DATA);
		break;
	case EFA_TEST_RTM_RUNTREAD_TAG:
		out->ret = efa_rdm_pke_init_runtread_tagrtm(
			pkt_entry, txe, EFA_TEST_RTM_RUNT_SEG,
			EFA_TEST_RTM_RUNT_DATA);
		break;
	}

	if (out->ret)
		goto release;

	out->payload_size = pkt_entry->payload_size;
	out->dc_requested = !!(txe->internal_flags &
			       EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED);

	// The zero-hdr eager packet carries no header so stop before dereferencing wiredata
	if (EFA_TEST_RTM_IS_ZERO_HDR(variant))
		goto release;

	base_hdr = (struct efa_rdm_base_hdr *) pkt_entry->wiredata;
	out->base_type = base_hdr->type;
	out->has_msg_flag = !!(base_hdr->flags & EFA_RDM_REQ_MSG);
	out->has_tagged_flag = !!(base_hdr->flags & EFA_RDM_REQ_TAGGED);
	out->msg_id = efa_rdm_pke_get_rtm_base_hdr(pkt_entry)->msg_id;
	if (EFA_TEST_RTM_IS_TAGGED(variant))
		out->tag = efa_rdm_pke_get_rtm_tag(pkt_entry);

	switch (family) {
	case EFA_TEST_RTM_FAM_EAGER:
		/* Only the DC eager header carries send_id. */
		if (EFA_TEST_RTM_IS_DC(variant))
			out->send_id =
				efa_rdm_pke_get_dc_eager_rtm_base_hdr(pkt_entry)
					->send_id;
		break;
	case EFA_TEST_RTM_FAM_MEDIUM:
		if (EFA_TEST_RTM_IS_DC(variant)) {
			struct efa_rdm_dc_medium_rtm_base_hdr *h =
				efa_rdm_pke_get_dc_medium_rtm_base_hdr(pkt_entry);
			out->msg_length = h->msg_length;
			out->seg_offset = h->seg_offset;
			out->send_id = h->send_id;
		} else {
			struct efa_rdm_medium_rtm_base_hdr *h =
				efa_rdm_pke_get_medium_rtm_base_hdr(pkt_entry);
			out->msg_length = h->msg_length;
			out->seg_offset = h->seg_offset;
		}
		break;
	case EFA_TEST_RTM_FAM_LONGCTS: {
		struct efa_rdm_longcts_rtm_base_hdr *h =
			efa_rdm_pke_get_longcts_rtm_base_hdr(pkt_entry);
		out->msg_length = h->msg_length;
		out->send_id = h->send_id;
		out->credit_request = h->credit_request;
		break;
	}
	case EFA_TEST_RTM_FAM_LONGREAD: {
		struct efa_rdm_longread_rtm_base_hdr *h =
			efa_rdm_pke_get_longread_rtm_base_hdr(pkt_entry);
		out->msg_length = h->msg_length;
		out->send_id = h->send_id;
		out->read_iov_count = h->read_iov_count;
		break;
	}
	case EFA_TEST_RTM_FAM_RUNTREAD: {
		struct efa_rdm_runtread_rtm_base_hdr *h =
			efa_rdm_pke_get_runtread_rtm_base_hdr(pkt_entry);
		out->msg_length = h->msg_length;
		out->send_id = h->send_id;
		out->seg_offset = h->seg_offset;
		out->runt_length = h->runt_length;
		out->read_iov_count = h->read_iov_count;
		break;
	}
	}

release:
	efa_rdm_pke_release_tx(pkt_entry);
	efa_rdm_txe_release(txe);
}
