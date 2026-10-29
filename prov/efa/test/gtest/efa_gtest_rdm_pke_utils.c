/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_rdm_pke_utils.h"
#include "efa.h"
#include "efa_env.h"
#include "ofi_mem.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_mr.h"
#include "rdm/efa_rdm_ope.h"
#include "rdm/efa_rdm_pke.h"
#include "rdm/efa_rdm_pke_cmd.h"
#include "rdm/efa_rdm_pke_req.h"
#include "rdm/efa_rdm_pke_rta.h"
#include "rdm/efa_rdm_pke_rtm.h"
#include "rdm/efa_rdm_pke_utils.h"
#include "rdm/efa_rdm_protocol.h"
#include "rdm/efa_rdm_srx.h"
#include "rdm/protocols/efa_rdm_proto_eager.h"
#include "rdm/protocols/efa_rdm_proto_longcts.h"
#include "rdm/protocols/efa_rdm_proto_medium.h"
#include "rdm/protocols/efa_rdm_proto_runtread.h"

int efa_test_rtm_read_nack_missing_rxe(struct fid_ep *ep, fi_addr_t peer_addr,
				       int tagged, ssize_t *ret)
{
	struct efa_rdm_ep *efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct efa_rdm_peer *peer = efa_rdm_ep_get_peer_explicit(efa_rdm_ep, peer_addr);
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

	pke->proto = &efa_rdm_proto_longcts;
	pke->handle_pke = efa_rdm_pke_proc_rtm_after_robuf;
	*ret = pke->handle_pke(pke);
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
	struct efa_rdm_peer *peer = efa_rdm_ep_get_peer_explicit(efa_rdm_ep, peer_addr);
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
	efa_rdm_pke_handle_rtm_recv(pke, &efa_rdm_proto_longcts);
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
	struct efa_rdm_peer *peer = efa_rdm_ep_get_peer_explicit(efa_rdm_ep, peer_addr);
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
	efa_rdm_pke_handle_rtm_recv(pke, &efa_rdm_proto_longcts);
	*to_post_after = efa_rdm_ep->efa_rx_pkts_to_post;
	*overflow_free_after =
		efa_test_bufpool_free_count(efa_rdm_ep->overflow_pke_pool);
	return 0;
}

static struct efa_rdm_pke *
efa_test_reorder_pke(struct efa_rdm_ep *ep, struct efa_rdm_peer *peer,
		     uint32_t msg_id)
{
	struct efa_rdm_eager_msgrtm_hdr hdr = {0};
	struct efa_rdm_req_opt_connid_hdr connid = {0};
	struct efa_rdm_pke *pke =
		efa_rdm_pke_alloc(ep, ep->efa_rx_pkt_pool,
				  EFA_RDM_PKE_FROM_EFA_RX_POOL);

	if (!pke)
		return NULL;

	ep->efa_rx_pkts_posted = efa_base_ep_get_rx_pool_size(&ep->base_ep);
	pke->peer = peer;
	hdr.hdr.version = EFA_RDM_PROTOCOL_VERSION;
	hdr.hdr.type = EFA_RDM_EAGER_MSGRTM_PKT;
	hdr.hdr.flags = EFA_RDM_PKT_CONNID_HDR | EFA_RDM_REQ_MSG;
	hdr.hdr.msg_id = msg_id;
	connid.connid = 0x1234;
	memcpy(pke->wiredata, &hdr, sizeof hdr);
	memcpy(pke->wiredata + sizeof hdr, &connid, sizeof connid);
	pke->pkt_size = sizeof hdr + sizeof connid;
	return pke;
}

int efa_test_reordered_packet_retains_callback(
	struct fid_ep *ep_fid, fi_addr_t peer_addr,
	struct efa_test_reorder_callback_result *out)
{
	struct efa_rdm_ep *ep = container_of(
		ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct efa_rdm_peer *peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);

	memset(out, 0, sizeof *out);
	if (!peer)
		return -FI_EINVAL;
	struct efa_rdm_pke *pke = efa_test_reorder_pke(ep, peer, 1);

	if (!pke)
		return -FI_ENOMEM;

	int saved_rx_copy_ooo = efa_env.rx_copy_ooo;

	efa_env.rx_copy_ooo = 0;
	efa_rdm_pke_proc_received(pke);
	efa_env.rx_copy_ooo = saved_rx_copy_ooo;

	pke = *ofi_recvwin_get_msg(&peer->robuf, 1);
	if (!pke)
		return -FI_EINVAL;
	out->callback_set = pke->handle_pke != NULL;
	out->protocol_correct = pke->proto == &efa_rdm_proto_eager;
	*ofi_recvwin_get_msg(&peer->robuf, 1) = NULL;
	efa_rdm_pke_release_rx(pke);
	return 0;
}

int efa_test_reordered_rta_retains_callback(
	struct fid_ep *ep_fid, fi_addr_t peer_addr,
	struct efa_test_reorder_callback_result *out)
{
	struct efa_rdm_ep *ep = container_of(
		ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct efa_rdm_peer *peer =
		efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	const struct {
		int pkt_type;
		efa_rdm_pke_callback callback;
	} cases[] = {
		{EFA_RDM_WRITE_RTA_PKT, efa_rdm_pke_proc_write_rta},
		{EFA_RDM_DC_WRITE_RTA_PKT, efa_rdm_pke_proc_dc_write_rta},
		{EFA_RDM_FETCH_RTA_PKT, efa_rdm_pke_proc_fetch_rta},
		{EFA_RDM_COMPARE_RTA_PKT, efa_rdm_pke_proc_compare_rta},
	};
	struct efa_rdm_pke *pke;
	struct efa_rdm_rta_hdr hdr = {0};
	uint32_t exp_msg_id;
	size_t pending_wcnt;
	size_t i;
	int saved_rx_copy_ooo;

	memset(out, 0, sizeof *out);
	if (!peer)
		return -FI_EINVAL;

	hdr.version = EFA_RDM_PROTOCOL_VERSION;
	hdr.flags = EFA_RDM_REQ_ATOMIC;
	saved_rx_copy_ooo = efa_env.rx_copy_ooo;
	efa_env.rx_copy_ooo = 0;
	out->callback_set = 1;
	out->protocol_correct = 1;

	for (i = 0; i < ARRAY_SIZE(cases); i++) {
		pke = efa_rdm_pke_alloc(ep, ep->efa_rx_pkt_pool,
					EFA_RDM_PKE_FROM_EFA_RX_POOL);
		if (!pke) {
			efa_env.rx_copy_ooo = saved_rx_copy_ooo;
			return -FI_ENOMEM;
		}

		exp_msg_id = ofi_recvwin_next_exp_id(&peer->robuf);
		pending_wcnt = peer->robuf.pending->wcnt;
		pke->peer = peer;
		hdr.type = cases[i].pkt_type;
		hdr.msg_id = exp_msg_id + 1;
		memcpy(pke->wiredata, &hdr, sizeof hdr);
		pke->pkt_size = sizeof hdr;

		efa_rdm_pke_proc_received(pke);
		pke = *ofi_recvwin_get_msg(&peer->robuf, hdr.msg_id);
		if (!pke) {
			peer->robuf.pending->wcnt = pending_wcnt;
			efa_env.rx_copy_ooo = saved_rx_copy_ooo;
			return -FI_EINVAL;
		}

		out->callback_set &=
			pke->handle_pke == cases[i].callback;
		out->protocol_correct &=
			pke->proto == NULL &&
			ofi_recvwin_next_exp_id(&peer->robuf) == exp_msg_id;
		*ofi_recvwin_get_msg(&peer->robuf, hdr.msg_id) = NULL;
		peer->robuf.pending->wcnt = pending_wcnt;
		efa_rdm_pke_release_rx(pke);
	}

	efa_env.rx_copy_ooo = saved_rx_copy_ooo;
	return 0;
}

struct efa_test_order_callback_ctx {
	struct efa_test_order_result *out;
	uint32_t first_msg_id;
	uint32_t fail_msg_id;
};

static struct efa_test_order_callback_ctx efa_test_order_ctx;

static ssize_t efa_test_order_callback(struct efa_rdm_pke *pke){
	struct efa_test_order_result *out = efa_test_order_ctx.out;
	uint32_t msg_id = efa_rdm_pke_get_rtm_msg_id(pke);
	size_t index = msg_id - efa_test_order_ctx.first_msg_id;

	assert(index < ARRAY_SIZE(out->log) &&
	       out->log_len < ARRAY_SIZE(out->log));
	out->log[out->log_len++] = msg_id;
	out->callback_count[index]++;
	efa_rdm_pke_release_rx(pke);
	return msg_id == efa_test_order_ctx.fail_msg_id ? -FI_EIO : 0;
}

static struct efa_rdm_pke *efa_test_order_pke(struct efa_rdm_ep *ep,
					      struct efa_rdm_peer *peer,
					      uint32_t msg_id, int atomic){
	struct efa_rdm_pke *pke = efa_rdm_pke_alloc(
		ep, ep->rx_unexp_pkt_pool, EFA_RDM_PKE_FROM_UNEXP_POOL);
	struct efa_rdm_rtm_base_hdr *hdr;

	if (!pke)
		return NULL;

	pke->peer = peer;
	hdr = (struct efa_rdm_rtm_base_hdr *) pke->wiredata;
	memset(hdr, 0, atomic ? sizeof(struct efa_rdm_rta_hdr) : sizeof(*hdr));
	hdr->version = EFA_RDM_PROTOCOL_VERSION;
	hdr->type = atomic ? EFA_RDM_WRITE_RTA_PKT : EFA_RDM_EAGER_MSGRTM_PKT;
	hdr->flags = atomic ? EFA_RDM_REQ_ATOMIC : EFA_RDM_REQ_MSG;
	hdr->msg_id = msg_id;
	pke->pkt_size = atomic ? sizeof(struct efa_rdm_rta_hdr) : sizeof(*hdr);

	return pke;
}

static int efa_test_order_take_queued(struct efa_rdm_pke **pke, int ret){
	if (ret != 1)
		return ret ? ret : -FI_EINVAL;
	*pke = NULL;
	return 0;
}

static int efa_test_order_cleanup(struct efa_rdm_peer *peer,
				  struct efa_test_order_result *out,
				  struct efa_rdm_pke **pkes,
				  struct efa_rdm_pke ***slots, size_t count){
	int drained = ofi_recvwin_is_empty((&peer->robuf)) &&
		      dlist_empty(&peer->overflow_pke_list);
	size_t i;

	for (i = 0; i < count; i++) {
		if (!slots[i])
			drained = 0;
		else if (*slots[i]) {
			drained = 0;
			if (!out->callback_count[i])
				efa_rdm_pke_release_rx(*slots[i]);
			*slots[i] = NULL;
		}
		if (pkes[i])
			efa_rdm_pke_release_rx(pkes[i]);
	}
	return drained;
}

static void efa_test_order_receive(struct efa_rdm_pke **pke){
	struct efa_rdm_pke *tmp = *pke;

	*pke = NULL;
	efa_rdm_pke_handle_rta_recv(tmp, efa_test_order_callback);
}

int efa_test_ordered_receive(struct fid_ep *ep_fid, fi_addr_t peer_addr,
			     enum efa_test_order_scenario scenario,
			     struct efa_test_order_result *out){
	struct efa_test_order_case {
		uint32_t first_msg_id;
		uint32_t exp_msg_id_start;
		uint32_t fail_msg_id;
		size_t count;
		bool copy_ooo;
		bool atomic[3];
	};
	static const struct efa_test_order_case cases[] = {
		[EFA_TEST_ORDER_MIXED_RTM_RTA] =
			{50, 50, UINT32_MAX, 3, false, {true, false, true}},
		[EFA_TEST_ORDER_DRAIN_ERROR] =
			{1, 1, 1, 2, false, {true, false}},
		[EFA_TEST_ORDER_IN_ORDER_ERROR] =
			{10, 10, 10, 2, false, {true, true}},
		[EFA_TEST_ORDER_SINGLE_DRAIN] =
			{1, 0, UINT32_MAX, 1, true, {false}},
	};
	const struct efa_test_order_case *test;
	struct efa_rdm_ep *ep =
		container_of(ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct efa_rdm_peer *peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	struct efa_rdm_pke *pkes[3] = {NULL};
	struct efa_rdm_pke **slots[3] = {NULL};
	int saved_rx_copy_ooo = efa_env.rx_copy_ooo;
	int ret = 0;
	size_t i;

	memset(out, 0, sizeof *out);
	if (!peer || (unsigned int) scenario >= ARRAY_SIZE(cases))
		return -FI_EINVAL;
	test = &cases[scenario];
	efa_test_order_ctx = (struct efa_test_order_callback_ctx) {
		out, test->first_msg_id, test->fail_msg_id};

	peer->robuf.exp_msg_id = test->exp_msg_id_start;
	for (i = 0; i < test->count; i++) {
		uint32_t msg_id = test->first_msg_id + i;

		pkes[i] = test->copy_ooo ?
				  efa_test_reorder_pke(ep, peer, msg_id) :
				  efa_test_order_pke(ep, peer, msg_id,
						     test->atomic[i]);
		slots[i] = ofi_recvwin_get_msg(&peer->robuf, msg_id);
		if (!pkes[i]) {
			ret = -FI_ENOMEM;
			goto out;
		}
	}

	efa_env.rx_copy_ooo = test->copy_ooo;
	switch (scenario) {
	case EFA_TEST_ORDER_MIXED_RTM_RTA:
		efa_test_order_receive(&pkes[2]);
		pkes[1]->handle_pke = efa_test_order_callback;
		ret = efa_test_order_take_queued(
			&pkes[1], efa_rdm_peer_reorder_msg(peer, ep, pkes[1]));
		break;
	case EFA_TEST_ORDER_DRAIN_ERROR:
		for (i = 0; i < test->count && !ret; i++) {
			pkes[i]->handle_pke = efa_test_order_callback;
			ret = efa_test_order_take_queued(
				&pkes[i],
				efa_rdm_peer_recvwin_queue_or_append_pke(
					pkes[i], test->first_msg_id + i,
					&peer->robuf));
		}
		break;
	case EFA_TEST_ORDER_IN_ORDER_ERROR:
		efa_test_order_receive(&pkes[1]);
		break;
	case EFA_TEST_ORDER_SINGLE_DRAIN:
		pkes[0]->handle_pke = efa_test_order_callback;
		ret = efa_test_order_take_queued(
			&pkes[0], efa_rdm_peer_reorder_msg(peer, ep, pkes[0]));
		break;
	}
	if (ret)
		goto out;
	out->pre_log_len = out->log_len;
	out->exp_msg_id_before = ofi_recvwin_next_exp_id(&peer->robuf);
	if (scenario == EFA_TEST_ORDER_SINGLE_DRAIN)
		ofi_recvwin_slide(&peer->robuf);
	if (scenario == EFA_TEST_ORDER_SINGLE_DRAIN ||
	    scenario == EFA_TEST_ORDER_DRAIN_ERROR)
		efa_rdm_peer_proc_pending_items_in_robuf(peer, ep);
	else
		efa_test_order_receive(&pkes[0]);
out:
	out->exp_msg_id = ofi_recvwin_next_exp_id(&peer->robuf);
	efa_test_order_ctx.out = NULL;
	efa_env.rx_copy_ooo = saved_rx_copy_ooo;
	out->drained =
		efa_test_order_cleanup(peer, out, pkes, slots, test->count);
	return ret;
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
	peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);

	iov.iov_base = buf;
	iov.iov_len = len;
	msg.msg_iov = &iov;
	msg.iov_count = 1;
	msg.addr = peer_addr;

	txe = ofi_buf_alloc(ep->base_ep.txe_pool);
	if (!txe)
		return NULL;
	efa_rdm_txe_construct(txe, ep, peer, &msg, op, 0, 0);

	txe->msg_id = EFA_TEST_RTM_MSG_ID;
	txe->tx_id = EFA_TEST_RTM_SEND_ID;
	txe->tag = EFA_TEST_RTM_TAG;
	return txe;
}

/*
 * Build an eager RTM packet the way the eager protocol does.
 *
 * The eager protocol moved to the refactored code path
 * (efa_rdm_proto_eager_construct_tx_pkes), which allocates its own packet
 * entry and so cannot stamp a caller-supplied one. This mirrors the header and
 * payload it writes, so the assertions below still describe the eager wire
 * format.
 */
static ssize_t efa_test_rtm_init_eager(struct efa_rdm_pke *pkt_entry,
				       struct efa_rdm_ope *txe,
				       enum efa_test_rtm_variant variant)
{
	int pkt_type = efa_test_rtm_pkt_type(variant);
	ssize_t ret;

	if (pkt_entry->flags & EFA_RDM_PKE_HAS_NO_BASE_HDR) {
		/* Headerless: raw payload, no protocol header at all. */
		efa_rdm_pke_set_ope(pkt_entry, txe);
		pkt_entry->peer = txe->peer;
		return efa_rdm_pke_init_payload_from_ope(pkt_entry, txe, 0, 0,
							 txe->total_len);
	}

	if (EFA_TEST_RTM_IS_DC(variant))
		txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	ret = efa_rdm_pke_init_rtm_with_payload(pkt_entry, pkt_type, txe, 0, -1);
	if (ret)
		return ret;

	if (EFA_TEST_RTM_IS_TAGGED(variant)) {
		efa_rdm_pke_get_base_hdr(pkt_entry)->flags |= EFA_RDM_REQ_TAGGED;
		efa_rdm_pke_set_rtm_tag(pkt_entry, txe->tag);
	}

	/* Only the DC eager header carries send_id. */
	if (EFA_TEST_RTM_IS_DC(variant))
		efa_rdm_pke_get_dc_eager_rtm_base_hdr(pkt_entry)->send_id =
			txe->tx_id;

	return 0;
}

/*
 * Build one segment of a medium RTM message the way the medium protocol does.
 *
 * Like eager, the medium protocol moved to the refactored code path
 * (efa_rdm_proto_medium_construct_tx_pkes), which allocates its own packet
 * entries and so cannot stamp a caller-supplied one. This mirrors the header and
 * payload it writes for a single segment, so the assertions below still describe
 * the medium wire format.
 */
static ssize_t efa_test_rtm_init_medium(struct efa_rdm_pke *pkt_entry,
					struct efa_rdm_ope *txe,
					enum efa_test_rtm_variant variant,
					size_t segment_offset, size_t data_size)
{
	int pkt_type = efa_test_rtm_pkt_type(variant);
	struct efa_rdm_rtm_base_hdr *rtm_hdr;

	if (EFA_TEST_RTM_IS_DC(variant))
		txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	efa_rdm_pke_init_req_hdr_common(pkt_entry, pkt_type, txe);

	/* The DC and non-DC medium headers share this prefix. */
	rtm_hdr = efa_rdm_pke_get_rtm_base_hdr(pkt_entry);
	rtm_hdr->flags |= EFA_RDM_REQ_MSG;
	rtm_hdr->msg_id = txe->msg_id;

	if (EFA_TEST_RTM_IS_TAGGED(variant)) {
		rtm_hdr->flags |= EFA_RDM_REQ_TAGGED;
		efa_rdm_pke_set_rtm_tag(pkt_entry, txe->tag);
	}

	if (EFA_TEST_RTM_IS_DC(variant)) {
		struct efa_rdm_dc_medium_rtm_base_hdr *h =
			efa_rdm_pke_get_dc_medium_rtm_base_hdr(pkt_entry);

		h->send_id = txe->tx_id;
		h->msg_length = txe->total_len;
		h->seg_offset = segment_offset;
	} else {
		struct efa_rdm_medium_rtm_base_hdr *h =
			efa_rdm_pke_get_medium_rtm_base_hdr(pkt_entry);

		h->msg_length = txe->total_len;
		h->seg_offset = segment_offset;
	}

	return efa_rdm_pke_init_payload_from_ope(
		pkt_entry, txe, efa_rdm_pke_get_req_hdr_size(pkt_entry),
		segment_offset, data_size);
}

/*
 * Build one segment of a runt read RTM message the way the runt read protocol
 * does.
 *
 * Like eager and medium, the runt read protocol moved to the refactored code
 * path (efa_rdm_proto_runtread_construct_tx_pkes), which allocates its own
 * packet entries and so cannot stamp a caller-supplied one. This mirrors the
 * header, read iov array and payload it writes for a single segment, so the
 * assertions below still describe the runt read wire format.
 */
static ssize_t efa_test_rtm_init_runtread(struct efa_rdm_pke *pkt_entry,
					  struct efa_rdm_ope *txe,
					  enum efa_test_rtm_variant variant,
					  size_t segment_offset, size_t data_size)
{
	int pkt_type = efa_test_rtm_pkt_type(variant);
	struct efa_rdm_runtread_rtm_base_hdr *rtm_hdr;
	struct fi_rma_iov *read_iov;
	size_t hdr_size;
	ssize_t ret;

	efa_rdm_pke_init_req_hdr_common(pkt_entry, pkt_type, txe);

	rtm_hdr = efa_rdm_pke_get_runtread_rtm_base_hdr(pkt_entry);
	rtm_hdr->hdr.flags |= EFA_RDM_REQ_MSG;
	rtm_hdr->hdr.msg_id = txe->msg_id;
	rtm_hdr->msg_length = txe->total_len;
	rtm_hdr->send_id = txe->tx_id;
	rtm_hdr->seg_offset = segment_offset;
	rtm_hdr->runt_length = txe->bytes_runt;
	rtm_hdr->read_iov_count = txe->iov_count;

	if (EFA_TEST_RTM_IS_TAGGED(variant)) {
		rtm_hdr->hdr.flags |= EFA_RDM_REQ_TAGGED;
		efa_rdm_pke_set_rtm_tag(pkt_entry, txe->tag);
	}

	hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);
	read_iov = (struct fi_rma_iov *) (pkt_entry->wiredata + hdr_size);
	ret = efa_rdm_txe_prepare_to_be_read(txe, read_iov);
	if (ret)
		return ret;

	return efa_rdm_pke_init_payload_from_ope(
		pkt_entry, txe,
		hdr_size + txe->iov_count * sizeof(struct fi_rma_iov),
		segment_offset, data_size);
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
	case EFA_TEST_RTM_EAGER_TAG:
	case EFA_TEST_RTM_DC_EAGER_MSG:
	case EFA_TEST_RTM_DC_EAGER_TAG:
		out->ret = efa_test_rtm_init_eager(pkt_entry, txe, variant);
		break;
	case EFA_TEST_RTM_MEDIUM_MSG:
	case EFA_TEST_RTM_MEDIUM_TAG:
	case EFA_TEST_RTM_DC_MEDIUM_MSG:
	case EFA_TEST_RTM_DC_MEDIUM_TAG:
		out->ret = efa_test_rtm_init_medium(pkt_entry, txe, variant,
						   EFA_TEST_RTM_MEDIUM_SEG,
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
	case EFA_TEST_RTM_RUNTREAD_TAG:
		out->ret = efa_test_rtm_init_runtread(
			pkt_entry, txe, variant, EFA_TEST_RTM_RUNT_SEG,
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

static int efa_test_txe_on_ope_list(struct efa_rdm_ep *ep,
				    struct efa_rdm_ope *txe)
{
	struct dlist_entry *item;

	dlist_foreach(&ep->base_ep.ope_list, item) {
		if (container_of(item, struct efa_rdm_ope, ep_entry) == txe)
			return 1;
	}
	return 0;
}

void efa_test_rtm_sent_build(struct fid_ep *ep, struct fid_av *av,
			     enum efa_test_rtm_variant variant,
			     enum efa_test_rtm_sent_op op, size_t payload_size,
			     size_t bytes_already,
			     struct efa_test_rtm_sent_result *out)
{
	struct efa_rdm_ep *efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct efa_rdm_domain *domain = efa_rdm_ep_rdm_domain(efa_rdm_ep);
	enum efa_test_rtm_family family = EFA_TEST_RTM_FAMILY(variant);
	uint32_t fi_op =
		EFA_TEST_RTM_IS_TAGGED(variant) ? ofi_op_tagged : ofi_op_msg;
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_peer *peer;
	bool pkt_entry_released = false;
	static char buf[EFA_TEST_RTM_LONG_LEN];

	memset(out, 0, sizeof(*out));

	txe = efa_test_rtm_alloc_txe(efa_rdm_ep, av, fi_op,
				     EFA_TEST_RTM_LONG_LEN, buf);
	if (!txe)
		return;
	peer = txe->peer;
	if (family == EFA_TEST_RTM_FAM_RUNTREAD)
		txe->bytes_runt = EFA_TEST_RTM_RUNT_LEN;
	/* A boundary completion writes to the CQ only when completion is
	 * requested; ask for it so the test can observe the SEND completion. */
	txe->fi_flags |= FI_COMPLETION;
	txe->bytes_sent = bytes_already;
	txe->bytes_acked = bytes_already;

	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	if (!pkt_entry) {
		efa_rdm_txe_release(txe);
		return;
	}
	efa_rdm_pke_set_ope(pkt_entry, txe);
	pkt_entry->peer = peer;
	pkt_entry->payload_size = payload_size;

	if (op == EFA_TEST_RTM_OP_SENT) {
		/* Report deltas: num_read_msg_in_flight is a domain counter that
		 * the "sent" handler only bumps (drained elsewhere), so repeated
		 * builds in one test would see a growing absolute value. The
		 * fresh peer per build makes its counter effectively a delta. */
		uint64_t read_in_flight_before = ofi_atomic_get64(&domain->num_read_msg_in_flight);

		switch (family) {
		case EFA_TEST_RTM_FAM_MEDIUM:
			/*
			 * The refactored medium protocol posts every packet of
			 * the message in one go, so its post-send hook accounts
			 * for the whole message rather than for one packet.
			 */
			efa_rdm_proto_medium_handle_tx_pkes_posted(efa_rdm_ep,
								   txe);
			break;
		case EFA_TEST_RTM_FAM_LONGCTS:
			efa_rdm_pke_handle_longcts_rtm_sent(pkt_entry);
			break;
		case EFA_TEST_RTM_FAM_LONGREAD:
			efa_rdm_pke_handle_longread_rtm_sent(pkt_entry);
			break;
		case EFA_TEST_RTM_FAM_RUNTREAD:
			/*
			 * The refactored runt read protocol posts the whole runt
			 * in one go, so its post-send hook accounts for
			 * bytes_runt rather than for one packet, and it bumps the
			 * read count once per message instead of on the first
			 * segment.
			 */
			efa_rdm_proto_runtread_handle_tx_pkes_posted(efa_rdm_ep,
								    txe);
			break;
		case EFA_TEST_RTM_FAM_EAGER:
			break;
		}
		/* sent handlers never free the txe; just read the counters. */
		out->bytes_sent = txe->bytes_sent;
		out->bytes_acked = txe->bytes_acked;
		out->num_read_msg_in_flight =
			ofi_atomic_get64(&domain->num_read_msg_in_flight) - read_in_flight_before;
		out->num_runt_bytes_in_flight = peer->num_runt_bytes_in_flight;
		out->txe_on_ope_list = 1;
		efa_rdm_pke_release_tx(pkt_entry);
		efa_rdm_txe_release(txe);
		return;
	}

	/* Completion handlers will free txe, so capture the expected post-state first. */
	out->bytes_acked = bytes_already + payload_size;
	out->num_runt_bytes_in_flight =
		peer->num_runt_bytes_in_flight; /* refreshed below if alive */

	/* The runtread completion handler will assert that this is >= payload_size */
	if (family == EFA_TEST_RTM_FAM_RUNTREAD)
		peer->num_runt_bytes_in_flight = payload_size;

	switch (family) {
	case EFA_TEST_RTM_FAM_EAGER:
		/*
		 * The eager protocol moved to the refactored code path, where
		 * the send completion handler lives on the packet entry. It
		 * releases the packet entry itself, unlike the handlers below,
		 * so remember not to release it again.
		 */
		pkt_entry->handle_pke =
			&efa_rdm_proto_eager_handle_rtm_send_completion;
		pkt_entry_released = true;
		pkt_entry->handle_pke(pkt_entry);
		break;
	case EFA_TEST_RTM_FAM_MEDIUM:
		/* Also refactored; see the eager case above. */
		pkt_entry->handle_pke =
			&efa_rdm_proto_medium_handle_rtm_send_completion;
		pkt_entry_released = true;
		pkt_entry->handle_pke(pkt_entry);
		break;
	case EFA_TEST_RTM_FAM_LONGCTS:
		efa_rdm_pke_handle_longcts_rtm_send_completion(pkt_entry);
		break;
	case EFA_TEST_RTM_FAM_RUNTREAD:
		/* Also refactored; see the eager case above. */
		pkt_entry->handle_pke =
			&efa_rdm_proto_runtread_handle_rtm_send_completion;
		pkt_entry_released = true;
		pkt_entry->handle_pke(pkt_entry);
		break;
	case EFA_TEST_RTM_FAM_LONGREAD:
		/* no dedicated completion handler */
		break;
	}

	out->txe_on_ope_list = efa_test_txe_on_ope_list(efa_rdm_ep, txe);
	out->send_completed = !out->txe_on_ope_list;
	out->cq_has_completion =
		(ofi_cq_read_entries(efa_rdm_ep->base_ep.util_ep.tx_cq, NULL, 0,
				     NULL) != -FI_EAGAIN);
	out->num_runt_bytes_in_flight = peer->num_runt_bytes_in_flight;
	if (out->txe_on_ope_list) {
		out->bytes_acked = txe->bytes_acked;
		if (!pkt_entry_released)
			efa_rdm_pke_release_tx(pkt_entry);
		efa_rdm_txe_release(txe);
	} else if (!pkt_entry_released) {
		efa_rdm_pke_release_tx(pkt_entry);
	}
}

void efa_test_selects_copy_method(struct fid_ep *ep, struct fid_av *av,
				  size_t total_len, enum efa_test_copy_src src,
				  bool cuda_api_permitted,
				  struct efa_test_copy_from_hmem_result *out)
{
	struct efa_rdm_ep *efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	static char buf[EFA_TEST_RTM_LONG_LEN];
	struct efa_rdm_mr mr = {0};
	struct efa_rdm_ope *txe;
	struct efa_rdm_pke *pke;
	int saved_p2p_opt;
	bool saved_cuda_api_permitted;

	memset(out, 0, sizeof(*out));

	if (total_len > sizeof(buf) ||
	    total_len < EFA_TEST_COPY_HMEM_DATA_SIZE) {
		out->ret = -FI_EINVAL;
		return;
	}

	mr.efa_mr.iface = FI_HMEM_CUDA;
	mr.device = EFA_TEST_COPY_HMEM_DEVICE;
	if (src == EFA_TEST_COPY_SRC_CUDA_GDRCOPY) {
		mr.flags = OFI_HMEM_DATA_DEV_REG_HANDLE;
		mr.hmem_data = (void *) (uintptr_t) EFA_TEST_COPY_HMEM_HANDLE;
	}

	txe = efa_test_rtm_alloc_txe(efa_rdm_ep, av, ofi_op_msg, total_len,
				     buf);
	if (!txe) {
		out->ret = -FI_ENOMEM;
		return;
	}
	txe->desc[0] = (src == EFA_TEST_COPY_SRC_HOST) ? NULL : &mr;

	pke = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_tx_pkt_pool,
				EFA_RDM_PKE_FROM_EFA_TX_POOL);
	if (!pke) {
		txe->desc[0] = NULL;
		efa_rdm_txe_release(txe);
		out->ret = -FI_ENOMEM;
		return;
	}

	/* Force the bounce-buffer copy regardless of the host's P2P support */
	saved_p2p_opt = efa_rdm_ep->hmem_p2p_opt;
	saved_cuda_api_permitted = efa_rdm_ep->cuda_api_permitted;
	efa_rdm_ep->hmem_p2p_opt = FI_HMEM_P2P_DISABLED;
	efa_rdm_ep->cuda_api_permitted = cuda_api_permitted;

	out->ret = efa_rdm_pke_init_payload_from_ope(
		pke, txe, 0, 0, EFA_TEST_COPY_HMEM_DATA_SIZE);

	efa_rdm_ep->hmem_p2p_opt = saved_p2p_opt;
	efa_rdm_ep->cuda_api_permitted = saved_cuda_api_permitted;

	out->payload_size = pke->payload_size;
	out->payload_in_wiredata = pke->payload == pke->wiredata;

	txe->desc[0] = NULL;
	efa_rdm_pke_release_tx(pke);
	efa_rdm_txe_release(txe);
}

size_t efa_test_get_max_gdrcopy_from_hmem_size(void)
{
	return efa_env.efa_max_gdrcopy_from_hmem_size;
}

void efa_test_set_max_gdrcopy_from_hmem_size(size_t size)
{
	efa_env.efa_max_gdrcopy_from_hmem_size = size;
}

int efa_test_flushes_when_queue_full(struct fid_ep *ep, size_t num_pkts,
				     struct efa_test_queued_copy_step *steps)
{
	struct efa_rdm_ep *efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	static char buf[(EFA_TEST_QUEUED_COPY_MAX_PKTS + 1) *
			EFA_TEST_COPY_HMEM_DATA_SIZE];
	struct efa_rdm_mr mr = {0};
	struct efa_rdm_ope *rxe;
	struct efa_rdm_pke *pke;
	size_t i, to_post_before;

	if (num_pkts > EFA_TEST_QUEUED_COPY_MAX_PKTS)
		return -FI_EINVAL;

	/* non-cuda, non-neuron hmem always takes the queued blocking copy */
	mr.efa_mr.iface = FI_HMEM_ROCR;
	mr.device = EFA_TEST_COPY_HMEM_DEVICE;

	rxe = efa_rdm_ep_alloc_rxe(efa_rdm_ep, NULL, ofi_op_msg);
	if (!rxe)
		return -FI_ENOMEM;
	rxe->iov_count = 1;
	rxe->iov[0].iov_base = buf;
	rxe->iov[0].iov_len = sizeof(buf);
	rxe->cq_entry.len = sizeof(buf);
	rxe->desc[0] = &mr;
	rxe->total_len = sizeof(buf);

	to_post_before = efa_rdm_ep->efa_rx_pkts_to_post;

	for (i = 0; i < num_pkts; i++) {
		pke = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
					EFA_RDM_PKE_FROM_EFA_RX_POOL);
		if (!pke) {
			rxe->desc[0] = NULL;
			efa_rdm_rxe_release(rxe);
			return -FI_ENOMEM;
		}
		/* eager rtm carries data with no segment offset */
		efa_rdm_pke_get_base_hdr(pke)->type = EFA_RDM_EAGER_MSGRTM_PKT;
		pke->payload = pke->wiredata;
		pke->payload_size = EFA_TEST_COPY_HMEM_DATA_SIZE;

		steps[i].ret = efa_rdm_pke_copy_payload_to_ope(pke, rxe);
		steps[i].queued_copy_num = efa_rdm_ep->queued_copy_num;
		steps[i].bytes_copied = rxe->bytes_copied;
		steps[i].bytes_queued_blocking_copy =
			rxe->bytes_queued_blocking_copy;
		steps[i].rx_pkts_released =
			efa_rdm_ep->efa_rx_pkts_to_post - to_post_before;
	}

	/* drops any copies still queued for the rxe */
	rxe->desc[0] = NULL;
	efa_rdm_rxe_release(rxe);
	return 0;
}

size_t efa_test_get_max_queued_copy(void)
{
	return efa_env.max_queued_copy;
}

void efa_test_set_max_queued_copy(size_t value)
{
	efa_env.max_queued_copy = value;
}

size_t efa_test_get_internal_rx_refill_threshold(void)
{
	return efa_env.internal_rx_refill_threshold;
}

static size_t efa_test_dlist_count(struct dlist_entry *head)
{
	struct dlist_entry *item;
	size_t count = 0;

	dlist_foreach(head, item)
		count++;
	return count;
}

int efa_test_av_remove_releases_unexp_pkt(
	struct fid_ep *ep_fid, struct fid_av *av, fi_addr_t peer_addr,
	int copy_unexp, struct efa_test_av_remove_unexp_result *out)
{
	struct efa_rdm_ep *ep = container_of(
		ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct util_srx_ctx *srx = efa_rdm_ep_get_peer_srx_ctx(ep);
	struct efa_rdm_peer *peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	int saved_rx_copy_unexp = efa_env.rx_copy_unexp;
	struct efa_rdm_pke *pke;
	struct efa_rdm_ope *rxe;
	ssize_t ret;

	memset(out, 0, sizeof *out);
	if (!peer)
		return -FI_EINVAL;

	pke = efa_test_reorder_pke(ep, peer, 0);
	if (!pke)
		return -FI_ENOMEM;
	pke->proto = &efa_rdm_proto_eager;

	out->unexp_pool_outstanding_before =
		efa_test_ep_unexp_pool_outstanding(ep_fid);
	out->rx_pkts_held_before = ep->efa_rx_pkts_held;
	out->ope_count_before = efa_test_dlist_count(&ep->base_ep.ope_list);

	ofi_genlock_lock(srx->lock);
	efa_env.rx_copy_unexp = copy_unexp;
	ret = efa_rdm_pke_proc_msgrtm(pke);
	efa_env.rx_copy_unexp = saved_rx_copy_unexp;
	if (ret) {
		ofi_genlock_unlock(srx->lock);
		return ret;
	}
	if (dlist_empty(&ep->base_ep.ope_list)) {
		ofi_genlock_unlock(srx->lock);
		return -FI_ENOENT;
	}
	rxe = container_of(ep->base_ep.ope_list.prev, struct efa_rdm_ope,
			   ep_entry);
	out->was_unexpected =
		rxe->state == EFA_RDM_RXE_UNEXP && rxe->unexp_pkt &&
		(copy_unexp ? rxe->unexp_pkt->alloc_type ==
				      EFA_RDM_PKE_FROM_UNEXP_POOL
			    : rxe->unexp_pkt == pke);

	out->unexp_pool_outstanding_queued =
		efa_test_ep_unexp_pool_outstanding(ep_fid);
	out->rx_pkts_held_queued = ep->efa_rx_pkts_held;
	out->rx_pkts_to_post_queued = ep->efa_rx_pkts_to_post;
	out->ope_count_queued = efa_test_dlist_count(&ep->base_ep.ope_list);

	ret = fi_av_remove(av, &peer_addr, 1, 0);
	ofi_genlock_unlock(srx->lock);
	if (ret)
		return ret;

	out->unexp_pool_outstanding_after =
		efa_test_ep_unexp_pool_outstanding(ep_fid);
	out->rx_pkts_held_after = ep->efa_rx_pkts_held;
	out->rx_pkts_to_post_after = ep->efa_rx_pkts_to_post;
	out->ope_count_after = efa_test_dlist_count(&ep->base_ep.ope_list);
	out->unspec_unexp_queue_empty =
		dlist_empty(&srx->unspec_unexp_msg_queue);
	return 0;
}

static ssize_t efa_test_rta_process_request(struct efa_rdm_pke *request,
					    enum fi_op op)
{
	switch (op) {
	case FI_ATOMIC_WRITE:
		return efa_rdm_pke_proc_write_rta(request);
	case FI_SUM:
		return efa_rdm_pke_proc_fetch_rta(request);
	case FI_CSWAP:
		return efa_rdm_pke_proc_compare_rta(request);
	default:
		return -FI_EINVAL;
	}
}

static size_t efa_test_bufpool_used_count(struct ofi_bufpool *pool)
{
	return pool->entry_cnt - efa_test_bufpool_free_count(pool);
}

static void efa_test_rta_fill_resource_state(
	struct efa_rdm_ep *ep, struct efa_rdm_peer *peer,
	struct efa_test_rta_responder_result *out)
{
	out->ope_count = efa_test_dlist_count(&ep->base_ep.ope_list);
	out->tx_pkt_pool_used =
		efa_test_bufpool_used_count(ep->efa_tx_pkt_pool);
	out->rxe_pool_used =
		efa_test_bufpool_used_count(ep->base_ep.rxe_pool);
	out->atomrsp_pool_used =
		efa_test_bufpool_used_count(ep->rx_atomrsp_pool);
	out->unexp_pool_used =
		efa_test_ep_unexp_pool_outstanding(&ep->base_ep.util_ep.ep_fid);
	out->outstanding_tx_ops = ep->efa_outstanding_tx_ops;
	out->peer_outstanding_tx_ops = peer->efa_outstanding_tx_ops;
}

static int efa_test_rta_responder(
	struct fid_ep *ep_fid, struct fid_domain *domain, fi_addr_t peer_addr,
	enum fi_op op, struct efa_test_rta_responder_result *out)
{
	struct efa_rdm_ep *ep = container_of(
		ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct efa_rdm_peer *peer =
		efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	struct efa_rdm_mr verified_mr = {0};
	struct efa_mr *registered_mr;
	struct efa_rdm_pke *request = NULL;
	struct efa_rdm_pke *response;
	struct efa_rdm_rta_hdr *rta_hdr;
	struct efa_rdm_atomrsp_pkt *atomrsp;
	struct fi_eq_err_entry eq_err = {0};
	struct fid_mr *mr = NULL;
	uint8_t target[16] = {1};
	void *saved_mem_desc;
	char *data;
	size_t hdr_size;
	size_t data_size;
	size_t unexp_pool_used_before;
	int pkt_type;
	int close_ret;
	int ret;

	memset(out, 0, sizeof(*out));
	if (!peer)
		return -FI_EINVAL;

	switch (op) {
	case FI_ATOMIC_WRITE:
		pkt_type = EFA_RDM_WRITE_RTA_PKT;
		data_size = 1;
		break;
	case FI_SUM:
		pkt_type = EFA_RDM_FETCH_RTA_PKT;
		data_size = 1;
		break;
	case FI_CSWAP:
		pkt_type = EFA_RDM_COMPARE_RTA_PKT;
		data_size = 2;
		break;
	default:
		return -FI_EINVAL;
	}

	ret = fi_mr_reg(domain, target, sizeof(target),
			FI_SEND | FI_RECV | FI_REMOTE_READ | FI_REMOTE_WRITE,
			0, 0, 0, &mr, NULL);
	if (ret)
		return ret;

	registered_mr = container_of(mr, struct efa_mr, mr_fid);
	out->registered_iface = registered_mr->iface;
	saved_mem_desc = registered_mr->mr_fid.mem_desc;
	verified_mr.efa_mr.iface = FI_HMEM_CUDA;
	registered_mr->mr_fid.mem_desc = &verified_mr.efa_mr;

	unexp_pool_used_before =
		efa_test_bufpool_used_count(ep->rx_unexp_pkt_pool);
	request = efa_rdm_pke_alloc(
		ep, ep->rx_unexp_pkt_pool, EFA_RDM_PKE_FROM_UNEXP_POOL);
	if (!request) {
		ret = -FI_ENOMEM;
		goto out;
	}
	request->peer = peer;

	rta_hdr = (struct efa_rdm_rta_hdr *) request->wiredata;
	memset(rta_hdr, 0,
	       sizeof(*rta_hdr) + sizeof(struct efa_rma_iov) + data_size);
	rta_hdr->type = pkt_type;
	rta_hdr->version = EFA_RDM_PROTOCOL_VERSION;
	rta_hdr->flags = EFA_RDM_REQ_ATOMIC;
	rta_hdr->rma_iov_count = 1;
	rta_hdr->atomic_datatype = FI_UINT8;
	rta_hdr->atomic_op = op;
	rta_hdr->recv_id = EFA_TEST_RTA_RECV_ID;
	rta_hdr->rma_iov[0].addr = (uintptr_t) target;
	rta_hdr->rma_iov[0].len = 1;
	rta_hdr->rma_iov[0].key = fi_mr_key(mr);

	hdr_size = efa_rdm_pke_get_req_hdr_size(request);
	data = request->wiredata + hdr_size;
	data[0] = 2;
	if (op == FI_CSWAP)
		data[1] = 1;
	request->pkt_size = hdr_size + data_size;

	out->ret = efa_test_rta_process_request(request, op);
	if (efa_test_bufpool_used_count(ep->rx_unexp_pkt_pool) ==
	    unexp_pool_used_before)
		request = NULL;

	if (!out->ret && op != FI_ATOMIC_WRITE) {
		if (ep->send_pkt_entry_vec_size != 1 ||
		    !ep->send_pkt_entry_vec[0]) {
			ret = -FI_EINVAL;
			goto out;
		}

		response = ep->send_pkt_entry_vec[0];
		atomrsp = (struct efa_rdm_atomrsp_pkt *) response->wiredata;
		out->response_valid =
			atomrsp->hdr.type == EFA_RDM_ATOMRSP_PKT &&
			atomrsp->hdr.seg_length == 1;
		out->response = atomrsp->data[0];
		out->response_recv_id = atomrsp->hdr.recv_id;
		efa_rdm_pke_handle_send_completion(response);
	}

	out->target = target[0];
	out->eq_read_ret = fi_eq_readerr(
		&ep->base_ep.util_ep.eq->eq_fid, &eq_err, 0);
	ret = 0;

out:
	if (request)
		efa_rdm_pke_release_rx(request);
	registered_mr->mr_fid.mem_desc = saved_mem_desc;
	close_ret = fi_close(&mr->fid);
	if (!ret && close_ret)
		ret = close_ret;
	efa_test_rta_fill_resource_state(ep, peer, out);
	return ret;
}

int efa_test_rta_responder_uses_verified_mr_desc(
	struct fid_ep *ep, struct fid_domain *domain, fi_addr_t peer_addr,
	enum fi_op op, struct efa_test_rta_responder_result *out)
{
	return efa_test_rta_responder(ep, domain, peer_addr, op, out);
}
