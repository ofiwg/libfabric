/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_rdm_pke_utils.h"
#include "efa.h"
#include "efa_env.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_ope.h"
#include "rdm/efa_rdm_pke.h"
#include "rdm/efa_rdm_pke_rtm.h"
#include "rdm/efa_rdm_pke_utils.h"
#include "rdm/efa_rdm_protocol.h"

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
	fi_addr_t peer_addr = 0;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_rdm_peer *peer;
	struct iovec iov;
	struct fi_msg msg = {0};
	struct efa_rdm_ope *txe;

	if (fi_getname(&ep->base_ep.util_ep.ep_fid.fid, &raw_addr,
		       &raw_addr_len))
		return NULL;
	raw_addr.qpn = 0;
	raw_addr.qkey = 0x1234;
	if (fi_av_insert(av, &raw_addr, 1, &peer_addr, 0, NULL) != 1)
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
			out->msg_length =
				efa_rdm_pke_get_dc_medium_rtm_base_hdr(
					pkt_entry)
					->msg_length;
			out->seg_offset =
				efa_rdm_pke_get_dc_medium_rtm_base_hdr(
					pkt_entry)
					->seg_offset;
			out->send_id = efa_rdm_pke_get_dc_medium_rtm_base_hdr(
					       pkt_entry)
					       ->send_id;
		} else {
			out->msg_length =
				efa_rdm_pke_get_medium_rtm_base_hdr(pkt_entry)
					->msg_length;
			out->seg_offset =
				efa_rdm_pke_get_medium_rtm_base_hdr(pkt_entry)
					->seg_offset;
		}
		break;
	case EFA_TEST_RTM_FAM_LONGCTS:
		out->msg_length =
			efa_rdm_pke_get_longcts_rtm_base_hdr(pkt_entry)
				->msg_length;
		out->send_id = efa_rdm_pke_get_longcts_rtm_base_hdr(pkt_entry)
				       ->send_id;
		out->credit_request =
			efa_rdm_pke_get_longcts_rtm_base_hdr(pkt_entry)
				->credit_request;
		break;
	case EFA_TEST_RTM_FAM_LONGREAD:
		out->msg_length =
			efa_rdm_pke_get_longread_rtm_base_hdr(pkt_entry)
				->msg_length;
		out->send_id = efa_rdm_pke_get_longread_rtm_base_hdr(pkt_entry)
				       ->send_id;
		out->read_iov_count =
			efa_rdm_pke_get_longread_rtm_base_hdr(pkt_entry)
				->read_iov_count;
		break;
	case EFA_TEST_RTM_FAM_RUNTREAD:
		out->msg_length =
			efa_rdm_pke_get_runtread_rtm_base_hdr(pkt_entry)
				->msg_length;
		out->send_id = efa_rdm_pke_get_runtread_rtm_base_hdr(pkt_entry)
				       ->send_id;
		out->seg_offset =
			efa_rdm_pke_get_runtread_rtm_base_hdr(pkt_entry)
				->seg_offset;
		out->runt_length =
			efa_rdm_pke_get_runtread_rtm_base_hdr(pkt_entry)
				->runt_length;
		out->read_iov_count =
			efa_rdm_pke_get_runtread_rtm_base_hdr(pkt_entry)
				->read_iov_count;
		break;
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
			     size_t bytes_already, size_t seg_offset,
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

	/* only the runtread send handler checks seg_offset */
	if (family == EFA_TEST_RTM_FAM_RUNTREAD) {
		struct efa_rdm_base_hdr *base_hdr =
			(struct efa_rdm_base_hdr *) pkt_entry->wiredata;
		base_hdr->type = efa_test_rtm_pkt_type(variant);
		efa_rdm_pke_get_runtread_rtm_base_hdr(pkt_entry)->seg_offset =
			seg_offset;
	}

	if (op == EFA_TEST_RTM_OP_SENT) {
		/* Report deltas: num_read_msg_in_flight is a domain counter that
		 * the "sent" handler only bumps (drained elsewhere), so repeated
		 * builds in one test would see a growing absolute value. The
		 * fresh peer per build makes its counter effectively a delta. */
		uint64_t read_in_flight_before = domain->num_read_msg_in_flight;

		switch (family) {
		case EFA_TEST_RTM_FAM_MEDIUM:
			efa_rdm_pke_handle_medium_rtm_sent(pkt_entry);
			break;
		case EFA_TEST_RTM_FAM_LONGCTS:
			efa_rdm_pke_handle_longcts_rtm_sent(pkt_entry);
			break;
		case EFA_TEST_RTM_FAM_LONGREAD:
			efa_rdm_pke_handle_longread_rtm_sent(pkt_entry);
			break;
		case EFA_TEST_RTM_FAM_RUNTREAD:
			efa_rdm_pke_handle_runtread_rtm_sent(pkt_entry, peer);
			break;
		case EFA_TEST_RTM_FAM_EAGER:
			break;
		}
		/* sent handlers never free the txe; just read the counters. */
		out->bytes_sent = txe->bytes_sent;
		out->bytes_acked = txe->bytes_acked;
		out->num_read_msg_in_flight =
			domain->num_read_msg_in_flight - read_in_flight_before;
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
	case EFA_TEST_RTM_FAM_MEDIUM:
		efa_rdm_pke_handle_medium_rtm_send_completion(pkt_entry);
		break;
	case EFA_TEST_RTM_FAM_LONGCTS:
		efa_rdm_pke_handle_longcts_rtm_send_completion(pkt_entry);
		break;
	case EFA_TEST_RTM_FAM_RUNTREAD:
		efa_rdm_pke_handle_runtread_rtm_send_completion(pkt_entry);
		break;
	case EFA_TEST_RTM_FAM_LONGREAD:
	case EFA_TEST_RTM_FAM_EAGER:
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
		efa_rdm_pke_release_tx(pkt_entry);
		efa_rdm_txe_release(txe);
	} else {
		efa_rdm_pke_release_tx(pkt_entry);
	}
}
