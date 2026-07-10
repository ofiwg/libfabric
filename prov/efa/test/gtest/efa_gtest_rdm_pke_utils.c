/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_rdm_pke_utils.h"
#include "efa.h"
#include "efa_env.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_ope.h"
#include "rdm/efa_rdm_pke.h"
#include "rdm/efa_rdm_pke_rtm.h"
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
