/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_rdm_ope_helpers.h"
#include "efa_gtest_common_helpers.h"
#include "efa.h"
#include "efa_av.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_ope.h"
#include "rdm/efa_rdm_mr.h"
#include "rdm/efa_rdm_pke.h"
#include "rdm/efa_rdm_pke_utils.h"
#include "rdm/efa_rdm_cq.h"
#include "rdm/efa_rdm_peer.h"
#include "rdm/efa_rdm_protocol.h"
#include "rdm/efa_rdm_pke_nonreq.h"
#include "rdm/efa_rdm_rma.h"
#include "rdm/efa_rdm_srx.h"
#include "ofi_util.h"

int efa_test_drive_rxe_unexp_handle_error(struct fid_ep *ep, void *op_context,
					  int err, int *prov_errno_out)
{
	struct efa_rdm_ep *efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	fi_addr_t peer_addr = 0;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *rxe;
	int prov_errno = EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE;
	int ret;

	ret = efa_test_av_insert_self(
		ep, &efa_rdm_ep->base_ep.util_ep.av->av_fid, &peer_addr);
	if (ret != 1)
		return -FI_EINVAL;

	peer = efa_rdm_ep_get_peer_explicit(efa_rdm_ep, peer_addr);
	if (!peer)
		return -FI_EINVAL;

	rxe = efa_rdm_ep_alloc_rxe(efa_rdm_ep, peer, ofi_op_tagged);
	if (!rxe)
		return -FI_ENOMEM;

	rxe->state = EFA_RDM_RXE_UNEXP;
	rxe->cq_entry.op_context = op_context;

	efa_rdm_rxe_handle_error(rxe, err, prov_errno);
	efa_rdm_rxe_release(rxe);

	if (prov_errno_out)
		*prov_errno_out = prov_errno;

	return 0;
}

int efa_test_queue_op_with_fi_more(struct fid_ep *ep_fid, struct fid_av *av_fid,
				   struct fid_domain *domain_fid, int op_kind,
				   struct efa_test_queued_op *qop)
{
	struct efa_rdm_ep *ep = container_of(ep_fid, struct efa_rdm_ep,
					     base_ep.util_ep.ep_fid);
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = FI_ADDR_NOTAVAIL;
	void *desc;
	struct iovec iov;
	int ret;

	memset(qop, 0, sizeof(*qop));
	qop->ep = ep_fid;

	/* Own GID with a different QPN: AH creation succeeds against the
	 * real device, but the peer is not self, so handshake is enforced. */
	ret = fi_getname(&ep_fid->fid, &raw_addr, &raw_addr_len);
	if (ret)
		return ret;
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	if (fi_av_insert(av_fid, &raw_addr, 1, &peer_addr, 0, NULL) != 1)
		return -FI_EINVAL;

	qop->peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	if (!qop->peer)
		return -FI_EINVAL;
	/* REQ already sent, so enforce_handshake queues instead of posting a
	 * handshake REQ; no handshake received yet. */
	qop->peer->flags = EFA_RDM_PEER_REQ_SENT;
	qop->peer->av_entry->shm_fi_addr = FI_ADDR_NOTAVAIL;

	ret = fi_mr_reg(domain_fid, qop->buf, sizeof(qop->buf),
			FI_SEND | FI_RECV | FI_READ | FI_WRITE, 0, 0, 0,
			&qop->mr, NULL);
	if (ret)
		return ret;
	desc = fi_mr_desc(qop->mr);

	iov.iov_base = qop->buf;
	iov.iov_len = sizeof(qop->buf);

	if (op_kind == EFA_TEST_QUEUED_OP_SEND) {
		/* Force the send path's handshake enforcement */
		ep->peer_may_have_zcpy_rx = true;

		struct fi_msg msg = {
			.msg_iov = &iov,
			.desc = &desc,
			.iov_count = 1,
			.addr = peer_addr,
			.context = NULL,
			.data = 0,
		};
		ret = fi_sendmsg(ep_fid, &msg, FI_MORE);
	} else {
		struct fi_rma_iov rma_iov = {
			.addr = (uint64_t) qop->buf,
			.len = sizeof(qop->buf),
			.key = 0x1234,
		};
		struct fi_msg_rma msg = {
			.msg_iov = &iov,
			.desc = &desc,
			.iov_count = 1,
			.addr = peer_addr,
			.rma_iov = &rma_iov,
			.rma_iov_count = 1,
			.context = NULL,
			.data = 0,
		};
		if (op_kind == EFA_TEST_QUEUED_OP_READ)
			ret = fi_readmsg(ep_fid, &msg, FI_MORE);
		else
			ret = fi_writemsg(ep_fid, &msg, FI_MORE);
	}
	if (ret)
		return ret;

	if (dlist_empty(&ep->ope_queued_list))
		return -FI_EINVAL;
	qop->txe = container_of(ep->ope_queued_list.next,
				struct efa_rdm_ope, queued_entry);
	if (!(qop->txe->internal_flags & EFA_RDM_OPE_QUEUED_BEFORE_HANDSHAKE))
		return -FI_EINVAL;

	qop->fi_more_was_set = !!(qop->txe->fi_flags & FI_MORE);
	return 0;
}

int efa_test_process_queued_ope_after_handshake(struct efa_test_queued_op *qop)
{
	struct efa_rdm_ep *ep = container_of(qop->ep, struct efa_rdm_ep,
					     base_ep.util_ep.ep_fid);

	/* Simulate the handshake landing: peer advertises device RDMA
	 * read/write and p2p, so the repost takes the device data path. */
	qop->peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	qop->peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_RDMA_READ |
				    EFA_RDM_EXTRA_FEATURE_RDMA_WRITE;
	qop->peer->p2p_supported = true;
	/* use_device_rdma defaults off on some platforms, which would route
	 * the repost through the emulated (send-based) protocols instead of
	 * efa_qp_post_read/write. The device caps are verified by the test's
	 * skip gate; force the software toggle so the device path is taken. */
	ep->use_device_rdma = true;

	return efa_rdm_ope_process_queued_ope(qop->txe);
}

void efa_test_queued_op_cleanup(struct efa_test_queued_op *qop, uint64_t wr_id)
{
	struct efa_rdm_ep *ep = container_of(qop->ep, struct efa_rdm_ep,
					     base_ep.util_ep.ep_fid);
	struct efa_rdm_pke *pkt_entry;

	if (wr_id) {
		pkt_entry = efa_rdm_cq_get_pke_from_wr_id_solicited(wr_id);
		if (pkt_entry)
			efa_rdm_pke_release_tx(pkt_entry);
	}
	if (qop->txe)
		efa_rdm_txe_release(qop->txe);
	ep->efa_outstanding_tx_ops = 0;

	if (qop->mr) {
		fi_close(&qop->mr->fid);
		qop->mr = NULL;
	}
}

static void efa_test_fill_process_queued_result(
	struct efa_rdm_ep *ep, struct efa_rdm_ope *txe,
	struct efa_test_process_queued_result *res)
{
	res->before_handshake_flag_set =
		!!(txe->internal_flags & EFA_RDM_OPE_QUEUED_BEFORE_HANDSHAKE);
	res->any_queued_flag_set =
		!!(txe->internal_flags & EFA_RDM_OPE_QUEUED_FLAGS);
	res->queued_list_empty = dlist_empty(&ep->ope_queued_list);
	res->fi_more_still_set = !!(txe->fi_flags & FI_MORE);
	res->before_handshake_cnt = ep->ope_queued_before_handshake_cnt;
}

int efa_test_process_queued_ope_derives_before_handshake_flag(
	struct efa_test_queued_op *qop,
	struct efa_test_process_queued_result *res)
{
	struct efa_rdm_ep *ep = container_of(qop->ep, struct efa_rdm_ep,
					     base_ep.util_ep.ep_fid);

	if (!qop->txe || !qop->peer)
		return -FI_EINVAL;
	if (qop->peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED)
		return -FI_EINVAL;
	if (!(qop->txe->internal_flags & EFA_RDM_OPE_QUEUED_BEFORE_HANDSHAKE))
		return -FI_EINVAL;

	res->ret = efa_rdm_ope_process_queued_ope(qop->txe);
	efa_test_fill_process_queued_result(ep, qop->txe, res);
	return 0;
}

int efa_test_process_queued_ope_after_handshake_result(
	struct efa_test_queued_op *qop,
	struct efa_test_process_queued_result *res)
{
	struct efa_rdm_ep *ep = container_of(qop->ep, struct efa_rdm_ep,
					     base_ep.util_ep.ep_fid);

	if (!qop->txe || !qop->peer)
		return -FI_EINVAL;

	res->ret = efa_test_process_queued_ope_after_handshake(qop);
	efa_test_fill_process_queued_result(ep, qop->txe, res);
	return 0;
}

int efa_test_queue_ope_with_flag(struct fid_ep *ep_fid, struct fid_av *av_fid,
				 int flag_kind, struct efa_test_queued_op *qop)
{
	struct efa_rdm_ep *ep = container_of(ep_fid, struct efa_rdm_ep,
					     base_ep.util_ep.ep_fid);
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = FI_ADDR_NOTAVAIL;
	struct efa_rdm_pke *pkt_entry;
	struct fi_msg msg = {0};
	struct iovec iov;
	int ret;

	memset(qop, 0, sizeof(*qop));
	qop->ep = ep_fid;

	ret = fi_getname(&ep_fid->fid, &raw_addr, &raw_addr_len);
	if (ret)
		return ret;
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	if (fi_av_insert(av_fid, &raw_addr, 1, &peer_addr, 0, NULL) != 1)
		return -FI_EINVAL;

	qop->peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	if (!qop->peer)
		return -FI_EINVAL;
	qop->peer->av_entry->shm_fi_addr = FI_ADDR_NOTAVAIL;

	iov.iov_base = qop->buf;
	iov.iov_len = sizeof(qop->buf);
	msg.msg_iov = &iov;
	msg.iov_count = 1;
	msg.addr = peer_addr;

	qop->txe = ofi_buf_alloc(ep->base_ep.txe_pool);
	if (!qop->txe)
		return -FI_ENOMEM;
	efa_rdm_txe_construct(qop->txe, ep, qop->peer, &msg, ofi_op_msg, 0, 0);

	switch (flag_kind) {
	case EFA_TEST_QUEUED_FLAG_RNR:
		pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
					      EFA_RDM_PKE_FROM_EFA_TX_POOL);
		if (!pkt_entry)
			return -FI_ENOMEM;
		efa_rdm_pke_set_ope(pkt_entry, qop->txe);
		pkt_entry->peer = qop->peer;
		efa_rdm_ep_queue_rnr_pkt(ep, pkt_entry);
		break;
	case EFA_TEST_QUEUED_FLAG_CTRL:
		qop->queued_ctrl_type = EFA_RDM_CTS_PKT;
		qop->txe->queued_ctrl_type = qop->queued_ctrl_type;
		qop->txe->internal_flags |= EFA_RDM_OPE_QUEUED_CTRL;
		dlist_insert_tail(&qop->txe->queued_entry, &ep->ope_queued_list);
		break;
	case EFA_TEST_QUEUED_FLAG_READ:
		/*
		 * efa_rdm_ope_post_read() asserts a non-empty rma_iov, and
		 * takes its zero-byte branch while bytes_read_total_len is 0.
		 */
		qop->txe->rma_iov_count = 1;
		qop->txe->rma_iov[0].addr = (uint64_t) qop->buf;
		qop->txe->rma_iov[0].len = sizeof(qop->buf);
		qop->txe->rma_iov[0].key = 0x1234;
		qop->txe->internal_flags |= EFA_RDM_OPE_QUEUED_READ;
		dlist_insert_tail(&qop->txe->queued_entry, &ep->ope_queued_list);
		break;
	default:
		return -FI_EINVAL;
	}

	if (dlist_empty(&ep->ope_queued_list))
		return -FI_EINVAL;

	return 0;
}

int efa_test_process_queued_flag_op(struct efa_test_queued_op *qop,
				    struct efa_test_process_queued_result *res)
{
	struct efa_rdm_ep *ep = container_of(qop->ep, struct efa_rdm_ep,
					     base_ep.util_ep.ep_fid);

	if (!qop->txe)
		return -FI_EINVAL;

	res->ret = efa_rdm_ope_process_queued_ope(qop->txe);
	efa_test_fill_process_queued_result(ep, qop->txe, res);
	return 0;
}

void efa_test_simulate_source_mr_canceled(struct efa_test_queued_op *qop)
{
	/*
	 * Mirrors the cmocka helper: a source MR whose current generation no
	 * longer matches the snapshot taken when the op was dispatched, which
	 * is how a closed MR presents to the repost path.
	 */
	static struct efa_rdm_mr stale_source_mr;

	stale_source_mr.gen = 1;
	qop->txe->iov_count = 1;
	qop->txe->desc[0] = &stale_source_mr;
	qop->txe->desc_gen[0] = 2;
}

int efa_test_peer_abort_prov_errno(void)
{
	return FI_EFA_ERR_PEER_ABORTED;
}

/**
 * @brief Build a recv matched through the SRX, so the abort path has a
 * peer_rxe to return and the caller's op_context to report.
 */
static int efa_test_alloc_matched_rxe(struct efa_rdm_ep *ep, struct fid_av *av,
				      char *buf, size_t len, void *op_context,
				      struct efa_rdm_ope **rxe_out)
{
	struct util_srx_ctx *srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);
	struct fid_peer_srx *peer_srx = util_get_peer_srx(ep->peer_srx_ep);
	struct fi_peer_match_attr match_attr = {0};
	struct fi_peer_rx_entry *peer_rxe = NULL;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_peer *peer;
	fi_addr_t peer_addr = 0;
	struct iovec iov;
	void *desc = NULL;
	int ret;

	ret = efa_test_av_insert_self(&ep->base_ep.util_ep.ep_fid, av,
				      &peer_addr);
	if (ret != 1)
		return -FI_EINVAL;

	peer = efa_rdm_ep_get_peer_explicit(ep, peer_addr);
	if (!peer)
		return -FI_EINVAL;

	iov.iov_base = buf;
	iov.iov_len = len;
	ret = util_srx_generic_recv(ep->peer_srx_ep, &iov, &desc, 1,
				    FI_ADDR_UNSPEC, op_context, 0);
	if (ret)
		return ret;

	match_attr.addr = FI_ADDR_UNSPEC;
	match_attr.msg_size = len;

	ofi_genlock_lock(srx_ctx->lock);
	ret = peer_srx->owner_ops->get_msg(peer_srx, &match_attr, &peer_rxe);
	if (ret || !peer_rxe) {
		ofi_genlock_unlock(srx_ctx->lock);
		return ret ? ret : -FI_EINVAL;
	}

	rxe = efa_rdm_ep_alloc_rxe(ep, peer, ofi_op_msg);
	if (!rxe) {
		ofi_genlock_unlock(srx_ctx->lock);
		return -FI_ENOMEM;
	}

	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->peer_rxe = peer_rxe;
	rxe->cq_entry.op_context = peer_rxe->context;
	rxe->cq_entry.flags = FI_RECV | FI_MSG;
	rxe->cq_entry.len = len;
	rxe->total_len = len;
	rxe->iov_count = 1;
	rxe->iov[0] = iov;
	ofi_genlock_unlock(srx_ctx->lock);

	*rxe_out = rxe;
	return 0;
}

static void efa_test_mark_and_drain(struct efa_rdm_ep *ep,
				    struct efa_rdm_ope *rxe)
{
	struct util_srx_ctx *srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);

	ofi_genlock_lock(srx_ctx->lock);
	efa_rdm_rxe_mark_peer_aborted_if_needed(
		rxe, EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);
	efa_rdm_rxe_release_peer_abort_if_drained(rxe);
	ofi_genlock_unlock(srx_ctx->lock);
}

static struct efa_rdm_ope *efa_test_find_local_read_txe(struct efa_rdm_ep *ep,
						       struct efa_rdm_pke *pke)
{
	struct efa_rdm_ope *ope;
	struct dlist_entry *item;

	dlist_foreach(&ep->base_ep.ope_list, item) {
		ope = container_of(item, struct efa_rdm_ope, ep_entry);
		if (ope->type == EFA_RDM_TXE &&
		    ope->local_read_pkt_entry == pke)
			return ope;
	}

	return NULL;
}

int efa_test_ope_list_rxe_count(struct fid_ep *ep)
{
	struct efa_rdm_ep *efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct efa_rdm_ope *ope;
	struct dlist_entry *item;
	int count = 0;

	dlist_foreach(&efa_rdm_ep->base_ep.ope_list, item) {
		ope = container_of(item, struct efa_rdm_ope, ep_entry);
		if (ope->type == EFA_RDM_RXE)
			count++;
	}

	return count;
}

int efa_test_abort_waits_for_local_read_copy_setup(
	struct fid_ep *ep, struct fid_av *av, void *op_context,
	struct efa_test_local_read_abort *state)
{
	struct efa_rdm_ep *efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	static struct efa_rdm_mr hmem_mr;
	int ret;

	state->ep = ep;
	ret = efa_test_alloc_matched_rxe(efa_rdm_ep, av, state->buf,
					 sizeof(state->buf), op_context,
					 &state->rxe);
	if (ret)
		return ret;

	/* An rx pool packet is registered, so the copy read needs no read-copy
	 * clone (which would require an FI_HMEM domain). */
	state->data_pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep,
						  efa_rdm_ep->efa_rx_pkt_pool,
						  EFA_RDM_PKE_FROM_EFA_RX_POOL);
	if (!state->data_pkt_entry)
		return -FI_ENOMEM;

	state->data_pkt_entry->payload = state->data_pkt_entry->wiredata;
	state->data_pkt_entry->payload_size = sizeof(state->buf);
	efa_rdm_pke_set_ope(state->data_pkt_entry, state->rxe);

	/* The copy read is only taken for device memory. */
	hmem_mr.efa_mr.iface = FI_HMEM_CUDA;
	state->rxe->desc[0] = &hmem_mr;

	ret = efa_rdm_rxe_post_local_read_or_queue(state->rxe, 0,
						   state->data_pkt_entry,
						   state->data_pkt_entry->payload,
						   sizeof(state->buf));
	if (ret)
		return ret;

	state->read_txe = efa_test_find_local_read_txe(efa_rdm_ep,
						      state->data_pkt_entry);
	if (!state->read_txe)
		return -FI_EINVAL;

	efa_test_mark_and_drain(efa_rdm_ep, state->rxe);
	return 0;
}

void efa_test_abort_waits_for_local_read_copy_retire(
	struct efa_test_local_read_abort *state,
	struct efa_rdm_pke *ctx_pkt_entry)
{
	struct efa_rdm_ep *ep = container_of(state->ep, struct efa_rdm_ep,
					     base_ep.util_ep.ep_fid);
	struct util_srx_ctx *srx_ctx = efa_rdm_ep_get_peer_srx_ctx(ep);

	/* efa_rdm_pke_read() would have set this when it posted the WR. */
	ctx_pkt_entry->flags |= EFA_RDM_PKE_LOCAL_READ;

	ofi_genlock_lock(srx_ctx->lock);
	efa_rdm_pke_handle_rma_completion(ctx_pkt_entry);
	ofi_genlock_unlock(srx_ctx->lock);
}

int efa_test_abort_without_local_read_copy_completes_now(
	struct fid_ep *ep, struct fid_av *av, void *op_context,
	struct efa_test_local_read_abort *state)
{
	struct efa_rdm_ep *efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	int ret;

	state->ep = ep;
	ret = efa_test_alloc_matched_rxe(efa_rdm_ep, av, state->buf,
					 sizeof(state->buf), op_context,
					 &state->rxe);
	if (ret)
		return ret;

	efa_test_mark_and_drain(efa_rdm_ep, state->rxe);
	return 0;
}
