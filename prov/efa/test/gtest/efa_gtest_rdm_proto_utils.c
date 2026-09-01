/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_rdm_proto_utils.h"
#include "efa.h"
#include "ofi_mem.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_ope.h"
#include "rdm/efa_rdm_pke.h"
#include "rdm/efa_rdm_pke_cmd.h"
#include "rdm/efa_rdm_pke_rtm.h"
#include "rdm/efa_rdm_pkt_type.h"
#include "rdm/efa_rdm_proto.h"
#include "rdm/efa_rdm_proto_medium.h"
#include "rdm/efa_rdm_protocol.h"
#include <rdma/fi_errno.h>
#include <stdlib.h>
#include <string.h>

struct efa_test_proto_ctx {
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	fi_addr_t peer_addr;
	void *buf;
	size_t len;
	struct fid_mr *mr;
};

static struct efa_rdm_ep *efa_test_proto_ep(struct fid_ep *ep)
{
	return container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
}

static size_t efa_test_proto_ope_list_count(struct efa_rdm_ep *ep)
{
	struct dlist_entry *item;
	size_t count = 0;

	dlist_foreach(&ep->base_ep.ope_list, item)
		count++;

	return count;
}

/*
 * Insert a handshaked peer that will not be routed to shm, so a send reaches
 * the EFA protocol path.
 */
static int efa_test_proto_setup_peer(struct fid_ep *ep, struct fid_av *av,
				     struct efa_test_proto_ctx *ctx)
{
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	int ret;

	ctx->ep = efa_test_proto_ep(ep);

	/*
	 * Own GID with a different QPN, not a fabricated GID: these tests run
	 * against the real device, which rejects an AH for a GID that is not on
	 * the fabric, and a different QPN is enough to keep the peer from being
	 * self.
	 */
	ret = fi_getname(&ep->fid, &raw_addr, &raw_addr_len);
	if (ret)
		return ret;
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	if (fi_av_insert(av, &raw_addr, 1, &ctx->peer_addr, 0, NULL) != 1)
		return -FI_EINVAL;

	ctx->peer = efa_rdm_ep_get_peer_explicit(ctx->ep, ctx->peer_addr);
	if (!ctx->peer)
		return -FI_EINVAL;

	ctx->peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	ctx->peer->av_entry->shm_fi_addr = FI_ADDR_NOTAVAIL;
	return 0;
}

static int efa_test_proto_setup_buf(struct fid_domain *domain,
				    struct efa_test_proto_ctx *ctx, size_t len)
{
	int err;

	ctx->len = len;
	ctx->buf = malloc(len);
	if (!ctx->buf)
		return -FI_ENOMEM;
	memset(ctx->buf, 0, len);

	err = fi_mr_reg(domain, ctx->buf, len, FI_SEND | FI_RECV, 0, 0, 0,
			&ctx->mr, NULL);
	if (err) {
		free(ctx->buf);
		ctx->buf = NULL;
		return err;
	}
	return 0;
}

static void efa_test_proto_teardown_buf(struct efa_test_proto_ctx *ctx)
{
	if (ctx->mr) {
		fi_close(&ctx->mr->fid);
		ctx->mr = NULL;
	}
	free(ctx->buf);
	ctx->buf = NULL;
}

int efa_test_proto_medium_len_in_band(struct fid_ep *ep, size_t len)
{
	struct efa_rdm_ep *efa_rdm_ep = efa_test_proto_ep(ep);

	return len > efa_rdm_ep->mtu_size &&
	       len <= g_efa_hmem_info[FI_HMEM_SYSTEM].max_medium_msg_size;
}

int efa_test_proto_peer_abort_prov_errno(void)
{
	return FI_EFA_ERR_PEER_ABORTED;
}

int efa_test_proto_medium_msgrtm_pkt_type(void)
{
	return EFA_RDM_MEDIUM_MSGRTM_PKT;
}

int efa_test_proto_medium_plan(struct fid_ep *ep, struct fid_av *av,
			       enum fi_hmem_iface iface, int align128,
			       size_t total_len, int leave_one_tx_pkt,
			       struct efa_test_proto_plan_result *out)
{
	struct efa_rdm_ep *efa_rdm_ep = efa_test_proto_ep(ep);
	struct efa_rdm_mr mock_mr;
	struct efa_rdm_ope mock_txe;
	struct efa_rdm_peer mock_peer = {0};
	size_t saved_outstanding, pkt_entry_cnt = 0;
	size_t data_sizes[EFA_TEST_PROTO_MAX_PKES] = {0};
	bool saved_align128;
	size_t i;

	memset(out, 0, sizeof(*out));
	memset(&mock_mr, 0, sizeof(mock_mr));
	mock_mr.efa_mr.iface = iface;

	memset(&mock_txe, 0, sizeof(mock_txe));
	mock_txe.ep = efa_rdm_ep;
	mock_txe.peer = &mock_peer;
	mock_txe.total_len = total_len;
	mock_txe.iov_count = 1;
	mock_txe.iov[0].iov_base = NULL;
	mock_txe.iov[0].iov_len = total_len;
	mock_txe.desc[0] = &mock_mr;

	saved_align128 = efa_rdm_ep->sendrecv_in_order_aligned_128_bytes;
	saved_outstanding = efa_rdm_ep->efa_outstanding_tx_ops;
	efa_rdm_ep->sendrecv_in_order_aligned_128_bytes = !!align128;
	if (leave_one_tx_pkt)
		efa_rdm_ep->efa_outstanding_tx_ops =
			efa_rdm_ep->efa_max_outstanding_tx_ops - 1;

	out->ret = efa_rdm_proto_medium_plan_tx_pkes(efa_rdm_ep, &mock_txe,
						    EFA_RDM_MEDIUM_MSGRTM_PKT,
						    &pkt_entry_cnt, data_sizes);

	efa_rdm_ep->sendrecv_in_order_aligned_128_bytes = saved_align128;
	efa_rdm_ep->efa_outstanding_tx_ops = saved_outstanding;

	if (out->ret)
		return 0;

	if (pkt_entry_cnt > EFA_TEST_PROTO_MAX_PKES)
		return -FI_ETOOSMALL;

	out->pkt_entry_cnt = pkt_entry_cnt;
	for (i = 0; i < pkt_entry_cnt; ++i)
		out->data_sizes[i] = data_sizes[i];

	return 0;
}

/*
 * Select, fill and construct on a caller-owned txe, then snapshot the packets.
 * Packets are released here so the caller can construct again on the same txe.
 */
static int efa_test_proto_medium_build(struct efa_test_proto_ctx *ctx,
				       struct efa_rdm_ope *txe,
				       int assign_msg_id,
				       struct efa_test_proto_construct_result *out)
{
	struct efa_rdm_ep *ep = ctx->ep;
	struct efa_rdm_proto *proto = NULL;
	struct fi_msg msg = {0};
	struct iovec iov;
	void *desc;
	size_t i;
	int err;

	memset(out, 0, sizeof(*out));

	iov.iov_base = ctx->buf;
	iov.iov_len = ctx->len;
	desc = fi_mr_desc(ctx->mr);
	msg.msg_iov = &iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = ctx->peer_addr;

	if (assign_msg_id) {
		err = efa_rdm_proto_select_send_protocol(
			ep, ctx->peer, &msg, ofi_op_msg, 0, txe, &proto);
		if (err)
			return err;

		out->selected_medium = (proto == &efa_rdm_proto_medium);
		if (!out->selected_medium)
			return 0;
		out->txe_filled = 1;

		efa_rdm_proto_txe_fill(txe, ep, ctx->peer, &msg, ofi_op_msg, 0,
				       0, 0, proto);
		txe->msg_id = ctx->peer->next_msg_id++;
	} else {
		out->selected_medium = (txe->proto == &efa_rdm_proto_medium);
		proto = txe->proto;
	}

	/* No fi_msg, exactly as the repost path calls it. */
	out->ret = proto->construct_tx_pkes(ep, ctx->peer, NULL, txe->op,
					    txe->tag, txe->fi_flags,
					    txe->internal_flags, txe);
	if (out->ret)
		return 0;

	out->req_pkt_type = txe->req_pkt_type;
	out->total_len = txe->total_len;
	out->pke_cnt = ep->send_pkt_entry_vec_size;
	if (out->pke_cnt > EFA_TEST_PROTO_MAX_PKES)
		return -FI_ETOOSMALL;

	for (i = 0; i < out->pke_cnt; ++i) {
		struct efa_rdm_pke *pke = ep->send_pkt_entry_vec[i];
		struct efa_rdm_medium_rtm_base_hdr *hdr;

		if (pke->handle_pke ==
		    &efa_rdm_proto_medium_handle_rtm_send_completion)
			out->callbacks_set++;
		if (pke->ope == txe)
			out->ope_backrefs_set++;

		hdr = efa_rdm_pke_get_medium_rtm_base_hdr(pke);
		out->msg_lengths[i] = hdr->msg_length;
		out->seg_offsets[i] = hdr->seg_offset;
		out->payload_sizes[i] = pke->payload_size;
	}

	for (i = 0; i < out->pke_cnt; ++i)
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[i]);

	return 0;
}

int efa_test_proto_medium_construct(struct fid_ep *ep, struct fid_av *av,
				    struct fid_domain *domain,
				    struct efa_test_proto_construct_result *out)
{
	struct efa_test_proto_ctx ctx = {0};
	struct efa_rdm_ope *txe;
	int err;

	err = efa_test_proto_setup_peer(ep, av, &ctx);
	if (err)
		return err;

	err = efa_test_proto_setup_buf(domain, &ctx, EFA_TEST_PROTO_MEDIUM_LEN);
	if (err)
		return err;

	txe = ofi_buf_alloc(ctx.ep->base_ep.txe_pool);
	if (!txe) {
		efa_test_proto_teardown_buf(&ctx);
		return -FI_ENOMEM;
	}

	err = efa_test_proto_medium_build(&ctx, txe, 1, out);

	/*
	 * efa_rdm_txe_release() unlinks list entries that only
	 * efa_rdm_txe_construct_common() initialises, so a txe the build bailed
	 * out of before the fill goes straight back to the pool.
	 */
	if (out->txe_filled)
		efa_rdm_txe_release(txe);
	else
		ofi_buf_free(txe);
	efa_test_proto_teardown_buf(&ctx);
	return err;
}

int efa_test_proto_medium_construct_repost(
	struct fid_ep *ep, struct fid_av *av, struct fid_domain *domain,
	struct efa_test_proto_construct_result *first,
	struct efa_test_proto_construct_result *second)
{
	struct efa_test_proto_ctx ctx = {0};
	struct efa_rdm_ope *txe;
	int err;

	err = efa_test_proto_setup_peer(ep, av, &ctx);
	if (err)
		return err;

	err = efa_test_proto_setup_buf(domain, &ctx, EFA_TEST_PROTO_MEDIUM_LEN);
	if (err)
		return err;

	txe = ofi_buf_alloc(ctx.ep->base_ep.txe_pool);
	if (!txe) {
		efa_test_proto_teardown_buf(&ctx);
		return -FI_ENOMEM;
	}

	err = efa_test_proto_medium_build(&ctx, txe, 1, first);
	if (!err && !first->ret)
		err = efa_test_proto_medium_build(&ctx, txe, 0, second);

	/*
	 * The second attempt is the one that reaches the device, so it is also
	 * the only one that runs the post-send hook.
	 */
	if (!err && !second->ret) {
		txe->proto->handle_tx_pkes_posted(ctx.ep, txe);
		second->bytes_sent = txe->bytes_sent;
	}

	if (first->txe_filled)
		efa_rdm_txe_release(txe);
	else
		ofi_buf_free(txe);
	efa_test_proto_teardown_buf(&ctx);
	return err;
}

/*
 * fi_send a medium message and snapshot the packets it posted. The callbacks
 * release the packet they are given, so they must be read beforehand.
 */
static int efa_test_proto_medium_send(struct efa_test_proto_ctx *ctx,
				      struct fid_ep *ep,
				      struct efa_rdm_pke **pkes,
				      size_t *pke_cnt)
{
	size_t i;
	int err;

	err = fi_send(ep, ctx->buf, ctx->len, fi_mr_desc(ctx->mr),
		      ctx->peer_addr, NULL);
	if (err)
		return err;

	*pke_cnt = ctx->ep->send_pkt_entry_vec_size;
	if (*pke_cnt < 2 || *pke_cnt > EFA_TEST_PROTO_MAX_PKES)
		return -FI_ETOOSMALL;

	for (i = 0; i < *pke_cnt; ++i) {
		pkes[i] = ctx->ep->send_pkt_entry_vec[i];
		if (!pkes[i] || !pkes[i]->handle_pke)
			return -FI_EINVAL;
	}
	return 0;
}

int efa_test_proto_medium_completion(
	struct fid_ep *ep, struct fid_av *av, struct fid_domain *domain,
	struct efa_test_proto_completion_result *out)
{
	struct efa_test_proto_ctx ctx = {0};
	struct efa_rdm_pke *pkes[EFA_TEST_PROTO_MAX_PKES];
	struct efa_rdm_ope *txe;
	size_t i;
	int err;

	memset(out, 0, sizeof(*out));

	err = efa_test_proto_setup_peer(ep, av, &ctx);
	if (err)
		return err;

	err = efa_test_proto_setup_buf(domain, &ctx, EFA_TEST_PROTO_MEDIUM_LEN);
	if (err)
		return err;

	err = efa_test_proto_medium_send(&ctx, ep, pkes, &out->pke_cnt);
	if (err)
		goto out;

	out->ope_list_after_send = efa_test_proto_ope_list_count(ctx.ep);

	txe = pkes[0]->ope;
	if (!txe || txe->bytes_acked != 0) {
		err = -FI_EINVAL;
		goto out;
	}
	out->total_len = txe->total_len;

	for (i = 0; i < out->pke_cnt; ++i) {
		out->payload_sizes[i] = pkes[i]->payload_size;

		efa_rdm_ep_record_tx_op_completed(ctx.ep, pkes[i]);
		pkes[i]->handle_pke(pkes[i]);

		/*
		 * The last callback releases the txe, so read bytes_acked from
		 * it only while a packet still holds a reference.
		 */
		out->bytes_acked_after[i] =
			(i + 1 < out->pke_cnt) ? txe->bytes_acked : 0;
		out->ope_list_after[i] = efa_test_proto_ope_list_count(ctx.ep);
	}

out:
	efa_test_proto_teardown_buf(&ctx);
	return err;
}

int efa_test_proto_medium_peer_abort(
	struct fid_ep *ep, struct fid_av *av, struct fid_cq *cq,
	struct fid_domain *domain,
	struct efa_test_proto_peer_abort_result *out)
{
	static struct efa_rdm_mr stale_source_mr;
	struct efa_test_proto_ctx ctx = {0};
	struct efa_rdm_pke *pkes[EFA_TEST_PROTO_MAX_PKES];
	struct fi_cq_err_entry err_entry;
	struct efa_rdm_ope *txe;
	size_t i;
	int err;

	memset(out, 0, sizeof(*out));

	err = efa_test_proto_setup_peer(ep, av, &ctx);
	if (err)
		return err;

	/* The peer must advertise PEER_ERROR support or the emit is skipped. */
	ctx.peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_PEER_ERROR;

	err = efa_test_proto_setup_buf(domain, &ctx, EFA_TEST_PROTO_MEDIUM_LEN);
	if (err)
		return err;

	err = efa_test_proto_medium_send(&ctx, ep, pkes, &out->pke_cnt);
	if (err)
		goto out;

	txe = pkes[0]->ope;
	if (!txe) {
		err = -FI_EINVAL;
		goto out;
	}
	out->req_pkt_type_is_rtm = efa_rdm_pkt_type_is_rtm(txe->req_pkt_type);

	/*
	 * The application closes the source MR while the send is in flight: the
	 * MR's generation no longer matches the snapshot taken at dispatch.
	 */
	stale_source_mr.gen = 1;
	txe->iov_count = 1;
	txe->desc[0] = &stale_source_mr;
	txe->desc_gen[0] = 2;

	efa_rdm_txe_handle_error(txe, FI_ECANCELED, FI_EFA_ERR_PKT_POST);
	out->abort_pending_after_error =
		!!(txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING);
	out->emitted_after_error =
		!!(txe->internal_flags &
		   EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);
	memset(&err_entry, 0, sizeof(err_entry));
	out->readerr_after_error = fi_cq_readerr(cq, &err_entry, 0);

	/*
	 * Every in-flight data WR now completes successfully. Only the last one
	 * drains the txe, so only it may emit.
	 */
	for (i = 0; i < out->pke_cnt; ++i) {
		efa_rdm_ep_record_tx_op_completed(ctx.ep, pkes[i]);
		pkes[i]->handle_pke(pkes[i]);

		out->emitted_after[i] =
			!!(txe->internal_flags &
			   EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED);
		memset(&err_entry, 0, sizeof(err_entry));
		out->readerr_after[i] = fi_cq_readerr(cq, &err_entry, 0);
	}

	out->ope_list_after_data = efa_test_proto_ope_list_count(ctx.ep);
	memset(&err_entry, 0, sizeof(err_entry));
	out->readerr_after_data = fi_cq_readerr(cq, &err_entry, 0);

	/*
	 * The PEER_ERROR_PKT's own send completion releases the txe and writes
	 * the single peer-abort error completion. Emitting it reused
	 * ep->send_pkt_entry_vec, so its packet is at slot 0.
	 */
	efa_rdm_pke_handle_send_completion(ctx.ep->send_pkt_entry_vec[0]);

	out->ope_list_final = efa_test_proto_ope_list_count(ctx.ep);
	memset(&err_entry, 0, sizeof(err_entry));
	out->readerr_final = fi_cq_readerr(cq, &err_entry, 0);
	if (out->readerr_final == 1) {
		out->final_err = err_entry.err;
		out->final_prov_errno = err_entry.prov_errno;
	}

out:
	efa_test_proto_teardown_buf(&ctx);
	return err;
}
