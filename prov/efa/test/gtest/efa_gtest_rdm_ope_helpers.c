/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_rdm_ope_helpers.h"
#include "efa_gtest_common_helpers.h"
#include "efa.h"
#include "efa_av.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_domain.h"
#include "rdm/efa_rdm_ope.h"
#include "rdm/efa_rdm_pke.h"
#include "rdm/efa_rdm_cq.h"
#include "rdm/efa_rdm_peer.h"
#include "rdm/efa_rdm_protocol.h"

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
	struct efa_rdm_domain *rdm_domain = efa_rdm_ep_rdm_domain(ep);
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
	qop->peer->conn->shm_fi_addr = FI_ADDR_NOTAVAIL;

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

	if (dlist_empty(&rdm_domain->ope_queued_list))
		return -FI_EINVAL;
	qop->txe = container_of(rdm_domain->ope_queued_list.next,
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

	return efa_rdm_ope_process_queued_ope(qop->txe,
					      EFA_RDM_OPE_QUEUED_BEFORE_HANDSHAKE);
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
