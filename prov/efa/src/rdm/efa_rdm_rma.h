/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#if HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */

#ifndef _EFA_RDM_RMA_H_
#define _EFA_RDM_RMA_H_

#include <rdma/fi_rma.h>

int efa_rdm_rma_verified_copy_iov(struct efa_rdm_ep *ep, struct efa_rma_iov *rma,
			      size_t count, uint32_t flags,
			      struct iovec *iov, void **desc);

extern struct fi_ops_rma efa_rdm_rma_ops;

struct efa_rdm_ope *
efa_rdm_rma_alloc_txe(struct efa_rdm_ep *efa_rdm_ep,
		      struct efa_rdm_peer *peer,
		      const struct fi_msg_rma *msg_rma,
		      uint32_t op,
		      uint64_t fi_flags,
		      uint32_t internal_flags);

ssize_t efa_rdm_rma_post_write(struct efa_rdm_ep *ep, struct efa_rdm_ope *txe);

ssize_t efa_rdm_rma_post_read(struct efa_rdm_ep *ep, struct efa_rdm_ope *txe);

/**
 * @brief Determine if one-sided fi_read should use device RDMA read.
 *
 * For one-sided fi_read, there is no NACK fallback. The sender must
 * know upfront whether the peer's buffers are NIC-accessible (p2p).
 *
 * @param[in] ep	Endpoint
 * @param[in] peer	Peer to read from
 * @param[in] use_p2p	Whether local p2p is available
 * @return bool		Whether device RDMA read should be used
 */
static inline
bool efa_rdm_rma_should_read_using_rdma(struct efa_rdm_ep *ep,
					struct efa_rdm_peer *peer,
					bool use_p2p)
{
	return efa_rdm_interop_rdma_read(ep, peer) &&
	       efa_both_support_p2p(ep, peer, use_p2p);
}

static inline
bool efa_rdm_rma_should_write_using_rdma(struct efa_rdm_ep *ep, struct efa_rdm_ope *txe,
					  struct efa_rdm_peer *peer, bool use_p2p)
{
	if (txe->fi_flags & FI_REMOTE_CQ_DATA) {
		/*
		 * Because EFA is unordered and EFA iov descriptions can be more
		 * expressive than the IBV sge's, we only implement
		 * FI_REMOTE_CQ_DATA using RDMA_WRITE_WITH_IMM when a single iov
		 * is given, otherwise we use sends to emulate it.
		 */
		if (txe->iov_count > 1 || txe->rma_iov_count > 1)
			return false;
		/**
		 * Unsolicited write recv is a feature that makes rdma-write with
		 * imm not consume an rx buffer on the responder side, and this
		 * feature requires consistent support status on both sides.
		 */
		if (!ep->homogeneous_peers &&
		    (efa_rdm_ep_support_unsolicited_write_recv(ep) != efa_rdm_peer_support_unsolicited_write_recv(peer))) {
			char err_msg[EFA_ERROR_MSG_BUFFER_LENGTH] = {0};
			(void) efa_rdm_construct_msg_with_local_and_peer_information(ep, peer, err_msg, "", EFA_ERROR_MSG_BUFFER_LENGTH);
			EFA_INFO(FI_LOG_EP_DATA,
				"Inconsistent support status detected on unsolicited write recv.\n"
				"My support status: %d, peer support status: %d. %s.\n"
				"This is usually caused by inconsistent efa driver, libfabric, or rdma-core versions.\n"
				"Libfabric will use emulated write protocols and the performance is suboptimal.\n"
				"Please use consistent software versions on both hosts, or disable the unsolicited write "
				"recv feature by setting environment variable FI_EFA_USE_UNSOLICITED_WRITE_RECV=0\n",
				ep->base_ep.qp->unsolicited_write_recv_enabled, efa_rdm_peer_support_unsolicited_write_recv(peer),
				err_msg);
			return false;
		}
	}

	return efa_both_support_rdma_write(ep, peer) &&
	       efa_both_support_p2p(ep, peer, use_p2p);
}

#endif
