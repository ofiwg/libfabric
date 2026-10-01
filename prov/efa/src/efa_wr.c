/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_data_path_ops.h"
#include "efa_io_defs.h"
#include "efa_wr.h"

/**
 * @brief Initiate transmit work the provider has deferred
 *
 * Completes the work request chain of sends posted with FI_MORE, or on the
 * direct data path rings the send queue doorbell for the entries batched so
 * far.
 *
 * @param flags	reserved, must be 0
 */
ssize_t efa_wr_tx_flush(struct fid_ep *ep_fid, uint64_t flags)
{
	struct efa_base_ep *base_ep;
	struct efa_qp *qp;
	int err = 0;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	qp = base_ep->qp;

	ofi_genlock_lock(&base_ep->util_ep.lock);

#if HAVE_EFA_DATA_PATH_DIRECT
	if (qp->data_path_direct_enabled) {
		if (qp->data_path_direct_qp.sq.num_wqe_pending)
			efa_data_path_direct_send_wr_ring_db(&qp->data_path_direct_qp.sq);
		ofi_genlock_unlock(&base_ep->util_ep.lock);
		return 0;
	}
#endif
	if (qp->base_ep->is_wr_started) {
		qp->base_ep->is_wr_started = false;
		err = ibv_wr_complete(qp->ibv_qp_ex);
	}
	ofi_genlock_unlock(&base_ep->util_ep.lock);

	return err ? -err : 0;
}

/**
 * @brief Initiate receive work the provider has deferred
 *
 * Posts the receive work request chain accumulated by fi_recv with FI_MORE.
 * The receive queue has no deferred-doorbell state of its own: the underlying
 * post always rings the RQ doorbell, so once the staged chain is posted the
 * receives are fully initiated.
 *
 * @param flags	reserved, must be 0
 */
ssize_t efa_wr_rx_flush(struct fid_ep *ep_fid, uint64_t flags)
{
	struct efa_base_ep *base_ep;
	struct ibv_recv_wr *bad_wr;
	int err = 0;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);

	ofi_genlock_lock(&base_ep->util_ep.lock);

	if (base_ep->recv_wr_index > 0) {
		err = efa_qp_post_recv(base_ep->qp,
				       &base_ep->efa_recv_wr_vec[0].wr, &bad_wr);
		if (OFI_UNLIKELY(err))
			err = (err == ENOMEM) ? -FI_EAGAIN : -err;
		base_ep->recv_wr_index = 0;
	}

	ofi_genlock_unlock(&base_ep->util_ep.lock);

	return err;
}
