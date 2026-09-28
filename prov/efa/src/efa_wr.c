/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_data_path_ops.h"
#include "efa_io_defs.h"
#include "efa_wr.h"
#include <rdma/fi_wr.h>

/**
 * @brief Initiate transmit work the provider has deferred
 *
 * Completes the work request chain of sends posted with FI_MORE, or on the
 * direct data path rings the send queue doorbell for the entries batched so
 * far.
 *
 * @param flags	reserved, must be 0
 *
 * @return 0 on success, or a negative fabric errno on fatal failure.
 */
int efa_wr_tx_flush(struct fid_ep *ep_fid, uint64_t flags)
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
		/*
		 * No error is expected here: ibv_wr_complete does not fail for
		 * the doorbell write, and the work requests were already
		 * validated when they were posted.
		 */
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
 *
 * @return 0 on success, or a negative fabric errno on fatal failure.
 */
int efa_wr_rx_flush(struct fid_ep *ep_fid, uint64_t flags)
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

static int efa_wr_prepare(struct fid_ep *ep_fid, const struct fi_wr_attr *attr,
			  fi_wr wr, size_t *wr_len)
{
	return -FI_ENOSYS;
}

static ssize_t efa_wr_queue_tx(struct fid_ep *ep_fid, const fi_wr wr,
			       void *context)
{
	return -FI_ENOSYS;
}

static ssize_t efa_wr_queue_recv(struct fid_ep *ep_fid, const fi_wr wr,
				 void *context)
{
	return -FI_ENOSYS;
}

static ssize_t efa_wr_queue_trecv(struct fid_ep *ep_fid, const fi_wr wr,
				  void *context)
{
	return -FI_ENOSYS;
}

static int efa_wr_modify_addr(struct fid_ep *ep_fid, fi_wr wr,
			      enum fi_op_type op_type, fi_addr_t addr)
{
	return -FI_ENOSYS;
}

static int efa_wr_modify_iov(struct fid_ep *ep_fid, fi_wr wr,
			     enum fi_op_type op_type, const struct iovec *iov,
			     void **desc, size_t count)
{
	return -FI_ENOSYS;
}

static int efa_wr_modify_rma_iov(struct fid_ep *ep_fid, fi_wr wr,
				 enum fi_op_type op_type,
				 const struct fi_rma_iov *rma_iov, size_t count)
{
	return -FI_ENOSYS;
}

static int efa_wr_modify_tag(struct fid_ep *ep_fid, fi_wr wr,
			     enum fi_op_type op_type, uint64_t tag,
			     uint64_t ignore)
{
	return -FI_ENOSYS;
}

static int efa_wr_modify_data(struct fid_ep *ep_fid, fi_wr wr,
			      enum fi_op_type op_type, uint64_t data)
{
	return -FI_ENOSYS;
}

static int efa_wr_modify_flags(struct fid_ep *ep_fid, fi_wr wr,
			       enum fi_op_type op_type, uint64_t flags)
{
	return -FI_ENOSYS;
}

struct fi_ops_wr efa_wr_ops = {
	.size = sizeof(struct fi_ops_wr),
	.prepare = efa_wr_prepare,
	.queue_tx = efa_wr_queue_tx,
	.queue_recv = efa_wr_queue_recv,
	.queue_trecv = efa_wr_queue_trecv,
	.modify_addr = efa_wr_modify_addr,
	.modify_iov = efa_wr_modify_iov,
	.modify_rma_iov = efa_wr_modify_rma_iov,
	.modify_tag = efa_wr_modify_tag,
	.modify_data = efa_wr_modify_data,
	.modify_flags = efa_wr_modify_flags,
};

size_t efa_wr_tx_size(void)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	return sizeof(struct efa_io_tx_wqe_128);
#else
	return 0;
#endif
}

size_t efa_wr_rx_size(size_t num_sge)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	return num_sge * sizeof(struct efa_io_rx_desc);
#else
	return 0;
#endif
}
