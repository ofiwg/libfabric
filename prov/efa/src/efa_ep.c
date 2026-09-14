/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved. */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "config.h"
#include "efa.h"
#include "efa_av.h"
#include "efa_cq.h"

#include <infiniband/efadv.h>

extern struct fi_ops_msg efa_msg_ops;
extern struct fi_ops_rma efa_rma_ops;

/**
 * @brief Query the inline-data size actually available on this endpoint.
 *
 * When completion-with-signal is enabled the QP uses wide WQEs whose signal
 * feature blocks reduce the inline region below the device-advertised
 * inline_buf_size_ex. This queries the effective inline size for the endpoint's
 * QP configuration via the efadv_get_max_inline_data verb and caps it by
 * @p configured (the endpoint's msg or rma inject size). Falls back to
 * @p configured when the verb is unavailable or signals are not enabled.
 *
 * @param[in]  ep		efa base endpoint
 * @param[in]  configured	the endpoint's configured inject size to cap by
 *				(ep->inject_msg_size or ep->inject_rma_size)
 * @param[out] inline_size	effective inline size in bytes
 * @return 0 on success, negative fi errno on failure
 */
static int efa_ep_query_inline_size(struct efa_base_ep *ep, size_t configured,
				    size_t *inline_size)
{
	uint32_t qp_flags = 0;
	uint32_t wr_flags = 0;
	int ret;

#if HAVE_EFADV_COMP_SIGNAL
#if HAVE_CAPS_UNSOLICITED_WRITE_RECV
	if (ep->use_unsolicited_write_recv)
		qp_flags |= EFADV_QP_FLAGS_UNSOLICITED_WRITE_RECV;
#endif
	if (ep->info->tx_attr->inject_size >
	    ep->domain->device->efa_attr.inline_buf_size &&
	    efa_device_support_rdma_write())
		qp_flags |= EFADV_QP_FLAGS_INLINE_WRITE;

	if (ep->comp_signal_enabled)
		wr_flags = EFADV_WR_EX_WITH_COMP_SIGNAL |
			   EFADV_WR_EX_WITH_COMP_SIGNAL_WITH_DATA;
#endif

	/*
	 * Query with the QP's actual flags. Returns the max inline (>= 0), or a
	 * negative errno (incl. -FI_ENOSYS on builds without efadv comp-signal
	 * support), in which case fall back to the configured inject size.
	 */
	ret = efa_query_max_inline_data(ep->domain->device->ibv_ctx, qp_flags,
					wr_flags);
	if (ret < 0) {
		*inline_size = configured;
		return FI_SUCCESS;
	}

	/* The effective inject size is capped by the endpoint configuration. */
	*inline_size = MIN((size_t) ret, configured);
	return FI_SUCCESS;
}

/**
 * @brief Query the send-queue depth (tx size) actually available on this endpoint.
 *
 * Wide (128-byte) WQEs consume more send-queue memory per entry, so the maximum
 * SQ depth is lower than the device-advertised default. A WQE is wide when the
 * endpoint uses a large inline (inject) size (> the device's inline_buf_size)
 * and/or has completion-with-signal enabled. This queries the effective SQ
 * depth for the endpoint's actual configuration via efadv_get_max_sq_depth and
 * caps it by the device-advertised tx queue depth (info->tx_attr->size, i.e.
 * max_sq_wr), matching the libfabric fi_getopt contract.
 *
 * The inject-size reduction is already reflected in info->tx_attr->size at
 * fi_getinfo (the inject size is known from hints then). The signal reduction
 * is not: signals are a per-endpoint opt-in applied after fi_getinfo, so it is
 * only observable here. Falls back to the device-advertised tx size when no
 * wide-WQE feature applies, the verb is unavailable, or on a build without the
 * required efadv support.
 *
 * @param[in]  ep		efa base endpoint
 * @param[out] tx_size	effective send-queue depth (number of entries)
 * @return 0 on success, negative fi errno on failure
 */
static int efa_ep_query_tx_size(struct efa_base_ep *ep, size_t *tx_size)
{
	size_t configured = ep->info->tx_attr->size;

#if HAVE_INLINE_BUF_SIZE_EX
	uint32_t sq_flags = 0;
	uint32_t qp_flags = 0;
	uint32_t max_inline = (uint32_t) ep->info->tx_attr->inject_size;
	int ret;

#if HAVE_CAPS_UNSOLICITED_WRITE_RECV
	if (ep->use_unsolicited_write_recv)
		qp_flags |= EFADV_QP_FLAGS_UNSOLICITED_WRITE_RECV;
#endif

	/* Large inline data uses wide WQEs (mirrors the QP-creation flag). */
	if (ep->info->tx_attr->inject_size >
	    ep->domain->device->efa_attr.inline_buf_size &&
	    efa_device_support_rdma_write()) {
		sq_flags |= EFADV_SQ_DEPTH_ATTR_INLINE_WRITE;
		qp_flags |= EFADV_QP_FLAGS_INLINE_WRITE;
	}

#if HAVE_EFADV_COMP_SIGNAL
	if (ep->comp_signal_enabled) {
		int signal_inline;

		sq_flags |= EFADV_SQ_DEPTH_ATTR_COMP_SIGNAL;

		/*
		 * With signals the QP's inline size is the signal-adjusted
		 * value, so query it (with the QP's actual flags) to feed the
		 * SQ-depth query, mirroring QP creation. A failure leaves the
		 * configured inject size.
		 */
		signal_inline = efa_query_max_inline_data(
			ep->domain->device->ibv_ctx, qp_flags,
			EFADV_WR_EX_WITH_COMP_SIGNAL |
				EFADV_WR_EX_WITH_COMP_SIGNAL_WITH_DATA);
		if (signal_inline >= 0)
			max_inline = (uint32_t) signal_inline;
	}
#endif

	/* No wide-WQE feature: the device-advertised tx size already applies. */
	if (!sq_flags) {
		*tx_size = configured;
		return FI_SUCCESS;
	}

	/* Returns the max SQ depth (>= 0) or a negative errno. */
	ret = efa_query_max_sq_depth(ep->domain->device->ibv_ctx, sq_flags,
				     max_inline);
	if (ret < 0) {
		EFA_INFO(FI_LOG_EP_CTRL,
			 "efadv_get_max_sq_depth failed (%d); reporting "
			 "device-advertised tx size\n", ret);
		*tx_size = configured;
		return FI_SUCCESS;
	}

	/* The effective tx size is capped by the device-advertised maximum. */
	*tx_size = MIN((size_t) ret, configured);
	return FI_SUCCESS;
#else
	*tx_size = configured;
	return FI_SUCCESS;
#endif
}

static int efa_ep_getopt(fid_t fid, int level, int optname,
			 void *optval, size_t *optlen)
{
	struct efa_base_ep *ep;

	ep = container_of(fid, struct efa_base_ep, util_ep.ep_fid.fid);

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_EFA_RNR_RETRY:
		if (*optlen < sizeof(size_t))
			return -FI_ETOOSMALL;
		*(size_t *)optval = ep->rnr_retry;
		*optlen = sizeof(size_t);
		break;
	/* p2p is required for efa direct ep */
	case FI_OPT_FI_HMEM_P2P:
		if (*optlen < sizeof(int))
			return -FI_ETOOSMALL;
		*(int *)optval = FI_HMEM_P2P_REQUIRED;
		*optlen = sizeof(int);
		break;
	case FI_OPT_MAX_MSG_SIZE:
		if (*optlen < sizeof (size_t))
			return -FI_ETOOSMALL;
		*(size_t *) optval = ep->max_msg_size;
		*optlen = sizeof (size_t);
		break;
	case FI_OPT_MAX_RMA_SIZE:
		if (*optlen < sizeof (size_t))
			return -FI_ETOOSMALL;
		*(size_t *) optval = ep->max_rma_size;
		*optlen = sizeof (size_t);
		break;
	case FI_OPT_INJECT_MSG_SIZE:
		if (*optlen < sizeof (size_t))
			return -FI_ETOOSMALL;
		efa_ep_query_inline_size(ep, ep->inject_msg_size,
					 (size_t *) optval);
		*optlen = sizeof (size_t);
		break;
	case FI_OPT_INJECT_RMA_SIZE:
		if (*optlen < sizeof (size_t))
			return -FI_ETOOSMALL;
		efa_ep_query_inline_size(ep, ep->inject_rma_size,
					 (size_t *) optval);
		*optlen = sizeof (size_t);
		break;
	case FI_OPT_TX_SIZE:
		if (*optlen < sizeof (size_t))
			return -FI_ETOOSMALL;
		efa_ep_query_tx_size(ep, (size_t *) optval);
		*optlen = sizeof (size_t);
		break;
	/* Emulated read/write is NOT used for efa direct ep */
	case FI_OPT_EFA_EMULATED_READ: /* fall through */
	case FI_OPT_EFA_EMULATED_WRITE:
		if (*optlen < sizeof(bool))
			return -FI_ETOOSMALL;
		*(bool *)optval = false;
		*optlen = sizeof(bool);
		break;
	default:
		EFA_INFO(FI_LOG_EP_CTRL, "Unknown / unsupported endpoint option\n");
		return -FI_ENOPROTOOPT;
	}

	return FI_SUCCESS;
}

static int efa_ep_setopt(fid_t fid, int level, int optname, const void *optval, size_t optlen)
{
	int ret, intval;
	struct efa_base_ep *ep;

	ep = container_of(fid, struct efa_base_ep, util_ep.ep_fid.fid);

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_EFA_RNR_RETRY:
		if (optlen != sizeof(size_t))
			return -FI_EINVAL;

		/*
		 * Application is required to call to fi_setopt before EP
		 * enabled. If it's calling to fi_setopt after EP enabled,
		 * fail the call.
		 *
		 * efa_ep->qp will be NULL before EP enabled, use it to check
		 * if the call to fi_setopt is before or after EP enabled for
		 * convience, instead of calling to ibv_query_qp
		 */
		if (ep->efa_qp_enabled) {
			EFA_WARN(FI_LOG_EP_CTRL,
				"The option FI_OPT_EFA_RNR_RETRY is required "
				"to be set before EP enabled\n");
			return -FI_EINVAL;
		}

		if (!efa_domain_support_rnr_retry_modify(ep->domain)) {
			EFA_WARN(FI_LOG_EP_CTRL,
				"RNR capability is not supported\n");
			return -FI_ENOSYS;
		}
		ep->rnr_retry = *(size_t *)optval;
		break;
	case FI_OPT_FI_HMEM_P2P:
		if (optlen != sizeof(int))
			return -FI_EINVAL;

		intval = *(int *)optval;

		if (intval == FI_HMEM_P2P_DISABLED) {
			EFA_WARN(FI_LOG_EP_CTRL, "p2p is required by implementation\n");
			return -FI_EOPNOTSUPP;
		}
		break;
	case FI_OPT_MAX_MSG_SIZE:
		EFA_EP_SETOPT_THRESHOLD(MAX_MSG_SIZE, ep->max_msg_size, (size_t) ep->domain->device->ibv_port_attr.max_msg_sz)
		break;
	case FI_OPT_MAX_RMA_SIZE:
		EFA_EP_SETOPT_THRESHOLD(MAX_RMA_SIZE, ep->max_rma_size, (size_t) ep->domain->device->max_rdma_size)
		break;
	case FI_OPT_INJECT_MSG_SIZE:
		EFA_EP_SETOPT_THRESHOLD(INJECT_MSG_SIZE, ep->inject_msg_size, (size_t) ep->info->tx_attr->inject_size)
		break;
	case FI_OPT_INJECT_RMA_SIZE:
		EFA_EP_SETOPT_THRESHOLD(INJECT_RMA_SIZE, ep->inject_rma_size, ep->inject_rma_size)
		break;
	/* no op as efa direct ep will not use cuda api and shm in data transfer */
	case FI_OPT_CUDA_API_PERMITTED: /* fall through */
	case FI_OPT_SHARED_MEMORY_PERMITTED:
		break;
	/* no op as efa direct ep will always use rdma for rma operations in data transfer */
	case FI_OPT_EFA_USE_DEVICE_RDMA:
		if (optlen != sizeof(bool))
			return -FI_EINVAL;
		if (!(*(bool *)optval) && (ep->info->caps & FI_RMA)) {
			EFA_WARN(FI_LOG_EP_CTRL, "Device rdma is required for rma operations\n");
			return -FI_EOPNOTSUPP;
		}
		break;
	case FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES:
		if (optlen != sizeof(bool))
			return -FI_EINVAL;
		if (*(bool *)optval) {
			ret = efa_base_ep_check_qp_in_order_aligned_128_bytes(ep, IBV_WR_SEND);
			if (ret)
				return ret;
		}
		break;
	case FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES:
		if (optlen != sizeof(bool))
			return -FI_EINVAL;
		if (*(bool *)optval) {
			ret = efa_base_ep_check_qp_in_order_aligned_128_bytes(ep, IBV_WR_RDMA_WRITE);
			if (ret)
				return ret;
		}
		break;
	/* no op as efa direct ep will not handshake with peers */
	case FI_OPT_EFA_HOMOGENEOUS_PEERS:
		break;
	case FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV:
		if (optlen != sizeof(bool))
			return -FI_EINVAL;
		if (!(ep->info->mode & FI_RX_CQ_DATA) && !*(bool *)optval) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "FI_RX_CQ_DATA is required when unsolicited "
				 "write recv is disabled.\n");
			return -FI_EOPNOTSUPP;
		}
		ep->use_unsolicited_write_recv = *(bool *)optval;
		break;
	case FI_OPT_EFA_COMP_SIGNAL:
		if (optlen != sizeof(bool))
			return -FI_EINVAL;
		/*
		 * Signal support governs how the send queue is allocated
		 * (wide 128-byte WQEs), so it must be set before the endpoint
		 * is enabled (before the QP is created).
		 */
		if (ep->efa_qp_enabled) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "The option FI_OPT_EFA_COMP_SIGNAL is required "
				 "to be set before EP enabled\n");
			return -FI_EINVAL;
		}
		if (*(bool *)optval && !efa_device_support_comp_signal()) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "Completion with signal is not supported by "
				 "the device\n");
			return -FI_EOPNOTSUPP;
		}
		ep->comp_signal_enabled = *(bool *)optval;
		break;
	default:
		EFA_INFO(FI_LOG_EP_CTRL, "Unknown / unsupported endpoint option\n");
		return -FI_ENOPROTOOPT;
	}

	return FI_SUCCESS;
}

static struct fi_ops_ep efa_ep_base_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_no_cancel,
	.getopt = efa_ep_getopt,
	.setopt = efa_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static int efa_ep_close(fid_t fid)
{
	struct efa_base_ep *ep;
	int ret;

	ep = container_of(fid, struct efa_base_ep, util_ep.ep_fid.fid);

	/* We need to free the util_ep first to avoid race conditions
	 * with other threads progressing the cntr. */
	efa_base_ep_close_util_ep(ep);

	efa_base_ep_remove_cntr_ibv_cq_poll_list(ep);

	ret = efa_base_ep_destruct(ep);
	if (ret) {
		EFA_WARN(FI_LOG_EP_CTRL, "Unable to close base endpoint\n");
	}
	if (efa_env.track_mr)
		efa_direct_ope_pool_destroy(ep);
	free(ep);

	return 0;
}

/**
 * @brief Commit the endpoint's FI_CONTEXT2 mode onto a CQ it is binding to
 *
 * All endpoints sharing a CQ must agree on FI_CONTEXT2 mode: the completion
 * path interprets wr_id differently for each mode, so a single CQ cannot serve
 * both. The first endpoint bound commits the CQ's mode; a later endpoint of a
 * different mode is rejected. The mode is never reset (a late completion could
 * otherwise dereference a context the new mode doesn't own).
 *
 * @param ep		efa base endpoint
 * @param ibv_cq	CQ being bound
 * @return 0 on success, -FI_EOPNOTSUPP if the endpoint's mode conflicts with a
 *         mode already committed on the CQ
 */
static int efa_ep_commit_cq_context_mode(struct efa_base_ep *ep,
					 struct efa_ibv_cq *ibv_cq)
{
	if (ibv_cq->context_mode != UNASSIGNED &&
	    ibv_cq->context_mode != ep->context_mode) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "CQ is already in use by an endpoint with a different "
			 "FI_CONTEXT2 mode\n");
		return -FI_EINVAL;
	}
	ibv_cq->context_mode = ep->context_mode;
	return 0;
}

static int efa_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct efa_base_ep *ep;
	struct efa_cq *cq;
	struct efa_av *av;
	struct efa_domain *efa_domain;
	struct util_eq *eq;
	struct util_cntr *cntr;
	int ret;

	ep = container_of(fid, struct efa_base_ep, util_ep.ep_fid.fid);
	ret = ofi_ep_bind_valid(&efa_prov, bfid, flags);
	if (ret)
		return ret;

	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		/* Must bind a CQ to either RECV or SEND completions */
		if (!(flags & (FI_RECV | FI_TRANSMIT)))
			return -FI_EBADFLAGS;

		cq = container_of(bfid, struct efa_cq, util_cq.cq_fid);
		efa_domain = container_of(cq->util_cq.domain, struct efa_domain, util_domain);
		if (ep->domain != efa_domain)
			return -FI_EINVAL;

		/*
		 * Selective completion requires the provider to suppress
		 * completions per-operation, which efa-direct implements via the
		 * caller-supplied context buffer. Without FI_CONTEXT2 there is no
		 * such buffer, so selective completion cannot be honored.
		 */
		if ((flags & FI_SELECTIVE_COMPLETION) &&
		    ep->context_mode != USE_CONTEXT2) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "FI_SELECTIVE_COMPLETION requires FI_CONTEXT2 "
				 "for efa-direct endpoints\n");
			return -FI_EOPNOTSUPP;
		}

		ret = efa_ep_commit_cq_context_mode(ep, &cq->ibv_cq);
		if (ret)
			return ret;

		ret = ofi_ep_bind_cq(&ep->util_ep, &cq->util_cq, flags);
		if (ret)
			return ret;

		break;
	case FI_CLASS_AV:
		av = container_of(bfid, struct efa_av, util_av.av_fid.fid);
		/* Bind util provider endpoint and av */
		ret = ofi_ep_bind_av(&ep->util_ep, &av->util_av);
		if (ret)
			return ret;

		ret = efa_base_ep_bind_av(ep, av);
		if (ret)
			return ret;
		break;
	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct util_cntr, cntr_fid.fid);

		ret = ofi_ep_bind_cntr(&ep->util_ep, cntr, flags);
		if (ret)
			return ret;
		break;
	case FI_CLASS_EQ:
		eq = container_of(bfid, struct util_eq, eq_fid.fid);

		ret = ofi_ep_bind_eq(&ep->util_ep, eq);
		if (ret)
			return ret;
		break;
	default:
		EFA_WARN(FI_LOG_EP_CTRL, "invalid fid class\n");
		return -EINVAL;
	}

	return 0;
}

static int efa_ep_getflags(struct fid_ep *ep_fid, uint64_t *flags)
{
	struct efa_base_ep *ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	struct fi_tx_attr *tx_attr = ep->info->tx_attr;
	struct fi_rx_attr *rx_attr = ep->info->rx_attr;

	if ((*flags & FI_TRANSMIT) && (*flags & FI_RECV)) {
		EFA_WARN(FI_LOG_EP_CTRL, "Both Tx/Rx flags cannot be specified\n");
		return -FI_EINVAL;
	} else if (tx_attr && (*flags & FI_TRANSMIT)) {
		*flags = tx_attr->op_flags;
	} else if (rx_attr && (*flags & FI_RECV)) {
		*flags = rx_attr->op_flags;
	} else {
		EFA_WARN(FI_LOG_EP_CTRL, "Tx/Rx flags not specified\n");
		return -FI_EINVAL;
	}
	return 0;
}

static int efa_ep_setflags(struct fid_ep *ep_fid, uint64_t flags)
{
	struct efa_base_ep *ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	struct fi_tx_attr *tx_attr = ep->info->tx_attr;
	struct fi_rx_attr *rx_attr = ep->info->rx_attr;

	if ((flags & FI_TRANSMIT) && (flags & FI_RECV)) {
		EFA_WARN(FI_LOG_EP_CTRL, "Both Tx/Rx flags cannot be specified.\n");
		return -FI_EINVAL;
	} else if (tx_attr && (flags & FI_TRANSMIT)) {
		tx_attr->op_flags = flags;
		tx_attr->op_flags &= ~FI_TRANSMIT;
	} else if (rx_attr && (flags & FI_RECV)) {
		rx_attr->op_flags = flags;
		rx_attr->op_flags &= ~FI_RECV;
	} else {
		EFA_WARN(FI_LOG_EP_CTRL, "Tx/Rx flags not specified\n");
		return -FI_EINVAL;
	}

	return 0;
}

static int efa_ep_enable(struct fid_ep *ep_fid)
{
	struct efa_base_ep *base_ep;
	int err;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);

	/* No context2 means no context buffer, which the mr tracking
	 * builds on top of
	 */
	if (efa_env.track_mr && base_ep->context_mode != USE_CONTEXT2) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "FI_EFA_TRACK_MR is not supported for efa-direct "
			 "endpoints without FI_CONTEXT2\n");
		return -FI_EOPNOTSUPP;
	}

	err = efa_base_ep_create_and_enable_qp(base_ep);
	if (err)
		return err;

	err = efa_base_ep_insert_cntr_ibv_cq_poll_list(base_ep);
	if (err) {
		efa_base_ep_destruct_qp(base_ep);
		return err;
	}

	if (efa_env.track_mr)
		err = efa_direct_ope_pool_create(base_ep);

	return err;
}

static int efa_ep_control(struct fid *fid, int command, void *arg)
{
	struct fid_ep *ep_fid;

	switch (fid->fclass) {
	case FI_CLASS_EP:
		ep_fid = container_of(fid, struct fid_ep, fid);
		switch (command) {
		case FI_GETOPSFLAG:
			return efa_ep_getflags(ep_fid, (uint64_t *)arg);
		case FI_SETOPSFLAG:
			return efa_ep_setflags(ep_fid, *(uint64_t *)arg);
		case FI_ENABLE:
			return efa_ep_enable(ep_fid);
		default:
			return -FI_ENOSYS;
		}
		break;
	default:
		return -FI_ENOSYS;
	}
}

static struct fi_ops efa_ep_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_ep_close,
	.bind = efa_ep_bind,
	.control = efa_ep_control,
	.ops_open = fi_no_ops_open,
};

/**
 * @brief progress engine for the EFA dgram endpoint
 *
 * This function now a no-op.
 *
 * @param[in] util_ep The endpoint FID to progress
 */
static
void efa_ep_progress_no_op(struct util_ep *util_ep)
{
	return;
}

static struct fi_ops_atomic efa_atomic_ops = {
	.size = sizeof(struct fi_ops_atomic),
	.write = fi_no_atomic_write,
	.writev = fi_no_atomic_writev,
	.writemsg = fi_no_atomic_writemsg,
	.inject = fi_no_atomic_inject,
	.readwrite = fi_no_atomic_readwrite,
	.readwritev = fi_no_atomic_readwritev,
	.readwritemsg = fi_no_atomic_readwritemsg,
	.compwrite = fi_no_atomic_compwrite,
	.compwritev = fi_no_atomic_compwritev,
	.compwritemsg = fi_no_atomic_compwritemsg,
	.writevalid = fi_no_atomic_writevalid,
	.readwritevalid = fi_no_atomic_readwritevalid,
	.compwritevalid = fi_no_atomic_compwritevalid,
};

struct fi_ops_cm efa_ep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = efa_base_ep_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

int efa_ep_open(struct fid_domain *domain_fid, struct fi_info *user_info,
		struct fid_ep **ep_fid, void *context)
{
	struct efa_base_ep *ep;
	int ret;

	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return -FI_ENOMEM;

	ret = efa_base_ep_construct(ep, domain_fid, user_info, efa_ep_progress_no_op, context);
	if (ret)
		goto err_ep_destroy;

	*ep_fid = &ep->util_ep.ep_fid;
	(*ep_fid)->fid.fclass = FI_CLASS_EP;
	(*ep_fid)->fid.context = context;
	(*ep_fid)->fid.ops = &efa_ep_ops;
	(*ep_fid)->ops = &efa_ep_base_ops;
	(*ep_fid)->msg = &efa_msg_ops;
	(*ep_fid)->cm = &efa_ep_cm_ops;
	(*ep_fid)->rma = &efa_rma_ops;
	(*ep_fid)->atomic = &efa_atomic_ops;

	return 0;

err_ep_destroy:
	efa_base_ep_destruct(ep);
	if (ep)
		free(ep);
	return ret;
}
