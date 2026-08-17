/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. */

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "efa.h"
#include "efa_base_ep.h"
#include "efa_cq.h"
#include "efa_cntr.h"
#include "efa_av.h"
#include "efa_mr.h"
#include "efa_xpu.h"

#include <rdma/fi_xpu_device_efa.h>

/*
 * efadv.h is already pulled in by efa_base_ep.h -> infiniband/efadv.h.
 * No additional include needed here.
 */

/*
 * Exporting EFA resources to an XPU needs efadv_query_qp_wqs() and
 * efadv_query_cq() to find the hardware queues to map. Without them the
 * provider still builds - it just has no XPU support to offer, so every entry
 * point below reports that and the rest of the file drops out.
 */
#if HAVE_EFA_XPU

/* ----------------------------------------------------------------
 * Memory allocation / import helpers
 * ---------------------------------------------------------------- */

/**
 * xpu_alloc - allocate device memory via user-provided ops or HMEM.
 *
 * If the XPU attr has ops->alloc, use that (user-managed allocation).
 * Otherwise fall back to the HMEM device ops, which take the same arguments.
 */
static int xpu_alloc(struct efa_xpu_ctx *xctx, uint64_t size,
		     uint64_t alignment, uint64_t flags,
		     void **addr, int *fd, uint64_t *offset)
{
	struct fi_xpu_attr *attr = &xctx->attr;

	if (attr->ops && attr->ops->alloc)
		return attr->ops->alloc(attr->device, size, alignment,
					flags, addr, fd, offset);

	return ofi_hmem_dev_alloc(attr->iface, attr->device, size, alignment,
				  flags, addr, fd, offset);
}

/**
 * xpu_import - map host address into device address space.
 */
static int xpu_import(struct efa_xpu_ctx *xctx, void *host_addr,
		      uint64_t size, uint64_t flags, void **dev_addr)
{
	struct fi_xpu_attr *attr = &xctx->attr;

	if (attr->ops && attr->ops->import)
		return attr->ops->import(attr->device, host_addr,
					 size, flags, dev_addr);

	/*
	 * Provider-managed import: register the host/MMIO region with the
	 * HMEM layer and return a device-accessible pointer (cuMemHostRegister
	 * + cuMemHostGetDevicePointer for CUDA).
	 */
	return ofi_hmem_dev_import(attr->iface, attr->device, host_addr, size,
				   flags, dev_addr);
}

/**
 * xpu_unimport - release a mapping made by xpu_import.
 *
 * Whoever made the mapping releases it. An application that supplied ->import
 * but no ->unimport keeps the mapping: there is nothing the provider can call,
 * and the HMEM ops did not make it.
 */
static void xpu_unimport(struct efa_xpu_ctx *xctx, void *host_addr)
{
	struct fi_xpu_attr *attr = &xctx->attr;
	int ret;

	if (!host_addr)
		return;

	if (attr->ops && attr->ops->import) {
		if (!attr->ops->unimport)
			return;
		ret = attr->ops->unimport(attr->device, host_addr);
	} else {
		ret = ofi_hmem_dev_unimport(attr->iface, attr->device,
					    host_addr);
	}

	if (ret)
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Failed to release XPU mapping of %p: %d\n",
			 host_addr, ret);
}

/**
 * xpu_free - free device memory via user-provided ops or HMEM.
 */
static void xpu_free(struct efa_xpu_ctx *xctx, void *addr)
{
	struct fi_xpu_attr *attr = &xctx->attr;

	if (attr->ops && attr->ops->free) {
		attr->ops->free(attr->device, addr);
		return;
	}

	if (!addr)
		return;

	/* Provider-managed: release the device allocation via HMEM ops. */
	ofi_hmem_dev_free(attr->iface, attr->device, addr);
}

/**
 * xpu_put_fd - release a dmabuf fd xpu_alloc exported.
 *
 * The fd is only needed until the NIC object the memory was allocated for has
 * been created over it and holds its own reference, so it is released there
 * rather than at free time. Whichever side allocated the memory, the fd names a
 * dmabuf of the context iface and is released the way that iface releases one.
 */
static void xpu_put_fd(struct efa_xpu_ctx *xctx, int fd)
{
	int ret;

	if (fd < 0)
		return;

	ret = ofi_hmem_put_dmabuf_fd(xctx->attr.iface, fd);
	if (ret)
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Failed to release dmabuf fd %d: %d\n", fd, ret);
}

/*
 * The CQ ring and the counters are allocated where they are opened, not here,
 * because the NIC wants their dmabuf fd at creation time. They still have to
 * come from - and go back to - whichever side owns the XPU context memory, so
 * those paths reach the same helpers through here.
 */
int efa_xpu_mem_alloc(struct fid_xpu_ctx *ctx, uint64_t size,
		      uint64_t alignment, uint64_t flags, void **addr, int *fd,
		      uint64_t *offset)
{
	return xpu_alloc(container_of(ctx, struct efa_xpu_ctx, ctx), size,
			 alignment, flags, addr, fd, offset);
}

void efa_xpu_mem_free(struct fid_xpu_ctx *ctx, void *addr)
{
	xpu_free(container_of(ctx, struct efa_xpu_ctx, ctx), addr);
}

void efa_xpu_mem_put_fd(struct fid_xpu_ctx *ctx, int fd)
{
	if (fd < 0)
		return;

	xpu_put_fd(container_of(ctx, struct efa_xpu_ctx, ctx), fd);
}

/* ----------------------------------------------------------------
 * XPU Context (fid_xpu_ctx) management
 * ---------------------------------------------------------------- */

static int efa_xpu_ctx_query(struct fid_xpu_ctx *ctx,
			     struct fi_xpu_ctx_attr *attr)
{
	attr->caps = FI_XPU_CAP_EP | FI_XPU_CAP_CQ | FI_XPU_CAP_CNTR;
	attr->av_addr_size = sizeof(struct efa_xpu_peer);
	attr->mr_desc_size = sizeof(struct efa_xpu_desc);
	return 0;
}

static struct fi_ops_xpu_ctx efa_xpu_ctx_ops = {
	.size = sizeof(struct fi_ops_xpu_ctx),
	.query = efa_xpu_ctx_query,
};

static int efa_xpu_ctx_close(struct fid *fid)
{
	struct efa_xpu_ctx *ctx;

	ctx = container_of(fid, struct efa_xpu_ctx, ctx.fid);
	free(ctx);
	return 0;
}

static struct fi_ops efa_xpu_ctx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_xpu_ctx_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int efa_xpu_ctx_open(struct fid_domain *domain, struct fi_xpu_attr *attr,
		     struct fid_xpu_ctx **ctx_out, void *context)
{
	struct efa_xpu_ctx *ctx;
	struct efa_domain *efa_domain;

	if (!domain || !attr || !ctx_out)
		return -FI_EINVAL;

	/*
	 * Only efa-direct exports its queues to a device. The rdm and dgram
	 * paths keep the send queue behind the provider's own protocol state,
	 * so a kernel posting into it would corrupt them. FI_XPU is advertised
	 * on the efa-direct info alone (see efa_prov.c), but a domain op can be
	 * reached from any domain, so the domain is checked here too.
	 */
	efa_domain = container_of(domain, struct efa_domain,
				  util_domain.domain_fid);
	if (efa_domain->info_type != EFA_INFO_DIRECT) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "FI_XPU is only supported on an efa-direct domain\n");
		return -FI_EOPNOTSUPP;
	}

	ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return -FI_ENOMEM;

	ctx->ctx.fid.fclass = FI_CLASS_XPU_CTX;
	ctx->ctx.fid.context = context;
	ctx->ctx.fid.ops = &efa_xpu_ctx_fi_ops;
	ctx->ctx.ops = &efa_xpu_ctx_ops;
	ctx->attr = *attr;
	ctx->domain = domain;

	*ctx_out = &ctx->ctx;
	return 0;
}

/* ----------------------------------------------------------------
 * State lifecycle
 * ---------------------------------------------------------------- */

struct efa_xpu_ep_state *efa_xpu_ep_state_create(struct fid_xpu_ctx *ctx)
{
	struct efa_xpu_ep_state *state;

	state = calloc(1, sizeof(*state));
	if (!state)
		return NULL;

	state->xpu_ctx = ctx;
	return state;
}

struct efa_xpu_cq_state *efa_xpu_cq_state_create(struct fid_xpu_ctx *ctx)
{
	struct efa_xpu_cq_state *state;

	state = calloc(1, sizeof(*state));
	if (!state)
		return NULL;

	state->xpu_ctx = ctx;
	return state;
}

struct efa_xpu_cntr_state *efa_xpu_cntr_state_create(struct fid_xpu_ctx *ctx)
{
	struct efa_xpu_cntr_state *state;

	state = calloc(1, sizeof(*state));
	if (!state)
		return NULL;

	state->xpu_ctx = ctx;
	return state;
}

void efa_xpu_ep_state_destroy(struct efa_xpu_ep_state *state)
{
	struct efa_xpu_ctx *xctx;
	size_t i;

	if (!state)
		return;

	xctx = container_of(state->xpu_ctx, struct efa_xpu_ctx, ctx);

	/* Unmap the hardware queues before the QP they belong to goes away. */
	for (i = 0; i < state->imported_cnt; i++)
		xpu_unimport(xctx, state->imported[i]);

	if (state->dev_ep)
		xpu_free(xctx, state->dev_ep);

	free(state);
}

void efa_xpu_cq_state_destroy(struct efa_xpu_cq_state *state)
{
	struct efa_xpu_ctx *xctx;

	if (!state)
		return;

	xctx = container_of(state->xpu_ctx, struct efa_xpu_ctx, ctx);

	if (state->dev_cq)
		xpu_free(xctx, state->dev_cq);

	if (state->cq_buf_dev)
		xpu_free(xctx, state->cq_buf_dev);

	free(state);
}

void efa_xpu_cntr_state_destroy(struct efa_xpu_cntr_state *state)
{
	struct efa_xpu_ctx *xctx;

	if (!state)
		return;

	xctx = container_of(state->xpu_ctx, struct efa_xpu_ctx, ctx);

	if (state->dev_cntr)
		xpu_free(xctx, state->dev_cntr);

	if (state->cntr_alloc_addr)
		xpu_free(xctx, state->cntr_alloc_addr);
	if (state->cntr_err_alloc_addr)
		xpu_free(xctx, state->cntr_err_alloc_addr);

	free(state);
}

/* ----------------------------------------------------------------
 * EP export
 * ---------------------------------------------------------------- */

int efa_ep_export_xpu(struct fid_ep *ep, uint64_t flags,
		      struct fid_xpu_ep *xpu_ep)
{
	struct efa_base_ep *base_ep;
	struct efa_xpu_ctx *xctx;
	struct efa_xpu_ep *dev_ep;
	struct fid_xpu_ctx *fid_ctx;
	struct efa_xpu_ep_state *state;
	void *dev_buf = NULL;
	int ret;

	if (!ep || !xpu_ep)
		return -FI_EINVAL;

	base_ep = container_of(ep, struct efa_base_ep, util_ep.ep_fid);

	/*
	 * Only an endpoint opened for an XPU has anything to export. It was
	 * opened with fi_endpoint2(FI_XPU), which recorded the flag, and with
	 * the context in info->ep_attr->xpu_ctx (not the endpoint context
	 * argument), preserved in base_ep->info by efa_base_ep_construct().
	 */
	if (!(base_ep->util_ep.flags & FI_XPU))
		return -FI_EINVAL;

	fid_ctx = (base_ep->info && base_ep->info->ep_attr) ?
		  base_ep->info->ep_attr->xpu_ctx : NULL;
	if (!fid_ctx)
		return -FI_EINVAL;

	xctx = container_of(fid_ctx, struct efa_xpu_ctx, ctx);

#if HAVE_EFADV_QUERY_QP_WQS
	{
		struct efadv_wq_attr qp_sq_attr = {0};
		struct efadv_wq_attr qp_rq_attr = {0};
		struct efa_xpu_ep h_ep = {0};

		if (!base_ep->qp || !base_ep->qp->ibv_qp)
			return -FI_EINVAL;

		/* Query hardware queue geometry for this QP. */
		ret = efadv_query_qp_wqs(base_ep->qp->ibv_qp, &qp_sq_attr,
					 &qp_rq_attr, sizeof(qp_sq_attr));
		if (ret) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "efadv_query_qp_wqs failed: %d\n", ret);
			return (ret == EOPNOTSUPP) ? -FI_EOPNOTSUPP : -FI_EINVAL;
		}

		if (base_ep->xpu_state) {
			/*
			 * Re-export: release the descriptor and the queue
			 * mappings the previous export took.
			 */
			efa_xpu_ep_state_destroy(base_ep->xpu_state);
			base_ep->xpu_state = NULL;
		}

		state = efa_xpu_ep_state_create(fid_ctx);
		if (!state)
			return -FI_ENOMEM;

		/*
		 * Allocate the device-side EP state struct on the XPU. It is
		 * plain device memory that only the XPU kernel reads, so it
		 * needs no dmabuf fd - unlike the CQ ring and the counters,
		 * which the NIC itself writes. dev_buf must NOT be dereferenced
		 * from the host; we build the contents in the host struct h_ep
		 * and copy it to the device below.
		 */
		ret = xpu_alloc(xctx, sizeof(struct efa_xpu_ep), 64, 0,
				&dev_buf, NULL, NULL);
		if (ret)
			goto err_destroy_state;
		state->dev_ep = dev_buf;
		dev_ep = (struct efa_xpu_ep *)dev_buf;

		/* Populate the XPU header (host side). prov_ctx points at the
		 * device-accessible copy of this struct. The version lets a
		 * kernel built against a newer header refuse, or adapt to, a
		 * handle exported by an older library. */
		h_ep.version = fi_version();
		h_ep.xpu_ep.fid.fclass = FI_CLASS_EP;
		h_ep.xpu_ep.fid.prov_id = FI_XPU_PROV_EFA;
		h_ep.xpu_ep.fid.prov_ctx = (uint64_t)(uintptr_t)dev_ep;

		/* SQ geometry */
		h_ep.sq.queue_mask = qp_sq_attr.num_entries - 1;
		h_ep.sq.queue_size_shift = __builtin_ctz(qp_sq_attr.num_entries);
		h_ep.sq.max_batch = qp_sq_attr.max_batch;
		h_ep.sq.entry_size = qp_sq_attr.entry_size;
		h_ep.sq.pc = 0;
		h_ep.sq.released = 0;
		h_ep.sq.db_rung = 0;
		h_ep.sq.init_phase = 0;
		h_ep.sq_size = qp_sq_attr.num_entries;
		h_ep.submitted_count = 0;
		h_ep.local_cntr = NULL;

		/*
		 * A device-built WQE carries the application's context as the
		 * request ID when the SQ supports 64-bit request IDs. That is what
		 * lets a host-side fi_cq_read() complete an operation the XPU
		 * posted - the device has no wr_id pool to translate a 16-bit
		 * index through. Follow the same runtime knob the host data path
		 * uses, so both halves agree on how a request ID is encoded.
		 */
		h_ep.sq_req_id_64_bit = 0;
#if HAVE_EFADV_WQ_ATTR_CAPS
		h_ep.sq_req_id_64_bit =
			!!(qp_sq_attr.caps & EFADV_WQ_CAPS_64_BIT_REQ_ID) &&
			efa_env.use_sq_req_id_64_bit;
#endif
		if (!h_ep.sq_req_id_64_bit)
			EFA_INFO(FI_LOG_EP_CTRL,
				 "SQ has no 64-bit request ID support: operations "
				 "posted from the XPU can only be completed on the "
				 "XPU, not through a host CQ\n");

		/* RQ geometry */
		if (qp_rq_attr.num_entries) {
			h_ep.rq.queue_mask = qp_rq_attr.num_entries - 1;
			h_ep.rq.queue_size_shift =
				__builtin_ctz(qp_rq_attr.num_entries);
			h_ep.rq.max_batch = qp_rq_attr.max_batch;
			h_ep.rq.entry_size = qp_rq_attr.entry_size;
			h_ep.rq.pc = 0;
			h_ep.rq.released = 0;
			h_ep.rq.db_rung = 0;
			h_ep.rq.init_phase = 0;
		}

		/*
		 * Map the SQ ring buffer (BAR MMIO) into device-accessible
		 * memory so the XPU kernel can build WQEs directly.
		 */
		{
			void *sq_buf_dev = NULL;

			ret = xpu_import(xctx, qp_sq_attr.buffer,
					 (size_t) qp_sq_attr.num_entries *
						 qp_sq_attr.entry_size,
					 FI_XPU_IMPORT_IOMEMORY |
						 FI_XPU_IMPORT_DEVICEMAP,
					 &sq_buf_dev);
			if (ret) {
				EFA_WARN(FI_LOG_EP_CTRL,
					 "map SQ buffer failed: %d\n", ret);
				goto err_destroy_state;
			}
			state->imported[state->imported_cnt++] =
				qp_sq_attr.buffer;
			h_ep.sq.buf = (uint8_t *) sq_buf_dev;
		}

		/*
		 * Map the SQ doorbell (BAR MMIO). Register only the 4-byte
		 * doorbell register: the doorbell VA is not page-aligned and
		 * registering a full page would cross into an unmapped page.
		 */
		{
			void *sq_db_dev = NULL;

			ret = xpu_import(xctx, qp_sq_attr.doorbell,
					 sizeof(uint32_t),
					 FI_XPU_IMPORT_IOMEMORY |
						 FI_XPU_IMPORT_DEVICEMAP,
					 &sq_db_dev);
			if (ret) {
				EFA_WARN(FI_LOG_EP_CTRL,
					 "map SQ doorbell failed: %d\n", ret);
				goto err_destroy_state;
			}
			state->imported[state->imported_cnt++] =
				qp_sq_attr.doorbell;
			h_ep.sq.db = (uint32_t *) sq_db_dev;
		}

		/* Map RQ buffer (host RAM) and doorbell (BAR MMIO) if present. */
		if (qp_rq_attr.buffer) {
			void *rq_buf_dev = NULL;
			void *rq_db_dev = NULL;

			ret = xpu_import(xctx, qp_rq_attr.buffer,
					 (size_t) qp_rq_attr.num_entries *
						 qp_rq_attr.entry_size,
					 FI_XPU_IMPORT_DEVICEMAP, &rq_buf_dev);
			if (ret) {
				EFA_WARN(FI_LOG_EP_CTRL,
					 "map RQ buffer failed: %d\n", ret);
				goto err_destroy_state;
			}
			state->imported[state->imported_cnt++] =
				qp_rq_attr.buffer;
			h_ep.rq.buf = (uint8_t *) rq_buf_dev;

			ret = xpu_import(xctx, qp_rq_attr.doorbell,
					 sizeof(uint32_t),
					 FI_XPU_IMPORT_IOMEMORY |
						 FI_XPU_IMPORT_DEVICEMAP,
					 &rq_db_dev);
			if (ret) {
				EFA_WARN(FI_LOG_EP_CTRL,
					 "map RQ doorbell failed: %d\n", ret);
				goto err_destroy_state;
			}
			state->imported[state->imported_cnt++] =
				qp_rq_attr.doorbell;
			h_ep.rq.db = (uint32_t *) rq_db_dev;
		}

		/*
		 * Link the bound write counter's device pointer for SQ
		 * backpressure, when a hw counter was bound to this EP.
		 */
		{
			struct util_cntr *wr_cntr =
				base_ep->util_ep.cntrs[CNTR_WR];
			if (wr_cntr) {
				struct efa_cntr *efa_cntr = container_of(
					wr_cntr, struct efa_cntr, util_cntr);
				if (efa_cntr->xpu_state &&
				    efa_cntr->xpu_state->cntr_value_dev)
					h_ep.local_cntr = (volatile uint64_t *)
						efa_cntr->xpu_state->cntr_value_dev;
			}
		}

		/* Transfer the fully-populated host struct to device memory. */
		ret = ofi_copy_to_hmem(xctx->attr.iface, xctx->attr.device,
				       dev_ep, &h_ep, sizeof(h_ep));
		if (ret) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "ofi_copy_to_hmem failed for XPU ep: %d\n",
				 ret);
			goto err_destroy_state;
		}

		xpu_ep->fid.fclass = FI_CLASS_EP;
		xpu_ep->fid.prov_id = FI_XPU_PROV_EFA;
		xpu_ep->fid.prov_ctx = (uint64_t)(uintptr_t)dev_ep;

		/* The endpoint owns these resources until it is closed. */
		base_ep->xpu_state = state;

		return FI_SUCCESS;

err_destroy_state:
		efa_xpu_ep_state_destroy(state);
		return ret;
	}
#else
	return -FI_EOPNOTSUPP;
#endif
}

/* ----------------------------------------------------------------
 * CQ export
 * ---------------------------------------------------------------- */

int efa_cq_export_xpu(struct fid_cq *cq, uint64_t flags,
		      struct fid_xpu_cq *xpu_cq)
{
	struct efa_cq *efa_cq;
	struct efa_xpu_ctx *xctx;
	struct efa_xpu_cq *dev_cq;
	struct fid_xpu_ctx *fid_ctx;
	void *dev_buf = NULL;
	int ret;

	if (!cq || !xpu_cq)
		return -FI_EINVAL;

	efa_cq = container_of(cq, struct efa_cq, util_cq.cq_fid);

	if (!efa_cq->xpu_state)
		return -FI_ENODATA;
	fid_ctx = efa_cq->xpu_state->xpu_ctx;
	if (!fid_ctx)
		return -FI_EINVAL;

	xctx = container_of(fid_ctx, struct efa_xpu_ctx, ctx);

#if HAVE_EFADV_QUERY_CQ
	{
		struct efadv_cq_attr efadv_attr = {0};
		struct efa_xpu_cq h_cq = {0};

		if (!efa_cq->xpu_state->cq_buf_dev)
			return -FI_ENODATA;

		ret = efadv_query_cq(ibv_cq_ex_to_cq(efa_cq->ibv_cq.ibv_cq_ex),
				     &efadv_attr, sizeof(efadv_attr));
		if (ret)
			return (ret == EOPNOTSUPP) ? -FI_EOPNOTSUPP : -FI_EINVAL;

		/*
		 * Allocate the device-side CQ descriptor. Plain device memory:
		 * only the ring buffer it points at, which the NIC writes, was
		 * allocated as a dmabuf.
		 */
		if (efa_cq->xpu_state->dev_cq) {
			/* Re-export: release the previous descriptor. */
			xpu_free(xctx, efa_cq->xpu_state->dev_cq);
			efa_cq->xpu_state->dev_cq = NULL;
		}

		ret = xpu_alloc(xctx, sizeof(struct efa_xpu_cq), 64, 0,
				&dev_buf, NULL, NULL);
		if (ret)
			return ret;
		efa_cq->xpu_state->dev_cq = dev_buf;
		dev_cq = (struct efa_xpu_cq *)dev_buf;

		/*
		 * Build the descriptor on the host. The CQ ring buffer is the
		 * device-resident buffer allocated at open time (EXT_MEM_DMABUF),
		 * so the XPU kernel polls the hardware CQ directly.
		 */
		h_cq.version = fi_version();
		h_cq.xpu_cq.fid.fclass = FI_CLASS_CQ;
		h_cq.xpu_cq.fid.prov_id = FI_XPU_PROV_EFA;
		h_cq.xpu_cq.fid.prov_ctx = (uint64_t)(uintptr_t)dev_cq;
		h_cq.buf = (uint8_t *)efa_cq->xpu_state->cq_buf_dev;
		h_cq.entry_size = efadv_attr.entry_size;
		h_cq.queue_mask = efadv_attr.num_entries - 1;
		h_cq.queue_size_shift = __builtin_ctz(efadv_attr.num_entries);
		h_cq.cc = 0;
		h_cq.init_phase = 1;
		/*
		 * What a device poll writes into the caller's buffer, as
		 * opposed to entry_size above, which is what the NIC writes
		 * into the ring.
		 */
		h_cq.format = efa_cq->format;
		h_cq.user_entry_size = efa_cq->entry_size;

		ret = ofi_copy_to_hmem(xctx->attr.iface, xctx->attr.device,
				       dev_cq, &h_cq, sizeof(h_cq));
		if (ret) {
			EFA_WARN(FI_LOG_CQ,
				 "ofi_copy_to_hmem failed for XPU cq: %d\n",
				 ret);
			xpu_free(xctx, dev_buf);
			efa_cq->xpu_state->dev_cq = NULL;
			return ret;
		}

		xpu_cq->fid.fclass = FI_CLASS_CQ;
		xpu_cq->fid.prov_id = FI_XPU_PROV_EFA;
		xpu_cq->fid.prov_ctx = (uint64_t)(uintptr_t)dev_cq;

		return FI_SUCCESS;
	}
#else
	return -FI_EOPNOTSUPP;
#endif
}

/* ----------------------------------------------------------------
 * Counter export
 * ---------------------------------------------------------------- */

int efa_cntr_export_xpu(struct fid_cntr *cntr, uint64_t flags,
			struct fid_xpu_cntr *xpu_cntr)
{
	struct efa_cntr *efa_cntr;
	struct efa_xpu_ctx *xctx;
	struct efa_xpu_cntr *dev_cntr;
	struct fid_xpu_ctx *fid_ctx;
	void *dev_buf = NULL;
	int ret;

	if (!cntr || !xpu_cntr)
		return -FI_EINVAL;

	efa_cntr = container_of(cntr, struct efa_cntr,
				util_cntr.cntr_fid);

	if (!efa_cntr->xpu_state || !efa_cntr->xpu_state->cntr_value_dev)
		return -FI_ENODATA;
	fid_ctx = efa_cntr->xpu_state->xpu_ctx;
	if (!fid_ctx)
		return -FI_EINVAL;

	xctx = container_of(fid_ctx, struct efa_xpu_ctx, ctx);

	{
		struct efa_xpu_cntr h_cntr = {0};

		/*
		 * Allocate the device-side counter descriptor. Plain device
		 * memory: only the counter values it points at, which the NIC
		 * writes, were allocated as dmabufs.
		 */
		if (efa_cntr->xpu_state->dev_cntr) {
			/* Re-export: release the previous descriptor. */
			xpu_free(xctx, efa_cntr->xpu_state->dev_cntr);
			efa_cntr->xpu_state->dev_cntr = NULL;
		}

		ret = xpu_alloc(xctx, sizeof(struct efa_xpu_cntr), 64, 0,
				&dev_buf, NULL, NULL);
		if (ret)
			return ret;
		efa_cntr->xpu_state->dev_cntr = dev_buf;
		dev_cntr = (struct efa_xpu_cntr *)dev_buf;

		/*
		 * Build the descriptor on the host. The counter values live in
		 * the device-resident buffers allocated at open time (EFADV
		 * external DMABUF); the NIC writes them and the XPU kernel reads
		 * them directly.
		 */
		h_cntr.version = fi_version();
		h_cntr.xpu_cntr.fid.fclass = FI_CLASS_CNTR;
		h_cntr.xpu_cntr.fid.prov_id = FI_XPU_PROV_EFA;
		h_cntr.xpu_cntr.fid.prov_ctx = (uint64_t)(uintptr_t)dev_cntr;
		h_cntr.value = (volatile uint64_t *)
			efa_cntr->xpu_state->cntr_value_dev;
		h_cntr.err_value = (volatile uint64_t *)
			efa_cntr->xpu_state->cntr_err_dev;

		ret = ofi_copy_to_hmem(xctx->attr.iface, xctx->attr.device,
				       dev_cntr, &h_cntr, sizeof(h_cntr));
		if (ret) {
			EFA_WARN(FI_LOG_CNTR,
				 "ofi_copy_to_hmem failed for XPU cntr: %d\n",
				 ret);
			xpu_free(xctx, dev_buf);
			efa_cntr->xpu_state->dev_cntr = NULL;
			return ret;
		}

		xpu_cntr->fid.fclass = FI_CLASS_CNTR;
		xpu_cntr->fid.prov_id = FI_XPU_PROV_EFA;
		xpu_cntr->fid.prov_ctx = (uint64_t)(uintptr_t)dev_cntr;

		return FI_SUCCESS;
	}
}

#else /* !HAVE_EFA_XPU */

int efa_xpu_ctx_open(struct fid_domain *domain, struct fi_xpu_attr *attr,
		     struct fid_xpu_ctx **ctx, void *context)
{
	EFA_WARN(FI_LOG_DOMAIN,
		 "FI_XPU is not supported: libfabric was built against an "
		 "rdma-core without efadv_query_qp_wqs()/efadv_query_cq()\n");
	return -FI_EOPNOTSUPP;
}

int efa_ep_export_xpu(struct fid_ep *ep, uint64_t flags,
		      struct fid_xpu_ep *xpu_ep)
{
	return -FI_EOPNOTSUPP;
}

int efa_cq_export_xpu(struct fid_cq *cq, uint64_t flags,
		      struct fid_xpu_cq *xpu_cq)
{
	return -FI_EOPNOTSUPP;
}

int efa_cntr_export_xpu(struct fid_cntr *cntr, uint64_t flags,
			struct fid_xpu_cntr *xpu_cntr)
{
	return -FI_EOPNOTSUPP;
}

int efa_xpu_mem_alloc(struct fid_xpu_ctx *ctx, uint64_t size,
		      uint64_t alignment, uint64_t flags, void **addr, int *fd,
		      uint64_t *offset)
{
	return -FI_EOPNOTSUPP;
}

void efa_xpu_mem_free(struct fid_xpu_ctx *ctx, void *addr)
{
}

void efa_xpu_mem_put_fd(struct fid_xpu_ctx *ctx, int fd)
{
}

/*
 * An XPU context can never be created, so no object can ever hold XPU state:
 * the creates always fail and the destroys always get NULL.
 */
struct efa_xpu_ep_state *efa_xpu_ep_state_create(struct fid_xpu_ctx *ctx)
{
	return NULL;
}

struct efa_xpu_cq_state *efa_xpu_cq_state_create(struct fid_xpu_ctx *ctx)
{
	return NULL;
}

struct efa_xpu_cntr_state *efa_xpu_cntr_state_create(struct fid_xpu_ctx *ctx)
{
	return NULL;
}

void efa_xpu_ep_state_destroy(struct efa_xpu_ep_state *state)
{
}

void efa_xpu_cq_state_destroy(struct efa_xpu_cq_state *state)
{
}

void efa_xpu_cntr_state_destroy(struct efa_xpu_cntr_state *state)
{
}

#endif /* HAVE_EFA_XPU */
