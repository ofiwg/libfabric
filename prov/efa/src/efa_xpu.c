/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "config.h"

#include <ofi_util.h>
#include <rdma/fi_xpu_device_efa.h>

#include "efa.h"
#include "efa_domain.h"
#include "efa_xpu.h"

/*
 * What the context can do. The exports of endpoints, CQs and counters are
 * not implemented yet, so it reports none of them; the sizes are of the
 * forms fi_av_lookup2() and fi_mr_get_xpu_desc() return.
 */
static int efa_xpu_ctx_query(struct fid_xpu_ctx *ctx,
			     struct fi_xpu_ctx_attr *attr)
{
	if (!ctx || !attr)
		return -FI_EINVAL;

	attr->caps = 0;
	attr->av_addr_size = sizeof(struct efa_xpu_peer);
	attr->mr_desc_size = sizeof(struct efa_xpu_desc);
	return 0;
}

static int efa_xpu_ctx_close(struct fid *fid)
{
	struct efa_xpu_ctx *xctx = container_of(fid, struct efa_xpu_ctx,
						ctx.fid);
	struct efa_domain *domain = xctx->domain;

	ofi_atomic_dec32(&domain->xpu_ctx_cnt);
	ofi_atomic_dec32(&domain->util_domain.ref);

	free(xctx);
	return 0;
}

static struct fi_ops efa_xpu_ctx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_xpu_ctx_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_xpu_ctx efa_xpu_ctx_ops = {
	.size = sizeof(struct fi_ops_xpu_ctx),
	.query = efa_xpu_ctx_query,
};

/*
 * The memory callbacks an application supplies must be complete enough to be
 * used: each allocation must have a way back, and so must each import.
 */
static int efa_xpu_ops_valid(const struct fi_xpu_ops *ops)
{
	if (ops->size < sizeof(struct fi_xpu_ops)) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "fi_xpu_ops size %zu is smaller than the %zu its "
			 "callbacks need\n", ops->size,
			 sizeof(struct fi_xpu_ops));
		return 0;
	}
	if (!ops->alloc != !ops->free) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "fi_xpu_ops must supply alloc and free together\n");
		return 0;
	}
	if (ops->unimport && !ops->import) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "fi_xpu_ops supplies unimport without import\n");
		return 0;
	}
	return 1;
}

/**
 * @brief Open an XPU context on an efa-direct domain.
 *
 * Checks that the memory callbacks in @attr are complete, claims one of the
 * domain's EFA_XPU_MAX_CTX_CNT context slots, records @attr, and takes a
 * reference on the domain so that it cannot close while the context is open.
 *
 * @return 0, -FI_EINVAL for a missing argument or incomplete memory
 * callbacks, -FI_EOPNOTSUPP on a domain other than efa-direct, -FI_EBUSY if
 * the domain already has its context, or -FI_ENOMEM.
 */
int efa_xpu_ctx_open(struct fid_domain *domain_fid, struct fi_xpu_attr *attr,
		     struct fid_xpu_ctx **ctx, void *context)
{
	struct efa_domain *domain;
	struct efa_xpu_ctx *xctx;

	if (!domain_fid || !attr || !ctx)
		return -FI_EINVAL;
	if (attr->ops && !efa_xpu_ops_valid(attr->ops))
		return -FI_EINVAL;

	domain = container_of(domain_fid, struct efa_domain,
			      util_domain.domain_fid);
	if (domain->info_type != EFA_INFO_DIRECT) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "FI_XPU is only supported on an efa-direct domain\n");
		return -FI_EOPNOTSUPP;
	}

	/*
	 * Claim a context slot, and give it back if that took the count past
	 * the limit: of two opens racing for the last slot, the second to
	 * count backs out.
	 */
	if (ofi_atomic_inc32(&domain->xpu_ctx_cnt) > EFA_XPU_MAX_CTX_CNT) {
		ofi_atomic_dec32(&domain->xpu_ctx_cnt);
		EFA_WARN(FI_LOG_DOMAIN,
			 "the domain already has its %d XPU context(s)\n",
			 EFA_XPU_MAX_CTX_CNT);
		return -FI_EBUSY;
	}

	xctx = calloc(1, sizeof(*xctx));
	if (!xctx) {
		ofi_atomic_dec32(&domain->xpu_ctx_cnt);
		return -FI_ENOMEM;
	}

	xctx->ctx.fid.fclass = FI_CLASS_XPU_CTX;
	xctx->ctx.fid.context = context;
	xctx->ctx.fid.ops = &efa_xpu_ctx_fi_ops;
	xctx->ctx.ops = &efa_xpu_ctx_ops;
	xctx->attr = *attr;
	xctx->domain = domain;
	ofi_atomic_inc32(&domain->util_domain.ref);

	*ctx = &xctx->ctx;
	return 0;
}
