/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_XPU_H
#define EFA_XPU_H

#include <ofi_atom.h>
#include <rdma/fi_xpu.h>

struct efa_domain;

/* The XPU contexts an EFA domain supports: one, for its one device. */
#define EFA_XPU_MAX_CTX_CNT 1

/*
 * An XPU context: the accelerator the application named, and the memory
 * callbacks it supplied, for the objects of one efa-direct domain.
 *
 * The context holds a reference on its domain, so the domain cannot be closed
 * under it.
 */
struct efa_xpu_ctx {
	struct fid_xpu_ctx	ctx;		/* must be first */
	struct fi_xpu_attr	attr;		/* copy of the application's */
	struct efa_domain	*domain;
};

int efa_xpu_ctx_open(struct fid_domain *domain, struct fi_xpu_attr *attr,
		     struct fid_xpu_ctx **ctx, void *context);

/**
 * @brief The EFA XPU context @ctx names, if it is one of @domain's.
 *
 * @return the context, or NULL if @ctx is not an XPU context or belongs to
 * another domain.
 */
static inline struct efa_xpu_ctx *
efa_xpu_ctx_of(struct fid_xpu_ctx *ctx, struct efa_domain *domain)
{
	struct efa_xpu_ctx *xctx;

	if (!ctx || ctx->fid.fclass != FI_CLASS_XPU_CTX)
		return NULL;

	xctx = container_of(ctx, struct efa_xpu_ctx, ctx);
	return xctx->domain == domain ? xctx : NULL;
}

#endif /* EFA_XPU_H */
