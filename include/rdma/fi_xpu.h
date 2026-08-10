/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef FI_XPU_H
#define FI_XPU_H

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Flags for fi_xpu_ops callbacks.
 */

/* alloc: allocation must be exportable as a DMA-BUF fd */
#define FI_XPU_ALLOC_DMABUF	(1ULL << 0)

/* import: host address is PCIe BAR MMIO (device I/O memory) */
#define FI_XPU_IMPORT_IOMEMORY	(1ULL << 0)
/* import: resulting pointer must be accessible from XPU kernels */
#define FI_XPU_IMPORT_DEVICEMAP	(1ULL << 1)

/*
 * XPU Memory Operations
 */
struct fi_xpu_ops {
	size_t	size;
	int	(*alloc)(uint64_t device, uint64_t size,
			 uint64_t alignment, uint64_t flags,
			 void **addr, int *fd, uint64_t *offset);
	int	(*import)(uint64_t device, void *host_addr,
			  uint64_t size, uint64_t flags,
			  void **dev_addr);
	void	(*free)(uint64_t device, void *addr);
};

/*
 * XPU Attribute — input to fi_xpu_ctx() creation
 */
struct fi_xpu_attr {
	enum fi_hmem_iface	iface;
	uint64_t		device;
	struct fi_xpu_ops	*ops;
};

/*
 * XPU Context capability flags — returned by fi_xpu_ctx_query()
 * to indicate which XPU objects the provider supports.
 */
#define FI_XPU_CAP_EP		(1ULL << 0)
#define FI_XPU_CAP_CQ		(1ULL << 1)
#define FI_XPU_CAP_CNTR		(1ULL << 2)

/*
 * XPU Context Attribute — output from fi_xpu_ctx_query()
 */
struct fi_xpu_ctx_attr {
	uint64_t		caps;
	size_t			av_addr_size;
	size_t			mr_desc_size;
};

/*
 * XPU Context Operations and Object
 */
struct fi_ops_xpu_ctx {
	size_t	size;
	int	(*query)(struct fid_xpu_ctx *ctx,
			struct fi_xpu_ctx_attr *attr);
};

struct fid_xpu_ctx {
	struct fid			fid;
	struct fi_ops_xpu_ctx		*ops;
};

/*
 * Inline wrappers
 */
static inline int
fi_xpu_ctx(struct fid_domain *domain, struct fi_xpu_attr *attr,
	   struct fid_xpu_ctx **ctx, void *context)
{
	return domain->ops->xpu_ctx(domain, attr, ctx, context);
}

static inline int
fi_xpu_ctx_query(struct fid_xpu_ctx *ctx, struct fi_xpu_ctx_attr *attr)
{
	return ctx->ops->query(ctx, attr);
}

#ifdef __cplusplus
}
#endif

#endif /* FI_XPU_H */
