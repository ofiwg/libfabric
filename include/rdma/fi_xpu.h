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
 * Provider identifiers for device-side dispatch.
 * Values start at 1 and form a tight range suitable for a jump table.
 * The provider must set prov_id to its assigned value during the
 * export call (fi_ep_export_xpu, fi_cq_export_xpu, fi_cntr_export_xpu).
 *
 * This list only includes providers that have added XPU device-side
 * dispatch support; it grows as additional providers do so.
 */
enum fi_xpu_provider {
	FI_XPU_PROV_EFA		= 1,
};

/*
 * XPU exported object base type.
 * Every handle returned by fi_ep_export_xpu / fi_cq_export_xpu /
 * fi_cntr_export_xpu embeds this as its first member.  The provider
 * populates fclass, prov_id, and prov_ctx during the export call.
 *
 * fclass identifies the object type (FI_CLASS_EP, FI_CLASS_CQ, FI_CLASS_CNTR).
 * prov_id identifies the provider for device-side dispatch.
 * prov_ctx holds a provider-internal value (typically a device-accessible
 * address) that the provider's device-side implementation uses to locate
 * all resources and state for this object.
 *
 * The caller provides the structure and passes it to the export call;
 * the provider fills it out.  Device-side dispatch casts the handle
 * to struct fid_xpu * to read prov_id and route to the correct
 * provider implementation.
 */
struct fid_xpu {
	uint32_t	fclass;		/* FI_CLASS_EP, _CQ, _CNTR */
	uint32_t	prov_id;	/* enum fi_xpu_provider */
	uint64_t	prov_ctx;	/* provider-internal context/pointer */
};

struct fid_xpu_ep {
	struct fid_xpu	fid;
};

struct fid_xpu_cq {
	struct fid_xpu	fid;
};

struct fid_xpu_cntr {
	struct fid_xpu	fid;
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
