/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef FI_XPU_H
#define FI_XPU_H

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * OFI Accelerator API — Host-Side Structures
 */

#ifndef FI_XPU
#define FI_XPU			(1ULL << 44)
#endif

#define FI_XPU_ALLOC_DMABUF	(1ULL << 0)

#define FI_XPU_IMPORT_IOMEMORY	(1ULL << 0)
#define FI_XPU_IMPORT_DEVICEMAP	(1ULL << 1)

struct fi_xpu_attr {
	enum fi_hmem_iface   iface;
	uint64_t             device;

	int (*alloc)(uint64_t device, uint64_t size,
		     uint64_t alignment, uint64_t flags,
		     void **addr, int *fd, uint64_t *offset);

	int (*import)(uint64_t device, void *host_addr,
		      uint64_t size, uint64_t flags,
		      void **dev_addr);

	void (*free)(uint64_t device, void *addr);

	size_t               xpu_key_size;
};

/*
 * Host-side export wrappers
 */

static inline int fi_ep_export_xpu(struct fid_ep *ep, uint64_t flags,
				   void **xpu_ep, size_t *size)
{
	return FI_CHECK_OP(ep->ops, struct fi_ops_ep, export_xpu) ?
		ep->ops->export_xpu(ep, flags, xpu_ep, size) : -FI_ENOSYS;
}

static inline int fi_cq_export_xpu(struct fid_cq *cq, uint64_t flags,
				   void **xpu_cq, size_t *size)
{
	return FI_CHECK_OP(cq->ops, struct fi_ops_cq, export_xpu) ?
		cq->ops->export_xpu(cq, flags, xpu_cq, size) : -FI_ENOSYS;
}

static inline int fi_cntr_export_xpu(struct fid_cntr *cntr, uint64_t flags,
				     void **xpu_cntr, size_t *size)
{
	return FI_CHECK_OP(cntr->ops, struct fi_ops_cntr, export_xpu) ?
		cntr->ops->export_xpu(cntr, flags, xpu_cntr, size) :
		-FI_ENOSYS;
}

static inline int fi_av_get_xpu_addr(struct fid_av *av, fi_addr_t fi_addr,
				     void *buf, size_t *len)
{
	return FI_CHECK_OP(av->ops, struct fi_ops_av, get_xpu_addr) ?
		av->ops->get_xpu_addr(av, fi_addr, buf, len) : -FI_ENOSYS;
}

static inline int fi_mr_get_xpu_desc(struct fid_mr *mr, void *buf, size_t *len)
{
	struct fid_domain *domain = (struct fid_domain *)mr->fid.context;
	return FI_CHECK_OP(domain->mr, struct fi_ops_mr, get_xpu_desc) ?
		domain->mr->get_xpu_desc(mr, buf, len) : -FI_ENOSYS;
}

#ifdef __cplusplus
}
#endif

#endif /* FI_XPU_H */
