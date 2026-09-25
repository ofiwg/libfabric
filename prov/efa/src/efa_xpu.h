/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. */

#ifndef EFA_XPU_H
#define EFA_XPU_H

#include <rdma/fi_xpu.h>

/*
 * The hardware queues an endpoint export maps for the device: SQ buffer, SQ
 * doorbell, RQ buffer, RQ doorbell.
 */
#define EFA_XPU_EP_MAX_IMPORTS 4

/*
 * Per-endpoint XPU state. Holds what fi_ep_export_xpu() obtained from the XPU
 * so that closing the endpoint gives it back.
 */
struct efa_xpu_ep_state {
	struct fid_xpu_ctx *xpu_ctx;
	void *dev_ep;
	/* Host addresses of the queue regions mapped for the device. */
	void *imported[EFA_XPU_EP_MAX_IMPORTS];
	size_t imported_cnt;
};

/* Per-CQ XPU state */
struct efa_xpu_cq_state {
	struct fid_xpu_ctx *xpu_ctx;
	void *cq_buf_dev;
	/* Device-side descriptor handed out by fi_cq_export_xpu() */
	void *dev_cq;
};

/* Per-counter XPU state */
struct efa_xpu_cntr_state {
	struct fid_xpu_ctx *xpu_ctx;
	void *cntr_value_dev;
	void *cntr_err_dev;
	void *cntr_alloc_addr;
	void *cntr_err_alloc_addr;
	/* Device-side descriptor handed out by fi_cntr_export_xpu() */
	void *dev_cntr;
};

/* XPU context provider-private data (stored in fid_xpu_ctx) */
struct efa_xpu_ctx {
	struct fid_xpu_ctx ctx;        /* must be first */
	struct fi_xpu_attr attr;       /* copy of user's attr */
	struct fid_domain *domain;
};

/*
 * Export implementations.
 *
 * The caller allocates the (fixed-size) struct fid_xpu_ep/cq/cntr and
 * passes a pointer to it. The provider fills fclass/prov_id/prov_ctx.
 * prov_ctx is set to the device-accessible address of the full
 * provider-specific state (struct efa_xpu_ep/cq/cntr, allocated via
 * the XPU memory ops), which the device-side dispatch functions use
 * to locate all resources for the object.
 */
int efa_ep_export_xpu(struct fid_ep *ep, uint64_t flags,
                      struct fid_xpu_ep *xpu_ep);
int efa_cq_export_xpu(struct fid_cq *cq, uint64_t flags,
                      struct fid_xpu_cq *xpu_cq);
int efa_cntr_export_xpu(struct fid_cntr *cntr, uint64_t flags,
                        struct fid_xpu_cntr *xpu_cntr);

/* AV and MR ops */
int efa_av_lookup2(struct fid_av *av, fi_addr_t fi_addr,
                   void *buf, size_t *len, uint64_t flags,
                   struct fid_xpu_ctx *ctx);
int efa_mr_control(struct fid *fid, int command, void *arg);

/* Domain op: create XPU context */
int efa_xpu_ctx_open(struct fid_domain *domain, struct fi_xpu_attr *attr,
                     struct fid_xpu_ctx **ctx, void *context);

/*
 * State lifecycle. Each destroy releases every XPU resource the matching
 * object took: the device-side descriptor an export allocated, the device
 * mappings of an endpoint's hardware queues, and the device buffers the NIC
 * writes completions and counter values into.
 */
int efa_xpu_mem_alloc(struct fid_xpu_ctx *ctx, uint64_t size,
		      uint64_t alignment, uint64_t flags, void **addr, int *fd,
		      uint64_t *offset);
void efa_xpu_mem_free(struct fid_xpu_ctx *ctx, void *addr);
void efa_xpu_mem_put_fd(struct fid_xpu_ctx *ctx, int fd);

struct efa_xpu_ep_state *efa_xpu_ep_state_create(struct fid_xpu_ctx *ctx);
struct efa_xpu_cq_state *efa_xpu_cq_state_create(struct fid_xpu_ctx *ctx);
struct efa_xpu_cntr_state *efa_xpu_cntr_state_create(struct fid_xpu_ctx *ctx);
void efa_xpu_ep_state_destroy(struct efa_xpu_ep_state *state);
void efa_xpu_cq_state_destroy(struct efa_xpu_cq_state *state);
void efa_xpu_cntr_state_destroy(struct efa_xpu_cntr_state *state);

#endif /* EFA_XPU_H */
