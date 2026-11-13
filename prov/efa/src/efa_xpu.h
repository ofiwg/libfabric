/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. */

#ifndef EFA_XPU_H
#define EFA_XPU_H

#include <rdma/fi_xpu.h>
#include <stdbool.h>
#include <stdint.h>

struct efa_base_ep;
struct efa_cq;

/*
 * The hardware queues an endpoint export maps for the device: SQ buffer, SQ
 * doorbell, RQ buffer, RQ doorbell.
 */
#define EFA_XPU_EP_MAX_IMPORTS 4

/*
 * Per-endpoint XPU state. Holds what fi_ep_export_xpu() obtained from the XPU
 * so that closing the endpoint gives it back.
 */
enum {
	EFA_XPU_SQ_DONE,
	EFA_XPU_RQ_DONE,
	EFA_XPU_DONE_CNT,
};

struct efa_xpu_ep_state {
	struct fid_xpu_ctx *xpu_ctx;
	void *dev_ep;
	/* Host addresses of the queue regions mapped for the device. */
	void *imported[EFA_XPU_EP_MAX_IMPORTS];
	size_t imported_cnt;

	/*
	 * The done cursors the device waits on before reusing a queue slot:
	 * how many of each queue's descriptors have had their completions
	 * consumed by a host CQ read (see efa_xpu_ep_note_done()). They live
	 * in device memory the host stores into through a registration, or,
	 * where the device memory cannot be registered, in a host page mapped
	 * for the device. A queue whose completions the device reads has none.
	 */
	struct efa_cq *done_cq[EFA_XPU_DONE_CNT];	/* NULL: not published */
	bool done_bound[EFA_XPU_DONE_CNT];		/* done is published */
	bool done_listed[EFA_XPU_DONE_CNT];		/* on its CQ's flush list */
	uint32_t done[EFA_XPU_DONE_CNT];		/* host copy */
	void *done_dev;					/* device address */
	void *done_host;				/* mapped host memory */
	uint64_t done_handle;				/* registration */
	bool done_registered;

	/*
	 * The per-slot completion counts of a queue with a context table (see
	 * efa_xpu_wq_wait_slot()). Where the host reads the completions they
	 * share the done cursors' memory, at slot_off, and slot_cnt is the
	 * host's copy; where the device reads them they are device memory of
	 * their own, slot_dev, which the device CQ read advances.
	 */
	uint32_t *slot_cnt[EFA_XPU_DONE_CNT];		/* host copy */
	size_t slot_off[EFA_XPU_DONE_CNT];		/* in done_dev */
	uint32_t slot_mask[EFA_XPU_DONE_CNT];
	void *slot_dev[EFA_XPU_DONE_CNT];

	/*
	 * The context tables (see efa_xpu_ep_ctx_setup()): a host page mapped
	 * for the device when the host reads the queue's completions, device
	 * memory when the device does.
	 */
	void *ctx_host[EFA_XPU_DONE_CNT];
	struct efa_cq *ctx_host_cq[EFA_XPU_DONE_CNT];	/* polls ctx_host */
	void *ctx_dev[EFA_XPU_DONE_CNT];

	/*
	 * The done cursors of the queues whose completions the device reads,
	 * in device memory the device-side CQ read advances, and the XPU CQ
	 * each such queue reports to.
	 */
	void *ddone_dev;
	uint32_t queue_mask[EFA_XPU_DONE_CNT];	/* ring slots - 1 */
	struct efa_cq *dev_cq[EFA_XPU_DONE_CNT];
};

/* Per-CQ XPU state */
struct efa_xpu_cq_state {
	struct fid_xpu_ctx *xpu_ctx;
	void *cq_buf_dev;
	/* Device-side descriptor handed out by fi_cq_export_xpu() */
	void *dev_cq;
	/*
	 * The QP map a device-side read finds each QP's done cursors and
	 * context tables through, and the host's copy of its rows (struct
	 * efa_xpu_cq_qp), filled from the bound endpoints by
	 * fi_cq_export_xpu().
	 */
	void *qp_map_dev;
	void *qp_map_host;
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
void efa_xpu_ep_detach(struct efa_base_ep *base_ep);
void efa_xpu_ep_note_done(struct efa_xpu_ep_state *state, int queue,
			  uint16_t req_id);
void efa_xpu_ep_publish_done(struct efa_xpu_ep_state *state, int queue);
void efa_xpu_cq_state_destroy(struct efa_xpu_cq_state *state);
void efa_xpu_cntr_state_destroy(struct efa_xpu_cntr_state *state);

#endif /* EFA_XPU_H */
