/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2026 Hewlett Packard Enterprise Development LP
 */

/*
 * fi_cxi_gpu_ext.h - GPU Data Structures for CXI KI Ops
 *
 * This header provides:
 *   1. GPU data structures (fi_cxi_gpu_cmdq, fi_cxi_gpu_ep_info, etc.)
 *   2. Host-side inline helpers for command queue device remapping
 */

#ifndef FI_CXI_GPU_EXT_H
#define FI_CXI_GPU_EXT_H

#include <stddef.h>
#include <stdint.h>

/*
 * Layout assertions.  This header is compiled by both the host and the
 * GPU toolchains (nvcc, hipcc, as C++), and the structs below are shared
 * between them, so each must see the same layout.  Compilers that
 * support neither form skip the checks.
 */
#if defined(__cplusplus)
#define FI_CXI_GPU_STATIC_ASSERT(expr, msg) static_assert(expr, msg)
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define FI_CXI_GPU_STATIC_ASSERT(expr, msg) _Static_assert(expr, msg)
#else
#define FI_CXI_GPU_STATIC_ASSERT(expr, msg)
#endif

/*
 * Command Queue State
 *
 * The hardware command queue state a GPU kernel needs to emit commands.
 * This is a provider-owned subset of the libcxi command queue, so the
 * application ABI does not depend on libcxi's internal layout.
 */

struct fi_cxi_gpu_cmdq_state {
	void     *cmds32;          /* Command buffer (device VA after
				    * set_buf)
				    */
	uint64_t *wp_addr;         /* Doorbell (device VA after set_db) */
	uint64_t  wp32;            /* Next slot to write */
	uint64_t  hw_wp32;         /* Last slot rung to hardware */
	uint64_t  rp32;            /* Cached hardware read pointer */
	uint32_t  size32;          /* Buffer size in slots */
};

/*
 * Command Queue Wrapper
 *
 * Wraps the command queue state to keep host and device pointers separate.
 * The inner cq is what the GPU kernel operates on directly.
 */

struct fi_cxi_gpu_cmdq {
	/* Host pointers - set by provider, never overwritten */
	void     *hptr_buff;       /* Host cmd buffer (from ki_ep_hints);
				    * NULL for a device-memory buffer
				    */
	uint64_t *hptr_mmio_db;    /* Host MMIO doorbell */

	/* Device pointers - set by user via set helpers */
	void     *dptr_buff;       /* Device cmd buffer */
	uint64_t *dptr_mmio_db;    /* Device MMIO doorbell */

	/* HW command queue state - GPU kernel operates on this */
	struct fi_cxi_gpu_cmdq_state cq;
};

/*
 * GPU Data Structures - passed to GPU kernel
 */

/* EP info - my hardware resources */
struct fi_cxi_gpu_ep_info {
	struct fi_cxi_gpu_cmdq tx_cmdq;   /* Transmit command queue */
	struct fi_cxi_gpu_cmdq trig_cmdq; /* Triggered transmit command queue */
	uint32_t eqn;              /* Event queue for error events */
	uint32_t ctn;              /* Counter for success completions */
	uint32_t pid_bits;         /* PID bit width for DFA computation */
	uint64_t user_ptr;         /* Error reporting handle */
};

/* Target addressing - pure scalars, no MR, no DFA. A command carries no
 * VNI; the KI command queues send on the EP's VNI.
 */
struct fi_cxi_gpu_target_info {
	uint32_t nid;              /* NIC address */
	uint32_t pid;              /* Portals ID (endpoint on the NIC) */
};

/* Remote MR - memory attributes only (DFA computed on GPU from target +
 * ptl_idx). A Put or Get to this MR sets match_bits as given and
 * remote_offset = offset_adjust + the MR-relative offset. An MR with
 * unrestricted set must be targeted with an unrestricted traffic class
 * type.
 */
struct fi_cxi_gpu_remote_mr {
	uint64_t iova;             /* IO Virtual Address of the MR start */
	uint64_t match_bits;       /* DMA match bits (standard MRs) */
	uint64_t offset_adjust;    /* Added to every remote offset */
	uint16_t write_ptl_idx;    /* Portal table index for Puts */
	uint16_t read_ptl_idx;     /* Portal table index for Gets */
	uint8_t  unrestricted;     /* Nonzero: needs unrestricted TC type */
};

/* Local MR - GPU-accessible local memory region info */
struct fi_cxi_gpu_mr_info {
	void     *buf;             /* Local buffer pointer */
	uint64_t  iova;            /* IO Virtual Address */
	uint32_t  lac;             /* Logical Address Context */
	uint32_t  flags;           /* Reserved */
};

/*
 * Pin the shared layouts (LP64).  A failure here means the host and GPU
 * compilers disagree, or a struct changed without a version bump.
 */
FI_CXI_GPU_STATIC_ASSERT(sizeof(struct fi_cxi_gpu_cmdq_state) == 48,
			 "fi_cxi_gpu_cmdq_state size");
FI_CXI_GPU_STATIC_ASSERT(offsetof(struct fi_cxi_gpu_cmdq_state, wp_addr) == 8,
			 "fi_cxi_gpu_cmdq_state.wp_addr offset");
FI_CXI_GPU_STATIC_ASSERT(offsetof(struct fi_cxi_gpu_cmdq_state, size32) == 40,
			 "fi_cxi_gpu_cmdq_state.size32 offset");

FI_CXI_GPU_STATIC_ASSERT(sizeof(struct fi_cxi_gpu_cmdq) == 80,
			 "fi_cxi_gpu_cmdq size");
FI_CXI_GPU_STATIC_ASSERT(offsetof(struct fi_cxi_gpu_cmdq, cq) == 32,
			 "fi_cxi_gpu_cmdq.cq offset");

FI_CXI_GPU_STATIC_ASSERT(sizeof(struct fi_cxi_gpu_ep_info) == 184,
			 "fi_cxi_gpu_ep_info size");
FI_CXI_GPU_STATIC_ASSERT(offsetof(struct fi_cxi_gpu_ep_info, trig_cmdq) == 80,
			 "fi_cxi_gpu_ep_info.trig_cmdq offset");
FI_CXI_GPU_STATIC_ASSERT(offsetof(struct fi_cxi_gpu_ep_info, eqn) == 160,
			 "fi_cxi_gpu_ep_info.eqn offset");
FI_CXI_GPU_STATIC_ASSERT(offsetof(struct fi_cxi_gpu_ep_info, user_ptr) == 176,
			 "fi_cxi_gpu_ep_info.user_ptr offset");

FI_CXI_GPU_STATIC_ASSERT(sizeof(struct fi_cxi_gpu_target_info) == 8,
			 "fi_cxi_gpu_target_info size");

FI_CXI_GPU_STATIC_ASSERT(sizeof(struct fi_cxi_gpu_remote_mr) == 32,
			 "fi_cxi_gpu_remote_mr size");
FI_CXI_GPU_STATIC_ASSERT(offsetof(struct fi_cxi_gpu_remote_mr,
				  write_ptl_idx) == 24,
			 "fi_cxi_gpu_remote_mr.write_ptl_idx offset");
FI_CXI_GPU_STATIC_ASSERT(offsetof(struct fi_cxi_gpu_remote_mr,
				  unrestricted) == 28,
			 "fi_cxi_gpu_remote_mr.unrestricted offset");

FI_CXI_GPU_STATIC_ASSERT(sizeof(struct fi_cxi_gpu_mr_info) == 24,
			 "fi_cxi_gpu_mr_info size");
FI_CXI_GPU_STATIC_ASSERT(offsetof(struct fi_cxi_gpu_mr_info, lac) == 16,
			 "fi_cxi_gpu_mr_info.lac offset");

/*
 * Command Queue Selector
 *
 * Defined here, alongside the get_db/set_db/set_buf helpers that are its
 * only consumers.
 */

enum fi_cxi_ki_cmdq_type {
	FI_CXI_KI_TXC,             /* TX command queue */
	FI_CXI_KI_TRIGC,           /* Triggered command queue */
};

/*
 * Inline Helpers - Command Queue Device Remap (Host-side)
 *
 * get reads hptr (stable, never overwritten)
 * set writes dptr AND updates cq for GPU use
 */

/*
 * fi_cxi_ki_cmdq_get_db - Get host VA of doorbell MMIO register.
 */
static inline void
fi_cxi_ki_cmdq_get_db(struct fi_cxi_gpu_ep_info *gpu_ep,
		      enum fi_cxi_ki_cmdq_type type,
		      uint64_t **db)
{
	struct fi_cxi_gpu_cmdq *cmdq = (type == FI_CXI_KI_TXC)
				       ? &gpu_ep->tx_cmdq : &gpu_ep->trig_cmdq;
	*db = cmdq->hptr_mmio_db;
}

/*
 * fi_cxi_ki_cmdq_set_db - Set device-mapped doorbell register pointer.
 */
static inline void
fi_cxi_ki_cmdq_set_db(struct fi_cxi_gpu_ep_info *gpu_ep,
		      enum fi_cxi_ki_cmdq_type type,
		      uint64_t *dev_db)
{
	struct fi_cxi_gpu_cmdq *cmdq = (type == FI_CXI_KI_TXC)
				       ? &gpu_ep->tx_cmdq : &gpu_ep->trig_cmdq;
	cmdq->dptr_mmio_db = dev_db;
	cmdq->cq.wp_addr = dev_db;
}

/*
 * fi_cxi_ki_cmdq_set_buf - Set device-mapped command buffer pointer.
 *
 * Updates dptr_buff and the inner cq.cmds32 pointer used by the GPU kernel.
 * Required for a device-memory buffer, which the provider returns with
 * cq.cmds32 NULL.  Skip if ki_ep_hints already provided a
 * device-accessible pointer.
 */
static inline void
fi_cxi_ki_cmdq_set_buf(struct fi_cxi_gpu_ep_info *gpu_ep,
			enum fi_cxi_ki_cmdq_type type,
			void *dev_buf)
{
	struct fi_cxi_gpu_cmdq *cmdq = (type == FI_CXI_KI_TXC)
				       ? &gpu_ep->tx_cmdq : &gpu_ep->trig_cmdq;
	cmdq->dptr_buff = dev_buf;
	cmdq->cq.cmds32 = dev_buf;
}

#endif /* FI_CXI_GPU_EXT_H */
