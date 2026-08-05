/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef FI_XPU_DEVICE_H
#define FI_XPU_DEVICE_H

/*
 * =============================================================================
 * OFI Accelerator API — Unified Device Header
 *
 * This is the ONLY file consumers include from their device kernels:
 *
 *     #include <rdma/fi_xpu_device.h>
 *
 * It works with any accelerator compiler (nvcc, hipcc, SYCL, or plain C)
 * and defines the device-side function interfaces for accelerator-initiated
 * communication.
 * =============================================================================
 */

#include <stdint.h>
#include <stddef.h>
#include <rdma/fi_errno.h>

/*
 * =============================================================================
 * Device function qualifier — adapts to the active compiler
 * =============================================================================
 */
#if defined(__CUDACC__) || (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__)
  #define FI_XPU_FUNC __device__ static inline
#elif defined(__SYCL_DEVICE_ONLY__)
  #define FI_XPU_FUNC static inline
#else
  #define FI_XPU_FUNC static inline
#endif

/*
 * =============================================================================
 * Scope hint for cooperative operations
 * =============================================================================
 */
#ifndef FI_XPU_SCOPE_DEFINED
#define FI_XPU_SCOPE_DEFINED
enum fi_xpu_scope {
	FI_XPU_WORK_ITEM  = 0,   /* Thread (CUDA) / Work item (SYCL) */
	FI_XPU_SUBGROUP   = 1,   /* Warp (CUDA) / Subgroup (SYCL) */
	FI_XPU_WORK_GROUP = 2,   /* Thread block (CUDA) / Work group (SYCL) */
	FI_XPU_DEVICE     = 3,   /* Device (all thread blocks) */
};
#endif

/*
 * =============================================================================
 * Post operations — RMA
 * =============================================================================
 */

FI_XPU_FUNC int
fi_xpu_write(void *xpu_ep, const void *buf, void *desc, uint64_t size,
	     uint64_t data, void *peer, uint64_t raddr, uint64_t rkey,
	     void *ctxt, int scope, uint64_t flags)
{
	return -FI_ENOSYS;
}

FI_XPU_FUNC int
fi_xpu_read(void *xpu_ep, void *buf, void *desc, uint64_t size,
	    void *peer, uint64_t raddr, uint64_t rkey,
	    void *ctxt, int scope, uint64_t flags)
{
	return -FI_ENOSYS;
}

/*
 * =============================================================================
 * Post operations — Message
 * =============================================================================
 */

FI_XPU_FUNC int
fi_xpu_send(void *xpu_ep, const void *buf, uint64_t size, void *desc,
	    uint64_t data, void *peer, void *ctxt, int scope, uint64_t flags)
{
	return -FI_ENOSYS;
}

FI_XPU_FUNC int
fi_xpu_recv(void *xpu_ep, void *buf, void *desc, uint64_t size,
	    void *peer, void *ctxt, int scope, uint64_t flags)
{
	return -FI_ENOSYS;
}

/*
 * =============================================================================
 * Post operations — Tagged
 * =============================================================================
 */

FI_XPU_FUNC int
fi_xpu_tsend(void *xpu_ep, const void *buf, uint64_t size, void *desc,
	     uint64_t data, void *peer, uint64_t tag, void *ctxt,
	     int scope, uint64_t flags)
{
	return -FI_ENOSYS;
}

FI_XPU_FUNC int
fi_xpu_trecv(void *xpu_ep, void *buf, uint64_t size, void *desc,
	     void *peer, uint64_t tag, uint64_t ignore, void *ctxt,
	     int scope, uint64_t flags)
{
	return -FI_ENOSYS;
}

/*
 * =============================================================================
 * Post operations — Atomic
 * =============================================================================
 */

FI_XPU_FUNC int
fi_xpu_atomic(void *xpu_ep, const void *buf, size_t count, void *desc,
	      void *peer, uint64_t addr, uint64_t key,
	      int datatype, int op, void *ctxt,
	      int scope, uint64_t flags)
{
	return -FI_ENOSYS;
}

FI_XPU_FUNC int
fi_xpu_fetch_atomic(void *xpu_ep, const void *buf, size_t count, void *desc,
		    void *result, void *result_desc,
		    void *peer, uint64_t addr, uint64_t key,
		    int datatype, int op, void *ctxt,
		    int scope, uint64_t flags)
{
	return -FI_ENOSYS;
}

FI_XPU_FUNC int
fi_xpu_compare_atomic(void *xpu_ep, const void *buf, size_t count, void *desc,
		      const void *compare, void *compare_desc,
		      void *result, void *result_desc,
		      void *peer, uint64_t addr, uint64_t key,
		      int datatype, int op, void *ctxt,
		      int scope, uint64_t flags)
{
	return -FI_ENOSYS;
}

/*
 * =============================================================================
 * Completion — Counter
 * =============================================================================
 */

FI_XPU_FUNC uint64_t
fi_xpu_cntr_read(void *xpu_cntr, int scope)
{
	return 0;
}

FI_XPU_FUNC uint64_t
fi_xpu_cntr_readerr(void *xpu_cntr, int scope)
{
	return 0;
}

FI_XPU_FUNC void
fi_xpu_cntr_wait(void *xpu_cntr, uint64_t threshold, int timeout, int scope)
{
}

FI_XPU_FUNC int
fi_xpu_cntr_add(void *xpu_cntr, uint64_t value, int scope)
{
	return -FI_ENOSYS;
}

FI_XPU_FUNC int
fi_xpu_cntr_set(void *xpu_cntr, uint64_t value, int scope)
{
	return -FI_ENOSYS;
}

FI_XPU_FUNC int
fi_xpu_cntr_adderr(void *xpu_cntr, uint64_t value, int scope)
{
	return -FI_ENOSYS;
}

FI_XPU_FUNC int
fi_xpu_cntr_seterr(void *xpu_cntr, uint64_t value, int scope)
{
	return -FI_ENOSYS;
}

/*
 * =============================================================================
 * Completion — CQ
 * =============================================================================
 */

FI_XPU_FUNC int64_t
fi_xpu_cq_read(void *xpu_cq, void *buf, size_t count, int scope)
{
	return -FI_ENOSYS;
}

FI_XPU_FUNC int64_t
fi_xpu_cq_readfrom(void *xpu_cq, void *buf, size_t count,
		   void *src_addr, int scope)
{
	return -FI_ENOSYS;
}

FI_XPU_FUNC int64_t
fi_xpu_cq_readerr(void *xpu_cq, void *buf, uint64_t flags, int scope)
{
	return -FI_ENOSYS;
}

FI_XPU_FUNC int64_t
fi_xpu_cq_sread(void *xpu_cq, void *buf, size_t count,
		uint64_t threshold, int scope)
{
	return -FI_ENOSYS;
}

FI_XPU_FUNC int64_t
fi_xpu_cq_sreadfrom(void *xpu_cq, void *buf, size_t count,
		    void *src_addr, uint64_t threshold, int scope)
{
	return -FI_ENOSYS;
}

#endif /* FI_XPU_DEVICE_H */
