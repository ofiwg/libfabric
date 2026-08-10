/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef FI_XPU_DEVICE_H
#define FI_XPU_DEVICE_H

/*
 * OFI XPU Device-Side API
 *
 * This header is compiled with a single XPU kernel compiler at a time.
 * It supports XPU programming environments with a C/C++ interface:
 *
 *   - NVIDIA CUDA (nvcc)
 *   - AMD ROCm HIP (hipcc)
 *   - Intel oneAPI Level Zero / SYCL (icpx -fsycl)
 *
 * The FI_XPU_FUNC macro adapts the function qualifier based on the
 * compiler detected at build time.
 *
 * Each exported XPU handle (fid_xpu_ep, fid_xpu_cq, fid_xpu_cntr) embeds
 * struct fid_xpu as its first member. The dispatch functions read
 * fid.prov_id from the typed handle to route to the correct
 * provider-specific implementation.
 *
 * Provider-specific headers (fi_xpu_device_efa.h, etc.) define the
 * per-provider fi_xpu_<op>_<prov>() functions. When a provider implements
 * device-side support, its header is included here and corresponding cases
 * are added to each dispatch switch.
 */

#include <rdma/fi_xpu.h>

#if defined(__CUDACC__) || (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__)
  #define FI_XPU_FUNC __device__ static inline
#elif defined(__SYCL_DEVICE_ONLY__)
  #define FI_XPU_FUNC static inline
#else
  #define FI_XPU_FUNC static inline
#endif

/*
 * Provider-specific device headers go here.
 *
 * Example: when a provider implements its device-side header
 * (e.g. fi_xpu_device_efa.h), it defines fi_xpu_<op>_efa() functions.
 * Then include the header here and add a case in each dispatch switch:
 *
 *   #include <rdma/fi_xpu_device_efa.h>
 *
 *   case FI_XPU_PROV_EFA:
 *       return fi_xpu_send_efa(ep, ...);
 */


FI_XPU_FUNC int
fi_xpu_write(struct fid_xpu_ep *ep, const void *buf, size_t len, void *desc,
	     uint64_t data, void *dest_addr, uint64_t addr, uint64_t key,
	     void *context, uint64_t flags, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_read(struct fid_xpu_ep *ep, void *buf, size_t len, void *desc,
	    void *src_addr, uint64_t addr, uint64_t key,
	    void *context, uint64_t flags, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}


FI_XPU_FUNC int
fi_xpu_send(struct fid_xpu_ep *ep, const void *buf, size_t len, void *desc,
	    uint64_t data, void *dest_addr, void *context,
	    uint64_t flags, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_recv(struct fid_xpu_ep *ep, void *buf, size_t len, void *desc,
	    void *src_addr, void *context, uint64_t flags, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}


FI_XPU_FUNC int
fi_xpu_tsend(struct fid_xpu_ep *ep, const void *buf, size_t len, void *desc,
	     uint64_t data, void *dest_addr, uint64_t tag, void *context,
	     uint64_t flags, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_trecv(struct fid_xpu_ep *ep, void *buf, size_t len, void *desc,
	     void *src_addr, uint64_t tag, uint64_t ignore, void *context,
	     uint64_t flags, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}


FI_XPU_FUNC int
fi_xpu_atomic(struct fid_xpu_ep *ep, const void *buf, size_t count, void *desc,
	      void *dest_addr, uint64_t addr, uint64_t key,
	      int datatype, int op, void *context,
	      uint64_t flags, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_fetch_atomic(struct fid_xpu_ep *ep, const void *buf, size_t count,
		    void *desc, void *result, void *result_desc,
		    void *dest_addr, uint64_t addr, uint64_t key,
		    int datatype, int op, void *context,
		    uint64_t flags, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_compare_atomic(struct fid_xpu_ep *ep, const void *buf, size_t count,
		      void *desc, const void *compare, void *compare_desc,
		      void *result, void *result_desc,
		      void *dest_addr, uint64_t addr, uint64_t key,
		      int datatype, int op, void *context,
		      uint64_t flags, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}


FI_XPU_FUNC uint64_t
fi_xpu_cntr_read(struct fid_xpu_cntr *cntr, int scope)
{
	switch (cntr->fid.prov_id) {
	default:
		return 0;
	}
}

FI_XPU_FUNC uint64_t
fi_xpu_cntr_readerr(struct fid_xpu_cntr *cntr, int scope)
{
	switch (cntr->fid.prov_id) {
	default:
		return 0;
	}
}

FI_XPU_FUNC void
fi_xpu_cntr_wait(struct fid_xpu_cntr *cntr, uint64_t threshold, int timeout,
		 int scope)
{
	switch (cntr->fid.prov_id) {
	default:
		return;
	}
}

FI_XPU_FUNC int
fi_xpu_cntr_add(struct fid_xpu_cntr *cntr, uint64_t value, int scope)
{
	switch (cntr->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_cntr_set(struct fid_xpu_cntr *cntr, uint64_t value, int scope)
{
	switch (cntr->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_cntr_adderr(struct fid_xpu_cntr *cntr, uint64_t value, int scope)
{
	switch (cntr->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_cntr_seterr(struct fid_xpu_cntr *cntr, uint64_t value, int scope)
{
	switch (cntr->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}


FI_XPU_FUNC int64_t
fi_xpu_cq_read(struct fid_xpu_cq *cq, void *buf, size_t count, int scope)
{
	switch (cq->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int64_t
fi_xpu_cq_readfrom(struct fid_xpu_cq *cq, void *buf, size_t count,
		   void *src_addr, int scope)
{
	switch (cq->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int64_t
fi_xpu_cq_readerr(struct fid_xpu_cq *cq, void *buf, uint64_t flags, int scope)
{
	switch (cq->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int64_t
fi_xpu_cq_sread(struct fid_xpu_cq *cq, void *buf, size_t count,
		uint64_t threshold, int scope)
{
	switch (cq->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int64_t
fi_xpu_cq_sreadfrom(struct fid_xpu_cq *cq, void *buf, size_t count,
		    void *src_addr, uint64_t threshold, int scope)
{
	switch (cq->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

#endif /* FI_XPU_DEVICE_H */
