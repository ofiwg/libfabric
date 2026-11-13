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
 * per-provider <prov>_xpu_<op>() functions. When a provider implements
 * device-side support, its header is included here and corresponding cases
 * are added to each dispatch switch.
 */

#include <rdma/fi_xpu.h>
#include <rdma/fi_trigger.h>

typedef void *fi_wr;
struct fi_wr_attr;
struct fi_rma_iov;

/*
 * EFA device-side dispatch is only relevant on the CPU architectures
 * where EFA hardware is deployed (x86_64 and aarch64 EC2 instances).
 * Gating on these GCC/Clang architecture macros (rather than, say,
 * __CUDACC__) also naturally excludes MSVC, which does not define
 * them, so a plain Windows host build/IntelliSense pass never parses
 * fi_xpu_device_efa.h.
 */
/*
 * EFA's device path is written for CUDA and HIP. A SYCL device compile defines
 * the host's architecture macros too, so without the exclusion it would build
 * the EFA path with the host stand-ins for its fences and atomics: code that
 * compiles and is wrong on the device. Leave it out instead, so the calls return
 * -FI_ENOSYS.
 */
#if (defined(__x86_64__) || defined(__aarch64__)) && \
    !defined(__SYCL_DEVICE_ONLY__)
#define FI_XPU_HAVE_EFA 1
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
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
 * (e.g. fi_xpu_device_efa.h), it defines efa_xpu_<op>() functions.
 * Then include the header here and add a case in each dispatch switch:
 *
 *   #include <rdma/fi_xpu_device_efa.h>
 *
 *   case FI_XPU_PROV_EFA:
 *       return efa_xpu_send(ep, ...);
 */

#ifdef FI_XPU_HAVE_EFA
#include <rdma/fi_xpu_device_efa.h>
#endif


FI_XPU_FUNC int
fi_xpu_write(struct fid_xpu_ep *ep, const void *buf, size_t len, void *desc,
	     uint64_t data, void *dest_addr, uint64_t addr, uint64_t key,
	     void *context, uint64_t flags, int scope)
{
	switch (ep->fid.prov_id) {
#ifdef FI_XPU_HAVE_EFA
	case FI_XPU_PROV_EFA:
		return efa_xpu_write(ep, buf, len, desc, data, dest_addr,
					addr, key, context, flags, scope);
#endif
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
#ifdef FI_XPU_HAVE_EFA
	case FI_XPU_PROV_EFA:
		return efa_xpu_read(ep, buf, len, desc, src_addr,
				       addr, key, context, flags, scope);
#endif
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
#ifdef FI_XPU_HAVE_EFA
	case FI_XPU_PROV_EFA:
		return efa_xpu_send(ep, buf, len, desc, data, dest_addr,
				       context, flags, scope);
#endif
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_recv(struct fid_xpu_ep *ep, void *buf, size_t len, void *desc,
	    void *src_addr, void *context, uint64_t flags, int scope)
{
	switch (ep->fid.prov_id) {
#ifdef FI_XPU_HAVE_EFA
	case FI_XPU_PROV_EFA:
		return efa_xpu_recv(ep, buf, len, desc, src_addr,
				       context, flags, scope);
#endif
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


/*
 * Work Request (WR) operations.
 */
FI_XPU_FUNC int
fi_xpu_wr_prepare(struct fid_xpu_ep *ep, const struct fi_wr_attr *attr,
		  fi_wr wr, size_t *wr_len, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_wr_queue_tx(struct fid_xpu_ep *ep, const fi_wr wr,
		   void *context, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_wr_queue_recv(struct fid_xpu_ep *ep, const fi_wr wr,
		     void *context, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_wr_queue_trecv(struct fid_xpu_ep *ep, const fi_wr wr,
		      void *context, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_tx_flush(struct fid_xpu_ep *ep, uint64_t flags, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_recv_flush(struct fid_xpu_ep *ep, uint64_t flags, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_trecv_flush(struct fid_xpu_ep *ep, uint64_t flags, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

/* xpu_addr references fi_xpu_ctx_attr::av_addr_size bytes. */
FI_XPU_FUNC int
fi_xpu_wr_modify_addr(struct fid_xpu_ep *ep, fi_wr wr,
		      enum fi_op_type op_type, const void *xpu_addr, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

/* desc references count * fi_xpu_ctx_attr::mr_desc_size bytes. */
FI_XPU_FUNC int
fi_xpu_wr_modify_iov(struct fid_xpu_ep *ep, fi_wr wr,
		     enum fi_op_type op_type, const struct iovec *iov,
		     const void *desc, size_t count, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_wr_modify_rma_iov(struct fid_xpu_ep *ep, fi_wr wr,
			 enum fi_op_type op_type,
			 const struct fi_rma_iov *rma_iov, size_t count,
			 int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_wr_modify_tag(struct fid_xpu_ep *ep, fi_wr wr,
		     enum fi_op_type op_type, uint64_t tag, uint64_t ignore,
		     int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_wr_modify_data(struct fid_xpu_ep *ep, fi_wr wr,
		      enum fi_op_type op_type, uint64_t data, int scope)
{
	switch (ep->fid.prov_id) {
	default:
		return -FI_ENOSYS;
	}
}

FI_XPU_FUNC int
fi_xpu_wr_modify_flags(struct fid_xpu_ep *ep, fi_wr wr,
		       enum fi_op_type op_type, uint64_t flags, int scope)
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
#ifdef FI_XPU_HAVE_EFA
	case FI_XPU_PROV_EFA:
		return efa_xpu_cntr_read(cntr, scope);
#endif
	default:
		return 0;
	}
}

FI_XPU_FUNC uint64_t
fi_xpu_cntr_readerr(struct fid_xpu_cntr *cntr, int scope)
{
	switch (cntr->fid.prov_id) {
#ifdef FI_XPU_HAVE_EFA
	case FI_XPU_PROV_EFA:
		return efa_xpu_cntr_readerr(cntr, scope);
#endif
	default:
		return 0;
	}
}

FI_XPU_FUNC void
fi_xpu_cntr_wait(struct fid_xpu_cntr *cntr, uint64_t threshold, int timeout,
		 int scope)
{
	switch (cntr->fid.prov_id) {
#ifdef FI_XPU_HAVE_EFA
	case FI_XPU_PROV_EFA:
		efa_xpu_cntr_wait(cntr, threshold, timeout, scope);
		return;
#endif
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
#ifdef FI_XPU_HAVE_EFA
	case FI_XPU_PROV_EFA:
		return efa_xpu_cq_read(cq, buf, count, scope);
#endif
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
#ifdef FI_XPU_HAVE_EFA
	case FI_XPU_PROV_EFA:
		return efa_xpu_cq_readerr(cq, buf, flags, scope);
#endif
	default:
		return -FI_ENOSYS;
	}
}

/*
 * The blocking CQ reads have no provider implementation yet. A kernel has no
 * wait object to block on, so the device form of these is a spin on
 * fi_xpu_cq_read() until the threshold is met - an increment to add, needing
 * nothing new from a provider's exported state. It stays unimplemented because
 * fi_xpu_cntr_wait() is the cheap way for a kernel to wait: one load of a
 * counter word, against an I/O memory read, a phase check and a CAS per
 * completion on the CQ path.
 */
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
