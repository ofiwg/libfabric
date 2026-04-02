/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "config.h"
#include <ofi_util.h>
#include "efa.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_ope.h"
#if HAVE_CUDA
#include <cuda.h>
#endif

static int efa_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
			  uint64_t flags, struct fid_mr **mr_fid);

/* Common validation for MR registration attributes */
int efa_mr_regattr_validate(struct fid *fid, const struct fi_mr_attr *attr,
				   uint64_t flags)
{
	struct efa_domain *domain;
	uint64_t supported_flags;
	uint32_t api_version;

	if (fid->fclass != FI_CLASS_DOMAIN) {
		EFA_WARN(FI_LOG_MR, "Unsupported domain. requested"
			 "[0x%" PRIx64 "] supported[0x%" PRIx64 "]\n",
			 fid->fclass, (uint64_t) FI_CLASS_DOMAIN);
		return -FI_EINVAL;
	}

	domain = container_of(fid, struct efa_domain, util_domain.domain_fid.fid);

	/*
	 * Notes supported memory registration flags:
	 *
	 * FI_HMEM_DEVICE_ONLY:
	 * This flag is used by some provider that need to distinguish
	 * whether a device memory can be accessed from device only, or
	 * can be access from host. EFA provider considers all device memory
	 * to be accessed by device only. Therefore, this function claim
	 * support of this flag, but do not save it in efa_mr.
	 *
	 * FI_MR_DMABUF:
	 * This flag indicates that the memory region to registered is
	 * a DMA-buf backed region. When set, the region is specified through
	 * the dmabuf field of the fi_mr_attr structure. This flag is only
	 * usable for domains opened with FI_HMEM capability support.
	 * This flag is introduced since Libfabric 1.20.
	 */
	supported_flags = FI_HMEM_DEVICE_ONLY;
	api_version = domain->util_domain.fabric->fabric_fid.api_version;

	if (FI_VERSION_GE(api_version, FI_VERSION(1, 20)))
		supported_flags |= FI_MR_DMABUF;

	if (flags & (~supported_flags)) {
		EFA_WARN(FI_LOG_MR, "Unsupported flag type. requested"
			 "[0x%" PRIx64 "] supported[0x%" PRIx64 "]\n",
			 flags, supported_flags);
		return -FI_EBADFLAGS;
	}

	if (attr->iov_count > EFA_MR_IOV_LIMIT) {
		EFA_WARN(FI_LOG_MR, "iov count > %d not supported\n",
			 EFA_MR_IOV_LIMIT);
		return -FI_EINVAL;
	}

	if (attr->iface >= OFI_HMEM_MAX || !g_efa_hmem_info[attr->iface].initialized) {
		EFA_WARN(FI_LOG_MR,
			 "Cannot register memory for uninitialized iface (%s)\n",
			 fi_tostr(&attr->iface, FI_TYPE_HMEM_IFACE));
		return -FI_ENOSYS;
	}

	return 0;
}

/**
 * @brief Validate HMEM attributes and populate efa_mr struct
 *
 * Check if FI_HMEM is enabled for the domain, validate whether the specific
 * device type requested is currently supported by the provider, and update the
 * efa_mr structure based on the attributes requested by the user.
 *
 * @param[in]	efa_mr	efa_mr structure to be updated
 * @param[in]	attr	a copy of fi_mr_attr updated from the user's registration call
 * @param[in]	flags   MR flags
 *
 * @return FI_SUCCESS or negative FI error code
 */
int efa_mr_hmem_setup(struct efa_mr *efa_mr,
		       const struct fi_mr_attr *attr,
		       uint64_t flags)
{
	if (attr->iface == FI_HMEM_SYSTEM) {
		efa_mr->iface = FI_HMEM_SYSTEM;
		return FI_SUCCESS;
	}

	if (efa_mr->domain->util_domain.info_domain_caps & FI_HMEM) {
		if (g_efa_hmem_info[attr->iface].initialized) {
			efa_mr->iface = attr->iface;
		} else {
			EFA_WARN(FI_LOG_MR,
				"%s is not initialized\n",
				fi_tostr(&attr->iface, FI_TYPE_HMEM_IFACE));
			return -FI_ENOSYS;
		}
	} else {
		/*
		 * It's possible that attr->iface is not initialized when
		 * FI_HMEM is off, so this can't be a fatal error. Print a
		 * warning in case this value is not FI_HMEM_SYSTEM for
		 * whatever reason.
		 */
		EFA_WARN_ONCE(FI_LOG_MR,
		             "FI_HMEM support is disabled, assuming FI_HMEM_SYSTEM instead of %s\n",
		             fi_tostr(&attr->iface, FI_TYPE_HMEM_IFACE));
		efa_mr->iface = FI_HMEM_SYSTEM;
	}

	return FI_SUCCESS;
}

int efa_mr_dereg_impl(struct efa_mr *efa_mr)
{
	int ret = 0;
	int err;
	size_t ibv_mr_size;
	int64_t reg_ct, reg_sz;

	if (efa_mr->ibv_mr) {
		ibv_mr_size = efa_mr->ibv_mr->length;
		err = -ibv_dereg_mr(efa_mr->ibv_mr);
		if (err) {
			EFA_WARN_ERRNO(FI_LOG_MR,
				"ibv_dereg_mr failed", -err);
			ret = err;
		} else {
			reg_ct = ofi_atomic_dec64(&efa_mr->domain->ibv_mr_reg_ct);
			reg_sz = ofi_atomic_sub64(&efa_mr->domain->ibv_mr_reg_sz, ibv_mr_size);
			EFA_INFO(FI_LOG_MR, "Deregistered memory of size %zu for ibv pd %p, total mr reg size %zd, mr reg count %zd\n",
				 ibv_mr_size, efa_mr->domain->ibv_pd, reg_sz, reg_ct);
		}
	}

	efa_mr->ibv_mr = NULL;

	efa_mr->mr_fid.mem_desc = NULL;
	efa_mr->mr_fid.key = FI_KEY_NOTAVAIL;
	return ret;
}

void efa_mr_close_warn_inflight_ope(void **desc, size_t iov_count,
				    struct fi_cq_tagged_entry *cq_entry,
				    struct efa_mr *efa_mr,
				    struct efa_base_ep *base_ep)
{
	size_t i;
	char ep_addr_str[OFI_ADDRSTRLEN] = {0};
	size_t ep_addr_strlen = sizeof(ep_addr_str);

	for (i = 0; i < iov_count; i++) {
		if (desc[i] == efa_mr) {
			efa_base_ep_raw_addr_str(base_ep, ep_addr_str, &ep_addr_strlen);
			EFA_WARN(FI_LOG_MR,
				 "MR %p (key: %ld) on EP %s is being closed "
				 "while an operation is still in flight: "
				 "context=%p, flags=%s, len=%zu, "
				 "buf=%p, data=%lx, tag=%lx.\n",
				 efa_mr, efa_mr->mr_fid.key,
				 ep_addr_str,
				 cq_entry->op_context,
				 fi_tostr(&cq_entry->flags, FI_TYPE_CAPS),
				 cq_entry->len, cq_entry->buf,
				 cq_entry->data, cq_entry->tag);
			desc[i] = NULL;
		}
	}
}

/**
 * @brief Check all in-flight operations for references to a closing EFA direct MR
 *
 * Iterates across all endpoints in the domain and warns about any
 * in-flight operations that still reference the MR being closed.
 *
 * @param[in] efa_mr	The MR being closed
 */
static void efa_mr_close_check_inflight_ope(struct efa_mr *efa_mr)
{
	struct efa_domain *efa_domain = efa_mr->domain;
	struct efa_base_ep *base_ep;
	struct dlist_entry *tmp;

	ofi_genlock_lock(&efa_domain->util_domain.lock);
	dlist_foreach_container(&efa_domain->base_ep_list, struct efa_base_ep,
				base_ep, base_ep_entry) {
		struct efa_direct_ope *direct_ope;
		dlist_foreach_container_safe (
			&base_ep->efa_direct_ope_list,
			struct efa_direct_ope, direct_ope,
			entry, tmp) {
			efa_mr_close_warn_inflight_ope(direct_ope->desc, direct_ope->iov_count, &direct_ope->cq_entry, efa_mr, base_ep);
		}
	}
	ofi_genlock_unlock(&efa_domain->util_domain.lock);
}

static int efa_mr_close(fid_t fid)
{
	struct efa_mr *efa_mr;
	int ret;

	efa_mr = container_of(fid, struct efa_mr, mr_fid.fid);
	if (efa_env.track_mr)
		efa_mr_close_check_inflight_ope(efa_mr);

	ret = efa_mr_dereg_impl(efa_mr);
	if (ret)
		EFA_WARN_FI_ERRNO(FI_LOG_MR, "Unable to close efa_mr", -ret);
	free(efa_mr);
	return ret;
}

struct fi_ops efa_mr_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_mr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

#if HAVE_EFA_DMABUF_MR

static inline
struct ibv_mr *efa_mr_reg_ibv_dmabuf_mr(struct ibv_pd *pd, uint64_t offset,
					size_t len, uint64_t iova, int fd, int access)
{
	return ibv_reg_dmabuf_mr(pd, offset, len, iova, fd, access);
}

#else

static inline
struct ibv_mr *efa_mr_reg_ibv_dmabuf_mr(struct ibv_pd *pd, uint64_t offset,
					size_t len, uint64_t iova, int fd, int access)
{
	EFA_WARN(FI_LOG_MR,
		"ibv_reg_dmabuf_mr is required for memory"
		" registration with FI_MR_DMABUF flags, but "
		" not available in the current rdma-core library."
		" please build libfabric with rdma-core >= 34.0\n");
	return NULL;
}

#endif

static inline void efa_mark_dmabuf_fail(struct efa_hmem_info *i, enum fi_hmem_iface iface)
{
	if (DMABUF_IS_NOT_SUPPORTED(i)) {
		return;
	} else if (DMABUF_IS_ASSUMED(i)) {
		EFA_INFO(FI_LOG_MR,
			 "assumed dmabuf support for %s was incorrect\n",
			 fi_tostr(&iface, FI_TYPE_HMEM_IFACE));
	} else {
		EFA_WARN(FI_LOG_MR,
			 "Explicit dmabuf support for %s was incorrect\n",
			 fi_tostr(&iface, FI_TYPE_HMEM_IFACE));
	}
	i->dmabuf_supported_by_device = EFA_DMABUF_NOT_SUPPORTED;
}

/**
 * @brief Register a memory buffer with rdma-core api.
 *
 * @param efa_mr the ptr to the efa_mr object
 * @param mr_attr the ptr to the fi_mr_attr object
 * @param access the desired memory protection attributes
 * @param flags flags in fi_mr_reg/fi_mr_regattr
 * @return struct ibv_mr* the ptr to the registered MR
 */
static struct ibv_mr *efa_mr_reg_ibv_mr(struct efa_mr *efa_mr,
					 struct fi_mr_attr *mr_attr,
					 int access, const uint64_t flags)
{
	struct efa_hmem_info *p_info = &g_efa_hmem_info[efa_mr->iface];
	struct ibv_mr *dmabuf_mr;
	uint64_t offset;
	int dmabuf_fd;
	int ret;

	/* Explicit dmabuf registration */
	if (flags & FI_MR_DMABUF) {
		if (!mr_attr->dmabuf) {
			EFA_WARN(FI_LOG_MR, "FI_MR_DMABUF set but mr_attr->dmabuf == NULL\n");
			return NULL;
		}
		if (DMABUF_IS_NOT_SUPPORTED(p_info)) {
			EFA_WARN(FI_LOG_MR,
				 "Requested FI_MR_DMABUF, but dmabuf not supported for %s\n",
				 fi_tostr(&efa_mr->iface, FI_TYPE_HMEM_IFACE));
			return NULL;
		}

		EFA_INFO(FI_LOG_MR,
			 "FI_MR_DMABUF: fd=%d offset=%lu len=%zu\n",
			 mr_attr->dmabuf->fd, mr_attr->dmabuf->offset,
			 mr_attr->dmabuf->len);

		dmabuf_mr = efa_mr_reg_ibv_dmabuf_mr(
			efa_mr->domain->ibv_pd,
			mr_attr->dmabuf->offset,
			mr_attr->dmabuf->len,
			(uintptr_t) mr_attr->dmabuf->base_addr + mr_attr->dmabuf->offset,
			mr_attr->dmabuf->fd,
			access);

		if (!dmabuf_mr)  {
			efa_mark_dmabuf_fail(p_info, efa_mr->iface);
		} else {
			p_info->dmabuf_supported_by_device = EFA_DMABUF_SUPPORTED;
		}

		return dmabuf_mr;
	}

	/* Implicit VA path with dmabuf-first */
	if (DMABUF_IS_SUPPORTED(p_info)) {
		ret = ofi_hmem_get_dmabuf_fd(
			efa_mr->iface,
			mr_attr->mr_iov->iov_base,
			(uint64_t) mr_attr->mr_iov->iov_len,
			&dmabuf_fd, &offset);
		if (ret == FI_SUCCESS) {
			/* get fd succeeded */
			EFA_INFO(FI_LOG_MR,
				 "Registering dmabuf MR: fd=%d offset=%lu len=%zu\n",
				 dmabuf_fd, offset, mr_attr->mr_iov->iov_len);

			dmabuf_mr = efa_mr_reg_ibv_dmabuf_mr(
				efa_mr->domain->ibv_pd, offset,
				mr_attr->mr_iov->iov_len,
				(uint64_t)mr_attr->mr_iov->iov_base,
				dmabuf_fd, access);

			/* Close the dmabuf file descriptor - it's no longer needed
			 * after registration
			 */
			(void) ofi_hmem_put_dmabuf_fd(efa_mr->iface, dmabuf_fd);

			if (!dmabuf_mr) {
				efa_mark_dmabuf_fail(p_info, efa_mr->iface);
				if (p_info->dmabuf_fallback_enabled) {
					return ibv_reg_mr(efa_mr->domain->ibv_pd,
						  	  (void *)mr_attr->mr_iov->iov_base,
						  	  mr_attr->mr_iov->iov_len, access);
				}
				return NULL;
			}

			p_info->dmabuf_supported_by_device = EFA_DMABUF_SUPPORTED;
			return dmabuf_mr;

		} else if (p_info->dmabuf_fallback_enabled) {
			/* get fd failed, fallback */
			EFA_INFO(FI_LOG_MR,
				 "Unable to get dmabuf fd for %s device buffer, "
				 "Fall back to ibv_reg_mr\n",
				 fi_tostr(&efa_mr->iface, FI_TYPE_HMEM_IFACE));
			efa_mark_dmabuf_fail(p_info, efa_mr->iface);
			return ibv_reg_mr(efa_mr->domain->ibv_pd,
					  (void *)mr_attr->mr_iov->iov_base,
					  mr_attr->mr_iov->iov_len, access);
		}
		
		/* get fd failed, no fallback */
		EFA_WARN(FI_LOG_MR,
			 "ofi_hmem_get_dmabuf_fd failed for %s: ret=%d (%s)\n",
			 fi_tostr(&efa_mr->iface, FI_TYPE_HMEM_IFACE), ret, fi_strerror(-ret));
		p_info->dmabuf_supported_by_device = EFA_DMABUF_NOT_SUPPORTED;

		return NULL;
	} else {
		/* dmabuf is not supported */
		return ibv_reg_mr(efa_mr->domain->ibv_pd,
				  (void *)mr_attr->mr_iov->iov_base,
				  mr_attr->mr_iov->iov_len, access);
	}
}

/*
 * Set ofi_access to FI_SEND | FI_RECV if not already set,
 * Convert ofi_access flags to ibv_access flags.
 * TODO: Figure out how to split this call for efa-direct and efa-protocol
 * (support emulation) access modes
 */
int efa_mr_ofi_to_ibv_access(uint64_t ofi_access,
			     bool device_support_rdma_read,
			     bool device_support_rdma_write)
{
	int ibv_access = 0;

	/* To support Emulated RMA path, if the access is not supported
	 * by EFA, modify it to FI_SEND | FI_RECV
	 */
	if (!ofi_access || (ofi_access & ~EFA_MR_SUPPORTED_PERMISSIONS))
		ofi_access = FI_SEND | FI_RECV;

	if (ofi_access & FI_RECV)
		ibv_access |= IBV_ACCESS_LOCAL_WRITE;

	if (device_support_rdma_read) {
		if (ofi_access & (FI_REMOTE_WRITE | FI_READ))
			ibv_access |= IBV_ACCESS_LOCAL_WRITE;

		if (ofi_access & (FI_REMOTE_READ | FI_SEND) ||
		    (ofi_access & FI_WRITE && !device_support_rdma_write))
			/* IBV_ACCESS_REMOTE_READ is needed for emulating
			 * fi_send/fi_write with RDMA read */
			ibv_access |= IBV_ACCESS_REMOTE_READ;
	}

	if (device_support_rdma_write && (ofi_access & FI_REMOTE_WRITE))
		/* If IBV_ACCESS_REMOTE_WRITE is set, then
		 * IBV_ACCESS_LOCAL_WRITE must be set too since remote write
		 * should be allowed only if local write is allowed. */
		ibv_access |= IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

	return ibv_access;
}

/*
 * Set the fi_ibv_access modes and do real registration (ibv_mr_reg)
 * Insert the key returned by ibv_mr_reg into efa mr_map and shm mr_map
 */
int efa_mr_reg_impl(struct efa_mr *efa_mr, uint64_t flags, const struct fi_mr_attr *mr_attr)
{
	int64_t reg_sz, reg_ct;
	int ret = 0;
	bool device_support_rdma_read = false;
	bool device_support_rdma_write = false;

	efa_mr->ibv_mr = NULL;
	efa_mr->mr_fid.mem_desc = NULL;
	efa_mr->mr_fid.key = FI_KEY_NOTAVAIL;

	ret = efa_mr_hmem_setup(efa_mr, mr_attr, flags);
	if (ret)
		return ret;

	device_support_rdma_read = efa_mr->domain->device->device_caps & EFADV_DEVICE_ATTR_CAPS_RDMA_READ;
#if HAVE_CAPS_RDMA_WRITE
	device_support_rdma_write = efa_mr->domain->device->device_caps & EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE;
#endif

	efa_mr->ibv_mr = efa_mr_reg_ibv_mr(
		efa_mr, (struct fi_mr_attr *)mr_attr,
		efa_mr_ofi_to_ibv_access(mr_attr->access,
					 device_support_rdma_read,
					 device_support_rdma_write),
		flags);
	if (!efa_mr->ibv_mr) {
		/* error path that doesn't call ibv_reg doesn't set errno */
		if (!errno)
			errno = ENOTSUP;
		EFA_WARN(FI_LOG_MR,
			 "Unable to register MR of %zu bytes: %s, "
			 "flags %ld, ibv pd: %p, total mr "
			 "reg size %zd, mr reg count %zd\n",
			 (flags & FI_MR_DMABUF) ?
				 mr_attr->dmabuf->len :
				 mr_attr->mr_iov->iov_len,
			 strerror(errno), flags,
			 efa_mr->domain->ibv_pd,
			 ofi_atomic_get64(&efa_mr->domain->ibv_mr_reg_sz),
			 ofi_atomic_get64(&efa_mr->domain->ibv_mr_reg_ct));
		return -errno;
	}
	reg_ct = ofi_atomic_inc64(&efa_mr->domain->ibv_mr_reg_ct);
	reg_sz = ofi_atomic_add64(&efa_mr->domain->ibv_mr_reg_sz, efa_mr->ibv_mr->length);
	EFA_INFO(FI_LOG_MR,
		 "Registered memory at %p of size %zu"
		 "flags %ld for ibv pd %p, "
		 "total mr reg size %zd, mr reg count %zd\n",
		 efa_mr->ibv_mr->addr, efa_mr->ibv_mr->length,
		 flags, efa_mr->domain->ibv_pd,
		 reg_sz,
		 reg_ct);
	efa_mr->mr_fid.key = efa_mr->ibv_mr->rkey;
	efa_mr->mr_fid.mem_desc = efa_mr;
	assert(efa_mr->mr_fid.key != FI_KEY_NOTAVAIL);

	return 0;
}

/* Core EFA MR registration - basic ibv_mr_reg/dereg operations */
static int efa_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
			  uint64_t flags, struct fid_mr **mr_fid)
{
	struct efa_domain *domain;
	struct efa_mr *efa_mr = NULL;
	int ret = 0;
	struct fi_mr_attr mr_attr = {0};
	bool device_support_rdma_read = false;
	bool device_support_rdma_write = false;

	ret = efa_mr_regattr_validate(fid, attr, flags);
	if (ret)
		return ret;

	domain = container_of(fid, struct efa_domain, util_domain.domain_fid.fid);

	device_support_rdma_read = domain->device->device_caps & EFADV_DEVICE_ATTR_CAPS_RDMA_READ;
#if HAVE_CAPS_RDMA_WRITE
	device_support_rdma_write = domain->device->device_caps & EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE;
#endif

	/* For efa-direct, fail registration if RDMA operations are requested 
	 * but hardware doesn't support them
	 */
	if ((attr->access & (FI_READ | FI_REMOTE_READ)) &&
		!device_support_rdma_read) {
		EFA_WARN(FI_LOG_MR, "FI_READ or FI_REMOTE_READ "
					"requested but hardware does not "
					"support RDMA read operations\n");
		return -FI_EOPNOTSUPP;
	}
	if (attr->access & (FI_WRITE | FI_REMOTE_WRITE) &&
		!device_support_rdma_write) {
		EFA_WARN(FI_LOG_MR, "FI_WRITE or FI_REMOTE_WRITE "
					"requested but hardware does not "
					"support RDMA write operations\n");
		return -FI_EOPNOTSUPP;
	}

	efa_mr = calloc(1, sizeof(*efa_mr));
	if (!efa_mr) {
		EFA_WARN(FI_LOG_MR, "Unable to initialize md\n");
		return -FI_ENOMEM;
	}

	efa_mr->domain = domain;
	efa_mr->mr_fid.fid.fclass = FI_CLASS_MR;
	efa_mr->mr_fid.fid.context = attr->context;
	efa_mr->mr_fid.fid.ops = &efa_mr_ops;

	/* Update attr for ABI compatibility */
	ofi_mr_update_attr(
		efa_mr->domain->util_domain.fabric->fabric_fid.api_version,
		efa_mr->domain->util_domain.info_domain_caps,
		attr, &mr_attr, flags);

	ret = efa_mr_reg_impl(efa_mr, flags, &mr_attr);
	if (ret)
		goto err;

	*mr_fid = &efa_mr->mr_fid;
	return 0;
err:
	EFA_WARN_FI_ERRNO(FI_LOG_MR, "Unable to register efa_mr", -ret);
	free(efa_mr);
	return ret;
}

static int efa_mr_regv(struct fid *fid, const struct iovec *iov,
		       size_t count, uint64_t access, uint64_t offset,
		       uint64_t requested_key, uint64_t flags,
		       struct fid_mr **mr_fid, void *context)
{
	struct fi_mr_attr attr = EFA_MR_ATTR_INIT_SYSTEM(iov, count, access, offset, requested_key, context);

	return efa_mr_regattr(fid, &attr, flags, mr_fid);
}

static int efa_mr_reg(struct fid *fid, const void *buf, size_t len,
		      uint64_t access, uint64_t offset,
		      uint64_t requested_key, uint64_t flags,
		      struct fid_mr **mr_fid, void *context)
{
	struct iovec iov = {.iov_base = (void *)buf, .iov_len = len};
	struct fi_mr_attr attr = EFA_MR_ATTR_INIT_SYSTEM(&iov, 1, access, offset, requested_key, context);

	return efa_mr_regattr(fid, &attr, flags, mr_fid);
}

struct fi_ops_mr efa_domain_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = efa_mr_reg,
	.regv = efa_mr_regv,
	.regattr = efa_mr_regattr,
};

