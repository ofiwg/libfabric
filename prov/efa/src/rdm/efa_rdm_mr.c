/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "config.h"
#include <ofi_util.h>
#include "efa.h"
#include "rdm/efa_rdm_mr.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_ope.h"
#if HAVE_CUDA
#include <cuda.h>
#endif

/*
 * Initial values for internal keygen functions to generate MR keys
 * (efa_mr->mr_fid.key)
 *
 * Typically the rkey returned from ibv_reg_mr() (ibv_mr->rkey) would be used.
 * In cases where ibv_reg_mr() should be avoided, we use proprietary MR key
 * generation instead.
 *
 * Initial values should be > UINT32_MAX to avoid collisions with ibv_mr rkeys,
 * and should be sufficiently spaced apart s.t. they don't collide with each
 * other.
 */
#define NON_P2P_MR_KEYGEN_INIT	BIT_ULL(32)

/*
 * Disable MR cache by default under ASAN. The memhooks monitor is
 * incompatible with ASAN, and userfaultfd may not be available on
 * all systems. Without a working monitor, efa_mr_cache_open fails.
 * Can be overridden at runtime with FI_EFA_MR_CACHE_ENABLE=1.
 */
#ifdef ENABLE_ASAN
int efa_mr_cache_enable	= 0;
#else
int efa_mr_cache_enable	= 1;
#endif
size_t efa_mr_max_cached_count;
size_t efa_mr_max_cached_size;

/*
 * Initial values for internal keygen functions to generate MR keys
 * (efa_mr->mr_fid.key)
 *
 * Typically the rkey returned from ibv_reg_mr() (ibv_mr->rkey) would be used.
 * In cases where ibv_reg_mr() should be avoided, we use proprietary MR key
 * generation instead.
 *
 * Initial values should be > UINT32_MAX to avoid collisions with ibv_mr rkeys,
 * and should be sufficiently spaced apart s.t. they don't collide with each
 * other.
 */
#define NON_P2P_MR_KEYGEN_INIT	BIT_ULL(32)

/* @brief Setup the MR cache.
 *
 * This function enables the MR cache using the util MR cache code.
 *
 * @param cache		The ofi_mr_cache that is to be set up.
 * @param domain	The EFA domain where cache will be used.
 * @return 0 on success, fi_errno on failure.
 */
int efa_rdm_mr_cache_open(struct ofi_mr_cache **cache, struct efa_domain *domain)
{
	struct ofi_mem_monitor *memory_monitors[OFI_HMEM_MAX] = {
		[FI_HMEM_SYSTEM] = default_monitor,
		[FI_HMEM_CUDA] = cuda_monitor,
		[FI_HMEM_ROCR] = rocr_monitor,
	};
	int err;

	/* Both Open MPI (and possibly other MPI implementations) and
	 * Libfabric use the same live binary patching to enable memory
	 * monitoring, but the patching technique only allows a single
	 * "winning" patch.  The Libfabric memhooks monitor will not
	 * overwrite a previous patch, but instead return
	 * -FI_EALREADY.  There are three cases of concern, and in all
	 * but one of them, we can avoid changing the default monitor.
	 *
	 * (1) Upper layer does not patch, such as Open MPI 4.0 and
	 * earlier.  In this case, the default monitor will be used,
	 * as the default monitor is either not the memhooks monitor
	 * (because the user specified a different monitor) or the
	 * default monitor is the memhooks monitor, but we were able
	 * to install the patches.  We will use the default monitor in
	 * this case.
	 *
	 * (2) Upper layer does patch, but does not export a memory
	 * monitor, such as Open MPI 4.1.0 and 4.1.1.  In this case,
	 * if the default memory monitor is not memhooks, we will use
	 * the default monitor.  If the default monitor is memhooks,
	 * the patch will fail to apply, and we will change the
	 * requested monitor to UFFD to avoid a broken configuration.
	 * If the user explicitly requested memhooks, we will return
	 * an error, as we can not satisfy that request.
	 *
	 * (3) Upper layer does patch and exports a memory monitor,
	 * such as Open MPI 4.1.2 and later.  In this case, the
	 * default monitor will have been changed from the memhooks
	 * monitor to the imported monitor, so we will use the
	 * imported monitor.
	 *
	 * The only known cases in which we will not use the default
	 * monitor are Open MPI 4.1.0/4.1.1.
	 *
	 * It is possible that this could be better handled at the
	 * mem_monitor level in Libfabric, but so far we have not
	 * reached agreement on how that would work.
	 */
	if (default_monitor == memhooks_monitor) {
		err = memhooks_monitor->start(memhooks_monitor);
		if (err == -FI_EALREADY) {
			if (cache_params.monitor) {
				EFA_WARN(FI_LOG_DOMAIN,
					 "Memhooks monitor requested via FI_MR_CACHE_MONITOR, but memhooks failed to\n"
					 "install.  No working monitor availale.\n");
				return -FI_ENOSYS;
			}
			EFA_INFO(FI_LOG_DOMAIN,
				 "Detected potential memhooks monitor conflict. Switching to UFFD.\n");
			memory_monitors[FI_HMEM_SYSTEM] = uffd_monitor;
		}
	} else if (default_monitor == NULL) {
		/* TODO: Fail if we don't find a system monitor.  This
		 * is a debatable decision, as the VERBS provider
		 * falls back to a no-cache mode in this case.  We
		 * fail the domain creation because the rest of the MR
		 * code hasn't been audited to deal with a NULL
		 * monitor.
		 */
		EFA_WARN(FI_LOG_DOMAIN,
			 "No default SYSTEM monitor available.\n");
		return -FI_ENOSYS;
	}

	*cache = (struct ofi_mr_cache *)calloc(1, sizeof(struct ofi_mr_cache));
	if (!*cache)
		return -FI_ENOMEM;

	if (!efa_mr_max_cached_count)
		efa_mr_max_cached_count = domain->device->ibv_attr.max_mr *
					  EFA_MR_CACHE_LIMIT_MULT;
	if (!efa_mr_max_cached_size)
		efa_mr_max_cached_size = domain->device->ibv_attr.max_mr_size *
					 EFA_MR_CACHE_LIMIT_MULT;
	/*
	 * TODO: we're modifying a global in the util mr cache? do we need an
	 * API here instead?
	 */
	cache_params.max_cnt = efa_mr_max_cached_count;
	cache_params.max_size = efa_mr_max_cached_size;
	(*cache)->entry_data_size = sizeof(struct efa_rdm_mr);
	(*cache)->add_region = efa_rdm_mr_cache_entry_reg;
	(*cache)->delete_region = efa_rdm_mr_cache_entry_dereg;
	err = ofi_mr_cache_init(&domain->util_domain, memory_monitors,
				*cache);
	if (err) {
		EFA_WARN(FI_LOG_DOMAIN, "EFA MR cache init failed: %s\n",
		         fi_strerror(err));
		free(*cache);
		*cache = NULL;
		return -err;
	}

	EFA_INFO(FI_LOG_DOMAIN, "EFA MR cache enabled, max_cnt: %zu max_size: %zu\n",
		 cache_params.max_cnt, cache_params.max_size);
	return 0;
}

static int efa_rdm_mr_cache_close(fid_t fid)
{
	struct efa_mr *efa_mr = container_of(fid, struct efa_mr,
					       mr_fid.fid);

	/* Safe cast: efa_mr is first member of efa_rdm_mr, verified by static assertion */
	ofi_mr_cache_delete(efa_mr->domain->cache, ((struct efa_rdm_mr *) efa_mr)->entry);

	return 0;
}

static struct fi_ops efa_rdm_mr_cache_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_rdm_mr_cache_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

/*
 * Since ibv_reg_mr() will fail for accelerator buffers when p2p is unavailable
 * (and thus isn't called), generate a proprietary internal key for
 * efa_mr->mr_fid.key. The key must be larger than UINT32_MAX to avoid potential
 * collisions with keys returned by ibv_reg_mr() for standard MR registrations.
 */
static uint64_t efa_rdm_mr_non_p2p_keygen(void) {
	static uint64_t NON_P2P_MR_KEYGEN = NON_P2P_MR_KEYGEN_INIT;
	return NON_P2P_MR_KEYGEN++;
}

#if HAVE_CUDA
static inline
int efa_rdm_mr_is_cuda_memory_freed(struct efa_rdm_mr *efa_rdm_mr, bool *freed)
{
	int err;
	uint64_t buffer_id;

	err = ofi_cuPointerGetAttribute(&buffer_id, CU_POINTER_ATTRIBUTE_BUFFER_ID,
					(CUdeviceptr)efa_rdm_mr->efa_mr.ibv_mr->addr);

	if (err == CUDA_ERROR_INVALID_VALUE) {
		/* According to CUDA document, the return code of ofi_cuPointerGetAttribute() being CUDA_ERROR_INVALID_VALUE
		 * means the efa_mr's pointer is NOT allocated by, mapped by or registered with CUDA.
		 * Because the address was registered, the only possiblity is that the memory has been freed.
		 */
		*freed = true;
		return 0;
	}

	if (!err) {
		/* Buffer ID mismatch means the original buffer was freed, and a new buffer has been
		 * allocated with the same address
		 */
		*freed = (buffer_id != efa_rdm_mr->entry->hmem_info.cuda_id);
		return 0;
	}

	EFA_WARN(FI_LOG_DOMAIN, "cuPointerGetAttribute() failed with error code: %d, error message: %s\n",
		 err, ofi_cudaGetErrorString(err));
	return -FI_EINVAL;
}

/**
 * @brief update the mr_map inside util_domain with a new memory registration.
 *
 * The mr_map is in util domain is a map between MR key and MR, and has all
 * the active MR in it. This function add the information of a new MR into
 * the mr_map.
 *
 * @param	efa_rdm_mr	the pointer to the efa_rdm_mr object
 * @param	mr_attr		the pointer to an fi_mr_attr object.
 * @return	0 on success.
 * 		negative libfabric error code on failure.
 */
static
int efa_rdm_mr_update_domain_mr_map(struct efa_rdm_mr *efa_rdm_mr, struct fi_mr_attr *mr_attr,
				     uint64_t flags)
{
	struct fid_mr *existing_mr_fid;
	struct efa_mr *existing_mr;
	bool cuda_memory_freed;
	int err;
	struct efa_mr *efa_mr = &efa_rdm_mr->efa_mr;

	mr_attr->requested_key = efa_mr->mr_fid.key;
	ofi_genlock_lock(&efa_mr->domain->util_domain.lock);
	err = ofi_mr_map_insert(&efa_mr->domain->util_domain.mr_map, mr_attr,
				&efa_mr->mr_fid.key, &efa_mr->mr_fid, flags);
	ofi_genlock_unlock(&efa_mr->domain->util_domain.lock);
	if (!err)
		return 0;

	/* There is a special error that we can recover from, which is:
	 * 1. error code is FI_ENOKEY, which means there is already a MR in mr_map with the same key
	 * 2. that MR is for a CUDA memory region
	 * 3. that CUDA memory region has been freed.
	 *
	 * This situation can happen because the combination of the following 3 things:
	 * 1. cuda memory uses memory cache.
	 * 2. cuda memory cache's monitor is not monitoring call to cudaFree().
	 * 3. kernel released the kernel space of memory registration when cudaFree() is called on a registered cuda memory.
	 *
	 * Therefore, when application call cudaFree() on a registered CUDA memory region, the EFA kernel module
	 * device released the kernel space memory region, made the key available for reuse.
	 * However, libfabric is not aware of the incident, and kept the user space memory registration in its
	 * MR cache and MR map.
	 *
	 * After we implement cuda memhook monitor (which monitors call to cudaFree() and remove dead region from MR cache),
	 * we should NOT encounter this special situation any more. At that time, this function should be removed.
	 */
	if (err != -FI_ENOKEY) {
		/* no way we can recover from this error, return error code */
		EFA_WARN(FI_LOG_MR,
			"Unable to add MR to map. errno: %d errmsg: (%s) key: %ld buff: %p hmem_iface: %s len: %zu\n",
			err,
			fi_strerror(-err),
			efa_mr->mr_fid.key,
			mr_attr->mr_iov->iov_base,
			fi_tostr(&mr_attr->iface, FI_TYPE_HMEM_IFACE),
			mr_attr->mr_iov->iov_len);
		return err;
	}

	existing_mr_fid = ofi_mr_map_get(&efa_mr->domain->util_domain.mr_map, efa_mr->mr_fid.key);
	assert(existing_mr_fid);
	existing_mr = container_of(existing_mr_fid, struct efa_mr, mr_fid);

	if (existing_mr->iface != FI_HMEM_CUDA) {
		/* no way we can recover from this situation, return error code */
		EFA_WARN(FI_LOG_DOMAIN, "key %ld already assigned to buffer: %p hmem_iface: %s length: %ld\n",
			 existing_mr->mr_fid.key,
			 existing_mr->ibv_mr->addr,
			 fi_tostr(&existing_mr->iface, FI_TYPE_HMEM_IFACE),
			 existing_mr->ibv_mr->length);
		return -FI_ENOKEY;
	}

	err = efa_rdm_mr_is_cuda_memory_freed((struct efa_rdm_mr *)existing_mr, &cuda_memory_freed);
	if (err)
		return err;

	if (!cuda_memory_freed) {
		/* The same key was assigned to two valid cuda memory region,
		 * there is no way we can recover from this situation, return error code */
		EFA_WARN(FI_LOG_DOMAIN, "key %ld has already assigned to another cuda buffer: %p length: %ld\n",
			 existing_mr->mr_fid.key,
			 existing_mr->ibv_mr->addr,
			 existing_mr->ibv_mr->length);
		return -FI_ENOKEY;
	}

	EFA_INFO(FI_LOG_DOMAIN, "key %ld has been assigned to cuda buffer: %p length: %ld, which has since been freed\n",
		 existing_mr->mr_fid.key,
		 existing_mr->ibv_mr->addr,
		 existing_mr->ibv_mr->length);

	/* this can only happen when MR cache is enabled, hence the assertion */
	assert(efa_mr->domain->cache);
	pthread_mutex_lock(&mm_lock);
	ofi_mr_cache_notify(efa_mr->domain->cache, existing_mr->ibv_mr->addr, existing_mr->ibv_mr->length);
	pthread_mutex_unlock(&mm_lock);

	/* due to MR cache's deferred de-registration, ofi_mr_cache_notify() only move the region to dead_region_list
	 * ofi_mr_cache_flush() will actually remove the region from cache.
	 * lru is a list of regions that are still active, so we set flush_lru to false.
	 */
	ofi_mr_cache_flush(efa_mr->domain->cache, false /*flush_lru */);

	/*
	 * When MR cache removes a MR, it will call its delete_region() call back. delete_region() calls efa_mr_dereg_impl(),
	 * which should remove the staled entry from MR map. So insert again here.
	 */
	ofi_genlock_lock(&efa_mr->domain->util_domain.lock);
	err = ofi_mr_map_insert(&efa_mr->domain->util_domain.mr_map, mr_attr,
				&efa_mr->mr_fid.key, &efa_mr->mr_fid, flags);
	ofi_genlock_unlock(&efa_mr->domain->util_domain.lock);
	if (err) {
		EFA_WARN(FI_LOG_MR,
			"Unable to add MR to map, even though we already tried to evict staled memory registration."
			"errno: %d errmsg: (%s) key: %ld buff: %p hmem_iface: %s len: %zu\n",
			err,
			fi_strerror(-err),
			efa_mr->mr_fid.key,
			mr_attr->mr_iov->iov_base,
			fi_tostr(&mr_attr->iface, FI_TYPE_HMEM_IFACE),
			mr_attr->mr_iov->iov_len);
		return err;
	}

	return 0;
}
#else /* HAVE_CUDA */
static
int efa_rdm_mr_update_domain_mr_map(struct efa_rdm_mr *efa_rdm_mr, struct fi_mr_attr *mr_attr,
				     uint64_t flags)
{
	int err;
	struct efa_mr *efa_mr = &efa_rdm_mr->efa_mr;

	mr_attr->requested_key = efa_mr->mr_fid.key;
	ofi_genlock_lock(&efa_mr->domain->util_domain.lock);
	err = ofi_mr_map_insert(&efa_mr->domain->util_domain.mr_map, mr_attr,
				&efa_mr->mr_fid.key, &efa_mr->mr_fid, flags);
	ofi_genlock_unlock(&efa_mr->domain->util_domain.lock);
	if (err) {
		EFA_WARN(FI_LOG_MR,
			"Unable to add MR to map. errno: %d errmsg: (%s) key: %ld buff: %p hmem_iface: %s len: %zu\n",
			err,
			fi_strerror(-err),
			efa_mr->mr_fid.key,
			mr_attr->mr_iov->iov_base,
			fi_tostr(&mr_attr->iface, FI_TYPE_HMEM_IFACE),
			mr_attr->mr_iov->iov_len);
		return err;
	}

	return 0;
}

#endif /* HAVE_CUDA */

/* RDM MR deregistration implementation */
static int efa_rdm_mr_dereg_impl(struct efa_rdm_mr *efa_rdm_mr)
{
	struct efa_domain *efa_domain;
	int ret = 0;
	int err;

	if (efa_rdm_mr->shm_mr) {
		ret = fi_close(&efa_rdm_mr->shm_mr->fid);
		if (ret)
			return ret;
		efa_rdm_mr->shm_mr = NULL;
	}

	if (efa_rdm_mr->inserted_to_mr_map) {
		efa_domain = efa_rdm_mr->efa_mr.domain;
		ofi_genlock_lock(&efa_domain->util_domain.lock);
		err = ofi_mr_map_remove(&efa_domain->util_domain.mr_map,
					efa_rdm_mr->efa_mr.mr_fid.key);
		ofi_genlock_unlock(&efa_domain->util_domain.lock);

		if (err) {
			EFA_WARN(FI_LOG_MR,
				"Unable to remove MR entry from util map (%s)\n",
				fi_strerror(-err));
			ret = err;
		}
		efa_rdm_mr->inserted_to_mr_map = false;
	}

	/* RDM-specific: GDRCopy cleanup */
	if (efa_rdm_mr->efa_mr.iface == FI_HMEM_CUDA &&
	    (efa_rdm_mr->flags & OFI_HMEM_DATA_DEV_REG_HANDLE)) {
		assert(efa_rdm_mr->hmem_data);
		int err = ofi_hmem_dev_unregister(FI_HMEM_CUDA, (uint64_t)efa_rdm_mr->hmem_data);
		if (err) {
			EFA_WARN(FI_LOG_MR, "Unable to de-register cuda handle\n");
			ret = err;
		}
		efa_rdm_mr->hmem_data = NULL;
	}

	return efa_mr_dereg_impl(&efa_rdm_mr->efa_mr);
}

/**
 * @brief Populate efa_rdm_mr struct's hmem fields
 *
 * Update the efa_rdm_mr structure based on the attributes requested by the user.
 *
 * @param[in]	efa_rdm_mr	efa_rdm_mr structure to be updated
 * @param[in]	attr	a copy of fi_mr_attr updated from the user's registration call
 * @param[in]	flags   MR flags
 *
 */
static void efa_rdm_mr_hmem_setup(struct efa_rdm_mr *efa_rdm_mr,
                             const struct fi_mr_attr *attr,
							 uint64_t flags)
{
	efa_rdm_mr->needs_sync = false;
	efa_rdm_mr->hmem_data = NULL;
	efa_rdm_mr->flags = flags & ~OFI_HMEM_DATA_DEV_REG_HANDLE;
	efa_rdm_mr->device = 0;

	/* RDM-specific: GDRCopy registration for CUDA */
	if (attr->iface == FI_HMEM_CUDA) {
		efa_rdm_mr->device = attr->device.cuda;
		efa_rdm_mr->needs_sync = true;

		if (!(flags & FI_MR_DMABUF) && cuda_is_gdrcopy_enabled()) {
			struct iovec mr_iov = *attr->mr_iov;
			int err = ofi_hmem_dev_register(FI_HMEM_CUDA, mr_iov.iov_base, mr_iov.iov_len,
							(uint64_t *)&efa_rdm_mr->hmem_data);
			efa_rdm_mr->flags |= OFI_HMEM_DATA_DEV_REG_HANDLE;
			if (err) {
				EFA_WARN(FI_LOG_MR,
					 "Unable to register handle for GPU memory. err: %d buf: %p len: %zu\n",
					 err, mr_iov.iov_base, mr_iov.iov_len);
				/* When gdrcopy pin buf failed, fallback to cudaMemcpy */
				efa_rdm_mr->hmem_data = NULL;
				efa_rdm_mr->flags &= ~OFI_HMEM_DATA_DEV_REG_HANDLE;
			}
		}
	} else if (attr->iface == FI_HMEM_ROCR) {
		efa_rdm_mr->device = attr->device.rocr;
	} else if (attr->iface == FI_HMEM_NEURON) {
		efa_rdm_mr->device = attr->device.neuron;
	} else if (attr->iface == FI_HMEM_SYNAPSEAI) {
		efa_rdm_mr->device = attr->device.synapseai;
	}
}

/* RDM MR registration implementation - wraps core EFA MR registration */
static int efa_rdm_mr_reg_impl(struct efa_rdm_mr *efa_rdm_mr, uint64_t flags,
			       const struct fi_mr_attr *mr_attr)
{
	int ret;

	/* RDM-specific: MR cache flush */
	if (efa_rdm_mr->efa_mr.domain->cache)
		ofi_mr_cache_flush(efa_rdm_mr->efa_mr.domain->cache, false);

	/*
	 * For cuda and rocr iface when p2p is unavailable, skip ibv_reg_mr() and
	 * generate proprietary mr_fid key.
	 */
	if ((mr_attr->iface == FI_HMEM_CUDA || mr_attr->iface == FI_HMEM_ROCR)
		&& !g_efa_hmem_info[mr_attr->iface].p2p_supported_by_device) {
		efa_rdm_mr->efa_mr.mr_fid.key = efa_rdm_mr_non_p2p_keygen();
	} else {
		/* base mr registration (ibv mr), must be called the first before RDM specific fields are setup */
		ret = efa_mr_reg_impl(&efa_rdm_mr->efa_mr, flags, mr_attr);
		if (ret)
			return ret;
	}

	/* Initialize RDM-specific fields */
	efa_rdm_mr->inserted_to_mr_map = false;
	efa_rdm_mr->shm_mr = NULL;
	efa_rdm_mr->entry = NULL;

	/* RDM specific mr hmem setup */
	efa_rdm_mr_hmem_setup(efa_rdm_mr, mr_attr, flags);
	/* RDM-specific: Update domain MR map */
	assert(efa_rdm_mr->efa_mr.mr_fid.key != FI_KEY_NOTAVAIL);
	ret = efa_rdm_mr_update_domain_mr_map(efa_rdm_mr, (struct fi_mr_attr *)mr_attr, flags);
	if (ret) {
		ret = efa_rdm_mr_dereg_impl(efa_rdm_mr);
		return ret;
	}

	efa_rdm_mr->inserted_to_mr_map = true;

	return 0;
}

/**
 * @brief Check all in-flight operations for references to a closing EFA RDM MR
 *
 * Iterates across all endpoints in the domain and warns about any
 * in-flight operations that still reference the MR being closed.
 *
 * @param[in] efa_mr	The MR being closed
 */
static void efa_rdm_mr_close_check_inflight_ope(struct efa_mr *efa_mr)
{
	struct efa_domain *efa_domain = efa_mr->domain;
	struct efa_base_ep *base_ep;
	struct dlist_entry *tmp;

	ofi_genlock_lock(&efa_domain->util_domain.lock);
	dlist_foreach_container(&efa_domain->base_ep_list, struct efa_base_ep,
				base_ep, base_ep_entry) {
		struct efa_rdm_ep *rdm_ep;
		struct efa_rdm_ope *ope;
		rdm_ep = container_of(base_ep, struct efa_rdm_ep, base_ep);
		dlist_foreach_container_safe (
			&rdm_ep->txe_list, struct efa_rdm_ope,
			ope, ep_entry, tmp) {
			efa_mr_close_warn_inflight_ope(ope->desc, ope->iov_count, &ope->cq_entry, efa_mr, base_ep);
		}
		dlist_foreach_container_safe (
			&rdm_ep->rxe_list, struct efa_rdm_ope,
			ope, ep_entry, tmp) {
			efa_mr_close_warn_inflight_ope(ope->desc, ope->iov_count, &ope->cq_entry, efa_mr, base_ep);
		}
	}
	ofi_genlock_unlock(&efa_domain->util_domain.lock);
}

/* RDM MR close operation */
static int efa_rdm_mr_close(fid_t fid)
{
	struct efa_rdm_mr *efa_rdm_mr;
	int ret, err;

	efa_rdm_mr = container_of(fid, struct efa_rdm_mr, efa_mr.mr_fid.fid);
	if (efa_env.track_mr)
		efa_rdm_mr_close_check_inflight_ope(&efa_rdm_mr->efa_mr);

	if (efa_rdm_mr->shm_mr) {
		err = fi_close(&efa_rdm_mr->shm_mr->fid);
		if (err) {
			EFA_WARN(FI_LOG_MR,
				"Unable to close shm MR: %s\n", fi_strerror(err));
			ret = err;
		}
		efa_rdm_mr->shm_mr = NULL;
	}

	ret = efa_rdm_mr_dereg_impl(efa_rdm_mr);
	if (ret)
		EFA_WARN(FI_LOG_MR, "Unable to close efa MR: %s\n", fi_strerror(ret));

	free(efa_rdm_mr);
	return 0;
}

static struct fi_ops efa_rdm_mr_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_rdm_mr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int efa_rdm_mr_cache_entry_reg(struct ofi_mr_cache *cache,
			       struct ofi_mr_entry *entry)
{
	int ret = 0;
	/* TODO
	 * Since, access is not passed as a parameter to efa_rdm_mr_cache_entry_reg,
	 * for now we will set access to all supported access modes. Once access
	 * information is available this can be removed.
	 * Issue: https://github.com/ofiwg/libfabric/issues/5677
	 */
	uint64_t access = EFA_MR_SUPPORTED_PERMISSIONS;
	struct fi_mr_attr attr = {0};
	struct efa_mr *efa_mr = (struct efa_mr *)entry->data;

	efa_mr->domain = container_of(cache->domain, struct efa_domain,
				      util_domain);
	efa_mr->mr_fid.fid.ops = &efa_rdm_mr_cache_ops;
	efa_mr->mr_fid.fid.fclass = FI_CLASS_MR;
	efa_mr->mr_fid.fid.context = NULL;

	attr.mr_iov = &entry->info.iov;
	/* ofi_mr_info only stores one iov */
	attr.iov_count = 1;
	attr.access = access;
	attr.offset = 0;
	attr.requested_key = 0;
	attr.context = NULL;
	attr.iface = entry->info.iface;

	if (attr.iface == FI_HMEM_CUDA)
		attr.device.cuda = entry->info.device;
	else if (attr.iface == FI_HMEM_NEURON)
		attr.device.neuron = entry->info.device;
	else if (attr.iface == FI_HMEM_SYNAPSEAI)
		attr.device.synapseai = entry->info.device;

	/* Safe cast: MR cache allocates full efa_rdm_mr structure (entry_data_size) */
	ret = efa_rdm_mr_reg_impl((struct efa_rdm_mr *)efa_mr, 0, &attr);
	return ret;
}

void efa_rdm_mr_cache_entry_dereg(struct ofi_mr_cache *cache,
				   struct ofi_mr_entry *entry)
{
	struct efa_mr *efa_mr = (struct efa_mr *)entry->data;
	int ret;

	/* Safe cast: MR cache allocates full efa_rdm_mr structure (entry_data_size) */
	ret = efa_rdm_mr_dereg_impl((struct efa_rdm_mr *)efa_mr);
	if (ret)
		EFA_WARN(FI_LOG_MR, "Unable to dereg mr: %d\n", ret);
}

static int efa_rdm_mr_cache_regattr(struct fid *fid, const struct fi_mr_attr *attr,
				     uint64_t flags, struct fid_mr **mr_fid)
{
	struct efa_domain *domain;
	struct efa_rdm_mr *efa_rdm_mr;
	struct ofi_mr_entry *entry;
	struct ofi_mr_info info = {0};
	int ret;

	if (attr->iov_count > EFA_MR_IOV_LIMIT) {
		EFA_WARN(FI_LOG_MR, "iov count > %d not supported\n",
			 EFA_MR_IOV_LIMIT);
		return -FI_EINVAL;
	}

	if (!ofi_hmem_is_initialized(attr->iface)) {
		EFA_WARN(FI_LOG_MR,
			"Cannot register memory for uninitialized iface (%s)\n",
			fi_tostr(&attr->iface, FI_TYPE_HMEM_IFACE));
		return -FI_ENOSYS;
	}

	domain = container_of(fid, struct efa_domain,
			      util_domain.domain_fid.fid);

	assert(attr->iov_count > 0 && attr->iov_count <= domain->info->domain_attr->mr_iov_limit);
	ofi_mr_info_get_iov_from_mr_attr(&info, attr, flags);
	info.iface = attr->iface;
	info.device = attr->device.reserved;
	ret = ofi_mr_cache_search(domain->cache, &info, &entry);
	if (OFI_UNLIKELY(ret))
		return ret;

	/* Safe cast: MR cache allocates full efa_rdm_mr structure (entry_data_size) */
	efa_rdm_mr = (struct efa_rdm_mr *)entry->data;
	efa_rdm_mr->entry = entry;

	*mr_fid = &efa_rdm_mr->efa_mr.mr_fid;
	return 0;
}

int efa_rdm_mr_cache_regv(struct fid_domain *domain_fid, const struct iovec *iov,
			  size_t count, uint64_t access, uint64_t offset,
			  uint64_t requested_key, uint64_t flags,
			  struct fid_mr **mr, void *context)
{
	struct fi_mr_attr attr = EFA_MR_ATTR_INIT_SYSTEM(iov, count, access, offset, requested_key, context);

	return efa_rdm_mr_cache_regattr(&domain_fid->fid, &attr, flags, mr);
}

int efa_rdm_mr_internal_regv(struct fid_domain *domain_fid, const struct iovec *iov,
			     size_t count, uint64_t access, uint64_t offset,
			     uint64_t requested_key, uint64_t flags,
			     struct fid_mr **mr_fid, void *context)
{
	struct fi_mr_attr attr = EFA_MR_ATTR_INIT_SYSTEM(iov, count, access, offset, requested_key, context);
	struct efa_rdm_mr *efa_rdm_mr;
	struct efa_domain *domain;
	int ret;
	*mr_fid = NULL;

	domain = container_of(domain_fid, struct efa_domain,
			      util_domain.domain_fid);
	efa_rdm_mr = calloc(1, sizeof(*efa_rdm_mr));
	if (!efa_rdm_mr) {
		EFA_WARN(FI_LOG_MR, "Unable to initialize MR\n");
		return -FI_ENOMEM;
	}

	efa_rdm_mr->efa_mr.domain = domain;
	efa_rdm_mr->efa_mr.mr_fid.fid.fclass = FI_CLASS_MR;
	efa_rdm_mr->efa_mr.mr_fid.fid.context = attr.context;
	efa_rdm_mr->efa_mr.mr_fid.fid.ops = &efa_rdm_mr_ops;

	ret = efa_rdm_mr_reg_impl(efa_rdm_mr, flags, &attr);
	if (ret) {
		EFA_WARN(FI_LOG_MR, "Unable to register MR: %s\n",
			fi_strerror(-ret));
		free(efa_rdm_mr);
		return ret;
	}
	*mr_fid = &efa_rdm_mr->efa_mr.mr_fid;
	return 0;
}


/* RDM MR registration - handles SHM integration and advanced features */
static int efa_rdm_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
			      uint64_t flags, struct fid_mr **mr_fid)
{
	struct efa_domain *domain;
	struct efa_rdm_mr *efa_rdm_mr;
	int ret;
	struct fi_mr_attr mr_attr = {0};

	ret = efa_mr_regattr_validate(fid, attr, flags);
	if (ret)
		return ret;

	domain = container_of(fid, struct efa_domain, util_domain.domain_fid.fid);

	efa_rdm_mr = calloc(1, sizeof(*efa_rdm_mr));
	if (!efa_rdm_mr) {
		EFA_WARN(FI_LOG_MR, "Unable to initialize MR\n");
		return -FI_ENOMEM;
	}

	efa_rdm_mr->efa_mr.domain = domain;
	efa_rdm_mr->efa_mr.mr_fid.fid.fclass = FI_CLASS_MR;
	efa_rdm_mr->efa_mr.mr_fid.fid.context = attr->context;
	efa_rdm_mr->efa_mr.mr_fid.fid.ops = &efa_rdm_mr_ops;

	/* Update attr for ABI compatibility */
	ofi_mr_update_attr(domain->util_domain.fabric->fabric_fid.api_version,
			    domain->util_domain.info_domain_caps,
			    attr, &mr_attr, flags);

	ret = efa_rdm_mr_reg_impl(efa_rdm_mr, flags, &mr_attr);
	if (ret) {
		EFA_WARN(FI_LOG_MR, "Unable to register MR: %s\n",
			 fi_strerror(-ret));
		free(efa_rdm_mr);
		return ret;
	}

	/* RDM-specific: SHM MR registration */
	if (domain->shm_domain) {
		uint64_t shm_flags = efa_rdm_mr->flags;
		struct fi_mr_attr shm_attr = mr_attr;

		if (mr_attr.iface != FI_HMEM_SYSTEM)
			shm_flags |= FI_HMEM_DEVICE_ONLY;

		shm_attr.hmem_data = efa_rdm_mr->hmem_data;
		ret = fi_mr_regattr(efa_rdm_mr->efa_mr.domain->shm_domain,
				    &shm_attr, shm_flags,
				    &efa_rdm_mr->shm_mr);
		if (ret) {
			EFA_WARN(FI_LOG_MR,
				 "Unable to register shm MR. errno: %d "
				 "err_msg: (%s) key: %ld buf: %p len: %zu "
				 "flags %ld\n",
				 ret, fi_strerror(-ret), efa_rdm_mr->efa_mr.mr_fid.key,
				 mr_attr.mr_iov ? mr_attr.mr_iov->iov_base :
						  NULL,
				 mr_attr.mr_iov ? mr_attr.mr_iov->iov_len : 0,
				 flags);
			efa_rdm_mr_dereg_impl(efa_rdm_mr);
			free(efa_rdm_mr);
			return ret;
		}
	}

	*mr_fid = &efa_rdm_mr->efa_mr.mr_fid;
	return 0;
}

static int efa_rdm_mr_regv(struct fid *fid, const struct iovec *iov,
			   size_t count, uint64_t access, uint64_t offset,
			   uint64_t requested_key, uint64_t flags,
			   struct fid_mr **mr_fid, void *context)
{
	struct fi_mr_attr attr = EFA_MR_ATTR_INIT_SYSTEM(iov, count, access,
							  offset,
							  requested_key,
							  context);

	return efa_rdm_mr_regattr(fid, &attr, flags, mr_fid);
}

static int efa_rdm_mr_reg(struct fid *fid, const void *buf, size_t len,
			  uint64_t access, uint64_t offset,
			  uint64_t requested_key, uint64_t flags,
			  struct fid_mr **mr_fid, void *context)
{
	struct iovec iov = {.iov_base = (void *)buf, .iov_len = len};
	struct fi_mr_attr attr = EFA_MR_ATTR_INIT_SYSTEM(&iov, 1, access,
							  offset,
							  requested_key,
							  context);

	return efa_rdm_mr_regattr(fid, &attr, flags, mr_fid);
}

struct fi_ops_mr efa_rdm_domain_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = efa_rdm_mr_reg,
	.regv = efa_rdm_mr_regv,
	.regattr = efa_rdm_mr_regattr,
};