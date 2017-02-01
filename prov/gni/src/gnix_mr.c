/*
 * Copyright (c) 2015-2016 Cray Inc. All rights reserved.
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
 *
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdlib.h>
#include <string.h>

#include "gnix.h"
#include "gnix_nic.h"
#include "gnix_util.h"
#include "gnix_mr.h"

/* forward declarations */

static int fi_gnix_mr_close(fid_t fid);

/* global declarations */
/* memory registration operations */
static struct fi_ops fi_gnix_mr_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_gnix_mr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

/**
 * Sign extends the value passed into up to length parameter
 *
 * @param[in]  val  value to be sign extended
 * @param[in]  len  length to sign extend the value
 * @return          sign extended value to length, len
 */
static inline int64_t __sign_extend(
		uint64_t val,
		int len)
{
	int64_t m = 1UL << (len - 1);
	int64_t r = (val ^ m) - m;

	return r;
}

/**
 * Converts a key to a gni memory handle without calculating crc
 *
 * @param key   gnix memory registration key
 * @param mhdl  gni memory handle
 */
void _gnix_convert_key_to_mhdl_no_crc(
		gnix_mr_key_t *key,
		gni_mem_handle_t *mhdl)
{
	uint64_t va = key->pfn;
	uint8_t flags = 0;

	va = (uint64_t) __sign_extend(va << GNIX_MR_PAGE_SHIFT,
				      GNIX_MR_VA_BITS);

	if (key->flags & GNIX_MR_FLAG_READONLY)
		flags |= GNI_MEMHNDL_ATTR_READONLY;

	GNI_MEMHNDL_INIT((*mhdl));
	GNI_MEMHNDL_SET_VA((*mhdl), va);
	GNI_MEMHNDL_SET_MDH((*mhdl), key->mdd);
	GNI_MEMHNDL_SET_NPAGES((*mhdl), GNI_MEMHNDL_NPGS_MASK);
	GNI_MEMHNDL_SET_FLAGS((*mhdl), flags);
	GNI_MEMHNDL_SET_PAGESIZE((*mhdl), GNIX_MR_PAGE_SHIFT);
}

/**
 * Converts a key to a gni memory handle
 *
 * @param key   gnix memory registration key
 * @param mhdl  gni memory handle
 */
void _gnix_convert_key_to_mhdl(
		gnix_mr_key_t *key,
		gni_mem_handle_t *mhdl)
{
	_gnix_convert_key_to_mhdl_no_crc(key, mhdl);
	compiler_barrier();
	GNI_MEMHNDL_SET_CRC((*mhdl));
}

/**
 * Converts a gni memory handle to gnix memory registration key
 *
 * @param mhdl  gni memory handle
 * @return uint64_t representation of a gnix memory registration key
 */
uint64_t _gnix_convert_mhdl_to_key(gni_mem_handle_t *mhdl)
{
	gnix_mr_key_t key = {{{{0}}}};
	key.pfn = GNI_MEMHNDL_GET_VA((*mhdl)) >> GNIX_MR_PAGE_SHIFT;
	key.mdd = GNI_MEMHNDL_GET_MDH((*mhdl));
	//key->format = GNI_MEMHNDL_NEW_FRMT((*mhdl));
	key.flags = 0;

	if (GNI_MEMHNDL_GET_FLAGS((*mhdl)) & GNI_MEMHNDL_FLAG_READONLY)
		key.flags |= GNIX_MR_FLAG_READONLY;

	return key.value;
}

/**
 * Helper function to calculate the length of a potential registration
 * based on some rules of the registration cache.
 *
 * Registrations should be page aligned and contain all of page(s)
 *
 * @param address   base address of the registration
 * @param length    length of the registration
 * @param pagesize  assumed page size of the registration
 * @return length for the new registration
 */
static inline uint64_t __calculate_length(
		uint64_t address,
		uint64_t length,
		uint64_t pagesize)
{
	uint64_t baseaddr = address & ~(pagesize - 1);
	uint64_t reg_len = (address + length) - baseaddr;
	uint64_t pages = reg_len / pagesize;

	if (reg_len % pagesize != 0)
		pages += 1;

	return pages * pagesize;
}

static int __mr_reg(struct fid *fid, const void *buf, size_t len,
			  uint64_t access, uint64_t offset,
			  uint64_t requested_key, uint64_t flags,
			  struct fid_mr **mr_o, void *context)
{
	struct gnix_fid_mem_desc *mr = NULL;
	struct gnix_fid_domain *domain;
	int rc;
	uint64_t reg_addr, reg_len;
	struct _gnix_fi_reg_context fi_reg_context = {
			.access = access,
			.offset = offset,
			.requested_key = requested_key,
			.flags = flags,
			.context = context,
	};

	GNIX_TRACE(FI_LOG_MR, "\n");

	/* Flags are reserved for future use and must be 0. */
	if (unlikely(flags))
		return -FI_EBADFLAGS;

	/* The offset parameter is reserved for future use and must be 0.
	 *   Additionally, check for invalid pointers, bad access flags and the
	 *   correct fclass on associated fid
	 */
	if (offset || !buf || !mr_o || !access ||
			(access & ~(FI_READ | FI_WRITE | FI_RECV | FI_SEND |
						FI_REMOTE_READ |
						FI_REMOTE_WRITE)) ||
			(fid->fclass != FI_CLASS_DOMAIN))

		return -FI_EINVAL;

	domain = container_of(fid, struct gnix_fid_domain, domain_fid.fid);

	reg_addr = ((uint64_t) buf) & ~((1 << GNIX_MR_PAGE_SHIFT) - 1);
	reg_len = __calculate_length((uint64_t) buf, len,
			1 << GNIX_MR_PAGE_SHIFT);

	/* call cache register op to retrieve the right entry */
	fastlock_acquire(&domain->mr_cache_lock);
	if (unlikely(!domain->mr_ops))
		_gnix_open_cache(domain, GNIX_DEFAULT_CACHE_TYPE);

	if (unlikely(!domain->mr_ops->is_init(domain))) {
		rc = domain->mr_ops->init(domain);
		if (rc != FI_SUCCESS) {
			fastlock_release(&domain->mr_cache_lock);
			goto err;
		}
	}

	rc = domain->mr_ops->reg_mr(domain,
			(uint64_t) reg_addr, reg_len, &fi_reg_context,
			(void **) &mr);
	fastlock_release(&domain->mr_cache_lock);

	/* check retcode */
	if (unlikely(rc != FI_SUCCESS))
		goto err;

	/* md.mr_fid */
	mr->mr_fid.mem_desc = mr;
	mr->mr_fid.fid.fclass = FI_CLASS_MR;
	mr->mr_fid.fid.context = context;
	mr->mr_fid.fid.ops = &fi_gnix_mr_ops;

	/* setup internal key structure */
	mr->mr_fid.key = _gnix_convert_mhdl_to_key(&mr->mem_hndl);

	_gnix_ref_get(mr->domain);

	/* set up mr_o out pointer */
	*mr_o = &mr->mr_fid;
	return FI_SUCCESS;

err:
	return rc;
}

DIRECT_FN int gnix_mr_reg(struct fid *fid, const void *buf, size_t len,
	uint64_t access, uint64_t offset,
	uint64_t requested_key, uint64_t flags,
	struct fid_mr **mr, void *context)
{
	const struct iovec mr_iov = {
		.iov_base = (void *) buf,
		.iov_len = len,
	};
	const struct fi_mr_attr attr = {
		.mr_iov = &mr_iov,
		.iov_count = 1,
		.access = access, 
		.offset = offset,
		.requested_key = requested_key, 
		.context = context,
	};

	return gnix_mr_regattr(fid, &attr, flags, mr);
}

DIRECT_FN int gnix_mr_regv(struct fid *fid, const struct iovec *iov,
	size_t count, uint64_t access,	
	uint64_t offset, uint64_t requested_key,
	uint64_t flags, struct fid_mr **mr, void *context)
{
	const struct fi_mr_attr attr = {
		.mr_iov = iov, 
		.iov_count = count,
		.access = access,
		.offset = offset,
		.requested_key = requested_key, 
		.context = context,
	};

	return gnix_mr_regattr(fid, &attr, flags, mr);
}


DIRECT_FN int gnix_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
	uint64_t flags, struct fid_mr **mr)
{
	struct gnix_fid_domain *domain = container_of(fid, 
		struct gnix_fid_domain, domain_fid.fid);

	if (!attr)
		return -FI_EINVAL;
	if (!attr->mr_iov || !attr->iov_count)
		return -FI_EINVAL;

	if (domain->mr_iov_limit < attr->iov_count)
		return -FI_EOPNOTSUPP;

	if (attr->iov_count == 1)
		return __mr_reg(fid, attr->mr_iov[0].iov_base, 
			attr->mr_iov[0].iov_len, attr->access, attr->offset,
			attr->requested_key, flags, mr, attr->context);
	
	/* regv limited to one iov at this time */
	return -FI_EOPNOTSUPP;
}

/**
 * Closes and deallocates a libfabric memory registration in the internal cache
 *
 * @param[in]  fid  libfabric memory registration fid
 *
 * @return     FI_SUCCESS on success
 *             -FI_EINVAL on invalid fid
 *             -FI_NOENT when there isn't a matching registration for the
 *               provided fid
 *             Otherwise, GNI_RC_* ret codes converted to FI_* err codes
 */
static int fi_gnix_mr_close(fid_t fid)
{
	struct gnix_fid_mem_desc *mr;
	gni_return_t ret;
	struct gnix_fid_domain *domain;

	GNIX_TRACE(FI_LOG_MR, "\n");

	if (unlikely(fid->fclass != FI_CLASS_MR))
		return -FI_EINVAL;

	mr = container_of(fid, struct gnix_fid_mem_desc, mr_fid.fid);

	domain = mr->domain;

	/* call cache deregister op */
	fastlock_acquire(&domain->mr_cache_lock);
	ret = domain->mr_ops->dereg_mr(domain, mr);
	fastlock_release(&domain->mr_cache_lock);

	/* check retcode */
	if (likely(ret == FI_SUCCESS)) {
		/* release references to the domain and nic */
		_gnix_ref_put(domain);
	} else {
		GNIX_INFO(FI_LOG_MR, "failed to deregister memory, "
			  "ret=%i\n", ret);
	}

	return ret;
}

static inline void *__gnix_generic_register(
		struct gnix_fid_domain *domain,
		struct gnix_fid_mem_desc *md,
		void *address,
		size_t length,
		gni_cq_handle_t dst_cq_hndl,
		int flags,
		int vmdh_index)
{
	struct gnix_nic *nic;
	gni_return_t grc = GNI_RC_SUCCESS;
	int rc;

	pthread_mutex_lock(&gnix_nic_list_lock);

	/* If the nic list is empty, create a nic */
	if (unlikely((dlist_empty(&gnix_nic_list_ptag[domain->ptag])))) {
		/* release the lock because we are not checking the list after
			this point. Additionally, gnix_nic_alloc takes the 
			lock to add the nic. */
		pthread_mutex_unlock(&gnix_nic_list_lock);

		rc = gnix_nic_alloc(domain, NULL, &nic);
		if (rc) {
			GNIX_INFO(FI_LOG_MR,
				  "could not allocate nic to do mr_reg,"
				  " ret=%i\n", rc);
			return NULL;
		}
	} else {
		nic = dlist_first_entry(&gnix_nic_list_ptag[domain->ptag], 
			struct gnix_nic, ptag_nic_list);
		if (unlikely(nic == NULL)) {
			GNIX_ERR(FI_LOG_MR, "Failed to find nic on "
				"ptag list\n");
			pthread_mutex_unlock(&gnix_nic_list_lock);
			return NULL;
		}
		_gnix_ref_get(nic);	
		pthread_mutex_unlock(&gnix_nic_list_lock);
        }

	COND_ACQUIRE(nic->requires_lock, &nic->lock);
	grc = GNI_MemRegister(nic->gni_nic_hndl, (uint64_t) address,
				  length,	dst_cq_hndl, flags,
				  vmdh_index, &md->mem_hndl);
	COND_RELEASE(nic->requires_lock, &nic->lock);

	if (unlikely(grc != GNI_RC_SUCCESS)) {
		GNIX_INFO(FI_LOG_MR, "failed to register memory with uGNI, "
			  "ret=%s\n", gni_err_str[grc]);
		_gnix_ref_put(nic);

		return NULL;
	}

	/* set up the mem desc */
	md->nic = nic;
	md->domain = domain;

	/* take references on domain */
	_gnix_ref_get(md->domain);

	return md;
}

static void *__gnix_register_region(
		void *handle,
		void *address,
		size_t length,
		struct _gnix_fi_reg_context *fi_reg_context,
		void *context)
{
	struct gnix_fid_mem_desc *md = (struct gnix_fid_mem_desc *) handle;
	struct gnix_fid_domain *domain = context;
	gni_cq_handle_t dst_cq_hndl = NULL;
	int flags = 0;
	int vmdh_index = -1;

	/* If network would be able to write to this buffer, use read-write */
	if (fi_reg_context->access & (FI_RECV | FI_READ | FI_REMOTE_WRITE))
		flags |= GNI_MEM_READWRITE;
	else
		flags |= GNI_MEM_READ_ONLY;

	return __gnix_generic_register(domain, md, address, length, dst_cq_hndl,
			flags, vmdh_index);
}

static int __gnix_deregister_region(
		void *handle,
		void *context)
{
	struct gnix_fid_mem_desc *mr = (struct gnix_fid_mem_desc *) handle;
	gni_return_t ret;
	struct gnix_fid_domain *domain;
	struct gnix_nic *nic;

	domain = mr->domain;
	nic = mr->nic;

	COND_ACQUIRE(nic->requires_lock, &nic->lock);
	ret = GNI_MemDeregister(nic->gni_nic_hndl, &mr->mem_hndl);
	COND_RELEASE(nic->requires_lock, &nic->lock);
	if (ret == GNI_RC_SUCCESS) {
		/* release reference to domain */
		_gnix_ref_put(domain);

		/* release reference to nic */
		_gnix_ref_put(nic);
	} else {
		GNIX_INFO(FI_LOG_MR, "failed to deregister memory"
			  " region, entry=%p ret=%i\n", handle, ret);
	}

	return ret;
}

/**
 * Associates a registered memory region with a completion counter.
 *
 * @param[in] fid		the memory region
 * @param[in] bfid		the fabric identifier for the memory region
 * @param[in] flags		flags to apply to the registration
 *
 * @return FI_SUCCESS		Upon successfully registering the memory region
 * @return -FI_ENOSYS		If binding of the memory region is not supported
 * @return -FI_EBADFLAGS	If the flags are not supported
 * @return -FI_EKEYREJECTED	If the key is not available
 * @return -FI_ENOKEY		If the key is already in use
 */
DIRECT_FN int gnix_mr_bind(fid_t fid, struct fid *bfid, uint64_t flags)
{
	return -FI_ENOSYS;
}

static int __gnix_destruct_registration(void *context)
{
	return GNI_RC_SUCCESS;
}


#ifdef HAVE_UDREG
void *__udreg_register(void *addr, uint64_t length, void *context)
{
	struct gnix_fid_mem_desc *md;
	struct gnix_fid_domain *domain;

	domain = (struct gnix_fid_domain *) context;

    /* Allocate an udreg info block for this registration. */
    md = calloc(1, sizeof(*md));
    if (!md) {
	GNIX_WARN(FI_LOG_MR, "failed to allocate memory for registration\n");
	return NULL;
    }

    return __gnix_generic_register(domain, md, addr, length, NULL,
		GNI_MEM_READWRITE, -1);
}


uint32_t __udreg_deregister(void *registration, void *context)
{
	gni_return_t grc;

	grc = __gnix_deregister_region(registration, NULL);

	free(registration);

	return (grc == GNI_RC_SUCCESS) ? 0 : 1;
}


/* Called via dreg when a cache is destroyed. */
void __udreg_cache_destructor(void *context)
{
    /*  Nothing needed here. */
}

static int __udreg_init(struct gnix_fid_domain *domain)
{
	udreg_return_t urc;

	udreg_cache_attr_t udreg_cache_attr = {
		.cache_name =           {"gnix_app_cache"},
		.max_entries =          domain->udreg_reg_limit,
		.modes =                UDREG_CC_MODE_USE_LARGE_PAGES,
		.debug_mode =           0,
		.debug_rank =           0,
		.reg_context =          (void *) domain,
		.dreg_context =         (void *) domain,
		.destructor_context =   (void *) domain,
		.device_reg_func =      __udreg_register,
		.device_dereg_func =    __udreg_deregister,
		.destructor_callback =  __udreg_cache_destructor,
	};

	if (domain->mr_cache_attr.lazy_deregistration)
		udreg_cache_attr.modes |= UDREG_CC_MODE_USE_LAZY_DEREG;

	/*
	 * Create a udreg cache for application memory registrations.
	 */
	urc = UDREG_CacheCreate(&udreg_cache_attr);
	if (urc != UDREG_RC_SUCCESS) {
		GNIX_FATAL(FI_LOG_MR,
				"Could not initialize udreg application cache, urc=%d\n",
				urc);
	}

	urc = UDREG_CacheAccess(udreg_cache_attr.cache_name, &domain->udreg_cache);
	if (urc != UDREG_RC_SUCCESS) {
		GNIX_FATAL(FI_LOG_MR,
				"Could not access udreg application cache, urc=%d",
				urc);
	}

	domain->mr_is_init = 1;

	return FI_SUCCESS;
}

static int __udreg_is_init(struct gnix_fid_domain *domain)
{
	return domain->udreg_cache != NULL;
}

static int __udreg_reg_mr(
		struct gnix_fid_domain     *domain,
		uint64_t                    address,
		uint64_t                    length,
		struct _gnix_fi_reg_context *fi_reg_context,
		void                        **handle) {

	udreg_return_t urc;
	udreg_entry_t *udreg_entry;
	struct gnix_fid_mem_desc *md;

	urc = UDREG_Register(domain->udreg_cache, (void *) address, length, &udreg_entry);
	if (unlikely(urc != UDREG_RC_SUCCESS))
		return -FI_EIO;

	md = udreg_entry->device_data;
	md->entry = udreg_entry;

	*handle = md;

	return FI_SUCCESS;
}

static int __udreg_dereg_mr(struct gnix_fid_domain *domain,
		struct gnix_fid_mem_desc *md)
{
	udreg_return_t urc;

	urc = UDREG_Unregister(domain->udreg_cache,
			(udreg_entry_t *) md->entry);
	if (urc != UDREG_RC_SUCCESS) {
		GNIX_WARN(FI_LOG_MR, "UDREG_Unregister() returned %d\n", urc);
		return -FI_ENOENT;
	}

	return urc;
}

static int __udreg_close(struct gnix_fid_domain *domain)
{
	udreg_return_t ret;

	if (domain->udreg_cache) {
		ret = UDREG_CacheRelease(domain->udreg_cache);
		if (unlikely(ret != UDREG_RC_SUCCESS))
			GNIX_FATAL(FI_LOG_DOMAIN, "failed to release from "
					"mr cache during domain destruct, dom=%p rc=%d\n",
					domain, ret);

		ret = UDREG_CacheDestroy(domain->udreg_cache);
		if (unlikely(ret != UDREG_RC_SUCCESS))
			GNIX_FATAL(FI_LOG_DOMAIN, "failed to destroy mr "
					"cache during domain destruct, dom=%p rc=%d\n",
					domain, ret);
	}

	return FI_SUCCESS;
}
#else
static int __udreg_init(struct gnix_fid_domain *domain)
{
	return -FI_ENOSYS;
}

static int __udreg_is_init(struct gnix_fid_domain *domain)
{
	return FI_SUCCESS;
}

static int __udreg_reg_mr(struct gnix_fid_domain *domain,
		uint64_t                    address,
		uint64_t                    length,
		struct _gnix_fi_reg_context *fi_reg_context,
		void                        **handle) {

	return -FI_ENOSYS;
}

static int __udreg_dereg_mr(struct gnix_fid_domain *domain,
		struct gnix_fid_mem_desc *md)
{
	return -FI_ENOSYS;
}

static int __udreg_close(struct gnix_fid_domain *domain)
{
	return FI_SUCCESS;
}
#endif

struct gnix_mr_ops udreg_mr_ops = {
	.init = __udreg_init,
	.is_init = __udreg_is_init,
	.reg_mr = __udreg_reg_mr,
	.dereg_mr = __udreg_dereg_mr,
	.destroy_cache = __udreg_close,
	.flush_cache = NULL, // UDREG doesn't support cache flush
};

static int __cache_init(struct gnix_fid_domain *domain) {
	int ret;

	ret = _gnix_mr_cache_init(&domain->mr_cache,
			&domain->mr_cache_attr);
	if (ret == FI_SUCCESS)
		domain->mr_is_init = 1;

	return ret;
}

static int __cache_is_init(struct gnix_fid_domain *domain) {
	return domain->mr_cache != NULL;
}

static int __cache_reg_mr(
		struct gnix_fid_domain      *domain,
		uint64_t                    address,
		uint64_t                    length,
		struct _gnix_fi_reg_context *fi_reg_context,
		void                        **handle) {

	return _gnix_mr_cache_register(domain->mr_cache, address, length,
			fi_reg_context, handle);
}

static int __cache_dereg_mr(struct gnix_fid_domain *domain,
		struct gnix_fid_mem_desc *md)
{
	return _gnix_mr_cache_deregister(domain->mr_cache, md);
}

static int __cache_close(struct gnix_fid_domain *domain)
{
	int ret;

	if (domain->mr_cache) {
		ret = _gnix_mr_cache_destroy(domain->mr_cache);
		if (ret != FI_SUCCESS)
			GNIX_FATAL(FI_LOG_DOMAIN, "failed to destroy mr cache "
					"during domain destruct, dom=%p ret=%d\n",
					domain, ret);
	}

	return FI_SUCCESS;
}

static int __cache_flush(struct gnix_fid_domain *domain)
{
	int ret;

	fastlock_acquire(&domain->mr_cache_lock);
	ret = _gnix_mr_cache_flush(domain->mr_cache);
	fastlock_release(&domain->mr_cache_lock);

	return ret;
}

struct gnix_mr_ops cache_mr_ops = {
	.init = __cache_init,
	.is_init = __cache_is_init,
	.reg_mr = __cache_reg_mr,
	.dereg_mr = __cache_dereg_mr,
	.destroy_cache = __cache_close,
	.flush_cache = __cache_flush,
};


static int __basic_mr_init(struct gnix_fid_domain *domain) {
	domain->mr_is_init = 1;
	return FI_SUCCESS;
}

static int __basic_mr_is_init(struct gnix_fid_domain *domain) {
	return domain->mr_is_init;
}

static int __basic_mr_reg_mr(
		struct gnix_fid_domain      *domain,
		uint64_t                    address,
		uint64_t                    length,
		struct _gnix_fi_reg_context *fi_reg_context,
		void                        **handle) {

	struct gnix_fid_mem_desc *md, *ret;

	md = calloc(1, sizeof(*md));
	if (!md) {
		GNIX_WARN(FI_LOG_MR, "failed to allocate memory");
		return -FI_ENOMEM;
	}
	ret = __gnix_register_region((void *) md, (void *) address, length,
			fi_reg_context, (void *) domain);
	if (!ret) {
		GNIX_WARN(FI_LOG_MR, "failed to register memory");
		free(md);
		return -FI_ENOSPC;
	}

	*handle = (void *) md;

	return FI_SUCCESS;
}

static int __basic_mr_dereg_mr(struct gnix_fid_domain *domain,
		struct gnix_fid_mem_desc *md)
{
	int ret; 
	
	ret = __gnix_deregister_region((void *) md, NULL);
	if (ret == FI_SUCCESS)
		free((void *) md);

	return ret;
}

struct gnix_mr_ops basic_mr_ops = {
	.init = __basic_mr_init,
	.is_init = __basic_mr_is_init,
	.reg_mr = __basic_mr_reg_mr,
	.dereg_mr = __basic_mr_dereg_mr,
	.flush_cache = NULL, // unsupported since there is no caching here
};


int _gnix_open_cache(struct gnix_fid_domain *domain, int type)
{
	if (type < 0 || type >= GNIX_MR_MAX_TYPE)
		return -FI_EINVAL;

	if (domain->mr_ops && domain->mr_ops->is_init(domain))
		return -FI_EBUSY;

	switch(type) {
	case GNIX_MR_TYPE_UDREG:
		domain->mr_ops = &udreg_mr_ops;
		break;
	case GNIX_MR_TYPE_NONE:
		domain->mr_ops = &basic_mr_ops;
		break;
	default:
		domain->mr_ops = &cache_mr_ops;
		break;
	}

	domain->mr_cache_type = type;
	return FI_SUCCESS;
}


int _gnix_flush_registration_cache(struct gnix_fid_domain *domain)
{
	if (domain->mr_ops && domain->mr_ops->flush_cache)
		return domain->mr_ops->flush_cache(domain);

	return FI_SUCCESS;  // if no flush was present, silently pass
}

int _gnix_close_cache(struct gnix_fid_domain *domain)
{
	/* if the domain isn't being destructed by close, we need to check the
	 * cache again. This isn't a likely case. Destroy must succeed since we
	 * are in the destruct path */
	if (domain->mr_ops && domain->mr_ops->destroy_cache)
		return domain->mr_ops->destroy_cache(domain);

	return FI_SUCCESS;
}

gnix_mr_cache_attr_t _gnix_default_mr_cache_attr = {
		.soft_reg_limit      = 4096,
		.hard_reg_limit      = -1,
		.hard_stale_limit    = 128,
		.lazy_deregistration = 1,
		.reg_callback        = __gnix_register_region,
		.dereg_callback      = __gnix_deregister_region,
		.destruct_callback   = __gnix_destruct_registration,
		.elem_size           = sizeof(struct gnix_fid_mem_desc),
};
