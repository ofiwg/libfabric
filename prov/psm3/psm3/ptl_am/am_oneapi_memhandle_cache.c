/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2022 Intel Corporation.

  This program is free software; you can redistribute it and/or modify
  it under the terms of version 2 of the GNU General Public License as
  published by the Free Software Foundation.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  Contact Information:
  Intel Corporation, www.intel.com

  BSD LICENSE

  Copyright(c) 2022 Intel Corporation.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifdef PSM_ONEAPI

#include "psm_user.h"
#include "psm_am_internal.h"
#include "am_oneapi_memhandle_cache.h"
#include <fcntl.h>
#include <unistd.h>
#ifdef HAVE_DRM
#include <sys/ioctl.h>
#include <drm/i915_drm.h>
#endif
#ifdef HAVE_LIBDRM
#include <sys/ioctl.h>
#include <libdrm/i915_drm.h>
#endif

#if defined(HAVE_DRM) || defined(HAVE_LIBDRM)
/*
 * rbtree cruft
 */
struct _cl_map_item;

typedef struct
{
	unsigned long           start;           /* start virtual address */
	ze_ipc_mem_handle_t     ze_ipc_handle;   /* ze ipc mem handle */
	void                    *ze_ipc_dev_ptr; /* ze device pointer */
	uint16_t                length;          /* length*/
	psm2_epid_t             epid;
	struct _cl_map_item*    i_prev;          /* idle queue previous */
	struct _cl_map_item*    i_next;          /* idle queue next */
}__attribute__ ((aligned (128))) rbtree_ze_memhandle_cache_mapitem_pl_t;

typedef struct {
	uint32_t                nelems;          /* number of elements in the cache */
} rbtree_ze_memhandle_cache_map_pl_t;

static psm2_error_t am_ze_memhandle_mpool_init(uint32_t memcache_size);

/*
 * Custom comparator
 */
typedef rbtree_ze_memhandle_cache_mapitem_pl_t ze_cache_item;

static int ze_cache_key_cmp(const ze_cache_item *a, const ze_cache_item *b)
{
	// When multi-ep is disabled, cache can assume
	//   1 epid == 1 remote process == 1 ONEAPI address space
	// But when multi-ep is enabled, one process can have many epids, so in this case
	// cannot use epid as part of cache key.
	if (!psm3_multi_ep_enabled) {
		switch (psm3_epid_cmp_internal(a->epid, b->epid)) {
		case -1: return -1;
		case 1: return 1;
		default:
			break;
		}
	}

	unsigned long a_end, b_end;
	// normalize into inclusive upper bounds to handle
	// 0-length entries
	a_end = (a->start + a->length);
	b_end = (b->start + b->length);
	if (a->length > 0)
		a_end--;

	if (b->length > 0)
		b_end--;

	if (a_end < b->start)
		return -1;
	if (b_end < a->start)
		return 1;

	return 0;
}


/*
 * Necessary rbtree cruft
 */
#define RBTREE_MI_PL  rbtree_ze_memhandle_cache_mapitem_pl_t
#define RBTREE_MAP_PL rbtree_ze_memhandle_cache_map_pl_t
#define RBTREE_CMP(a,b) ze_cache_key_cmp((a), (b))
#define RBTREE_ASSERT                     psmi_assert
#define RBTREE_MAP_COUNT(PAYLOAD_PTR)     ((PAYLOAD_PTR)->nelems)
#define RBTREE_NO_EMIT_IPS_CL_QMAP_PREDECESSOR

#include "psm3_rbtree.h"
#include "psm3_rbtree.c"

/*
 * Convenience rbtree cruft
 */
#define NELEMS			ze_memhandle_cachemap.payload.nelems

#define IHEAD			ze_memhandle_cachemap.root
#define LAST			IHEAD->payload.i_prev
#define FIRST			IHEAD->payload.i_next
#define INEXT(x)		x->payload.i_next
#define IPREV(x)		x->payload.i_prev

/*
 * Actual module data
 */
static cl_qmap_t ze_memhandle_cachemap; /* Global cache */
static uint8_t ze_memhandle_cache_enabled;
static mpool_t ze_memhandle_mpool;
static uint32_t ze_memhandle_cache_size;

static uint64_t cache_hit_counter;
static uint64_t cache_miss_counter;
static uint64_t cache_evict_counter;
static uint64_t cache_collide_counter;
static uint64_t cache_clear_counter;

static void print_ze_memhandle_cache_stats(void)
{
	_HFI_DBG("enabled=%u,size=%u,hit=%lu,miss=%lu,evict=%lu,collide=%lu,clear=%lu\n",
		ze_memhandle_cache_enabled, ze_memhandle_cache_size,
		cache_hit_counter, cache_miss_counter,
		cache_evict_counter, cache_collide_counter, cache_clear_counter);
}

/*
 * This is the callback function when mempool are resized or destroyed.
 * Upon calling cache fini mpool is detroyed which in turn calls this callback
 * which helps in closing all memhandles.
 */
static void
psmi_ze_memhandle_cache_alloc_func(int is_alloc, void* context, void* obj)
{
	cl_map_item_t* memcache_item = (cl_map_item_t*)obj;
	if (!is_alloc) {
		if(memcache_item->payload.start)
			PSMI_ONEAPI_ZE_CALL(zeMemCloseIpcHandle, ze_context,
				       memcache_item->payload.ze_ipc_dev_ptr);
	}
}

/*
 * Creating mempool for ze memhandle cache nodes.
 */
static psm2_error_t
am_ze_memhandle_mpool_init(uint32_t memcache_size)
{
	psm2_error_t err;
	if (memcache_size < 1)
		return PSM2_PARAM_ERR;

	ze_memhandle_cache_size = memcache_size;
	/* Creating a memory pool of size PSM3_ONEAPI_MEMCACHE_SIZE
	 * which includes the Root and NIL items
	 */
	ze_memhandle_mpool = psm3_mpool_create_for_gpu(sizeof(cl_map_item_t),
					ze_memhandle_cache_size,
					ze_memhandle_cache_size, 0,
					UNDEFINED, NULL, NULL,
					psmi_ze_memhandle_cache_alloc_func,
					NULL);
	if (ze_memhandle_mpool == NULL) {
		err = psm3_handle_error(PSMI_EP_NORETURN, PSM2_NO_MEMORY,
				"Couldn't allocate ONEAPI host receive buffer pool");
		return err;
	}
	return PSM2_OK;
}

/*
 * Initialize rbtree.
 */
psm2_error_t am_ze_memhandle_cache_init(uint32_t memcache_size)
{
	psm2_error_t err = am_ze_memhandle_mpool_init(memcache_size);
	if (err != PSM2_OK)
		return err;

	cl_map_item_t *root, *nil_item;
	root = (cl_map_item_t *)psmi_calloc(NULL, UNDEFINED, 1, sizeof(cl_map_item_t));
	if (root == NULL)
		return PSM2_NO_MEMORY;
	nil_item = (cl_map_item_t *)psmi_calloc(NULL, UNDEFINED, 1, sizeof(cl_map_item_t));
	if (nil_item == NULL) {
		psmi_free(root);
		return PSM2_NO_MEMORY;
	}

	nil_item->payload.start = 0;
	nil_item->payload.epid = psm3_epid_zeroed_internal();
	nil_item->payload.length = 0;
	ze_memhandle_cache_enabled = 1;
	ips_cl_qmap_init(&ze_memhandle_cachemap,root,nil_item);
	NELEMS = 0;

	cache_hit_counter = 0;
	cache_miss_counter = 0;
	cache_evict_counter = 0;
	cache_collide_counter = 0;
	cache_clear_counter = 0;

	return PSM2_OK;
}

void am_ze_memhandle_cache_map_fini()
{
	print_ze_memhandle_cache_stats();

	if (ze_memhandle_cachemap.nil_item) {
		psmi_free(ze_memhandle_cachemap.nil_item);
		ze_memhandle_cachemap.nil_item = NULL;
	}

	if (ze_memhandle_cachemap.root) {
		psmi_free(ze_memhandle_cachemap.root);
		ze_memhandle_cachemap.root = NULL;
	}

	if (ze_memhandle_cache_enabled) {
		psm3_mpool_destroy(ze_memhandle_mpool);
		ze_memhandle_cache_enabled = 0;
	}

	ze_memhandle_cache_size = 0;
}

/*
 * Insert at the head of Idleq.
 */
static void
am_ze_idleq_insert(cl_map_item_t* memcache_item)
{
	if (FIRST == NULL) {
		FIRST = memcache_item;
		LAST = memcache_item;
		return;
	}
	INEXT(FIRST) = memcache_item;
	IPREV(memcache_item) = FIRST;
	FIRST = memcache_item;
	INEXT(FIRST) = NULL;
	return;
}

/*
 * Remove least recent used element.
 */
static void
am_ze_idleq_remove_last(cl_map_item_t* memcache_item)
{
	if (!INEXT(memcache_item)) {
		LAST = NULL;
		FIRST = NULL;
	} else {
		LAST = INEXT(memcache_item);
		IPREV(LAST) = NULL;
	}
	// Null-out now-removed memcache_item's next and prev pointers out of
	// an abundance of caution
	INEXT(memcache_item) = IPREV(memcache_item) = NULL;
}

static void
am_ze_idleq_remove(cl_map_item_t* memcache_item)
{
	if (LAST == memcache_item) {
		am_ze_idleq_remove_last(memcache_item);
	} else if (FIRST == memcache_item) {
		FIRST = IPREV(memcache_item);
		INEXT(FIRST) = NULL;
	} else {
		INEXT(IPREV(memcache_item)) = INEXT(memcache_item);
		IPREV(INEXT(memcache_item)) = IPREV(memcache_item);
	}
	// Null-out now-removed memcache_item's next and prev pointers out of
	// an abundance of caution
	INEXT(memcache_item) = IPREV(memcache_item) = NULL;
}

static void
am_ze_idleq_reorder(cl_map_item_t* memcache_item)
{
	if (FIRST == memcache_item && LAST == memcache_item ) {
		return;
	}
	am_ze_idleq_remove(memcache_item);
	am_ze_idleq_insert(memcache_item);
	return;
}

/*
 * After a successful cache hit, item is validated by doing a
 * memcmp on the handle stored and the handle we recieve from the
 * sender. If the validation fails the item is removed from the idleq,
 * the rbtree, is put back into the mpool and IpcCloseMemHandle function
 * is called.
 */

static psm2_error_t
am_ze_memhandle_cache_validate(cl_map_item_t* memcache_item,
				 uintptr_t sbuf, ze_ipc_mem_handle_t* handle,
				 uint32_t length, psm2_epid_t epid)
{
	if ((0 == memcmp(handle, &memcache_item->payload.ze_ipc_handle,
			 sizeof(ze_ipc_mem_handle_t)))
			 && sbuf == memcache_item->payload.start
			 && !psm3_epid_cmp_internal(epid, memcache_item->payload.epid)) {
		return PSM2_OK;
	}
	_HFI_DBG("cache collision: new entry start=%lu,length=%u\n", sbuf, length);

	cache_collide_counter++;
	ips_cl_qmap_remove_item(&ze_memhandle_cachemap, memcache_item);
	PSMI_ONEAPI_ZE_CALL(zeMemCloseIpcHandle, ze_context,
		       memcache_item->payload.ze_ipc_dev_ptr);
	am_ze_idleq_remove(memcache_item);
	memset(memcache_item, 0, sizeof(*memcache_item));
	psm3_mpool_put(memcache_item);
	return PSM2_OK_NO_PROGRESS;
}

/*
 * Current eviction policy: Least Recently Used.
 */
static void
am_ze_memhandle_cache_evict(void)
{
	cache_evict_counter++;
	cl_map_item_t *p_item = LAST;
	_HFI_VDBG("Removing (epid=%s,start=%lu,length=%u,dev_ptr=%p,it=%p) from ze_memhandle_cachemap.\n",
			psm3_epid_fmt_internal(p_item->payload.epid, 0), p_item->payload.start, p_item->payload.length,
			p_item->payload.ze_ipc_dev_ptr, p_item);
	ips_cl_qmap_remove_item(&ze_memhandle_cachemap, p_item);
	PSMI_ONEAPI_ZE_CALL(zeMemCloseIpcHandle, ze_context, p_item->payload.ze_ipc_dev_ptr);
	am_ze_idleq_remove_last(p_item);
	memset(p_item, 0, sizeof(*p_item));
	psm3_mpool_put(p_item);
}

static psm2_error_t
am_ze_memhandle_cache_register(uintptr_t sbuf, ze_ipc_mem_handle_t* handle,
				 uint32_t length, psm2_epid_t epid,
				 void *ze_ipc_dev_ptr)
{
	if (NELEMS == ze_memhandle_cache_size)
		am_ze_memhandle_cache_evict();

	cl_map_item_t* memcache_item = psm3_mpool_get(ze_memhandle_mpool);
	/* memcache_item cannot be NULL as we evict
	 * before the call to mpool_get. Check has
	 * been fixed to help with klockwork analysis.
	 */
	if (memcache_item == NULL)
		return PSM2_NO_MEMORY;
	memcache_item->payload.start = sbuf;
	memcpy(&memcache_item->payload.ze_ipc_handle, handle, sizeof(ze_ipc_mem_handle_t));
	memcache_item->payload.ze_ipc_dev_ptr = ze_ipc_dev_ptr;
	memcache_item->payload.length = length;
	memcache_item->payload.epid = epid;
	ips_cl_qmap_insert_item(&ze_memhandle_cachemap, memcache_item);
	am_ze_idleq_insert(memcache_item);
	return PSM2_OK;
}

static inline psm2_error_t am_ze_prepare_fds_for_ipc_open(struct ptl_am *ptl, ze_ipc_mem_handle_t *handle,
                                                          int device_index, int *ipc_fd, psm2_epaddr_t epaddr,
                                                          ze_ipc_mem_handle_t *ze_handle)
{
	am_epaddr_t *am_epaddr = (am_epaddr_t*)epaddr;
	int fd;
	struct drm_prime_handle open_fd = {0, 0, -1};

	if (device_index >= num_ze_devices) {
		_HFI_ERROR("am_ze_memhandle_acquire received invalid device_index from peer: %d\n",
			device_index);
		psm3_handle_error(ptl->ep, PSM2_INTERNAL_ERR,
			"device_index "
			"invalid - received from peer: %d",
			device_index);
		return PSM2_INTERNAL_ERR;
	}
	fd = am_epaddr->peer_fds[device_index];
	cur_ze_dev = &ze_devices[device_index];
	open_fd.flags = DRM_CLOEXEC | DRM_RDWR;
	open_fd.handle = *(int *)handle;
	if (ioctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &open_fd) < 0) {
		_HFI_ERROR("ioctl failed for DRM_IOCTL_PRIME_HANDLE_TO_FD: %s\n", strerror(errno));
		psm3_handle_error(ptl->ep, PSM2_INTERNAL_ERR,
			"ioctl "
			"failed for DRM_IOCTL_PRIME_HANDLE_TO_FD errno=%d",
			errno);
		return PSM2_INTERNAL_ERR;
	}
	memset(ze_handle, 0, sizeof(*ze_handle));
	memcpy(ze_handle, &open_fd.fd, sizeof(open_fd.fd));
	*ipc_fd = open_fd.fd;									\
	return PSM2_OK;
}
#endif /* HAVE_DRM || HAVE_LIBDRM */

/*
 * The key used to search the cache is the senders buf address pointer.
 * Upon a succesful hit in the cache, additional validation is required
 * as multiple senders could potentially send the same buf address value.
 */
ze_device_handle_t*
am_ze_memhandle_acquire(struct ptl_am *ptl, uintptr_t sbuf, ze_ipc_mem_handle_t *handle,
				uint32_t length, psm2_epaddr_t epaddr, int device_index)
{
	void *ze_ipc_dev_ptr = NULL;
	psm2_epid_t epid = epaddr->epid;
#if HAVE_DRM || HAVE_LIBDRM
	ze_ipc_mem_handle_t ze_handle;
	int ipc_fd = -1;
#endif
	_HFI_VDBG("sbuf=%lu,handle=%p,length=%u,epid=%s\n",
			sbuf, handle, length, psm3_epid_fmt_internal(epid, 0));
#if HAVE_DRM || HAVE_LIBDRM

	if (!ze_memhandle_cache_enabled) {
		if (am_ze_prepare_fds_for_ipc_open(ptl, handle, device_index, &ipc_fd, epaddr,
											&ze_handle) == PSM2_OK) {
			PSMI_ONEAPI_ZE_CALL(zeMemOpenIpcHandle, ze_context, cur_ze_dev->dev, ze_handle, 0,
			 		(void **)&ze_ipc_dev_ptr);
			if (ipc_fd >= 0) {
				if (close(ipc_fd) < 0) {
					_HFI_ERROR("close failed for ipc_fd: %s\n", strerror(errno));
					psm3_handle_error(ptl->ep, PSM2_INTERNAL_ERR,
						"close "
						"failed for ipc_fd %d errno=%d",
						ipc_fd, errno);
					return NULL;
				}
			}
		}
		return ze_ipc_dev_ptr;
	}

	ze_cache_item key = {
		.start = (unsigned long) sbuf,
		.length= length,
		.epid = epid
	};

	/*
	 * preconditions:
	 *  1) newrange [start,end) may or may not be in cachemap already
	 *  2) there are no overlapping address ranges in cachemap
	 * postconditions:
	 *  1) newrange is in cachemap
	 *  2) there are no overlapping address ranges in cachemap
	 *
	 * The key used to search the cache is the senders buf address pointer.
	 * Upon a succesful hit in the cache, additional validation is required
	 * as multiple senders could potentially send the same buf address value.
	 */
	cl_map_item_t *p_item = ips_cl_qmap_searchv(&ze_memhandle_cachemap, &key);
	while (p_item->payload.start) {
		// Since a precondition is that there are no overlapping ranges in cachemap,
		// an exact match implies no need to check further
		if (am_ze_memhandle_cache_validate(p_item, sbuf, handle, length, epid) == PSM2_OK) {
			cache_hit_counter++;
			am_ze_idleq_reorder(p_item);
			return p_item->payload.ze_ipc_dev_ptr;
		}

		// newrange is not in the cache and overlaps at least one existing range.
		// am_ze_memhandle_cache_validate() closed and removed existing range.
		// Continue searching for more overlapping ranges
		p_item = ips_cl_qmap_searchv(&ze_memhandle_cachemap, &key);
	}
	cache_miss_counter++;

	if (am_ze_prepare_fds_for_ipc_open(ptl, handle, device_index, &ipc_fd, epaddr,
										&ze_handle) == PSM2_OK) {
		PSMI_ONEAPI_ZE_CALL(zeMemOpenIpcHandle, ze_context, cur_ze_dev->dev, ze_handle, 0,
		 		(void **)&ze_ipc_dev_ptr);
		if (ipc_fd >= 0) {
			if (close(ipc_fd) < 0) {
				_HFI_ERROR("close failed for ipc_fd: %s\n", strerror(errno));
				psm3_handle_error(ptl->ep, PSM2_INTERNAL_ERR,
					"close "
					"failed for ipc_fd %d errno=%d",
					ipc_fd, errno);
				return NULL;
			}
		}
	}

	am_ze_memhandle_cache_register(sbuf, handle,
					   length, epid, ze_ipc_dev_ptr);
	return ze_ipc_dev_ptr;
#else // if no drm, set up to return NULL as oneapi ipc handles don't work without drm
	ze_ipc_dev_ptr = NULL;
	return ze_ipc_dev_ptr;
#endif // HAVE_DRM || HAVE_LIBDRM

}

void
am_ze_memhandle_release(ze_device_handle_t *ze_ipc_dev_ptr)
{
#if HAVE_DRM || HAVE_LIBDRM
	if (!ze_memhandle_cache_enabled)
		PSMI_ONEAPI_ZE_CALL(zeMemCloseIpcHandle, ze_context, ze_ipc_dev_ptr);
#endif // HAVE_DRM || HAVE_LIBDRM
	return;
}

#endif /* PSM_ONEAPI */
