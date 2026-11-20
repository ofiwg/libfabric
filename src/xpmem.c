/*
 * (C) Copyright 2023 UT-Battelle, LLC. All rights reserved.
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

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <rdma/fi_errno.h>
#include <ofi_xpmem.h>
#include <ofi_iov.h>
#include <ofi_mem.h>
#include <ofi_mr.h>
#include <ofi_tree.h>
#include <ofi.h>
#include <stdio.h>

#define XPMEM_DEFAULT_MEMCPY_CHUNK_SIZE 262144
struct ofi_xpmem *xpmem = NULL;

#if HAVE_XPMEM
/* global cache for use with xpmem */
static struct ofi_mr_cache *xpmem_cache;

int ofi_xpmem_init(void)
{
	/* Any attachment that goes past the Linux TASK_SIZE will always
	 * fail. To prevent this we need to determine the value of
	 * TASK_SIZE. On x86_64 the value was hard-coded in sm to be
	 * 0x7ffffffffffful but this approach does not work with AARCH64
	 * (and possibly other architectures). Since there is really no
	 * way to directly determine the value we can (in all cases?) look
	 * through the mapping for this process to determine what the
	 * largest address is.  This should be the top of the stack. No
	 * heap allocations should be larger than this value.  Since the
	 * largest address may differ between processes the value must be
	 * shared as part of the modex and stored in the endpoint. */
	int ret = 0;
	char buffer[1024];
	uintptr_t address_max = 0;
	FILE *fh;
	uintptr_t high;
	char *tmp;

	fi_param_define(&core_prov, "xpmem_memcpy_chunksize", FI_PARAM_SIZE_T,
			"Maximum size used for a single memcpy call. "
			"(default: %d)", XPMEM_DEFAULT_MEMCPY_CHUNK_SIZE);

	xpmem = calloc(sizeof(*xpmem), 1);
	if (!xpmem)
		return -FI_ENOMEM;

	fh = fopen("/proc/self/maps", "r");
	if (NULL == fh) {
		FI_WARN(&core_prov, FI_LOG_CORE,
				"could not open /proc/self/maps for reading");
		ret = -FI_EIO;
		goto fail;
	}

	while (fgets(buffer, sizeof(buffer), fh)) {
		/* each line of /proc/self/maps starts with low-high in
		 * hexidecimal (without a 0x) */
		(void) strtoul(buffer, &tmp, 16);
		high = strtoul(tmp + 1, NULL, 16);
		if (address_max < high)
			address_max = high;
	}
	fclose(fh);

	if (address_max == 0) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"could not determine the address max");
		ret = -FI_EIO;
		goto fail;
	}

	/* save the calcuated maximum */
	xpmem->pinfo.address_max = address_max - 1;

	/* export the process virtual address space for use with xpmem */
	xpmem->pinfo.seg_id = xpmem_make(0, XPMEM_MAXADDR_SIZE, XPMEM_PERMIT_MODE,
					 (void *) 0666);
	if (xpmem->pinfo.seg_id == -1) {
		FI_INFO(&core_prov, FI_LOG_CORE,
			"Failed to export process virtual address space for use with xpmem\n");
		ret = -FI_ENODATA;
		goto fail;
	}

	ret = fi_param_get_size_t(&core_prov, "xpmem_memcpy_chunksize",
				  &xpmem->memcpy_chunk_size);
	if (ret)
		xpmem->memcpy_chunk_size = XPMEM_DEFAULT_MEMCPY_CHUNK_SIZE;

	ret = ofi_xpmem_cache_open(&xpmem_cache);
	if (ret)
		FI_WARN(&core_prov, FI_LOG_CORE,
			"could not open xpmem cache");

	return ret;

fail:
	free(xpmem);
	xpmem = NULL;
	return ret;
}

int ofi_xpmem_cleanup(void)
{
	int ret = 0;

	if (xpmem_cache)
		ofi_xpmem_cache_destroy(xpmem_cache);

	if (xpmem)
		ret = xpmem_remove(xpmem->pinfo.seg_id);

	return (ret) ? -FI_EINVAL : ret;
}

static inline void xpmem_memcpy(void *dst, void *src, size_t size)
{
	size_t copy_size;

	while (size > 0) {
		copy_size = MIN(size, xpmem->memcpy_chunk_size);
		memcpy(dst, src, copy_size);
		dst = (void *) ((uintptr_t) dst + copy_size);
		src = (void *) ((uintptr_t) src + copy_size);
		size -= copy_size;
	}
}

int ofi_xpmem_copy(struct iovec *local, size_t local_cnt,
                   struct iovec *remote, size_t remote_cnt,
                   size_t total, pid_t pid, bool write, void *user_data)
{
	int ret;
	size_t i, j = 0, local_off = 0, rem;
	size_t offset, remote_len, chunk, delta;
	struct iovec iov;
	void *mapped_addr;
	struct ofi_mr_entry *mr_entry;
	long page_size = ofi_get_page_size();

	rem = MIN3(total, ofi_total_iov_len(local, local_cnt),
		   ofi_total_iov_len(remote, remote_cnt));

	for (i = 0; i < remote_cnt && rem > 0; i++) {
		remote_len = MIN(remote[i].iov_len, rem);
		if (!remote_len)
			continue;

		iov.iov_base = (void *) ofi_get_page_start(remote[i].iov_base,
							   page_size);
		iov.iov_len = ofi_get_page_bytes(remote[i].iov_base, remote_len,
						 page_size);

		ret = ofi_xpmem_cache_search(xpmem_cache, &iov, pid, &mr_entry,
					     (struct ofi_xpmem_client *)user_data);
		if (ret)
			return ret;

		delta = (uintptr_t) iov.iov_base -
			(uintptr_t) mr_entry->info.iov.iov_base;
		offset = (uintptr_t) remote[i].iov_base -
			 (uintptr_t) iov.iov_base;

		mapped_addr = (char *) (uintptr_t)mr_entry->info.mapped_addr +
				offset + delta;

		rem -= remote_len;
		while (remote_len > 0 && j < local_cnt) {
			chunk = MIN(remote_len, local[j].iov_len - local_off);
			if (chunk > 0) {
				if (write)
					xpmem_memcpy(mapped_addr,
						     (char *)local[j].iov_base +
						     local_off, chunk);
				else
					xpmem_memcpy((char *)local[j].iov_base +
						     local_off, mapped_addr,
						     chunk);
				mapped_addr = (char *)mapped_addr + chunk;
				remote_len -= chunk;
				local_off += chunk;
			}
			if (local_off == local[j].iov_len) {
				j++;
				local_off = 0;
			}
		}
		assert(!remote_len);

		ofi_mr_cache_delete(xpmem_cache, mr_entry);
	}

	return 0;
}

int ofi_xpmem_enable(struct ofi_xpmem_pinfo *peer,
		     struct ofi_xpmem_client *xpmem)
{
	xpmem->apid = xpmem_get(peer->seg_id,
				XPMEM_RDWR, XPMEM_PERMIT_MODE, (void *) 0666);
	if (xpmem->apid == -1)
		return -FI_ENOSYS;

	return FI_SUCCESS;
}

void ofi_xpmem_release(struct ofi_xpmem_client *xpmem)
{
	struct ofi_mr_entry *entry;
	struct dlist_entry *tmp;

	if (xpmem_cache) {
		pthread_mutex_lock(&mm_lock);
		dlist_foreach_container_safe(&xpmem_cache->lru_list,
					     struct ofi_mr_entry, entry,
					     list_entry, tmp) {
			if (entry->info.peer_id != (uint64_t) xpmem->apid)
				continue;
			dlist_remove(&entry->list_entry);
			ofi_rbmap_delete(&xpmem_cache->tree, entry->node);
			entry->node = NULL;
			xpmem_cache->cached_cnt--;
			xpmem_cache->cached_size -= entry->info.iov.iov_len;
			dlist_insert_tail(&entry->list_entry,
					  &xpmem_cache->dead_region_list);
		}
		pthread_mutex_unlock(&mm_lock);
		ofi_mr_cache_flush(xpmem_cache, false);
	}
	xpmem_release(xpmem->apid);
}

#else

int ofi_xpmem_init(void)
{
	return -FI_ENOSYS;
}

int ofi_xpmem_cleanup(void)
{
	return -FI_ENOSYS;
}

int ofi_xpmem_copy(struct iovec *local, size_t local_cnt,
                   struct iovec *remote, size_t remote_cnt,
                   size_t total, pid_t pid, bool write, void *user_data)
{
	return -FI_ENOSYS;
}

int ofi_xpmem_enable(struct ofi_xpmem_pinfo *peer,
		     struct ofi_xpmem_client *xpmem)
{
	return -FI_ENOSYS;
}

void ofi_xpmem_release(struct ofi_xpmem_client *xpmem)
{
}

#endif /* HAVE_XPMEM */
