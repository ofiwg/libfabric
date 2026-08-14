/*
 * Copyright (C) 2022-2024,2026 Cornelis Networks.
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
#ifndef _FI_PROV_OPX_TID_CACHE_H_
#define _FI_PROV_OPX_TID_CACHE_H_

#include "config.h"
#include <ofi_util.h>
#include "rdma/opx/fi_opx_tid_domain.h"
#include "fi_opx_tid.h"

/* @brief Setup the MR cache.
 *
 * This function enables the MR cache using the util MR cache code.
 *
 * @param cache		The ofi_mr_cache that is to be set up.
 * @param domain	The EFA domain where cache will be used.
 * @return 0 on success, fi_errno on failure.
 */
int opx_tid_cache_setup(struct ofi_mr_cache **cache, struct opx_tid_domain *domain);

int  opx_tid_cache_add_abort(struct ofi_mr_cache *cache, struct ofi_mr_entry *entry);
void opx_tid_cache_delete_abort(struct ofi_mr_cache *cache, struct ofi_mr_entry *entry);

enum opx_tid_cache_entry_status {
	OPX_TID_CACHE_ENTRY_NOT_FOUND = 0,
	OPX_TID_CACHE_ENTRY_FOUND,
	OPX_TID_CACHE_ENTRY_OVERLAP_LEFT,
	OPX_TID_CACHE_ENTRY_OVERLAP_RIGHT,
	OPX_TID_CACHE_ENTRY_IN_USE,
	OPX_TID_CACHE_ENTRY_LAST
};

struct opx_tid_cache_chain {
	uint32_t	     entry_count;
	struct iovec	     range;
	struct ofi_mr_entry *entries[OPX_MAX_TID_COUNT];
};

/* Flush cache entries */
int opx_tid_cache_flush_all(struct ofi_mr_cache *cache, const bool flush_lru, const bool flush_all);

__OPX_FORCE_INLINE__
int opx_tid_cache_flush(struct ofi_mr_cache *cache, const bool flush_lru)
{
	/* Nothing to do, early exit */
	if (dlist_empty(&cache->dead_region_list) && (!flush_lru || dlist_empty(&cache->lru_list))) {
		return 0;
	}

	pthread_mutex_unlock(&mm_lock);

	/* Flush dead list and possibly one lru entry */
	int freed_entries = opx_tid_cache_flush_all(cache, flush_lru, false);

	pthread_mutex_lock(&mm_lock);
	return freed_entries;
}

/*
 * Force a contiguous run of pages resident from the calling (application)
 * thread by issuing a non-destructive write access to one byte of every page
 * in the byte range [start, end).
 *
 * A WRITE access is required: the userfaultfd memory monitor only services
 * UFFD_PAGEFAULT_FLAG_WRITE faults, so a read access would raise a fault the
 * handler refuses to resolve and the page would stay absent. The hfi1
 * TID_UPDATE ioctl pins with FOLL_WRITE, so it likewise needs writable,
 * present pages. __atomic_fetch_or() with an operand of 0 performs a real
 * read-modify-write store yet leaves the byte value unchanged, so existing
 * buffer contents are preserved.
 *
 * Faulting the pages in lets pin_user_pages cover the range in a single ioctl
 * instead of stopping at the first not-present page (partial update or
 * EFAULT).
 *
 * The page target MUST be cast through a volatile-qualified pointer. To the C
 * abstract machine "x | 0 == x" with a discarded result is a no-op, so an
 * optimizing compiler (clang at -O3 in particular; gcc happens to keep it) is
 * free to delete the atomic entirely. The page-fault side effect we depend on
 * is invisible to the optimizer, so without volatile the prefault loop is
 * elided, TID_UPDATE then faults on not-present pages, and the transfer hangs.
 * The volatile qualifier marks the access as observable and forces the
 * compiler to emit the read-modify-write store at every optimization level. */
static inline void opx_write_prefault(uint64_t vaddr, uint32_t start, uint32_t end)
{
	FI_DBG(&fi_opx_provider, FI_LOG_MR, "prefault %p bytes [%u - %u] (pages %lu)\n", (char *) (vaddr + start),
	       start, end, (unsigned long) ((end - start) / PAGE_SIZE));
	for (uint32_t off = start; off < end; off += PAGE_SIZE) {
		__atomic_fetch_or((volatile char *) (vaddr + off), 0, __ATOMIC_RELAXED);
	}
}

/* Purge all entries for the specified endpoint */
void opx_tid_cache_purge_ep(struct ofi_mr_cache *cache, struct fi_opx_ep *opx_ep);

/* Cleanup the cache at exit/finalize */
void opx_tid_cache_cleanup(struct ofi_mr_cache *cache);

/* De-register a TID rendezvous registration by releasing the owned cache entries */
void opx_deregister_entries_for_rzv(struct fi_opx_ep *opx_ep, struct ofi_mr_entry **entries, const uint8_t nentries);

/* Register a memory region for TID rendezvous,
 * return 0 on success
 * returns non-zero on failure
 */
int opx_register_for_rzv(struct fi_opx_ep *opx_ep, struct fi_opx_hmem_iov *cur_addr_range,
			 struct opx_tid_addr_block *tid_addr_block, const struct opx_tid_dmabuf_ref *dmabuf);
#endif /* _FI_PROV_OPX_TID_CACHE_H_ */
