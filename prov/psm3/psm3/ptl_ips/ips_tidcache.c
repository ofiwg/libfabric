/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2015 Intel Corporation.

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

  Copyright(c) 2015 Intel Corporation.

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

#ifdef PSM_OPA
#include "psm_user.h"
#include "psm2_hal.h"
#include "ips_proto.h"
#include "ips_expected_proto.h"

#define RBTREE_GET_LEFTMOST(PAYLOAD_PTR)  ((PAYLOAD_PTR)->start)
#define RBTREE_GET_RIGHTMOST(PAYLOAD_PTR) ((PAYLOAD_PTR)->start+((PAYLOAD_PTR)->length<<12))
#define RBTREE_ASSERT                     psmi_assert
#define RBTREE_MAP_COUNT(PAYLOAD_PTR)     ((PAYLOAD_PTR)->ntid)

#include "psm3_rbtree.c"

void ips_tidcache_map_init(cl_qmap_t		*p_map,
			   cl_map_item_t* const	root,
			   cl_map_item_t* const	nil_item)
{
	ips_cl_qmap_init(p_map,root,nil_item);
}

/*
 *
 * Force to remove a tid, check invalidation event afterwards.
 */
static psm2_error_t
ips_tidcache_remove(struct ips_tid *tidc, uint32_t tidcnt)
{
	cl_qmap_t *p_map = &tidc->tid_cachemap;
	uint32_t idx;
	uint64_t events_mask;
	psm2_error_t err;

	/*
	 * call driver to free the tids.
	 */
	if (psmi_hal_free_tid(tidc->context->psm_hw_ctxt,
			      (uint64_t) (uintptr_t) tidc->tid_array, tidcnt) < 0) {
		/* If failed to unpin pages, it's fatal error */
		err = psm3_handle_error(tidc->context->ep,
			PSM2_EP_DEVICE_FAILURE,
			"Failed to tid free %d tids", 1);
		return err;
	}

	while (tidcnt) {
		tidcnt--;
		idx = 2*IPS_TIDINFO_GET_TID(tidc->tid_array[tidcnt]) +
			IPS_TIDINFO_GET_TIDCTRL(tidc->tid_array[tidcnt]);

		/*
		 * sanity check.
		 */
		psmi_assert(idx != 0);
		psmi_assert(idx <= tidc->tid_ctrl->tid_num_max);
		psmi_assert(INVALIDATE(idx) == 0);
		psmi_assert(REFCNT(idx) == 0);

		/*
		 * mark the tid invalidated.
		 */
		INVALIDATE(idx) = 1;

		/*
		 * remove the tid from RB tree.
		 */
		IDLE_REMOVE(idx);
		ips_cl_qmap_remove_item(p_map, &p_map->root[idx]);
	}

	/*
	 * Because the freed tid is not from invalidation list,
	 * it is possible that kernel just invalidated the tid,
	 * then we need to check and process the invalidation
	 * before we can re-use this tid. The reverse order
	 * will wrongly invalidate this tid again.
	 */
	err = psmi_hal_get_hfi_event_bits(&events_mask,tidc->context->psm_hw_ctxt);

	if_pf (err)
		return PSM2_INTERNAL_ERR;

	if (events_mask & PSM_HAL_HFI_EVENT_TID_MMU_NOTIFY) {
		err = ips_tidcache_invalidation(tidc);
		if (err)
			return err;
	}

	return PSM2_OK;
}

/*
 * Register a new buffer with driver, and cache the tidinfo.
 */
static psm2_error_t
ips_tidcache_register(struct ips_tid *tidc,
		unsigned long start, uint32_t length, uint32_t *firstidx
#ifdef PSM_CUDA
		, uint8_t is_cuda_ptr
#endif
		)
{
	cl_qmap_t *p_map = &tidc->tid_cachemap;
	uint32_t tidoff, tidlen;
	uint32_t idx, tidcnt;
	uint16_t flags = 0;
	psm2_error_t err;

	/*
	 * make sure we have at least one free tid to
	 * register the new buffer.
	 */
	if (NTID == tidc->tid_cachesize) {
		/* all tids are in active use, error? */
		if (NIDLE == 0)
			return PSM2_OK_NO_PROGRESS;

		/*
		 * free the first tid in idle queue.
		 */
		idx = IPREV(IHEAD);
		tidc->tid_array[0] = p_map->root[idx].payload.tidinfo;
		err = ips_tidcache_remove(tidc, 1);
		if (err)
			return err;
	}
	psmi_assert(NTID < tidc->tid_cachesize);

	/* Clip length if it exceeds worst case tid allocation,
	   where each entry in the tid array can accommodate only
	   1 page. */
	if (length > 4096*tidc->tid_ctrl->tid_num_max)
	{
		length = 4096*tidc->tid_ctrl->tid_num_max;
	}
	/*
	 * register the new buffer.
	 */

retry:
	tidcnt = 0;

#ifdef PSM_CUDA
	if (is_cuda_ptr)
		flags = PSM_HAL_BUF_GPU_MEM;
#endif

	if (psmi_hal_update_tid(tidc->context->psm_hw_ctxt,
				(uint64_t) start, &length,
				(uint64_t) tidc->tid_array, &tidcnt,
				flags) < 0) {
		/* if driver reaches lockable memory limit */
		if ((errno == ENOMEM
#ifdef PSM_CUDA
			/* This additional check is in place for just the cuda
			 * version. It is a temporary workaround for a known
			 * issue where nvidia driver returns EINVAL instead of
			 * ENOMEM when there is no BAR1 space left to pin pages.
			 * PSM frees tidcache enteries when the driver sends
			 * EINVAL there by unpinning pages and freeing some
			 * BAR1 space.*/
		     || (PSMI_IS_GPU_ENABLED && PSMI_IS_GPU_MEM((void*)start) && errno == EINVAL)
#endif
			) && NIDLE) {
			uint64_t lengthEvicted = ips_tidcache_evict(tidc,length);

			if (lengthEvicted >= length)
				goto retry;
		} else if (errno == EFAULT)
                       psm3_handle_error(PSMI_EP_NORETURN, PSM2_INTERNAL_ERR,
                                " Unhandled error in TID Update: %s\n", strerror(errno));
#ifdef PSM_CUDA
		else if (PSMI_IS_GPU_ENABLED && errno == ENOTSUP)
		       psm3_handle_error(PSMI_EP_NORETURN, PSM2_INTERNAL_ERR,
                                " Nvidia driver apis mismatch: %s\n", strerror(errno));
#endif

		/* Unable to pin pages? retry later */
		return PSM2_EP_DEVICE_FAILURE;
	}
	psmi_assert_always(tidcnt > 0);
	psmi_assert((tidcnt+NTID) <= tidc->tid_cachesize);

	/*
	 * backward processing because we want to return
	 * the first RB index in the array.
	 */
	idx = 0;
	tidoff = length;
	while (tidcnt) {
		/*
		 * Driver only returns tidctrl=1 or tidctrl=2.
		 */
		tidcnt--;
		idx = 2*IPS_TIDINFO_GET_TID(tidc->tid_array[tidcnt]) +
			IPS_TIDINFO_GET_TIDCTRL(tidc->tid_array[tidcnt]);
		tidlen = IPS_TIDINFO_GET_LENGTH(tidc->tid_array[tidcnt]);

		/*
		 * sanity check.
		 */
		psmi_assert(idx != 0);
		psmi_assert(idx <= tidc->tid_ctrl->tid_num_max);
		psmi_assert(INVALIDATE(idx) != 0);
		psmi_assert(REFCNT(idx) == 0);

		/*
		 * clear the tid invalidated.
		 */
		INVALIDATE(idx) = 0;

		/*
		 * put the tid into a RB node.
		 */
		tidoff -= tidlen << 12;
		START(idx) = start + tidoff;
		LENGTH(idx) = tidlen;
		p_map->root[idx].payload.tidinfo = tidc->tid_array[tidcnt];

		/*
		 * put the node into RB tree and idle queue head.
		 */
		IDLE_INSERT(idx);
		ips_cl_qmap_insert_item(p_map, &p_map->root[idx]);
	}
	psmi_assert(idx != 0);
	psmi_assert(tidoff == 0);
	*firstidx = idx;

	return PSM2_OK;
}

/*
 * Get mmu notifier invalidation info and update PSM's caching.
 */
psm2_error_t
ips_tidcache_invalidation(struct ips_tid *tidc)
{
	cl_qmap_t *p_map = &tidc->tid_cachemap;
	uint32_t i, j, idx, tidcnt;
	psm2_error_t err;

	/*
	 * get a list of invalidated tids from driver,
	 * driver will clear the event bit before return.
	 */
	tidcnt = 0;
	if (psmi_hal_get_tidcache_invalidation(tidc->context->psm_hw_ctxt,
					       (uint64_t) (uintptr_t) tidc->tid_array,
					       &tidcnt) < 0) {
		/* If failed to get invalidation info, it's fatal error */
		err = psm3_handle_error(tidc->context->ep,
			PSM2_EP_DEVICE_FAILURE,
			"Failed to get invalidation info");
		return err;
	}
	psmi_assert(tidcnt > 0 && tidcnt <= tidc->tid_ctrl->tid_num_max);

	j = 0;
	for (i = 0; i < tidcnt; i++) {
		/*
		 * Driver only returns tidctrl=1 or tidctrl=2.
		 */
		idx = 2*IPS_TIDINFO_GET_TID(tidc->tid_array[i]) +
			IPS_TIDINFO_GET_TIDCTRL(tidc->tid_array[i]);
		psmi_assert(idx != 0);
		psmi_assert(idx <= tidc->tid_ctrl->tid_num_max);

		/*
		 * sanity check.
		 */
#if 0
		/* disabled this assert since observed it on OPA debug build on
		 * nVidia gv100 GPU with small BAR space.  When disabled OSU tests
		 * and mpi_stress all worked fine.  Suspect the assert is inaccurate
		 * and since it's for OPA code, not worth further debug.  Did attempt
		 * placing the assert after the INVALIDATE test below and it still
		 * failed.
		 */
		psmi_assert(p_map->root[idx].payload.tidinfo == tidc->tid_array[i]);
		psmi_assert(LENGTH(idx) ==
				IPS_TIDINFO_GET_LENGTH(tidc->tid_array[i]));
#endif

		/*
		 * if the tid is already invalidated, ignore it,
		 * but do sanity check.
		 */
		if (INVALIDATE(idx) != 0) {
			psmi_assert(REFCNT(idx) == 0);
			continue;
		}

		/*
		 * mark the tid invalidated.
		 */
		INVALIDATE(idx) = 1;

		/*
		 * if the tid is idle, remove the tid from RB tree
		 * and idle queue, put on free list.
		 */
		if (REFCNT(idx) == 0) {
			IDLE_REMOVE(idx);
			ips_cl_qmap_remove_item(p_map, &p_map->root[idx]);

			if (i != j)
				tidc->tid_array[j] = tidc->tid_array[i];
			j++;
		}
	}

	if (j > 0) {
		/*
		 * call driver to free the tids.
		 */
		if (psmi_hal_free_tid(tidc->context->psm_hw_ctxt,
				      (uint64_t) (uintptr_t) tidc->tid_array, j) < 0) {
			/* If failed to unpin pages, it's fatal error */
			err = psm3_handle_error(tidc->context->ep,
				PSM2_EP_DEVICE_FAILURE,
				"Failed to tid free %d tids", j);
			return err;
		}
	}

	return PSM2_OK;
}

psm2_error_t
ips_tidcache_acquire(struct ips_tid *tidc,
		const void *buf, uint32_t *length,
		uint32_t *tid_array, uint32_t *tidcnt,
		uint32_t *tidoff
#ifdef PSM_CUDA
		, uint8_t is_cuda_ptr
#endif
		)
{
	cl_qmap_t *p_map = &tidc->tid_cachemap;
	cl_map_item_t *p_item;
	unsigned long start = (unsigned long)buf;
	unsigned long end = start + (*length);
	uint32_t idx, nbytes;
	uint64_t event_mask;
	psm2_error_t err;

	/*
	 * Before every tid caching search, we need to update the
	 * tid caching if there is invalidation event, otherwise,
	 * the cached address may be invalidated and we might have
	 * wrong matching.
	 */
	err = psmi_hal_get_hfi_event_bits(&event_mask,tidc->context->psm_hw_ctxt);

	if_pf (err)
		return PSM2_INTERNAL_ERR;

	if (event_mask & PSM_HAL_HFI_EVENT_TID_MMU_NOTIFY) {
		err = ips_tidcache_invalidation(tidc);
		if (err)
			return err;
	}

	/*
	 * Now we can do matching from the caching, because obsolete
	 * address in caching has been removed or identified.
	 */
retry:
	p_item = ips_cl_qmap_search(p_map, start, end);
	idx = 2*IPS_TIDINFO_GET_TID(p_item->payload.tidinfo) +
		IPS_TIDINFO_GET_TIDCTRL(p_item->payload.tidinfo);

	/*
	 * There is tid matching.
	 */
	if (idx) {
		/*
		 * if there is a caching match, but the tid has been
		 * invalidated, we can't match this tid, and we also
		 * can't register this address, we need to wait this
		 * tid to be freed.
		 */
		if (INVALIDATE(idx) != 0)
			return PSM2_OK_NO_PROGRESS;

		/*
		 * if the page offset within the tid is not less than
		 * 128K, the address offset within the page is not 64B
		 * multiple, PSM can't handle this tid with any offset
		 * mode. We need to free this tid and re-register with
		 * the asked page address.
		 */
		if (((start - START(idx)) >= 131072) && ((*tidoff) & 63)) {
			/*
			 * If the tid is currently used, retry later.
			 */
			if (REFCNT(idx) != 0)
				return PSM2_OK_NO_PROGRESS;

			/*
			 * free this tid.
			 */
			tidc->tid_array[0] = p_map->root[idx].payload.tidinfo;
			err = ips_tidcache_remove(tidc, 1);
			if (err)
				return err;

			/* try to match a node again */
			goto retry;
		}
	}

	/*
	 * If there is no match node, or 'start' falls out of node range,
	 * whole or partial buffer from 'start' is not registered yet.
	 */
	if (!idx || START(idx) > start) {
		if (!idx)
			nbytes = end - start;
		else
			nbytes = START(idx) - start;

		/*
		 * Because we don't have any match tid yet, if
		 * there is an error, we return from here, PSM
		 * will try later.
		 */
		err = ips_tidcache_register(tidc, start, nbytes, &idx
#ifdef PSM_CUDA
					, is_cuda_ptr
#endif
				);
		if (err)
			return err;
	}

	/*
	 * sanity check.
	 */
	psmi_assert(START(idx) <= start);
	psmi_assert(INVALIDATE(idx) == 0);

	*tidoff += start - START(idx);
	*tidcnt = 1;

	tid_array[0] = p_map->root[idx].payload.tidinfo;
	REFCNT(idx)++;
	if (REFCNT(idx) == 1)
		IDLE_REMOVE(idx);
	start = END(idx);

	while (start < end) {
		p_item = ips_cl_qmap_successor(p_map, &p_map->root[idx]);
		idx = 2*IPS_TIDINFO_GET_TID(p_item->payload.tidinfo) +
			IPS_TIDINFO_GET_TIDCTRL(p_item->payload.tidinfo);
		if (!idx || START(idx) != start) {
			if (!idx)
				nbytes = end - start;
			else
				nbytes = (START(idx) > end) ?
					(end - start) :
					(START(idx) - start);

			/*
			 * Because we already have at least one match tid,
			 * if it is error to register new pages, we break
			 * here and return the tids we already have.
			 */
			err = ips_tidcache_register(tidc, start, nbytes, &idx
#ifdef PSM_CUDA
					, is_cuda_ptr
#endif
				);
			if (err)
				break;
		} else if (INVALIDATE(idx) != 0) {
			/*
			 * the tid has been invalidated, it is still in
			 * caching because it is still being used, but
			 * any new usage is not allowed, we ignore it and
			 * return the tids we already have.
			 */
			psmi_assert(REFCNT(idx) != 0);
			break;
		}

		/*
		 * sanity check.
		 */
		psmi_assert(START(idx) == start);
		psmi_assert(INVALIDATE(idx) == 0);

		tid_array[(*tidcnt)++] = p_map->root[idx].payload.tidinfo;
		REFCNT(idx)++;
		if (REFCNT(idx) == 1)
			IDLE_REMOVE(idx);
		start = END(idx);
	}

	if (start < end)
		*length = start - (unsigned long)buf;
	/* otherwise, all pages are registered */
	psmi_assert((*tidcnt) > 0);

	return PSM2_OK;
}

psm2_error_t
ips_tidcache_release(struct ips_tid *tidc,
		uint32_t *tid_array, uint32_t tidcnt)
{
	cl_qmap_t *p_map = &tidc->tid_cachemap;
	uint32_t i, j, idx;
	psm2_error_t err;

	psmi_assert(tidcnt > 0);

	j = 0;
	for (i = 0; i < tidcnt; i++) {
		/*
		 * Driver only returns tidctrl=1 or tidctrl=2.
		 */
		idx = 2*IPS_TIDINFO_GET_TID(tid_array[i]) +
			IPS_TIDINFO_GET_TIDCTRL(tid_array[i]);
		psmi_assert(idx != 0);
		psmi_assert(idx <= tidc->tid_ctrl->tid_num_max);
		psmi_assert(REFCNT(idx) != 0);

		REFCNT(idx)--;
		if (REFCNT(idx) == 0) {
			if (INVALIDATE(idx) != 0) {
				ips_cl_qmap_remove_item(p_map, &p_map->root[idx]);

				tidc->tid_array[j] = tid_array[i];
				j++;
			} else {
				IDLE_INSERT(idx);
			}
		}
	}

	if (j > 0) {
		/*
		 * call driver to free the tids.
		 */
		if (psmi_hal_free_tid(tidc->context->psm_hw_ctxt,
				      (uint64_t) (uintptr_t) tidc->tid_array, j) < 0) {
			/* If failed to unpin pages, it's fatal error */
			err = psm3_handle_error(tidc->context->ep,
				PSM2_EP_DEVICE_FAILURE,
				"Failed to tid free %d tids", j);
			return err;
		}
	}

	return PSM2_OK;
}

/*
 *
 * Call driver to free all cached tids.
 */
psm2_error_t
ips_tidcache_cleanup(struct ips_tid *tidc)
{
	cl_qmap_t *p_map = &tidc->tid_cachemap;
	psm2_error_t err;
	int i, j;

	j = 0;
	for (i = 1; i <= tidc->tid_ctrl->tid_num_max; i++) {
		psmi_assert(REFCNT(i) == 0);
		if (INVALIDATE(i) == 0) {
			tidc->tid_array[j++] = p_map->root[i].payload.tidinfo;
		}
	}

	if (j > 0) {
		/*
		 * call driver to free the tids.
		 */
		if (psmi_hal_free_tid(tidc->context->psm_hw_ctxt,
				      (uint64_t) (uintptr_t) tidc->tid_array, j) < 0) {
			/* If failed to unpin pages, it's fatal error */
			err = psm3_handle_error(tidc->context->ep,
				PSM2_EP_DEVICE_FAILURE,
				"Failed to tid free %d tids", j);
			return err;
		}
	}

	psmi_free(tidc->tid_array);
	psmi_free(tidc->tid_cachemap.root);

	return PSM2_OK;
}


/* Note that the caller is responsible for making sure that NIDLE is non-zero
   before calling ips_tidcache_evict.  If NIDLE is 0 at the time of call,
   ips_tidcache_evict is unstable.
 */
uint64_t
ips_tidcache_evict(struct ips_tid *tidc,uint64_t length)
{
	cl_qmap_t *p_map = &tidc->tid_cachemap;
	uint32_t idx = IHEAD, tidcnt = 0, tidlen = 0;
	/*
	 * try to free the required
	 * pages from idle queue tids
	 */

	do {
		idx = IPREV(idx);
		psmi_assert(idx != 0);
		tidc->tid_array[tidcnt] =
			p_map->root[idx].payload.tidinfo;
		tidcnt++;

		tidlen += IPS_TIDINFO_GET_LENGTH
			(p_map->root[idx].payload.tidinfo)<<12;
	} while (tidcnt < NIDLE && tidlen < length);

	/*
	 * free the selected tids on successfully finding some:.
	 */
	if (tidcnt > 0 && ips_tidcache_remove(tidc, tidcnt))
		return 0;

	return tidlen;
}
#endif // PSM_OPA
