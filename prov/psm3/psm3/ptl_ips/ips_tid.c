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

/* Copyright (c) 2003-2014 Intel Corporation. All rights reserved. */

#ifdef PSM_OPA
#include "psm_user.h"
#include "psm2_hal.h"
#include "ips_tid.h"
#include "ips_proto.h"
#include "ips_expected_proto.h"

psm2_error_t
ips_tid_init(const psmi_context_t *context, struct ips_protoexp *protoexp,
	     ips_tid_avail_cb_fn_t cb, void *cb_context)
{
	struct ips_tid *tidc = &protoexp->tidc;

	struct psmi_stats_entry entries[] = {
		PSMI_STATS_DECL("tid_update_count", MPSPAWN_STATS_REDUCTION_ALL,
				NULL, &tidc->tid_num_total),
	};

	tidc->context = context;
	tidc->protoexp = protoexp;
	tidc->tid_num_total = 0;
	tidc->tid_num_inuse = 0;
	tidc->tid_avail_cb = cb;
	tidc->tid_avail_context = cb_context;
	tidc->tid_array = NULL;

	/*
	 * PSM uses tid registration caching only if driver has enabled it.
	 */
	if (!psmi_hal_has_cap(PSM_HAL_CAP_TID_UNMAP)) {
		int i;
		cl_qmap_t *p_map;
		cl_map_item_t *root,*nil_item;

		tidc->tid_array = (uint32_t *)
			psmi_calloc(context->ep, UNDEFINED,
				    psmi_hal_get_tid_exp_cnt(context->psm_hw_ctxt),
				    sizeof(uint32_t));
		if (tidc->tid_array == NULL)
			return PSM2_NO_MEMORY;

		/*
		 * first is root node, last is terminator node.
		 */
		p_map = &tidc->tid_cachemap;
		root = (cl_map_item_t *)
			psmi_calloc(context->ep, UNDEFINED,
				    psmi_hal_get_tid_exp_cnt(context->psm_hw_ctxt) + 2,
				    sizeof(cl_map_item_t));

		if (root == NULL)
			return PSM2_NO_MEMORY;

		nil_item = &root
			[psmi_hal_get_tid_exp_cnt(context->psm_hw_ctxt) + 1];

		ips_tidcache_map_init(p_map,root,nil_item);

		NTID = 0;
		NIDLE = 0;
		IPREV(IHEAD) = INEXT(IHEAD) = IHEAD;
		for (i = 1; i <= psmi_hal_get_tid_exp_cnt(context->psm_hw_ctxt); i++) {
			INVALIDATE(i) = 1;
		}

		/*
		 * if not shared context, all tids are used by the same
		 * process. Otherwise, subcontext process can only cache
		 * its own portion. Driver makes the same tid number
		 * assignment to subcontext processes.
		 */
		tidc->tid_cachesize = psmi_hal_get_tid_exp_cnt(context->psm_hw_ctxt);
		if (psmi_hal_get_subctxt_cnt(context->psm_hw_ctxt) > 0) {
			uint16_t remainder = tidc->tid_cachesize %
					psmi_hal_get_subctxt_cnt(context->psm_hw_ctxt);
			tidc->tid_cachesize /= psmi_hal_get_subctxt_cnt(context->psm_hw_ctxt);
			if (psmi_hal_get_subctxt(context->psm_hw_ctxt) < remainder)
				tidc->tid_cachesize++;
		}
	}

	/*
	 * Setup shared control structure.
	 */
	tidc->tid_ctrl = (struct ips_tid_ctrl *)context->tid_ctrl;
	if (!tidc->tid_ctrl) {
		tidc->tid_ctrl = (struct ips_tid_ctrl *)
		    psmi_calloc(context->ep, UNDEFINED, 1,
				sizeof(struct ips_tid_ctrl));
		if (tidc->tid_ctrl == NULL) {
			return PSM2_NO_MEMORY;
		}
	}

	/*
	 * Only the master process can initialize.
	 */
	if (psmi_hal_get_subctxt(context->psm_hw_ctxt) == 0) {
		pthread_spin_init(&tidc->tid_ctrl->tid_ctrl_lock,
					PTHREAD_PROCESS_SHARED);

		tidc->tid_ctrl->tid_num_max =
			    psmi_hal_get_tid_exp_cnt(context->psm_hw_ctxt);
		tidc->tid_ctrl->tid_num_avail = tidc->tid_ctrl->tid_num_max;
	}

	return psm3_stats_register_type("TID_Statistics",
					PSMI_STATSTYPE_RDMA,
					entries,
					PSMI_HOWMANY(entries),
					psm3_epid_fmt_internal(protoexp->proto->ep->epid, 0), tidc,
					protoexp->proto->ep->dev_name);
}

psm2_error_t ips_tid_fini(struct ips_tid *tidc)
{
	psm3_stats_deregister_type(PSMI_STATSTYPE_RDMA, tidc);

	if (tidc->tid_array)
		ips_tidcache_cleanup(tidc);

	if (!tidc->context->tid_ctrl)
		psmi_free(tidc->tid_ctrl);

	return PSM2_OK;
}

psm2_error_t
ips_tid_acquire(struct ips_tid *tidc,
		const void *buf, uint32_t *length,
		uint32_t *tid_array, uint32_t *tidcnt
#ifdef PSM_CUDA
		, uint8_t is_cuda_ptr
#endif
		)
{
	struct ips_tid_ctrl *ctrl = tidc->tid_ctrl;
	psm2_error_t err = PSM2_OK;
	uint16_t flags = 0;
	int rc;

	psmi_assert(((uintptr_t) buf & 0xFFF) == 0);
	psmi_assert(((*length) & 0xFFF) == 0);

	if (tidc->context->tid_ctrl)
		pthread_spin_lock(&ctrl->tid_ctrl_lock);

	if (!ctrl->tid_num_avail) {
		err = PSM2_EP_NO_RESOURCES;
		goto fail;
	}

	/* Clip length if it exceeds worst case tid allocation,
	   where each entry in the tid array can accommodate only
	   1 page. */
	if (*length > 4096*tidc->tid_ctrl->tid_num_max)
	{
		*length = 4096*tidc->tid_ctrl->tid_num_max;
	}

#ifdef PSM_CUDA
	if (is_cuda_ptr)
		flags = PSM_HAL_BUF_GPU_MEM;
#endif

	rc = psmi_hal_update_tid(tidc->context->psm_hw_ctxt,
				 (uint64_t) (uintptr_t) buf, length,
				 (uint64_t) (uintptr_t) tid_array, tidcnt, flags);

	if (rc < 0) {
		/* Unable to pin pages? retry later */
		err = PSM2_EP_DEVICE_FAILURE;
		goto fail;
	}

	psmi_assert_always((*tidcnt) > 0);
	psmi_assert(ctrl->tid_num_avail >= (*tidcnt));
	ctrl->tid_num_avail -= (*tidcnt);
	tidc->tid_num_total += (*tidcnt);
	tidc->tid_num_inuse += (*tidcnt);

fail:
	if (tidc->context->tid_ctrl)
		pthread_spin_unlock(&ctrl->tid_ctrl_lock);

	return err;
}

psm2_error_t
ips_tid_release(struct ips_tid *tidc,
		uint32_t *tid_array, uint32_t tidcnt)
{
	struct ips_tid_ctrl *ctrl = tidc->tid_ctrl;
	psm2_error_t err = PSM2_OK;

	psmi_assert(tidcnt > 0);
	if (tidc->context->tid_ctrl)
		pthread_spin_lock(&ctrl->tid_ctrl_lock);

	if (psmi_hal_free_tid(tidc->context->psm_hw_ctxt,
			      (uint64_t) (uintptr_t) tid_array, tidcnt) < 0) {
		if (tidc->context->tid_ctrl)
			pthread_spin_unlock(&ctrl->tid_ctrl_lock);

		/* If failed to unpin pages, it's fatal error */
		err = psm3_handle_error(tidc->context->ep,
			PSM2_EP_DEVICE_FAILURE,
			"Failed to tid free %d tids",
			tidcnt);
		goto fail;
	}

	ctrl->tid_num_avail += tidcnt;
	if (tidc->context->tid_ctrl)
		pthread_spin_unlock(&ctrl->tid_ctrl_lock);

	tidc->tid_num_inuse -= tidcnt;
	/* If an available callback is registered invoke it */
	if (((tidc->tid_num_inuse + tidcnt) == ctrl->tid_num_max)
	    && tidc->tid_avail_cb)
		tidc->tid_avail_cb(tidc, tidc->tid_avail_context);

fail:
	return err;
}
#endif // PSM_OPA
