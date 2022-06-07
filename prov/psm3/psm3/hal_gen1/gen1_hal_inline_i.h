#ifdef PSM_OPA
/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2017 Intel Corporation.

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

  Copyright(c) 2017 Intel Corporation.

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

#include "gen1_hal.h"

static PSMI_HAL_INLINE int      psm3_hfp_gen1_get_jkey(psm2_ep_t ep);

extern size_t psm3_gen1_arrsz[MAPSIZE_MAX];

static void psm3_gen1_free_egr_buffs(hfp_gen1_pc_private *psm_hw_ctxt)
{
#define FREE_EGR_BUFFS_TABLE(cl_qs_arr, index)          psm3_ips_recvq_egrbuf_table_free(((cl_qs_arr)[index]).egr_buffs)
	size_t i, index, subctxt_cnt;
	psm3_gen1_cl_q_t *cl_qs;

	cl_qs = psm_hw_ctxt->cl_qs;
	index = PSM3_GEN1_CL_Q_RX_EGR_Q;
	FREE_EGR_BUFFS_TABLE(cl_qs, index);

	subctxt_cnt = psm_hw_ctxt->user_info.subctxt_cnt;
	for (i = 0; i < subctxt_cnt; i++) {
		index = PSM3_GEN1_GET_SC_CL_Q_RX_EGR_Q(i);
		FREE_EGR_BUFFS_TABLE(cl_qs, index);
	}
#undef FREE_EGR_BUFFS_TABLE
}

static void psm3_gen1_unmap_hfi_mem(hfp_gen1_pc_private *psm_hw_ctxt)
{
	size_t subctxt_cnt = psm_hw_ctxt->user_info.subctxt_cnt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;
	struct hfi1_base_info *binfo = &ctrl->base_info;
	struct hfi1_ctxt_info *cinfo = &ctrl->ctxt_info;

	/* 1. Unmap the PIO credits address */
	HFI_MUNMAP_ERRCHECK(binfo, sc_credits_addr, psm3_gen1_arrsz[SC_CREDITS]);

	/* 2. Unmap the PIO buffer SOP address */
	HFI_MUNMAP_ERRCHECK(binfo, pio_bufbase_sop, psm3_gen1_arrsz[PIO_BUFBASE_SOP]);

	/* 3. Unmap the PIO buffer address */
	HFI_MUNMAP_ERRCHECK(binfo, pio_bufbase, psm3_gen1_arrsz[PIO_BUFBASE]);

	/* 4. Unmap the receive header queue */
	HFI_MUNMAP_ERRCHECK(binfo, rcvhdr_bufbase, psm3_gen1_arrsz[RCVHDR_BUFBASE]);

	/* 5. Unmap the receive eager buffer */
	HFI_MUNMAP_ERRCHECK(binfo, rcvegr_bufbase, psm3_gen1_arrsz[RCVEGR_BUFBASE]);

	/* 6. Unmap the sdma completion queue */
	HFI_MUNMAP_ERRCHECK(binfo, sdma_comp_bufbase, psm3_gen1_arrsz[SDMA_COMP_BUFBASE]);

	/* 7. Unmap RXE per-context CSRs */
	HFI_MUNMAP_ERRCHECK(binfo, user_regbase, psm3_gen1_arrsz[USER_REGBASE]);
	ctrl->__hfi_rcvhdrtail = NULL;
	ctrl->__hfi_rcvhdrhead = NULL;
	ctrl->__hfi_rcvegrtail = NULL;
	ctrl->__hfi_rcvegrhead = NULL;
	ctrl->__hfi_rcvofftail = NULL;
	if (cinfo->runtime_flags & HFI1_CAP_HDRSUPP) {
		ctrl->__hfi_rcvtidflow = NULL;
	}

	/* 8. Unmap the rcvhdrq tail register address */
	if (cinfo->runtime_flags & HFI1_CAP_DMA_RTAIL) {
		/* only unmap the RTAIL if it was enabled in the first place */
		HFI_MUNMAP_ERRCHECK(binfo, rcvhdrtail_base, psm3_gen1_arrsz[RCVHDRTAIL_BASE]);
	} else {
		binfo->rcvhdrtail_base = 0;
	}

	/* 9. Unmap the event page */
	HFI_MUNMAP_ERRCHECK(binfo, events_bufbase, psm3_gen1_arrsz[EVENTS_BUFBASE]);

	/* 10. Unmap the status page */
	HFI_MUNMAP_ERRCHECK(binfo, status_bufbase, psm3_gen1_arrsz[STATUS_BUFBASE]);

	/* 11. If subcontext is used, unmap the buffers */
	if (subctxt_cnt > 0) {
		/* only unmap subcontext-related stuff it subcontexts are enabled */
		HFI_MUNMAP_ERRCHECK(binfo, subctxt_uregbase, psm3_gen1_arrsz[SUBCTXT_UREGBASE]);
		HFI_MUNMAP_ERRCHECK(binfo, subctxt_rcvhdrbuf, psm3_gen1_arrsz[SUBCTXT_RCVHDRBUF]);
		HFI_MUNMAP_ERRCHECK(binfo, subctxt_rcvegrbuf, psm3_gen1_arrsz[SUBCTXT_RCVEGRBUF]);
	}
}

#include "gen1_spio.c"

static PSMI_HAL_INLINE int psm3_hfp_gen1_close_context(psm2_ep_t ep)
{
	hfp_gen1_pc_private *psm_hw_ctxt = (hfp_gen1_pc_private *)(ep->context.psm_hw_ctxt);

	if (!psm_hw_ctxt)
		return PSM_HAL_ERROR_OK;
	/* Free the egress buffers */
	psm3_gen1_free_egr_buffs(psm_hw_ctxt);

	/* Unmap the HFI memory */
	psm3_gen1_unmap_hfi_mem(psm_hw_ctxt);

	/* Clean up the rest */
	close(psm_hw_ctxt->ctrl->fd);
	free(psm_hw_ctxt->ctrl);
	psmi_free(psm_hw_ctxt);
	ep->context.psm_hw_ctxt = 0;

	return PSM_HAL_ERROR_OK;
}

/* Check NIC and context status, returns one of
 *
 * PSM2_OK: Port status is ok (or context not initialized yet but still "ok")
 * PSM2_OK_NO_PROGRESS: Cable pulled
 * PSM2_EP_NO_NETWORK: No network, no lid, ...
 * PSM2_EP_DEVICE_FAILURE: Chip failures, rxe/txe parity, etc.
 */
static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_context_check_status(struct ptl_ips *ptl)
{
	psm2_error_t err = psm3_gen1_context_check_hw_status(ptl->ep);
	if (err == PSM2_OK || err == PSM2_OK_NO_PROGRESS)
	{
		int rc = psm3_gen1_spio_process_events((struct ptl *)ptl);
		err = rc >= 0 ? PSM2_OK : PSM2_INTERNAL_ERR;
	}
	return err;
}

#ifdef PSM_FI
static PSMI_HAL_INLINE int psm3_hfp_gen1_faultinj_allowed(const char *name,
			psm2_ep_t ep)
{
	return 1;
}
#endif

/* Moved from psm_context.c */

ustatic PSMI_HAL_INLINE
int MOCKABLE(psm3_gen1_sharedcontext_params)(int *nranks, int *rankid);
MOCK_DCL_EPILOGUE(psm3_gen1_sharedcontext_params);
ustatic PSMI_HAL_INLINE psm2_error_t psm3_gen1_init_userinfo_params(psm2_ep_t ep,
					     int unit_id,
					     psm2_uuid_t const unique_job_key,
					     struct hfi1_user_info_dep *user_info);

/*
 * Prepare user_info params for driver open, used only in psm3_context_open
 */
ustatic PSMI_HAL_INLINE
psm2_error_t
psm3_gen1_init_userinfo_params(psm2_ep_t ep, int unit_id,
			  psm2_uuid_t const unique_job_key,
			  struct hfi1_user_info_dep *user_info)
{
	// TBD - known issue, when HAL is built as pure inline
	// can't declare static variables in an inline function
	// (and shouldn't delcare in a header file in general)
	/* static variables, shared among rails */
	static int shcontexts_enabled = -1, rankid, nranks;

	int avail_contexts = 0, max_contexts, ask_contexts;
	int ranks_per_context = 0;
	psm2_error_t err = PSM2_OK;
	union psmi_envvar_val env_maxctxt, env_ranks_per_context;
	static int subcontext_id_start;

	memset(user_info, 0, sizeof(*user_info));
	user_info->userversion = HFI1_USER_SWMINOR|(psm3_gen1_get_user_major_version()<<HFI1_SWMAJOR_SHIFT);

	user_info->subctxt_id = 0;
	user_info->subctxt_cnt = 0;
	memcpy(user_info->uuid, unique_job_key, sizeof(user_info->uuid));

	if (shcontexts_enabled == -1) {
		shcontexts_enabled =
		    psm3_gen1_sharedcontext_params(&nranks, &rankid);
	}
	if (!shcontexts_enabled)
		return err;

	avail_contexts = psm3_hfp_gen1_get_num_contexts(unit_id);

	if (avail_contexts == 0) {
		err = psm3_handle_error(NULL, PSM2_EP_NO_DEVICE,
					"PSM3 found 0 available contexts on opa device(s).");
		goto fail;
	}

	/* See if the user wants finer control over context assignments */
	if (!psm3_getenv("PSM3_MAX_CONTEXTS_PER_JOB",
			 "Maximum number of contexts for this PSM3 job",
			 PSMI_ENVVAR_LEVEL_USER, PSMI_ENVVAR_TYPE_INT,
			 (union psmi_envvar_val)avail_contexts, &env_maxctxt)) {
		max_contexts = max(env_maxctxt.e_int, 1);		/* needs to be non-negative */
		ask_contexts = min(max_contexts, avail_contexts);	/* needs to be available */
	} else if (!psm3_getenv("PSM3_SHAREDCONTEXTS_MAX",
				"",  /* deprecated */
				PSMI_ENVVAR_LEVEL_HIDDEN | PSMI_ENVVAR_LEVEL_NEVER_PRINT,
				PSMI_ENVVAR_TYPE_INT,
				(union psmi_envvar_val)avail_contexts, &env_maxctxt)) {

		_HFI_INFO
		    ("The PSM3_SHAREDCONTEXTS_MAX env variable is deprecated. Please use PSM3_MAX_CONTEXTS_PER_JOB in future.\n");

		max_contexts = max(env_maxctxt.e_int, 1);		/* needs to be non-negative */
		ask_contexts = min(max_contexts, avail_contexts);	/* needs to be available */
	} else
		ask_contexts = max_contexts = avail_contexts;

	if (!psm3_getenv("PSM3_RANKS_PER_CONTEXT",
			 "Number of ranks per context",
			 PSMI_ENVVAR_LEVEL_USER, PSMI_ENVVAR_TYPE_INT,
			 (union psmi_envvar_val)1, &env_ranks_per_context)) {
		ranks_per_context = max(env_ranks_per_context.e_int, 1);
		ranks_per_context = min(ranks_per_context, HFI1_MAX_SHARED_CTXTS);
	}

	/*
	 * See if we could get a valid ppn.  If not, approximate it to be the
	 * number of cores.
	 */
	if (nranks == -1) {
		long nproc = sysconf(_SC_NPROCESSORS_ONLN);
		if (nproc < 1)
			nranks = 1;
		else
			nranks = nproc;
	}

	/*
	 * Make sure that our guesses are good educated guesses
	 */
	if (rankid >= nranks) {
		_HFI_PRDBG
		    ("PSM3_SHAREDCONTEXTS disabled because lrank=%d,ppn=%d\n",
		     rankid, nranks);
		goto fail;
	}

	if (ranks_per_context) {
		int contexts =
		    (nranks + ranks_per_context - 1) / ranks_per_context;
		if (contexts > ask_contexts) {
			err = psm3_handle_error(NULL, PSM2_EP_NO_DEVICE,
						"Incompatible settings for "
						"PSM3_MAX_CONTEXTS_PER_JOB and PSM3_RANKS_PER_CONTEXT");
			goto fail;
		}
		ask_contexts = contexts;
	}

	/* group id based on total groups and local rank id */
	user_info->subctxt_id = subcontext_id_start + rankid % ask_contexts;
	/* this is for multi-rail, when we setup a new rail,
	 * we can not use the same subcontext ID as the previous
	 * rail, otherwise, the driver will match previous rail
	 * and fail.
	 */
	subcontext_id_start += ask_contexts;

	/* Need to compute with how many *other* peers we will be sharing the
	 * context */
	if (nranks > ask_contexts) {
		user_info->subctxt_cnt = nranks / ask_contexts;
		/* If ppn != multiple of contexts, some contexts get an uneven
		 * number of subcontexts */
		if (nranks % ask_contexts > rankid % ask_contexts)
			user_info->subctxt_cnt++;
		/* The case of 1 process "sharing" a context (giving 1 subcontext)
		 * is supcontexted by the driver and PSM. However, there is no
		 * need to share in this case so disable context sharing. */
		if (user_info->subctxt_cnt == 1)
			user_info->subctxt_cnt = 0;
		if (user_info->subctxt_cnt > HFI1_MAX_SHARED_CTXTS) {
			err = psm3_handle_error(NULL, PSM2_INTERNAL_ERR,
						"Calculation of subcontext count exceeded maximum supported");
			goto fail;
		}
	}
	/* else subcontext_cnt remains 0 and context sharing is disabled. */

	_HFI_PRDBG("PSM3_SHAREDCONTEXTS lrank=%d,ppn=%d,avail_contexts=%d,"
		   "max_contexts=%d,ask_contexts=%d,"
		   "ranks_per_context=%d,id=%u,cnt=%u\n",
		   rankid, nranks, avail_contexts, max_contexts,
		   ask_contexts, ranks_per_context,
		   user_info->subctxt_id, user_info->subctxt_cnt);
fail:
	return err;
}

ustatic
int MOCKABLE(psm3_gen1_sharedcontext_params)(int *nranks, int *rankid)
{
	union psmi_envvar_val enable_shcontexts;

	*rankid = -1;
	*nranks = -1;

	/* We do not support context sharing for multiple endpoints */
	if (psm3_multi_ep_enabled) {
		return 0;
	}

	/* New name in 2.0.1, keep observing old name */
	psm3_getenv("PSM3_SHAREDCONTEXTS", "Enable shared contexts",
		    PSMI_ENVVAR_LEVEL_USER, PSMI_ENVVAR_TYPE_YESNO,
		    (union psmi_envvar_val)
		    PSMI_SHARED_CONTEXTS_ENABLED_BY_DEFAULT,
		    &enable_shcontexts);
	if (!enable_shcontexts.e_int)
		return 0;

	if (psm3_get_mylocalrank() >= 0 && psm3_get_mylocalrank_count() >= 0) {
		*rankid = psm3_get_mylocalrank();
		*nranks = psm3_get_mylocalrank_count();
		return 1;
	} else
		return 0;
}
MOCK_DEF_EPILOGUE(psm3_gen1_sharedcontext_params);

/* moved from ips_subcontext.c */
static PSMI_HAL_INLINE psm2_error_t
psm3_gen1_divvy_shared_mem_ptrs(hfp_gen1_pc_private *pc_private,
		      psmi_context_t *context,
		      const struct hfi1_base_info *base_info)
{
	struct gen1_ips_hwcontext_ctrl **hwcontext_ctrl = &pc_private->hwcontext_ctrl;
	uint32_t subcontext_cnt                    = pc_private->user_info.subctxt_cnt;
	struct gen1_ips_subcontext_ureg **uregp         = &pc_private->subcontext_ureg[0];

	uintptr_t all_subcontext_uregbase =
	    (uintptr_t) base_info->subctxt_uregbase;
	int i;

	psmi_assert_always(all_subcontext_uregbase != 0);
	for (i = 0; i < HFI1_MAX_SHARED_CTXTS; i++) {
		struct gen1_ips_subcontext_ureg *subcontext_ureg =
		    (struct gen1_ips_subcontext_ureg *)all_subcontext_uregbase;
		*uregp++ = (i < subcontext_cnt) ? subcontext_ureg : NULL;
		all_subcontext_uregbase += sizeof(struct gen1_ips_subcontext_ureg);
	}

	*hwcontext_ctrl =
	    (struct gen1_ips_hwcontext_ctrl *)all_subcontext_uregbase;
	all_subcontext_uregbase += sizeof(struct gen1_ips_hwcontext_ctrl);

	context->spio_ctrl = (void *)all_subcontext_uregbase;
	all_subcontext_uregbase += sizeof(struct psm3_gen1_spio_ctrl);

	context->tid_ctrl = (void *)all_subcontext_uregbase;
	all_subcontext_uregbase += sizeof(struct ips_tid_ctrl);

	context->tf_ctrl = (void *)all_subcontext_uregbase;
	all_subcontext_uregbase += sizeof(struct ips_tf_ctrl);

	psmi_assert((all_subcontext_uregbase -
		     (uintptr_t) base_info->subctxt_uregbase) <= PSMI_PAGESIZE);

	return PSM2_OK;
}

static PSMI_HAL_INLINE
uint64_t psm3_gen1_get_cap_mask(uint64_t gen1_mask)
{
	// TBD - known issue, when HAL is built as pure inline
	// can't declare static variables in an inline function
	// (and shouldn't delcare in a header file in general)
	static  const struct
	{
		uint64_t gen1_bit;
		uint32_t psmi_hal_bit;
	} bit_map[] =
	  {
		  { HFI1_CAP_SDMA,		  PSM_HAL_CAP_SDMA		     },
		  { HFI1_CAP_SDMA_AHG,		  PSM_HAL_CAP_SDMA_AHG	     },
		  { HFI1_CAP_EXTENDED_PSN,	  PSM_HAL_CAP_EXTENDED_PSN	     },
		  { HFI1_CAP_HDRSUPP,		  PSM_HAL_CAP_HDRSUPP	     },
		  { HFI1_CAP_USE_SDMA_HEAD,	  PSM_HAL_CAP_USE_SDMA_HEAD       },
		  { HFI1_CAP_MULTI_PKT_EGR,	  PSM_HAL_CAP_MULTI_PKT_EGR       },
		  { HFI1_CAP_NODROP_RHQ_FULL,	  PSM_HAL_CAP_NODROP_RHQ_FULL     },
		  { HFI1_CAP_NODROP_EGR_FULL,	  PSM_HAL_CAP_NODROP_EGR_FULL     },
		  { HFI1_CAP_TID_UNMAP,		  PSM_HAL_CAP_TID_UNMAP           },
		  { HFI1_CAP_PRINT_UNIMPL,	  PSM_HAL_CAP_PRINT_UNIMPL        },
		  { HFI1_CAP_ALLOW_PERM_JKEY,	  PSM_HAL_CAP_ALLOW_PERM_JKEY     },
		  { HFI1_CAP_NO_INTEGRITY,	  PSM_HAL_CAP_NO_INTEGRITY        },
		  { HFI1_CAP_PKEY_CHECK,	  PSM_HAL_CAP_PKEY_CHECK          },
		  { HFI1_CAP_STATIC_RATE_CTRL,	  PSM_HAL_CAP_STATIC_RATE_CTRL    },
		  { HFI1_CAP_SDMA_HEAD_CHECK,	  PSM_HAL_CAP_SDMA_HEAD_CHECK     },
		  { HFI1_CAP_EARLY_CREDIT_RETURN, PSM_HAL_CAP_EARLY_CREDIT_RETURN },
#ifdef HFI1_CAP_GPUDIRECT_OT
		  { HFI1_CAP_GPUDIRECT_OT,        PSM_HAL_CAP_GPUDIRECT           },
		  { HFI1_CAP_GPUDIRECT_OT,        PSM_HAL_CAP_GPUDIRECT_RDMA      },
#else /* #ifdef HFI1_CAP_GPUDIRECT_OT */
#ifndef PSM_CUDA
		  /* lifted from hfi1_user.h */
		  { (1UL << 63),                  PSM_HAL_CAP_GPUDIRECT           },
		  { (1UL << 63),                  PSM_HAL_CAP_GPUDIRECT_RDMA      },
#else /* #ifndef PSM_CUDA */
#error "Inconsistent build.  HFI1_CAP_GPUDIRECT_OT must be defined for CUDA builds. Must use CUDA enabled driver headers"
#endif /* #ifndef PSM_CUDA */
#endif /* #ifdef HFI1_CAP_GPUDIRECT_OT */
	  };
	uint64_t rv = 0;
	int i;
	for (i=0;i < sizeof(bit_map)/sizeof(bit_map[0]);i++)
	{
		if (bit_map[i].gen1_bit & gen1_mask)
			rv |= bit_map[i].psmi_hal_bit;
	}
	return rv;
}

static PSMI_HAL_INLINE int psm3_hfp_gen1_context_open(int unit,
				 int port, int addr_index,
				 uint64_t open_timeout,
				 psm2_ep_t ep,
				 psm2_uuid_t const job_key,
				 unsigned retryCnt)
{
	psm2_error_t err = PSM2_OK;
	int fd = -1;
	psmi_context_t *psm_ctxt = &ep->context;
	hfp_gen1_pc_private *pc_private = psmi_malloc(ep, UNDEFINED, sizeof(hfp_gen1_pc_private));

	psmi_assert_always(!ep->context.psm_hw_ctxt);
	psmi_assert_always(psm3_epid_zero_internal(ep->epid));
	if_pf (!pc_private) {
		//err = -PSM_HAL_ERROR_CANNOT_OPEN_CONTEXT;
		goto bail_fd;
	}

	memset(pc_private, 0, sizeof(hfp_gen1_pc_private));

	ep->rdmamode = psm3_gen1_parse_tid(0);
	// MR cache N/A (gen1 uses TID cache), leave ep->mr_cache_mode and
	// ep->rv_gpu_cache_size as set by caller (NONE, 0)

	char dev_name[PATH_MAX];
	fd = psm3_gen1_nic_context_open_ex(unit, port, open_timeout,
					       dev_name, sizeof(dev_name));
	if (fd < 0)
	{
		err = -PSM_HAL_ERROR_CANNOT_OPEN_DEVICE;
		goto bail_fd;
	}

	err = psm3_gen1_init_userinfo_params(ep,
						     unit,
						     job_key,
						     &pc_private->user_info);
	if (err) {
		err = -PSM_HAL_ERROR_GENERAL_ERROR;
		goto bail_fd;
	}

	cpu_set_t mycpuset;
	if (psm3_sysfs_get_unit_cpumask(unit, &mycpuset)) {
		_HFI_ERROR( "Failed to get %s (unit %d) cpu set\n", ep->dev_name, unit);
		//err = -PSM_HAL_ERROR_GENERAL_ERROR;
		goto bail_fd;
	}

	if (psm3_context_set_affinity(ep, mycpuset))
		goto bail_fd;

	/* attempt to assign the context via psm3_gen1_userinit_internal()
	 * and mmap the HW resources */
	int retry = 0;
	do {
		if (retry > 0)
			_HFI_INFO("psm3_gen1_userinit_internal: failed, trying again (%d/%d)\n",
				  retry, retryCnt);
		pc_private->ctrl = psm3_gen1_userinit_internal(fd, ep->skip_affinity,
				&pc_private->user_info);
	} while (pc_private->ctrl == NULL && ++retry <= retryCnt);

	if (!pc_private->ctrl)
	{
		err = -PSM_HAL_ERROR_CANNOT_OPEN_CONTEXT;
		goto bail_fd;
	}
	else
	{

		if (psm3_parse_identify()) {
			printf("%s %s run-time driver interface v%d.%d\n",
			       psm3_get_mylabel(), psm3_ident_tag,
			       psm3_gen1_get_user_major_version(),
			       psm3_gen1_get_user_minor_version());
		}

		struct _hfi_ctrl *ctrl = pc_private->ctrl;
		int i;
		int lid;

		if ((lid = psm3_gen1_get_port_lid(ctrl->__hfi_unit,
				     ctrl->__hfi_port, addr_index, GEN1_FILTER)) <= 0) {
			err = psm3_handle_error(NULL,
						PSM2_EP_DEVICE_FAILURE,
						"Can't get HFI LID in psm3_ep_open: is SMA running?");
			goto bail;
		}
		if (psm3_hfp_gen1_get_port_subnet(ctrl->__hfi_unit, ctrl->__hfi_port, addr_index,
				    &ep->subnet, &ep->addr,
				    NULL, &ep->gid) == -1) {
			err =
				psm3_handle_error(NULL, PSM2_EP_DEVICE_FAILURE,
						  "Can't get HFI GID in psm3_ep_open: is SMA running?");
			goto bail;
		}
		ep->unit_id = ctrl->__hfi_unit;
		ep->portnum = ctrl->__hfi_port;
		ep->addr_index = addr_index;
		ep->dev_name = psm3_sysfs_unit_dev_name(ep->unit_id);

		/* Endpoint out_sl contains the default SL to use for this endpoint. */
		/* Get the MTU for this SL. */
		int sc;
		if ((sc=psm3_gen1_get_port_sl2sc(ep->unit_id,
				       ctrl->__hfi_port,
				       ep->out_sl)) < 0) {
			sc = PSMI_SC_DEFAULT;
		}
		int vl;
		if ((vl = psm3_gen1_get_port_sc2vl(ep->unit_id,
					     ctrl->__hfi_port,
					     sc)) < 0) {
			vl = PSMI_VL_DEFAULT;
		}
		if (sc == PSMI_SC_ADMIN ||
		    vl == PSMI_VL_ADMIN) {
			err = psm3_handle_error(NULL, PSM2_INTERNAL_ERR,
						"Invalid sl: %d, please specify correct sl via PSM3_NIC_SL",
						ep->out_sl);
			goto bail;
		}

		if ((ep->mtu = psm3_gen1_get_port_vl2mtu(ep->unit_id,
						   ctrl->__hfi_port,
						   vl)) < 0) {
			err =
				psm3_handle_error(NULL, PSM2_EP_DEVICE_FAILURE,
						  "Can't get MTU for VL %d",
						  vl);
			goto bail;
		}

		get_psm_gen1_hi()->phi.params.cap_mask |=
			psm3_gen1_get_cap_mask(ctrl->ctxt_info.runtime_flags)
			| PSM_HAL_CAP_MERGED_TID_CTRLS
			| PSM_HAL_CAP_RSM_FECN_SUPP;

		int driver_major = psm3_gen1_get_user_major_version();
		int driver_minor = psm3_gen1_get_user_minor_version();

		if ((driver_major > 6) ||
		    ((driver_major == 6) &&
		     (driver_minor >= 3)))
		{
			get_psm_gen1_hi()->phi.params.cap_mask |= PSM_HAL_CAP_DMA_HSUPP_FOR_32B_MSGS;
		}

		get_psm_gen1_hi()->hfp_private.sdmahdr_req_size = HFI_SDMA_HDR_SIZE;

		if (psm3_gen1_check_non_dw_mul_sdma())
			get_psm_gen1_hi()->phi.params.cap_mask |= PSM_HAL_CAP_NON_DW_MULTIPLE_MSG_SIZE;
		/* The dma_rtail member is: 1 when the HFI1_CAP_DMA_RTAIL bit is     set.
					    0 when the HFI1_CAP_DMA_RTAIL bit is NOT set. */
		get_psm_gen1_hi()->hfp_private.dma_rtail = 0 != (HFI1_CAP_DMA_RTAIL & ctrl->ctxt_info.runtime_flags);

		psm_ctxt->psm_hw_ctxt = pc_private;
		if (pc_private->user_info.subctxt_cnt > 0)
			psm3_gen1_divvy_shared_mem_ptrs(pc_private,
					      psm_ctxt,
					      &ctrl->base_info);

		/* Initialize all of the cl q's. */

		get_psm_gen1_hi()->hfp_private.hdrq_rhf_off = (ctrl->ctxt_info.rcvhdrq_entsize - 8) >> BYTE2DWORD_SHIFT;

		/* The following guard exists to workaround a critical issue flagged by KW to prevent
		   subscripting past the end of the cl_qs[] array in the following for () loop. */
		if (pc_private->user_info.subctxt_cnt <= HFI1_MAX_SHARED_CTXTS)
		{
			/* Here, we are initializing only the rx hdrq rhf seq for all subcontext
			   cl q's: */
			for (i=PSM3_GEN1_CL_Q_RX_HDR_Q_SC_0; i <
				     PSM3_GEN1_GET_SC_CL_Q_RX_HDR_Q(pc_private->user_info.subctxt_cnt); i += 2)
			{
				psm3_gen1_cl_q_t *pcl_q = &(pc_private->cl_qs[i]);

				pcl_q->hdr_qe.p_rx_hdrq_rhf_seq = &pcl_q->hdr_qe.rx_hdrq_rhf_seq;
				if (get_psm_gen1_hi()->hfp_private.dma_rtail)
					pcl_q->hdr_qe.rx_hdrq_rhf_seq = 0;
				else
					pcl_q->hdr_qe.rx_hdrq_rhf_seq = 1;
			}
		}
		/* Next, initialize the hw rx hdr q and egr buff q: */
		{
			/* base address of user registers */
			volatile uint64_t *uregbase = (volatile uint64_t *)(uintptr_t) (ctrl->base_info.user_regbase);
			/* hw rx hdr q: */
			psm3_gen1_cl_q_t *pcl_q = &(pc_private->cl_qs[PSM3_GEN1_CL_Q_RX_HDR_Q]);
			pcl_q->cl_q_head = (volatile uint64_t *)&(uregbase[ur_rcvhdrhead]);
			pcl_q->cl_q_tail = (volatile uint64_t *)&(uregbase[ur_rcvhdrtail]);
			pcl_q->hdr_qe.hdrq_base_addr       = (uint32_t *) (ctrl->base_info.rcvhdr_bufbase);

			/* Initialize the ptr to the rx hdrq rhf seq: */
			if (pc_private->user_info.subctxt_cnt > 0)
				/* During sharing of a context, the h/w hdrq rhf_seq is placed in shared memory and is shared
				   by all subcontexts: */
				pcl_q->hdr_qe.p_rx_hdrq_rhf_seq    = &pc_private->hwcontext_ctrl->rx_hdrq_rhf_seq;
			else
				pcl_q->hdr_qe.p_rx_hdrq_rhf_seq    = &pcl_q->hdr_qe.rx_hdrq_rhf_seq;

			if (get_psm_gen1_hi()->hfp_private.dma_rtail)
				*pcl_q->hdr_qe.p_rx_hdrq_rhf_seq = 0;
			else
				*pcl_q->hdr_qe.p_rx_hdrq_rhf_seq = 1;
			/* hw egr buff q: */
			pcl_q = &pc_private->cl_qs[PSM3_GEN1_CL_Q_RX_EGR_Q];
			pcl_q->cl_q_head = (volatile uint64_t *)&(uregbase[ur_rcvegrindexhead]);
			pcl_q->cl_q_tail = (volatile uint64_t *)&(uregbase[ur_rcvegrindextail]);
			pcl_q->egr_buffs = psm3_ips_recvq_egrbuf_table_alloc(ep,
									  (void*)(ctrl->base_info.rcvegr_bufbase),
									  ctrl->ctxt_info.egrtids,
									  ctrl->ctxt_info.rcvegr_size);
		}
		/* Next, initialize the subcontext's rx hdr q and egr buff q: */
		for (i=0; i < pc_private->user_info.subctxt_cnt;i++)
		{
			/* Subcontexts mimic the HW registers but use different addresses
			 * to avoid cache contention. */
			volatile uint64_t *subcontext_uregbase;
			uint32_t *rcv_hdr, *rcv_egr;
			unsigned hdrsize, egrsize;
			unsigned pagesize = getpagesize();
			uint32_t subcontext = i;
			unsigned i = pagesize - 1;
			hdrsize =
				(ctrl->ctxt_info.rcvhdrq_cnt * ctrl->ctxt_info.rcvhdrq_entsize + i) & ~i;
			egrsize =
				(ctrl->ctxt_info.egrtids * ctrl->ctxt_info.rcvegr_size + i) & ~i;

			subcontext_uregbase = (uint64_t *)
			  (((uintptr_t) (ctrl->base_info.subctxt_uregbase)) +
			   (sizeof(struct gen1_ips_subcontext_ureg) * subcontext));
			{
				struct gen1_ips_subcontext_ureg *pscureg = (struct gen1_ips_subcontext_ureg *)subcontext_uregbase;

				if (subcontext == ctrl->ctxt_info.subctxt)
				{
					memset(pscureg, 0, sizeof(*pscureg));
					if (get_psm_gen1_hi()->hfp_private.dma_rtail)
						pscureg->writeq_state.hdrq_rhf_seq = 0;
					else
						pscureg->writeq_state.hdrq_rhf_seq = 1;
				}
			}

			rcv_hdr = (uint32_t *)
			  (((uintptr_t) (ctrl->base_info.subctxt_rcvhdrbuf)) +
			   (hdrsize * subcontext));
			rcv_egr = (uint32_t *)
				(((uintptr_t) ctrl->base_info.subctxt_rcvegrbuf +
				  (egrsize * subcontext)));

			/* rx hdr q: */
			psm3_gen1_cl_q_t *pcl_q = &(pc_private->cl_qs[PSM3_GEN1_GET_SC_CL_Q_RX_HDR_Q(subcontext)]);
			pcl_q->hdr_qe.hdrq_base_addr = rcv_hdr;
			pcl_q->cl_q_head = (volatile uint64_t *)&subcontext_uregbase[ur_rcvhdrhead * 8];
			pcl_q->cl_q_tail = (volatile uint64_t *)&subcontext_uregbase[ur_rcvhdrtail * 8];

			/* egr q: */
			pcl_q = &(pc_private->cl_qs[PSM3_GEN1_GET_SC_CL_Q_RX_EGR_Q(subcontext)]);
			pcl_q->cl_q_head = (volatile uint64_t *)&subcontext_uregbase[ur_rcvegrindexhead * 8];
			pcl_q->cl_q_tail = (volatile uint64_t *)&subcontext_uregbase[ur_rcvegrindextail * 8];
			pcl_q->egr_buffs = psm3_ips_recvq_egrbuf_table_alloc(
				ep,
				(void*)rcv_egr,
				ctrl->ctxt_info.egrtids,
				ctrl->ctxt_info.rcvegr_size);
		}

		/* Construct epid for this Endpoint */
		ep->epid = psm_ctxt->epid = psm3_epid_pack_ips(lid, ctrl->ctxt_info.ctxt,
					ctrl->ctxt_info.subctxt, ep->unit_id,
					ep->addr);

		_HFI_VDBG("construct epid v%u: %s: lid %d ctxt %d subctxt %d hcatype %d addr %s mtu %d\n",
		     ep->addr.fmt,
		     psm3_epid_fmt_internal(ep->epid, 0), lid,
		     ctrl->ctxt_info.ctxt, ctrl->ctxt_info.subctxt,
		     PSMI_HFI_TYPE_OPA1,
		     psm3_naddr128_fmt(ep->addr, 1),  ep->mtu);
	}
	ep->wiremode = 0; // Only 1 mode for OPA
	ep->context.ep = ep;
	return PSM_HAL_ERROR_OK;

	/* no failure possible after alloc egr_buffs */
	//psm3_gen1_free_egr_buffs(pc_private);
bail:
	/* Unmap the HFI memory mapped by userinit_internal */
	psm3_gen1_unmap_hfi_mem(pc_private);
bail_fd:
	if (fd >0) close(fd);
	if (pc_private) {
		if (pc_private->ctrl) free(pc_private->ctrl);
		psmi_free(pc_private);
		psm_ctxt->psm_hw_ctxt = NULL;
	}

	return -PSM_HAL_ERROR_GENERAL_ERROR;
}

static PSMI_HAL_INLINE int psm3_hfp_gen1_get_port_index2pkey(psm2_ep_t ep, int index)
{
	return psm3_gen1_get_port_index2pkey(ep->unit_id, ep->portnum, index);
}

static PSMI_HAL_INLINE int psm3_hfp_gen1_set_pkey(psmi_hal_hw_context ctxt, uint16_t pkey)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	return psm3_gen1_set_pkey(psm_hw_ctxt->ctrl, pkey);
}

/* Tell the driver to change the way packets can generate interrupts.

 HFI1_POLL_TYPE_URGENT: Generate interrupt only when send with
			IPS_SEND_FLAG_INTR (HFI_KPF_INTR)
 HFI1_POLL_TYPE_ANYRCV: wakeup on any rcv packet (when polled on). [not used]

 PSM: Uses TYPE_URGENT in ips protocol
*/

static PSMI_HAL_INLINE int psm3_hfp_gen1_poll_type(uint16_t poll_type, psm2_ep_t ep)
{
	if (poll_type == PSMI_HAL_POLL_TYPE_URGENT)
		poll_type = HFI1_POLL_TYPE_URGENT;
	else
		poll_type = 0;
	hfp_gen1_pc_private *psm_hw_ctxt = ep->context.psm_hw_ctxt;
	return psm3_gen1_poll_type(psm_hw_ctxt->ctrl, poll_type);
}

// initialize HAL specific parts of ptl_ips
// This is called after most of the generic aspects have been initialized
// so we can use ptl->ep, ptl->ctl, etc as needed
// However it is called prior to ips_proto_init.  ips_proto_init requires some
// ips_ptl items such as ptl->spioc
static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_ips_ptl_init_pre_proto_init(struct ptl_ips *ptl)
{
	return psm3_gen1_ips_ptl_init_pre_proto_init(ptl);
}

// initialize HAL specific parts of ptl_ips
// This is called after after ips_proto_init and after most of the generic
// aspects of ips_ptl have been initialized
// so we can use ptl->ep and ptl->proto as needed
static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_ips_ptl_init_post_proto_init(struct ptl_ips *ptl)
{
	return psm3_gen1_ips_ptl_init_post_proto_init(ptl);
}

// finalize HAL specific parts of ptl_ips
// This is called before the generic aspects have been finalized
// but after ips_proto has been finalized
// so we can use ptl->ep as needed
static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_ips_ptl_fini(struct ptl_ips *ptl)
{
	return psm3_gen1_ips_ptl_fini(ptl);
}

// initialize HAL specific details in ips_proto.
// called after many of ips_proto parameters parsed and initialized
static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_ips_proto_init(
				struct ips_proto *proto, uint32_t cksum_sz)
{
	psm2_error_t err = PSM2_OK;
	hfp_gen1_pc_private *psm_hw_ctxt = proto->ep->context.psm_hw_ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;
	union psmi_envvar_val env_mtu;

	// defaults for SDMA thresholds.  These may be updated when
	// PSM3_* env for SDMA are parsed later in psm3_ips_proto_init.
	if(psm3_cpu_model == CPUID_MODEL_PHI_GEN2 || psm3_cpu_model == CPUID_MODEL_PHI_GEN2M)
	{
		proto->iovec_thresh_eager = 65536;
		proto->iovec_thresh_eager_blocking = 200000;
	} else {
		proto->iovec_thresh_eager = 16384;
		proto->iovec_thresh_eager_blocking = 34000;
	}

	// set basic HW info, some of which is used for dispersive routing hash
	proto->epinfo.ep_baseqp = ctrl->base_info.bthqp;
	proto->epinfo.ep_context = ctrl->ctxt_info.ctxt; /* "real" context */
	proto->epinfo.ep_hash = proto->epinfo.ep_context;
	proto->epinfo.ep_subcontext = ctrl->ctxt_info.subctxt;
	proto->epinfo.ep_hfi_type = PSMI_HFI_TYPE_OPA1;
	proto->epinfo.ep_jkey = psm3_hfp_gen1_get_jkey(proto->ep);

	// at this point ep->mtu is our PSM payload HW capability found during
	// open (not yet adjusted for optional cksum_sz)

	/* See if user specifies a lower MTU to use */
	if (!psm3_getenv("PSM3_MTU",
		"Upper bound on packet MTU (<=0 uses port MTU): 1-7,256,512,1024,2048,4096,8192,10240]",
	     PSMI_ENVVAR_LEVEL_USER, PSMI_ENVVAR_TYPE_INT,
	     (union psmi_envvar_val)-1, &env_mtu)) {
		if (env_mtu.e_int >= OPA_MTU_MIN && env_mtu.e_int <= OPA_MTU_MAX) //enum
			env_mtu.e_int = opa_mtu_enum_to_int((enum opa_mtu)env_mtu.e_int);
		else if (env_mtu.e_int < OPA_MTU_MIN) // pick default
			env_mtu.e_int = 8192;
		else // wash through enum to force round up to next valid MTU
			env_mtu.e_int = opa_mtu_enum_to_int(opa_mtu_int_to_enum(env_mtu.e_int));
		if (proto->ep->mtu > env_mtu.e_int)
			proto->ep->mtu = env_mtu.e_int;
	}
	/* allow space for optional software managed checksum (for debug) */
	proto->ep->mtu -= cksum_sz;
	// ep->mtu is our final choice of local PSM payload we can support
	proto->epinfo.ep_mtu = proto->ep->mtu;

#ifdef PSM_BYTE_FLOW_CREDITS
	// for OPA we let flow_credits be the control
	proto->flow_credit_bytes = proto->ep->mtu * proto->flow_credits;
#endif
	/*
	 * The PIO size should not include the ICRC because it is
	 * stripped by HW before delivering to receiving buffer.
	 * We decide to use minimum 2 PIO buffers so that PSM has
	 * turn-around time to do PIO transfer. Each credit is a
	 * block of 64 bytes. Also PIO buffer size must not be
	 * bigger than MTU.
	 */
	proto->epinfo.ep_piosize = psmi_hal_get_pio_size(psm_hw_ctxt) - cksum_sz;
	proto->epinfo.ep_piosize =
	    min(proto->epinfo.ep_piosize, proto->epinfo.ep_mtu);

	/* Keep PIO as multiple of cache line size */
	if (proto->epinfo.ep_piosize > PSM_CACHE_LINE_BYTES)
		proto->epinfo.ep_piosize &= ~(PSM_CACHE_LINE_BYTES - 1);

	/* Save back to hfi level. */
	ctrl->__hfi_mtusize = proto->epinfo.ep_mtu;
	ctrl->__hfi_piosize = proto->epinfo.ep_piosize;

	/* sdma queue size */
	proto->sdma_queue_size = psm3_gen1_get_sdma_ring_size(psm_hw_ctxt);
	/* don't use the last slot */
	if (proto->sdma_queue_size > 8) {
		/* configure sdma_avail_counter */
		union psmi_envvar_val env_sdma_avail;
		int tmp_queue_size = 8;

		psm3_getenv("PSM3_MAX_PENDING_SDMA_REQS",
			"PSM maximum pending SDMA requests",
			PSMI_ENVVAR_LEVEL_USER, PSMI_ENVVAR_TYPE_INT,
			(union psmi_envvar_val) tmp_queue_size,
			&env_sdma_avail);

		if ((env_sdma_avail.e_int < 8) || (env_sdma_avail.e_int > (proto->sdma_queue_size - 1)))
			proto->sdma_avail_counter = 8;
		else
			proto->sdma_avail_counter = env_sdma_avail.e_int;
	} else {
		err = PSM2_PARAM_ERR;
		goto fail;
	}


	proto->sdma_fill_index = 0;
	proto->sdma_done_index = 0;
	proto->sdma_scb_queue = (struct ips_scb **)
		psmi_calloc(proto->ep, UNDEFINED,
		proto->sdma_queue_size, sizeof(struct ips_scb *));
	if (proto->sdma_scb_queue == NULL) {
		err = PSM2_NO_MEMORY;
		goto fail;
	}

	/*
	 * Pre-calculate the PSN mask to support 24 or 31 bit PSN.
	 */
	if (psmi_hal_has_cap(PSM_HAL_CAP_EXTENDED_PSN)) {
		proto->psn_mask = 0x7FFFFFFF;
	} else {
		proto->psn_mask = 0xFFFFFF;
	}
	/* 12 bit pktlen (limit to <= 4095 32 bit words per packet */
	proto->pktlen_mask = 0xFFF;
fail:
	return err;
}

// Fetch current link state to update linkinfo fields in ips_proto:
// 	ep_base_lid, ep_lmc, ep_link_rate, QoS tables, CCA tables
// These are all fields which can change during a link bounce.
// Note "active" state is not adjusted as on link down PSM will wait for
// the link to become usable again so it's always a viable/active device
// afer initial PSM startup has selected devices.
// Called during initialization of ips_proto during ibta_init as well
// as during a link bounce.
// TBD - may be able to call this from HAL ips_proto_init as well as
// directly within HAL event processing, in which case this could
// be completely internal to HAL and not exposed in HAL API
static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_ips_proto_update_linkinfo(
				struct ips_proto *proto)
{
	return psm3_gen1_ptl_ips_update_linkinfo(proto);
}

// Indicate if all underlying connections are now established
// (eg. RV connections)
// return:
//	0 - not yet connected
//	1 - connected (or nothing extra needed)
//	-1 - failure to check or connect (errno is status)
//		EIO is connection error other values are more serious
//		(invalid call, etc)
static PSMI_HAL_INLINE int psm3_hfp_gen1_ips_fully_connected(ips_epaddr_t *ipsaddr)
{
	return 1;
}

/* handle HAL specific connection processing as part of processing an
 * inbound PSM connect Request or Reply when connection not yet established
 * save the negotiated parameters
 */
static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_ips_ipsaddr_set_req_params(
				struct ips_proto *proto,
				ips_epaddr_t *ipsaddr,
				const struct ips_connect_reqrep *req)
{
	return PSM2_OK;
}

/* handle HAL specific connection processing as part of processing an
 * inbound PSM connect Reply which completes establishment of on outgoing
 * connection.
 */
static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_ips_ipsaddr_process_connect_reply(
				struct ips_proto *proto,
				ips_epaddr_t *ipsaddr,
				const struct ips_connect_reqrep *req)
{
	return PSM2_OK;
}

/* build HAL specific portion of an outbound PSM connect message
 * for PSM Connect or Disconnect Request or Reply
 */
static PSMI_HAL_INLINE void psm3_hfp_gen1_ips_proto_build_connect_message(
			struct ips_proto *proto,
			ips_epaddr_t *ipsaddr, uint8_t opcode,
			struct ips_connect_reqrep *req)
{
	switch (opcode) {
	case OPCODE_CONNECT_REPLY:
	case OPCODE_CONNECT_REQUEST:
		memset(req->hal_pad, 0, sizeof(req->hal_pad));
		break;
	case OPCODE_DISCONNECT_REQUEST:
	case OPCODE_DISCONNECT_REPLY:
		// placeholder, but typically nothing to be done
		// as the ips_connect_hdr is sufficient
		break;
	default:
		psmi_assert_always(0);
		break;
	}
}

/* handle HAL specific ipsaddr initialization for addressing, including
 * parts of ipsaddr needed for path record query
 * For ipsaddr created just for a disconnect, ips_ipsaddr_init_connections
 * is not called. In which case ips_ipsaddr_init_addressing and ips_flow_init
 * need to do what is needed to allow spio_transfer_frame to send the
 * disconnect control packet.
 */
static PSMI_HAL_INLINE void psm3_hfp_gen1_ips_ipsaddr_init_addressing(
			struct ips_proto *proto, psm2_epid_t epid,
			ips_epaddr_t *ipsaddr, uint16_t *lidp
			)
{
	/* Actual context of peer */
	ipsaddr->opa.context = psm3_epid_context(epid);
	/* Subcontext */
	ipsaddr->opa.subcontext = psm3_epid_subcontext(epid);
	ipsaddr->hash = ipsaddr->opa.context;

	// for OPA, just need lid
	*lidp = psm3_epid_lid(epid);
}

/* handle HAL specific ipsaddr initialization for any HAL specific connections
 * underlying the ipsaddr (RC QPs, TCP sockets, etc)
 * This is not called for an ipsaddr created just for a disconnect.  In which
 * case ips_ipsaddr_init_addressing and ips_flow_init need to do what is
 * needed to allow spio_transfer_frame to send the disconnect control packet.
 */
static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_ips_ipsaddr_init_connections(
			struct ips_proto *proto, psm2_epid_t epid,
			ips_epaddr_t *ipsaddr)
{
	return PSM2_OK;
}

/* handle HAL specific ipsaddr free for any HAL specific information
 * in ipsaddr (from ipsaddr_init_*, set_req_params, etc
 */
static PSMI_HAL_INLINE void psm3_hfp_gen1_ips_ipsaddr_free(
			ips_epaddr_t *ipsaddr, struct ips_proto *proto)
{
}

/* handle HAL specific ips_flow initialization
 */
static PSMI_HAL_INLINE void psm3_hfp_gen1_ips_flow_init(
			struct ips_flow *flow, struct ips_proto *proto)
{
	if (flow->transfer == PSM_TRANSFER_PIO) {
		flow->flush = psm3_ips_proto_flow_flush_pio;
	} else {
		flow->flush = ips_proto_flow_flush_dma;
	}

	/* if PIO, need to consider local pio buffer size */
	if (flow->transfer == PSM_TRANSFER_PIO) {
		flow->frag_size = min(flow->frag_size, proto->epinfo.ep_piosize);
		_HFI_CONNDBG("[ipsaddr=%p] PIO flow->frag_size: %u = min("
			"proto->epinfo.ep_mtu(%u), flow->path->pr_mtu(%u), proto->epinfo.ep_piosize(%u))\n",
			flow->ipsaddr, flow->frag_size, proto->epinfo.ep_mtu,
			flow->path->pr_mtu, proto->epinfo.ep_piosize);
	} else {
		_HFI_CONNDBG("[ipsaddr=%p] SDMA flow->frag_size: %u = min("
			"proto->epinfo.ep_mtu(%u), flow->path->pr_mtu(%u))\n",
			flow->ipsaddr, flow->frag_size, proto->epinfo.ep_mtu,
			flow->path->pr_mtu);
	}

	flow->cca_ooo_pkts = 0;
}

/* handle HAL specific connection processing as part of processing an
 * outbound PSM disconnect Request or Reply or an inbound disconnect request
 */
static PSMI_HAL_INLINE void psm3_hfp_gen1_ips_ipsaddr_disconnect(
			struct ips_proto *proto, ips_epaddr_t *ipsaddr)
{
}

/* Handle HAL specific initialization of ibta path record query, CCA
 * and dispersive routing
 */
static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_ips_ibta_init(
				struct ips_proto *proto)
{
	psm2_error_t err = PSM2_OK;
	union psmi_envvar_val psm_path_policy;
	union psmi_envvar_val disable_cca;
	union psmi_envvar_val cca_prescan;

	/* Get the path selection policy */
	psm3_getenv("PSM3_PATH_SELECTION",
		    "Policy to use if multiple paths are available between endpoints. Options are adaptive, static_src, static_dest, static_base. Default is adaptive.",
		    PSMI_ENVVAR_LEVEL_USER, PSMI_ENVVAR_TYPE_STR,
		    (union psmi_envvar_val)"adaptive", &psm_path_policy);

	if (!strcasecmp((const char *)psm_path_policy.e_str, "adaptive"))
		proto->flags |= IPS_PROTO_FLAG_PPOLICY_ADAPTIVE;
	else if (!strcasecmp((const char *)psm_path_policy.e_str, "static_src"))
		proto->flags |= IPS_PROTO_FLAG_PPOLICY_STATIC_SRC;
	else if (!strcasecmp
		 ((const char *)psm_path_policy.e_str, "static_dest"))
		proto->flags |= IPS_PROTO_FLAG_PPOLICY_STATIC_DST;
	else if (!strcasecmp
		 ((const char *)psm_path_policy.e_str, "static_base"))
		proto->flags |= IPS_PROTO_FLAG_PPOLICY_STATIC_BASE;

	if (proto->flags & IPS_PROTO_FLAG_PPOLICY_ADAPTIVE)
		_HFI_PRDBG("Using adaptive path selection.\n");
	if (proto->flags & IPS_PROTO_FLAG_PPOLICY_STATIC_SRC)
		_HFI_PRDBG("Static path selection: Src Context\n");
	if (proto->flags & IPS_PROTO_FLAG_PPOLICY_STATIC_DST)
		_HFI_PRDBG("Static path selection: Dest Context\n");
	if (proto->flags & IPS_PROTO_FLAG_PPOLICY_STATIC_BASE)
		_HFI_PRDBG("Static path selection: Base LID\n");

	psm3_getenv("PSM3_DISABLE_CCA",
		    "Disable use of Congestion Control Architecture (CCA) [enabled] ",
		    PSMI_ENVVAR_LEVEL_USER, PSMI_ENVVAR_TYPE_UINT,
		    (union psmi_envvar_val)0, &disable_cca);
	if (disable_cca.e_uint)
		_HFI_CCADBG("CCA is disabled for congestion control.\n");
	else {
		int i;
		char ccabuf[256];
		uint8_t *p;

		/* Start out by turning on both styles of congestion control.
		 * Later, we will eliminate the correct one. */
		proto->flags |= IPS_PROTO_FLAG_CCA | IPS_PROTO_FLAG_CC_REPL_BECN;
/*
 * If user set any environment variable, use self CCA.
 */
		if (getenv("PSM3_CCTI_INCREMENT") || getenv("PSM3_CCTI_TIMER")
		    || getenv("PSM3_CCTI_TABLE_SIZE")) {
			goto disablecca;
		}

		psm3_getenv("PSM3_CCA_PRESCAN",
                    "Enable Congestion Control Prescanning (disabled by default) ",
                    PSMI_ENVVAR_LEVEL_USER, PSMI_ENVVAR_TYPE_UINT,
                    (union psmi_envvar_val)0, &cca_prescan);

		if (cca_prescan.e_uint)
			proto->flags |= IPS_PROTO_FLAG_CCA_PRESCAN;

/*
 * Check qib driver CCA setting, and try to use it if available.
 * Fall to self CCA setting if errors.
 */
		i = psm3_gen1_get_cc_settings_bin(proto->ep->unit_id,
			proto->ep->portnum, ccabuf, sizeof(ccabuf));

		if (i <= 0) {
			goto disablecca;
		}
		p = (uint8_t *) ccabuf;
		memcpy(&proto->ccti_ctrlmap, p, 4);
		p += 4;
		memcpy(&proto->ccti_portctrl, p, 2);
		p += 2;
		for (i = 0; i < 32; i++) {
			proto->cace[i].ccti_increase = *p;
			p++;
			/* skip reserved u8 */
			p++;
			memcpy(&proto->cace[i].ccti_timer_cycles, p, 2);
			p += 2;
			proto->cace[i].ccti_timer_cycles =
			    us_2_cycles(proto->cace[i].ccti_timer_cycles);
			proto->cace[i].ccti_threshold = *p;
			p++;
			proto->cace[i].ccti_min = *p;
			p++;
		}

		i = psm3_gen1_get_cc_table_bin(proto->ep->unit_id, proto->ep->portnum,
					      &proto->cct);
		if (i < 0) {
			err = PSM2_NO_MEMORY;
			goto fail;
		} else if (i == 0) {
			goto disablecca;
		}
		proto->ccti_limit = i;
		proto->ccti_size = proto->ccti_limit + 1;

		_HFI_CCADBG("ccti_limit = %d\n", (int) proto->ccti_limit);
		for (i = 0; i < proto->ccti_limit; i++)
			_HFI_CCADBG("cct[%d] = 0x%04x\n", i, (int) proto->cct[i]);

		/* Note, here, we are leaving CC style(s):
		   (IPS_PROTO_FLAG_CCA | IPS_PROTO_FLAG_CCA_PRESCAN) */
		proto->flags &= ~IPS_PROTO_FLAG_CC_REPL_BECN;
		goto finishcca;

/*
 * Disable CCA.
 */
disablecca:
		/* Note, here, we are leaving CC style:
		   IPS_PROTO_FLAG_CC_REPL_BECN */
		proto->flags &= ~(IPS_PROTO_FLAG_CCA | IPS_PROTO_FLAG_CCA_PRESCAN);
	}

finishcca:
fail:
	return err;

}

/* Handle HAL specific initialization of an ips_path_rec
 * as part of fetching or hand building a path record.
 * Responsible for all fields in the HAL specific union and any tweaks to
 * other fields which may be HAL specific (such as pr_mtu).
 * response is only provided when we are building a ips_path_rec from a
 * fetched ibta_path_rec.  Otherwise we are building it solely based on
 * our own end point and what our caller knows from the EPID.
 */
static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_ips_path_rec_init(
				struct ips_proto *proto,
				struct ips_path_rec *path_rec,
				struct _ibta_path_rec *response)
{
	psm2_error_t err = PSM2_OK;
	/* Setup CCA parameters for path */
	if (!(proto->ccti_ctrlmap & (1 << path_rec->pr_sl))) {
		_HFI_CCADBG("No CCA for sl %d, disable CCA\n",
			    path_rec->pr_sl);
		proto->flags &= ~IPS_PROTO_FLAG_CCA;
		proto->flags &= ~IPS_PROTO_FLAG_CCA_PRESCAN;
	}
	if (!psmi_hal_has_cap(PSM_HAL_CAP_STATIC_RATE_CTRL)) {
		_HFI_CCADBG("No Static-Rate-Control, disable CCA\n");
		proto->flags &= ~IPS_PROTO_FLAG_CCA;
		proto->flags &= ~IPS_PROTO_FLAG_CCA_PRESCAN;
	}

	path_rec->opa.pr_proto = proto;
	path_rec->opa.pr_ccti = proto->cace[path_rec->pr_sl].ccti_min;
	path_rec->opa.pr_timer_cca = NULL;

	/* Determine active IPD for path. Is max of static rate and CCT table */
	if (!(proto->flags & IPS_PROTO_FLAG_CCA)) {
		_HFI_CCADBG("No IPS_PROTO_FLAG_CCA\n");

		path_rec->opa.pr_active_ipd = 0;
		path_rec->opa.pr_cca_divisor = 0;

		_HFI_CCADBG("pr_active_ipd = %d\n", (int) path_rec->opa.pr_active_ipd);
		_HFI_CCADBG("pr_cca_divisor = %d\n", (int) path_rec->opa.pr_cca_divisor);
	} else if ((path_rec->pr_static_ipd) &&
		    ((path_rec->pr_static_ipd + 1) >
		     (proto->cct[path_rec->opa.pr_ccti] & CCA_IPD_MASK))) {
		_HFI_CCADBG("IPS_PROTO_FLAG_CCA set, Setting pr_active_ipd.\n");

		path_rec->opa.pr_active_ipd = path_rec->pr_static_ipd + 1;
		path_rec->opa.pr_cca_divisor = 0;

		_HFI_CCADBG("pr_active_ipd = %d\n", (int) path_rec->opa.pr_active_ipd);
		_HFI_CCADBG("pr_cca_divisor = %d\n", (int) path_rec->opa.pr_cca_divisor);
	} else {
		/* Pick it from the CCT table */
		_HFI_CCADBG("Picking up active IPD from CCT table, index %d, value 0x%x\n",
			    (int) path_rec->opa.pr_ccti, (int) proto->cct[path_rec->opa.pr_ccti]);

		path_rec->opa.pr_active_ipd =
		    proto->cct[path_rec->opa.pr_ccti] & CCA_IPD_MASK;
		path_rec->opa.pr_cca_divisor =
		    proto->cct[path_rec->opa.pr_ccti] >> CCA_DIVISOR_SHIFT;

		_HFI_CCADBG("pr_active_ipd = %d\n", (int) path_rec->opa.pr_active_ipd);
		_HFI_CCADBG("pr_cca_divisor = %d\n", (int) path_rec->opa.pr_cca_divisor);
	}
	return err;
}

static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_ips_ptl_pollintr(
		psm2_ep_t ep, struct ips_recvhdrq *recvq,
		int fd_pipe, int next_timeout,
		uint64_t *pollok, uint64_t *pollcyc)
{
	return psm3_gen1_ips_ptl_pollintr(ep, recvq, fd_pipe,
					 next_timeout, pollok, pollcyc);
}

#ifdef PSM_CUDA
static PSMI_HAL_INLINE void psm3_hfp_gen1_gdr_close(void)
{
	psm3_gen1_gdr_close();
}
static PSMI_HAL_INLINE void* psm3_hfp_gen1_gdr_convert_gpu_to_host_addr(unsigned long buf,
                                size_t size, int flags, psm2_ep_t ep)
{
	return psm3_gen1_gdr_convert_gpu_to_host_addr(buf, size, flags, ep);
}
#endif /* PSM_CUDA */

static PSMI_HAL_INLINE int psm3_hfp_gen1_free_tid(psmi_hal_hw_context ctxt, uint64_t tidlist, uint32_t tidcnt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	return psm3_gen1_free_tid(psm_hw_ctxt->ctrl, tidlist, tidcnt);
}

static PSMI_HAL_INLINE int psm3_hfp_gen1_get_tidcache_invalidation(psmi_hal_hw_context ctxt, uint64_t tidlist, uint32_t *tidcnt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	return psm3_gen1_get_invalidation(psm_hw_ctxt->ctrl, tidlist, tidcnt);
}

static PSMI_HAL_INLINE int psm3_hfp_gen1_update_tid(psmi_hal_hw_context ctxt, uint64_t vaddr, uint32_t *length,
					       uint64_t tidlist, uint32_t *tidcnt, uint16_t flags)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;

	return psm3_gen1_update_tid(psm_hw_ctxt->ctrl, vaddr, length, tidlist, tidcnt, flags);
}

static PSMI_HAL_INLINE int psm3_hfp_gen1_get_hfi_event_bits(uint64_t *event_bits, psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;
	uint64_t *pevents_mask = (uint64_t *)ctrl->base_info.events_bufbase;
	uint64_t events_mask   = *pevents_mask;
	uint64_t hal_hfi_event_bits = 0;
	int i;

	if (!events_mask)
	{
		*event_bits = 0;
		return PSM_HAL_ERROR_OK;
	}

	/* Encode hfi1_events as HAL event codes here */
	for (i = 0; i < sizeof(hfi1_events_map)/sizeof(hfi1_events_map[0]); i++)
	{
		if (events_mask & hfi1_events_map[i].hfi1_event_bit)
			hal_hfi_event_bits |=
				hfi1_events_map[i].psmi_hal_hfi_event_bit;
	}

	*event_bits = hal_hfi_event_bits;

	return PSM_HAL_ERROR_OK;
}

static PSMI_HAL_INLINE int psm3_hfp_gen1_tidflow_set_entry(uint32_t flowid, uint32_t genval, uint32_t seqnum, psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;

	psm3_gen1_tidflow_set_entry(ctrl, flowid, genval, seqnum);
	return PSM_HAL_ERROR_OK;
}

static PSMI_HAL_INLINE int psm3_hfp_gen1_tidflow_reset(psmi_hal_hw_context ctxt, uint32_t flowid, uint32_t genval, uint32_t seqnum)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;

	psm3_gen1_tidflow_reset(ctrl, flowid, genval, seqnum);
	return PSM_HAL_ERROR_OK;
}

static PSMI_HAL_INLINE int psm3_hfp_gen1_tidflow_get(uint32_t flowid, uint64_t *ptf, psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;

	*ptf = psm3_gen1_tidflow_get(ctrl, flowid);
	return PSM_HAL_ERROR_OK;
}

static PSMI_HAL_INLINE int psm3_hfp_gen1_tidflow_get_hw(uint32_t flowid, uint64_t *ptf, psmi_hal_hw_context ctxt)
{
	return psm3_hfp_gen1_tidflow_get(flowid, ptf, ctxt);
}

static PSMI_HAL_INLINE int psm3_hfp_gen1_tidflow_get_seqnum(uint64_t val, uint32_t *pseqn)
{
	*pseqn = psm3_gen1_tidflow_get_seqnum(val);
	return PSM_HAL_ERROR_OK;
}

static PSMI_HAL_INLINE int psm3_hfp_gen1_tidflow_check_update_pkt_seq(void *vpprotoexp
							      /* actually a:
								 struct ips_protoexp *protoexp */,
							      psmi_seqnum_t sequence_num,
							      void *vptidrecvc
							      /* actually a:
								 struct ips_tid_recv_desc *tidrecvc */,
							      struct ips_message_header *p_hdr,
							      void (*ips_protoexp_do_tf_generr)
							      (void *vpprotoexp
							       /* actually a:
								  struct ips_protoexp *protoexp */,
							       void *vptidrecvc
							       /* actually a:
								  struct ips_tid_recv_desc *tidrecvc */,
							       struct ips_message_header *p_hdr),
							      void (*ips_protoexp_do_tf_seqerr)
							      (void *vpprotoexp
							       /* actually a:
								  struct ips_protoexp *protoexp */,
							       void *vptidrecvc
							       /* actually a:
								  struct ips_tid_recv_desc *tidrecvc */,
							       struct ips_message_header *p_hdr)
		)
{
	struct ips_protoexp *protoexp = (struct ips_protoexp *) vpprotoexp;
	struct ips_tid_recv_desc *tidrecvc = (struct ips_tid_recv_desc *) vptidrecvc;

	if_pf(psmi_hal_has_sw_status(PSM_HAL_HDRSUPP_ENABLED)) {
		/* Drop packet if generation number does not match. There
		 * is a window that before we program the hardware tidflow
		 * table with new gen/seq, hardware might receive some
		 * packets with the old generation.
		 */
		if (sequence_num.psn_gen != tidrecvc->tidflow_genseq.psn_gen)
		{
			PSM2_LOG_MSG("leaving");
			return PSM_HAL_ERROR_GENERAL_ERROR;
		}

#ifdef PSM_DEBUG
		/* Check if new packet falls into expected seq range, we need
		 * to deal with wrap around of the seq value from 2047 to 0
		 * because seq is only 11 bits. */
		int16_t seq_off = (int16_t)(sequence_num.psn_seq -
					tidrecvc->tidflow_genseq.psn_seq);
		if (seq_off < 0)
			seq_off += 2048; /* seq is 11 bits */
		psmi_assert(seq_off < 1024);
#endif
		/* NOTE: with RSM in use, we should not automatically update
		 * our PSN from the HFI's PSN.  The HFI doesn't know about
		 * RSM interceptions.
		 */
		/* (DON'T!) Update the shadow tidflow_genseq */
		/* tidrecvc->tidflow_genseq.psn_seq = sequence_num.psn_seq + 1; */

	}
	/* Always check the sequence number if we get a header, even if SH. */
	if_pt(sequence_num.psn_num == tidrecvc->tidflow_genseq.psn_num) {
		/* Update the shadow tidflow_genseq */
		tidrecvc->tidflow_genseq.psn_seq = sequence_num.psn_seq + 1;

		/* update the fake tidflow table with new seq, this is for
		 * seqerr and err_chk_gen processing to get the latest
		 * valid sequence number */
		psm3_hfp_gen1_tidflow_set_entry(
			tidrecvc->rdescid._desc_idx,
			tidrecvc->tidflow_genseq.psn_gen,
			tidrecvc->tidflow_genseq.psn_seq,
			tidrecvc->context->psm_hw_ctxt);
	} else {
		/* Generation mismatch */
		if (sequence_num.psn_gen != tidrecvc->tidflow_genseq.psn_gen) {
			ips_protoexp_do_tf_generr(protoexp,
						tidrecvc, p_hdr);
			PSM2_LOG_MSG("leaving");
			return PSM_HAL_ERROR_GENERAL_ERROR;
		} else {
			/* Possible sequence mismatch error */
			/* First, check if this is a recoverable SeqErr -
			 * caused by a good packet arriving in a tidflow that
			 * has had a FECN bit set on some earlier packet.
			 */

			/* If this is the first RSM packet, our own PSN state
			 * is probably old.  Pull from the HFI if it has
			 * newer data.
			 */
			uint64_t tf;
			psmi_seqnum_t tf_sequence_num;

			psm3_hfp_gen1_tidflow_get(tidrecvc->rdescid._desc_idx, &tf,
					     tidrecvc->context->psm_hw_ctxt);
			psm3_hfp_gen1_tidflow_get_seqnum(tf, &tf_sequence_num.psn_val);

			if (tf_sequence_num.psn_val > tidrecvc->tidflow_genseq.psn_seq)
				tidrecvc->tidflow_genseq.psn_seq = tf_sequence_num.psn_seq;

			/* Now re-check the sequence numbers. */
			if (sequence_num.psn_seq > tidrecvc->tidflow_genseq.psn_seq) {
				/* It really was a sequence error.  Restart. */
				ips_protoexp_do_tf_seqerr(protoexp, tidrecvc, p_hdr);
				PSM2_LOG_MSG("leaving");
				return PSM_HAL_ERROR_GENERAL_ERROR;
			} else {
				/* False SeqErr.  We can accept this packet. */
				if (sequence_num.psn_seq == tidrecvc->tidflow_genseq.psn_seq)
					tidrecvc->tidflow_genseq.psn_seq++;
			}
		}
	}

	return PSM_HAL_ERROR_OK;
}

static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_spio_transfer_frame(struct ips_proto *proto,
					struct ips_flow *flow, struct ips_scb *scb,
					uint32_t *payload, uint32_t length,
					uint32_t isCtrlMsg, uint32_t cksum_valid,
					uint32_t cksum
#ifdef PSM_CUDA
				, uint32_t is_cuda_payload
#endif
	)
{
	return psm3_gen1_spio_transfer_frame(proto, flow, scb,
					 payload, length, isCtrlMsg,
					 cksum_valid, cksum
#ifdef PSM_CUDA
				, is_cuda_payload
#endif
	);
} 

static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_transfer_frame(struct ips_proto *proto,
					struct ips_flow *flow, struct ips_scb *scb,
					uint32_t *payload, uint32_t length,
					uint32_t isCtrlMsg, uint32_t cksum_valid,
					uint32_t cksum
#ifdef PSM_CUDA
				, uint32_t is_cuda_payload
#endif
	)
{
	switch (flow->transfer) {
	case PSM_TRANSFER_PIO:
		return psm3_gen1_spio_transfer_frame(proto, flow, scb,
					 payload, length, isCtrlMsg,
					 cksum_valid, cksum
#ifdef PSM_CUDA
					, is_cuda_payload
#endif
		);
		break;
	case PSM_TRANSFER_DMA:
		return psm3_gen1_dma_transfer_frame(proto, flow, scb,
					 payload, length, cksum_valid, cksum);
		break;
	default:
		return PSM2_INTERNAL_ERR;
		break;
	}
}

static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_dma_send_pending_scbs(struct ips_proto *proto,
					struct ips_flow *flow, struct ips_scb_pendlist *slist,
					int *num_sent)
{
	return psm3_gen1_dma_send_pending_scbs(proto, flow, slist, num_sent);
}

static PSMI_HAL_INLINE psm2_error_t psm3_hfp_gen1_drain_sdma_completions(struct ips_proto *proto)
{
	return psm3_gen1_dma_completion_update(proto);
}

static PSMI_HAL_INLINE int psm3_hfp_gen1_get_node_id(int unit, int *nodep)
{
	int64_t node_id = psm3_sysfs_unit_read_node_s64(unit);
	*nodep = (int)node_id;
	if (node_id != -1)
		return PSM_HAL_ERROR_OK;
	else
		return -PSM_HAL_ERROR_GENERAL_ERROR;
}

static PSMI_HAL_INLINE int      psm3_hfp_gen1_get_jkey(psm2_ep_t ep)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ep->context.psm_hw_ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;

	return ctrl->base_info.jkey;
}

static PSMI_HAL_INLINE int      psm3_hfp_gen1_get_pio_size(psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;

	return (ctrl->ctxt_info.credits / 2) * 64 -
		(sizeof(struct ips_message_header) + HFI_PCB_SIZE_IN_BYTES);
}

static PSMI_HAL_INLINE int      psm3_hfp_gen1_get_subctxt(psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;

	return ctrl->ctxt_info.subctxt;
}

static PSMI_HAL_INLINE int      psm3_hfp_gen1_get_subctxt_cnt(psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;

	return psm_hw_ctxt->user_info.subctxt_cnt;
}

static PSMI_HAL_INLINE int      psm3_hfp_gen1_get_tid_exp_cnt(psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;

	return ctrl->__hfi_tidexpcnt;
}

static PSMI_HAL_INLINE int      psm3_hfp_gen1_get_pio_stall_cnt(psmi_hal_hw_context ctxt, uint64_t **pio_stall_cnt)
{
	if (!ctxt)
		return -PSM_HAL_ERROR_GENERAL_ERROR;

	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;

	*pio_stall_cnt = &psm_hw_ctxt->spio_ctrl.spio_num_stall_total;

	return PSM_HAL_ERROR_OK;
}
#endif /* PSM_OPA */
