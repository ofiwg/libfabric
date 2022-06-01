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

#include "psm_user.h"
#include "psm2_hal.h"
#include "gen1_user.h"

#if PSMI_HAL_INST_CNT > 1 || defined(PSM_DEBUG)
// declare all the HAL_INLINE functions and pull in implementation as non-inline
#define PSMI_HAL_CAT_INL_SYM(KERNEL) psm3_hfp_gen1_ ## KERNEL
#include "psm2_hal_inline_t.h"
#include "gen1_hal_inline_i.h"
#endif

static int psm3_hfp_gen1_initialize(psmi_hal_instance_t *phi,
											int devid_enabled[PTL_MAX_INIT])
{
	/* psm3_hal_current_hal_instance is not yet initialized, so
	 * we can't call psmi_hal_* routines to set cap or sw_status
	 */

	/* we initialize a few HAL software specific capabilities which
	 * are known before context_open can open RV or parse HAL specific
	 * env variables.  Additional flags may be added to cap_mask by
	 * context_open.
	 * Any flags which influence PSM env variable parsing prior to
	 * context_open must be set here
	 */
	phi->params.cap_mask = 0;

#if 0
	// this may have been an OPA bug, but may be hiding other bugs
	// This was guarded by a test of PSM_HAL_CAP_HDRSUPP, however that cap_mask
	// is not set until context_open so this code was never run and
	// the PSM_HAL_HDRSUPP_ENABLED sw_status was never set.  Error handling code
	// for packet sequence errors uses if_pf testing PSM_HAL_HDRSUPP_ENABLED
	{
		union psmi_envvar_val env_hdrsupp;

		psm3_getenv("PSM3_HDRSUPP",
			    "Receive header suppression. Default is 1 (enabled),"
					" 0 to disable.\n",
			    PSMI_ENVVAR_LEVEL_USER, PSMI_ENVVAR_TYPE_UINT_FLAGS,
			    (union psmi_envvar_val)1, &env_hdrsupp);
		if (env_hdrsupp.e_uint)
			phi->params.sw_status |= PSM_HAL_HDRSUPP_ENABLED;
	}
#endif

	return 0;
}

/* functions called vis DISPATCH_FUNC */
static int psm3_hfp_gen1_finalize_(void)
{
	return 0;
}

static const char* psm3_hfp_gen1_identify(void)
{
	static char buf[100];

/* we test NVIDIA_GPU_DIRECT here instead of PSM_CUDA since that define
 * controls the hfi1 header file interface
 */
	snprintf(buf, sizeof(buf), "HAL: %s (%s) built against driver interface v%d.%d"
#ifdef NVIDIA_GPU_DIRECT
			" gpu cuda"
#endif
			,
			psmi_hal_get_hal_instance_name(),
			psmi_hal_get_hal_instance_description(),
			HFI1_USER_SWMAJOR, HFI1_USER_SWMINOR);
	return buf;
}

// used as domain.name for fi_info
static const char *psm3_hfp_gen1_get_unit_name(int unit)
{
	return psm3_sysfs_unit_dev_name(unit);
}

// used as fabric.name for fi_info
static int psm3_hfp_gen1_get_port_subnet_name(int unit, int port, int addr_index, char *buf, size_t bufsize)
{
	psmi_subnet128_t subnet;

	if (psm3_hfp_gen1_get_port_subnet(unit, 1, addr_index, &subnet, NULL, NULL, NULL))
		return -1;

	psm3_subnet128_fmt_name(subnet, buf, bufsize);
	return 0;
}

static int psm3_hfp_gen1_get_port_lid(int unit, int port, int addr_index)
{
	return psm3_gen1_get_port_lid(unit, port, addr_index, GEN1_FILTER);
}

// initialize default MQ thresholds
// This is called prior to parsing PSM3_ env variables for MQ and
// also prior to the EP being opened (eg. NIC not yet selected).
static void psm3_hfp_gen1_mq_init_defaults(struct psm2_mq *mq)
{
	unsigned rdmamode = psm3_gen1_parse_tid(1);

	/* These values may be changed by initialize_params if user specifies
	 * corresponding PSM3_* env variables.
	 * Otherwise these defaults are used.
	 */
	if(psm3_cpu_model == CPUID_MODEL_PHI_GEN2 || psm3_cpu_model == CPUID_MODEL_PHI_GEN2M)
	{
		mq->hfi_thresh_rv = 200000;
		mq->hfi_base_window_rv = 4194304;
	} else {
		mq->hfi_thresh_rv = 64000;
		mq->hfi_base_window_rv = 131072;
	}
	// hfi_base_window_rv may be further reduced in protoexp_init to account
	// for max TID resources allowed per IO

	// reload env var cache once per MQ so don't report in VERBOSE_ENV per rail
	if (! (rdmamode & IPS_PROTOEXP_FLAG_ENABLED)) {
		// Retain existing gen1 behavior and leave rendezvous enabled.  It
		// will use LONG_DATA mechanism which provides receive side pacing
		//mq->hfi_thresh_rv = (~(uint32_t)0); // disable rendezvous
	}
	mq->hfi_thresh_tiny = PSM_MQ_NIC_MAX_TINY;
#ifdef PSM_CUDA
	if (PSMI_IS_GPU_ENABLED)
		mq->hfi_base_window_rv = 2097152;
#endif
}

// initialize default EP Open options
// This is called in psm3_ep_open_internal prior to parsing PSM3_ env variables
// and also prior to the EP being opened (eg. NIC not yet selected).
static void psm3_hfp_gen1_ep_open_opts_get_defaults(struct psm3_ep_open_opts *opts)
{
	opts->imm_size = 128;
}

static void psm3_hfp_gen1_context_initstats(psm2_ep_t ep)
{
	// Noop
}

/* functions called vis DISPATCH_PI */
static int psm3_hfp_gen1_get_num_ports(void)
{
	return HFI_NUM_PORTS_GEN1;
}

static int psm3_hfp_gen1_get_unit_active(int unit)
{
	return psm3_gen1_get_unit_active(unit, GEN1_FILTER);
}

static int psm3_hfp_gen1_get_num_free_contexts(int unit)
{
	int64_t nfreectxts=0;

	if (!psm3_sysfs_unit_read_s64(unit, "nfreectxts",
				     &nfreectxts, 0))
	{
		return (int)nfreectxts;
	}
	return -PSM_HAL_ERROR_GENERAL_ERROR;
}

static int psm3_hfp_gen1_get_default_pkey(void)
{
	return 0x8001;	/* fabric default pkey for app traffic */
}

static int psm3_hfp_gen1_get_unit_pci_bus(int unit, uint32_t *domain,
	uint32_t *bus, uint32_t *device, uint32_t *function)
{
	return psm3_sysfs_get_unit_pci_bus(unit, domain, bus, device, function);
}

static int psm3_hfp_gen1_get_unit_device_id(int unit, char *buf, size_t bufsize)
{
	return psm3_sysfs_get_unit_device_id(unit, buf, bufsize);
}

static int psm3_hfp_gen1_get_unit_device_version(int unit, char *buf, size_t bufsize)
{
	return psm3_sysfs_get_unit_device_version(unit, buf, bufsize);
}

static int psm3_hfp_gen1_get_unit_vendor_id(int unit, char *buf, size_t bufsize)
{
	return psm3_sysfs_get_unit_vendor_id(unit, buf, bufsize);
}

static int psm3_hfp_gen1_get_unit_driver(int unit, char *buf, size_t bufsize)
{
	return psm3_sysfs_get_unit_driver(unit, buf, bufsize);
}

/* define the singleton that implements hal for gen1 */
static hfp_gen1_t psm3_gen1_hi = {
	/* start of public psmi_hal_instance_t data */
	.phi = {
		.hal_index				  = PSM_HAL_INDEX_OPA,
		.description				  = "OPA100"
#ifdef PSM_CUDA
								" (cuda)"
#endif
									,
		.nic_sys_class_path			  = "/sys/class/infiniband",
		.nic_sys_port_path_fmt			  = PSM3_PORT_PATH_TYPE_IB,
		.params					  = {0},

	/* functions called directly, no DISPATCH macro */
		.hfp_initialize				  = psm3_hfp_gen1_initialize,
		.hfp_have_active_unit			  = psm3_hfp_gen1_have_active_unit,

	/* called via DISPATCH_FUNC */
		.hfp_finalize_				  = psm3_hfp_gen1_finalize_,
		.hfp_identify				  = psm3_hfp_gen1_identify,
		.hfp_get_unit_name			  = psm3_hfp_gen1_get_unit_name,
		.hfp_get_port_subnet_name		  = psm3_hfp_gen1_get_port_subnet_name,
		.hfp_get_port_speed			  = psm3_hfp_gen1_get_port_speed,
		.hfp_get_port_lid			  = psm3_hfp_gen1_get_port_lid,
		.hfp_mq_init_defaults			  = psm3_hfp_gen1_mq_init_defaults,
		.hfp_ep_open_opts_get_defaults		  = psm3_hfp_gen1_ep_open_opts_get_defaults,
		.hfp_context_initstats			  = psm3_hfp_gen1_context_initstats,
#ifdef PSM_CUDA
		.hfp_gdr_open				  = psm3_hfp_gen1_gdr_open,
#endif

	/* called via DISPATCH_PI */
		.hfp_get_num_units			  = psm3_hfp_gen1_get_num_units,
		.hfp_get_num_ports			  = psm3_hfp_gen1_get_num_ports,
		.hfp_get_unit_active			  = psm3_hfp_gen1_get_unit_active,
		.hfp_get_port_active			  = psm3_hfp_gen1_get_port_active,
		.hfp_get_num_contexts			  = psm3_hfp_gen1_get_num_contexts,
		.hfp_get_num_free_contexts		  = psm3_hfp_gen1_get_num_free_contexts,
		.hfp_get_default_pkey			  = psm3_hfp_gen1_get_default_pkey,
		.hfp_get_port_subnet			  = psm3_hfp_gen1_get_port_subnet,
		.hfp_get_unit_pci_bus			  = psm3_hfp_gen1_get_unit_pci_bus,
		.hfp_get_unit_device_id			  = psm3_hfp_gen1_get_unit_device_id,
		.hfp_get_unit_device_version		  = psm3_hfp_gen1_get_unit_device_version,
		.hfp_get_unit_vendor_id			  = psm3_hfp_gen1_get_unit_vendor_id,
		.hfp_get_unit_driver			  = psm3_hfp_gen1_get_unit_driver,

	/* called via DISPATCH, may be inline */
#if PSMI_HAL_INST_CNT > 1 || defined(PSM_DEBUG)
		.hfp_context_open			  = psm3_hfp_gen1_context_open,
		.hfp_close_context			  = psm3_hfp_gen1_close_context,
		.hfp_context_check_status		  = psm3_hfp_gen1_context_check_status,
#ifdef PSM_FI
		.hfp_faultinj_allowed		  = psm3_hfp_gen1_faultinj_allowed,
#endif
		.hfp_ips_ptl_init_pre_proto_init	  = psm3_hfp_gen1_ips_ptl_init_pre_proto_init,
		.hfp_ips_ptl_init_post_proto_init	  = psm3_hfp_gen1_ips_ptl_init_post_proto_init,
		.hfp_ips_ptl_fini			  = psm3_hfp_gen1_ips_ptl_fini,
		.hfp_ips_proto_init			  = psm3_hfp_gen1_ips_proto_init,
		.hfp_ips_proto_update_linkinfo		  = psm3_hfp_gen1_ips_proto_update_linkinfo,
		.hfp_ips_fully_connected		  = psm3_hfp_gen1_ips_fully_connected,
		.hfp_ips_ipsaddr_set_req_params		  = psm3_hfp_gen1_ips_ipsaddr_set_req_params,
		.hfp_ips_ipsaddr_process_connect_reply	  = psm3_hfp_gen1_ips_ipsaddr_process_connect_reply,
		.hfp_ips_proto_build_connect_message	  = psm3_hfp_gen1_ips_proto_build_connect_message,
		.hfp_ips_ipsaddr_init_addressing	  = psm3_hfp_gen1_ips_ipsaddr_init_addressing,
		.hfp_ips_ipsaddr_init_connections	  = psm3_hfp_gen1_ips_ipsaddr_init_connections,
		.hfp_ips_ipsaddr_free			  = psm3_hfp_gen1_ips_ipsaddr_free,
		.hfp_ips_flow_init			  = psm3_hfp_gen1_ips_flow_init,
		.hfp_ips_ipsaddr_disconnect		  = psm3_hfp_gen1_ips_ipsaddr_disconnect,
		.hfp_ips_ibta_init			  = psm3_hfp_gen1_ips_ibta_init,
		.hfp_ips_path_rec_init			  = psm3_hfp_gen1_ips_path_rec_init,
		.hfp_ips_ptl_pollintr			  = psm3_hfp_gen1_ips_ptl_pollintr,
#ifdef PSM_CUDA
		.hfp_gdr_close				  = psm3_hfp_gen1_gdr_close,
		.hfp_gdr_convert_gpu_to_host_addr	  = psm3_hfp_gen1_gdr_convert_gpu_to_host_addr,
#endif /* PSM_CUDA */
		.hfp_get_port_index2pkey		  = psm3_hfp_gen1_get_port_index2pkey,
		.hfp_poll_type				  = psm3_hfp_gen1_poll_type,
		.hfp_free_tid				  = psm3_hfp_gen1_free_tid,
		.hfp_get_tidcache_invalidation		  = psm3_hfp_gen1_get_tidcache_invalidation,
		.hfp_update_tid				  = psm3_hfp_gen1_update_tid,
		.hfp_tidflow_check_update_pkt_seq	  = psm3_hfp_gen1_tidflow_check_update_pkt_seq,
		.hfp_tidflow_get			  = psm3_hfp_gen1_tidflow_get,
		.hfp_tidflow_get_hw			  = psm3_hfp_gen1_tidflow_get_hw,
		.hfp_tidflow_get_seqnum			  = psm3_hfp_gen1_tidflow_get_seqnum,
		.hfp_tidflow_reset			  = psm3_hfp_gen1_tidflow_reset,
		.hfp_tidflow_set_entry			  = psm3_hfp_gen1_tidflow_set_entry,
		.hfp_get_hfi_event_bits			  = psm3_hfp_gen1_get_hfi_event_bits,
		.hfp_spio_transfer_frame		  = psm3_hfp_gen1_spio_transfer_frame,
		.hfp_transfer_frame			  = psm3_hfp_gen1_transfer_frame,
		.hfp_dma_send_pending_scbs		  = psm3_hfp_gen1_dma_send_pending_scbs,
		.hfp_drain_sdma_completions		  = psm3_hfp_gen1_drain_sdma_completions,
		.hfp_get_node_id			  = psm3_hfp_gen1_get_node_id,
		.hfp_get_jkey				  = psm3_hfp_gen1_get_jkey,
		.hfp_get_pio_size			  = psm3_hfp_gen1_get_pio_size,
		.hfp_get_pio_stall_cnt			  = psm3_hfp_gen1_get_pio_stall_cnt,
		.hfp_get_subctxt			  = psm3_hfp_gen1_get_subctxt,
		.hfp_get_subctxt_cnt			  = psm3_hfp_gen1_get_subctxt_cnt,
		.hfp_get_tid_exp_cnt			  = psm3_hfp_gen1_get_tid_exp_cnt,
		.hfp_set_pkey				  = psm3_hfp_gen1_set_pkey,
#endif /* PSMI_HAL_INST_CNT > 1 || defined(PSM_DEBUG) */
	},
	/* start of private hfp_gen1_private data */
	.hfp_private = {
		.sdmahdr_req_size	= 0,
		.dma_rtail		= 0,
		.hdrq_rhf_off		= 0,
	}
};

static void __attribute__ ((constructor)) __psmi_hal_gen1_constructor(void)
{
	psm3_hal_register_instance((psmi_hal_instance_t*)&psm3_gen1_hi);
}
#endif /* PSM_OPA */
