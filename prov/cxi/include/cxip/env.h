/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_ENV_H_
#define _CXIP_ENV_H_

#include <stddef.h>

/* Type definitions */
struct cxip_environment {
	/* Translation */
	int odp;
	int force_odp;
	int ats;
	int iotlb;
	int disable_dmabuf_cuda;
	int disable_dmabuf_rocr;
	enum cxip_ats_mlock_mode ats_mlock_mode;

	/* Messaging */
	int fork_safe_requested;
	enum cxip_ep_ptle_mode rx_match_mode;
	int msg_offload;
	int trunc_ok;
	int hybrid_preemptive;
	int hybrid_recv_preemptive;
	size_t rdzv_threshold;
	size_t rdzv_get_min;
	size_t rdzv_eager_size;
	int rdzv_aligned_sw_rget;
	int rnr_max_timeout_us;
	int disable_non_inject_msg_idc;
	int disable_non_inject_rma_idc;
	int disable_non_inject_amo_idc;
	int disable_host_register;
	size_t oflow_buf_size;
	size_t oflow_buf_min_posted;
	size_t oflow_buf_max_cached;
	size_t safe_devmem_copy_threshold;
	size_t req_buf_size;
	size_t req_buf_min_posted;
	size_t req_buf_max_cached;
	int sw_rx_tx_init_max;
	int msg_lossless;
	size_t default_cq_size;
	size_t default_tx_size;
	size_t default_rx_size;
	int optimized_mrs;
	int prov_key_cache;
	int mr_match_events;
	int disable_eq_hugetlb;
	int zbcoll_radix;

	enum cxip_llring_mode llring_mode;

	int cq_policy;

	size_t default_vni;

	size_t eq_ack_batch_size;
	size_t cq_batch_size;
	int fc_retry_usec_delay;
	int cntr_spin_before_yield;
	size_t ctrl_rx_eq_max_size;
	char *device_name;
	size_t cq_fill_percent;
	int rget_tc;
	int cacheline_size;

	char *coll_job_id;
	char *coll_job_step_id;
	size_t coll_retry_usec;
	size_t coll_timeout_usec;
	char *coll_fabric_mgr_url;
	char *coll_mcast_token;
	size_t hwcoll_addrs_per_job;
	size_t hwcoll_min_nodes;
	int coll_use_dma_put;

	char hostname[255];
	char *telemetry;
	int telemetry_rgid;
	int disable_hmem_dev_register;
	int ze_hmem_supported;
	enum cxip_rdzv_proto rdzv_proto;
	int disable_alt_read_cmdq;
	int cntr_trig_cmdq;
	int enable_trig_op_limit;
	int hybrid_posted_recv_preemptive;
	int hybrid_unexpected_msg_preemptive;
	size_t mr_cache_events_disable_poll_nsecs;
	size_t mr_cache_events_disable_le_poll_nsecs;
	int force_dev_reg_copy;
	enum cxip_mr_target_ordering mr_target_ordering;
	int disable_cuda_sync_memops;
};

#endif /* _CXIP_ENV_H_ */
