// SPDX-License-Identifier: GPL-2.0

/*
 * Cassini QoS Profiles
 * Copyright 2023 Hewlett Packard Enterprise Development LP
 */
#include "cass_core.h"

#define SRB_MAX_SIZE 2048U

/* Reserve half the SRB space to cover C2 expected bandwidth delay product. The
 * remainder of the space should be split up between the RDMA traffic classes.
 */
#define SRB_SHARED_MAX (SRB_MAX_SIZE / 2)

/* Since low-latency does not need to sustain bandwidth, reserve only a few
 * MTU sized packets worth of SRB.
 */
#define LOW_LATENCY_SRB_RSVD 32U

/* SRB Space Reserved for the Global Restricted BC - C1 Only */
#define C1_SRB_GLOBAL_RES_BC_RSVD 39U

/* SRB Space Available to Split Between other RDMA TCs */
#define C1_SRB_RESERVED (SRB_MAX_SIZE - SRB_SHARED_MAX - C1_SRB_GLOBAL_RES_BC_RSVD - LOW_LATENCY_SRB_RSVD)
#define C2_SRB_RESERVED (SRB_MAX_SIZE - SRB_SHARED_MAX - LOW_LATENCY_SRB_RSVD)


struct qos_profile profiles[CXI_QOS_NUM_PROF] = {
	/* HPC Profile */
	[CXI_QOS_HPC] = {
		/* Shared with low latency */
		.pct_control_pcp = 5,
		.untagged_eth_pcp = 6, /* Shared with Eth1 */
		.tcs_active = {
			[CXI_TC_DEDICATED_ACCESS] = true,
			[CXI_TC_LOW_LATENCY] = true,
			[CXI_TC_BULK_DATA] = true,
			[CXI_TC_BEST_EFFORT] = true,

			[CXI_ETH_SHARED] = true,
			[CXI_ETH_TC1] = true,
		},
		.tcs = {
			[CXI_TC_DEDICATED_ACCESS] = {
				.dscp_pcp_settings = {
					.req_pcp = 6,
					.rsp_pcp = 7,
					.res_rsp_dscp = 13,
					.unres_rsp_dscp = 15,
					.res_req_dscp = 12,
					.unres_req_dscp = 14,
					.hrp_res_req_dscp = -1,
					.coll_leaf_res_req_dscp = -1,
					.restricted_unres_req_dscp = -1,
				},
				.ixe_settings = {
					.fq_count = 10,
				},
				.lpe_settings = {
					.fq_count = 8,
				},
				.pct_settings = {
					.trs_rsvd = 64,
					.mst_rsvd = 64,
					.tct_rsvd = 256,
				},
				.oxe_settings = {
					.smt_rsvd = 4,
					.sct_rsvd = 4,
					.pbuf_rsvd = 39,
					.assured_percent = 50,
					.ceiling_percent = 100,
					.bucket_limit = DIV_ROUND_UP(4 * PORTALS_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 2,
					.leaf_response_priority = 1,
					.branch_priority = 1,
					.mfs_index = 0,
				},
				.cq_settings = {
					.dynamic_fq_count = 8,
					.fq_buf_reserved = 32,
					.pfq_high_thresh = 64,
				},
				.hni_settings = {
					.pbuf_rsvd = 128,
					.mfs = PORTALS_MAX_FRAME_SIZE,
				},
			},
			[CXI_TC_LOW_LATENCY] = {
				.dscp_pcp_settings = {
					.req_pcp = 4,
					.rsp_pcp = 5,
					.res_rsp_dscp = 9,
					.unres_rsp_dscp = 11,
					.res_req_dscp = 8,
					.unres_req_dscp = 10,
					.hrp_res_req_dscp = -1,
					.coll_leaf_res_req_dscp = -1,
					.restricted_unres_req_dscp = -1,
				},
				.ixe_settings = {
					.fq_count = 2,
				},
				.lpe_settings = {

					.fq_count = 2,
				},
				.pct_settings = {
					.trs_rsvd = 64,
					.mst_rsvd = 64,
					.tct_rsvd = 256,
				},
				.oxe_settings = {
					.spt_rsvd = 64,
					.smt_rsvd = 4,
					.sct_rsvd = 4,
					.srb_rsvd = LOW_LATENCY_SRB_RSVD,
					.pbuf_rsvd = 13,
					.assured_percent = 15,
					.ceiling_percent = 30,
					.bucket_limit = DIV_ROUND_UP(4 * PORTALS_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 7,
					.leaf_response_priority = 6,
					.branch_priority = 6,
					.mfs_index = 0,
				},
				.cq_settings = {
					.dynamic_fq_count = 2,
					.fq_buf_reserved = 32,
					.pfq_high_thresh = 64,
				},
				.hni_settings = {
					.pbuf_rsvd = 128,
					.mfs = PORTALS_MAX_FRAME_SIZE,
				},
			},
			[CXI_TC_BULK_DATA] = {
				.dscp_pcp_settings = {
					.req_pcp = 2,
					.rsp_pcp = 3,
					.res_rsp_dscp = 5,
					.unres_rsp_dscp = 7,
					.res_req_dscp = 4,
					.unres_req_dscp = 6,
					.hrp_res_req_dscp = -1,
					.coll_leaf_res_req_dscp = -1,
					.restricted_unres_req_dscp = -1,
				},
				.ixe_settings = {
					.fq_count = 4,
				},
				.lpe_settings = {
					.fq_count = 2,
				},
				.pct_settings = {
					.trs_rsvd = 32,
					.mst_rsvd = 32,
					.tct_rsvd = 128,
				},
				.oxe_settings = {
					.smt_rsvd = 4,
					.sct_rsvd = 4,
					.pbuf_rsvd = 39,
					.assured_percent = 15,
					.ceiling_percent = 100,
					.bucket_limit = DIV_ROUND_UP(2 * PORTALS_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 2,
					.leaf_response_priority = 1,
					.branch_priority = 1,
					.mfs_index = 0,
				},
				.cq_settings = {
					.dynamic_fq_count = 4,
					.fq_buf_reserved = 32,
					.pfq_high_thresh = 64,
				},
				.hni_settings = {
					.pbuf_rsvd = 64,
					.mfs = PORTALS_MAX_FRAME_SIZE,
				},
			},
			[CXI_TC_BEST_EFFORT] = {
				.dscp_pcp_settings = {
					.req_pcp = 0,
					.rsp_pcp = 1,
					.res_rsp_dscp = 1,
					.unres_rsp_dscp = 3,
					.res_req_dscp = 0,
					.unres_req_dscp = 2,
					.hrp_res_req_dscp = 50,
					.coll_leaf_res_req_dscp = 52,
					.restricted_unres_req_dscp = 30,
				},
				.ixe_settings = {
					.fq_count = 8,
				},
				.lpe_settings = {
					.fq_count = 4,
				},
				.pct_settings = {
					.trs_rsvd = 4,
					.mst_rsvd = 4,
					.tct_rsvd = 16,
				},
				.oxe_settings = {
					.smt_rsvd = 4,
					.sct_rsvd = 4,
					.pbuf_rsvd = 13,
					.assured_percent = 10,
					.ceiling_percent = 100,
					.bucket_limit = DIV_ROUND_UP(2 * PORTALS_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 2,
					.leaf_response_priority = 1,
					.branch_priority = 1,
					.mfs_index = 0,
				},
				.cq_settings = {
					.dynamic_fq_count = 8,
					.static_fq_count = 16,
					.fq_buf_reserved = 32,
					.pfq_high_thresh = 64,
				},
				.hni_settings = {
					.pbuf_rsvd = 17,
					.mfs = PORTALS_MAX_FRAME_SIZE,
				},
			},
			[CXI_TC_ETH] = {
				.oxe_settings = {
					.spt_rsvd = 30,
					.pbuf_rsvd = DIV_ROUND_UP(ETHERNET_MAX_FRAME_SIZE, 256),
				},
				.cq_settings = {
					.dynamic_fq_count = 2,
					.static_fq_count = 2,
					.fq_buf_reserved = 32,
					.pfq_high_thresh = 64,
				},
				.hni_settings = {
					.pbuf_rsvd = DIV_ROUND_UP(ETHERNET_MAX_FRAME_SIZE, 128),
					.mfs = ETHERNET_MAX_FRAME_SIZE,
				},

			},
			[CXI_ETH_SHARED] = {
				.oxe_settings = {
					.assured_percent = 10,
					.ceiling_percent = 100,
					.bucket_limit = DIV_ROUND_UP(2 * ETHERNET_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 1,
					.branch_priority = 1,
					.mfs_index = 1,
				},
			},
			[CXI_ETH_TC1] = {
				.eth_settings = {
					.pcp =  6,
				},
				.oxe_settings = {
					.assured_percent = 10,
					.ceiling_percent = 100,
					.bucket_limit = DIV_ROUND_UP(2 * ETHERNET_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 1,
					.branch_priority = 1,
					.mfs_index = 1,
				},

			},
		},
	},
	[CXI_QOS_LL_BE_BD_ET] = {
		.pct_control_pcp = 7,
		.untagged_eth_pcp = 6, /* Shared with Eth1 */
		.tcs_active = {
			[CXI_TC_LOW_LATENCY] = true,
			[CXI_TC_BULK_DATA] = true,
			[CXI_TC_BEST_EFFORT] = true,

			[CXI_ETH_SHARED] = true,
			[CXI_ETH_TC1] = true,
		},
		.tcs = {
			[CXI_TC_LOW_LATENCY] = {
				.dscp_pcp_settings = {
					.req_pcp = 4,
					.rsp_pcp = 5,
					.res_rsp_dscp = 9,
					.unres_rsp_dscp = 11,
					.res_req_dscp = 8,
					.unres_req_dscp = 10,
					.hrp_res_req_dscp = -1,
					.coll_leaf_res_req_dscp = 52,
					.restricted_unres_req_dscp = -1,
				},
				.ixe_settings = {
					.fq_count = 2,
				},
				.lpe_settings = {

					.fq_count = 2,
				},
				.pct_settings = {
					.trs_rsvd = 64,
					.mst_rsvd = 64,
					.tct_rsvd = 256,
				},
				.oxe_settings = {
					.spt_rsvd = 64,
					.smt_rsvd = 4,
					.sct_rsvd = 4,
					.srb_rsvd = LOW_LATENCY_SRB_RSVD,
					.pbuf_rsvd = 13,
					.assured_percent = 2,
					.ceiling_percent = 5,
					.bucket_limit = DIV_ROUND_UP(4 * PORTALS_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 7,
					.leaf_response_priority = 6,
					.branch_priority = 6,
					.mfs_index = 0,
				},
				.cq_settings = {
					.dynamic_fq_count = 2,
					.static_fq_count = 4,
					.fq_buf_reserved = 32,
					.pfq_high_thresh = 64,
				},
				.hni_settings = {
					.pbuf_rsvd = 128,
					.mfs = PORTALS_MAX_FRAME_SIZE,
				},
			},
			[CXI_TC_BULK_DATA] = {
				.dscp_pcp_settings = {
					.req_pcp = 2,
					.rsp_pcp = 3,
					.res_rsp_dscp = 5,
					.unres_rsp_dscp = 7,
					.res_req_dscp = 4,
					.unres_req_dscp = 6,
					.hrp_res_req_dscp = -1,
					.coll_leaf_res_req_dscp = -1,
					.restricted_unres_req_dscp = -1,
				},
				.ixe_settings = {
					.fq_count = 4,
				},
				.lpe_settings = {
					.fq_count = 2,
				},
				.pct_settings = {
					.trs_rsvd = 32,
					.mst_rsvd = 32,
					.tct_rsvd = 128,
				},
				.oxe_settings = {
					.smt_rsvd = 4,
					.sct_rsvd = 4,
					.pbuf_rsvd = 39,
					.assured_percent = 10, /* Fabric: 20 */
					.ceiling_percent = 100,
					.bucket_limit = DIV_ROUND_UP(2 * PORTALS_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 2,
					.leaf_response_priority = 1,
					.branch_priority = 1,
					.mfs_index = 0,
				},
				.cq_settings = {
					.dynamic_fq_count = 4,
					.fq_buf_reserved = 32,
					.pfq_high_thresh = 64,
				},
				.hni_settings = {
					.pbuf_rsvd = 64,
					.mfs = PORTALS_MAX_FRAME_SIZE,
				},
			},
			[CXI_TC_BEST_EFFORT] = {
				.dscp_pcp_settings = {
					.req_pcp = 0,
					.rsp_pcp = 1,
					.res_rsp_dscp = 1,
					.unres_rsp_dscp = 3,
					.res_req_dscp = 0,
					.unres_req_dscp = 2,
					.hrp_res_req_dscp = 50,
					.coll_leaf_res_req_dscp = -1,
					.restricted_unres_req_dscp = 30,
				},
				.ixe_settings = {
					.fq_count = 8,
				},
				.lpe_settings = {
					.fq_count = 4,
				},
				.pct_settings = {
					.trs_rsvd = 4,
					.mst_rsvd = 4,
					.tct_rsvd = 16,
				},
				.oxe_settings = {
					.smt_rsvd = 4,
					.sct_rsvd = 4,
					.pbuf_rsvd = 13,
					.assured_percent = 10, /* Fabric: 50 */
					.ceiling_percent = 100,
					.bucket_limit = DIV_ROUND_UP(2 * PORTALS_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 2,
					.leaf_response_priority = 1,
					.branch_priority = 1,
					.mfs_index = 0,
				},
				.cq_settings = {
					.dynamic_fq_count = 8,
					.static_fq_count = 16,
					.fq_buf_reserved = 32,
					.pfq_high_thresh = 64,
				},
				.hni_settings = {
					.pbuf_rsvd = 17,
					.mfs = PORTALS_MAX_FRAME_SIZE,
				},
			},
			[CXI_TC_ETH] = {
				.oxe_settings = {
					.spt_rsvd = 30,
					.pbuf_rsvd = DIV_ROUND_UP(ETHERNET_MAX_FRAME_SIZE, 256),
				},
				.cq_settings = {
					.dynamic_fq_count = 2,
					.static_fq_count = 2,
					.fq_buf_reserved = 32,
					.pfq_high_thresh = 64,
				},
				.hni_settings = {
					.pbuf_rsvd = DIV_ROUND_UP(ETHERNET_MAX_FRAME_SIZE, 128),
					.mfs = ETHERNET_MAX_FRAME_SIZE,
				},

			},
			[CXI_ETH_SHARED] = {
				.oxe_settings = {
					.assured_percent = 0,
					.ceiling_percent = 100,
					.bucket_limit = DIV_ROUND_UP(2 * ETHERNET_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 2,
					.branch_priority = 1,
					.mfs_index = 1,
				},
			},
			[CXI_ETH_TC1] = {
				.eth_settings = {
					.pcp =  6,
				},
				.oxe_settings = {
					.assured_percent = 10, /* Fabric: 20 */
					.ceiling_percent = 100,
					.bucket_limit = DIV_ROUND_UP(2 * ETHERNET_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 2,
					.branch_priority = 1,
					.mfs_index = 1,
				},

			},
		},
	},
	[CXI_QOS_LL_BE_BD_ET1_ET2] = {
		.pct_control_pcp = 7,
		.untagged_eth_pcp = 2, /* Shared with Eth2 */
		.tcs_active = {
			[CXI_TC_LOW_LATENCY] = true,
			[CXI_TC_BULK_DATA] = true,
			[CXI_TC_BEST_EFFORT] = true,

			[CXI_ETH_SHARED] = true,
			[CXI_ETH_TC1] = true,
			[CXI_ETH_TC2] = true,
		},
		.tcs = {
			[CXI_TC_LOW_LATENCY] = {
				.dscp_pcp_settings = {
					.req_pcp = 4,
					.rsp_pcp = 5,
					.res_rsp_dscp = 40,
					.unres_rsp_dscp = 41,
					.res_req_dscp = 32,
					.unres_req_dscp = 33,
					.hrp_res_req_dscp = -1,
					.coll_leaf_res_req_dscp = 52,
					.restricted_unres_req_dscp = -1,
				},
				.ixe_settings = {
					.fq_count = 2,
				},
				.lpe_settings = {

					.fq_count = 2,
				},
				.pct_settings = {
					.trs_rsvd = 64,
					.mst_rsvd = 64,
					.tct_rsvd = 256,
				},
				.oxe_settings = {
					.spt_rsvd = 64,
					.smt_rsvd = 4,
					.sct_rsvd = 4,
					.srb_rsvd = LOW_LATENCY_SRB_RSVD,
					.pbuf_rsvd = 13,
					.assured_percent = 2,
					.ceiling_percent = 5,
					.bucket_limit = DIV_ROUND_UP(4 * PORTALS_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 7,
					.leaf_response_priority = 6,
					.branch_priority = 6,
					.mfs_index = 0,
				},
				.cq_settings = {
					.dynamic_fq_count = 2,
					.static_fq_count = 4,
					.fq_buf_reserved = 32,
					.pfq_high_thresh = 64,
				},
				.hni_settings = {
					.pbuf_rsvd = 128,
					.mfs = PORTALS_MAX_FRAME_SIZE,
				},
			},
			[CXI_TC_BULK_DATA] = {
				.dscp_pcp_settings = {
					.req_pcp = 2,
					.rsp_pcp = 3,
					.res_rsp_dscp = 24,
					.unres_rsp_dscp = 26,
					.res_req_dscp = 16,
					.unres_req_dscp = 18,
					.hrp_res_req_dscp = -1,
					.coll_leaf_res_req_dscp = -1,
					.restricted_unres_req_dscp = -1,
				},
				.ixe_settings = {
					.fq_count = 4,
				},
				.lpe_settings = {
					.fq_count = 2,
				},
				.pct_settings = {
					.trs_rsvd = 32,
					.mst_rsvd = 32,
					.tct_rsvd = 128,
				},
				.oxe_settings = {
					.smt_rsvd = 4,
					.sct_rsvd = 4,
					.pbuf_rsvd = 39,
					.assured_percent = 10, /* Fabric: 20 */
					.ceiling_percent = 100,
					.bucket_limit = DIV_ROUND_UP(2 * PORTALS_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 2,
					.leaf_response_priority = 1,
					.branch_priority = 1,
					.mfs_index = 0,
				},
				.cq_settings = {
					.dynamic_fq_count = 4,
					.fq_buf_reserved = 32,
					.pfq_high_thresh = 64,
				},
				.hni_settings = {
					.pbuf_rsvd = 64,
					.mfs = PORTALS_MAX_FRAME_SIZE,
				},
			},
			[CXI_TC_BEST_EFFORT] = {
				.dscp_pcp_settings = {
					.req_pcp = 0,
					.rsp_pcp = 1,
					.res_rsp_dscp = 1,
					.unres_rsp_dscp = 3,
					.res_req_dscp = 0,
					.unres_req_dscp = 2,
					.hrp_res_req_dscp = 7,
					.coll_leaf_res_req_dscp = -1,
					.restricted_unres_req_dscp = 30,
				},
				.ixe_settings = {
					.fq_count = 8,
				},
				.lpe_settings = {
					.fq_count = 4,
				},
				.pct_settings = {
					.trs_rsvd = 4,
					.mst_rsvd = 4,
					.tct_rsvd = 16,
				},
				.oxe_settings = {
					.smt_rsvd = 4,
					.sct_rsvd = 4,
					.pbuf_rsvd = 13,
					.assured_percent = 10, /* Fabric: 40 */
					.ceiling_percent = 100,
					.bucket_limit = DIV_ROUND_UP(2 * PORTALS_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 2,
					.leaf_response_priority = 1,
					.branch_priority = 1,
					.mfs_index = 0,
				},
				.cq_settings = {
					.dynamic_fq_count = 8,
					.static_fq_count = 16,
					.fq_buf_reserved = 32,
					.pfq_high_thresh = 64,
				},
				.hni_settings = {
					.pbuf_rsvd = 17,
					.mfs = PORTALS_MAX_FRAME_SIZE,
				},
			},
			[CXI_TC_ETH] = {
				.oxe_settings = {
					.spt_rsvd = 30,
					.pbuf_rsvd = DIV_ROUND_UP(ETHERNET_MAX_FRAME_SIZE, 256),
				},
				.cq_settings = {
					.dynamic_fq_count = 2,
					.static_fq_count = 2,
					.fq_buf_reserved = 32,
					.pfq_high_thresh = 64,
				},
				.hni_settings = {
					.pbuf_rsvd = DIV_ROUND_UP(ETHERNET_MAX_FRAME_SIZE, 128),
					.mfs = ETHERNET_MAX_FRAME_SIZE,
				},

			},
			[CXI_ETH_SHARED] = {
				.oxe_settings = {
					.assured_percent = 0,
					.ceiling_percent = 100,
					.bucket_limit = DIV_ROUND_UP(2 * ETHERNET_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 2,
					.branch_priority = 1,
					.mfs_index = 1,
				},
			},
			[CXI_ETH_TC1] = {
				.eth_settings = {
					.pcp =  6,
				},
				.oxe_settings = {
					.assured_percent = 10, /* Fabric: 25 */
					.ceiling_percent = 100,
					.bucket_limit = DIV_ROUND_UP(2 * ETHERNET_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 2,
					.branch_priority = 1,
					.mfs_index = 1,
				},

			},
			[CXI_ETH_TC2] = {
				.eth_settings = {
					.pcp =  2,
				},
				.oxe_settings = {
					.assured_percent = 10, /* Fabric: 13 */
					.ceiling_percent = 100,
					.bucket_limit = DIV_ROUND_UP(2 * ETHERNET_MAX_FRAME_SIZE, 1024),
					.leaf_request_priority = 2,
					.branch_priority = 1,
					.mfs_index = 1,
				},

			},

		},
	}

};

/* Handle any non hardcoded QoS limits.
 * Sharing SRB space for example depends on the number of RDMA TCs
 * that need to sustain max bandwidth and whether or not a C1/C2 dev is in use.
 */
void cxi_qos_calculate_limits(struct qos_profile *qos, bool is_c2)
{
	unsigned int tc;
	unsigned int num_active_tcs = 0;
	unsigned int srb_rsvd;

	for (tc = 0; tc < CXI_TC_MAX; tc++) {
		if (!qos->tcs_active[tc])
			continue;
		num_active_tcs++;
	}

	/* Each active TC needs to sustain max bandwidth.
	 * Divide up the non-shared srb space between them.
	 */
	if (is_c2)
		srb_rsvd = C1_SRB_RESERVED / num_active_tcs;
	else
		srb_rsvd = C2_SRB_RESERVED / num_active_tcs;

	for (tc = 0; tc < CXI_TC_MAX; tc++) {
		if (!qos->tcs_active[tc])
			continue;
		qos->tcs->oxe_settings.srb_rsvd = srb_rsvd;
	}
}
