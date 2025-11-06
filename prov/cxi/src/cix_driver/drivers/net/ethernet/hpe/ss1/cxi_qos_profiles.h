/* SPDX-License-Identifier: GPL-2.0 */

/*
 * Cassini QoS Profiles
 * Copyright 2023 Hewlett Packard Enterprise Development LP
 */

#define CXI_MAX_RDMA_TCS 4
#define PCP_COUNT 8U
#define CXI_ETH_TC_MAX (CXI_ETH_TC2 + 1)

struct cass_tc_oxe_settings {
	unsigned int spt_rsvd;
	unsigned int smt_rsvd;
	unsigned int sct_rsvd;
	unsigned int srb_rsvd;
	unsigned int pbuf_rsvd;
	unsigned int assured_percent;
	unsigned int ceiling_percent;
	unsigned int bucket_limit;
	unsigned int leaf_request_priority;
	unsigned int leaf_response_priority;
	unsigned int branch_priority;
	unsigned int mfs_index;
};

struct cass_tc_dscp_pcp_settings {
	/* Required request and response PCP and DSCP for all traffic classes
	 * and traffic class types.
	 */
	unsigned int req_pcp;
	unsigned int rsp_pcp;
	unsigned int res_rsp_dscp;
	unsigned int unres_rsp_dscp;

	/* Restricted request DSCP used for CXI_TC_TYPE_DEFAULT and
	 * CXI_TC_TYPE_RESTRICTED traffic class types.
	 */
	unsigned int res_req_dscp;

	/* Unrestricted request DSCP used for CXI_TC_TYPE_DEFAULT and
	 * CXI_TC_TYPE_HRP traffic class types.
	 */
	unsigned int unres_req_dscp;

	/* Restricted request DSCP used only for CXI_TC_TYPE_HRP traffic class
	 * type. -1 disables the CXI_TC_TYPE_HRP traffic class type for a given
	 * traffic class.
	 */
	int hrp_res_req_dscp;

	/* Restricted request DSCP used only for CXI_TC_TYPE_COLL_LEAF
	 * traffic class type. -1 disables the CXI_TC_TYPE COLL_LEAF for a
	 * given traffic class.
	 */
	int coll_leaf_res_req_dscp;

	/* Unrestricted request DSCP used only for CXI_TC_TYPE_RESTRICTED
	 * traffic class type. -1 disables the CXI_TC_TYPE_RESTRICTED traffic
	 * class type for a given traffic class.
	 */
	int restricted_unres_req_dscp;
};

struct cass_tc_cq_settings {
	unsigned int dynamic_fq_count;
	unsigned int static_fq_count;
	unsigned int fq_buf_reserved;
	unsigned int pfq_high_thresh;
};

struct cass_tc_ixe_settings {
	unsigned int fq_count;
};

struct cass_tc_lpe_settings {
	unsigned int fq_count;
};

struct cass_tc_pct_settings {
	unsigned int trs_rsvd;
	unsigned int mst_rsvd;
	unsigned int tct_rsvd;
};

struct cass_tc_hni_settings {
	unsigned int pbuf_rsvd;
	unsigned int mfs;
};

enum cxi_qos_profile {
	CXI_QOS_HPC = 1, /* Leave 0 as Invalid to catch errors */
	CXI_QOS_LL_BE_BD_ET,
	CXI_QOS_LL_BE_BD_ET1_ET2,
	CXI_QOS_NUM_PROF
};

struct cass_tc_eth_settings {
	unsigned int pcp;
	bool tc_pcps[8]; /* Used only by ETH_SHARED */
};

struct cxi_tc {
	enum cxi_traffic_class tc;
	struct kobject kobj;
	unsigned int default_ocuset;
	unsigned int default_cq_tc;
	unsigned int restricted_ocuset;
	unsigned int restricted_cq_tc;
	unsigned int branch;
	unsigned int leaf[2]; /* assumes always 2 leaves per TC */
	unsigned int req_bc;

	/* RDMA Only Settings */
	struct cass_tc_dscp_pcp_settings dscp_pcp_settings;
	struct cass_tc_ixe_settings ixe_settings;
	struct cass_tc_lpe_settings lpe_settings;
	struct cass_tc_pct_settings pct_settings;

	/* Settings for RDMA + "Global" Ethernet */
	struct cass_tc_oxe_settings oxe_settings;
	struct cass_tc_cq_settings cq_settings;
	struct cass_tc_hni_settings hni_settings;

	/* Eth only settings */
	struct cass_tc_eth_settings eth_settings;
};

struct qos_profile {
	struct mutex mutex;
	bool tcs_active[CXI_ETH_TC_MAX];
	unsigned int pct_control_pcp;
	unsigned int pct_control_res_rsp_dscp;
	unsigned int pct_control_unres_rsp_dscp;
	unsigned int tc_restricted_oxe_req_bc;
	unsigned int untagged_eth_pcp;
	struct cxi_tc tcs[CXI_ETH_TC_MAX];
};

extern struct qos_profile profiles[];
int tc_cfg_show(struct seq_file *s, void *unused);

/* Check if a TC is an ethernet TC. Excludes the general CXI_ETH class */
static inline bool is_eth_tc(unsigned int tc)
{
	return (tc > CXI_TC_ETH && tc < CXI_ETH_TC_MAX);
}

void cxi_qos_calculate_limits(struct qos_profile *qos, bool is_c2);
