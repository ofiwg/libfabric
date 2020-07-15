/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2020 Cray Inc. All rights reserved.
 */

#ifndef _FI_CXI_EXT_H_
#define _FI_CXI_EXT_H_


#define FI_CXI_DOM_OPS_1 "dom_ops_v1"

struct fi_cxi_dom_ops {
	int (*cntr_read)(struct fid *fid, unsigned int cntr, uint64_t *value,
		      struct timespec *ts);
};

/*
 * CXI Authorization Key
 */
struct cxi_auth_key {
	/* The CXI service assigned to the Domain and Endpoints. A CXI service
	 * is associated with a set of local resource limits, VNIs, and Traffic
	 * Classes.
	 *
	 * The svc_id used by an OFI Domain must match all Endpoints belonging
	 * to the Domain.
	 */
	uint32_t svc_id;

	/* The Virtual Network ID (VNI) assigned to the Endpoint. Two Endpoints
	 * must use the same VNI in order to communicate.
	 *
	 * Note that while the CXI service may define one or more VNIs which a
	 * process can access, an Endpoint is assigned to only one.
	 */
	uint16_t vni;
};

/*
 * CXI Collectives
 */

/* Extended reduction opcodes.
 *
 * Only the following standard FI_ATOMIC operations are supported:
 * - FI_MIN	: INT or FLT
 * - FI_MAX	: INT or FLT
 * - FI_SUM	: INT or FLT
 * - FI_BOR	: INT
 * - FI_BAND	: INT
 * - FI_BXOR	: INT
 *
 * The codes below extend this standard set to explicitly take advantage of
 * extended hardware operations.
 */

enum cxip_coll_op {
	CXI_COLL_BARRIER = FI_ATOMIC_OP_LAST,
	CXI_COLL_MINMAXLOC,	// FLT or INT
	CXI_COLL_MINNUM,	// FLT only
	CXI_COLL_MAXNUM,	// FLT only
	CXI_COLL_MINMAXNUMLOC,	// FLT only
	CXI_COLL_OP_LAST
};

/* The codes below define the different float sum rounding modes.
 */
enum cxip_coll_flt_sum_mode {
	CXI_COLL_FLT_SUM_DEFAULT = 0,
	CXI_COLL_FLT_SUM_NOFTZ_NEAR,
	CXI_COLL_FLT_SUM_NOFTZ_CEIL,
	CXI_COLL_FLT_SUM_NOFTZ_FLOOR,
	CXI_COLL_FLT_SUM_NOFTZ_CHOP,
	CXI_COLL_FLT_SUM_FTZ_NEAR,
	CXI_COLL_FLT_SUM_FTZ_CEIL,
	CXI_COLL_FLT_SUM_FTZ_FLOOR,
	CXI_COLL_FLT_SUM_FTZ_CHOP,
	CXI_COLL_FLT_REPSUM,
	CXI_COLL_FLT_LAST,
};

/*
 * Exported comm_key structure. Use initializion routines below to prepare this.
 *
 * The address to a comm_key structure can be passed through the info->comm_key
 * structure when initializing an av_set. It is copied to the av_set structure
 * and can be reused or freed after the av_set is created.
 */
struct cxip_coll_comm_key {
	uint8_t data[64];
};

/**
 * Initialize a multicast comm_key structure.
 *
 * The mcast_id and hwroot_nic values are provided by the Rosetta configuration
 * service, and together represent the multicast collective acceleration tree
 * set up for use by this av_set.
 *
 * @param comm_key - space to contain an intialized comm_key
 * @param round - floating-point sum rounding mode (0 = no-flush, nearest)
 * @param mcast_id - 13-bit multicast address
 * @param hwroot_nic 20-bit NIC address of the hardware root node
 *
 * @return size_t size of comm_key structure initialized
 */
size_t cxip_coll_init_mcast_comm_key(struct cxip_coll_comm_key *comm_key,
				     enum cxip_coll_flt_sum_mode round,
				     uint32_t mcast_id,
				     uint32_t hwroot_nic);

#endif /* _FI_CXI_EXT_H_ */
