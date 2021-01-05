/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2020 Cray Inc. All rights reserved.
 */

#ifndef _FI_CXI_EXT_H_
#define _FI_CXI_EXT_H_

/*
 * Use CXI High Rate Puts (HRP). Increases message rate performance. Applies to
 * RMA and unreliable AMO operations.
 */
#define FI_CXI_HRP (1ULL << 60)

/*
 * Disable AMO reliability. Increases message rate performance. Applies to
 * non-fetching AMOs. Required for HRP AMOs.
 */
#define FI_CXI_UNRELIABLE (1ULL << 61)

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
 * The codes below extend this standard FI_ATOMIC set to explicitly take
 * advantage of extended hardware operations. These can be used as opcodes for
 * any of the collective operations, just like FI_MIN or FI_SUM.
 *
 * Note that the current FI_ATOMIC set ends at opcode == 19. We start this one
 * at 32, to accommodate possible expansion of the FI_ATOMIC set, and check for
 * overlap during initialization.
 */
enum cxip_coll_op {
	CXI_FI_MINMAXLOC = 32,	// FLT or INT
	CXI_FI_MINNUM,		// FLT only
	CXI_FI_MAXNUM,		// FLT only
	CXI_FI_MINMAXNUMLOC,	// FLT only
	CXI_FI_REPSUM,		// FLT only
	CXI_FI_BARRIER,		// no data
	CXI_FI_OP_LAST
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
 * The mcast_ref, mcast_id, and hwroot_idx values are provided by the Rosetta
 * configuration service, and together represent the multicast collective
 * acceleration tree set up for use by this av_set.
 *
 * @param comm_key - space to contain an intialized comm_key
 * @param mcast_ref - multicast reference id (from service)
 * @param mcast_id - 13-bit multicast address
 * @param hwroot_idx - index (rank) of hwroot in fi_av
 *
 * @return size_t size of comm_key structure initialized
 */
size_t cxip_coll_init_mcast_comm_key(struct cxip_coll_comm_key *comm_key,
				     uint32_t mcast_ref,
				     uint32_t mcast_id,
				     uint32_t hwroot_idx);

#endif /* _FI_CXI_EXT_H_ */
