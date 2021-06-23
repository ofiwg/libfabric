/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2020 Cray Inc. All rights reserved.
 */

#ifndef _FI_CXI_EXT_H_
#define _FI_CXI_EXT_H_

/*
 * Use CXI High Rate Puts (HRP). Increases message rate performance. Applies to
 * RMA and unreliable, non-fetching AMO operations.
 */
#define FI_CXI_HRP (1ULL << 60)

/*
 * Disable AMO reliability. Increases message rate performance. Applies to
 * non-fetching AMOs. Required for HRP AMOs.
 */
#define FI_CXI_UNRELIABLE (1ULL << 61)

/*
 * Used in conjunction with the deferred work queue API. If a deferred work
 * queue operation has this flag set, the CXI provider will ensure a counter
 * writeback occurs once the deferred work queue operation completes.
 * Note: Addition hardware resources will be used to ensure a counter writeback
 * occurs at the completion of the deferred work queue operation.
 */
#define FI_CXI_CNTR_WB (1ULL << 62)
#define FI_CXI_COUNTER_OPS "cxi_counter_ops"

struct fi_cxi_cntr_ops {
	/* Set the counter writeback address to a client provided address. */
	int (*set_wb_buffer)(struct fid *fid, void *buf, size_t len);

	/* Get the counter MMIO region. */
	int (*get_mmio_addr)(struct fid *fid, void **addr, size_t *len);
};

/* Success values cannot exceed FI_CXI_CNTR_SUCCESS_MAX */
#define FI_CXI_CNTR_SUCCESS_MAX ((1ULL << 48) - 1)

/* Failure values cannot exceed FI_CXI_CNTR_FAILURE_MAX */
#define FI_CXI_CNTR_FAILURE_MAX ((1ULL << 7) - 1)

/* fi_cntr_read() equivalent but for the writeback buffer. */
static inline uint64_t fi_cxi_cntr_wb_read(const void *wb_buf)
{
	return (*(uint64_t *)wb_buf) & FI_CXI_CNTR_SUCCESS_MAX;
};

/* fi_cntr_reader() equivalent but for the writeback buffer. */
static inline uint64_t fi_cxi_cntr_wb_readerr(const void *wb_buf)
{
	return ((*(uint64_t *)wb_buf) >> 48) & FI_CXI_CNTR_FAILURE_MAX;
};

/* Generate a counter success value which can be polled on. */
static inline int fi_cxi_gen_cntr_success(uint64_t value, uint64_t *cxi_value)
{
	if (value > FI_CXI_CNTR_SUCCESS_MAX)
		return -FI_EINVAL;

	*cxi_value = (1ULL << 63) | value;
	return FI_SUCCESS;
};

/* fi_cntr_add() equivalent but for the MMIO region. */
static inline int fi_cxi_cntr_add(void *cntr_mmio, uint64_t value)
{
	/* Success counter is only 48 bits wide. */
	if (value > FI_CXI_CNTR_SUCCESS_MAX)
		return -FI_EINVAL;

	*((uint64_t *)cntr_mmio) = value;
	return FI_SUCCESS;
}

/* fi_cntr_adderr() equivalent but for the MMIO region. */
static inline int fi_cxi_cntr_adderr(void *cntr_mmio, uint64_t value)
{
	/* Error counter is only 7 bits wide. */
	if (value > FI_CXI_CNTR_FAILURE_MAX)
		return -FI_EINVAL;

	*((uint64_t *)cntr_mmio + 8) = value;
	return FI_SUCCESS;
}

/* fi_cntr_set() equivalent but for the MMIO region. */
static inline int fi_cxi_cntr_set(void *cntr_mmio, uint64_t value)
{
	/* Only set of zero is supported through MMIO region. */
	if (value > 0)
		return -FI_EINVAL;

	*((uint64_t *)cntr_mmio + 16) = 0;
	return FI_SUCCESS;
}

/* fi_cntr_seterr() equivalent but for MMIO region. */
static inline int fi_cxi_cntr_seterr(void *cntr_mmio, uint64_t value)
{
	/* Only set of zero is supported through MMIO region. */
	if (value > 0)
		return -FI_EINVAL;

	*((uint64_t *)cntr_mmio + 24) = 0;
	return FI_SUCCESS;
}

/* fi_cntr_add() equivalent but for the MMIO region. */
static inline void *fi_cxi_get_cntr_add_addr(void *cntr_mmio)
{
	return cntr_mmio;
}

/* fi_cntr_adderr() equivalent but for the MMIO region. */
static inline void *fi_cxi_get_cntr_adderr_addr(void *cntr_mmio)
{
	return (void *)((uint64_t *)cntr_mmio + 8);
}

/* fi_cntr_set() equivalent but for the MMIO region reset.
 * NOTE: CXI does not support set to counter MMIO region. Only reset.
 */
static inline void *fi_cxi_get_cntr_reset_addr(void *cntr_mmio)
{
	return (void *)((uint64_t *)cntr_mmio + 16);
}

/* fi_cntr_seterr() equivalent but for MMIO region reset.
 * NOTE: CXI does not support set to counter MMIO region. Only reset.
 */
static inline void *fi_cxi_get_cntr_reseterr_addr(void *cntr_mmio)
{
	return (void *)((uint64_t *)cntr_mmio + 24);
}

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
