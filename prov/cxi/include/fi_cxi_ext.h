/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2020 Cray Inc. All rights reserved.
 * Copyright (c) 2021-2022 Hewlett Packard Enterprise Development LP
 *
 */

#ifndef _FI_CXI_EXT_H_
#define _FI_CXI_EXT_H_

/*
 * TODO: The following should be integrated into the include/rdma/fi_ext.h
 * and are use for provider specific fi_control() operations.
 */
#define FI_PROV_SPECIFIC_CXI	(0xccc << 16)

enum {
	FI_OPT_CXI_SET_TCLASS = -FI_PROV_SPECIFIC_CXI,	/* uint32_t */
	FI_OPT_CXI_SET_MSG_ORDER,			/* uint64_t */
};

/*
 * Execute a given libfabric atomic memory operation as a PCIe operation as
 * compared to a NIC operation.
 *
 * Note: Ordering between PCIe atomic operations and NIC atomic/RMA operations
 * is undefined.
 *
 * Note: This flag overloads the bit used for FI_SOURCE. But, since FI_SOURCE
 * is invalid for AMO operations, overloading this bit is not an issue.
 */
#define FI_CXI_PCIE_AMO (1ULL << 57)

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
 * Request a provider specific weak FENCE operation to facilitate an
 * EP alias ordering point, when the original EP utilizes PCIe RO=1.
 */
#define FI_CXI_WEAK_FENCE (1ULL << 63)

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
#define FI_CXI_DOM_OPS_2 "dom_ops_v2"
#define FI_CXI_DOM_OPS_3 "dom_ops_v3"

/* v1 and v2 can use the same struct since v2 only appended a routine */
struct fi_cxi_dom_ops {
	int (*cntr_read)(struct fid *fid, unsigned int cntr, uint64_t *value,
		      struct timespec *ts);
	int (*topology)(struct fid *fid, unsigned int *group_id,
			unsigned int *switch_id, unsigned int *port_id);

	/* Enable hybrid MR desc mode. Hybrid MR desc allows for libfabric users
	 * to optionally pass in a valid MR desc for local communication
	 * operations.
	 *
	 * When enabled, if the MR desc is NULL, the provider will
	 * perform internal memory registration. Else, the provider will assume
	 * the MR desc field is valid and skip internal memory registration.
	 *
	 * When disabled, the provider will ignore the MR desc field and always
	 * perform internal memory registration. This is the default behavior.
	 *
	 * All child endpoints will inherit the current domain status of hybrid
	 * MR desc only during endpoint creation. Dynamically changing the
	 * domain hybrid MR desc status with endpoint allocate may not propagate
	 * to child endpoints. Thus, it is recommended to set hybrid MR desc
	 * status prior to allocating endpoints.
	 */
	int (*enable_hybrid_mr_desc)(struct fid *fid, bool enable);
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

#endif /* _FI_CXI_EXT_H_ */
