/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_MR_H
#define EFA_MR_H

#include <stdbool.h>
#include <ofi_mr.h>
#include "efa_base_ep.h"

struct efa_mr {
	struct fid_mr		mr_fid;
	struct ibv_mr		*ibv_mr;
	struct efa_domain	*domain;
	/* HMEM interface fields */
	enum fi_hmem_iface	iface;
};

int efa_mr_reg_impl(struct efa_mr *efa_mr, uint64_t flags, const struct fi_mr_attr *mr_attr);
int efa_mr_dereg_impl(struct efa_mr *efa_mr);
int efa_mr_hmem_setup(struct efa_mr *efa_mr, const struct fi_mr_attr *attr, uint64_t flags);
int efa_mr_validate_regattr(struct fid *fid, const struct fi_mr_attr *attr, uint64_t flags);

#define EFA_MR_ATTR_INIT_SYSTEM(iov, count, access, offset, requested_key, context) \
	{                                                                              \
		.mr_iov = iov,                                                             \
		.iov_count = count,                                                        \
		.access = access,                                                          \
		.offset = offset,                                                          \
		.requested_key = requested_key,                                            \
		.context = context,                                                        \
		.iface = FI_HMEM_SYSTEM,                                                   \
	}

struct efa_domain;

extern struct fi_ops_mr efa_domain_mr_ops;

int efa_mr_regattr_validate(struct fid *fid, const struct fi_mr_attr *attr, uint64_t flags);

static inline bool efa_mr_is_non_system_hmem(struct efa_mr *efa_mr)
{
	return efa_mr && (
		efa_mr->iface == FI_HMEM_CUDA ||
		efa_mr->iface == FI_HMEM_ROCR ||
		efa_mr->iface == FI_HMEM_NEURON ||
		efa_mr->iface == FI_HMEM_SYNAPSEAI);
}

static inline bool efa_mr_is_iface(struct efa_mr *efa_mr, enum fi_hmem_iface iface)
{
	return efa_mr && efa_mr->iface == iface;
}

/*
 * Return true if any descriptor in @desc (size @count) refers to
 * non-system HMEM.  A NULL @desc array is treated as no HMEM.
 */
static inline bool efa_mr_any_is_non_system_hmem(void **desc, size_t count)
{
	size_t i;

	if (!desc)
		return false;
	for (i = 0; i < count; i++)
		if (efa_mr_is_non_system_hmem(desc[i]))
			return true;
	return false;
}

static inline bool efa_mr_any_is_iface(void **desc, size_t count,
				       enum fi_hmem_iface iface)
{
	size_t i;

	if (!desc)
		return false;
	for (i = 0; i < count; i++)
		if (efa_mr_is_iface(desc[i], iface))
			return true;
	return false;
}

#define EFA_MR_IOV_LIMIT 1
#define EFA_MR_SUPPORTED_PERMISSIONS (FI_SEND | FI_RECV | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE)

int efa_mr_ofi_to_ibv_access(uint64_t ofi_access,
			     bool device_support_rdma_read,
			     bool device_support_rdma_write);

/**
 * @brief Warn about MR descriptor references in an in-flight operation
 *
 * This helper function iterates through an operation's descriptor array and
 * clears any references to the specified MR that is being closed. It logs a
 * warning when such references are found, as this indicates the MR is being
 * closed while operations are still in flight.
 *
 * @param[in,out] desc      The descriptor array
 * @param[in]     iov_count Number of IOVs in the operation
 * @param[in]     cq_entry  CQ entry with diagnostic info for the warning
 * @param[in]     efa_mr    The MR being closed
 * @param[in]     base_ep   The base ep
 */
void efa_mr_close_warn_inflight_ope(void **desc, size_t iov_count,
					       struct fi_cq_tagged_entry *cq_entry,
					       struct efa_mr *efa_mr,
					       struct efa_base_ep *base_ep);
#endif
