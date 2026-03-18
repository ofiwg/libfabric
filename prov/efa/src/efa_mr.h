/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_MR_H
#define EFA_MR_H

#include <stdbool.h>
#include <ofi_mr.h>

struct efa_mr {
	struct fid_mr		mr_fid;
	struct ibv_mr		*ibv_mr;
	struct efa_domain	*domain;
	/* HMEM interface fields */
	enum fi_hmem_iface	iface;
};

int efa_mr_reg_impl(struct efa_mr *efa_mr, uint64_t flags, const struct fi_mr_attr *mr_attr);
int efa_mr_dereg_impl(struct efa_mr *efa_mr);
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

static inline bool efa_mr_is_hmem(struct efa_mr *efa_mr)
{
	return efa_mr && (
		efa_mr->iface == FI_HMEM_CUDA ||
		efa_mr->iface == FI_HMEM_ROCR ||
		efa_mr->iface == FI_HMEM_NEURON ||
		efa_mr->iface == FI_HMEM_SYNAPSEAI);
}

static inline bool efa_mr_is_cuda(struct efa_mr *efa_mr)
{
	return efa_mr ? (efa_mr->iface == FI_HMEM_CUDA) : false;
}

static inline bool efa_mr_is_neuron(struct efa_mr *efa_mr)
{
	return efa_mr ? (efa_mr->iface == FI_HMEM_NEURON) : false;
}

static inline bool efa_mr_is_synapseai(struct efa_mr *efa_mr)
{
	return efa_mr ? (efa_mr->iface == FI_HMEM_SYNAPSEAI) : false;
}

static inline bool efa_mr_is_rocr(struct efa_mr *efa_mr)
{
	return efa_mr && efa_mr->iface == FI_HMEM_ROCR;
}

#define EFA_MR_IOV_LIMIT 1
#define EFA_MR_SUPPORTED_PERMISSIONS (FI_SEND | FI_RECV | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE)

int efa_mr_ofi_to_ibv_access(uint64_t ofi_access,
			     bool device_support_rdma_read,
			     bool device_support_rdma_write);

#endif
