/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_MR_H
#define EFA_MR_H

#include <stdbool.h>
#include <stddef.h>
#include <ofi_mr.h>

/*
 * Descriptor returned for FI_HMEM peer memory registrations
 */
struct efa_mr_peer {
	enum fi_hmem_iface  iface;
	uint64_t			device;
	uint64_t            flags;
	void                *hmem_data;
};

struct efa_mr {
	struct fid_mr		mr_fid;
	struct ibv_mr		*ibv_mr;
	struct efa_domain	*domain;
	struct efa_mr_peer	peer;
};

struct efa_rdm_mr {
	struct efa_mr		efa_mr;
	/* Used only in MR cache */
	struct ofi_mr_entry	*entry;
	/* Used only in rdm */
	struct fid_mr		*shm_mr;
	bool			inserted_to_mr_map;
	bool 			needs_sync;
};

/* Compile-time assertion to ensure safe casting between efa_mr and efa_rdm_mr */
_Static_assert(offsetof(struct efa_rdm_mr, efa_mr) == 0,
               "efa_mr must be the first member of efa_rdm_mr for safe casting");

extern int efa_mr_cache_enable;
extern size_t efa_mr_max_cached_count;
extern size_t efa_mr_max_cached_size;

int efa_mr_reg_impl(struct efa_mr *efa_mr, uint64_t flags, const struct fi_mr_attr *mr_attr);
int efa_mr_dereg_impl(struct efa_mr *efa_mr);
int efa_mr_validate_regattr(struct fid *fid, const struct fi_mr_attr *attr, uint64_t flags);

#define EFA_MR_ATTR_INIT_SYSTEM(iov, count, access, offset, requested_key, context) \
	{ \
		.mr_iov = iov, \
		.iov_count = count, \
		.access = access, \
		.offset = offset, \
		.requested_key = requested_key, \
		.context = context, \
		.iface = FI_HMEM_SYSTEM, \
	}

struct efa_domain;

extern struct fi_ops_mr efa_domain_mr_ops;
extern struct fi_ops_mr efa_domain_mr_cache_ops;
extern struct fi_ops_mr efa_rdm_domain_mr_ops;

int efa_rdm_mr_cache_entry_reg(struct ofi_mr_cache *cache,
			   struct ofi_mr_entry *entry);

void efa_rdm_mr_cache_entry_dereg(struct ofi_mr_cache *cache,
			      struct ofi_mr_entry *entry);

static inline bool efa_mr_is_hmem(struct efa_mr *efa_mr)
{
	return efa_mr && (
		efa_mr->peer.iface == FI_HMEM_CUDA ||
		efa_mr->peer.iface == FI_HMEM_ROCR ||
		efa_mr->peer.iface == FI_HMEM_NEURON ||
		efa_mr->peer.iface == FI_HMEM_SYNAPSEAI);
}

int efa_rdm_mr_cache_regv(struct fid_domain *domain_fid, const struct iovec *iov,
		      size_t count, uint64_t access, uint64_t offset,
		      uint64_t requested_key, uint64_t flags,
		      struct fid_mr **mr_fid, void *context);

int efa_rdm_mr_internal_regv(struct fid_domain *domain_fid, const struct iovec *iov,
		      size_t count, uint64_t access, uint64_t offset,
		      uint64_t requested_key, uint64_t flags,
		      struct fid_mr **mr_fid, void *context);

static inline bool efa_mr_is_cuda(struct efa_mr *efa_mr)
{
	return efa_mr ? (efa_mr->peer.iface == FI_HMEM_CUDA) : false;
}

static inline bool efa_mr_is_neuron(struct efa_mr *efa_mr)
{
	return efa_mr ? (efa_mr->peer.iface == FI_HMEM_NEURON) : false;
}

static inline bool efa_mr_is_synapseai(struct efa_mr *efa_mr)
{
	return efa_mr ? (efa_mr->peer.iface == FI_HMEM_SYNAPSEAI) : false;
}

static inline bool efa_mr_is_rocr(struct efa_mr *efa_mr)
{
	return efa_mr && efa_mr->peer.iface == FI_HMEM_ROCR;
}

#define EFA_MR_IOV_LIMIT 1
#define EFA_MR_SUPPORTED_PERMISSIONS (FI_SEND | FI_RECV | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE)

/*
 * Multiplier to give some room in the device memory registration limits
 * to allow processes added to a running job to bootstrap.
 */
#define EFA_MR_CACHE_LIMIT_MULT (.9)

int efa_mr_ofi_to_ibv_access(uint64_t ofi_access,
			     bool device_support_rdma_read,
			     bool device_support_rdma_write);

#endif
