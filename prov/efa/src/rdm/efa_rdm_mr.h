/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_RDM_MR_H
#define EFA_RDM_MR_H

#include "efa.h"

/* RDM MR cache functions */
int efa_rdm_mr_cache_open(struct ofi_mr_cache **cache, struct efa_domain *domain);
int efa_rdm_mr_cache_entry_reg(struct ofi_mr_cache *cache,
			       struct ofi_mr_entry *entry);
void efa_rdm_mr_cache_entry_dereg(struct ofi_mr_cache *cache,
				  struct ofi_mr_entry *entry);
int efa_rdm_mr_cache_regv(struct fid_domain *domain_fid, const struct iovec *iov,
			  size_t count, uint64_t access, uint64_t offset,
			  uint64_t requested_key, uint64_t flags,
			  struct fid_mr **mr, void *context);

/* RDM MR operations */
extern struct fi_ops_mr efa_rdm_domain_mr_ops;

/* RDM MR internal functions */
int efa_rdm_mr_internal_regv(struct fid_domain *domain_fid, const struct iovec *iov,
			     size_t count, uint64_t access, uint64_t offset,
			     uint64_t requested_key, uint64_t flags,
			     struct fid_mr **mr_fid, void *context);

#endif /* EFA_RDM_MR_H */