/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_RDM_FABRIC_H
#define EFA_RDM_FABRIC_H

#include <stddef.h>

#include "efa.h"

/* efa_rdm-fabric extends struct efa_fabric */
struct efa_rdm_fabric {
	struct efa_fabric	efa_fabric;

	struct fid_fabric	*shm_fabric;
#ifdef EFA_PERF_ENABLED
	struct ofi_perfset	perf_set;
#endif
};

_Static_assert(offsetof(struct efa_rdm_fabric, efa_fabric) == 0,
	       "efa_fabric must be the first member of efa_rdm_fabric for safe casting");

int efa_rdm_fabric_open(struct fi_fabric_attr *attr,
			struct fid_fabric **fabric_fid, void *context);

#ifdef EFA_PERF_ENABLED
static inline void efa_perfset_start(struct efa_rdm_ep *ep, size_t index)
{
	struct efa_domain *domain = efa_rdm_ep_domain(ep);
	struct efa_rdm_fabric *rdm_fabric = (struct efa_rdm_fabric *) domain->fabric;
	ofi_perfset_start(&rdm_fabric->perf_set, index);
}

static inline void efa_perfset_end(struct efa_rdm_ep *ep, size_t index)
{
	struct efa_domain *domain = efa_rdm_ep_domain(ep);
	struct efa_rdm_fabric *rdm_fabric = (struct efa_rdm_fabric *) domain->fabric;
	ofi_perfset_end(&rdm_fabric->perf_set, index);
}
#else
#define efa_perfset_start(ep, index) do {} while (0)
#define efa_perfset_end(ep, index) do {} while (0)
#endif

#endif /* EFA_RDM_FABRIC_H */
