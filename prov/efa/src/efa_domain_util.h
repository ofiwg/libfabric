/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_DOMAIN_UTIL_H
#define EFA_DOMAIN_UTIL_H

#include "efa_domain.h"

/**
 * @brief Run the shared base-domain initialization steps.
 *
 * Used by both efa_domain_open (efa-direct, dgram) and efa_rdm_domain_open
 * (efa-rdm) for setup that doesn't depend on which extended struct embeds
 * the base efa_domain.
 *
 * Caller must allocate efa_domain, initialize srx_lock, and validate the
 * info type before calling. After this returns 0 the caller installs
 * path-specific ops tables, runs path-specific init, and then calls
 * efa_domain_finalize_base.
 *
 * On failure, the caller's err_free path should call the path's close
 * function. See the TODO above efa_domain_open's err_free for the
 * partial-init fragility caveat.
 *
 * @return 0 on success, negative errno on failure.
 */
int efa_domain_init_base(struct efa_domain *efa_domain,
			 struct fid_fabric *fabric_fid,
			 struct fi_info *info,
			 void *context);

/**
 * @brief Final post-init steps shared by both open paths.
 *
 * Installs the fork-support handler (no-op on Windows) and inserts the
 * domain onto the global g_efa_domain_list under g_efa_domain_list_lock.
 *
 * @return 0 on success, negative errno on failure.
 */
int efa_domain_finalize_base(struct efa_domain *efa_domain);

/**
 * @brief Remove the domain from g_efa_domain_list.
 *
 * Always safe to call after efa_domain_init_base initialized list_entry
 * (the dlist_init self-loop makes this a no-op when the entry was never
 * inserted).
 */
void efa_domain_remove_from_global_list(struct efa_domain *efa_domain);

/**
 * @brief Destroy any AHs left in efa_domain->ah_map.
 */
void efa_domain_cleanup_ah_map(struct efa_domain *efa_domain);

void efa_domain_destruct(struct efa_domain *efa_domain);

#endif
