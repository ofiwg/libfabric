/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_RDM_DOMAIN_H
#define EFA_RDM_DOMAIN_H

#include "efa_domain.h"

struct efa_rdm_domain {
	struct efa_domain	efa_domain;

	struct fid_domain	*shm_domain;
	struct ofi_mr_cache	*cache;
	size_t			mtu_size;
	size_t			addrlen;
	size_t			rdm_cq_size;
	/* number of rdma-read messages in flight */
	uint64_t		num_read_msg_in_flight;
	/* queued op entries */
	struct dlist_entry ope_queued_list;
	/* tx/rx_entries used by long CTS msg/write/read protocol
         * which have data to be sent */
	struct dlist_entry ope_longcts_send_list;
	/* list of #efa_rdm_peer that are in backoff due to RNR */
	struct dlist_entry peer_backoff_list;
	/* list of #efa_rdm_peer that will retry posting handshake pkt */
	struct dlist_entry handshake_queued_peer_list;
	/* LRU list of AH entries in this domain */
	struct dlist_entry ah_lru_list;
};

/**
 * @brief get the extended efa_rdm_domain from a base efa_domain pointer.
 *
 * Caller must ensure the base struct is actually embedded in an efa_rdm_domain
 * (i.e. opened via efa_rdm_domain_open); calling on a direct/dgram base struct
 * yields an invalid pointer.
 *
 * @param[in]	efa_domain	base efa_domain
 * @return	pointer to the containing efa_rdm_domain
 */
static inline
struct efa_rdm_domain *efa_rdm_domain_from_efa_domain(struct efa_domain *efa_domain)
{
	return container_of(efa_domain, struct efa_rdm_domain, efa_domain);
}

/*
 * efa_is_cache_available() is a check to see whether a memory registration
 * cache is available to be used by this domain.
 *
 * Return value:
 *    return true if a memory registration cache exists in this domain.
 *    return false if a memory registration cache does not exist in this domain.
 */
static inline bool efa_is_cache_available(struct efa_rdm_domain *rdm_domain)
{
	return rdm_domain->cache;
}

int efa_rdm_domain_open(struct fid_fabric *fabric_fid, struct fi_info *info,
			struct fid_domain **domain_fid, void *context);

void efa_rdm_domain_progress_peers_and_queues(struct efa_rdm_domain *rdm_domain);

static inline void efa_rdm_domain_ope_list_lock(struct efa_rdm_domain *rdm_domain)
{
	if (efa_env.track_mr)
		ofi_genlock_lock(&rdm_domain->efa_domain.util_domain.lock);
}

static inline void efa_rdm_domain_ope_list_unlock(struct efa_rdm_domain *rdm_domain)
{
	if (efa_env.track_mr)
		ofi_genlock_unlock(&rdm_domain->efa_domain.util_domain.lock);
}

#endif
