/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_RDM_AV_H
#define EFA_RDM_AV_H

#include "ofi_util.h"
#include "../efa_av.h"
#include "efa_thread_annotations.h"

struct efa_rdm_pke;

/**
 * @brief RDM address vector
 *
 * Embeds the base efa_av as its first member and adds the RDM-only state: the
 * implicit AV (peers that send to us before the application inserts them), the
 * connid-aware previous-connection reverse maps, the SHM sub-AV, the implicit
 * AV LRU eviction list and the evicted-peers hashset.
 */
struct efa_rdm_av {
	struct efa_av efa_av;

	struct fid_av *shm_rdm_av;
	size_t shm_used;

	/* prv_reverse_av is a map from (ahn + qpn + connid) to all previous
	 * efa_av_entries, used only by the connid-aware RDM reverse lookup. */
	struct efa_prv_reverse_av *prv_reverse_av OFI_TSA_GUARDED_BY(efa_util_av_lock_sym);

	/* implicit AV is used when receiving messages from peers not explicitly
	 * inserted by the application */
	struct util_av util_av_implicit;
	struct efa_av_array *addr_to_entry_map_implicit;
	struct efa_cur_reverse_av *cur_reverse_av_implicit;
	struct efa_prv_reverse_av *prv_reverse_av_implicit;

	size_t implicit_av_size;
	struct dlist_entry implicit_av_lru_list OFI_TSA_GUARDED_BY(efa_implicit_av_lock_sym);
	struct efa_ep_addr_hashable *evicted_peers_hashset OFI_TSA_GUARDED_BY(efa_implicit_av_lock_sym);
};

_Static_assert(offsetof(struct efa_rdm_av, efa_av) == 0,
	       "efa_av must be the first member of efa_rdm_av");

/**
 * @brief RDM address vector entry
 *
 * Embeds the base efa_av_entry as its first member and adds the RDM-only state.
 */
struct efa_rdm_av_entry {
	struct efa_av_entry	efa_av_entry;
	struct efa_rdm_av	*av;
	fi_addr_t		implicit_fi_addr;
	fi_addr_t		shm_fi_addr;
	struct dlist_entry	implicit_av_lru_entry;
	struct dlist_entry	ah_implicit_conn_list_entry OFI_TSA_GUARDED_BY(efa_util_domain_lock_sym);
};

_Static_assert(offsetof(struct efa_rdm_av_entry, efa_av_entry) == 0,
	       "efa_av_entry must be the first member of efa_rdm_av_entry");

/**
 * @brief RDM address handle
 *
 * Embeds the base efa_ah as its first member and adds the RDM-only split
 * reference counts, the list of implicit AV entries using this AH, and the
 * position in the domain's AH LRU list used for out-of-memory eviction.
 */
struct efa_rdm_ah {
	struct efa_ah	efa_ah;
	/* Number of explicit AV entries associated with this AH */
	int explicit_refcnt OFI_TSA_GUARDED_BY(efa_util_domain_lock_sym);
	/* Number of implicit AV entries associated with this AH */
	int implicit_refcnt OFI_TSA_GUARDED_BY(efa_util_domain_lock_sym);
	/* dlist of all implicit AV entries associated with this AH entry */
	struct dlist_entry implicit_conn_list OFI_TSA_GUARDED_BY(efa_util_domain_lock_sym);
	/* dlist entry in domain's LRU AH list */
	struct dlist_entry domain_lru_ah_list_entry OFI_TSA_GUARDED_BY(efa_util_domain_lock_sym);
};

_Static_assert(offsetof(struct efa_rdm_ah, efa_ah) == 0,
	       "efa_ah must be the first member of efa_rdm_ah");

int efa_rdm_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		    struct fid_av **av_fid, void *context);

int efa_rdm_av_reverse_av_add(struct efa_cur_reverse_av **cur_reverse_av,
			      struct efa_prv_reverse_av **prv_reverse_av,
			      struct efa_av_entry *entry);

void efa_rdm_av_reverse_av_remove(struct efa_cur_reverse_av **cur_reverse_av,
				  struct efa_prv_reverse_av **prv_reverse_av,
				  struct efa_av_entry *entry);

int efa_rdm_av_insert_one_implicit(struct efa_av *av, struct efa_ep_addr *addr,
				   fi_addr_t *fi_addr, uint64_t flags,
				   void *context)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

struct efa_rdm_av_entry *efa_rdm_av_addr_to_entry_implicit(struct efa_av *av,
							   fi_addr_t fi_addr);

fi_addr_t efa_rdm_av_reverse_lookup(struct efa_av *av, uint16_t ahn,
				    uint16_t qpn, struct efa_rdm_pke *pkt_entry);

fi_addr_t efa_rdm_av_reverse_lookup_implicit(struct efa_av *av, uint16_t ahn,
					     uint16_t qpn,
					     struct efa_rdm_pke *pkt_entry);

void efa_rdm_av_implicit_av_lru_move(struct efa_av *av,
				     struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym);

struct efa_rdm_av_entry *efa_rdm_av_entry_alloc_explicit(struct efa_av *av,
						   struct efa_ep_addr *raw_addr,
						   uint64_t flags, void *context)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

struct efa_rdm_av_entry *efa_rdm_av_entry_alloc_implicit(struct efa_av *av,
						   struct efa_ep_addr *raw_addr,
						   uint64_t flags, void *context)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_rdm_av_entry_release_explicit(struct efa_av *av,
				 struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_util_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_rdm_av_entry_release_implicit(struct efa_av *av,
				 struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_rdm_av_entry_release_implicit_ah_unsafe(struct efa_av *av,
					   struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_rdm_ah_implicit_av_lru_ah_move(struct efa_domain *domain,
					struct efa_ah *ah)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

struct efa_ah *efa_rdm_ah_alloc(struct efa_domain *domain, const uint8_t *gid,
				bool insert_implicit_av)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_rdm_ah_release(struct efa_domain *domain, struct efa_ah *ah,
			bool release_from_implicit_av)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

#endif
