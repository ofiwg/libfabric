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
 *
 * Every remote endpoint known to the AV is one efa_rdm_av_entry (a "conn")
 * allocated from conn_pool. The explicit and implicit util AVs only provide the
 * raw address hash and the fi_addr allocator; their entries carry no context.
 * fi_addr -> conn goes through addr_to_entry_map (explicit) and
 * addr_to_entry_map_implicit (implicit), and (AHN, QPN) -> conn through
 * cur_reverse_av and cur_reverse_av_implicit. All four are efa_av_arrays with
 * lock-free readers. A conn keeps its address and its peer_idx from the moment
 * it enters the AV until it leaves it; promotion from the implicit to the
 * explicit AV only changes which of these maps point at it.
 *
 * Writers of the explicit maps hold util_av.lock; writers of the implicit maps
 * hold util_av_implicit.lock. Deciding that an address is in neither AV (and
 * inserting it implicitly) requires both. See
 * prov/efa/docs/efa_rdm_av_locking.md.
 */
struct efa_rdm_av {
	struct efa_av efa_av;

	struct fid_av *shm_rdm_av;
	size_t shm_used OFI_TSA_GUARDED_BY(efa_util_av_lock_sym);

	/* prv_reverse_av is a map from (ahn + qpn + connid) to all previous
	 * explicit conns, used only by the connid-aware RDM reverse lookup. */
	struct efa_prv_reverse_av *prv_reverse_av OFI_TSA_GUARDED_BY(efa_util_av_lock_sym);

	/*
	 * Owns every conn. Indexed, so ofi_buf_index() of a conn is its
	 * peer_idx: dense, and reused only after the conn is freed. Pool memory
	 * stays mapped until the AV is closed. conn_pool_lock is a leaf held
	 * only around ofi_ibuf_alloc/free: conns are created under util_av.lock
	 * (explicit) or util_av_implicit.lock (implicit), and AH eviction frees
	 * implicit conns of any AV in the domain holding only that AV's
	 * util_av_implicit.lock.
	 */
	struct ofi_genlock conn_pool_lock;
	struct ofi_bufpool *conn_pool OFI_TSA_GUARDED_BY(efa_av_conn_pool_lock_sym);

	/* implicit AV is used when receiving messages from peers not explicitly
	 * inserted by the application */
	struct util_av util_av_implicit;
	struct efa_av_array *addr_to_entry_map_implicit;
	struct efa_av_array *cur_reverse_av_implicit;
	struct efa_prv_reverse_av *prv_reverse_av_implicit OFI_TSA_GUARDED_BY(efa_implicit_av_lock_sym);

	size_t implicit_av_size;
	struct dlist_entry implicit_av_lru_list OFI_TSA_GUARDED_BY(efa_implicit_av_lock_sym);
	struct efa_ep_addr_hashable *evicted_peers_hashset OFI_TSA_GUARDED_BY(efa_implicit_av_lock_sym);
};

_Static_assert(offsetof(struct efa_rdm_av, efa_av) == 0,
	       "efa_av must be the first member of efa_rdm_av");

/**
 * @brief RDM AV entry ("conn"): one remote endpoint known to the AV
 *
 * Allocated from efa_rdm_av->conn_pool when the remote first enters the AV,
 * either explicitly through fi_av_insert() or implicitly when a packet arrives
 * from an unknown sender, and freed when it leaves the AV. The same object
 * represents the remote while it is implicit, across promotion to the explicit
 * AV, and while it is explicit, so a peer's av_entry pointer never changes.
 *
 * efa_av_entry.fi_addr and implicit_fi_addr are written under the AV locks but
 * read on the data path without them (completion source addresses, SRX
 * matching, logging), so every access goes through the atomic accessors below.
 */
struct efa_rdm_av_entry {
	struct efa_av_entry	efa_av_entry;
	struct efa_rdm_av	*av;
	fi_addr_t		implicit_fi_addr;
	fi_addr_t		shm_fi_addr;
	/*
	 * Index of this conn in every endpoint's peer_map. Assigned at
	 * allocation and immutable until the conn is freed.
	 */
	uint32_t		peer_idx;
	/*
	 * Set when the conn has been unpublished and its peers are being
	 * destroyed. Read by efa_rdm_ep_get_peer under ctrl_lock so that a
	 * creator that raced with the release does not leave a peer behind.
	 */
	bool			released;
	struct dlist_entry	implicit_av_lru_entry OFI_TSA_GUARDED_BY(efa_implicit_av_lock_sym);
	struct dlist_entry	ah_implicit_conn_list_entry OFI_TSA_GUARDED_BY(efa_util_domain_lock_sym);
};

_Static_assert(offsetof(struct efa_rdm_av_entry, efa_av_entry) == 0,
	       "efa_av_entry must be the first member of efa_rdm_av_entry");

/* fi_addr of the conn in the explicit AV, or FI_ADDR_NOTAVAIL if it is not in
 * the explicit AV. Safe without any lock. */
static inline fi_addr_t
efa_rdm_av_entry_fi_addr(const struct efa_rdm_av_entry *av_entry)
{
	return __atomic_load_n(&av_entry->efa_av_entry.fi_addr, __ATOMIC_ACQUIRE);
}

/* fi_addr of the conn in the implicit AV, or FI_ADDR_NOTAVAIL. Safe without any
 * lock, but only stable under util_av_implicit.lock. */
static inline fi_addr_t
efa_rdm_av_entry_implicit_fi_addr(const struct efa_rdm_av_entry *av_entry)
{
	return __atomic_load_n(&av_entry->implicit_fi_addr, __ATOMIC_ACQUIRE);
}

static inline bool
efa_rdm_av_entry_is_released(const struct efa_rdm_av_entry *av_entry)
{
	return __atomic_load_n(&av_entry->released, __ATOMIC_ACQUIRE);
}

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

int efa_rdm_av_reverse_av_add(struct efa_av_array *cur_reverse_av,
			      struct efa_prv_reverse_av **prv_reverse_av,
			      struct efa_av_entry *entry);

void efa_rdm_av_reverse_av_remove(struct efa_av_array *cur_reverse_av,
				  struct efa_prv_reverse_av **prv_reverse_av,
				  struct efa_av_entry *entry);

int efa_rdm_av_insert_one_implicit(struct efa_av *av, struct efa_ep_addr *addr,
				   fi_addr_t *fi_addr, uint64_t flags,
				   void *context)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_util_av_lock_sym,
			 efa_implicit_av_lock_sym);

/* fi_addr -> conn in the explicit AV. Lock free. */
static inline struct efa_rdm_av_entry *
efa_rdm_av_addr_to_entry(struct efa_av *av, fi_addr_t fi_addr)
{
	struct efa_av_entry *entry = efa_av_addr_to_entry(av, fi_addr);

	return entry ? container_of(entry, struct efa_rdm_av_entry, efa_av_entry) : NULL;
}

/* fi_addr -> conn in the implicit AV. Lock free. */
struct efa_rdm_av_entry *efa_rdm_av_addr_to_entry_implicit(struct efa_av *av,
							   fi_addr_t fi_addr);

/*
 * Reverse lookups. The _entry variants return the conn and are what the CQ read
 * path uses; the fi_addr variants are thin wrappers kept for FI_SOURCE
 * reporting and for tests.
 */
struct efa_rdm_av_entry *efa_rdm_av_reverse_lookup_entry(struct efa_av *av,
							 uint16_t ahn, uint16_t qpn,
							 struct efa_rdm_pke *pkt_entry);

struct efa_rdm_av_entry *efa_rdm_av_reverse_lookup_entry_unsafe(struct efa_av *av,
								uint16_t ahn, uint16_t qpn,
								struct efa_rdm_pke *pkt_entry)
	OFI_TSA_REQUIRES(efa_util_av_lock_sym);

struct efa_rdm_av_entry *
efa_rdm_av_reverse_lookup_entry_implicit_unsafe(struct efa_av *av,
						uint16_t ahn, uint16_t qpn,
						struct efa_rdm_pke *pkt_entry)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym);

struct efa_rdm_av_entry *efa_rdm_av_addr_lookup_entry_unsafe(struct efa_av *av,
							     struct efa_ep_addr *addr)
	OFI_TSA_REQUIRES(efa_util_av_lock_sym);

struct efa_rdm_av_entry *
efa_rdm_av_addr_lookup_entry_implicit_unsafe(struct efa_av *av,
					     struct efa_ep_addr *addr)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym);

fi_addr_t efa_rdm_av_reverse_lookup(struct efa_av *av, uint16_t ahn,
				    uint16_t qpn, struct efa_rdm_pke *pkt_entry);

fi_addr_t efa_rdm_av_reverse_lookup_unsafe(struct efa_av *av, uint16_t ahn,
				    uint16_t qpn, struct efa_rdm_pke *pkt_entry)
	OFI_TSA_REQUIRES(efa_util_av_lock_sym);

fi_addr_t efa_rdm_av_reverse_lookup_implicit_unsafe(struct efa_av *av,
						    uint16_t ahn, uint16_t qpn,
						    struct efa_rdm_pke *pkt_entry)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym);

void efa_rdm_av_implicit_av_lru_move(struct efa_av *av,
				     struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_implicit_av_lock_sym);

struct efa_rdm_av_entry *efa_rdm_av_entry_alloc_explicit(struct efa_av *av,
						   struct efa_ep_addr *raw_addr,
						   uint64_t flags, void *context)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_util_av_lock_sym);

struct efa_rdm_av_entry *efa_rdm_av_entry_alloc_implicit(struct efa_av *av,
						   struct efa_ep_addr *raw_addr,
						   uint64_t flags, void *context)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_implicit_av_lock_sym);

void efa_rdm_av_entry_release_explicit(struct efa_av *av,
				 struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_util_av_lock_sym);

void efa_rdm_av_entry_release_implicit(struct efa_av *av,
				 struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_implicit_av_lock_sym);

void efa_rdm_av_entry_release_implicit_ah_unsafe(struct efa_av *av,
					   struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_implicit_av_lock_sym);

struct efa_ah *efa_rdm_ah_alloc(struct efa_domain *domain, const uint8_t *gid,
				bool insert_implicit_av)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_rdm_ah_release(struct efa_domain *domain, struct efa_ah *ah,
			bool release_from_implicit_av)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

#endif
