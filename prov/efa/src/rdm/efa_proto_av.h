/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_PROTO_AV_H
#define EFA_PROTO_AV_H

#include "efa_av.h"

struct efa_rdm_ep;
struct efa_rdm_peer;

/**
 * @brief Protocol AV entry — flat layout with same field prefix as efa_av_entry
 *
 * pahole:
 *   size: 112, cachelines: 2, members: 9
 *
 * Cache line 0 (64 bytes): hot fields
 *   ep_addr[32]        off=0   — TX hot (qpn@+16, qkey@+20)
 *   ah*                off=32  — TX hot (EFA send path)
 *   shm_fi_addr        off=40  — TX hot (SHM send path)
 *   fi_addr            off=48  — RX hot (explicit AV)
 *   implicit_fi_addr   off=56  — RX hot (implicit AV / CQ progress)
 *
 * Cache line 1 (48 bytes, cold, control path only):
 *   implicit_av_lru_entry          off=64
 *   ah_implicit_conn_list_entry    off=80
 *   ep_peer_map*                   off=96
 *   av*                            off=104 — back-pointer for AH eviction
 */
struct efa_proto_av_entry {
	uint8_t			ep_addr[EFA_EP_ADDR_LEN]; /*     0    32  must be first (util_av) */
	struct efa_ah		*ah;                       /*    32     8 */
	fi_addr_t		shm_fi_addr;               /*    40     8 */
	fi_addr_t		fi_addr;                   /*    48     8 */
	fi_addr_t		implicit_fi_addr;          /*    56     8 */
	/* --- cacheline 1 boundary (64 bytes) --- */
	struct dlist_entry	implicit_av_lru_entry;     /*    64    16 */
	struct dlist_entry	ah_implicit_conn_list_entry; /*  80    16 */
	struct efa_proto_av_entry_ep_peer_map_entry *ep_peer_map; /* 96  8 */
	struct efa_proto_av	*av;                       /*   104     8 */
};

/**
 * @brief Peer map entry — maps (ep_ptr) to efa_rdm_peer for a given AV entry
 *
 * pahole: size: 328, cachelines: 6
 */
struct efa_proto_av_entry_ep_peer_map_entry {
	struct efa_rdm_ep *ep_ptr;                     /*     0     8 */
	struct efa_rdm_peer peer;                      /*     8   264 */
	UT_hash_handle hh;                             /*   272    56 */
};

/**
 * @brief Protocol AV — embeds efa_av as first member (castable)
 *
 * pahole:
 *   size: 672, cachelines: 11, members: 10
 *
 *   efa_av                    off=0     size=320 (cachelines 0-4)
 *     domain*                 off=0     — cacheline 0
 *     cur_reverse_av*         off=24    — RX hot: explicit peer reverse lookup
 *     prv_reverse_av*         off=32    — RX hot: QPN reuse fallback
 *     util_av                 off=40    size=280
 *   --- cacheline 5 boundary (320 bytes) ---
 *   shm_rdm_av*               off=320   — control path only
 *   util_av_implicit          off=328   size=280
 *   --- cacheline 9 boundary (576 bytes) + 32 ---
 *   cur_reverse_av_implicit*  off=608   — RX hot (implicit peers only)
 *   prv_reverse_av_implicit*  off=616   — RX hot (implicit peers only)
 *   implicit_av_lru_list      off=624   — implicit RX: LRU reorder
 *   --- cacheline 10 boundary (640 bytes) ---
 *   used_implicit             off=640
 *   shm_used                  off=648
 *   implicit_av_size          off=656
 *   evicted_peers_hashset*    off=664
 *
 * RX hot path (every RX completion):
 *   efa_av.cur_reverse_av (off=24) — HASH_FIND for explicit peer reverse lookup
 *   efa_av.prv_reverse_av (off=32) — HASH_FIND fallback for QPN reuse (connid mismatch)
 *   These are in cacheline 0 — explicit peer reverse lookup stays in one line.
 *
 * RX hot path for implicit (unknown) peers:
 *   cur_reverse_av_implicit (off=608) — HASH_FIND for implicit peer reverse lookup
 *   prv_reverse_av_implicit (off=616) — HASH_FIND fallback
 *   implicit_av_lru_list (off=624) — LRU reorder on every implicit RX
 *   All three are in cacheline 9 — implicit peer reverse lookup + LRU
 *   update stays in one cache line.
 *
 * Control path only (AV insert/remove/close):
 *   shm_rdm_av, util_av_implicit, used_implicit, shm_used,
 *   implicit_av_size, evicted_peers_hashset
 */
struct efa_proto_av {
	struct efa_av		efa_av;                    /*     0   320 */
	/* --- cacheline 5 boundary (320 bytes) --- */
	struct fid_av		*shm_rdm_av;               /*   320     8 */
	/* implicit AV is used when receiving messages from peers not
	 * explicitly inserted by the application */
	struct util_av		util_av_implicit;          /*   328   280 */
	struct efa_cur_reverse_av *cur_reverse_av_implicit; /* 608   8 */
	struct efa_prv_reverse_av *prv_reverse_av_implicit; /* 616   8 */
	struct dlist_entry	implicit_av_lru_list;      /*   624    16 */
	/* --- cacheline 10 boundary (640 bytes) --- */
	size_t			used_implicit;             /*   640     8 */
	size_t			shm_used;                  /*   648     8 */
	size_t			implicit_av_size;          /*   656     8 */
	struct efa_ep_addr_hashable *evicted_peers_hashset; /* 664   8 */
};

/**
 * @brief typed accessor for the ep_addr field of a proto AV entry
 *
 * @param[in]	entry	proto AV entry
 * @return	pointer to the efa_ep_addr embedded in the entry
 */
static inline struct efa_ep_addr *
efa_proto_av_entry_ep_addr(struct efa_proto_av_entry *entry)
{
	return (struct efa_ep_addr *)entry->ep_addr;
}

/* Address lookup */
struct efa_proto_av_entry *efa_proto_av_addr_to_entry(struct efa_proto_av *av,
						      fi_addr_t fi_addr);

struct efa_proto_av_entry *efa_proto_av_addr_to_entry_implicit(
	struct efa_proto_av *av, fi_addr_t fi_addr);

/* Peer map operations */
void efa_proto_av_entry_ep_peer_map_insert(
	struct efa_proto_av_entry *entry,
	struct efa_proto_av_entry_ep_peer_map_entry *map_entry);

struct efa_rdm_peer *efa_proto_av_entry_ep_peer_map_lookup(
	struct efa_proto_av_entry *entry, struct efa_rdm_ep *ep);

void efa_proto_av_entry_ep_peer_map_remove(
	struct efa_proto_av_entry *entry, struct efa_rdm_ep *ep);

/* SHM AV operations */
int efa_proto_av_entry_insert_shm_av(struct efa_proto_av *av,
					  struct efa_proto_av_entry *entry);

void efa_proto_av_entry_deinit(struct efa_proto_av *av,
				   struct efa_proto_av_entry *entry);

/* Implicit AV LRU */
void efa_proto_av_implicit_av_lru_entry_move(struct efa_proto_av *av,
					     struct efa_proto_av_entry *entry);

/* Reverse lookup for protocol path */
fi_addr_t efa_proto_av_reverse_lookup(struct efa_proto_av *av,
					  uint16_t ahn, uint16_t qpn,
					  struct efa_rdm_pke *pkt_entry);

fi_addr_t efa_proto_av_reverse_lookup_implicit(struct efa_proto_av *av,
						   uint16_t ahn, uint16_t qpn,
						   struct efa_rdm_pke *pkt_entry);

/* Entry alloc/release */
struct efa_proto_av_entry *efa_proto_av_entry_alloc(
	struct efa_proto_av *av, struct efa_ep_addr *raw_addr,
	uint64_t flags, void *context, bool insert_shm_av,
	bool insert_implicit_av);

void efa_proto_av_entry_release(struct efa_proto_av *av,
				struct efa_proto_av_entry *entry,
				bool release_from_implicit_av);

void efa_proto_av_entry_release_ah_unsafe(struct efa_proto_av *av,
					  struct efa_proto_av_entry *entry,
					  bool release_from_implicit_av);

/* Implicit to explicit migration */
int efa_proto_av_entry_implicit_to_explicit(struct efa_proto_av *av,
					    struct efa_ep_addr *raw_addr,
					    fi_addr_t implicit_fi_addr,
					    fi_addr_t *fi_addr);

/* AV open/close/insert/remove for protocol path */
int efa_proto_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		      struct fid_av **av_fid, void *context);

int efa_proto_av_insert_one(struct efa_proto_av *av, struct efa_ep_addr *addr,
			    fi_addr_t *fi_addr, uint64_t flags, void *context,
			    bool insert_shm_av, bool insert_implicit_av);

#endif
