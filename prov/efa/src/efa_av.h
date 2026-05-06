/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_AV_H
#define EFA_AV_H

#include <infiniband/verbs.h>
#include "efa_ah.h"

#define EFA_MIN_AV_SIZE (16384)
#define EFA_SHM_MAX_AV_COUNT       (256)

struct efa_ep_addr {
	uint8_t			raw[EFA_GID_LEN];
	uint16_t		qpn;
	uint16_t		pad;
	uint32_t		qkey;
	struct efa_ep_addr	*next;
};

struct efa_ep_addr_hashable {
	struct efa_ep_addr addr;
	UT_hash_handle	hh;
};

#define EFA_EP_ADDR_LEN sizeof(struct efa_ep_addr)

/**
 * @brief Base AV entry (efa-direct)
 *
 * pahole:
 *   size: 48, cachelines: 1, members: 3
 *   ep_addr[32]  off=0   — TX hot (qpn@+16, qkey@+20)
 *   ah*          off=32  — TX hot
 *   fi_addr      off=40  — RX hot
 */
struct efa_av_entry {
	uint8_t			ep_addr[EFA_EP_ADDR_LEN]; /*     0    32  must be first (util_av) */
	struct efa_ah		*ah;                       /*    32     8 */
	fi_addr_t		fi_addr;                   /*    40     8 */
};

/* pahole: size: 4, no holes */
struct efa_cur_reverse_av_key {
	uint16_t ahn;
	uint16_t qpn;
};

/**
 * @brief Reverse AV entry keyed by (AHN, QPN) — points to current peer
 *
 * pahole: size: 72, cachelines: 2 (4-byte hole after key)
 */
struct efa_cur_reverse_av {
	struct efa_cur_reverse_av_key key;              /*     0     4 */
	/* 4-byte hole */
	struct efa_av_entry *av_entry;                  /*     8     8 */
	UT_hash_handle hh;                              /*    16    56 */
};

/* pahole: size: 8, no holes */
struct efa_prv_reverse_av_key {
	uint16_t ahn;
	uint16_t qpn;
	uint32_t connid;
};

/**
 * @brief Reverse AV entry keyed by (AHN, QPN, connid) — points to previous peer
 *
 * pahole: size: 72, cachelines: 2
 */
struct efa_prv_reverse_av {
	struct efa_prv_reverse_av_key key;              /*     0     8 */
	struct efa_av_entry *av_entry;                  /*     8     8 */
	UT_hash_handle hh;                              /*    16    56 */
};

/**
 * @brief Base AV — contains only what efa-direct needs
 *
 * pahole:
 *   size: 320, cachelines: 5
 *   domain*          off=0    — cacheline 0
 *   used             off=8
 *   type             off=16
 *   (4-byte hole)    off=20
 *   cur_reverse_av*  off=24   — RX hot: reverse lookup hash head
 *   prv_reverse_av*  off=32   — RX hot: QPN reuse fallback hash head
 *   util_av          off=40   — 280 bytes (contains bufpool, locks, ep_list)
 */
struct efa_av {
	struct efa_domain *domain;                      /*     0     8 */
	size_t used;                                    /*     8     8 */
	enum fi_av_type type;                           /*    16     4 */
	/* 4-byte hole */
	/* cur_reverse_av is a map from (ahn + qpn) to current (latest) efa_av_entry.
	 * prv_reverse_av is a map from (ahn + qpn + connid) to all previous efa_av_entries.
	 * cur_reverse_av is faster to search because its key size is smaller.
	 */
	struct efa_cur_reverse_av *cur_reverse_av;      /*    24     8 */
	struct efa_prv_reverse_av *prv_reverse_av;      /*    32     8 */
	struct util_av util_av;                         /*    40   280 */
};

int efa_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		struct fid_av **av_fid, void *context);

int efa_av_init_util_av(struct efa_domain *efa_domain,
			struct fi_av_attr *attr,
			struct util_av *util_av,
			void *context,
			size_t context_len);

struct efa_av_entry *efa_av_addr_to_entry(struct efa_av *av, fi_addr_t fi_addr);

fi_addr_t efa_av_reverse_lookup(struct efa_av *av, uint16_t ahn, uint16_t qpn);

int efa_av_reverse_av_add(struct efa_av *av,
			  struct efa_cur_reverse_av **cur_reverse_av,
			  struct efa_prv_reverse_av **prv_reverse_av,
			  struct efa_av_entry *av_entry);

void efa_av_reverse_av_remove(struct efa_cur_reverse_av **cur_reverse_av,
			      struct efa_prv_reverse_av **prv_reverse_av,
			      struct efa_av_entry *av_entry);

/**
 * @brief typed accessor for the ep_addr field of an AV entry
 *
 * @param[in]	entry	AV entry
 * @return	pointer to the efa_ep_addr embedded in the entry
 */
static inline struct efa_ep_addr *efa_av_entry_ep_addr(struct efa_av_entry *entry)
{
	return (struct efa_ep_addr *)entry->ep_addr;
}

/**
 * @brief check if an efa_ep_addr has a non-zero GID
 *
 * @param[in]	addr	address to check
 * @return	non-zero if valid, 0 if all-zeros
 */
static inline int efa_av_is_valid_address(struct efa_ep_addr *addr)
{
	struct efa_ep_addr all_zeros = { 0 };

	return memcmp(addr->raw, all_zeros.raw, sizeof(addr->raw));
}

#endif
