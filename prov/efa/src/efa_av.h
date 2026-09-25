/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_AV_H
#define EFA_AV_H

#include <infiniband/verbs.h>
#include "efa_ah.h"
#include "efa_av_array.h"
#include "efa_thread_annotations.h"

/*
 * BMI2 gives a single-instruction bit deposit (PDEP) for the reverse AV key.
 * It is only used when the build explicitly targets BMI2 (-mbmi2 or a
 * -march that implies it); otherwise the portable shift/mask form is used.
 */
#if defined(__BMI2__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define EFA_AV_HAVE_PDEP 1
#endif

#define EFA_MIN_AV_SIZE (16384)
#define EFA_SHM_MAX_AV_COUNT       (256)

struct efa_rdm_av;
struct efa_rdm_pke;

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

/* util_av implementation requires the first element of efa_av_entry to be
 * ep_addr */
struct efa_av_entry {
	uint8_t			ep_addr[EFA_EP_ADDR_LEN];
	struct efa_ah		*ah;
	fi_addr_t		fi_addr;
};

_Static_assert(offsetof(struct efa_av_entry, ep_addr) == 0,
	       "ep_addr must be the first member of efa_av_entry");

/**
 * @brief return the raw endpoint address stored in an efa_av_entry
 *
 * The raw address is stored as a byte array whose first element is required
 * to be ep_addr by the util_av implementation. This accessor provides a typed
 * view over those bytes.
 *
 * @param[in]	entry	efa_av_entry
 */
static inline struct efa_ep_addr *efa_av_entry_ep_addr(struct efa_av_entry *entry)
{
	return (struct efa_ep_addr *) entry->ep_addr;
}

/*
 * The current reverse AV is an efa_av_array indexed by a 32-bit key built from
 * the 16-bit AHN and the 16-bit QPN, so every entry in the whole 32-bit key
 * space is addressable and no key can collide.
 */
#define EFA_REVERSE_AV_MAX_IDX ((uint64_t) UINT32_MAX)

/*
 * Spread the low 16 bits of x over the even bit positions of the result, i.e.
 * b15..b0 becomes 0 b15 0 b14 ... 0 b0.
 */
static inline uint32_t efa_av_spread_bits16(uint32_t x)
{
	x &= 0x0000ffffu;
	x = (x | (x << 8)) & 0x00ff00ffu;
	x = (x | (x << 4)) & 0x0f0f0f0fu;
	x = (x | (x << 2)) & 0x33333333u;
	x = (x | (x << 1)) & 0x55555555u;
	return x;
}

/**
 * @brief build the reverse AV index for an (AHN, QPN) pair
 *
 * The two 16-bit halves are bit-interleaved (a Morton code) rather than packed
 * as (ahn << 16 | qpn). Interleaving keeps the index small whenever both halves
 * are small and keeps neighboring (ahn, qpn) pairs neighbors in the index space,
 * so the backing efa_av_array only ever allocates the handful of chunks the live
 * keys land in. Packing by shifting would instead place consecutive AHNs 64K
 * indexes apart and burn one chunk per AHN.
 *
 * @param[in]	ahn	address handle number
 * @param[in]	qpn	QP number
 */
static inline uint64_t efa_av_reverse_av_key(uint16_t ahn, uint16_t qpn)
{
#ifdef EFA_AV_HAVE_PDEP
	return _pdep_u32(ahn, 0xaaaaaaaau) | _pdep_u32(qpn, 0x55555555u);
#else
	return ((uint64_t) efa_av_spread_bits16(ahn) << 1) |
	       efa_av_spread_bits16(qpn);
#endif
}

struct efa_prv_reverse_av_key {
	uint16_t ahn;
	uint16_t qpn;
	uint32_t connid;
};

struct efa_prv_reverse_av {
	struct efa_prv_reverse_av_key key;
	struct efa_av_entry *entry;
	UT_hash_handle hh;
};

/**
 * @brief base address vector
 *
 * Holds the efa-direct-only forward and reverse AV state. The RDM layer embeds
 * this as the first member of struct efa_rdm_av (see rdm/efa_rdm_av.h) and layers on
 * the implicit AV, SHM AV, connid-aware reverse lookup and per-endpoint peer
 * maps.
 */
struct efa_av {
	struct efa_domain *domain;
	enum fi_av_type type;
	/* cur_reverse_av is a map from (ahn + qpn) to the current (latest)
	 * efa_av_entry. */
	struct efa_av_array *cur_reverse_av;
	struct util_av util_av;
	struct efa_av_array *addr_to_entry_map;
};

int efa_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		struct fid_av **av_fid, void *context);

/**
 * @brief shared fi_av_open attr prologue for the base and RDM open paths
 *
 * Validates and normalizes @p attr (name/flags rejection, count clamping to
 * EFA_MIN_AV_SIZE, FI_AV_MAP deprecation, forcing FI_AV_TABLE, universe_size
 * handling) and resolves the owning efa_domain. The calloc and init bodies
 * stay in the respective open functions.
 */
int efa_av_open_prepare_attr(struct fid_domain *domain_fid,
			     struct fi_av_attr *attr,
			     struct efa_domain **efa_domain_out);

struct efa_av_entry *efa_av_addr_to_entry(struct efa_av *av, fi_addr_t fi_addr);

struct efa_av_entry *efa_av_addr_to_entry_impl(struct efa_av_array *entry_map,
					       fi_addr_t fi_addr);

int efa_av_is_valid_address(struct efa_ep_addr *addr);

int efa_av_insert_one_validate(struct efa_ep_addr *addr, fi_addr_t *fi_addr,
			       char *raw_gid_str);

fi_addr_t efa_av_reverse_lookup(struct efa_av *av, uint16_t ahn, uint16_t qpn);

/**
 * @brief the reverse AV index of an existing AV entry
 *
 * @param[in]	entry	efa_av_entry
 */
static inline uint64_t efa_av_entry_reverse_av_key(struct efa_av_entry *entry)
{
	return efa_av_reverse_av_key(entry->ah->ahn,
				     efa_av_entry_ep_addr(entry)->qpn);
}

int efa_av_lookup(struct fid_av *av_fid, fi_addr_t fi_addr,
		  void *addr, size_t *addrlen);

const char *efa_av_straddr(struct fid_av *av_fid, const void *addr,
			   char *buf, size_t *len);

/* Allocate a current reverse AV sized for the full (ahn, qpn) key space. */
int efa_av_reverse_av_init(struct efa_av_array **cur_reverse_av);

int efa_av_reverse_av_add(struct efa_av_array *cur_reverse_av,
			  struct efa_av_entry *entry);

bool efa_av_reverse_av_remove(struct efa_av_array *cur_reverse_av,
			      struct efa_av_entry *entry);

int efa_av_init_util_av(struct efa_domain *efa_domain,
			struct fi_av_attr *attr,
			struct util_av *util_av,
			void *context,
			size_t context_len);

int efa_av_init_base(struct efa_av *av, struct efa_domain *efa_domain,
		     struct fi_av_attr *attr, void *context, size_t entry_size);

struct efa_av_entry *efa_av_entry_alloc_explicit(struct efa_av *av,
						 struct efa_ep_addr *raw_addr,
						 fi_addr_t *fi_addr_out)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

int efa_av_entry_base_construct(struct efa_av *av, struct efa_av_entry *entry,
				struct efa_ah *ah, fi_addr_t fi_addr)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_av_entry_remove_from_util_av(struct efa_av_array *entry_map,
				      struct util_av *util_av,
				      struct efa_av_entry *entry,
				      fi_addr_t fi_addr);

void efa_av_entry_release_explicit(struct efa_av *av, struct efa_av_entry *entry,
				   fi_addr_t fi_addr)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

#endif
