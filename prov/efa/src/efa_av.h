/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_AV_H
#define EFA_AV_H

#include <infiniband/verbs.h>
#include "efa_ah.h"
#include "efa_av_array.h"
#include "efa_thread_annotations.h"

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
 */
static inline struct efa_ep_addr *efa_av_entry_ep_addr(struct efa_av_entry *entry)
{
	return (struct efa_ep_addr *) entry->ep_addr;
}

struct efa_cur_reverse_av_key {
	uint16_t ahn;
	uint16_t qpn;
};

struct efa_cur_reverse_av {
	struct efa_cur_reverse_av_key key;
	struct efa_av_entry *entry;
	UT_hash_handle hh;
};

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
	struct efa_cur_reverse_av *cur_reverse_av OFI_TSA_GUARDED_BY(efa_util_av_lock_sym);
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

int efa_av_lookup(struct fid_av *av_fid, fi_addr_t fi_addr,
		  void *addr, size_t *addrlen);

const char *efa_av_straddr(struct fid_av *av_fid, const void *addr,
			   char *buf, size_t *len);

int efa_av_reverse_av_add(struct efa_cur_reverse_av **cur_reverse_av,
			  struct efa_av_entry *entry);

bool efa_av_reverse_av_remove(struct efa_cur_reverse_av **cur_reverse_av,
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

void efa_av_entry_remove_from_util_av(struct efa_av_array *entry_map,
				      struct util_av *util_av,
				      struct efa_av_entry *entry,
				      fi_addr_t fi_addr);

void efa_av_entry_release_explicit(struct efa_av *av, struct efa_av_entry *entry,
				   fi_addr_t fi_addr)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

#endif
