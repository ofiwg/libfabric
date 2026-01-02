/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_AV_H_
#define _CXIP_AV_H_


#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <pthread.h>
#include <ofi_list.h>
#include <ofi_atom.h>

/* Forward declarations */
struct cxip_addr;
struct cxip_coll_mc;
struct cxip_domain;
struct cxip_ep;

/* Type definitions */
struct cxip_av_auth_key_entry {
	ofi_atomic32_t use_cnt;
	ofi_atomic32_t ref_cnt;
	UT_hash_handle hh;
	struct dlist_entry entry;
	struct cxi_auth_key key;
	fi_addr_t fi_addr;
};

struct cxip_av_entry {
	ofi_atomic32_t use_cnt;
	UT_hash_handle hh;
	struct cxip_addr addr;
	fi_addr_t fi_addr;
	struct cxip_av_auth_key_entry *auth_key;
};

struct cxip_av {
	struct fid_av av_fid;
	struct cxip_domain *domain;

	/* List of endpoints bound to this AV. Each bind takes a reference
	 * as well.
	 */
	struct dlist_entry ep_list;
	ofi_atomic32_t ref;

	/* Memory used to implement lookups. Two data structures are used.
	 * 1. ibuf pool for O(1) lookup on the data path
	 * 2. hash table for O(1) on the receive path
	 */
	struct cxip_av_entry *av_entry_hash;
	struct ofi_bufpool *av_entry_pool;
	ofi_atomic32_t av_entry_cnt;

	/* Memory used to support AV authorization key. Three data structures
	 * are needed.
	 * 1. ibuf pool for memory allocation and lookup O(1) access.
	 * 2. hash table for O(1) reverse lookup
	 * 3. List for iterating
	 */
	struct cxip_av_auth_key_entry *auth_key_entry_hash;
	struct ofi_bufpool *auth_key_entry_pool;
	struct dlist_entry auth_key_entry_list;
	ofi_atomic32_t auth_key_entry_cnt;
	size_t auth_key_entry_max;

	/* Single lock is used to protect entire AV. With domain level
	 * threading, this lock is not used.
	 */
	bool lockless;
	pthread_rwlock_t lock;

	/* AV is configured as symmetric. This is an optimization which enables
	 * endpoints to use logical address.
	 */
	bool symmetric;

	/* Address vector type. */
	enum fi_av_type type;

	/* Whether or not the AV is operating in FI_AV_AUTH_KEY mode. */
	bool av_auth_key;

	/* Whether or not the AV was opened with FI_AV_USER_ID. */
	bool av_user_id;
};

struct cxip_av_set {
	struct fid_av_set av_set_fid;
	struct cxip_av *cxi_av;		// associated AV
	struct cxip_coll_mc *mc_obj;	// reference MC
	fi_addr_t *fi_addr_ary;		// addresses in set
	size_t fi_addr_cnt;		// count of addresses
	struct cxip_comm_key comm_key;	// communication key
	uint64_t flags;
};

/* Function declarations */
int cxip_av_auth_key_get_vnis(struct cxip_av *av, uint16_t **vni,
			      size_t *vni_count);

void cxip_av_auth_key_put_vnis(struct cxip_av *av, uint16_t *vni,
			       size_t vni_count);

extern struct cxip_addr *(*cxip_av_addr_in)(const void *addr);

extern void (*cxip_av_addr_out)(struct cxip_addr *addr_out,
				struct cxip_addr *addr);

int cxip_av_lookup_addr(struct cxip_av *av, fi_addr_t fi_addr,
			struct cxip_addr *addr);

fi_addr_t cxip_av_lookup_fi_addr(struct cxip_av *av,
				 const struct cxip_addr *addr);

fi_addr_t cxip_av_lookup_auth_key_fi_addr(struct cxip_av *av, unsigned int vni);

int cxip_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		 struct fid_av **av, void *context);

int cxip_av_bind_ep(struct cxip_av *av, struct cxip_ep *ep);

void cxip_av_unbind_ep(struct cxip_av *av, struct cxip_ep *ep);

int cxip_av_set(struct fid_av *av, struct fi_av_set_attr *attr,
	        struct fid_av_set **av_set_fid, void * context);

#endif /* _CXIP_AV_H_ */
