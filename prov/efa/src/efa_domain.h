/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_DOMAIN_H
#define EFA_DOMAIN_H

#include <infiniband/verbs.h>
#include "efa_device.h"
#include "efa_hmem.h"
#include "efa_env.h"
#include "ofi_hmem.h"
#include "ofi_util.h"
#include "ofi_lock.h"
#include "ofi_atom.h"

enum efa_domain_info_type {
	EFA_INFO_RDM,
	EFA_INFO_DIRECT,
	EFA_INFO_DGRAM,
};

struct efa_domain {
	struct util_domain	util_domain;
	struct efa_device	*device;
	struct ibv_pd		*ibv_pd;
	struct fi_info		*info;
	struct efa_fabric	*fabric;
	bool 			mr_local;
	struct dlist_entry	list_entry; /* linked to g_efa_domain_list */
	struct efa_ah		*ah_map;
	/* Total count of ibv memory registrations */
	ofi_atomic64_t ibv_mr_reg_ct;
	/* Total size of memory registrations (in bytes) */
	ofi_atomic64_t ibv_mr_reg_sz;
	/* info_type is used to distinguish between the rdm, dgram and
	 * efa-direct paths */
	enum efa_domain_info_type info_type;
	/* list of enabled efa_base_ep in this domain */
	struct dlist_entry base_ep_list;
	/* Bounce buffer for 0-byte inject operations (efa-direct only) */
	void *zero_byte_bounce_buf;
	struct efa_mr *zero_byte_bounce_buf_mr;
	/*
	 * Pool backing MR structs. The pool entries (efa_mr/efa_rdm_mr) are
	 * determined by info_type at creation time. Slots are recycled on MR
	 * close and stale desc pointers from in-flight ops remain dereferenceable.
	 * For efa_rdm the embedded gen counter detects slot reuse.
	 */
	struct ofi_bufpool *mr_pool;
};

extern struct dlist_entry g_efa_domain_list;
extern ofi_mutex_t g_efa_domain_list_lock;
extern struct fi_efa_ops_domain efa_ops_domain;

/**
 * @brief domain name suffix according to endpoint type
 *
 * @param	ep_type[in]		end point type
 * @return	a string to be append to domain name
 */
static inline
const char *efa_domain_name_suffix(enum fi_ep_type ep_type)
{
	assert(ep_type == FI_EP_RDM || ep_type == FI_EP_DGRAM);
	return (ep_type == FI_EP_RDM) ? "-rdm" : "-dgrm";
}

/**
 * @brief get prov_info according to endpoint type
 *
 * @param	efa_domain[in]		EFA domain
 * @param	ep_type[in]		end point type
 * @return	pointer to prov_info
 */
static inline
struct fi_info *efa_domain_get_prov_info(struct efa_domain *efa_domain, enum fi_ep_type ep_type)
{
	assert(ep_type == FI_EP_RDM || ep_type == FI_EP_DGRAM);
	return (ep_type == FI_EP_RDM) ? efa_domain->device->rdm_info : efa_domain->device->dgram_info;
}

static inline
bool efa_domain_support_rnr_retry_modify(struct efa_domain *domain)
{
#if HAVE_CAPS_RNR_RETRY
	return domain->device->device_caps & EFADV_DEVICE_ATTR_CAPS_RNR_RETRY;
#else
	return false;
#endif
}

int efa_domain_open(struct fid_fabric *fabric_fid, struct fi_info *info,
		    struct fid_domain **domain_fid, void *context);

int efa_domain_init_device_and_pd(struct efa_domain *efa_domain,
				  const char *domain_name,
				  enum fi_ep_type ep_type);

#endif
