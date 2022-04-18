/*
 * Copyright (c) 2022 ORNL. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "config.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>

#include <rdma/fi_errno.h>
#include "ofi_util.h"
#include "ofi.h"
#include "shared/ofi_str.h"
#include "ofi_prov.h"
#include "ofi_perf.h"
#include "ofi_hmem.h"
#include "rdma/fi_ext.h"
#include "linkx.h"

#define LNX_PASSTHRU_TX_OP_FLAGS	(FI_INJECT_COMPLETE | \
					 FI_TRANSMIT_COMPLETE | \
					 FI_DELIVERY_COMPLETE)
#define LNX_PASSTHRU_RX_OP_FLAGS	(0ULL)
#define LNX_TX_OP_FLAGS		(FI_INJECT | FI_COMPLETION)
#define LNX_RX_OP_FLAGS		(FI_COMPLETION)
#define LNX_IOV_LIMIT		5

struct local_prov *shm_prov;
struct util_fabric lnx_fabric_info;

DEFINE_LIST(local_prov_table);

struct fi_tx_attr lnx_tx_attr = {
	.caps 		= ~0x0ULL,
	.op_flags	= LNX_PASSTHRU_TX_OP_FLAGS | LNX_TX_OP_FLAGS,
	.msg_order 	= ~0x0ULL,
	.comp_order 	= ~0x0ULL,
	.inject_size 	= SIZE_MAX,
	.size 		= SIZE_MAX,
	.iov_limit 	= LNX_IOV_LIMIT,
	.rma_iov_limit 	= SIZE_MAX,
};

struct fi_rx_attr lnx_rx_attr = {
	.caps 			= ~0x0ULL,
	.op_flags		= LNX_PASSTHRU_RX_OP_FLAGS | LNX_RX_OP_FLAGS,
	.msg_order 		= ~0x0ULL,
	.comp_order 		= ~0x0ULL,
	.total_buffered_recv 	= SIZE_MAX,
	.size 			= SIZE_MAX,
	.iov_limit		= SIZE_MAX,
};

struct fi_ep_attr lnx_ep_attr = {
	.type 			= FI_EP_UNSPEC,
	.protocol 		= FI_PROTO_LINKX,
	.protocol_version 	= 1,
	.max_msg_size 		= SIZE_MAX,
	.msg_prefix_size	= SIZE_MAX,
	.max_order_raw_size 	= SIZE_MAX,
	.max_order_war_size 	= SIZE_MAX,
	.max_order_waw_size 	= SIZE_MAX,
	.mem_tag_format = FI_TAG_GENERIC,
	.tx_ctx_cnt 		= SIZE_MAX,
	.rx_ctx_cnt 		= SIZE_MAX,
	.auth_key_size		= SIZE_MAX,
};

struct fi_domain_attr lnx_domain_attr = {
	.name			= "ofi_lnx_domain",
	.threading 		= FI_THREAD_SAFE,
	.control_progress 	= FI_PROGRESS_AUTO,
	.data_progress 		= FI_PROGRESS_AUTO,
	.resource_mgmt 		= FI_RM_ENABLED,
	.av_type 		= FI_AV_UNSPEC,
	.mr_mode 		= FI_MR_BASIC | FI_MR_SCALABLE | FI_MR_RAW,
	.mr_key_size		= SIZE_MAX,
	.cq_data_size 		= SIZE_MAX,
	.cq_cnt 		= SIZE_MAX,
	.ep_cnt 		= SIZE_MAX,
	.tx_ctx_cnt 		= SIZE_MAX,
	.rx_ctx_cnt 		= SIZE_MAX,
	.max_ep_tx_ctx 		= SIZE_MAX,
	.max_ep_rx_ctx 		= SIZE_MAX,
	.max_ep_stx_ctx 	= SIZE_MAX,
	.max_ep_srx_ctx 	= SIZE_MAX,
	.cntr_cnt 		= SIZE_MAX,
	.mr_iov_limit 		= SIZE_MAX,
	.caps			= ~0x0ULL,
	.auth_key_size		= SIZE_MAX,
	.max_err_data		= SIZE_MAX,
	.mr_cnt			= SIZE_MAX,
};

struct fi_fabric_attr lnx_fabric_attr = {
	.prov_version = OFI_VERSION_DEF_PROV,
	.name = "ofi_lnx_fabric",
};

struct fi_info lnx_info = {
	.caps = ~0x0ULL,
	.tx_attr = &lnx_tx_attr,
	.rx_attr = &lnx_rx_attr,
	.ep_attr = &lnx_ep_attr,
	.domain_attr = &lnx_domain_attr,
	.fabric_attr = &lnx_fabric_attr
};

static struct fi_ops lnx_fabric_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = lnx_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_fabric lnx_fabric_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = lnx_domain_open,
	.passive_ep = fi_no_passive_ep,
	.eq_open = fi_no_eq_open,
	.wait_open = fi_no_wait_open,
	.trywait = fi_no_trywait
};

struct fi_provider lnx_prov = {
	.name = OFI_UTIL_PREFIX "linkx",
	.version = OFI_VERSION_DEF_PROV,
	.fi_version = OFI_VERSION_LATEST,
	.getinfo = lnx_getinfo,
	.fabric = lnx_fabric,
	.cleanup = lnx_fini
};

struct util_prov lnx_util_prov = {
	.prov = &lnx_prov,
	.info = &lnx_info,
	.flags = 0
};

struct lnx_fi_info_cache {
	char cache_name[FI_NAME_MAX];
	struct fi_info *cache_info;
};

/*
 * For the fi_getinfo() -> fi_fabric() -> fi_domain() path, we need to
 * keep track of the fi_info in case we need them later on when linking in
 * the fi_fabric() function.
 *
 * This cache gets cleared after we use the ones we need, or when the
 * library exists, if LINKx is never used.
 */
static struct lnx_fi_info_cache lnx_fi_info_cache[LNX_MAX_LOCAL_EPS] = {0};

static void lnx_free_info_cache(void)
{
	int i;

	/* free the cache if there are any left */
	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		if (lnx_fi_info_cache[i].cache_info) {
			fi_freeinfo(lnx_fi_info_cache[i].cache_info);
			lnx_fi_info_cache[i].cache_info = NULL;
		}
	}
}

static int lnx_cache_info(struct fi_info *info, int idx)
{
	struct fi_info *prov_info;

	/* exceeded the number of supported providers */
	if (idx >= LNX_MAX_LOCAL_EPS)
		return -FI_ENODATA;

	/* stash this fi info */
	lnx_fi_info_cache[idx].cache_info = fi_dupinfo(info);
	if (!lnx_fi_info_cache[idx].cache_info)
		return -FI_ENODATA;

	prov_info = lnx_fi_info_cache[idx].cache_info;
	if (!strcmp(info->fabric_attr->prov_name, "shm"))
		snprintf(lnx_fi_info_cache[idx].cache_name, FI_NAME_MAX, "%s%d",
				 prov_info->fabric_attr->prov_name,
				 idx);
	else
		snprintf(lnx_fi_info_cache[idx].cache_name, FI_NAME_MAX, "%s",
				 prov_info->fabric_attr->prov_name);

	FI_INFO(&lnx_prov, FI_LOG_CORE, "Caching %s\n",
			prov_info->fabric_attr->prov_name);
	prov_info->next = NULL;

	return 0;
}

static struct fi_info *
lnx_get_cache_entry(char *prov_name)
{
	int i;

	/* free the cache if there are any left */
	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		struct fi_info *info = lnx_fi_info_cache[i].cache_info;

		if (info && info->fabric_attr) {
			if (!strcmp(prov_name,
						lnx_fi_info_cache[i].cache_name)) {
				lnx_fi_info_cache[i].cache_info = NULL;
				FI_INFO(&lnx_prov, FI_LOG_CORE, "Found %s\n",
						info->fabric_attr->prov_name);
				return info;
			}
		}
	}

	return NULL;
}

static int lnx_generate_info(struct fi_info *ci, struct fi_info **info,
							 int idx)
{
	struct fi_info *itr, *fi, *tail;
	char *s, *prov_name, *domain;
	int rc, num = idx, num_shm = idx, i, incr = 0;

	*info = tail = NULL;
	for (itr = ci; itr; itr = itr->next) {
		if (itr->fabric_attr->prov_name &&
			!strcmp(itr->fabric_attr->prov_name, "shm"))
			continue;

		rc = lnx_cache_info(itr, num);
		if (rc)
			goto err;

		for (i = 0; i < num_shm; i++) {
			fi = fi_dupinfo(itr);
			if (!fi)
				return -FI_ENOMEM;

			incr += i;

			free(fi->fabric_attr->name);
			domain = fi->domain_attr->name;
			prov_name = fi->fabric_attr->prov_name;

			fi->fabric_attr->name = NULL;
			fi->domain_attr->name = NULL;
			fi->fabric_attr->prov_name = NULL;

			if (asprintf(&s, "shm%d+%s", i, prov_name) < 0) {
				free(prov_name);
				fi_freeinfo(fi);
				goto err;
			}
			free(prov_name);
			fi->fabric_attr->prov_name = s;

			if (asprintf(&s, "%s_%d", lnx_info.fabric_attr->name,
						 num + incr) < 0) {
				fi_freeinfo(fi);
				goto err;
			}
			fi->fabric_attr->name = s;

			if (asprintf(&s, "shm%d+%s;%s", i, domain, lnx_info.domain_attr->name) < 0) {
				free(domain);
				fi_freeinfo(fi);
				goto err;
			}
			free(domain);
			fi->domain_attr->name = s;

			/* TODO: ofi_endpoint_init() looks at the ep_attr in detail to
			* make sure it matches between what's passed in by the user and
			* what's given by the provider. That's why we just copy the
			* provider ep_attr into what we return to the user.
			*/
			memcpy(fi->ep_attr, lnx_info.ep_attr, sizeof(*lnx_info.ep_attr));
			fi->fabric_attr->prov_version = lnx_info.fabric_attr->prov_version;

			if (!tail)
				*info = fi;
			else
				tail->next = fi;
			tail = fi;
		}

		num++;
	}

	if (num == idx)
		return -FI_ENODATA;

	return 0;

err:
	fi_freeinfo(*info);
	lnx_free_info_cache();

	return -FI_ENODATA;
}

int lnx_getinfo(uint32_t version, const char *node, const char *service,
	     uint64_t flags, const struct fi_info *hints,
	     struct fi_info **info)
{
	int rc, num;
	char *orig_prov_name = NULL;
	struct fi_info *core_info, *lnx_hints, *itr;
	uint64_t caps;

	/* If the hints are not provided then we endup with a new block */
	lnx_hints = fi_dupinfo(hints);
	if (!lnx_hints)
		return -FI_ENOMEM;

	/* get the providers which support linking */
	lnx_hints->caps |= FI_LINK;
	/* we need to lookup the shm as well, so turn off FI_REMOTE_COMM
	 * and FI_LOCAL_COMM if they are set.
	 */
	caps = lnx_hints->caps;
	lnx_hints->caps &= ~(FI_REMOTE_COMM | FI_LOCAL_COMM);

	FI_INFO(&lnx_prov, FI_LOG_FABRIC, "LINKX START -------------------\n");

	if (lnx_hints->fabric_attr->prov_name) {
		/* find the shm memory provider. There could be more than one. We need
		* to look it up ahead so we can generate all possible combination
		* between shm and other providers which it can link against.
		*/
		orig_prov_name = lnx_hints->fabric_attr->prov_name;
		lnx_hints->fabric_attr->prov_name = NULL;
	}

	lnx_hints->fabric_attr->prov_name = strdup("shm");
	rc = fi_getinfo(version, NULL, NULL, OFI_GETINFO_INTERNAL,
					lnx_hints, &core_info);
	if (rc) {
		lnx_hints->fabric_attr->prov_name = orig_prov_name;
		goto free_hints;
	}

	num = 0;
	for (itr = core_info; itr; itr = itr->next) {
		rc = lnx_cache_info(itr, num);
		num++;
	}
	free(lnx_hints->fabric_attr->prov_name);

	if (!num) {
		FI_WARN(&lnx_prov, FI_LOG_FABRIC, "No SHM provider available");
		rc = -FI_ENODATA;
		goto free_hints;
	}

	lnx_hints->fabric_attr->prov_name = orig_prov_name;

	rc = ofi_exclude_prov_name(&lnx_hints->fabric_attr->prov_name,
			lnx_prov.name);
	if (rc)
		goto free_hints;

	lnx_hints->caps = caps;
	rc = fi_getinfo(version, NULL, NULL,
				OFI_GETINFO_INTERNAL, lnx_hints,
				&core_info);
	if (rc)
		goto free_hints;

	FI_INFO(&lnx_prov, FI_LOG_FABRIC, "LINKX END -------------------\n");

	/* The list pointed to by core_info can all be coupled with shm. Note
	 * that shm will be included in that list, so we need to exclude it
	 * from the list
	 */
	rc = lnx_generate_info(core_info, info, num);

	fi_freeinfo(core_info);

free_hints:
	fi_freeinfo(lnx_hints);
	return rc;
}

static int
lnx_parse_prov_name(char *name, char **shm, char **prov)
{
	char *sub1, *sub2, *delim;

	/* the name comes in as: shm+<prov>;ofi_linkx */
	sub1 = strtok(name, ";");
	if (!sub1)
		return -FI_ENODATA;

	delim = strchr(sub1, '+');
	if (!delim)
		return -FI_ENODATA;

	sub2 = delim+1;
	*delim = '\0';

	*shm = sub1;
	*prov = sub2;

	return 0;
}

static struct local_prov *
lnx_get_local_prov(char *prov_name)
{
	struct local_prov *entry;

	/* close all the open core fabrics */
	dlist_foreach_container(&local_prov_table, struct local_prov,
							entry, lpv_entry) {
		if (!strncasecmp(entry->lpv_prov_name, prov_name, FI_NAME_MAX))
			return entry;
	}

	return NULL;
}

static int
lnx_add_ep_to_prov(struct local_prov *prov,
				   struct local_prov_ep *ep)
{
	int i;

	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		if (prov->lpv_prov_eps[i])
			continue;
		prov->lpv_prov_eps[i] = ep;
		prov->lpv_ep_count++;
		return 0;
	}

	return -FI_ENOENT;
}

static int
lnx_setup_core_prov(struct fi_info *info, void *context)
{
	int rc;
	struct local_prov_ep *entry = NULL;
	struct local_prov *lprov, *new_lprov = NULL;

	entry = calloc(sizeof(*entry), 1);
	if (!entry)
		return -FI_ENOMEM;

	new_lprov = calloc(sizeof(*new_lprov), 1);
	if (!new_lprov)
		goto free_entry;

	rc = fi_fabric(info->fabric_attr, &entry->lpe_fabric, context);
	if (rc)
		return rc;

	entry->lpe_fi_info = info;
	strncpy(entry->lpe_fabric_name, info->fabric_attr->name,
			FI_NAME_MAX - 1);

	lprov = lnx_get_local_prov(info->fabric_attr->prov_name);
	if (!lprov) {
		lprov = new_lprov;
		new_lprov = NULL;
		strncpy(lprov->lpv_prov_name, info->fabric_attr->prov_name,
				FI_NAME_MAX - 1);
	} else {
		free(new_lprov);
	}

	/* indicate that this fabric can be used for on-node communication */
	if (!strncasecmp(lprov->lpv_prov_name, "shm", 3)) {
		shm_prov = lprov;
		entry->lpe_local = true;
	}

	rc = lnx_add_ep_to_prov(lprov, entry);
	if (rc)
		goto free_all;

	dlist_insert_after(&lprov->lpv_entry, &local_prov_table);

	return 0;

free_all:
	if (new_lprov)
		free(new_lprov);
free_entry:
	if (entry)
		free(entry);

	return rc;
}

int lnx_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		void *context)
{
	struct fi_info *info = NULL;
	char *dup, *shm, *prov;
	int rc;
	/*
	 * provider: shm1+cxi;ofi_linkx
	 *     fabric: ofi_lnx_fabric_3
	 *     domain: shm1+cxi3;ofi_lnx_domain
	 *     version: 115.0
	 *     type: FI_EP_RDM
	 *     protocol: FI_PROTO_LINKX
	 *
	 * Parse out the provider name. It should be shm+<prov>
	 *
	 * Create a fabric for shm and one for the other provider.
	 *
	 * When fi_domain() is called, we get the fi_info for the
	 * second provider, which we should've returned as part of the
	 * fi_getinfo() call.
	 */
	/* create a new entry for shm.
	 * Create its fabric.
	 * insert fabric in the global table
	 */
	dup = strdup(attr->prov_name);
	rc = lnx_parse_prov_name(dup, &shm, &prov);
	if (rc)
		goto fail;

	info = lnx_get_cache_entry(shm);
	if (!info) {
		rc = -FI_ENODATA;
		goto fail;
	}

	rc = lnx_setup_core_prov(info, context);
	if (rc)
		goto fail;

	info = lnx_get_cache_entry(prov);
	if (!info) {
		rc = -FI_ENODATA;
		goto fail;
	}

	rc = lnx_setup_core_prov(info, context);
	if (rc)
		goto fail;

	memset(&lnx_fabric_info, 0, sizeof(lnx_fabric_info));

	rc = ofi_fabric_init(&lnx_prov, lnx_info.fabric_attr,
						 lnx_info.fabric_attr, &lnx_fabric_info, context);
	if (rc)
		goto fail;

	lnx_fabric_info.fabric_fid.fid.ops = &lnx_fabric_fi_ops;
	lnx_fabric_info.fabric_fid.ops = &lnx_fabric_ops;
	*fabric = &lnx_fabric_info.fabric_fid;

	free(dup);

	return 0;

fail:
	free(dup);
	fi_freeinfo(info);
	return rc;
}

void lnx_fini(void)
{
	lnx_free_info_cache();
}

static int lnx_cleanup_eps(struct local_prov *prov)
{
	int i;
	int rc, frc = 0;
	struct local_prov_ep *ep;

	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		ep = prov->lpv_prov_eps[i];
		if (!ep)
			continue;
		rc = fi_close(&ep->lpe_fabric->fid);
		if (rc)
			frc = rc;
		fi_freeinfo(ep->lpe_fi_info);
		free(ep);
		prov->lpv_prov_eps[i] = NULL;
		prov->lpv_ep_count--;
	}

	return frc;
}

int ofi_create_link(struct fi_info *prov_list,
					struct fid_fabric **fabric,
					uint64_t caps, void *context)
{
	int rc;
	struct fi_info *prov;

	/* create the fabric for the list of providers 
	 * TODO: modify the code to work with the new data structures */
	for (prov = prov_list; prov; prov = prov->next) {
		struct fi_info *info = fi_dupinfo(prov);

		if (!info)
			return -FI_ENODATA;

		rc = lnx_setup_core_prov(prov, context);
		if (rc)
			return rc;
	}

	memset(&lnx_fabric_info, 0, sizeof(lnx_fabric_info));

	rc = ofi_fabric_init(&lnx_prov, lnx_info.fabric_attr,
						 lnx_info.fabric_attr, &lnx_fabric_info, context);
	if (rc)
		return rc;

	lnx_fabric_info.fabric_fid.fid.ops = &lnx_fabric_fi_ops;
	lnx_fabric_info.fabric_fid.ops = &lnx_fabric_ops;
	*fabric = &lnx_fabric_info.fabric_fid;

	return 0;
}

int lnx_fabric_close(struct fid *fid)
{
	int rc = 0;
	struct util_fabric *fabric;
	struct local_prov *entry;
	struct dlist_entry *tmp;

	/* close all the open core fabrics */
	dlist_foreach_container_safe(&local_prov_table, struct local_prov,
								 entry, lpv_entry, tmp) {
		dlist_remove(&entry->lpv_entry);
		rc = lnx_cleanup_eps(entry);
		if (rc)
			FI_WARN(&lnx_prov, FI_LOG_CORE, "Failed to close provider %s\n",
					entry->lpv_prov_name);

		free(entry);
	}

	fabric = container_of(fid, struct util_fabric, fabric_fid.fid);
	rc = ofi_fabric_close(fabric);

	return rc;
}

void ofi_link_fini(void)
{
	lnx_prov.cleanup();
}

LNX_INI
{
	return &lnx_prov;
}
