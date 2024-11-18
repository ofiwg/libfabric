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
#include "ofi_str.h"
#include "ofi_prov.h"
#include "ofi_perf.h"
#include "ofi_hmem.h"
#include "rdma/fi_ext.h"
#include "lnx.h"

#define LNX_PASSTHRU_TX_OP_FLAGS	(FI_INJECT_COMPLETE | \
					 FI_TRANSMIT_COMPLETE | \
					 FI_DELIVERY_COMPLETE)
#define LNX_PASSTHRU_RX_OP_FLAGS	(0ULL)
#define LNX_TX_OP_FLAGS		(FI_INJECT_COMPLETE | FI_COMPLETION | \
				 FI_DELIVERY_COMPLETE | FI_TRANSMIT_COMPLETE)
#define LNX_RX_OP_FLAGS		(FI_COMPLETION)

ofi_spin_t global_bplock;
struct ofi_bufpool *global_recv_bp = NULL;

struct util_fabric lnx_fabric_info;

struct fi_tx_attr lnx_tx_attr = {
	.caps 		= ~0x0ULL,
	.op_flags	= LNX_PASSTHRU_TX_OP_FLAGS | LNX_TX_OP_FLAGS,
	.msg_order 	= ~0x0ULL,
	.comp_order 	= 0,
	.inject_size 	= SIZE_MAX,
	.size 		= SIZE_MAX,
	.iov_limit 	= LNX_IOV_LIMIT,
	.rma_iov_limit = LNX_IOV_LIMIT,
};

struct fi_rx_attr lnx_rx_attr = {
	.caps 			= ~0x0ULL,
	.op_flags		= LNX_PASSTHRU_RX_OP_FLAGS | LNX_RX_OP_FLAGS,
	.msg_order 		= ~0x0ULL,
	.comp_order 		= 0,
	.total_buffered_recv 	= 0,
	.size 			= 1024,
	.iov_limit		= LNX_IOV_LIMIT,
};

struct fi_ep_attr lnx_ep_attr = {
	.type 			= FI_EP_UNSPEC,
	.protocol 		= FI_PROTO_LNX,
	.protocol_version 	= 1,
	.max_msg_size 		= SIZE_MAX,
	.msg_prefix_size	= SIZE_MAX,
	.max_order_raw_size 	= SIZE_MAX,
	.max_order_war_size 	= SIZE_MAX,
	.max_order_waw_size 	= SIZE_MAX,
	.mem_tag_format = FI_TAG_GENERIC,
	.tx_ctx_cnt 		= SIZE_MAX,
	.rx_ctx_cnt 		= SIZE_MAX,
	.auth_key		= NULL,
	.auth_key_size		= 0,
};

struct fi_domain_attr lnx_domain_attr = {
	.name			= "ofi_lnx_domain",
	.threading 		= FI_THREAD_SAFE,
	.control_progress 	= FI_PROGRESS_AUTO,
	.data_progress 		= FI_PROGRESS_AUTO,
	.resource_mgmt 		= FI_RM_ENABLED,
	.av_type 		= FI_AV_TABLE,
	.mr_mode 		= FI_MR_RAW,
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
	.name = OFI_LNX,
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

/*
 * For the fi_getinfo() -> fi_fabric() -> fi_domain() path, we need to
 * keep track of the fi_info in case we need them later on when linking in
 * the fi_fabric() function.
 *
 * This cache gets cleared after we use the ones we need, or when the
 * library exists, if LNX is never used.
 */
struct dlist_entry lnx_fi_info_cache;
/* this is a list of all possible links */
struct dlist_entry lnx_links;
struct dlist_entry lnx_links_meta;

struct lnx_fi_cache_entry {
	struct dlist_entry entry;
	struct fi_info *fi;
};

struct lnx_fi_info_meta {
	struct dlist_entry entry;
	struct fi_info *lnx_rep;
	struct fi_info *lnx_link;
};

static int lnx_get_cache_meta(struct dlist_entry *head, int *size)
{
	int num_prov = 0;
	struct dlist_entry *e;

	dlist_foreach(head, e)
		num_prov++;

	*size = num_prov;

	return FI_SUCCESS;
}

static void lnx_free_meta(void)
{
	struct lnx_fi_info_meta *e;
	struct dlist_entry *tmp;

	dlist_foreach_container_safe(&lnx_links_meta, struct lnx_fi_info_meta, e,
				     entry, tmp) {
		dlist_remove(&e->entry);
		free(e);
	}
}

static void lnx_free_info_cache(struct dlist_entry *head, bool meta)
{
	struct lnx_fi_cache_entry *e;
	struct dlist_entry *tmp;

	dlist_foreach_container_safe(head, struct lnx_fi_cache_entry, e,
				     entry, tmp) {
		fi_freeinfo(e->fi);
		dlist_remove(&e->entry);
		free(e);
	}

	if (meta)
		lnx_free_meta();
}

static int lnx_cache_info(struct dlist_entry *head,
			  struct fi_info *info)
{
	struct lnx_fi_cache_entry *e = calloc(1, sizeof(*e));

	if (!e)
		return -FI_ENOMEM;
	dlist_init(&e->entry);
	e->fi = info;

	dlist_insert_tail(&e->entry, head);

	return 0;
}

struct fi_info *
lnx_get_link_by_dom(char *domain_name)
{
	struct fi_info *info;
	struct lnx_fi_info_meta *e;

	dlist_foreach_container(&lnx_links_meta, struct lnx_fi_info_meta, e,
				entry) {
		info = e->lnx_rep;
		if (info && info->domain_attr) {
			if (!strcmp(domain_name,
				    info->domain_attr->name)) {
				FI_INFO(&lnx_prov, FI_LOG_CORE, "Found %s\n",
						info->fabric_attr->prov_name);
				return e->lnx_link;
			}
		}
	}

	return NULL;
}

static void lnx_insert_tail(struct fi_info *head, struct fi_info *item)
{
	struct fi_info *itr = head;

	while (itr->next)
		itr = itr->next;
	itr->next = item;
}

static void lnx_remove_tail(struct fi_info **head)
{
	struct fi_info *itr = *head, *prev = NULL;

	while (itr->next) {
		prev = itr;
		itr = itr->next;
	}

	if (prev)
		prev->next = NULL;
	else
		*head = NULL;
	free(itr);
}

static struct fi_info *lnx_dupinfo_list(struct fi_info *l)
{
	struct fi_info *itr, *new, *prev = NULL, *head = NULL;

	for (itr = l; itr; itr = itr->next) {
		new = fi_dupinfo(itr);
		if (!new) {
			if (head)
				fi_freeinfo(head);
			return NULL;
		}

		if (!head)
			head = new;

		if (prev) {
			prev->next = new;
			prev = new;
		} else {
			prev = new;
		}
	}

	return head;
}

static int gen_links_rec(struct dlist_entry *current, struct dlist_entry *head,
			 struct dlist_entry *result,  struct fi_info *l,
			 int depth, int target_depth)
{
	int rc;
	struct fi_info *itr;
	struct fi_info *fi_copy, *dup;
	struct lnx_fi_cache_entry *e, *new;

	while(current->next != head) {
		e = container_of(current->next, struct lnx_fi_cache_entry, entry);
		for (itr = e->fi; itr; itr = itr->next) {
			fi_copy = fi_dupinfo(itr);
			if (l) {
				lnx_insert_tail(l, fi_copy);
			} else {
				l = fi_copy;
			}
			if (current->next->next == head &&
			    depth == target_depth) {
				dup = lnx_dupinfo_list(l);
				if (!dup)
					return -FI_ENOMEM;
				new = calloc(1, sizeof(*new));
				if (!new)
					return -FI_ENOMEM;
				new->fi = dup;
				dlist_init(&new->entry);
				dlist_insert_tail(&new->entry, result);
			}
			rc = gen_links_rec(current->next, head, result, l,
					   depth+1, target_depth);
			lnx_remove_tail(&l);
			if (rc)
				return rc;
		}
		current = current->next;
	}

	return FI_SUCCESS;
}

static int gen_links(struct dlist_entry *head, struct dlist_entry *result,
	       int target_depth)
{
	return gen_links_rec(head, head, result, NULL, 1, target_depth);
}

static int lnx_form_info(struct fi_info *fi, struct fi_info **out)
{
	int size_prov = 0, size_dom = 0, rc = FI_SUCCESS;
	struct lnx_fi_info_meta *meta = NULL;
	char *lnx_prov, *lnx_dom, *s;
	struct fi_info *itr, *r = NULL;
	bool copy = false;
	uint64_t min_inject_size = SIZE_MAX;

	for (itr = fi; itr; itr = itr->next) {
		size_prov += strlen(itr->fabric_attr->prov_name)+1;
		size_dom += strlen(itr->domain_attr->name)+1;
		if (itr->tx_attr && itr->tx_attr->inject_size < min_inject_size)
			min_inject_size = itr->tx_attr->inject_size;
	}

	lnx_dom = calloc(size_dom, sizeof(char));
	lnx_prov = calloc(size_prov, sizeof(char));
	if (!lnx_prov || !lnx_dom)
		return -FI_ENOMEM;

	for (itr = fi; itr; itr = itr->next) {
		strcat(lnx_prov, itr->fabric_attr->prov_name);
		strcat(lnx_dom, itr->domain_attr->name);
		if (itr->next) {
			strcat(lnx_dom, "+");
			strcat(lnx_prov, "+");
		}
		if (!strncmp(itr->fabric_attr->prov_name, "shm", 3))
			continue;

		if (!copy) {
			meta = calloc(1, sizeof(*meta));
			r = fi_dupinfo(itr);
			if (!r || !meta) {
				rc = -FI_ENOMEM;
				goto fail;
			}
			r->domain_attr->av_type = FI_AV_TABLE;
			meta->lnx_rep = r;
			meta->lnx_link = fi;
			if (r->tx_attr)
				r->tx_attr->inject_size = min_inject_size;
			dlist_init(&meta->entry);
			dlist_insert_tail(&meta->entry, &lnx_links_meta);
			copy = true;
		}
	}

	if (!r) {
		rc = -FI_ENODATA;
		goto fail;
	}

	free(r->fabric_attr->prov_name);
	free(r->fabric_attr->name);
	free(r->domain_attr->name);

	r->fabric_attr->name = NULL;
	r->domain_attr->name = NULL;
	r->fabric_attr->prov_name = lnx_prov;

	if (asprintf(&s, "%s", lnx_info.fabric_attr->name) < 0)
		goto fail;
	r->fabric_attr->name = s;

	if (asprintf(&s, "%s:%s", lnx_dom, lnx_info.domain_attr->name) < 0)
		goto fail;
	r->domain_attr->name = s;
	free(lnx_dom);

	*out = r;
	return FI_SUCCESS;

fail:
	if (meta)
		free(meta);
	if (r)
		fi_freeinfo(r);
	free(lnx_dom);
	return rc;
}

static int lnx_generate_info(struct fi_info **info)
{
	struct fi_info *fi = NULL, *head = NULL, *prev = NULL;
	struct lnx_fi_cache_entry *e;
	int rc, size;

	/* we need at least 2 providers to link */
	rc = lnx_get_cache_meta(&lnx_fi_info_cache, &size);
	if (rc || size < 2)
		return -FI_ENODATA;

	rc = gen_links(&lnx_fi_info_cache, &lnx_links, size);
	if (rc)
		return rc;

	/*
	 * 1. Iterate over the links and create a linked list of fi_infos
	 *    each fi_info in the list represents one of the links
	 * 2. Have metadata associated with each fi_info to refer back to
	 *    an entry in the lnx_links cache.
	 * 3. When the application selects one of these fi_infos, we can
	 *    then find the appropriate link in the cache and be able to
	 *    create the underlying core providers correctly.
	*/
	dlist_foreach_container(&lnx_links, struct lnx_fi_cache_entry, e,
				entry) {
		rc = lnx_form_info(e->fi, &fi);
		if (rc)
			goto err;

		if (prev) {
			prev->next = fi;
			prev = fi;
		} else {
			prev = fi;
			head = fi;
		}
	}

	*info = head;

	return FI_SUCCESS;

err:
	if (fi)
		fi_freeinfo(fi);
	lnx_free_info_cache(&lnx_fi_info_cache, false);
	lnx_free_info_cache(&lnx_links, true);

	return -FI_ENODATA;
}

int lnx_getinfo_helper(uint32_t version, char *prov, struct fi_info *lnx_hints)
{
	int rc;
	char *orig_prov_name = NULL;
	struct fi_info *core_info;
	uint64_t caps, mr_mode;
	bool shm = false;

	caps = lnx_hints->caps;
	mr_mode = lnx_hints->domain_attr->mr_mode;

	if (lnx_hints->fabric_attr->prov_name) {
		orig_prov_name = lnx_hints->fabric_attr->prov_name;
		lnx_hints->fabric_attr->prov_name = NULL;
	}

	lnx_hints->fabric_attr->prov_name = prov;
	if (!strncmp(prov, "shm", 3)) {
		shm = true;
		/* make sure we get the correct shm provider */
		lnx_hints->caps &= ~(FI_REMOTE_COMM | FI_LOCAL_COMM);
		lnx_hints->caps |= FI_HMEM;
		lnx_hints->domain_attr->mr_mode |= (FI_MR_VIRT_ADDR | FI_MR_HMEM
						| FI_MR_PROV_KEY);
	}
	rc = fi_getinfo(version, NULL, NULL, OFI_GETINFO_HIDDEN,
			lnx_hints, &core_info);

	lnx_hints->fabric_attr->prov_name = orig_prov_name;
	if (rc)
		return rc;

	if (shm) {
		lnx_hints->caps = caps;
		lnx_hints->domain_attr->mr_mode = mr_mode;
	}

	rc = lnx_cache_info(&lnx_fi_info_cache, core_info);

	return rc;
}

int lnx_getinfo(uint32_t version, const char *node, const char *service,
		uint64_t flags, const struct fi_info *hints,
		struct fi_info **info)
{
	int rc;
	struct fi_info *lnx_hints;
	char *linked_provs, *linked_provs_cp, *token, *exclude = NULL;

	rc = fi_param_get_str(&lnx_prov, "prov_links",
			      &linked_provs);
	if (rc)
		return rc;

	if (strstr(linked_provs, "lnx")) {
		FI_WARN(&lnx_prov, FI_LOG_FABRIC,
			"Can't specify the lnx provider as part of the link: %s\n",
			linked_provs);
		return -FI_EINVAL;
	}

	linked_provs_cp = strdup(linked_provs);
	if (!linked_provs_cp)
		return -FI_ENOMEM;

	/* The assumption is that the entire series of
	 * lnx_getinfo()->lnx_fabric()->lnx_domain()->lnx_endpoint() are
	 * going to be called before another lnx_getinfo() is called again.
	 * Based on this assumption, we will free the cache whenever
	 * lnx_getinfo() is called
	 */
	lnx_free_info_cache(&lnx_fi_info_cache, false);
	lnx_free_info_cache(&lnx_links, true);

	/* If the hints are not provided then we endup with a new block */
	lnx_hints = fi_dupinfo(hints);
	if (!lnx_hints)
		return -FI_ENOMEM;

	rc = ofi_exclude_prov_name(&lnx_hints->fabric_attr->prov_name, lnx_prov.name);
	if (rc)
		return rc;

	/* get the providers which support peer functionality. These are
	 * the only ones we can link*/
	lnx_hints->caps |= FI_PEER;

	token = strtok(linked_provs_cp, "+");
	while (token) {
		lnx_getinfo_helper(version, token, lnx_hints);
		rc = ofi_exclude_prov_name(&lnx_hints->fabric_attr->prov_name, token);
		if (rc)
			goto free_hints;
		token = strtok(NULL, "+");
	}
	free(linked_provs_cp);

	/* Generate the lnx info which represents all possible combination
	 * of domains which are to be linked.
	 */
	rc = lnx_generate_info(info);

free_hints:
	free(exclude);
	fi_freeinfo(lnx_hints);
	return rc;
}

static struct local_prov *
lnx_get_local_prov(struct dlist_entry *prov_table, char *prov_name)
{
	struct local_prov *entry;

	/* close all the open core fabrics */
	dlist_foreach_container(prov_table, struct local_prov,
				entry, lpv_entry) {
		if (!strncasecmp(entry->lpv_prov_name, prov_name, FI_NAME_MAX))
			return entry;
	}

	return NULL;
}

static int
lnx_add_ep_to_prov(struct local_prov *prov, struct local_prov_ep *ep)
{
	dlist_insert_tail(&ep->entry, &prov->lpv_prov_eps);
	ep->lpe_parent = prov;
	prov->lpv_ep_count++;

	return FI_SUCCESS;
}

static int
lnx_setup_core_prov(struct fi_info *info, struct dlist_entry *prov_table,
		    struct local_prov **shm_prov, void *context)
{
	int rc = -FI_EINVAL;
	struct local_prov_ep *ep = NULL;
	struct local_prov *lprov, *new_lprov = NULL;

	ep = calloc(sizeof(*ep), 1);
	if (!ep)
		return -FI_ENOMEM;

	new_lprov = calloc(sizeof(*new_lprov), 1);
	if (!new_lprov)
		goto free_entry;

	dlist_init(&new_lprov->lpv_prov_eps);

	rc = fi_fabric(info->fabric_attr, &ep->lpe_fabric, context);
	if (rc)
		return rc;

	ep->lpe_fi_info = info;
	strncpy(ep->lpe_fabric_name, info->fabric_attr->name,
		FI_NAME_MAX - 1);

	lprov = lnx_get_local_prov(prov_table, info->fabric_attr->prov_name);
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
		*shm_prov = lprov;
		ep->lpe_local = true;
	}

	dlist_init(&ep->entry);
	rc = lnx_add_ep_to_prov(lprov, ep);
	if (rc)
		goto free_all;

	dlist_insert_after(&lprov->lpv_entry, prov_table);

	return 0;

free_all:
	if (new_lprov)
		free(new_lprov);
free_entry:
	if (ep)
		free(ep);

	return rc;
}

int
lnx_setup_core_fabrics(char *name, struct lnx_fabric *lnx_fab,
		       void *context)
{
	int rc;
	struct fi_info *link, *itr;

	link = lnx_get_link_by_dom(name);
	if (!link)
		return -FI_ENODATA;

	for (itr = link; itr; itr = itr->next) {
		rc = lnx_setup_core_prov(itr, &lnx_fab->local_prov_table,
					 &lnx_fab->shm_prov, context);
		if (rc)
			return rc;
	}

	return FI_SUCCESS;
}

int lnx_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		void *context)
{
	struct ofi_bufpool_attr bp_attrs = {};
	struct lnx_fabric *lnx_fab;
	int rc;

	lnx_fab = calloc(sizeof(*lnx_fab), 1);
	if (!lnx_fab)
		return -FI_ENOMEM;

	bp_attrs.size = sizeof(struct lnx_mr);
	bp_attrs.alignment = 8;
	bp_attrs.max_cnt = UINT32_MAX;
	bp_attrs.chunk_cnt = 64;
	bp_attrs.flags = OFI_BUFPOOL_NO_TRACK;
	rc = ofi_bufpool_create_attr(&bp_attrs, &lnx_fab->mem_reg_bp);
	if (rc) {
		FI_WARN(&lnx_prov, FI_LOG_FABRIC,
			"Failed to create memory registration buffer pool");
		free(lnx_fab);
		return -FI_ENOMEM;
	}

	/* initialize the provider table */
	dlist_init(&lnx_fab->local_prov_table);

	rc = ofi_fabric_init(&lnx_prov, lnx_info.fabric_attr,
			     lnx_info.fabric_attr,
			     &lnx_fab->util_fabric, context);
	if (rc)
		goto fail;

	lnx_fab->util_fabric.fabric_fid.fid.ops = &lnx_fabric_fi_ops;
	lnx_fab->util_fabric.fabric_fid.ops = &lnx_fabric_ops;
	*fabric = &lnx_fab->util_fabric.fabric_fid;

	return 0;

fail:
	return rc;
}

void lnx_fini(void)
{
	lnx_free_info_cache(&lnx_fi_info_cache, false);
	lnx_free_info_cache(&lnx_links, true);
	ofi_bufpool_destroy(global_recv_bp);
}

static int lnx_free_ep(struct local_prov *prov, struct local_prov_ep *ep)
{
	int rc;

	if (!prov || !ep)
		return FI_SUCCESS;

	rc = fi_close(&ep->lpe_fabric->fid);
	fi_freeinfo(ep->lpe_fi_info);
	free(ep);
	prov->lpv_ep_count--;

	if (prov->lpv_ep_count == 0)
		dlist_remove(&prov->lpv_entry);

	return rc;
}

static int lnx_free_eps(struct local_prov *prov)
{
	int rc, frc = 0;
	struct dlist_entry *tmp;
	struct local_prov_ep *ep;

	dlist_foreach_container_safe(&prov->lpv_prov_eps,
		struct local_prov_ep, ep, entry, tmp) {
		dlist_remove(&ep->entry);
		rc = lnx_free_ep(prov, ep);
		if (rc)
			frc = rc;
	}

	return frc;
}

int lnx_fabric_close(struct fid *fid)
{
	int rc = 0;
	struct util_fabric *fabric;
	struct lnx_fabric *lnx_fab;
	struct local_prov *entry;
	struct dlist_entry *tmp;

	fabric = container_of(fid, struct util_fabric, fabric_fid.fid);
	lnx_fab = container_of(fabric, struct lnx_fabric, util_fabric);

	/* close all the open core fabrics */
	dlist_foreach_container_safe(&lnx_fab->local_prov_table,
				     struct local_prov, entry, lpv_entry, tmp) {
		dlist_remove(&entry->lpv_entry);
		rc = lnx_free_eps(entry);
		if (rc)
			FI_WARN(&lnx_prov, FI_LOG_CORE,
				"Failed to close provider %s\n",
				entry->lpv_prov_name);

		free(entry);
	}

	/* free mr registration pool */
	ofi_bufpool_destroy(lnx_fab->mem_reg_bp);

	rc = ofi_fabric_close(fabric);

	return rc;
}

void ofi_link_fini(void)
{
	lnx_prov.cleanup();
}

LNX_INI
{
	struct ofi_bufpool_attr bp_attrs = {};
	int ret;

	fi_param_define(&lnx_prov, "prov_links", FI_PARAM_STRING,
			"Specify which providers LNX will link together. Format: "
			"<prov 1>+<prov 2>+...+<prov N>. EX: shm+cxi");

	fi_param_define(&lnx_prov, "disable_shm", FI_PARAM_BOOL,
			"Turn off SHM support. Defaults to 0");

	fi_param_define(&lnx_prov, "use_srq", FI_PARAM_BOOL,
			"Turns shared receive queue support on and off. By default it is on. "
			"When SRQ is turned on some Hardware offload capability will not "
			"work. EX: Hardware Tag matching");

	dlist_init(&lnx_fi_info_cache);
	dlist_init(&lnx_links);
	dlist_init(&lnx_links_meta);

	if (!global_recv_bp) {
		bp_attrs.size = sizeof(struct lnx_rx_entry);
		bp_attrs.alignment = 8;
		bp_attrs.max_cnt = UINT16_MAX;
		bp_attrs.chunk_cnt = 64;
		bp_attrs.flags = OFI_BUFPOOL_NO_TRACK;
		ret = ofi_bufpool_create_attr(&bp_attrs, &global_recv_bp);
		if (ret) {
			FI_WARN(&lnx_prov, FI_LOG_FABRIC,
				"Failed to create receive buffer pool");
			return NULL;
		}
		ofi_spin_init(&global_bplock);
	}

	return &lnx_prov;
}
