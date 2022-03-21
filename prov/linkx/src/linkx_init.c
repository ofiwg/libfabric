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

struct util_fabric lnx_fabric_info;

DEFINE_LIST(local_prov_table);

struct fi_fabric_attr lnx_fabric_attr = {
	.name = "linkx",
	.prov_version = OFI_VERSION_DEF_PROV
};

struct fi_domain_attr lnx_domain_attr = {
	.name = "linkx",
	.threading = FI_THREAD_SAFE,
	.control_progress = FI_PROGRESS_AUTO,
	.data_progress = FI_PROGRESS_MANUAL,
	.resource_mgmt = FI_RM_ENABLED,
	.av_type = FI_AV_UNSPEC,
	.mr_mode = FI_MR_UNSPEC,
};

struct fi_info lnx_info = {
	.caps = FI_LNX_BASIC,
	.addr_format = FI_FORMAT_UNSPEC,
	.fabric_attr = &lnx_fabric_attr,
	.domain_attr = &lnx_domain_attr,
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
	.name = "linkx",
	.version = OFI_VERSION_DEF_PROV,
	.fi_version = OFI_VERSION_LATEST,
	.getinfo = lnx_getinfo,
	.fabric = lnx_fabric,
	.cleanup = lnx_fini
};

int lnx_getinfo(uint32_t version, const char *node, const char *service,
	     uint64_t flags, const struct fi_info *hints,
	     struct fi_info **info)
{
	/* LINKx provider doesn't show up in fi_getinfo(). It can not be
	 * selected explicitly. It's a hidden provider of sorts.
	 *
	 * TODO: should it be shown in the fi_getinof() result? It doesn't
	 * make sense, since the intent is to make this an undercover
	 * provider, which is returned only as part of fi_link()
	 */
	return -FI_EOPNOTSUPP;
}

int lnx_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		void *context)
{
	/* LINKx provider's fabric is not explicitly initialized using
	 * lnx_fabric(). It gets initialized implicitly with ofi_link
	 */
	return -FI_EOPNOTSUPP;
}

void lnx_fini(void)
{
	/* TODO clean up */
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

static int lnx_add_ep_to_prov(struct local_prov *prov,
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
	struct local_prov *lprov, *new_lprov = NULL;
	struct local_prov_ep *entry = NULL;

	memset(&lnx_fabric_info, 0, sizeof(lnx_fabric_info));

	/* create the fabric for the list of providers */
	for (prov = prov_list; prov; prov = prov->next) {
		entry = calloc(sizeof(*entry), 1);
		if (!entry)
			return -FI_ENOMEM;

		new_lprov = calloc(sizeof(*new_lprov), 1);
		if (!new_lprov)
			goto free_entry;

		rc = fi_fabric(prov->fabric_attr, &entry->lpe_fabric, context);
		if (rc)
			goto free_all;

		entry->lpe_fi_info = prov;
		strncpy(entry->lpe_fabric_name, prov->fabric_attr->name,
				FI_NAME_MAX - 1);

		lprov = lnx_get_local_prov(prov->fabric_attr->prov_name);
		if (!lprov) {
			lprov = new_lprov;
			new_lprov = NULL;
			strncpy(lprov->lpv_prov_name, prov->fabric_attr->prov_name,
					FI_NAME_MAX - 1);
		} else {
			free(new_lprov);
		}

		/* indicate that this fabric can be used for on-node communication */
		if (!strncasecmp(lprov->lpv_prov_name, "shm", 3))
			entry->lpe_local = true;

		rc = lnx_add_ep_to_prov(lprov, entry);
		if (rc)
			goto free_all;

		dlist_insert_after(&lprov->lpv_entry, &local_prov_table);
	}

	rc = ofi_fabric_init(&lnx_prov, lnx_info.fabric_attr,
						 lnx_info.fabric_attr, &lnx_fabric_info, context);
	if (rc)
		goto free_all;

	lnx_fabric_info.fabric_fid.fid.ops = &lnx_fabric_fi_ops;
	lnx_fabric_info.fabric_fid.ops = &lnx_fabric_ops;
	*fabric = &lnx_fabric_info.fabric_fid;

	return 0;

free_all:
	if (new_lprov)
		free(new_lprov);
free_entry:
	if (entry)
		free(entry);

	return rc;
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
		if (rc) {
			FI_INFO(&lnx_prov, FI_LOG_CORE, "Failed to close provider %s\n",
					entry->lpv_prov_name);
		}

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


