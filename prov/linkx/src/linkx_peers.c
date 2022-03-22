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

static int lnx_cleanup_avs(struct local_prov *prov)
{
	int i;
	int rc, frc = 0;
	struct local_prov_ep *ep;

	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		ep = prov->lpv_prov_eps[i];
		if (!ep)
			continue;
		rc = fi_close(&ep->lpe_av->fid);
		if (rc)
			frc = rc;
	}

	return frc;
}

int lnx_av_close(struct fid *fid)
{
	int rc;
	struct local_prov *entry;
	struct lnx_peer_table *peer_tbl;

	peer_tbl = container_of(fid, struct lnx_peer_table, lpt_av_fid.fid);

	/* walk through the rest of the core providers and open their
	 * respective address vector tables
	 */
	dlist_foreach_container(&local_prov_table, struct local_prov,
							entry, lpv_entry) {
		rc = lnx_cleanup_avs(entry);
		if (rc) {
			FI_INFO(&lnx_prov, FI_LOG_CORE, "Failed to close av for %s\n",
					entry->lpv_prov_name);
		}
	}

	ofi_atomic_dec32(&peer_tbl->lpt_domain->ref);
	free(peer_tbl);

	return 0;
}

static struct fi_ops lnx_av_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = lnx_av_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int lnx_av_insert(struct fid_av *av, const void *addr, size_t count,
				  fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	return -FI_EOPNOTSUPP;
}

int lnx_av_remove(struct fid_av *av, fi_addr_t *fi_addr, size_t count,
				  uint64_t flags)
{
	return -FI_EOPNOTSUPP;
}

static const char *
lnx_av_straddr(struct fid_av *av, const void *addr,
			   char *buf, size_t *len)
{
	/* TODO: implement */
	return NULL;
}

static int
lnx_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr,
			  size_t *addrlen)
{
	/* TODO: implement */
	return -FI_EOPNOTSUPP;
}

static struct fi_ops_av lnx_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = lnx_av_insert,
	.remove = lnx_av_remove,
	.insertsvc = fi_no_av_insertsvc,
	.insertsym = fi_no_av_insertsym,
	.remove = fi_no_av_remove,
	.lookup = lnx_av_lookup,
	.straddr = lnx_av_straddr,
};

static int lnx_open_avs(struct local_prov *prov, struct fi_av_attr *attr,
						void *context)
{
	int i;
	int rc = 0;
	struct local_prov_ep *ep;

	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		ep = prov->lpv_prov_eps[i];
		if (!ep)
			continue;
		rc = fi_av_open(ep->lpe_domain, attr,
					    &ep->lpe_av, context);
		if (rc)
			return rc;
	}

	return 0;
}

int lnx_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
				struct fid_av **av, void *context)
{
	struct util_domain *util_domain;
	struct lnx_peer_table *peer_tbl;
	struct local_prov *entry;
	size_t peer_tbl_size;
	int rc = 0;

	if (!attr)
		return -FI_EINVAL;

	if (attr->name)
		return -FI_ENOSYS;

	if (attr->type == FI_AV_UNSPEC)
		attr->type = FI_AV_TABLE;

	peer_tbl_size = (sizeof(struct lnx_peer_entry) * attr->count)
	  + sizeof(*peer_tbl);

	peer_tbl = calloc(peer_tbl_size, 1);
	if (!peer_tbl)
		return -FI_ENOMEM;

	util_domain = container_of(domain, struct util_domain, domain_fid.fid);

	peer_tbl->lpt_size = attr->count;
	peer_tbl->lpt_domain = util_domain;
	peer_tbl->lpt_av_fid.fid.fclass = FI_CLASS_AV;
	peer_tbl->lpt_av_fid.fid.context = context;
	peer_tbl->lpt_av_fid.fid.ops = &lnx_av_fi_ops;
	peer_tbl->lpt_av_fid.ops = &lnx_av_ops;

	/* walk through the rest of the core providers and open their
	 * respective address vector tables
	 */
	dlist_foreach_container(&local_prov_table, struct local_prov,
							entry, lpv_entry) {
		rc = lnx_open_avs(entry, attr, context);
		if (rc) {
			FI_INFO(&lnx_prov, FI_LOG_CORE, "Failed to initialize domain for %s\n",
					entry->lpv_prov_name);
			goto failed;
		}
	}

	*av = &peer_tbl->lpt_av_fid;
	ofi_atomic_inc32(&util_domain->ref);

	return 0;

failed:
	free(peer_tbl);
	return rc;
}


