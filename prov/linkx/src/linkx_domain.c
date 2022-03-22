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

static struct fi_ops_domain lnx_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = lnx_av_open,
	.cq_open = lnx_cq_open,
	.endpoint = lnx_endpoint,
	.scalable_ep = lnx_scalable_ep,
	.cntr_open = fi_no_cntr_open,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = fi_no_query_atomic,
	.query_collective = fi_no_query_collective,
};

static int lnx_cleanup_domains(struct local_prov *prov)
{
	int i, rc, frc = 0;
	struct local_prov_ep *ep;

	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		ep = prov->lpv_prov_eps[i];
		if (!ep)
			continue;
		rc = fi_close(&ep->lpe_domain->fid);
		if (rc)
			frc = rc;
	}

	return frc;
}

static int lnx_domain_close(fid_t fid)
{
	int rc = 0;
	struct local_prov *entry;
	struct util_domain *domain;

	/* close all the open core domains */
	dlist_foreach_container(&local_prov_table, struct local_prov,
							entry, lpv_entry) {
		rc = lnx_cleanup_domains(entry);
		if (rc)
			FI_INFO(&lnx_prov, FI_LOG_CORE, "Failed to close domain for %s\n",
					entry->lpv_prov_name);
	}

	domain = container_of(fid, struct util_domain, domain_fid.fid);
	rc = ofi_domain_close(domain);

	free(domain);

	return rc;
}

static struct fi_ops lnx_domain_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = lnx_domain_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_mr lnx_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = fi_no_mr_reg,
	.regv = fi_no_mr_regv,
	.regattr = fi_no_mr_regattr,
};

static int lnx_open_core_domains(struct local_prov *prov,
								 void *context)
{
	int i, rc;
	struct local_prov_ep *ep;

	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		ep = prov->lpv_prov_eps[i];
		if (!ep)
			continue;
		rc = fi_domain(ep->lpe_fabric, ep->lpe_fi_info,
					   &ep->lpe_domain, context);
		if (rc)
			return rc;
	}

	return 0;
}

int lnx_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **domain, void *context)
{
	int rc = 0;
	struct local_prov *entry;
	struct util_domain *lnx_domain_info;

	lnx_domain_info = calloc(sizeof(*lnx_domain_info), 1);
	if (!lnx_domain_info)
		return FI_ENOMEM;

	dlist_foreach_container(&local_prov_table, struct local_prov,
							entry, lpv_entry) {
		rc = lnx_open_core_domains(entry, context);
		if (rc) {
			FI_INFO(&lnx_prov, FI_LOG_CORE, "Failed to initialize domain for %s\n",
					entry->lpv_prov_name);
			return rc;
		}
	}

	rc = ofi_domain_init(fabric, info, lnx_domain_info, context,
			      OFI_DOMAIN_SPINLOCK);
	if (rc)
		return rc;

	lnx_domain_info->threading = FI_THREAD_SAFE;

	lnx_domain_info->domain_fid.fid.ops = &lnx_domain_fi_ops;
	lnx_domain_info->domain_fid.ops = &lnx_domain_ops;
	lnx_domain_info->domain_fid.mr = &lnx_mr_ops;
	*domain = &lnx_domain_info->domain_fid;

	return 0;
}

