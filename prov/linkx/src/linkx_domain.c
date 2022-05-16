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

static int
lnx_core_mr_regattr(struct lnx_mem_desc *mem_desc,
					const struct fi_mr_attr *attr, uint64_t flags,
					int idx)
{
	struct local_prov_ep *ep = mem_desc->ep[idx];
	struct fi_mr_attr *core_attr = ofi_dup_mr_attr(attr);
	int rc;

	/* Look at:
	 * https://ofiwg.github.io/libfabric/v1.15.0/man/fi_mr.3.html
	 */
	if (ep->lpe_fi_info->domain_attr->mr_mode == 0 ||
		ep->lpe_fi_info->domain_attr->mr_mode & FI_MR_ENDPOINT)
		return 0;
	core_attr->iface = ofi_get_hmem_iface(attr->mr_iov->iov_base,
										  &core_attr->device.reserved,
										  &flags);
	rc = fi_mr_regattr(ep->lpe_domain, core_attr,
					   flags, &mem_desc->core_mr[idx]);

	free(core_attr);

	return rc;
}

static int
lnx_mr_regattrs_all(struct local_prov *prov, const struct fi_mr_attr *attr,
					uint64_t flags, struct lnx_mem_desc *mem_desc)
{
	int i, rc = 0;
	struct local_prov_ep *ep;

	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		ep = prov->lpv_prov_eps[i];
		if (!ep)
			continue;
		mem_desc->ep[i] = ep;
		mem_desc->peer_addr[i] = FI_ADDR_UNSPEC;
		rc = lnx_core_mr_regattr(mem_desc, attr, flags, i);
		/* TODO: SHM provider returns FI_ENOKEY if requested_key is the
		 * same as the previous call. Application, like OMPI, might not
		 * specify the requested key in fi_mr_attr, so for now ignore that
		 * error.
		 * We need a better way of handling this.
		 * if (rc == -FI_ENOKEY)
		 *		rc = 0;
		 * I made a change in SHM to support FI_MR_PROV_KEY if set by the
		 * application. This tells ofi to generate its own requested_key
		 * for each fi_mr_regattr call
		*/
		if (rc) {
			FI_WARN(&lnx_prov, FI_LOG_CORE, "%s mr_regattr() failed: %d\n",
					ep->lpe_fabric_name, rc);
			return rc;
		}
	}

	return rc;
}

static int
lnx_mr_close_all(struct lnx_mem_desc *mem_desc)
{
	int i, rc, frc = 0;
	struct fid_mr *mr;

	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		mr = mem_desc->core_mr[i];
		if (!mr)
			return frc;
		rc = fi_close(&mr->fid);
		if (rc) {
			FI_WARN(&lnx_prov, FI_LOG_CORE, "%s mr_close() failed: %d\n",
					mem_desc->ep[i]->lpe_fabric_name, rc);
			frc = rc;
		}
	}

	return frc;
}

int lnx_mr_close(struct fid *fid)
{
	struct ofi_mr *mr;
	int rc, frc = 0;

	mr = container_of(fid, struct ofi_mr, mr_fid.fid);

	rc = lnx_mr_close_all(mr->mr_fid.mem_desc);
	if (rc) {
		FI_INFO(&lnx_prov, FI_LOG_CORE, "Failed to complete Memory Deregistration\n");
		frc = rc;
	}

	ofi_atomic_dec32(&mr->domain->ref);

	free(mr->mr_fid.mem_desc);
	free(mr);

	return frc;
}

static struct fi_ops lnx_mr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = lnx_mr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open
};

static int
lnx_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
			   uint64_t flags, struct fid_mr **mr_fid)
{
	/*
	 * If the address is specified then use it to find out which
	 * domain to register the memory against. LINKx can be managing
	 * multiple underlying core provider endpoints, I need to register the
	 * memory against the correct one.
	 *
	 * Once the domain is determined, I need to set the mr->mem_desc to
	 * point to a structure which contains my local endpoint I'll end up
	 * using (which is the same one that I registered the memory against)
	 * and the associate fid_mr which the core provider set for me.
	 *
	 * I return that to the application.
	 *
	 * When the application calls back into the data operations API it'll
	 * pass the mr. I can then pull out a pointer to my local endpoint
	 * which I'll use in the data operation and pass it the correct mr.
	 *
	 * If the address is not provided, then I'll register the memory
	 * buffer against all my core domains, store those and return them to
	 * the user
	 */

	struct util_domain *domain;
	struct ofi_mr *mr;
	struct lnx_mem_desc *mem_desc;
	int rc = 0;

	if (fid->fclass != FI_CLASS_DOMAIN || !attr || attr->iov_count <= 0)
		return -FI_EINVAL;

	domain = container_of(fid, struct util_domain, domain_fid.fid);
	mr = calloc(1, sizeof(*mr));
	mem_desc = calloc(1, sizeof(*mem_desc));
	if (!mr || !mem_desc) {
		rc = -FI_ENOMEM;
		goto fail;
	}

	mr->mr_fid.fid.fclass = FI_CLASS_MR;
	mr->mr_fid.fid.context = attr->context;
	mr->mr_fid.fid.ops = &lnx_mr_fi_ops;
	mr->mr_fid.mem_desc = mem_desc;
	mr->domain = domain;
	mr->flags = flags;

	if (attr->addr == FI_ADDR_UNSPEC) {
		struct local_prov *entry;

		/* register against all domains */
		dlist_foreach_container(&local_prov_table, struct local_prov,
								entry, lpv_entry) {
			rc = lnx_mr_regattrs_all(entry, attr, flags, mem_desc);
			if (rc) {
				FI_INFO(&lnx_prov, FI_LOG_CORE, "Failed to complete Memory Registration %s\n",
						entry->lpv_prov_name);
				return rc;
			}
		}
	} else {
		rc = lnx_select_send_pathway(lnx_get_peer(lnx_peer_tbl->lpt_entries,
					attr->addr), NULL, &mem_desc->ep[0],
					&mem_desc->peer_addr[0], NULL);
		if (rc)
			goto fail;

		rc = lnx_core_mr_regattr(mem_desc, attr, flags, 0);
		if (rc)
			goto fail;
	}

	*mr_fid = &mr->mr_fid;
	ofi_atomic_inc32(&domain->ref);

	return 0;

fail:
	if (mr)
		free(mr);
	if (mem_desc)
		free(mem_desc);
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
	.regattr = lnx_mr_regattr,
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

