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
	int rc, frc = 0;
	struct local_prov_ep *ep;

	dlist_foreach_container(&prov->lpv_prov_eps,
				struct local_prov_ep, ep, entry) {
		if (!ep->lpe_domain)
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
	struct lnx_domain *domain;

	domain = container_of(fid, struct lnx_domain, ld_domain.domain_fid.fid);

	/* close all the open core domains */
	dlist_foreach_container(&domain->ld_fabric->local_prov_table,
				struct local_prov,
				entry, lpv_entry) {
		rc = lnx_cleanup_domains(entry);
		if (rc)
			FI_WARN(&lnx_prov, FI_LOG_CORE, "Failed to close domain for %s\n",
					entry->lpv_prov_name);
	}

	ofi_mr_cache_cleanup(&domain->ld_mr_cache);

	rc = ofi_domain_close(&domain->ld_domain);

	free(domain);

	return rc;
}

static int
lnx_mr_regattrs_all(struct local_prov *prov, const struct fi_mr_attr *attr,
		    uint64_t flags, struct lnx_mem_desc_prov *desc)
{
	int rc = 0;
	struct local_prov_ep *ep;

	desc->prov = prov;

	/* TODO: This is another issue here because MR registration can happen
	 * quiet often
	 */
	dlist_foreach_container(&prov->lpv_prov_eps,
				struct local_prov_ep, ep, entry) {
		rc = fi_mr_regattr(ep->lpe_domain, attr,
				flags, &desc->core_mr);

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

	for (i = 0; i < mem_desc->desc_count; i++) {
		mr = mem_desc->desc[i].core_mr;
		if (!mr)
			continue;
		rc = fi_close(&mr->fid);
		if (rc) {
			FI_WARN(&lnx_prov, FI_LOG_CORE, "%s mr_close() failed: %d\n",
					mem_desc->desc[i].prov->lpv_prov_name, rc);
			frc = rc;
		}
	}

	return frc;
}

int lnx_mr_close(struct fid *fid)
{
	struct lnx_mr *lnx_mr;
	struct ofi_mr *mr;
	int rc, frc = 0;

	mr = container_of(fid, struct ofi_mr, mr_fid.fid);
	lnx_mr = container_of(mr, struct lnx_mr, mr);

	rc = lnx_mr_close_all(mr->mr_fid.mem_desc);
	if (rc) {
		FI_WARN(&lnx_prov, FI_LOG_CORE, "Failed to complete Memory Deregistration\n");
		frc = rc;
	}

	ofi_atomic_dec32(&mr->domain->ref);

	ofi_buf_free(lnx_mr);

	return frc;
}

static int lnx_mr_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	int i, rc, frc = 0;
	struct local_prov_ep *ep;
	struct fid_mr *mr, *cmr;
	struct lnx_mem_desc *mem_desc;
	struct lnx_mem_desc_prov *desc;

	mr = container_of(fid, struct fid_mr, fid);

	mem_desc = mr->mem_desc;

	/* TODO: This is another issue here because MR registration can happen
	 * quiet often
	 */
	for (i = 0; i < mem_desc->desc_count; i++) {
		desc = &mem_desc->desc[i];
		cmr = desc->core_mr;
		if (!cmr)
			continue;
		dlist_foreach_container(&desc->prov->lpv_prov_eps,
					struct local_prov_ep, ep, entry) {
			rc = fi_mr_bind(cmr, &ep->lpe_ep->fid, flags);
			if (rc) {
				FI_WARN(&lnx_prov, FI_LOG_CORE,
					"%s lnx_mr_bind() failed: %d\n",
					mem_desc->desc[i].prov->lpv_prov_name, rc);
				frc = rc;
			}
		}
	}

	return frc;
}

static int lnx_mr_control(struct fid *fid, int command, void *arg)
{
	int i, rc, frc = 0;
	struct fid_mr *mr, *cmr;
	struct lnx_mem_desc *mem_desc;
	struct lnx_mem_desc_prov *desc;

	if (command != FI_ENABLE)
		return -FI_ENOSYS;

	mr = container_of(fid, struct fid_mr, fid);

	mem_desc = mr->mem_desc;

	/* TODO: This is another issue here because MR registration can happen
	 * quiet often
	 */
	for (i = 0; i < mem_desc->desc_count; i++) {
		desc = &mem_desc->desc[i];
		cmr = desc->core_mr;
		if (!cmr)
			continue;
		rc = fi_mr_enable(cmr);
		if (rc) {
			FI_WARN(&lnx_prov, FI_LOG_CORE, "%s lnx_mr_control() failed: %d\n",
				mem_desc->desc[i].prov->lpv_prov_name, rc);
			frc = rc;
		}
	}

	return frc;
}

static struct fi_ops lnx_mr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = lnx_mr_close,
	.bind = lnx_mr_bind,
	.control = lnx_mr_control,
	.ops_open = fi_no_ops_open
};

static int
lnx_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
	       uint64_t flags, struct fid_mr **mr_fid)
{
	/*
	 * If the address is specified then use it to find out which
	 * domain to register the memory against. LNX can be managing
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

	struct lnx_domain *domain;
	struct lnx_fabric *fabric;
	struct lnx_mr *lnx_mr = NULL;;
	struct ofi_mr *mr;
	struct lnx_mem_desc *mem_desc;
	struct local_prov *entry;
	int rc = 0, i = 1;
	bool shm = false;

	if (fid->fclass != FI_CLASS_DOMAIN || !attr || attr->iov_count <= 0)
		return -FI_EINVAL;

	domain = container_of(fid, struct lnx_domain, ld_domain.domain_fid.fid);
	fabric = domain->ld_fabric;

	lnx_mr = ofi_buf_alloc(fabric->mem_reg_bp);
	if (!lnx_mr) {
		rc = -FI_ENOMEM;
		goto fail;
	}

	mr = &lnx_mr->mr;
	mem_desc = &lnx_mr->desc;

	mr->mr_fid.fid.fclass = FI_CLASS_MR;
	mr->mr_fid.fid.context = attr->context;
	mr->mr_fid.fid.ops = &lnx_mr_fi_ops;
	mr->mr_fid.mem_desc = mem_desc;
	mr->domain = &domain->ld_domain;
	mr->flags = flags;

	/* TODO: What's gonna happen if you try to register the same piece
	 * of memory via multiple providers?
	 * TODO 2: We need a better way to handle memory registration.
	 * This is simply not very good. We need to have a peer interface
	 * to memory registration
	 */
	/* register against all domains */
	dlist_foreach_container(&fabric->local_prov_table,
				struct local_prov,
				entry, lpv_entry) {
		if (!strcmp(entry->lpv_prov_name, "shm"))
			shm = true;
		else
			shm = false;
		if (i >= LNX_MAX_LOCAL_EPS) {
			FI_WARN(&lnx_prov, FI_LOG_CORE,
				"Exceeded number of allowed memory registrations %s\n",
				entry->lpv_prov_name);
			rc = -FI_ENOSPC;
			goto fail;
		}
		rc = lnx_mr_regattrs_all(entry, attr, flags,
					 (shm) ? &mem_desc->desc[0] :
					 &mem_desc->desc[i]);
		if (rc) {
			FI_WARN(&lnx_prov, FI_LOG_CORE,
				"Failed to complete Memory Registration %s\n",
				entry->lpv_prov_name);
			goto fail;
		}
		if (!shm)
			i++;
	}

	mem_desc->desc_count = i;
	if (shm)
		mr->mr_fid.key = mem_desc->desc[0].core_mr->key;
	else
		mr->mr_fid.key = mem_desc->desc[1].core_mr->key;
	*mr_fid = &mr->mr_fid;
	ofi_atomic_inc32(&domain->ld_domain.ref);

	return 0;

fail:
	if (lnx_mr)
		ofi_buf_free(lnx_mr);
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

static int lnx_setup_core_domain(struct local_prov_ep *ep, struct fi_info *info)
{
	struct fi_info *fi, *itr;

	fi = lnx_get_link_by_dom(info->domain_attr->name);
	if (!fi)
		return -FI_ENODATA;

	for (itr = fi; itr; itr = itr->next) {
		if (!strcmp(itr->fabric_attr->name, ep->lpe_fabric_name)) {
			ep->lpe_fi_info = fi_dupinfo(itr);
			return FI_SUCCESS;
		}
	}

	ep->lpe_fi_info = NULL;

	return -FI_ENOENT;
}

static struct fi_ops_srx_owner lnx_srx_ops = {
	.size = sizeof(struct fi_ops_srx_owner),
	.get_msg = lnx_get_msg,
	.get_tag = lnx_get_tag,
	.queue_msg = lnx_queue_msg,
	.queue_tag = lnx_queue_tag,
	.free_entry = lnx_free_entry,
	.foreach_unspec_addr = lnx_foreach_unspec_addr,
};

static int lnx_open_core_domains(struct local_prov *prov,
				void *context, struct lnx_domain *lnx_domain,
				struct fi_info *info)
{
	int rc;
	struct local_prov_ep *ep;
	struct fi_rx_attr attr = {0};
	struct fi_peer_srx_context peer_srx;
	struct dlist_entry *tmp;
	int srq_support = 1;

	fi_param_get_bool(&lnx_prov, "use_srq", &srq_support);

	attr.op_flags = FI_PEER;
	peer_srx.size = sizeof(peer_srx);

	if (srq_support)
		lnx_domain->ld_srx_supported = true;
	else
		lnx_domain->ld_srx_supported = false;

	dlist_foreach_container_safe(&prov->lpv_prov_eps,
		struct local_prov_ep, ep, entry, tmp) {
		/* the fi_info we setup when we created the fabric might not
		 * necessarily be the correct one. It'll have the same fabric
		 * information, since the fabric information is common among all
		 * the domains the provider manages. However at this point we need
		 * to get the fi_info that the application is requesting */
		rc = lnx_setup_core_domain(ep, info);
		if (rc)
			return rc;

		if (srq_support) {
			/* special case for CXI provider. We need to turn off tag
			 * matching HW offload if we're going to support shared
			 * receive queues.
			 */
			if (strstr(ep->lpe_fabric_name, "cxi"))
				setenv("FI_CXI_RX_MATCH_MODE", "software", 1);
		}

		rc = fi_domain(ep->lpe_fabric, ep->lpe_fi_info,
			       &ep->lpe_domain, context);

		if (!rc && srq_support) {
			ep->lpe_srx.owner_ops = &lnx_srx_ops;
			peer_srx.srx = &ep->lpe_srx;
			rc = fi_srx_context(ep->lpe_domain, &attr, NULL, &peer_srx);
		}

		/* if one of the constituent endpoints doesn't support shared
		 * receive context, then fail, as we can't continue with this
		 * inconsistency
		 */
		if (rc) {
			FI_WARN(&lnx_prov, FI_LOG_CORE, "%s does not support shared"
				" receive queues. Failing\n", ep->lpe_fabric_name);
			return rc;
		}
	}

	return 0;
}

static int lnx_addr_add_region_noop(struct ofi_mr_cache *cache,
			     struct ofi_mr_entry *entry)
{
	return FI_SUCCESS;
}

static void lnx_addr_del_region(struct ofi_mr_cache *cache,
			 struct ofi_mr_entry *entry)
{
	struct ofi_mr *mr = (struct ofi_mr *)entry->data;

	ofi_hmem_dev_unregister(mr->iface, (uint64_t) mr->hmem_data);
}

/*
 * provider: shm+cxi:lnx
 *     fabric: ofi_lnx_fabric
 *     domain: shm+cxi3:ofi_lnx_domain
 *     version: 120.0
 *     type: FI_EP_RDM
 *     protocol: FI_PROTO_LNX
 *
 * Parse out the provider name. It should be shm+<prov>
 *
 * Create a fabric for shm and one for the other provider.
 *
 * When fi_domain() is called, we get the fi_info for the
 * second provider, which we should've returned as part of the
 * fi_getinfo() call.
 */
int lnx_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **domain, void *context)
{
	int rc = 0;
	struct local_prov *entry;
	struct lnx_domain *lnx_domain;
	struct util_domain *lnx_domain_info;
	struct lnx_fabric *lnx_fab = container_of(fabric, struct lnx_fabric,
					util_fabric.fabric_fid);
	struct ofi_mem_monitor *memory_monitors[OFI_HMEM_MAX] = {
		[FI_HMEM_SYSTEM] = default_monitor,
		[FI_HMEM_CUDA] = default_cuda_monitor,
		[FI_HMEM_ROCR] = default_rocr_monitor,
		[FI_HMEM_ZE] = default_ze_monitor,
	};

	/* create a new entry for shm.
	 * Create its fabric.
	 * insert fabric in the global table
	 */
	rc = lnx_setup_core_fabrics(info->domain_attr->name, lnx_fab, context);
	if (rc)
		goto fail;

	rc = -FI_ENOMEM;
	lnx_domain = calloc(sizeof(*lnx_domain), 1);
	if (!lnx_domain)
		goto fail;

	lnx_domain_info = &lnx_domain->ld_domain;
	lnx_domain->ld_fabric = lnx_fab;

	rc = ofi_domain_init(fabric, info, lnx_domain_info, context,
			     OFI_LOCK_SPINLOCK);
	if (rc)
		goto fail;

	dlist_foreach_container(&lnx_domain->ld_fabric->local_prov_table,
				struct local_prov, entry, lpv_entry) {
		rc = lnx_open_core_domains(entry, context, lnx_domain, info);
		if (rc) {
			FI_INFO(&lnx_prov, FI_LOG_CORE, "Failed to initialize domain for %s\n",
				entry->lpv_prov_name);
			goto close_domain;
		}
	}

	lnx_domain_info->domain_fid.fid.ops = &lnx_domain_fi_ops;
	lnx_domain_info->domain_fid.ops = &lnx_domain_ops;
	lnx_domain_info->domain_fid.mr = &lnx_mr_ops;

	lnx_domain->ld_mr_cache.add_region = lnx_addr_add_region_noop;
	lnx_domain->ld_mr_cache.delete_region = lnx_addr_del_region;
	lnx_domain->ld_mr_cache.entry_data_size = sizeof(struct ofi_mr);
	rc = ofi_mr_cache_init(&lnx_domain->ld_domain, memory_monitors,
			       &lnx_domain->ld_mr_cache);
	if (rc)
		goto close_domain;

	*domain = &lnx_domain_info->domain_fid;

	return 0;

close_domain:
	lnx_domain_close(&(lnx_domain_info->domain_fid.fid));
fail:
	return rc;
}

