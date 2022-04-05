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

ssize_t lnx_peer_cq_write(struct fid_peer_cq *cq, void *context, uint64_t flags,
						  size_t len, void *buf, uint64_t data, uint64_t tag,
						  fi_addr_t src)
{
	struct lnx_peer_cq *lnx_cq;
	int rc;

	/* TODO: We need to decide if we should have multiple write/write
	 * error callbacks
	 */
	lnx_cq = container_of(cq, struct lnx_peer_cq, lpc_cq);

	rc = ofi_cq_write(lnx_cq->lpc_shared_cq, context,
					  flags, len, buf, data, tag);

	return rc;
}

ssize_t lnx_peer_cq_writeerr(struct fid_peer_cq *cq,
							 const struct fi_cq_err_entry *err_entry)
{
	struct lnx_peer_cq *lnx_cq;
	int rc;

	/* TODO: We need to decide if we should have multiple write/write
	 * error callbacks
	 */
	lnx_cq = container_of(cq, struct lnx_peer_cq, lpc_cq);

	rc = ofi_cq_write_error(lnx_cq->lpc_shared_cq, err_entry);

	return rc;
}

static int lnx_cleanup_cqs(struct local_prov *prov)
{
	int i, rc, frc = 0;
	struct local_prov_ep *ep;

	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		ep = prov->lpv_prov_eps[i];
		if (!ep)
			continue;
		rc = fi_close(&ep->lpe_cq.lpc_core_cq->fid);
		if (rc)
			frc = rc;
		ep->lpe_cq.lpc_core_cq = NULL;
	}

	return frc;
}

static int lnx_cq_close(struct fid *fid)
{
	int rc;
	struct util_cq *util_cq;
	struct local_prov *entry;

	/* close all the open core cqs */
	dlist_foreach_container(&local_prov_table, struct local_prov,
							entry, lpv_entry) {
		rc = lnx_cleanup_cqs(entry);
		if (rc) {
			FI_INFO(&lnx_prov, FI_LOG_CORE, "Failed to close domain for %s\n",
					entry->lpv_prov_name);
			return rc;
		}
	}

	util_cq = container_of(fid, struct util_cq, cq_fid.fid);

	rc = ofi_cq_cleanup(util_cq);
	if (rc)
		return rc;

	free(util_cq);
	return 0;
}

struct fi_ops_cq_owner lnx_cq_write = {
	.size = sizeof(lnx_cq_write),
	.write = lnx_peer_cq_write,
	.writeerr = lnx_peer_cq_writeerr,
};

static struct fi_ops lnx_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = lnx_cq_close,
	.bind = fi_no_bind,
	.control = ofi_cq_control,
	.ops_open = fi_no_ops_open,
};

static void lnx_cq_progress(struct util_cq *cq)
{
	int i;
	struct local_prov_ep *ep;
	struct local_prov *entry;

	/* Kick the core provider endpoints to progress */
	dlist_foreach_container(&local_prov_table, struct local_prov,
							entry, lpv_entry) {
		for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
			ep = entry->lpv_prov_eps[i];
			if (!ep)
				continue;
			fi_cq_read(ep->lpe_cq.lpc_core_cq, NULL, 0);
		}
	}
}

/* TODO: we might need to define all core providers the LINKx supports and
 * define the object attributes ahead of time, so we can pass them in
 * functions like this one
 */
static int lnx_cq_open_core_prov(struct util_cq *cq,
								 struct fi_cq_attr *attr)
{
	int rc, i;
	struct local_prov_ep *ep;
	struct local_prov *entry;

	/* tell the core providers to import my CQ */
	attr->flags |= FI_PEER_CQ;

	/* create all the core provider endpoints */
	dlist_foreach_container(&local_prov_table, struct local_prov,
							entry, lpv_entry) {
		for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
			struct fid_cq *core_cq;
			struct fi_peer_cq_context cq_ctxt;

			ep = entry->lpv_prov_eps[i];
			if (!ep)
				continue;

			ep->lpe_cq.lpc_shared_cq = cq;
			ep->lpe_cq.lpc_cq.owner_ops = &lnx_cq_write;

			cq_ctxt.size = sizeof(cq_ctxt);
			cq_ctxt.cq = &ep->lpe_cq.lpc_cq;

			/* pass my CQ into the open and get back the core's cq */
			rc = fi_cq_open(ep->lpe_domain, attr, &core_cq, &cq_ctxt);
			if (rc)
				return rc;

			/* before the fi_cq_open() returns the core provider should
			 * have called fi_export_fid() and got a pointer to the peer
			 * CQ which we have allocated for this core provider
			 */

			ep->lpe_cq.lpc_core_cq = core_cq;
		}
	}

	return 0;
}

int lnx_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context)
{
	struct util_cq *util_cq;
	int rc;

	util_cq = calloc(1, sizeof(*util_cq));
	if (!util_cq)
		return -FI_ENOMEM;

	/* this is going to be a standard CQ from the read side. From the
	 * write side, it'll use the peer_cq callbacks to write 
	 */
	rc = ofi_cq_init(&lnx_prov, domain, attr, util_cq,
			  &lnx_cq_progress, context);
	if (rc)
		goto free;

	util_cq->cq_fid.fid.ops = &lnx_cq_fi_ops;
	(*cq_fid) = &util_cq->cq_fid;

	/* open core CQs and tell them to import my CQ */
	rc = lnx_cq_open_core_prov(util_cq, attr);

	return rc;

free:
	free(util_cq);
	return rc;
}
