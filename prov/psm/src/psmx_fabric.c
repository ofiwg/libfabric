/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

#include "psmx.h"

struct psmx_fid_fabric *psmx_active_fabric = NULL;

void psmx_fabric_release(struct psmx_fid_fabric *fabric)
{
	void *exit_code;
	int ret;

	FI_INFO(&psmx_prov, FI_LOG_CORE, "refcnt=%d\n", fabric->refcnt);

	if (--fabric->refcnt)
		return;

	if (psmx_env.name_server &&
	    !pthread_equal(fabric->name_server_thread, pthread_self())) {
		ret = pthread_cancel(fabric->name_server_thread);
		if (ret) {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"pthread_cancel returns %d\n", ret);
		}
		ret = pthread_join(fabric->name_server_thread, &exit_code);
		if (ret) {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"pthread_join returns %d\n", ret);
		}
		else {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"name server thread exited with code %ld (%s)\n",
				(uintptr_t)exit_code,
				(exit_code == PTHREAD_CANCELED) ? "PTHREAD_CANCELED" : "?");
		}
	}
	if (fabric->active_domain) {
		FI_WARN(&psmx_prov, FI_LOG_CORE, "forced closing of active_domain\n");
		fi_close(&fabric->active_domain->domain.fid);
	}
	assert(fabric == psmx_active_fabric);
	psmx_active_fabric = NULL;
	free(fabric);
}

static int psmx_fabric_close(fid_t fid)
{
	struct psmx_fid_fabric *fabric;

	FI_INFO(&psmx_prov, FI_LOG_CORE, "\n");

	fabric = container_of(fid, struct psmx_fid_fabric, fabric.fid);

	psmx_fabric_release(fabric);

	return 0;
}

static struct fi_ops psmx_fabric_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx_fabric_close,
};

static struct fi_ops_fabric psmx_fabric_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = psmx_domain_open,
	.passive_ep = fi_no_passive_ep,
	.eq_open = psmx_eq_open,
	.wait_open = psmx_wait_open,
};

int psmx_fabric(struct fi_fabric_attr *attr,
		struct fid_fabric **fabric, void *context)
{
	struct psmx_fid_fabric *fabric_priv;
	int ret;

	FI_INFO(&psmx_prov, FI_LOG_CORE, "\n");

	if (strcmp(attr->name, PSMX_FABRIC_NAME))
		return -FI_ENODATA;

	if (psmx_active_fabric) {
		psmx_fabric_acquire(psmx_active_fabric);
		*fabric = &psmx_active_fabric->fabric;
		return 0;
	}

	fabric_priv = calloc(1, sizeof(*fabric_priv));
	if (!fabric_priv)
		return -FI_ENOMEM;

	fabric_priv->fabric.fid.fclass = FI_CLASS_FABRIC;
	fabric_priv->fabric.fid.context = context;
	fabric_priv->fabric.fid.ops = &psmx_fabric_fi_ops;
	fabric_priv->fabric.ops = &psmx_fabric_ops;

	psmx_get_uuid(fabric_priv->uuid);

	if (psmx_env.name_server) {
		ret = pthread_create(&fabric_priv->name_server_thread, NULL,
				     psmx_name_server, (void *)fabric_priv);
		if (ret) {
			FI_INFO(&psmx_prov, FI_LOG_CORE, "pthread_create returns %d\n", ret);
			/* use the main thread's ID as invalid value for the new thread */
			fabric_priv->name_server_thread = pthread_self();
		}
	}

	psmx_query_mpi();

	fabric_priv->refcnt = 1;
	*fabric = &fabric_priv->fabric;
	psmx_active_fabric = fabric_priv;

	return 0;
}

