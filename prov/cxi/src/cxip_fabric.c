/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2017 DataDirect Networks, Inc. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>

#include "ofi_prov.h"
#include "ofi_osd.h"

#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_FABRIC, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_FABRIC, __VA_ARGS__)

int cxip_av_def_sz = CXIP_AV_DEF_SZ;
int cxip_cq_def_sz = CXIP_CQ_DEF_SZ;
int cxip_eq_def_sz = CXIP_EQ_DEF_SZ;

static int read_default_params;

static struct fi_ops_fabric cxip_fab_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = cxip_domain,
	.passive_ep = fi_no_passive_ep,
	.eq_open = fi_no_eq_open,
	.wait_open = fi_no_wait_open,
	.trywait = fi_no_trywait
};

static int cxip_fabric_close(fid_t fid)
{
	struct cxip_fabric *fab;

	fab = container_of(fid, struct cxip_fabric, util_fabric.fabric_fid);
	if (ofi_atomic_get32(&fab->ref))
		return -FI_EBUSY;

	ofi_fabric_close(&fab->util_fabric);
	free(fab);

	return 0;
}

static struct fi_ops cxip_fab_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static void cxip_read_default_params(void)
{
	if (!read_default_params)
		read_default_params = 1;
}

int cxip_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		void *context)
{
	struct cxip_fabric *fab;
	int ret;

	fab = calloc(1, sizeof(*fab));
	if (!fab)
		return -FI_ENOMEM;

	ret = ofi_fabric_init(&cxip_prov, &cxip_fabric_attr, attr,
			      &fab->util_fabric, context);
	if (ret != FI_SUCCESS)
		goto free_fab;

	cxip_read_default_params();

	ofi_atomic_initialize32(&fab->ref, 0);

	fab->util_fabric.fabric_fid.fid.ops = &cxip_fab_fi_ops;
	fab->util_fabric.fabric_fid.ops = &cxip_fab_ops;

	*fabric = &fab->util_fabric.fabric_fid;

	return 0;

free_fab:
	free(fab);
	return ret;
}
