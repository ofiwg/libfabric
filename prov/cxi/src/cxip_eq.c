/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2020 Cray Inc. All rights reserved.
 */

 /*
  * Notes:
  *
  * Implemented as an extension of util_eq.
  *
  * At present, the cxip_wait objects are not implemented as extensions of the
  * util_wait object, so we cannot currently fully implement the EQ with wait
  * states. However, the non-blocking read() and peek() functions work.
  */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>

#include <ofi_list.h>
#include <ofi.h>

#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_CQ, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_CQ, __VA_ARGS__)

static int cxip_eq_close(struct fid *fid)
{
	struct cxip_eq *cxi_eq;

	cxi_eq = container_of(fid, struct cxip_eq, util_eq.eq_fid.fid);

	ofi_eq_cleanup(&cxi_eq->util_eq.eq_fid.fid);

	free(cxi_eq);

	return FI_SUCCESS;
}

static struct fi_ops cxi_eq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_eq_close,
	.bind = fi_no_bind,
	.control = ofi_eq_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_eq_attr cxip_eq_def_attr = {
	.size = CXIP_EQ_DEF_SZ,
	.flags = 0,
	.wait_obj = FI_WAIT_FD,
	.signaling_vector = 0,
	.wait_set = NULL
};

int cxip_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		 struct fid_eq **eq, void *context)
{
	struct cxip_eq *cxi_eq;
	int ret;

	cxi_eq = calloc(1, sizeof(*cxi_eq));
	if (!cxi_eq)
		return -FI_ENOMEM;

	if (!attr)
		cxi_eq->attr = cxip_eq_def_attr;
	else
		cxi_eq->attr = *attr;

	ret = ofi_eq_init(fabric, &cxi_eq->attr, &cxi_eq->util_eq.eq_fid, context);
	if (ret != FI_SUCCESS)
		goto err0;

	cxi_eq->util_eq.eq_fid.fid.ops = &cxi_eq_fi_ops;

	*eq = &cxi_eq->util_eq.eq_fid;

	return FI_SUCCESS;
err0:
	free(cxi_eq);
	return ret;
}
