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

static int cxip_eq_close(struct fid *fid)
{
	struct cxip_eq *cxi_eq;

	cxi_eq = container_of(fid, struct cxip_eq, util_eq.eq_fid.fid);

	ofi_eq_cleanup(&cxi_eq->util_eq.eq_fid.fid);

	free(cxi_eq);

	return FI_SUCCESS;
}

ssize_t cxip_eq_read(struct fid_eq *eq_fid, uint32_t *event,
		     void *buf, size_t len, uint64_t flags)
{
	struct cxip_curl_handle *handle;
	int ret;

	ret = cxip_curl_progress(&handle);
	if (ret == FI_SUCCESS) {
		// callback has run, handle is valid
		// TODO process data in handle
	} else if (ret == -FI_EAGAIN) {
		// nothing avail, but some processing
		// handle invalid, do nothing
	} else if (ret == -FI_ENODATA) {
		// nothing avail, nothing processing
		// handle invalid, do nothing
	} else {
		// curl failure
		// TODO handle error condition
	}
	/* pass control to the OFI EQ handler */
	return ofi_eq_read(eq_fid, event, buf, len, flags);
}

static struct fi_ops_eq cxi_eq_ops = {
	.size = sizeof(struct fi_ops_eq),
	.read = cxip_eq_read,		// customized
	.readerr = ofi_eq_readerr,
	.sread = ofi_eq_sread,
	.write = ofi_eq_write,
	.strerror = ofi_eq_strerror,
};

static struct fi_ops cxi_eq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_eq_close,		// customized
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

	ret = ofi_eq_init(fabric, &cxi_eq->attr, &cxi_eq->util_eq.eq_fid,
			  context);
	if (ret != FI_SUCCESS)
		goto err0;

	/* custom operations */
	cxi_eq->util_eq.eq_fid.fid.ops = &cxi_eq_fi_ops;
	cxi_eq->util_eq.eq_fid.ops = &cxi_eq_ops;

	*eq = &cxi_eq->util_eq.eq_fid;

	return FI_SUCCESS;
err0:
	free(cxi_eq);
	return ret;
}
