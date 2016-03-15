/*
 * Copyright (c) 2016 Cisco Systems, Inc.  All rights reserved.
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

#include "fi.h"

#include "usdf.h"
#include "fi_ext_usnic.h"

static int
usdf_usnic_getinfo(uint32_t version, struct fid_fabric *fabric,
			struct fi_usnic_info *uip)
{
	struct usdf_fabric *fp;
	struct usd_device_attrs *dap;

	USDF_TRACE("\n");

	fp = fab_ftou(fabric);
	dap = fp->fab_dev_attrs;

	if (version > FI_EXT_USNIC_INFO_VERSION) {
		return -FI_EINVAL;
	}

	/* this assignment was missing in libfabric v1.1.1 and earlier */
	uip->ui_version = FI_EXT_USNIC_INFO_VERSION;

	uip->ui.v1.ui_link_speed = dap->uda_bandwidth;
	uip->ui.v1.ui_netmask_be = dap->uda_netmask_be;
	strcpy(uip->ui.v1.ui_ifname, dap->uda_ifname);
	uip->ui.v1.ui_num_vf = dap->uda_num_vf;
	uip->ui.v1.ui_qp_per_vf = dap->uda_qp_per_vf;
	uip->ui.v1.ui_cq_per_vf = dap->uda_cq_per_vf;

	return 0;
}

static struct fi_usnic_ops_fabric usdf_usnic_ops_fabric = {
	.size = sizeof(struct fi_usnic_ops_fabric),
	.getinfo = usdf_usnic_getinfo
};

int
usdf_fabric_ops_open(struct fid *fid, const char *ops_name, uint64_t flags,
		void **ops, void *context)
{
	USDF_TRACE("\n");

	if (strcmp(ops_name, FI_USNIC_FABRIC_OPS_1) == 0) {
		*ops = &usdf_usnic_ops_fabric;
	} else {
		return -FI_EINVAL;
	}

	return 0;
}
