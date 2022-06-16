/*
 * Copyright (c) 2017-2022 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *	   Redistribution and use in source and binary forms, with or
 *	   without modification, are permitted provided that the following
 *	   conditions are met:
 *
 *		- Redistributions of source code must retain the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer.
 *
 *		- Redistributions in binary form must reproduce the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer in the documentation and/or other materials
 *		  provided with the distribution.
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

#include <rdma/fi_errno.h>

#include <ofi_prov.h>
#include "tcp2.h"
#include <poll.h>

#include <sys/types.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <ofi_util.h>

struct fi_ops_fabric tcp2_fabric_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = tcp2_domain_open,
	.passive_ep = tcp2_passive_ep,
	.eq_open = tcp2_eq_create,
	.wait_open = ofi_wait_fd_open,
	.trywait = tcp2_trywait,
};

static int tcp2_fabric_close(fid_t fid)
{
	int ret;
	struct tcp2_fabric *fabric;

	fabric = container_of(fid, struct tcp2_fabric,
			      util_fabric.fabric_fid.fid);

	ret = ofi_fabric_close(&fabric->util_fabric);
	if (ret)
		return ret;

	tcp2_close_progress(&fabric->progress);
	free(fabric);
	return 0;
}

struct fi_ops tcp2_fabric_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = tcp2_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int tcp2_create_fabric(struct fi_fabric_attr *attr,
		       struct fid_fabric **fabric_fid, void *context)
{
	struct tcp2_fabric *fabric;
	int ret;

	fabric = calloc(1, sizeof(*fabric));
	if (!fabric)
		return -FI_ENOMEM;

	ret = ofi_fabric_init(&tcp2_prov, tcp2_info.fabric_attr, attr,
			      &fabric->util_fabric, context);
	if (ret)
		goto free;

	ret = tcp2_init_progress(&fabric->progress, false);
	if (ret)
		goto close;

	fabric->util_fabric.fabric_fid.fid.ops = &tcp2_fabric_fi_ops;
	fabric->util_fabric.fabric_fid.ops = &tcp2_fabric_ops;
	*fabric_fid = &fabric->util_fabric.fabric_fid;

	return 0;

close:
	(void) ofi_fabric_close(&fabric->util_fabric);
free:
	free(fabric);
	return ret;
}
