/*
 * Copyright (c) 2015-2016 Intel Corporation. All rights reserved.
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

#include <prov.h>
#include "tcpx.h"

#include <sys/types.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <fi_util.h>

static int tcpx_getinfo (uint32_t version, const char *node, const char *service,
						uint64_t flags, struct fi_info *hints,
						struct fi_info **info)
{
	return util_getinfo(&tcpx_util_prov, version, node, service, flags,
					   hints, info);
}

static void fi_tcp_fini(void)
{
	/* empty as of now */
}

struct fi_provider tcpx_prov = {
	.name = "tcp",
	.version = FI_VERSION(1,0),
	.fi_version = FI_VERSION(1,5),
	.getinfo = tcpx_getinfo,
	.fabric = tcpx_fabric,
	.cleanup = fi_tcp_fini,
};

TCP_INI
{
	return &tcpx_prov;
}
