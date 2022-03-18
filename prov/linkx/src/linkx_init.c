/*
 * Copyright (c) 2016 Intel Corporation. All rights reserved.
 * (C) Copyright 2020 Hewlett Packard Enterprise Development LP.
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

static int
lnx_getinfo(uint32_t version, const char *node, const char *service,
	     uint64_t flags, const struct fi_info *hints,
	     struct fi_info **info)
{
	/* LINKx provider doesn't show up in fi_getinfo(). It can not be
	 * selected explicitly. It's a hidden provider of sorts.
	 *
	 * TODO: should it be shown in the fi_getinof() result? It doesn't
	 * make sense, since the intent is to make this an undercover
	 * provider, which is returned only as part of fi_link()
	 */
	return -FI_EOPNOTSUPP;
}

int lnx_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		void *context)
{
	/* LINKx provider's fabric is not explicitly initialized using
	 * lnx_fabric(). It gets initialized implicitly with ofi_link
	 */
	return -FI_EOPNOTSUPP;
}

static void lnx_fini(void)
{
}

struct fi_provider lnx_prov = {
	.name = "linkx",
	.version = OFI_VERSION_DEF_PROV,
	.fi_version = OFI_VERSION_LATEST,
	.getinfo = lnx_getinfo,
	.fabric = lnx_fabric,
	.cleanup = lnx_fini
};

int ofi_create_link(struct fi_info *prov_list,
					struct fid_fabric **fabric,
					uint64_t caps, void *context)
{
	return -FI_EOPNOTSUPP;
}

void ofi_link_fini(void)
{
	lnx_prov.cleanup();
}


