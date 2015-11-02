/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#include <stdlib.h>
#include <sys/epoll.h>

#include <rdma/rdma_cma.h>

#include <prov.h>
#include <fi_enosys.h>
#include <rdma/fi_prov.h>
#include <prov/verbs/src/fi_verbs.h>

#include <prov/verbs/src/verbs_checks.h>

extern struct fi_info *verbs_info;

/* TODO: verbs_info.h> */
int fi_ibv_getinfo(uint32_t version, const char *node, const char *service,
			  uint64_t flags, struct fi_info *hints, struct fi_info **info);
int fi_ibv_init_info(void);
struct fi_info *fi_ibv_search_verbs_info(const char *fabric_name,
		const char *domain_name);

static int fi_ibv_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
			 void *context);

int fi_ibv_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
            struct fid_eq **eq, void *context);
int fi_ibv_domain(struct fid_fabric *fabric, struct fi_info *info,
            struct fid_domain **domain, void *context);
int fi_ibv_passive_ep(struct fid_fabric *fabric, struct fi_info *info,
	      struct fid_pep **pep, void *context);

static int fi_ibv_fabric_close(fid_t fid)
{
	free(fid);
	return 0;
}

static struct fi_ops fi_ibv_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static void fi_ibv_fini(void)
{
	fi_freeinfo(verbs_info);
}

struct fi_provider fi_ibv_prov = {
	.name = VERBS_PROV_NAME,
	.version = VERBS_PROV_VERS,
	.fi_version = FI_VERSION(1, 1),
	.getinfo = fi_ibv_getinfo,
	.fabric = fi_ibv_fabric,
	.cleanup = fi_ibv_fini
};

static struct fi_ops_fabric fi_ibv_ops_fabric = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = fi_ibv_domain,
	.passive_ep = fi_ibv_passive_ep,
	.eq_open = fi_ibv_eq_open,
	.wait_open = fi_no_wait_open,
};

static int fi_ibv_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
			 void *context)
{
	struct fi_ibv_fabric *fab;
	struct fi_info *info;
	int ret;

	ret = fi_ibv_init_info();
	if (ret)
		return ret;

	info = fi_ibv_search_verbs_info(attr->name, NULL);
	if (!info)
		return -FI_ENODATA;

	ret = fi_ibv_check_fabric_attr(attr, info);
	if (ret)
		return -FI_ENODATA;

	fab = calloc(1, sizeof(*fab));
	if (!fab)
		return -FI_ENOMEM;

	fab->fabric_fid.fid.fclass = FI_CLASS_FABRIC;
	fab->fabric_fid.fid.context = context;
	fab->fabric_fid.fid.ops = &fi_ibv_fi_ops;
	fab->fabric_fid.ops = &fi_ibv_ops_fabric;
	*fabric = &fab->fabric_fid;
	return 0;
}

VERBS_INI
{
	return &fi_ibv_prov;
}

