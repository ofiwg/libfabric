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

#include <fi_enosys.h>
#include <rdma/fabric.h>

#include <stdlib.h>
#include <rdma/rdma_cma.h>

#include <prov/verbs/src/fi_verbs.h>
#include <prov/verbs/src/verbs_checks.h>

extern struct fi_provider fi_ibv_prov;
extern struct fi_ops_mr fi_ibv_domain_mr_ops;

static int fi_ibv_open_device_by_name(struct fi_ibv_domain *domain,
                                      const char *name)
{
	struct ibv_context **dev_list;
	int i, ret = -FI_ENODEV;

	if (!name)
		return -FI_EINVAL;

	dev_list = rdma_get_devices(NULL);
	if (!dev_list)
		return -errno;

	for (i = 0; dev_list[i]; i++) {
		if (!strcmp(name, ibv_get_device_name(dev_list[i]->device))) {
			domain->verbs = dev_list[i];
			ret = 0;
			break;
		}
	}
	rdma_free_devices(dev_list);
	return ret;
}

static int fi_ibv_close(fid_t fid)
{
	struct fi_ibv_domain *domain;
	int ret;

	domain = container_of(fid, struct fi_ibv_domain, domain_fid.fid);
	if (domain->pd) {
		ret = ibv_dealloc_pd(domain->pd);
		if (ret)
			return -ret;
		domain->pd = NULL;
	}

	free(domain);
	return 0;
}

static struct fi_ops fi_ibv_fid_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int fi_ibv_open_ep(struct fid_domain *domain, struct fi_info *info,
                          struct fid_ep **ep, void *context)
{
    assert(info);
    assert(info->ep_attr);
    assert(info->domain_attr);
    if (info->ep_attr->type == FI_EP_RDM) {
        return fi_ibv_rdm_tagged_open_ep(domain, info, ep, context);
    } else {
        return fi_ibv_msg_open_ep(domain, info, ep, context);
    }
}

static struct fi_ops_domain fi_ibv_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = fi_ibv_av_open,
	.cq_open = fi_ibv_cq_open,
	.endpoint = fi_ibv_open_ep,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = fi_no_cntr_open,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
};

int fi_ibv_domain(struct fid_fabric *fabric, struct fi_info *info,
	   struct fid_domain **domain, void *context)
{
	struct fi_ibv_domain *_domain;
	struct fi_info *fi;
	int ret;

	fi = fi_ibv_search_verbs_info(NULL, info->domain_attr->name);
	if (!fi)
		return -FI_EINVAL;

	ret = fi_ibv_check_domain_attr(info->domain_attr, fi);
	if (ret)
		return ret;

	_domain = calloc(1, sizeof *_domain);
	if (!_domain)
		return -FI_ENOMEM;

	ret = fi_ibv_open_device_by_name(_domain, info->domain_attr->name);
	if (ret)
		goto err;

	_domain->pd = ibv_alloc_pd(_domain->verbs);
	if (!_domain->pd) {
		ret = -errno;
		goto err;
	}

	_domain->domain_fid.fid.fclass = FI_CLASS_DOMAIN;
	_domain->domain_fid.fid.context = context;
	_domain->domain_fid.fid.ops = &fi_ibv_fid_ops;
	_domain->domain_fid.ops = &fi_ibv_domain_ops;
	_domain->domain_fid.mr = &fi_ibv_domain_mr_ops;

	*domain = &_domain->domain_fid;
	return 0;
err:
	free(_domain);
	return ret;
}
