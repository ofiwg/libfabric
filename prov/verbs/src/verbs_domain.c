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

#include "config.h"

#include "fi_verbs.h"
#include "ep_rdm/verbs_rdm.h"

static int fi_ibv_mr_close(fid_t fid)
{
	struct fi_ibv_mem_desc *mr;
	int ret;

	mr = container_of(fid, struct fi_ibv_mem_desc, mr_fid.fid);
	ret = -ibv_dereg_mr(mr->mr);
	if (!ret)
		free(mr);
	return ret;
}

static struct fi_ops fi_ibv_mr_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_mr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int
fi_ibv_mr_reg(struct fid *fid, const void *buf, size_t len,
	   uint64_t access, uint64_t offset, uint64_t requested_key,
	   uint64_t flags, struct fid_mr **mr, void *context)
{
	struct fi_ibv_mem_desc *md;
	int fi_ibv_access = 0;
	struct fid_domain *domain;

	if (flags)
		return -FI_EBADFLAGS;

	if (fid->fclass != FI_CLASS_DOMAIN) {
		return -FI_EINVAL;
	}
	domain = container_of(fid, struct fid_domain, fid);

	md = calloc(1, sizeof *md);
	if (!md)
		return -FI_ENOMEM;

	md->domain = container_of(domain, struct fi_ibv_domain, domain_fid);
	md->mr_fid.fid.fclass = FI_CLASS_MR;
	md->mr_fid.fid.context = context;
	md->mr_fid.fid.ops = &fi_ibv_mr_ops;

	/* Local read access to an MR is enabled by default in verbs */

	if (access & FI_RECV)
		fi_ibv_access |= IBV_ACCESS_LOCAL_WRITE;

	/* iWARP spec requires Remote Write access for an MR that is used
	 * as a data sink for a Remote Read */
	if (access & FI_READ) {
		fi_ibv_access |= IBV_ACCESS_LOCAL_WRITE;
		if (md->domain->verbs->device->transport_type == IBV_TRANSPORT_IWARP)
			fi_ibv_access |= IBV_ACCESS_REMOTE_WRITE;
	}

	if (access & FI_WRITE)
		fi_ibv_access |= IBV_ACCESS_LOCAL_WRITE;

	if (access & FI_REMOTE_READ)
		fi_ibv_access |= IBV_ACCESS_REMOTE_READ;

	/* Verbs requires Local Write access too for Remote Write access */
	if (access & FI_REMOTE_WRITE)
		fi_ibv_access |= IBV_ACCESS_LOCAL_WRITE |
			IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

	md->mr = ibv_reg_mr(md->domain->pd, (void *) buf, len, fi_ibv_access);
	if (!md->mr)
		goto err;

	md->mr_fid.mem_desc = (void *) (uintptr_t) md->mr->lkey;
	md->mr_fid.key = md->mr->rkey;
	*mr = &md->mr_fid;
	return 0;

err:
	free(md);
	return -errno;
}

static int fi_ibv_domain_close(fid_t fid)
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

	fi_freeinfo(domain->info);
	free(domain);
	return 0;
}

static int fi_ibv_open_device_by_name(struct fi_ibv_domain *domain, const char *name)
{
	struct ibv_context **dev_list;
	int i, ret = -FI_ENODEV;

	if (!name)
		return -FI_EINVAL;

	dev_list = rdma_get_devices(NULL);
	if (!dev_list)
		return -errno;

	for (i = 0; dev_list[i] && ret; i++) {
		if (domain->rdm) {
			ret = strncmp(name, ibv_get_device_name(dev_list[i]->device),
				      strlen(name) - strlen(verbs_rdm_domain.suffix));

		} else {
			ret = strcmp(name, ibv_get_device_name(dev_list[i]->device));
		}

		if (!ret)
			domain->verbs = dev_list[i];
	}
	rdma_free_devices(dev_list);
	return ret;
}

static struct fi_ops fi_ibv_fid_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_domain_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_mr fi_ibv_domain_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = fi_ibv_mr_reg,
	.regv = fi_no_mr_regv,
	.regattr = fi_no_mr_regattr,
};

static struct fi_ops_domain fi_ibv_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = fi_no_av_open,
	.cq_open = fi_ibv_cq_open,
	.endpoint = fi_ibv_open_ep,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = fi_no_cntr_open,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_ibv_srq_context,
};

static struct fi_ops_domain fi_ibv_rdm_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = fi_ibv_av_open,
	.cq_open = fi_ibv_rdm_cq_open,
	.endpoint = fi_ibv_rdm_open_ep,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = fi_no_cntr_open,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
};

static int
fi_ibv_domain(struct fid_fabric *fabric, struct fi_info *info,
	   struct fid_domain **domain, void *context)
{
	struct fi_ibv_domain *_domain;
	struct fi_info *fi;
	int ret;

	fi = fi_ibv_get_verbs_info(info->domain_attr->name);
	if (!fi)
		return -FI_EINVAL;

	ret = fi_ibv_check_domain_attr(info->domain_attr, fi);
	if (ret)
		return ret;

	_domain = calloc(1, sizeof *_domain);
	if (!_domain)
		return -FI_ENOMEM;

	_domain->info = fi_dupinfo(info);
	if (!_domain->info)
		goto err1;

	_domain->rdm = FI_IBV_EP_TYPE_IS_RDM(info);
	ret = fi_ibv_open_device_by_name(_domain, info->domain_attr->name);
	if (ret)
		goto err2;

	_domain->pd = ibv_alloc_pd(_domain->verbs);
	if (!_domain->pd) {
		ret = -errno;
		goto err2;
	}

	_domain->domain_fid.fid.fclass = FI_CLASS_DOMAIN;
	_domain->domain_fid.fid.context = context;
	_domain->domain_fid.fid.ops = &fi_ibv_fid_ops;
	_domain->domain_fid.ops = _domain->rdm ? &fi_ibv_rdm_domain_ops :
						 &fi_ibv_domain_ops;
	_domain->domain_fid.mr = &fi_ibv_domain_mr_ops;
	_domain->fab = container_of(fabric, struct fi_ibv_fabric, fabric_fid);

	*domain = &_domain->domain_fid;
	return 0;
err2:
	fi_freeinfo(_domain->info);
err1:
	free(_domain);
	return ret;
}

static int fi_ibv_trywait(struct fid_fabric *fabric, struct fid **fids, int count)
{
	struct fi_ibv_cq *cq;
	int ret, i;

	for (i = 0; i < count; i++) {
		switch (fids[i]->fclass) {
		case FI_CLASS_CQ:
			cq = container_of(fids[i], struct fi_ibv_cq, cq_fid.fid);
			ret = cq->trywait(fids[i]);
			if (ret)
				return ret;
			break;
		case FI_CLASS_EQ:
			/* We are always ready to wait on an EQ since
			 * rdmacm EQ is based on an fd */
			continue;
		case FI_CLASS_CNTR:
		case FI_CLASS_WAIT:
			return -FI_ENOSYS;
		default:
			return -FI_EINVAL;
		}

	}
	return FI_SUCCESS;
}

static int fi_ibv_fabric_close(fid_t fid)
{
	struct fi_ibv_fabric *fab;

	fab = container_of(fid, struct fi_ibv_fabric, fabric_fid.fid);
	util_buf_pool_destroy(fab->wce_pool);
	util_buf_pool_destroy(fab->epe_pool);
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

static struct fi_ops_fabric fi_ibv_ops_fabric = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = fi_ibv_domain,
	.passive_ep = fi_ibv_passive_ep,
	.eq_open = fi_ibv_eq_open,
	.wait_open = fi_no_wait_open,
	.trywait = fi_ibv_trywait
};

int fi_ibv_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		  void *context)
{
	struct fi_ibv_fabric *fab;
	int ret;

	ret = fi_ibv_init_info();
	if (ret)
		return ret;

	ret = fi_ibv_find_fabric(attr);
	if (ret)
		return ret;

	fab = calloc(1, sizeof(*fab));
	if (!fab)
		return -FI_ENOMEM;

	fab->fabric_fid.fid.fclass = FI_CLASS_FABRIC;
	fab->fabric_fid.fid.context = context;
	fab->fabric_fid.fid.ops = &fi_ibv_fi_ops;
	fab->fabric_fid.ops = &fi_ibv_ops_fabric;

	fab->wce_pool = util_buf_pool_create(sizeof(struct fi_ibv_wce), 16, 0, VERBS_WCE_CNT);
	if (!fab->wce_pool) {
		FI_WARN(&fi_ibv_prov, FI_LOG_FABRIC, "Failed to create wce_pool\n");
		ret = -FI_ENOMEM;
		goto err1;
	}

	fab->epe_pool = util_buf_pool_create(sizeof(struct fi_ibv_msg_epe), 16, 0, VERBS_EPE_CNT);
	if (!fab->epe_pool) {
		FI_WARN(&fi_ibv_prov, FI_LOG_FABRIC, "Failed to create epe_pool\n");
		ret = -FI_ENOMEM;
		goto err2;
	}

	*fabric = &fab->fabric_fid;
	return 0;
err2:
	util_buf_pool_destroy(fab->wce_pool);
err1:
	free(fab);
	return ret;
}
