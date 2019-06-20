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

#include "ofi_iov.h"

#include "fi_verbs.h"
#include <malloc.h>


static int fi_ibv_domain_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct fi_ibv_domain *domain;
	struct fi_ibv_eq *eq;

	domain = container_of(fid, struct fi_ibv_domain,
			      util_domain.domain_fid.fid);

	switch (bfid->fclass) {
	case FI_CLASS_EQ:
		switch (domain->ep_type) {
		case FI_EP_MSG:
			eq = container_of(bfid, struct fi_ibv_eq, eq_fid);
			domain->eq = eq;
			domain->eq_flags = flags;
			break;
		case FI_EP_DGRAM:
			return -FI_EINVAL;
		default:
			/* Shouldn't go here */
			assert(0);
			return -FI_EINVAL;
		}
		break;

	default:
		return -EINVAL;
	}

	return 0;
}

static int fi_ibv_domain_close(fid_t fid)
{
	int ret;
	struct fi_ibv_fabric *fab;
	struct fi_ibv_domain *domain =
		container_of(fid, struct fi_ibv_domain,
			     util_domain.domain_fid.fid);

	switch (domain->ep_type) {
	case FI_EP_DGRAM:
		fab = container_of(&domain->util_domain.fabric->fabric_fid,
				   struct fi_ibv_fabric,
				   util_fabric.fabric_fid.fid);
		/* Even if it's invoked not for the first time
		 * (e.g. multiple domains per fabric), it's safe
		 */
		if (fi_ibv_gl_data.dgram.use_name_server)
			ofi_ns_stop_server(&fab->name_server);
		break;
	case FI_EP_MSG:
		if (domain->use_xrc) {
			ret = fi_ibv_domain_xrc_cleanup(domain);
			if (ret)
				return ret;
		}
		break;
	default:
		/* Never should go here */
		assert(0);
		return -FI_EINVAL;
	}

	ofi_mr_cache_cleanup(&domain->cache);

	if (domain->pd) {
		ret = ibv_dealloc_pd(domain->pd);
		if (ret)
			return -ret;
		domain->pd = NULL;
	}

	ret = ofi_domain_close(&domain->util_domain);
	if (ret)
		return ret;

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
		const char *rdma_name = ibv_get_device_name(dev_list[i]->device);
		switch (domain->ep_type) {
		case FI_EP_MSG:
			ret = domain->use_xrc ?
				fi_ibv_cmp_xrc_domain_name(name, rdma_name) :
				strcmp(name, rdma_name);
			break;
		case FI_EP_DGRAM:
			ret = strncmp(name, rdma_name,
				      strlen(name) - strlen(verbs_dgram_domain.suffix));
			break;
		default:
			VERBS_WARN(FI_LOG_DOMAIN,
				   "Unsupported EP type - %d\n", domain->ep_type);
			/* Never should go here */
			assert(0);
			ret = -FI_EINVAL;
			break;
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
	.bind = fi_ibv_domain_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_domain fi_ibv_msg_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = fi_no_av_open,
	.cq_open = fi_ibv_cq_open,
	.endpoint = fi_ibv_open_ep,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = fi_no_cntr_open,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_ibv_srq_context,
	.query_atomic = fi_ibv_query_atomic,
};

static struct fi_ops_domain fi_ibv_dgram_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = fi_ibv_dgram_av_open,
	.cq_open = fi_ibv_cq_open,
	.endpoint = fi_ibv_open_ep,
	.scalable_ep = fi_no_scalable_ep,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = fi_no_query_atomic,
};

static void fi_ibv_domain_process_exp(struct fi_ibv_domain *domain)
{
#ifdef HAVE_VERBS_EXP_H
	struct ibv_exp_device_attr exp_attr= {
		.comp_mask = IBV_EXP_DEVICE_ATTR_ODP |
			     IBV_EXP_DEVICE_ATTR_EXP_CAP_FLAGS,
	};
	domain->use_odp = (!ibv_exp_query_device(domain->verbs, &exp_attr) &&
			   exp_attr.exp_device_cap_flags & IBV_EXP_DEVICE_ODP);
#else /* HAVE_VERBS_EXP_H */
	domain->use_odp = 0;
#endif /* HAVE_VERBS_EXP_H */
	if (!domain->use_odp && fi_ibv_gl_data.use_odp) {
		VERBS_WARN(FI_LOG_CORE,
			   "ODP is not supported on this configuration, ignore \n");
		return;
	}
	domain->use_odp = fi_ibv_gl_data.use_odp;
}

static int
fi_ibv_post_send_track_credits(struct ibv_qp *qp, struct ibv_send_wr *wr,
			       struct ibv_send_wr **bad_wr)
{
	struct fi_ibv_cq *cq =
		container_of(((struct fi_ibv_ep *)qp->qp_context)->util_ep.tx_cq,
			     struct fi_ibv_cq, util_cq);
	int credits = (int)ofi_atomic_dec32(&cq->credits);
	int ret;

	if (credits < 0) {
		FI_DBG(&fi_ibv_prov, FI_LOG_EP_DATA, "CQ credits not available,"
		       " retry later\n");
		ofi_atomic_inc32(&cq->credits);
		return ENOMEM;
	}
	ret = ibv_post_send(qp, wr, bad_wr);
	if (ret)
		ofi_atomic_inc32(&cq->credits);
	return ret;
}

static int
fi_ibv_poll_cq_track_credits(struct ibv_cq *cq, int num_entries,
			     struct ibv_wc *wc)
{
	struct fi_ibv_cq *verbs_cq = (struct fi_ibv_cq *)cq->cq_context;
	int i, ret;

	ret = ibv_poll_cq(cq, num_entries, wc);
	for (i = 0; i < ret; i++) {
		if (!(wc[i].opcode & IBV_WC_RECV))
			ofi_atomic_inc32(&verbs_cq->credits);
	}
	return ret;
}


static int
fi_ibv_domain(struct fid_fabric *fabric, struct fi_info *info,
	      struct fid_domain **domain, void *context)
{
	struct fi_ibv_domain *_domain;
	int ret;
	struct fi_ibv_fabric *fab =
		 container_of(fabric, struct fi_ibv_fabric,
			      util_fabric.fabric_fid);
	const struct fi_info *fi = fi_ibv_get_verbs_info(fi_ibv_util_prov.info,
							 info->domain_attr->name);
	if (!fi)
		return -FI_EINVAL;

	ret = ofi_check_domain_attr(&fi_ibv_prov, fabric->api_version,
				    fi->domain_attr, info);
	if (ret)
		return ret;

	_domain = calloc(1, sizeof *_domain);
	if (!_domain)
		return -FI_ENOMEM;

	ret = ofi_domain_init(fabric, info, &_domain->util_domain, context);
	if (ret)
		goto err1;

	_domain->info = fi_dupinfo(info);
	if (!_domain->info)
		goto err2;

	_domain->ep_type = FI_IBV_EP_TYPE(info);
	_domain->use_xrc = fi_ibv_is_xrc(info);

	ret = fi_ibv_open_device_by_name(_domain, info->domain_attr->name);
	if (ret)
		goto err3;

	_domain->pd = ibv_alloc_pd(_domain->verbs);
	if (!_domain->pd) {
		ret = -errno;
		goto err3;
	}

	_domain->util_domain.domain_fid.fid.fclass = FI_CLASS_DOMAIN;
	_domain->util_domain.domain_fid.fid.context = context;
	_domain->util_domain.domain_fid.fid.ops = &fi_ibv_fid_ops;

	fi_ibv_domain_process_exp(_domain);

	_domain->cache.entry_data_size = sizeof(struct fi_ibv_mem_desc);
	_domain->cache.add_region = fi_ibv_mr_cache_entry_reg;
	_domain->cache.delete_region = fi_ibv_mr_cache_entry_dereg;
	ret = ofi_mr_cache_init(&_domain->util_domain, uffd_monitor,
				&_domain->cache);
	if (!ret) {
		_domain->util_domain.domain_fid.mr = fi_ibv_mr_internal_cache_ops.fi_ops;
		_domain->internal_mr_reg = fi_ibv_mr_internal_cache_ops.internal_mr_reg;
		_domain->internal_mr_dereg = fi_ibv_mr_internal_cache_ops.internal_mr_dereg;
	} else {
		_domain->util_domain.domain_fid.mr = fi_ibv_mr_internal_ops.fi_ops;
		_domain->internal_mr_reg = fi_ibv_mr_internal_ops.internal_mr_reg;
		_domain->internal_mr_dereg = fi_ibv_mr_internal_ops.internal_mr_dereg;
	}

	switch (_domain->ep_type) {
	case FI_EP_DGRAM:
		if (fi_ibv_gl_data.dgram.use_name_server) {
			/* Even if it's invoked not for the first time
			 * (e.g. multiple domains per fabric), it's safe
			 */
			fab->name_server.port =
					fi_ibv_gl_data.dgram.name_server_port;
			fab->name_server.name_len = sizeof(struct ofi_ib_ud_ep_name);
			fab->name_server.service_len = sizeof(int);
			fab->name_server.service_cmp = fi_ibv_dgram_ns_service_cmp;
			fab->name_server.is_service_wildcard =
					fi_ibv_dgram_ns_is_service_wildcard;

			ofi_ns_init(&fab->name_server);
			ofi_ns_start_server(&fab->name_server);
		}
		_domain->util_domain.domain_fid.ops = &fi_ibv_dgram_domain_ops;
		break;
	case FI_EP_MSG:
		if (_domain->use_xrc) {
			ret = fi_ibv_domain_xrc_init(_domain);
			if (ret)
				goto err4;
		}
		_domain->util_domain.domain_fid.ops = &fi_ibv_msg_domain_ops;
		break;
	default:
		VERBS_INFO(FI_LOG_DOMAIN, "Ivalid EP type is provided, "
			   "EP type :%d\n", _domain->ep_type);
		ret = -FI_EINVAL;
		goto err4;
	}

	if (!strncmp(info->domain_attr->name, "hfi1", strlen("hfi1")) ||
	    !strncmp(info->domain_attr->name, "qib", strlen("qib"))) {
		_domain->post_send = fi_ibv_post_send_track_credits;
		_domain->poll_cq = fi_ibv_poll_cq_track_credits;
	} else {
		_domain->post_send = ibv_post_send;
		_domain->poll_cq = ibv_poll_cq;
	}

	*domain = &_domain->util_domain.domain_fid;
	return FI_SUCCESS;
err4:
	ofi_mr_cache_cleanup(&_domain->cache);
	if (ibv_dealloc_pd(_domain->pd))
		VERBS_INFO_ERRNO(FI_LOG_DOMAIN,
				 "ibv_dealloc_pd", errno);
err3:
	fi_freeinfo(_domain->info);
err2:
	if (ofi_domain_close(&_domain->util_domain))
		VERBS_INFO(FI_LOG_DOMAIN,
			   "ofi_domain_close fails");
err1:
	free(_domain);
	return ret;
}

static int fi_ibv_trywait(struct fid_fabric *fabric, struct fid **fids, int count)
{
	struct fi_ibv_cq *cq;
	struct fi_ibv_eq *eq;
	int ret, i;

	for (i = 0; i < count; i++) {
		switch (fids[i]->fclass) {
		case FI_CLASS_CQ:
			cq = container_of(fids[i], struct fi_ibv_cq, util_cq.cq_fid.fid);
			ret = fi_ibv_cq_trywait(cq);
			if (ret)
				return ret;
			break;
		case FI_CLASS_EQ:
			eq = container_of(fids[i], struct fi_ibv_eq, eq_fid.fid);
			ret = fi_ibv_eq_trywait(eq);
			if (ret)
				return ret;
			break;
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
	int ret;

	fab = container_of(fid, struct fi_ibv_fabric, util_fabric.fabric_fid.fid);
	ret = ofi_fabric_close(&fab->util_fabric);
	if (ret)
		return ret;
	free(fab);

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
	const struct fi_info *cur, *info = fi_ibv_util_prov.info;
	int ret = FI_SUCCESS;

	fab = calloc(1, sizeof(*fab));
	if (!fab)
		return -FI_ENOMEM;

	for (cur = info; cur; cur = info->next) {
		ret = ofi_fabric_init(&fi_ibv_prov, cur->fabric_attr, attr,
				      &fab->util_fabric, context);
		if (ret != -FI_ENODATA)
			break;
	}
	if (ret) {
		free(fab);
		return ret;
	}

	fab->info = cur;

	*fabric = &fab->util_fabric.fabric_fid;
	(*fabric)->fid.fclass = FI_CLASS_FABRIC;
	(*fabric)->fid.ops = &fi_ibv_fi_ops;
	(*fabric)->ops = &fi_ibv_ops_fabric;

	return 0;
}
