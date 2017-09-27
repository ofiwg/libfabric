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

#include <fi_util.h>
#include "fi_verbs.h"
#include "ep_rdm/verbs_rdm.h"

#include "fi_verbs.h"

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

	/* Enable local write access by default for FI_EP_RDM which hides local
	 * registration requirements. This allows to avoid buffering or double
	 * registration */
	if (!(md->domain->info->caps & FI_LOCAL_MR) ||
	    (md->domain->info->domain_attr->mr_mode & FI_MR_LOCAL))
		fi_ibv_access |= IBV_ACCESS_LOCAL_WRITE;

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
	if(md->domain->eq && (md->domain->eq_flags & FI_REG_MR)) {
		struct fi_eq_entry entry = {
			.fid = &md->mr_fid.fid,
			.context = context
		};
		fi_ibv_eq_write_event(md->domain->eq, FI_MR_COMPLETE,
			 	      &entry, sizeof(entry));
	}
	return 0;

err:
	free(md);
	return -errno;
}

static int fi_ibv_mr_regv(struct fid *fid, const struct iovec * iov,
		size_t count, uint64_t access, uint64_t offset, uint64_t requested_key,
		uint64_t flags, struct fid_mr **mr, void *context)
{
	if (count > VERBS_MR_IOV_LIMIT) {
		VERBS_WARN(FI_LOG_FABRIC,
			   "iov count > %d not supported\n",
			   VERBS_MR_IOV_LIMIT);
		return -FI_EINVAL;
	}
	return fi_ibv_mr_reg(fid, iov->iov_base, iov->iov_len, access, offset,
			requested_key, flags, mr, context);
}

static int fi_ibv_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
		uint64_t flags, struct fid_mr **mr)
{
	return fi_ibv_mr_regv(fid, attr->mr_iov, attr->iov_count, attr->access,
			0, attr->requested_key, flags, mr, attr->context);
}

static int fi_ibv_domain_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct fi_ibv_domain *domain;
	struct fi_ibv_eq *eq;

	domain = container_of(fid, struct fi_ibv_domain, domain_fid.fid);

	switch (bfid->fclass) {
	case FI_CLASS_EQ:
		eq = container_of(bfid, struct fi_ibv_eq, eq_fid);
		domain->eq = eq;
		domain->eq_flags = flags;
		break;
	default:
		return -EINVAL;
	}

	return 0;
}

static void *fi_ibv_rdm_cm_progress_thread(void *dom)
{
	struct fi_ibv_domain *domain =
		(struct fi_ibv_domain *)dom;
	struct slist_entry *item, *prev;
	while (domain->rdm_cm->fi_ibv_rdm_tagged_cm_progress_running) {
		struct fi_ibv_rdm_ep *ep = NULL;
		slist_foreach(&domain->ep_list, item, prev) {
			(void) prev;
			ep = container_of(item, struct fi_ibv_rdm_ep,
					  list_entry);
			if (fi_ibv_rdm_cm_progress(ep)) {
				VERBS_INFO (FI_LOG_EP_DATA,
				            "fi_ibv_rdm_cm_progress error\n");
				abort();
			}
		}
		usleep(domain->rdm_cm->cm_progress_timeout);
	}
	return NULL;
}

static int fi_ibv_domain_close(fid_t fid)
{
	struct fi_ibv_domain *domain;
	struct fi_ibv_rdm_av_entry *av_entry = NULL;
	struct slist_entry *item;
	void *status = NULL;
	int ret;

	domain = container_of(fid, struct fi_ibv_domain, domain_fid.fid);

	if (domain->rdm) {
		domain->rdm_cm->fi_ibv_rdm_tagged_cm_progress_running = 0;
		pthread_join(domain->rdm_cm->cm_progress_thread, &status);
		pthread_mutex_destroy(&domain->rdm_cm->cm_lock);

		for (item = slist_remove_head(
				&domain->rdm_cm->av_removed_entry_head);
	     	     item;
	     	     item = slist_remove_head(
				&domain->rdm_cm->av_removed_entry_head)) {
			av_entry = container_of(item,
						struct fi_ibv_rdm_av_entry,
						removed_next);
			fi_ibv_rdm_overall_conn_cleanup(av_entry);
			ofi_freealign(av_entry);
		}
		rdma_destroy_ep(domain->rdm_cm->listener);
		free(domain->rdm_cm);
	}

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
	.bind = fi_ibv_domain_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_mr fi_ibv_domain_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = fi_ibv_mr_reg,
	.regv = fi_ibv_mr_regv,
	.regattr = fi_ibv_mr_regattr,
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
	.query_atomic = fi_ibv_query_atomic,
};

static struct fi_ops_domain fi_ibv_rdm_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = fi_ibv_rdm_av_open,
	.cq_open = fi_ibv_rdm_cq_open,
	.endpoint = fi_ibv_rdm_open_ep,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = fi_rbv_rdm_cntr_open,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = fi_ibv_query_atomic,
};

static int
fi_ibv_domain(struct fid_fabric *fabric, struct fi_info *info,
	   struct fid_domain **domain, void *context)
{
	struct fi_ibv_domain *_domain;
	struct fi_ibv_fabric *fab;
	struct fi_info *fi;
	int param = 0, ret;

	fi = fi_ibv_get_verbs_info(info->domain_attr->name);
	if (!fi)
		return -FI_EINVAL;

	fab = container_of(fabric, struct fi_ibv_fabric,
			   util_fabric.fabric_fid);
	ret = ofi_check_domain_attr(&fi_ibv_prov, fabric->api_version,
				    fi->domain_attr, info->domain_attr);
	if (ret)
		return ret;

	_domain = calloc(1, sizeof *_domain);
	if (!_domain)
		return -FI_ENOMEM;

	_domain->info = fi_dupinfo(info);
	if (!_domain->info)
		goto err1;

	_domain->rdm = FI_IBV_EP_TYPE_IS_RDM(info);
	if (_domain->rdm) {
		_domain->rdm_cm = calloc(1, sizeof(*_domain->rdm_cm));
		if (!_domain->rdm_cm) {
			ret = -FI_ENOMEM;
			goto err2;
		}
		_domain->rdm_cm->cm_progress_timeout =
			FI_IBV_RDM_CM_THREAD_TIMEOUT;
		if (!fi_param_get_int(&fi_ibv_prov,
				      "rdm_thread_timeout",
				      &param)) {
			if (param < 0) {
				VERBS_INFO(FI_LOG_CORE,
				   	   "invalid value of "
					   "rdm_thread_timeout\n");
				ret = -FI_EINVAL;
				goto err2;
			} else {
				_domain->rdm_cm->cm_progress_timeout = param;
			}
		}
		slist_init(&_domain->rdm_cm->av_removed_entry_head);

		pthread_mutex_init(&_domain->rdm_cm->cm_lock, NULL);
		_domain->rdm_cm->fi_ibv_rdm_tagged_cm_progress_running = 1;
		ret = pthread_create(&_domain->rdm_cm->cm_progress_thread,
				     NULL, &fi_ibv_rdm_cm_progress_thread,
				     (void *)_domain);
		if (ret) {
			VERBS_INFO(FI_LOG_EP_CTRL,
				   "Failed to launch CM progress thread, "
				   "err :%d\n", ret);
			ret = -FI_EOTHER;
			goto err2;
		}
	}
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
	_domain->domain_fid.mr = &fi_ibv_domain_mr_ops;
	if (_domain->rdm) {
		_domain->domain_fid.ops = &fi_ibv_rdm_domain_ops;

		_domain->rdm_cm->ec = rdma_create_event_channel();

		if (!_domain->rdm_cm->ec) {
			VERBS_INFO(FI_LOG_EP_CTRL,
				"Failed to create listener event channel: %s\n",
				strerror(errno));
			ret = -FI_EOTHER;
			goto err2;
		}

		if (fi_fd_nonblock(_domain->rdm_cm->ec->fd) != 0) {
			VERBS_INFO_ERRNO(FI_LOG_EP_CTRL, "fcntl", errno);
			ret = -FI_EOTHER;
			goto err3;
		}

		if (rdma_create_id(_domain->rdm_cm->ec,
				   &_domain->rdm_cm->listener, NULL, RDMA_PS_TCP))
		{
			VERBS_INFO(FI_LOG_EP_CTRL, "Failed to create cm listener: %s\n",
				   strerror(errno));
			ret = -FI_EOTHER;
			goto err3;
		}
		_domain->rdm_cm->is_bound = 0;
	} else {
		_domain->domain_fid.ops = &fi_ibv_domain_ops;
	}
	_domain->fab = fab;

	*domain = &_domain->domain_fid;
	return 0;
err3:
	if (_domain->rdm)
		rdma_destroy_event_channel(_domain->rdm_cm->ec);
err2:
	if (_domain->rdm)
		free(_domain->rdm_cm);
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
	struct fi_info *info;
	int ret;

	ret = fi_ibv_init_info();
	if (ret)
		return ret;

	fab = calloc(1, sizeof(*fab));
	if (!fab)
		return -FI_ENOMEM;

	for (info = verbs_info; info; info = info->next) {
		ret = ofi_fabric_init(&fi_ibv_prov, info->fabric_attr, attr,
				      &fab->util_fabric, context);
		if (ret != -FI_ENODATA)
			break;
	}
	if (ret) {
		free(fab);
		return ret;
	}

	*fabric = &fab->util_fabric.fabric_fid;
	(*fabric)->fid.ops = &fi_ibv_fi_ops;
	(*fabric)->ops = &fi_ibv_ops_fabric;

	return 0;
}
