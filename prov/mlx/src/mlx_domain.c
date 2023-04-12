/*
 * Copyright (c) 2016 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
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
#include "mlx.h"

static int mlx_domain_close(fid_t fid)
{
	struct mlx_domain *domain;
	int status;

	domain = container_of( fid,
				struct mlx_domain,
				u_domain.domain_fid.fid);

	ucp_cleanup(domain->context);
	status = ofi_domain_close( &(domain->u_domain));
	if (!status) {
		free(domain);
	}
	return status;
}

int mlx_dom_control(struct fid *fid, int command, void *arg)
{
	ucs_status_t status;
	struct mlx_domain *domain = container_of(fid, struct mlx_domain,
			u_domain.domain_fid.fid);
	struct mlx_mr_key_descr *mr_dsc = (struct mlx_mr_key_descr*)arg;
	struct mlx_mr_rkey *rkey;

	if (command == FI_MLX_MR_ADD_KEY) {
		struct mlx_ave *ep_ave = (struct mlx_ave *)mr_dsc->owner_addr;
		rkey = malloc(sizeof(struct mlx_mr_rkey));
		status = ucp_ep_rkey_unpack(ep_ave->uep,
				mr_dsc->pkey, &(rkey->rkey));
		if (status != UCS_OK) {
			free (rkey);
			return MLX_TRANSLATE_ERRCODE(status);
		}

		rkey->id.owner_addr = mr_dsc->owner_addr;
		rkey->id.key = mr_dsc->mr_key;
		HASH_ADD(hh, domain->remote_keys, id, sizeof(rkey->id), rkey);

		FI_DBG( &mlx_prov,FI_LOG_DEBUG,
			"MLX/RMA: added key {%llu:%llu}\n",
			rkey->id.owner_addr,
			rkey->id.key);
		return FI_SUCCESS;
	} else if (command == FI_MLX_MR_DEL_KEY) {
		struct mlx_mr_rkey tmp_rkey;
		tmp_rkey.id.owner_addr = mr_dsc->owner_addr;
		tmp_rkey.id.key = mr_dsc->mr_key;
		HASH_FIND(hh, domain->remote_keys, &tmp_rkey.id, sizeof(tmp_rkey.id),
				rkey);
		if (rkey) {
			FI_DBG( &mlx_prov,FI_LOG_DEBUG,
				"MLX/RMA: removed key {%llu:%llu}\n",
				tmp_rkey.id.owner_addr,
				tmp_rkey.id.key);
			HASH_DEL(domain->remote_keys, rkey);
			ucp_rkey_destroy(rkey->rkey);
			free(rkey);
		}
		return FI_SUCCESS;
	}
	return -FI_ENOSYS;
}

static struct fi_ops mlx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = mlx_domain_close,
	.control = mlx_dom_control,
};

struct fi_ops_domain mlx_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = mlx_av_open,
	.cq_open = mlx_cq_open,
	.endpoint = mlx_ep_open,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = mlx_cntr_open,
	.poll_open = fi_poll_create,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = fi_no_query_atomic,
};


int mlx_mr_close(struct fid *fid)
{
	struct mlx_mr *mr;
	struct ofi_mr *omr;
	int ret;
	struct mlx_domain *domain;
	ucs_status_t status = UCS_OK;

	mr = container_of(fid, struct mlx_mr, omr.mr_fid.fid);
	omr = container_of(fid, struct ofi_mr, mr_fid.fid);

	ofi_spin_lock(&omr->domain->lock);
	ret = ofi_mr_map_remove(&omr->domain->mr_map, omr->key);
	ofi_spin_unlock(&omr->domain->lock);
	if (ret)
		return ret;

	domain = container_of( omr->domain,
				struct mlx_domain,
				u_domain);
	status = ucp_mem_unmap(domain->context, mr->memh);
	ofi_atomic_dec32(&omr->domain->ref);
	free(mr);
	return MLX_TRANSLATE_ERRCODE(status);
}

int mlx_mr_control(struct fid *fid, int command, void *arg)
{
	if (command == FI_MLX_MR_GET_KEY) {
		ucs_status_t status;
		void * tmp = NULL;
		size_t tmp_size = 0;
		struct mlx_domain *domain;
		struct mlx_mr_key_descr *mr_dsc = (struct mlx_mr_key_descr*)arg;
		struct mlx_mr *mlx_mr = container_of(fid, struct mlx_mr, omr.mr_fid.fid);

		domain = container_of(mlx_mr->omr.domain, struct mlx_domain, u_domain);
		status = ucp_rkey_pack(domain->context,
					mlx_mr->memh,
					&tmp,
					&tmp_size);
		if (status != UCS_OK)
			return MLX_TRANSLATE_ERRCODE(status);
		if (mr_dsc->pkey_size < tmp_size) {
			FI_WARN( &mlx_prov, FI_LOG_MR,
				"Rkey buffer is too small: expected %lu, provided %lu\n",
				mr_dsc->pkey_size, tmp_size);
			return -FI_ENOMEM;
		}

		memcpy(mr_dsc->pkey, tmp, tmp_size);
		mr_dsc->pkey_size = tmp_size;
		ucp_rkey_buffer_release(tmp);
		return FI_SUCCESS;
	}
	return -FI_ENOSYS;
}

static struct fi_ops mlx_mr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = mlx_mr_close,
	.control = mlx_mr_control,
	.ops_open = fi_no_ops_open,
};

int mlx_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
		   uint64_t flags, struct fid_mr **mr_fid)
{
	struct mlx_domain *m_domain;
	struct util_domain *domain;
	struct mlx_mr *mlx_mr;
	struct ofi_mr *mr;
	ucp_mem_map_params_t um_params;
	uint64_t key;
	int ret = 0;
	ucs_status_t status = UCS_OK;

	if (fid->fclass != FI_CLASS_DOMAIN || !attr || attr->iov_count <= 0)
		return -FI_EINVAL;

	domain = container_of(fid, struct util_domain, domain_fid.fid);
	m_domain = container_of(domain, struct mlx_domain, u_domain);
	mlx_mr = calloc(1, sizeof(*mlx_mr));
	if (!mlx_mr)
		return -FI_ENOMEM;
	mr = &mlx_mr->omr;

	ofi_spin_lock(&domain->lock);

	mr->mr_fid.fid.fclass = FI_CLASS_MR;
	mr->mr_fid.fid.context = attr->context;
	mr->mr_fid.fid.ops = &mlx_mr_fi_ops;
	mr->domain = domain;
	mr->flags = flags;

	um_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
			UCP_MEM_MAP_PARAM_FIELD_LENGTH | UCP_MEM_MAP_PARAM_FIELD_FLAGS;
	um_params.address = attr->mr_iov->iov_base;
	um_params.length = attr->mr_iov->iov_len;
	um_params.flags = 0;

	status = ucp_mem_map(m_domain->context, &um_params, &mlx_mr->memh);
	if (status != UCS_OK) {
		ret = MLX_TRANSLATE_ERRCODE(status);
		free(mlx_mr);
		goto out;
	}

	ret = ofi_mr_map_insert(&domain->mr_map, attr, &key, mr);
	if (ret) {
		ucp_mem_unmap(m_domain->context, mlx_mr->memh);
		free(mlx_mr);
		goto out;
	}

	mr->mr_fid.key = mr->key = key;
	mr->mr_fid.mem_desc = (void *) (uintptr_t) key;

	*mr_fid = &mr->mr_fid;
	ofi_atomic_inc32(&domain->ref);

out:
	ofi_spin_unlock(&domain->lock);
	return ret;
}


int mlx_mr_regv(struct fid *fid, const struct iovec *iov,
	        size_t count, uint64_t access, uint64_t offset,
		uint64_t requested_key, uint64_t flags,
		struct fid_mr **mr_fid, void *context)
{
	struct fi_mr_attr attr;

	attr.mr_iov = iov;
	attr.iov_count = count;
	attr.access = access;
	attr.offset = offset;
	attr.requested_key = requested_key;
	attr.context = context;
	return mlx_mr_regattr(fid, &attr, flags, mr_fid);
}

int mlx_mr_reg(struct fid *fid, const void *buf, size_t len,
	       uint64_t access, uint64_t offset, uint64_t requested_key,
	       uint64_t flags, struct fid_mr **mr_fid, void *context)
{
	struct iovec iov;

	iov.iov_base = (void *) buf;
	iov.iov_len = len;
	return mlx_mr_regv(fid, &iov, 1, access, offset, requested_key, flags,
			   mr_fid, context);
}

struct fi_ops_mr mlx_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = mlx_mr_reg,
	.regv = mlx_mr_regv,
	.regattr = mlx_mr_regattr,
};

void mlx_req_reset(void *request)
{
	struct mlx_request* mreq = (struct mlx_request*)request;
	mreq->type = MLX_FI_REQ_UNINITIALIZED;
	mreq->cq = NULL;
	mreq->ep = NULL;
}

int mlx_domain_open(struct fid_fabric *fabric, struct fi_info *info,
                     struct fid_domain **fid, void *context)
{
	ucs_status_t status = UCS_OK;
	int ofi_status;
	struct mlx_domain* domain;
	size_t univ_size;
	ucp_params_t params = {
		.features = UCP_FEATURE_TAG | UCP_FEATURE_RMA,
		.request_size = sizeof(struct mlx_request),
		.request_init = mlx_req_reset,
		.field_mask = UCP_PARAM_FIELD_FEATURES |
			UCP_PARAM_FIELD_REQUEST_SIZE | UCP_PARAM_FIELD_REQUEST_INIT,
	};

	if (!info->domain_attr->name ||
	    strcmp(info->domain_attr->name, FI_MLX_FABRIC_NAME)) {
		return -FI_EINVAL;
	}

	ofi_status = ofi_prov_check_info(&mlx_util_prov,
					 fabric->api_version,
					 info);
	if (ofi_status) {
		return ofi_status;
	}

	domain = calloc(1, sizeof(struct mlx_domain));
	if (!domain) {
		return -ENOMEM;
	}

	ofi_status = fi_param_get_size_t(NULL, "universe_size", &univ_size);
	if (ofi_status) {
		params.estimated_num_eps = univ_size;
		params.field_mask |= UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
	}

	ofi_status = ofi_domain_init(fabric, info,
				     &(domain->u_domain), context, OFI_LOCK_MUTEX);
	if (ofi_status) {
		goto domain_free;
	}

	status = ucp_init_version(FI_MLX_VERSION_MAJOR, FI_MLX_VERSION_MINOR,
			&params, mlx_descriptor.config,
			 &(domain->context));
	if (status != UCS_OK) {
		ofi_status = MLX_TRANSLATE_ERRCODE(status);
		goto destroy_domain;
	}

	domain->u_domain.domain_fid.fid.ops = &mlx_fi_ops;
	domain->u_domain.domain_fid.ops = &mlx_domain_ops;
	domain->u_domain.domain_fid.mr = &mlx_mr_ops;

	*fid = &(domain->u_domain.domain_fid);
	return FI_SUCCESS;

destroy_domain:
	ofi_domain_close(&(domain->u_domain));
domain_free:
	free(domain);
	if (!ofi_status) {
		ofi_status = FI_ENETUNREACH;
	}
	return ofi_status;
}

