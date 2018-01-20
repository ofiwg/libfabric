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
		util_buf_pool_destroy(domain->fast_path_pool);
		free(domain);
	}
	return status;
}

static struct fi_ops mlx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = mlx_domain_close,
};

struct fi_ops_domain mlx_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = mlx_av_open,
	.cq_open = mlx_cq_open,
	.endpoint = mlx_ep_open,
	.poll_open = fi_poll_create,
};


struct fi_ops_mr mlx_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = fi_no_mr_reg,
	.regv = fi_no_mr_regv,
	.regattr = fi_no_mr_regattr,
};

int mlx_domain_open(struct fid_fabric *fabric, struct fi_info *info,
                     struct fid_domain **fid, void *context)
{
	ucs_status_t status = UCS_OK;
	int ofi_status;
	struct mlx_domain* domain;
	const ucp_params_t params = {
		.features = UCP_FEATURE_TAG,
		.request_size = sizeof(struct mlx_request),
		.request_init = NULL,
		.request_cleanup = NULL,
		.field_mask = UCP_PARAM_FIELD_FEATURES |
			      UCP_PARAM_FIELD_REQUEST_SIZE,
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

	ofi_status = ofi_domain_init(fabric, info,
				     &(domain->u_domain), context);
	if (ofi_status) {
		goto domain_free;
	}

	status = ucp_init(&params, mlx_descriptor.config,
			  &(domain->context));
	if (status != UCS_OK) {
		ofi_status = MLX_TRANSLATE_ERRCODE(status);
		goto destroy_domain;
	}
	fastlock_init(&(domain->fpp_lock));

	ofi_status = util_buf_pool_create(
			&domain->fast_path_pool,
			sizeof(struct mlx_request),
			16, 0, 1024 );
	if (ofi_status)
		goto cleanup_mlx;

	domain->u_domain.domain_fid.fid.ops = &mlx_fi_ops;
	domain->u_domain.domain_fid.ops = &mlx_domain_ops;
	domain->u_domain.domain_fid.mr = &mlx_mr_ops;

	*fid = &(domain->u_domain.domain_fid);
	return FI_SUCCESS;

cleanup_mlx:
	ucp_cleanup(domain->context);
destroy_domain:
	ofi_domain_close(&(domain->u_domain));
domain_free:
	free(domain);
	if (!ofi_status) {
		ofi_status = FI_ENETUNREACH;
	}
	return ofi_status;
}

