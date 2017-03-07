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

int mlx_fabric_close(struct fid *fid)
{
	int status;
	status = ofi_fabric_close(
			container_of(fid, struct util_fabric, fabric_fid.fid));
	return status;
}

static struct fi_ops mlx_fabric_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = mlx_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_fabric mlx_fabric_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = mlx_domain_open,
	.passive_ep = fi_no_passive_ep,
	.eq_open = ofi_eq_create,
	.wait_open = ofi_wait_fd_open,
	.trywait = fi_no_trywait,
};

int mlx_fabric_open(
		struct fi_fabric_attr *attr,
		struct fid_fabric **fabric,
		void *context)
{
	struct mlx_fabric *fabric_priv;
	int status;

	FI_INFO( &mlx_prov, FI_LOG_CORE, "\n" );

	if (strcmp(attr->name, FI_MLX_FABRIC_NAME)) {
		return -FI_ENODATA;
	}

	fabric_priv = calloc(1, sizeof(struct mlx_fabric));
	if (!fabric_priv) {
		return -FI_ENOMEM;
	}

	status = ofi_fabric_init(&mlx_prov, &mlx_fabric_attrs, attr,
				 &(fabric_priv->u_fabric), context);
	if (status) {
		FI_INFO( &mlx_prov, FI_LOG_CORE,
			"Error in ofi_fabric_init: %d\n", status);
		free(fabric_priv);
		return status;
	}

	fabric_priv->u_fabric.fabric_fid.fid.ops = &mlx_fabric_fi_ops;
	fabric_priv->u_fabric.fabric_fid.ops = &mlx_fabric_ops;
	*fabric = &(fabric_priv->u_fabric.fabric_fid);

	return FI_SUCCESS;
}
