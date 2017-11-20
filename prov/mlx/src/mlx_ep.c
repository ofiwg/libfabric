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

static void mlx_ep_progress( struct util_ep *util_ep)
{
	struct mlx_ep *ep;
	ep = container_of(util_ep, struct mlx_ep, ep);
	ucp_worker_progress(ep->worker);
}


static ssize_t mlx_ep_cancel( fid_t fid, void *ctx)
{
	struct mlx_ep *ep;
	void *req;
	struct fi_context *context = (struct fi_context*)ctx;

	ep = container_of( fid, struct mlx_ep, ep.ep_fid.fid);
	if (!ep->ep.domain)
		return -EBADF;
	if (!context)
		return -EINVAL;
	if (context->internal[0] == NULL)
		return -FI_EINVAL;

	req = context->internal[0];
	ucp_request_cancel(ep->worker, req);

	return FI_SUCCESS;
}

static int mlx_ep_getopt( fid_t fid, int level, int optname,
			void *optval, size_t *optlen)
{
	return -ENOSYS;
}

static int mlx_ep_setopt(fid_t fid, int level, int optname,
		const void *optval, size_t optlen)
{
	return FI_SUCCESS;
}

static int mlx_ep_close(fid_t fid)
{
	struct mlx_ep *ep;
	ucs_status_t status = UCS_OK;
	void *addr_local = NULL;
	size_t addr_len_local;

	ep = container_of(fid, struct mlx_ep, ep.ep_fid.fid);

	if (mlx_descriptor.use_ns) {
		status = ucp_worker_get_address( ep->worker,
						(ucp_address_t **)&addr_local,
						(size_t*) &addr_len_local );
		if (status != UCS_OK)
			return MLX_TRANSLATE_ERRCODE(status);

		ofi_ns_del_local_name(&mlx_descriptor.name_serv,
					  &ep->service, addr_local);

		ucp_worker_release_address(
					ep->worker,
					(ucp_address_t *)addr_local);
	}

	ucp_worker_flush(ep->worker);
	ucp_worker_destroy(ep->worker);

	ofi_endpoint_close(&ep->ep);
	free(ep);
	return FI_SUCCESS;
}

static int mlx_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct mlx_ep *ep;
	struct util_cq *cq;

	ep = container_of(fid, struct mlx_ep, ep.ep_fid.fid);
	int status = FI_SUCCESS;

	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct util_cq, cq_fid.fid);
		status = ofi_ep_bind_cq(&ep->ep, cq, flags);
		break;
	case FI_CLASS_AV:
		if (ep->av) {
			FI_WARN( &mlx_prov, FI_LOG_EP_CTRL,
				"AV already binded\n");
			status = -FI_EINVAL;
			break;
		}
		ep->av = container_of(bfid, struct mlx_av, av.fid);
		ep->av->ep = ep;
		break;
	default:
		status = -FI_EINVAL;
		break;
	}
	return status;
}


static int mlx_ep_control(fid_t fid, int command, void *arg)
{

	struct mlx_ep *ep;

	ep = container_of(fid, struct mlx_ep, ep.ep_fid.fid);
	switch (command) {
	case FI_ENABLE:
		if (!ep->ep.rx_cq || !ep->ep.tx_cq)
			return -FI_ENOCQ;
		if (!ep->av)
			return -FI_EOPBADSTATE; /* TODO: Add FI_ENOAV */
		break;
	default:
		return -FI_ENOSYS;
	}
	return FI_SUCCESS;
}

struct fi_ops_ep mlx_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = mlx_ep_cancel,
	.getopt = mlx_ep_getopt,
	.setopt = mlx_ep_setopt,
};

static struct fi_ops mlx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = mlx_ep_close,
	.bind = mlx_ep_bind,
	.control = mlx_ep_control,
};

int mlx_ep_open( struct fid_domain *domain, struct fi_info *info,
		struct fid_ep **fid, void *context)
{
	struct mlx_ep     *ep;
	struct mlx_domain *u_domain;
	int ofi_status = FI_SUCCESS;
	ucs_status_t status = UCS_OK;
	ucp_worker_params_t worker_params = { };
	worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
	worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
	u_domain = container_of( domain, struct mlx_domain, u_domain.domain_fid);

	void *addr_local = NULL;
	size_t addr_len_local;


	ep = (struct mlx_ep *) calloc(1, sizeof (struct mlx_ep));
	if (!ep) {
		return -ENOMEM;
	}

	ofi_status = ofi_endpoint_init(domain, &mlx_util_prov, info,
				       &ep->ep, context, mlx_ep_progress);
	if (ofi_status) {
		goto free_ep;
	}

	status = ucp_worker_create( u_domain->context,
				&worker_params,
				&(ep->worker));
	if (status != UCS_OK) {
		ofi_status = MLX_TRANSLATE_ERRCODE(status);
		ofi_atomic_dec32(&(u_domain->u_domain.ref));
		goto free_ep;
	}

	if (mlx_descriptor.use_ns) {
		char tmpb [FI_MLX_MAX_NAME_LEN]={0};
		status = ucp_worker_get_address( ep->worker,
			(ucp_address_t **)&addr_local,
			(size_t*) &addr_len_local );
		if (status != UCS_OK)
			return MLX_TRANSLATE_ERRCODE(status);
		ep->service = (short)((getpid() & 0xFFFF ));
		memcpy(tmpb,addr_local,addr_len_local);
		FI_INFO(&mlx_prov, FI_LOG_CORE,
			"PUBLISHED UCP address(size=%zd): [%hu] %s\n",
			addr_len_local,ep->service,(char*)(addr_local));

		ofi_ns_add_local_name(&mlx_descriptor.name_serv,
			&ep->service, tmpb);

		ucp_worker_release_address( ep->worker,
			(ucp_address_t *)addr_local);
	}

	ep->ep.ep_fid.fid.ops = &mlx_fi_ops;
	ep->ep.ep_fid.ops = &mlx_ep_ops;
	ep->ep.ep_fid.cm = &mlx_cm_ops;
	ep->ep.ep_fid.tagged = &mlx_tagged_ops;
	ep->ep.flags = info->mode;
	ep->ep.caps = u_domain->u_domain.info_domain_caps;

	*fid = &(ep->ep.ep_fid);

	return FI_SUCCESS;
free_ep:
	free(ep);
	return ofi_status;
}
