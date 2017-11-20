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

static int mlx_av_write_event(
				struct mlx_av *av, uint64_t data,
				int err, void *context)
{
	struct fi_eq_err_entry entry;
	size_t size;
	uint64_t flags;

	entry.fid = &(av->av.fid);
	entry.context = context;
	entry.data = data;

	if (err) {
		entry.err = err;
		size = sizeof(struct fi_eq_err_entry);
		flags = UTIL_FLAG_ERROR;
	} else {
		size = sizeof(struct fi_eq_entry);
		flags = 0;
	}

	fi_eq_write(
		&(av->eq->eq_fid), FI_AV_COMPLETE,
		&entry, size, flags);
	return FI_SUCCESS;
}

static int mlx_av_remove(
			struct fid_av *fi_av, fi_addr_t *fi_addr, size_t count,
			uint64_t flags)
{
	struct mlx_av *av;
	int i;

	av = container_of(fi_av, struct mlx_av, av);
	if ((av->async) && (!av->eq)) {
		return -FI_ENOEQ;
	}

	for (i = 0; i < count; ++i) {
		ucp_ep_destroy((ucp_ep_h)(fi_addr[i]));
	}
	return FI_SUCCESS;
}


static inline int mlx_av_resolve_if_addr(
		const struct sockaddr *saddr,
		char **address)
{
	char peer_host[INET_ADDRSTRLEN] = {0};
	char peer_serv[INET_ADDRSTRLEN] = {0};
	int intserv, peer_host_len, peer_serv_len;
	peer_host_len = peer_serv_len = INET_ADDRSTRLEN;
	int rv;

	rv = getnameinfo(saddr, sizeof(struct sockaddr_in),
		peer_host, peer_host_len,
		peer_serv, peer_serv_len,
		NI_NUMERICSERV|NI_NUMERICHOST);
	if (0 != rv) {
		FI_WARN( &mlx_prov, FI_LOG_CORE,
			"Unable to resolve address: %s \n",
			 gai_strerror(rv));
		return -FI_EINVAL;
	}

	intserv = atoi(peer_serv);
	(*address) = ofi_ns_resolve_name(
		&mlx_descriptor.name_serv,
		peer_host, &intserv);
	if (!(*address)) {
		FI_WARN( &mlx_prov, FI_LOG_CORE,
			"Unable to resolve address: %s:%s\n",
			peer_host, peer_serv);
		return -FI_EINVAL;
	}
	return FI_SUCCESS;
}

static int mlx_av_insert(
			struct fid_av *fi_av, const void *addr, size_t count,
			fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	struct mlx_av *av;
	struct mlx_ep *ep;
	int i;
	ucs_status_t status = UCS_OK;
	int added = 0;

	av = container_of(fi_av, struct mlx_av, av);
	ep = av->ep;

	if ((av->async) && (!av->eq)) {
		return -FI_ENOEQ;
	}

	for ( i = 0; i < count ; ++i) {
		ucp_ep_params_t ep_params = {};

		if (mlx_descriptor.use_ns) {
			if (mlx_av_resolve_if_addr(
				(struct sockaddr*)
				  (&(((struct sockaddr_in*)addr)[i])),
				(char**)&ep_params.address) != FI_SUCCESS)
				break;
		} else {
			ep_params.address = (const ucp_address_t *)
				(&(((const char *)addr)[i * FI_MLX_MAX_NAME_LEN]));
		}

		ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
		FI_WARN( &mlx_prov, FI_LOG_CORE,
			"Try to insert address #%d, offset=%d (size=%ld)"
			" fi_addr=%p \naddr = %s\n",
			i, i * FI_MLX_MAX_NAME_LEN, count,
			fi_addr, &(((const char *)addr)[i * FI_MLX_MAX_NAME_LEN]));

		status = ucp_ep_create( ep->worker,
					&ep_params,
					(ucp_ep_h *)(&(fi_addr[i])));
		if (mlx_descriptor.use_ns) {
			free((void*)ep_params.address);
		}
		if (status == UCS_OK) {
			FI_WARN( &mlx_prov, FI_LOG_CORE, "address inserted\n");
			added++;
		} else {
			if (av->eq) {
				mlx_av_write_event( av, i,
					MLX_TRANSLATE_ERRCODE(status),
					context);
			}
			break;
		}
	}

	if (av->eq) {
		mlx_av_write_event(av, added, 0, context);
		count = 0;
	} else {
		count = added;
	}
	return count;
}


static int mlx_av_close(fid_t fid)
{
	struct mlx_av *fid_av;
	fid_av = container_of(fid, struct mlx_av, av);
	free (fid_av);
	return FI_SUCCESS;
}

static int mlx_av_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct mlx_av *av;
	struct util_eq *eq;

	av = container_of(fid, struct mlx_av, av.fid);
	if ((!(av->async)) || (bfid->fclass != FI_CLASS_EQ)){
		FI_WARN( &mlx_prov, FI_LOG_EP_CTRL,
			"Try to bind not a EQ to AV, "
			"or attemt to bind EQ and syncronious AV\n");
		return -FI_EINVAL;
	}
	eq = container_of(bfid, struct util_eq, eq_fid.fid);
	av->eq = eq;
	return FI_SUCCESS;
}

static struct fi_ops mlx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = mlx_av_close,
	.bind = mlx_av_bind,
};

static struct fi_ops_av mlx_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = mlx_av_insert,
	.remove = mlx_av_remove,
};

int mlx_av_open(
		struct fid_domain *fi_domain, struct fi_av_attr *attr,
		struct fid_av **fi_av, void *context)
{
	struct mlx_domain *domain;
	struct mlx_av *av;
	int type = FI_AV_MAP;
	size_t count = 64;
	domain = container_of(fi_domain, struct mlx_domain, u_domain.domain_fid);

	int is_async = 0;
	if (attr) {
		switch (attr->type) {
		case FI_AV_MAP:
			type = attr->type;
			break;
		case FI_AV_UNSPEC:
			/* Set FI_AV_MAP by default */
			type = FI_AV_MAP;
			break;
		default:
			return -EINVAL;
		}
		if (attr->flags & FI_EVENT){
			is_async = 1;
		}
		count = attr->count;
	}

	av = (struct mlx_av *)calloc(1, sizeof(struct mlx_av));
	if (!av)
		return -ENOMEM;

	av->domain = domain;
	av->async = is_async;
	av->type = type;
	av->eq = NULL;

	if (mlx_descriptor.use_ns) {
		av->addr_len = sizeof(struct sockaddr_in);
	} else {
		av->addr_len = FI_MLX_MAX_NAME_LEN;
	}

	av->count = count;
	av->av.fid.fclass = FI_CLASS_AV;
	av->av.fid.context = context;
	av->av.fid.ops = &mlx_fi_ops;
	av->av.ops = &mlx_av_ops;

	*fi_av = &av->av;
	return FI_SUCCESS;
}


