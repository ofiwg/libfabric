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
	struct mlx_ave *ep_ave;
	int i;

	av = container_of(fi_av, struct mlx_av, av);
	if ((av->async) && (!av->eq)) {
		return -FI_ENOEQ;
	}

	for (i = 0; i < count; ++i) {
		ep_ave = (struct mlx_ave *)fi_addr[i];
		ucp_ep_destroy((ucp_ep_h)(ep_ave->uep));
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
	struct mlx_avblock *avb_eps;
	struct mlx_avblock *avb_addrs;
	size_t i;
	ucs_status_t status = UCS_OK;
	int added = 0;

	av = container_of(fi_av, struct mlx_av, av);
	ep = av->ep;

	if ((av->async) && (!av->eq)) {
		return -FI_ENOEQ;
	}

	avb_eps = malloc(sizeof(struct mlx_ave) * count + sizeof(struct mlx_avblock));
	if (!avb_eps) {
		free(avb_eps);
		return - FI_ENOMEM;
	}
	avb_eps->next = av->ep_block;
	av->ep_block = avb_eps;

	if (mlx_descriptor.enable_spawn) {
		avb_addrs = malloc(av->addr_len * count + sizeof(struct mlx_avblock));
		if (!avb_addrs) {
			free(avb_addrs);
			return - FI_ENOMEM;
		}
		avb_addrs->next = av->addr_blocks;
		av->addr_blocks = avb_addrs;
	}

	for (i = 0; i < count ; ++i) {
		struct mlx_ave *ep_ave;
		ucp_ep_params_t ep_params = { 0 };

		if (mlx_descriptor.use_ns) {
			if (mlx_av_resolve_if_addr(
				(struct sockaddr*)
				  (&(((struct sockaddr_in *) addr)[i])),
				(char**) &ep_params.address) != FI_SUCCESS)
				break;
		} else {
			ep_params.address = (const ucp_address_t *)
				(&(((const char *) addr)[i * av->addr_len]));
		}

		ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
		FI_WARN(&mlx_prov, FI_LOG_CORE,
			"Try to insert address #%zd, offset=%zd (size=%zd)"
			" fi_addr=%p \n",
			i, i * av->addr_len, count,
			fi_addr);
		ep_ave = &(((struct mlx_ave *)(avb_eps->payload))[i]);

		ep_ave->addr = (mlx_descriptor.enable_spawn) ?
			(&avb_addrs->payload[i * av->addr_len]) : NULL;
		fi_addr[i] = (fi_addr_t)ep_ave;

		status = ucp_ep_create(ep->worker, &ep_params,
						&(ep_ave->uep));
		if (status == UCS_OK) {
			FI_WARN(&mlx_prov, FI_LOG_CORE, "address inserted\n");
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

static inline void mlx_del_avb_list(struct mlx_avblock *avb)
{
	struct mlx_avblock *avb_tmp;
	if (!avb)
		return;
	do {
		avb_tmp = avb->next;
		free(avb);
		avb = avb_tmp;
	} while (avb);
}

static int mlx_av_close(fid_t fid)
{
	struct mlx_av *av;
	av = container_of(fid, struct mlx_av, av);
	mlx_del_avb_list(av->addr_blocks);
	mlx_del_avb_list(av->ep_block);
	free (av);
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

static int mlx_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr,
			size_t *addrlen)
{
	struct mlx_ave *ave;
	struct mlx_av *mav;
	size_t realsz;

	ave = (struct mlx_ave*) fi_addr;
	mav = container_of(av, struct mlx_av, av.fid);
	realsz = MIN(*addrlen, mav->addr_len);
	memcpy(addr, ave->addr, realsz);
	*addrlen = mav->addr_len;
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
	.lookup = mlx_av_lookup,
};

int mlx_av_open(
		struct fid_domain *fi_domain, struct fi_av_attr *attr,
		struct fid_av **fi_av, void *context)
{
	struct mlx_domain *domain;
	struct mlx_av *av;
	int type = FI_AV_MAP;
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
		if (attr->name || attr->map_addr) {
			return -EINVAL;
		}
		if (attr->flags & FI_EVENT){
			is_async = 1;
		}
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

	av->av.fid.fclass = FI_CLASS_AV;
	av->av.fid.context = context;
	av->av.fid.ops = &mlx_fi_ops;
	av->av.ops = &mlx_av_ops;

	*fi_av = &av->av;
	return FI_SUCCESS;
}


