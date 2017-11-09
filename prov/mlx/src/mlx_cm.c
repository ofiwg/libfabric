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
#include <inttypes.h>

static int mlx_cm_getname_mlx_format(
			fid_t fid,
			void *addr,
			size_t *addrlen)
{
	ucs_status_t status = UCS_OK;
	void *addr_local = NULL;
	size_t addr_len_local;
	struct mlx_ep* ep;
	int ofi_status = FI_SUCCESS;

	ep = container_of(fid, struct mlx_ep, ep.ep_fid.fid); 

	status = ucp_worker_get_address( ep->worker,
					(ucp_address_t **)&addr_local,
					(size_t*) &addr_len_local );
	if (status != UCS_OK) {
		FI_WARN( &mlx_prov, FI_LOG_CORE,
			"ucp_worker_get_address error!!!\n");
		return MLX_TRANSLATE_ERRCODE(status);
	}

	if (addr_len_local > FI_MLX_MAX_NAME_LEN) {
		FI_WARN( &mlx_prov, FI_LOG_CORE,
			"Address returned by UCX is too long %"PRIu64"\n",
			 addr_len_local);
		return -FI_EINVAL;
	}

	if ((*addrlen) < FI_MLX_MAX_NAME_LEN) {
		FI_WARN( &mlx_prov, FI_LOG_CORE,
			"Buffer storage for ep address is too small %"PRIu64
			" instead of %d [%s]\n",
			*addrlen, FI_MLX_MAX_NAME_LEN, (char *)addr_local);
		ofi_status = -FI_ETOOSMALL;
	}
	FI_INFO(&mlx_prov, FI_LOG_CORE, 
		"Loaded UCP address: [%"PRIu64"]%s\n",
		addr_len_local, (char *)addr_local);

	if (addr_local != NULL)
		memcpy(addr, addr_local, (((*addrlen) < addr_len_local) ?
					  (*addrlen) : addr_len_local));

	*addrlen = FI_MLX_MAX_NAME_LEN;
	ucp_worker_release_address(
				ep->worker,
				(ucp_address_t *)addr_local);
	return ofi_status;
}

static int mlx_cm_getname_ai_format(
			fid_t fid,
			void *addr,
			size_t *addrlen)
{
	int ofi_status = FI_SUCCESS;
	struct mlx_ep* ep;
	ep = container_of(fid, struct mlx_ep, ep.ep_fid.fid);
	if (ep->addr) {
		if (ep->addr_len > *addrlen) {
			ofi_status = -FI_ETOOSMALL;
		} else {
			memcpy(addr, ep->addr, ep->addr_len);
		}
		*addrlen = ep->addr_len;
	} else {
		char *hostname = mlx_descriptor.localhost;
		int service = (((getpid() & 0xFFFF)));
		struct addrinfo hints;
		struct addrinfo *res;

		memset(&hints, 0, sizeof(hints));
		hints.ai_flags = 0;
		hints.ai_family = AF_INET;
		hints.ai_socktype = SOCK_STREAM;
		hints.ai_protocol = IPPROTO_TCP;
		hints.ai_addrlen = 0;
		hints.ai_addr = NULL;
		hints.ai_canonname = NULL;
		hints.ai_next = NULL;

		if(getaddrinfo(hostname, NULL, &hints, &res) != 0) {
			FI_WARN( &mlx_prov, FI_LOG_CORE,
					"Unable to resolve hostname:%s\n",hostname);
		}
		FI_INFO(&mlx_prov, FI_LOG_CORE,
			"Loaded IPv4 address: [%"PRIu64"]%s:%d\n",
			res->ai_addrlen, hostname, service);

		memcpy(addr,res->ai_addr,res->ai_addrlen);
		((struct sockaddr_in*)addr)->sin_port = htons((short)service);
		freeaddrinfo(res);

		*addrlen = sizeof(struct sockaddr);
	}

	return ofi_status;
}

static int mlx_cm_getname(
			fid_t fid,
			void *addr,
			size_t *addrlen)
{
	int ofi_status = FI_SUCCESS;
	if (mlx_descriptor.use_ns) {
		ofi_status = mlx_cm_getname_ai_format(fid, addr, addrlen);
	} else {
		ofi_status = mlx_cm_getname_mlx_format(fid, addr, addrlen);
	}
	return ofi_status;
}



struct fi_ops_cm mlx_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.getname = mlx_cm_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
};
