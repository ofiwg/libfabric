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

static int mlx_cm_getname(
			fid_t fid,
			void *addr,
			size_t *addrlen)
{
	ucs_status_t status = UCS_OK;
	void* addr_local = NULL;
	size_t addr_len_local;
	struct mlx_ep* ep;
	int ofi_status = FI_SUCCESS;

	ep = container_of(fid, struct mlx_ep, ep.ep_fid.fid); 

	status = ucp_worker_get_address( ep->worker,
					(ucp_address_t **)&addr_local,
					(size_t*) &addr_len_local );
	if (status != UCS_OK) {
		return MLX_TRANSLATE_ERRCODE(status);
	}

	if ((*addrlen) < addr_len_local) {
		FI_WARN( &mlx_prov, FI_LOG_CORE,
			"Buffer storage for ep address is too small %d "
			"instead of %d [%s]\n",
			addrlen, addr_len_local, (char *)addr_local);
		ofi_status = -FI_ETOOSMALL;
	}
	FI_INFO( &mlx_prov, FI_LOG_CORE, 
		"Loaded UCP adress: [%d]%s\n",
		addr_len_local, (char *)addr_local);

	memcpy( addr, addr_local,
		(((*addrlen)<addr_len_local) ? (*addrlen):addr_len_local));

	*addrlen = addr_len_local;
	ucp_worker_release_address(
				ep->worker,
				(ucp_address_t *)addr_local);
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
