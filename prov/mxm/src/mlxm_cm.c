/*
 * Copyright (c) 2013 Intel Corporation. All rights reserved.
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

#include "mlxm.h"


static int mlxm_cm_getname(fid_t fid, void *addr, size_t *addrlen)
{
    mlxm_fid_ep_t	*fid_ep;
    mxm_error_t		mxm_err;
    size_t mxm_addrlen = *addrlen;

    fid_ep = container_of(fid, struct mlxm_fid_ep, ep.fid);
    if (!fid_ep->domain)
        return -EBADF;

    mxm_err = mxm_ep_get_address(mlxm_globals.mxm_ep,
                                 addr, &mxm_addrlen);
    if (mxm_err == MXM_ERR_BUFFER_TOO_SMALL) {
            FI_WARN(&mlxm_prov,FI_LOG_CORE,
                    "Buffer storage for ep address is too small\n");
    } else {
            FI_INFO(&mlxm_prov, FI_LOG_CORE,
                    "got self ep addr, %s\n",(char*)addr);
            fid_ep->domain->mxm_addrlen = *addrlen;
    }

    return (mxm_err ? mlxm_errno(mxm_err) : 0);
}


struct fi_ops_cm mlxm_cm_ops = {
	.size    = sizeof(struct fi_ops_cm),
	.getname = mlxm_cm_getname,
        // .connect = mlxm_cm_connect,
        // .listen = mlxm_cm_listen,
        // .accept = mlxm_cm_accept,
        // .reject = mlxm_cm_reject,
        // .shutdown = mlxm_cm_shutdown,
        // .join = mlxm_cm_join,
        // .leave = mlxm_cm_leave,
};
