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

static int mlxm_av_insert(struct fid_av *av, const void *addr, size_t count,
                          fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	mlxm_fid_av_t	*fid_av;
	mlxm_fid_ep_t	*fid_ep;
	mxm_error_t		mxm_err;
	void                *mxm_addr;
	size_t	        mxm_addrlen;
	int i, err;

        fid_av = container_of(av, mlxm_fid_av_t, av);
	fid_ep = fid_av->ep;
	mxm_addrlen = fid_av->domain->mxm_addrlen;

	for (i = 0; i < count; ++i) {
                mxm_addr = (void*)&((char *)addr)[i*mxm_addrlen];
                mxm_err = mxm_ep_connect(mlxm_globals.mxm_ep, mxm_addr,
					 (mxm_conn_h*)&fi_addr[i]);
		if (mxm_err != MXM_OK) {
			err = mlxm_errno(mxm_err);
			goto err_out;
		}
                FI_INFO(&mlxm_prov, FI_LOG_AV, "connected to %s, conn %p\n",
                        (char*)mxm_addr+8,
                        *((mxm_conn_h*)&fi_addr[i]));

	}

	return 0;

err_out:
	return err;
}

static int mlxm_av_remove(struct fid_av *av, fi_addr_t *fi_addr, size_t count,
			  uint64_t flags)
{
	mxm_error_t	mxm_err;
	int		i;
	mlxm_fid_av_t *fid_av;
        fid_av = container_of(av, mlxm_fid_av_t, av.fid);

        if (mlxm_globals.mxm_ep) {
                for (i = 0; i < count; ++i) {
                        mxm_err = mxm_ep_disconnect(((mxm_conn_h *)fi_addr)[i]);
                        if (mxm_err)
                                return mlxm_errno(mxm_err);
                        FI_INFO(&mlxm_prov, FI_LOG_AV,
                                "disconnected from %p\n",
                                ((mxm_conn_h *)fi_addr)[i]);
                }
        }
	return 0;
}

static int mlxm_av_close(fid_t fid)
{
	mlxm_fid_av_t *fid_av;
	fid_av = container_of(fid, mlxm_fid_av_t, av.fid);
	free(fid_av);
	return 0;
}

static int mlxm_av_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	/* no need to bind an EQ since insert/remove is synchronous */
        return -FI_ENOSYS;
}


static struct fi_ops mlxm_fi_ops = {
	.size  = sizeof(struct fi_ops),
	.close = mlxm_av_close,
	.bind  = mlxm_av_bind,
};

static struct fi_ops_av mlxm_av_ops = {
	.size   = sizeof(struct fi_ops_av),
	.insert = mlxm_av_insert,
	.remove = mlxm_av_remove,
};

int mlxm_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		 struct fid_av **av, void *context)
{
	mlxm_fid_domain_t *fid_domain;
	mlxm_fid_av_t *fid_av;
	int type = FI_AV_MAP;
	size_t count = 64;
        fid_domain = container_of(domain, mlxm_fid_domain_t, domain);

	if (attr) {
                switch (attr->type) {
		case FI_AV_MAP:
                        type = attr->type;
			break;
		default:
			return -EINVAL;
		}
                count = attr->count;
        }

	fid_av = (mlxm_fid_av_t *) calloc(1, sizeof *fid_av);
	if (!fid_av)
		return -ENOMEM;

	fid_av->domain	= fid_domain;
	fid_av->type	= type;
	fid_av->addrlen = sizeof(mxm_conn_h);
	fid_av->count   = count;

	fid_av->av.fid.fclass  = FI_CLASS_AV;
	fid_av->av.fid.context = context;
	fid_av->av.fid.ops     = &mlxm_fi_ops;
	fid_av->av.ops         = &mlxm_av_ops;

	*av = &fid_av->av;
	return 0;
}
