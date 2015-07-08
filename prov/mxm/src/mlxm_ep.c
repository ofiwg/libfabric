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


static ssize_t mlxm_ep_cancel(fid_t fid, void *ctx)
{
	mlxm_fid_ep_t     *fid_ep;
	mlxm_req_t        *req;
        struct fi_context *context = (struct fi_context*)ctx;
	int err;

        fid_ep = container_of(fid, mlxm_fid_ep_t, ep.fid);
	if (!fid_ep->domain)
		return -EBADF;

	if (!context)
		return -EINVAL;

	if (context->internal[0] == NULL)
                return -FI_EINVAL;

        req =(mlxm_req_t *)context->internal[1];
        if (FI_RECV == (uint64_t)(context->internal[3])) {
                err = mxm_req_cancel_recv(&req->mxm_req.rreq);
        } else {
                err = mxm_req_cancel_send(&req->mxm_req.sreq);
        }
        if (err == MXM_OK) {
                mxm_req_wait(&req->mxm_req.rreq.base);
        }
	return mlxm_errno(err);
}

static int mlxm_ep_getopt(fid_t fid, int level, int optname,
			    void *optval, size_t *optlen)
{
	return -ENOSYS;
}

static int mlxm_ep_setopt(fid_t fid, int level, int optname,
			    const void *optval, size_t optlen)
{
        return 0;
}

static int mlxm_ep_close(fid_t fid)
{
	mlxm_fid_ep_t   	*fid_ep;
        fid_ep = container_of(fid, mlxm_fid_ep_t, ep.fid);
        mlxm_mq_storage_fini(fid_ep);
        free(fid_ep);
        return 0;
}

static int mlxm_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
    mlxm_fid_ep_t	*fid_ep;
    int			err;


    fid_ep = container_of(fid, mlxm_fid_ep_t, ep.fid);

    switch (bfid->fclass) {
    case FI_CLASS_CQ:
        /* TODO: check ress flags for send/recv ECs */
        fid_ep->cq = container_of(bfid, mlxm_fid_cq_t, cq.fid);
        break;
    case FI_CLASS_AV:
        fid_ep->av = container_of(bfid, mlxm_fid_av_t, av.fid);
        fid_ep->domain = fid_ep->av->domain;
        fid_ep->av->ep = fid_ep;
        break;
    default:
        return -ENOSYS;
    }

    return 0;

    return err;
}


static int mlxm_ep_control(fid_t fid, int command, void *arg)
{
        switch (command) {
	case FI_ENABLE:
		return 0;

	default:
		return -FI_ENOSYS;

        }
        return 0;
}

static int mlxm_ep_enable(struct fid_ep *ep) {
	return 0;
}

static struct fi_ops_ep mlxm_ep_ops = {
	.size   = sizeof(struct fi_ops_ep),
	.cancel = mlxm_ep_cancel,
	.getopt = mlxm_ep_getopt,
	.setopt = mlxm_ep_setopt,
};

static struct fi_ops mlxm_fi_ops = {
	.size    = sizeof(struct fi_ops),
	.close   = mlxm_ep_close,
	.bind    = mlxm_ep_bind,
        .control = mlxm_ep_control,
};


static inline
int mlxm_check_mem_tag_format(uint64_t format) {
    if (format == MLXM_MEM_TAG_FORMAT)
        return 0;
    else
        return 1;
}


int mlxm_ep_open(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **fid, void *context)
{
        mlxm_fid_ep_t	*fid_ep;
        mlxm_fid_domain_t *mlxm_domain;

        if (mlxm_check_mem_tag_format(mlxm_mem_tag_format)) {
                FI_WARN(&mlxm_prov, FI_LOG_CORE, "unsupported mem_tag_format: 0x%llx, supported: 0x%llx\n",
                        (long long unsigned)mlxm_mem_tag_format,
                        MLXM_MEM_TAG_FORMAT);
                return -EINVAL;
        }

        fid_ep = (mlxm_fid_ep_t *) calloc(1, sizeof *fid_ep);
	if (!fid_ep)
		return -ENOMEM;

        mlxm_domain = container_of(domain, mlxm_fid_domain_t, domain);
        fid_ep->ep.fid.fclass	= FI_CLASS_EP;
	fid_ep->ep.fid.context	= context;
	fid_ep->ep.fid.ops	= &mlxm_fi_ops;
	fid_ep->ep.ops		= &mlxm_ep_ops;
	fid_ep->ep.cm		= &mlxm_cm_ops;
        fid_ep->ep.tagged	= &mlxm_tagged_ops;
        fid_ep->domain		= mlxm_domain;

        if (info) {
                if (info->tx_attr)
                        fid_ep->flags = info->tx_attr->op_flags;
                if (info->rx_attr)
                        fid_ep->flags |= info->rx_attr->op_flags;
                
		if (info->dest_addr) {
			/* Connected mode: store the address until bind() */
			/* The user passes
			 * hints.dest_addr = <address given by mxm_ep_address()>
                         * TODO: clarify this flow */
                }
	}

        *fid = &fid_ep->ep;

        mpool_init(&mlxm_globals.req_pool, sizeof(struct mlxm_req), 32*4);
        fid_ep->mxm_mqs = &mlxm_globals.mq_storage;
        return 0;
}
