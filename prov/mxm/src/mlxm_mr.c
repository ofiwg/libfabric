/*
 * Copyright (c) 2015 Intel Corporation. All rights reserved.
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
#include "fi_enosys.h"

static int mlxm_mr_reg(struct fid *domain, const void *buf, size_t len,
                       uint64_t access, uint64_t offset, uint64_t requested_key,
                       uint64_t flags, struct fid_mr **mr, void *context) {
        mlxm_fid_domain_t *domain_priv;
        mlxm_fid_mr_t *mr_priv = NULL;
        uint64_t key;
        int err;
        domain_priv = container_of(domain, mlxm_fid_domain_t, domain);

        mr_priv = (mlxm_fid_mr_t *) calloc(1, sizeof(*mr_priv) + sizeof(struct iovec));
        if (!mr_priv)
                return -ENOMEM;
        mr_priv->mr.fid.fclass = FI_CLASS_MR;
        mr_priv->mr.fid.context = context;
        mr_priv->mr.mem_desc = mr_priv;
        key = (uint64_t)(uintptr_t)mr_priv;
        mr_priv->mr.key = key;
        mr_priv->domain = domain_priv;
        mr_priv->iov_count = 1;
        mr_priv->iov[0].iov_base = (void *)buf;
        mr_priv->iov[0].iov_len = len;
        err = mxm_mem_get_key(mlxm_globals.mxm_context, (void*)buf,
                              &mr_priv->mxm_key);
        if (MXM_OK != err) {
                FI_WARN(&mlxm_prov,FI_LOG_MR,
                        "Failed to get memory key: %s", mxm_error_string(err));
                goto error_out;
        }
        *mr = &mr_priv->mr;
        return 0;
error_out:
        if (mr_priv)
                free(mr_priv);
        return FI_ENOKEY;
}

struct fi_ops_mr mlxm_mr_ops = {
        .size = sizeof(struct fi_ops_mr),
        .reg = mlxm_mr_reg,
        .regv = fi_no_mr_regv,
        .regattr = fi_no_mr_regattr,
};
