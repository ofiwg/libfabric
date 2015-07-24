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

static int mlxm_domain_close(fid_t fid)
{
        mlxm_fid_domain_t *fid_domain;
        fid_domain = container_of(fid, mlxm_fid_domain_t, domain.fid);
        free(fid_domain);
        return 0;
}

static struct fi_ops mlxm_fi_ops = {
        .size = sizeof(struct fi_ops),
        .close = mlxm_domain_close,
};

static struct fi_ops_domain mlxm_domain_ops = {
        .size = sizeof(struct fi_ops_domain),
        .av_open = mlxm_av_open,
        .cq_open = mlxm_cq_open,
        .endpoint = mlxm_ep_open,
};

int mlxm_domain_open(struct fid_fabric *fabric, struct fi_info *info,
                     struct fid_domain **domain, void *context)
{
        mlxm_fid_domain_t       *fid_domain;
        FI_INFO(&mlxm_prov, FI_LOG_DOMAIN, "\n");

        if (!info->domain_attr->name ||
            strncmp(info->domain_attr->name, "mxm", 3))
                return -FI_EINVAL;
        fid_domain = (mlxm_fid_domain_t*) calloc(1, sizeof(*fid_domain));
        if (!fid_domain)
                return -ENOMEM;
        fid_domain->domain.fid.fclass  = FI_CLASS_DOMAIN;
        fid_domain->domain.fid.context = context;
        fid_domain->domain.fid.ops     = &mlxm_fi_ops;
        fid_domain->domain.ops         = &mlxm_domain_ops;

        *domain = &fid_domain->domain;
        return 0;
}
