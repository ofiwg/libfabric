/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
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

#include <prov/verbs/src/fi_verbs.h>

static int fi_ibv_pep_getname(fid_t pep, void *addr, size_t * addrlen)
{
    struct fi_ibv_pep *_pep;
    struct sockaddr *sa;

    _pep = container_of(pep, struct fi_ibv_pep, pep_fid);
    sa = rdma_get_local_addr(_pep->id);
    return fi_ibv_copy_addr(addr, addrlen, sa);
}

static int fi_ibv_pep_listen(struct fid_pep *pep_fid)
{
    struct fi_ibv_pep *pep;
    struct sockaddr *addr;
    pep = container_of(pep_fid, struct fi_ibv_pep, pep_fid);
    addr = rdma_get_local_addr(pep->id);

    if (addr) {
        FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "Listening on %s:%d\n",
                inet_ntoa(((struct sockaddr_in *)addr)->sin_addr),
                ntohs(((struct sockaddr_in *)addr)->sin_port));
    }

    return rdma_listen(pep->id, 0) ? -errno : 0;
}

static int
fi_ibv_msg_ep_reject(struct fid_pep *pep, fid_t handle,
                     const void *param, size_t paramlen)
{
    struct fi_ibv_connreq *connreq;
    int ret;

    connreq = container_of(handle, struct fi_ibv_connreq, handle);
    ret = rdma_reject(connreq->id, param, (uint8_t) paramlen) ? -errno : 0;
    free(connreq);
    return ret;
}

static int fi_ibv_pep_setname(fid_t pep_fid, void *addr, size_t addrlen)
{
    struct fi_ibv_pep *pep;
    int ret;

    pep = container_of(pep_fid, struct fi_ibv_pep, pep_fid);
    if (pep->src_addrlen && (addrlen != pep->src_addrlen)) {
        FI_INFO(&fi_ibv_prov, FI_LOG_FABRIC,
                "addrlen expected: %d, got: %d.\n", pep->src_addrlen, addrlen);
        return -FI_EINVAL;
    }

    /* Re-create id if already bound */
    if (pep->bound) {
        ret = rdma_destroy_id(pep->id);
        if (ret) {
            FI_INFO(&fi_ibv_prov, FI_LOG_FABRIC,
                    "Unable to destroy previous rdma_cm_id\n");
            return -errno;
        }
        ret = rdma_create_id(NULL, &pep->id, NULL, RDMA_PS_TCP);
        if (ret) {
            FI_INFO(&fi_ibv_prov, FI_LOG_FABRIC,
                    "Unable to create rdma_cm_id\n");
            return -errno;
        }
    }

    ret = rdma_bind_addr(pep->id, (struct sockaddr *)addr);
    if (ret) {
        FI_INFO(&fi_ibv_prov, FI_LOG_FABRIC,
                "Unable to bind addres to rdma_cm_id\n");
        return -errno;
    }

    return 0;
}

static struct fi_ops_cm fi_ibv_pep_cm_ops = {
    .size = sizeof(struct fi_ops_cm),
    .setname = fi_ibv_pep_setname,
    .getname = fi_ibv_pep_getname,
    .getpeer = fi_no_getpeer,
    .connect = fi_no_connect,
    .listen = fi_ibv_pep_listen,
    .accept = fi_no_accept,
    .reject = fi_ibv_msg_ep_reject,
    .shutdown = fi_no_shutdown,
};

static int fi_ibv_pep_bind(fid_t fid, struct fid *bfid, uint64_t flags)
{
    struct fi_ibv_pep *pep;
    int ret;

    pep = container_of(fid, struct fi_ibv_pep, pep_fid.fid);
    if (bfid->fclass != FI_CLASS_EQ)
        return -FI_EINVAL;

    pep->eq = container_of(bfid, struct fi_ibv_eq, eq_fid.fid);
    ret = rdma_migrate_id(pep->id, pep->eq->channel);
    if (ret)
        return -errno;

    return 0;
}

static int fi_ibv_pep_close(fid_t fid)
{
    struct fi_ibv_pep *pep;

    pep = container_of(fid, struct fi_ibv_pep, pep_fid.fid);
    if (pep->id)
        rdma_destroy_ep(pep->id);

    free(pep);
    return 0;
}

static struct fi_ops fi_ibv_pep_ops = {
    .size = sizeof(struct fi_ops),
    .close = fi_ibv_pep_close,
    .bind = fi_ibv_pep_bind,
    .control = fi_no_control,
    .ops_open = fi_no_ops_open,
};

int fi_ibv_passive_ep(struct fid_fabric *fabric, struct fi_info *info,
                      struct fid_pep **pep, void *context)
{
    struct fi_ibv_pep *_pep;
    int ret;

    _pep = calloc(1, sizeof *_pep);
    if (!_pep)
        return -FI_ENOMEM;

    ret = rdma_create_id(NULL, &_pep->id, NULL, RDMA_PS_TCP);
    if (ret) {
        FI_INFO(&fi_ibv_prov, FI_LOG_DOMAIN, "Unable to create rdma_cm_id\n");
        goto err1;
    }

    if (info->src_addr) {
        ret = rdma_bind_addr(_pep->id, (struct sockaddr *)info->src_addr);
        if (ret) {
            FI_INFO(&fi_ibv_prov, FI_LOG_DOMAIN,
                    "Unable to bind addres to rdma_cm_id\n");
            goto err2;
        }
        _pep->bound = 1;
    }

    _pep->id->context = &_pep->pep_fid.fid;

    _pep->pep_fid.fid.fclass = FI_CLASS_PEP;
    _pep->pep_fid.fid.context = context;
    _pep->pep_fid.fid.ops = &fi_ibv_pep_ops;
    _pep->pep_fid.cm = &fi_ibv_pep_cm_ops;

    _pep->src_addrlen = info->src_addrlen;

    *pep = &_pep->pep_fid;
    return 0;

 err2:
    rdma_destroy_id(_pep->id);
 err1:
    free(_pep);
    return ret;
}
