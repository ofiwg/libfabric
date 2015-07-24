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

static ssize_t mlxm_cq_readfrom(struct fid_cq *cq, void *buf, size_t len,
                                fi_addr_t *src_addr)
{
        mlxm_fid_cq_t     *fid_cq;
        mlxm_req_t        *mlxm_req;
        mxm_send_req_t    *mxm_sreq;
        mxm_recv_req_t    *mxm_rreq;
        struct fi_context *ctx;
        struct fi_cq_tagged_entry *cqe =
                (struct fi_cq_tagged_entry *) buf;

        fid_cq = container_of(cq, mlxm_fid_cq_t, cq);
        if (fid_cq->err_q.head)
                return -FI_EAVAIL;
        mxm_progress(fid_cq->mxm_context);

        if (!fid_cq->ok_q.head) {
                return 0;
        }

        MLXM_CQ_DEQUEUE(fid_cq->ok_q, ctx);
        mlxm_req = ctx->internal[1];
        ctx->internal[1] = NULL;
        cqe->flags      = 0;
        cqe->op_context = ctx;

        if ((uint64_t)(ctx->internal[3]) == FI_SEND) {
                mxm_sreq = &mlxm_req->mxm_req.sreq;
                assert (mxm_sreq->base.error == MXM_OK);
                cqe->flags      |= FI_SEND;
                cqe->len        = mxm_sreq->base.data.buffer.length;
                cqe->data       = mxm_sreq->op.send.imm_data;
                cqe->tag        = mxm_sreq->op.send.tag;
                if (src_addr) {
                        memcpy(src_addr, &(mxm_sreq->base.conn), sizeof(mxm_conn_h));
                }
        } else {
                mxm_rreq = &mlxm_req->mxm_req.rreq;
                assert(mxm_rreq->base.error != MXM_OK);
                cqe->flags      |= FI_RECV;
                cqe->len        = mxm_rreq->completion.actual_len;
                cqe->data       = mxm_rreq->completion.sender_imm;
                cqe->tag        = mxm_rreq->completion.sender_tag;
                if (src_addr) {
                        memcpy(src_addr, &(mxm_rreq->completion.source), sizeof(mxm_conn_h));
                }
        }
        cqe->tag |= (((uint64_t)mlxm_req->mq_id) << 32);
        MPOOL_RETURN(mlxm_globals.req_pool, struct mlxm_req, mlxm_req);
        return 1;
}

static ssize_t mlxm_cq_read(struct fid_cq *cq, void *buf, size_t len)
{
        return mlxm_cq_readfrom(cq, buf, len, NULL);
}

static int mlxm_cq_close(fid_t fid)
{
        mlxm_fid_cq_t   *fid_cq;
        fid_cq = container_of(fid, mlxm_fid_cq_t, cq.fid);
        free(fid_cq);
        return 0;
}

static ssize_t  mlxm_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *cqe,
                                uint64_t flags)
{
        struct mlxm_fid_cq              *fid_cq;
        mlxm_req_t                      *mlxm_req;
        mxm_send_req_t                  *mxm_sreq = NULL;
        mxm_recv_req_t                  *mxm_rreq = NULL;
        struct fi_context *ctx;
        fid_cq = container_of(cq, mlxm_fid_cq_t, cq);
        if (!fid_cq->err_q.head) {
                return 0;
        }

        MLXM_CQ_DEQUEUE(fid_cq->err_q, ctx);
        mlxm_req = ctx->internal[1];
        ctx->internal[1] = NULL;
        cqe->op_context = ctx;
        cqe->flags      = FI_TAGGED;
        cqe->buf = NULL;
        if ((uint64_t)(ctx->internal[3]) == FI_SEND) {
                mxm_sreq = &mlxm_req->mxm_req.sreq;
                assert(mxm_sreq->base.error != MXM_OK);
                cqe->prov_errno = mxm_sreq->base.error;
                cqe->len        = mxm_sreq->base.data.buffer.length;
                cqe->olen       = 0;
                cqe->data       = mxm_sreq->op.send.imm_data;
                cqe->tag        = mxm_sreq->op.send.tag;
                cqe->flags |= FI_SEND;
        } else {
                mxm_rreq = &mlxm_req->mxm_req.rreq;
                assert(mxm_rreq->base.error != MXM_OK);
                cqe->prov_errno = mxm_sreq->base.error;
                cqe->len        = mxm_rreq->completion.actual_len;
                cqe->olen       = mxm_rreq->completion.sender_len -
                        cqe->len;
                cqe->data       = mxm_rreq->completion.sender_imm;
                cqe->tag        = mxm_rreq->completion.sender_tag;
                cqe->flags |= FI_RECV;
        }
        cqe->err = -mlxm_errno(cqe->prov_errno);
        cqe->tag |= (((uint64_t)mlxm_req->mq_id) << 32);
        MPOOL_RETURN(mlxm_globals.req_pool, struct mlxm_req, mlxm_req);
        return 1;
}



static const char *mlxm_cq_strerror(struct fid_cq *cq, int prov_errno,
                                    const void *prov_data, char *buf, size_t len)
{
        return mxm_error_string(prov_errno);
}


static int mlxm_cq_control(fid_t fid, int command, void *arg)
{
        return -ENOSYS;
}

static struct fi_ops mlxm_fi_ops = {
        .size    = sizeof(struct fi_ops),
        .close   = mlxm_cq_close,
        .control = mlxm_cq_control,
};

static struct fi_ops_cq mlxm_cq_ops = {
        .size     = sizeof(struct fi_ops_cq),
        .readfrom = mlxm_cq_readfrom,
        .read     = mlxm_cq_read,
        .readerr  = mlxm_cq_readerr,
        .strerror = mlxm_cq_strerror,
};

int mlxm_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
                 struct fid_cq **cq, void *context)
{
        mlxm_fid_cq_t   *fid_cq;
        fid_cq = (mlxm_fid_cq_t *) calloc(1, sizeof *fid_cq);
        if (!fid_cq)
                return -ENOMEM;
        fid_cq->cq.fid.fclass   = FI_CLASS_CQ;
        fid_cq->cq.fid.context  = context;
        fid_cq->cq.fid.ops              = &mlxm_fi_ops;
        fid_cq->cq.ops          = &mlxm_cq_ops;
        fid_cq->mxm_context     = mlxm_globals.mxm_context;
        fid_cq->ok_q.head = NULL;
        fid_cq->ok_q.tail = NULL;
        fid_cq->err_q.head = NULL;
        fid_cq->err_q.tail = NULL;

        *cq = &fid_cq->cq;
        return 0;
}
