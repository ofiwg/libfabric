#ifndef _MLXM_HELPERS_H
#define _MLXM_HELPERS_H
#include "mlxm.h"

static void mlxm_completion_cb(void *context)
{
    struct fi_context *ctx = (struct fi_context *)context;
    mlxm_fid_cq_t		*fid_cq =
        (mlxm_fid_cq_t *) ctx->internal[2];

    mlxm_req_t *mlxm_req = (mlxm_req_t*)ctx->internal[1];
    int err = ((mxm_req_base_t*)&mlxm_req->mxm_req)->error;
    if (err == MXM_OK) {
            MLXM_CQ_ENQUEUE(fid_cq->ok_q, ctx);
    } else if (err == MXM_ERR_CANCELED) {
            MPOOL_RETURN(mlxm_globals.req_pool, struct mlxm_req, mlxm_req);
    } else {
            MLXM_CQ_ENQUEUE(fid_cq->err_q, ctx);
    }
}

static void mlxm_completion_cb_v(void *context)
{
    struct fi_context *ctx = (struct fi_context *)context;
    mlxm_fid_cq_t		*fid_cq =
        (mlxm_fid_cq_t *) ctx->internal[2];

    mlxm_req_t *mlxm_req = (mlxm_req_t*)ctx->internal[1];

    free(((mxm_req_base_t*)&mlxm_req->mxm_req)->data.iov.vector);
    int err = ((mxm_req_base_t*)&mlxm_req->mxm_req)->error;
    if (err == MXM_OK) {
            MLXM_CQ_ENQUEUE(fid_cq->ok_q, ctx);
    } else if (err == MXM_ERR_CANCELED) {
            MPOOL_RETURN(mlxm_globals.req_pool, struct mlxm_req, mlxm_req);
    } else {
            MLXM_CQ_ENQUEUE(fid_cq->err_q, ctx);
    }
}

static inline int _mlxm_do_send(mlxm_fid_ep_t *fid_ep, mxm_mq_h mq,
                                int mq_id,
                                const void *buf, size_t len,
                                mxm_conn_h conn, uint32_t mxm_tag,
                                void *context, uint32_t data,
                                const int is_blocking)
{

    mlxm_fid_cq_t	*fid_cq;
    mlxm_req_t		*mlxm_req = NULL;
    mxm_send_req_t		*mxm_req = NULL;
    mxm_error_t		mxm_err;
    int			err;
    mxm_send_req_t stack_req;


    assert(fid_ep->domain);
    fid_cq = fid_ep->cq;


    if (!is_blocking) {
        // MLXM_REQ_GET(mlxm_req);
        MPOOL_ALLOC(mlxm_globals.req_pool, struct mlxm_req, mlxm_req);
        if (!mlxm_req)
            return -ENOMEM;
        mlxm_req->mq_id         = mq_id;
        mlxm_req->pool = mlxm_globals.req_pool;
        mxm_req = &mlxm_req->mxm_req.sreq;
    } else {
        mxm_req = &stack_req;
    }


    mxm_req->base.state			= MXM_REQ_NEW;
    mxm_req->base.mq			= mq;
    mxm_req->base.conn			= conn;
    mxm_req->flags		        	=
        is_blocking ? MXM_REQ_SEND_FLAG_BLOCKING : 0;
    mxm_req->base.data_type			= MXM_REQ_DATA_BUFFER;
    mxm_req->base.data.buffer.ptr		= (void *)buf;
    mxm_req->base.data.buffer.length	= len;
    mxm_req->base.data.buffer.memh		= MXM_INVALID_MEM_HANDLE;
    mxm_req->opcode				= MXM_REQ_OP_SEND;
    mxm_req->op.send.tag			= mxm_tag;
    mxm_req->op.send.imm_data		= (mxm_imm_t) data;

    if (is_blocking) {
        mxm_req->base.completed_cb	= NULL;
        mxm_req->base.context		= NULL;
    } else {
        mxm_req->base.completed_cb	= mlxm_completion_cb;
        mxm_req->base.context		= (void *)context;
        ((struct fi_context *)context)->internal[0] = context;
        ((struct fi_context *)context)->internal[1] = mlxm_req;
        ((struct fi_context *)context)->internal[2] = fid_cq;
        ((struct fi_context *)context)->internal[3] = (void*)FI_SEND;
    }

    FI_DBG(&mlxm_prov,FI_LOG_CORE,
           "sendto mq 0x%x, conn %p, buf %p,"
           "len %zd, tag 0x%x, imm 0x%x\n",
           mq_id, mxm_req->base.conn, buf, len,
           mxm_req->op.send.tag, mxm_req->op.send.imm_data);


    mxm_err = mxm_req_send(mxm_req);


    if (mxm_err != MXM_OK) {
        err = mlxm_errno(mxm_err);
        goto err_out_destroy_req;
    }

    if (is_blocking)  {
        mxm_req_wait(&mxm_req->base);
        return 0;
    }

    return 0;
err_out_destroy_req:
    if (!is_blocking)
        MPOOL_RETURN(mlxm_globals.req_pool, struct mlxm_req, mlxm_req);
    return err;
}

static int _mlxm_do_recv(mlxm_fid_ep_t *fid_ep, void *buf, size_t len,
                         mxm_mq_h mq, int mq_id, mxm_conn_h conn,
                         uint32_t mxm_tag,
                         uint32_t mxm_tagmask, void *context)
{
    mlxm_fid_cq_t	*fid_cq;
    mlxm_req_t	*mlxm_req = NULL;
    mxm_recv_req_t	*mxm_req = NULL;
    mxm_error_t	mxm_err;
    int		err;


    fid_cq = fid_ep->cq;
    MPOOL_ALLOC(mlxm_globals.req_pool, struct mlxm_req, mlxm_req);

    if (!mlxm_req)
        return -ENOMEM;

    mlxm_req->mq_id         = mq_id;
    mxm_req = &mlxm_req->mxm_req.rreq;
    mxm_req->base.state			= MXM_REQ_NEW;
    mxm_req->base.mq			= mq;
    mxm_req->base.conn			= conn;

    mxm_req->base.data_type			= MXM_REQ_DATA_BUFFER;
    mxm_req->base.data.buffer.ptr		= buf;
    mxm_req->base.data.buffer.length	= len;
    mxm_req->base.data.buffer.memh		= MXM_INVALID_MEM_HANDLE;

    mxm_req->base.completed_cb		= mlxm_completion_cb;
    mxm_req->base.context		= (void *)context;
    ((struct fi_context *)context)->internal[0] = context;
    ((struct fi_context *)context)->internal[1] = mlxm_req;
    ((struct fi_context *)context)->internal[2] = fid_cq;
    ((struct fi_context *)context)->internal[3] = (void*)FI_RECV;

    mxm_req->tag				= mxm_tag;
    mxm_req->tag_mask			= mxm_tagmask;

    FI_DBG(&mlxm_prov,FI_LOG_CORE,
           "recv mq 0x%x, conn %p, buf %p,"
           "len %zd, tag 0x%x, tagmask 0x%x\n",
           mq_id, mxm_req->base.conn, buf, len,
           mxm_req->tag, mxm_req->tag_mask);

    mxm_err = mxm_req_recv(mxm_req);

    if (mxm_err != MXM_OK) {
        err = mlxm_errno(mxm_err);
        goto err_out_destroy_req;
    }

    // mxm_progress(mlxm_globals.mxm_context);
    assert(mxm_req->base.state != MXM_REQ_NEW);
    return 0;

err_out_destroy_req:
    if (mlxm_req)
        MPOOL_RETURN(mlxm_globals.req_pool, struct mlxm_req, mlxm_req);
    return err;
}


static inline int _mlxm_do_send_v(mlxm_fid_ep_t *fid_ep, mxm_mq_h mq,
                                  int mq_id,
                                  const int iov_num,
                                  const struct iovec * iov,
                                  mxm_conn_h conn, uint32_t mxm_tag,
                                  void *context, uint32_t data,
                                  const int is_blocking)
{

    mlxm_fid_cq_t	*fid_cq;
    mlxm_req_t		*mlxm_req = NULL;
    mxm_send_req_t		*mxm_req = NULL;
    mxm_error_t		mxm_err;
    int			err;
    mxm_send_req_t stack_req;
    int i;

    assert(fid_ep->domain);
    fid_cq = fid_ep->cq;


    if (!is_blocking) {
        // MLXM_REQ_GET(mlxm_req);
        MPOOL_ALLOC(mlxm_globals.req_pool, struct mlxm_req, mlxm_req);
        if (!mlxm_req)
            return -ENOMEM;
        mlxm_req->mq_id         = mq_id;
        mlxm_req->pool = mlxm_globals.req_pool;
        mxm_req = &mlxm_req->mxm_req.sreq;
    } else {
        mxm_req = &stack_req;
    }


    mxm_req->base.state			= MXM_REQ_NEW;
    mxm_req->base.mq			= mq;
    mxm_req->base.conn			= conn;
    mxm_req->flags		        	=
        is_blocking ? MXM_REQ_SEND_FLAG_BLOCKING : 0;
    mxm_req->base.data_type			= MXM_REQ_DATA_IOV;
    mxm_req->base.data.iov.count = iov_num;
    mxm_req->base.data.iov.vector =
        (mxm_req_buffer_t*)malloc(iov_num*sizeof(mxm_req_buffer_t));
    for (i=0; i<iov_num; i++) {
        mxm_req->base.data.iov.vector[i].ptr =
            iov[i].iov_base;
        mxm_req->base.data.iov.vector[i].length =
            iov[i].iov_len;
        mxm_req->base.data.iov.vector[i].memh =
            MXM_INVALID_MEM_HANDLE;
    }

    mxm_req->opcode				= MXM_REQ_OP_SEND;
    mxm_req->op.send.tag			= mxm_tag;
    mxm_req->op.send.imm_data		= (mxm_imm_t) data;

    if (is_blocking) {
        mxm_req->base.completed_cb	= NULL;
        mxm_req->base.context		= NULL;
    } else {
        mxm_req->base.completed_cb	= mlxm_completion_cb_v;
        mxm_req->base.context		= (void *)context;
        ((struct fi_context *)context)->internal[0] = context;
        ((struct fi_context *)context)->internal[1] = mlxm_req;
        ((struct fi_context *)context)->internal[2] = fid_cq;
        ((struct fi_context *)context)->internal[3] = (void*)FI_SEND;
    }

    FI_DBG(&mlxm_prov,FI_LOG_CORE,
           "sendto mq 0x%x, conn %p, iov_count %d,"
           "tag 0x%x, imm 0x%x\n",
           mq_id, mxm_req->base.conn, iov_num,
           mxm_req->op.send.tag, mxm_req->op.send.imm_data);


    mxm_err = mxm_req_send(mxm_req);


    if (mxm_err != MXM_OK) {
        err = mlxm_errno(mxm_err);
        goto err_out_destroy_req;
    }

    if (is_blocking)  {
        mxm_req_wait(&mxm_req->base);
        return 0;
    }

    return 0;
err_out_destroy_req:
    if (!is_blocking)
        MPOOL_RETURN(mlxm_globals.req_pool, struct mlxm_req, mlxm_req);
    return err;
}

static int _mlxm_do_recv_v(mlxm_fid_ep_t *fid_ep, const int iov_num,
                           const struct iovec *iov,
                           mxm_mq_h mq, int mq_id, mxm_conn_h conn,
                           uint32_t mxm_tag,
                           uint32_t mxm_tagmask, void *context)
{
    mlxm_fid_cq_t	*fid_cq;
    mlxm_req_t	*mlxm_req = NULL;
    mxm_recv_req_t	*mxm_req = NULL;
    mxm_error_t	mxm_err;
    int		err, i;


    fid_cq = fid_ep->cq;
    MPOOL_ALLOC(mlxm_globals.req_pool, struct mlxm_req, mlxm_req);

    if (!mlxm_req)
        return -ENOMEM;

    mlxm_req->mq_id         = mq_id;
    mxm_req = &mlxm_req->mxm_req.rreq;
    mxm_req->base.state			= MXM_REQ_NEW;
    mxm_req->base.mq			= mq;
    mxm_req->base.conn			= conn;

    mxm_req->base.data_type			= MXM_REQ_DATA_IOV;
    mxm_req->base.data.iov.count = iov_num;
    mxm_req->base.data.iov.vector =
        (mxm_req_buffer_t*)malloc(iov_num*sizeof(mxm_req_buffer_t));
    for (i=0; i<iov_num; i++) {
        mxm_req->base.data.iov.vector[i].ptr =
            iov[i].iov_base;
        mxm_req->base.data.iov.vector[i].length =
            iov[i].iov_len;
        mxm_req->base.data.iov.vector[i].memh =
            MXM_INVALID_MEM_HANDLE;
    }

    mxm_req->base.completed_cb		= mlxm_completion_cb_v;
    mxm_req->base.context		= (void *)context;
    ((struct fi_context *)context)->internal[0] = context;
    ((struct fi_context *)context)->internal[1] = mlxm_req;
    ((struct fi_context *)context)->internal[2] = fid_cq;
    ((struct fi_context *)context)->internal[3] = (void*)FI_RECV;

    mxm_req->tag				= mxm_tag;
    mxm_req->tag_mask			= mxm_tagmask;

    FI_DBG(&mlxm_prov,FI_LOG_CORE,
           "recv mq 0x%x, conn %p, iov_num %d,"
           "tag 0x%x, tagmask 0x%x\n",
           mq_id, mxm_req->base.conn, iov_num,
           mxm_req->tag, mxm_req->tag_mask);

    mxm_err = mxm_req_recv(mxm_req);

    if (mxm_err != MXM_OK) {
        err = mlxm_errno(mxm_err);
        goto err_out_destroy_req;
    }

    // mxm_progress(mlxm_globals.mxm_context);
    assert(mxm_req->base.state != MXM_REQ_NEW);
    return 0;

err_out_destroy_req:
    if (mlxm_req)
        MPOOL_RETURN(mlxm_globals.req_pool, struct mlxm_req, mlxm_req);
    return err;
}

static ssize_t
_mlxm_do_probe(mlxm_fid_ep_t *fid_ep, mxm_mq_h mq,
               int mq_id, mxm_conn_h conn,
               uint32_t mxm_tag, uint32_t mxm_tagmask,
               void *context)
{
    mxm_error_t	mxm_err;
    mlxm_req_t	*mlxm_req = NULL;
    mxm_recv_req_t	*mxm_req = NULL;

    MPOOL_ALLOC(mlxm_globals.req_pool, struct mlxm_req, mlxm_req);

    if (!mlxm_req)
        return -ENOMEM;

    mlxm_req->mq_id         = mq_id;
    mxm_req = &mlxm_req->mxm_req.rreq;
    mxm_req->base.state			= MXM_REQ_NEW;
    mxm_req->base.mq			= mq;
    mxm_req->base.conn			= conn;

    mxm_req->base.context		= (void *)context;

    ((struct fi_context *)context)->internal[0] = context;
    ((struct fi_context *)context)->internal[1] = mlxm_req;
    ((struct fi_context *)context)->internal[2] = fid_ep->cq;
    ((struct fi_context *)context)->internal[3] = (void*)FI_RECV;

    mxm_req->tag				= mxm_tag;
    mxm_req->tag_mask			= mxm_tagmask;


    mxm_err = mxm_req_probe(mxm_req);
    switch (mxm_err) {
    case MXM_OK:
        mlxm_completion_cb((void*)context);
        return 0;
    case MXM_ERR_NO_MESSAGE:
        return -FI_ENOMSG;
    default:
        return mlxm_errno(mxm_err);
    }

}

#endif


