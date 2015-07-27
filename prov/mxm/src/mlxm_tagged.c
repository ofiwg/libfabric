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
#include "mlxm_helpers.h"

#define GET_MQ_ID(_tag, _mq_id)  do{                                    \
                (_mq_id) =  (( (_tag) & MLXM_MEM_TAG_FORMAT) >> 32);    \
        }while(0)
#define GET_32BIT_TAG(_tag, _tag32) do{                                 \
                (_tag32) = (uint32_t)(_tag & (~MLXM_MEM_TAG_FORMAT));   \
        }while(0)

static inline ssize_t _mlxm_tagged_recvfrom(struct fid_ep *ep, void *buf, size_t len,
                                            void *desc, fi_addr_t src_addr, uint64_t tag,
                                            uint64_t ignore, void *context, uint64_t flags)
{
        struct mlxm_fid_ep *fid_ep;
        mxm_mq_h       mq = NULL;
        uint32_t mxm_tag;
        uint32_t mxm_tagmask;
        uint16_t mq_id = 0;
        mxm_conn_h conn = (FI_ADDR_UNSPEC == src_addr) ?
                NULL : (mxm_conn_h)src_addr;

        fid_ep = container_of(ep, struct mlxm_fid_ep, ep);
        assert(fid_ep->domain);

        GET_MQ_ID(tag, mq_id);
        GET_32BIT_TAG(tag, mxm_tag);
        GET_32BIT_TAG((~ignore), mxm_tagmask);
        if(!mlxm_find_mq(fid_ep->mxm_mqs, mq_id, &mq)) {
                mlxm_mq_add_to_storage(fid_ep->mxm_mqs, mq_id, &mq);
        }
        return _mlxm_do_recv(fid_ep, buf, len, mq,
                             mq_id, conn,
                             mxm_tag, mxm_tagmask, context);
}

static inline ssize_t _mlxm_tagged_recvfrom_v(struct fid_ep *ep, const int iov_num,
                                              const struct iovec *iov,
                                              void *desc, fi_addr_t src_addr, uint64_t tag,
                                              uint64_t ignore, void *context, uint64_t flags)
{
        struct mlxm_fid_ep *fid_ep;
        mxm_mq_h       mq = NULL;
        uint32_t mxm_tag;
        uint32_t mxm_tagmask;
        uint16_t mq_id = 0;
        mxm_conn_h conn = (FI_ADDR_UNSPEC == src_addr) ?
                NULL : (mxm_conn_h)src_addr;

        fid_ep = container_of(ep, struct mlxm_fid_ep, ep);
        assert(fid_ep->domain);

        GET_MQ_ID(tag, mq_id);
        GET_32BIT_TAG(tag, mxm_tag);
        GET_32BIT_TAG((~ignore), mxm_tagmask);
        if(!mlxm_find_mq(fid_ep->mxm_mqs, mq_id, &mq)) {
                mlxm_mq_add_to_storage(fid_ep->mxm_mqs, mq_id, &mq);
        }
        return _mlxm_do_recv_v(fid_ep, iov_num, iov, mq,
                               mq_id, conn,
                               mxm_tag, mxm_tagmask, context);
}

static ssize_t mlxm_tagged_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
                                fi_addr_t src_addr,
                                uint64_t tag, uint64_t ignore, void *context)
{
        struct mlxm_fid_ep *ep_priv;
        ep_priv = container_of(ep, struct mlxm_fid_ep, ep);
        return _mlxm_tagged_recvfrom(ep, buf, len, desc, src_addr, tag, ignore,
                                     context, ep_priv->flags);
}

static inline
ssize_t _mlxm_tagged_peek(struct fid_ep *ep, const struct fi_msg_tagged *msg,
                          uint64_t flags)
{
        struct mlxm_fid_ep *fid_ep;
        mxm_mq_h mq = NULL;
        uint32_t mxm_tag = 0;
        uint32_t mxm_tagmask = 0;
        uint16_t mq_id  = 0;
        if (flags & (FI_CLAIM | FI_DISCARD))
                return -FI_EOPNOTSUPP;

        fid_ep = container_of(ep, struct mlxm_fid_ep, ep);
        assert(fid_ep->domain);
        GET_MQ_ID(msg->tag, mq_id);
        GET_32BIT_TAG(msg->tag, mxm_tag);
        GET_32BIT_TAG((~(msg->ignore)), mxm_tagmask);
        if(!mlxm_find_mq(fid_ep->mxm_mqs, mq_id, &mq)) {
                mlxm_mq_add_to_storage(fid_ep->mxm_mqs, mq_id, &mq);
        }
        return _mlxm_do_probe(fid_ep, mq, mq_id,
                              msg->addr == FI_ADDR_UNSPEC ? NULL : (mxm_conn_h)msg->addr,
                              mxm_tag, mxm_tagmask, msg->context);
}

static ssize_t mlxm_tagged_recvmsg(struct fid_ep *ep,
                                   const struct fi_msg_tagged *msg,uint64_t flags)
{
        if (flags & FI_PEEK)
                return _mlxm_tagged_peek(ep,msg,flags);
        if (msg->iov_count == 1) {
                return _mlxm_tagged_recvfrom(ep, msg->msg_iov[0].iov_base,
                                             msg->msg_iov[0].iov_len,
                                             msg->desc ? msg->desc[0] : NULL,
                                             msg->addr, msg->tag, msg->ignore,
                                             msg->context, flags);
        }else{
                return _mlxm_tagged_recvfrom_v(ep, msg->iov_count,
                                               msg->msg_iov,
                                               msg->desc ? msg->desc[0] : NULL,
                                               msg->addr, msg->tag, msg->ignore,
                                               msg->context, flags);
        }
}

static ssize_t mlxm_tagged_recvv(struct fid_ep *ep, const struct iovec *iov,
                                 void **desc, size_t count, fi_addr_t src_addr,
                                 uint64_t tag, uint64_t ignore, void *context)
{
        if (count == 1) {
                return _mlxm_tagged_recvfrom(ep, iov[0].iov_base,
                                             iov[0].iov_len,
                                             desc, src_addr, tag, ignore,
                                             context, 0);
        }else{
                return _mlxm_tagged_recvfrom_v(ep, count,iov,
                                               desc, src_addr, tag, ignore,
                                               context, 0);
        }
}


static inline ssize_t _mlxm_tagged_sendto(struct fid_ep *ep, const void *buf, size_t len,
                                          void *desc, fi_addr_t dest_addr,
                                          uint64_t tag, void *context, uint64_t flags,
                                          uint32_t data, const int is_blocking)
{
        struct mlxm_fid_ep   *fid_ep;
        mxm_mq_h mq = NULL;
        uint32_t mxm_tag;
        uint16_t mq_id = 0;
        fid_ep = container_of(ep, struct mlxm_fid_ep, ep);
        assert(fid_ep->domain);
        assert(dest_addr != FI_ADDR_UNSPEC);

        GET_MQ_ID(tag, mq_id);
        GET_32BIT_TAG(tag, mxm_tag);
        if(!mlxm_find_mq(fid_ep->mxm_mqs, mq_id, &mq)) {
                mlxm_mq_add_to_storage(fid_ep->mxm_mqs, mq_id, &mq);
        }
        return _mlxm_do_send(fid_ep, mq, mq_id,
                             buf, len,(mxm_conn_h)dest_addr,
                             mxm_tag, context, data,
                             is_blocking);
}

static inline ssize_t _mlxm_tagged_sendto_v(struct fid_ep *ep, const int iov_num, const struct iovec* iov,
                                            void *desc, fi_addr_t dest_addr,
                                            uint64_t tag, void *context, uint64_t flags,
                                            uint32_t data, const int is_blocking)
{
        struct mlxm_fid_ep   *fid_ep;
        mxm_mq_h mq = NULL;
        uint32_t mxm_tag;
        uint16_t mq_id = 0;
        fid_ep = container_of(ep, struct mlxm_fid_ep, ep);
        assert(fid_ep->domain);
        assert(dest_addr != FI_ADDR_UNSPEC);

        GET_MQ_ID(tag, mq_id);
        GET_32BIT_TAG(tag, mxm_tag);
        if(!mlxm_find_mq(fid_ep->mxm_mqs, mq_id, &mq)) {
                mlxm_mq_add_to_storage(fid_ep->mxm_mqs, mq_id, &mq);
        }
        return _mlxm_do_send_v(fid_ep, mq, mq_id,
                               iov_num, iov,(mxm_conn_h)dest_addr,
                               mxm_tag, context, data,
                               is_blocking);
}


static ssize_t mlxm_tagged_sendv(struct fid_ep *ep,
                                 const struct iovec *iov, void **desc,
                                 size_t count, fi_addr_t src_addr,
                                 uint64_t tag, void *context)
{
        if (count == 1) {
                return _mlxm_tagged_sendto(ep, iov[0].iov_base, iov[0].iov_len,
                                           desc, src_addr,
                                           tag, context, 0, 0,0);
        } else {
                return _mlxm_tagged_sendto_v(ep, count, iov,
                                             desc, src_addr,
                                             tag, context, 0, 0,0);
        }
}



static ssize_t mlxm_tagged_injectdata(struct fid_ep *ep, const void *buf, size_t len,
                                      uint64_t data, fi_addr_t dest_addr, uint64_t tag) {
        return _mlxm_tagged_sendto(ep, buf, len, NULL, dest_addr, tag, NULL,
                                   0, data,1);
}

static ssize_t mlxm_tagged_sendto(struct fid_ep *ep, const void *buf, size_t len,
                                  void *desc, fi_addr_t dest_addr,
                                  uint64_t tag, void *context)
{
        return _mlxm_tagged_sendto(ep, buf, len, desc, dest_addr, tag, context,
                                   0, 0, 0);
}

static ssize_t mlxm_tagged_senddatato(struct fid_ep *ep, const void *buf, size_t len,
                                      void *desc, uint64_t data, fi_addr_t dest_addr,
                                      uint64_t tag, void *context)
{
        assert(data < (1ULL << 32));
        return _mlxm_tagged_sendto(ep, buf, len, desc, dest_addr, tag, context,
                                   0,  (uint32_t)data, 0);
}


static ssize_t mlxm_tagged_sendmsg(struct fid_ep *ep,
                                   const struct fi_msg_tagged *msg,uint64_t flags)
{
        if (msg->iov_count == 1) {
                return _mlxm_tagged_sendto(ep, msg->msg_iov[0].iov_base,
                                           msg->msg_iov[0].iov_len,
                                           msg->desc ? msg->desc[0] : NULL, msg->addr,
                                           msg->tag, msg->context, flags,
                                           (uint32_t)msg->data,flags & FI_INJECT);

        }else{
                return _mlxm_tagged_sendto_v(ep, msg->iov_count, msg->msg_iov,
                                             msg->desc ? msg->desc[0] : NULL, msg->addr,
                                             msg->tag, msg->context, flags,
                                             (uint32_t)msg->data,flags & FI_INJECT);
        }
}

struct fi_ops_tagged mlxm_tagged_ops = {
        .size = sizeof(struct fi_ops_tagged),
        .recv = mlxm_tagged_recv,
        .recvv = mlxm_tagged_recvv,
        .recvmsg = mlxm_tagged_recvmsg,
        .send = mlxm_tagged_sendto,
        .senddata = mlxm_tagged_senddatato,
        .sendv = mlxm_tagged_sendv,
        .sendmsg = mlxm_tagged_sendmsg,
        .injectdata = mlxm_tagged_injectdata,
};

