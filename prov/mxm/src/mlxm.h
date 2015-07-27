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

#ifndef _FI_MLXM_H
#define _FI_MLXM_H

#ifdef __cplusplus
extern "C" {
#endif

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <rdma/fabric.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_log.h>
#include <fi_list.h>
#include <mxm/api/mxm_api.h>
#include "mpool.h"
#include "uthash.h"
extern int mlxm_errno_table[MXM_ERR_LAST];
static inline
int mlxm_errno(int err)
{
        if (err>=0 && err < MXM_ERR_LAST)
                return mlxm_errno_table[err];
        else
                return -FI_EOTHER;
}



#define MLXM_SUPPORTED_FLAGS ( FI_SEND | FI_RECV )
#define MLXM_DEFAULT_FLAGS   (0)
extern struct fi_provider mlxm_prov;

struct mlxm_fid_domain {
        struct fid_domain       domain;
        size_t                  mxm_addrlen;
};

struct mlxm_fid_ep {
        struct fid_ep           ep;
        struct mlxm_fid_domain  *domain;
        struct mlxm_fid_cq      *cq;
        struct mlxm_fid_av      *av;
        uint64_t                flags;
        struct mlxm_mq_storage* mxm_mqs;
};

struct mlxm_fid_av {
        struct fid_av           av;
        struct mlxm_fid_domain  *domain;
        struct mlxm_fid_ep      *ep;
        int                     type;
        size_t                  count;
        size_t                  addrlen;
};

struct mlxm_fid_fabric {
        struct fid_fabric       fabric;
};

struct mlxm_fid_mr {
        struct fid_mr           mr;
        struct mlxm_fid_domain  *domain;
        mxm_mem_key_t mxm_key;
        size_t iov_count;
        struct iovec            iov[0]; /* must be the last field */
};

struct mlxm_cq_entry_queue {
        size_t                  size;
        size_t                  n;
        struct mlxm_cq_entry    *head;
        struct mlxm_cq_entry    *tail;
};

struct mlxm_fid_cq {
        struct fid_cq cq;
        mxm_h mxm_context;
        struct {
                void *head;
                void *tail;
        } ok_q;
        struct {
                void *head;
                void *tail;
        } err_q;
};

struct mlxm_cq_entry {
        void                 *ptr;
        mxm_conn_h            src_addr;
        struct mlxm_cq_entry *next;
};

struct mlxm_req {
        union {
                mxm_send_req_t  sreq;
                mxm_recv_req_t  rreq;
        } mxm_req;
        struct mpool* pool;
        uint16_t   mq_id;
};

struct mlxm_mq_entry {
        uint16_t mq_key;
        mxm_mq_h mq;
        UT_hash_handle hh;
};

struct mlxm_mq_storage{
        struct mlxm_mq_entry *hash;
};

struct mlxm_globals{
        mxm_h mxm_context;
        mxm_ep_h mxm_ep;
        struct mlxm_mq_storage mq_storage;
        struct mpool *req_pool;
};
extern struct mlxm_globals mlxm_globals;

static inline
int mlxm_find_mq(struct mlxm_mq_storage *storage,
                 uint16_t id, mxm_mq_h *mq) {
        struct mlxm_mq_entry *mq_e = NULL;
        HASH_FIND(hh, storage->hash, &id, sizeof(uint16_t), mq_e);
        if (mq_e) {
                *mq = mq_e->mq;
                return 1;
        } else {
                return 0;
        }
};

static inline
int mlxm_mq_add_to_storage(struct mlxm_mq_storage *storage,
                           uint16_t id, mxm_mq_h *mq) {
        int mxm_err;
        struct mlxm_mq_entry *mq_entry;
        mq_entry = (struct mlxm_mq_entry*)malloc(sizeof(*mq_entry));
        mxm_err = mxm_mq_create(mlxm_globals.mxm_context,
                                id,
                                &mq_entry->mq);
        if (mxm_err) {
                FI_WARN(&mlxm_prov,FI_LOG_CORE,
                        "mxm_mq_create failed: mq_id %d, errno %d:%s\n",
                        id, mxm_err, mxm_error_string(mxm_err));
                return mlxm_errno(mxm_err);
        }
        FI_INFO(&mlxm_prov,FI_LOG_CORE,
                "MXM mq created, id 0x%x, %p\n",id , mq_entry->mq);

        mq_entry->mq_key = id;
        HASH_ADD(hh, storage->hash, mq_key, sizeof(uint16_t), mq_entry);
        *mq = mq_entry->mq;
        return 0;
};

static inline
void mlxm_mq_storage_init() {
        mlxm_globals.mq_storage.hash = NULL;
}

static inline
void mlxm_mq_storage_fini(struct mlxm_fid_ep *fid_ep) {
        struct mlxm_mq_entry *mq_e, *tmp;
        HASH_ITER(hh, mlxm_globals.mq_storage.hash, mq_e, tmp) {
                mxm_mq_destroy(mq_e->mq);
                HASH_DEL(mlxm_globals.mq_storage.hash,
                         mq_e);
                free(mq_e);
        }
}

#define MLXM_CQ_ENQUEUE(_queue, __ctx)                                  \
        do{                                                             \
                if (_queue.head == NULL) {                              \
                        _queue.head = __ctx->internal[0];               \
                        _queue.tail = __ctx->internal[0];               \
                } else {                                                \
                        ((struct fi_context *)(_queue.tail))->internal[0] = \
                                __ctx;                                  \
                        _queue.tail = __ctx;                            \
                }                                                       \
        }while(0)

#define MLXM_CQ_DEQUEUE(_queue, __ctx)                          \
        do{                                                     \
                __ctx = (struct fi_context*)(_queue.head);      \
                assert(__ctx);                                  \
                if (__ctx->internal[0] == __ctx) {              \
                        _queue.head = NULL;                     \
                } else {                                        \
                        _queue.head = __ctx->internal[0];       \
                }                                               \
        }while(0)

extern struct fi_ops_cm         mlxm_cm_ops;
extern struct fi_ops_tagged     mlxm_tagged_ops;
extern struct fi_ops_mr         mlxm_mr_ops;

#define MLXM_MEM_TAG_FORMAT (0xFFFF00000000LLU) /* MXM support 16 bit field for MQ, and 32 more bits for tag */
extern uint64_t mlxm_mem_tag_format;

int mlxm_domain_open(struct fid_fabric *fabric, struct fi_info *info,
                     struct fid_domain **fid, void *context);
int mlxm_ep_open(struct fid_domain *domain, struct fi_info *info,
                 struct fid_ep **fid, void *context);
int mlxm_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
                 struct fid_cq **cq, void *context);
int mlxm_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
                 struct fid_av **av, void *context);

#ifdef __cplusplus
}
#endif

#endif
