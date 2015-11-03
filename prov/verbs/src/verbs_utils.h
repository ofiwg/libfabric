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

#ifndef _VERBS_UTILS_H
#define _VERBS_UTILS_H

#include <alloca.h>
#include <malloc.h>
#include <stddef.h>
#include <string.h>
#include <sys/types.h>

#include <infiniband/verbs.h>

#include <rdma/fi_log.h>

struct iovec;
struct sockaddr;
struct fi_ibv_msg_ep;

extern struct fi_provider fi_ibv_prov;

#define fi_ibv_set_sge(sge, buf, len, desc)		\
    do {                                        \
        sge.addr = (uintptr_t)buf;              \
        sge.length = (uint32_t)len;             \
        sge.lkey = (uint32_t)(uintptr_t)desc;   \
    } while (0)

#define fi_ibv_set_sge_iov(sg_list, iov, count, desc, len)		\
    do {                                                        \
        int i;                                                  \
        if (count) {                                            \
            sg_list = alloca(sizeof(*sg_list) * count);         \
            for (i = 0; i < count; i++) {                       \
                fi_ibv_set_sge(sg_list[i],                      \
                               iov[i].iov_base,                 \
                               iov[i].iov_len,                  \
                               desc[i]);                        \
                len += iov[i].iov_len;                          \
            }                                                   \
        }                                                       \
    } while (0)

#define fi_ibv_set_sge_inline(sge, buf, len)                    \
    do {                                                        \
        sge.addr = (uintptr_t)buf;                              \
        sge.length = (uint32_t)len;                             \
    } while (0)

#define fi_ibv_set_sge_iov_inline(sg_list, iov, count, len)		\
    do {                                                        \
        int i;                                                  \
        if (count) {                                            \
            sg_list = alloca(sizeof(*sg_list) * count);         \
            for (i = 0; i < count; i++) {                       \
                fi_ibv_set_sge_inline(sg_list[i],               \
                iov[i].iov_base,                                \
                iov[i].iov_len);                                \
                len += iov[i].iov_len;                          \
            }                                                   \
        }                                                       \
    } while (0)

ssize_t fi_ibv_send(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr, size_t len,
                    int count, void *context);
ssize_t fi_ibv_send_buf_inline(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
                               const void *buf, size_t len);
int fi_ibv_sockaddr_len(struct sockaddr *addr);
int fi_ibv_copy_addr(void *dst_addr, size_t *dst_addrlen, void *src_addr);
ssize_t fi_ibv_send_buf(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
                        const void *buf, size_t len, void *desc, void *context);
ssize_t fi_ibv_send_iov_flags(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
                              const struct iovec *iov, void **desc, int count,
                              void *context, uint64_t flags);

#define fi_ibv_send_iov(ep, wr, iov, desc, count, context)	\
	fi_ibv_send_iov_flags(ep, wr, iov, desc, count, context,\
			ep->info->tx_attr->op_flags)

#define fi_ibv_send_msg(ep, wr, msg, flags)                                 \
	fi_ibv_send_iov_flags(ep, wr, msg->msg_iov, msg->desc, msg->iov_count,	\
			msg->context, flags)

#define FI_IBV_RDM_TAGGED_MAX(a, b) (((a) < (b)) ? (b) : (a))
#define FI_IBV_RDM_TAGGED_MIN(a, b) (((a) < (b)) ? (a) : (b))

#define FI_IBV_RDM_DFLT_ADDRLEN                                         \
    (/*ep->my_ipoib_addr.sin_addr.s_addr)*/ sizeof(in_addr_t) +         \
     /*ep->cm_listener_port */              sizeof(uint16_t))

#define FI_IBV_RDM_TRUE (1)
#define FI_IBV_RDM_FALSE (0)

#define FI_IBV_RDM_CM_THREAD_TIMEOUT (100)
#define FI_IBV_RDM_MEM_ALIGNMENT (4096)

#define FI_IBV_RDM_TAGGED_DFLT_BUFFER_NUM (4)

#define FI_IBV_RDM_TAGGED_DFLT_BUFFER_SIZE                              \
    ((8 * 1024 + FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE) +            \
     (8 * 1024 + FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE) %            \
     FI_IBV_RDM_MEM_ALIGNMENT)

#define FI_IBV_RDM_TAGGED_DFLT_RQ_SIZE  (1000)

/* TODO: CQs depths increased from 100 to 1000 to prevent
 *      "Work Request Flushed Error" in stress tests like alltoall.
 */
#define FI_IBV_RDM_TAGGED_DFLT_SCQ_SIZE (1000)
#define FI_IBV_RDM_TAGGED_DFLT_RCQ_SIZE (1000)

#define FI_IBV_RDM_CM_RESOLVEADDR_TIMEOUT (30000)
#define FI_IBV_ERROR(format, ...) do {                                  \
        char _hostname[100];                                            \
        gethostname(_hostname, sizeof(_hostname));                      \
        fprintf(stderr, "[%s:%d] libfabric:verbs:%s:%d:<error>:"format, \
                _hostname, getpid(),                                    \
                __FUNCTION__, __LINE__, ##__VA_ARGS__);                 \
}while(0)

#define FI_VERBS_DBG(...) \
        FI_DBG(&fi_ibv_prov, FI_LOG_CORE, ##__VA_ARGS__)

/* memory pool utils
 */

struct fi_ibv_mem_pool_entry {
    struct fi_ibv_mem_pool_entry *next;
    uint64_t is_malloced;
};

struct fi_ibv_mem_pool {
    struct fi_ibv_mem_pool_entry *head;
    int current_size;
    int max_size;
    int entry_size;
    void *storage;
};

/* memory pool utils
 */

static inline void fi_ibv_mem_pool_init(struct fi_ibv_mem_pool *pool,
                                        int init_size, int max_size,
                                        int entry_size)
{
    int size = init_size < max_size ? init_size : max_size;
    int i;

    pool->head = memalign(FI_IBV_RDM_MEM_ALIGNMENT, entry_size * size);
    memset(pool->head, 0, entry_size * size);
    pool->storage = (void *)pool->head;
    struct fi_ibv_mem_pool_entry *tmp = pool->head;
    for (i = 1; i < size; i++) {
        tmp->next = (struct fi_ibv_mem_pool_entry *)((char *)tmp + entry_size);
        tmp = tmp->next;
    }
    tmp->next = NULL;
    pool->current_size = size;
    pool->max_size = max_size;
    pool->entry_size = entry_size;
}

static inline struct fi_ibv_mem_pool_entry *
fi_verbs_mem_pool_get(struct fi_ibv_mem_pool *pool)
{
    struct fi_ibv_mem_pool_entry *rst;
    if (pool->head != NULL) {
        rst = pool->head;
        pool->head = rst->next;
    } else {
        rst = (struct fi_ibv_mem_pool_entry *)calloc(1, pool->entry_size);
        rst->is_malloced = 1;
        pool->current_size++;
        FI_VERBS_DBG("MALLOCED: %p, %d\n", rst, pool->current_size);
    }
    return rst;
}

static inline void fi_ibv_mem_pool_return(struct fi_ibv_mem_pool_entry
                                          *entry,
                                          struct fi_ibv_mem_pool *pool)
{
    if (entry->is_malloced) {
        pool->current_size--;
        FI_VERBS_DBG("FREED: %p, %d\n", entry, pool->current_size);
        free(entry);
    } else {
        entry->next = pool->head;
        pool->head = entry;
    }
}

static inline void fi_ibv_mem_pool_fini(struct fi_ibv_mem_pool *pool)
{
    if (pool->storage) {
        free(pool->storage);
    }
}

#if ENABLE_DEBUG
#define FI_IBV_PRINT_LIST(list, name) do {              \
        int count;                                      \
        struct fi_ibv_rdm_tagged_request *tmp;          \
                                                        \
        DL_COUNT(list, tmp, count);                     \
        char str[1000];                                 \
        sprintf(str, "list %s, head %p, count %d: ",    \
                name, list, count);                     \
                                                        \
        DL_FOREACH(list, tmp)                           \
        {                                               \
            char str2[100];                             \
            sprintf(str2, "%p, ", (void*)tmp);          \
            strcat(str, str2);                          \
        }                                               \
                                                        \
        FI_VERBS_DBG("%s\n", str);                      \
    } while(0)
#else                           // ENABLE_DEBUG
#define FI_IBV_PRINT_LIST(list, name)
#endif                          // ENABLE_DEBUG

#define FI_IBV_DBG_OPCODE(wc_opcode, str)                                      \
        FI_VERBS_DBG("CQ COMPL: "str" -> %s\n",                                \
        wc_opcode == IBV_WC_SEND       ? "IBV_WC_SEND"       :                 \
        wc_opcode == IBV_WC_RDMA_WRITE ? "IBV_WC_RDMA_WRITE" :                 \
        wc_opcode == IBV_WC_RDMA_READ  ? "IBV_WC_RDMA_READ"  :                 \
        wc_opcode == IBV_WC_COMP_SWAP  ? "IBV_WC_COMP_SWAP"  :                 \
        wc_opcode == IBV_WC_FETCH_ADD  ? "IBV_WC_FETCH_ADD"  :                 \
        wc_opcode == IBV_WC_BIND_MW    ? "IBV_WC_BIND_MW"    :                 \
        wc_opcode == IBV_WC_RECV       ? "IBV_WC_RECV"       :                 \
        wc_opcode == IBV_WC_RECV_RDMA_WITH_IMM ? "IBV_WC_RECV_RDMA_WITH_IMM" : \
        "IBV_WC_UNKNOWN!!!");

#if ENABLE_DEBUG

#define FI_IBV_RDM_TAGGED_DBG_REQUEST(prefix, request, level)               \
do {                                                                        \
    const size_t max_str_len = 1024;                                        \
    char str[max_str_len];                                                  \
    snprintf(str, max_str_len,                                              \
            "%s request: %p, eager_state: %s, rndv_state: %s, tag: 0x%llx," \
            "len: %d context: %p, connection: %p\n",                        \
            prefix,                                                         \
            request,                                                        \
            fi_ibv_rdm_tagged_req_eager_state_to_str(request->state.eager), \
            fi_ibv_rdm_tagged_req_rndv_state_to_str(request->state.rndv),   \
            request->tag,                                                   \
            request->len,                                                   \
            request->context,                                               \
            request->conn);                                                 \
                                                                            \
    switch (level)                                                          \
    {                                                                       \
        case FI_LOG_WARN:                                                   \
        case FI_LOG_TRACE:                                                  \
        case FI_LOG_INFO:                                                   \
            fprintf(stderr, str);                                           \
            break;                                                          \
        case FI_LOG_DEBUG:                                                  \
        default:                                                            \
            FI_VERBS_DBG(str);                                              \
            break;                                                          \
    }                                                                       \
} while (0);

#else                           // ENABLE_DEBUG

#define FI_IBV_RDM_TAGGED_DBG_REQUEST(prefix, request, level)

#endif                          // ENABLE_DEBUG

#endif /* _VERBS_UTILS_H */
