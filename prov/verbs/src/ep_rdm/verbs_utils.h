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
#include <inttypes.h>
#include <string.h>
#include <sys/types.h>

#include <infiniband/verbs.h>

#include <rdma/fi_log.h>
#include "../fi_verbs.h"

#if defined(__ICC) || defined(__INTEL_COMPILER) || \
 defined(__GNUC__) || defined(__GNUG__)
#include "xmmintrin.h"
#define FI_IBV_PREFETCH_ADDR(_addr) {                    \
        _mm_prefetch((const char *)(_addr), _MM_HINT_T0);\
}
#else /* ICC || GCC */
#define FI_IBV_PREFETCH_ADDR(_addr)
#endif /* ICC || GCC */

/* TODO: Merge anything useful into verbs_rdm.h */

struct fi_ibv_msg_ep;

#define FI_IBV_RDM_DFLT_ADDRLEN                                         \
    (/*ep->my_ipoib_addr.sin_addr.s_addr)*/ sizeof(in_addr_t) +         \
     /*ep->cm_listener_port */              sizeof(uint16_t))

#define FI_IBV_RDM_CM_THREAD_TIMEOUT (100)
#define FI_IBV_RDM_MEM_ALIGNMENT (64)
#define FI_IBV_RDM_BUF_ALIGNMENT (4096)

#define FI_IBV_RDM_TAGGED_DFLT_BUFFER_NUM (8)

#define FI_IBV_RDM_TAGGED_DFLT_BUFFER_SIZE                              \
	((8 * 1024 + FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE) +        \
	 (8 * 1024 + FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE) %        \
	  FI_IBV_RDM_BUF_ALIGNMENT)

#define FI_IBV_RDM_TAGGED_DFLT_RQ_SIZE  (1000)

/* TODO: CQs depths increased from 100 to 1000 to prevent
 *      "Work Request Flushed Error" in stress tests like alltoall.
 */
#define FI_IBV_RDM_TAGGED_DFLT_SCQ_SIZE (1000)
#define FI_IBV_RDM_TAGGED_DFLT_RCQ_SIZE (1000)

#define FI_IBV_RDM_CM_RESOLVEADDR_TIMEOUT (30000)

/* TODO: Create and use a common abstraction */

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

/* TODO: Remove overuse of static inlines */
static inline void fi_ibv_mem_pool_init(struct fi_ibv_mem_pool *pool,
                                        int init_size, int max_size,
                                        int entry_size)
{
	int size = init_size < max_size ? init_size : max_size;
	int i;
	int entry_asize = entry_size % FI_IBV_RDM_MEM_ALIGNMENT;
	entry_asize += entry_size;

	pool->head = memalign(FI_IBV_RDM_BUF_ALIGNMENT, entry_asize * size);
	memset(pool->head, 0, entry_asize * size);
	pool->storage = (void *)pool->head;
	struct fi_ibv_mem_pool_entry *tmp = pool->head;
	for (i = 1; i < size; i++) {
		tmp->next = (struct fi_ibv_mem_pool_entry *)
				((char *)tmp + entry_asize);
		tmp = tmp->next;
	}
	tmp->next = NULL;
	pool->current_size = size;
	pool->max_size = max_size;
	pool->entry_size = entry_asize;
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
		VERBS_DBG(FI_LOG_FABRIC, "MALLOCED: %p, %d\n", rst, pool->current_size);
	}
	return rst;
}

static inline void
fi_ibv_mem_pool_return(struct fi_ibv_mem_pool_entry *entry,
		       struct fi_ibv_mem_pool *pool)
{
	if (entry->is_malloced) {
		pool->current_size--;
		VERBS_DBG(FI_LOG_FABRIC, "FREED: %p, %d\n", entry, pool->current_size);
		free(entry);
	} else {
		entry->next = pool->head;
		pool->head = entry;
	}
}

static inline void fi_ibv_mem_pool_fini(struct fi_ibv_mem_pool *pool)
{
	free(pool->storage);
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
        VERBS_DBG(FI_LOG_EP_DATA, "%s\n", str);         \
    } while(0)
#else                           // ENABLE_DEBUG
#define FI_IBV_PRINT_LIST(list, name)
#endif                          // ENABLE_DEBUG

/* TODO: Holy macro batman, use verbs calls */
#define FI_IBV_DBG_OPCODE(wc_opcode, str)                                      \
        VERBS_DBG(FI_LOG_CQ, "CQ COMPL: "str" -> %s\n",                        \
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
            "%s request: %p, eager_state: %s, rndv_state: %s, tag: 0x%" PRIx64 ", len: %"PRIx64" context: %p, connection: %p\n",	\
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
            VERBS_INFO(FI_LOG_EP_DATA, "%s", str);                          \
            break;                                                          \
        case FI_LOG_DEBUG:                                                  \
        default:                                                            \
            VERBS_DBG(FI_LOG_EP_DATA, "%s", str);                           \
            break;                                                          \
    }                                                                       \
} while (0);

#else                           // ENABLE_DEBUG

#define FI_IBV_RDM_TAGGED_DBG_REQUEST(prefix, request, level)

#endif                          // ENABLE_DEBUG

struct fi_verbs_rdm_tagged_request_minfo {
	struct fi_ibv_rdm_tagged_conn	*conn;
	uint64_t			tag;
	uint64_t			tagmask;
} ;

int fi_ibv_rdm_tagged_req_match(struct dlist_entry *item, const void *other);
int fi_ibv_rdm_tagged_req_match_by_info(struct dlist_entry *item,
                                        const void *info);
int fi_ibv_rdm_tagged_req_match_by_info2(struct dlist_entry *item,
                                         const void *info);
void fi_ibv_rdm_tagged_send_postponed_process(struct dlist_entry *item,
                                              const void *arg);

#endif /* _VERBS_UTILS_H */
