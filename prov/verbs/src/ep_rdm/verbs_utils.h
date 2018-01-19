/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
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

#include "../fi_verbs.h"

#if (defined(__ICC) || defined(__INTEL_COMPILER) ||	\
 defined(__GNUC__) || defined(__GNUG__)) &&		\
 defined(__x86_64__)
#include "xmmintrin.h"
#define FI_IBV_PREFETCH_ADDR(_addr) {                    \
        _mm_prefetch((const char *)(_addr), _MM_HINT_T0);\
}
#else /* ICC || GCC && x86_64 */
#define FI_IBV_PREFETCH_ADDR(_addr)
#endif /* ICC || GCC && x86_64 */

/* TODO: Merge anything useful into verbs_rdm.h */

struct fi_ibv_msg_ep;

#define FI_IBV_RDM_DFLT_ADDRLEN	(sizeof (struct sockaddr_in))

#define FI_IBV_RDM_CM_THREAD_TIMEOUT (100)

#define FI_IBV_RDM_TAGGED_DFLT_BUFFER_NUM (8)
#define FI_IBV_RDM_DFLT_CQREAD_BUNCH_SIZE (FI_IBV_RDM_TAGGED_DFLT_BUFFER_NUM)

#define FI_IBV_RDM_DFLT_BUFFER_SIZE					\
	(3 * FI_IBV_BUF_ALIGNMENT)

#define FI_IBV_RDM_DFLT_BUFFERED_SIZE					\
	(FI_IBV_RDM_DFLT_BUFFER_SIZE -					\
	 FI_IBV_RDM_BUFF_SERVICE_DATA_SIZE -				\
	 sizeof(struct fi_ibv_rdm_header))

/*
 * calculates internal buffer size from user defined buffered send size in
 * consideration of wired protocols, alignment, etc
 */
size_t rdm_buffer_size(size_t buf_send_size);

#ifdef HAVE_VERBS_EXP_H
/* 128MB is ODP MR limitation */
#define FI_IBV_RDM_SEG_MAXSIZE (128*1024*1024)
#else /* HAVE_VERBS_EXP_H */
/* 1GB is RC_QP limitation */
#define FI_IBV_RDM_SEG_MAXSIZE (1024*1024*1024)
#endif /* HAVE_VERBS_EXP_H */

/* TODO: CQs depths increased from 100 to 1000 to prevent
 *      "Work Request Flushed Error" in stress tests like alltoall.
 */
#define FI_IBV_RDM_TAGGED_DFLT_SCQ_SIZE (1000)
#define FI_IBV_RDM_TAGGED_DFLT_RCQ_SIZE (1000)

#define FI_IBV_RDM_CM_RESOLVEADDR_TIMEOUT (30000)

#if ENABLE_DEBUG
#define FI_IBV_RDM_CHECK_RECV_WC(wc)						\
	((wc->status == IBV_WC_SUCCESS) &&					\
	(wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM || wc->opcode == IBV_WC_RECV))
#else
#define FI_IBV_RDM_CHECK_RECV_WC(wc) (wc->status == IBV_WC_SUCCESS)
#endif /* ENABLE_DEBUG */

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

#define FI_IBV_RDM_DBG_REQUEST(prefix, request, level)				\
do {										\
	const size_t max_str_len = 1024;					\
	char str[max_str_len];							\
	snprintf(str, max_str_len,						\
		 "%s request: %p, eager_state: %s, rndv_state: %s,"		\
		 " err_state: %ld, tag: 0x%lx, len: %lu, rest: %lu,"		\
		 "context: %p, connection: %p ep: %p\n",			\
		 prefix,							\
		 request,							\
		 fi_ibv_rdm_req_eager_state_to_str(request->state.eager),	\
		 fi_ibv_rdm_req_rndv_state_to_str(request->state.rndv),		\
		 request->state.err,						\
		 request->minfo.tag,						\
		 request->len,							\
		 request->rest_len,						\
		 request->context,						\
		 request->minfo.conn,						\
		 request->ep);							\
										\
	switch (level)								\
	{									\
	case FI_LOG_WARN:							\
	case FI_LOG_TRACE:							\
	case FI_LOG_INFO:							\
		VERBS_INFO(FI_LOG_EP_DATA, "%s", str);				\
		break;								\
	case FI_LOG_DEBUG:							\
	default:								\
		VERBS_DBG(FI_LOG_EP_DATA, "%s", str);				\
		break;								\
	}									\
} while (0);

#else                           // ENABLE_DEBUG

#define FI_IBV_RDM_DBG_REQUEST(prefix, request, level)

#endif                          // ENABLE_DEBUG

struct fi_ibv_rdm_minfo {
	struct fi_ibv_rdm_conn	*conn;
	uint64_t		is_tagged; /* TODO: unexpected RTS for MSG */
	uint64_t		tag;
	uint64_t		tagmask;
};

struct fi_ibv_rdm_tagged_peek_data {
	struct fi_ibv_rdm_minfo minfo;
	void *context;
	uint64_t flags;
};

struct fi_ibv_rdm_cm;
struct fi_ibv_rdm_request;
struct fi_ibv_rdm_send_start_data;

int fi_ibv_rdm_req_match(struct dlist_entry *item, const void *other);
int fi_ibv_rdm_req_match_by_info(struct dlist_entry *item, const void *info);
int fi_ibv_rdm_req_match_by_info2(struct dlist_entry *item, const void *info);
int fi_ibv_rdm_req_match_by_info3(struct dlist_entry *item, const void *info);
int fi_ibv_rdm_postponed_process(struct dlist_entry *item, const void *arg);
void fi_ibv_rdm_conn_init_cm_role(struct fi_ibv_rdm_conn *conn,
				  struct fi_ibv_rdm_ep *ep);

ssize_t fi_ibv_rdm_send_common(struct fi_ibv_rdm_send_start_data* sdata);
ssize_t rdm_trecv_second_event(struct fi_ibv_rdm_request *request,
			       struct fi_ibv_rdm_ep *ep);

#endif /* _VERBS_UTILS_H */
