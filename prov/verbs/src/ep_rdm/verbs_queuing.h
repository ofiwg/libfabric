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

#ifndef _VERBS_QUEING_H
#define _VERBS_QUEING_H

#include <stdlib.h>
#include <ofi_list.h>
#include "verbs_rdm.h"

static inline void
fi_ibv_rdm_move_to_cq(struct fi_ibv_rdm_cq *cq,
		      struct fi_ibv_rdm_request *request)
{
	FI_IBV_RDM_DBG_REQUEST("move_to_cq: ", request, FI_LOG_DEBUG);
	dlist_insert_tail(&request->queue_entry, &cq->request_cq);
}

static inline void
fi_ibv_rdm_remove_from_cq(struct fi_ibv_rdm_request *request)
{
	FI_IBV_RDM_DBG_REQUEST("remove_from_cq: ", request, FI_LOG_DEBUG);
	dlist_remove(&request->queue_entry);
}

static inline struct fi_ibv_rdm_request *
fi_ibv_rdm_take_first_from_cq(struct fi_ibv_rdm_cq *cq)
{
	if (cq && !dlist_empty(&cq->request_cq)) {
		struct fi_ibv_rdm_request *entry =
			container_of(cq->request_cq.next,
				     struct fi_ibv_rdm_request, queue_entry);
		fi_ibv_rdm_remove_from_cq(entry);
		return entry;
	}
	return NULL;
}

static inline void
fi_ibv_rdm_move_to_errcq(struct fi_ibv_rdm_cq *cq,
			 struct fi_ibv_rdm_request *request, ssize_t err)
{
	request->state.err = llabs(err);
	FI_IBV_RDM_DBG_REQUEST("move_to_errcq: ", request, FI_LOG_DEBUG);
	assert(request->context);
	dlist_insert_tail(&request->queue_entry,
			  &cq->request_errcq);
}

static inline void
fi_ibv_rdm_remove_from_errcq(struct fi_ibv_rdm_request *request)
{
	FI_IBV_RDM_DBG_REQUEST("remove_from_errcq: ", request, FI_LOG_DEBUG);
	dlist_remove(&request->queue_entry);
}

static inline struct fi_ibv_rdm_request *
fi_ibv_rdm_take_first_from_errcq(struct fi_ibv_rdm_cq *cq)
{
	if (cq && !dlist_empty(&cq->request_errcq)) {
		struct fi_ibv_rdm_request *entry =
			container_of(cq->request_errcq.next,
				     struct fi_ibv_rdm_request, queue_entry);
		fi_ibv_rdm_remove_from_errcq(entry);
		return entry;
	}
	return NULL;
}

static inline void
fi_ibv_rdm_move_to_unexpected_queue(struct fi_ibv_rdm_request *request,
				    struct fi_ibv_rdm_ep *ep)
{
	FI_IBV_RDM_DBG_REQUEST("move_to_unexpected_queue: ", request,
				FI_LOG_DEBUG);
	dlist_insert_tail(&request->queue_entry, &ep->fi_ibv_rdm_unexp_queue);
#if ENABLE_DEBUG
	request->minfo.conn->unexp_counter++;
#endif // ENABLE_DEBUG
}

static inline void
fi_ibv_rdm_remove_from_unexp_queue(struct fi_ibv_rdm_request *request)
{
	FI_IBV_RDM_DBG_REQUEST("remove_from_unexpected_queue: ", request,
				FI_LOG_DEBUG);
	dlist_remove(&request->queue_entry);
}

static inline struct fi_ibv_rdm_request *
fi_ibv_rdm_take_first_from_unexp_queue(struct fi_ibv_rdm_ep *ep)
{
	if (!dlist_empty(&ep->fi_ibv_rdm_unexp_queue)) {
		struct fi_ibv_rdm_request *entry =
			container_of(ep->fi_ibv_rdm_unexp_queue.next,
				     struct fi_ibv_rdm_request, queue_entry);
		fi_ibv_rdm_remove_from_unexp_queue(entry);
		return entry;
	}
	return NULL;
}

static inline void
fi_ibv_rdm_move_to_posted_queue(struct fi_ibv_rdm_request *request,
				struct fi_ibv_rdm_ep *ep)
{
	FI_IBV_RDM_DBG_REQUEST("move_to_posted_queue: ", request, FI_LOG_DEBUG);
	dlist_insert_tail(&request->queue_entry, &ep->fi_ibv_rdm_posted_queue);
	ep->posted_recvs++;
#if ENABLE_DEBUG
	if (request->minfo.conn) {
		request->minfo.conn->exp_counter++;
	}
#endif // ENABLE_DEBUG
}

static inline uint32_t
fi_ibv_rdm_remove_from_posted_queue(struct fi_ibv_rdm_request *request,
				    struct fi_ibv_rdm_ep *ep)
{
	FI_IBV_RDM_DBG_REQUEST("remove_from_posted_queue: ", request,
				FI_LOG_DEBUG);
	dlist_remove(&request->queue_entry);
	return ep->posted_recvs--;
}

static inline struct fi_ibv_rdm_request *
fi_ibv_rdm_take_first_from_posted_queue(struct fi_ibv_rdm_ep* ep)
{
	if (!dlist_empty(&ep->fi_ibv_rdm_posted_queue)) {
		struct fi_ibv_rdm_request *entry =
			container_of(ep->fi_ibv_rdm_posted_queue.next,
				     struct fi_ibv_rdm_request, queue_entry);
		fi_ibv_rdm_remove_from_posted_queue(entry, ep);
		return entry;
	}
	return NULL;
}

static inline int
fi_ibv_rdm_move_to_postponed_queue(struct fi_ibv_rdm_request *request)
{
	FI_IBV_RDM_DBG_REQUEST("move_to_postponed_queue: ", request, 
				FI_LOG_DEBUG);
	assert(request && request->minfo.conn);

	struct fi_ibv_rdm_conn *conn = request->minfo.conn;

	if (dlist_empty(&conn->postponed_requests_head)) {
		struct fi_ibv_rdm_postponed_entry *entry =
			util_buf_alloc(conn->ep->fi_ibv_rdm_postponed_pool);
		if (OFI_UNLIKELY(!entry)) {
			VERBS_WARN(FI_LOG_EP_DATA, "Unable to alloc buffer");
			return -FI_ENOMEM;
		}

		entry->conn = conn;	
		conn->postponed_entry = entry;

		dlist_insert_tail(&entry->queue_entry,
				  &conn->ep->fi_ibv_rdm_postponed_queue);
	}
	dlist_insert_tail(&request->queue_entry,
			  &conn->postponed_requests_head);

	return FI_SUCCESS;
}

static inline void
fi_ibv_rdm_remove_from_multi_recv_list(struct fi_ibv_rdm_multi_request *request,
				       struct fi_ibv_rdm_ep *ep)
{
	dlist_remove(&request->list_entry);
}

static inline void
fi_ibv_rdm_add_to_multi_recv_list(struct fi_ibv_rdm_multi_request *request,
				  struct fi_ibv_rdm_ep *ep)
{
	dlist_insert_tail(&request->list_entry,
			  &ep->fi_ibv_rdm_multi_recv_list);
}

static inline struct fi_ibv_rdm_multi_request *
fi_ibv_rdm_take_first_from_multi_recv_list(struct fi_ibv_rdm_ep *ep)
{
	if (!dlist_empty(&ep->fi_ibv_rdm_multi_recv_list)) {
		struct fi_ibv_rdm_multi_request *entry;
		dlist_pop_front(&ep->fi_ibv_rdm_multi_recv_list,
				struct fi_ibv_rdm_multi_request,
				entry, list_entry);
		return entry;
	}
	return NULL;
}

static inline void
fi_ibv_rdm_remove_from_postponed_queue(struct fi_ibv_rdm_request *request)
{
	FI_IBV_RDM_DBG_REQUEST("remove_from_postponed_queue: ", request,
				FI_LOG_DEBUG);

	struct fi_ibv_rdm_conn *conn = request->minfo.conn;
	assert(conn);
	assert(!dlist_empty(&conn->postponed_requests_head));

	/* 
	 * remove from conn->postponed_requests_head at first
	 * then if conn->postponed_requests_head is empty
	 * clean fi_ibv_rdm_postponed_queue
	 */

	dlist_remove(&request->queue_entry);
	request->queue_entry.next = request->queue_entry.prev = NULL;

	if (dlist_empty(&conn->postponed_requests_head)) {
		dlist_remove(&conn->postponed_entry->queue_entry);
		conn->postponed_entry->queue_entry.next = 
		conn->postponed_entry->queue_entry.prev = NULL;
		conn->postponed_entry->conn = NULL;

		util_buf_release(conn->ep->fi_ibv_rdm_postponed_pool,
				 conn->postponed_entry);
		conn->postponed_entry = NULL;
	}
}

static inline struct fi_ibv_rdm_request *
fi_ibv_rdm_take_first_from_postponed_queue(struct fi_ibv_rdm_ep *ep)
{
	if (!dlist_empty(&ep->fi_ibv_rdm_postponed_queue)) {
		struct fi_ibv_rdm_postponed_entry *entry = 
			container_of(ep->fi_ibv_rdm_postponed_queue.next,
				struct fi_ibv_rdm_postponed_entry, queue_entry);

		if (!dlist_empty(&entry->conn->postponed_requests_head)) {
			struct dlist_entry *req_entry = 
				entry->conn->postponed_requests_head.next;

			struct fi_ibv_rdm_request *request =
			    container_of(req_entry, struct fi_ibv_rdm_request,
					 queue_entry);
			fi_ibv_rdm_remove_from_postponed_queue(request);
			return request;
		}
	}

	return NULL;
}

static inline struct fi_ibv_rdm_request *
fi_ibv_rdm_take_first_match_from_postponed_queue(dlist_func_t *match,
						 const void *arg,
						 struct fi_ibv_rdm_ep *ep)
{
	struct dlist_entry *i, *j;
	dlist_foreach((&ep->fi_ibv_rdm_postponed_queue), i) {
		 struct fi_ibv_rdm_postponed_entry *entry = 
			container_of(i, struct fi_ibv_rdm_postponed_entry,
				     queue_entry);

		j = dlist_find_first_match((&entry->conn->postponed_requests_head),
					   match, arg);
		if (j) {
			struct fi_ibv_rdm_request *request =
				container_of(j, struct fi_ibv_rdm_request,
					     queue_entry);
			fi_ibv_rdm_remove_from_postponed_queue(request);
			return request;
		}
	}

	return NULL;
}

void fi_ibv_rdm_clean_queues(struct fi_ibv_rdm_ep* ep);

#endif   // _VERBS_QUEING_H
