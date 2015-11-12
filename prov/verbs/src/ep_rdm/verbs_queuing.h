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
#include <fi_list.h>
#include "verbs_rdm.h"

/* managing of queues
 */

extern struct dlist_entry fi_ibv_rdm_tagged_request_ready_queue;
extern struct dlist_entry fi_ibv_rdm_tagged_recv_unexp_queue;
extern struct dlist_entry fi_ibv_rdm_tagged_recv_posted_queue;
extern struct dlist_entry fi_ibv_rdm_tagged_send_postponed_queue;

extern struct fi_ibv_mem_pool fi_ibv_rdm_tagged_postponed_pool;


static inline void
fi_ibv_rdm_tagged_move_to_ready_queue(struct fi_ibv_rdm_tagged_request *request)
{
	FI_IBV_RDM_TAGGED_DBG_REQUEST("move_to_ready_queue: ",
				      request, FI_LOG_DEBUG);
	dlist_insert_tail(&request->queue_entry,
			  &fi_ibv_rdm_tagged_request_ready_queue);
}

static inline void
fi_ibv_rdm_tagged_remove_from_ready_queue(
		struct fi_ibv_rdm_tagged_request *request)
{
	FI_IBV_RDM_TAGGED_DBG_REQUEST("remove_from_ready_queue: ",
				      request, FI_LOG_DEBUG);
	dlist_remove(&request->queue_entry);
}

static inline void
fi_ibv_rdm_tagged_move_to_unexpected_queue(
		struct fi_ibv_rdm_tagged_request *request)
{
	FI_IBV_RDM_TAGGED_DBG_REQUEST("move_to_unexpected_queue: ",
				      request, FI_LOG_DEBUG);
	dlist_insert_tail(&request->queue_entry,
			  &fi_ibv_rdm_tagged_recv_unexp_queue);
}

static inline void
fi_ibv_rdm_tagged_remove_from_unexp_queue(
		struct fi_ibv_rdm_tagged_request *request)
{
	FI_IBV_RDM_TAGGED_DBG_REQUEST("remove_from_unexpected_queue: ", request,
				      FI_LOG_DEBUG);
	dlist_remove(&request->queue_entry);
}

static inline void
fi_ibv_rdm_tagged_move_to_posted_queue(
		struct fi_ibv_rdm_tagged_request *request,
		struct fi_ibv_rdm_ep *ep)
{
	FI_IBV_RDM_TAGGED_DBG_REQUEST("move_to_posted_queue: ", request,
				      FI_LOG_DEBUG);
	dlist_insert_tail(&request->queue_entry,
			  &fi_ibv_rdm_tagged_recv_posted_queue);
	ep->pend_recv++;
}

static inline void
fi_ibv_rdm_tagged_remove_from_posted_queue(
		struct fi_ibv_rdm_tagged_request *request,
		struct fi_ibv_rdm_ep *ep)
{
	FI_IBV_RDM_TAGGED_DBG_REQUEST("remove_from_posted_queue: ", request,
				      FI_LOG_DEBUG);
	dlist_remove(&request->queue_entry);
	ep->pend_recv--;
}

static inline void
fi_ibv_rdm_tagged_move_to_postponed_queue(
		struct fi_ibv_rdm_tagged_request *request)
{
	FI_IBV_RDM_TAGGED_DBG_REQUEST("move_to_postponed_queue: ", request,
				      FI_LOG_DEBUG);
	assert(request->conn);

	if (dlist_empty(&request->conn->postponed_requests_head)) {
		struct fi_ibv_rdm_tagged_postponed_entry *entry =
			(struct fi_ibv_rdm_tagged_postponed_entry *)
			fi_verbs_mem_pool_get(&fi_ibv_rdm_tagged_postponed_pool);

		entry->conn = request->conn;
		entry->conn->postponed_entry = entry;
		dlist_insert_tail(&entry->queue_entry,
				  &fi_ibv_rdm_tagged_send_postponed_queue);
	}
	dlist_insert_tail(&request->queue_entry,
			  &request->conn->postponed_requests_head);
}

static inline void
fi_ibv_rdm_tagged_remove_from_postponed_queue(
		struct fi_ibv_rdm_tagged_request *request)
{
	FI_IBV_RDM_TAGGED_DBG_REQUEST("remove_from_postponed_queue: ", request,
				      FI_LOG_DEBUG);

	assert(!dlist_empty(&request->conn->postponed_requests_head));

	dlist_remove(&request->queue_entry);
	if (dlist_empty(&request->conn->postponed_requests_head)) {
		dlist_remove(&request->conn->postponed_entry->queue_entry);
		request->conn->postponed_entry->conn = NULL;
		fi_ibv_mem_pool_return(&request->conn->postponed_entry->mpe,
					&fi_ibv_rdm_tagged_postponed_pool);
		request->conn->postponed_entry = NULL;
	}
}

#endif   // _VERBS_QUEING_H
