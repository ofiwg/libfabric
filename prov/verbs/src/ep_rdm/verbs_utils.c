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

#include <rdma/fi_errno.h>
#include "../fi_verbs.h"
#include "verbs_utils.h"
#include "verbs_rdm.h"

void fi_ibv_mem_pool_init(struct fi_ibv_mem_pool *pool, int init_size,
			  int max_size, int entry_size)
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

	pool->current_size = 0;
	pool->max_size = max_size;
	pool->entry_size = entry_asize;
}

int fi_ibv_rdm_tagged_req_match(struct dlist_entry *item, const void *other)
{
	const struct fi_ibv_rdm_tagged_request *req = other;
	return (item == &req->queue_entry);
}

int fi_ibv_rdm_tagged_req_match_by_info(struct dlist_entry *item,
					  const void *info)
{
	struct fi_ibv_rdm_tagged_request *request =
	    container_of(item, struct fi_ibv_rdm_tagged_request, queue_entry);

	const struct fi_verbs_rdm_tagged_request_minfo *minfo = info;

	return (((request->conn == NULL) || (request->conn == minfo->conn)) &&
		((request->tag & request->tagmask) ==
		 (minfo->tag   & request->tagmask)));
}

/*
 * The same as fi_ibv_rdm_tagged_req_match_by_info but conn and tagmask fields
 * are used for matching instead of request's ones
 */
int fi_ibv_rdm_tagged_req_match_by_info2(struct dlist_entry *item,
					 const void *info)
{
	struct fi_ibv_rdm_tagged_request *request =
	    container_of(item, struct fi_ibv_rdm_tagged_request, queue_entry);

	const struct fi_verbs_rdm_tagged_request_minfo *minfo = info;

	return (((minfo->conn == NULL) || (request->conn == minfo->conn)) &&
		((request->tag & minfo->tagmask) ==
		 (minfo->tag   & minfo->tagmask)));
}

int fi_ibv_rdm_tagged_send_postponed_process(struct dlist_entry *postponed_item,
					     const void *arg)
{
	const struct fi_ibv_rdm_tagged_send_ready_data *send_data = arg;

	struct fi_ibv_rdm_tagged_postponed_entry *postponed_entry =
	    container_of(postponed_item,
			 struct fi_ibv_rdm_tagged_postponed_entry, queue_entry);
	int ret = 0;
	if (!dlist_empty(&postponed_entry->conn->postponed_requests_head)) {
		struct dlist_entry *req_entry = 
			postponed_entry->conn->postponed_requests_head.next;

		struct fi_ibv_rdm_tagged_request *request =
		    container_of(req_entry, struct fi_ibv_rdm_tagged_request,
				 queue_entry);

		int res = fi_ibv_rdm_tagged_prepare_send_request(request,
								 send_data->ep);
		if (res) {
			fi_ibv_rdm_tagged_req_hndl(request,
						   FI_IBV_EVENT_SEND_READY,
						   (void *) send_data);
			ret++;
		}
	}
	return ret;
}
