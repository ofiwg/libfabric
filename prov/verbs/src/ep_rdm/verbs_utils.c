/*
 * Copyright (c) 2016, Cisco Systems, Inc. All rights reserved.
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

#include <ifaddrs.h>
#include <net/if.h>

#include <rdma/fi_errno.h>
#include "../fi_verbs.h"
#include "verbs_utils.h"
#include "verbs_rdm.h"
#include "verbs_queuing.h"

size_t rdm_buffer_size(size_t buf_send_size)
{
	size_t size = buf_send_size + FI_IBV_RDM_BUFF_SERVICE_DATA_SIZE +
		sizeof(struct fi_ibv_rdm_header) + FI_IBV_BUF_ALIGNMENT;
	size -= (size % FI_IBV_BUF_ALIGNMENT);
	return size;
}

int fi_ibv_rdm_req_match(struct dlist_entry *item, const void *other)
{
	const struct fi_ibv_rdm_request *req = other;
	return (item == &req->queue_entry);
}

int fi_ibv_rdm_req_match_by_info(struct dlist_entry *item, const void *info)
{
	struct fi_ibv_rdm_request *request =
		container_of(item, struct fi_ibv_rdm_request, queue_entry);

	const struct fi_ibv_rdm_minfo *minfo = info;

	return	(
			((request->minfo.conn == NULL) ||
			(request->minfo.conn == minfo->conn))
			&&
			(request->minfo.is_tagged ?
			((request->minfo.tag & request->minfo.tagmask) ==
			(minfo->tag          & request->minfo.tagmask)) :
			(request->minfo.is_tagged == minfo->is_tagged))
		);
}

/*
 * The same as fi_ibv_rdm_req_match_by_info but conn and tagmask fields
 * are used for matching instead of request's ones
 */
int fi_ibv_rdm_req_match_by_info2(struct dlist_entry *item, const void *info)
{
	struct fi_ibv_rdm_request *request =
		container_of(item, struct fi_ibv_rdm_request, queue_entry);

	const struct fi_ibv_rdm_minfo *minfo = info;

	return	(
			((minfo->conn == NULL) ||
			(request->minfo.conn == minfo->conn))
			&&
			(minfo->is_tagged ?
			((request->minfo.tag & minfo->tagmask) ==
			(minfo->tag          & minfo->tagmask)) :
			(request->minfo.is_tagged == minfo->is_tagged))
		);
}

/*
 * The same as fi_ibv_rdm_tagged_req_match_by_info2 but context field is added  
 * to compare
 */
int fi_ibv_rdm_req_match_by_info3(struct dlist_entry *item, const void *info)
{
	struct fi_ibv_rdm_request *request =
		container_of(item, struct fi_ibv_rdm_request, queue_entry);

	const struct fi_ibv_rdm_tagged_peek_data *peek_data = info;
	const void *context = (peek_data->flags & FI_CLAIM) ?
		peek_data->context : NULL;

	return ((request->context == context) && 
		fi_ibv_rdm_req_match_by_info2(item, &peek_data->minfo));
}

int fi_ibv_rdm_postponed_process(struct dlist_entry *postponed_item,
				 const void *arg)
{
	const struct fi_ibv_rdm_tagged_send_ready_data *send_data = arg;

	struct fi_ibv_rdm_postponed_entry *postponed_entry =
	    container_of(postponed_item,
			 struct fi_ibv_rdm_postponed_entry, queue_entry);
	int ret = 0;
	if (!dlist_empty(&postponed_entry->conn->postponed_requests_head)) {
		struct dlist_entry *req_entry = 
			postponed_entry->conn->postponed_requests_head.next;

		struct fi_ibv_rdm_request *request =
			container_of(req_entry, struct fi_ibv_rdm_request,
				     queue_entry);

		int res = 0;
		if ((request->state.eager < FI_IBV_STATE_EAGER_RMA_INJECT) &&
		    (request->sbuf == NULL)) {
			res = fi_ibv_rdm_tagged_prepare_send_request(request,
								 send_data->ep);
		} else  {
			/*
			 * This case is possible only for segmented RNDV msg or 
			 * RMA operation (> 1GB), connection must be already 
			 * established
			 */
			assert(request->state.rndv != FI_IBV_STATE_RNDV_NOT_USED);
			assert(fi_ibv_rdm_check_connection(request->minfo.conn));
			if (request->state.eager <= FI_IBV_STATE_EAGER_RECV_END) {
				res = !TSEND_RESOURCES_IS_BUSY(request->minfo.conn,
							      send_data->ep);
			} else {
				res = !RMA_RESOURCES_IS_BUSY(request->minfo.conn,
							      send_data->ep);
			}
		}

		if (res) {
			fi_ibv_rdm_req_hndl(request, FI_IBV_EVENT_POST_READY,
					    (void *) send_data);
			ret++;
		}
	}
	return ret;
}

void fi_ibv_rdm_conn_init_cm_role(struct fi_ibv_rdm_conn *conn,
				  struct fi_ibv_rdm_ep *ep)
{
	const int addr_cmp = memcmp(&conn->addr, &ep->my_addr,
				    FI_IBV_RDM_DFLT_ADDRLEN);

	if (addr_cmp < 0) {
		conn->cm_role = FI_VERBS_CM_ACTIVE;
	} else if (addr_cmp > 0) {
		conn->cm_role = FI_VERBS_CM_PASSIVE;
	} else {
		conn->cm_role = FI_VERBS_CM_SELF;
	}
}

void fi_ibv_rdm_clean_queues(struct fi_ibv_rdm_ep *ep)
{
	struct fi_ibv_rdm_request *request;
	struct fi_ibv_rdm_multi_request *multi_request;

	while ((request = fi_ibv_rdm_take_first_from_unexp_queue(ep))) {
		if (request->unexp_rbuf) {
			util_buf_release(ep->fi_ibv_rdm_extra_buffers_pool,
					 request->unexp_rbuf);
		}
		FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
		util_buf_release(ep->fi_ibv_rdm_request_pool, request);
	}

	while ((multi_request = fi_ibv_rdm_take_first_from_multi_recv_list(ep)))
		util_buf_release(ep->fi_ibv_rdm_multi_request_pool, multi_request);

	while ((request = fi_ibv_rdm_take_first_from_posted_queue(ep))) {
 		/* Check `request->context->internal[0] == NULL` in fi_cancel
		 * will handle the case that request was already canceled
		 * internally by provider */
		request->context->internal[0] = NULL;
		if (request->iov_count > 0) {
			util_buf_release(ep->fi_ibv_rdm_extra_buffers_pool,
					 request->unexp_rbuf);
		}
		FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
		util_buf_release(ep->fi_ibv_rdm_request_pool, request);
	}

	while ((request = fi_ibv_rdm_take_first_from_postponed_queue(ep))) {
		if (request->iov_count > 0) {
			util_buf_release(ep->fi_ibv_rdm_extra_buffers_pool,
					 request->unexp_rbuf);
		}
		FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
		util_buf_release(ep->fi_ibv_rdm_request_pool, request);
	}

	while ((request = fi_ibv_rdm_take_first_from_cq(ep->fi_scq))) {
		if (request->iov_count > 0) {
			util_buf_release(ep->fi_ibv_rdm_extra_buffers_pool,
					 request->unexp_rbuf);
		}
		FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
		util_buf_release(ep->fi_ibv_rdm_request_pool, request);
	}

	while ((request = fi_ibv_rdm_take_first_from_cq(ep->fi_rcq))) {
		if (request->iov_count > 0) {
			util_buf_release(ep->fi_ibv_rdm_extra_buffers_pool,
					 request->unexp_rbuf);
		}
		FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
		util_buf_release(ep->fi_ibv_rdm_request_pool, request);
	}

	while ((request = fi_ibv_rdm_take_first_from_errcq(ep->fi_scq))) {
		FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
		util_buf_release(ep->fi_ibv_rdm_request_pool, request);
	}

	while ((request = fi_ibv_rdm_take_first_from_errcq(ep->fi_rcq))) {
		FI_IBV_RDM_DBG_REQUEST("to_pool: ", request, FI_LOG_DEBUG);
		util_buf_release(ep->fi_ibv_rdm_request_pool, request);
	}
}

ssize_t
fi_ibv_rdm_send_common(struct fi_ibv_rdm_send_start_data* sdata)
{
	struct fi_ibv_rdm_request *request =
		util_buf_alloc(sdata->ep_rdm->fi_ibv_rdm_request_pool);
	if (OFI_UNLIKELY(!request))
		return -FI_EAGAIN;
	request->ep = sdata->ep_rdm;
	FI_IBV_RDM_DBG_REQUEST("get_from_pool: ", request, FI_LOG_DEBUG);

	/* Initial state */
	request->state.eager = FI_IBV_STATE_EAGER_BEGIN;
	request->state.rndv  = FI_IBV_STATE_RNDV_NOT_USED;
	request->state.err   = FI_SUCCESS;

	/* postponed_entry means that there are elements postponed to
	 * send & current request must be queued */
	const int in_order = (sdata->conn->postponed_entry) ? 0 : 1;
	int ret = fi_ibv_rdm_req_hndl(request, FI_IBV_EVENT_SEND_START, sdata);

	if (!ret && in_order &&
	    fi_ibv_rdm_tagged_prepare_send_request(request, sdata->ep_rdm)) {
		struct fi_ibv_rdm_tagged_send_ready_data req_data = 
			{ .ep = sdata->ep_rdm };
		ret = fi_ibv_rdm_req_hndl(request, FI_IBV_EVENT_POST_READY,
					  &req_data);
	}

	return ret;
}

ssize_t
rdm_trecv_second_event(struct fi_ibv_rdm_request *request, 
			struct fi_ibv_rdm_ep *ep)
{
	ssize_t ret = FI_SUCCESS;

	switch (request->state.rndv)
	{
	case FI_IBV_STATE_RNDV_NOT_USED:
		if (request->state.eager != FI_IBV_STATE_EAGER_RECV_WAIT4PKT) {
			struct fi_ibv_recv_got_pkt_process_data data = {
				.ep = ep
			};
			ret = fi_ibv_rdm_req_hndl(request,
						  FI_IBV_EVENT_RECV_START,
						  &data);
		}
		break;
	case FI_IBV_STATE_RNDV_RECV_WAIT4RES:
		if (fi_ibv_rdm_tagged_prepare_send_request(request, ep)) {
			struct fi_ibv_rdm_tagged_send_ready_data data = {
				.ep = ep
			};
			ret = fi_ibv_rdm_req_hndl(request,
						  FI_IBV_EVENT_POST_READY,
						  &data);
		}
		break;
	default:
		break;
	}

	return ret;
}
