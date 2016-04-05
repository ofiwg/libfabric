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

#include <ifaddrs.h>
#include <rdma/fi_errno.h>
#include "../fi_verbs.h"
#include "verbs_utils.h"
#include "verbs_rdm.h"

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

void fi_ibv_rdm_conn_init_cm_role(struct fi_ibv_rdm_tagged_conn *conn,
				  struct fi_ibv_rdm_ep *ep)
{
	const int addr_cmp = memcmp(&conn->addr, &ep->cm.my_addr,
				    FI_IBV_RDM_DFLT_ADDRLEN);

	if (addr_cmp < 0) {
		conn->cm_role = FI_VERBS_CM_ACTIVE;
	} else if (addr_cmp > 0) {
		conn->cm_role = FI_VERBS_CM_PASSIVE;
	} else {
		conn->cm_role = FI_VERBS_CM_SELF;
	}
}

/* find the IPoIB address of the device opened in the fi_domain call. The name
 * of this device is _domain->verbs->device->name. The logic of the function is:
 * iterate through all the available network interfaces, find those having "ib"
 * in the name, then try to test the IB device that correspond to each address.
 * If the name is the desired one then we're done.
 */
int fi_ibv_rdm_tagged_find_ipoib_addr(const struct sockaddr_in *addr,
				      struct fi_ibv_rdm_cm* cm)
{
	struct ifaddrs *addrs, *tmp;
	struct sockaddr_in lh;
	int found = 0;

	inet_pton(AF_INET, "127.0.0.1", &lh.sin_addr);

	getifaddrs(&addrs);
	tmp = addrs;
	while (tmp) {
		if (tmp->ifa_addr && tmp->ifa_addr->sa_family == AF_INET) {
			struct sockaddr_in *paddr =
			    (struct sockaddr_in *) tmp->ifa_addr;
			if (!strncmp(tmp->ifa_name, "ib", 2)) {
				int ret = 0;

				if (addr && addr->sin_addr.s_addr) {
					ret = !memcmp(&addr->sin_addr,
						      &paddr->sin_addr,
						      sizeof(addr->sin_addr)) ||
					      !memcmp(&addr->sin_addr,
						      &lh.sin_addr,
						      sizeof(addr->sin_addr))
					      ? 1 : 0;
				}

				if (ret == 1) {
					memcpy(&(cm->my_addr), paddr,
					       sizeof(cm->my_addr));
					found = 1;
					break;
				}
			}
		}

		tmp = tmp->ifa_next;
	}

	freeifaddrs(addrs);

	if (found) {
		assert(cm->my_addr.sin_family == AF_INET);
	}
	return !found;
}
