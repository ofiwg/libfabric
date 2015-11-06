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


#include <prov/verbs/src/fi_verbs.h>
#include <prov/verbs/src/verbs_utils.h>

#include <alloca.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <infiniband/ib.h>
#include <rdma/rdma_cma.h>

#include <rdma/fi_errno.h>

int fi_ibv_sockaddr_len(struct sockaddr *addr)
{
	if (!addr)
		return 0;

	switch (addr->sa_family) {
	case AF_INET:
		return sizeof(struct sockaddr_in);
	case AF_INET6:
		return sizeof(struct sockaddr_in6);
	case AF_IB:
		return sizeof(struct sockaddr_ib);
	default:
		return 0;
	}
}

int fi_ibv_copy_addr(void *dst_addr, size_t *dst_addrlen, void *src_addr)
{
	size_t src_addrlen = fi_ibv_sockaddr_len(src_addr);

	if (*dst_addrlen == 0) {
		*dst_addrlen = src_addrlen;
		return -FI_ETOOSMALL;
	}

	if (*dst_addrlen < src_addrlen) {
		memcpy(dst_addr, src_addr, *dst_addrlen);
	} else {
		memcpy(dst_addr, src_addr, src_addrlen);
	}
	*dst_addrlen = src_addrlen;
	return 0;
}

ssize_t fi_ibv_send(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr, size_t len,
		int count, void *context)
{
	struct ibv_send_wr *bad_wr;
	int ret;

	wr->num_sge = count;
	wr->wr_id = (uintptr_t) context;

	ret = ibv_post_send(ep->id->qp, wr, &bad_wr);
	switch (ret) {
	case ENOMEM:
		return -FI_EAGAIN;
	case -1:
		/* Deal with non-compliant libibverbs drivers which set errno
		 * instead of directly returning the error value */
		return (errno == ENOMEM) ? -FI_EAGAIN : -errno;
	default:
		return -ret;
	}
}

ssize_t
fi_ibv_send_buf_inline(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
		const void *buf, size_t len)
{
	struct ibv_sge sge;

	fi_ibv_set_sge_inline(sge, buf, len);
	wr->sg_list = &sge;

	return fi_ibv_send(ep, wr, len, 1, NULL);
}

ssize_t fi_ibv_send_buf(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
		const void *buf, size_t len, void *desc, void *context)
{
	struct ibv_sge sge;

	fi_ibv_set_sge(sge, buf, len, desc);
	wr->sg_list = &sge;

	return fi_ibv_send(ep, wr, len, 1, context);
}

ssize_t fi_ibv_send_iov_flags(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
		const struct iovec *iov, void **desc, int count, void *context,
		uint64_t flags)
{
	size_t len = 0;

	if (!desc)
		fi_ibv_set_sge_iov_inline(wr->sg_list, iov, count, len);
	else
		fi_ibv_set_sge_iov(wr->sg_list, iov, count, desc, len);

	wr->send_flags
        = VERBS_INJECT_FLAGS(ep, len, flags) | VERBS_COMP_FLAGS(ep, flags);

	return fi_ibv_send(ep, wr, len, count, context);
}

/* dlist comparators, operators, etc
 */

int fi_ibv_rdm_tagged_match_requests(struct dlist_entry *item,
                                     const void *other)
{
    struct fi_ibv_rdm_tagged_request *req =
        (struct fi_ibv_rdm_tagged_request *)other;
    return (item == &req->queue_entry);
}

int fi_verbs_rdm_tagged_match_request_by_minfo(struct dlist_entry *item,
                                               const void *other)
{
    struct fi_ibv_rdm_tagged_request *request =
        container_of(item, struct fi_ibv_rdm_tagged_request, queue_entry);

    fi_verbs_rdm_tagged_request_minfo_t *minfo =
        (fi_verbs_rdm_tagged_request_minfo_t *)other;

    return (((request->conn == NULL) || (request->conn == minfo->conn)) &&
            ((request->tag & request->tagmask) ==
             (minfo->tag   & request->tagmask)));
}

int fi_verbs_rdm_tagged_match_request_by_minfo_with_tagmask
    (struct dlist_entry *item, const void *other)
{
    struct fi_ibv_rdm_tagged_request *request =
        container_of(item, struct fi_ibv_rdm_tagged_request, queue_entry);

    fi_verbs_rdm_tagged_request_minfo_t *minfo =
        (fi_verbs_rdm_tagged_request_minfo_t *)other;

    return (((minfo->conn == NULL) || (request->conn == minfo->conn)) &&
            ((request->tag & minfo->tagmask) ==
            (minfo->tag   & minfo->tagmask)));
}

void fi_ibv_rdm_tagged_send_postponed_process(struct dlist_entry *item,
                                              const void *arg)
{
    fi_ibv_rdm_tagged_send_ready_data_t* send_data =
        (fi_ibv_rdm_tagged_send_ready_data_t*)arg;

    struct fi_ibv_rdm_tagged_postponed_entry *postponed_entry =
        container_of(item, struct fi_ibv_rdm_tagged_postponed_entry,
                     queue_entry);

    if (!dlist_empty(&postponed_entry->conn->postponed_requests_head)) {
        struct fi_ibv_rdm_tagged_request *request =
            container_of(postponed_entry->conn->postponed_requests_head.next,
                         struct fi_ibv_rdm_tagged_request,
                         queue_entry);

        int res = fi_ibv_rdm_tagged_prepare_send_resources(request,
                                                           send_data->ep);
        if (res) {
            fi_ibv_rdm_tagged_req_hndl(request, FI_IBV_EVENT_SEND_READY,
                                       send_data);
        }
    }
}
