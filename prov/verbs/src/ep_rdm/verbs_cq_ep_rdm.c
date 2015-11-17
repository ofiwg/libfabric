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

#include <stdlib.h>

#include <fi_enosys.h>
#include <fi_list.h>
#include "../fi_verbs.h"
#include "verbs_queuing.h"


extern struct fi_ibv_mem_pool fi_ibv_rdm_tagged_request_pool;


static ssize_t fi_ibv_rdm_tagged_cq_readfrom(struct fid_cq *cq, void *buf,
                                             size_t count, fi_addr_t * src_addr)
{
	struct fi_ibv_cq *_cq = container_of(cq, struct fi_ibv_cq, cq_fid);
	struct fi_cq_tagged_entry *entry = buf;
	size_t i;

	for (i = 0;
	     i < count && !dlist_empty(&fi_ibv_rdm_tagged_request_ready_queue);
	     ++i)
	{
		struct fi_ibv_rdm_tagged_request *completed_req =
			container_of(fi_ibv_rdm_tagged_request_ready_queue.next,
				     struct fi_ibv_rdm_tagged_request, queue_entry);

		fi_ibv_rdm_tagged_remove_from_ready_queue(completed_req);

		FI_DBG(&fi_ibv_prov, FI_LOG_CQ,
		       "\t\t-> found match in ready: op_ctx %p, len %d, tag 0x%llx\n",
		       completed_req->context, completed_req->len,
		       (long long unsigned int) completed_req->tag);

		src_addr[i] = (fi_addr_t) (uintptr_t) completed_req->conn;
		entry[i].op_context = completed_req->context;
		entry[i].flags = 0;
		entry[i].len = completed_req->len;
		entry[i].data = completed_req->imm;
		entry[i].tag = completed_req->tag;

		if (completed_req->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE) {
			FI_IBV_RDM_TAGGED_DBG_REQUEST("to_pool: ", completed_req,
						      FI_LOG_DEBUG);
			fi_ibv_mem_pool_return(&completed_req->mpe,
						&fi_ibv_rdm_tagged_request_pool);
		} else {
			completed_req->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;
		}
	}

	if (i == 0) {
		int err = fi_ibv_rdm_tagged_poll(_cq->ep);
		if (err < 0) {
			VERBS_INFO(FI_LOG_CQ, "fi_ibv_rdm_tagged_poll failed\n");
		}
	}

	return i;
}

static ssize_t fi_ibv_rdm_tagged_cq_read(struct fid_cq *cq, void *buf,
                                         size_t count)
{
	fi_addr_t src_addr;
	return fi_ibv_rdm_tagged_cq_readfrom(cq, buf, count, &src_addr);
}

#if 0
static const char *fi_ibv_cq_strerror(struct fid_cq *eq, int prov_errno,
                                      const void *err_data, char *buf,
                                      size_t len)
{
	if (buf && len)
		strncpy(buf, ibv_wc_status_str(prov_errno), len);
	return ibv_wc_status_str(prov_errno);
}
#endif                          /* 0 */

static ssize_t
fi_ibv_rdm_tagged_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *entry,
                             uint64_t flags)
{
#if 0
	struct fi_ibv_cq *_cq;
	_cq = container_of(cq, struct fi_ibv_cq, cq_fid);
#endif                          /* 0 */
	/* TODO */
	return sizeof(*entry);
}

static struct fi_ops_cq fi_ibv_rdm_tagged_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = fi_ibv_rdm_tagged_cq_read,
	.readfrom = fi_ibv_rdm_tagged_cq_readfrom,
	.readerr = fi_ibv_rdm_tagged_cq_readerr,
	.sread = fi_no_cq_sread,
	.strerror = fi_cq_strerror
};

#if 0
static int fi_ibv_cq_close(fid_t fid)
{
	struct fi_ibv_cq *cq;
	int ret;

	cq = container_of(fid, struct fi_ibv_cq, cq_fid.fid);
	assert(cq->ep->type == FI_EP_RDM);
	while (cq->ep->ep_rdm.conn_active) {
		if (0 != (ret = fi_ibv_rdm_tagged_cm_progress(cq->ep)))
			return ret;

	}

	while (cq->ep->ep_rdm.pend_send || cq->ep->ep_rdm.pend_recv)
		fi_ibv_rdm_tagged_poll(cq->ep);

	if (cq->cq) {
		ret = ibv_destroy_cq(cq->cq);
		if (ret) {
			FI_IBV_ERROR("ibv_destroy_cq returned: "
				     "ret %d, errno %d, errstr %s", ret,
				     errno, strerror(errno));
			return -ret;
		}
	}

	free(cq);
	return 0;
}
#endif

struct fi_ops_cq *fi_ibv_cq_ops_tagged(struct fi_ibv_cq *cq)
{
	return &fi_ibv_rdm_tagged_cq_ops;
}
