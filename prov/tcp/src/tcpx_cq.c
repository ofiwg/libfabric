/*
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
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
#include <string.h>

#include "tcpx.h"

#define TCPX_DEF_CQ_SIZE (1024)

static int tcpx_cq_close(struct fid *fid)
{
	int ret;
	struct tcpx_cq *tcpx_cq;

	tcpx_cq = container_of(fid, struct tcpx_cq, util_cq.cq_fid.fid);
	util_buf_pool_destroy(tcpx_cq->pe_entry_pool);
	ret = ofi_cq_cleanup(&tcpx_cq->util_cq);
	if (ret)
		return ret;

	free(tcpx_cq);
	return 0;
}

struct tcpx_pe_entry *tcpx_pe_entry_alloc(struct tcpx_cq *tcpx_cq)
{
	struct tcpx_pe_entry *pe_entry;

	fastlock_acquire(&tcpx_cq->util_cq.cq_lock);

	/* optimization: don't allocate queue_entry when cq is full */
	if (ofi_cirque_isfull(tcpx_cq->util_cq.cirq)) {
		fastlock_release(&tcpx_cq->util_cq.cq_lock);
		return NULL;
	}

	pe_entry = util_buf_alloc(tcpx_cq->pe_entry_pool);
	if (!pe_entry) {
		fastlock_release(&tcpx_cq->util_cq.cq_lock);
		FI_INFO(&tcpx_prov, FI_LOG_DOMAIN,"failed to get buffer\n");
		return NULL;
	}
	fastlock_release(&tcpx_cq->util_cq.cq_lock);
	memset(pe_entry, 0, sizeof(*pe_entry));
	return pe_entry;
}

void tcpx_pe_entry_release(struct tcpx_pe_entry *pe_entry)
{
	struct util_cq *cq;
	struct tcpx_cq *tcpx_cq;

	switch (pe_entry->msg_hdr.hdr.op_data) {
	case TCPX_OP_MSG_SEND:
	case TCPX_OP_WRITE:
		cq = pe_entry->ep->util_ep.tx_cq;
		break;
	case TCPX_OP_MSG_RECV:
	case TCPX_OP_REMOTE_WRITE:
		cq = pe_entry->ep->util_ep.rx_cq;
		break;
	default:
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,"invalid pe_entry\n");
		return;
	}

	tcpx_cq = container_of(cq, struct tcpx_cq, util_cq);

	if (pe_entry->ep->cur_rx_entry == pe_entry)
		pe_entry->ep->cur_rx_entry = NULL;

	fastlock_acquire(&cq->cq_lock);
	util_buf_release(tcpx_cq->pe_entry_pool, pe_entry);
	fastlock_release(&cq->cq_lock);
}

void tcpx_cq_report_completion(struct util_cq *cq,
			       struct tcpx_pe_entry *pe_entry,
			       int err)
{
	struct fi_cq_err_entry err_entry;

	if (pe_entry->flags & TCPX_NO_COMPLETION)
		return;

	if (err) {
		err_entry.op_context = pe_entry->context;
		err_entry.flags = pe_entry->flags;
		err_entry.len = 0;
		err_entry.buf = NULL;
		err_entry.data = pe_entry->msg_hdr.hdr.data;
		err_entry.tag = 0;
		err_entry.olen = 0;
		err_entry.err = err;
		err_entry.prov_errno = errno;
		err_entry.err_data = NULL;
		err_entry.err_data_size = 0;

		ofi_cq_write_error(cq, &err_entry);
	} else {
		ofi_cq_write(cq, pe_entry->context,
			      pe_entry->flags, 0, NULL,
			      pe_entry->msg_hdr.hdr.data, 0);

		if (cq->wait)
			ofi_cq_signal(&cq->cq_fid);
	}
}

static int tcpx_cq_control(struct fid *fid, int command, void *arg)
{
	struct util_cq *cq;
	int ret;

	cq = container_of(fid, struct util_cq, cq_fid.fid);

	switch(command) {
	case FI_GETWAIT:
		if (!cq->wait)
			return -FI_ENOSYS;

		ret = fi_control(&cq->wait->wait_fid.fid,
				 command, arg);
		if (ret)
			return ret;

		return FI_SUCCESS;
	default:
		return -FI_ENOSYS;
	}
}

static struct fi_ops tcpx_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = tcpx_cq_close,
	.bind = fi_no_bind,
	.control = tcpx_cq_control,
	.ops_open = fi_no_ops_open,
};

int tcpx_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq_fid, void *context)
{
	int ret;
	struct tcpx_cq *tcpx_cq;

	tcpx_cq = calloc(1, sizeof(*tcpx_cq));
	if (!tcpx_cq)
		return -FI_ENOMEM;

	if (!attr->size)
		attr->size = TCPX_DEF_CQ_SIZE;

	ret =  util_buf_pool_create(&tcpx_cq->pe_entry_pool,
				    sizeof(struct tcpx_pe_entry),
				    16, 0, 1024);
	if (ret)
		goto free_cq;

	ret = ofi_cq_init(&tcpx_prov, domain, attr, &tcpx_cq->util_cq,
			   &ofi_cq_progress, context);
	if (ret)
		goto destroy_pool;

	*cq_fid = &tcpx_cq->util_cq.cq_fid;
	(*cq_fid)->fid.ops = &tcpx_cq_fi_ops;
	return 0;

destroy_pool:
	util_buf_pool_destroy(tcpx_cq->pe_entry_pool);
free_cq:
	free(tcpx_cq);
	return ret;
}
