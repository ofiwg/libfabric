
/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2018-2020 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>

#include <ofi_list.h>
#include <ofi.h>

#include "cxip.h"

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_CQ, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_CQ, __VA_ARGS__)

/*
 * cxip_cq_req_complete() - Generate a completion event for the request.
 */
int cxip_cq_req_complete(struct cxip_req *req)
{
	if (req->discard) {
		CXIP_DBG("Event discarded: %p\n", req);
		return FI_SUCCESS;
	}

	return ofi_cq_write(&req->cq->util_cq, (void *)req->context,
			    req->flags, req->data_len, (void *)req->buf,
			    req->data, req->tag);
}

/*
 * cxip_cq_req_complete() - Generate a completion event with source address for
 * the request.
 */
int cxip_cq_req_complete_addr(struct cxip_req *req, fi_addr_t src)
{
	if (req->discard) {
		CXIP_DBG("Event discarded: %p\n", req);
		return FI_SUCCESS;
	}

	return ofi_cq_write_src(&req->cq->util_cq, (void *)req->context,
				req->flags, req->data_len, (void *)req->buf,
				req->data, req->tag, src);
}

/*
 * cxip_cq_req_complete() - Generate an error event for the request.
 */
int cxip_cq_req_error(struct cxip_req *req, size_t olen,
		      int err, int prov_errno, void *err_data,
		      size_t err_data_size)
{
	struct fi_cq_err_entry err_entry;

	if (req->discard) {
		CXIP_DBG("Event discarded: %p\n", req);
		return FI_SUCCESS;
	}

	err_entry.err = err;
	err_entry.olen = olen;
	err_entry.err_data = err_data;
	err_entry.err_data_size = err_data_size;
	err_entry.len = req->data_len;
	err_entry.prov_errno = prov_errno;
	err_entry.flags = req->flags;
	err_entry.data = req->data;
	err_entry.tag = req->tag;
	err_entry.op_context = (void *)(uintptr_t)req->context;
	err_entry.buf = (void *)(uintptr_t)req->buf;

	return ofi_cq_write_error(&req->cq->util_cq, &err_entry);
}

/*
 * cxip_cq_progress() - Progress the CXI Completion Queue.
 *
 * Process events on the underlying Cassini event queue.
 */
void cxip_cq_progress(struct cxip_cq *cq)
{
	ofi_spin_lock(&cq->lock);

	if (!cq->enabled)
		goto out;

	cxip_cq_eq_progress(cq, &cq->eq);

out:
	ofi_spin_unlock(&cq->lock);
}

/*
 * cxip_util_cq_progress() - Progress function wrapper for utility CQ.
 */
void cxip_util_cq_progress(struct util_cq *util_cq)
{
	struct cxip_cq *cq = container_of(util_cq, struct cxip_cq, util_cq);

	cxip_cq_progress(cq);

	/* TODO support multiple EPs/CQ */
	if (cq->ep_obj)
		cxip_ep_ctrl_progress(cq->ep_obj);
}

/*
 * cxip_cq_strerror() - Converts provider specific error information into a
 * printable string.
 */
static const char *cxip_cq_strerror(struct fid_cq *cq, int prov_errno,
				    const void *err_data, char *buf,
				    size_t len)
{
	return cxi_rc_to_str(prov_errno);
}

/*
 * cxip_cq_trywait - Return success if able to block waiting for CQ events.
 */
static int cxip_cq_trywait(void *arg)
{
	struct cxip_cq *cq = (struct cxip_cq *)arg;

	assert(cq->util_cq.wait);

	if (!cq->priv_wait) {
		CXIP_WARN("No CXI wait object\n");
		return -FI_EINVAL;
	}

	if (cxi_eq_peek_event(cq->eq.eq))
		return -FI_EAGAIN;

	/* Clear wait, and check for any events */
	ofi_spin_lock(&cq->lock);
	cxil_clear_wait_obj(cq->priv_wait);

	if (cxi_eq_peek_event(cq->eq.eq)) {
		ofi_spin_unlock(&cq->lock);
		return -FI_EAGAIN;
	}
	ofi_spin_unlock(&cq->lock);

	return FI_SUCCESS;
}

/*
 * cxip_cq_close() - Destroy the Completion Queue object.
 */
static int cxip_cq_close(struct fid *fid)
{
	struct cxip_cq *cq;
	int ret;

	cq = container_of(fid, struct cxip_cq, util_cq.cq_fid.fid);
	if (ofi_atomic_get32(&cq->ref))
		return -FI_EBUSY;

	cxip_cq_disable(cq);

	if (cq->priv_wait) {
		ret = ofi_wait_del_fd(cq->util_cq.wait,
				      cxil_get_wait_obj_fd(cq->priv_wait));
		if (ret)
			CXIP_WARN("Wait FD delete error: %d\n", ret);

		ret = cxil_destroy_wait_obj(cq->priv_wait);
		if (ret)
			CXIP_WARN("Release CXI wait object failed: %d\n", ret);
	}

	ofi_cq_cleanup(&cq->util_cq);

	ofi_spin_destroy(&cq->lock);
	ofi_spin_destroy(&cq->ibuf_lock);
	ofi_spin_destroy(&cq->req_lock);

	cxip_domain_remove_cq(cq->domain, cq);

	free(cq);

	return 0;
}

static struct fi_ops cxip_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_cq_close,
	.bind = fi_no_bind,
	.control = ofi_cq_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_cq_attr cxip_cq_def_attr = {
	.flags = 0,
	.format = FI_CQ_FORMAT_CONTEXT,
	.wait_obj = FI_WAIT_NONE,
	.signaling_vector = 0,
	.wait_cond = FI_CQ_COND_NONE,
	.wait_set = NULL,
};

/*
 * cxip_cq_verify_attr() - Verify input Completion Queue attributes.
 */
static int cxip_cq_verify_attr(struct fi_cq_attr *attr)
{
	if (!attr)
		return FI_SUCCESS;

	switch (attr->format) {
	case FI_CQ_FORMAT_CONTEXT:
	case FI_CQ_FORMAT_MSG:
	case FI_CQ_FORMAT_DATA:
	case FI_CQ_FORMAT_TAGGED:
		break;
	case FI_CQ_FORMAT_UNSPEC:
		attr->format = cxip_cq_def_attr.format;
		break;
	default:
		CXIP_WARN("Unsupported CQ attribute format: %d\n",
			  attr->format);
		return -FI_ENOSYS;
	}

	/* Applications should set wait_obj == FI_WAIT_NONE for best
	 * performance. However, if a wait_obj is required and not
	 * specified, default to FI_WAIT_FD.
	 */
	switch (attr->wait_obj) {
	case FI_WAIT_UNSPEC:
		attr->wait_obj = FI_WAIT_FD;
		break;
	case FI_WAIT_NONE:
	case FI_WAIT_FD:
	case FI_WAIT_POLLFD:
		break;
	default:
		CXIP_WARN("Unsupported CQ wait object: %d\n",
			  attr->wait_obj);
		return -FI_ENOSYS;
	}

	/* Use environment variable to allow for dynamic setting of default CQ
	 * size.
	 */
	if (!attr->size)
		attr->size = cxip_env.default_cq_size;

	return FI_SUCCESS;
}

/*
 * cxip_cq_alloc_priv_wait - Allocate an internal wait channel for the CQ.
 */
static int cxip_cq_alloc_priv_wait(struct cxip_cq *cq)
{
	int ret;
	int wait_fd;

	assert(cq->domain);

	/* Not required or already created */
	if (!cq->util_cq.wait || cq->priv_wait)
		return FI_SUCCESS;

	ret = cxil_alloc_wait_obj(cq->domain->lni->lni, &cq->priv_wait);
	if (ret) {
		CXIP_WARN("Allocation of internal wait object failed %d\n",
			  ret);
		return ret;
	}

	wait_fd = cxil_get_wait_obj_fd(cq->priv_wait);
	ret = fi_fd_nonblock(wait_fd);
	if (ret) {
		CXIP_WARN("Unable to set CQ wait non-blocking mode: %d\n", ret);
		goto destroy_wait;
	}

	ret = ofi_wait_add_fd(cq->util_cq.wait, wait_fd, POLLIN,
			      cxip_cq_trywait, cq, &cq->util_cq.cq_fid.fid);
	if (ret) {
		CXIP_WARN("Add FD of internal wait object failed: %d\n", ret);
		goto destroy_wait;
	}

	CXIP_DBG("Add CQ private wait object, CQ intr FD: %d\n", wait_fd);

	return FI_SUCCESS;

destroy_wait:
	cxil_destroy_wait_obj(cq->priv_wait);
	cq->priv_wait = NULL;

	return ret;
}

/*
 * cxip_cq_open() - Allocate a new Completion Queue object.
 */
int cxip_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq, void *context)
{
	struct cxip_domain *cxi_dom;
	struct cxip_cq *cxi_cq;
	int ret;

	if (!domain || !cq)
		return -FI_EINVAL;

	cxi_dom = container_of(domain, struct cxip_domain,
			       util_domain.domain_fid);

	ret = cxip_cq_verify_attr(attr);
	if (ret != FI_SUCCESS)
		return ret;

	cxi_cq = calloc(1, sizeof(*cxi_cq));
	if (!cxi_cq)
		return -FI_ENOMEM;

	if (!attr) {
		cxi_cq->attr = cxip_cq_def_attr;
		cxi_cq->attr.size = cxip_env.default_cq_size;
	} else {
		cxi_cq->attr = *attr;
	}

	ret = ofi_cq_init(&cxip_prov, domain, &cxi_cq->attr, &cxi_cq->util_cq,
			  cxip_util_cq_progress, context);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("ofi_cq_init() failed: %d\n", ret);
		goto err_util_cq;
	}

	cxi_cq->util_cq.cq_fid.ops->strerror = &cxip_cq_strerror;
	cxi_cq->util_cq.cq_fid.fid.ops = &cxip_cq_fi_ops;

	cxi_cq->domain = cxi_dom;
	cxi_cq->ack_batch_size = cxip_env.eq_ack_batch_size;
	ofi_atomic_initialize32(&cxi_cq->ref, 0);
	ofi_spin_init(&cxi_cq->lock);
	ofi_spin_init(&cxi_cq->req_lock);
	ofi_spin_init(&cxi_cq->ibuf_lock);

	if (cxi_cq->util_cq.wait) {
		ret = cxip_cq_alloc_priv_wait(cxi_cq);
		if (ret != FI_SUCCESS) {
			CXIP_WARN("Unable to allocate CXI wait obj: %d\n",
				  ret);
			goto err_wait_alloc;
		}
	}

	cxip_domain_add_cq(cxi_dom, cxi_cq);
	*cq = &cxi_cq->util_cq.cq_fid;

	return FI_SUCCESS;

err_wait_alloc:
	ofi_cq_cleanup(&cxi_cq->util_cq);
err_util_cq:
	free(cxi_cq);

	return ret;
}
