/*
 * Copyright (C) 2025-2026 Cornelis Networks.
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
#include "rdma/opx/fi_opx_domain.h"
#include "rdma/opx/opx_hfisvc.h"

#if HAVE_HFISVC
void opx_hfisvc_mr_open(struct fi_opx_domain *opx_domain, struct fi_opx_mr *opx_mr)
{
	assert(opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_NOT_REGISTERED);

	struct hfisvc_client_completion completion = {
		.flags		= HFISVC_CLIENT_COMPLETION_FLAG_CQ,
		.cq.handle	= opx_domain->hfisvc.mr_completion_queue,
		.cq.app_context = (uint64_t) opx_mr,
	};

	void			 *base;
	size_t			  len;
	struct hfisvc_client_hmem hmem;

	if (opx_mr->dmabuf.fd != -1) {
#if HAVE_HFISVC_DMABUF
		hmem.iface = HFISVC_CLIENT_HMEM_IFACE_DMABUF;
#endif
		hmem.device.reserved = (int64_t) opx_mr->dmabuf.fd;
		base		     = opx_mr->dmabuf.base_addr;
		len		     = opx_mr->dmabuf.len + opx_mr->dmabuf.offset;
	} else {
		switch (opx_mr->attr.iface) {
		case FI_HMEM_CUDA:
			hmem.iface	 = HFISVC_CLIENT_HMEM_IFACE_CUDA;
			hmem.device.cuda = opx_mr->attr.device.cuda;
			break;
		case FI_HMEM_ROCR:
			hmem.iface	     = HFISVC_CLIENT_HMEM_IFACE_ROCR;
			hmem.device.reserved = opx_mr->attr.device.reserved;
			break;
		case FI_HMEM_ZE:
			hmem.iface     = HFISVC_CLIENT_HMEM_IFACE_ZE;
			hmem.device.ze = opx_mr->attr.device.ze;
			break;
		default:
			hmem.iface	     = HFISVC_CLIENT_HMEM_IFACE_SYSTEM;
			hmem.device.reserved = 0UL;
		};
		base = opx_mr->iov.iov_base;
		len  = opx_mr->iov.iov_len;
	}

	OPX_HFISVC_DEBUG_LOG(
		"Opening MR opx_mr=%p hmem.iface-%d buf=%p-%p (%lu bytes offset=%lu) attr.iface=%d fd=%d\n", opx_mr,
		hmem.iface, base, (void *) ((uintptr_t) base + len), len,
		(opx_mr->dmabuf.fd == -1) ? 0 : opx_mr->dmabuf.offset, opx_mr->attr.iface, opx_mr->dmabuf.fd);

	int ret = (opx_domain->hfisvc.cmd_mr_open)(opx_domain->hfisvc.mr_command_queue, completion, 0UL /* flags */,
						   len, base, hmem);
	if (ret) {
		FI_WARN(fi_opx_global.prov, FI_LOG_MR, "Error opening opx_mr=%p buf=%p-%p iface=%d\n", opx_mr, base,
			(void *) ((uintptr_t) base + len), opx_mr->attr.iface);
	} else {
		(*opx_domain->hfisvc.doorbell)(opx_domain->hfisvc.ctxs[0].ctx);
		OPX_HFISVC_DEBUG_LOG("MR State transition opx_mr=%p state=NOT_REGISTERED -> PENDING_OPEN\n", opx_mr);
		opx_mr->hfisvc.state = OPX_MR_HFISVC_STATE_PENDING_OPEN;
	}
}

int opx_hfisvc_mr_enable_access_key(struct fi_opx_domain *opx_domain, struct fi_opx_mr *opx_mr)
{
	assert(opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_PENDING_KEY_ALLOC);

	uint32_t access_key;
	// TODO: Figure out how to pass in counters
	int ret = opx_hfisvc_keyset_alloc_key(&opx_domain->hfisvc.ctxs[0].access_key_set, &access_key, NULL);
	if (ret) {
		OPX_HFISVC_DEBUG_LOG("Unable to allocate access_key for mr=%p buf=%p-%p (EAGAIN)\n", opx_mr,

				     (opx_mr->dmabuf.fd == -1) ? opx_mr->iov.iov_base : opx_mr->dmabuf.base_addr,
				     (opx_mr->dmabuf.fd == -1) ?
					     (void *) ((uintptr_t) opx_mr->iov.iov_base + opx_mr->iov.iov_len) :
					     (void *) ((uintptr_t) opx_mr->dmabuf.base_addr + opx_mr->dmabuf.len +
						       opx_mr->dmabuf.offset));
		return -FI_EAGAIN;
	}

	struct hfisvc_client_completion enable_completion = {
		.flags		= HFISVC_CLIENT_COMPLETION_FLAG_CQ,
		.cq.handle	= opx_domain->hfisvc.mr_completion_queue,
		.cq.app_context = (uint64_t) opx_mr,
	};

	struct hfisvc_client_completion mr_notification = {
		.flags		= HFISVC_CLIENT_COMPLETION_FLAG_CQ,
		.cq.handle	= opx_domain->hfisvc.mr_completion_queue,
		.cq.app_context = (uint64_t) opx_mr,
	};

	ret = (opx_domain->hfisvc.cmd_dma_access_enable)(
		opx_domain->hfisvc.mr_command_queue, enable_completion, 0UL /* flags */, access_key,
		(opx_mr->dmabuf.fd == -1) ? opx_mr->iov.iov_len : (opx_mr->dmabuf.len + opx_mr->dmabuf.offset),
		opx_mr->hfisvc.mr_handle, 0UL, mr_notification);

	if (ret) {
		OPX_HFISVC_DEBUG_LOG(
			"Error enabling access_key=%u for mr=%p buf=%p-%p (returned %d) (EAGAIN)\n", access_key, opx_mr,
			(opx_mr->dmabuf.fd == -1) ? opx_mr->iov.iov_base : opx_mr->dmabuf.base_addr,
			(opx_mr->dmabuf.fd == -1) ? (void *) ((uintptr_t) opx_mr->iov.iov_base + opx_mr->iov.iov_len) :
						    (void *) ((uintptr_t) opx_mr->dmabuf.base_addr +
							      opx_mr->dmabuf.len + opx_mr->dmabuf.offset),
			ret);
		opx_hfisvc_keyset_free_key(opx_domain->hfisvc.ctxs[0].access_key_set, access_key,
					   NULL); // TODO: Debug counters parm
		return -FI_EAGAIN;
	}
	(*opx_domain->hfisvc.doorbell)(opx_domain->hfisvc.ctxs[0].ctx);
	opx_mr->hfisvc.access_key = access_key;

	OPX_HFISVC_DEBUG_LOG("MR State transition opx_mr=%p key=%u state=PENDING_KEY_ALLOC -> PENDING_KEY_ENABLE\n",
			     opx_mr, opx_mr->hfisvc.access_key);

	opx_mr->hfisvc.state = OPX_MR_HFISVC_STATE_PENDING_KEY_ENABLE;

	return 0;
}

void opx_hfisvc_mr_disable_access_key(struct fi_opx_domain *opx_domain, struct fi_opx_mr *opx_mr)
{
	assert(opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_OPENED);

	struct hfisvc_client_completion disable_completion = {
		.flags		= HFISVC_CLIENT_COMPLETION_FLAG_CQ,
		.cq.handle	= opx_domain->hfisvc.mr_completion_queue,
		.cq.app_context = (uint64_t) opx_mr,
	};

	int ret = (opx_domain->hfisvc.cmd_dma_access_disable)(opx_domain->hfisvc.mr_command_queue, disable_completion,
							      0UL /* flags */, opx_mr->hfisvc.access_key);
	if (ret) {
		OPX_HFISVC_DEBUG_LOG("Error disabling access_key=%u for mr=%p buf=%p-%p (returned %d) (EAGAIN)\n",
				     opx_mr->hfisvc.access_key, opx_mr,
				     (opx_mr->dmabuf.fd == -1) ? opx_mr->iov.iov_base : opx_mr->dmabuf.base_addr,
				     (opx_mr->dmabuf.fd == -1) ?
					     (void *) ((uintptr_t) opx_mr->iov.iov_base + opx_mr->iov.iov_len) :
					     (void *) ((uintptr_t) opx_mr->dmabuf.base_addr + opx_mr->dmabuf.len +
						       opx_mr->dmabuf.offset),
				     ret);
	} else {
		(*opx_domain->hfisvc.doorbell)(opx_domain->hfisvc.ctxs[0].ctx);
		OPX_HFISVC_DEBUG_LOG("MR State transition opx_mr=%p state=OPEN -> PENDING_KEY_DISABLE\n", opx_mr);
		opx_mr->hfisvc.state = OPX_MR_HFISVC_STATE_PENDING_KEY_DISABLE;
	}
}

void opx_hfisvc_mr_deregister_mr(struct fi_opx_domain *opx_domain, struct fi_opx_mr *opx_mr)
{
	assert(opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_PENDING_DEREGISTER);

	struct hfisvc_client_completion completion = {
		.flags		= HFISVC_CLIENT_COMPLETION_FLAG_CQ,
		.cq.handle	= opx_domain->hfisvc.mr_completion_queue,
		.cq.app_context = (uint64_t) opx_mr,
	};

	OPX_HFISVC_DEBUG_LOG(
		"Closing MR opx_mr=%p buf=%p-%p\n", opx_mr,
		(opx_mr->dmabuf.fd == -1) ? opx_mr->iov.iov_base : opx_mr->dmabuf.base_addr,
		(opx_mr->dmabuf.fd == -1) ?
			(void *) ((uintptr_t) opx_mr->iov.iov_base + opx_mr->iov.iov_len) :
			(void *) ((uintptr_t) opx_mr->dmabuf.base_addr + opx_mr->dmabuf.len + opx_mr->dmabuf.offset));
	int ret = (*opx_domain->hfisvc.cmd_mr_close)(opx_domain->hfisvc.mr_command_queue, completion, 0UL /* flags */,
						     opx_mr->hfisvc.mr_handle);
	if (ret) {
		FI_WARN(fi_opx_global.prov, FI_LOG_MR, "Error closing opx_mr=%p buf=%p-%p\n", opx_mr,
			(opx_mr->dmabuf.fd == -1) ? opx_mr->iov.iov_base : opx_mr->dmabuf.base_addr,
			(opx_mr->dmabuf.fd == -1) ? (void *) ((uintptr_t) opx_mr->iov.iov_base + opx_mr->iov.iov_len) :
						    (void *) ((uintptr_t) opx_mr->dmabuf.base_addr +
							      opx_mr->dmabuf.len + opx_mr->dmabuf.offset));
	} else {
		(*opx_domain->hfisvc.doorbell)(opx_domain->hfisvc.ctxs[0].ctx);
		OPX_HFISVC_DEBUG_LOG("MR State transition opx_mr=%p state=PENDING_DEREGISTER -> PENDING_CLOSE\n",
				     opx_mr);
		opx_mr->hfisvc.state = OPX_MR_HFISVC_STATE_PENDING_CLOSE;
	}
}
#endif

int opx_hfisvc_mr_deferred_open(struct opx_domain_deferred_work *work)
{
#if HAVE_HFISVC
	struct fi_opx_mr *opx_mr = work->opx_mr;

	if (opx_mr->hfisvc.state >= OPX_MR_HFISVC_STATE_OPENED) {
		work->opx_mr = NULL;
		return FI_SUCCESS;
	}

	if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_NOT_REGISTERED) {
		opx_hfisvc_mr_open(opx_mr->domain, opx_mr);
	} else if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_PENDING_KEY_ALLOC) {
		opx_hfisvc_mr_enable_access_key(opx_mr->domain, opx_mr);
	}

#endif
	return -FI_EAGAIN;
}

int opx_hfisvc_mr_deferred_close(struct opx_domain_deferred_work *work)
{
#if HAVE_HFISVC
	struct fi_opx_mr *opx_mr = work->opx_mr;

	if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_CLOSED) {
		free(opx_mr);
		work->opx_mr = NULL;
		return FI_SUCCESS;
	}

	if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_OPENED) {
		opx_hfisvc_mr_disable_access_key(opx_mr->domain, opx_mr);
	} else if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_PENDING_DEREGISTER) {
		opx_hfisvc_mr_deregister_mr(opx_mr->domain, opx_mr);
	}

#endif
	return -FI_EAGAIN;
}
