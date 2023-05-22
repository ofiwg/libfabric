/*
 * Copyright (c) Intel Corporation. All rights reserved
 * Copyright (c) Amazon.com, Inc. or its affiliates. All rights reserved.
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
#include <sys/uio.h>

#include "ofi_atom.h"
#include "ofi_hmem.h"
#include "ofi_iov.h"
#include "ofi_mr.h"
#include "sm2.h"
#include "sm2_fifo.h"

static int sm2_ipc_dev_to_host_send_ipc_handle(
	struct sm2_ep *ep, struct sm2_xfer_entry *xfer_entry,
	struct fi_peer_rx_entry *rx_entry, void *host_ptr)
{
	sm2_gid_t sender_gid = xfer_entry->hdr.sender_gid;
	struct sm2_ipc_dev_to_host_data *dev_to_host_data;
	struct ipc_info *ipc_info;
	uint64_t device;
	void *device_ptr = rx_entry->iov[0].iov_base;
	void *base;
	int ret;

	if (cuda_is_addr_valid(device_ptr, &device, NULL)) {
		/* The destination address is a valid CUDA address */
		xfer_entry->hdr.proto = sm2_proto_ipc_dev_to_host;
		xfer_entry->hdr.sender_gid = ep->gid;

		dev_to_host_data = ((struct sm2_ipc_dev_to_host_data *)
					    xfer_entry->user_data);
		dev_to_host_data->host_ptr = host_ptr;
		dev_to_host_data->rx_context = rx_entry->context;
		dev_to_host_data->rx_flags = rx_entry->flags;

		ipc_info = &dev_to_host_data->ipc_info;
		ipc_info->iface = FI_HMEM_CUDA;
		ipc_info->device = device;

		/* Open IPC handle */
		ret = ofi_hmem_get_base_addr(ipc_info->iface, device_ptr, &base,
					     &ipc_info->base_length);
		if (ret)
			return ret;

		ret = ofi_hmem_get_handle(ipc_info->iface, base,
					  ipc_info->base_length,
					  (void **) &ipc_info->ipc_handle);
		if (ret)
			return ret;

		ipc_info->base_addr = (uintptr_t) base;
		ipc_info->offset = (uintptr_t) device_ptr -
				   (uintptr_t) ipc_info->base_addr;

		/* Send xfer_entry with IPC handle back to sender */
		sm2_fifo_write(ep, sender_gid, xfer_entry);

		return FI_SUCCESS;
	} else {
		assert(1 == 0);
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL, "CMA error %d\n", errno);
		return -FI_EIO;
	}
}

static int sm2_ipc_dev_to_host_cuda_memcpy(struct sm2_ep *ep,
					   struct sm2_xfer_entry *xfer_entry)
{
	struct sm2_domain *domain;
	struct sm2_ipc_dev_to_host_data *dev_to_host_data;
	struct ipc_info *ipc_info;
	struct ofi_mr_entry *mr_entry;
	void *dest;
	int ret;

	dev_to_host_data =
		(struct sm2_ipc_dev_to_host_data *) xfer_entry->user_data;
	ipc_info = &dev_to_host_data->ipc_info;

	domain = container_of(ep->util_ep.domain, struct sm2_domain,
			      util_domain);

	ret = ofi_ipc_cache_search(domain->ipc_cache,
				   xfer_entry->hdr.sender_gid, ipc_info,
				   &mr_entry);
	if (ret)
		return ret;

	dest = (char *) (uintptr_t) mr_entry->info.ipc_mapped_addr +
	       (uintptr_t) ipc_info->offset;

	ret = ofi_copy_to_hmem(ipc_info->iface, ipc_info->device, dest,
			       dev_to_host_data->host_ptr,
			       xfer_entry->hdr.size);

	ofi_mr_cache_delete(domain->ipc_cache, mr_entry);

	if (ret != 0) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"IPC device to host CUDA memcpy failed with code %d\n",
			ret);
		return ret;
	}

	return FI_SUCCESS;
}

static inline int sm2_cma_loop(struct sm2_ep *ep,
			       struct sm2_xfer_entry *xfer_entry,
			       struct fi_peer_rx_entry *rx_entry, size_t total,
			       bool write, bool *ipc_dev_to_host)
{
	int err;
	ssize_t ret;
	struct sm2_av *sm2_av =
		container_of(ep->util_ep.av, struct sm2_av, util_av);
	struct sm2_ep_allocation_entry *entries =
		sm2_mmap_entries(&sm2_av->mmap);
	struct sm2_cma_data *cma_data =
		(struct sm2_cma_data *) xfer_entry->user_data;

	pid_t pid = entries[xfer_entry->hdr.sender_gid].pid;

	while (1) {
		if (write)
			ret = ofi_process_vm_writev(
				pid, rx_entry->iov, rx_entry->count,
				cma_data->iov, cma_data->iov_count, 0);
		else
			ret = ofi_process_vm_readv(
				pid, rx_entry->iov, rx_entry->count,
				cma_data->iov, cma_data->iov_count, 0);
		if (ret < 0) {
			if (errno == 14) {
				/* The sender might be trying to send from host
				 * memory to device memory. CMA does not support
				 * device memory, so we open the IPC handle,
				 * return the handle to the sender for the
				 * sender to a cudaMeCpy */
				assert(rx_entry->count == 1 &&
				       cma_data->iov_count == 1);

				if (!(rx_entry->count == 1 &&
				      cma_data->iov_count == 1)) {
					FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
						"CMA error %d\n", errno);
					return -FI_EIO;
				}

				err = sm2_ipc_dev_to_host_send_ipc_handle(
					ep, xfer_entry, rx_entry,
					cma_data->iov[0].iov_base);
				if (err == FI_SUCCESS)
					*ipc_dev_to_host = true;
				return err;
			} else {
				FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
					"CMA error %d\n", errno);
				return -FI_EIO;
			}
		}

		total -= ret;
		if (!total)
			return FI_SUCCESS;

		ofi_consume_iov(rx_entry->iov, &rx_entry->count, (size_t) ret);
		ofi_consume_iov(cma_data->iov, &cma_data->iov_count,
				(size_t) ret);
	}
}

static int sm2_progress_cma(struct sm2_ep *ep,
			    struct sm2_xfer_entry *xfer_entry,
			    struct fi_peer_rx_entry *rx_entry,
			    size_t *total_len, int err, bool *ipc_dev_to_host)
{
	int ret;

	/* TODO Need to update last argument for RMA support (as well as generic
	 * format) */
	ret = sm2_cma_loop(ep, xfer_entry, rx_entry, xfer_entry->hdr.size,
			   false, ipc_dev_to_host);
	if (!ret)
		*total_len = xfer_entry->hdr.size;

	return -ret;
}

static int sm2_progress_inject(struct sm2_xfer_entry *xfer_entry,
			       struct ofi_mr **mr, struct iovec *iov,
			       size_t iov_count, size_t *total_len,
			       struct sm2_ep *ep, int err)
{
	ssize_t hmem_copy_ret;

	hmem_copy_ret =
		ofi_copy_to_mr_iov(mr, iov, iov_count, 0, xfer_entry->user_data,
				   xfer_entry->hdr.size);

	if (hmem_copy_ret < 0) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"Inject recv failed with code %d\n",
			(int) (-hmem_copy_ret));
		return hmem_copy_ret;
	} else if (hmem_copy_ret != xfer_entry->hdr.size) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL, "Inject recv truncated\n");
		return -FI_ETRUNC;
	}

	*total_len = hmem_copy_ret;

	return FI_SUCCESS;
}

static int sm2_progress_ipc(struct sm2_xfer_entry *xfer_entry,
			    struct ofi_mr **mr, struct iovec *iov,
			    size_t iov_count, size_t *total_len,
			    struct sm2_ep *ep)
{
	void *ptr;
	ssize_t hmem_copy_ret;
	int ret, err;
	struct sm2_domain *domain;
	struct ofi_mr_entry *mr_entry;
	struct ipc_info *ipc_info = (struct ipc_info *) xfer_entry->user_data;

	domain = container_of(ep->util_ep.domain, struct sm2_domain,
			      util_domain);

	ret = ofi_ipc_cache_search(
		domain->ipc_cache, xfer_entry->hdr.sender_gid,
		(struct ipc_info *) &xfer_entry->user_data, &mr_entry);

	ptr = (char *) (uintptr_t) mr_entry->info.ipc_mapped_addr +
	      (uintptr_t) ipc_info->offset;

	hmem_copy_ret =
		ofi_copy_to_hmem_iov(ipc_info->iface, ipc_info->device, iov,
				     iov_count, 0, ptr, xfer_entry->hdr.size);

	ofi_mr_cache_delete(domain->ipc_cache, mr_entry);

	if (hmem_copy_ret < 0) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"IPC recv failed with code %d\n",
			(int) (-hmem_copy_ret));
		err = hmem_copy_ret;
	} else if (hmem_copy_ret != xfer_entry->hdr.size) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL, "IPC recv truncated\n");
		err = -FI_ETRUNC;
	} else
		err = ret;

	*total_len = hmem_copy_ret;

	if (err)
		return err;
	else
		return FI_SUCCESS;
}

static int sm2_start_common(struct sm2_ep *ep,
			    struct sm2_xfer_entry *xfer_entry,
			    struct fi_peer_rx_entry *rx_entry,
			    bool return_xfer_entry)
{
	size_t total_len = 0;
	uint64_t comp_flags;
	void *comp_buf;
	int err = 0, ret = 0;
	bool ipc_dev_to_host = false;

	switch (xfer_entry->hdr.proto) {
	case sm2_proto_inject:
		err = sm2_progress_inject(
			xfer_entry, (struct ofi_mr **) rx_entry->desc,
			rx_entry->iov, rx_entry->count, &total_len, ep, err);
		break;
	case sm2_proto_ipc:
		err = sm2_progress_ipc(
			xfer_entry, (struct ofi_mr **) rx_entry->desc,
			rx_entry->iov, rx_entry->count, &total_len, ep);
		break;
	case sm2_proto_cma:
		err = sm2_progress_cma(ep, xfer_entry, rx_entry, &total_len, 0,
				       &ipc_dev_to_host);
		break;
	default:
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"Unidentified operation type\n");
		err = -FI_EINVAL;
	}

	comp_buf = rx_entry->iov[0].iov_base;
	comp_flags = sm2_rx_cq_flags(xfer_entry->hdr.op, rx_entry->flags,
				     xfer_entry->hdr.op_flags);

	if (err) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL, "Error processing op\n");
		ret = sm2_write_err_comp(ep->util_ep.rx_cq, rx_entry->context,
					 comp_flags, rx_entry->tag, err);
	} else {
		/* If the IPC device to host protocol is used, the receive
		 * completion is not generated at this stage. Instead, it's
		 * generated after the sender completes the CUDA memcpy */
		if (!ipc_dev_to_host)
			ret = sm2_complete_rx(
				ep, rx_entry->context, xfer_entry->hdr.op,
				comp_flags, total_len, comp_buf,
				xfer_entry->hdr.sender_gid, xfer_entry->hdr.tag,
				xfer_entry->hdr.cq_data);
	}
	if (ret) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"Unable to process rx completion\n");
	} else if (return_xfer_entry) {
		/* Return Free Queue Entries for all protocols except IPC device
		 * to host */
		if (!ipc_dev_to_host)
			sm2_fifo_write_back(ep, xfer_entry);
	}

	sm2_get_peer_srx(ep)->owner_ops->free_entry(rx_entry);

	return 0;
}

int sm2_unexp_start(struct fi_peer_rx_entry *rx_entry)
{
	struct sm2_xfer_ctx *xfer_ctx = rx_entry->peer_context;
	int ret;

	ret = sm2_start_common(xfer_ctx->ep, &xfer_ctx->xfer_entry, rx_entry,
			       false);
	ofi_buf_free(xfer_ctx);

	return ret;
}

static int sm2_alloc_xfer_entry_ctx(struct sm2_ep *ep,
				    struct fi_peer_rx_entry *rx_entry,
				    struct sm2_xfer_entry *xfer_entry)
{
	struct sm2_xfer_ctx *xfer_ctx;

	xfer_ctx = ofi_buf_alloc(ep->xfer_ctx_pool);
	if (!xfer_ctx) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"Error allocating xfer_entry ctx\n");
		return -FI_ENOMEM;
	}

	memcpy(&xfer_ctx->xfer_entry, xfer_entry, sizeof(*xfer_entry));
	xfer_ctx->ep = ep;

	rx_entry->peer_context = xfer_ctx;

	return FI_SUCCESS;
}

static int sm2_progress_recv_msg(struct sm2_ep *ep,
				 struct sm2_xfer_entry *xfer_entry)
{
	struct sm2_av *av =
		container_of(ep->util_ep.av, struct sm2_av, util_av);
	struct sm2_mmap *map = &av->mmap;
	struct fid_peer_srx *peer_srx = sm2_get_peer_srx(ep);
	struct fi_peer_rx_entry *rx_entry;
	struct sm2_av *sm2_av;
	fi_addr_t addr;
	int ret;

	/* sm2_proto_ipc_dev_to_host protocol means that the sender received an
	 * IPC handle from the receiver. So the sender needs to do a CUDA memcpy
	 * and doesn't need to look for an rx entry */
	if (xfer_entry->hdr.proto == sm2_proto_ipc_dev_to_host) {
		ret = sm2_ipc_dev_to_host_cuda_memcpy(ep, xfer_entry);
		if (ret) {
			FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
				"CUDA memcpy in device to host IPC protocol "
				"failed\n");
			smr_freestack_push(
				sm2_freestack(sm2_mmap_ep_region(map, ep->gid)),
				xfer_entry);
			goto out;
		}

		/* Generate send completion on sender side */
		ret = sm2_complete_tx(ep, (void *) xfer_entry->hdr.context,
				      xfer_entry->hdr.op,
				      xfer_entry->hdr.op_flags);

		/* Send a message to receiver with protocol
		 * sm2_proto_ipc_dev_to_host_ack so that receiver can
		 * generate a receive completion */
		sm2_fifo_write_back_ipc_dev_to_host(ep, xfer_entry);
		goto out;
	}

	sm2_av = container_of(ep->util_ep.av, struct sm2_av, util_av);
	addr = sm2_av->reverse_lookup[xfer_entry->hdr.sender_gid];

	if (xfer_entry->hdr.op == ofi_op_tagged) {
		ret = peer_srx->owner_ops->get_tag(
			peer_srx, addr, xfer_entry->hdr.size,
			xfer_entry->hdr.tag, &rx_entry);
		if (ret == -FI_ENOENT) {
			ret = sm2_alloc_xfer_entry_ctx(ep, rx_entry,
						       xfer_entry);
			sm2_fifo_write_back(ep, xfer_entry);
			if (ret)
				return ret;

			ret = peer_srx->owner_ops->queue_tag(rx_entry);
			goto out;
		}
	} else {
		ret = peer_srx->owner_ops->get_msg(
			peer_srx, addr, xfer_entry->hdr.size, &rx_entry);
		if (ret == -FI_ENOENT) {
			ret = sm2_alloc_xfer_entry_ctx(ep, rx_entry,
						       xfer_entry);
			sm2_fifo_write_back(ep, xfer_entry);
			if (ret)
				return ret;

			ret = peer_srx->owner_ops->queue_msg(rx_entry);
			goto out;
		}
	}
	if (ret) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL, "Error getting rx_entry\n");
		return ret;
	}
	ret = sm2_start_common(ep, xfer_entry, rx_entry, true);

out:
	return ret < 0 ? ret : 0;
}

void sm2_progress_recv(struct sm2_ep *ep)
{
	struct sm2_av *av =
		container_of(ep->util_ep.av, struct sm2_av, util_av);
	struct sm2_mmap *map = &av->mmap;
	struct sm2_xfer_entry *xfer_entry;
	struct sm2_ipc_dev_to_host_data *dev_to_host_data;
	uint64_t comp_flags;
	int ret = 0, i;

	for (i = 0; i < MAX_SM2_MSGS_PROGRESSED; i++) {
		xfer_entry = sm2_fifo_read(ep);
		if (!xfer_entry)
			break;

		if (xfer_entry->hdr.proto == sm2_proto_return) {
			if (xfer_entry->hdr.op_flags & FI_DELIVERY_COMPLETE) {
				ret = sm2_complete_tx(
					ep, (void *) xfer_entry->hdr.context,
					xfer_entry->hdr.op,
					xfer_entry->hdr.op_flags);
				if (ret)
					FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
						"Unable to process "
						"FI_DELIVERY_COMPLETE "
						"completion\n");
			}

			smr_freestack_push(
				sm2_freestack(sm2_mmap_ep_region(map, ep->gid)),
				xfer_entry);
			continue;
		} else if (xfer_entry->hdr.proto ==
			   sm2_proto_ipc_dev_to_host_ack) {
			/* Generate receive completion on the receiver because
			 * the receiver received a IPC device to host ack
			 * message */
			dev_to_host_data = (struct sm2_ipc_dev_to_host_data *)
						   xfer_entry->user_data;

			comp_flags = sm2_rx_cq_flags(xfer_entry->hdr.op,
						     dev_to_host_data->rx_flags,
						     xfer_entry->hdr.op_flags);

			ret = sm2_complete_rx(ep, dev_to_host_data->rx_context,
					      xfer_entry->hdr.op, comp_flags,
					      xfer_entry->hdr.size,
					      dev_to_host_data->host_ptr,
					      xfer_entry->hdr.sender_gid,
					      xfer_entry->hdr.tag,
					      xfer_entry->hdr.cq_data);

			if (ret) {
				FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
					"Unable to process rx completion\n");
			}

			sm2_fifo_write_back(ep, xfer_entry);
			continue;
		}

		switch (xfer_entry->hdr.op) {
		case ofi_op_msg:
		case ofi_op_tagged:
			ret = sm2_progress_recv_msg(ep, xfer_entry);
			break;
		default:
			FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
				"Unidentified operation type\n");
			ret = -FI_EINVAL;
		}
		if (ret) {
			if (ret != -FI_EAGAIN) {
				FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
					"Error processing command\n");
			}
			break;
		}
	}
}

void sm2_ep_progress(struct util_ep *util_ep)
{
	struct sm2_ep *ep;

	ep = container_of(util_ep, struct sm2_ep, util_ep);
	sm2_progress_recv(ep);
}
