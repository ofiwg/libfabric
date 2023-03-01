/*
 * Copyright (c) 2013-2020 Intel Corporation. All rights reserved
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

#include "ofi_iov.h"
#include "ofi_hmem.h"
#include "ofi_atom.h"
#include "ofi_mr.h"
#include "sm2.h"
#include "sm2_fifo.h"

static int sm2_progress_inject(struct sm2_free_queue_entry *fqe, enum fi_hmem_iface iface,
			       uint64_t device, struct iovec *iov,
			       size_t iov_count, size_t *total_len,
			       struct sm2_ep *ep, int err)
{
	ssize_t hmem_copy_ret;


	hmem_copy_ret = ofi_copy_to_hmem_iov(iface, device, iov,
						     iov_count, 0, fqe->data,
						     fqe->protocol_hdr.size);

	if (hmem_copy_ret < 0) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"inject recv failed with code %d\n",
			(int)(-hmem_copy_ret));
		return hmem_copy_ret;
	} else if (hmem_copy_ret != fqe->protocol_hdr.size) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"inject recv truncated\n");
		return -FI_ETRUNC;
	}

	*total_len = hmem_copy_ret;

	return FI_SUCCESS;
}

static int sm2_start_common(struct sm2_ep *ep, struct sm2_free_queue_entry *fqe,
		struct fi_peer_rx_entry *rx_entry)
{
	struct sm2_sar_entry *sar = NULL;
	size_t total_len = 0;
	uint64_t comp_flags;
	void *comp_buf;
	int ret;
	uint64_t err = 0;

	switch (fqe->protocol_hdr.op_src) {
	case sm2_src_inject:
		err = sm2_progress_inject(fqe, 0, 0,
					  rx_entry->iov, rx_entry->count,
					  &total_len, ep, 0);
		break;
	case sm2_buffer_return:
		// TODO This is currently not being used b/c of hack
		smr_freestack_push(sm2_free_stack(ep->region), fqe);
		break;
	default:
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"unidentified operation type\n");
		err = -FI_EINVAL;
	}

	comp_buf = rx_entry->iov[0].iov_base;
	comp_flags = sm2_rx_cq_flags(fqe->protocol_hdr.op, rx_entry->flags,
				     fqe->protocol_hdr.op_flags);
	if (!sar) {
		if (err) {
			FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
				"error processing op\n");
			ret = sm2_write_err_comp(ep->util_ep.rx_cq,
						 rx_entry->context,
						 comp_flags, rx_entry->tag,
						 err);
		} else {
			ret = sm2_complete_rx(ep, rx_entry->context, fqe->protocol_hdr.op,
					      comp_flags, total_len, comp_buf,
					      fqe->protocol_hdr.id, fqe->protocol_hdr.tag,
					      fqe->protocol_hdr.data);
		}
		if (ret) {
			FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
				"unable to process rx completion\n");
		} else {
			/* Return Free Queue Entries here */
			// TODO Shouldn't need this hack... should just be able to write FQE back with FQE
			struct sm2_region *owning_region = sm2_peer_region(ep->region, 0);
			sm2_fifo_write_back(fqe, owning_region);
		}

		sm2_get_peer_srx(ep)->owner_ops->free_entry(rx_entry);
	}

	return 0;
}

int sm2_unexp_start(struct fi_peer_rx_entry *rx_entry)
{
	struct sm2_cmd_ctx *cmd_ctx = rx_entry->peer_context;
	int ret;

	ret = sm2_start_common(cmd_ctx->ep, &cmd_ctx->cmd, rx_entry);
	ofi_freestack_push(cmd_ctx->ep->cmd_ctx_fs, cmd_ctx);

	return ret;
}

static int sm2_alloc_cmd_ctx(struct sm2_ep *ep,
		struct fi_peer_rx_entry *rx_entry, struct sm2_free_queue_entry *fqe)
{
	struct sm2_cmd_ctx *cmd_ctx;

	if (ofi_freestack_isempty(ep->cmd_ctx_fs))
		return -FI_EAGAIN;

	cmd_ctx = ofi_freestack_pop(ep->cmd_ctx_fs);
	memcpy(&cmd_ctx->cmd, fqe, sizeof(*fqe));
	cmd_ctx->ep = ep;

	rx_entry->peer_context = cmd_ctx;

	return FI_SUCCESS;
}

static int sm2_progress_recv_msg(struct sm2_ep *ep, struct sm2_free_queue_entry *fqe)
{
	struct fid_peer_srx *peer_srx = sm2_get_peer_srx(ep);
	struct fi_peer_rx_entry *rx_entry;
	fi_addr_t addr;
	int ret;

	addr = ep->region->map->peers[fqe->protocol_hdr.id].fiaddr;
	if (fqe->protocol_hdr.op == ofi_op_tagged) {
		ret = peer_srx->owner_ops->get_tag(peer_srx, addr,
				fqe->protocol_hdr.tag, &rx_entry);
		if (ret == -FI_ENOENT) {
			ret = sm2_alloc_cmd_ctx(ep, rx_entry, fqe);
			if (ret)
				return ret;

			ret = peer_srx->owner_ops->queue_tag(rx_entry);
			goto out;
		}
	} else {
		ret = peer_srx->owner_ops->get_msg(peer_srx, addr,
				fqe->protocol_hdr.size, &rx_entry);
		if (ret == -FI_ENOENT) {
			ret = sm2_alloc_cmd_ctx(ep, rx_entry, fqe);
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
	ret = sm2_start_common(ep, fqe, rx_entry);

out:

	return ret < 0 ? ret : 0;
}

void sm2_progress_recv(struct sm2_ep *ep)
{
	struct sm2_free_queue_entry *fqe;
	// TODO Owning Region is part of hack!
	// TODO Should this be 1, is self 0?
	struct sm2_region *owning_region = sm2_peer_region(ep->region, 0);
	int ret = 0;

	// TODO SETH FIX THIS
	while (!sm2_fifo_empty(sm2_recv_queue(ep->region))) {
		// This will pop FQE off of FIFO recv queue, and we will own it until we return it
		fqe = sm2_fifo_read(sm2_recv_queue(ep->region), owning_region);

		switch (fqe->protocol_hdr.op) {
		case ofi_op_msg:
		case ofi_op_tagged:
			ret = sm2_progress_recv_msg(ep, fqe);
			break;
		default:
			FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
				"unidentified operation type\n");
			ret = -FI_EINVAL;
		}
		if (ret) {
			if (ret != -FI_EAGAIN) {
				FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
					"error processing command\n");
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
