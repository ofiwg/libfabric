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


static int sm2_progress_resp_entry(struct sm2_ep *ep, struct sm2_resp *resp,
				   struct sm2_tx_entry *pending, uint64_t *err)
{
	int i;
	struct sm2_region *peer_smr;
	size_t inj_offset;
	struct sm2_inject_buf *tx_buf = NULL;
	struct sm2_sar_buf *sar_buf = NULL;
	uint8_t *src;
	ssize_t hmem_copy_ret;

	peer_smr = sm2_peer_region(ep->region, pending->peer_id);

	switch (pending->cmd.msg.hdr.op_src) {
	case sm2_src_inject:
		inj_offset = (size_t) pending->cmd.msg.hdr.src_data;
		tx_buf = sm2_get_ptr(peer_smr, inj_offset);
		if (*err || pending->bytes_done == pending->cmd.msg.hdr.size ||
		    pending->cmd.msg.hdr.op == ofi_op_atomic)
			break;

		src = pending->cmd.msg.hdr.op == ofi_op_atomic_compare ?
		      tx_buf->buf : tx_buf->data;
		hmem_copy_ret  = ofi_copy_to_hmem_iov(pending->iface, pending->device,
						      pending->iov, pending->iov_count,
						      0, src, pending->cmd.msg.hdr.size);

		if (hmem_copy_ret < 0) {
			FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
				"RMA read/fetch failed with code %d\n",
				(int)(-hmem_copy_ret));
			*err = hmem_copy_ret;
		} else if (hmem_copy_ret != pending->cmd.msg.hdr.size) {
			FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
				"Incomplete rma read/fetch buffer copied\n");
			*err = -FI_ETRUNC;
		} else {
			pending->bytes_done = (size_t) hmem_copy_ret;
		}
		break;
	default:
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"unidentified operation type\n");
	}

	//Skip locking on transfers from self since we already have
	//the ep->region->lock
	if (peer_smr != ep->region) {
		if (pthread_spin_trylock(&peer_smr->lock)) {
			sm2_signal(ep->region);
			return -FI_EAGAIN;
		}
	}

	peer_smr->cmd_cnt++;
	if (tx_buf) {
		smr_freestack_push(sm2_inject_pool(peer_smr), tx_buf);
	} else if (sar_buf) {
		for (i = pending->cmd.msg.data.buf_batch_size - 1; i >= 0; i--) {
			smr_freestack_push_by_index(sm2_sar_pool(peer_smr),
					pending->cmd.msg.data.sar[i]);
		}
		peer_smr->sar_cnt++;
		sm2_peer_data(ep->region)[pending->peer_id].sar_status = 0;
	}

	if (peer_smr != ep->region)
		pthread_spin_unlock(&peer_smr->lock);

	return FI_SUCCESS;
}

static void sm2_progress_resp(struct sm2_ep *ep)
{
	struct sm2_resp *resp;
	struct sm2_tx_entry *pending;
	int ret;

	pthread_spin_lock(&ep->region->lock);
	ofi_spin_lock(&ep->tx_lock);
	while (!ofi_cirque_isempty(sm2_resp_queue(ep->region))) {
		resp = ofi_cirque_head(sm2_resp_queue(ep->region));
		if (resp->status == FI_EBUSY)
			break;

		pending = (struct sm2_tx_entry *) resp->msg_id;
		if (sm2_progress_resp_entry(ep, resp, pending, &resp->status))
			break;

		if (-resp->status) {
			ret = sm2_write_err_comp(ep->util_ep.tx_cq, pending->context,
					 pending->op_flags, pending->cmd.msg.hdr.tag,
					 -(resp->status));
		} else {
			ret = sm2_complete_tx(ep, pending->context,
					  pending->cmd.msg.hdr.op, pending->op_flags);
		}
		if (ret) {
			FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
				"unable to process tx completion\n");
			break;
		}
		ofi_freestack_push(ep->pend_fs, pending);
		ofi_cirque_discard(sm2_resp_queue(ep->region));
	}
	ofi_spin_unlock(&ep->tx_lock);
	pthread_spin_unlock(&ep->region->lock);
}

static int sm2_progress_inject(struct sm2_cmd *cmd, enum fi_hmem_iface iface,
			       uint64_t device, struct iovec *iov,
			       size_t iov_count, size_t *total_len,
			       struct sm2_ep *ep, int err)
{
	struct sm2_inject_buf *tx_buf;
	size_t inj_offset;
	ssize_t hmem_copy_ret;

	inj_offset = (size_t) cmd->msg.hdr.src_data;
	tx_buf = sm2_get_ptr(ep->region, inj_offset);

	if (err) {
		smr_freestack_push(sm2_inject_pool(ep->region), tx_buf);
		return err;
	}

	if (cmd->msg.hdr.op == ofi_op_read_req) {
		hmem_copy_ret = ofi_copy_from_hmem_iov(tx_buf->data,
						       cmd->msg.hdr.size,
						       iface, device, iov,
						       iov_count, 0);
	} else {
		hmem_copy_ret = ofi_copy_to_hmem_iov(iface, device, iov,
						     iov_count, 0, tx_buf->data,
						     cmd->msg.hdr.size);
		smr_freestack_push(sm2_inject_pool(ep->region), tx_buf);
	}

	if (hmem_copy_ret < 0) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"inject recv failed with code %d\n",
			(int)(-hmem_copy_ret));
		return hmem_copy_ret;
	} else if (hmem_copy_ret != cmd->msg.hdr.size) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"inject recv truncated\n");
		return -FI_ETRUNC;
	}

	*total_len = hmem_copy_ret;

	return FI_SUCCESS;
}

static int sm2_start_common(struct sm2_ep *ep, struct sm2_cmd *cmd,
		struct fi_peer_rx_entry *rx_entry)
{
	struct sm2_sar_entry *sar = NULL;
	size_t total_len = 0;
	uint64_t comp_flags;
	void *comp_buf;
	int ret;
	uint64_t err = 0, device;
	enum fi_hmem_iface iface;

	iface = sm2_get_mr_hmem_iface(ep->util_ep.domain, rx_entry->desc,
				      &device);

	switch (cmd->msg.hdr.op_src) {
	case sm2_src_inject:
		err = sm2_progress_inject(cmd, iface, device,
					  rx_entry->iov, rx_entry->count,
					  &total_len, ep, 0);
		ep->region->cmd_cnt++;
		break;
	default:
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"unidentified operation type\n");
		err = -FI_EINVAL;
	}

	comp_buf = rx_entry->iov[0].iov_base;
	comp_flags = sm2_rx_cq_flags(cmd->msg.hdr.op, rx_entry->flags,
				     cmd->msg.hdr.op_flags);
	if (!sar) {
		if (err) {
			FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
				"error processing op\n");
			ret = sm2_write_err_comp(ep->util_ep.rx_cq,
						 rx_entry->context,
						 comp_flags, rx_entry->tag,
						 err);
		} else {
			ret = sm2_complete_rx(ep, rx_entry->context, cmd->msg.hdr.op,
					      comp_flags, total_len, comp_buf,
					      cmd->msg.hdr.id, cmd->msg.hdr.tag,
					      cmd->msg.hdr.data);
		}
		if (ret) {
			FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
				"unable to process rx completion\n");
		}
		sm2_get_peer_srx(ep)->owner_ops->free_entry(rx_entry);
	}

	return 0;
}

int sm2_unexp_start(struct fi_peer_rx_entry *rx_entry)
{
	struct sm2_cmd_ctx *cmd_ctx = rx_entry->peer_context;
	int ret;

	pthread_spin_lock(&cmd_ctx->ep->region->lock);
	ret = sm2_start_common(cmd_ctx->ep, &cmd_ctx->cmd, rx_entry);
	ofi_freestack_push(cmd_ctx->ep->cmd_ctx_fs, cmd_ctx);
	pthread_spin_unlock(&cmd_ctx->ep->region->lock);

	return ret;
}

static void sm2_progress_connreq(struct sm2_ep *ep, struct sm2_cmd *cmd)
{
	struct sm2_region *peer_smr;
	struct sm2_inject_buf *tx_buf;
	size_t inj_offset;
	int64_t idx = -1;
	int ret = 0;

	inj_offset = (size_t) cmd->msg.hdr.src_data;
	tx_buf = sm2_get_ptr(ep->region, inj_offset);

	ret = sm2_map_add(&sm2_prov, ep->region->map,
			  (char *) tx_buf->data, &idx);
	if (ret)
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"Error processing mapping request\n");

	peer_smr = sm2_peer_region(ep->region, idx);

	if (peer_smr->pid != (int) cmd->msg.hdr.data) {
		//TODO track and update/complete in error any transfers
		//to or from old mapping
		munmap(peer_smr, peer_smr->total_size);
		sm2_map_to_region(&sm2_prov, ep->region->map, idx);
		peer_smr = sm2_peer_region(ep->region, idx);
	}
	sm2_peer_data(peer_smr)[cmd->msg.hdr.id].addr.id = idx;
	sm2_peer_data(ep->region)[idx].addr.id = cmd->msg.hdr.id;

	smr_freestack_push(sm2_inject_pool(ep->region), tx_buf);
	ofi_cirque_discard(sm2_cmd_queue(ep->region));
	ep->region->cmd_cnt++;
	assert(ep->region->map->num_peers > 0);
	ep->region->max_sar_buf_per_peer = SM2_MAX_PEERS /
		ep->region->map->num_peers;
}

static int sm2_alloc_cmd_ctx(struct sm2_ep *ep,
		struct fi_peer_rx_entry *rx_entry, struct sm2_cmd *cmd)
{
	struct sm2_cmd_ctx *cmd_ctx;

	if (ofi_freestack_isempty(ep->cmd_ctx_fs))
		return -FI_EAGAIN;

	cmd_ctx = ofi_freestack_pop(ep->cmd_ctx_fs);
	memcpy(&cmd_ctx->cmd, cmd, sizeof(*cmd));
	cmd_ctx->ep = ep;

	rx_entry->peer_context = cmd_ctx;

	return FI_SUCCESS;
}

static int sm2_progress_cmd_msg(struct sm2_ep *ep, struct sm2_cmd *cmd)
{
	struct fid_peer_srx *peer_srx = sm2_get_peer_srx(ep);
	struct fi_peer_rx_entry *rx_entry;
	fi_addr_t addr;
	int ret;

	addr = ep->region->map->peers[cmd->msg.hdr.id].fiaddr;
	if (cmd->msg.hdr.op == ofi_op_tagged) {
		ret = peer_srx->owner_ops->get_tag(peer_srx, addr,
				cmd->msg.hdr.tag, &rx_entry);
		if (ret == -FI_ENOENT) {
			ret = sm2_alloc_cmd_ctx(ep, rx_entry, cmd);
			if (ret)
				return ret;

			ret = peer_srx->owner_ops->queue_tag(rx_entry);
			goto out;
		}
	} else {
		ret = peer_srx->owner_ops->get_msg(peer_srx, addr,
				cmd->msg.hdr.size, &rx_entry);
		if (ret == -FI_ENOENT) {
			ret = sm2_alloc_cmd_ctx(ep, rx_entry, cmd);
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
	ret = sm2_start_common(ep, cmd, rx_entry);

out:
	ofi_cirque_discard(sm2_cmd_queue(ep->region));
	return ret < 0 ? ret : 0;
}

static int sm2_progress_cmd_rma(struct sm2_ep *ep, struct sm2_cmd *cmd)
{
	struct sm2_region *peer_smr;
	struct sm2_domain *domain;
	struct sm2_cmd *rma_cmd;
	struct sm2_resp *resp;
	struct iovec iov[SM2_IOV_LIMIT];
	size_t iov_count;
	size_t total_len = 0;
	int err = 0, ret = 0;
	struct ofi_mr *mr;
	enum fi_hmem_iface iface = FI_HMEM_SYSTEM;
	uint64_t device = 0;

	domain = container_of(ep->util_ep.domain, struct sm2_domain,
			      util_domain);

	ofi_cirque_discard(sm2_cmd_queue(ep->region));
	ep->region->cmd_cnt++;
	rma_cmd = ofi_cirque_head(sm2_cmd_queue(ep->region));

	ofi_genlock_lock(&domain->util_domain.lock);
	for (iov_count = 0; iov_count < rma_cmd->rma.rma_count; iov_count++) {
		ret = ofi_mr_map_verify(&domain->util_domain.mr_map,
				(uintptr_t *) &(rma_cmd->rma.rma_iov[iov_count].addr),
				rma_cmd->rma.rma_iov[iov_count].len,
				rma_cmd->rma.rma_iov[iov_count].key,
				ofi_rx_mr_reg_flags(cmd->msg.hdr.op, 0), (void **) &mr);
		if (ret)
			break;

		iov[iov_count].iov_base = (void *) rma_cmd->rma.rma_iov[iov_count].addr;
		iov[iov_count].iov_len = rma_cmd->rma.rma_iov[iov_count].len;

		if (!iov_count) {
			iface = mr->iface;
			device = mr->device;
		} else {
			assert(mr->iface == iface && mr->device == device);
		}
	}
	ofi_genlock_unlock(&domain->util_domain.lock);

	ofi_cirque_discard(sm2_cmd_queue(ep->region));
	if (ret) {
		ep->region->cmd_cnt++;
		return ret;
	}

	switch (cmd->msg.hdr.op_src) {
	case sm2_src_inject:
		err = sm2_progress_inject(cmd, iface, device, iov, iov_count,
					  &total_len, ep, ret);
		if (cmd->msg.hdr.op == ofi_op_read_req && cmd->msg.hdr.data) {
			peer_smr = sm2_peer_region(ep->region, cmd->msg.hdr.id);
			resp = sm2_get_ptr(peer_smr, cmd->msg.hdr.data);
			resp->status = -err;
			sm2_signal(peer_smr);
		} else {
			ep->region->cmd_cnt++;
		}
		break;
	default:
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"unidentified operation type\n");
		err = -FI_EINVAL;
	}

	if (err) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"error processing rma op\n");
		ret = sm2_write_err_comp(ep->util_ep.rx_cq, NULL,
					 sm2_rx_cq_flags(cmd->msg.hdr.op, 0,
					 cmd->msg.hdr.op_flags), 0, err);
	} else {
		ret = sm2_complete_rx(ep, (void *) cmd->msg.hdr.msg_id,
			      cmd->msg.hdr.op, sm2_rx_cq_flags(cmd->msg.hdr.op,
			      0, cmd->msg.hdr.op_flags), total_len,
			      iov_count ? iov[0].iov_base : NULL,
			      cmd->msg.hdr.id, 0, cmd->msg.hdr.data);
	}
	if (ret) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
		"unable to process rx completion\n");
	}

	return ret;
}

static void sm2_progress_cmd(struct sm2_ep *ep)
{
	struct sm2_cmd *cmd;
	int ret = 0;

	pthread_spin_lock(&ep->region->lock);
	while (!ofi_cirque_isempty(sm2_cmd_queue(ep->region))) {
		cmd = ofi_cirque_head(sm2_cmd_queue(ep->region));

		switch (cmd->msg.hdr.op) {
		case ofi_op_msg:
		case ofi_op_tagged:
			ret = sm2_progress_cmd_msg(ep, cmd);
			break;
		case ofi_op_write:
		case ofi_op_read_req:
			ret = sm2_progress_cmd_rma(ep, cmd);
			break;
		case ofi_op_write_async:
		case ofi_op_read_async:
			ofi_ep_rx_cntr_inc_func(&ep->util_ep,
						cmd->msg.hdr.op);
			ofi_cirque_discard(sm2_cmd_queue(ep->region));
			ep->region->cmd_cnt++;
			break;
		case SM2_OP_MAX + ofi_ctrl_connreq:
			sm2_progress_connreq(ep, cmd);
			break;
		default:
			FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
				"unidentified operation type\n");
			ret = -FI_EINVAL;
		}
		if (ret) {
			sm2_signal(ep->region);
			if (ret != -FI_EAGAIN) {
				FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
					"error processing command\n");
			}
			break;
		}
	}
	pthread_spin_unlock(&ep->region->lock);
}

void sm2_ep_progress(struct util_ep *util_ep)
{
	struct sm2_ep *ep;

	ep = container_of(util_ep, struct sm2_ep, util_ep);

	if (ofi_atomic_cas_bool32(&ep->region->signal, 1, 0)) {
		sm2_progress_resp(ep);
		sm2_progress_cmd(ep);
	}
}
