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
#include "ofi_mb.h"
#include "ofi_mr.h"
#include "ofi_shm_p2p.h"
#include "smr.h"
#include "smr_dsa.h"

static inline void
smr_try_progress_to_sar(struct smr_region *smr, struct smr_cmd *cmd,
			struct ofi_mr **mr, struct iovec *iov, size_t iov_count,
                        size_t *bytes_done, void *entry_ptr)
{
	if (*bytes_done < cmd->msg.hdr.size) {
		if (smr_env.use_dsa_sar && ofi_mr_all_host(mr, iov_count))
			(void) smr_dsa_copy_to_sar(smr, cmd, iov, iov_count,
						   bytes_done, entry_ptr);
		else
			smr_copy_to_sar(smr, cmd, mr, iov, iov_count,
					bytes_done);
	}
}

static inline void
smr_try_progress_from_sar(struct smr_region *smr, struct smr_cmd *cmd,
			struct ofi_mr **mr, struct iovec *iov, size_t iov_count,
                        size_t *bytes_done, void *entry_ptr)
{
	if (*bytes_done < cmd->msg.hdr.size) {
		if (smr_env.use_dsa_sar && ofi_mr_all_host(mr, iov_count))
			(void) smr_dsa_copy_from_sar(smr, cmd, iov, iov_count,
						     bytes_done, entry_ptr);
		else
			smr_copy_from_sar(smr, cmd, mr, iov, iov_count,
					  bytes_done);

	}
}

static int smr_progress_inline(struct smr_ep *ep, struct smr_cmd *cmd, struct fi_peer_rx_entry *rx_entry,
		struct ofi_mr **mr, struct iovec *iov, size_t iov_count,
		size_t *total_len, void *context)
{
	ssize_t hmem_copy_ret;

	hmem_copy_ret = ofi_copy_to_mr_iov(mr, iov, iov_count, 0,
				cmd->msg.data.msg, cmd->msg.hdr.size);
	if (hmem_copy_ret < 0) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"inline recv failed with code %d\n",
			(int)(-hmem_copy_ret));
		return hmem_copy_ret;
	} else if (hmem_copy_ret != cmd->msg.hdr.size) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"inline recv truncated\n");
		return -FI_ETRUNC;
	}

	*total_len = hmem_copy_ret;

	return FI_SUCCESS;
}

static int smr_progress_inject(struct smr_ep *ep, struct smr_cmd *cmd, struct fi_peer_rx_entry *rx_entry,
		struct ofi_mr **mr, struct iovec *iov, size_t iov_count,
		size_t *total_len, void *context)
{
	struct smr_inject_buf *tx_buf;
	ssize_t hmem_copy_ret;
	struct smr_region *peer_smr = smr_peer_region(ep->region, cmd->msg.hdr.id);

	assert(cmd->msg.hdr.op != ofi_op_read_req);

	tx_buf = smr_get_ptr(peer_smr, cmd->msg.hdr.src_data);
	hmem_copy_ret = ofi_copy_to_mr_iov(mr, iov, iov_count, 0, tx_buf->data,
	                                   cmd->msg.hdr.size);
	if (hmem_copy_ret < 0) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"inject recv failed with code %d\n",
			(int)(-hmem_copy_ret));
		return hmem_copy_ret;
	} else if (hmem_copy_ret != cmd->msg.hdr.size) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"inject recv truncated\n");
		return -FI_ETRUNC;
	}

	*total_len = hmem_copy_ret;

	return FI_SUCCESS;
}

static int smr_progress_iov(struct smr_ep *ep, struct smr_cmd *cmd, struct fi_peer_rx_entry *rx_entry,
		struct ofi_mr **mr, struct iovec *iov, size_t iov_count,
		size_t *total_len, void *context)
{
	struct smr_region *peer_smr;
	struct xpmem_client *xpmem;
	int ret;

	peer_smr = smr_peer_region(ep->region, cmd->msg.hdr.id);


	xpmem = &smr_peer_data(ep->region)[cmd->msg.hdr.id].xpmem;

	ret = ofi_shm_p2p_copy(ep->p2p_type, iov, iov_count, cmd->msg.data.iov,
			       cmd->msg.data.iov_count, cmd->msg.hdr.size,
			       peer_smr->pid, cmd->msg.hdr.op == ofi_op_read_req,
			       xpmem);
	if (!ret) {
		*total_len = cmd->msg.hdr.size;
		return FI_SUCCESS;
	}
	return -FI_ETRUNC;
}


static int smr_progress_sar(struct smr_ep *ep, struct smr_cmd *cmd, struct fi_peer_rx_entry *rx_entry,
		struct ofi_mr **mr, struct iovec *iov, size_t iov_count,
		size_t *total_len, void *context)
{
	struct smr_pend_entry *sar_entry;
	struct smr_region *peer_smr;
	struct iovec sar_iov[SMR_IOV_LIMIT];

	peer_smr = smr_peer_region(ep->region, cmd->msg.hdr.id);

	// TODO Do I need this?
	/* Nothing to do for 0 byte transfer */
	if (!cmd->msg.hdr.size) {
		return FI_SUCCESS;
	}

	memcpy(sar_iov, iov, sizeof(*iov) * iov_count);
	(void) ofi_truncate_iov(sar_iov, &iov_count, cmd->msg.hdr.size);

	// TODO try moving this alloc to after progress?
	sar_entry = ofi_buf_alloc(ep->pend_buf_pool);
	dlist_insert_tail(&sar_entry->entry, &ep->sar_list);

	if (cmd->msg.hdr.op == ofi_op_read_req)
		smr_try_progress_to_sar(peer_smr, cmd, mr, sar_iov,
					iov_count, total_len, sar_entry);
	else
		smr_try_progress_from_sar(peer_smr, cmd, mr, sar_iov,
					  iov_count, total_len, sar_entry);

	if (*total_len == cmd->msg.hdr.size) {
		dlist_remove(&sar_entry->entry);
		ofi_buf_free(sar_entry);
		return FI_SUCCESS;
	}

	// TODO Verify that this is correct
	sar_entry->cmd = *cmd; // TODO Do I need this?
	sar_entry->cmd_ctx = NULL; // TODO DO I need this?
	sar_entry->bytes_done = *total_len;
	sar_entry->rx_entry = rx_entry;
	cmd->msg.hdr.rx_ctx = (uintptr_t) sar_entry;
	memcpy(sar_entry->iov, sar_iov, sizeof(*sar_iov) * iov_count);
	sar_entry->iov_count = iov_count;
	sar_entry->context = context;
	// TODO Do I need an rx_entry here?
	if (mr) //try to get rid of this memcpy
		memcpy(sar_entry->mr, mr, sizeof(*mr) * iov_count);
	else
		memset(sar_entry->mr, 0, sizeof(*mr) * iov_count);

	*total_len = cmd->msg.hdr.size;
	return FI_SUCCESS;
}

static int
smr_ipc_async_copy(struct smr_ep *ep, struct smr_cmd *cmd,
		   struct ofi_mr_entry *mr_entry, struct iovec *iov,
		   size_t iov_count, void *ptr)
{
	struct smr_pend_entry *ipc_entry;
	enum fi_hmem_iface iface = cmd->msg.data.ipc_info.iface;
	uint64_t device = cmd->msg.data.ipc_info.device;
	int ret;

	ipc_entry = ofi_buf_alloc(ep->pend_buf_pool);
	if (!ipc_entry)
		return -FI_ENOMEM;

	ipc_entry->cmd = *cmd;
	ipc_entry->ipc_entry = mr_entry;
	ipc_entry->bytes_done = 0;
	memcpy(ipc_entry->iov, iov, sizeof(*iov) * iov_count);
	ipc_entry->iov_count = iov_count;
	ipc_entry->rx_entry->flags |= cmd->msg.hdr.op_flags;

	ret = ofi_create_async_copy_event(iface, device,
					  &ipc_entry->async_event);
	if (ret < 0)
		goto fail;

	if (cmd->msg.hdr.op == ofi_op_read_req) {
		ret = ofi_async_copy_from_hmem_iov(ptr, cmd->msg.hdr.size,
				iface, device, iov, iov_count, 0,
				ipc_entry->async_event);
	} else {
		ret = ofi_async_copy_to_hmem_iov(iface, device, iov, iov_count,
				0, ptr, cmd->msg.hdr.size,
				ipc_entry->async_event);
	}

	if (ret < 0) {
		(void) ofi_free_async_copy_event(iface, device,
					ipc_entry->async_event);
		goto fail;
	}

	dlist_insert_tail(&ipc_entry->entry, &ep->ipc_cpy_pend_list);
	return FI_SUCCESS;
fail:
	ofi_buf_free(ipc_entry);
	return ret;
}

static int smr_progress_ipc(struct smr_ep *ep, struct smr_cmd *cmd, struct fi_peer_rx_entry *rx_entry,
		struct ofi_mr **mr, struct iovec *iov, size_t iov_count,
		size_t *total_len, void *context)
{
	void *base, *ptr;
	uint64_t ipc_device;
	int64_t id;
	int ret, fd, ipc_fd;
	ssize_t hmem_copy_ret;
	struct ofi_mr_entry *mr_entry;
	struct smr_domain *domain;

	domain = container_of(ep->util_ep.domain, struct smr_domain,
			      util_domain);

	//TODO disable IPC if more than 1 interface is initialized
	if (cmd->msg.data.ipc_info.iface == FI_HMEM_ZE) {
		id = cmd->msg.hdr.id;
		ipc_device = cmd->msg.data.ipc_info.device;
		fd = ep->sock_info->peers[id].device_fds[ipc_device];
		ret = ze_hmem_open_shared_handle(fd,
				(void **) &cmd->msg.data.ipc_info.ipc_handle,
				&ipc_fd, ipc_device, &base);
	} else {
		ret = ofi_ipc_cache_search(domain->ipc_cache,
					   cmd->msg.hdr.id,
					   &cmd->msg.data.ipc_info,
					   &mr_entry);
		base = mr_entry->info.mapped_addr;
	}
	if (ret)
		return ret;

	ptr = (char *) base + (uintptr_t) cmd->msg.data.ipc_info.offset;
	if (cmd->msg.data.ipc_info.iface == FI_HMEM_ROCR) {//TODO fix rocr async
		*total_len = 0;
		ret = smr_ipc_async_copy(ep, cmd, mr_entry, iov, iov_count, ptr);
	}

	if (cmd->msg.hdr.op == ofi_op_read_req) {
		hmem_copy_ret = ofi_copy_from_hmem_iov(ptr, cmd->msg.hdr.size,
					cmd->msg.data.ipc_info.iface,
					cmd->msg.data.ipc_info.device, iov,
					iov_count, 0);
	} else {
		hmem_copy_ret = ofi_copy_to_hmem_iov(cmd->msg.data.ipc_info.iface,
					cmd->msg.data.ipc_info.device, iov,
					iov_count, 0, ptr, cmd->msg.hdr.size);
	}

	if (cmd->msg.data.ipc_info.iface == FI_HMEM_ZE) {
		close(ipc_fd);
		/* Truncation error takes precedence over close_handle error */
		ret = ofi_hmem_close_handle(cmd->msg.data.ipc_info.iface, base);
	} else {
		ofi_mr_cache_delete(domain->ipc_cache, mr_entry);
	}

	if (hmem_copy_ret < 0)
		return hmem_copy_ret;

	*total_len = hmem_copy_ret;

	return hmem_copy_ret == cmd->msg.hdr.size ? FI_SUCCESS : -FI_ETRUNC;
}

static void smr_do_atomic(void *src, void *dst, void *cmp, enum fi_datatype datatype,
			  enum fi_op op, size_t cnt, bool read)
{
	char tmp_result[SMR_INJECT_SIZE];

	if (ofi_atomic_isswap_op(op)) {
		ofi_atomic_swap_handler(op, datatype, dst, src, cmp,
					tmp_result, cnt);
	} else if (read) {
	 	ofi_atomic_readwrite_handler(op, datatype, dst, src,
	 				     tmp_result, cnt);
	} else if (ofi_atomic_iswrite_op(op)) {
		ofi_atomic_write_handler(op, datatype, dst, src, cnt);
	} else {
		FI_WARN(&smr_prov, FI_LOG_EP_DATA,
			"invalid atomic operation\n");
	}

	if (read)
	 	memcpy(src, op == FI_ATOMIC_READ ? dst : tmp_result,
	 	       cnt * ofi_datatype_size(datatype));

}

static int smr_progress_inline_atomic(struct smr_cmd *cmd, struct fi_ioc *ioc,
			       size_t ioc_count, size_t *len)
{
	uint8_t *src = cmd->msg.data.msg;
	int i;

	assert(cmd->msg.hdr.op == ofi_op_atomic);

	for (i = *len = 0; i < ioc_count && *len < cmd->msg.hdr.size; i++) {
		smr_do_atomic(&src[*len], ioc[i].addr, NULL,
			      cmd->msg.hdr.datatype, cmd->msg.hdr.atomic_op,
			      ioc[i].count, false);
		*len += ioc[i].count * ofi_datatype_size(cmd->msg.hdr.datatype);
	}

	if (*len != cmd->msg.hdr.size) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"recv truncated");
		return -FI_ETRUNC;
	}
	return FI_SUCCESS;
}

static int smr_progress_inject_atomic(struct smr_cmd *cmd, struct fi_ioc *ioc,
			       size_t ioc_count, size_t *len,
			       struct smr_ep *ep, int err)
{
	struct smr_inject_buf *tx_buf;
	size_t inj_offset;
	uint8_t *src, *comp;
	bool read = false;
	int i;
	struct smr_region *peer_smr = smr_peer_region(ep->region, cmd->msg.hdr.id);

	inj_offset = (size_t) cmd->msg.hdr.src_data;
	tx_buf = smr_get_ptr(peer_smr, inj_offset);
	if (err)
		goto out;

	switch (cmd->msg.hdr.op) {
	case ofi_op_atomic_compare:
		src = tx_buf->buf;
		comp = tx_buf->comp;
		break;
	default:
		src = tx_buf->data;
		comp = NULL;
		break;
	}

	if (cmd->msg.hdr.op == ofi_op_atomic_compare ||
	    cmd->msg.hdr.op == ofi_op_atomic_fetch)
		read = true;

	for (i = *len = 0; i < ioc_count && *len < cmd->msg.hdr.size; i++) {
		smr_do_atomic(&src[*len], ioc[i].addr, comp ? &comp[*len] : NULL,
			      cmd->msg.hdr.datatype, cmd->msg.hdr.atomic_op,
			      ioc[i].count, read);
		*len += ioc[i].count * ofi_datatype_size(cmd->msg.hdr.datatype);
	}

	if (*len != cmd->msg.hdr.size) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"recv truncated");
		err = -FI_ETRUNC;
	}

out:
	return err;
}

typedef int (*smr_progress_func)(struct smr_ep *ep, struct smr_cmd *cmd, struct fi_peer_rx_entry *rx_entry,
		struct ofi_mr **mr, struct iovec *iov, size_t iov_count,
		size_t *total_len, void *context);

static smr_progress_func smr_progress_ops[smr_src_max] = {
	[smr_src_inline] = &smr_progress_inline,
	[smr_src_inject] = &smr_progress_inject,
	[smr_src_iov] = &smr_progress_iov,
	[smr_src_sar] = &smr_progress_sar,
	[smr_src_ipc] = &smr_progress_ipc,
};

static int smr_start_common(struct smr_ep *ep, struct smr_cmd *cmd,
		struct fi_peer_rx_entry *rx_entry)
{
	size_t total_len = 0;
	uint64_t comp_flags;
	void *comp_buf;
	int ret;

	cmd->msg.hdr.rx_ctx = 0;
	ret = smr_progress_ops[cmd->msg.hdr.op_src](ep, cmd, rx_entry,
			(struct ofi_mr **) rx_entry->desc, rx_entry->iov,
			rx_entry->count, &total_len, rx_entry->context);
	if (cmd->msg.hdr.rx_ctx)
		goto out;

	comp_buf = rx_entry->iov[0].iov_base;
	comp_flags = smr_rx_cq_flags(cmd->msg.hdr.op, rx_entry->flags,
				     cmd->msg.hdr.op_flags);
	if (ret) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL, "error processing op\n");
		ret = smr_write_err_comp(ep->util_ep.rx_cq, rx_entry->context,
					comp_flags, rx_entry->tag, ret);
	} else {
		ret = smr_complete_rx(ep, rx_entry->context, cmd->msg.hdr.op,
				      comp_flags, total_len, comp_buf,
				      cmd->msg.hdr.id, cmd->msg.hdr.tag,
				      cmd->msg.hdr.data);
	}
	if (ret) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unable to process rx completion\n");
	}
	smr_get_peer_srx(ep)->owner_ops->free_entry(rx_entry);
out:
	return ret;
}

static int smr_copy_saved(struct smr_cmd_ctx *cmd_ctx,
		struct fi_peer_rx_entry *rx_entry)
{
	struct smr_unexp_buf *sar_buf;
	size_t bytes = 0;
	uint64_t comp_flags;
	int ret;

	while (!slist_empty(&cmd_ctx->buf_list)) {
		slist_remove_head_container(&cmd_ctx->buf_list,
				struct smr_unexp_buf, sar_buf, entry);

		bytes += ofi_copy_to_mr_iov((struct ofi_mr **) rx_entry->desc,
				rx_entry->iov, rx_entry->count, bytes,
				sar_buf->buf,
				MIN(cmd_ctx->cmd.msg.hdr.size - bytes,
					SMR_SAR_SIZE));
		ofi_buf_free(sar_buf);
	}
	if (bytes != cmd_ctx->cmd.msg.hdr.size) {
		assert(cmd_ctx->sar_entry);
		cmd_ctx->sar_entry->cmd_ctx = NULL;
		cmd_ctx->sar_entry->rx_entry = rx_entry;
		memcpy(cmd_ctx->sar_entry->iov, rx_entry->iov,
		       sizeof(*rx_entry->iov) * rx_entry->count);
		cmd_ctx->sar_entry->iov_count = rx_entry->count;
		(void) ofi_truncate_iov(cmd_ctx->sar_entry->iov,
					&cmd_ctx->sar_entry->iov_count,
					cmd_ctx->cmd.msg.hdr.size);
		memcpy(cmd_ctx->sar_entry->mr, rx_entry->desc,
		       sizeof(*rx_entry->desc) * cmd_ctx->sar_entry->iov_count);
		return FI_SUCCESS;
	}

	comp_flags = smr_rx_cq_flags(cmd_ctx->cmd.msg.hdr.op,
			rx_entry->flags, cmd_ctx->cmd.msg.hdr.op_flags);

	ret = smr_complete_rx(cmd_ctx->ep, rx_entry->context,
			      cmd_ctx->cmd.msg.hdr.op, comp_flags,
			      bytes, rx_entry->iov[0].iov_base,
			      cmd_ctx->cmd.msg.hdr.id,
			      cmd_ctx->cmd.msg.hdr.tag,
			      cmd_ctx->cmd.msg.hdr.data);
	if (ret) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unable to process rx completion\n");
		return ret;
	}
	smr_get_peer_srx(cmd_ctx->ep)->owner_ops->free_entry(rx_entry);

	return FI_SUCCESS;
}

int smr_unexp_start(struct fi_peer_rx_entry *rx_entry)
{
	struct smr_cmd_ctx *cmd_ctx = rx_entry->peer_context;
	int ret;

	if (cmd_ctx->cmd.msg.hdr.op_src == smr_src_sar ||
	    cmd_ctx->cmd.msg.hdr.op_src == smr_src_inject)
		ret = smr_copy_saved(cmd_ctx, rx_entry);
	else
		ret = smr_start_common(cmd_ctx->ep, &cmd_ctx->cmd, rx_entry);

	ofi_buf_free(cmd_ctx);

	return ret;
}

static void smr_buffer_sar(struct smr_ep *ep, struct smr_region *peer_smr,
		           struct smr_pend_entry *sar_entry)
{
	struct smr_sar_buf *sar_buf;
	struct smr_unexp_buf *buf;
	size_t bytes;
	int next_buf = 0;

	while (next_buf < sar_entry->cmd.msg.data.buf_batch_size &&
	       sar_entry->bytes_done < sar_entry->cmd.msg.hdr.size) {
		buf = ofi_buf_alloc(ep->unexp_buf_pool);
		if (!buf) {
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"Error allocating buffer\n");
			assert(0);
		}
		slist_insert_tail(&buf->entry,
			&sar_entry->cmd_ctx->buf_list);

		sar_buf = smr_freestack_get_entry_from_index(
				smr_sar_pool(peer_smr),
				sar_entry->cmd.msg.data.sar[next_buf]);
		bytes = MIN(sar_entry->cmd.msg.hdr.size -
				sar_entry->bytes_done,
				SMR_SAR_SIZE);

		memcpy(buf->buf, sar_buf->buf, bytes);

		sar_entry->bytes_done += bytes;
		next_buf++;
	}
	ofi_wmb();
}

static void smr_progress_pending_rx(struct smr_ep *ep, struct smr_cmd *cmd)
{
	struct smr_pend_entry *pend_entry;
	struct smr_region *peer_smr;
	uint64_t comp_flags;
	int ret = FI_SUCCESS;

	pend_entry = (struct smr_pend_entry *) cmd->msg.hdr.rx_ctx;
	switch (cmd->msg.hdr.op_src) {
	case smr_src_iov:
		assert(0);
		break; // TODO add fallback proto processing, separate this function into other functions
	case smr_src_sar:
		peer_smr = smr_peer_region(ep->region, cmd->msg.hdr.id);
		if (cmd->msg.hdr.op == ofi_op_read_req) {
			smr_try_progress_to_sar(peer_smr, cmd, pend_entry->mr,
					pend_entry->iov, pend_entry->iov_count,
					&pend_entry->bytes_done, pend_entry);
		} else {
			if (pend_entry->cmd_ctx)
				smr_buffer_sar(ep, peer_smr, pend_entry);
			else
				smr_try_progress_from_sar(peer_smr, cmd, pend_entry->mr,
					pend_entry->iov, pend_entry->iov_count,
					&pend_entry->bytes_done, pend_entry);
		}

		if (pend_entry->bytes_done == cmd->msg.hdr.size) {
			if (pend_entry->rx_entry) {
				comp_flags = smr_rx_cq_flags(
					cmd->msg.hdr.op,
					pend_entry->rx_entry->flags,
					cmd->msg.hdr.op_flags);
			} else {
				comp_flags = smr_rx_cq_flags(
					cmd->msg.hdr.op, 0,
					cmd->msg.hdr.op_flags);
			}
			ret = smr_complete_rx(ep, pend_entry->context,
					cmd->msg.hdr.op, comp_flags,
					pend_entry->bytes_done,
					pend_entry->iov[0].iov_base,
					cmd->msg.hdr.id, cmd->msg.hdr.tag,
					cmd->msg.hdr.data);
			if (ret) {
				FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
					"unable to process rx completion\n");
			}
			if (pend_entry->rx_entry)
				smr_get_peer_srx(ep)->owner_ops->free_entry(
							pend_entry->rx_entry);

			dlist_remove(&pend_entry->entry);
			ofi_buf_free(pend_entry);
		}
		break;
	default:
		assert(0);
	}

	smr_return_cmd(ep->region, cmd);
}

static int smr_alloc_cmd_ctx(struct smr_ep *ep,
		struct fi_peer_rx_entry *rx_entry, struct smr_cmd *cmd)
{
	struct smr_cmd_ctx *cmd_ctx;
	struct smr_pend_entry *sar_entry;
	struct smr_inject_buf *tx_buf;
	struct smr_unexp_buf *buf;
	struct smr_region *peer_smr = smr_peer_region(ep->region, cmd->msg.hdr.id);

	cmd_ctx = ofi_buf_alloc(ep->cmd_ctx_pool);
	if (!cmd_ctx) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"Error allocating cmd ctx\n");
		return -FI_ENOMEM;
	}
	memcpy(&cmd_ctx->cmd, cmd, sizeof(*cmd));
	cmd_ctx->ep = ep;

	// TODO I do not think we need this block b/c rx_entry->size isn't used, and I think we have a test that is failing without this block.
	// We also don't have SMR_REMOTE_CQ_DATA flags
	// rx_entry->size = cmd->msg.hdr.size;
	// if (cmd->msg.hdr.op_flags & SMR_REMOTE_CQ_DATA) {
	// 	rx_entry->flags |= FI_REMOTE_CQ_DATA;
	// 	rx_entry->cq_data = cmd->msg.hdr.data;
	// }

	if (cmd->msg.hdr.op_src == smr_src_inject) {
		buf = ofi_buf_alloc(ep->unexp_buf_pool);
		if (!buf) {
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"Error allocating buffer\n");
			assert(0);
		}
		cmd_ctx->sar_entry = NULL;
		slist_init(&cmd_ctx->buf_list);
		slist_insert_tail(&buf->entry, &cmd_ctx->buf_list);
		tx_buf = smr_get_ptr(peer_smr, (size_t) cmd->msg.hdr.src_data);
		memcpy(buf->buf, tx_buf->buf, cmd->msg.hdr.size);
	} else if (cmd->msg.hdr.op_src == smr_src_sar) {
		slist_init(&cmd_ctx->buf_list);

		if (cmd->msg.hdr.size) {
			sar_entry = ofi_buf_alloc(ep->pend_buf_pool);

			memcpy(&sar_entry->cmd, cmd, sizeof(*cmd));
			sar_entry->cmd_ctx = cmd_ctx;
			sar_entry->bytes_done = 0;
			sar_entry->rx_entry = rx_entry;

			dlist_insert_tail(&sar_entry->entry, &ep->sar_list);

			cmd_ctx->sar_entry = sar_entry;
			smr_buffer_sar(ep, peer_smr, sar_entry);

			if (sar_entry->bytes_done < cmd->msg.hdr.size)
				cmd->msg.hdr.rx_ctx = (uintptr_t) sar_entry;
		}
	}

	rx_entry->peer_context = cmd_ctx;
	return FI_SUCCESS;
}

static void smr_progress_cmd_msg(struct smr_ep *ep, struct smr_cmd *cmd)
{
	struct fid_peer_srx *peer_srx = smr_get_peer_srx(ep);
	struct fi_peer_rx_entry *rx_entry;
	fi_addr_t addr;
	int ret;

	if (cmd->msg.hdr.rx_ctx) {
		smr_progress_pending_rx(ep, cmd);
		return;
	}

	addr = ep->region->map->peers[cmd->msg.hdr.id].fiaddr;
	if (cmd->msg.hdr.op == ofi_op_tagged) {
		ret = peer_srx->owner_ops->get_tag(peer_srx, addr,
				cmd->msg.hdr.tag, &rx_entry);
		if (ret == -FI_ENOENT) {
			ret = smr_alloc_cmd_ctx(ep, rx_entry, cmd);
			if (ret) {
				peer_srx->owner_ops->free_entry(rx_entry);
				goto out;
			}

			ret = peer_srx->owner_ops->queue_tag(rx_entry);
			if (ret) {
				peer_srx->owner_ops->free_entry(rx_entry);
			}
			goto out;
		}
	} else {
		ret = peer_srx->owner_ops->get_msg(peer_srx, addr,
				cmd->msg.hdr.size, &rx_entry);
		if (ret == -FI_ENOENT) {
			ret = smr_alloc_cmd_ctx(ep, rx_entry, cmd);
			if (ret) {
				peer_srx->owner_ops->free_entry(rx_entry);
				goto out;
			}

			ret = peer_srx->owner_ops->queue_msg(rx_entry);
			if (ret) {
				peer_srx->owner_ops->free_entry(rx_entry);
			}
			goto out;
		}
	}

	if (ret) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL, "Error getting rx_entry\n");
		goto out;
	}

	// TODO do I need to do anything with this return value? We already log it in smr_start_common()
	ret = smr_start_common(ep, cmd, rx_entry);
out:
	smr_return_cmd(ep->region, cmd);
}

static void smr_progress_cmd_rma(struct smr_ep *ep, struct smr_cmd *cmd)
{
	struct smr_cmd *rma_cmd = (struct smr_cmd *) cmd->msg.hdr.rma_cmd;
	struct smr_domain *domain;
	struct iovec iov[SMR_IOV_LIMIT];
	size_t iov_count;
	size_t total_len = 0;
	int err = 0, ret = 0;
	int64_t id = cmd->msg.hdr.id;
	struct ofi_mr *mr[SMR_IOV_LIMIT];

	if (cmd->msg.hdr.rx_ctx) {
		smr_progress_pending_rx(ep, cmd);
		return;
	}

	domain = container_of(ep->util_ep.domain, struct smr_domain,
			      util_domain);
	ofi_genlock_lock(&domain->util_domain.lock);
	for (iov_count = 0; iov_count < rma_cmd->rma.rma_count; iov_count++) {
		ret = ofi_mr_map_verify(&domain->util_domain.mr_map,
				(uintptr_t *) &(rma_cmd->rma.rma_iov[iov_count].addr),
				rma_cmd->rma.rma_iov[iov_count].len,
				rma_cmd->rma.rma_iov[iov_count].key,
				ofi_rx_mr_reg_flags(cmd->msg.hdr.op, 0),
				(void **) &mr[iov_count]);
		if (ret)
			break;

		iov[iov_count].iov_base = (void *) rma_cmd->rma.rma_iov[iov_count].addr;
		iov[iov_count].iov_len = rma_cmd->rma.rma_iov[iov_count].len;
	}
	ofi_genlock_unlock(&domain->util_domain.lock);

	if (ret)
		goto out;

	ret = smr_progress_ops[cmd->msg.hdr.op_src](ep, cmd, NULL, mr, iov, iov_count,
						    &total_len, NULL);


	// TODO Look at ERROR Completions b/c they are bad
	if (ret) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL, "error processing rma op\n");
		ret = smr_write_err_comp(ep->util_ep.rx_cq, NULL,
					 smr_rx_cq_flags(cmd->msg.hdr.op, 0,
					 cmd->msg.hdr.op_flags), 0, err);
	} else {
		// TODO This being in the else will cause us to have multiple error completions on bad SAR (maybe)
		if (cmd->msg.hdr.rx_ctx)
			goto out;

		ret = smr_complete_rx(ep, NULL,
			      cmd->msg.hdr.op, smr_rx_cq_flags(cmd->msg.hdr.op,
			      0, cmd->msg.hdr.op_flags), total_len,
			      iov_count ? iov[0].iov_base : NULL,
			      cmd->msg.hdr.id, 0, cmd->msg.hdr.data);
	}
	if (ret) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unable to process rx completion\n");
	}

out:
	/* Set RMA Pointer back to host memory, so host can return to its free stack */
	assert(cmd->msg.hdr.rma_cmd);
	cmd->msg.hdr.rma_cmd = smr_get_owner_ptr(ep->region, id,
	                                         smr_peer_data(ep->region)[id].addr.id,
	                                         rma_cmd);
	smr_return_cmd(ep->region, cmd);
}

static void smr_progress_cmd_atomic(struct smr_ep *ep, struct smr_cmd *cmd)
{
	struct smr_cmd *rma_cmd = (struct smr_cmd *) cmd->msg.hdr.rma_cmd;
	struct smr_domain *domain;
	struct fi_ioc ioc[SMR_IOV_LIMIT];
	size_t ioc_count;
	size_t total_len = 0;
	int ret = 0;
	int64_t id = cmd->msg.hdr.id;

	domain = container_of(ep->util_ep.domain, struct smr_domain,
			      util_domain);

	ofi_genlock_lock(&domain->util_domain.lock);
	for (ioc_count = 0; ioc_count < rma_cmd->rma.rma_count; ioc_count++) {
		ret = ofi_mr_map_verify(&domain->util_domain.mr_map,
				(uintptr_t *) &(rma_cmd->rma.rma_ioc[ioc_count].addr),
				rma_cmd->rma.rma_ioc[ioc_count].count *
				ofi_datatype_size(cmd->msg.hdr.datatype),
				rma_cmd->rma.rma_ioc[ioc_count].key,
				ofi_rx_mr_reg_flags(cmd->msg.hdr.op,
				cmd->msg.hdr.atomic_op), NULL);
		if (ret)
			break;

		ioc[ioc_count].addr = (void *) rma_cmd->rma.rma_ioc[ioc_count].addr;
		ioc[ioc_count].count = rma_cmd->rma.rma_ioc[ioc_count].count;
	}
	ofi_genlock_unlock(&domain->util_domain.lock);

	if (ret)
		goto out;

	switch (cmd->msg.hdr.op_src) {
	case smr_src_inline:
		ret = smr_progress_inline_atomic(cmd, ioc, ioc_count,
						 &total_len);
		break;
	case smr_src_inject:
		ret = smr_progress_inject_atomic(cmd, ioc, ioc_count,
						 &total_len, ep, ret);
		break;
	default:
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unidentified operation type\n");
		ret = -FI_EINVAL;
	}

	if (cmd->msg.hdr.data) {
		/*
		 * smr_do_atomic will do memcpy when flags has SMR_RMA_REQ.
		 * Add a memory barrier before updating resp status to ensure
		 * the buffer is ready before the status update.
		 */
		// TODO Only do memory barrier for FETCH and compare ops that do RMA Read Operations.
		// if (cmd->msg.hdr.op_flags & SMR_RMA_REQ)
		ofi_wmb();
	}

	if (ret) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"error processing atomic op\n");
		ret = smr_write_err_comp(ep->util_ep.rx_cq, NULL,
					 smr_rx_cq_flags(cmd->msg.hdr.op, 0,
					 cmd->msg.hdr.op_flags), 0, ret);
	} else {
		ret = smr_complete_rx(ep, NULL, cmd->msg.hdr.op,
				      smr_rx_cq_flags(cmd->msg.hdr.op, 0,
				      cmd->msg.hdr.op_flags), total_len,
				      ioc_count ? ioc[0].addr : NULL,
				      cmd->msg.hdr.id, 0, cmd->msg.hdr.data);
	}
	if (ret) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unable to process rx completion\n");
	}

out:
	/* Set RMA Pointer back to host memory, so host can return to its free stack */
	assert(cmd->msg.hdr.rma_cmd);
	cmd->msg.hdr.rma_cmd = smr_get_owner_ptr(ep->region, id,
	                                         smr_peer_data(ep->region)[id].addr.id,
	                                         rma_cmd);
	smr_return_cmd(ep->region, cmd);
}

static void smr_progress_connreq(struct smr_ep *ep)
{
	struct smr_conn_req *conn_req;
	struct smr_region *peer_smr;
	int64_t idx = -1, pos;
	int ret = 0;

	while (1) {
		ret = smr_conn_queue_head(smr_conn_queue(ep->region), &conn_req,
					  &pos);
		if (ret == -FI_ENOENT)
			break;

		ret = smr_map_add(&smr_prov, ep->region->map,
				(char *) conn_req->name, &idx);
		if (ret)
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"Error processing mapping request\n");

		peer_smr = smr_peer_region(ep->region, idx);

		if (peer_smr->pid != conn_req->pid) {
			//TODO track and update/complete in error any transfers
			//to or from old mapping
			munmap(peer_smr, peer_smr->total_size);
			smr_map_to_region(&smr_prov, ep->region->map, idx);
			peer_smr = smr_peer_region(ep->region, idx);
		}
		smr_peer_data(ep->region)[idx].local_region =
					(uintptr_t) peer_smr;
		smr_peer_data(peer_smr)[conn_req->id].addr.id = idx;
		smr_peer_data(ep->region)[idx].addr.id = conn_req->id;

		assert(ep->region->map->num_peers > 0);
		ep->region->max_sar_buf_per_peer = SMR_MAX_PEERS /
			ep->region->map->num_peers;

		smr_conn_queue_release(smr_conn_queue(ep->region), conn_req,
				       pos);
	}
}

static void smr_progress_return(struct smr_ep *ep)
{
	struct smr_cmd *cmd;
	struct smr_tx_entry *pending;
	struct smr_inject_buf *tx_buf = NULL;
	uint8_t *src;
	int i;
	int ret;

	while (1) {
		cmd = smr_read_return(ep->region);
		if (!cmd)
			break;

		pending = (struct smr_tx_entry *) cmd->msg.hdr.tx_ctx;

		if (cmd->msg.hdr.rma_cmd) {
			smr_freestack_push(smr_cmd_pool(ep->region),
					(struct smr_cmd *) cmd->msg.hdr.rma_cmd);
			cmd->msg.hdr.rma_cmd = 0;
		}

		switch (cmd->msg.hdr.op_src) {
		case smr_src_inline:
			break;
		case smr_src_iov:
			//TODO deal with IPC fallback here
			break;
		case smr_src_ipc:
			assert(pending->mr[0]);
			if (pending->mr[0]->iface == FI_HMEM_ZE)
				close(pending->fd);
			break;
		case smr_src_sar:
			// TODO Refactor this to get rid of copy/paste
			if (cmd->msg.hdr.op == ofi_op_read_req) {
				smr_try_progress_from_sar(ep->region, cmd,
							pending->mr, pending->iov,
							pending->iov_count,
							&pending->bytes_done, pending);

				if (pending->bytes_done == cmd->msg.hdr.size) {
					for (i = cmd->msg.data.buf_batch_size - 1; i >= 0; i--) {
						smr_freestack_push_by_index(
							smr_sar_pool(ep->region),
							cmd->msg.data.sar[i]);
					}
					break;
				}
			} else {
				if (pending->bytes_done == cmd->msg.hdr.size) {
					for (i = cmd->msg.data.buf_batch_size - 1; i >= 0; i--) {
						smr_freestack_push_by_index(
							smr_sar_pool(ep->region),
							cmd->msg.data.sar[i]);
					}
					break;
				}

				smr_try_progress_to_sar(ep->region, cmd, pending->mr,
							pending->iov, pending->iov_count,
							&pending->bytes_done, pending);

			}
			smr_commit_cmd(ep->region, pending->peer_id, cmd);
			return;
		case smr_src_inject:
			tx_buf = smr_get_ptr(ep->region, cmd->msg.hdr.src_data);

			if (cmd->msg.hdr.op == ofi_op_atomic_fetch ||
			cmd->msg.hdr.op == ofi_op_atomic_compare) {
				src = cmd->msg.hdr.op == ofi_op_atomic_compare ? tx_buf->buf : tx_buf->data;
				ret  = ofi_copy_to_mr_iov(pending->mr, pending->iov,
							pending->iov_count, 0, src,
							cmd->msg.hdr.size);

				if (ret < 0)
					FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
						"Atomic read/fetch failed with code %d\n",
						-ret);
				else if (ret != cmd->msg.hdr.size)
					FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
						"Incomplete atomic fetch/compare buffer copied\n");
			}

			smr_freestack_push(smr_inject_pool(ep->region), tx_buf);
			break;
		default:
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"unidentified operation type\n");
		}

		if (!(cmd->msg.hdr.op_flags & FI_DELIVERY_COMPLETE)) {
			smr_freestack_push(smr_cmd_pool(ep->region), cmd);
			return;
		}

		assert(pending);  //do I even need pending anymore? context?

		// TODO DO SOMETHING ABOUT ERR Completions here
		// if (ret) {
		// 	ret = smr_write_err_comp(ep->util_ep.tx_cq, pending->context,
		// 			cmd->msg.hdr.op_flags, cmd->msg.hdr.tag, ret);
		// } else {smr_complete_tx(...)}
		ret = smr_complete_tx(ep, pending->context, cmd->msg.hdr.op,
				cmd->msg.hdr.op_flags);

		smr_peer_data(ep->region)[pending->peer_id].sar = false;
		if (ret) {
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"unable to process tx completion\n");
		}

		smr_freestack_push(smr_cmd_pool(ep->region), cmd);
		ofi_freestack_push(ep->tx_fs, pending);
	}
}

static void smr_progress_cmd(struct smr_ep *ep)
{
	struct smr_cmd *cmd;

	/* ep->util_ep.lock is used to serialize the message/tag matching.
	 * We keep the lock until the matching is complete. This will
	 * ensure that commands are matched in the order they are
	 * received, if there are multiple progress threads.
	 *
	 * This lock should be low cost because it's only used by this
	 * single process. It is also optimized to be a noop if
	 * multi-threading is disabled.
	 *
	 * Other processes are free to post on the queue without the need
	 * for locking the queue.
	 */
	while (1) {
		cmd = smr_read_cmd(ep->region);
		if (!cmd)
			break;

		switch (cmd->msg.hdr.op) {
		case ofi_op_msg:
		case ofi_op_tagged:
			smr_progress_cmd_msg(ep, cmd);
			break;
		case ofi_op_write:
		case ofi_op_read_req:
			smr_progress_cmd_rma(ep, cmd);
			break;
		case ofi_op_write_async:
		case ofi_op_read_async:
			ofi_ep_peer_rx_cntr_inc(&ep->util_ep,
						cmd->msg.hdr.op);
			break;
		case ofi_op_atomic:
		case ofi_op_atomic_fetch:
		case ofi_op_atomic_compare:
			smr_progress_cmd_atomic(ep, cmd);
			break;
		default:
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"unidentified operation type\n");
		}
	}
}

void smr_progress_ipc_list(struct smr_ep *ep)
{
	struct smr_pend_entry *ipc_entry;
	struct smr_domain *domain;
	enum fi_hmem_iface iface;
	struct dlist_entry *tmp;
	uint64_t device;
	uint64_t flags;
	void *context;
	int ret;

	domain = container_of(ep->util_ep.domain, struct smr_domain,
			      util_domain);

	/* after the synchronize all operations should be complete */
	dlist_foreach_container_safe(&ep->ipc_cpy_pend_list,
				     struct smr_pend_entry,
				     ipc_entry, entry, tmp) {
		iface = ipc_entry->cmd.msg.data.ipc_info.iface;
		device = ipc_entry->cmd.msg.data.ipc_info.device;
		if (ofi_async_copy_query(iface, ipc_entry->async_event))
			continue;

		if (ipc_entry->rx_entry) {
			context = ipc_entry->rx_entry->context;
			flags = smr_rx_cq_flags(ipc_entry->cmd.msg.hdr.op,
					ipc_entry->rx_entry->flags,
					ipc_entry->cmd.msg.hdr.op_flags);
		} else {
			context = NULL;
			flags = smr_rx_cq_flags(ipc_entry->cmd.msg.hdr.op,
					0, ipc_entry->cmd.msg.hdr.op_flags);
		}

		ret = smr_complete_rx(ep, context, ipc_entry->cmd.msg.hdr.op,
				flags, ipc_entry->cmd.msg.hdr.size,
				ipc_entry->iov[0].iov_base,
				ipc_entry->cmd.msg.hdr.id,
				ipc_entry->cmd.msg.hdr.tag,
				ipc_entry->cmd.msg.hdr.data);
		if (ret) {
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"unable to process rx completion\n");
		}

		ofi_mr_cache_delete(domain->ipc_cache, ipc_entry->ipc_entry);
		ofi_free_async_copy_event(iface, device,
					  ipc_entry->async_event);
		dlist_remove(&ipc_entry->entry);
		if (ipc_entry->rx_entry)
			smr_get_peer_srx(ep)->owner_ops->free_entry(
						ipc_entry->rx_entry);
		smr_return_cmd(ep->region, &ipc_entry->cmd);
		ofi_buf_free(ipc_entry);
	}
}

void smr_ep_progress(struct util_ep *util_ep)
{
	struct smr_ep *ep;

	ep = container_of(util_ep, struct smr_ep, util_ep);

	if (smr_env.use_dsa_sar)
		smr_dsa_progress(ep);

	ofi_genlock_lock(&ep->util_ep.lock);
	smr_progress_connreq(ep);
	smr_progress_return(ep);
	smr_progress_cmd(ep);
	ofi_genlock_unlock(&ep->util_ep.lock);

	/* always drive forward the ipc list since the completion is
	 * independent of any action by the provider */
	ep->smr_progress_ipc_list(ep);
}
