/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2017-2019 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include "config.h"

#include <ofi_mem.h>

#include "efa.h"
#include "efa_cmd.h"
#include "efa_ib.h"
#include "efa_io_defs.h"

static __u32 efa_cq_sub_cq_get_current_index(struct efa_sub_cq *sub_cq)
{
	return sub_cq->consumed_cnt & sub_cq->qmask;
}

static int efa_cq_cqe_is_pending(struct efa_io_cdesc_common *cqe_common, int phase)
{
	return (cqe_common->flags & EFA_IO_CDESC_COMMON_PHASE_MASK) == phase;
}

static struct efa_io_cdesc_common *efa_cq_sub_cq_get_cqe(struct efa_sub_cq *sub_cq, int entry)
{
	return (struct efa_io_cdesc_common *)(sub_cq->buf + (entry * sub_cq->cqe_size));
}

static void efa_cq_sub_cq_initialize(struct efa_sub_cq *sub_cq, uint8_t *buf,
				     int sub_cq_size, int cqe_size)
{
	sub_cq->consumed_cnt = 0;
	sub_cq->phase = 1;
	sub_cq->buf = buf;
	sub_cq->qmask = sub_cq_size - 1;
	sub_cq->cqe_size = cqe_size;
	sub_cq->ref_cnt = 0;
}

static int efa_cq_create(struct efa_cq *cq, struct efa_context *ctx, unsigned int cq_size)
{
	struct ibv_context *ibctx = &ctx->ibv_ctx;
	int err, sub_cq_size, sub_buf_size;
	uint64_t q_mmap_key, q_mmap_size;
	uint16_t i, num_sub_cqs;
	int fd = ibctx->cmd_fd;
	uint8_t *buf;
	uint32_t cqn;

	pthread_mutex_lock(&ibctx->mutex);

	cq->num_sub_cqs = ctx->sub_cqs_per_cq;
	cq->cqe_size    = ctx->cqe_size;

	cq_size = align_up_queue_size(cq_size);
	err = efa_cmd_create_cq(cq, cq_size, &q_mmap_key, &q_mmap_size, &cqn);
	if (err) {
		EFA_WARN(FI_LOG_CQ, "efa_cmd_create_cq failed[%u].\n", err);
		goto err_unlock;
	}

	cq->cqn = cqn;
	cq->buf_size = q_mmap_size;
	num_sub_cqs = cq->num_sub_cqs;
	sub_cq_size = cq->ibv_cq.cqe;

	err = fastlock_init(&cq->inner_lock);
	if (err) {
		err = -err;
		EFA_WARN(FI_LOG_CQ, "cq spin lock init failed[%d]!\n", err);
		goto err_destroy_cq;
	}

	cq->buf = mmap(NULL, cq->buf_size, PROT_WRITE, MAP_SHARED, fd, q_mmap_key);
	if (cq->buf == MAP_FAILED) {
		EFA_WARN(FI_LOG_CQ, "cq buffer mmap failed[%d]!\n", errno);
		err = -EINVAL;
		goto err_destroy_lock;
	}

	cq->sub_cq_arr = calloc(num_sub_cqs, sizeof(*cq->sub_cq_arr));
	if (!cq->sub_cq_arr) {
		err = -ENOMEM;
		EFA_WARN(FI_LOG_CQ, "sub cq allocation failed.\n");
		goto err_unmap_buf;
	}

	buf = cq->buf;
	sub_buf_size = cq->cqe_size * sub_cq_size;
	for (i = 0; i < num_sub_cqs; i++) {
		efa_cq_sub_cq_initialize(&cq->sub_cq_arr[i], buf, sub_cq_size, cq->cqe_size);
		buf += sub_buf_size;
	}

	pthread_mutex_unlock(&ibctx->mutex);
	return 0;

err_unmap_buf:
	munmap(cq->buf, cq->buf_size);
err_destroy_lock:
	fastlock_destroy(&cq->inner_lock);
err_destroy_cq:
	efa_cmd_destroy_cq(cq);
err_unlock:
	pthread_mutex_unlock(&ibctx->mutex);
	return err;
}

static int efa_cq_destroy(struct efa_cq *cq)
{
	int err;

	pthread_mutex_lock(&cq->domain->ctx->ibv_ctx.mutex);

	free(cq->sub_cq_arr);
	if (munmap(cq->buf, cq->buf_size))
		EFA_WARN(FI_LOG_CQ, "cq[%u]: buffer unmap failed!\n", cq->cqn);

	fastlock_destroy(&cq->inner_lock);
	err = efa_cmd_destroy_cq(cq);

	pthread_mutex_unlock(&cq->domain->ctx->ibv_ctx.mutex);

	return err;
}

static ssize_t efa_cq_readerr(struct fid_cq *cq_fid, struct fi_cq_err_entry *entry,
			      uint64_t flags)
{
	struct efa_cq *cq;
	struct efa_wce *wce;
	struct slist_entry *slist_entry;
	uint32_t api_version;

	cq = container_of(cq_fid, struct efa_cq, cq_fid);

	fastlock_acquire(&cq->outer_lock);
	if (slist_empty(&cq->wcq))
		goto err;

	wce = container_of(cq->wcq.head, struct efa_wce, entry);
	if (!wce->wc.comp_status)
		goto err;

	api_version = cq->domain->fab->util_fabric.fabric_fid.api_version;

	slist_entry = slist_remove_head(&cq->wcq);
	fastlock_release(&cq->outer_lock);

	wce = container_of(slist_entry, struct efa_wce, entry);

	entry->op_context = (void *)(uintptr_t)wce->wc.wr_id;
	entry->flags = wce->wc.flags;
	entry->err = EIO;
	entry->prov_errno = wce->wc.comp_status;

	/* We currently don't have err_data to give back to the user. */
	if (FI_VERSION_GE(api_version, FI_VERSION(1, 5)))
		entry->err_data_size = 0;

	ofi_buf_free(wce);
	return sizeof(*entry);
err:
	fastlock_release(&cq->outer_lock);
	return -FI_EAGAIN;
}

static void efa_cq_read_context_entry(struct efa_wc *wc, int i, void *buf)
{
	struct fi_cq_entry *entry = buf;

	entry[i].op_context = (void *)(uintptr_t)wc->wr_id;
}

static void efa_cq_read_msg_entry(struct efa_wc *wc, int i, void *buf)
{
	struct fi_cq_msg_entry *entry = buf;

	entry[i].op_context = (void *)(uintptr_t)wc->wr_id;
	entry[i].flags = wc->flags;
	entry[i].len = (uint64_t)wc->byte_len;
}

static void efa_cq_read_data_entry(struct efa_wc *wc, int i, void *buf)
{
	struct fi_cq_data_entry *entry = buf;

	entry[i].op_context = (void *)(uintptr_t)wc->wr_id;
	entry[i].flags = wc->flags;

	entry[i].data = (wc->flags & FI_REMOTE_CQ_DATA) ? ntohl(wc->imm_data) : 0;

	entry->len = (wc->flags & FI_RECV) ? wc->byte_len : 0;
}

static struct efa_io_cdesc_common *cq_next_sub_cqe_get(struct efa_sub_cq *sub_cq)
{
	struct efa_io_cdesc_common *cqe;
	__u32 current_index;
	int is_pending;

	current_index = efa_cq_sub_cq_get_current_index(sub_cq);
	cqe = efa_cq_sub_cq_get_cqe(sub_cq, current_index);
	is_pending = efa_cq_cqe_is_pending(cqe, sub_cq->phase);
	/* We need the rmb() to ensure that the rest of the completion
	* entry is only read after the phase bit has been validated.
	* We unconditionally call rmb rather than leave it in the for
	* loop to prevent the compiler from optimizing out loads of
	* the flag if the caller is in a tight loop.
	*/
	rmb();
	if (is_pending) {
		sub_cq->consumed_cnt++;
		if (efa_cq_sub_cq_get_current_index(sub_cq) == 0)
			sub_cq->phase = 1 - sub_cq->phase;
		return cqe;
	}

	return NULL;
}

static int efa_cq_poll_sub_cq(struct efa_cq *cq, struct efa_sub_cq *sub_cq,
			      struct efa_qp **cur_qp, struct efa_wc *wc)
{
	struct efa_context *ctx = to_efa_ctx(cq->ibv_cq.context);
	struct efa_io_cdesc_common *cqe;
	struct efa_wq *wq;
	uint32_t qpn, wrid_idx;

	cqe = cq_next_sub_cqe_get(sub_cq);
	if (!cqe)
		return -FI_EAGAIN;

	qpn = cqe->qp_num;
	if (!*cur_qp || (qpn != (*cur_qp)->qp_num)) {
		/* We do not have to take the QP table lock here,
		 * because CQs will be locked while QPs are removed
		 * from the table.
		 */
		*cur_qp = ctx->qp_table[qpn];
		if (!*cur_qp)
			return -FI_EOTHER;
	}

	wrid_idx = cqe->req_id;
	wc->comp_status = cqe->status;
	wc->flags = 0;
	if (get_efa_io_cdesc_common_q_type(cqe) == EFA_IO_SEND_QUEUE) {
		wq = &(*cur_qp)->sq.wq;
		wc->flags = FI_SEND | FI_MSG;
		wc->efa_ah = 0; /* AH report is valid for RX only */
		wc->src_qp = 0;
	} else {
		struct efa_io_rx_cdesc *rcqe =
			container_of(cqe, struct efa_io_rx_cdesc, common);

		wq = &(*cur_qp)->rq.wq;
		wc->byte_len = cqe->length;
		wc->flags = FI_RECV | FI_MSG;
		if (get_efa_io_cdesc_common_has_imm(cqe)) {
			wc->flags |= FI_REMOTE_CQ_DATA;
			wc->imm_data = rcqe->imm;
		}
		wc->efa_ah = rcqe->ah;
		wc->src_qp = rcqe->src_qp_num;
	}

	wc->qp = *cur_qp;
	wq->wrid_idx_pool_next--;
	wq->wrid_idx_pool[wq->wrid_idx_pool_next] = wrid_idx;
	wc->wr_id = wq->wrid[wrid_idx];
	wq->wqe_completed++;

	return FI_SUCCESS;
}

/* Must call with cq->outer_lock held */
ssize_t efa_poll_cq(struct efa_cq *cq, struct efa_wc *wc)
{
	uint16_t num_sub_cqs = cq->num_sub_cqs;
	struct efa_sub_cq *sub_cq;
	struct efa_qp *qp = NULL;
	int err = FI_SUCCESS;
	uint16_t sub_cq_idx;

	fastlock_acquire(&cq->inner_lock);
	for (sub_cq_idx = 0; sub_cq_idx < num_sub_cqs; ++sub_cq_idx) {
		sub_cq = &cq->sub_cq_arr[cq->next_poll_idx++];
		cq->next_poll_idx %= num_sub_cqs;

		if (!sub_cq->ref_cnt)
			continue;

		err = efa_cq_poll_sub_cq(cq, sub_cq, &qp, wc);
		if (err != -FI_EAGAIN)
			break;
	}
	fastlock_release(&cq->inner_lock);

	return err;
}

static ssize_t efa_cq_readfrom(struct fid_cq *cq_fid, void *buf, size_t count,
			       fi_addr_t *src_addr)
{
	struct efa_cq *cq;
	struct efa_wce *wce;
	struct slist_entry *entry;
	struct efa_wc wc;
	ssize_t ret = 0, i;

	cq = container_of(cq_fid, struct efa_cq, cq_fid);

	fastlock_acquire(&cq->outer_lock);

	for (i = 0; i < count; i++) {
		if (!slist_empty(&cq->wcq)) {
			wce = container_of(cq->wcq.head, struct efa_wce, entry);
			if (wce->wc.comp_status) {
				ret = -FI_EAVAIL;
				break;
			}
			entry = slist_remove_head(&cq->wcq);
			wce = container_of(entry, struct efa_wce, entry);
			cq->read_entry(&wce->wc, i, buf);
			ofi_buf_free(wce);
			continue;
		}

		ret = efa_poll_cq(cq, &wc);
		if (ret)
			break;

		/* Insert error entry into wcq */
		if (wc.comp_status) {
			wce = ofi_buf_alloc(cq->wce_pool);
			if (!wce) {
				fastlock_release(&cq->outer_lock);
				return -FI_ENOMEM;
			}
			memset(wce, 0, sizeof(*wce));
			memcpy(&wce->wc, &wc, sizeof(wc));
			slist_insert_tail(&wce->entry, &cq->wcq);
			ret = -FI_EAVAIL;
			break;
		}

		if (src_addr)
			src_addr[i] = efa_ah_qpn_to_addr(wc.qp->ep, wc.efa_ah,
							 wc.src_qp);
		cq->read_entry(&wc, i, buf);
	}

	fastlock_release(&cq->outer_lock);
	return i ? i : ret;
}

static ssize_t efa_cq_read(struct fid_cq *cq_fid, void *buf, size_t count)
{
	return efa_cq_readfrom(cq_fid, buf, count, NULL);
}

static const char *efa_cq_strerror(struct fid_cq *cq_fid,
				   int prov_errno,
				   const void *err_data,
				   char *buf, size_t len)
{
	static const char *const status_str[] = {
		[EFA_IO_COMP_STATUS_OK]                            = "Success",
		[EFA_IO_COMP_STATUS_FLUSHED]                       = "Flushed during qp destroy",
		[EFA_IO_COMP_STATUS_LOCAL_ERROR_QP_INTERNAL_ERROR] = "Internal qp error",
		[EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_OP_TYPE]   = "Invalid op type",
		[EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_AH]        = "Invalid ah",
		[EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY]      = "Invalid lkey",
		[EFA_IO_COMP_STATUS_LOCAL_ERROR_BAD_LENGTH]        = "Local message too long",
		[EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS]      = "Bad remote address",
		[EFA_IO_COMP_STATUS_REMOTE_ERROR_ABORT]            = "Remote aborted",
		[EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_DEST_QPN]     = "Bad dest qpn",
		[EFA_IO_COMP_STATUS_REMOTE_ERROR_RNR]              = "Destination rnr",
		[EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_LENGTH]       = "Remote message too long",
		[EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_STATUS]       = "Unexpected status by responder",
	};
	const char *strerr;

	if (prov_errno < EFA_IO_COMP_STATUS_OK ||
	    prov_errno > EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_STATUS ||
	    !status_str[prov_errno])
		strerr = "unknown error";
	else
		strerr = status_str[prov_errno];

	if (buf && len)
		strncpy(buf, strerr, len);
	return strerr;
}

static struct fi_ops_cq efa_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = efa_cq_read,
	.readfrom = efa_cq_readfrom,
	.readerr = efa_cq_readerr,
	.sread = fi_no_cq_sread,
	.sreadfrom = fi_no_cq_sreadfrom,
	.signal = fi_no_cq_signal,
	.strerror = efa_cq_strerror
};

static int efa_cq_control(fid_t fid, int command, void *arg)
{
	int ret = 0;

	switch (command) {
	default:
		ret = -FI_ENOSYS;
		break;
	}

	return ret;
}

static int efa_cq_close(fid_t fid)
{
	struct efa_cq *cq;
	struct efa_wce *wce;
	struct slist_entry *entry;
	int ret;

	cq = container_of(fid, struct efa_cq, cq_fid.fid);

	fastlock_acquire(&cq->outer_lock);
	while (!slist_empty(&cq->wcq)) {
		entry = slist_remove_head(&cq->wcq);
		wce = container_of(entry, struct efa_wce, entry);
		ofi_buf_free(wce);
	}
	fastlock_release(&cq->outer_lock);

	ofi_bufpool_destroy(cq->wce_pool);

	fastlock_destroy(&cq->outer_lock);

	ret = efa_cq_destroy(cq);
	if (ret)
		return ret;

	free(cq);
	return 0;
}

static struct fi_ops efa_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_cq_close,
	.bind = fi_no_bind,
	.control = efa_cq_control,
	.ops_open = fi_no_ops_open,
};

int efa_cq_open(struct fid_domain *domain_fid, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context)
{
	struct efa_cq *cq;
	size_t size;
	int ret;

	cq = calloc(1, sizeof(*cq));
	if (!cq)
		return -FI_ENOMEM;

	cq->domain = container_of(domain_fid, struct efa_domain,
				  util_domain.domain_fid);

	switch (attr->wait_obj) {
	case FI_WAIT_NONE:
		break;
	default:
		ret = -FI_ENOSYS;
		goto err_free_cq;
	}

	size = attr->size ? attr->size : EFA_DEF_CQ_SIZE;
	ret = efa_cq_create(cq, cq->domain->ctx, size);
	if (ret) {
		EFA_WARN(FI_LOG_CQ, "Unable to create CQ\n");
		goto err_free_cq;
	}

	ret = ofi_bufpool_create(&cq->wce_pool, sizeof(struct efa_wce), 16, 0,
				 EFA_WCE_CNT, 0);
	if (ret) {
		EFA_WARN(FI_LOG_CQ, "Failed to create wce_pool\n");
		goto err_destroy_cq;
	}

	cq->next_poll_idx = 0;
	cq->cq_fid.fid.fclass = FI_CLASS_CQ;
	cq->cq_fid.fid.context = context;
	cq->cq_fid.fid.ops = &efa_cq_fi_ops;
	cq->cq_fid.ops = &efa_cq_ops;

	switch (attr->format) {
	case FI_CQ_FORMAT_UNSPEC:
	case FI_CQ_FORMAT_CONTEXT:
		cq->read_entry = efa_cq_read_context_entry;
		cq->entry_size = sizeof(struct fi_cq_entry);
		break;
	case FI_CQ_FORMAT_MSG:
		cq->read_entry = efa_cq_read_msg_entry;
		cq->entry_size = sizeof(struct fi_cq_msg_entry);
		break;
	case FI_CQ_FORMAT_DATA:
		cq->read_entry = efa_cq_read_data_entry;
		cq->entry_size = sizeof(struct fi_cq_data_entry);
		break;
	case FI_CQ_FORMAT_TAGGED:
	default:
		ret = -FI_ENOSYS;
		goto err_destroy_pool;
	}

	fastlock_init(&cq->outer_lock);

	slist_init(&cq->wcq);

	*cq_fid = &cq->cq_fid;
	return 0;

err_destroy_pool:
	ofi_bufpool_destroy(cq->wce_pool);
err_destroy_cq:
	efa_cq_destroy(cq);
err_free_cq:
	free(cq);
	return ret;
}

void efa_cq_inc_ref_cnt(struct efa_cq *cq, uint8_t sub_cq_idx)
{
	cq->sub_cq_arr[sub_cq_idx].ref_cnt++;
}

void efa_cq_dec_ref_cnt(struct efa_cq *cq, uint8_t sub_cq_idx)
{
	cq->sub_cq_arr[sub_cq_idx].ref_cnt--;
}
