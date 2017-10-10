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

#include "verbs_dgram.h"

static inline
int fi_ibv_dgram_cq_cntr_comp(struct util_cq *util_cq,
			      struct util_cntr *util_cntr,
			      struct ibv_wc *wc)
{
	struct fi_cq_tagged_entry *comp;
	int ret = FI_SUCCESS;
	struct fi_ibv_dgram_wr_entry *wr_entry =
		(struct fi_ibv_dgram_wr_entry *)(uintptr_t)wc->wr_id;

	if (util_cntr)
		util_cntr->cntr_fid.ops->add(&util_cntr->cntr_fid, 1);

	fastlock_acquire(&util_cq->cq_lock);
	if (OFI_UNLIKELY(ofi_cirque_isfull(util_cq->cirq))) {
		VERBS_DBG(FI_LOG_CQ, "util_cq cirq is full!\n");
		ret = -FI_EAGAIN;
		goto out;
	}

	comp = ofi_cirque_tail(util_cq->cirq);
	comp->op_context = wr_entry->hdr.context;
	comp->flags = wr_entry->hdr.flags;
	comp->len = (uint64_t)wc->byte_len;
	comp->buf = NULL;
	if (wc->wc_flags & IBV_WC_WITH_IMM)
		comp->data = ntohl(wc->imm_data);
	ofi_cirque_commit(util_cq->cirq);
	fi_ibv_dgram_wr_entry_release(
		&wr_entry->hdr.ep->grh_pool,
		(struct fi_ibv_dgram_wr_entry_hdr *)wr_entry
	);
out:
	fastlock_release(&util_cq->cq_lock);
	return ret;
}

static inline
int fi_ibv_dgram_cq_cntr_report_error(struct util_cq *util_cq,
				      struct util_cntr *util_cntr,
				      struct ibv_wc *wc)
{
	struct fi_cq_err_entry err_entry = {
		.err 		= EIO,
		.prov_errno	= wc->status,
		.err_data_size	= sizeof(wc->vendor_err),
		.err_data	= (void *)&wc->vendor_err,
	};
	struct fi_cq_tagged_entry *comp;
	struct fi_ibv_dgram_wr_entry *wr_entry =
		(struct fi_ibv_dgram_wr_entry *)(uintptr_t)wc->wr_id;
	struct util_cq_err_entry *err = calloc(1, sizeof(*err));
	if (!err) {
		VERBS_WARN(FI_LOG_CQ, "Unable to allocate "
				      "util_cq_err_entry\n");
		goto out;
	}

	err_entry.op_context = wr_entry->hdr.context;
	err_entry.flags = wr_entry->hdr.flags;
	err->err_entry = err_entry;

	if (util_cntr)
		util_cntr->cntr_fid.ops->adderr(&util_cntr->cntr_fid, 1);

	fastlock_acquire(&util_cq->cq_lock);
	slist_insert_tail(&err->list_entry, &util_cq->err_list);

	/* Signal that there is err entry */
	comp = ofi_cirque_tail(util_cq->cirq);
	comp->flags = UTIL_FLAG_ERROR;
	ofi_cirque_commit(util_cq->cirq);

	fastlock_release(&util_cq->cq_lock);

out:
	fi_ibv_dgram_wr_entry_release(
		&wr_entry->hdr.ep->grh_pool,
		(struct fi_ibv_dgram_wr_entry_hdr *)wr_entry
	);

	return FI_SUCCESS;
}

int fi_ibv_dgram_rx_cq_comp(struct util_cq *util_cq,
			    struct util_cntr *util_cntr,
			    struct ibv_wc *wc)
{
	return fi_ibv_dgram_cq_cntr_comp(util_cq, util_cntr, wc);
}

int fi_ibv_dgram_tx_cq_comp(struct util_cq *util_cq,
			    struct util_cntr *util_cntr,
			    struct ibv_wc *wc)
{
	struct fi_ibv_dgram_wr_entry *wr_entry =
		(struct fi_ibv_dgram_wr_entry *)(uintptr_t)wc->wr_id;

	ofi_atomic_sub32(&wr_entry->hdr.ep->unsignaled_send_cnt,
			 wr_entry->hdr.ep->max_unsignaled_send_cnt);

	return fi_ibv_dgram_cq_cntr_comp(util_cq, util_cntr, wc);
}

int fi_ibv_dgram_tx_cq_report_error(struct util_cq *util_cq,
				    struct util_cntr *util_cntr,
				    struct ibv_wc *wc)
{
	struct fi_ibv_dgram_wr_entry *wr_entry =
		(struct fi_ibv_dgram_wr_entry *)(uintptr_t)wc->wr_id;

	ofi_atomic_sub32(&wr_entry->hdr.ep->unsignaled_send_cnt,
			 wr_entry->hdr.ep->max_unsignaled_send_cnt);

	return fi_ibv_dgram_cq_cntr_report_error(util_cq, util_cntr, wc);
}

int fi_ibv_dgram_rx_cq_report_error(struct util_cq *util_cq,
				    struct util_cntr *util_cntr,
				    struct ibv_wc *wc)
{
	return fi_ibv_dgram_cq_cntr_report_error(util_cq, util_cntr, wc);
}

int fi_ibv_dgram_tx_cq_no_action(struct util_cq *util_cq,
				 struct util_cntr *util_cntr,
				 struct ibv_wc *wc)
{
	struct fi_ibv_dgram_wr_entry *wr_entry =
		(struct fi_ibv_dgram_wr_entry *)(uintptr_t)wc->wr_id;
	

	ofi_atomic_sub32(&wr_entry->hdr.ep->unsignaled_send_cnt,
			 wr_entry->hdr.ep->max_unsignaled_send_cnt);

	fi_ibv_dgram_wr_entry_release(
		&wr_entry->hdr.ep->grh_pool,
		(struct fi_ibv_dgram_wr_entry_hdr *)wr_entry
	);

	return FI_SUCCESS;
}

int fi_ibv_dgram_rx_cq_no_action(struct util_cq *util_cq,
				 struct util_cntr *util_cntr,
				 struct ibv_wc *wc)
{
	struct fi_ibv_dgram_wr_entry *wr_entry =
		(struct fi_ibv_dgram_wr_entry *)(uintptr_t)wc->wr_id;

	fi_ibv_dgram_wr_entry_release(
		&wr_entry->hdr.ep->grh_pool,
		(struct fi_ibv_dgram_wr_entry_hdr *)wr_entry
	);

	return FI_SUCCESS;
}

static inline
void fi_ibv_dgram_cq_handle_wc(struct util_cq *util_cq,
			       struct util_cntr *util_cntr,
			       struct ibv_wc *wc)
{
	struct fi_ibv_dgram_wr_entry *wr_entry =
		(struct fi_ibv_dgram_wr_entry *)(uintptr_t)wc->wr_id;
	if (OFI_LIKELY(wc->status == IBV_WC_SUCCESS))
		wr_entry->hdr.suc_cb(util_cq, util_cntr, wc);
	else
		wr_entry->hdr.err_cb(util_cq, util_cntr, wc);
}

void fi_ibv_dgram_recv_cq_progress(struct util_ep *util_ep)
{
	struct fi_ibv_dgram_cq *cq;
	int num_ent, i;
	struct ibv_wc *wcs = alloca(fi_ibv_gl_data.cqread_bunch_size *
				    sizeof(struct ibv_wc));

	cq = container_of(&util_ep->rx_cq->cq_fid, struct fi_ibv_dgram_cq,
			  util_cq.cq_fid);

	num_ent = ibv_poll_cq(cq->ibv_cq,
			      fi_ibv_gl_data.cqread_bunch_size,
			      wcs);
	for (i = 0; i < num_ent; i++)
		fi_ibv_dgram_cq_handle_wc(util_ep->rx_cq,
					  util_ep->rx_cntr,
					  &wcs[i]);
}

void fi_ibv_dgram_send_cq_progress(struct util_ep *util_ep)
{
	struct fi_ibv_dgram_cq *cq;
	int num_ent, i;
	struct ibv_wc *wcs = alloca(fi_ibv_gl_data.cqread_bunch_size *
				    sizeof(struct ibv_wc));

	cq = container_of(&util_ep->tx_cq->cq_fid, struct fi_ibv_dgram_cq,
			  util_cq.cq_fid);

	num_ent = ibv_poll_cq(cq->ibv_cq,
			      fi_ibv_gl_data.cqread_bunch_size,
			      wcs);
	for (i = 0; i < num_ent; i++)
		fi_ibv_dgram_cq_handle_wc(util_ep->tx_cq,
					  util_ep->tx_cntr,
					  &wcs[i]);
}

void fi_ibv_dgram_send_recv_cq_progress(struct util_ep *util_ep)
{
	struct fi_ibv_dgram_cq *tx_cq, *rx_cq;
	int num_ent, i;
	struct ibv_wc *wcs = alloca(fi_ibv_gl_data.cqread_bunch_size *
				    sizeof(struct ibv_wc));

	tx_cq = container_of(&util_ep->tx_cq->cq_fid, struct fi_ibv_dgram_cq,
			     util_cq.cq_fid);
	rx_cq = container_of(&util_ep->rx_cq->cq_fid, struct fi_ibv_dgram_cq,
			     util_cq.cq_fid);

	/* Poll Transmit events */
	num_ent = ibv_poll_cq(tx_cq->ibv_cq,
			      fi_ibv_gl_data.cqread_bunch_size,
			      wcs);
	for (i = 0; i < num_ent; i++)
		fi_ibv_dgram_cq_handle_wc(util_ep->tx_cq,
					  util_ep->tx_cntr,
					  &wcs[i]);

	/* Poll Receive events */
	num_ent = ibv_poll_cq(rx_cq->ibv_cq,
			      fi_ibv_gl_data.cqread_bunch_size,
			      wcs);
	for (i = 0; i < num_ent; i++)
		fi_ibv_dgram_cq_handle_wc(util_ep->rx_cq,
					  util_ep->rx_cntr,
					  &wcs[i]);
}

static inline
const char *fi_ibv_dgram_cq_strerror(struct fid_cq *cq_fid, int prov_errno,
				     const void *err_data, char *buf, size_t len)
{
	if (buf && len)
		strncpy(buf, ibv_wc_status_str(prov_errno), len);
	return ibv_wc_status_str(prov_errno);
}

static int fi_ibv_dgram_cq_close(fid_t cq_fid)
{
	int ret = FI_SUCCESS;
	struct fi_ibv_dgram_cq *cq;
	struct fi_ibv_domain *domain;

	cq = container_of(cq_fid, struct fi_ibv_dgram_cq, util_cq.cq_fid.fid);
	if (!cq)
		return -FI_EINVAL;

	domain = container_of(cq->util_cq.domain, struct fi_ibv_domain,
			      util_domain.domain_fid);
	if (!domain)
		return -FI_EINVAL;

	ret = ofi_cq_cleanup(&cq->util_cq);
	if (ret)
		return ret;

	if (ibv_destroy_cq(cq->ibv_cq)) {
		VERBS_WARN(FI_LOG_CQ,
			   "unable to destroy completion queue "
			   "(errno %d)\n", errno);
		ret = -errno;
	}

	free(cq);

	return ret;
}

static struct fi_ops fi_ibv_dgram_fi_ops = {
	.size		= sizeof(fi_ibv_dgram_fi_ops),
	.close		= fi_ibv_dgram_cq_close,
	.bind		= fi_no_bind,
	.control	= fi_no_control,
	.ops_open	= fi_no_ops_open,
};

static struct fi_ops_cq fi_ibv_dgram_cq_ops = {
	.size		= sizeof(fi_ibv_dgram_cq_ops),
	.read		= ofi_cq_read,
	.readfrom	= ofi_cq_readfrom,
	.readerr	= ofi_cq_readerr,
	.sread		= ofi_cq_sread,
	.sreadfrom	= ofi_cq_sreadfrom,
	.signal		= ofi_cq_signal,
	.strerror	= fi_ibv_dgram_cq_strerror
};

int fi_ibv_dgram_cq_open(struct fid_domain *domain_fid, struct fi_cq_attr *attr,
			 struct fid_cq **cq_fid, void *context)
{
	struct fi_ibv_dgram_cq *cq;
	struct fi_ibv_domain *domain;
	int ret;
	size_t cq_size;

	cq = calloc(1, sizeof(*cq));
	if (!cq)
		return -FI_ENOMEM;

	domain = container_of(domain_fid, struct fi_ibv_domain,
			      util_domain.domain_fid);
	if (!domain || (domain->ep_type != FI_EP_DGRAM)) {
		ret = -FI_EINVAL;
		goto err1;
	}

	assert(domain->ep_type == FI_EP_DGRAM);

	ret = ofi_cq_init(&fi_ibv_prov, domain_fid, attr,
			  &cq->util_cq, &ofi_cq_progress,
			  context);
	if (ret)
		goto err1;

	cq_size = attr->size ?
		  attr->size : MIN(VERBS_DEF_CQ_SIZE,
				   domain->info->domain_attr->cq_cnt);

	cq->ibv_cq = ibv_create_cq(domain->verbs, cq_size, cq,
					   NULL, attr->signaling_vector);
	if (!cq->ibv_cq) {
		VERBS_WARN(FI_LOG_CQ,
			   "unable to create completion queue for "
			   "transsmission (errno %d)\n", errno);
		ret = -errno;
		goto err2;
	}

	*cq_fid = &cq->util_cq.cq_fid;
	(*cq_fid)->fid.ops = &fi_ibv_dgram_fi_ops;
	(*cq_fid)->ops = &fi_ibv_dgram_cq_ops;

	return ret;
err2:
	ofi_cq_cleanup(&cq->util_cq);
err1:
	free(cq);
	return ret;
}
