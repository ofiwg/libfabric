/*
 * Copyright (c) 2015 Cray Inc. All rights reserved.
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
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
#include <assert.h>

#include "gnix.h"
#include "gnix_nic.h"
#include "gnix_vc.h"
#include "gnix_ep.h"
#include "gnix_mr.h"
#include "gnix_cm_nic.h"
#include "gnix_mbox_allocator.h"
#include "gnix_cntr.h"

#include <gni_pub.h>

/* Threshold to switch from indirect transfer to chained transfer to move
 * unaligned read data. */
#define GNIX_RMA_UREAD_CHAINED_THRESH		60
#define GNI_READ_ALIGN				4
#define GNI_READ_ALIGN_MASK			(GNI_READ_ALIGN - 1)

static int __gnix_rma_send_err(struct gnix_fid_ep *ep,
			       struct gnix_fab_req *req)
{
	struct gnix_fid_cntr *cntr = NULL;
	int rc = FI_SUCCESS;
	uint64_t flags = req->flags & GNIX_RMA_COMPLETION_FLAGS;

	if (ep->send_cq) {
		rc = _gnix_cq_add_error(ep->send_cq, req->user_context,
					flags, 0, 0, 0, 0, 0, FI_ECANCELED,
					GNI_RC_TRANSACTION_ERROR, NULL);
		if (rc) {
			GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_cq_add_error() failed: %d\n", rc);
		}
	}

	if ((req->type == GNIX_FAB_RQ_RDMA_WRITE) &&
	    ep->write_cntr)
		cntr = ep->write_cntr;

	if ((req->type == GNIX_FAB_RQ_RDMA_READ) &&
	    ep->read_cntr)
		cntr = ep->read_cntr;

	if (cntr) {
		rc = _gnix_cntr_inc_err(cntr);
		if (rc)
			GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_cntr_inc_err() failed: %d\n", rc);
	}

	return rc;
}

static int __gnix_rma_send_completion(struct gnix_fid_ep *ep,
				      struct gnix_fab_req *req)
{
	struct gnix_fid_cntr *cntr = NULL;
	int rc = FI_SUCCESS;
	uint64_t flags = req->flags & GNIX_RMA_COMPLETION_FLAGS;

	if ((req->flags & FI_COMPLETION) && ep->send_cq) {
		rc = _gnix_cq_add_event(ep->send_cq, req->user_context,
					flags, 0, 0, 0, 0, FI_ADDR_NOTAVAIL);
		if (rc) {
			GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_cq_add_event() failed: %d\n", rc);
		}
	}

	if ((req->type == GNIX_FAB_RQ_RDMA_WRITE) &&
	    ep->write_cntr)
		cntr = ep->write_cntr;

	if ((req->type == GNIX_FAB_RQ_RDMA_READ) &&
	    ep->read_cntr)
		cntr = ep->read_cntr;

	if (cntr) {
		rc = _gnix_cntr_inc(cntr);
		if (rc)
			GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_cntr_inc() failed: %d\n", rc);
	}

	return FI_SUCCESS;
}

static void __gnix_rma_copy_indirect_get_data(struct gnix_fab_req *req)
{
	int head_off = req->rma.rem_addr & GNI_READ_ALIGN_MASK;

	memcpy((void *)req->rma.loc_addr,
	       req->rma.align_buf + head_off,
	       req->rma.len);
}

static void __gnix_rma_copy_chained_get_data(struct gnix_fab_req *req)
{
	int head_off, head_len, tail_len;
	void *addr;

	head_off = req->rma.rem_addr & GNI_READ_ALIGN_MASK;
	head_len = GNI_READ_ALIGN - head_off;
	tail_len = (req->rma.rem_addr + req->rma.len) & GNI_READ_ALIGN_MASK;

	if (head_off) {
		GNIX_INFO(0, "writing %d bytes to %p\n",
			  head_len, req->rma.loc_addr);
		memcpy((void *)req->rma.loc_addr,
		       req->rma.align_buf + head_off,
		       head_len);
	}

	if (tail_len) {
		addr = (void *)req->rma.loc_addr +
			       req->rma.len -
			       tail_len;

		GNIX_INFO(0, "writing %d bytes to %p\n", tail_len, addr);
		memcpy((void *)addr,
		       req->rma.align_buf + GNI_READ_ALIGN,
		       tail_len);
	}
}

static void __gnix_rma_fr_complete(struct gnix_fab_req *req,
				   struct gnix_tx_descriptor *txd)
{
	if (req->flags & FI_LOCAL_MR) {
		GNIX_INFO(FI_LOG_EP_DATA, "freeing auto-reg MR: %p\n",
			  req->rma.loc_md);
		fi_close(&req->rma.loc_md->mr_fid.fid);
	}

	atomic_dec(&req->vc->outstanding_tx_reqs);
	_gnix_nic_tx_free(req->vc->ep->nic, txd);

	/* Schedule VC TX queue in case the VC is 'fenced'. */
	_gnix_vc_tx_schedule(req->vc);

	_gnix_fr_free(req->vc->ep, req);
}

static int __gnix_rma_post_err(struct gnix_tx_descriptor *txd)
{
	struct gnix_fab_req *req = txd->req;
	int rc;

	req->tx_failures++;
	if (req->tx_failures < req->gnix_ep->domain->params.max_retransmits) {
		_gnix_nic_tx_free(req->gnix_ep->nic, txd);

		GNIX_INFO(FI_LOG_EP_DATA,
			  "Requeueing failed request: %p\n", req);
		return _gnix_vc_queue_work_req(req);
	}

	GNIX_INFO(FI_LOG_EP_DATA, "Failed %d transmits: %p\n",
		  req->tx_failures, req);
	rc = __gnix_rma_send_err(req->vc->ep, req);
	if (rc != FI_SUCCESS)
		GNIX_WARN(FI_LOG_EP_DATA,
			  "__gnix_rma_send_err() failed: %d\n",
			  rc);

	__gnix_rma_fr_complete(req, txd);
	return FI_SUCCESS;
}

/* __gnix_rma_txd_data_complete() should match __gnix_rma_txd_complete() except
 * for checking whether to send immediate data. */
static int __gnix_rma_txd_data_complete(void *arg, gni_return_t tx_status)
{
	struct gnix_tx_descriptor *txd = (struct gnix_tx_descriptor *)arg;
	struct gnix_fab_req *req = txd->req;
	int rc;

	if (tx_status != GNI_RC_SUCCESS) {
		return __gnix_rma_post_err(txd);
	}

	/* Successful data delivery.  Generate local completion. */
	rc = __gnix_rma_send_completion(req->gnix_ep, req);
	if (rc != FI_SUCCESS)
		GNIX_WARN(FI_LOG_EP_DATA,
			  "__gnix_rma_send_completion() failed: %d\n",
			  rc);

	__gnix_rma_fr_complete(req, txd);

	return FI_SUCCESS;
}

static int __gnix_rma_send_data_req(void *arg)
{
	struct gnix_fab_req *req = (struct gnix_fab_req *)arg;
	struct gnix_fid_ep *ep = req->gnix_ep;
	struct gnix_nic *nic = ep->nic;
	struct gnix_tx_descriptor *txd;
	gni_return_t status;
	int rc;
	int inject_err = _gnix_req_inject_err(req);

	rc = _gnix_nic_tx_alloc(nic, &txd);
	if (rc) {
		GNIX_INFO(FI_LOG_EP_DATA,
				"_gnix_nic_tx_alloc() failed: %d\n",
				rc);
		return -FI_ENOSPC;
	}

	txd->req = req;
	txd->completer_fn = __gnix_rma_txd_data_complete;

	txd->rma_data_hdr.flags = FI_RMA | FI_REMOTE_CQ_DATA;
	if (req->type == GNIX_FAB_RQ_RDMA_WRITE) {
		txd->rma_data_hdr.flags |= FI_REMOTE_WRITE;
	} else {
		txd->rma_data_hdr.flags |= FI_REMOTE_READ;
	}
	txd->rma_data_hdr.data = req->rma.imm;

	fastlock_acquire(&nic->lock);
	if (inject_err) {
		_gnix_nic_txd_err_inject(nic, txd);
		status = GNI_RC_SUCCESS;
	} else {
		status = GNI_SmsgSendWTag(req->vc->gni_ep,
					  &txd->rma_data_hdr,
					  sizeof(txd->rma_data_hdr),
					  NULL, 0, txd->id,
					  GNIX_SMSG_T_RMA_DATA);
	}
	fastlock_release(&nic->lock);

	if (status == GNI_RC_NOT_DONE) {
		_gnix_nic_tx_free(nic, txd);
		GNIX_INFO(FI_LOG_EP_DATA,
			  "GNI_SmsgSendWTag returned %s\n",
			  gni_err_str[status]);
	} else if (status != GNI_RC_SUCCESS) {
		_gnix_nic_tx_free(nic, txd);
		GNIX_WARN(FI_LOG_EP_DATA,
			  "GNI_SmsgSendWTag returned %s\n",
			  gni_err_str[status]);
	} else {
		GNIX_INFO(FI_LOG_EP_DATA, "Sent RMA CQ data, req: %p\n", req);
	}

	return gnixu_to_fi_errno(status);
}

static int __gnix_rma_txd_complete(void *arg, gni_return_t tx_status)
{
	struct gnix_tx_descriptor *txd = (struct gnix_tx_descriptor *)arg;
	struct gnix_fab_req *req = txd->req;
	int rc = FI_SUCCESS;

	/* Wait for both TXDs before processing RDMA chained requests. */
	if (req->flags & GNIX_RMA_CHAINED && req->flags & GNIX_RMA_RDMA) {
		req->rma.status |= tx_status;

		atomic_dec(&req->rma.outstanding_txds);
		if (atomic_get(&req->rma.outstanding_txds)) {
			_gnix_nic_tx_free(req->gnix_ep->nic, txd);
			GNIX_INFO(FI_LOG_EP_DATA,
				  "Received first RDMA chain TXD, req: %p\n",
				  req);
			return FI_SUCCESS;
		}

		tx_status = req->rma.status;
	}

	if (tx_status != GNI_RC_SUCCESS) {
		return __gnix_rma_post_err(txd);
	}

	/* Successful delivery.  Progress request. */
	if (req->flags & FI_REMOTE_CQ_DATA) {
		/* initiate immediate data transfer */
		req->tx_failures = 0;
		req->work_fn = __gnix_rma_send_data_req;
		_gnix_vc_queue_work_req(req);
	} else {
		if (req->flags & GNIX_RMA_INDIRECT) {
			__gnix_rma_copy_indirect_get_data(req);

			GNIX_INFO(FI_LOG_EP_DATA,
				  "freeing indirect align MR: %p addr: %p\n",
				  req->rma.align_md, req->rma.align_buf);
			fi_close(&req->rma.align_md->mr_fid.fid);
		} else if (req->flags & GNIX_RMA_CHAINED) {
			__gnix_rma_copy_chained_get_data(req);

			GNIX_INFO(FI_LOG_EP_DATA,
				  "freeing chained align MR: %p addr: %p\n",
				  req->rma.align_md, req->rma.align_buf);
			fi_close(&req->rma.align_md->mr_fid.fid);
		}

		/* complete request */
		rc = __gnix_rma_send_completion(req->vc->ep, req);
		if (rc != FI_SUCCESS)
			GNIX_WARN(FI_LOG_EP_DATA,
				  "__gnix_rma_send_completion() failed: %d\n",
				  rc);

		__gnix_rma_fr_complete(req, txd);
	}

	return FI_SUCCESS;
}

static gni_post_type_t __gnix_fr_post_type(int fr_type, int rdma)
{
	switch (fr_type) {
	case GNIX_FAB_RQ_RDMA_WRITE:
		return rdma ? GNI_POST_RDMA_PUT : GNI_POST_FMA_PUT;
	case GNIX_FAB_RQ_RDMA_READ:
		return rdma ? GNI_POST_RDMA_GET : GNI_POST_FMA_GET;
	default:
		break;
	}

	GNIX_WARN(FI_LOG_EP_DATA, "Unsupported post type: %d", fr_type);
	assert(0);
	return -FI_ENOSYS;
}

static void __gnix_rma_fill_pd_chained_get(struct gnix_fab_req *req,
					   struct gnix_tx_descriptor *txd,
					   gni_mem_handle_t *rem_mdh)
{
	int head_off, head_len, tail_len, desc_idx = 0;

	/* Copy head and tail through intermediate buffer.  Copy
	 * aligned data directly to user buffer. */
	head_off = req->rma.rem_addr & GNI_READ_ALIGN_MASK;
	head_len = head_off ? GNI_READ_ALIGN - head_off : 0;
	tail_len = (req->rma.rem_addr + req->rma.len) & GNI_READ_ALIGN_MASK;

	/* Use full post descriptor for aligned data */
	txd->gni_desc.local_addr = (uint64_t)req->rma.loc_addr + head_len;
	txd->gni_desc.remote_addr = (uint64_t)req->rma.rem_addr + head_len;
	txd->gni_desc.length = req->rma.len - head_len - tail_len;
	assert(txd->gni_desc.length);
	txd->gni_desc.next_descr = &txd->gni_ct_descs[0];

	if (head_off) {
		txd->gni_ct_descs[0].ep_hndl = req->vc->gni_ep;
		txd->gni_ct_descs[0].length = GNI_READ_ALIGN;
		txd->gni_ct_descs[0].remote_addr =
				req->rma.rem_addr & ~GNI_READ_ALIGN_MASK;
		txd->gni_ct_descs[0].remote_mem_hndl = *rem_mdh;
		txd->gni_ct_descs[0].local_addr =
				(uint64_t)req->rma.align_buf;
		txd->gni_ct_descs[0].local_mem_hndl =
				req->rma.align_md->mem_hndl;

		if (tail_len)
			txd->gni_ct_descs[0].next_descr =
					&txd->gni_ct_descs[1];
		else
			txd->gni_ct_descs[0].next_descr = NULL;

		desc_idx++;
	}

	if (tail_len) {
		txd->gni_ct_descs[desc_idx].ep_hndl = req->vc->gni_ep;
		txd->gni_ct_descs[desc_idx].length = GNI_READ_ALIGN;
		txd->gni_ct_descs[desc_idx].remote_addr =
				(req->rma.rem_addr +
				 req->rma.len) & ~GNI_READ_ALIGN_MASK;
		txd->gni_ct_descs[desc_idx].remote_mem_hndl = *rem_mdh;
		txd->gni_ct_descs[desc_idx].local_addr =
				(uint64_t)req->rma.align_buf + GNI_READ_ALIGN;
		txd->gni_ct_descs[desc_idx].local_mem_hndl =
				req->rma.align_md->mem_hndl;
		txd->gni_ct_descs[desc_idx].next_descr = NULL;
	}

	GNIX_INFO(0, "ct_rem_addr[0] = %p %p, ct_rem_addr[1] = %p %p\n",
		  txd->gni_ct_descs[0].remote_addr,
		  txd->gni_ct_descs[0].local_addr,
		  txd->gni_ct_descs[1].remote_addr,
		  txd->gni_ct_descs[1].local_addr);
}

static void __gnix_rma_fill_pd_indirect_get(struct gnix_fab_req *req,
					    struct gnix_tx_descriptor *txd)
{
	int head_off = req->rma.rem_addr & GNI_READ_ALIGN_MASK;

	/* Copy all data through an intermediate buffer. */
	txd->gni_desc.local_addr = (uint64_t)req->rma.align_buf;
	txd->gni_desc.local_mem_hndl = req->rma.align_md->mem_hndl;
	txd->gni_desc.length = CEILING(req->rma.len + head_off, GNI_READ_ALIGN);
	txd->gni_desc.remote_addr =
			(uint64_t)req->rma.rem_addr & ~GNI_READ_ALIGN_MASK;
}

int _gnix_rma_post_rdma_chain_req(void *data)
{
	struct gnix_fab_req *req = (struct gnix_fab_req *)data;
	struct gnix_fid_ep *ep = req->gnix_ep;
	struct gnix_nic *nic = ep->nic;
	struct gnix_tx_descriptor *bte_txd, *ct_txd;
	gni_mem_handle_t mdh;
	gni_return_t status;
	int rc;
	int inject_err = _gnix_req_inject_err(req);
	int head_off, head_len, tail_len;
	int fma_chain = 0;

	rc = _gnix_nic_tx_alloc(nic, &bte_txd);
	if (rc) {
		GNIX_INFO(FI_LOG_EP_DATA,
			  "BTE _gnix_nic_tx_alloc() failed: %d\n",
			  rc);
		return -FI_ENOSPC;
	}

	rc = _gnix_nic_tx_alloc(nic, &ct_txd);
	if (rc) {
		_gnix_nic_tx_free(nic, bte_txd);
		GNIX_INFO(FI_LOG_EP_DATA,
			  "CT _gnix_nic_tx_alloc() failed: %d\n",
			  rc);
		return -FI_ENOSPC;
	}

	_gnix_convert_key_to_mhdl(
			(gnix_mr_key_t *)&req->rma.rem_mr_key,
			&mdh);

	/* BTE TXD */
	bte_txd->completer_fn = __gnix_rma_txd_complete;
	bte_txd->req = req;
	bte_txd->gni_desc.type = GNI_POST_RDMA_GET;
	bte_txd->gni_desc.cq_mode = GNI_CQMODE_GLOBAL_EVENT; /* check flags */
	bte_txd->gni_desc.dlvr_mode = GNI_DLVMODE_PERFORMANCE; /* check flags */

	head_off = req->rma.rem_addr & GNI_READ_ALIGN_MASK;
	head_len = head_off ? GNI_READ_ALIGN - head_off : 0;
	tail_len = (req->rma.rem_addr + req->rma.len) & GNI_READ_ALIGN_MASK;

	bte_txd->gni_desc.local_addr = (uint64_t)req->rma.loc_addr + head_len;
	bte_txd->gni_desc.remote_addr = (uint64_t)req->rma.rem_addr + head_len;
	bte_txd->gni_desc.length = req->rma.len - head_len - tail_len;

	bte_txd->gni_desc.remote_mem_hndl = mdh;
	bte_txd->gni_desc.rdma_mode = 0; /* check flags */
	bte_txd->gni_desc.src_cq_hndl = nic->tx_cq; /* check flags */
	bte_txd->gni_desc.local_mem_hndl = req->rma.loc_md->mem_hndl;

	/* FMA TXD */
	ct_txd->completer_fn = __gnix_rma_txd_complete;
	ct_txd->req = req;
	ct_txd->gni_desc.type = GNI_POST_FMA_GET;
	ct_txd->gni_desc.cq_mode = GNI_CQMODE_GLOBAL_EVENT; /* check flags */
	ct_txd->gni_desc.dlvr_mode = GNI_DLVMODE_PERFORMANCE; /* check flags */

	ct_txd->gni_desc.remote_mem_hndl = mdh;
	ct_txd->gni_desc.rdma_mode = 0; /* check flags */
	ct_txd->gni_desc.src_cq_hndl = nic->tx_cq; /* check flags */
	ct_txd->gni_desc.local_mem_hndl = req->rma.align_md->mem_hndl;

	ct_txd->gni_desc.length = GNI_READ_ALIGN;
	ct_txd->gni_desc.local_mem_hndl = req->rma.align_md->mem_hndl;

	if (head_off) {
		ct_txd->gni_desc.remote_addr =
				req->rma.rem_addr & ~GNI_READ_ALIGN_MASK;
		ct_txd->gni_desc.local_addr = (uint64_t)req->rma.align_buf;

		if (tail_len) {
			ct_txd->gni_desc.next_descr = &ct_txd->gni_ct_descs[0];
			ct_txd->gni_ct_descs[0].ep_hndl = req->vc->gni_ep;
			ct_txd->gni_ct_descs[0].length = GNI_READ_ALIGN;
			ct_txd->gni_ct_descs[0].remote_addr =
					(req->rma.rem_addr +
					 req->rma.len) & ~GNI_READ_ALIGN_MASK;
			ct_txd->gni_ct_descs[0].remote_mem_hndl = mdh;
			ct_txd->gni_ct_descs[0].local_addr =
					(uint64_t)req->rma.align_buf +
					GNI_READ_ALIGN;
			ct_txd->gni_ct_descs[0].local_mem_hndl =
					req->rma.align_md->mem_hndl;
			ct_txd->gni_ct_descs[0].next_descr = NULL;
			fma_chain = 1;
		}
	} else {
		ct_txd->gni_desc.remote_addr =
				(req->rma.rem_addr +
				 req->rma.len) & ~GNI_READ_ALIGN_MASK;
		ct_txd->gni_desc.local_addr =
				(uint64_t)req->rma.align_buf + GNI_READ_ALIGN;
	}

	fastlock_acquire(&nic->lock);

	if (unlikely(inject_err)) {
		_gnix_nic_txd_err_inject(nic, bte_txd);
		status = GNI_RC_SUCCESS;
	} else {
		status = GNI_PostRdma(req->vc->gni_ep,
				      &bte_txd->gni_desc);
	}

	if (status != GNI_RC_SUCCESS) {
		fastlock_release(&nic->lock);
		_gnix_nic_tx_free(nic, ct_txd);
		_gnix_nic_tx_free(nic, bte_txd);

		GNIX_INFO(FI_LOG_EP_DATA, "GNI_Post*() failed: %s\n",
			  gni_err_str[status]);
		return gnixu_to_fi_errno(status);
	}

	if (unlikely(inject_err)) {
		_gnix_nic_txd_err_inject(nic, ct_txd);
		status = GNI_RC_SUCCESS;
	} else if (fma_chain) {
		status = GNI_CtPostFma(req->vc->gni_ep,
				       &ct_txd->gni_desc);
	} else {
		status = GNI_PostFma(req->vc->gni_ep,
				     &ct_txd->gni_desc);
	}

	if (status != GNI_RC_SUCCESS) {
		fastlock_release(&nic->lock);
		_gnix_nic_tx_free(nic, ct_txd);

		/* Wait for the first TX to complete, then retransmit the
		 * entire thing. */
		atomic_set(&req->rma.outstanding_txds, 1);
		req->rma.status = GNI_RC_TRANSACTION_ERROR;

		GNIX_INFO(FI_LOG_EP_DATA, "GNI_Post*() failed: %s\n",
			  gni_err_str[status]);
		return FI_SUCCESS;
	}

	fastlock_release(&nic->lock);

	/* Wait for both TXs to complete, then process the request. */
	atomic_set(&req->rma.outstanding_txds, 2);
	req->rma.status = 0;

	return FI_SUCCESS;
}

int _gnix_rma_post_req(void *data)
{
	struct gnix_fab_req *fab_req = (struct gnix_fab_req *)data;
	struct gnix_fid_ep *ep = fab_req->gnix_ep;
	struct gnix_nic *nic = ep->nic;
	struct gnix_fid_mem_desc *loc_md;
	struct gnix_tx_descriptor *txd;
	gni_mem_handle_t mdh;
	gni_return_t status;
	int rc;
	int rdma = !!(fab_req->flags & GNIX_RMA_RDMA);
	int indirect = !!(fab_req->flags & GNIX_RMA_INDIRECT);
	int chained = !!(fab_req->flags & GNIX_RMA_CHAINED);
	int inject_err = _gnix_req_inject_err(fab_req);

	rc = _gnix_nic_tx_alloc(nic, &txd);
	if (rc) {
		GNIX_INFO(FI_LOG_EP_DATA,
				"_gnix_nic_tx_alloc() failed: %d\n",
				rc);
		return -FI_ENOSPC;
	}

	txd->completer_fn = __gnix_rma_txd_complete;
	txd->req = fab_req;

	if (rdma) {
		_gnix_convert_key_to_mhdl(
				(gnix_mr_key_t *)&fab_req->rma.rem_mr_key,
				&mdh);
	} else {
		/* Mem handle CRC is not validated during FMA operations.  Skip
		 * this costly calculation. */
		_gnix_convert_key_to_mhdl_no_crc(
				(gnix_mr_key_t *)&fab_req->rma.rem_mr_key,
				&mdh);
	}

	txd->gni_desc.type = __gnix_fr_post_type(fab_req->type, rdma);
	txd->gni_desc.cq_mode = GNI_CQMODE_GLOBAL_EVENT; /* check flags */
	txd->gni_desc.dlvr_mode = GNI_DLVMODE_PERFORMANCE; /* check flags */

	if (indirect) {
		__gnix_rma_fill_pd_indirect_get(fab_req, txd);
	} else if (chained) {
		__gnix_rma_fill_pd_chained_get(fab_req, txd, &mdh);
	} else {
		txd->gni_desc.local_addr = (uint64_t)fab_req->rma.loc_addr;
		txd->gni_desc.length = fab_req->rma.len;
		txd->gni_desc.remote_addr = (uint64_t)fab_req->rma.rem_addr;

		loc_md = (struct gnix_fid_mem_desc *)fab_req->rma.loc_md;
		if (loc_md) {
			txd->gni_desc.local_mem_hndl = loc_md->mem_hndl;
		}
	}

	txd->gni_desc.remote_mem_hndl = mdh;
	txd->gni_desc.rdma_mode = 0; /* check flags */
	txd->gni_desc.src_cq_hndl = nic->tx_cq; /* check flags */

	{
		gni_mem_handle_t *tl_mdh = &txd->gni_desc.local_mem_hndl;
		gni_mem_handle_t *tr_mdh = &txd->gni_desc.remote_mem_hndl;
		GNIX_INFO(FI_LOG_EP_DATA, "la: %llx ra: %llx len: %d\n",
			  txd->gni_desc.local_addr, txd->gni_desc.remote_addr,
			  txd->gni_desc.length);
		GNIX_INFO(FI_LOG_EP_DATA, "lmdh: %llx:%llx rmdh: %llx:%llx key: %llx\n",
			  *(uint64_t *)tl_mdh, *(((uint64_t *)tl_mdh) + 1),
			  *(uint64_t *)tr_mdh, *(((uint64_t *)tr_mdh) + 1),
			  fab_req->rma.rem_mr_key);
	}

	fastlock_acquire(&nic->lock);

	if (unlikely(inject_err)) {
		_gnix_nic_txd_err_inject(nic, txd);
		status = GNI_RC_SUCCESS;
	} else if (chained) {
		status = GNI_CtPostFma(fab_req->vc->gni_ep, &txd->gni_desc);
	} else if (rdma) {
		status = GNI_PostRdma(fab_req->vc->gni_ep, &txd->gni_desc);
	} else {
		status = GNI_PostFma(fab_req->vc->gni_ep, &txd->gni_desc);
	}

	fastlock_release(&nic->lock);

	if (status != GNI_RC_SUCCESS) {
		_gnix_nic_tx_free(nic, txd);
		GNIX_INFO(FI_LOG_EP_DATA, "GNI_Post*() failed: %s\n",
			  gni_err_str[status]);
	}

	return gnixu_to_fi_errno(status);
}

/*
 * @brief Create an RMA request.
 *
 * Creates a new RMA request.  Reads and writes are supported.  GNI supports
 * writing to local and remote addresses with any alignment and length.  GNI
 * requires reads to use four byte aligned remote address and length.  For
 * reads smaller than one cacheline, aligned data is read into an intermediate
 * buffer, then partially copied to the user buffer (the code terms this an
 * 'INDIRECT' transfer).  For larger unaligned reads, the interior, aligned
 * portion of remote data is pulled directly into the user provided buffer.
 * The four bytes at the head and tail of an unaliged read are pulled into an
 * intermediate buffer, then partially copied into the user buffer.  This
 * method is termed a 'CHAINED' transfer in the code.  Unaligned reads smaller
 * than the RDMA threshold can perform these 3 distinct transactions (head,
 * middle, tail) in a single GNI chained FMA operation (resulting in a single
 * GNI CQE).  For unaligned reads larger than the RDMA threshold, two GNI posts
 * are used, one RDMA TX to transfer the bulk of the data, another FMA TX to
 * transfer the head and/or tail data.
 *
 * @param ep The endpiont to use for the RMA request.
 * @param fr_type RMA request type.
 * @param loc_addr Local address for the RMA request.
 * @param len Length of the RMA request.
 * @param mdesc Local memory descriptor for the RMA request.
 * @param dest_addr Remote endpiont address for the RMA request.
 * @param rem_addr Remote address for the RMA request.
 * @param mkey Remote memory key for the RMA request.
 * @param context Event context for the RMA request.
 * @param flags Flags for the RMA request
 * @param data Remote event data for the RMA request.
 *
 * @return FI_SUCCESS on success.  FI_EINVAL for invalid parameter.  -FI_ENOSPC
 *         for low memory.
 */

ssize_t _gnix_rma(struct gnix_fid_ep *ep, enum gnix_fab_req_type fr_type,
		  uint64_t loc_addr, size_t len, void *mdesc,
		  uint64_t dest_addr, uint64_t rem_addr, uint64_t mkey,
		  void *context, uint64_t flags, uint64_t data)
{
	struct gnix_vc *vc;
	struct gnix_fab_req *req;
	struct gnix_fid_mem_desc *md = NULL;
	int rc;
	int rdma;
	struct fid_mr *auto_mr = NULL, *align_mr = NULL;
	void *align_buf = NULL;

	if (!ep) {
		return -FI_EINVAL;
	}

	if ((flags & FI_INJECT) && (len > GNIX_INJECT_SIZE)) {
		GNIX_INFO(FI_LOG_EP_DATA,
			  "RMA length %d exceeds inject max size: %d\n",
			  len, GNIX_INJECT_SIZE);
		return -FI_EINVAL;
	}

	/* find VC for target */
	rc = _gnix_ep_get_vc(ep, dest_addr, &vc);
	if (rc) {
		GNIX_INFO(FI_LOG_EP_DATA,
			  "_gnix_ep_get_vc() failed, addr: %lx, rc:\n",
			  dest_addr, rc);
		return rc;
	}

	/* setup fabric request */
	req = _gnix_fr_alloc(ep);
	if (!req) {
		GNIX_INFO(FI_LOG_EP_DATA, "_gnix_fr_alloc() failed\n");
		return -FI_ENOSPC;
	}

	rdma = len >= ep->domain->params.rma_rdma_thresh;

	req->type = fr_type;
	req->gnix_ep = ep;
	req->vc = vc;
	req->user_context = context;
	req->work_fn = _gnix_rma_post_req;

	if (fr_type == GNIX_FAB_RQ_RDMA_READ &&
	    (rem_addr & GNI_READ_ALIGN_MASK || len & GNI_READ_ALIGN_MASK)) {
		/* Copy unaligned data through the inject buffer. */
		align_buf = req->inject_buf;
		rc = gnix_mr_reg(&ep->domain->domain_fid.fid, align_buf,
				GNIX_INJECT_SIZE, FI_READ, 0, 0, 0, &align_mr,
				NULL);
		if (rc != FI_SUCCESS) {
			GNIX_INFO(FI_LOG_EP_DATA,
				  "Failed to auto-register align buffer: %d\n",
				  rc);
			goto err_unalign_reg;
		}

		if (len >= GNIX_RMA_UREAD_CHAINED_THRESH) {
			GNIX_INFO(FI_LOG_EP_DATA,
				  "Using CT for unaligned GET, req: %p\n",
				  req);
			flags |= GNIX_RMA_CHAINED;
		} else {
			GNIX_INFO(FI_LOG_EP_DATA,
				  "Using tmp buf for unaligned GET, req: %p\n",
				  req);
			flags |= GNIX_RMA_INDIRECT;
		}

		if (rdma)
			req->work_fn = _gnix_rma_post_rdma_chain_req;

		GNIX_INFO(FI_LOG_EP_DATA, "align MR: %p\n", align_mr);
	}

	if (!(flags & GNIX_RMA_INDIRECT) && !mdesc &&
	    (rdma || fr_type == GNIX_FAB_RQ_RDMA_READ)) {
		/* We need to auto-register the source buffer. */
		rc = gnix_mr_reg(&ep->domain->domain_fid.fid, (void *)loc_addr,
				 len, FI_READ | FI_WRITE, 0, 0, 0, &auto_mr,
				 NULL);
		if (rc != FI_SUCCESS) {
			GNIX_INFO(FI_LOG_EP_DATA,
				  "Failed to auto-register local buffer: %d\n",
				  rc);
			goto err_auto_reg;
		}
		flags |= FI_LOCAL_MR;
		mdesc = (void *)auto_mr;
		GNIX_INFO(FI_LOG_EP_DATA, "auto-reg MR: %p\n", auto_mr);
	}

	if (mdesc)
		md = container_of(mdesc, struct gnix_fid_mem_desc, mr_fid);
	req->rma.loc_md = (void *)md;

	req->rma.rem_addr = rem_addr;
	req->rma.rem_mr_key = mkey;
	req->rma.len = len;
	req->rma.imm = data;
	req->flags = flags;

	req->rma.align_buf = align_buf;
	if (align_mr)
		md = container_of(align_mr, struct gnix_fid_mem_desc, mr_fid);
	req->rma.align_md = md;

	if (req->flags & FI_INJECT) {
		memcpy(req->inject_buf, (void *)loc_addr, len);
		req->rma.loc_addr = (uint64_t)req->inject_buf;
	} else {
		req->rma.loc_addr = loc_addr;
	}

	/* Inject interfaces always suppress completions.  If
	 * SELECTIVE_COMPLETION is set, honor any setting.  Otherwise, always
	 * deliver a completion. */
	if ((flags & GNIX_SUPPRESS_COMPLETION) ||
	    (ep->send_selective_completion && !(flags & FI_COMPLETION))) {
		req->flags &= ~FI_COMPLETION;
	} else {
		req->flags |= FI_COMPLETION;
	}

	if (rdma) {
		req->flags |= GNIX_RMA_RDMA;
	}

	GNIX_INFO(0, "Queuing (%p %p %d)\n",
		  (void *) loc_addr, (void *)rem_addr, len);

	return _gnix_vc_queue_tx_req(req);

err_auto_reg:
	if (align_mr)
		fi_close(&align_mr->fid);
err_unalign_reg:
	_gnix_fr_free(req->vc->ep, req);
	return rc;
}

