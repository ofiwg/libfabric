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
#include "gnix_cm_nic.h"
#include "gnix_nic.h"
#include "gnix_util.h"
#include "gnix_ep.h"
#include "gnix_hashtable.h"
#include "gnix_vc.h"
#include "gnix_cntr.h"
#include "gnix_av.h"
#include "gnix_rma.h"

#define INVALID_PEEK_FORMAT(fmt) \
	((fmt) == FI_CQ_FORMAT_CONTEXT || (fmt) == FI_CQ_FORMAT_MSG)

#define GNIX_TAGGED_PCD_COMPLETION_FLAGS	(FI_MSG | FI_RECV | FI_TAGGED)

/*******************************************************************************
 * helper functions
 ******************************************************************************/

static struct gnix_fab_req *__gnix_msg_dup_req(struct gnix_fab_req *req)
{
	struct gnix_fab_req *new_req;

	new_req = _gnix_fr_alloc(req->gnix_ep);
	if (new_req == NULL) {
		GNIX_WARN(FI_LOG_EP_DATA, "Failed to allocate request\n");
		return NULL;
	}

	/* TODO: selectively copy fields. */
	memcpy((void *)new_req, (void *)req, sizeof(*req));

	return new_req;
}

static void __gnix_msg_queues(struct gnix_fid_ep *ep,
			      int tagged,
			      fastlock_t **queue_lock,
			      struct gnix_tag_storage **posted_queue,
			      struct gnix_tag_storage **unexp_queue)
{
	if (tagged) {
		*queue_lock = &ep->tagged_queue_lock;
		*posted_queue = &ep->tagged_posted_recv_queue;
		*unexp_queue = &ep->tagged_unexp_recv_queue;
	} else {
		*queue_lock = &ep->recv_queue_lock;
		*posted_queue = &ep->posted_recv_queue;
		*unexp_queue = &ep->unexp_recv_queue;
	}
}

static void __gnix_msg_send_fr_complete(struct gnix_fab_req *req,
					struct gnix_tx_descriptor *txd)
{
	atomic_dec(&req->vc->outstanding_tx_reqs);
	_gnix_nic_tx_free(req->gnix_ep->nic, txd);

	/* Schedule VC TX queue in case the VC is 'fenced'. */
	_gnix_vc_tx_schedule(req->vc);

	_gnix_fr_free(req->gnix_ep, req);
}

static int __recv_err(struct gnix_fid_ep *ep, void *context, uint64_t flags,
		      size_t len, void *addr, uint64_t data, uint64_t tag,
		      size_t olen, int err, int prov_errno, void *err_data)
{
	int rc;

	if (ep->recv_cq) {
		rc = _gnix_cq_add_error(ep->recv_cq, context, flags, len,
					addr, data, tag, olen, err,
					prov_errno, err_data);
		if (rc != FI_SUCCESS)  {
			GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_cq_add_error returned %d\n",
				  rc);
		}
	}

	if (ep->recv_cntr) {
		rc = _gnix_cntr_inc_err(ep->recv_cntr);
		if (rc != FI_SUCCESS)
			GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_cntr_inc_err() failed: %d\n",
				  rc);
	}

	return FI_SUCCESS;
}

static int __gnix_msg_recv_err(struct gnix_fid_ep *ep, struct gnix_fab_req *req)
{
	uint64_t flags = FI_RECV | FI_MSG;

	flags |= req->msg.send_flags & FI_TAGGED;

	return __recv_err(ep, req->user_context, flags, req->msg.recv_len,
			  (void *)req->msg.recv_addr, req->msg.imm,
			  req->msg.tag, 0, FI_ECANCELED,
			  GNI_RC_TRANSACTION_ERROR, NULL);
}

static int __recv_completion(
		struct gnix_fid_ep *ep,
		struct gnix_fab_req *req,
		void *context,
		uint64_t flags,
		size_t len,
		void *addr,
		uint64_t data,
		uint64_t tag,
		fi_addr_t src_addr)
{
	int rc;

	if ((req->msg.recv_flags & FI_COMPLETION) && ep->recv_cq) {
		rc = _gnix_cq_add_event(ep->recv_cq, context, flags, len,
					addr, data, tag, src_addr);
		if (rc != FI_SUCCESS)  {
			GNIX_WARN(FI_LOG_EP_DATA,
					"_gnix_cq_add_event returned %d\n",
					rc);
		}
	}

	if (ep->recv_cntr) {
		rc = _gnix_cntr_inc(ep->recv_cntr);
		if (rc != FI_SUCCESS)
			GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_cntr_inc() failed: %d\n",
				  rc);
	}

	return FI_SUCCESS;
}

static inline int __gnix_msg_recv_completion(struct gnix_fid_ep *ep,
					     struct gnix_fab_req *req)
{
	uint64_t flags = FI_RECV | FI_MSG;

	flags |= req->msg.send_flags & (FI_TAGGED | FI_REMOTE_CQ_DATA);
	flags |= req->msg.recv_flags & (FI_PEEK | FI_CLAIM | FI_DISCARD |
					FI_MULTI_RECV);

	return __recv_completion(ep, req, req->user_context, flags,
				 req->msg.send_len, (void *)req->msg.recv_addr,
				 req->msg.imm, req->msg.tag,
				 _gnix_vc_peer_fi_addr(req->vc));
}

static int __gnix_msg_send_err(struct gnix_fid_ep *ep, struct gnix_fab_req *req)
{
	uint64_t flags = FI_SEND | FI_MSG;
	int rc;

	flags |= req->msg.send_flags & FI_TAGGED;

	if (ep->send_cq) {
		rc = _gnix_cq_add_error(ep->send_cq, req->user_context,
					flags, 0, 0, 0, 0, 0, FI_ECANCELED,
					GNI_RC_TRANSACTION_ERROR, NULL);
		if (rc != FI_SUCCESS)  {
			GNIX_WARN(FI_LOG_EP_DATA,
				   "_gnix_cq_add_error() returned %d\n",
				   rc);
		}
	}

	if (ep->send_cntr) {
		rc = _gnix_cntr_inc_err(ep->send_cntr);
		if (rc != FI_SUCCESS)
			GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_cntr_inc() failed: %d\n",
				  rc);
	}

	return FI_SUCCESS;
}

static int __gnix_msg_send_completion(struct gnix_fid_ep *ep,
				      struct gnix_fab_req *req)
{
	uint64_t flags = FI_SEND | FI_MSG;
	int rc;

	flags |= req->msg.send_flags & FI_TAGGED;

	if ((req->msg.send_flags & FI_COMPLETION) && ep->send_cq) {
		rc = _gnix_cq_add_event(ep->send_cq,
				req->user_context,
				flags, 0, 0, 0, 0, FI_ADDR_NOTAVAIL);
		if (rc != FI_SUCCESS)  {
			GNIX_WARN(FI_LOG_EP_DATA,
					"_gnix_cq_add_event returned %d\n",
					rc);
		}
	}

	if (ep->send_cntr) {
		rc = _gnix_cntr_inc(ep->send_cntr);
		if (rc != FI_SUCCESS)
			GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_cntr_inc() failed: %d\n",
				  rc);
	}

	return FI_SUCCESS;
}

static int __gnix_rndzv_req_send_fin(void *arg)
{
	struct gnix_fab_req *req = (struct gnix_fab_req *)arg;
	struct gnix_nic *nic;
	struct gnix_fid_ep *ep;
	struct gnix_tx_descriptor *txd;
	gni_return_t status;
	int rc;

	GNIX_TRACE(FI_LOG_EP_DATA, "\n");

	ep = req->gnix_ep;
	assert(ep != NULL);

	nic = ep->nic;
	assert(nic != NULL);

	rc = _gnix_nic_tx_alloc(nic, &txd);
	if (rc) {
		GNIX_INFO(FI_LOG_EP_DATA,
				"_gnix_nic_tx_alloc() failed: %d\n",
				rc);
		return -FI_ENOSPC;
	}

	txd->rndzv_fin_hdr.req_addr = req->msg.rma_id;

	txd->req = req;
	txd->completer_fn = gnix_ep_smsg_completers[GNIX_SMSG_T_RNDZV_FIN];

	fastlock_acquire(&nic->lock);
	status = GNI_SmsgSendWTag(req->vc->gni_ep,
			&txd->rndzv_fin_hdr, sizeof(txd->rndzv_fin_hdr),
			NULL, 0, txd->id, GNIX_SMSG_T_RNDZV_FIN);
	if ((status == GNI_RC_SUCCESS) &&
		(ep->domain->data_progress == FI_PROGRESS_AUTO))
		_gnix_rma_post_irq(req->vc);
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
	}

	GNIX_INFO(FI_LOG_EP_DATA, "Initiated RNDZV_FIN, req: %p\n", req);

	return gnixu_to_fi_errno(status);
}

static void __gnix_msg_copy_unaligned_get_data(struct gnix_fab_req *req)
{
	int head_off, head_len, tail_len;
	void *addr;

	head_off = req->msg.send_addr & GNI_READ_ALIGN_MASK;
	head_len = head_off ? GNI_READ_ALIGN - head_off : 0;
	tail_len = (req->msg.send_addr + req->msg.send_len) &
			GNI_READ_ALIGN_MASK;

	if (head_off) {
		addr = (void *)&req->msg.rndzv_head + head_off;

		GNIX_INFO(FI_LOG_EP_DATA,
			  "writing %d bytes to head (%p, 0x%x)\n",
			  head_len, req->msg.recv_addr, *(uint32_t *)addr);
		memcpy((void *)req->msg.recv_addr, addr, head_len);
	}

	if (tail_len) {
		addr = (void *)(req->msg.recv_addr +
				req->msg.send_len -
				tail_len);

		GNIX_INFO(FI_LOG_EP_DATA,
			  "writing %d bytes to tail (%p, 0x%x)\n",
			  tail_len, addr, req->msg.rndzv_tail);
		memcpy((void *)addr, &req->msg.rndzv_tail, tail_len);
	}
}

static int __gnix_rndzv_req_complete(void *arg, gni_return_t tx_status)
{
	struct gnix_tx_descriptor *txd = (struct gnix_tx_descriptor *)arg;
	struct gnix_fab_req *req = txd->req;
	int ret;

	if (req->msg.recv_flags & GNIX_MSG_GET_TAIL) {
		/* There are two TXDs involved with this request, an RDMA
		 * transfer to move the middle block and an FMA transfer to
		 * move unaligned tail data.  If this is the FMA TXD, store the
		 * unaligned bytes.  Bytes are copied from the request to the
		 * user buffer once both TXDs arrive. */
		if (txd->gni_desc.type == GNI_POST_FMA_GET)
			req->msg.rndzv_tail = *(uint32_t *)txd->int_buf;

		/* Remember any failure.  Retransmit both TXDs once both are
		 * complete. */
		req->msg.status |= tx_status;

		atomic_dec(&req->msg.outstanding_txds);
		if (atomic_get(&req->msg.outstanding_txds)) {
			_gnix_nic_tx_free(req->gnix_ep->nic, txd);
			GNIX_INFO(FI_LOG_EP_DATA,
				  "Received first RDMA chain TXD, req: %p\n",
				  req);
			return FI_SUCCESS;
		}

		tx_status = req->msg.status;
	}

	_gnix_nic_tx_free(req->gnix_ep->nic, txd);

	if (tx_status != GNI_RC_SUCCESS) {
		req->tx_failures++;
		if (req->tx_failures <
		    req->gnix_ep->domain->params.max_retransmits) {

			GNIX_INFO(FI_LOG_EP_DATA,
				  "Requeueing failed request: %p\n", req);
			return _gnix_vc_queue_work_req(req);
		}

		/* TODO should this be fatal? A request will sit waiting at the
		 * peer. */
		return __gnix_msg_recv_err(req->gnix_ep, req);
	}

	__gnix_msg_copy_unaligned_get_data(req);

	GNIX_INFO(FI_LOG_EP_DATA, "Completed RNDZV GET, req: %p\n", req);

	if (req->msg.recv_flags & FI_LOCAL_MR) {
		GNIX_INFO(FI_LOG_EP_DATA, "freeing auto-reg MR: %p\n",
			  req->msg.recv_md);
		fi_close(&req->msg.recv_md->mr_fid.fid);
	}

	req->work_fn = __gnix_rndzv_req_send_fin;
	ret = _gnix_vc_queue_work_req(req);

	return ret;
}

static int __gnix_rndzv_req(void *arg)
{
	struct gnix_fab_req *req = (struct gnix_fab_req *)arg;
	struct gnix_fid_ep *ep = req->gnix_ep;
	struct gnix_nic *nic = ep->nic;
	struct gnix_tx_descriptor *txd, *tail_txd = NULL;
	gni_return_t status;
	int rc;
	int use_tx_cq_blk = 0;
	struct fid_mr *auto_mr = NULL;
	int inject_err = _gnix_req_inject_err(req);
	int head_off, head_len, tail_len;
	void *tail_data = NULL;

	if (!req->msg.recv_md) {
		rc = gnix_mr_reg(&ep->domain->domain_fid.fid,
				 (void *)req->msg.recv_addr, req->msg.recv_len,
				 FI_READ | FI_WRITE, 0, 0, 0, &auto_mr, NULL);
		if (rc != FI_SUCCESS) {
			GNIX_INFO(FI_LOG_EP_DATA,
				  "Failed to auto-register local buffer: %d\n",
				  rc);
			return -FI_EAGAIN;
		}
		req->msg.recv_flags |= FI_LOCAL_MR;
		req->msg.recv_md = container_of(auto_mr,
						struct gnix_fid_mem_desc,
						mr_fid);
		GNIX_INFO(FI_LOG_EP_DATA, "auto-reg MR: %p\n", auto_mr);
	}

	rc = _gnix_nic_tx_alloc(nic, &txd);
	if (rc) {
		GNIX_INFO(FI_LOG_EP_DATA, "_gnix_nic_tx_alloc() failed: %d\n",
			  rc);
		return -FI_ENOSPC;
	}

	txd->completer_fn = __gnix_rndzv_req_complete;
	txd->req = req;

	use_tx_cq_blk = (ep->domain->data_progress == FI_PROGRESS_AUTO) ? 1 : 0;

	txd->gni_desc.type = GNI_POST_RDMA_GET;
	txd->gni_desc.cq_mode = GNI_CQMODE_GLOBAL_EVENT;
	txd->gni_desc.dlvr_mode = GNI_DLVMODE_PERFORMANCE;
	txd->gni_desc.local_mem_hndl = req->msg.recv_md->mem_hndl;
	txd->gni_desc.remote_mem_hndl = req->msg.rma_mdh;
	txd->gni_desc.rdma_mode = 0;
	txd->gni_desc.src_cq_hndl = (use_tx_cq_blk) ?
					nic->tx_cq_blk : nic->tx_cq;

	head_off = req->msg.send_addr & GNI_READ_ALIGN_MASK;
	head_len = head_off ? GNI_READ_ALIGN - head_off : 0;
	tail_len = (req->msg.send_addr + req->msg.send_len) &
			GNI_READ_ALIGN_MASK;

	txd->gni_desc.local_addr = (uint64_t)req->msg.recv_addr + head_len;
	txd->gni_desc.remote_addr = (uint64_t)req->msg.send_addr + head_len;
	txd->gni_desc.length = req->msg.send_len - head_len - tail_len;

	if (req->msg.recv_flags & GNIX_MSG_GET_TAIL) {
		/* The user ended up with a send matching a receive with a
		 * buffer that is too short and unaligned... what a way to
		 * behave.  We could not have forseen which unaligned data to
		 * send across with the rndzv_start request, so we do an extra
		 * TX here to pull the random unaligned bytes. */
		rc = _gnix_nic_tx_alloc(nic, &tail_txd);
		if (rc) {
			_gnix_nic_tx_free(nic, txd);
			GNIX_INFO(FI_LOG_EP_DATA,
				  "_gnix_nic_tx_alloc() failed (tail): %d\n",
				  rc);
			return -FI_ENOSPC;
		}

		tail_txd->completer_fn = __gnix_rndzv_req_complete;
		tail_txd->req = req;

		tail_data = (void *)((req->msg.send_addr + req->msg.send_len) &
				      ~GNI_READ_ALIGN_MASK);

		tail_txd->gni_desc.type = GNI_POST_FMA_GET;
		tail_txd->gni_desc.cq_mode = GNI_CQMODE_GLOBAL_EVENT;
		tail_txd->gni_desc.dlvr_mode = GNI_DLVMODE_PERFORMANCE;
		tail_txd->gni_desc.local_mem_hndl = nic->int_bufs_mdh;
		tail_txd->gni_desc.remote_mem_hndl = req->msg.rma_mdh;
		tail_txd->gni_desc.rdma_mode = 0;
		tail_txd->gni_desc.src_cq_hndl = nic->tx_cq;
		tail_txd->gni_desc.local_addr = (uint64_t)tail_txd->int_buf;
		tail_txd->gni_desc.remote_addr = (uint64_t)tail_data;
		tail_txd->gni_desc.length = GNI_READ_ALIGN;

		GNIX_INFO(FI_LOG_EP_DATA, "Using two GETs\n");
	}

	fastlock_acquire(&nic->lock);

	if (inject_err) {
		_gnix_nic_txd_err_inject(nic, txd);
		status = GNI_RC_SUCCESS;
	} else {
		status = GNI_PostRdma(req->vc->gni_ep, &txd->gni_desc);
	}

	if (status != GNI_RC_SUCCESS) {
		fastlock_release(&nic->lock);
		if (tail_txd)
			_gnix_nic_tx_free(nic, tail_txd);
		_gnix_nic_tx_free(nic, txd);
		GNIX_INFO(FI_LOG_EP_DATA, "GNI_PostRdma failed: %s\n",
			  gni_err_str[status]);
		return gnixu_to_fi_errno(status);
	}

	if (req->msg.recv_flags & GNIX_MSG_GET_TAIL) {
		if (unlikely(inject_err)) {
			_gnix_nic_txd_err_inject(nic, tail_txd);
			status = GNI_RC_SUCCESS;
		} else {
			status = GNI_PostFma(req->vc->gni_ep,
					     &tail_txd->gni_desc);
		}

		if (status != GNI_RC_SUCCESS) {
			fastlock_release(&nic->lock);
			_gnix_nic_tx_free(nic, tail_txd);

			/* Wait for the first TX to complete, then retransmit
			 * the entire thing. */
			atomic_set(&req->msg.outstanding_txds, 1);
			req->msg.status = GNI_RC_TRANSACTION_ERROR;

			GNIX_INFO(FI_LOG_EP_DATA, "GNI_PostFma() failed: %s\n",
				  gni_err_str[status]);
			return FI_SUCCESS;
		}

		/* Wait for both TXs to complete, then process the request. */
		atomic_set(&req->msg.outstanding_txds, 2);
		req->msg.status = 0;

	}

	fastlock_release(&nic->lock);

	GNIX_INFO(FI_LOG_EP_DATA, "Initiated RNDZV GET, req: %p\n", req);

	return gnixu_to_fi_errno(status);
}

/*******************************************************************************
 * GNI SMSG callbacks invoked upon completion of an SMSG message at the sender.
 ******************************************************************************/

static int __comp_eager_msg_w_data(void *data, gni_return_t tx_status)
{
	struct gnix_tx_descriptor *tdesc = (struct gnix_tx_descriptor *)data;
	struct gnix_fab_req *req = tdesc->req;
	int ret = FI_SUCCESS;

	if (tx_status != GNI_RC_SUCCESS) {
		GNIX_INFO(FI_LOG_EP_DATA, "Failed transaction: %p\n", req);
		ret = __gnix_msg_send_err(req->gnix_ep, req);
		if (ret != FI_SUCCESS)
			GNIX_WARN(FI_LOG_EP_DATA,
				  "__gnix_msg_send_err() failed: %d\n",
				  ret);
	} else {
		/* Successful delivery.  Generate completions. */
		ret = __gnix_msg_send_completion(req->gnix_ep, req);
		if (ret != FI_SUCCESS)
			GNIX_WARN(FI_LOG_EP_DATA,
				  "__gnix_msg_send_completion() failed: %d\n",
				  ret);
	}

	__gnix_msg_send_fr_complete(req, tdesc);

	return FI_SUCCESS;
}

static int __comp_eager_msg_w_data_ack(void *data, gni_return_t tx_status)
{
	return -FI_ENOSYS;
}

static int __comp_eager_msg_data_at_src(void *data, gni_return_t tx_status)
{
	return -FI_ENOSYS;
}

static int __comp_eager_msg_data_at_src_ack(void *data, gni_return_t tx_status)
{
	return -FI_ENOSYS;
}

static int __comp_rndzv_msg_rts(void *data, gni_return_t tx_status)
{
	return -FI_ENOSYS;
}

static int __comp_rndzv_msg_rtr(void *data, gni_return_t tx_status)
{
	return -FI_ENOSYS;
}

static int __comp_rndzv_msg_cookie(void *data, gni_return_t tx_status)
{
	return -FI_ENOSYS;
}

static int __comp_rndzv_msg_send_done(void *data, gni_return_t tx_status)
{
	return -FI_ENOSYS;
}

static int __comp_rndzv_msg_recv_done(void *data, gni_return_t tx_status)
{
	return -FI_ENOSYS;
}

/* Completed request to start rendezvous send. */
static int __comp_rndzv_start(void *data, gni_return_t tx_status)
{
	struct gnix_tx_descriptor *txd = (struct gnix_tx_descriptor *)data;
	struct gnix_fab_req *req = txd->req;
	int ret;

	if (tx_status != GNI_RC_SUCCESS) {
		GNIX_INFO(FI_LOG_EP_DATA, "Failed transaction: %p\n", txd->req);
		ret = __gnix_msg_send_err(req->gnix_ep, req);
		if (ret != FI_SUCCESS)
			GNIX_WARN(FI_LOG_EP_DATA,
				  "__gnix_msg_send_err() failed: %d\n",
				  ret);
		__gnix_msg_send_fr_complete(req, txd);
	} else {
		/* Just free the TX descriptor for now.  The request remains
		 * active until the remote peer notifies us that they're done
		 * with the send buffer. */
		_gnix_nic_tx_free(txd->req->gnix_ep->nic, txd);

		GNIX_INFO(FI_LOG_EP_DATA, "Completed RNDZV_START, req: %p\n",
			  txd->req);
	}

	return FI_SUCCESS;
}

/* Notified sender that rendezvous data has been moved.  Rendezvous send
 * complete.  Generate Completions. */
static int __comp_rndzv_fin(void *data, gni_return_t tx_status)
{
	int ret = FI_SUCCESS;
	struct gnix_tx_descriptor *tdesc = (struct gnix_tx_descriptor *)data;
	struct gnix_fab_req *req = tdesc->req;

	if (tx_status != GNI_RC_SUCCESS) {
		/* TODO should this be fatal? A request will sit waiting at the
		 * peer. */
		GNIX_WARN(FI_LOG_EP_DATA, "Failed transaction: %p\n", req);
		ret = __gnix_msg_recv_err(req->gnix_ep, req);
		if (ret != FI_SUCCESS)
			GNIX_WARN(FI_LOG_EP_DATA,
					"__gnix_msg_recv_err() failed: %d\n",
					ret);
	} else {
		GNIX_INFO(FI_LOG_EP_DATA, "Completed RNDZV_FIN, req: %p\n",
			  req);

		ret = __gnix_msg_recv_completion(req->gnix_ep, req);
		if (ret != FI_SUCCESS)
			GNIX_WARN(FI_LOG_EP_DATA,
				  "__gnix_msg_recv_completion() failed: %d\n",
				  ret);
	}

	_gnix_nic_tx_free(req->gnix_ep->nic, tdesc);
	_gnix_fr_free(req->gnix_ep, req);

	return FI_SUCCESS;
}

smsg_completer_fn_t gnix_ep_smsg_completers[] = {
	[GNIX_SMSG_T_EGR_W_DATA] = __comp_eager_msg_w_data,
	[GNIX_SMSG_T_EGR_W_DATA_ACK] = __comp_eager_msg_w_data_ack,
	[GNIX_SMSG_T_EGR_GET] = __comp_eager_msg_data_at_src,
	[GNIX_SMSG_T_EGR_GET_ACK] = __comp_eager_msg_data_at_src_ack,
	[GNIX_SMSG_T_RNDZV_RTS] = __comp_rndzv_msg_rts,
	[GNIX_SMSG_T_RNDZV_RTR] = __comp_rndzv_msg_rtr,
	[GNIX_SMSG_T_RNDZV_COOKIE] = __comp_rndzv_msg_cookie,
	[GNIX_SMSG_T_RNDZV_SDONE] = __comp_rndzv_msg_send_done,
	[GNIX_SMSG_T_RNDZV_RDONE] = __comp_rndzv_msg_recv_done,
	[GNIX_SMSG_T_RNDZV_START] = __comp_rndzv_start,
	[GNIX_SMSG_T_RNDZV_FIN] = __comp_rndzv_fin,
};


/*******************************************************************************
 * GNI SMSG callbacks invoked upon receipt of an SMSG message.
 * These callback functions are invoked with the lock for the nic
 * associated with the vc already held.
 ******************************************************************************/

/*
 * Handle SMSG message with tag GNIX_SMSG_T_EGR_W_DATA
 */

static int __smsg_eager_msg_w_data(void *data, void *msg)
{
	int ret = FI_SUCCESS;
	gni_return_t status;
	struct gnix_vc *vc = (struct gnix_vc *)data;
	struct gnix_smsg_eager_hdr *hdr = (struct gnix_smsg_eager_hdr *)msg;
	struct gnix_fid_ep *ep;
	struct gnix_fab_req *req = NULL;
	void *data_ptr;
	struct gnix_tag_storage *unexp_queue;
	struct gnix_tag_storage *posted_queue;
	fastlock_t *queue_lock;
	int tagged;

	ep = vc->ep;
	assert(ep);

	data_ptr = (void *)((char *)msg + sizeof(*hdr));

	tagged = !!(hdr->flags & FI_TAGGED);
	__gnix_msg_queues(ep, tagged, &queue_lock, &posted_queue, &unexp_queue);

	fastlock_acquire(queue_lock);

	/* Lookup a matching posted request. */
	req = _gnix_match_tag(posted_queue, hdr->msg_tag, 0, FI_PEEK, NULL,
			      &vc->peer_addr);
	if (req) {
		req->addr = vc->peer_addr;
		req->gnix_ep = ep;
		req->vc = vc;

		req->msg.send_len = MIN(hdr->len, req->msg.recv_len);
		req->msg.send_flags = hdr->flags;
		req->msg.tag = hdr->msg_tag;
		req->msg.imm = hdr->imm;

		GNIX_INFO(FI_LOG_EP_DATA, "Matched req: %p (%p, %u)\n",
			  req, req->msg.recv_addr, req->msg.send_len);

		memcpy((void *)req->msg.recv_addr, data_ptr, req->msg.send_len);
		__gnix_msg_recv_completion(ep, req);

		/* Check if we're using FI_MULTI_RECV and there is space left
		 * in the receive buffer. */
		if ((req->msg.recv_flags & FI_MULTI_RECV) &&
		    ((req->msg.recv_len - req->msg.send_len) >=
		     ep->min_multi_recv)) {
			GNIX_INFO(FI_LOG_EP_DATA, "Re-using req: %p\n", req);

			/* Adjust receive buffer for the next match. */
			req->msg.recv_addr += req->msg.send_len;
			req->msg.recv_len -= req->msg.send_len;
		} else {
			GNIX_INFO(FI_LOG_EP_DATA, "Freeing req: %p\n", req);

			/* Dequeue and free the request. */
			req = _gnix_match_tag(posted_queue, hdr->msg_tag, 0, 0,
					      NULL, &vc->peer_addr);
			_gnix_fr_free(ep, req);
		}
	} else {
		/* Add new unexpected receive request. */
		req = _gnix_fr_alloc(ep);
		if (req == NULL) {
			fastlock_release(queue_lock);
			return -FI_ENOMEM;
		}

		req->msg.send_addr = (uint64_t)malloc(hdr->len);
		if (unlikely(req->msg.send_addr == 0ULL)) {
			fastlock_release(queue_lock);
			_gnix_fr_free(ep, req);
			return -FI_ENOMEM;
		}

		req->type = GNIX_FAB_RQ_RECV;
		req->addr = vc->peer_addr;
		req->gnix_ep = ep;
		req->vc = vc;

		req->msg.send_len = hdr->len;
		req->msg.send_flags = hdr->flags;
		req->msg.tag = hdr->msg_tag;
		req->msg.imm = hdr->imm;

		memcpy((void *)req->msg.send_addr, data_ptr, hdr->len);
		req->addr = vc->peer_addr;

		_gnix_insert_tag(unexp_queue, req->msg.tag, req, ~0);

		GNIX_INFO(FI_LOG_EP_DATA, "New req: %p (%u)\n",
			  req, req->msg.send_len);
	}

	fastlock_release(queue_lock);

	status = GNI_SmsgRelease(vc->gni_ep);
	if (unlikely(status != GNI_RC_SUCCESS)) {
		GNIX_WARN(FI_LOG_EP_DATA,
				"GNI_SmsgRelease returned %s\n",
				gni_err_str[status]);
		ret = gnixu_to_fi_errno(status);
	}

	return ret;
}

/*
 * this function will probably not be used unless we need
 * some kind of explicit flow control to handle unexpected
 * receives
 */

static int __smsg_eager_msg_w_data_ack(void *data, void *msg)
{
	return -FI_ENOSYS;
}

/*
 * Handle SMSG message with tag GNIX_SMSG_T_EGR_GET
 */
static int __smsg_eager_msg_data_at_src(void *data, void *msg)
{
	return -FI_ENOSYS;
}

/*
 * Handle SMSG message with tag GNIX_SMSG_T_EGR_GET_ACK
 */
static int  __smsg_eager_msg_data_at_src_ack(void *data, void *msg)
{
	return -FI_ENOSYS;
}

static int __smsg_rndzv_msg_rts(void *data, void *msg)
{
	return -FI_ENOSYS;
}

static int __smsg_rndzv_msg_rtr(void *data, void *msg)
{
	return -FI_ENOSYS;
}

static int __smsg_rndzv_msg_cookie(void *data, void *msg)
{
	return -FI_ENOSYS;
}

static int __smsg_rndzv_msg_send_done(void *data, void *msg)
{
	return -FI_ENOSYS;
}

static int __smsg_rndzv_msg_recv_done(void *data, void *msg)
{
	return -FI_ENOSYS;
}

/* Received SMSG rendezvous start message.  Try to match a posted receive and
 * start pulling data. */
static int __smsg_rndzv_start(void *data, void *msg)
{
	int ret = FI_SUCCESS;
	gni_return_t status;
	struct gnix_vc *vc = (struct gnix_vc *)data;
	struct gnix_smsg_rndzv_start_hdr *hdr =
			(struct gnix_smsg_rndzv_start_hdr *)msg;
	struct gnix_fid_ep *ep;
	struct gnix_fab_req *req = NULL, *dup_req;
	struct gnix_tag_storage *unexp_queue;
	struct gnix_tag_storage *posted_queue;
	fastlock_t *queue_lock;
	int tagged;

	ep = vc->ep;
	assert(ep);

	tagged = !!(hdr->flags & FI_TAGGED);
	__gnix_msg_queues(ep, tagged, &queue_lock, &posted_queue, &unexp_queue);

	fastlock_acquire(queue_lock);

	req = _gnix_match_tag(posted_queue, hdr->msg_tag, 0, FI_PEEK, NULL,
			      &vc->peer_addr);
	if (req) {
		req->addr = vc->peer_addr;
		req->gnix_ep = ep;
		req->vc = vc;
		req->tx_failures = 0;

		/* Check if a second GET for unaligned data is needed. */
		if (hdr->len > req->msg.recv_len &&
		    ((hdr->addr + req->msg.recv_len) & GNI_READ_ALIGN_MASK)) {
			req->msg.recv_flags |= GNIX_MSG_GET_TAIL;
		}

		req->msg.send_addr = hdr->addr;
		req->msg.send_len = MIN(hdr->len, req->msg.recv_len);
		req->msg.send_flags = hdr->flags;
		req->msg.tag = hdr->msg_tag;
		req->msg.imm = hdr->imm;
		req->msg.rma_mdh = hdr->mdh;
		req->msg.rma_id = hdr->req_addr;
		req->msg.rndzv_head = hdr->head;
		req->msg.rndzv_tail = hdr->tail;

		GNIX_INFO(FI_LOG_EP_DATA, "Matched req: %p (%p, %u)\n",
			  req, req->msg.recv_addr, req->msg.send_len);

		/* Check if we're using FI_MULTI_RECV and there is space left
		 * in the receive buffer. */
		if ((req->msg.recv_flags & FI_MULTI_RECV) &&
		    ((req->msg.recv_len - req->msg.send_len) >=
		     ep->min_multi_recv)) {
			/* Allocate new request for this transfer. */
			dup_req = __gnix_msg_dup_req(req);
			if (!dup_req) {
				fastlock_release(queue_lock);
				return -FI_ENOMEM;
			}

			/* Adjust receive buffer for the next match. */
			req->msg.recv_addr += req->msg.send_len;
			req->msg.recv_len -= req->msg.send_len;

			/* 'req' remains queued for more matches while the
			 * duplicated request is processed. */
			req = dup_req;
		} else {
			/* Dequeue the request. */
			req = _gnix_match_tag(posted_queue, hdr->msg_tag, 0, 0,
					      NULL, &vc->peer_addr);
		}

		/* Queue request to initiate pull of source data. */
		req->work_fn = __gnix_rndzv_req;
		ret = _gnix_vc_queue_work_req(req);
	} else {
		/* Add new unexpected receive request. */
		req = _gnix_fr_alloc(ep);
		if (req == NULL) {
			fastlock_release(queue_lock);
			return -FI_ENOMEM;
		}

		req->type = GNIX_FAB_RQ_RECV;
		req->addr = vc->peer_addr;
		req->gnix_ep = ep;
		req->vc = vc;

		req->msg.send_addr = hdr->addr;
		req->msg.send_len = hdr->len;
		req->msg.send_flags = hdr->flags;
		req->msg.tag = hdr->msg_tag;
		req->msg.imm = hdr->imm;
		req->msg.rma_mdh = hdr->mdh;
		req->msg.rma_id = hdr->req_addr;
		req->msg.rndzv_head = hdr->head;
		req->msg.rndzv_tail = hdr->tail;
		atomic_initialize(&req->msg.outstanding_txds, 0);

		_gnix_insert_tag(unexp_queue, req->msg.tag, req, ~0);

		GNIX_INFO(FI_LOG_EP_DATA, "New req: %p (%u)\n",
			  req, req->msg.send_len);
	}

	fastlock_release(queue_lock);

	status = GNI_SmsgRelease(vc->gni_ep);
	if (unlikely(status != GNI_RC_SUCCESS)) {
		GNIX_WARN(FI_LOG_EP_DATA,
			  "GNI_SmsgRelease returned %s\n",
			  gni_err_str[status]);
		ret = gnixu_to_fi_errno(status);
	}

	return ret;
}

static int __gnix_rndzv_fin_cleanup(void *arg)
{
	struct gnix_fab_req *req = (struct gnix_fab_req *)arg;

	GNIX_INFO(FI_LOG_EP_DATA, "freeing auto-reg MR: %p\n",
		  req->msg.send_md);
	fi_close(&req->msg.send_md->mr_fid.fid);
	_gnix_fr_free(req->gnix_ep, req);

	return FI_SUCCESS;
}

/* Received SMSG rendezvous fin message.  The peer has finished pulling send
 * data.  Free the send request and generate completions. */
static int __smsg_rndzv_fin(void *data, void *msg)
{
	int ret = FI_SUCCESS;
	gni_return_t status;
	struct gnix_vc *vc = (struct gnix_vc *)data;
	struct gnix_smsg_rndzv_fin_hdr *hdr =
			(struct gnix_smsg_rndzv_fin_hdr *)msg;
	struct gnix_fab_req *req;
	struct gnix_fid_ep *ep;

	req = (struct gnix_fab_req *)hdr->req_addr;
	assert(req);

	GNIX_INFO(FI_LOG_EP_DATA, "Received RNDZV_FIN, req: %p\n", req);

	ep = req->gnix_ep;
	assert(ep != NULL);

	__gnix_msg_send_completion(ep, req);

	atomic_dec(&req->vc->outstanding_tx_reqs);

	/* Schedule VC TX queue in case the VC is 'fenced'. */
	_gnix_vc_tx_schedule(req->vc);

	if (req->msg.send_flags & FI_LOCAL_MR) {
		/* Defer freeing the MR and request. */
		req->work_fn = __gnix_rndzv_fin_cleanup;
		ret = _gnix_vc_queue_work_req(req);
	} else {
		_gnix_fr_free(ep, req);
	}

	status = GNI_SmsgRelease(vc->gni_ep);
	if (unlikely(status != GNI_RC_SUCCESS)) {
		GNIX_WARN(FI_LOG_EP_DATA,
				"GNI_SmsgRelease returned %s\n",
				gni_err_str[status]);
		ret = gnixu_to_fi_errno(status);
	}

	return ret;
}

/* TODO: This is kind of out of place. */
static int __smsg_rma_data(void *data, void *msg)
{
	int ret = FI_SUCCESS;
	struct gnix_vc *vc = (struct gnix_vc *)data;
	struct gnix_smsg_rma_data_hdr *hdr =
			(struct gnix_smsg_rma_data_hdr *)msg;
	struct gnix_fid_ep *ep = vc->ep;
	gni_return_t status;

	if (ep->recv_cq) {
		ret = _gnix_cq_add_event(ep->recv_cq, NULL, hdr->flags, 0,
					 0, hdr->data, 0, FI_ADDR_NOTAVAIL);
		if (ret != FI_SUCCESS)  {
			GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_cq_add_event returned %d\n",
				  ret);
		}
	}

	status = GNI_SmsgRelease(vc->gni_ep);
	if (unlikely(status != GNI_RC_SUCCESS)) {
		GNIX_WARN(FI_LOG_EP_DATA,
			  "GNI_SmsgRelease returned %s\n",
			  gni_err_str[status]);
		ret = gnixu_to_fi_errno(status);
	}

	return ret;
}

smsg_callback_fn_t gnix_ep_smsg_callbacks[] = {
	[GNIX_SMSG_T_EGR_W_DATA] = __smsg_eager_msg_w_data,
	[GNIX_SMSG_T_EGR_W_DATA_ACK] = __smsg_eager_msg_w_data_ack,
	[GNIX_SMSG_T_EGR_GET] = __smsg_eager_msg_data_at_src,
	[GNIX_SMSG_T_EGR_GET_ACK] = __smsg_eager_msg_data_at_src_ack,
	[GNIX_SMSG_T_RNDZV_RTS] = __smsg_rndzv_msg_rts,
	[GNIX_SMSG_T_RNDZV_RTR] = __smsg_rndzv_msg_rtr,
	[GNIX_SMSG_T_RNDZV_COOKIE] = __smsg_rndzv_msg_cookie,
	[GNIX_SMSG_T_RNDZV_SDONE] = __smsg_rndzv_msg_send_done,
	[GNIX_SMSG_T_RNDZV_RDONE] = __smsg_rndzv_msg_recv_done,
	[GNIX_SMSG_T_RNDZV_START] = __smsg_rndzv_start,
	[GNIX_SMSG_T_RNDZV_FIN] = __smsg_rndzv_fin,
	[GNIX_SMSG_T_RMA_DATA] = __smsg_rma_data
};

static int __gnix_peek_request(struct gnix_fab_req *req)
{
	struct gnix_fid_cq *recv_cq = req->gnix_ep->recv_cq;
	int rendezvous = !!(req->msg.send_flags & GNIX_MSG_RENDEZVOUS);
	int ret;

	/* All claim work is performed by the tag storage, so nothing special
	 * here.  If no CQ, no data is to be returned.  Just inform the user
	 * that a message is present. */
	GNIX_INFO(FI_LOG_EP_DATA, "peeking req=%p\n", req);
	if (!recv_cq)
		return FI_SUCCESS;

	/* Rendezvous messages on the unexpected queue won't have data.
	 * Additionally, if the CQ format doesn't support passing a buffer
	 * location and length, then data will not be copied. */
	if (!rendezvous && req->msg.recv_addr &&
	    !INVALID_PEEK_FORMAT(recv_cq->attr.format)) {
		int copy_len = MIN(req->msg.send_len, req->msg.recv_len);

		memcpy((void *)req->msg.recv_addr, (void *)req->msg.send_addr,
		       copy_len);
	} else {
		/* The CQE should not contain a valid buffer. */
		req->msg.recv_addr = 0;
	}

	ret = __gnix_msg_recv_completion(req->gnix_ep, req);
	if (ret != FI_SUCCESS)
		GNIX_WARN(FI_LOG_EP_DATA,
			  "__gnix_msg_recv_completion() failed: %d\n",
			  ret);

	return ret;
}

static int __gnix_discard_request(struct gnix_fab_req *req)
{
	int ret = FI_SUCCESS;
	int rendezvous = !!(req->msg.send_flags & GNIX_MSG_RENDEZVOUS);

	/* The CQE should not contain a valid buffer. */
	req->msg.recv_addr = 0;
	req->msg.send_len = 0;

	GNIX_INFO(FI_LOG_EP_DATA, "discarding req=%p\n", req);
	if (rendezvous) {
		GNIX_INFO(FI_LOG_EP_DATA,
			  "returning rndzv completion for req, %p", req);

		/* Complete rendezvous request, skipping data transfer. */
		req->work_fn = __gnix_rndzv_req_send_fin;
		ret = _gnix_vc_queue_work_req(req);
	} else {
		/* Data has already been delivered, so just discard it and
		 * generate a CQE. */
		ret = __gnix_msg_recv_completion(req->gnix_ep, req);
		if (ret != FI_SUCCESS)
			GNIX_WARN(FI_LOG_EP_DATA,
				  "__gnix_msg_recv_completion() failed: %d\n",
				  ret);

		/* Free unexpected eager receive buffer. */
		free((void *)req->msg.send_addr);
		_gnix_fr_free(req->gnix_ep, req);
	}

	return ret;
}

static int __gnix_msg_addr_lookup(struct gnix_fid_ep *ep, uint64_t src_addr,
				  struct gnix_address *gnix_addr)
{
	int ret;
	struct gnix_fid_av *av;
	struct gnix_av_addr_entry *av_entry;

	/* Translate source address. */
	if (ep->type == FI_EP_RDM) {
		if ((ep->caps & FI_DIRECTED_RECV) &&
		    (src_addr != FI_ADDR_UNSPEC)) {
			av = ep->av;
			assert(av != NULL);
			ret = _gnix_av_lookup(av, src_addr, &av_entry);
			if (ret != FI_SUCCESS) {
				GNIX_WARN(FI_LOG_AV,
					  "_gnix_av_lookup returned %d\n",
					  ret);
				return ret;
			}
			*gnix_addr = av_entry->gnix_addr;
		} else {
			*(uint64_t *)gnix_addr = FI_ADDR_UNSPEC;
		}
	} else {
		assert(ep->vc != NULL);
		*gnix_addr = ep->vc->peer_addr;
	}

	return FI_SUCCESS;
}

/*******************************************************************************
 * Generic EP recv handling
 ******************************************************************************/

ssize_t _gnix_recv(struct gnix_fid_ep *ep, uint64_t buf, size_t len,
		   void *mdesc, uint64_t src_addr, void *context,
		   uint64_t flags, uint64_t tag, uint64_t ignore)
{
	int ret;
	struct gnix_fab_req *req = NULL;
	struct gnix_address gnix_addr;
	fastlock_t *queue_lock = NULL;
	struct gnix_tag_storage *posted_queue = NULL;
	struct gnix_tag_storage *unexp_queue = NULL;
	uint64_t match_flags;
	struct gnix_fid_mem_desc *md = NULL;
	int tagged = !!(flags & FI_TAGGED);

	match_flags = flags & (FI_CLAIM | FI_DISCARD | FI_PEEK);

	ret = __gnix_msg_addr_lookup(ep, src_addr, &gnix_addr);
	if (ret != FI_SUCCESS)
		return ret;

	__gnix_msg_queues(ep, tagged, &queue_lock, &posted_queue, &unexp_queue);

	if (!tagged) {
		tag = 0;
		ignore = ~0;
	}

	fastlock_acquire(queue_lock);

retry_match:
	/* Look for a matching unexpected receive request. */
	req = _gnix_match_tag(unexp_queue, tag, ignore,
			      match_flags, context, &gnix_addr);
	if (req) {
		/* Found matching request, populate local fields. */
		req->gnix_ep = ep;
		req->user_context = context;

		req->msg.recv_addr = (uint64_t)buf;
		req->msg.recv_len = len;
		if (mdesc) {
			md = container_of(mdesc,
					struct gnix_fid_mem_desc,
					mr_fid);
		}
		req->msg.recv_md = md;
		req->msg.recv_flags = flags;
		req->msg.ignore = ignore;

		if ((flags & GNIX_SUPPRESS_COMPLETION) ||
		    (ep->recv_selective_completion &&
		    !(flags & FI_COMPLETION))) {
			req->msg.recv_flags &= ~FI_COMPLETION;
		} else {
			req->msg.recv_flags |= FI_COMPLETION;
		}

		/* Check to see if we are using P/C/D matching flags. */
		if (match_flags & FI_DISCARD) {
			ret = __gnix_discard_request(req);
			goto pdc_exit;
		} else if (match_flags & FI_PEEK) {
			ret = __gnix_peek_request(req);
			goto pdc_exit;
		}

		if (req->msg.send_flags & GNIX_MSG_RENDEZVOUS) {
			/* Matched rendezvous request.  Start data movement. */
			GNIX_INFO(FI_LOG_EP_DATA, "matched RNDZV, req: %p\n",
				  req);

			/* TODO: prevent re-lookup of src_addr */
			ret = _gnix_ep_get_vc(ep, src_addr, &req->vc);
			if (ret) {
				GNIX_INFO(FI_LOG_EP_DATA,
					  "_gnix_ep_get_vc failed: %dn",
					  ret);
				return ret;
			}

			/* Check if second GET for unaligned data is needed. */
			if (req->msg.send_len > req->msg.recv_len &&
			    ((req->msg.send_addr + req->msg.recv_len) &
			     GNI_READ_ALIGN_MASK)) {
				req->msg.recv_flags |= GNIX_MSG_GET_TAIL;
			}

			/* Send length is truncated to receive buffer size. */
			req->msg.send_len = MIN(req->msg.send_len,
						req->msg.recv_len);

			/* Initiate pull of source data. */
			req->work_fn = __gnix_rndzv_req;
			ret = _gnix_vc_queue_work_req(req);

			/* If using FI_MULTI_RECV and there is space left in
			 * the receive buffer, try to match another unexpected
			 * request. */
			if ((req->msg.recv_flags & FI_MULTI_RECV) &&
			    ((len - req->msg.send_len) >= ep->min_multi_recv)) {
				buf += req->msg.send_len;
				len -= req->msg.send_len;

				GNIX_INFO(FI_LOG_EP_DATA,
					  "Attempting additional matches, "
					  "req: %p (%p %u)\n",
					  req, buf, len);
				goto retry_match;
			}
		} else {
			/* Matched eager request.  Copy data and generate
			 * completions. */
			GNIX_INFO(FI_LOG_EP_DATA, "Matched recv, req: %p\n",
				  req);

			/* Send length is truncated to receive buffer size. */
			req->msg.send_len = MIN(req->msg.send_len,
						req->msg.recv_len);

			/* Copy data from unexpected eager receive buffer. */
			memcpy((void *)buf, (void *)req->msg.send_addr,
			       req->msg.send_len);
			free((void *)req->msg.send_addr);

			__gnix_msg_recv_completion(ep, req);

			/* If using FI_MULTI_RECV and there is space left in
			 * the receive buffer, try to match another unexpected
			 * request. */
			if ((req->msg.recv_flags & FI_MULTI_RECV) &&
			    ((len - req->msg.send_len) >= ep->min_multi_recv)) {
				buf += req->msg.send_len;
				len -= req->msg.send_len;

				GNIX_INFO(FI_LOG_EP_DATA,
					  "Attempting additional matches, "
					  "req: %p (%p %u)\n",
					  req, buf, len);
				goto retry_match;
			}

			_gnix_fr_free(ep, req);
		}
	} else {
		/* if peek/claim/discard, we didn't find what we
		 * were looking for, return FI_ENOMSG
		 */
		if (match_flags) {
			__recv_err(ep, context, flags, len,
				   (void *)buf, 0, tag, len, FI_ENOMSG,
				   FI_ENOMSG, NULL);

			/* if handling trecvmsg flags, return here
			 * Never post a receive request from this type of context
			 */
			ret = FI_SUCCESS;
			goto pdc_exit;
		}

		/* Add new posted receive request. */
		req = _gnix_fr_alloc(ep);
		if (req == NULL) {
			ret = -FI_EAGAIN;
			goto err;
		}

		GNIX_INFO(FI_LOG_EP_DATA, "New recv, req: %p\n", req);

		req->type = GNIX_FAB_RQ_RECV;

		req->addr = gnix_addr;
		req->gnix_ep = ep;
		req->user_context = context;

		req->msg.recv_addr = (uint64_t)buf;
		req->msg.recv_len = len;
		if (mdesc) {
			md = container_of(mdesc,
					struct gnix_fid_mem_desc,
					mr_fid);
		}
		req->msg.recv_md = md;
		req->msg.recv_flags = flags;
		req->msg.tag = tag;
		req->msg.ignore = ignore;
		atomic_initialize(&req->msg.outstanding_txds, 0);

		if ((flags & GNIX_SUPPRESS_COMPLETION) ||
		    (ep->recv_selective_completion &&
		    !(flags & FI_COMPLETION))) {
			req->msg.recv_flags &= ~FI_COMPLETION;
		} else {
			req->msg.recv_flags |= FI_COMPLETION;
		}

		_gnix_insert_tag(posted_queue, tag, req, ignore);
	}

pdc_exit:
err:
	fastlock_release(queue_lock);

	return ret;
}

/*******************************************************************************
 * Generic EP send handling
 ******************************************************************************/

static int _gnix_send_req(void *arg)
{
	struct gnix_fab_req *req = (struct gnix_fab_req *)arg;
	struct gnix_nic *nic;
	struct gnix_fid_ep *ep;
	struct gnix_tx_descriptor *tdesc;
	gni_return_t status;
	int rc;
	int rendezvous = !!(req->msg.send_flags & GNIX_MSG_RENDEZVOUS);
	int hdr_len, data_len;
	void *hdr, *data;
	int tag;
	int inject_err = _gnix_req_inject_smsg_err(req);

	ep = req->gnix_ep;
	assert(ep != NULL);

	nic = ep->nic;
	assert(nic != NULL);

	rc = _gnix_nic_tx_alloc(nic, &tdesc);
	if (rc != FI_SUCCESS) {
		GNIX_INFO(FI_LOG_EP_DATA, "_gnix_nic_tx_alloc() failed: %d\n",
			  rc);
		return -FI_ENOSPC;
	}
	assert(rc == FI_SUCCESS);

	if (rendezvous) {
		assert(req->msg.send_md);

		tag = GNIX_SMSG_T_RNDZV_START;
		tdesc->rndzv_start_hdr.flags = req->msg.send_flags;
		tdesc->rndzv_start_hdr.imm = req->msg.imm;
		tdesc->rndzv_start_hdr.msg_tag = req->msg.tag;
		tdesc->rndzv_start_hdr.mdh = req->msg.send_md->mem_hndl;
		tdesc->rndzv_start_hdr.addr = req->msg.send_addr;
		tdesc->rndzv_start_hdr.len = req->msg.send_len;
		tdesc->rndzv_start_hdr.req_addr = (uint64_t)req;

		if (req->msg.send_addr & GNI_READ_ALIGN_MASK) {
			tdesc->rndzv_start_hdr.head =
				*(uint32_t *)(req->msg.send_addr &
					      ~GNI_READ_ALIGN_MASK);
			GNIX_INFO(FI_LOG_EP_DATA,
				  "Sending %d unaligned head bytes (%x)\n",
				  GNI_READ_ALIGN -
				  (req->msg.send_addr & GNI_READ_ALIGN_MASK),
				  tdesc->rndzv_start_hdr.head);
		}

		if ((req->msg.send_addr + req->msg.send_len) &
		    GNI_READ_ALIGN_MASK) {
			tdesc->rndzv_start_hdr.tail =
				*(uint32_t *)((req->msg.send_addr +
					       req->msg.send_len) &
					      ~GNI_READ_ALIGN_MASK);
			GNIX_INFO(FI_LOG_EP_DATA,
				  "Sending %d unaligned tail bytes (%x)\n",
				  (req->msg.send_addr + req->msg.send_len) &
				  GNI_READ_ALIGN_MASK,
				  tdesc->rndzv_start_hdr.tail);
		}

		hdr = &tdesc->rndzv_start_hdr;
		hdr_len = sizeof(tdesc->rndzv_start_hdr);
		data = NULL;
		data_len = 0;
	} else {
		tag = GNIX_SMSG_T_EGR_W_DATA;
		tdesc->eager_hdr.flags = req->msg.send_flags;
		tdesc->eager_hdr.imm = req->msg.imm;
		tdesc->eager_hdr.msg_tag = req->msg.tag;
		tdesc->eager_hdr.len = req->msg.send_len;

		hdr = &tdesc->eager_hdr;
		hdr_len = sizeof(tdesc->eager_hdr);
		data = (void *)req->msg.send_addr;
		data_len = req->msg.send_len;
	}
	tdesc->req = req;
	tdesc->completer_fn = gnix_ep_smsg_completers[tag];

	fastlock_acquire(&nic->lock);

	if (inject_err) {
		_gnix_nic_txd_err_inject(nic, tdesc);
		status = GNI_RC_SUCCESS;
	} else {
		status = GNI_SmsgSendWTag(req->vc->gni_ep,
					  hdr, hdr_len, data, data_len,
					  tdesc->id, tag);
	}

	/*
 	 * if this is a rendezvous message, or GNI_RC_NOT_DONE
 	 * returned, we might want to generate IRQ at remote
 	 * peer.
 	 */
	if (((status == GNI_RC_SUCCESS) &&
		(tag == GNIX_SMSG_T_RNDZV_START)) ||
	    (status == GNI_RC_NOT_DONE))
		_gnix_rma_post_irq(req->vc);

	fastlock_release(&nic->lock);

	if (status == GNI_RC_NOT_DONE) {
		_gnix_nic_tx_free(nic, tdesc);
		GNIX_INFO(FI_LOG_EP_DATA,
			  "GNI_SmsgSendWTag returned %s\n",
			  gni_err_str[status]);
	} else if (status != GNI_RC_SUCCESS) {
		_gnix_nic_tx_free(nic, tdesc);
		GNIX_WARN(FI_LOG_EP_DATA,
			  "GNI_SmsgSendWTag returned %s\n",
			  gni_err_str[status]);
	}

	return gnixu_to_fi_errno(status);
}

ssize_t _gnix_send(struct gnix_fid_ep *ep, uint64_t loc_addr, size_t len,
		   void *mdesc, uint64_t dest_addr, void *context,
		   uint64_t flags, uint64_t data, uint64_t tag)
{
	int ret = FI_SUCCESS;
	struct gnix_vc *vc = NULL;
	struct gnix_fab_req *req;
	struct gnix_fid_mem_desc *md = NULL;
	int rendezvous;
	struct fid_mr *auto_mr = NULL;

	if (!ep) {
		return -FI_EINVAL;
	}

	if ((flags & FI_INJECT) && (len > GNIX_INJECT_SIZE)) {
		GNIX_INFO(FI_LOG_EP_DATA,
			  "Send length %d exceeds inject max size: %d\n",
			  len, GNIX_INJECT_SIZE);
		return -FI_EINVAL;
	}

	rendezvous = len >= ep->domain->params.msg_rendezvous_thresh;

	/* need a memory descriptor for large sends */
	if (rendezvous && !mdesc) {
		ret = gnix_mr_reg(&ep->domain->domain_fid.fid, (void *)loc_addr,
				 len, FI_READ | FI_WRITE, 0, 0, 0,
				 &auto_mr, NULL);
		if (ret != FI_SUCCESS) {
			GNIX_INFO(FI_LOG_EP_DATA,
				  "Failed to auto-register local buffer: %d\n",
				  ret);
			return ret;
		}
		flags |= FI_LOCAL_MR;
		mdesc = (void *)auto_mr;
		GNIX_INFO(FI_LOG_EP_DATA, "auto-reg MR: %p\n", auto_mr);
	}

	ret = _gnix_ep_get_vc(ep, dest_addr, &vc);
	if (ret) {
		goto err_get_vc;
	}

	req = _gnix_fr_alloc(ep);
	if (req == NULL) {
		ret = -FI_ENOSPC;
		goto err_fr_alloc;
	}

	req->type = GNIX_FAB_RQ_SEND;
	req->gnix_ep = ep;
	req->vc = vc;
	req->user_context = context;
	req->work_fn = _gnix_send_req;

	if (flags & FI_TAGGED) {
		req->msg.tag = tag;
	} else {
		/* Make sure zeroed tag ends up in the send CQE. */
		req->msg.tag = 0;
	}

	if (mdesc) {
		md = container_of(mdesc, struct gnix_fid_mem_desc, mr_fid);
	}
	req->msg.send_md = md;
	req->msg.send_len = len;
	req->msg.send_flags = flags;
	req->msg.imm = data;
	req->flags = 0;

	if (flags & FI_INJECT) {
		memcpy(req->inject_buf, (void *)loc_addr, len);
		req->msg.send_addr = (uint64_t)req->inject_buf;
	} else {
		req->msg.send_addr = loc_addr;
	}

	if ((flags & GNIX_SUPPRESS_COMPLETION) ||
	    (ep->send_selective_completion &&
	    !(flags & FI_COMPLETION))) {
		req->msg.send_flags &= ~FI_COMPLETION;
	} else {
		req->msg.send_flags |= FI_COMPLETION;
	}

	if (rendezvous) {
		/*
		 * this initialization is not necessary currently
		 * but is a place holder in the event a RDMA write
		 * path is implemented for rendezvous
		 */
		atomic_initialize(&req->msg.outstanding_txds, 0);
		req->msg.send_flags |= GNIX_MSG_RENDEZVOUS;
	}

	GNIX_INFO(FI_LOG_EP_DATA, "Queuing (%p %d)\n",
		  (void *)loc_addr, len);

	return _gnix_vc_queue_tx_req(req);

err_fr_alloc:
err_get_vc:
	if (auto_mr) {
		fi_close(&auto_mr->fid);
	}
	return ret;
}

