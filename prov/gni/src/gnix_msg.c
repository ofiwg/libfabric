/*
 * Copyright (c) 2015-2016 Cray Inc. All rights reserved.
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
#include "gnix_atomic.h"

#define INVALID_PEEK_FORMAT(fmt) \
	((fmt) == FI_CQ_FORMAT_CONTEXT || (fmt) == FI_CQ_FORMAT_MSG)

#define GNIX_TAGGED_PCD_COMPLETION_FLAGS	(FI_MSG | FI_RECV | FI_TAGGED)

/*******************************************************************************
 * helper functions
 ******************************************************************************/
static inline void __gnix_msg_free_rma_txd(struct gnix_fab_req *req,
					   struct gnix_tx_descriptor *txd)
{
	gni_ct_get_post_descriptor_t *cur_ct, *nxt_ct;

	if (txd->gni_desc.type & GNI_POST_FMA_GET) {
		for (cur_ct = txd->gni_desc.next_descr;
		     cur_ct != NULL;
		     cur_ct = nxt_ct) {
			nxt_ct = cur_ct->next_descr;
			free(cur_ct);
		}
	}
	_gnix_nic_tx_free(req->gnix_ep->nic, txd);
}

static inline void __gnix_msg_free_iov_txds(struct gnix_fab_req *req,
					    size_t txd_cnt)
{
	int i;

	for (i = 0; i < txd_cnt; i++) {
		__gnix_msg_free_rma_txd(req, req->iov_txds[i]);
	}
}

static void __gnix_msg_unpack_data_into_iov(const struct recv_info_t *dest,
					    size_t dest_cnt,
					    uint64_t src, size_t src_len)
{
	static int i;
	static size_t cum_len;
	static size_t offset;
	size_t bwrite;

	GNIX_DEBUG(FI_LOG_EP_DATA, "Unpacking data for recvv count (%lu)"
		   " src = %p, dest = %p,"
		   " i = %d, cum_len = %lu, offset = %lu\n",
		   dest_cnt, src, dest, i, cum_len, offset);

	/* Pull out each iov len and base field into the iov */
	while (i < dest_cnt) {
		bwrite = dest[i].recv_len - offset < src_len - cum_len ?
			dest[i].recv_len - offset : src_len - cum_len;

		memcpy((uint8_t *) dest[i].recv_addr + offset,
		       (void *) (src + cum_len), bwrite);

		cum_len += bwrite;

		/* Src exhausted */
		if (cum_len == src_len) {
			offset = dest[i].recv_len - bwrite;
			cum_len = 0;

			/* dest exhausted too, reset the counters */
			if (!offset) {
				i = cum_len = 0;
			}

			return;
		}

		i++;
	}
}

static void __gnix_msg_pack_data_from_iov(uint64_t dest, size_t dest_len,
					  const struct iovec *src,
					  size_t src_cnt)
{
	int i;
	size_t cum_len = 0;

	GNIX_DEBUG(FI_LOG_EP_CTRL, "Packing data for sendv count (%d)\n",
		   src_cnt);

	/* Pull out each iov len and base field into the dest buffer */
	for (i = 0; i < src_cnt && cum_len < dest_len; i++) {
		memcpy((void *) (dest + cum_len), src[i].iov_base,
		       src[i].iov_len < dest_len - cum_len ?
		       src[i].iov_len : dest_len - cum_len);
		cum_len += src[i].iov_len;
	}
}

static void __gnix_msg_copy_data_to_recv_addr(struct gnix_fab_req *req,
					      void *data)
{
	GNIX_TRACE(FI_LOG_EP_DATA, "\n");

	switch(req->type) {
	case GNIX_FAB_RQ_RECV:
		memcpy((void *)req->msg.recv_info[0].recv_addr, data,
		       req->msg.cum_send_len);
		break;

	case GNIX_FAB_RQ_RECVV:
	case GNIX_FAB_RQ_TRECVV:
		__gnix_msg_unpack_data_into_iov(req->msg.recv_info,
						req->msg.recv_iov_cnt,
						(uint64_t) data,
						req->msg.cum_send_len);
		break;

	default:
		GNIX_FATAL(FI_LOG_EP_DATA, "Invalid request type: %d", req->type);
		return;
	}
}

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
	struct gnix_vc *vc = req->vc;

	atomic_dec(&vc->outstanding_tx_reqs);
	_gnix_nic_tx_free(req->gnix_ep->nic, txd);

	_gnix_fr_free(req->gnix_ep, req);

	/* Schedule VC TX queue in case the VC is 'fenced'. */
	_gnix_vc_tx_schedule(vc);

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

	/* TODO: Check for FI_MULTI_RECV? */
	return __recv_err(ep, req->user_context, flags, req->msg.cum_recv_len,
			  (void *)req->msg.recv_info[0].recv_addr, req->msg.imm,
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
				 req->msg.cum_send_len,
				 req->msg.recv_flags & FI_MULTI_RECV ?
				 (void *)req->msg.recv_info[0].recv_addr :
				 NULL,
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

	GNIX_DEBUG(FI_LOG_EP_DATA, "send_cq = %p\n", ep->send_cq);

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
	txd->rndzv_fin_hdr.status = req->msg.status;

	GNIX_DEBUG(FI_LOG_EP_DATA, "req_addr = %p\n",
		   (void *) txd->rndzv_fin_hdr.req_addr);

	txd->req = req;

	txd->completer_fn = gnix_ep_smsg_completers[GNIX_SMSG_T_RNDZV_FIN];

	COND_ACQUIRE(nic->requires_lock, &nic->lock);
	status = GNI_SmsgSendWTag(req->vc->gni_ep,
			&txd->rndzv_fin_hdr, sizeof(txd->rndzv_fin_hdr),
			NULL, 0, txd->id, GNIX_SMSG_T_RNDZV_FIN);
	if ((status == GNI_RC_SUCCESS) &&
		(ep->domain->data_progress == FI_PROGRESS_AUTO))
		_gnix_rma_post_irq(req->vc);
	COND_RELEASE(nic->requires_lock, &nic->lock);

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

	head_off = req->msg.send_info[0].send_addr & GNI_READ_ALIGN_MASK;
	head_len = head_off ? GNI_READ_ALIGN - head_off : 0;
	tail_len = (req->msg.send_info[0].send_addr + req->msg.send_info[0].send_len) &
			GNI_READ_ALIGN_MASK;

	if (head_off) {
		addr = (uint8_t *)&req->msg.rndzv_head + head_off;

		GNIX_INFO(FI_LOG_EP_DATA,
			  "writing %d bytes to head (%p, %hxx)\n",
			  head_len, req->msg.recv_info[0].recv_addr,
			  *(uint32_t *)addr);
		memcpy((void *)req->msg.recv_info[0].recv_addr, addr, head_len);
	}

	if (tail_len) {
		addr = (void *)(req->msg.recv_info[0].recv_addr +
				req->msg.send_info[0].send_len -
				tail_len);

		GNIX_INFO(FI_LOG_EP_DATA,
			  "writing %d bytes to tail (%p, %hxx)\n",
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

		if (atomic_dec(&req->msg.outstanding_txds) == 1) {
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
		if (GNIX_EP_RDM(req->gnix_ep->type) &&
			req->tx_failures <
				req->gnix_ep->domain->params.max_retransmits) {

			GNIX_INFO(FI_LOG_EP_DATA,
				  "Requeueing failed request: %p\n", req);
			return _gnix_vc_queue_work_req(req);
		}

		if (!GNIX_EP_DGM(req->gnix_ep->type))
			GNIX_INFO(FI_LOG_EP_DATA,
				  "Dropping failed request: %p\n", req);
		req->msg.status = tx_status;
		req->work_fn = __gnix_rndzv_req_send_fin;
		return _gnix_vc_queue_work_req(req);
	}

	__gnix_msg_copy_unaligned_get_data(req);

	GNIX_INFO(FI_LOG_EP_DATA, "Completed RNDZV GET, req: %p\n", req);

	if (req->msg.recv_flags & FI_LOCAL_MR) {
		GNIX_INFO(FI_LOG_EP_DATA, "freeing auto-reg MR: %p\n",
			  req->msg.recv_md[0]);
		fi_close(&req->msg.recv_md[0]->mr_fid.fid);
	}

	req->work_fn = __gnix_rndzv_req_send_fin;
	ret = _gnix_vc_queue_work_req(req);

	return ret;
}

/*
 * As the completer fn is called in the nic's progress loop, the remote
 * side's fabric request will keep track of the state of this rndzv iov
 * transaction, once the outstanding_txds reaches zero for this fabric
 * request, we are either ready to generate CQEs and send the fin msg
 * back to the sender or rebuild and retransmit the iov txds.
 */
static int __gnix_rndzv_iov_req_build(void *arg);
static int __gnix_rndzv_iov_req_complete(void *arg, gni_return_t tx_status)
{
	struct gnix_tx_descriptor *txd = (struct gnix_tx_descriptor *)arg;
	struct gnix_fab_req *req = txd->req;
	int i;

	GNIX_TRACE(FI_LOG_EP_DATA, "\n");

	GNIX_DEBUG(FI_LOG_EP_DATA, "req->msg.outstanding_txds = %d\n",
		   atomic_get(&req->msg.outstanding_txds));

	req->msg.status |= tx_status;
	__gnix_msg_free_rma_txd(req, txd);

	if (atomic_dec(&req->msg.outstanding_txds) == 0) {
		GNIX_DEBUG(FI_LOG_EP_DATA, "req->msg.recv_flags == FI_LOCAL_MR "
			   "is %s, req->msg.recv_iov_cnt = %lu\n",
			   req->msg.recv_flags & FI_LOCAL_MR ? "true" : "false",
			   req->msg.recv_iov_cnt);

		if (req->msg.status != FI_SUCCESS) {
			req->tx_failures++;

			if (GNIX_EP_RDM(req->gnix_ep->type) &&
			    req->tx_failures <
			    req->gnix_ep->domain->params.max_retransmits) {
				/* Build and re-tx the entire iov request if the
				 * ep type is "reliable datagram" */
				req->work_fn = __gnix_rndzv_iov_req_build;
				return _gnix_vc_queue_work_req(req);
			}

			if (!GNIX_EP_DGM(req->gnix_ep->type)) {
				GNIX_INFO(FI_LOG_EP_DATA,
					  "Dropping failed request: %p\n", req);
			}
		} else {
			if (req->msg.recv_flags & FI_LOCAL_MR) {
				for (i = 0; i < req->msg.recv_iov_cnt; i++) {
					GNIX_INFO(FI_LOG_EP_DATA, "freeing auto"
						  "-reg MR: %p\n",
						  req->msg.recv_md[i]);
					fi_close(&req->msg.recv_md[i]->mr_fid.fid);
				}
			}
		}
		/* Generate remote CQE and send fin msg back to sender */
		req->work_fn = __gnix_rndzv_req_send_fin;
		return _gnix_vc_queue_work_req(req);
	}

	/*
	 * Successful tx, continue until the txd counter reaches zero
	 * or we can't recover from the error.
	 */
	return FI_SUCCESS;
}

static int __gnix_rndzv_req_xpmem(struct gnix_fab_req *req)
{
	int ret = FI_SUCCESS, i = 0, j = 0;
	size_t cpy_len, recv_len;
	uint64_t recv_ptr = 0UL;
	struct gnix_xpmem_access_handle *access_hndl;

	recv_len = req->msg.recv_info[0].recv_len;
	recv_ptr = req->msg.recv_info[0].recv_addr;

	/* Copy data from/to (>=1) iovec entries */
	while (i < req->msg.send_iov_cnt) {
		cpy_len = MIN(recv_len, req->msg.send_info[i].send_len);

		/*
		 * look up mapping from other EP
		 */
		ret = _gnix_xpmem_access_hndl_get(req->gnix_ep->xpmem_hndl,
						  req->vc->peer_apid,
						  req->msg.send_info[i].send_addr,
						  req->msg.recv_info[j].recv_len,
						  &access_hndl);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_DATA, "_gnix_xpmem_access_hndl_get failed %s\n",
				  fi_strerror(-ret));
			req->msg.status = GNI_RC_TRANSACTION_ERROR;
			return ret;
		}

		/*
		 * pull the data from the other process' address space
		 */
		ret = _gnix_xpmem_copy(access_hndl,
				       (void *)recv_ptr,
				       (void *)req->msg.send_info[i].send_addr,
				       cpy_len);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_DATA, "_gnix_xpmem_vaddr_copy failed %s\n",
				  fi_strerror(-ret));
			req->msg.status = GNI_RC_TRANSACTION_ERROR;
			_gnix_xpmem_access_hndl_put(access_hndl);
			return ret;
		}

		ret = _gnix_xpmem_access_hndl_put(access_hndl);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_DATA, "_gnix_xpmem_access_hndl_put failed %s\n",
				  fi_strerror(-ret));
		}

		/* Update the local and remote addresses */
		recv_len -= cpy_len;

		/* We have exhausted the current recv (and possibly send)
		 * buffer */
		if (recv_len == 0) {
			j++;

			/* We cannot receive any more. */
			if (j == req->msg.recv_iov_cnt)
				break;

			recv_ptr = req->msg.recv_info[j].recv_addr;
			recv_len = req->msg.recv_info[j].recv_len;

			/* Also exhausted send buffer */
			if (cpy_len == req->msg.send_info[i].send_len) {
				i++;
			} else {
				req->msg.send_info[i].send_addr += cpy_len;
			}
		} else {	/* Just exhausted current send buffer. */
			i++;
			recv_ptr += cpy_len;
		}
		GNIX_DEBUG(FI_LOG_EP_DATA, "i = %d, j = %d\n", i, j);
	}

	/*
	 * set the req send fin and reschedule req
	 */

	req->msg.status = GNI_RC_SUCCESS;  /* hmph */
	req->work_fn = __gnix_rndzv_req_send_fin;
	return _gnix_vc_queue_work_req(req);
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

	GNIX_TRACE(FI_LOG_EP_DATA, "\n");
	/*
	 * TODO: xpmem intercept here
	 */

	if (req->vc->modes & GNIX_VC_MODE_XPMEM)
		return  __gnix_rndzv_req_xpmem(req);

	if (!req->msg.recv_md[0]) {
		rc = gnix_mr_reg(&ep->domain->domain_fid.fid,
				 (void *)req->msg.recv_info[0].recv_addr,
				 req->msg.recv_info[0].recv_len,
				 FI_READ | FI_WRITE, 0, 0, 0, &auto_mr, NULL);
		if (rc != FI_SUCCESS) {
			GNIX_INFO(FI_LOG_EP_DATA,
				  "Failed to auto-register local buffer: %d\n",
				  rc);
			return -FI_EAGAIN;
		}
		req->msg.recv_flags |= FI_LOCAL_MR;
		req->msg.recv_md[0] = container_of(auto_mr,
						   struct gnix_fid_mem_desc,
						   mr_fid);
		req->msg.recv_info[0].mem_hndl = req->msg.recv_md[0]->mem_hndl;
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
	txd->gni_desc.local_mem_hndl = req->msg.recv_info[0].mem_hndl;
	txd->gni_desc.remote_mem_hndl = req->msg.rma_mdh;
	txd->gni_desc.rdma_mode = 0;
	txd->gni_desc.src_cq_hndl = (use_tx_cq_blk) ?
					nic->tx_cq_blk : nic->tx_cq;

	head_off = req->msg.send_info[0].send_addr & GNI_READ_ALIGN_MASK;
	head_len = head_off ? GNI_READ_ALIGN - head_off : 0;
	tail_len = (req->msg.send_info[0].send_addr + req->msg.send_info[0].send_len) &
			GNI_READ_ALIGN_MASK;

	txd->gni_desc.local_addr = (uint64_t)req->msg.recv_info[0].recv_addr + head_len;
	txd->gni_desc.remote_addr = (uint64_t)req->msg.send_info[0].send_addr + head_len;
	txd->gni_desc.length = req->msg.send_info[0].send_len - head_len - tail_len;

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

		tail_data = (void *)((req->msg.send_info[0].send_addr +
				      req->msg.send_info[0].send_len) &
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

	COND_ACQUIRE(nic->requires_lock, &nic->lock);

	if (inject_err) {
		_gnix_nic_txd_err_inject(nic, txd);
		status = GNI_RC_SUCCESS;
	} else {
		status = GNI_PostRdma(req->vc->gni_ep, &txd->gni_desc);
	}

	if (status != GNI_RC_SUCCESS) {
		COND_RELEASE(nic->requires_lock, &nic->lock);
		if (tail_txd)
			_gnix_nic_tx_free(nic, tail_txd);
		_gnix_nic_tx_free(nic, txd);
		GNIX_INFO(FI_LOG_EP_DATA, "GNI_PostRdma failed: %s\n",
			  gni_err_str[status]);

		GNIX_DEBUG(FI_LOG_EP_DATA, "\n");
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
			COND_RELEASE(nic->requires_lock, &nic->lock);
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
		req->msg.status = GNI_RC_SUCCESS;

	}

	COND_RELEASE(nic->requires_lock, &nic->lock);

	GNIX_INFO(FI_LOG_EP_DATA, "Initiated RNDZV GET, req: %p\n", req);

	return gnixu_to_fi_errno(status);
}

/* Attempt to post the iov txds */
static int __gnix_rndzv_iov_req_post(void *arg)
{
	struct gnix_fab_req *req = (struct gnix_fab_req *)arg;
	struct gnix_tx_descriptor *txd;
	gni_return_t status;
	struct gnix_nic *nic = req->gnix_ep->nic;
	int i, iov_txd_cnt = atomic_get(&req->msg.outstanding_txds);

	assert(nic != NULL);

	GNIX_TRACE(FI_LOG_EP_DATA, "\n");

	COND_ACQUIRE(nic->requires_lock, &nic->lock);

	for (i = 0, txd = req->iov_txds[0];
	     i < iov_txd_cnt;
	     txd = req->iov_txds[++i]) {
		if (txd->gni_desc.type & GNI_POST_RDMA_GET) {
			status = GNI_PostRdma(req->vc->gni_ep,
					      &txd->gni_desc);
		} else {
			status = GNI_CtPostFma(req->vc->gni_ep,
					       &txd->gni_desc);
		}

		if (status != GNI_RC_SUCCESS) {
			COND_RELEASE(nic->requires_lock, &nic->lock);

			GNIX_WARN(FI_LOG_EP_DATA, "%s failed on txd %p: with "
				  "%s\n",
				  txd->gni_desc.type & GNI_POST_RDMA_GET ?
				  "GNI_PostRdma" : "GNI_CtPostFma", txd,
				  gni_err_str[status]);

			return gnixu_to_fi_errno(status);
		}
	}

	COND_RELEASE(nic->requires_lock, &nic->lock);

	return FI_SUCCESS;
}

/*
 * Dequeued a iov work req on the remote endpoint, initiate the pull of data.
 */
static int __gnix_rndzv_iov_req_build(void *arg)
{
	int ret = FI_SUCCESS, i = 0, j = 0, use_tx_cq_blk;
	struct gnix_fab_req *req = (struct gnix_fab_req *)arg;
	struct gnix_fid_ep *ep = req->gnix_ep;
	struct gnix_nic *nic = ep->nic;
	gni_ep_handle_t gni_ep = req->vc->gni_ep;
	struct gnix_tx_descriptor *txd = NULL;
	size_t recv_len, get_len, ct_size, send_cnt, recv_cnt, txd_cnt;
	uint64_t recv_ptr = 0UL;
	/* TODO: Should this be the sender's rndzv thresh instead? */
	size_t rndzv_thresh = ep->domain->params.msg_rendezvous_thresh;
	gni_ct_get_post_descriptor_t *cur_ct = NULL;
	void **next_ct = NULL;
	uint32_t max_ct_size = nic->gni_cdm_modes & GNI_CDM_MODE_FMA_SHARED ?
		1024*1024 : 1<<30;

	GNIX_TRACE(FI_LOG_EP_DATA, "\n");
	/*
	 * TODO: xpmem intercept here
	 */

	if (req->vc->modes & GNIX_VC_MODE_XPMEM)
		return  __gnix_rndzv_req_xpmem(req);

	txd_cnt = ct_size = 0;
	send_cnt = req->msg.send_iov_cnt;
	recv_ptr = req->msg.recv_info[0].recv_addr;
	recv_len = req->msg.recv_info[0].recv_len;
	recv_cnt = req->msg.recv_iov_cnt;
	use_tx_cq_blk = (ep->domain->data_progress == FI_PROGRESS_AUTO);

	GNIX_DEBUG(FI_LOG_EP_DATA, "send_cnt = %lu, recv_cnt = %lu\n",
		   send_cnt, recv_cnt);

	/* Ensure the user's recv buffer is registered for recv/recvv */
	if (!req->msg.recv_md[0]) {
		struct fid_mr *auto_mr;

		for (i = 0; i < recv_cnt; i++) {
			auto_mr = NULL;
			ret = gnix_mr_reg(&ep->domain->domain_fid.fid,
					  (void *)
					  req->msg.recv_info[i].recv_addr,
					  req->msg.recv_info[i].recv_len,
					  FI_READ | FI_WRITE, 0, 0, 0,
					  &auto_mr, NULL);

			if (ret != FI_SUCCESS) {
				GNIX_DEBUG(FI_LOG_EP_DATA,
					   "Failed to auto-register"
					   " local buffer: %s\n",
					   fi_strerror(-ret));

				for (i--; i >= 0; i--) {
					fi_close(&req->msg.recv_md[i]->mr_fid.fid);
				}

				return ret;
			}

			req->msg.recv_md[i] = container_of(
				(void *) auto_mr,
				struct gnix_fid_mem_desc,
				mr_fid);

			req->msg.recv_info[i].mem_hndl =
				req->msg.recv_md[i]->mem_hndl;

			GNIX_DEBUG(FI_LOG_EP_DATA, "auto-reg MR: %p\n",
				   req->msg.recv_md[i]);

		}
		req->msg.recv_flags |= FI_LOCAL_MR;
	}

	i = 0;

	/* Iterate through the buffers and build the Fma and Rdma requests! */
	while (i < send_cnt) {
		get_len = MIN(recv_len, req->msg.send_info[i].send_len);

		GNIX_DEBUG(FI_LOG_EP_DATA, "send_info[%d].send_len = %lu,"
			   " recv_len = %lu, get_len = %lu\n", i,
			   req->msg.send_info[i].send_len, recv_len, get_len);

		if (get_len >= rndzv_thresh) { /* Build the rdma txd */
			ret = _gnix_nic_tx_alloc(nic, &txd);

			if (ret != FI_SUCCESS) {
				/* We'll try again. */
				GNIX_INFO(FI_LOG_EP_DATA, "_gnix_nic_tx_alloc()"
					  " returned %s\n",
					  fi_strerror(-ret));

				__gnix_msg_free_iov_txds(req, txd_cnt);
				return -FI_ENOSPC;
			}

			txd->completer_fn = __gnix_rndzv_iov_req_complete;
			txd->req = req;

			txd->gni_desc.type = GNI_POST_RDMA_GET;
			txd->gni_desc.cq_mode = GNI_CQMODE_GLOBAL_EVENT;
			txd->gni_desc.dlvr_mode = GNI_DLVMODE_PERFORMANCE;
			txd->gni_desc.local_mem_hndl =
				req->msg.recv_info[j].mem_hndl;
			txd->gni_desc.remote_mem_hndl = req->msg.send_info[i].mem_hndl;
			txd->gni_desc.rdma_mode = 0;
			txd->gni_desc.src_cq_hndl = (use_tx_cq_blk) ?
				nic->tx_cq_blk : nic->tx_cq;

			/* TODO: handle alignment! */
			txd->gni_desc.local_addr =
				(uint64_t) recv_ptr;
			txd->gni_desc.remote_addr =
				req->msg.send_info[i].send_addr;
			txd->gni_desc.length = get_len;

			req->iov_txds[txd_cnt++] = txd;
			txd = NULL;
		} else {		       /* Build the Ct txd */
			/*
			 * The txd is ready to be queued if the max size has
			 * been reached.
			 */
			if (ct_size + get_len >= max_ct_size) {
				*next_ct = txd = NULL;
				req->iov_txds[txd_cnt++] = txd;
				ct_size = 0;
			} else {
				if (!txd) {
					GNIX_DEBUG(FI_LOG_EP_DATA, "New FMA"
						   " CT\n");
					ret = _gnix_nic_tx_alloc(nic, &txd);
					if (ret != FI_SUCCESS) {
						/* We'll try again. */
						GNIX_INFO(FI_LOG_EP_DATA,
							  "_gnix_nic_tx_alloc()"
							  " returned %s\n",
							  fi_strerror(-ret));

						__gnix_msg_free_iov_txds(req, txd_cnt);
						return -FI_ENOSPC;
					}

					txd->completer_fn =
						__gnix_rndzv_iov_req_complete;
					txd->req = req;

					txd->gni_desc.type = GNI_POST_FMA_GET;
					txd->gni_desc.cq_mode =
						GNI_CQMODE_GLOBAL_EVENT;
					txd->gni_desc.dlvr_mode =
						GNI_DLVMODE_PERFORMANCE;
					txd->gni_desc.local_mem_hndl =
						req->msg.recv_info[j].
						mem_hndl;

					txd->gni_desc.remote_mem_hndl =
						req->msg.send_info[i].mem_hndl;
					txd->gni_desc.rdma_mode = 0;
					txd->gni_desc.src_cq_hndl =
						(use_tx_cq_blk) ?
						nic->tx_cq_blk : nic->tx_cq;

					/* TODO: handle alignment! */
					txd->gni_desc.local_addr =
						(uint64_t) recv_ptr;
					txd->gni_desc.remote_addr =
						req->msg.send_info[i].send_addr;
					txd->gni_desc.length = get_len;
					ct_size += get_len;

					next_ct = &txd->gni_desc.next_descr;
				} else {
					cur_ct = *next_ct = malloc(sizeof(gni_ct_get_post_descriptor_t));

					if (cur_ct == NULL) {
						GNIX_DEBUG(FI_LOG_EP_DATA,
							  "Failed to allocate "
							  "gni FMA get chained "
							  "descriptor.");

						/* +1 to ensure we free the
						 * current chained txd */
						__gnix_msg_free_iov_txds(req, txd_cnt + 1);
						return -FI_ENOSPC;
					}

					cur_ct->ep_hndl = gni_ep;
					cur_ct->length = get_len;
					ct_size += get_len;
					cur_ct->remote_addr = req->msg.send_info[i].send_addr;
					cur_ct->remote_mem_hndl =
						req->msg.send_info[i].mem_hndl;
					cur_ct->local_addr =
						(uint64_t) recv_ptr;
					cur_ct->local_mem_hndl =
						req->msg.recv_info[j].
						mem_hndl;

					next_ct = &cur_ct->next_descr;
				}
			}
		}

		/* Update the local and remote addresses */
		recv_len -= get_len;

		/* We have exhausted the current recv (and possibly send)
		 * buffer */
		if (recv_len == 0) {
			j++;

			/* We cannot receive any more. */
			if (j == recv_cnt)
				break;

			recv_ptr = req->msg.recv_info[j].recv_addr;
			recv_len = req->msg.recv_info[j].recv_len;

			/* Also exhausted send buffer */
			if (get_len == req->msg.send_info[i].send_len) {
				i++;
			} else {
				req->msg.send_info[i].send_addr += get_len;
			}
		} else {	/* Just exhausted current send buffer. */
			i++;
			recv_ptr += get_len;
		}
		GNIX_DEBUG(FI_LOG_EP_DATA, "i = %d, j = %d\n", i, j);
	}

	/*
	 * If we ran out of buffer space on the sender's/receiver's side in the
	 * middle of building the ct, we must terminate and add that ct to the
	 * queue. Note that if the last txd built was a rdma txd then the txd
	 * will have been queued and txd will have a NULL value.
	 */
	if (txd) {
		*next_ct = NULL;
		req->iov_txds[txd_cnt++] = txd;
	}

	atomic_set(&req->msg.outstanding_txds, txd_cnt);

	/* All the txd's are built, update the work_fn */
	req->work_fn = __gnix_rndzv_iov_req_post;

	/* Put this request back on work Q.
	 * TODO: Should we put it at the beginning of the work Q? */
	ret = _gnix_vc_queue_work_req(req);
	return ret;
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

		/*
		 * For fi_sendv we must free the temporary buf used to flatten
		 * the user's iovec.
		 */
		if (req->type == GNIX_FAB_RQ_SENDV ||
		    req->type == GNIX_FAB_RQ_TSENDV) {
			free((void *) req->msg.send_info[0].send_addr);
		}

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

	if (tx_status != GNI_RC_SUCCESS || req->msg.status != GNI_RC_SUCCESS) {
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
	[GNIX_SMSG_T_RNDZV_IOV_START] = __comp_rndzv_start
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

	GNIX_TRACE(FI_LOG_EP_DATA, "\n");

	ep = vc->ep;
	assert(ep);

	data_ptr = (void *)((char *)msg + sizeof(*hdr));

	tagged = !!(hdr->flags & FI_TAGGED);
	__gnix_msg_queues(ep, tagged, &queue_lock, &posted_queue, &unexp_queue);

	COND_ACQUIRE(ep->requires_lock, queue_lock);

	/* Lookup a matching posted request. */
	req = _gnix_match_tag(posted_queue, hdr->msg_tag, 0, FI_PEEK, NULL,
			      &vc->peer_addr);
	if (req) {
		req->addr = vc->peer_addr;
		req->gnix_ep = ep;
		req->vc = vc;

		req->msg.cum_send_len = MIN(hdr->len, req->msg.cum_recv_len);
		req->msg.send_flags = hdr->flags;
		req->msg.send_iov_cnt = 1;
		req->msg.tag = hdr->msg_tag;
		req->msg.imm = hdr->imm;

		GNIX_DEBUG(FI_LOG_EP_DATA, "Matched req: %p (%p, %u)\n",
			   req, req->msg.recv_info[0].recv_addr,
			   req->msg.cum_send_len);

		__gnix_msg_copy_data_to_recv_addr(req, data_ptr);
		__gnix_msg_recv_completion(ep, req);

		/* Check if we're using FI_MULTI_RECV and there is space left
		 * in the receive buffer. */
		if ((req->msg.recv_flags & FI_MULTI_RECV) &&
		    ((req->msg.cum_recv_len - req->msg.cum_send_len) >=
		     ep->min_multi_recv)) {
			GNIX_DEBUG(FI_LOG_EP_DATA, "Re-using req: %p\n", req);

			/* Adjust receive buffer for the next match. */
			req->msg.recv_info[0].recv_addr += req->msg.send_info[0].send_len;
			req->msg.recv_info[0].recv_len -= req->msg.cum_send_len;
			req->msg.cum_recv_len = req->msg.recv_info[0].recv_len;
		} else {
			GNIX_DEBUG(FI_LOG_EP_DATA, "Freeing req: %p\n", req);

			/* Dequeue and free the request. */
			_gnix_remove_tag(posted_queue, req);
			_gnix_fr_free(ep, req);
		}
	} else {
		/* Add new unexpected receive request. */
		req = _gnix_fr_alloc(ep);
		if (req == NULL) {
			COND_RELEASE(ep->requires_lock, queue_lock);
			return -FI_ENOMEM;
		}

		/* TODO: Buddy alloc */
		req->msg.send_info[0].send_addr = (uint64_t)malloc(hdr->len);
		if (unlikely(req->msg.send_info[0].send_addr == 0ULL)) {
			COND_RELEASE(ep->requires_lock, queue_lock);
			_gnix_fr_free(ep, req);
			return -FI_ENOMEM;
		}

		req->type = GNIX_FAB_RQ_RECV;
		req->addr = vc->peer_addr;
		req->gnix_ep = ep;
		req->vc = vc;

		req->msg.cum_send_len = hdr->len;
		req->msg.send_info[0].send_len = hdr->len;
		req->msg.send_iov_cnt = 1;
		req->msg.send_flags = hdr->flags;
		req->msg.tag = hdr->msg_tag;
		req->msg.imm = hdr->imm;

		memcpy((void *)req->msg.send_info[0].send_addr, data_ptr, hdr->len);
		req->addr = vc->peer_addr;

		_gnix_insert_tag(unexp_queue, req->msg.tag, req, ~0);

		GNIX_DEBUG(FI_LOG_EP_DATA, "New req: %p (%u)\n",
			  req, req->msg.cum_send_len);
	}

	COND_RELEASE(ep->requires_lock, queue_lock);

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

	GNIX_TRACE(FI_LOG_EP_DATA, "\n");

	ep = vc->ep;
	assert(ep);

	tagged = !!(hdr->flags & FI_TAGGED);
	__gnix_msg_queues(ep, tagged, &queue_lock, &posted_queue, &unexp_queue);

	COND_ACQUIRE(ep->requires_lock, queue_lock);

	req = _gnix_match_tag(posted_queue, hdr->msg_tag, 0, FI_PEEK, NULL,
			      &vc->peer_addr);

	if (req) {
		req->addr = vc->peer_addr;
		req->gnix_ep = ep;
		req->vc = vc;
		req->tx_failures = 0;

		/* Check if a second GET for unaligned data is needed. */
		if (hdr->len > req->msg.recv_info[0].recv_len &&
		    ((hdr->addr + req->msg.recv_info[0].recv_len) & GNI_READ_ALIGN_MASK)) {
			req->msg.recv_flags |= GNIX_MSG_GET_TAIL;
		}

		req->msg.send_info[0].send_addr = hdr->addr;
		req->msg.send_info[0].send_len =
			MIN(hdr->len, req->msg.cum_recv_len);
		req->msg.send_info[0].mem_hndl = hdr->mdh;
		req->msg.cum_send_len = req->msg.send_info[0].send_len;
		req->msg.send_iov_cnt = 1;
		req->msg.send_flags = hdr->flags;
		req->msg.tag = hdr->msg_tag;
		req->msg.imm = hdr->imm;
		req->msg.rma_mdh = hdr->mdh;
		req->msg.rma_id = hdr->req_addr;
		req->msg.rndzv_head = hdr->head;
		req->msg.rndzv_tail = hdr->tail;

		if (req->type == GNIX_FAB_RQ_RECV) {
			/* fi_send is rndzv with recv */

			GNIX_DEBUG(FI_LOG_EP_DATA, "recv rndzv start - POSTED,"
				   " req = %p\n", req);
			req->work_fn = __gnix_rndzv_req;
		} else if (req->type == GNIX_FAB_RQ_RECVV ||
			   req->type == GNIX_FAB_RQ_TRECVV) {
			/* fi_send is rndzv with recvv */

			GNIX_DEBUG(FI_LOG_EP_DATA, "recvv rndzv start - POSTED,"
				   " req = %p\n", req);

			req->work_fn = __gnix_rndzv_iov_req_build;
		}

		GNIX_INFO(FI_LOG_EP_DATA, "Matched req: %p (%p, %u)\n",
			  req, req->msg.recv_info[0].recv_addr,
			  req->msg.send_info[0].send_len);

		/* Check if we're using FI_MULTI_RECV and there is space left
		 * in the receive buffer. */
		if (req->type == GNIX_FAB_RQ_RECV &&
		    (req->msg.recv_flags & FI_MULTI_RECV) &&
		    ((req->msg.cum_recv_len - req->msg.cum_send_len) >=
		     ep->min_multi_recv)) {
			/* Allocate new request for this transfer. */
			dup_req = __gnix_msg_dup_req(req);
			if (!dup_req) {
				COND_RELEASE(ep->requires_lock, queue_lock);
				return -FI_ENOMEM;
			}

			/* Adjust receive buffer for the next match. */
			req->msg.recv_info[0].recv_addr += req->msg.send_info[0].send_len;
			req->msg.recv_info[0].recv_len -= req->msg.send_info[0].send_len;
			req->msg.cum_recv_len = req->msg.recv_info[0].recv_len;

			/* 'req' remains queued for more matches while the
			 * duplicated request is processed. */
			req = dup_req;
		} else {
			/* Dequeue the request. */
			_gnix_remove_tag(posted_queue, req);
		}

		/* Queue request to initiate pull of source data. */
		ret = _gnix_vc_queue_work_req(req);

		GNIX_DEBUG(FI_LOG_EP_DATA,
			   "_gnix_vc_queue_work_req returned %s\n",
			   fi_strerror(-ret));
	} else {
		/* Add new unexpected receive request. */
		req = _gnix_fr_alloc(ep);
		if (req == NULL) {
			COND_RELEASE(ep->requires_lock, queue_lock);
			return -FI_ENOMEM;
		}

		req->type = GNIX_FAB_RQ_RECV;
		req->addr = vc->peer_addr;
		req->gnix_ep = ep;
		req->vc = vc;

		req->msg.send_info[0].send_addr = hdr->addr;
		req->msg.send_info[0].send_len = hdr->len;
		req->msg.send_info[0].mem_hndl = hdr->mdh;
		req->msg.send_iov_cnt = 1;
		req->msg.cum_send_len = req->msg.send_info[0].send_len;
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
			  req, req->msg.send_info[0].send_len);
	}

	COND_RELEASE(ep->requires_lock, queue_lock);

	status = GNI_SmsgRelease(vc->gni_ep);
	if (unlikely(status != GNI_RC_SUCCESS)) {
		GNIX_WARN(FI_LOG_EP_DATA,
			  "GNI_SmsgRelease returned %s\n",
			  gni_err_str[status]);
		ret = gnixu_to_fi_errno(status);
	}

	return ret;
}

static int __smsg_rndzv_iov_start(void *data, void *msg)
{
	int ret = FI_SUCCESS;
	gni_return_t status;
	struct gnix_vc *vc = (struct gnix_vc *) data;
	struct gnix_smsg_rndzv_iov_start_hdr *hdr = msg;
	void *data_ptr = (void *) ((uint8_t *) msg + sizeof(*hdr));
	struct gnix_fid_ep *ep;
	struct gnix_fab_req *req = NULL;
	struct gnix_tag_storage *unexp_queue;
	struct gnix_tag_storage *posted_queue;
	fastlock_t *queue_lock;
	char is_req_posted = 0;

	GNIX_TRACE(FI_LOG_EP_DATA, "\n");

#if ENABLE_DEBUG
	int i;

	for (i = 0; i < hdr->iov_cnt; i++) {
		GNIX_DEBUG(FI_LOG_EP_DATA, "base[%d] = %p, len[%d] = %lu\n", i,
			   ((struct iovec *)data_ptr)[i].iov_base, i,
			   ((struct iovec *)data_ptr)[i].iov_len);
	}
#endif
	ep = vc->ep;
	assert(ep != NULL);

	__gnix_msg_queues(ep, hdr->flags & FI_TAGGED, &queue_lock,
			  &posted_queue, &unexp_queue);

	COND_ACQUIRE(ep->requires_lock, queue_lock);

	req = _gnix_match_tag(posted_queue, hdr->msg_tag, 0, FI_PEEK, NULL,
			      &vc->peer_addr);

	if (req) {		/* Found a request in the posted queue */
		is_req_posted = 1;
		req->tx_failures = 0;

		GNIX_INFO(FI_LOG_EP_DATA, "Matched req: %p (%p, %u)\n",
			  req, req->msg.recv_info[0].recv_addr, hdr->send_len);
		_gnix_remove_tag(posted_queue, req);
	} else {		/* Unexpected receive, enqueue it */
		req = _gnix_fr_alloc(ep);
		if (req == NULL) {
			COND_RELEASE(ep->requires_lock, queue_lock);
			return -FI_ENOMEM;
		}

		atomic_initialize(&req->msg.outstanding_txds, 0);

		GNIX_INFO(FI_LOG_EP_DATA, "New req: %p (%u)\n",
			  req, hdr->send_len);
	}

	req->addr = vc->peer_addr;
	req->gnix_ep = ep;
	req->vc = vc;
	req->work_fn = __gnix_rndzv_iov_req_build;

	req->msg.send_flags = hdr->flags;
	req->msg.imm = hdr->imm;
	req->msg.tag = hdr->msg_tag;
	req->msg.send_iov_cnt = hdr->iov_cnt;
	req->msg.rma_id = hdr->req_addr;
	req->msg.cum_send_len = hdr->send_len;
	memcpy(req->msg.send_info, data_ptr,
	       sizeof(struct send_info_t) * hdr->iov_cnt);

	if (is_req_posted)
		ret = _gnix_vc_queue_work_req(req);
	else
		_gnix_insert_tag(unexp_queue, req->msg.tag, req, ~0);


	COND_RELEASE(ep->requires_lock, queue_lock);

	/*
	 * Release the message buffer on the nic, need to copy the data
	 * section out before this.
	 */
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
	int i;
	struct gnix_fab_req *req = (struct gnix_fab_req *)arg;

	GNIX_TRACE(FI_LOG_EP_DATA, "\n");

	for (i = 0; i < req->msg.send_iov_cnt; i++) {
		GNIX_INFO(FI_LOG_EP_DATA, "freeing auto-reg MR: %p\n",
			  req->msg.send_md[i]);

		GNIX_DEBUG(FI_LOG_EP_DATA, "req->msg.send_md[%d] ="
			   " %p\n", i, req->msg.send_md[i]);
		fi_close(&req->msg.send_md[i]->mr_fid.fid);
	}

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

	GNIX_TRACE(FI_LOG_EP_DATA, "\n");

	req = (struct gnix_fab_req *)hdr->req_addr;
	assert(req);

	GNIX_INFO(FI_LOG_EP_DATA, "Received RNDZV_FIN, req: %p\n", req);

	ep = req->gnix_ep;
	assert(ep != NULL);

	if (hdr->status == GNI_RC_SUCCESS) {
		__gnix_msg_send_completion(ep, req);
	} else {
		__gnix_msg_send_err(ep, req);
	}

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
	[GNIX_SMSG_T_RMA_DATA] = __smsg_rma_data, /* defined in gnix_rma.c */
	[GNIX_SMSG_T_AMO_CNTR] = __smsg_amo_cntr, /* defined in gnix_amo.c */
	[GNIX_SMSG_T_RNDZV_IOV_START] = __smsg_rndzv_iov_start
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
	if (!rendezvous && req->msg.recv_info[0].recv_addr &&
	    !INVALID_PEEK_FORMAT(recv_cq->attr.format)) {
		int copy_len = MIN(req->msg.send_info[0].send_len,
				   req->msg.recv_info[0].recv_len);

		memcpy((void *)req->msg.recv_info[0].recv_addr,
		       (void *)req->msg.send_info[0].send_addr,
		       copy_len);
	} else {
		/* The CQE should not contain a valid buffer. */
		req->msg.recv_info[0].recv_addr = 0;
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
	req->msg.recv_info[0].recv_addr = 0;
	req->msg.cum_send_len = req->msg.send_info[0].send_len = 0;

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
		free((void *)req->msg.send_info[0].send_addr);
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
	if (GNIX_EP_RDM_DGM(ep->type)) {
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

	if (!ep->recv_cq) {
		return -FI_ENOCQ;
	}

	if (!tagged) {
		if (!ep->ep_ops.msg_recv_allowed)
			return -FI_EOPNOTSUPP;
	} else {
		if (!ep->ep_ops.tagged_recv_allowed)
			return -FI_EOPNOTSUPP;
	}

	match_flags = flags & (FI_CLAIM | FI_DISCARD | FI_PEEK);

	ret = __gnix_msg_addr_lookup(ep, src_addr, &gnix_addr);
	if (ret != FI_SUCCESS)
		return ret;

	__gnix_msg_queues(ep, tagged, &queue_lock, &posted_queue, &unexp_queue);

	GNIX_DEBUG(FI_LOG_EP_DATA, "posted_queue = %p\n", posted_queue);

	if (!tagged) {
		tag = 0;
		ignore = ~0;
	}

	COND_ACQUIRE(ep->requires_lock, queue_lock);

retry_match:
	/* Look for a matching unexpected receive request. */
	req = _gnix_match_tag(unexp_queue, tag, ignore,
			      match_flags, context, &gnix_addr);
	if (req) {
		/* Found matching request, populate local fields. */
		req->gnix_ep = ep;
		req->user_context = context;
		req->msg.recv_info[0].recv_addr = (uint64_t)buf;
		req->msg.recv_info[0].recv_len = len;
		req->msg.cum_recv_len = len;

		if (mdesc) {
			md = container_of(mdesc,
					  struct gnix_fid_mem_desc,
					  mr_fid);
			req->msg.recv_info[0].mem_hndl = md->mem_hndl;
		}
		req->msg.recv_md[0] = md;
		req->msg.recv_iov_cnt = 1;
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
			GNIX_DEBUG(FI_LOG_EP_DATA, "matched RNDZV, req: %p\n",
				  req);

			/*
			 * this shouldn't happen
			 */
			if (unlikely(req->vc == NULL)) {
				GNIX_ERR(FI_LOG_EP_DATA,
					 "fab req vc field NULL");
			}

			/* Check if second GET for unaligned data is needed. */
			if (req->msg.send_info[0].send_len > req->msg.recv_info[0].recv_len &&
			    ((req->msg.send_info[0].send_addr + req->msg.recv_info[0].recv_len) &
			     GNI_READ_ALIGN_MASK)) {
				req->msg.recv_flags |= GNIX_MSG_GET_TAIL;
			}

			/* Send length is truncated to receive buffer size. */
			req->msg.cum_send_len =
				MIN(req->msg.cum_send_len,
				    req->msg.recv_info[0].recv_len);

			/* Initiate pull of source data. */
			req->work_fn = req->msg.send_iov_cnt == 1 ?
				__gnix_rndzv_req : __gnix_rndzv_iov_req_build;

			ret = _gnix_vc_queue_work_req(req);

			/* If using FI_MULTI_RECV and there is space left in
			 * the receive buffer, try to match another unexpected
			 * request. */
			if ((req->msg.recv_flags & FI_MULTI_RECV) &&
			    ((len - req->msg.cum_send_len) >= ep->min_multi_recv)) {
				buf += req->msg.cum_send_len;
				len -= req->msg.cum_send_len;

				GNIX_DEBUG(FI_LOG_EP_DATA,
					  "Attempting additional matches, "
					  "req: %p (%p %u)\n",
					  req, buf, len);
				goto retry_match;
			}
		} else {
			/* Matched eager request.  Copy data and generate
			 * completions. */
			GNIX_DEBUG(FI_LOG_EP_DATA, "Matched recv, req: %p\n",
				  req);

			/* Send length is truncated to receive buffer size. */
			req->msg.send_info[0].send_len =
				MIN(req->msg.send_info[0].send_len,
				    req->msg.recv_info[0].recv_len);

			req->msg.cum_send_len = req->msg.send_info[0].send_len;

			/* Copy data from unexpected eager receive buffer. */
			memcpy((void *)buf, (void *)req->msg.send_info[0].send_addr,
			       req->msg.send_info[0].send_len);
			free((void *)req->msg.send_info[0].send_addr);

			__gnix_msg_recv_completion(ep, req);

			/* If using FI_MULTI_RECV and there is space left in
			 * the receive buffer, try to match another unexpected
			 * request. */
			if ((req->msg.recv_flags & FI_MULTI_RECV) &&
			    ((len - req->msg.cum_send_len) >= ep->min_multi_recv)) {
				buf += req->msg.cum_send_len;
				len -= req->msg.cum_send_len;

				GNIX_DEBUG(FI_LOG_EP_DATA,
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

		GNIX_DEBUG(FI_LOG_EP_DATA, "New recv, req: %p\n", req);

		req->type = GNIX_FAB_RQ_RECV;

		req->addr = gnix_addr;
		req->gnix_ep = ep;
		req->user_context = context;

		req->msg.recv_info[0].recv_addr = (uint64_t)buf;
		req->msg.recv_info[0].recv_len = len;
		req->msg.cum_recv_len = len;

		if (mdesc) {
			md = container_of(mdesc,
					struct gnix_fid_mem_desc,
					mr_fid);

			req->msg.recv_info[0].mem_hndl = md->mem_hndl;
		}

		req->msg.recv_md[0] = md;
		req->msg.send_iov_cnt = req->msg.recv_iov_cnt = 1;
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
	COND_RELEASE(ep->requires_lock, queue_lock);

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

	if (unlikely(rendezvous)) {
		switch(req->type) {

		case GNIX_FAB_RQ_SEND:
		case GNIX_FAB_RQ_TSEND:
			assert(req->msg.send_md);
			tag = GNIX_SMSG_T_RNDZV_START;
			tdesc->rndzv_start_hdr.flags = req->msg.send_flags;
			tdesc->rndzv_start_hdr.imm = req->msg.imm;
			tdesc->rndzv_start_hdr.msg_tag = req->msg.tag;
			tdesc->rndzv_start_hdr.mdh = req->msg.send_info[0].mem_hndl;
			tdesc->rndzv_start_hdr.addr = req->msg.send_info[0].send_addr;
			tdesc->rndzv_start_hdr.len = req->msg.send_info[0].send_len;
			tdesc->rndzv_start_hdr.req_addr = (uint64_t)req;

			if (req->msg.send_info[0].send_addr & GNI_READ_ALIGN_MASK) {
				tdesc->rndzv_start_hdr.head =
					*(uint32_t *)(req->msg.send_info[0].send_addr &
						      ~GNI_READ_ALIGN_MASK);
				GNIX_INFO(FI_LOG_EP_DATA,
					  "Sending %d unaligned head bytes (%x)\n",
					  GNI_READ_ALIGN -
					  (req->msg.send_info[0].send_addr &
					   GNI_READ_ALIGN_MASK),
					  tdesc->rndzv_start_hdr.head);
			}

			if ((req->msg.send_info[0].send_addr +
			     req->msg.send_info[0].send_len) &
			    GNI_READ_ALIGN_MASK) {
				tdesc->rndzv_start_hdr.tail =
					*(uint32_t *)((req->msg.send_info[0].send_addr +
						       req->msg.send_info[0].send_len) &
						      ~GNI_READ_ALIGN_MASK);
				GNIX_INFO(FI_LOG_EP_DATA,
					  "Sending %d unaligned tail bytes (%x)\n",
					  (req->msg.send_info[0].send_addr +
					   req->msg.send_info[0].send_len) &
					  GNI_READ_ALIGN_MASK,
					  tdesc->rndzv_start_hdr.tail);
			}

			hdr = &tdesc->rndzv_start_hdr;
			hdr_len = sizeof(tdesc->rndzv_start_hdr);
			/* TODO: Unify send&sendv/recv&recvv, so data will be
			 * req->msg.send_info */
			data = NULL;
			data_len = 0;

			break;

		case GNIX_FAB_RQ_SENDV:
		case GNIX_FAB_RQ_TSENDV:
			assert(req->msg.send_md[0]);
			tag = GNIX_SMSG_T_RNDZV_IOV_START;
			tdesc->rndzv_iov_start_hdr.flags = req->msg.send_flags;
			tdesc->rndzv_iov_start_hdr.imm = req->msg.imm;
			tdesc->rndzv_iov_start_hdr.msg_tag = req->msg.tag;
			tdesc->rndzv_iov_start_hdr.iov_cnt =
				req->msg.send_iov_cnt;
			tdesc->rndzv_iov_start_hdr.req_addr = (uint64_t) req;
			tdesc->rndzv_iov_start_hdr.send_len = req->msg.cum_send_len;

			data_len = sizeof(struct send_info_t) * req->msg.send_iov_cnt;
			data = (void *) req->msg.send_info;
			hdr_len = sizeof(tdesc->rndzv_iov_start_hdr);
			hdr = &tdesc->rndzv_iov_start_hdr;

#if ENABLE_DEBUG
			int i;
			for (i = 0; i < req->msg.send_iov_cnt; i++) {
				GNIX_DEBUG(FI_LOG_EP_DATA,
					   "data[%d].send_addr = 0x%ux, "
					   "data[%d].send_len = %lu, "
					   "data[%d].mem_hndl = %hxx\n", i,
					   ((struct send_info_t *)data)[i].send_addr,
					   i,
					   ((struct send_info_t *)data)[i].send_len,
					   i,
					   ((struct send_info_t *)data)[i].mem_hndl);

			}
#endif
			break;
		default:
			GNIX_FATAL(FI_LOG_EP_DATA, "Invalid request type: %d\n", req->type);
			return -FI_EINVAL;
		}
	} else {
		tag = GNIX_SMSG_T_EGR_W_DATA;

		tdesc->eager_hdr.flags = req->msg.send_flags;
		tdesc->eager_hdr.imm = req->msg.imm;
		tdesc->eager_hdr.msg_tag = req->msg.tag;
		tdesc->eager_hdr.len = req->msg.cum_send_len;

		hdr = &tdesc->eager_hdr;
		hdr_len = sizeof(tdesc->eager_hdr);
		data = (void *)req->msg.send_info[0].send_addr;
		/* If this is not rndzv the send length should always be the
		 * cumulative length of all the send_info lengths */
		data_len = req->msg.cum_send_len;
	}
	tdesc->req = req;
	tdesc->completer_fn = gnix_ep_smsg_completers[tag];

	COND_ACQUIRE(nic->requires_lock, &nic->lock);

	if (unlikely(inject_err)) {
		_gnix_nic_txd_err_inject(nic, tdesc);
		status = GNI_RC_SUCCESS;
	} else {
		status = GNI_SmsgSendWTag(req->vc->gni_ep,
					  hdr, hdr_len, data, data_len,
					  tdesc->id, tag);
	}

	/*
	 * if this is a rendezvous message, we want to generate
	 * IRQ at remote peer.
	 * TODO: Do we need to do this for sendv?
	 */
	if ((status == GNI_RC_SUCCESS) &&
		(tag == GNIX_SMSG_T_RNDZV_START ||
		 tag == GNIX_SMSG_T_RNDZV_IOV_START))
		_gnix_rma_post_irq(req->vc);

	COND_RELEASE(nic->requires_lock, &nic->lock);

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

	if (flags & FI_TRIGGER) {
		struct fi_triggered_context *trigger_context =
				(struct fi_triggered_context *)context;
		if ((trigger_context->event_type != FI_TRIGGER_THRESHOLD) ||
		    (flags & FI_INJECT)) {
			return -FI_EINVAL;
		}
	}

	if (!ep->send_cq) {
		return -FI_ENOCQ;
	}

	if ((flags & FI_INJECT) && (len > GNIX_INJECT_SIZE)) {
		GNIX_INFO(FI_LOG_EP_DATA,
			  "Send length %d exceeds inject max size: %d\n",
			  len, GNIX_INJECT_SIZE);
		return -FI_EINVAL;
	}

	if (!(flags & FI_TAGGED)) {
		if (!ep->ep_ops.msg_send_allowed)
			return -FI_EOPNOTSUPP;
	} else {
		if (!ep->ep_ops.tagged_send_allowed)
			return -FI_EOPNOTSUPP;
	}

	rendezvous = len >= ep->domain->params.msg_rendezvous_thresh;

	/* need a memory descriptor for large sends */
	if (rendezvous && !mdesc) {
		ret = gnix_mr_reg(&ep->domain->domain_fid.fid, (void *)loc_addr,
				 len, FI_READ | FI_WRITE, 0, 0, 0,
				 &auto_mr, NULL);
		if (ret != FI_SUCCESS) {
			GNIX_DEBUG(FI_LOG_EP_DATA,
				  "Failed to auto-register local buffer: %s\n",
				   fi_strerror(-ret));
			return ret;
		}
		flags |= FI_LOCAL_MR;
		mdesc = (void *)auto_mr;
		GNIX_DEBUG(FI_LOG_EP_DATA, "auto-reg MR: %p\n", auto_mr);
	}

	ret = _gnix_vc_ep_get_vc(ep, dest_addr, &vc);
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
		req->msg.send_info[0].mem_hndl = md->mem_hndl;
	}
	req->msg.send_md[0] = md;
	req->msg.send_iov_cnt = 1;
	req->msg.send_flags = flags;
	req->msg.send_info[0].send_len = len;
	req->msg.cum_send_len = len;
	req->msg.imm = data;
	req->flags = 0;

	if (flags & FI_INJECT) {
		memcpy(req->inject_buf, (void *)loc_addr, len);
		req->msg.send_info[0].send_addr = (uint64_t)req->inject_buf;
		req->flags |= FI_INJECT;
	} else {
		req->msg.send_info[0].send_addr = loc_addr;
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

	GNIX_DEBUG(FI_LOG_EP_DATA, "Queuing (%p %d)\n",
		  (void *)loc_addr, len);

	return _gnix_vc_queue_tx_req(req);

err_fr_alloc:
err_get_vc:
	if (auto_mr) {
		fi_close(&auto_mr->fid);
	}
	return ret;
}

ssize_t _gnix_recvv(struct gnix_fid_ep *ep, const struct iovec *iov,
		    void **desc, size_t count, uint64_t src_addr, void *context,
		    uint64_t flags, uint64_t ignore, uint64_t tag)
{
	int i, ret = FI_SUCCESS;
	size_t cum_len = 0;
	struct gnix_fab_req *req = NULL;
	struct gnix_address gnix_addr;
	fastlock_t *queue_lock = NULL;
	struct gnix_tag_storage *posted_queue = NULL;
	struct gnix_tag_storage *unexp_queue = NULL;
	uint64_t match_flags;
	int tagged = flags & FI_TAGGED;
	struct fid_mr *auto_mr;

	if (!ep->recv_cq) {
		return -FI_ENOCQ;
	}

	if (!tagged) {
		if (!ep->ep_ops.msg_send_allowed)
			return -FI_EOPNOTSUPP;

		tag = 0;
		ignore = ~0;
	} else {
		if (!ep->ep_ops.tagged_send_allowed)
			return -FI_EOPNOTSUPP;
	}

	match_flags = flags & (FI_CLAIM | FI_DISCARD | FI_PEEK);

	/*
	 * Lookup the gni addr in the av_table or av_hashtable.
	 * If the gni addr doesn't exist the addr is FI_ADDR_UNSPEC,
	 * meaning this remote node wants to receive from all senders?
	 */
	ret = __gnix_msg_addr_lookup(ep, src_addr, &gnix_addr);
	if (ret != FI_SUCCESS)
		return ret;

	/* calculate cumulative size of the iovec buf lens */
	for (i = 0; i < count; i++) {
		cum_len += iov[i].iov_len;
	}

	/*
	 * Initialize the tag storage objects.
	 * The posted_queue holds information about receives that have
	 * been posted on the remote endpoint.
	 *
	 * The unexp_queue holds information about data that has arrived
	 * prior to posting a receive on the remote endpoint.
	 *
	 * Both {unexp,posted}_queue objects have two sets, one for tagged
	 * messages and the other for untagged messages.
	 *
	 * The untagged queues match based off the source address.
	 *
	 * The tagged queues match based off the tag and source address (when
	 * the ep is created with FI_DIRECTED_RECV).
	 *
	 * A "message" is added to the unexpected queue when it arrives at a
	 * remote endpoint and the completer_fn doesn't find an existing request
	 * in the posted queue (i.e. no fi_recvs have been called (or posted)
	 * on the remote endpoint).
	 */
	__gnix_msg_queues(ep, tagged, &queue_lock, &posted_queue, &unexp_queue);

	COND_ACQUIRE(ep->requires_lock, queue_lock);

	/*
	 * Posting a recv, look for an existing request in the
	 * unexpected queue.
	 */
	req = _gnix_match_tag(unexp_queue, tag, ignore,
			      match_flags, context, &gnix_addr);

	if (req) {
		GNIX_DEBUG(FI_LOG_EP_DATA, "UNEXPECTED, req = %p\n", req);
		/* Found a matching request in the unexpected queue. */

		/*
		 * reset ep, it might be different than the ep the message came
		 * in on.
		 */
		req->gnix_ep = ep;
		req->user_context = context;
		req->flags = 0;
		req->msg.recv_flags = flags;
		req->msg.recv_iov_cnt = count;

		if (tagged) {
			req->type = GNIX_FAB_RQ_TRECVV;
		} else {
			req->type = GNIX_FAB_RQ_RECVV;
		}

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
			req->work_fn = __gnix_rndzv_iov_req_build;
			if (!desc) {	/* Register the memory for the user */
				for (i = 0; i < count; i++) {
					auto_mr = NULL;
					ret = gnix_mr_reg(&ep->domain->
							  domain_fid.fid,
							  iov[i].iov_base,
							  iov[i].iov_len,
							  FI_READ | FI_WRITE, 0,
							  0, 0, &auto_mr,
							  NULL);

					if (ret != FI_SUCCESS) {
						GNIX_DEBUG(FI_LOG_EP_DATA,
							   "Failed to "
							   "auto-register"
							   " local buffer: %s\n"
							   , fi_strerror(-ret));

						for (i--; i >= 0; i--) {
							fi_close(&req->msg.recv_md[i]->mr_fid.fid);
						}

						return ret;
					}

					req->msg.recv_md[i] = container_of(
						(void *) auto_mr,
						struct gnix_fid_mem_desc,
						mr_fid);

					req->msg.recv_info[i].recv_addr = (uint64_t) iov[i].iov_base;
					req->msg.recv_info[i].recv_len = iov[i].iov_len;
					req->msg.recv_info[i].mem_hndl =
						req->msg.recv_md[i]->mem_hndl;

					GNIX_DEBUG(FI_LOG_EP_DATA, "auto-reg MR"
						   ": %p\n",
						   req->msg.recv_md[i]);

				}

				req->msg.send_flags |= FI_LOCAL_MR;
			} else {	/* User registered their memory */
				for (i = 0; i < count; i++) {
					if (!desc[i]) {
						GNIX_WARN(FI_LOG_EP_DATA,
							  "invalid memory reg"
							  "istration (%p).\n",
							  desc[i]);
						return -FI_EINVAL;
					}

					req->msg.recv_md[i] =
						container_of(desc[i],
						struct gnix_fid_mem_desc,
						mr_fid);

					req->msg.recv_info[i].mem_hndl =
						req->msg.recv_md[i]->mem_hndl;
				}
			}

			ret = _gnix_vc_queue_work_req(req);
		} else {
			for (i = 0; i < count; i++) {
				req->msg.recv_info[i].recv_addr = (uint64_t) iov[i].iov_base;
				req->msg.recv_info[i].recv_len = iov[i].iov_len;
			}

			/*
			 * This request is associate with a regular eager smsg,
			 * the rndzv threshold on the sender was not reached or
			 * exceeded.
			 */
			__gnix_msg_unpack_data_into_iov(req->msg.recv_info,
							count,
							req->msg.send_info[0].send_addr,
							req->msg.send_info[0].send_len);

			__gnix_msg_recv_completion(ep, req);
			_gnix_fr_free(ep, req);
		}
	} else {
		/* if peek/claim/discard, we didn't find what we
		 * were looking for, return FI_ENOMSG
		 */
		if (match_flags) {
			__recv_err(ep, context, flags, cum_len,
				   (void *) iov, 0, tag, cum_len, FI_ENOMSG,
				   FI_ENOMSG, NULL);

			/* if handling trecvmsg flags, return here
			 * Never post a receive request from this type of
			 * context
			 */
			ret = FI_SUCCESS;
			goto pdc_exit;
		}

		/*
		 * No matching requests found, create a new one and enqueue
		 * it in the posted queue.
		 */
		req = _gnix_fr_alloc(ep);
		if (req == NULL) {
			ret = -FI_EAGAIN;
			goto err;
		}

		GNIX_DEBUG(FI_LOG_EP_DATA, "EXPECTED, req = %p\n", req);

		if (tagged) {
			req->type = GNIX_FAB_RQ_TRECVV;
		} else {
			req->type = GNIX_FAB_RQ_RECVV;
		}

		if ((flags & GNIX_SUPPRESS_COMPLETION) ||
		    (ep->recv_selective_completion &&
		     !(flags & FI_COMPLETION))) {
			req->msg.recv_flags &= ~FI_COMPLETION;
		} else {
			req->msg.recv_flags |= FI_COMPLETION;
		}

		req->addr = gnix_addr;
		req->gnix_ep = ep;
		req->user_context = context;
		req->flags = 0;

		for (i = 0; i < count; i++) {
			req->msg.recv_info[i].recv_addr = (uint64_t) iov[i].iov_base;
			req->msg.recv_info[i].recv_len = iov[i].iov_len;
			req->msg.recv_md[i] = NULL;
		}

		req->msg.recv_iov_cnt = count;
		req->msg.recv_flags = flags;
		req->msg.cum_recv_len = cum_len;
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
	COND_RELEASE(ep->requires_lock, queue_lock);

	return ret;
}

ssize_t _gnix_sendv(struct gnix_fid_ep *ep, const struct iovec *iov,
		    void **mdesc, size_t count, uint64_t dest_addr,
		    void *context, uint64_t flags, uint64_t tag)
{
	int i, ret = FI_SUCCESS;
	unsigned long long cum_len = 0;
	void *tmp = NULL;
	struct gnix_vc *vc = NULL;
	struct gnix_fab_req *req = NULL;
	struct fid_mr *auto_mr;

	if (!(flags & FI_TAGGED)) {
		if (!ep->ep_ops.msg_send_allowed)
			return -FI_EOPNOTSUPP;
	} else {
		if (!ep->ep_ops.tagged_send_allowed)
			return -FI_EOPNOTSUPP;
	}

	ret = _gnix_vc_ep_get_vc(ep, dest_addr, &vc);
	if (ret != FI_SUCCESS) {
		return ret;
	}

	req = _gnix_fr_alloc(ep);
	if (req == NULL) {
		return -FI_ENOSPC;
	}

	GNIX_DEBUG(FI_LOG_EP_DATA, "Created req - %p\n", req);

	/* calculate cumulative size of the iovec buf lens */
	for (i = 0; i < count; i++) {
		/* TODO: handle possible overflow */
		cum_len += iov[i].iov_len;
	}

	/* Fill out fabric request */
	if (flags & FI_TAGGED) {
		req->type = GNIX_FAB_RQ_TSENDV;
		req->msg.tag = tag;
		req->msg.ignore = 0;
	} else {
		req->type = GNIX_FAB_RQ_SENDV;
		req->msg.tag = 0;
		req->msg.ignore = ~0;
	}

	req->gnix_ep = ep;
	req->user_context = context;
	req->vc = vc;
	req->work_fn = _gnix_send_req;
	req->flags = 0; /* Flags that apply to all message types? */
	req->msg.send_flags = flags;
	req->msg.imm = 0;

	/*
	 * If the cum_len is >= ep->domain->params.msg_rendezvous_thresh
	 * transfer the iovec entries individually.
	 *
	 * For this case, use CtPostFma for iovec lengths that are smaller than
	 * the rendezvous thresh. For CtPostFma:
	 * the sum of the iov lens must be either <= 1GB or <= 1MB if the comm
	 * dom is configured with FmaSharing.
	 * otherwise use PostRdma.
	 */
	if (cum_len >= ep->domain->params.msg_rendezvous_thresh) {
		if (!mdesc) {	/* Register the memory for the user */
			for (i = 0; i < count; i++) {
				auto_mr = NULL;
				ret = gnix_mr_reg(&ep->domain->domain_fid.fid,
						  iov[i].iov_base,
						  iov[i].iov_len,
						  FI_READ | FI_WRITE, 0, 0, 0,
						  &auto_mr, NULL);

				if (ret != FI_SUCCESS) {
					GNIX_DEBUG(FI_LOG_EP_DATA,
						   "Failed to auto-register"
						   " local buffer: %s\n",
						   fi_strerror(-ret));

					for (i--; i >= 0; i--) {
						fi_close(&req->msg.send_md[i]->mr_fid.fid);
					}

					return ret;
				}

				req->msg.send_md[i] = container_of(
					(void *) auto_mr,
					struct gnix_fid_mem_desc,
					mr_fid);

				req->msg.send_info[i].send_addr = (uint64_t) iov[i].iov_base;
				req->msg.send_info[i].send_len = iov[i].iov_len;
				req->msg.send_info[i].mem_hndl =
					req->msg.send_md[i]->mem_hndl;

				GNIX_DEBUG(FI_LOG_EP_DATA, "req->msg.send_md[%d] "
					   "= %p\n", i,
					   req->msg.send_md[i]);

				GNIX_DEBUG(FI_LOG_EP_DATA, "auto-reg MR: %p\n",
					   req->msg.send_md[i]);

			}

			req->msg.send_flags |= FI_LOCAL_MR;
		} else {	/* User registered their memory */
			for (i = 0; i < count; i++) {
				if (!mdesc[i]) {
					GNIX_WARN(FI_LOG_EP_DATA,
						  "invalid memory reg"
						  "istration (%p).\n",
						  mdesc[i]);
					return -FI_EINVAL;
				}

				req->msg.send_md[i] =
					container_of(mdesc[i],
						     struct gnix_fid_mem_desc,
						     mr_fid);

				req->msg.send_info[i].send_addr = (uint64_t) iov[i].iov_base;
				req->msg.send_info[i].send_len = iov[i].iov_len;
				req->msg.send_info[i].mem_hndl =
					req->msg.send_md[i]->mem_hndl;
			}
		}

		req->msg.send_iov_cnt = count;
		req->msg.send_flags |= GNIX_MSG_RENDEZVOUS;
	} else {
		/*
		 * TODO: Use buddy allocator with max alloc lim of
		 * ep->domain->params.msg_rendezvous_thresh
		 */
		/* This is freed in __comp_eager_msg_w_data */
		tmp = malloc(cum_len);
		assert(tmp != NULL);

		__gnix_msg_pack_data_from_iov((uint64_t) tmp, cum_len,
					      iov, count);
		req->msg.send_info[0].send_addr = (uint64_t) tmp;
		req->msg.send_info[0].send_len = cum_len;
	}

	if ((flags & GNIX_SUPPRESS_COMPLETION) ||
	    (ep->send_selective_completion &&
	     !(flags & FI_COMPLETION))) {
		req->msg.send_flags &= ~FI_COMPLETION;
	} else {
		req->msg.send_flags |= FI_COMPLETION;
	}

	req->msg.cum_send_len = (size_t) cum_len;

	return _gnix_vc_queue_tx_req(req);
}
