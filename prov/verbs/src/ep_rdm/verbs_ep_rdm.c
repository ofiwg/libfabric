/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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
#include <arpa/inet.h>
#include <rdma/rdma_cma.h>

#include <fi_list.h>
#include <fi_file.h>
#include <fi_enosys.h>
#include "../fi_verbs.h"
#include "verbs_queuing.h"
#include "verbs_utils.h"


extern struct fi_ops_tagged fi_ibv_rdm_tagged_ops;
extern struct fi_ops_cm fi_ibv_rdm_tagged_ep_cm_ops;
extern struct dlist_entry fi_ibv_rdm_tagged_recv_posted_queue;
extern struct util_buf_pool* fi_ibv_rdm_tagged_request_pool;
extern struct util_buf_pool* fi_ibv_rdm_tagged_extra_buffers_pool;
extern struct fi_provider fi_ibv_prov;

struct fi_ibv_rdm_tagged_conn *fi_ibv_rdm_tagged_conn_hash = NULL;


static int
fi_ibv_rdm_find_max_inline(struct ibv_pd *pd, struct ibv_context *context)
{
	struct ibv_qp_init_attr qp_attr;
	struct ibv_qp *qp = NULL;
	struct ibv_cq *cq = ibv_create_cq(context, 1, NULL, NULL, 0);
	assert(cq);
	int max_inline = 2;
	int rst = 0;

	memset(&qp_attr, 0, sizeof(qp_attr));
	qp_attr.send_cq = cq;
	qp_attr.recv_cq = cq;
	qp_attr.qp_type = IBV_QPT_RC;
	qp_attr.cap.max_send_wr = 1;
	qp_attr.cap.max_recv_wr = 1;
	qp_attr.cap.max_send_sge = 1;
	qp_attr.cap.max_recv_sge = 1;

	do {
		if (qp)
			ibv_destroy_qp(qp);
		qp_attr.cap.max_inline_data = max_inline;
		qp = ibv_create_qp(pd, &qp_attr);
		if (qp) {
			/* 
			 * truescale returns max_inline_data 0
			 */
			if (qp_attr.cap.max_inline_data == 0)
				break;

			/*
			 * iWarp is able to create qp with unsupported
			 * max_inline, lets take first returned value.
			 */
			if (context->device->transport_type == IBV_TRANSPORT_IWARP) {
				max_inline = rst = qp_attr.cap.max_inline_data;
				break;
			}
			rst = max_inline;
		}
	} while (qp && (max_inline < INT_MAX / 2) && (max_inline *= 2));

	if (rst != 0) {
		int pos = rst, neg = max_inline;
		do {
			max_inline = pos + (neg - pos) / 2;
			if (qp)
				ibv_destroy_qp(qp);

			qp_attr.cap.max_inline_data = max_inline;
			qp = ibv_create_qp(pd, &qp_attr);
			if (qp)
				pos = max_inline;
			else
				neg = max_inline;

		} while (neg - pos > 2);

		rst = pos;
	}

	if (qp) {
		ibv_destroy_qp(qp);
	}

	if (cq) {
		ibv_destroy_cq(cq);
	}

	return rst;
}

static int fi_ibv_rdm_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct fi_ibv_rdm_ep *ep;
	struct fi_ibv_rdm_cq *cq;
	struct fi_ibv_av *av;
	int ret;

	ep = container_of(fid, struct fi_ibv_rdm_ep, ep_fid.fid);
	ret = ofi_ep_bind_valid(&fi_ibv_prov, bfid, flags);
	if (ret)
		return ret;

	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct fi_ibv_rdm_cq, cq_fid);

		if (flags & FI_RECV) {
			if (ep->fi_rcq)
				return -EINVAL;
			ep->fi_rcq = cq;
			ep->rx_selective_completion = 
				(flags & FI_SELECTIVE_COMPLETION) ? 1 : 0;
		}

		if (flags & FI_SEND) {
			if (ep->fi_scq)
				return -EINVAL;
			ep->fi_scq = cq;
			ep->tx_selective_completion = 
				(flags & FI_SELECTIVE_COMPLETION) ? 1 : 0;
		}

		/* TODO: this is wrong. CQ to EP is 1:n */
		cq->ep = ep;
		break;
	case FI_CLASS_AV:
		av = container_of(bfid, struct fi_ibv_av, av.fid);
		ep->av = av;

		/* TODO: this is wrong, AV to EP is 1:n */
		ep->av->ep = ep;
		break;
	default:
		return -EINVAL;
	}

	return 0;
}

static ssize_t fi_ibv_rdm_tagged_ep_cancel(fid_t fid, void *ctx)
{
	struct fi_context *context = (struct fi_context *)ctx;
	struct fi_ibv_rdm_ep *ep_rdm = 
		container_of(fid, struct fi_ibv_rdm_ep, ep_fid);
	int err = 1;

	if (!ep_rdm->domain)
		return -EBADF;

	if (!context)
		return -EINVAL;

	if (context->internal[0] == NULL)
		return 0;

	struct fi_ibv_rdm_tagged_request *request = context->internal[0];

	VERBS_DBG(FI_LOG_EP_DATA,
		  "ep_cancel, match %p, tag 0x%llx, len %d, ctx %p\n",
		  request, request->minfo.tag, request->len, request->context);

	struct dlist_entry *found =
		dlist_find_first_match(&fi_ibv_rdm_tagged_recv_posted_queue,
			fi_ibv_rdm_tagged_req_match, request);
	if (found) {
		assert(container_of(found, struct fi_ibv_rdm_tagged_request,
				    queue_entry) == request);
		fi_ibv_rdm_tagged_remove_from_posted_queue(request, ep_rdm);
		VERBS_DBG(FI_LOG_EP_DATA, "\t\t-> SUCCESS, post recv %d\n",
			ep_rdm->posted_recvs);
		err = 0;
	} else {
		request = fi_ibv_rdm_take_first_match_from_postponed_queue
				(fi_ibv_rdm_tagged_req_match, request);
		if (request) {
			fi_ibv_rdm_remove_from_postponed_queue(request);
			err = 0;
		}
	}

	if (!err) {
		fi_ibv_rdm_move_to_errcq(request, FI_ECANCELED);
	}

	return err;
}

static int fi_ibv_rdm_tagged_ep_getopt(fid_t fid, int level, int optname,
				       void *optval, size_t * optlen)
{
	switch (level) {
	case FI_OPT_ENDPOINT:
		return -FI_ENOPROTOOPT;
	default:
		return -FI_ENOPROTOOPT;
	}
	return 0;
}

static int fi_ibv_rdm_tagged_setopt(fid_t fid, int level, int optname,
				    const void *optval, size_t optlen)
{
	switch (level) {
	case FI_OPT_ENDPOINT:
		return -FI_ENOPROTOOPT;
	default:
		return -FI_ENOPROTOOPT;
	}
	return 0;
}

#if 0
static int fi_ibv_ep_enable(struct fid_ep *ep)
{
	struct fi_ibv_rdm_ep *_ep;

	_ep = container_of(ep, struct fi_ibv_rdm_ep, ep_fid);

	assert(_ep->type == FI_EP_RDM);
	return 0;
}
#endif /* 0 */

static int fi_ibv_rdm_tagged_control(fid_t fid, int command, void *arg)
{
	switch (command) {
	case FI_ENABLE:
		return 0;
	default:
		return -FI_ENOSYS;
	}

	return 0;
}

struct fi_ops_ep fi_ibv_rdm_tagged_ep_base_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_ibv_rdm_tagged_ep_cancel,
	.getopt = fi_ibv_rdm_tagged_ep_getopt,
	.setopt = fi_ibv_rdm_tagged_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static int _fi_ibv_rdm_tagged_cm_progress_running = 1;

static void *fi_ibv_rdm_tagged_cm_progress_thread(void *ctx)
{
	struct fi_ibv_rdm_ep *ep = (struct fi_ibv_rdm_ep *)ctx;
	while (_fi_ibv_rdm_tagged_cm_progress_running) {
		if (fi_ibv_rdm_cm_progress(ep)) {
			VERBS_INFO (FI_LOG_EP_DATA,
			"fi_ibv_rdm_cm_progress error\n");
			abort();
		}
		usleep(FI_IBV_RDM_CM_THREAD_TIMEOUT);
	}
	return NULL;
}

static int fi_ibv_rdm_ep_close(fid_t fid)
{
	int ret = FI_SUCCESS;
	int err = FI_SUCCESS;
	void *status = NULL;
	struct fi_ibv_rdm_ep *ep =
		container_of(fid, struct fi_ibv_rdm_ep, ep_fid.fid);

	ep->fi_scq->ep = ep->fi_rcq->ep = NULL;

	ep->is_closing = 1;
	_fi_ibv_rdm_tagged_cm_progress_running = 0;
	pthread_join(ep->cm_progress_thread, &status);
	pthread_mutex_destroy(&ep->cm_lock);

	/* All posted sends are waiting local completions */
	while (ep->posted_sends > 0 && ep->num_active_conns > 0) {
		fi_ibv_rdm_tagged_poll(ep);
	}

	struct fi_ibv_rdm_tagged_conn *conn = NULL, *tmp = NULL;

	HASH_ITER(hh, fi_ibv_rdm_tagged_conn_hash, conn, tmp) {
		HASH_DEL(fi_ibv_rdm_tagged_conn_hash, conn);
		switch (conn->state) {
		case FI_VERBS_CONN_ALLOCATED:
		case FI_VERBS_CONN_REMOTE_DISCONNECT:
		case FI_VERBS_CONN_ESTABLISHED:
			ret = fi_ibv_rdm_start_disconnection(conn);
			break;
		case FI_VERBS_CONN_STARTED:
			while (conn->state != FI_VERBS_CONN_ESTABLISHED &&
			       conn->state != FI_VERBS_CONN_REJECTED) {
				ret = fi_ibv_rdm_cm_progress(ep);
				if (ret) {
					VERBS_INFO(FI_LOG_AV, 
						   "cm progress failed\n");
					break;
				}
			}
			break;
		default:
			break;
		}
	}
	while (ep->num_active_conns) {
		err = fi_ibv_rdm_cm_progress(ep);
		if (err) {
			VERBS_INFO(FI_LOG_AV, "cm progress failed\n");
			ret = (ret == FI_SUCCESS) ? err : ret;
		}
	}

	assert(0 == HASH_COUNT(fi_ibv_rdm_tagged_conn_hash) &&
	       NULL == fi_ibv_rdm_tagged_conn_hash);

	VERBS_INFO(FI_LOG_AV, "DISCONNECT complete\n");
	assert(ep->scq && ep->rcq);
	if (ibv_destroy_cq(ep->scq) || ibv_destroy_cq(ep->rcq)) {
		VERBS_INFO_ERRNO(FI_LOG_AV, "ibv_destroy_cq failed\n", errno);
		ret = (ret == FI_SUCCESS) ? -errno : ret;
	}

	errno = 0;
	fi_ibv_destroy_ep(FI_EP_RDM, ep->cm.rai, &(ep->cm.listener));
	if (errno) {
		VERBS_INFO_ERRNO(FI_LOG_AV, "ibv_destroy_ep failed\n", errno);
		ret = (ret == FI_SUCCESS) ? -errno : ret;
	}

	/* TODO: move queues & related pools cleanup to close CQ*/
	fi_ibv_rdm_clean_queues();

	util_buf_pool_destroy(fi_ibv_rdm_tagged_request_pool);
	util_buf_pool_destroy(fi_ibv_rdm_tagged_extra_buffers_pool);
	util_buf_pool_destroy(fi_ibv_rdm_postponed_pool);

	free(ep);

	return ret;
}

#if 0
static int fi_ibv_ep_sync(fid_t fid, uint64_t flags, void *context)
{
	struct fi_ibv_rdm_ep *ep;

	ep = container_of(fid, struct fi_ibv_rdm_ep, ep_fid);

	if (ep->type == FI_EP_MSG) {
		return 0;
	} else if (ep->type == FI_EP_RDM) {
		if (!flags || (flags & FI_SEND)) {
			while (ep->pend_send) {
				fi_ibv_rdm_tagged_poll(ep);
			}
		}

		if (!flags || (flags & FI_RECV)) {
			while (ep->pend_recv) {
				fi_ibv_rdm_tagged_poll(ep);
			}
		}

		if (!flags || (flags & FI_READ)) {
		}

		if (!flags || (flags & FI_WRITE) || (flags & FI_WRITE)) {
		}
	}
	return 0;
}
#endif /* 0 */

struct fi_ops fi_ibv_rdm_tagged_ep_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_rdm_ep_close,
	.bind = fi_ibv_rdm_ep_bind,
	.control = fi_ibv_rdm_tagged_control,
	.ops_open = fi_no_ops_open,
};

int fi_ibv_open_rdm_ep(struct fid_domain *domain, struct fi_info *info,
			struct fid_ep **ep, void *context)
{
	struct fi_ibv_domain *_domain = 
		container_of(domain, struct fi_ibv_domain, domain_fid);
	int ret = 0;

	if (!info || !info->ep_attr || !info->domain_attr ||
	    strncmp(_domain->verbs->device->name, info->domain_attr->name,
		    strlen(_domain->verbs->device->name)) ||
	    (!info->rx_attr ||
	     info->rx_attr->size <= FI_IBV_RDM_TAGGED_DFLT_BUFFER_NUM))
	{
		return -FI_EINVAL;
	}

	struct fi_ibv_rdm_ep *_ep;
	_ep = calloc(1, sizeof *_ep);
	if (!_ep) {
		return -FI_ENOMEM;
	}

	_ep->domain = _domain;
	_ep->ep_fid.fid.fclass = FI_CLASS_EP;
	_ep->ep_fid.fid.context = context;
	_ep->ep_fid.fid.ops = &fi_ibv_rdm_tagged_ep_ops;
	_ep->ep_fid.ops = &fi_ibv_rdm_tagged_ep_base_ops;
	_ep->ep_fid.tagged = &fi_ibv_rdm_tagged_ops;
	_ep->ep_fid.rma = fi_ibv_rdm_ep_ops_rma(_ep);
	_ep->ep_fid.cm = &fi_ibv_rdm_tagged_ep_cm_ops;
	_ep->tx_selective_completion = 0;
	_ep->rx_selective_completion = 0;

	switch (info->ep_attr->protocol) {
	case FI_PROTO_IB_RDM:
		_ep->topcode = IBV_WR_RDMA_WRITE_WITH_IMM;
		break;
	case FI_PROTO_IWARP_RDM:
		_ep->topcode = IBV_WR_SEND;
		break;
	default:
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Unsupported protocol\n");
		ret = -FI_ENODATA;
		goto err;
	}

	ret = fi_ibv_create_ep(NULL, NULL, 0, info, &_ep->cm.rai, &_ep->cm.listener);
	if (ret) {
		goto err;
	}

	if (rdma_listen(_ep->cm.listener, 1024)) {
		VERBS_INFO(FI_LOG_EP_CTRL, "rdma_listen failed: %s\n",
			strerror(errno));
		ret = -FI_EOTHER;
		goto err;
	}

	_ep->n_buffs = FI_IBV_RDM_TAGGED_DFLT_BUFFER_NUM;
	_ep->buff_len = FI_IBV_RDM_DFLT_BUFFER_SIZE;
	_ep->rndv_threshold = FI_IBV_RDM_DFLT_BUFFERED_SSIZE;
	
	_ep->rq_wr_depth = info->rx_attr->size;

	/*
	 * max number of WRs in SQ is n_buffer for send and
	 * the same amount for buffer releasing from recv
	 */
	_ep->sq_wr_depth = 2 * (_ep->n_buffs + 1);

	_ep->posted_sends = 0;
	_ep->posted_recvs = 0;
	_ep->recv_preposted_threshold = MAX(0.2 * _ep->rq_wr_depth, _ep->n_buffs);
	VERBS_INFO(FI_LOG_EP_CTRL, "recv preposted threshold: %d\n",
		   _ep->recv_preposted_threshold);

	fi_ibv_rdm_tagged_request_pool = util_buf_pool_create(
		sizeof(struct fi_ibv_rdm_tagged_request),
		FI_IBV_RDM_MEM_ALIGNMENT, 0, 100);

	fi_ibv_rdm_postponed_pool = util_buf_pool_create(
		sizeof(struct fi_ibv_rdm_postponed_entry),
		FI_IBV_RDM_MEM_ALIGNMENT, 0, 100);

	fi_ibv_rdm_tagged_extra_buffers_pool = util_buf_pool_create(
		_ep->buff_len, FI_IBV_RDM_MEM_ALIGNMENT, 0, 100);

	_ep->max_inline_rc = 
		fi_ibv_rdm_find_max_inline(_ep->domain->pd, _ep->domain->verbs);

	_ep->scq_depth = FI_IBV_RDM_TAGGED_DFLT_SCQ_SIZE;
	_ep->rcq_depth = FI_IBV_RDM_TAGGED_DFLT_RCQ_SIZE;

	_ep->scq = ibv_create_cq(_ep->domain->verbs, _ep->scq_depth, _ep,
				 NULL, 0);
	if (_ep->scq == NULL) {
		VERBS_INFO_ERRNO(FI_LOG_EP_CTRL, "ibv_create_cq", errno);
		ret = -FI_EOTHER;
		goto err;
	}

	_ep->rcq =
	    ibv_create_cq(_ep->domain->verbs, _ep->rcq_depth, _ep, NULL, 0);
	if (_ep->rcq == NULL) {
		VERBS_INFO_ERRNO(FI_LOG_EP_CTRL, "ibv_create_cq", errno);
		ret = -FI_EOTHER;
		goto err;
	}

	*ep = &_ep->ep_fid;

	_ep->is_closing = 0;
	fi_ibv_rdm_tagged_req_hndls_init();

	pthread_mutex_init(&_ep->cm_lock, NULL);
	ret = pthread_create(&_ep->cm_progress_thread, NULL,
			     &fi_ibv_rdm_tagged_cm_progress_thread,
			     (void *)_ep);
	if (ret) {
		VERBS_INFO(FI_LOG_EP_CTRL,
			"Failed to launch CM progress thread, err :%d\n", ret);
		ret = -FI_EOTHER;
		goto err;
	}

	return ret;
err:
	free(_ep);
	return ret;
}
