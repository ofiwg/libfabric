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

#include <ofi_list.h>
#include <ofi_file.h>
#include <ofi_enosys.h>
#include "../fi_verbs.h"
#include "verbs_queuing.h"
#include "verbs_utils.h"


extern struct fi_ops_tagged fi_ibv_rdm_tagged_ops;
extern struct fi_ops_cm fi_ibv_rdm_tagged_ep_cm_ops;
extern struct fi_provider fi_ibv_prov;
extern struct fi_ops_msg fi_ibv_rdm_ep_msg_ops;
extern struct fi_ops_rma fi_ibv_rdm_ep_rma_ops;

static int fi_ibv_rdm_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct fi_ibv_rdm_ep *ep;
	struct fi_ibv_rdm_cq *cq;
	struct fi_ibv_av *av;
	struct fi_ibv_rdm_cntr *cntr;
	int ret;

	ep = container_of(fid, struct fi_ibv_rdm_ep, ep_fid.fid);
	ret = ofi_ep_bind_valid(&fi_ibv_prov, bfid, flags);
	if (ret)
		return ret;

	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct fi_ibv_rdm_cq, cq_fid);
		if (ep->domain != cq->domain) {
			return -FI_EINVAL;
		}

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
		av = container_of(bfid, struct fi_ibv_av, av_fid.fid);
		if (ep->domain != av->domain)
			return -FI_EINVAL;
		ep->av = av;
		break;
	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct fi_ibv_rdm_cntr, fid.fid);
		if (ep->domain != cntr->domain) {
			return -FI_EINVAL;
		}

		if ((flags & FI_REMOTE_READ) || (flags & FI_REMOTE_WRITE)) {
			return -FI_ENOSYS;
		}

		if (flags & FI_SEND) {
			ep->send_cntr = cntr;
			ofi_atomic_inc32(&ep->send_cntr->ep_ref);
		}
		if (flags & FI_RECV) {
			ep->recv_cntr = cntr;
			ofi_atomic_inc32(&ep->recv_cntr->ep_ref);
		}
		if (flags & FI_READ) {
			ep->read_cntr = cntr;
			ofi_atomic_inc32(&ep->read_cntr->ep_ref);
		}
		if (flags & FI_WRITE) {
			ep->write_cntr = cntr;
			ofi_atomic_inc32(&ep->write_cntr->ep_ref);
		}

		break;
	default:
		return -EINVAL;
	}

	return 0;
}

static ssize_t fi_ibv_rdm_cancel(fid_t fid, void *ctx)
{
	struct fi_context *context = (struct fi_context *)ctx;
	struct fi_ibv_rdm_ep *ep_rdm = 
		container_of(fid, struct fi_ibv_rdm_ep, ep_fid);
	int err = -FI_ENOENT;

	if (!ep_rdm->domain)
		return -EBADF;

	if (!context)
		return -EINVAL;

	if (context->internal[0] == NULL)
		return 0;

	struct fi_ibv_rdm_request *request = context->internal[0];

	VERBS_DBG(FI_LOG_EP_DATA,
		  "ep_cancel, match %p, tag 0x%" PRIx64 ", len %" PRIu64 ", ctx %p\n",
		  request, request->minfo.tag, request->len, request->context);

	struct dlist_entry *found =
		dlist_find_first_match(&ep_rdm->fi_ibv_rdm_posted_queue,
					fi_ibv_rdm_req_match, request);
	if (found) {
		uint32_t posted_recvs;

		assert(container_of(found, struct fi_ibv_rdm_request,
				    queue_entry) == request);
		request->context->internal[0] = NULL;
		posted_recvs = fi_ibv_rdm_remove_from_posted_queue(request, ep_rdm);
		(void)posted_recvs;
		VERBS_DBG(FI_LOG_EP_DATA, "\t\t-> SUCCESS, post recv %"PRIu32"\n",
			  posted_recvs);
		err = 0;
	} else {
		request = fi_ibv_rdm_take_first_match_from_postponed_queue
		    (fi_ibv_rdm_req_match, request, ep_rdm);
		if (request) {
			fi_ibv_rdm_remove_from_postponed_queue(request);
			err = 0;
		}
	}

	if (!err) {
		fi_ibv_rdm_cntr_inc_err(ep_rdm->recv_cntr);

		if (request->comp_flags & FI_COMPLETION) {
			fi_ibv_rdm_move_to_errcq(ep_rdm->fi_rcq, request,
						 FI_ECANCELED);
		}
		request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;
	}

	return err;
}

static int fi_ibv_rdm_getopt(fid_t fid, int level, int optname, void *optval,
			     size_t * optlen)
{
	struct fi_ibv_rdm_ep *ep_rdm = 
		container_of(fid, struct fi_ibv_rdm_ep, ep_fid);

	if (level != FI_OPT_ENDPOINT) {
		return -FI_ENOPROTOOPT;
	}

	if (optname != FI_OPT_MIN_MULTI_RECV) {
		return -FI_ENOPROTOOPT;
	}

	*(size_t *)optval = ep_rdm->min_multi_recv_size;
	*optlen = sizeof(size_t);

	return 0;
}

static int fi_ibv_rdm_setopt(fid_t fid, int level, int optname,
			     const void *optval, size_t optlen)
{
	struct fi_ibv_rdm_ep *ep_rdm =
		container_of(fid, struct fi_ibv_rdm_ep, ep_fid);

	if (level != FI_OPT_ENDPOINT) {
		return -FI_ENOPROTOOPT;
	}

	if (optname != FI_OPT_MIN_MULTI_RECV) {
		return -FI_ENOPROTOOPT;
	}

	ep_rdm->min_multi_recv_size = *(size_t *)optval;
	return 0;
}

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

struct fi_ops_ep fi_ibv_rdm_ep_base_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_ibv_rdm_cancel,
	.getopt = fi_ibv_rdm_getopt,
	.setopt = fi_ibv_rdm_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static int fi_ibv_rdm_ep_match(struct slist_entry *item,
			       const void *ep)
{
	const struct fi_ibv_rdm_ep *ep_obj = (struct fi_ibv_rdm_ep *)ep;
	return (item == &ep_obj->list_entry);
}

static int fi_ibv_rdm_ep_close(fid_t fid)
{
	int ret = FI_SUCCESS;
	int err = FI_SUCCESS;
	struct fi_ibv_rdm_ep *ep =
		container_of(fid, struct fi_ibv_rdm_ep, ep_fid.fid);
	struct fi_ibv_rdm_av_entry *av_entry = NULL, *tmp = NULL;
	struct slist_entry *item, *prev_item;
	struct fi_ibv_rdm_conn *conn = NULL;

	if (ep->fi_scq)
		ep->fi_scq->ep = NULL;
	if (ep->fi_rcq)
		ep->fi_rcq->ep = NULL;

	ep->is_closing = 1;

	/* All posted sends are waiting local completions */
	while (ep->posted_sends > 0 && ep->num_active_conns > 0)
		fi_ibv_rdm_tagged_poll(ep);

	if (ep->send_cntr) {
		ofi_atomic_dec32(&ep->send_cntr->ep_ref);
		ep->send_cntr = 0;
	}

	if (ep->recv_cntr) {
		ofi_atomic_dec32(&ep->recv_cntr->ep_ref);
		ep->recv_cntr = 0;
	}

	if (ep->read_cntr) {
		ofi_atomic_dec32(&ep->read_cntr->ep_ref);
		ep->read_cntr = 0;
	}

	if (ep->write_cntr) {
		ofi_atomic_dec32(&ep->write_cntr->ep_ref);
		ep->write_cntr = 0;
	}

	slist_remove_first_match(&ep->domain->ep_list,
				 fi_ibv_rdm_ep_match, ep);

	HASH_ITER(hh, ep->domain->rdm_cm->av_hash, av_entry, tmp) {
		pthread_mutex_lock(&ep->domain->rdm_cm->cm_lock);
		HASH_FIND(hh, av_entry->conn_hash, &ep,
			  sizeof(struct fi_ibv_rdm_ep *), conn);
		if (conn) {
			switch (conn->state) {
			case FI_VERBS_CONN_ALLOCATED:
			case FI_VERBS_CONN_ESTABLISHED:
				ret = fi_ibv_rdm_start_disconnection(conn);
				pthread_mutex_unlock(&ep->domain->rdm_cm->cm_lock);
				break;
			case FI_VERBS_CONN_STARTED:
				pthread_mutex_unlock(&ep->domain->rdm_cm->cm_lock);
				/* No need to hold `rdm_cm::cm_lock` during
				 * CM progressing of the EP */
				while (conn->state != FI_VERBS_CONN_ESTABLISHED &&
				       conn->state != FI_VERBS_CONN_REJECTED &&
				       conn->state != FI_VERBS_CONN_CLOSED) {
					ret = fi_ibv_rdm_cm_progress(ep);
					if (ret) {
						VERBS_INFO(FI_LOG_AV, 
							   "cm progress failed\n");
						break;
					}
				}
				break;
			default:
				pthread_mutex_unlock(&ep->domain->rdm_cm->cm_lock);
				break;
			}
		} else {
			pthread_mutex_unlock(&ep->domain->rdm_cm->cm_lock);
		}
	}

        /* ok, all connections are initiated to disconnect. now wait
	 * till all connections are switch to state 'closed' */
	HASH_ITER(hh, ep->domain->rdm_cm->av_hash, av_entry, tmp) {
		pthread_mutex_lock(&ep->domain->rdm_cm->cm_lock);
		HASH_FIND(hh, av_entry->conn_hash, &ep,
			  sizeof(struct fi_ibv_rdm_ep *), conn);
		pthread_mutex_unlock(&ep->domain->rdm_cm->cm_lock);
		if (conn) {
			/* No need to hold `rdm_cm::cm_lock` during
			 * polling of receive CQ */
			while(conn->state != FI_VERBS_CONN_CLOSED &&
			      conn->state != FI_VERBS_CONN_ALLOCATED) {
				fi_ibv_rdm_tagged_poll_recv(ep);
				err = fi_ibv_rdm_cm_progress(ep);
				if (err) {
					VERBS_INFO(FI_LOG_AV, "cm progress failed\n");
					ret = (ret == FI_SUCCESS) ? err : ret;
				}
			}
		}
	}

        /* now destroy all connections that wasn't removed from HASH */
	HASH_ITER(hh, ep->domain->rdm_cm->av_hash, av_entry, tmp) {
		pthread_mutex_lock(&ep->domain->rdm_cm->cm_lock);
		HASH_FIND(hh, av_entry->conn_hash, &ep,
			  sizeof(struct fi_ibv_rdm_ep *), conn);
		if (conn) {
			HASH_DEL(av_entry->conn_hash, conn);
			pthread_mutex_unlock(&ep->domain->rdm_cm->cm_lock);
			/* No need to hold `rdm_cm::cm_lock` during
			 * connection cleanup */
			fi_ibv_rdm_conn_cleanup(conn);
		} else {
			pthread_mutex_unlock(&ep->domain->rdm_cm->cm_lock);
		}
	}

	/* now destroy all connection that was removed from HASH */
	slist_foreach(&ep->domain->rdm_cm->av_removed_entry_head,
		      item, prev_item) {
		OFI_UNUSED(prev_item); /* Compiler complains about unused variable */
		av_entry = container_of(item,
					struct fi_ibv_rdm_av_entry,
					removed_next);
		pthread_mutex_lock(&ep->domain->rdm_cm->cm_lock);
		HASH_FIND(hh, av_entry->conn_hash, &ep,
			  sizeof(struct fi_ibv_rdm_ep *), conn);
		if (conn) {
			HASH_DEL(av_entry->conn_hash, conn);
			pthread_mutex_unlock(&ep->domain->rdm_cm->cm_lock);
			/* No need to hold `rdm_cm::cm_lock` during
			 * connection cleanup */
			fi_ibv_rdm_conn_cleanup(conn);
		} else {
			pthread_mutex_unlock(&ep->domain->rdm_cm->cm_lock);
		}
		/* do NOT free av_entry and do NOT remove from
		 * the List of removed AV entries, becasue we need to
		 * remove remaining connections during domain closure */
	}

	/* TODO: MUST be removed in DOMAIN_CLOSE */
	/*assert(HASH_COUNT(av_entry->conn_hash) == 0 &&
	       av_entry->conn_hash == NULL);*/
	free(ep->domain->rdm_cm->av_table);

	VERBS_INFO(FI_LOG_AV, "DISCONNECT complete\n");
	assert(ep->scq && ep->rcq);
	if (ibv_destroy_cq(ep->scq)) {
		VERBS_INFO_ERRNO(FI_LOG_AV, "ep->scq: ibv_destroy_cq failed",
				 errno);
		ret = (ret == FI_SUCCESS) ? -errno : ret;
	}

	if (ibv_destroy_cq(ep->rcq)) {
		VERBS_INFO_ERRNO(FI_LOG_AV, "ep->rcq: ibv_destroy_cq failed",
				 errno);
		ret = (ret == FI_SUCCESS) ? -errno : ret;
	}

	rdma_freeaddrinfo(ep->rai);

	/* TODO: move queues & related pools cleanup to close CQ*/
	fi_ibv_rdm_clean_queues(ep);

	util_buf_pool_destroy(ep->fi_ibv_rdm_request_pool);
	util_buf_pool_destroy(ep->fi_ibv_rdm_multi_request_pool);
	util_buf_pool_destroy(ep->fi_ibv_rdm_extra_buffers_pool);
	util_buf_pool_destroy(ep->fi_ibv_rdm_postponed_pool);

	fi_freeinfo(ep->info);

	free(ep);

	return ret;
}

struct fi_ops fi_ibv_rdm_ep_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_rdm_ep_close,
	.bind = fi_ibv_rdm_ep_bind,
	.control = fi_ibv_rdm_tagged_control,
	.ops_open = fi_no_ops_open,
};

int fi_ibv_rdm_open_ep(struct fid_domain *domain, struct fi_info *info,
			struct fid_ep **ep, void *context)
{
	struct fi_ibv_domain *_domain = 
		container_of(domain, struct fi_ibv_domain, util_domain.domain_fid);
	int ret = 0;

	if (!info || !info->ep_attr || !info->domain_attr ||
	    strncmp(_domain->verbs->device->name, info->domain_attr->name,
		    strlen(_domain->verbs->device->name)))
	{
		return -FI_EINVAL;
	}

	struct fi_ibv_rdm_ep *_ep;
	_ep = calloc(1, sizeof *_ep);
	if (!_ep)
		return -FI_ENOMEM;

	_ep->info = fi_dupinfo(info);
	if (!_ep->info) {
		ret = -FI_ENOMEM;
		goto err1;
	}
	_ep->domain = _domain;
	_ep->ep_fid.fid.fclass = FI_CLASS_EP;
	_ep->ep_fid.fid.context = context;
	_ep->ep_fid.fid.ops = &fi_ibv_rdm_ep_ops;
	_ep->ep_fid.ops = &fi_ibv_rdm_ep_base_ops;
	_ep->ep_fid.cm = &fi_ibv_rdm_tagged_ep_cm_ops;
	_ep->ep_fid.msg = &fi_ibv_rdm_ep_msg_ops;
	_ep->ep_fid.rma = &fi_ibv_rdm_ep_rma_ops;
	_ep->ep_fid.tagged = &fi_ibv_rdm_tagged_ops;
	_ep->tx_selective_completion = 0;
	_ep->rx_selective_completion = 0;
	_ep->n_buffs = fi_ibv_gl_data.rdm.buffer_num;

	VERBS_INFO(FI_LOG_EP_CTRL, "inject_size: %zu\n",
		   info->tx_attr->inject_size);

	_ep->rndv_threshold = info->tx_attr->inject_size;
	_ep->iov_per_rndv_thr =
		(_ep->rndv_threshold / sizeof(struct iovec));
	VERBS_INFO(FI_LOG_EP_CTRL, "rndv_threshold: %d\n",
		   _ep->rndv_threshold);

	_ep->buff_len = rdm_buffer_size(info->tx_attr->inject_size);
	VERBS_INFO(FI_LOG_EP_CTRL, "buff_len: %d\n", _ep->buff_len);

	_ep->tx_op_flags = info->tx_attr->op_flags;
	_ep->rx_op_flags = info->rx_attr->op_flags;
	_ep->min_multi_recv_size = (_ep->rx_op_flags & FI_MULTI_RECV) ?
				   info->tx_attr->inject_size : 0;
	_ep->rndv_seg_size = fi_ibv_gl_data.rdm.rndv_seg_size;
	_ep->rq_wr_depth = info->rx_attr->size;
	/* one more outstanding slot for releasing eager buffers */
	_ep->sq_wr_depth = _ep->n_buffs + 1;
	if (!strncmp(fi_ibv_gl_data.rdm.eager_send_opcode,
		     "IBV_WR_RDMA_WRITE_WITH_IMM",
		     strlen("IBV_WR_RDMA_WRITE_WITH_IMM"))) {
		_ep->eopcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	} else if (!strncmp(fi_ibv_gl_data.rdm.eager_send_opcode,
			    "IBV_WR_SEND",
			    strlen("IBV_WR_SEND"))) {
		_ep->eopcode = IBV_WR_SEND;
	} else {
		VERBS_INFO(FI_LOG_CORE,
			   "Invalid value of rdm_eager_send_opcode\n");
		ret = -FI_EINVAL;
		goto err2;
	}

	switch (info->ep_attr->protocol) {
	case FI_PROTO_IB_RDM:
		if (_ep->eopcode != IBV_WR_RDMA_WRITE_WITH_IMM &&
		    _ep->eopcode != IBV_WR_SEND) {
			VERBS_INFO(FI_LOG_CORE,
				   "Unsupported eager operation code\n");
			ret = -FI_ENODATA;
			goto err2;
		}
		break;
	case FI_PROTO_IWARP_RDM:
		if (_ep->eopcode != IBV_WR_SEND) {
			VERBS_INFO(FI_LOG_CORE,
				   "Unsupported eager operation code\n");
			ret = -FI_ENODATA;
			goto err1;
		}
		break;
	default:
		VERBS_INFO(FI_LOG_CORE, "Unsupported protocol\n");
		ret = -FI_ENODATA;
		goto err2;
	}

	ret = fi_ibv_get_rdma_rai(NULL, NULL, 0, info, &_ep->rai);
	if (ret) {
		goto err2;
	}
	ret = fi_ibv_rdm_cm_bind_ep(_ep->domain->rdm_cm, _ep);
	if (ret) {
		goto err2;
	}

	_ep->posted_sends = 0;
	_ep->posted_recvs = 0;
	_ep->recv_preposted_threshold = MAX(0.2 * _ep->rq_wr_depth, _ep->n_buffs);
	VERBS_INFO(FI_LOG_EP_CTRL, "recv preposted threshold: %d\n",
		   _ep->recv_preposted_threshold);

	ret = util_buf_pool_create(&_ep->fi_ibv_rdm_request_pool,
				   sizeof(struct fi_ibv_rdm_request),
				   FI_IBV_MEM_ALIGNMENT, 0,
				   FI_IBV_POOL_BUF_CNT);
	if (ret)
		goto err3;

	ret = util_buf_pool_create(&_ep->fi_ibv_rdm_multi_request_pool,
				   sizeof(struct fi_ibv_rdm_multi_request),
				   FI_IBV_MEM_ALIGNMENT, 0,
				   FI_IBV_POOL_BUF_CNT);
	if (ret)
		goto err3;

	ret = util_buf_pool_create(&_ep->fi_ibv_rdm_postponed_pool,
				   sizeof(struct fi_ibv_rdm_postponed_entry),
				   FI_IBV_MEM_ALIGNMENT, 0,
				   FI_IBV_POOL_BUF_CNT);
	if (ret)
		goto err3;

	ret = util_buf_pool_create(&_ep->fi_ibv_rdm_extra_buffers_pool,
				   _ep->buff_len, FI_IBV_MEM_ALIGNMENT,
				   0, FI_IBV_POOL_BUF_CNT);
	if (ret)
		goto err3;

	dlist_init(&_ep->fi_ibv_rdm_posted_queue);
	dlist_init(&_ep->fi_ibv_rdm_postponed_queue);
	dlist_init(&_ep->fi_ibv_rdm_unexp_queue);
	dlist_init(&_ep->fi_ibv_rdm_multi_recv_list);

	_ep->max_inline_rc =
		fi_ibv_find_max_inline(_ep->domain->pd, _ep->domain->verbs, IBV_QPT_RC);

	_ep->scq_depth = FI_IBV_RDM_TAGGED_DFLT_SCQ_SIZE;
	_ep->rcq_depth = FI_IBV_RDM_TAGGED_DFLT_RCQ_SIZE;

	_ep->scq = ibv_create_cq(_ep->domain->verbs, _ep->scq_depth, _ep,
				 NULL, 0);
	if (_ep->scq == NULL) {
		VERBS_INFO_ERRNO(FI_LOG_EP_CTRL, "ibv_create_cq", errno);
		ret = -FI_EOTHER;
		goto err3;
	}

	_ep->rcq =
	    ibv_create_cq(_ep->domain->verbs, _ep->rcq_depth, _ep, NULL, 0);
	if (_ep->rcq == NULL) {
		VERBS_INFO_ERRNO(FI_LOG_EP_CTRL, "ibv_create_cq", errno);
		ret = -FI_EOTHER;
		goto err4;
	}

	*ep = &_ep->ep_fid;

	_ep->is_closing = 0;
	fi_ibv_rdm_req_hndls_init();

	slist_insert_tail(&_ep->list_entry, &_domain->ep_list);

	return ret;
err4:
	if (ibv_destroy_cq(_ep->scq))
		VERBS_INFO_ERRNO(FI_LOG_EP_CTRL, "ibv_destroy_cq", errno);
err3:
	if (_ep->fi_ibv_rdm_request_pool)
		util_buf_pool_destroy(_ep->fi_ibv_rdm_request_pool);
	if (_ep->fi_ibv_rdm_multi_request_pool)
		util_buf_pool_destroy(_ep->fi_ibv_rdm_multi_request_pool);
	if (_ep->fi_ibv_rdm_postponed_pool)
		util_buf_pool_destroy(_ep->fi_ibv_rdm_postponed_pool);
	if (_ep->fi_ibv_rdm_extra_buffers_pool)
		util_buf_pool_destroy(_ep->fi_ibv_rdm_extra_buffers_pool);
err2:
	fi_freeinfo(_ep->info);
err1:
	free(_ep);
	return ret;
}
