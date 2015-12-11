/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#include <ifaddrs.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <rdma/rdma_cma.h>

#include <fi_list.h>
#include <fi_enosys.h>
#include "../fi_verbs.h"
#include "verbs_queuing.h"
#include "verbs_utils.h"


extern struct fi_ops_tagged fi_ibv_rdm_tagged_ops;
extern struct fi_ops_cm fi_ibv_rdm_tagged_ep_cm_ops;
extern struct dlist_entry fi_ibv_rdm_tagged_recv_posted_queue;
extern struct fi_ibv_mem_pool fi_ibv_rdm_tagged_request_pool;
extern struct fi_ibv_mem_pool fi_ibv_rdm_tagged_unexp_buffers_pool;
extern struct fi_provider fi_ibv_prov;

struct fi_ibv_rdm_tagged_conn *fi_ibv_rdm_tagged_conn_hash = NULL;


static int
fi_ibv_rdm_tagged_find_max_inline_size(struct ibv_pd *pd,
				       struct ibv_context *context)
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
		if (qp)
			rst = max_inline;
	} while (qp && (max_inline *= 2));

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

static int fi_ibv_rdm_tagged_ep_bind(struct fid *fid, struct fid *bfid,
				     uint64_t flags)
{
	struct fi_ibv_rdm_ep *ep;
	struct fi_ibv_cq *cq;
	struct fi_ibv_av *av;

	ep = container_of(fid, struct fi_ibv_rdm_ep, ep_fid.fid);

	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct fi_ibv_cq, cq_fid);

		if (flags & FI_RECV) {
			if (ep->fi_rcq)
				return -EINVAL;
			ep->fi_rcq = cq;
		}
		if (flags & FI_SEND) {
			if (ep->fi_scq)
				return -EINVAL;
			ep->fi_scq = cq;
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
	struct fi_ibv_rdm_ep *fid_ep;
	struct fi_context *context = (struct fi_context *)ctx;
	int err = 1;

	fid_ep = container_of(fid, struct fi_ibv_rdm_ep, ep_fid);
	if (!fid_ep->domain)
		return -EBADF;

	if (!context)
		return -EINVAL;

	if (context->internal[0] == NULL)
		return 0;

	struct fi_ibv_rdm_tagged_request *request = context->internal[0];

	VERBS_DBG(FI_LOG_EP_DATA,
		  "ep_cancel, match %p, tag 0x%llx, len %d, ctx %p\n",
		  request, (long long unsigned)request->tag,
		  request->len, request->context);

	struct dlist_entry *found =
	    dlist_find_first_match(&fi_ibv_rdm_tagged_recv_posted_queue,
				   fi_ibv_rdm_tagged_req_match, request);

	if (found) {
		assert(container_of(found, struct fi_ibv_rdm_tagged_request,
				    queue_entry) == request);

		fi_ibv_rdm_tagged_remove_from_posted_queue(request, fid_ep);

		assert(request->send_completions_wait == 0);
		FI_IBV_RDM_TAGGED_DBG_REQUEST("to_pool: ", request,
					      FI_LOG_DEBUG);

		fi_ibv_mem_pool_return(&request->mpe,
				       &fi_ibv_rdm_tagged_request_pool);

		VERBS_DBG(FI_LOG_EP_DATA,
			  "\t\t-> SUCCESS, pend recv %d\n", fid_ep->pend_recv);

		err = 0;
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
		if (fi_ibv_rdm_tagged_cm_progress(ep)) {
			VERBS_INFO (FI_LOG_EP_DATA,
			"fi_ibv_rdm_cm_progress error\n");
			abort();
		}
		usleep(FI_IBV_RDM_CM_THREAD_TIMEOUT);
	}
	return NULL;
}

static int fi_ibv_rdm_tagged_ep_close(fid_t fid)
{
	int ret = 0;
	struct fi_ibv_rdm_ep *ep;
	void *status;
	ep = container_of(fid, struct fi_ibv_rdm_ep, ep_fid.fid);

	ep->is_closing = 1;
	// assert(ep->pend_send == 0);
	// assert(ep->pend_recv == 0); //TODO
	_fi_ibv_rdm_tagged_cm_progress_running = 0;
	pthread_join(ep->cm_progress_thread, &status);
	pthread_mutex_destroy(&ep->cm_lock);

	struct fi_ibv_rdm_tagged_conn *conn, *tmp;

	HASH_ITER(hh, fi_ibv_rdm_tagged_conn_hash, conn, tmp) {
		HASH_DEL(fi_ibv_rdm_tagged_conn_hash, conn);
		switch (conn->state) {
		case FI_VERBS_CONN_ALLOCATED:
			free(conn);
			break;
		case FI_VERBS_CONN_REMOTE_DISCONNECT:
			fi_ibv_rdm_start_disconnection(ep, conn);
			fi_ibv_rdm_tagged_conn_cleanup(ep, conn);
			break;
		case FI_VERBS_CONN_STARTED:
			while (conn->state != FI_VERBS_CONN_ESTABLISHED &&
			       conn->state != FI_VERBS_CONN_REJECTED) {
				ret = fi_ibv_rdm_tagged_cm_progress(ep);
				if (ret) {
					VERBS_INFO(FI_LOG_AV,
						"cm progress failed\n");
					return ret;
				}
			}
			break;
		case FI_VERBS_CONN_ESTABLISHED:
			fi_ibv_rdm_start_disconnection(ep, conn);
			break;
		default:
			break;
		}
	}
	while (ep->num_active_conns) {
		ret = fi_ibv_rdm_tagged_cm_progress(ep);
		if (ret) {
			VERBS_INFO(FI_LOG_AV, "cm progress failed\n");
			return ret;
		}
	}

	assert(0 == HASH_COUNT(fi_ibv_rdm_tagged_conn_hash) &&
	       NULL == fi_ibv_rdm_tagged_conn_hash);

	VERBS_INFO(FI_LOG_AV, "DISCONNECT complete\n");
	rdma_destroy_id(ep->cm_listener);
	ibv_destroy_cq(ep->scq);
	ibv_destroy_cq(ep->rcq);

	fi_ibv_mem_pool_fini(&fi_ibv_rdm_tagged_request_pool);
	fi_ibv_mem_pool_fini(&fi_ibv_rdm_tagged_postponed_pool);
	fi_ibv_mem_pool_fini(&fi_ibv_rdm_tagged_unexp_buffers_pool);

	free(ep);

	return 0;
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
	.close = fi_ibv_rdm_tagged_ep_close,
	.bind = fi_ibv_rdm_tagged_ep_bind,
	.control = fi_ibv_rdm_tagged_control,
	.ops_open = fi_no_ops_open,
};

static inline int fi_ibv_rdm_tagged_test_addr(const char *devname,
					      struct sockaddr_in *addr)
{
	struct rdma_cm_id *test_id;
	int test = 0;
	if (rdma_create_id(NULL, &test_id, NULL, RDMA_PS_TCP)) {
		VERBS_INFO(FI_LOG_AV, "Failed to create test rdma cm id: %s\n",
			     strerror(errno));
		return -1;
	}

	if (rdma_bind_addr(test_id, (struct sockaddr *)addr)) {
		VERBS_INFO(FI_LOG_AV,
			"Failed to bind cm listener to  addr : %s\n",
			strerror(errno));
		rdma_destroy_id(test_id);
		return 0;
	}

	VERBS_INFO(FI_LOG_AV, "device name: %s %s\n",
		test_id->verbs->device->name, devname);

	if (!strcmp(test_id->verbs->device->name, devname)) {
		test = 1;
	}

	rdma_destroy_id(test_id);
	return test;
}

/* find the IPoIB address of the device opened in the fi_domain call. The name
 * of this device is _domain->verbs->device->name. The logic of the function is:
 * iterate through all the available network interfaces, find those having "ib"
 * in the name, then try to test the IB device that correspond to each address.
 * If the name is the desired one then we're done.
 */
static inline int fi_ibv_rdm_tagged_find_ipoib_addr(struct fi_ibv_rdm_ep *ep,
						    const char *devname)
{
	struct ifaddrs *addrs, *tmp;
	getifaddrs(&addrs);
	tmp = addrs;
	int found = 0;

	while (tmp) {
		if (tmp->ifa_addr && tmp->ifa_addr->sa_family == AF_INET) {
			struct sockaddr_in *paddr =
			    (struct sockaddr_in *) tmp->ifa_addr;
			if (!strncmp(tmp->ifa_name, "ib", 2)) {
				int ret;
				ret = fi_ibv_rdm_tagged_test_addr(devname, paddr);
				if (ret == 1) {
					memcpy(&ep->my_ipoib_addr, paddr,
						sizeof(struct sockaddr_in));
					found = 1;
					break;
				} else if (ret < 0) {
					return -1;
				}
			}
		}

		tmp = tmp->ifa_next;
	}

	if (found) {
		if (ep->my_ipoib_addr.sin_family == AF_INET) {
			inet_ntop(ep->my_ipoib_addr.sin_family,
				  &ep->my_ipoib_addr.sin_addr.s_addr,
				  ep->my_ipoib_addr_str, INET_ADDRSTRLEN);
		} else {
			assert(0);
		}
	}
	return !found;
}

int fi_ibv_open_rdm_ep(struct fid_domain *domain, struct fi_info *info,
			struct fid_ep **ep, void *context)
{
	struct fi_ibv_domain *_domain;
	int ret = 0;

	_domain = container_of(domain, struct fi_ibv_domain, domain_fid);
	if (strncmp(_domain->verbs->device->name, info->domain_attr->name,
                strlen(_domain->verbs->device->name))) {
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
	_ep->ep_fid.cm = &fi_ibv_rdm_tagged_ep_cm_ops;

	if (fi_ibv_rdm_tagged_find_ipoib_addr
	    (_ep, _domain->verbs->device->name)) {
		ret = -FI_ENODEV;
		VERBS_INFO(FI_LOG_EP_CTRL,
			   "Failed to find correct IPoIB address\n");
		goto err;
	}

	VERBS_INFO(FI_LOG_EP_CTRL, "My IPoIB: %s\n",
		_ep->my_ipoib_addr_str);
	_ep->cm_listener_ec = rdma_create_event_channel();
	if (!_ep->cm_listener_ec) {
		VERBS_INFO(FI_LOG_EP_CTRL,
			"Failed to create listener event channel: %s\n",
			strerror(errno));
		ret = -FI_EOTHER;
		goto err;
	}

	int fd = _ep->cm_listener_ec->fd;
	int flags = fcntl(fd, F_GETFL, 0);
	ret = fcntl(fd, F_SETFL, flags | O_NONBLOCK);
	if (ret == -1) {
		VERBS_INFO_ERRNO(FI_LOG_EP_CTRL, "fcntl", errno);
		ret = -FI_EOTHER;
		goto err;
	}

	if (rdma_create_id(_ep->cm_listener_ec,
			   &_ep->cm_listener, NULL, RDMA_PS_TCP)) {
		VERBS_INFO(FI_LOG_EP_CTRL, "Failed to create cm listener: %s\n",
			     strerror(errno));
		ret = -FI_EOTHER;
		goto err;
	}

	if (rdma_bind_addr(_ep->cm_listener,
			   (struct sockaddr *)&_ep->my_ipoib_addr)) {
		VERBS_INFO(FI_LOG_EP_CTRL,
			"Failed to bind cm listener to my IPoIB addr %s: %s\n",
			_ep->my_ipoib_addr_str, strerror(errno));
		ret = -FI_EOTHER;
		goto err;

	}
	if (rdma_listen(_ep->cm_listener, 1024)) {
		VERBS_INFO(FI_LOG_EP_CTRL, "rdma_listen failed: %s\n",
			strerror(errno));
		ret = -FI_EOTHER;
		goto err;
	}

	_ep->cm_listener_port = ntohs(rdma_get_src_port(_ep->cm_listener));
	VERBS_INFO(FI_LOG_EP_CTRL, "listener port: %d\n", _ep->cm_listener_port);

	size_t s1 = sizeof(_ep->my_ipoib_addr.sin_addr.s_addr);
	size_t s2 = sizeof(_ep->cm_listener_port);
	assert(FI_IBV_RDM_DFLT_ADDRLEN == s1 + s2);

	memcpy(_ep->my_rdm_addr, &_ep->my_ipoib_addr.sin_addr.s_addr, s1);
	memcpy(_ep->my_rdm_addr + s1, &_ep->cm_listener_port, s2);

	VERBS_INFO(FI_LOG_EP_CTRL,
		"My ep_addr: " FI_IBV_RDM_ADDR_STR_FORMAT "\n",
		FI_IBV_RDM_ADDR_STR(_ep->my_rdm_addr));

	_ep->n_buffs = FI_IBV_RDM_TAGGED_DFLT_BUFFER_NUM;
	const int header_size = sizeof(struct fi_ibv_rdm_tagged_header);
	_ep->buff_len = FI_IBV_RDM_TAGGED_DFLT_BUFFER_SIZE;
	_ep->rndv_threshold = _ep->buff_len -
	    FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE - header_size;

	_ep->rq_wr_depth = FI_IBV_RDM_TAGGED_DFLT_RQ_SIZE;

	/*
	 * max number of WRs in SQ is n_buffer for send and
	 * the same amount for buffer releasing from recv
	 */
	_ep->sq_wr_depth = 2 * (_ep->n_buffs + 1);

	_ep->total_outgoing_send = 0;
	_ep->pend_send = 0;
	_ep->pend_recv = 0;
	_ep->recv_preposted_threshold = MAX(0.2 * _ep->rq_wr_depth, 5);
	VERBS_INFO(FI_LOG_EP_CTRL, "recv preposted threshold: %d\n",
		   _ep->recv_preposted_threshold);

	fi_ibv_mem_pool_init(&fi_ibv_rdm_tagged_request_pool,
			     100, 100,
			     sizeof(struct fi_ibv_rdm_tagged_request));

	fi_ibv_mem_pool_init(&fi_ibv_rdm_tagged_postponed_pool,
			     100, 100,
			     sizeof(struct fi_ibv_rdm_tagged_postponed_entry));

	fi_ibv_mem_pool_init(&fi_ibv_rdm_tagged_unexp_buffers_pool,
			     100, 100, _ep->buff_len);

	_ep->max_inline_rc =
	    fi_ibv_rdm_tagged_find_max_inline_size(_ep->domain->pd,
						   _ep->domain->verbs);
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
out:
	return ret;
err:
	free(_ep);
	goto out;
}
