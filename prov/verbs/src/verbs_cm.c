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

#include "config.h"

#include "fi_verbs.h"


static int fi_ibv_copy_addr(void *dst_addr, size_t *dst_addrlen, void *src_addr)
{
	size_t src_addrlen = fi_ibv_sockaddr_len(src_addr);

	if (*dst_addrlen == 0) {
		*dst_addrlen = src_addrlen;
		return -FI_ETOOSMALL;
	}

	if (*dst_addrlen < src_addrlen) {
		memcpy(dst_addr, src_addr, *dst_addrlen);
	} else {
		memcpy(dst_addr, src_addr, src_addrlen);
	}
	*dst_addrlen = src_addrlen;
	return 0;
}

static int fi_ibv_msg_ep_setname(fid_t ep_fid, void *addr, size_t addrlen)
{
	struct fi_ibv_msg_ep *ep;
	void *save_addr;
	struct rdma_cm_id *id;
	int ret;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	if (addrlen != ep->info->src_addrlen) {
		VERBS_INFO(FI_LOG_EP_CTRL,"addrlen expected: %zu, got: %zu.\n",
			   ep->info->src_addrlen, addrlen);
		return -FI_EINVAL;
	}

	save_addr = ep->info->src_addr;

	ep->info->src_addr = malloc(ep->info->src_addrlen);
	if (!ep->info->src_addr) {
		ret = -FI_ENOMEM;
		goto err1;
	}

	memcpy(ep->info->src_addr, addr, ep->info->src_addrlen);

	ret = fi_ibv_create_ep(NULL, NULL, 0, ep->info, NULL, &id);
	if (ret)
		goto err2;

	if (ep->id)
		rdma_destroy_ep(ep->id);

	ep->id = id;
	free(save_addr);

	return 0;
err2:
	free(ep->info->src_addr);
err1:
	ep->info->src_addr = save_addr;
	return ret;
}

static int fi_ibv_msg_ep_getname(fid_t ep, void *addr, size_t *addrlen)
{
	struct fi_ibv_msg_ep *_ep;
	struct sockaddr *sa;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	sa = rdma_get_local_addr(_ep->id);
	return fi_ibv_copy_addr(addr, addrlen, sa);
}

static int fi_ibv_msg_ep_getpeer(struct fid_ep *ep, void *addr, size_t *addrlen)
{
	struct fi_ibv_msg_ep *_ep;
	struct sockaddr *sa;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	sa = rdma_get_peer_addr(_ep->id);
	return fi_ibv_copy_addr(addr, addrlen, sa);
}

static int
fi_ibv_msg_ep_connect(struct fid_ep *ep, const void *addr,
		   const void *param, size_t paramlen)
{
	struct fi_ibv_msg_ep *_ep;
	struct rdma_conn_param conn_param;
	struct sockaddr *src_addr, *dst_addr;
	int ret;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	if (!_ep->id->qp) {
		ret = ep->fid.ops->control(&ep->fid, FI_ENABLE, NULL);
		if (ret)
			return ret;
	}

	memset(&conn_param, 0, sizeof conn_param);
	conn_param.private_data = param;
	conn_param.private_data_len = paramlen;
	conn_param.responder_resources = RDMA_MAX_RESP_RES;
	conn_param.initiator_depth = RDMA_MAX_INIT_DEPTH;
	conn_param.flow_control = 1;
	conn_param.retry_count = 15;
	conn_param.rnr_retry_count = 7;

	if (_ep->srq_ep)
		conn_param.srq = 1;

	src_addr = rdma_get_local_addr(_ep->id);
	if (src_addr) {
		VERBS_INFO(FI_LOG_CORE, "src_addr: %s:%d\n",
			   inet_ntoa(((struct sockaddr_in *)src_addr)->sin_addr),
			   ntohs(((struct sockaddr_in *)src_addr)->sin_port));
	}

	dst_addr = rdma_get_peer_addr(_ep->id);
	if (dst_addr) {
		VERBS_INFO(FI_LOG_CORE, "dst_addr: %s:%d\n",
			   inet_ntoa(((struct sockaddr_in *)dst_addr)->sin_addr),
			   ntohs(((struct sockaddr_in *)dst_addr)->sin_port));
	}

	return rdma_connect(_ep->id, &conn_param) ? -errno : 0;
}

static int
fi_ibv_msg_ep_accept(struct fid_ep *ep, const void *param, size_t paramlen)
{
	struct fi_ibv_msg_ep *_ep;
	struct rdma_conn_param conn_param;
	struct fi_ibv_connreq *connreq;
	int ret;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	if (!_ep->id->qp) {
		ret = ep->fid.ops->control(&ep->fid, FI_ENABLE, NULL);
		if (ret)
			return ret;
	}

	memset(&conn_param, 0, sizeof conn_param);
	conn_param.private_data = param;
	conn_param.private_data_len = paramlen;
	conn_param.responder_resources = RDMA_MAX_RESP_RES;
	conn_param.initiator_depth = RDMA_MAX_INIT_DEPTH;
	conn_param.flow_control = 1;
	conn_param.rnr_retry_count = 7;

	if (_ep->srq_ep)
		conn_param.srq = 1;

	ret = rdma_accept(_ep->id, &conn_param);
	if (ret)
		return -errno;

	connreq = container_of(_ep->info->handle, struct fi_ibv_connreq, handle);
	free(connreq);

	return 0;
}

static int
fi_ibv_msg_ep_reject(struct fid_pep *pep, fid_t handle,
		  const void *param, size_t paramlen)
{
	struct fi_ibv_connreq *connreq;
	int ret;

	connreq = container_of(handle, struct fi_ibv_connreq, handle);
	ret = rdma_reject(connreq->id, param, (uint8_t) paramlen) ? -errno : 0;
	free(connreq);
	return ret;
}

static int fi_ibv_msg_ep_shutdown(struct fid_ep *ep, uint64_t flags)
{
	struct fi_ibv_msg_ep *_ep;
	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	return rdma_disconnect(_ep->id) ? -errno : 0;
}

struct fi_ops_cm fi_ibv_msg_ep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_ibv_msg_ep_setname,
	.getname = fi_ibv_msg_ep_getname,
	.getpeer = fi_ibv_msg_ep_getpeer,
	.connect = fi_ibv_msg_ep_connect,
	.listen = fi_no_listen,
	.accept = fi_ibv_msg_ep_accept,
	.reject = fi_no_reject,
	.shutdown = fi_ibv_msg_ep_shutdown,
	.join = fi_no_join,
};

static int fi_ibv_pep_setname(fid_t pep_fid, void *addr, size_t addrlen)
{
	struct fi_ibv_pep *pep;
	int ret;

	pep = container_of(pep_fid, struct fi_ibv_pep, pep_fid);

	if (pep->src_addrlen && (addrlen != pep->src_addrlen)) {
		VERBS_INFO(FI_LOG_FABRIC, "addrlen expected: %zu, got: %zu.\n",
			   pep->src_addrlen, addrlen);
		return -FI_EINVAL;
	}

	/* Re-create id if already bound */
	if (pep->bound) {
		ret = rdma_destroy_id(pep->id);
		if (ret) {
			VERBS_INFO(FI_LOG_FABRIC,
				   "Unable to destroy previous rdma_cm_id\n");
			return -errno;
		}
		ret = rdma_create_id(NULL, &pep->id, &pep->pep_fid.fid, RDMA_PS_TCP);
		if (ret) {
			VERBS_INFO(FI_LOG_FABRIC,
				   "Unable to create rdma_cm_id\n");
			return -errno;
		}
	}

	ret = rdma_bind_addr(pep->id, (struct sockaddr *)addr);
	if (ret) {
		VERBS_INFO(FI_LOG_FABRIC,
			   "Unable to bind address to rdma_cm_id\n");
		return -errno;
	}

	return 0;
}

static int fi_ibv_pep_getname(fid_t pep, void *addr, size_t *addrlen)
{
	struct fi_ibv_pep *_pep;
	struct sockaddr *sa;

	_pep = container_of(pep, struct fi_ibv_pep, pep_fid);
	sa = rdma_get_local_addr(_pep->id);
	return fi_ibv_copy_addr(addr, addrlen, sa);
}

static int fi_ibv_pep_listen(struct fid_pep *pep_fid)
{
	struct fi_ibv_pep *pep;
	struct sockaddr *addr;

	pep = container_of(pep_fid, struct fi_ibv_pep, pep_fid);

	addr = rdma_get_local_addr(pep->id);
	if (addr) {
		VERBS_INFO(FI_LOG_CORE, "Listening on %s:%d\n",
			   inet_ntoa(((struct sockaddr_in *)addr)->sin_addr),
			   ntohs(((struct sockaddr_in *)addr)->sin_port));
	}

	return rdma_listen(pep->id, pep->backlog) ? -errno : 0;
}

static struct fi_ops_cm fi_ibv_pep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_ibv_pep_setname,
	.getname = fi_ibv_pep_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_ibv_pep_listen,
	.accept = fi_no_accept,
	.reject = fi_ibv_msg_ep_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

struct fi_ops_cm *fi_ibv_pep_ops_cm(struct fi_ibv_pep *pep)
{
	return &fi_ibv_pep_cm_ops;
}
