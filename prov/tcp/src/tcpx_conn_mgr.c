/*
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *	   Redistribution and use in source and binary forms, with or
 *	   without modification, are permitted provided that the following
 *	   conditions are met:
 *
 *		- Redistributions of source code must retain the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer.
 *
 *		- Redistributions in binary form must reproduce the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer in the documentation and/or other materials
 *		  provided with the distribution.
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

#include <rdma/fi_errno.h>

#include <ofi_prov.h>
#include "tcpx.h"
#include <poll.h>
#include <sys/types.h>
#include <ofi_util.h>


static int read_cm_data(SOCKET fd, struct tcpx_cm_context *cm_ctx,
			struct ofi_ctrl_hdr *hdr)
{
	cm_ctx->cm_data_sz = ntohs(hdr->seg_size);
	if (cm_ctx->cm_data_sz) {
		size_t data_sz = MIN(cm_ctx->cm_data_sz,
				     TCPX_MAX_CM_DATA_SIZE);
		ssize_t ret = ofi_recv_socket(fd, cm_ctx->cm_data,
					      data_sz, MSG_WAITALL);
		if ((size_t) ret != data_sz)
			return -FI_EIO;
		cm_ctx->cm_data_sz = data_sz;

		if (OFI_UNLIKELY(cm_ctx->cm_data_sz >
					TCPX_MAX_CM_DATA_SIZE)) {
			ofi_discard_socket(fd, cm_ctx->cm_data_sz -
					   TCPX_MAX_CM_DATA_SIZE);
		}
	}
	return FI_SUCCESS;
}

static int rx_cm_data(SOCKET fd, struct ofi_ctrl_hdr *hdr,
		      int type, struct tcpx_cm_context *cm_ctx)
{
	ssize_t ret;

	ret = ofi_recv_socket(fd, hdr,
			      sizeof(*hdr), MSG_WAITALL);
	if (ret != sizeof(*hdr))
		return -FI_EIO;

	if (hdr->version != TCPX_CTRL_HDR_VERSION)
		return -FI_ENOPROTOOPT;

	ret = read_cm_data(fd, cm_ctx, hdr);
	if (hdr->type != type) {
		ret = -FI_ECONNREFUSED;
	}
	return ret;
}

static int tx_cm_data(SOCKET fd, uint8_t type, struct tcpx_cm_context *cm_ctx)
{
	struct ofi_ctrl_hdr hdr;
	ssize_t ret;

	memset(&hdr, 0, sizeof(hdr));
	hdr.version = TCPX_CTRL_HDR_VERSION;
	hdr.type = type;
	hdr.seg_size = htons((uint16_t) cm_ctx->cm_data_sz);
	hdr.conn_data = 1; /* For testing endianess mismatch at peer */

	ret = ofi_send_socket(fd, &hdr, sizeof(hdr), MSG_NOSIGNAL);
	if (ret != sizeof(hdr))
		return -FI_EIO;

	if (cm_ctx->cm_data_sz) {
		ret = ofi_send_socket(fd, cm_ctx->cm_data,
				      cm_ctx->cm_data_sz, MSG_NOSIGNAL);
		if ((size_t) ret != cm_ctx->cm_data_sz)
			return -FI_EIO;
	}
	return FI_SUCCESS;
}

static int tcpx_ep_msg_xfer_enable(struct tcpx_ep *ep)
{
	int ret;

	fastlock_acquire(&ep->lock);
	if (ep->cm_state != TCPX_EP_CONNECTING) {
		fastlock_release(&ep->lock);
		return -FI_EINVAL;
	}
	ep->progress_func = tcpx_ep_progress;
	ret = fi_fd_nonblock(ep->conn_fd);
	if (ret) {
		fastlock_release(&ep->lock);
		return ret;
	}
	ep->cm_state = TCPX_EP_CONNECTED;
	fastlock_release(&ep->lock);

	return tcpx_cq_wait_ep_add(ep);
}

static int proc_conn_resp(struct tcpx_cm_context *cm_ctx,
			  struct tcpx_ep *ep)
{
	struct ofi_ctrl_hdr conn_resp;
	struct fi_eq_cm_entry *cm_entry;
	ssize_t len;
	int ret = FI_SUCCESS;

	ret = rx_cm_data(ep->conn_fd, &conn_resp, ofi_ctrl_connresp, cm_ctx);
	if (ret)
		return ret;

	cm_entry = calloc(1, sizeof(*cm_entry) + cm_ctx->cm_data_sz);
	if (!cm_entry)
		return -FI_ENOMEM;

	cm_entry->fid = cm_ctx->fid;
	memcpy(cm_entry->data, cm_ctx->cm_data, cm_ctx->cm_data_sz);

	ep->hdr_bswap = (conn_resp.conn_data == 1)?
		tcpx_hdr_none:tcpx_hdr_bswap;

	ret = tcpx_ep_msg_xfer_enable(ep);
	if (ret)
		goto err;

	len = fi_eq_write(&ep->util_ep.eq->eq_fid, FI_CONNECTED, cm_entry,
			  sizeof(*cm_entry) + cm_ctx->cm_data_sz, 0);
	if (len < 0) {
		ret = (int) len;
		goto err;
	}
err:
	free(cm_entry);
	return ret;
}

int tcpx_eq_wait_try_func(void *arg)
{
	return FI_SUCCESS;
}

static void client_recv_connresp(struct util_wait *wait,
				 struct tcpx_cm_context *cm_ctx)
{
	struct fi_eq_err_entry err_entry = { 0 };
	struct tcpx_ep *ep;
	ssize_t ret;

	assert(cm_ctx->fid->fclass == FI_CLASS_EP);
	ep = container_of(cm_ctx->fid, struct tcpx_ep, util_ep.ep_fid.fid);

	ret = ofi_wait_fd_del(wait, ep->conn_fd);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"Could not remove fd from wait\n");
		goto err;
	}

	ret = proc_conn_resp(cm_ctx, ep);
	if (ret)
		goto err;

	FI_DBG(&tcpx_prov, FI_LOG_EP_CTRL, "Received Accept from server\n");
	free(cm_ctx);
	return;
err:
	err_entry.fid = cm_ctx->fid;
	err_entry.context = cm_ctx->fid->context;
	err_entry.err = -ret;
	if (cm_ctx->cm_data_sz) {
		err_entry.err_data = calloc(1, cm_ctx->cm_data_sz);
		if (OFI_LIKELY(err_entry.err_data != NULL)) {
			memcpy(err_entry.err_data, cm_ctx->cm_data,
			       cm_ctx->cm_data_sz);
			err_entry.err_data_size = cm_ctx->cm_data_sz;
		}
	}
	FI_DBG(&tcpx_prov, FI_LOG_EP_CTRL,
	       "fi_eq_write the conn refused %"PRId64"\n", ret);
	free(cm_ctx);
	/* `err_entry.err_data` must live until it is passed to user */
	ret = fi_eq_write(&ep->util_ep.eq->eq_fid, FI_NOTIFY,
			  &err_entry, sizeof(err_entry), UTIL_FLAG_ERROR);
	if (OFI_UNLIKELY(ret < 0)) {
		free(err_entry.err_data);
	}
}

static void server_send_cm_accept(struct util_wait *wait,
				  struct tcpx_cm_context *cm_ctx)
{
	struct fi_eq_cm_entry cm_entry = {0};
	struct fi_eq_err_entry err_entry;
	struct tcpx_ep *ep;
	int ret;

	assert(cm_ctx->fid->fclass == FI_CLASS_EP);
	ep = container_of(cm_ctx->fid, struct tcpx_ep, util_ep.ep_fid.fid);

	ret = tx_cm_data(ep->conn_fd, ofi_ctrl_connresp, cm_ctx);
	if (ret)
		goto err;

	cm_entry.fid =  cm_ctx->fid;
	ret = (int) fi_eq_write(&ep->util_ep.eq->eq_fid, FI_CONNECTED,
				&cm_entry, sizeof(cm_entry), 0);
	if (ret < 0) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL, "Error writing to EQ\n");
	}

	ret = ofi_wait_fd_del(wait, ep->conn_fd);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"Could not remove fd from wait\n");
		goto err;
	}

	ret = tcpx_ep_msg_xfer_enable(ep);
	if (ret)
		goto err;

	FI_DBG(&tcpx_prov, FI_LOG_EP_CTRL, "Connection Accept Successful\n");
	free(cm_ctx);
	return;
err:
	memset(&err_entry, 0, sizeof err_entry);
	err_entry.fid = cm_ctx->fid;
	err_entry.context = cm_ctx->fid->context;
	err_entry.err = -ret;

	free(cm_ctx);
	fi_eq_write(&ep->util_ep.eq->eq_fid, FI_NOTIFY,
		    &err_entry, sizeof(err_entry), UTIL_FLAG_ERROR);
}

static void server_recv_connreq(struct util_wait *wait,
				struct tcpx_cm_context *cm_ctx)
{
	struct tcpx_conn_handle *handle;
	struct fi_eq_cm_entry *cm_entry;
	struct ofi_ctrl_hdr conn_req;
	socklen_t len;
	int ret;

	assert(cm_ctx->fid->fclass == FI_CLASS_CONNREQ);

	handle  = container_of(cm_ctx->fid,
			       struct tcpx_conn_handle,
			       handle);

	ret = rx_cm_data(handle->conn_fd, &conn_req, ofi_ctrl_connreq, cm_ctx);
	if (ret)
		goto err1;

	cm_entry = calloc(1, sizeof(*cm_entry) + cm_ctx->cm_data_sz);
	if (!cm_entry)
		goto err1;

	cm_entry->fid = &handle->pep->util_pep.pep_fid.fid;
	cm_entry->info = fi_dupinfo(handle->pep->info);
	if (!cm_entry->info)
		goto err2;

	len = cm_entry->info->dest_addrlen = handle->pep->info->src_addrlen;
	cm_entry->info->dest_addr = malloc(len);
	if (!cm_entry->info->dest_addr)
		goto err3;

	ret = ofi_getpeername(handle->conn_fd, cm_entry->info->dest_addr, &len);
	if (ret)
		goto err3;

	handle->endian_match = (conn_req.conn_data == 1);
	cm_entry->info->handle = &handle->handle;
	memcpy(cm_entry->data, cm_ctx->cm_data, cm_ctx->cm_data_sz);

	ret = (int) fi_eq_write(&handle->pep->util_pep.eq->eq_fid, FI_CONNREQ, cm_entry,
				sizeof(*cm_entry) + cm_ctx->cm_data_sz, 0);
	if (ret < 0) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL, "Error writing to EQ\n");
		goto err3;
	}
	ret = ofi_wait_fd_del(wait, handle->conn_fd);
	if (ret)
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"fd deletion from ofi_wait failed\n");
	free(cm_entry);
	free(cm_ctx);
	return;
err3:
	fi_freeinfo(cm_entry->info);
err2:
	free(cm_entry);
err1:
	ofi_wait_fd_del(wait, handle->conn_fd);
	ofi_close_socket(handle->conn_fd);
	free(cm_ctx);
	free(handle);
}

static void client_send_connreq(struct util_wait *wait,
				struct tcpx_cm_context *cm_ctx)
{
	struct tcpx_ep *ep;
	struct fi_eq_err_entry err_entry;
	socklen_t len;
	int status, ret = FI_SUCCESS;

	FI_DBG(&tcpx_prov, FI_LOG_EP_CTRL, "client send connreq\n");
	assert(cm_ctx->fid->fclass == FI_CLASS_EP);

	ep = container_of(cm_ctx->fid, struct tcpx_ep, util_ep.ep_fid.fid);

	len = sizeof(status);
	ret = getsockopt(ep->conn_fd, SOL_SOCKET, SO_ERROR, (char *) &status, &len);
	if (ret < 0 || status) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL, "connection failure\n");
		ret = (ret < 0)? -ofi_sockerr() : status;
		goto err;
	}

	ret = tx_cm_data(ep->conn_fd, ofi_ctrl_connreq, cm_ctx);
	if (ret)
		goto err;

	ret = ofi_wait_fd_del(wait, ep->conn_fd);
	if (ret)
		goto err;

	cm_ctx->type = CLIENT_RECV_CONNRESP;
	ret = ofi_wait_fd_add(wait, ep->conn_fd, FI_EPOLL_IN,
			      tcpx_eq_wait_try_func, NULL, cm_ctx);
	if (ret)
		goto err;

	wait->signal(wait);
	return;
err:
	memset(&err_entry, 0, sizeof err_entry);
	err_entry.fid = cm_ctx->fid;
	err_entry.context = cm_ctx->fid->context;
	err_entry.err = -ret;

	free(cm_ctx);
	fi_eq_write(&ep->util_ep.eq->eq_fid, FI_NOTIFY,
		    &err_entry, sizeof(err_entry), UTIL_FLAG_ERROR);
}

static void server_sock_accept(struct util_wait *wait,
			       struct tcpx_cm_context *cm_ctx)
{
	struct tcpx_conn_handle *handle;
	struct tcpx_cm_context *rx_req_cm_ctx;
	struct tcpx_pep *pep;
	SOCKET sock;
	int ret;

	FI_DBG(&tcpx_prov, FI_LOG_EP_CTRL, "Received Connreq\n");
	assert(cm_ctx->fid->fclass == FI_CLASS_PEP);
	pep = container_of(cm_ctx->fid, struct tcpx_pep,
			   util_pep.pep_fid.fid);

	sock = accept(pep->sock, NULL, 0);
	if (sock < 0) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"accept error: %d\n", ofi_sockerr());
		return;
	}

	handle = calloc(1, sizeof(*handle));
	if (!handle) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"cannot allocate memory \n");
		goto err1;
	}

	rx_req_cm_ctx = calloc(1, sizeof(*rx_req_cm_ctx));
	if (!rx_req_cm_ctx)
		goto err2;

	handle->conn_fd = sock;
	handle->handle.fclass = FI_CLASS_CONNREQ;
	handle->pep = pep;
	rx_req_cm_ctx->fid = &handle->handle;
	rx_req_cm_ctx->type = SERVER_RECV_CONNREQ;

	ret = ofi_wait_fd_add(wait, sock, FI_EPOLL_IN,
			      tcpx_eq_wait_try_func,
			      NULL, (void *) rx_req_cm_ctx);
	if (ret)
		goto err3;
	wait->signal(wait);
	return;
err3:
	free(rx_req_cm_ctx);
err2:
	free(handle);
err1:
	ofi_close_socket(sock);
}

static void process_cm_ctx(struct util_wait *wait,
			   struct tcpx_cm_context *cm_ctx)
{
	switch (cm_ctx->type) {
	case SERVER_SOCK_ACCEPT:
		server_sock_accept(wait,cm_ctx);
		break;
	case CLIENT_SEND_CONNREQ:
		client_send_connreq(wait, cm_ctx);
		break;
	case SERVER_RECV_CONNREQ:
		server_recv_connreq(wait, cm_ctx);
		break;
	case SERVER_SEND_CM_ACCEPT:
		server_send_cm_accept(wait, cm_ctx);
		break;
	case CLIENT_RECV_CONNRESP:
		client_recv_connresp(wait, cm_ctx);
		break;
	default:
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"should never end up here\n");
	}
}

void tcpx_conn_mgr_run(struct util_eq *eq)
{
	struct util_wait_fd *wait_fd;
	struct tcpx_eq *tcpx_eq;
	void *wait_contexts[MAX_EPOLL_EVENTS];
	int num_fds = 0, i;

	assert(eq->wait != NULL);

	wait_fd = container_of(eq->wait, struct util_wait_fd,
			       util_wait);

	tcpx_eq = container_of(eq, struct tcpx_eq, util_eq);
	fastlock_acquire(&tcpx_eq->close_lock);
	num_fds = fi_epoll_wait(wait_fd->epoll_fd, wait_contexts,
				MAX_EPOLL_EVENTS, 0);
	if (num_fds < 0) {
		fastlock_release(&tcpx_eq->close_lock);
		return;
	}

	for ( i = 0; i < num_fds; i++) {

		/* skip wake up signals */
		if (&wait_fd->util_wait.wait_fid.fid == wait_contexts[i])
			continue;

		process_cm_ctx(eq->wait,
			       (struct tcpx_cm_context *)
			       wait_contexts[i]);
	}
	fastlock_release(&tcpx_eq->close_lock);
}
