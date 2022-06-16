/*
 * Copyright (c) 2017-2022 Intel Corporation. All rights reserved.
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

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/tcp.h>

#include <ofi_prov.h>
#include <ofi_iov.h>
#include "tcp2.h"
#include <errno.h>

extern struct fi_ops_rma tcp2_rma_ops;
extern struct fi_ops_msg tcp2_msg_ops;
extern struct fi_ops_tagged tcp2_tagged_ops;


void tcp2_hdr_none(struct tcp2_base_hdr *hdr)
{
	/* no-op */
}

void tcp2_hdr_bswap(struct tcp2_base_hdr *hdr)
{
	uint64_t *cur;
	int i, cnt;

	hdr->flags = ntohs(hdr->flags);
	hdr->size = ntohll(hdr->size);

	cnt = (hdr->hdr_size - sizeof(*hdr)) >> 3;
	cur = (uint64_t *) (hdr + 1);
	for (i = 0; i < cnt; i++)
		cur[i] = ntohll(cur[i]);
}

#ifdef MSG_ZEROCOPY
void tcp2_set_zerocopy(SOCKET sock)
{
	int val = 1;

	if (tcp2_zerocopy_size == SIZE_MAX)
		return;

	(void) setsockopt(sock, SOL_SOCKET, SO_ZEROCOPY, &val, sizeof(val));
}

static void tcp2_config_bsock(struct ofi_bsock *bsock)
{
	int ret, val = 0;
	socklen_t len = sizeof(val);

	if (tcp2_zerocopy_size == SIZE_MAX)
		return;

	ret = getsockopt(bsock->sock, SOL_SOCKET, SO_ZEROCOPY, &val, &len);
	if (!ret && val) {
		bsock->zerocopy_size = tcp2_zerocopy_size;
		FI_INFO(&tcp2_prov, FI_LOG_EP_CTRL,
			"zero copy enabled for transfers > %zu\n",
			bsock->zerocopy_size);
	}
}
#else
void tcp2_set_zerocopy(SOCKET sock)
{
	OFI_UNUSED(sock);
}

#define tcp2_config_bsock(bsock)
#endif

#ifdef IP_BIND_ADDRESS_NO_PORT
static void tcp2_set_no_port(SOCKET sock)
{
	int val = 1;

	(void) setsockopt(sock, IPPROTO_IP, IP_BIND_ADDRESS_NO_PORT,
			  &val, sizeof(val));
}
#else
#define tcp2_set_no_port(sock)
#endif

int tcp2_setup_socket(SOCKET sock, struct fi_info *info)
{
	int ret, optval = 1;

	ret = setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char *) &optval,
			 sizeof(optval));
	if (ret) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,"setsockopt reuseaddr failed\n");
		return -ofi_sockerr();
	}

	/* Do not enable nodelay for bulk data traffic class, unless nodelay
	 * has explicitly been requested.
	 */
	if (tcp2_nodelay && !((tcp2_nodelay < 0) &&
	    (info->fabric_attr->api_version >= FI_VERSION(1, 9) &&
	    info->tx_attr->tclass == FI_TC_BULK_DATA))) {

		ret = setsockopt(sock, IPPROTO_TCP, TCP_NODELAY,
				 (char *) &optval, sizeof(optval));
		if (ret) {
			FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
				"setsockopt nodelay failed\n");
			return -ofi_sockerr();
		}
	}

	ret = fi_fd_nonblock(sock);
	if (ret) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"failed to set socket to nonblocking\n");
		return ret;
	}

	return 0;
}

static int tcp2_ep_connect(struct fid_ep *ep_fid, const void *addr,
			   const void *param, size_t paramlen)
{
	struct tcp2_progress *progress;
	struct tcp2_ep *ep;
	int ret;

	FI_DBG(&tcp2_prov, FI_LOG_EP_CTRL, "connecting endpoint\n");
	ep = container_of(ep_fid, struct tcp2_ep, util_ep.ep_fid);
	if (!addr || (ep->bsock.sock == INVALID_SOCKET) ||
	    (paramlen > TCP2_MAX_CM_DATA_SIZE) || (ep->state != TCP2_IDLE))
		return -FI_EINVAL;

	ep->cm_msg->hdr.version = TCP2_CTRL_HDR_VERSION;
	ep->cm_msg->hdr.type = ofi_ctrl_connreq;
	ep->cm_msg->hdr.conn_data = 1; /* tests endianess mismatch at peer */
	if (paramlen) {
		memcpy(ep->cm_msg->data, param, paramlen);
		ep->cm_msg->hdr.seg_size = htons((uint16_t) paramlen);
	}

	ep->state = TCP2_CONNECTING;
	ret = connect(ep->bsock.sock, (struct sockaddr *) addr,
		      (socklen_t) ofi_sizeofaddr(addr));
	if (ret && !OFI_SOCK_TRY_CONN_AGAIN(ofi_sockerr())) {
		ep->state = TCP2_IDLE;
		ret = -ofi_sockerr();
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"connect failure %d(%s)\n", -ret, fi_strerror(-ret));
		return ret;
	}

	ep->pollout_set = true;
	progress = tcp2_ep2_progress(ep);
	ofi_mutex_lock(&progress->lock);
	ret = tcp2_monitor_sock(progress, ep->bsock.sock, POLLOUT,
				&ep->util_ep.ep_fid.fid);
	ofi_mutex_unlock(&progress->lock);
	if (ret)
		goto disable;

	return 0;

disable:
	ofi_mutex_lock(&progress->lock);
	ofi_mutex_lock(&ep->lock);
	tcp2_ep_disable(ep, -ret, NULL, 0);
	ofi_mutex_unlock(&ep->lock);
	ofi_mutex_unlock(&progress->lock);
	return ret;
}

static int
tcp2_ep_accept(struct fid_ep *ep_fid, const void *param, size_t paramlen)
{
	struct tcp2_progress *progress;
	struct tcp2_ep *ep;
	struct tcp2_conn_handle *conn;
	struct fi_eq_cm_entry cm_entry;
	int ret;

	FI_DBG(&tcp2_prov, FI_LOG_EP_CTRL, "accepting endpoint connection\n");
	ep = container_of(ep_fid, struct tcp2_ep, util_ep.ep_fid);
	conn = ep->conn;
	if (ep->bsock.sock == INVALID_SOCKET || ep->state != TCP2_ACCEPTING ||
	    !conn || (conn->fid.fclass != FI_CLASS_CONNREQ) ||
	    (paramlen > TCP2_MAX_CM_DATA_SIZE))
		return -FI_EINVAL;

	ep->conn = NULL;

	assert(ep->cm_msg);
	ep->cm_msg->hdr.version = TCP2_CTRL_HDR_VERSION;
	ep->cm_msg->hdr.type = ofi_ctrl_connresp;
	ep->cm_msg->hdr.conn_data = 1; /* tests endianess mismatch at peer */
	if (paramlen) {
		memcpy(ep->cm_msg->data, param, paramlen);
		ep->cm_msg->hdr.seg_size = htons((uint16_t) paramlen);
	}

	ret = tcp2_send_cm_msg(ep);
	if (ret)
		return ret;

	free(ep->cm_msg);
	ep->cm_msg = NULL;
	ep->state = TCP2_CONNECTED;

	progress = tcp2_ep2_progress(ep);
	ofi_mutex_lock(&progress->lock);
	ofi_mutex_lock(&ep->lock);
	ret = tcp2_monitor_sock(progress, ep->bsock.sock, POLLIN,
				&ep->util_ep.ep_fid.fid);
	if (!ret && tcp2_active_wait(ep)) {
		dlist_insert_tail(&ep->progress_entry,
				  &progress->active_wait_list);
		tcp2_signal_progress(progress);
	}
	ofi_mutex_unlock(&ep->lock);
	ofi_mutex_unlock(&progress->lock);
	if (ret)
		return ret;

	cm_entry.fid = &ep->util_ep.ep_fid.fid;
	cm_entry.info = NULL;
	ret = (int) fi_eq_write(&ep->util_ep.eq->eq_fid, FI_CONNECTED, &cm_entry,
				sizeof(cm_entry), 0);
	if (ret < 0) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL, "Error writing to EQ\n");
		return ret;
	}

	/* Only free conn on success; on failure, app may try to reject */
	free(conn);
	return 0;
}

/* must hold ep->lock */
static void tcp2_ep_flush_queue(struct slist *queue,
				struct tcp2_cq *cq)
{
	struct tcp2_xfer_entry *xfer_entry;

	while (!slist_empty(queue)) {
		xfer_entry = container_of(queue->head, struct tcp2_xfer_entry,
					  entry);
		slist_remove_head(queue);
		tcp2_cq_report_error(&cq->util_cq, xfer_entry, FI_ECANCELED);
		tcp2_free_xfer(cq, xfer_entry);
	}
}

static void tcp2_ep_flush_all_queues(struct tcp2_ep *ep)
{
	struct tcp2_cq *cq;

	assert(ofi_mutex_held(&ep->lock));
	cq = container_of(ep->util_ep.tx_cq, struct tcp2_cq, util_cq);
	if (ep->cur_tx.entry) {
		ep->hdr_bswap(&ep->cur_tx.entry->hdr.base_hdr);
		tcp2_cq_report_error(&cq->util_cq, ep->cur_tx.entry,
				     FI_ECANCELED);
		tcp2_free_xfer(cq, ep->cur_tx.entry);
		ep->cur_tx.entry = NULL;
	}

	tcp2_ep_flush_queue(&ep->tx_queue, cq);
	tcp2_ep_flush_queue(&ep->priority_queue, cq);
	tcp2_ep_flush_queue(&ep->rma_read_queue, cq);
	tcp2_ep_flush_queue(&ep->need_ack_queue, cq);
	tcp2_ep_flush_queue(&ep->async_queue, cq);

	cq = container_of(ep->util_ep.rx_cq, struct tcp2_cq, util_cq);
	if (ep->cur_rx.entry) {
		tcp2_cq_report_error(&cq->util_cq, ep->cur_rx.entry,
				     FI_ECANCELED);
		tcp2_free_xfer(cq, ep->cur_rx.entry);
	}
	tcp2_reset_rx(ep);
	tcp2_ep_flush_queue(&ep->rx_queue, cq);
	ofi_bsock_discard(&ep->bsock);
}

void tcp2_ep_disable(struct tcp2_ep *ep, int cm_err, void* err_data,
                     size_t err_data_size)
{
	struct fi_eq_cm_entry cm_entry = {0};
	struct fi_eq_err_entry err_entry = {0};
	int ret;

	assert(ofi_mutex_held(&tcp2_ep2_progress(ep)->lock));
	assert(ofi_mutex_held(&ep->lock));
	switch (ep->state) {
	case TCP2_CONNECTING:
	case TCP2_REQ_SENT:
	case TCP2_CONNECTED:
		break;
	default:
		return;
	};

	dlist_remove_init(&ep->progress_entry);
	tcp2_halt_sock(tcp2_ep2_progress(ep), ep->bsock.sock);

	ret = ofi_shutdown(ep->bsock.sock, SHUT_RDWR);
	if (ret && ofi_sockerr() != ENOTCONN)
		FI_WARN(&tcp2_prov, FI_LOG_EP_DATA, "shutdown failed\n");

	tcp2_ep_flush_all_queues(ep);

	if (cm_err) {
		err_entry.err = cm_err;
		err_entry.fid = &ep->util_ep.ep_fid.fid;
		err_entry.context = ep->util_ep.ep_fid.fid.context;
		if (err_data && err_data_size > 0) {
			err_entry.err_data = mem_dup(err_data, err_data_size);
			if (err_entry.err_data)
				err_entry.err_data_size = err_data_size;
		}
		(void) fi_eq_write(&ep->util_ep.eq->eq_fid, FI_SHUTDOWN,
				   &err_entry, sizeof(err_entry),
				   UTIL_FLAG_ERROR);
	} else {
		cm_entry.fid = &ep->util_ep.ep_fid.fid;
		(void) fi_eq_write(&ep->util_ep.eq->eq_fid, FI_SHUTDOWN,
				   &cm_entry, sizeof(cm_entry), 0);
	}
	ep->state = TCP2_DISCONNECTED;
}

static int tcp2_ep_shutdown(struct fid_ep *ep_fid, uint64_t flags)
{
	struct tcp2_progress *progress;
	struct tcp2_ep *ep;

	ep = container_of(ep_fid, struct tcp2_ep, util_ep.ep_fid);

	progress = tcp2_ep2_progress(ep);
	ofi_mutex_lock(&progress->lock);
	ofi_mutex_lock(&ep->lock);
	(void) ofi_bsock_flush(&ep->bsock);
	tcp2_ep_disable(ep, 0, NULL, 0);
	ofi_mutex_unlock(&ep->lock);
	ofi_mutex_unlock(&progress->lock);

	return FI_SUCCESS;
}

static int tcp2_ep_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct tcp2_ep *ep;
	size_t addrlen_in = *addrlen;
	int ret;

	ep = container_of(fid, struct tcp2_ep, util_ep.ep_fid);
	ret = ofi_getsockname(ep->bsock.sock, addr, (socklen_t *) addrlen);
	if (ret)
		return -ofi_sockerr();

	return (addrlen_in < *addrlen)? -FI_ETOOSMALL: FI_SUCCESS;
}

static int tcp2_ep_getpeer(struct fid_ep *ep_fid, void *addr, size_t *addrlen)
{
	struct tcp2_ep *ep;
	size_t addrlen_in = *addrlen;
	int ret;

	ep = container_of(ep_fid, struct tcp2_ep, util_ep.ep_fid);
	ret = ofi_getpeername(ep->bsock.sock, addr, (socklen_t *) addrlen);
	if (ret)
		return -ofi_sockerr();

	return (addrlen_in < *addrlen) ? -FI_ETOOSMALL: FI_SUCCESS;
}

static struct fi_ops_cm tcp2_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = tcp2_ep_getname,
	.getpeer = tcp2_ep_getpeer,
	.connect = tcp2_ep_connect,
	.listen = fi_no_listen,
	.accept = tcp2_ep_accept,
	.reject = fi_no_reject,
	.shutdown = tcp2_ep_shutdown,
	.join = fi_no_join,
};

void tcp2_reset_rx(struct tcp2_ep *ep)
{
	ep->cur_rx.handler = NULL;
	ep->cur_rx.entry = NULL;
	ep->cur_rx.hdr_done = 0;
	ep->cur_rx.hdr_len = sizeof(ep->cur_rx.hdr.base_hdr);
	OFI_DBG_SET(ep->cur_rx.hdr.base_hdr.version, 0);
}

static void tcp2_ep_cancel_rx(struct tcp2_ep *ep, void *context)
{
	struct slist_entry *cur, *prev;
	struct tcp2_xfer_entry *xfer_entry;
	struct tcp2_cq *cq;

	assert(ofi_mutex_held(&ep->lock));

	/* To cancel an active receive, we would need to flush the socket of
	 * all data associated with that message.  Since some of that data
	 * may not have arrived yet, this would require additional state
	 * tracking and complexity.  Fail the cancel in this case, since
	 * the receive is already in process anyway.
	 */
	slist_foreach(&ep->rx_queue, cur, prev) {
		xfer_entry = container_of(cur, struct tcp2_xfer_entry, entry);
		if (xfer_entry->context == context) {
			if (ep->cur_rx.entry != xfer_entry)
				goto found;
			break;
		}
	}

	return;

found:
	cq = container_of(ep->util_ep.rx_cq, struct tcp2_cq, util_cq);

	slist_remove(&ep->rx_queue, cur, prev);
	ep->rx_avail++;
	tcp2_cq_report_error(&cq->util_cq, xfer_entry, FI_ECANCELED);
	tcp2_free_xfer(cq, xfer_entry);
}

/* We currently only support canceling receives, which is the common case.
 * Canceling an operation from the other queues is not trivial,
 * especially if the operation has already been initiated.
 */
static ssize_t tcp2_ep_cancel(fid_t fid, void *context)
{
	struct tcp2_ep *ep;

	ep = container_of(fid, struct tcp2_ep, util_ep.ep_fid.fid);

	ofi_mutex_lock(&ep->lock);
	tcp2_ep_cancel_rx(ep, context);
	ofi_mutex_unlock(&ep->lock);

	return 0;
}

static int tcp2_ep_close(struct fid *fid)
{
	struct tcp2_progress *progress;
	struct tcp2_ep *ep;

	ep = container_of(fid, struct tcp2_ep, util_ep.ep_fid.fid);

	progress = tcp2_ep2_progress(ep);
	ofi_mutex_lock(&progress->lock);
	dlist_remove_init(&ep->progress_entry);
	tcp2_halt_sock(progress, ep->bsock.sock);
	ofi_mutex_unlock(&progress->lock);

	/* Lock not technically needed, since we're freeing the EP.  But it's
	 * harmless to acquire and silences static code analysis tools.
	 */
	ofi_mutex_lock(&ep->lock);
	tcp2_ep_flush_all_queues(ep);
	ofi_mutex_unlock(&ep->lock);

	if (ep->util_ep.eq) {
		ofi_eq_remove_fid_events(ep->util_ep.eq,
					 &ep->util_ep.ep_fid.fid);
		ofi_atomic_dec32(&ep->util_ep.eq->ref);
	}

	free(ep->cm_msg);
	ofi_close_socket(ep->bsock.sock);

	if (ep->util_ep.rx_cq)
		ofi_atomic_dec32(&ep->util_ep.rx_cq->ref);
	if (ep->util_ep.tx_cq)
		ofi_atomic_dec32(&ep->util_ep.tx_cq->ref);
	if (ep->util_ep.rx_cntr)
		ofi_atomic_dec32(&ep->util_ep.rx_cntr->ref);
	if (ep->util_ep.tx_cntr)
		ofi_atomic_dec32(&ep->util_ep.tx_cntr->ref);
	if (ep->util_ep.wr_cntr)
		ofi_atomic_dec32(&ep->util_ep.wr_cntr->ref);
	if (ep->util_ep.rd_cntr)
		ofi_atomic_dec32(&ep->util_ep.rd_cntr->ref);
	if (ep->util_ep.rem_wr_cntr)
		ofi_atomic_dec32(&ep->util_ep.rem_wr_cntr->ref);
	if (ep->util_ep.rem_rd_cntr)
		ofi_atomic_dec32(&ep->util_ep.rem_rd_cntr->ref);

	ofi_atomic_dec32(&ep->util_ep.domain->ref);
	ofi_mutex_destroy(&ep->util_ep.lock);
	ofi_mutex_destroy(&ep->lock);

	free(ep);
	return 0;
}

static int tcp2_ep_ctrl(struct fid *fid, int command, void *arg)
{
	struct tcp2_ep *ep;

	ep = container_of(fid, struct tcp2_ep, util_ep.ep_fid.fid);
	switch (command) {
	case FI_ENABLE:
		if ((ofi_needs_rx(ep->util_ep.caps) && !ep->util_ep.rx_cq) ||
		    (ofi_needs_tx(ep->util_ep.caps) && !ep->util_ep.tx_cq)) {
			FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
				"missing needed CQ binding\n");
			return -FI_ENOCQ;
		}
		break;
	default:
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL, "unsupported command\n");
		return -FI_ENOSYS;
	}
	return FI_SUCCESS;
}

static int tcp2_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct tcp2_ep *ep;
	struct tcp2_rx_ctx *rx_ctx;
	int ret;

	ep = container_of(fid, struct tcp2_ep, util_ep.ep_fid.fid);

	if (bfid->fclass == FI_CLASS_SRX_CTX) {
		rx_ctx = container_of(bfid, struct tcp2_rx_ctx, rx_fid.fid);
		ep->srx_ctx = rx_ctx;
		return FI_SUCCESS;
	}

	ret = ofi_ep_bind(&ep->util_ep, bfid, flags);
	if (!ret && (bfid->fclass == FI_CLASS_CNTR))
		ep->report_success = tcp2_report_cntr_success;

	return ret;
}

static struct fi_ops tcp2_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = tcp2_ep_close,
	.bind = tcp2_ep_bind,
	.control = tcp2_ep_ctrl,
	.ops_open = fi_no_ops_open,
};

static int tcp2_ep_getopt(fid_t fid, int level, int optname,
			  void *optval, size_t *optlen)
{
	struct tcp2_ep *ep;

	if (level != FI_OPT_ENDPOINT)
		return -ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_MIN_MULTI_RECV:
		if (*optlen < sizeof(size_t)) {
			*optlen = sizeof(size_t);
			return -FI_ETOOSMALL;
		}
		ep = container_of(fid, struct tcp2_ep,
				  util_ep.ep_fid.fid);
		*((size_t *) optval) = ep->min_multi_recv_size;
		*optlen = sizeof(size_t);
		break;
	case FI_OPT_CM_DATA_SIZE:
		if (*optlen < sizeof(size_t)) {
			*optlen = sizeof(size_t);
			return -FI_ETOOSMALL;
		}
		*((size_t *) optval) = TCP2_MAX_CM_DATA_SIZE;
		*optlen = sizeof(size_t);
		break;
	default:
		return -FI_ENOPROTOOPT;
	}
	return FI_SUCCESS;
}

int tcp2_ep_setopt(fid_t fid, int level, int optname,
		   const void *optval, size_t optlen)
{
	struct tcp2_ep *ep;

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	ep = container_of(fid, struct tcp2_ep, util_ep.ep_fid.fid);
	switch (optname) {
	case FI_OPT_MIN_MULTI_RECV:
		if (optlen != sizeof(size_t))
			return -FI_EINVAL;

		ep->min_multi_recv_size = *(size_t *) optval;
		FI_INFO(&tcp2_prov, FI_LOG_EP_CTRL,
			"FI_OPT_MIN_MULTI_RECV set to %zu\n",
			ep->min_multi_recv_size);
		break;
	case OFI_OPT_TCP_FI_ADDR:
		if (optlen != sizeof(fi_addr_t))
			return -FI_EINVAL;
		ep->src_addr = *(fi_addr_t *) optval;
		break;
	default:
		return -ENOPROTOOPT;
	}

	return FI_SUCCESS;
}

static struct fi_ops_ep tcp2_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = tcp2_ep_cancel,
	.getopt = tcp2_ep_getopt,
	.setopt = tcp2_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

int tcp2_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep_fid, void *context)
{
	struct tcp2_ep *ep;
	struct tcp2_pep *pep;
	struct tcp2_conn_handle *conn;
	int ret;

	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return -FI_ENOMEM;

	ret = ofi_endpoint_init(domain, &tcp2_util_prov, info, &ep->util_ep,
				context, NULL);
	if (ret)
		goto err1;

	ofi_bsock_init(&ep->bsock, tcp2_staging_sbuf_size,
		       tcp2_prefetch_rbuf_size);
	if (info->handle) {
		if (((fid_t) info->handle)->fclass == FI_CLASS_PEP) {
			pep = container_of(info->handle, struct tcp2_pep,
					   util_pep.pep_fid.fid);

			ep->bsock.sock = pep->sock;
			pep->sock = INVALID_SOCKET;
		} else {
			ep->state = TCP2_ACCEPTING;
			conn = container_of(info->handle,
					    struct tcp2_conn_handle, fid);
			/* EP now owns socket */
			ep->bsock.sock = conn->sock;
			conn->sock = INVALID_SOCKET;
			ep->hdr_bswap = conn->endian_match ?
					tcp2_hdr_none : tcp2_hdr_bswap;
			/* Save handle, but we only free if user calls accept.
			 * Otherwise, user will call reject, which will free it.
			 */
			ep->conn = conn;

			ret = tcp2_setup_socket(ep->bsock.sock, info);
			if (ret)
				goto err3;
		}
	} else {
		ep->bsock.sock = ofi_socket(ofi_get_sa_family(info), SOCK_STREAM, 0);
		if (ep->bsock.sock == INVALID_SOCKET) {
			ret = -ofi_sockerr();
			goto err2;
		}

		ret = tcp2_setup_socket(ep->bsock.sock, info);
		if (ret)
			goto err3;

		tcp2_set_zerocopy(ep->bsock.sock);

		if (info->src_addr && (!ofi_is_any_addr(info->src_addr) ||
					ofi_addr_get_port(info->src_addr))) {

			if (!ofi_addr_get_port(info->src_addr)) {
				tcp2_set_no_port(ep->bsock.sock);
			}

			ret = bind(ep->bsock.sock, info->src_addr,
				(socklen_t) info->src_addrlen);
			if (ret) {
				FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL, "bind failed\n");
				ret = -ofi_sockerr();
				goto err3;
			}
		}
	}

	ret = ofi_mutex_init(&ep->lock);
	if (ret)
		goto err3;

	ep->cm_msg = calloc(1, sizeof(*ep->cm_msg));
	if (!ep->cm_msg) {
		ret = -FI_ENOMEM;
		goto err4;
	}

	dlist_init(&ep->progress_entry);
	slist_init(&ep->rx_queue);
	slist_init(&ep->tx_queue);
	slist_init(&ep->priority_queue);
	slist_init(&ep->rma_read_queue);
	slist_init(&ep->need_ack_queue);
	slist_init(&ep->async_queue);

	if (info->ep_attr->rx_ctx_cnt != FI_SHARED_CONTEXT)
		ep->rx_avail = (int) info->rx_attr->size;

	ep->cur_rx.hdr_done = 0;
	ep->cur_rx.hdr_len = sizeof(ep->cur_rx.hdr.base_hdr);
	ep->min_multi_recv_size = TCP2_MIN_MULTI_RECV;
	tcp2_config_bsock(&ep->bsock);
	ep->report_success = tcp2_report_success;

	*ep_fid = &ep->util_ep.ep_fid;
	(*ep_fid)->fid.ops = &tcp2_ep_fi_ops;
	(*ep_fid)->ops = &tcp2_ep_ops;
	(*ep_fid)->cm = &tcp2_cm_ops;
	(*ep_fid)->msg = &tcp2_msg_ops;
	(*ep_fid)->rma = &tcp2_rma_ops;
	(*ep_fid)->tagged = &tcp2_tagged_ops;

	return 0;

err4:
	ofi_mutex_destroy(&ep->lock);
err3:
	ofi_close_socket(ep->bsock.sock);
err2:
	ofi_endpoint_close(&ep->util_ep);
err1:
	free(ep);
	return ret;
}
