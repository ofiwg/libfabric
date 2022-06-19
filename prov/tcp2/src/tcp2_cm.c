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

#include <rdma/fi_errno.h>

#include <ofi_prov.h>
#include "tcp2.h"
#include <poll.h>
#include <sys/types.h>
#include <ofi_util.h>


/* Must be castable to struct fi_eq_cm_entry */
struct tcp2_cm_entry {
	fid_t			fid;
	struct fi_info		*info;
	uint8_t			data[TCP2_MAX_CM_DATA_SIZE];
};

/* The underlying socket has the POLLIN event set.  The entire
 * CM message should be readable, as it fits within a single MTU
 * and is the first data transferred over the socket.
 */
static int
tcp2_recv_cm_msg(SOCKET sock, struct tcp2_cm_msg *msg, uint8_t exp_msg)
{
	size_t len;
	ssize_t ret;

	len = sizeof(msg->hdr);
	ret = ofi_recv_socket(sock, &msg->hdr, len, 0);
	if ((size_t) ret != len) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"Failed to read cm header\n");
		ret = ofi_sockerr() ? -ofi_sockerr() : -FI_EIO;
		goto err;
	}

	if (msg->hdr.version != TCP2_CTRL_HDR_VERSION) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"cm protocol version mismatch\n");
		ret = -FI_ENOPROTOOPT;
		goto err;
	}

	if (msg->hdr.type != exp_msg && msg->hdr.type != ofi_ctrl_nack) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"unexpected message, expected %d or %d got: %d\n",
			exp_msg, ofi_ctrl_nack, msg->hdr.type);
		ret = -FI_ECONNREFUSED;
		goto err;
	}

	len = ntohs(msg->hdr.seg_size);
	if (len) {
		if (len > TCP2_MAX_CM_DATA_SIZE) {
			FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
				"cm data size is too large\n");
			ret = -FI_ENOPROTOOPT;
			goto err;
		}

		ret = ofi_recv_socket(sock, msg->data, len, 0);
		if ((size_t) ret != len) {
			FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
				"Failed to read cm data\n");
			ret = ofi_sockerr() ? -ofi_sockerr() : -FI_EIO;
			goto err;
		}
	}

	if (msg->hdr.type == ofi_ctrl_nack) {
		FI_INFO(&tcp2_prov, FI_LOG_EP_CTRL,
			"Connection refused from remote\n");
		return -FI_ECONNREFUSED;
	}

	return 0;

err:
	/* For any unexpected error, we have no cm data to report */
	msg->hdr.seg_size = 0;
	return ret;
}

/* The underlying socket has the POLLOUT event set.  It is ready
 * to accept outbound data.  We expect to transfer the entire CM
 * message as it fits into a single MTU and is the first data
 * transferred over the socket.
 */
int tcp2_send_cm_msg(struct tcp2_ep *ep)
{
	size_t len;
	ssize_t ret;

	len = sizeof(ep->cm_msg->hdr) + ntohs(ep->cm_msg->hdr.seg_size);

	ret = ofi_send_socket(ep->bsock.sock, ep->cm_msg, len, MSG_NOSIGNAL);
	if ((size_t) ret != len)
		return ofi_sockerr() ? -ofi_sockerr() : -FI_EIO;

	return FI_SUCCESS;
}

void tcp2_req_done(struct tcp2_ep *ep)
{
	struct tcp2_progress *progress;
	struct tcp2_cm_entry cm_entry;
	uint16_t len;
	ssize_t ret;

	FI_DBG(&tcp2_prov, FI_LOG_EP_CTRL, "connect request done\n");
	assert(ofi_genlock_held(&tcp2_ep2_progress(ep)->lock));
	assert(ofi_mutex_held(&ep->lock));

	ret = tcp2_recv_cm_msg(ep->bsock.sock, ep->cm_msg, ofi_ctrl_connresp);
	if (ret) {
		if (ret == -FI_EAGAIN)
			return; /* This shouldn't happen */

		enum fi_log_level level = (ret == -FI_ECONNREFUSED) ?
				FI_LOG_INFO : FI_LOG_WARN;
		FI_LOG(&tcp2_prov, level, FI_LOG_EP_CTRL,
			"Failed to receive connect response\n");
		goto disable;
	}

	ep->hdr_bswap = (ep->cm_msg->hdr.conn_data == 1) ?
			tcp2_hdr_none : tcp2_hdr_bswap;

	len = ntohs(ep->cm_msg->hdr.seg_size);
	cm_entry.fid = &ep->util_ep.ep_fid.fid;
	cm_entry.info = NULL;
	if (len)
		memcpy(cm_entry.data, ep->cm_msg->data, len);

	ret = (int) fi_eq_write(&ep->util_ep.eq->eq_fid, FI_CONNECTED, &cm_entry,
				sizeof(struct fi_eq_cm_entry) + len, 0);
	if (ret < 0) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL, "Error writing to EQ\n");
		goto disable;
	}

	if (tcp2_active_wait(ep)) {
		progress = tcp2_ep2_progress(ep);
		dlist_insert_tail(&ep->progress_entry,
				  &progress->active_wait_list);
		tcp2_signal_progress(progress);
	}
	ep->state = TCP2_CONNECTED;
	free(ep->cm_msg);
	ep->cm_msg = NULL;
	return;

disable:
	tcp2_ep_disable(ep, -ret, ep->cm_msg->data,
			ntohs(ep->cm_msg->hdr.seg_size));
}

void tcp2_run_conn(struct tcp2_conn_handle *conn, bool pin, bool pout, bool perr)
{
	struct tcp2_cm_msg msg;
	struct tcp2_cm_entry cm_entry;
	socklen_t len;
	uint16_t datalen;
	int ret;

	FI_DBG(&tcp2_prov, FI_LOG_EP_CTRL, "Receiving connect request\n");
	assert(ofi_genlock_held(&conn->pep->progress->lock));

	/* Don't monitor the socket until the user calls fi_accept */
	tcp2_halt_sock(conn->pep->progress, conn->sock);

	if (perr) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL, "socket error\n");
		goto close;
	}

	ret = tcp2_recv_cm_msg(conn->sock, &msg, ofi_ctrl_connreq);
	if (ret) {
		if (ret == -FI_EAGAIN)
			return;
		goto close;
	}

	cm_entry.fid = &conn->pep->util_pep.pep_fid.fid;
	cm_entry.info = fi_dupinfo(conn->pep->info);
	if (!cm_entry.info)
		goto close;

	cm_entry.info->dest_addrlen = conn->pep->info->src_addrlen;
	len = (socklen_t) cm_entry.info->dest_addrlen;

	free(cm_entry.info->dest_addr);
	cm_entry.info->dest_addr = malloc(len);
	if (!cm_entry.info->dest_addr)
		goto freeinfo;

	ret = ofi_getpeername(conn->sock, cm_entry.info->dest_addr, &len);
	if (ret)
		goto freeinfo;

	conn->endian_match = (msg.hdr.conn_data == 1);
	cm_entry.info->handle = &conn->fid;
	datalen = ntohs(msg.hdr.seg_size);
	if (datalen)
		memcpy(cm_entry.data, msg.data, datalen);

	ret = (int) fi_eq_write(&conn->pep->util_pep.eq->eq_fid,
				FI_CONNREQ, &cm_entry,
				sizeof(struct fi_eq_cm_entry) + datalen, 0);
	if (ret < 0) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL, "Error writing to EQ\n");
		goto freeinfo;
	}

	return;

freeinfo:
	fi_freeinfo(cm_entry.info);
close:
	ofi_close_socket(conn->sock);
	free(conn);
}

void tcp2_connect_done(struct tcp2_ep *ep)
{
	socklen_t len;
	int status, ret;

	FI_DBG(&tcp2_prov, FI_LOG_EP_CTRL, "socket connected, sending req\n");
	assert(ofi_genlock_held(&tcp2_ep2_progress(ep)->lock));
	assert(ofi_mutex_held(&ep->lock));

	len = sizeof(status);
	ret = getsockopt(ep->bsock.sock, SOL_SOCKET, SO_ERROR,
			 (char *) &status, &len);
	if (ret < 0 || status) {
		ret = (ret < 0)? -ofi_sockerr() : -status;
		FI_WARN_SPARSE(&tcp2_prov, FI_LOG_EP_CTRL,
				"connection failure (sockerr %d)\n", ret);
		goto disable;
	}

	ret = tcp2_send_cm_msg(ep);
	if (ret)
		goto disable;

	ep->state = TCP2_REQ_SENT;
	tcp2_update_poll(ep);
	return;

disable:
	tcp2_ep_disable(ep, -ret, NULL, 0);
}

void tcp2_accept_sock(struct tcp2_pep *pep)
{
	struct tcp2_conn_handle *conn;
	SOCKET sock;
	int ret;

	FI_DBG(&tcp2_prov, FI_LOG_EP_CTRL, "accepting socket\n");
	assert(ofi_genlock_held(&pep->progress->lock));

	sock = accept(pep->sock, NULL, 0);
	if (sock < 0) {
		if (!OFI_SOCK_TRY_ACCEPT_AGAIN(ofi_sockerr())) {
			FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
				"accept error: %d\n", ofi_sockerr());
		}
		return;
	}

	conn = calloc(1, sizeof(*conn));
	if (!conn) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"cannot allocate memory\n");
		goto close;
	}

	conn->sock = sock;
	conn->fid.fclass = FI_CLASS_CONNREQ;
	/* TODO: We need to hold a reference on the pep to defer destruction */
	conn->pep = pep;

	ret = tcp2_monitor_sock(pep->progress, sock, POLLIN, &conn->fid);
	if (ret)
		goto free;

	return;

free:
	free(conn);
close:
	ofi_close_socket(sock);
}
