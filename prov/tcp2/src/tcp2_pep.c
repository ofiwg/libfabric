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


static int tcp2_pep_close(struct fid *fid)
{
	struct tcp2_progress *progress;
	struct tcp2_pep *pep;

	pep = container_of(fid, struct tcp2_pep, util_pep.pep_fid.fid);
	/* TODO: We need to abort any outstanding active connection requests.
	 * The tcp2_conn_handle points back to the pep and will dereference
	 * the freed memory if we continue.
	 */

	if (pep->state == TCP2_LISTENING) {
		progress = tcp2_pep2_progress(pep);
		ofi_mutex_lock(&progress->lock);
		tcp2_halt_sock(progress, pep->sock);
		ofi_mutex_unlock(&progress->lock);
	}

	ofi_close_socket(pep->sock);
	ofi_pep_close(&pep->util_pep);
	fi_freeinfo(pep->info);
	free(pep);
	return 0;
}

static int tcp2_pep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct tcp2_pep *pep;

	pep = container_of(fid, struct tcp2_pep, util_pep.pep_fid.fid);

	switch (bfid->fclass) {
	case FI_CLASS_EQ:
		return ofi_pep_bind_eq(&pep->util_pep,
				       container_of(bfid, struct util_eq,
						    eq_fid.fid), flags);
	default:
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"invalid FID class for binding\n");
		return -FI_EINVAL;
	}
}

static struct fi_ops tcp2_pep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = tcp2_pep_close,
	.bind = tcp2_pep_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int tcp2_bind_to_port_range(SOCKET sock, void* src_addr, size_t addrlen)
{
	int ret, i, rand_port_number;
	static uint32_t seed;
	if (!seed)
		seed = ofi_generate_seed();

	rand_port_number = ofi_xorshift_random_r(&seed) %
			   (tcp2_ports.high + 1 - tcp2_ports.low) +
			   tcp2_ports.low;

	for (i = tcp2_ports.low; i <= tcp2_ports.high; i++, rand_port_number++) {
		if (rand_port_number > tcp2_ports.high)
			rand_port_number = tcp2_ports.low;

		ofi_addr_set_port(src_addr, (uint16_t) rand_port_number);
		ret = bind(sock, src_addr, (socklen_t) addrlen);
		if (ret) {
			if (ofi_sockerr() == EADDRINUSE)
				continue;

			FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
				"failed to bind listener: %s\n",
				strerror(ofi_sockerr()));
			return -ofi_sockerr();
		}
		break;
	}
	return (i <= tcp2_ports.high) ? FI_SUCCESS : -FI_EADDRNOTAVAIL;
}

static int tcp2_pep_sock_create(struct tcp2_pep *pep)
{
	int ret, af;

	switch (pep->info->addr_format) {
	case FI_SOCKADDR:
	case FI_SOCKADDR_IN:
	case FI_SOCKADDR_IN6:
		af = ((struct sockaddr *)pep->info->src_addr)->sa_family;
		break;
	default:
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"invalid source address format\n");
		return -FI_EINVAL;
	}

	pep->sock = ofi_socket(af, SOCK_STREAM, 0);
	if (pep->sock == INVALID_SOCKET) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"failed to create listener: %s\n",
			strerror(ofi_sockerr()));
		return -FI_EIO;
	}
	ret = tcp2_setup_socket(pep->sock, pep->info);
	if (ret)
		goto err;

	tcp2_set_zerocopy(pep->sock);
	ret = fi_fd_nonblock(pep->sock);
	if (ret) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"failed to set listener socket to nonblocking\n");
		goto err;
	}

	if (ofi_addr_get_port(pep->info->src_addr) != 0 || tcp2_ports.high == 0) {
		ret = bind(pep->sock, pep->info->src_addr,
			  (socklen_t) pep->info->src_addrlen);
		if (ret)
			ret = -ofi_sockerr();
	} else {
		ret = tcp2_bind_to_port_range(pep->sock, pep->info->src_addr,
					      pep->info->src_addrlen);
	}

	if (ret) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"failed to bind listener: %s\n",
			strerror(ofi_sockerr()));
		goto err;
	}
	return FI_SUCCESS;
err:
	ofi_close_socket(pep->sock);
	pep->sock = INVALID_SOCKET;
	return ret;
}

static int tcp2_pep_setname(fid_t fid, void *addr, size_t addrlen)
{
	struct tcp2_pep *pep;

	if ((addrlen != sizeof(struct sockaddr_in)) &&
	    (addrlen != sizeof(struct sockaddr_in6)))
		return -FI_EINVAL;

	pep = container_of(fid, struct tcp2_pep,
				util_pep.pep_fid);

	if (pep->sock != INVALID_SOCKET) {
		ofi_close_socket(pep->sock);
		pep->sock = INVALID_SOCKET;
	}

	if (pep->info->src_addr) {
		free(pep->info->src_addr);
		pep->info->src_addrlen = 0;
	}

	pep->info->src_addr = mem_dup(addr, addrlen);
	if (!pep->info->src_addr)
		return -FI_ENOMEM;
	pep->info->src_addrlen = addrlen;

	return tcp2_pep_sock_create(pep);
}

static int tcp2_pep_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct tcp2_pep *pep;
	size_t addrlen_in = *addrlen;
	int ret;

	pep = container_of(fid, struct tcp2_pep, util_pep.pep_fid);
	ret = ofi_getsockname(pep->sock, addr, (socklen_t *) addrlen);
	if (ret)
		return -ofi_sockerr();

	return (addrlen_in < *addrlen) ? -FI_ETOOSMALL: FI_SUCCESS;
}

static int tcp2_pep_listen(struct fid_pep *pep_fid)
{
	struct tcp2_progress *progress;
	struct tcp2_pep *pep;
	int ret;

	pep = container_of(pep_fid, struct tcp2_pep, util_pep.pep_fid);
	if (pep->state != TCP2_IDLE) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"passive endpoint is not idle\n");
		return -FI_EINVAL;
	}

	/* arbitrary backlog value to support larger scale jobs */
	if (listen(pep->sock, 4096)) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"socket listen failed\n");
		return -ofi_sockerr();
	}

	progress = tcp2_pep2_progress(pep);
	ofi_mutex_lock(&progress->lock);
	ret = tcp2_monitor_sock(progress, pep->sock, POLLIN,
				&pep->util_pep.pep_fid.fid);
	ofi_mutex_unlock(&progress->lock);
	if (!ret)
		pep->state = TCP2_LISTENING;

	return ret;
}

static int tcp2_pep_reject(struct fid_pep *pep, fid_t fid_handle,
			   const void *param, size_t paramlen)
{
	struct tcp2_cm_msg msg;
	struct tcp2_conn_handle *conn;
	ssize_t size_ret;
	int ret;

	FI_DBG(&tcp2_prov, FI_LOG_EP_CTRL, "rejecting connection");
	conn = container_of(fid_handle, struct tcp2_conn_handle, fid);
	/* If we created an endpoint, it owns the socket */
	if (conn->sock == INVALID_SOCKET)
		goto free;

	memset(&msg.hdr, 0, sizeof(msg.hdr));
	msg.hdr.version = TCP2_CTRL_HDR_VERSION;
	msg.hdr.type = ofi_ctrl_nack;
	msg.hdr.seg_size = htons((uint16_t) paramlen);
	if (paramlen)
		memcpy(&msg.data, param, paramlen);

	size_ret = ofi_send_socket(conn->sock, &msg,
				   sizeof(msg.hdr) + paramlen, MSG_NOSIGNAL);
	if ((size_t) size_ret != sizeof(msg.hdr) + paramlen)
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"sending of reject message failed\n");

	ofi_shutdown(conn->sock, SHUT_RDWR);
	ret = ofi_close_socket(conn->sock);
	if (ret)
		return ret;

free:
	free(conn);
	return FI_SUCCESS;
}

static struct fi_ops_cm tcp2_pep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = tcp2_pep_setname,
	.getname = tcp2_pep_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = tcp2_pep_listen,
	.accept = fi_no_accept,
	.reject = tcp2_pep_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

static int  tcp2_pep_getopt(fid_t fid, int level, int optname,
			    void *optval, size_t *optlen)
{
	if ( level != FI_OPT_ENDPOINT ||
	     optname != FI_OPT_CM_DATA_SIZE)
		return -FI_ENOPROTOOPT;

	if (*optlen < sizeof(size_t)) {
		*optlen = sizeof(size_t);
		return -FI_ETOOSMALL;
	}

	*((size_t *) optval) = TCP2_MAX_CM_DATA_SIZE;
	*optlen = sizeof(size_t);
	return FI_SUCCESS;
}

static struct fi_ops_ep tcp2_pep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.getopt = tcp2_pep_getopt,
	.setopt = fi_no_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

int tcp2_passive_ep(struct fid_fabric *fabric, struct fi_info *info,
		    struct fid_pep **pep_fid, void *context)
{
	struct tcp2_pep *pep;
	int ret;

	if (!info) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,"invalid info\n");
		return -FI_EINVAL;
	}

	ret = ofi_prov_check_info(&tcp2_util_prov, fabric->api_version, info);
	if (ret)
		return ret;

	pep = calloc(1, sizeof(*pep));
	if (!pep)
		return -FI_ENOMEM;

	ret = ofi_pep_init(fabric, info, &pep->util_pep, context);
	if (ret)
		goto err1;

	pep->util_pep.pep_fid.fid.ops = &tcp2_pep_fi_ops;
	pep->util_pep.pep_fid.cm = &tcp2_pep_cm_ops;
	pep->util_pep.pep_fid.ops = &tcp2_pep_ops;

	pep->info = fi_dupinfo(info);
	if (!pep->info) {
		ret = -FI_ENOMEM;
		goto err2;
	}

	pep->sock = INVALID_SOCKET;
	pep->state = TCP2_IDLE;

	if (info->src_addr) {
		ret = tcp2_pep_sock_create(pep);
		if (ret)
			goto err3;
	}

	*pep_fid = &pep->util_pep.pep_fid;
	return FI_SUCCESS;
err3:
	fi_freeinfo(pep->info);
err2:
	ofi_pep_close(&pep->util_pep);
err1:
	free(pep);
	return ret;
}
