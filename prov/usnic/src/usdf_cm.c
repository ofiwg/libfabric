/*
 * Copyright (c) 2014-2017, Cisco Systems, Inc. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "config.h"

#include <asm/types.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/epoll.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_errno.h>
#include "ofi.h"
#include "ofi_file.h"

#include "usnic_direct.h"
#include "usdf.h"
#include "usdf_endpoint.h"
#include "usdf_dgram.h"
#include "usdf_msg.h"
#include "usdf_av.h"
#include "usdf_cm.h"

void
usdf_cm_msg_connreq_cleanup(struct usdf_connreq *crp)
{
	struct usdf_ep *ep;
	struct usdf_pep *pep;
	struct usdf_fabric *fp;

	ep = crp->cr_ep;
	pep = crp->cr_pep;
	if (pep != NULL) {
		fp = pep->pep_fabric;
	} else {
		fp = ep->ep_domain->dom_fabric;
	}

	if (crp->cr_pollitem.pi_rtn != NULL) {
		(void) epoll_ctl(fp->fab_epollfd, EPOLL_CTL_DEL, crp->cr_sockfd, NULL);
		crp->cr_pollitem.pi_rtn = NULL;
	}
	if (crp->cr_sockfd != -1) {
		close(crp->cr_sockfd);
		crp->cr_sockfd = -1;
	}

	/* If there is a passive endpoint, recycle the crp */
	if (pep != NULL) {
		if (TAILQ_ON_LIST(crp, cr_link)) {
			TAILQ_REMOVE(&pep->pep_cr_pending, crp, cr_link);
		}
		TAILQ_INSERT_TAIL(&pep->pep_cr_free, crp, cr_link);
	} else {
		free(crp);
	}
}

static int
usdf_cm_msg_accept_complete(struct usdf_connreq *crp)
{
	struct usdf_ep *ep;
	struct fi_eq_cm_entry entry;
	int ret;

	ep = crp->cr_ep;

	/* post EQ entry */
	entry.fid = ep_utofid(ep);
	entry.info = NULL;
	ret = usdf_eq_write_internal(ep->ep_eq, FI_CONNECTED, &entry,
			sizeof(entry), 0);
	if (ret != sizeof(entry)) {
		usdf_cm_report_failure(crp, ret, false);
		return 0;
	}

	usdf_cm_msg_connreq_cleanup(crp);

	return 0;
}

int
usdf_cm_msg_accept(struct fid_ep *fep, const void *param, size_t paramlen)
{
	struct usdf_ep *ep;
	struct usdf_rx *rx;
	struct usdf_domain *udp;
	struct usdf_fabric *fp;
	struct usdf_connreq *crp;
	struct usdf_connreq_msg *reqp;
	struct usd_qp_impl *qp;
	int ret;
	int n;

	USDF_TRACE_SYS(EP_CTRL, "\n");

	if (paramlen > USDF_MAX_CONN_DATA)
		return -FI_EINVAL;

	ep = ep_ftou(fep);
	udp = ep->ep_domain;
	fp = udp->dom_fabric;
	crp = ep->e.msg.ep_connreq;
	if (crp == NULL) {
		return -FI_ENOTCONN;
	}
	if (ep->ep_eq == NULL) {
		return -FI_ENOEQ;
	}
	crp->cr_ep = ep;
	reqp = (struct usdf_connreq_msg *)crp->cr_data;

	ep->e.msg.ep_lcl_peer_id = ntohs(reqp->creq_peer_id);

	/* start creating the dest early */
	ret = usd_create_dest(udp->dom_dev, reqp->creq_ipaddr,
			reqp->creq_port, &ep->e.msg.ep_dest);
	if (ret != 0) {
		goto fail;
	}

	ep->e.msg.ep_dest->ds_dest.ds_udp.u_hdr.uh_ip.frag_off |= htons(IP_DF);

	ret = usdf_ep_msg_get_queues(ep);
	if (ret != 0) {
		goto fail;
	}
	rx = ep->ep_rx;
	qp = to_qpi(rx->rx_qp);

	/* allocate a peer ID */
	ep->e.msg.ep_rem_peer_id = udp->dom_next_peer;
	udp->dom_peer_tab[udp->dom_next_peer] = ep;
	++udp->dom_next_peer;

	crp->cr_ptr = crp->cr_data;
	crp->cr_resid = sizeof(*reqp) + paramlen;

	reqp->creq_peer_id = htons(ep->e.msg.ep_rem_peer_id);
	reqp->creq_ipaddr = fp->fab_dev_attrs->uda_ipaddr_be;
	reqp->creq_port =
		qp->uq_attrs.uqa_local_addr.ul_addr.ul_udp.u_addr.sin_port;
	reqp->creq_result = htonl(0);
	reqp->creq_datalen = htonl(paramlen);
	memcpy(reqp->creq_data, param, paramlen);

	n = write(crp->cr_sockfd, crp->cr_ptr, crp->cr_resid);
	if (n == -1) {
		usdf_cm_msg_connreq_cleanup(crp);
		ret = -errno;
		goto fail;
	}

	crp->cr_resid -= n;
	if (crp->cr_resid == 0) {
		usdf_cm_msg_accept_complete(crp);
	} else {
		// XXX set up epoll junk to send rest
	}

	return 0;
fail:
	free(ep->e.msg.ep_dest);
	/* XXX release queues */
	return ret;
}

/* Given a connection request structure containing data, make a copy of the data
 * that can be accessed in error entries on the EQ. The return value is the size
 * of the data stored in the error entry. If the return value is a non-negative
 * value, then the function has suceeded and the size and output data can be
 * assumed to be valid. If the function fails, then the data will be NULL and
 * the size will be a negative error value.
 */
static int usdf_cm_generate_err_data(struct usdf_eq *eq,
		struct usdf_connreq *crp, void **data)
{
	struct usdf_err_data_entry *err_data_entry;
	struct usdf_connreq_msg *reqp;
	size_t entry_size;
	size_t data_size;

	if (!eq || !crp || !data) {
		USDF_DBG_SYS(EP_CTRL,
				"eq, crp, or data is NULL.\n");
		return -FI_EINVAL;
	}

	/* Initialize to NULL so data can't be used in the error case. */
	*data = NULL;

	reqp = (struct usdf_connreq_msg *) crp->cr_data;

	/* This is a normal case, maybe there was no data. */
	if (!reqp || !reqp->creq_datalen)
		return 0;

	data_size = reqp->creq_datalen;

	entry_size = sizeof(*err_data_entry) + data_size;

	err_data_entry = calloc(1, entry_size);
	if (!err_data_entry) {
		USDF_WARN_SYS(EP_CTRL,
				"failed to allocate err data entry\n");
		return -FI_ENOMEM;
	}

	/* This data should be copied and owned by the provider. Keep
	 * track of it in the EQ, this will be freed in the next EQ read
	 * call after it has been read.
	 */
	memcpy(err_data_entry->err_data, reqp->creq_data, data_size);
	slist_insert_tail(&err_data_entry->entry, &eq->eq_err_data);

	*data = err_data_entry->err_data;

	return data_size;
}

/* Report a connection management related failure. Sometimes there is connection
 * event data that should be copied into the generated event. If the copy_data
 * parameter evaluates to true, then the data will be copied.
 *
 * If data is to be generated for the error entry, then the connection request
 * is assumed to have the data size in host order. If something fails during
 * processing of the error data, then the EQ entry will still be generated
 * without the error data.
 */
void usdf_cm_report_failure(struct usdf_connreq *crp, int error, bool copy_data)
{
	struct fi_eq_err_entry err = {0};
        struct usdf_pep *pep;
        struct usdf_ep *ep;
        struct usdf_eq *eq;
	fid_t fid;
	int ret;

	USDF_DBG_SYS(EP_CTRL, "error=%d (%s)\n", error, fi_strerror(error));

        pep = crp->cr_pep;
        ep = crp->cr_ep;

	if (ep != NULL) {
		fid = ep_utofid(ep);
		eq = ep->ep_eq;
		ep->ep_domain->dom_peer_tab[ep->e.msg.ep_rem_peer_id] = NULL;
	} else {
		fid = pep_utofid(pep);
		eq = pep->pep_eq;
	}

	/* Try to generate the space necessary for the error data. If the
	 * function returns a number greater than or equal to 0, then it was a
	 * success. The return value is the size of the data.
	 */
	if (copy_data) {
		ret = usdf_cm_generate_err_data(eq, crp, &err.err_data);
		if (ret >= 0)
			err.err_data_size = ret;
	}

        err.fid = fid;
        err.err = -error;

        usdf_eq_write_internal(eq, 0, &err, sizeof(err), USDF_EVENT_FLAG_ERROR);

        usdf_cm_msg_connreq_cleanup(crp);
}

/*
 * read connection request response from the listener
 */
static int
usdf_cm_msg_connect_cb_rd(void *v)
{
	struct usdf_connreq *crp;
	struct usdf_ep *ep;
	struct usdf_fabric *fp;
	struct usdf_domain *udp;
	struct usdf_connreq_msg *reqp;
	struct fi_eq_cm_entry *entry;
	size_t entry_len;
	int ret;

	crp = v;
	ep = crp->cr_ep;
	fp = ep->ep_domain->dom_fabric;

	ret = read(crp->cr_sockfd, crp->cr_ptr, crp->cr_resid);
	if (ret == -1)
		goto report_failure_skip_data;

	crp->cr_ptr += ret;
	crp->cr_resid -= ret;

	reqp = (struct usdf_connreq_msg *)crp->cr_data;
	if (crp->cr_resid == 0 && crp->cr_ptr == crp->cr_data + sizeof(*reqp)) {
		reqp->creq_datalen = ntohl(reqp->creq_datalen);
		crp->cr_resid = reqp->creq_datalen;
	}

	/* if resid is 0 now, completely done */
	if (crp->cr_resid == 0) {
		reqp->creq_result = ntohl(reqp->creq_result);

		ret = epoll_ctl(fp->fab_epollfd, EPOLL_CTL_DEL,
				crp->cr_sockfd, NULL);
		close(crp->cr_sockfd);
		crp->cr_sockfd = -1;

		if (reqp->creq_result != FI_SUCCESS) {
			/* Copy the data since this was an explicit rejection.
			 */
			usdf_cm_report_failure(crp, reqp->creq_result, true);
			return 0;
		}

		entry_len = sizeof(*entry) + reqp->creq_datalen;
		entry = malloc(entry_len);
		if (entry == NULL)
			goto report_failure_skip_data;

		udp = ep->ep_domain;
		ep->e.msg.ep_lcl_peer_id = ntohs(reqp->creq_peer_id);
		ret = usd_create_dest(udp->dom_dev, reqp->creq_ipaddr,
				reqp->creq_port, &ep->e.msg.ep_dest);
		if (ret != 0)
			goto free_entry_and_report_failure;

		ep->e.msg.ep_dest->ds_dest.ds_udp.u_hdr.uh_ip.frag_off |=
			htons(IP_DF);

		entry->fid = ep_utofid(ep);
		entry->info = NULL;
		memcpy(entry->data, reqp->creq_data, reqp->creq_datalen);
		ret = usdf_eq_write_internal(ep->ep_eq, FI_CONNECTED, entry,
				entry_len, 0);
		if (ret != (int)entry_len) {
			free(ep->e.msg.ep_dest);
			ep->e.msg.ep_dest = NULL;

			goto free_entry_and_report_failure;
		}

		free(entry);
		usdf_cm_msg_connreq_cleanup(crp);
	}
	return 0;

free_entry_and_report_failure:
	free(entry);
report_failure_skip_data:
	usdf_cm_report_failure(crp, ret, false);
	return 0;
}

/*
 * Write connection request data to the listener
 * Once everything is written, switch over into listening mode to
 * capture the listener response.
 */
static int
usdf_cm_msg_connect_cb_wr(void *v)
{
	struct usdf_connreq *crp;
	struct usdf_ep *ep;
	struct usdf_fabric *fp;
	struct epoll_event ev;
	int ret;

	crp = v;
	ep = crp->cr_ep;
	fp = ep->ep_domain->dom_fabric;

	ret = write(crp->cr_sockfd, crp->cr_ptr, crp->cr_resid);
	if (ret == -1) {
		usdf_cm_report_failure(crp, -errno, false);
		return 0;
	}

	crp->cr_resid -= ret;
	if (crp->cr_resid == 0) {
		crp->cr_pollitem.pi_rtn = usdf_cm_msg_connect_cb_rd;
		crp->cr_ptr = crp->cr_data;
		crp->cr_resid = sizeof(struct usdf_connreq_msg);

		ev.events = EPOLLIN;
		ev.data.ptr = &crp->cr_pollitem;
		ret = epoll_ctl(fp->fab_epollfd, EPOLL_CTL_MOD,
				crp->cr_sockfd, &ev);
		if (ret != 0) {
			usdf_cm_report_failure(crp, -errno, false);
			return 0;
		}
	}
	return 0;
}

int
usdf_cm_msg_connect(struct fid_ep *fep, const void *addr,
		const void *param, size_t paramlen)
{
	struct usdf_connreq *crp;
	struct usdf_ep *ep;
	struct usdf_rx *rx;
	struct usdf_domain *udp;
	const struct sockaddr_in *sin;
	struct epoll_event ev;
	struct usdf_fabric *fp;
	struct usdf_connreq_msg *reqp;
	struct usd_qp_impl *qp;
	struct fi_info *info;
	size_t request_size;
	int ret;

	USDF_TRACE_SYS(EP_CTRL, "\n");

	if (paramlen > USDF_MAX_CONN_DATA)
		return -FI_EINVAL;

	ep = ep_ftou(fep);
	udp = ep->ep_domain;
	fp = udp->dom_fabric;
	info = ep->ep_domain->dom_info;

	sin = usdf_format_to_sin(info, addr);

	/* Although paramlen may be less than USDF_MAX_CONN_DATA, the same crp
	 * struct is used for receiving the accept and reject payload. The
	 * structure has to be prepared to receive the maximum allowable amount
	 * of data per transfer. The maximum size includes the connection
	 * request structure, the connection request message, and the maximum
	 * amount of data per connection request message.
	 */
	request_size = sizeof(*crp) + sizeof(*reqp) + USDF_MAX_CONN_DATA;
	crp = calloc(1, request_size);
	if (crp == NULL) {
		ret = -errno;
		goto fail;
	}
	ep->e.msg.ep_connreq = crp;

	crp->handle.fclass = FI_CLASS_CONNREQ;

	if (ep->e.msg.ep_cm_sock == -1) {
		crp->cr_sockfd = socket(AF_INET, SOCK_STREAM, 0);
		if (crp->cr_sockfd == -1) {
			ret = -errno;
			goto fail;
		}
	} else {
		crp->cr_sockfd = ep->e.msg.ep_cm_sock;
		ep->e.msg.ep_cm_sock = -1;
	}

	ret = fi_fd_nonblock(crp->cr_sockfd);
	if (ret) {
		ret = -errno;
		goto fail;
	}

	ret = usdf_ep_msg_get_queues(ep);
	if (ret != 0) {
		goto fail;
	}
	rx = ep->ep_rx;
	qp = to_qpi(rx->rx_qp);

	ret = connect(crp->cr_sockfd, (struct sockaddr *)sin, sizeof(*sin));
	if (ret != 0 && errno != EINPROGRESS) {
		ret = -errno;
		goto fail;
	}

	/* If cr_sockfd was previously unbound, connect(2) will do a a bind(2)
	 * for us.  Update our snapshot of the locally bound address. */
	ret = usdf_msg_upd_lcl_addr(ep);
	if (ret)
		goto fail;

	/* allocate remote peer ID */
	ep->e.msg.ep_rem_peer_id = udp->dom_next_peer;
	udp->dom_peer_tab[udp->dom_next_peer] = ep;
	++udp->dom_next_peer;

	crp->cr_ep = ep;
	reqp = (struct usdf_connreq_msg *)crp->cr_data;
	crp->cr_ptr = crp->cr_data;
	crp->cr_resid =  sizeof(*reqp) + paramlen;

	reqp->creq_peer_id = htons(ep->e.msg.ep_rem_peer_id);
	reqp->creq_ipaddr = fp->fab_dev_attrs->uda_ipaddr_be;
	reqp->creq_port =
		qp->uq_attrs.uqa_local_addr.ul_addr.ul_udp.u_addr.sin_port;
	reqp->creq_datalen = htonl(paramlen);
	memcpy(reqp->creq_data, param, paramlen);

	/* register for notification when connect completes */
	crp->cr_pollitem.pi_rtn = usdf_cm_msg_connect_cb_wr;
	crp->cr_pollitem.pi_context = crp;
	ev.events = EPOLLOUT;
	ev.data.ptr = &crp->cr_pollitem;
	ret = epoll_ctl(fp->fab_epollfd, EPOLL_CTL_ADD, crp->cr_sockfd, &ev);
	if (ret != 0) {
		crp->cr_pollitem.pi_rtn = NULL;
		ret = -errno;
		goto fail;
	}

	usdf_free_sin_if_needed(info, (struct sockaddr_in *)sin);

	return 0;

fail:
	usdf_free_sin_if_needed(info, (struct sockaddr_in *)sin);

	if (crp != NULL) {
		if (crp->cr_sockfd != -1) {
			close(crp->cr_sockfd);
		}
		free(crp);
		ep->e.msg.ep_connreq = NULL;
	}
	usdf_ep_msg_release_queues(ep);
	return ret;
}

/* A wrapper to core function to translate string address to
 * sockaddr_in type. We are expecting a NULL sockaddr_in**.
 * The core function will allocated it for us. The caller HAS TO FREE it.
 */
int usdf_str_toaddr(const char *str, struct sockaddr_in **outaddr)
{
	uint32_t type;
	size_t size;
	int ret;

	type = FI_SOCKADDR_IN;

	/* call the core function. The core always allocate the addr for us. */
	ret = ofi_str_toaddr(str, &type, (void **)outaddr, &size);

#if ENABLE_DEBUG
	char outstr[USDF_ADDR_STR_LEN];
	size_t out_size = USDF_ADDR_STR_LEN;

	inet_ntop(AF_INET, &((*outaddr)->sin_addr), outstr, out_size);
	USDF_DBG_SYS(EP_CTRL,
		    "%s(string) converted to addr :%s:%u(inet)\n",
		    str, outstr, ntohs((*outaddr)->sin_port));
#endif

	return ret;
}

/* A wrapper to core function to translate sockaddr_in address to
 * string. This function is not allocating any memory. We are expected
 * an allocated buffer.
 */
const char *usdf_addr_tostr(const struct sockaddr_in *sin,
			    char *addr_str, size_t *size)
{
	const char *ret;

	ret = ofi_straddr(addr_str, size, FI_SOCKADDR_IN, sin);

#if ENABLE_DEBUG
	char outstr[USDF_ADDR_STR_LEN];
	size_t out_size = USDF_ADDR_STR_LEN;

	inet_ntop(AF_INET, &sin->sin_addr, outstr, out_size);
	USDF_DBG_SYS(EP_CTRL,
		    "%s:%d(inet) converted to %s(string)\n",
		    outstr, ntohs(sin->sin_port), addr_str);
#endif

	return ret;
}

/*
 * Return local address of an EP
 */
static int usdf_cm_copy_name(struct fi_info *info, struct sockaddr_in *sin,
		void *addr, size_t *addrlen)
{
	int ret;
	char addr_str[USDF_ADDR_STR_LEN];
	size_t len;

	USDF_TRACE_SYS(EP_CTRL, "\n");

	ret = FI_SUCCESS;
	switch (info->addr_format) {
	case FI_ADDR_STR:
		len = USDF_ADDR_STR_LEN;
		usdf_addr_tostr(sin, addr_str, &len);
		snprintf(addr, MIN(len, *addrlen), "%s", addr_str);
		break;
	case FI_SOCKADDR:
	case FI_SOCKADDR_IN:
		len = sizeof(*sin);
		memcpy(addr, sin, MIN(len, *addrlen));
		break;
	default:
		return -FI_EINVAL;
	}

	/* If the buffer is too small, tell the user. */
	if (*addrlen < len)
		ret = -FI_ETOOSMALL;

	/* Always return the actual size. */
	*addrlen = len;
	return ret;
}

int usdf_cm_rdm_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct usdf_ep *ep;
	struct usdf_rx *rx;
	struct sockaddr_in sin;
	struct fi_info *info;

	USDF_TRACE_SYS(EP_CTRL, "\n");

	ep = ep_fidtou(fid);
	rx = ep->ep_rx;
	info = ep->ep_domain->dom_info;

	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET;
	sin.sin_addr.s_addr =
		ep->ep_domain->dom_fabric->fab_dev_attrs->uda_ipaddr_be;
	if (rx == NULL || rx->rx_qp == NULL) {
		sin.sin_port = 0;
	} else {
		sin.sin_port = to_qpi(rx->rx_qp)->uq_attrs.uqa_local_addr.ul_addr.ul_udp.u_addr.sin_port;
	}

	return usdf_cm_copy_name(info, &sin, addr, addrlen);
}

int usdf_cm_dgram_getname(fid_t fid, void *addr, size_t *addrlen)
{
	int ret;
	struct usdf_ep *ep;
	struct sockaddr_in sin;
	struct fi_info *info;
	socklen_t slen;

	USDF_TRACE_SYS(EP_CTRL, "\n");

	ep = ep_fidtou(fid);
	info = ep->ep_domain->dom_info;

	memset(&sin, 0, sizeof(sin));
	if (ep->e.dg.ep_qp == NULL) {
		sin.sin_family = AF_INET;
		sin.sin_addr.s_addr =
			ep->ep_domain->dom_fabric->fab_dev_attrs->uda_ipaddr_be;
		sin.sin_port = 0;
	} else {
		slen = sizeof(sin);
		ret = getsockname(ep->e.dg.ep_sock, (struct sockaddr *)&sin, &slen);
		if (ret == -1) {
			return -errno;
		}
		assert(((struct sockaddr *)&sin)->sa_family == AF_INET);
		assert(slen == sizeof(sin));
		assert(sin.sin_addr.s_addr ==
			ep->ep_domain->dom_fabric->fab_dev_attrs->uda_ipaddr_be);
	}

	return usdf_cm_copy_name(info, &sin, addr, addrlen);
}

int usdf_cm_msg_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct usdf_ep *ep;
	struct fi_info *info;

	USDF_TRACE_SYS(EP_CTRL, "\n");

	ep = ep_fidtou(fid);
	info = ep->ep_domain->dom_info;

	return usdf_cm_copy_name(info, &ep->e.msg.ep_lcl_addr, addr, addrlen);
}

/* Checks that the given address is actually a sockaddr_in of appropriate
 * length.  "addr_format" is an FI_ constant like FI_SOCKADDR_IN indicating the
 * claimed type of the given address.
 *
 * Returns true if address is actually a sockaddr_in, false otherwise.
 *
 * Upon successful return, "addr" can be safely cast to either
 * "struct sockaddr_in *" or "struct sockaddr *".
 *
 * "addr" should not be NULL.
 */
bool usdf_cm_addr_is_valid_sin(void *addr, size_t addrlen, uint32_t addr_format)
{
	assert(addr != NULL);

	switch (addr_format) {
	case FI_SOCKADDR_IN:
	case FI_SOCKADDR:
		if (addrlen != sizeof(struct sockaddr_in)) {
			USDF_WARN("addrlen is incorrect\n");
			return false;
		}
		if (((struct sockaddr *)addr)->sa_family != AF_INET) {
			USDF_WARN("unknown/unsupported addr_format\n");
			return false;
		}
		return true;
	default:
		USDF_WARN("unknown/unsupported addr_format\n");
		return false;
	}
}
