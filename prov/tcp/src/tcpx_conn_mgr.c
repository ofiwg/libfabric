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

static int poll_fd_resize(struct poll_fd_mgr *poll_mgr, int size)
{
	struct pollfd *new_poll_fds;
	struct poll_fd_info *new_poll_info;

	new_poll_fds = calloc(size, sizeof(*new_poll_fds));
	if (!new_poll_fds)
		return -FI_ENOMEM;

	new_poll_info = calloc(size, sizeof(*new_poll_info));
	if (!new_poll_info) {
		free(new_poll_fds);
		return -FI_ENOMEM;
	}

	if (poll_mgr->max_nfds) {
		memcpy(new_poll_fds, poll_mgr->poll_fds,
		       poll_mgr->max_nfds * sizeof(*new_poll_fds));
		free(poll_mgr->poll_fds);

		memcpy(new_poll_info, poll_mgr->poll_info,
		       poll_mgr->max_nfds * sizeof(*new_poll_info));
		free(poll_mgr->poll_info);
	}

	poll_mgr->poll_fds = new_poll_fds;
	poll_mgr->poll_info = new_poll_info;
	poll_mgr->max_nfds = size;

	return 0;
}

static void poll_fds_swap_del_last(struct poll_fd_mgr *poll_mgr, int index)
{
	poll_mgr->poll_fds[index] = poll_mgr->poll_fds[(poll_mgr->nfds) - 1];
	poll_mgr->poll_info[index] = poll_mgr->poll_info[(poll_mgr->nfds) - 1];
	poll_mgr->nfds--;
}

static int poll_fds_find_dup(struct poll_fd_mgr *poll_mgr,
			     struct poll_fd_info *fd_info_entry)
{
	struct tcpx_ep *tcpx_ep;
	struct tcpx_pep *tcpx_pep;
	int i;

	for (i = 1 ; i < poll_mgr->nfds ; i++) {
		switch (fd_info_entry->fid->fclass) {
		case FI_CLASS_EP:
			tcpx_ep = container_of(fd_info_entry->fid, struct tcpx_ep,
					       util_ep.ep_fid.fid);
			if (poll_mgr->poll_fds[i].fd == tcpx_ep->conn_fd)
				return i;
			break;
		case FI_CLASS_PEP:
			tcpx_pep = container_of(fd_info_entry->fid, struct tcpx_pep,
						util_pep.pep_fid.fid);
			if (poll_mgr->poll_fds[i].fd == tcpx_pep->sock)
				return i;
			break;
		default:
			continue;
		}
	}
	return -1;
}

static int poll_fds_add_item(struct poll_fd_mgr *poll_mgr,
			     struct poll_fd_info *poll_info)
{
	struct tcpx_ep *tcpx_ep;
	struct tcpx_pep *tcpx_pep;
	struct tcpx_conn_handle *handle;
	int ret;

	if (poll_mgr->nfds >= poll_mgr->max_nfds) {
		ret = poll_fd_resize(poll_mgr, poll_mgr->max_nfds << 1);
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"memory allocation failed\n");
		return ret;
	}

	poll_mgr->poll_info[poll_mgr->nfds] = *poll_info;
	poll_mgr->poll_fds[poll_mgr->nfds].revents = 0;

	switch (poll_mgr->poll_info[poll_mgr->nfds].type) {
	case CONNECT_SOCK:
	case ACCEPT_SOCK:
		tcpx_ep = container_of(poll_info->fid, struct tcpx_ep,
				       util_ep.ep_fid.fid);
		poll_mgr->poll_fds[poll_mgr->nfds].fd = tcpx_ep->conn_fd;
		poll_mgr->poll_fds[poll_mgr->nfds].events = POLLOUT;
		break;
	case PASSIVE_SOCK:
		tcpx_pep = container_of(poll_info->fid, struct tcpx_pep,
					util_pep.pep_fid.fid);
		if (tcpx_pep->state == TCPX_PEP_CLOSED)
			return -FI_EINVAL;

		poll_mgr->poll_fds[poll_mgr->nfds].fd = tcpx_pep->sock;
		poll_mgr->poll_fds[poll_mgr->nfds].events = POLLIN;
		break;
	case CONNREQ_HANDLE:
		handle = container_of(poll_info->fid, struct tcpx_conn_handle,
				      handle);
		poll_mgr->poll_fds[poll_mgr->nfds].fd = handle->conn_fd;
		poll_mgr->poll_fds[poll_mgr->nfds].events = POLLIN;
		break;
	default:
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"invalid fd\n");
		return -FI_EINVAL;
	}
	poll_mgr->nfds++;
	return 0;
}

static int handle_poll_list(struct poll_fd_mgr *poll_mgr)
{
	struct poll_fd_info *poll_item;
	int ret = FI_SUCCESS;
	int id = 0;

	while (!dlist_empty(&poll_mgr->list)) {
		poll_item = container_of(poll_mgr->list.next,
					 struct poll_fd_info, entry);
		dlist_remove_init(&poll_item->entry);

		if (poll_item->flags & POLL_MGR_DEL) {
			id = poll_fds_find_dup(poll_mgr, poll_item);
			assert(id > 0);
			if (id <= 0) {
				return -FI_EINVAL;
			}

			poll_fds_swap_del_last(poll_mgr, id);
			poll_item->flags |= POLL_MGR_ACK;
		} else {
			assert(poll_fds_find_dup(poll_mgr, poll_item) < 0);
			ret = poll_fds_add_item(poll_mgr, poll_item);
			if (ret) {
				FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
					"Failed to add fd to event polling\n");
			}
		}

		if (poll_item->flags & POLL_MGR_FREE)
			free(poll_item);
		else
			poll_item->flags |= POLL_MGR_ACK;
	}
	return ret;
}

static int rx_cm_data(SOCKET fd, struct ofi_ctrl_hdr *hdr,
		      int type, struct poll_fd_info *poll_info)
{
	ssize_t ret;

	ret = ofi_recv_socket(fd, hdr,
			      sizeof(*hdr), MSG_WAITALL);
	if (ret != sizeof(*hdr))
		return -FI_EIO;

	if (hdr->type != type)
		return -FI_ECONNREFUSED;

	if (hdr->version != OFI_CTRL_VERSION)
		return -FI_ENOPROTOOPT;

	poll_info->cm_data_sz = ntohs(hdr->seg_size);
	if (poll_info->cm_data_sz) {
		if (poll_info->cm_data_sz > TCPX_MAX_CM_DATA_SIZE)
			return -FI_EINVAL;

		ret = ofi_recv_socket(fd, poll_info->cm_data,
				      poll_info->cm_data_sz, MSG_WAITALL);
		if ((size_t) ret != poll_info->cm_data_sz)
			return -FI_EIO;
	}
	return FI_SUCCESS;
}

static int tx_cm_data(SOCKET fd, uint8_t type, struct poll_fd_info *poll_info)
{
	struct ofi_ctrl_hdr hdr;
	ssize_t ret;

	memset(&hdr, 0, sizeof(hdr));
	hdr.version = OFI_CTRL_VERSION;
	hdr.type = type;
	hdr.seg_size = htons((uint16_t) poll_info->cm_data_sz);

	ret = ofi_send_socket(fd, &hdr, sizeof(hdr), MSG_NOSIGNAL);
	if (ret != sizeof(hdr))
		return -FI_EIO;

	if (poll_info->cm_data_sz) {
		ret = ofi_send_socket(fd, poll_info->cm_data,
				      poll_info->cm_data_sz, MSG_NOSIGNAL);
		if ((size_t) ret != poll_info->cm_data_sz)
			return -FI_EIO;
	}
	return FI_SUCCESS;
}

static int send_conn_req(struct poll_fd_mgr *poll_mgr,
			 struct poll_fd_info *poll_info,
			 struct tcpx_ep *ep,
			 int index)
{
	socklen_t len;
	int status, ret = FI_SUCCESS;

	assert(poll_mgr->poll_fds[index].revents & POLLOUT);

	len = sizeof(status);
	ret = getsockopt(ep->conn_fd, SOL_SOCKET, SO_ERROR, (char *) &status, &len);
	if (ret < 0 || status) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL, "connection failure\n");
		return (ret < 0)? -ofi_sockerr() : status;
	}

	ret = tx_cm_data(ep->conn_fd, ofi_ctrl_connreq, poll_info);
	return ret;
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
	if (ret)
		goto err;

	ret = tcpx_progress_ep_add(ep);
	if (ret)
		goto err;

	ep->cm_state = TCPX_EP_CONNECTED;
err:
	fastlock_release(&ep->lock);
	return ret;
}

static int proc_conn_resp(struct poll_fd_mgr *poll_mgr,
			  struct poll_fd_info *poll_info,
			  struct tcpx_ep *ep,
			  int index)
{
	struct ofi_ctrl_hdr conn_resp;
	struct fi_eq_cm_entry *cm_entry;
	ssize_t len;
	int ret = FI_SUCCESS;

	assert(poll_mgr->poll_fds[index].revents & POLLIN);
	ret = rx_cm_data(ep->conn_fd, &conn_resp, ofi_ctrl_connresp, poll_info);
	if (ret)
		return ret;

	cm_entry = calloc(1, sizeof(*cm_entry) + poll_info->cm_data_sz);
	if (!cm_entry) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL, "mem alloc failed\n");
		return -FI_ENOMEM;
	}

	cm_entry->fid = poll_info->fid;
	memcpy(cm_entry->data, poll_info->cm_data, poll_info->cm_data_sz);

	ret = tcpx_ep_msg_xfer_enable(ep);
	if (ret)
		goto err;

	len = fi_eq_write(&ep->util_ep.eq->eq_fid, FI_CONNECTED, cm_entry,
			  sizeof(*cm_entry) + poll_info->cm_data_sz, 0);
	if (len < 0) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL, "Error writing to EQ\n");
		ret = (int) len;
		goto err;
	}
err:
	free(cm_entry);
	return ret;
}

static void handle_connect(struct poll_fd_mgr *poll_mgr,
			   int index)
{
	struct tcpx_ep *ep;
	struct poll_fd_info *poll_info = &poll_mgr->poll_info[index];
	struct fi_eq_err_entry err_entry;
	int ret;

	assert(poll_info->fid->fclass == FI_CLASS_EP);
	ep = container_of(poll_info->fid, struct tcpx_ep, util_ep.ep_fid.fid);

	switch (poll_info->state) {
	case ESTABLISH_CONN:
		ret = send_conn_req(poll_mgr, poll_info, ep, index);
		if (ret)
			goto err;

		poll_info->state = RCV_RESP;
		poll_mgr->poll_fds[index].events = POLLIN;
		FI_DBG(&tcpx_prov, FI_LOG_EP_CTRL, "Sent Connreq\n");
		break;
	case RCV_RESP:
		ret = proc_conn_resp(poll_mgr, poll_info, ep, index);
		if (ret)
			goto err;

		FI_DBG(&tcpx_prov, FI_LOG_EP_CTRL, "Received accept from server\n");
		poll_info->state = CONNECT_DONE;
		break;
	default:
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL, "Invalid connection state\n");
		ret = -FI_EINVAL;
		goto err;
	}
	return;
err:
	memset(&err_entry, 0, sizeof err_entry);
	err_entry.fid = poll_info->fid;
	err_entry.context = poll_info->fid->context;
	err_entry.err = -ret;

	poll_info->state = CONNECT_DONE;
	fi_eq_write(&ep->util_ep.eq->eq_fid, FI_NOTIFY,
		    &err_entry, sizeof(err_entry), UTIL_FLAG_ERROR);
}

static void handle_connreq(struct poll_fd_mgr *poll_mgr,
			   struct poll_fd_info *poll_info)
{
	struct tcpx_conn_handle *handle;
	struct poll_fd_info *fd_info;
	struct tcpx_pep *pep;
	SOCKET sock;

	assert(poll_info->fid->fclass == FI_CLASS_PEP);
	FI_DBG(&tcpx_prov, FI_LOG_EP_CTRL, "Received Connreq\n");
	pep = container_of(poll_info->fid, struct tcpx_pep, util_pep.pep_fid.fid);

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

	fd_info = calloc(1, sizeof(*fd_info));
	if (!fd_info) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"cannot allocate memory \n");
		goto err2;
	}

	handle->conn_fd = sock;
	handle->handle.fclass = FI_CLASS_CONNREQ;
	handle->pep = pep;
	fd_info->fid = &handle->handle;
	fd_info->flags = POLL_MGR_FREE;
	fd_info->type = CONNREQ_HANDLE;

	fastlock_acquire(&poll_mgr->lock);
	dlist_insert_tail(&fd_info->entry, &poll_mgr->list);
	fd_signal_set(&poll_mgr->signal);
	fastlock_release(&poll_mgr->lock);
	return;
err2:
	free(handle);
err1:
	ofi_close_socket(sock);
}

static void handle_conn_handle(struct poll_fd_mgr *poll_mgr,
			   struct poll_fd_info *poll_info)
{
	struct tcpx_conn_handle *handle;
	struct fi_eq_cm_entry *cm_entry;
	struct ofi_ctrl_hdr conn_req;
	int ret;

	assert(poll_info->fid->fclass == FI_CLASS_CONNREQ);
	handle  = container_of(poll_info->fid,
			       struct tcpx_conn_handle,
			       handle);

	ret = rx_cm_data(handle->conn_fd, &conn_req, ofi_ctrl_connreq, poll_info);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL, "cm data recv failed \n");
		goto err1;
	}

	cm_entry = calloc(1, sizeof(*cm_entry) + poll_info->cm_data_sz);
	if (!cm_entry)
		goto err1;

	cm_entry->fid = &handle->pep->util_pep.pep_fid.fid;
	cm_entry->info = fi_dupinfo(&handle->pep->info);
	if (!cm_entry->info)
		goto err2;

	cm_entry->info->handle = &handle->handle;
	memcpy(cm_entry->data, poll_info->cm_data, poll_info->cm_data_sz);

	ret = (int) fi_eq_write(&handle->pep->util_pep.eq->eq_fid, FI_CONNREQ, cm_entry,
				sizeof(*cm_entry) + poll_info->cm_data_sz, 0);
	if (ret < 0) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL, "Error writing to EQ\n");
		goto err3;
	}
	free(cm_entry);
	return;
err3:
	fi_freeinfo(cm_entry->info);
err2:
	free(cm_entry);
err1:
	ofi_close_socket(handle->conn_fd);
	free(handle);
}

static void handle_accept_conn(struct poll_fd_mgr *poll_mgr,
			       struct poll_fd_info *poll_info)
{
	struct fi_eq_cm_entry cm_entry = {0};
	struct fi_eq_err_entry err_entry;
	struct tcpx_ep *ep;
	int ret;

	assert(poll_info->fid->fclass == FI_CLASS_EP);
	ep = container_of(poll_info->fid, struct tcpx_ep, util_ep.ep_fid.fid);

	ret = tx_cm_data(ep->conn_fd, ofi_ctrl_connresp, poll_info);
	if (ret)
		goto err;

	cm_entry.fid =  poll_info->fid;

	ret = tcpx_ep_msg_xfer_enable(ep);
	if (ret)
		goto err;

	ret = (int) fi_eq_write(&ep->util_ep.eq->eq_fid, FI_CONNECTED,
				&cm_entry, sizeof(cm_entry), 0);
	if (ret < 0) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL, "Error writing to EQ\n");
	}
	FI_DBG(&tcpx_prov, FI_LOG_EP_CTRL, "Accepted Connection\n");
	return;
err:
	memset(&err_entry, 0, sizeof err_entry);
	err_entry.fid = poll_info->fid;
	err_entry.context = poll_info->fid->context;
	err_entry.err = ret;

	fi_eq_write(&ep->util_ep.eq->eq_fid, FI_NOTIFY,
		    &err_entry, sizeof(err_entry), UTIL_FLAG_ERROR);
}

static void handle_fd_events(struct poll_fd_mgr *poll_mgr)
{
	int i;

	/* Process the fd array from end to start.  This allows us to handle
	 * removing entries from the array. Also ignore the signal fd at index 0.
	 */
	for (i = poll_mgr->nfds-1; i > 0; i--) {
		if (!poll_mgr->poll_fds[i].revents)
			continue;

		switch (poll_mgr->poll_info[i].type) {
		case CONNECT_SOCK:
			handle_connect(poll_mgr, i);

			if (poll_mgr->poll_info[i].state == CONNECT_DONE)
				poll_fds_swap_del_last(poll_mgr, i);
			break;
		case PASSIVE_SOCK:
			handle_connreq(poll_mgr, &poll_mgr->poll_info[i]);
			break;
		case ACCEPT_SOCK:
			handle_accept_conn(poll_mgr, &poll_mgr->poll_info[i]);
			poll_fds_swap_del_last(poll_mgr, i);
			break;
		case CONNREQ_HANDLE:
			handle_conn_handle(poll_mgr, &poll_mgr->poll_info[i]);
			poll_fds_swap_del_last(poll_mgr, i);
			break;
		default:
			FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
				"should never end up here\n");
		}
	}
}

static void *tcpx_conn_mgr_thread(void *data)
{
	struct tcpx_fabric *tcpx_fabric = (struct tcpx_fabric *) data;
	struct poll_fd_mgr *poll_mgr = &tcpx_fabric->poll_mgr;
	int ret;

	ret = poll_fd_resize(poll_mgr, 64);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"poll_fd memory alloc failed\n");
		return NULL;
	}

	poll_mgr->poll_fds[0].fd = poll_mgr->signal.fd[FI_READ_FD];
	poll_mgr->poll_fds[0].events = POLLIN;
	poll_mgr->nfds = 1;

	while (poll_mgr->run) {
		ret = poll(poll_mgr->poll_fds, poll_mgr->nfds, -1);
		if (ret < 0) {
			FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
				"Poll failed\n");
			break;
		}

		if (poll_mgr->poll_fds[0].revents & POLLIN) {
			fastlock_acquire(&tcpx_fabric->poll_mgr.lock);
			fd_signal_reset(&poll_mgr->signal);
			if (handle_poll_list(poll_mgr)) {
				FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
					"fd list add or remove failed\n");
			}
			fastlock_release(&tcpx_fabric->poll_mgr.lock);
		}
		handle_fd_events(poll_mgr);
	}
	return NULL;
}

void tcpx_conn_mgr_close(struct tcpx_fabric *tcpx_fabric)
{
	struct poll_fd_info *poll_info;

	tcpx_fabric->poll_mgr.run = 0;
	fastlock_acquire(&tcpx_fabric->poll_mgr.lock);
	fd_signal_set(&tcpx_fabric->poll_mgr.signal);
	fastlock_release(&tcpx_fabric->poll_mgr.lock);

	if (tcpx_fabric->conn_mgr_thread &&
	    pthread_join(tcpx_fabric->conn_mgr_thread, NULL)) {
		FI_DBG(&tcpx_prov, FI_LOG_FABRIC,
		       "cm thread failed to join\n");
	}

	while (!dlist_empty(&tcpx_fabric->poll_mgr.list)) {
		poll_info = container_of(tcpx_fabric->poll_mgr.list.next,
					 struct poll_fd_info, entry);
		dlist_remove(&poll_info->entry);
		assert(poll_info->flags & POLL_MGR_FREE);
		free(poll_info);
	}

	fastlock_destroy(&tcpx_fabric->poll_mgr.lock);
	fd_signal_free(&tcpx_fabric->poll_mgr.signal);
}

int tcpx_conn_mgr_init(struct tcpx_fabric *tcpx_fabric)
{
	int ret;

	dlist_init(&tcpx_fabric->poll_mgr.list);
	fastlock_init(&tcpx_fabric->poll_mgr.lock);
	ret = fd_signal_init(&tcpx_fabric->poll_mgr.signal);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_FABRIC,"signal init failed\n");
		goto err;
	}

	tcpx_fabric->poll_mgr.run = 1;
	ret = pthread_create(&tcpx_fabric->conn_mgr_thread, 0,
			     tcpx_conn_mgr_thread, (void *) tcpx_fabric);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_FABRIC,
			"Failed creating tcpx connection manager thread");

		goto err1;
	}
	return 0;
err1:
	fd_signal_free(&tcpx_fabric->poll_mgr.signal);
err:
	fastlock_destroy(&tcpx_fabric->poll_mgr.lock);
	return ret;
}
