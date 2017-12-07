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

#include <prov.h>
#include "tcpx.h"
#include <poll.h>
#include <sys/types.h>
#include <fi_util.h>

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
	int ret;

	if (poll_mgr->nfds >= poll_mgr->max_nfds) {
		ret = poll_fd_resize(poll_mgr, poll_mgr->max_nfds << 1);
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"memory allocation failed\n");
		return ret;
	}

	poll_mgr->poll_info[poll_mgr->nfds] = *poll_info;

	switch (poll_info->fid->fclass) {
	case FI_CLASS_EP:
		tcpx_ep = container_of(poll_info->fid, struct tcpx_ep,
				       util_ep.ep_fid.fid);
		poll_mgr->poll_fds[poll_mgr->nfds].fd = tcpx_ep->conn_fd;
		poll_mgr->poll_fds[poll_mgr->nfds].events = POLLOUT;
		break;
	case FI_CLASS_PEP:
		tcpx_pep = container_of(poll_info->fid, struct tcpx_pep,
				       util_pep.pep_fid.fid);

		poll_mgr->poll_fds[poll_mgr->nfds].fd = tcpx_pep->sock;
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

	fastlock_acquire(&poll_mgr->lock);
	while (!dlist_empty(&poll_mgr->list)) {
		poll_item = container_of(poll_mgr->list.next,
					 struct poll_fd_info, entry);
		dlist_remove_init(&poll_item->entry);

		if (poll_item->flags & POLL_MGR_DEL) {
			ret = poll_fds_find_dup(poll_mgr, poll_item);
			assert(ret > 0);
			poll_fds_swap_del_last(poll_mgr, ret);
			poll_item->flags |= POLL_MGR_ACK;
		} else {
			assert(poll_fds_find_dup(poll_mgr, poll_item) < 0);
			ret = poll_fds_add_item(poll_mgr, poll_item);
			if (ret)
				FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
					"Failed to add fd to event polling\n");
		}

		if (poll_item->flags & POLL_MGR_FREE)
			free(poll_item);
		else
			poll_item->flags |= POLL_MGR_ACK;
	}
	fastlock_release(&poll_mgr->lock);
	return ret;
}

static void handle_connect(struct poll_fd_mgr *poll_mgr,
			   struct poll_fd_info *poll_info)
{
	struct tcpx_ep *ep;
	struct fi_eq_cm_entry cm_entry;
	struct fi_eq_err_entry err_entry;
	socklen_t len;
	int ret, status;

	assert(poll_info->fid->fclass == FI_CLASS_EP);
	ep = container_of(poll_info->fid, struct tcpx_ep, util_ep.ep_fid.fid);

	len = sizeof(status);
	ret = getsockopt(ep->conn_fd, SOL_SOCKET, SO_ERROR, &status, &len);
	if (ret < 0 || status) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL, "connection failure\n");

		memset(&err_entry, 0, sizeof err_entry);
		err_entry.fid = poll_info->fid;
		err_entry.context = poll_info->fid->context;
		err_entry.err = status ? status : errno;

		ret = fi_eq_write(&ep->util_ep.eq->eq_fid, FI_SHUTDOWN,
				  &err_entry, sizeof(err_entry), UTIL_FLAG_ERROR);
	} else {
		memset(&cm_entry, 0, sizeof cm_entry);
		cm_entry.fid = poll_info->fid;

		ret = fi_eq_write(&ep->util_ep.eq->eq_fid, FI_CONNECTED,
				  &cm_entry, sizeof(cm_entry), 0);
	}

	if (ret < 0)
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL, "Error writing to EQ\n");
}

static void handle_connreq(struct poll_fd_mgr *poll_mgr,
			   struct poll_fd_info *poll_info)
{
	struct tcpx_conn_handle *handle;
	struct tcpx_pep *pep;
	struct fi_eq_cm_entry cm_entry;
	SOCKET sock;
	int ret;

	assert(poll_info->fid->fclass == FI_CLASS_PEP);
	pep = container_of(poll_info->fid, struct tcpx_pep, util_pep.pep_fid.fid);

	sock = accept(pep->sock, NULL, 0);
	if (sock < 0) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL, "accept error: %d\n",
			ofi_sockerr());
		return;
	}

	handle = calloc(1, sizeof(*handle));
	if (!handle)
		goto err1;

	handle->conn_fd = sock;
	cm_entry.fid = poll_info->fid;
	cm_entry.info = fi_dupinfo(&pep->info);
	if (!cm_entry.info)
		goto err2;

	cm_entry.info->handle = &handle->handle;

	ret = fi_eq_write(&pep->util_pep.eq->eq_fid, FI_CONNREQ,
			  &cm_entry, sizeof(cm_entry), 0);
	if (ret < 0) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL, "Error writing to EQ\n");
		goto err3;
	}
	return;
err3:
	fi_freeinfo(cm_entry.info);
err2:
	free(handle);
err1:
	ofi_close_socket(sock);

}

static void handle_fd_events(struct poll_fd_mgr *poll_mgr)
{
	int i;

	/* Process the fd array from end to start.  This allows us to handle
	 * removing entries from the array.
	 */
	for (i = poll_mgr->nfds; i > 0; i--) {
		switch (poll_mgr->poll_fds[i].revents) {
		case POLLOUT:
			handle_connect(poll_mgr, &poll_mgr->poll_info[i]);
			poll_fds_swap_del_last(poll_mgr, i);
			break;
		case POLLIN:
			handle_connreq(poll_mgr, &poll_mgr->poll_info[i]);
			break;
		case POLLERR:
		case POLLHUP:
		case POLLNVAL:
			FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
				"polling error on socket\n");
			/* TODO: change state to error or handle somehow */
			poll_fds_swap_del_last(poll_mgr, i);
			break;
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
			fd_signal_reset(&poll_mgr->signal);
			if (handle_poll_list(poll_mgr)) {
				FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
					"fd list add or remove failed\n");
			}
		}

		handle_fd_events(poll_mgr);
	}

	return NULL;
}

void tcpx_conn_mgr_close(struct tcpx_fabric *tcpx_fabric)
{
	struct poll_fd_info *poll_info;

	tcpx_fabric->poll_mgr.run = 0;
	fd_signal_set(&tcpx_fabric->poll_mgr.signal);

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

	fd_signal_free(&tcpx_fabric->poll_mgr.signal);
	fastlock_destroy(&tcpx_fabric->poll_mgr.lock);
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
