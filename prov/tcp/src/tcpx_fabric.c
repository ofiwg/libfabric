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
#include <ifaddrs.h>
#include <net/if.h>
#include <fi_util.h>

struct fi_ops_fabric tcpx_fabric_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = tcpx_domain_open,
	.passive_ep = tcpx_passive_ep,
	.eq_open = ofi_eq_create,
	.wait_open = ofi_wait_fd_open,
	.trywait = ofi_trywait
};

static int tcpx_fabric_close(fid_t fid)
{
	int ret;
	struct tcpx_fabric *tcpx_fabric;
	struct dlist_entry *fd_entry;
	struct poll_fd_info *fd_info_entry;

	tcpx_fabric = container_of(fid, struct tcpx_fabric,
			      util_fabric.fabric_fid.fid);

	ret = ofi_fabric_close(&tcpx_fabric->util_fabric);
	if (ret)
		return ret;

	tcpx_fabric->run_cm_thread = 0;
	fd_signal_set(&tcpx_fabric->signal);

	if(tcpx_fabric->conn_mgr_thread &&
	   pthread_join(tcpx_fabric->conn_mgr_thread, NULL))
		FI_DBG(&tcpx_prov, FI_LOG_FABRIC,
		       "cm thread failed to join\n");
	while (!dlist_empty(&tcpx_fabric->fd_list)) {
		fd_entry = tcpx_fabric->fd_list.next;
		dlist_remove(&tcpx_fabric->fd_list);
		fd_info_entry = (container_of(fd_entry,
					struct poll_fd_info,
					entry));
		free(fd_info_entry);
	}

	fastlock_destroy(&tcpx_fabric->fd_list_lock);
	fd_signal_free(&tcpx_fabric->signal);

	free(tcpx_fabric);
	return 0;
}

struct fi_ops tcpx_fabric_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = tcpx_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int poll_fd_data_alloc(struct poll_fd_data *pf_data, int size)
{
	struct pollfd *new_poll_fds;
	struct poll_fd_info *new_fd_info;

	new_poll_fds = calloc(size,
			      sizeof(*new_poll_fds));
	new_fd_info = calloc(size,
			     sizeof(*new_fd_info));
	if (!new_poll_fds || !new_fd_info)
		return -FI_ENOMEM;

	pf_data->max_nfds = size;
	memcpy(new_poll_fds, pf_data->poll_fds,
	       pf_data->max_nfds*sizeof(*new_poll_fds));
	free(pf_data->poll_fds);
	pf_data->poll_fds = new_poll_fds;

	memcpy(new_fd_info, pf_data->fd_info,
	       pf_data->max_nfds*sizeof(*new_fd_info));
	free(pf_data->fd_info);
	pf_data->fd_info = new_fd_info;

	return 0;
}

static void poll_fds_swap_del_last(int index,
				struct poll_fd_data *pf_data)
{
	pf_data->poll_fds[index] = pf_data->poll_fds[(pf_data->nfds)-1];
	pf_data->fd_info[index] = pf_data->fd_info[(pf_data->nfds)-1];
	pf_data->nfds--;
}

static int poll_fds_find_dup(struct poll_fd_data *pf_data,
			      struct poll_fd_info *fd_info_entry)
{
	struct tcpx_ep *tcpx_ep;
	struct tcpx_pep *tcpx_pep;
	int i;

	for (i = 1 ; i < pf_data->nfds ; i++) {

		switch (fd_info_entry->fid->fclass) {
		case FI_CLASS_EP:
			tcpx_ep = container_of(fd_info_entry->fid, struct tcpx_ep,
					       util_ep.ep_fid.fid);
			if (pf_data->poll_fds[i].fd == tcpx_ep->conn_fd)
				return i;
			break;
		case FI_CLASS_PEP:
			tcpx_pep = container_of(fd_info_entry->fid, struct tcpx_pep,
						util_pep.pep_fid.fid);
			if (pf_data->poll_fds[i].fd == tcpx_pep->sock)
				return i;
			break;
		default:
			continue;
		}
	}
	return -1;
}

static int poll_fds_add_item(struct poll_fd_data *pf_data,
			      struct poll_fd_info *fd_info_entry)
{
	struct tcpx_ep *tcpx_ep;
	struct tcpx_pep *tcpx_pep;
	int ret = FI_SUCCESS;

	if (pf_data->nfds >= pf_data->max_nfds) {
		ret = poll_fd_data_alloc(pf_data, pf_data->max_nfds*2);
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"memory allocation failed\n");
		goto out;
	}

	pf_data->fd_info[pf_data->nfds] = *fd_info_entry;

	switch (fd_info_entry->fid->fclass) {
	case FI_CLASS_EP:
		tcpx_ep = container_of(fd_info_entry->fid, struct tcpx_ep,
				       util_ep.ep_fid.fid);
		pf_data->poll_fds[pf_data->nfds].fd = tcpx_ep->conn_fd;
		pf_data->poll_fds[pf_data->nfds].events = POLLOUT;
		break;
	case FI_CLASS_PEP:
		tcpx_pep = container_of(fd_info_entry->fid, struct tcpx_pep,
				       util_pep.pep_fid.fid);

		pf_data->poll_fds[pf_data->nfds].fd = tcpx_pep->sock;
		pf_data->poll_fds[pf_data->nfds].events = POLLIN;
		break;
	default:
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"invalid fd\n");
		ret = -FI_EINVAL;
	}
	pf_data->nfds++;
out:
	return ret;
}

static int handle_fd_list(struct tcpx_fabric *tcpx_fabric,
			   struct poll_fd_data *pf_data)
{
	int ret = FI_SUCCESS;
	struct dlist_entry *fd_entry;
	struct poll_fd_info *fd_info_entry;

	fastlock_acquire(&tcpx_fabric->fd_list_lock);
	while (!dlist_empty(&tcpx_fabric->fd_list)) {
		fd_entry = tcpx_fabric->fd_list.next;
		fd_info_entry = (container_of(fd_entry,
					struct poll_fd_info,
					entry));

		ret = poll_fds_find_dup(pf_data, fd_info_entry);
		if (ret >= 0) {
			if (fd_info_entry->flags & TCPX_SOCK_DEL) {
				poll_fds_swap_del_last(ret, pf_data);
			}
			dlist_remove(&tcpx_fabric->fd_list);
			free(fd_info_entry);
			continue;
		}

		ret = poll_fds_add_item(pf_data, fd_info_entry);
		if (ret)
			goto out;

		dlist_remove(&tcpx_fabric->fd_list);
		free(fd_info_entry);
	}
out:
	fastlock_release(&tcpx_fabric->fd_list_lock);
	return ret;
}

static void handle_fd_events(struct tcpx_fabric *fabric,
			     struct poll_fd_data *pf_data)
{
	int iter, status, ret;
	socklen_t len;
	struct tcpx_conn_handle *handle;
	struct util_ep *util_ep;
	struct tcpx_pep *tcpx_pep;
	struct fi_eq_cm_entry eq_entry;

	iter = pf_data->nfds;
	while (iter--) {

		/* skip processing socket pair Read FD*/
		if (0 == iter)
			continue;
		switch (pf_data->poll_fds[iter].revents) {

		case POLLOUT:
			assert(pf_data->fd_info[iter].fid->fclass == FI_CLASS_EP);
			util_ep = container_of(pf_data->fd_info[iter].fid,
					       struct util_ep,
					       ep_fid.fid);

			len = sizeof(status);
			ret = getsockopt(pf_data->poll_fds[iter].fd, SOL_SOCKET,
					 SO_ERROR, &status, &len);
			if (ret < 0) {
				eq_entry.fid = pf_data->fd_info[iter].fid;

				/* report the conn req to the associated eq */
				if (fi_eq_write(&util_ep->eq->eq_fid,
						FI_CONNECTED, &eq_entry,
						sizeof(eq_entry),
						UTIL_FLAG_ERROR) < 0) {
					FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
						"Error in writing to EQ\n");
				}
				FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
					"getsockopt failed\n");
			} else {
				eq_entry.fid = pf_data->fd_info[iter].fid;

				/* report the conn req to the associated eq */
				if (fi_eq_write(&util_ep->eq->eq_fid,
						FI_CONNECTED, &eq_entry,
						sizeof(eq_entry), 0) < 0) {
					FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
						"Error in writing to EQ\n");
				}
			}
			poll_fds_swap_del_last(iter, pf_data);
			break;

		case POLLIN:

			assert(pf_data->fd_info[iter].fid->fclass == FI_CLASS_PEP);

			tcpx_pep = container_of(pf_data->fd_info[iter].fid,
						struct tcpx_pep,
						util_pep.pep_fid.fid);

			handle = calloc(1, sizeof(*handle));
			if (!handle) {
				FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
					"cannot allocate memory\n");
				continue;
			}

			handle->conn_fd = accept(pf_data->poll_fds[iter].fd, NULL, 0);
			if (handle->conn_fd < 0) {
				FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
					"failed to accept: %d\n", errno);
				continue;
			}

			eq_entry.fid = pf_data->fd_info[iter].fid;
			eq_entry.info = &tcpx_pep->info;
			eq_entry.info->handle = &handle->handle;

			/* report the conn req to the associated eq */
			if (fi_eq_write(&tcpx_pep->util_pep.eq->eq_fid, FI_CONNREQ,
					&eq_entry, sizeof(eq_entry), 0) < 0) {

				FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
					"Error in writing to EQ\n");
				ofi_close_socket(handle->conn_fd);
				free(handle);
			}
			break;

		case POLLERR:
		case POLLHUP:
		case POLLNVAL:
			poll_fds_swap_del_last(iter, pf_data);
			break;
		}

	}
}

static void *tcpx_conn_mgr_thread(void *data)
{
	struct tcpx_fabric *tcpx_fabric = (struct tcpx_fabric *) data;
	struct poll_fd_data pf_data;
	int ret;

	ret = poll_fd_data_alloc(&pf_data, 64);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"poll_fd memory alloc failed\n");
		return NULL;
	}

	pf_data.poll_fds[0].fd = tcpx_fabric->signal.fd[FI_READ_FD];
	pf_data.poll_fds[0].events = POLLIN;
	pf_data.nfds = 1;

	while (tcpx_fabric->run_cm_thread) {
		ret = poll(pf_data.poll_fds, pf_data.nfds, -1);
		if (ret < 0) {
			FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
				"Poll failed\n");
			break;
		}

		if (pf_data.poll_fds[0].revents & POLLIN) {
			fd_signal_reset(&tcpx_fabric->signal);
			if (handle_fd_list(tcpx_fabric, &pf_data)) {
				FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
					"fd list add or remove failed\n");
				goto out;
			}
		}
		handle_fd_events(tcpx_fabric, &pf_data);
	}
out:
	return NULL;
}

int tcpx_create_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		void *context)
{
	int ret;
	struct tcpx_fabric *tcpx_fabric;

	tcpx_fabric = calloc(1, sizeof(*tcpx_fabric));
	if (!tcpx_fabric)
		return -FI_ENOMEM;

	ret = ofi_fabric_init(&tcpx_prov, tcpx_info.fabric_attr, attr,
			      &tcpx_fabric->util_fabric, context);
	if (ret)
		goto err;

	*fabric = &tcpx_fabric->util_fabric.fabric_fid;
	(*fabric)->fid.ops = &tcpx_fabric_fi_ops;
	(*fabric)->ops = &tcpx_fabric_ops;

	fastlock_init(&tcpx_fabric->fd_list_lock);
	ret = fd_signal_init(&tcpx_fabric->signal);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_FABRIC,"signal init failed\n");
		goto err1;
	}

	tcpx_fabric->run_cm_thread = 1;
	ret = pthread_create(&tcpx_fabric->conn_mgr_thread, 0,
			     tcpx_conn_mgr_thread, (void *)tcpx_fabric);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_FABRIC,
			"Failed creating tcpx connection manager thread");

		goto err1;
	}

	return 0;
err1:
	fd_signal_free(&tcpx_fabric->signal);
	fastlock_destroy(&tcpx_fabric->fd_list_lock);
	ofi_fabric_close(&tcpx_fabric->util_fabric);
err:
	free(tcpx_fabric);
	return ret;
}
