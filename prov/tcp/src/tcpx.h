/*
 * Copyright (c) 2017 Intel Corporation, Inc.  All rights reserved.
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
#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <sys/types.h>
#include <netdb.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/ip.h>

#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_trigger.h>

#include <fi.h>
#include <fi_enosys.h>
#include <fi_rbuf.h>
#include <fi_list.h>
#include <fi_signal.h>
#include <fi_util.h>

#ifndef _TCP_H_
#define _TCP_H_

#define TCPX_MAJOR_VERSION 0
#define TCPX_MINOR_VERSION 1


extern struct fi_provider tcpx_prov;
extern struct util_prov tcpx_util_prov;
extern struct fi_info tcpx_info;

#define TCPX_IOV_LIMIT 4
#define TCPX_MAX_SOCK_REQS (1<<10)

#define TCPX_NO_COMPLETION (1ULL << 30)

#define TCPX_SOCK_ADD (1ULL << 0)
#define TCPX_SOCK_DEL (1ULL << 1)

int tcpx_create_fabric(struct fi_fabric_attr *attr,
		struct fid_fabric **fabric,
		void *context);

int tcpx_passive_ep(struct fid_fabric *fabric, struct fi_info *info,
		    struct fid_pep **pep, void *context);

int tcpx_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		     struct fid_domain **domain, void *context);


int tcpx_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep_fid, void *context);


int tcpx_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq_fid, void *context);

struct poll_fd_info {
	fid_t fid;
	int flags;
	struct fi_info *info;
	struct dlist_entry entry;
};

struct poll_fd_data {
	struct pollfd *poll_fds;
	struct poll_fd_info *fd_info;
	int nfds;
	int max_nfds;
};

struct tcpx_conn_handle {
	struct fid handle;
	SOCKET conn_fd;
};

struct tcpx_pep {
	struct util_pep util_pep;
	struct fi_info info;
	SOCKET sock;
	int sock_fd_closed;
};

struct tcpx_ep {
	struct util_ep util_ep;
	struct fi_info info;
	SOCKET conn_fd;
};

struct tcpx_fabric {
	struct util_fabric util_fabric;
	struct fd_signal signal;
	struct dlist_entry fd_list;
	fastlock_t fd_list_lock;
	pthread_t conn_mgr_thread;
	int run_cm_thread;
};

#endif //_TCP_H_
