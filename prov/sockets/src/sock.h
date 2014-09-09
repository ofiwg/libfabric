/*
 * Copyright (c) 2014 Intel Corporation, Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
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

//#include <errno.h>
//#include <fcntl.h>
//#include <netdb.h>
//#include <netinet/in.h>
//#include <netinet/tcp.h>
//#include <poll.h>
#include <pthread.h>
//#include <stdarg.h>
//#include <stddef.h>
//#include <stdio.h>
//#include <string.h>
//#include <sys/select.h>
//#include <sys/socket.h>
//#include <sys/types.h>
//#include <sys/time.h>
//#include <unistd.h>

#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_trigger.h>

#include "fi.h"
#include "indexer.h"

static const char const fab_name[] = "IP";
static const char const dom_name[] = "sockets";


struct sock_fabric {
	struct fid_fabric	fab_fid;
	uint64_t		flags;
};

struct sock_domain {
	struct fid_domain	dom_fid;
	struct sock_fabric	*fab;
	fastlock_t		lock;
	atomic_t		ref;
	struct index_map	mr_idm;
};

struct sock_cntr {
	struct fid_cntr		cntr_fid;
	struct sock_domain	*dom;
	uint64_t		value;
	uint64_t		threshold;
	pthread_cond_t		cond;
	pthread_mutex_t		mut;
};

struct sock_eq {
	struct fid_eq		eq_fid;
	struct sock_domain	*dom;
};

struct sock_eq_comp {
	struct sock_eq		eq;
};

struct sock_mr {
	struct fid_mr		mr_fid;
	struct sock_domain	*dom;
	uint64_t		access;
	uint64_t		offset;
	uint64_t		key;
	size_t			iov_count;
	struct iovec		mr_iov[1];
};

struct sock_av {
	struct fid_av		av_fid;
	struct sock_domain	*dom;
	atomic_t		ref;
	struct fi_av_attr	attr;
};

struct sock_poll {
	struct fid_poll		poll_fid;
	struct sock_domain	*dom;
};

struct sock_wait {
	struct fid_wait		wait_fid;
	struct sock_domain	*dom;
};

struct sock_ep {
	struct fid_ep		ep_fid;
	struct sock_domain	*dom;
};

int sock_rdm_getinfo(int version, const char *node, const char *service,
		uint64_t flags, struct fi_info *hints, struct fi_info **info);
int sock_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		struct fid_av **av, void *context);
int sock_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		struct fid_cntr **cntr, void *context);
int sock_domain(struct fid_fabric *fabric, struct fi_domain_attr *attr,
		struct fid_domain **dom, void *context);
int sock_eq_open(struct fid_domain *domain, struct fi_eq_attr *attr,
		struct fid_eq **eq, void *context);
int sock_rdm_ep(struct fid_domain *domain, struct fi_info *info,
		struct fid_ep **ep, void *context);
int sock_poll_open(struct fid_domain *domain, struct fi_poll_attr *attr,
		struct fid_poll **pollset);
int sock_wait_open(struct fid_domain *domain, struct fi_wait_attr *attr,
		struct fid_wait **waitset);

