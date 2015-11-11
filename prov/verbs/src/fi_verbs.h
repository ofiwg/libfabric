/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#ifndef FI_VERBS_H
#define FI_VERBS_H

#include "config.h"

#include <asm/types.h>
#include <errno.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <sys/epoll.h>

#include <infiniband/ib.h>
#include <infiniband/verbs.h>
#include <infiniband/driver.h>
#include <rdma/rdma_cma.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_errno.h>

#include "fi.h"
#include "fi_enosys.h"
#include <rdma/fi_log.h>
#include "prov.h"
#include "fi_list.h"
#include "fi_signal.h"


#define VERBS_PROV_NAME "verbs"
#define VERBS_PROV_VERS FI_VERSION(1,0)

#define VERBS_DBG(subsys, ...) FI_DBG(&fi_ibv_prov, subsys, __VA_ARGS__)
#define VERBS_INFO(subsys, ...) FI_INFO(&fi_ibv_prov, subsys, __VA_ARGS__)
#define VERBS_INFO_ERRNO(subsys, fn, errno) VERBS_INFO(subsys, fn ": %s(%d)\n",	\
		strerror(errno), errno)


#define VERBS_INJECT_FLAGS(ep, len, flags) (((flags & FI_INJECT) || \
		len <= ep->info->tx_attr->inject_size) ? IBV_SEND_INLINE : 0)
#define VERBS_INJECT(ep, len) VERBS_INJECT_FLAGS(ep, len, ep->info->tx_attr->op_flags)

#define VERBS_SELECTIVE_COMP(ep) (ep->ep_flags & FI_SELECTIVE_COMPLETION)
#define VERBS_COMP_FLAGS(ep, flags) ((!VERBS_SELECTIVE_COMP(ep) || \
		(flags & (FI_COMPLETION | FI_TRANSMIT_COMPLETE))) ? \
		IBV_SEND_SIGNALED : 0)
#define VERBS_COMP(ep) VERBS_COMP_FLAGS(ep, ep->info->tx_attr->op_flags)


extern struct fi_provider fi_ibv_prov;


struct fi_ibv_fabric {
	struct fid_fabric	fabric_fid;
};

int fi_ibv_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		  void *context);
int fi_ibv_find_fabric(const struct fi_fabric_attr *attr);

struct fi_ibv_eq_entry {
	struct dlist_entry	item;
	uint32_t		event;
	size_t			len;
	char 			eq_entry[0];
};

struct fi_ibv_eq {
	struct fid_eq		eq_fid;
	struct fi_ibv_fabric	*fab;
	fastlock_t		lock;
	struct dlistfd_head	list_head;
	struct rdma_event_channel *channel;
	uint64_t		flags;
	struct fi_eq_err_entry	err;
	int			epfd;
};

int fi_ibv_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		   struct fid_eq **eq, void *context);

struct fi_ibv_av {
	struct fid_av		av;	/* TODO: rename to av_fid */
	struct fi_ibv_domain	*domain;
	struct fi_ibv_rdm_ep	*ep;	/* TODO: check usage */
	int			type;	/* TODO: AV enum? */
	size_t			count;
};

int fi_ibv_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		   struct fid_av **av, void *context);

struct fi_ibv_pep {
	struct fid_pep		pep_fid;
	struct fi_ibv_eq	*eq;
	struct rdma_cm_id	*id;
	int			backlog;
	int			bound;
	size_t			src_addrlen;
};

struct fi_ops_cm *fi_ibv_pep_ops_cm(struct fi_ibv_pep *pep);

struct fi_ibv_domain {
	struct fid_domain	domain_fid;
	struct ibv_context	*verbs;
	struct ibv_pd		*pd;
	int			rdm;
};

struct fi_ibv_cq {
	struct fid_cq		cq_fid;
	struct fi_ibv_domain	*domain;
	struct ibv_comp_channel	*channel;
	struct ibv_cq		*cq;
	size_t			entry_size;
	uint64_t		flags;
	enum fi_cq_wait_cond	wait_cond;
	struct ibv_wc		wc;
	int			signal_fd[2];
	/* RDM EP fields - TODO: check usage */
	struct fi_ibv_rdm_ep	*ep;
	int			format;
};

int fi_ibv_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		   struct fid_cq **cq, void *context);
struct fi_ops_cq *fi_ibv_cq_ops_tagged(struct fi_ibv_cq *cq);


struct fi_ibv_mem_desc {
	struct fid_mr		mr_fid;
	struct ibv_mr		*mr;
	struct fi_ibv_domain	*domain;
};

struct fi_ibv_msg_ep {
	struct fid_ep		ep_fid;
	struct rdma_cm_id	*id;
	struct fi_ibv_eq	*eq;
	struct fi_ibv_cq	*rcq;
	struct fi_ibv_cq	*scq;
	uint64_t		ep_flags;
	struct fi_info		*info;
};

int fi_ibv_open_ep(struct fid_domain *domain, struct fi_info *info,
		   struct fid_ep **ep, void *context);
int fi_ibv_passive_ep(struct fid_fabric *fabric, struct fi_info *info,
		      struct fid_pep **pep, void *context);
int fi_ibv_open_rdm_ep(struct fid_domain *domain, struct fi_info *info,
			struct fid_ep **ep, void *context);
int fi_ibv_create_ep(const char *node, const char *service,
		     uint64_t flags, const struct fi_info *hints,
		     struct rdma_addrinfo **rai, struct rdma_cm_id **id);

struct fi_ops_atomic *fi_ibv_msg_ep_ops_atomic(struct fi_ibv_msg_ep *ep);
struct fi_ops_cm *fi_ibv_msg_ep_ops_cm(struct fi_ibv_msg_ep *ep);
struct fi_ops_msg *fi_ibv_msg_ep_ops_msg(struct fi_ibv_msg_ep *ep);
struct fi_ops_rma *fi_ibv_msg_ep_ops_rma(struct fi_ibv_msg_ep *ep);


struct fi_ibv_connreq {
	struct fid		handle;
	struct rdma_cm_id	*id;
};

int fi_ibv_sockaddr_len(struct sockaddr *addr);


int fi_ibv_init_info();
void fi_ibv_free_info();
int fi_ibv_getinfo(uint32_t version, const char *node, const char *service,
		   uint64_t flags, struct fi_info *hints, struct fi_info **info);
struct fi_info *fi_ibv_get_verbs_info(const char *domain_name);
void fi_ibv_update_info(const struct fi_info *hints, struct fi_info *info);
int fi_ibv_fi_to_rai(const struct fi_info *fi, uint64_t flags,
		     struct rdma_addrinfo *rai);

struct verbs_ep_domain {
	char			*suffix;
	enum fi_ep_type		type;
	uint64_t		caps;
};

extern const struct verbs_ep_domain verbs_rdm_domain;

int fi_ibv_check_fabric_attr(const struct fi_fabric_attr *attr,
			     const struct fi_info *info);
int fi_ibv_check_domain_attr(const struct fi_domain_attr *attr,
			     const struct fi_info *info);
int fi_ibv_check_ep_attr(const struct fi_ep_attr *attr,
			 const struct fi_info *info);
int fi_ibv_check_rx_attr(const struct fi_rx_attr *attr,
			 const struct fi_info *hints, const struct fi_info *info);
int fi_ibv_check_tx_attr(const struct fi_tx_attr *attr,
			 const struct fi_info *hints, const struct fi_info *info);


ssize_t fi_ibv_send(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr, size_t len,
		    int count, void *context);
ssize_t fi_ibv_send_buf(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
			const void *buf, size_t len, void *desc, void *context);
ssize_t fi_ibv_send_buf_inline(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
			       const void *buf, size_t len);
ssize_t fi_ibv_send_iov_flags(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
			      const struct iovec *iov, void **desc, int count,
			      void *context, uint64_t flags);

#define fi_ibv_set_sge(sge, buf, len, desc)				\
	do {								\
		sge.addr = (uintptr_t)buf;				\
		sge.length = (uint32_t)len;				\
		sge.lkey = (uint32_t)(uintptr_t)desc;			\
	} while (0)

#define fi_ibv_set_sge_iov(sg_list, iov, count, desc, len)		\
	do {								\
		int i;							\
		if (count) {						\
			sg_list = alloca(sizeof(*sg_list) * count);	\
			for (i = 0; i < count; i++) {			\
				fi_ibv_set_sge(sg_list[i],		\
						iov[i].iov_base,	\
						iov[i].iov_len,		\
						desc[i]);		\
				len += iov[i].iov_len;			\
			}						\
		}							\
	} while (0)

#define fi_ibv_set_sge_inline(sge, buf, len)				\
	do {								\
		sge.addr = (uintptr_t)buf;				\
		sge.length = (uint32_t)len;				\
	} while (0)

#define fi_ibv_set_sge_iov_inline(sg_list, iov, count, len)		\
	do {								\
		int i;							\
		if (count) {						\
			sg_list = alloca(sizeof(*sg_list) * count);	\
			for (i = 0; i < count; i++) {			\
				fi_ibv_set_sge_inline(sg_list[i],	\
						iov[i].iov_base,	\
						iov[i].iov_len);	\
				len += iov[i].iov_len;			\
			}						\
		}							\
	} while (0)

#define fi_ibv_send_iov(ep, wr, iov, desc, count, context)		\
	fi_ibv_send_iov_flags(ep, wr, iov, desc, count, context,	\
			ep->info->tx_attr->op_flags)

#define fi_ibv_send_msg(ep, wr, msg, flags)				\
	fi_ibv_send_iov_flags(ep, wr, msg->msg_iov, msg->desc,		\
			msg->iov_count,	msg->context, flags)

#endif /* FI_VERBS_H */
