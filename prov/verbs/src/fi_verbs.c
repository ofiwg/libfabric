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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

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


static int fi_ibv_getinfo(uint32_t version, const char *node, const char *service,
			  uint64_t flags, struct fi_info *hints, struct fi_info **info);
static int fi_ibv_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
			 void *context);
static void fi_ibv_fini(void);

#define VERBS_PROV_NAME "verbs"
#define VERBS_PROV_VERS FI_VERSION(1,0)

static struct fi_provider fi_ibv_prov = {
	.name = VERBS_PROV_NAME,
	.version = VERBS_PROV_VERS,
	.fi_version = FI_VERSION(1, 1),
	.getinfo = fi_ibv_getinfo,
	.fabric = fi_ibv_fabric,
	.cleanup = fi_ibv_fini
};

#define VERBS_IB_PREFIX "IB-0x"
#define VERBS_IWARP_FABRIC "Ethernet-iWARP"
#define VERBS_ANY_FABRIC "Any RDMA fabric"
#define VERBS_CM_DATA_SIZE 56
#define VERBS_RESOLVE_TIMEOUT 2000	// ms

#define VERBS_CAPS (FI_MSG | FI_RMA | FI_ATOMICS | FI_READ | FI_WRITE | \
		FI_SEND | FI_RECV | FI_REMOTE_READ | FI_REMOTE_WRITE)
#define VERBS_MODE (FI_LOCAL_MR)
#define VERBS_TX_OP_FLAGS (FI_INJECT | FI_COMPLETION | FI_TRANSMIT_COMPLETE)
#define VERBS_TX_OP_FLAGS_IWARP (FI_INJECT | FI_COMPLETION)
#define VERBS_TX_MODE VERBS_MODE
#define VERBS_RX_MODE (FI_LOCAL_MR | FI_RX_CQ_DATA)
#define VERBS_MSG_ORDER (FI_ORDER_RAR | FI_ORDER_RAW | FI_ORDER_RAS | \
		FI_ORDER_WAW | FI_ORDER_WAS | FI_ORDER_SAW | FI_ORDER_SAS )

#define VERBS_INJECT_FLAGS(ep, len, flags) ((flags & FI_INJECT) || \
		len <= ep->info->tx_attr->inject_size)
#define VERBS_INJECT(ep, len) VERBS_INJECT_FLAGS(ep, len, ep->info->tx_attr->op_flags)

#define VERBS_SELECTIVE_COMP(ep) (ep->ep_flags & FI_SELECTIVE_COMPLETION)

#define VERBS_COMP_FLAGS(ep, flags) (!VERBS_SELECTIVE_COMP(ep) || \
		(flags & (FI_COMPLETION | FI_TRANSMIT_COMPLETE)))
#define VERBS_COMP(ep) VERBS_COMP_FLAGS(ep, ep->info->tx_attr->op_flags)

#define VERBS_COMP_READ_FLAGS(ep, flags) (!VERBS_SELECTIVE_COMP(ep) || \
		(flags & (FI_COMPLETION | FI_TRANSMIT_COMPLETE | FI_DELIVERY_COMPLETE)))
#define VERBS_COMP_READ(ep) VERBS_COMP_READ_FLAGS(ep, ep->info->tx_attr->op_flags)

#define VERBS_DBG(subsys, ...) FI_DBG(&fi_ibv_prov, subsys, __VA_ARGS__)
#define VERBS_INFO(subsys, ...) FI_INFO(&fi_ibv_prov, subsys, __VA_ARGS__)

#define VERBS_INFO_ERRNO(subsys, fn, errno) VERBS_INFO(subsys, fn ": %s(%d)\n",	\
		strerror(errno), errno)

#define set_sge(sge, buf, len, desc)			\
	do {						\
		sge.addr = (uintptr_t)buf;		\
		sge.length = (uint32_t)len;		\
		sge.lkey = (uint32_t)(uintptr_t)desc;	\
	} while (0)

struct fi_ibv_fabric {
	struct fid_fabric	fabric_fid;
};

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

struct fi_ibv_pep {
	struct fid_pep		pep_fid;
	struct fi_ibv_eq	*eq;
	struct rdma_cm_id	*id;
	int			bound;
	size_t			src_addrlen;
};

struct fi_ibv_domain {
	struct fid_domain	domain_fid;
	struct ibv_context	*verbs;
	struct ibv_pd		*pd;
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
};

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

struct fi_ibv_connreq {
	struct fid		handle;
	struct rdma_cm_id	*id;
};

static const char *local_node = "localhost";
static char def_tx_ctx_size[16] = "384";
static char def_rx_ctx_size[16] = "384";
static char def_tx_iov_limit[16] = "4";
static char def_rx_iov_limit[16] = "4";
static char def_inject_size[16] = "64";

const struct fi_fabric_attr verbs_fabric_attr = {
	.prov_version		= VERBS_PROV_VERS,
};

const struct fi_domain_attr verbs_domain_attr = {
	.threading		= FI_THREAD_SAFE,
	.control_progress	= FI_PROGRESS_AUTO,
	.data_progress		= FI_PROGRESS_AUTO,
	.mr_mode		= FI_MR_BASIC,
	.mr_key_size		= sizeof_field(struct ibv_sge, lkey),
	.cq_data_size		= sizeof_field(struct ibv_send_wr, imm_data),
	.tx_ctx_cnt		= 1024,
	.rx_ctx_cnt		= 1024,
	.max_ep_tx_ctx		= 1,
	.max_ep_rx_ctx		= 1,
};

const struct fi_ep_attr verbs_ep_attr = {
	.type			= FI_EP_MSG,
	.protocol_version	= 1,
	.msg_prefix_size	= 0,
	.max_order_war_size	= 0,
	.mem_tag_format		= 0,
	.tx_ctx_cnt		= 1,
	.rx_ctx_cnt		= 1,
};

const struct fi_rx_attr verbs_rx_attr = {
	.caps			= VERBS_CAPS,
	.mode			= VERBS_RX_MODE,
	.msg_order		= VERBS_MSG_ORDER,
	.total_buffered_recv	= 0,
};

const struct fi_tx_attr verbs_tx_attr = {
	.caps			= VERBS_CAPS,
	.mode			= VERBS_TX_MODE,
	.op_flags		= VERBS_TX_OP_FLAGS,
	.msg_order		= VERBS_MSG_ORDER,
	.inject_size		= 0,
	.rma_iov_limit		= 1,
};

static struct fi_info *verbs_info = NULL;
static pthread_mutex_t verbs_info_lock = PTHREAD_MUTEX_INITIALIZER;

static int fi_ibv_sockaddr_len(struct sockaddr *addr)
{
	if (!addr)
		return 0;

	switch (addr->sa_family) {
	case AF_INET:
		return sizeof(struct sockaddr_in);
	case AF_INET6:
		return sizeof(struct sockaddr_in6);
	case AF_IB:
		return sizeof(struct sockaddr_ib);
	default:
		return 0;
	}
}

static int fi_ibv_check_fabric_attr(const struct fi_fabric_attr *attr,
		const struct fi_info *info)
{
	if (attr->name && strcmp(attr->name, info->fabric_attr->name)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "Unknown fabric name\n");
		return -FI_ENODATA;
	}

	if (attr->prov_version > info->fabric_attr->prov_version) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Unsupported provider version\n");
		return -FI_ENODATA;
	}

	return 0;
}

static int fi_ibv_check_domain_attr(const struct fi_domain_attr *attr,
		const struct fi_info *info)
{
	if (attr->name && strcmp(attr->name, info->domain_attr->name)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "Unknown domain name\n");
		return -FI_ENODATA;
	}

	switch (attr->threading) {
	case FI_THREAD_UNSPEC:
	case FI_THREAD_SAFE:
	case FI_THREAD_FID:
	case FI_THREAD_DOMAIN:
	case FI_THREAD_COMPLETION:
		break;
	default:
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Invalid threading model\n");
		return -FI_ENODATA;
	}

	switch (attr->control_progress) {
	case FI_PROGRESS_UNSPEC:
	case FI_PROGRESS_AUTO:
	case FI_PROGRESS_MANUAL:
		break;
	default:
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given control progress mode not supported\n");
		return -FI_ENODATA;
	}

	switch (attr->data_progress) {
	case FI_PROGRESS_UNSPEC:
	case FI_PROGRESS_AUTO:
	case FI_PROGRESS_MANUAL:
		break;
	default:
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given data progress mode not supported!\n");
		return -FI_ENODATA;
	}

	switch (attr->mr_mode) {
	case FI_MR_UNSPEC:
	case FI_MR_BASIC:
		break;
	default:
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"MR mode not supported\n");
		return -FI_ENODATA;
	}

	if (attr->mr_key_size > info->domain_attr->mr_key_size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"MR key size too large\n");
		return -FI_ENODATA;
	}

	if (attr->cq_data_size > info->domain_attr->cq_data_size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"CQ data size too large\n");
		return -FI_ENODATA;
	}

	if (attr->cq_cnt > info->domain_attr->cq_cnt) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"cq_cnt exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->ep_cnt > info->domain_attr->ep_cnt) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"ep_cnt exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->max_ep_tx_ctx > info->domain_attr->max_ep_tx_ctx) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"domain_attr: max_ep_tx_ctx exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->max_ep_rx_ctx > info->domain_attr->max_ep_rx_ctx) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"domain_attr: max_ep_rx_ctx exceeds supported size\n");
		return -FI_ENODATA;
	}

	return 0;
}

static int fi_ibv_check_ep_attr(const struct fi_ep_attr *attr,
		const struct fi_info *info)
{
	switch (attr->type) {
	case FI_EP_UNSPEC:
	case FI_EP_MSG:
		break;
	default:
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Unsupported endpoint type\n");
		return -FI_ENODATA;
	}

	switch (attr->protocol) {
	case FI_PROTO_UNSPEC:
	case FI_PROTO_RDMA_CM_IB_RC:
	case FI_PROTO_IWARP:
	case FI_PROTO_IB_UD:
		break;
	default:
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Unsupported protocol\n");
		return -FI_ENODATA;
	}

	if (attr->protocol_version > 1) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Unsupported protocol version\n");
		return -FI_ENODATA;
	}

	if (attr->max_msg_size > info->ep_attr->max_msg_size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Max message size too large\n");
		return -FI_ENODATA;
	}

	if (attr->max_order_raw_size > info->ep_attr->max_order_raw_size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"max_order_raw_size exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->max_order_war_size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"max_order_war_size exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->max_order_waw_size > info->ep_attr->max_order_waw_size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"max_order_waw_size exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->tx_ctx_cnt > info->domain_attr->max_ep_tx_ctx) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"tx_ctx_cnt exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->rx_ctx_cnt > info->domain_attr->max_ep_rx_ctx) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"rx_ctx_cnt exceeds supported size\n");
		return -FI_ENODATA;
	}

	return 0;
}

static int fi_ibv_check_rx_attr(const struct fi_rx_attr *attr,
		const struct fi_info *hints, const struct fi_info *info)
{
	uint64_t compare_mode, check_mode;

	if (attr->caps & ~(info->rx_attr->caps)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given rx_attr->caps not supported\n");
		return -FI_ENODATA;
	}

	compare_mode = attr->mode ? attr->mode : hints->mode;
	check_mode = (hints->caps & FI_RMA) ?
		     info->rx_attr->mode : VERBS_MODE;
	if ((compare_mode & check_mode) != check_mode) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given rx_attr->mode not supported\n");
		return -FI_ENODATA;
	}

	if (attr->op_flags & ~(info->rx_attr->op_flags)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given rx_attr->op_flags not supported\n");
		return -FI_ENODATA;
	}

	if (attr->msg_order & ~(info->rx_attr->msg_order)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given rx_attr->msg_order not supported\n");
		return -FI_ENODATA;
	}

	if (attr->size > info->rx_attr->size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given rx_attr->size is greater than supported\n");
		return -FI_ENODATA;
	}

	if (attr->total_buffered_recv > info->rx_attr->total_buffered_recv) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given rx_attr->total_buffered_recv exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->iov_limit > info->rx_attr->iov_limit) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given rx_attr->iov_limit greater than supported\n");
		return -FI_ENODATA;
	}

	return 0;
}

static int fi_ibv_check_tx_attr(const struct fi_tx_attr *attr,
		const struct fi_info *hints, const struct fi_info *info)
{
	if (attr->caps & ~(info->tx_attr->caps)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given tx_attr->caps not supported\n");
		return -FI_ENODATA;
	}

	if (((attr->mode ? attr->mode : hints->mode) &
				info->tx_attr->mode) != info->tx_attr->mode) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given tx_attr->mode not supported\n");
		return -FI_ENODATA;
	}

	if (attr->op_flags & ~(info->tx_attr->op_flags)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given tx_attr->op_flags not supported\n");
		return -FI_ENODATA;
	}

	if (attr->msg_order & ~(info->tx_attr->msg_order)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given tx_attr->msg_order not supported\n");
		return -FI_ENODATA;
	}

	if (attr->size > info->tx_attr->size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given tx_attr->size is greater than supported\n");
		return -FI_ENODATA;
	}

	if (attr->iov_limit > info->tx_attr->iov_limit) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given tx_attr->iov_limit greater than supported\n");
		return -FI_ENODATA;
	}

	if (attr->rma_iov_limit > info->tx_attr->rma_iov_limit) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given tx_attr->rma_iov_limit greater than supported\n");
		return -FI_ENODATA;
	}

	return 0;
}

static int fi_ibv_check_hints(const struct fi_info *hints,
		const struct fi_info *info)
{
	int ret;

	if (hints->caps & ~(info->caps)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Unsupported capabilities\n");
		return -FI_ENODATA;
	}

	if ((hints->mode & info->mode) != info->mode) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Required hints mode bits not set. Expected:0x%llx"
			" Given:0x%llx\n", info->mode, hints->mode);
		return -FI_ENODATA;
	}

	if (hints->fabric_attr) {
		ret = fi_ibv_check_fabric_attr(hints->fabric_attr, info);
		if (ret)
			return ret;
	}

	if (hints->domain_attr) {
		ret = fi_ibv_check_domain_attr(hints->domain_attr, info);
		if (ret)
			return ret;
	}

	if (hints->ep_attr) {
		ret = fi_ibv_check_ep_attr(hints->ep_attr, info);
		if (ret)
			return ret;
	}

	if (hints->rx_attr) {
		ret = fi_ibv_check_rx_attr(hints->rx_attr, hints, info);
		if (ret)
			return ret;
	}

	if (hints->tx_attr) {
		ret = fi_ibv_check_tx_attr(hints->tx_attr, hints, info);
		if (ret)
			return ret;
	}

	return 0;
}

static int fi_ibv_fi_to_rai(const struct fi_info *fi, uint64_t flags, struct rdma_addrinfo *rai)
{
	memset(rai, 0, sizeof *rai);
	if (flags & FI_SOURCE)
		rai->ai_flags = RAI_PASSIVE;
	if (flags & FI_NUMERICHOST)
		rai->ai_flags |= RAI_NUMERICHOST;

	rai->ai_qp_type = IBV_QPT_RC;
	rai->ai_port_space = RDMA_PS_TCP;

	if (!fi)
		return 0;

	switch(fi->addr_format) {
	case FI_SOCKADDR_IN:
		rai->ai_family = AF_INET;
		rai->ai_flags |= RAI_FAMILY;
		break;
	case FI_SOCKADDR_IN6:
		rai->ai_family = AF_INET6;
		rai->ai_flags |= RAI_FAMILY;
		break;
	case FI_SOCKADDR_IB:
		rai->ai_family = AF_IB;
		rai->ai_flags |= RAI_FAMILY;
		break;
	case FI_SOCKADDR:
		if (fi->src_addrlen) {
			rai->ai_family = ((struct sockaddr *)fi->src_addr)->sa_family;
			rai->ai_flags |= RAI_FAMILY;
		} else if (fi->dest_addrlen) {
			rai->ai_family = ((struct sockaddr *)fi->dest_addr)->sa_family;
			rai->ai_flags |= RAI_FAMILY;
		}
		break;
	case FI_FORMAT_UNSPEC:
		break;
	default:
		VERBS_INFO(FI_LOG_FABRIC, "Unknown fi->addr_format\n");
	}

	if (fi->src_addrlen) {
		if (!(rai->ai_src_addr = malloc(fi->src_addrlen)))
			return -FI_ENOMEM;
		memcpy(rai->ai_src_addr, fi->src_addr, fi->src_addrlen);
		rai->ai_src_len = fi->src_addrlen;
	}
	if (fi->dest_addrlen) {
		if (!(rai->ai_dst_addr = malloc(fi->dest_addrlen)))
			return -FI_ENOMEM;
		memcpy(rai->ai_dst_addr, fi->dest_addr, fi->dest_addrlen);
		rai->ai_dst_len = fi->dest_addrlen;
	}

	return 0;
}

static int fi_ibv_rai_to_fi(struct rdma_addrinfo *rai, struct fi_info *fi)
{
	switch(rai->ai_family) {
	case AF_INET:
		fi->addr_format = FI_SOCKADDR_IN;
		break;
	case AF_INET6:
		fi->addr_format = FI_SOCKADDR_IN6;
		break;
	case AF_IB:
		fi->addr_format = FI_SOCKADDR_IB;
		break;
	default:
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "Unknown rai->ai_family\n");
	}

	if (rai->ai_src_len) {
 		if (!(fi->src_addr = malloc(rai->ai_src_len)))
 			return -FI_ENOMEM;
 		memcpy(fi->src_addr, rai->ai_src_addr, rai->ai_src_len);
 		fi->src_addrlen = rai->ai_src_len;
 	}
 	if (rai->ai_dst_len) {
		if (!(fi->dest_addr = malloc(rai->ai_dst_len)))
			return -FI_ENOMEM;
 		memcpy(fi->dest_addr, rai->ai_dst_addr, rai->ai_dst_len);
 		fi->dest_addrlen = rai->ai_dst_len;
 	}

 	return 0;
}

static inline int fi_ibv_get_qp_cap(struct ibv_context *ctx,
		struct ibv_device_attr *device_attr,
		struct fi_info *info)
{
	struct ibv_pd *pd;
	struct ibv_cq *cq;
	struct ibv_qp *qp;
	struct ibv_qp_init_attr init_attr;
	int ret = 0;

	pd = ibv_alloc_pd(ctx);
	if (!pd) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "ibv_alloc_pd", errno);
		return -errno;
	}

	cq = ibv_create_cq(ctx, 1, NULL, NULL, 0);
	if (!cq) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "ibv_create_cq", errno);
		ret = -errno;
		goto err1;
	}


	/* TODO: serialize access to string buffers */
	fi_read_file(FI_CONF_DIR, "def_tx_ctx_size",
			def_tx_ctx_size, sizeof def_tx_ctx_size);
	fi_read_file(FI_CONF_DIR, "def_rx_ctx_size",
			def_rx_ctx_size, sizeof def_rx_ctx_size);
	fi_read_file(FI_CONF_DIR, "def_tx_iov_limit",
			def_tx_iov_limit, sizeof def_tx_iov_limit);
	fi_read_file(FI_CONF_DIR, "def_rx_iov_limit",
			def_rx_iov_limit, sizeof def_rx_iov_limit);
	fi_read_file(FI_CONF_DIR, "def_inject_size",
			def_inject_size, sizeof def_inject_size);

	memset(&init_attr, 0, sizeof init_attr);
	init_attr.send_cq = cq;
	init_attr.recv_cq = cq;
	init_attr.cap.max_send_wr = atoi(def_tx_ctx_size);
	init_attr.cap.max_recv_wr = atoi(def_rx_ctx_size);
	init_attr.cap.max_send_sge = MIN(atoi(def_tx_iov_limit), device_attr->max_sge);
	init_attr.cap.max_recv_sge = MIN(atoi(def_rx_iov_limit), device_attr->max_sge);
	init_attr.cap.max_inline_data = atoi(def_inject_size);
	init_attr.qp_type = IBV_QPT_RC;

	qp = ibv_create_qp(pd, &init_attr);
	if (!qp) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "ibv_create_qp", errno);
		ret = -errno;
		goto err2;
	}

	info->tx_attr->inject_size	= init_attr.cap.max_inline_data;
	info->tx_attr->iov_limit 	= init_attr.cap.max_send_sge;
	info->tx_attr->size	 	= init_attr.cap.max_send_wr;

	info->rx_attr->iov_limit 	= init_attr.cap.max_recv_sge;
	info->rx_attr->size	 	= init_attr.cap.max_recv_wr;

	ibv_destroy_qp(qp);
err2:
	ibv_destroy_cq(cq);
err1:
	ibv_dealloc_pd(pd);

	return ret;
}

static int fi_ibv_get_device_attrs(struct ibv_context *ctx, struct fi_info *info)
{
	struct ibv_device_attr device_attr;
	struct ibv_port_attr port_attr;
	int ret = 0;

	ret = ibv_query_device(ctx, &device_attr);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "ibv_query_device", errno);
		return -errno;
	}

	info->domain_attr->cq_cnt 		= device_attr.max_cq;
	info->domain_attr->ep_cnt 		= device_attr.max_qp;
	info->domain_attr->tx_ctx_cnt 		= MIN(info->domain_attr->tx_ctx_cnt, device_attr.max_qp);
	info->domain_attr->rx_ctx_cnt 		= MIN(info->domain_attr->rx_ctx_cnt, device_attr.max_qp);
	info->domain_attr->max_ep_tx_ctx 	= device_attr.max_qp;
	info->domain_attr->max_ep_rx_ctx 	= device_attr.max_qp;

	ret = fi_ibv_get_qp_cap(ctx, &device_attr, info);
	if (ret)
		return ret;

	ret = ibv_query_port(ctx, 1, &port_attr);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "ibv_query_port", errno);
		return -errno;
	}

	info->ep_attr->max_msg_size 		= port_attr.max_msg_sz;
	info->ep_attr->max_order_raw_size 	= port_attr.max_msg_sz;
	info->ep_attr->max_order_waw_size	= port_attr.max_msg_sz;

	return 0;
}

/*
 * USNIC plugs into the verbs framework, but is not a usable device.
 * Manually check for devices and fail gracefully if none are present.
 * This avoids the lower libraries (libibverbs and librdmacm) from
 * reporting error messages to stderr.
 */
static int fi_ibv_have_device(void)
{
	struct ibv_device **devs;
	struct ibv_context *verbs;
	int i, ret = 0;

	devs = ibv_get_device_list(NULL);
	if (!devs)
		return 0;

	for (i = 0; devs[i]; i++) {
		verbs = ibv_open_device(devs[i]);
		if (verbs) {
			ibv_close_device(verbs);
			ret = 1;
			break;
		}
	}

	ibv_free_device_list(devs);
	return ret;
}

static int fi_ibv_get_info_ctx(struct ibv_context *ctx, struct fi_info **info)
{
	struct fi_info *fi;
	union ibv_gid gid;
	size_t name_len;
	int ret;

	if (!(fi = fi_allocinfo()))
		return -FI_ENOMEM;

	fi->caps		= VERBS_CAPS;
	fi->mode		= VERBS_MODE;
	fi->handle		= NULL;
	*(fi->tx_attr)		= verbs_tx_attr;
	*(fi->rx_attr)		= verbs_rx_attr;
	*(fi->ep_attr)		= verbs_ep_attr;
	*(fi->domain_attr)	= verbs_domain_attr;
	*(fi->fabric_attr)	= verbs_fabric_attr;

	ret = fi_ibv_get_device_attrs(ctx, fi);
	if (ret)
		goto err;

	switch (ctx->device->transport_type) {
	case IBV_TRANSPORT_IB:
		if(ibv_query_gid(ctx, 1, 0, &gid)) {
			VERBS_INFO_ERRNO(FI_LOG_FABRIC, "ibv_query_gid", errno);
			ret = -errno;
			goto err;
		}

		name_len =  strlen(VERBS_IB_PREFIX) + INET6_ADDRSTRLEN;

		if (!(fi->fabric_attr->name = calloc(1, name_len + 1))) {
			ret = -FI_ENOMEM;
			goto err;
		}

		snprintf(fi->fabric_attr->name, name_len, VERBS_IB_PREFIX "%lx",
			 gid.global.subnet_prefix);

		fi->ep_attr->protocol = FI_PROTO_RDMA_CM_IB_RC;
		break;
	case IBV_TRANSPORT_IWARP:
		fi->fabric_attr->name = strdup(VERBS_IWARP_FABRIC);
		if (!fi->fabric_attr->name) {
			ret = -FI_ENOMEM;
			goto err;
		}

		fi->ep_attr->protocol = FI_PROTO_IWARP;
		fi->tx_attr->op_flags = VERBS_TX_OP_FLAGS_IWARP;
		break;
	default:
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "Unknown transport type\n");
		ret = -FI_ENODATA;
		goto err;
	}

	if (!(fi->domain_attr->name = strdup(ctx->device->name))) {
		ret = -FI_ENOMEM;
		goto err;
	}

	*info = fi;
	return 0;
err:
	fi_freeinfo(fi);
	return ret;
}

static int fi_ibv_init_info(void)
{
	struct ibv_context **ctx_list;
	struct fi_info *fi = NULL, *tail = NULL;
	int ret = 0, i, num_devices;

	if (verbs_info)
		return 0;

	pthread_mutex_lock(&verbs_info_lock);
	if (verbs_info)
		goto unlock;

	if (!fi_ibv_have_device()) {
		VERBS_INFO(FI_LOG_FABRIC, "No RDMA devices found\n");
		ret = -FI_ENODATA;
		goto unlock;
	}

	ctx_list = rdma_get_devices(&num_devices);
	if (!num_devices) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "rdma_get_devices", errno);
		ret = -errno;
		goto unlock;
	}

	for (i = 0; i < num_devices; i++) {
		ret = fi_ibv_get_info_ctx(ctx_list[i], &fi);
		if (!ret) {
			if (!verbs_info)
				verbs_info = fi;
			else
				tail->next = fi;
			tail = fi;
		}
	}

	ret = verbs_info ? 0 : ret;

	rdma_free_devices(ctx_list);
unlock:
	pthread_mutex_unlock(&verbs_info_lock);
	return ret;
}

static inline void fi_ibv_update_info(const struct fi_info *hints, struct fi_info *info)
{
	if (hints) {
		if (hints->ep_attr) {
			if (hints->ep_attr->tx_ctx_cnt)
				info->ep_attr->tx_ctx_cnt = hints->ep_attr->tx_ctx_cnt;
			if (hints->ep_attr->rx_ctx_cnt)
				info->ep_attr->rx_ctx_cnt = hints->ep_attr->rx_ctx_cnt;
		}

		if (hints->tx_attr)
			info->tx_attr->op_flags = hints->tx_attr->op_flags;

		if (hints->rx_attr)
			info->rx_attr->op_flags = hints->rx_attr->op_flags;

		if (hints->handle)
			info->handle = hints->handle;
	} else {
		info->tx_attr->op_flags = 0;
		info->rx_attr->op_flags = 0;
	}
}

static int
fi_ibv_create_ep(const char *node, const char *service,
		 uint64_t flags, const struct fi_info *hints,
		 struct rdma_addrinfo **rai, struct rdma_cm_id **id)
{
	struct rdma_addrinfo rai_hints, *_rai;
	struct rdma_addrinfo **rai_current;
	int ret;

	ret = fi_ibv_fi_to_rai(hints, flags, &rai_hints);
	if (ret)
		goto out;

	if (!node && !rai_hints.ai_dst_addr) {
		if (!rai_hints.ai_src_addr) {
			node = local_node;
		}
		rai_hints.ai_flags |= RAI_PASSIVE;
	}

	ret = rdma_getaddrinfo((char *) node, (char *) service,
				&rai_hints, &_rai);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "rdma_getaddrinfo", errno);
		ret = -errno;
		goto out;
	}

	/*
	 * If caller requested rai, remove ib_rai entries added by IBACM to
	 * prevent wrong ib_connect_hdr from being sent in connect request.
	 */
	if (rai && hints && (hints->addr_format != FI_SOCKADDR_IB)) {
		for (rai_current = &_rai; *rai_current;) {
			struct rdma_addrinfo *rai_next;
			if ((*rai_current)->ai_family == AF_IB) {
				rai_next = (*rai_current)->ai_next;
				(*rai_current)->ai_next = NULL;
				rdma_freeaddrinfo(*rai_current);
				*rai_current = rai_next;
				continue;
			}
			rai_current = &(*rai_current)->ai_next;
		}
	}

	ret = rdma_create_ep(id, _rai, NULL, NULL);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "rdma_create_ep", errno);
		ret = -errno;
		goto err;
	}

	if (rai) {
		*rai = _rai;
		goto out;
	}
err:
	rdma_freeaddrinfo(_rai);
out:
	if (rai_hints.ai_src_addr)
		free(rai_hints.ai_src_addr);
	if (rai_hints.ai_dst_addr)
		free(rai_hints.ai_dst_addr);
	return ret;
}

static void fi_ibv_msg_ep_qp_init_attr(struct fi_ibv_msg_ep *ep,
		struct ibv_qp_init_attr *attr)
{
	attr->cap.max_send_wr		= ep->info->tx_attr->size;
	attr->cap.max_recv_wr		= ep->info->rx_attr->size;
	attr->cap.max_send_sge		= ep->info->tx_attr->iov_limit;
	attr->cap.max_recv_sge		= ep->info->rx_attr->iov_limit;
	attr->cap.max_inline_data	= ep->info->tx_attr->inject_size;

	attr->srq = NULL;
	attr->qp_type = IBV_QPT_RC;
	attr->sq_sig_all = 0;
	attr->qp_context = ep;
	attr->send_cq = ep->scq->cq;
	attr->recv_cq = ep->rcq->cq;
}

static struct fi_info *fi_ibv_search_verbs_info(const char *fabric_name,
		const char *domain_name)
{
	struct fi_info *info;

	for (info = verbs_info; info; info = info->next) {
		if ((!domain_name || !strcmp(info->domain_attr->name, domain_name)) &&
			(!fabric_name || !strcmp(info->fabric_attr->name, fabric_name))) {
			return info;
		}
	}

	return NULL;
}

static int fi_ibv_get_matching_info(struct fi_info *check_info,
		struct fi_info *hints, struct rdma_addrinfo *rai,
		struct fi_info **info)
{

	int ret;
	struct fi_info *fi, *tail;

	*info = tail = NULL;

	for (; check_info; check_info = check_info->next) {
		if (hints) {
			ret = fi_ibv_check_hints(hints, check_info);
			if (ret)
				continue;
		}

		if (!(fi = fi_dupinfo(check_info))) {
			ret = -FI_ENOMEM;
			goto err1;
		}

		ret = fi_ibv_rai_to_fi(rai, fi);
		if (ret)
			goto err2;

		fi_ibv_update_info(hints, fi);

		if (!*info)
			*info = fi;
		else
			tail->next = fi;
		tail = fi;
	}

	if (!*info)
		return -FI_ENODATA;

	return 0;
err2:
	fi_freeinfo(fi);
err1:
	fi_freeinfo(*info);
	return ret;
}

static int fi_ibv_getinfo(uint32_t version, const char *node, const char *service,
			  uint64_t flags, struct fi_info *hints, struct fi_info **info)
{
	struct rdma_cm_id *id;
	struct rdma_addrinfo *rai;
	struct fi_info *check_info;
	int ret;

	ret = fi_ibv_init_info();
	if (ret)
		goto err1;

	ret = fi_ibv_create_ep(node, service, flags, hints, &rai, &id);
	if (ret)
		goto err1;

	check_info = id->verbs ? fi_ibv_search_verbs_info(NULL,
			ibv_get_device_name(id->verbs->device)) : verbs_info;

	if (!check_info) {
		VERBS_DBG(FI_LOG_FABRIC, "Unable to find check_info\n");
		ret = -FI_ENODATA;
		goto err2;
	}

	ret = fi_ibv_get_matching_info(check_info, hints, rai, info);

err2:
	rdma_destroy_ep(id);
	rdma_freeaddrinfo(rai);
err1:
	if (!ret || ret == -FI_ENOMEM)
		return ret;
	else
		return -FI_ENODATA;
}

static int fi_ibv_msg_ep_create_qp(struct fi_ibv_msg_ep *ep)
{
	struct ibv_qp_init_attr attr;

	fi_ibv_msg_ep_qp_init_attr(ep, &attr);
	return rdma_create_qp(ep->id, ep->rcq->domain->pd, &attr) ? -errno : 0;
}

static int fi_ibv_msg_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct fi_ibv_msg_ep *ep;
	int ret;

	ep = container_of(fid, struct fi_ibv_msg_ep, ep_fid.fid);

	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		/* Must bind a CQ to either RECV or SEND completions, and
		 * the FI_SELECTIVE_COMPLETION flag is only valid when binding the
		 * FI_SEND CQ. */
		if (!(flags & (FI_RECV|FI_SEND))
				|| (flags & (FI_SEND|FI_SELECTIVE_COMPLETION))
							== FI_SELECTIVE_COMPLETION) {
			return -EINVAL;
		}
		if (flags & FI_RECV) {
			if (ep->rcq)
				return -EINVAL;
			ep->rcq = container_of(bfid, struct fi_ibv_cq, cq_fid.fid);
		}
		if (flags & FI_SEND) {
			if (ep->scq)
				return -EINVAL;
			ep->scq = container_of(bfid, struct fi_ibv_cq, cq_fid.fid);
			if (flags & FI_SELECTIVE_COMPLETION)
				ep->ep_flags |= FI_SELECTIVE_COMPLETION;
			else
				ep->info->tx_attr->op_flags |= FI_COMPLETION;
		}
		break;
	case FI_CLASS_EQ:
		ep->eq = container_of(bfid, struct fi_ibv_eq, eq_fid.fid);
		ret = rdma_migrate_id(ep->id, ep->eq->channel);
		if (ret)
			return -errno;
		break;
	default:
		return -EINVAL;
	}

	return 0;
}

static ssize_t
fi_ibv_msg_ep_recvmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_recv_wr wr, *bad;
	struct ibv_sge *sge = NULL;
	ssize_t ret;
	size_t i;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	if (msg->iov_count) {
		sge = alloca(sizeof(*sge) * msg->iov_count);
		for (i = 0; i < msg->iov_count; i++) {
			sge[i].addr = (uintptr_t) msg->msg_iov[i].iov_base;
			sge[i].length = (uint32_t) msg->msg_iov[i].iov_len;
			sge[i].lkey = (uint32_t) (uintptr_t) (msg->desc[i]);
		}

	}
	wr.sg_list = sge;
	wr.num_sge = msg->iov_count;

	ret = ibv_post_recv(_ep->id->qp, &wr, &bad);
	switch (ret) {
	case ENOMEM:
		return -FI_EAGAIN;
	case -1:
		/* Deal with non-compliant libibverbs drivers which set errno
		 * instead of directly returning the error value */
		return (errno == ENOMEM) ? -FI_EAGAIN : -errno;
	default:
		return -ret;
	}
}

static ssize_t
fi_ibv_msg_ep_recv(struct fid_ep *ep, void *buf, size_t len,
		void *desc, fi_addr_t src_addr, void *context)
{
	struct iovec iov;
	struct fi_msg msg;

	iov.iov_base = buf;
	iov.iov_len = len;

	msg.msg_iov = &iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = src_addr;
	msg.context = context;

	return fi_ibv_msg_ep_recvmsg(ep, &msg, 0);
}

static ssize_t
fi_ibv_msg_ep_recvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
                 size_t count, fi_addr_t src_addr, void *context)
{
	struct fi_msg msg;

	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = src_addr;
	msg.context = context;

	return fi_ibv_msg_ep_recvmsg(ep, &msg, 0);
}

static ssize_t fi_ibv_send(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr, size_t len,
		int count, void *context, int inject, int comp)
{
	struct ibv_send_wr *bad_wr;
	int ret;

	wr->num_sge = count;
	wr->wr_id = (uintptr_t) context;

	if (inject)
		wr->send_flags |= IBV_SEND_INLINE;

	if (comp)
		wr->send_flags |= IBV_SEND_SIGNALED;

	ret = ibv_post_send(ep->id->qp, wr, &bad_wr);
	switch (ret) {
	case ENOMEM:
		return -FI_EAGAIN;
	case -1:
		/* Deal with non-compliant libibverbs drivers which set errno
		 * instead of directly returning the error value */
		return (errno == ENOMEM) ? -FI_EAGAIN : -errno;
	default:
		return -ret;
	}
}

static ssize_t fi_ibv_send_buf(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
		const void *buf, size_t len, void *desc, void *context,
		int inject, int comp)
{
	struct ibv_sge sge;

	set_sge(sge, buf, len, desc);
	wr->sg_list = &sge;

	return fi_ibv_send(ep, wr, len, 1, context, inject, comp);
}

static ssize_t fi_ibv_send_iov(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
		const struct iovec *iov, void **desc, int count, void *context,
		int inject, int comp)
{
	size_t len = 0;
	int i;

	if (count) {
		wr->sg_list = alloca(sizeof(*wr->sg_list) * count);

		for (i = 0; i < count; i++) {
			set_sge(wr->sg_list[i], iov[i].iov_base, iov[i].iov_len, desc[i]);
			len += iov[i].iov_len;
		}
	}

	if (!inject && (wr->opcode != IBV_WR_RDMA_READ))
		inject = VERBS_INJECT(ep, len);

	return fi_ibv_send(ep, wr, len, count, context, inject, comp);
}

#define fi_ibv_send_msg(ep, wr, msg, inject, comp)		\
       fi_ibv_send_iov(ep, wr, (msg)->msg_iov, (msg)->desc,	\
		(msg)->iov_count, (msg)->context, inject, comp)

static ssize_t
fi_ibv_msg_ep_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg, uint64_t flags)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	if (flags & FI_REMOTE_CQ_DATA) {
		wr.opcode = IBV_WR_SEND_WITH_IMM;
		wr.imm_data = htonl((uint32_t)msg->data);
	} else {
		wr.opcode = IBV_WR_SEND;
	}

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	return fi_ibv_send_msg(ep, &wr, msg, flags & FI_INJECT,
			VERBS_COMP_FLAGS(ep, flags));
}

static ssize_t
fi_ibv_msg_ep_send(struct fid_ep *ep_fid, const void *buf, size_t len,
		void *desc, fi_addr_t dest_addr, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_SEND;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	return fi_ibv_send_buf(ep, &wr, buf, len, desc, context,
			VERBS_INJECT(ep, len), VERBS_COMP(ep));
}

static ssize_t
fi_ibv_msg_ep_senddata(struct fid_ep *ep_fid, const void *buf, size_t len,
		    void *desc, uint64_t data, fi_addr_t dest_addr, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_SEND_WITH_IMM;
	wr.imm_data = htonl((uint32_t)data);

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	return fi_ibv_send_buf(ep, &wr, buf, len, desc, context,
			VERBS_INJECT(ep, len), VERBS_COMP(ep));
}

static ssize_t
fi_ibv_msg_ep_sendv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
                 size_t count, fi_addr_t dest_addr, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_SEND;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	return fi_ibv_send_iov(ep, &wr, iov, desc, count, context, 0, VERBS_COMP(ep));
}

static struct fi_ops_msg fi_ibv_msg_ep_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = fi_ibv_msg_ep_recv,
	.recvv = fi_ibv_msg_ep_recvv,
	.recvmsg = fi_ibv_msg_ep_recvmsg,
	.send = fi_ibv_msg_ep_send,
	.sendv = fi_ibv_msg_ep_sendv,
	.sendmsg = fi_ibv_msg_ep_sendmsg,
	.inject = fi_no_msg_inject,
	.senddata = fi_ibv_msg_ep_senddata,
	.injectdata = fi_no_msg_injectdata,
};

static ssize_t
fi_ibv_msg_ep_rma_write(struct fid_ep *ep_fid, const void *buf, size_t len,
		     void *desc, fi_addr_t dest_addr,
		     uint64_t addr, uint64_t key, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_RDMA_WRITE;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	return fi_ibv_send_buf(ep, &wr, buf, len, desc, context,
			VERBS_INJECT(ep, len), VERBS_COMP(ep));
}

static ssize_t
fi_ibv_msg_ep_rma_writev(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		      size_t count, fi_addr_t dest_addr,
		      uint64_t addr, uint64_t key, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;


	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_RDMA_WRITE;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	return fi_ibv_send_iov(ep, &wr, iov, desc, count, context, 0, VERBS_COMP(ep));
}

static ssize_t
fi_ibv_msg_ep_rma_writemsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
			uint64_t flags)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	if (flags & FI_REMOTE_CQ_DATA) {
		wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
		wr.imm_data = htonl((uint32_t)msg->data);
	} else {
		wr.opcode = IBV_WR_RDMA_WRITE;
	}

	wr.wr.rdma.remote_addr = msg->rma_iov->addr;
	wr.wr.rdma.rkey = (uint32_t) msg->rma_iov->key;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	return fi_ibv_send_msg(ep, &wr, msg, flags & FI_INJECT,
			VERBS_COMP_FLAGS(ep, flags));
}

static ssize_t
fi_ibv_msg_ep_rma_read(struct fid_ep *ep_fid, void *buf, size_t len,
		    void *desc, fi_addr_t src_addr,
		    uint64_t addr, uint64_t key, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_RDMA_READ;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	return fi_ibv_send_buf(ep, &wr, buf, len, desc, context, 0,
			VERBS_COMP_READ(ep));
}

static ssize_t
fi_ibv_msg_ep_rma_readv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		     size_t count, fi_addr_t src_addr,
		     uint64_t addr, uint64_t key, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_RDMA_READ;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	return fi_ibv_send_iov(ep, &wr, iov, desc, count, context, 0,
			VERBS_COMP_READ(ep));
}

static ssize_t
fi_ibv_msg_ep_rma_readmsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
			uint64_t flags)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_RDMA_READ;
	wr.wr.rdma.remote_addr = msg->rma_iov->addr;
	wr.wr.rdma.rkey = (uint32_t) msg->rma_iov->key;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	return fi_ibv_send_msg(ep, &wr, msg, 0, VERBS_COMP_READ_FLAGS(ep, flags));
}

static ssize_t
fi_ibv_msg_ep_rma_writedata(struct fid_ep *ep_fid, const void *buf, size_t len,
			void *desc, uint64_t data, fi_addr_t dest_addr,
			uint64_t addr, uint64_t key, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	wr.imm_data = htonl((uint32_t)data);
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	return fi_ibv_send_buf(ep, &wr, buf, len, desc, context,
			VERBS_INJECT(ep, len), VERBS_COMP(ep));
}

static struct fi_ops_rma fi_ibv_msg_ep_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = fi_ibv_msg_ep_rma_read,
	.readv = fi_ibv_msg_ep_rma_readv,
	.readmsg = fi_ibv_msg_ep_rma_readmsg,
	.write = fi_ibv_msg_ep_rma_write,
	.writev = fi_ibv_msg_ep_rma_writev,
	.writemsg = fi_ibv_msg_ep_rma_writemsg,
	.inject = fi_no_rma_inject,
	.writedata = fi_ibv_msg_ep_rma_writedata,
	.injectdata = fi_no_rma_injectdata,
};

static int
fi_ibv_msg_ep_atomic_writevalid(struct fid_ep *ep, enum fi_datatype datatype,
				enum fi_op op, size_t *count)
{
	switch (op) {
	case FI_ATOMIC_WRITE:
		break;
	default:
		return -FI_ENOSYS;
	}

	switch (datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	if (count)
		*count = 1;
	return 0;
}

static int
fi_ibv_msg_ep_atomic_readwritevalid(struct fid_ep *ep, enum fi_datatype datatype,
				enum fi_op op, size_t *count)
{
	struct fi_ibv_msg_ep *_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);

	switch (op) {
	case FI_ATOMIC_READ:
		break;
	case FI_SUM:
		if (_ep->info->tx_attr->op_flags & FI_INJECT) {
			FI_INFO(&fi_ibv_prov, FI_LOG_EP_DATA,"FI_INJECT not "
				"supported for fi_fetch_atomic with FI_SUM op\n");
			return -FI_EINVAL;
		}
		break;
	default:
		return -FI_ENOSYS;
	}

	switch (datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	if (count)
		*count = 1;
	return 0;
}

static int
fi_ibv_msg_ep_atomic_compwritevalid(struct fid_ep *ep, enum fi_datatype datatype,
				enum fi_op op, size_t *count)
{
	struct fi_ibv_msg_ep *_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);

	if (op != FI_CSWAP)
		return -FI_ENOSYS;

	if (_ep->info->tx_attr->op_flags & FI_INJECT) {
		FI_INFO(&fi_ibv_prov, FI_LOG_EP_DATA, "FI_INJECT not supported "
			"for fi_compare_atomic\n");
		return -FI_EINVAL;
	}

	switch (datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	if (count)
		*count = 1;
	return 0;
}

static ssize_t
fi_ibv_msg_ep_atomic_write(struct fid_ep *ep_fid, const void *buf, size_t count,
			void *desc, fi_addr_t dest_addr, uint64_t addr, uint64_t key,
			enum fi_datatype datatype, enum fi_op op, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;
	size_t count_copy;
	int ret;

	if (count != 1)
		return -FI_E2BIG;

	count_copy = count;

	ret = fi_ibv_msg_ep_atomic_writevalid(ep_fid, datatype, op, &count_copy);
	if (ret)
		return ret;

	memset(&wr, 0, sizeof(wr));

	switch(op) {
	case FI_ATOMIC_WRITE:
		wr.opcode = IBV_WR_RDMA_WRITE;
		wr.wr.rdma.remote_addr = addr;
		wr.wr.rdma.rkey = (uint32_t) (uintptr_t) key;
		break;
	default:
		return -ENOSYS;
	}

	wr.send_flags = IBV_SEND_FENCE;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	return fi_ibv_send_buf(ep, &wr, buf, sizeof(uint64_t), desc, context,
			VERBS_INJECT(ep, sizeof(uint64_t)), VERBS_COMP(ep));
}

static ssize_t
fi_ibv_msg_ep_atomic_writev(struct fid_ep *ep,
                        const struct fi_ioc *iov, void **desc, size_t count,
                        fi_addr_t dest_addr, uint64_t addr, uint64_t key,
                        enum fi_datatype datatype, enum fi_op op, void *context)
{
	if (iov->count != 1)
		return -FI_E2BIG;

	return fi_ibv_msg_ep_atomic_write(ep, iov->addr, count, desc[0],
			dest_addr, addr, key, datatype, op, context);
}

static ssize_t
fi_ibv_msg_ep_atomic_writemsg(struct fid_ep *ep_fid,
                        const struct fi_msg_atomic *msg, uint64_t flags)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;
	size_t count_copy;
	int ret;

	if (msg->iov_count != 1 || msg->msg_iov->count != 1)
		return -FI_E2BIG;

	count_copy = msg->iov_count;

	ret = fi_ibv_msg_ep_atomic_writevalid(ep_fid, msg->datatype, msg->op,
			&count_copy);
	if (ret)
		return ret;

	memset(&wr, 0, sizeof(wr));

	switch (msg->op) {
	case FI_ATOMIC_WRITE:
		if (flags & FI_REMOTE_CQ_DATA) {
			wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
			wr.imm_data = htonl((uint32_t)msg->data);
		} else {
			wr.opcode = IBV_WR_RDMA_WRITE;
		}
		wr.wr.rdma.remote_addr = msg->rma_iov->addr;
		wr.wr.rdma.rkey = (uint32_t) (uintptr_t) msg->rma_iov->key;
		break;
	default:
		return -ENOSYS;
	}

	wr.send_flags = IBV_SEND_FENCE;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	return fi_ibv_send_buf(ep, &wr, msg->msg_iov->addr, sizeof(uint64_t),
			msg->desc[0], msg->context, flags & FI_INJECT,
			VERBS_COMP_FLAGS(ep, flags));
}

static ssize_t
fi_ibv_msg_ep_atomic_readwrite(struct fid_ep *ep_fid, const void *buf, size_t count,
			void *desc, void *result, void *result_desc,
			fi_addr_t dest_addr, uint64_t addr, uint64_t key,
			enum fi_datatype datatype,
			enum fi_op op, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;
	size_t count_copy;
	int ret;

	if (count != 1)
		return -FI_E2BIG;

	count_copy = count;

	ret = fi_ibv_msg_ep_atomic_readwritevalid(ep_fid, datatype, op,
			&count_copy);
	if (ret)
		return ret;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	memset(&wr, 0, sizeof(wr));

	switch (op) {
	case FI_ATOMIC_READ:
		wr.opcode = IBV_WR_RDMA_READ;
		wr.wr.rdma.remote_addr = addr;
		wr.wr.rdma.rkey = (uint32_t) (uintptr_t) key;
		break;
	case FI_SUM:
		wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
		wr.wr.atomic.remote_addr = addr;
		wr.wr.atomic.compare_add = (uintptr_t) buf;
		wr.wr.atomic.swap = 0;
		wr.wr.atomic.rkey = (uint32_t) (uintptr_t) key;
		break;
	default:
		return -ENOSYS;
	}

	wr.send_flags = IBV_SEND_FENCE;

	return fi_ibv_send_buf(ep, &wr, result, sizeof(uint64_t), result_desc,
		context, 0, VERBS_COMP(ep));
}

static ssize_t
fi_ibv_msg_ep_atomic_readwritev(struct fid_ep *ep, const struct fi_ioc *iov,
			void **desc, size_t count,
			struct fi_ioc *resultv, void **result_desc,
			size_t result_count, fi_addr_t dest_addr, uint64_t addr,
			uint64_t key, enum fi_datatype datatype,
			enum fi_op op, void *context)
{
	if (iov->count != 1)
		return -FI_E2BIG;

        return fi_ibv_msg_ep_atomic_readwrite(ep, iov->addr, count,
			desc[0], resultv->addr, result_desc[0],
			dest_addr, addr, key, datatype, op, context);
}

static ssize_t
fi_ibv_msg_ep_atomic_readwritemsg(struct fid_ep *ep_fid,
				const struct fi_msg_atomic *msg,
				struct fi_ioc *resultv, void **result_desc,
				size_t result_count, uint64_t flags)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;
	size_t count_copy;
	int ret;

	if (msg->iov_count != 1 || msg->msg_iov->count != 1)
		return -FI_E2BIG;

	count_copy = msg->iov_count;

	ret = fi_ibv_msg_ep_atomic_readwritevalid(ep_fid, msg->datatype, msg->op,
		       &count_copy);
	if (ret)
		return ret;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	memset(&wr, 0, sizeof(wr));

	switch (msg->op) {
	case FI_ATOMIC_READ:
		wr.opcode = IBV_WR_RDMA_READ;
		wr.wr.rdma.remote_addr = msg->rma_iov->addr;
		wr.wr.rdma.rkey = (uint32_t) (uintptr_t) msg->rma_iov->key;
		break;
	case FI_SUM:
		wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
		wr.wr.atomic.remote_addr = msg->rma_iov->addr;
		wr.wr.atomic.compare_add = (uintptr_t) msg->addr;
		wr.wr.atomic.swap = 0;
		wr.wr.atomic.rkey = (uint32_t) (uintptr_t) msg->rma_iov->key;
		break;
	default:
		return -ENOSYS;
	}

	wr.send_flags = IBV_SEND_FENCE;
	if (flags & FI_REMOTE_CQ_DATA)
		wr.imm_data = htonl((uint32_t) msg->data);

	return fi_ibv_send_buf(ep, &wr, resultv->addr, sizeof(uint64_t),
			result_desc[0], msg->context, 0,
			VERBS_COMP_FLAGS(ep, flags));
}

static ssize_t
fi_ibv_msg_ep_atomic_compwrite(struct fid_ep *ep_fid, const void *buf, size_t count,
			void *desc, const void *compare,
			void *compare_desc, void *result,
			void *result_desc,
			fi_addr_t dest_addr, uint64_t addr, uint64_t key,
			enum fi_datatype datatype,
			enum fi_op op, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;
	size_t count_copy;
	int ret;

	if (count != 1)
		return -FI_E2BIG;

	count_copy = count;

	ret = fi_ibv_msg_ep_atomic_compwritevalid(ep_fid, datatype, op, &count_copy);
	if (ret)
		return ret;

	memset(&wr, 0, sizeof(wr));

	wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
	wr.wr.atomic.remote_addr = addr;
	wr.wr.atomic.compare_add = (uintptr_t) compare;
	wr.wr.atomic.swap = (uintptr_t) buf;
	wr.wr.atomic.rkey = (uint32_t) (uintptr_t) key;

	wr.send_flags = IBV_SEND_FENCE;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	return fi_ibv_send_buf(ep, &wr, result, sizeof(uint64_t), result_desc,
		context, 0, VERBS_COMP(ep));
}

static ssize_t
fi_ibv_msg_ep_atomic_compwritev(struct fid_ep *ep, const struct fi_ioc *iov,
				void **desc, size_t count,
				const struct fi_ioc *comparev,
				void **compare_desc, size_t compare_count,
				struct fi_ioc *resultv, void **result_desc,
				size_t result_count,
				fi_addr_t dest_addr, uint64_t addr, uint64_t key,
				enum fi_datatype datatype,
				enum fi_op op, void *context)
{
	if (iov->count != 1)
		return -FI_E2BIG;

	return fi_ibv_msg_ep_atomic_compwrite(ep, iov->addr, count, desc[0],
				comparev->addr, compare_desc[0], resultv->addr,
				result_desc[0], dest_addr, addr, key,
                        	datatype, op, context);
}

static ssize_t
fi_ibv_msg_ep_atomic_compwritemsg(struct fid_ep *ep_fid,
				const struct fi_msg_atomic *msg,
				const struct fi_ioc *comparev,
				void **compare_desc, size_t compare_count,
				struct fi_ioc *resultv,
				void **result_desc, size_t result_count,
				uint64_t flags)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr;
	size_t count_copy;
	int ret;

	if (msg->iov_count != 1 || msg->msg_iov->count != 1)
		return -FI_E2BIG;

	count_copy = msg->iov_count;

	ret = fi_ibv_msg_ep_atomic_compwritevalid(ep_fid, msg->datatype, msg->op,
		       &count_copy);
	if (ret)
		return ret;

	memset(&wr, 0, sizeof(wr));

	wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
	wr.wr.atomic.remote_addr = msg->rma_iov->addr;
	wr.wr.atomic.compare_add = (uintptr_t) comparev->addr;
	wr.wr.atomic.swap = (uintptr_t) msg->addr;
	wr.wr.atomic.rkey = (uint32_t) (uintptr_t) msg->rma_iov->key;

	wr.send_flags = IBV_SEND_FENCE;
	if (flags & FI_REMOTE_CQ_DATA)
		wr.imm_data = htonl((uint32_t) msg->data);

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	return fi_ibv_send_buf(ep, &wr, resultv->addr, sizeof(uint64_t),
			result_desc[0], msg->context, 0, VERBS_COMP_FLAGS(ep, flags));
}

static struct fi_ops_atomic fi_ibv_msg_ep_atomic_ops = {
	.size		= sizeof(struct fi_ops_atomic),
	.write		= fi_ibv_msg_ep_atomic_write,
	.writev		= fi_ibv_msg_ep_atomic_writev,
	.writemsg	= fi_ibv_msg_ep_atomic_writemsg,
	.inject		= fi_no_atomic_inject,
	.readwrite	= fi_ibv_msg_ep_atomic_readwrite,
	.readwritev	= fi_ibv_msg_ep_atomic_readwritev,
	.readwritemsg	= fi_ibv_msg_ep_atomic_readwritemsg,
	.compwrite	= fi_ibv_msg_ep_atomic_compwrite,
	.compwritev	= fi_ibv_msg_ep_atomic_compwritev,
	.compwritemsg	= fi_ibv_msg_ep_atomic_compwritemsg,
	.writevalid	= fi_ibv_msg_ep_atomic_writevalid,
	.readwritevalid	= fi_ibv_msg_ep_atomic_readwritevalid,
	.compwritevalid = fi_ibv_msg_ep_atomic_compwritevalid
};

static int fi_ibv_copy_addr(void *dst_addr, size_t *dst_addrlen, void *src_addr)
{
	size_t src_addrlen = fi_ibv_sockaddr_len(src_addr);

	if (*dst_addrlen == 0) {
		*dst_addrlen = src_addrlen;
		return -FI_ETOOSMALL;
	}

	if (*dst_addrlen < src_addrlen) {
		memcpy(dst_addr, src_addr, *dst_addrlen);
	} else {
		memcpy(dst_addr, src_addr, src_addrlen);
	}
	*dst_addrlen = src_addrlen;
	return 0;
}

static int fi_ibv_msg_ep_setname(fid_t ep_fid, void *addr, size_t addrlen)
{
	struct fi_ibv_msg_ep *ep;
	void *save_addr;
	struct rdma_cm_id *id;
	int ret;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	if (addrlen != ep->info->src_addrlen) {
		FI_INFO(&fi_ibv_prov, FI_LOG_EP_CTRL,"addrlen expected: %d, got: %d.\n",
				ep->info->src_addrlen, addrlen);
		return -FI_EINVAL;
	}

	save_addr = ep->info->src_addr;

	ep->info->src_addr = malloc(ep->info->src_addrlen);
	if (!ep->info->src_addr) {
		ret = -FI_ENOMEM;
		goto err1;
	}

	memcpy(ep->info->src_addr, addr, ep->info->src_addrlen);

	ret = fi_ibv_create_ep(NULL, NULL, 0, ep->info, NULL, &id);
	if (ret)
		goto err2;

	if (ep->id)
		rdma_destroy_ep(ep->id);

	ep->id = id;
	free(save_addr);

	return 0;
err2:
	free(ep->info->src_addr);
err1:
	ep->info->src_addr = save_addr;
	return ret;
}

static int fi_ibv_msg_ep_getname(fid_t ep, void *addr, size_t *addrlen)
{
	struct fi_ibv_msg_ep *_ep;
	struct sockaddr *sa;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	sa = rdma_get_local_addr(_ep->id);
	return fi_ibv_copy_addr(addr, addrlen, sa);
}

static int fi_ibv_msg_ep_getpeer(struct fid_ep *ep, void *addr, size_t *addrlen)
{
	struct fi_ibv_msg_ep *_ep;
	struct sockaddr *sa;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	sa = rdma_get_peer_addr(_ep->id);
	return fi_ibv_copy_addr(addr, addrlen, sa);
}

static int
fi_ibv_msg_ep_connect(struct fid_ep *ep, const void *addr,
		   const void *param, size_t paramlen)
{
	struct fi_ibv_msg_ep *_ep;
	struct rdma_conn_param conn_param;
	struct sockaddr *src_addr, *dst_addr;
	int ret;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	if (!_ep->id->qp) {
		ret = ep->fid.ops->control(&ep->fid, FI_ENABLE, NULL);
		if (ret)
			return ret;
	}

	memset(&conn_param, 0, sizeof conn_param);
	conn_param.private_data = param;
	conn_param.private_data_len = paramlen;
	conn_param.responder_resources = RDMA_MAX_RESP_RES;
	conn_param.initiator_depth = RDMA_MAX_INIT_DEPTH;
	conn_param.flow_control = 1;
	conn_param.retry_count = 15;
	conn_param.rnr_retry_count = 7;

	src_addr = rdma_get_local_addr(_ep->id);
	if (src_addr) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "src_addr: %s:%d\n",
			inet_ntoa(((struct sockaddr_in *)src_addr)->sin_addr),
			ntohs(((struct sockaddr_in *)src_addr)->sin_port));
	}

	dst_addr = rdma_get_peer_addr(_ep->id);
	if (dst_addr) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "dst_addr: %s:%d\n",
			inet_ntoa(((struct sockaddr_in *)dst_addr)->sin_addr),
			ntohs(((struct sockaddr_in *)dst_addr)->sin_port));
	}

	return rdma_connect(_ep->id, &conn_param) ? -errno : 0;
}

static int
fi_ibv_msg_ep_accept(struct fid_ep *ep, const void *param, size_t paramlen)
{
	struct fi_ibv_msg_ep *_ep;
	struct rdma_conn_param conn_param;
	struct fi_ibv_connreq *connreq;
	int ret;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	if (!_ep->id->qp) {
		ret = ep->fid.ops->control(&ep->fid, FI_ENABLE, NULL);
		if (ret)
			return ret;
	}

	memset(&conn_param, 0, sizeof conn_param);
	conn_param.private_data = param;
	conn_param.private_data_len = paramlen;
	conn_param.responder_resources = RDMA_MAX_RESP_RES;
	conn_param.initiator_depth = RDMA_MAX_INIT_DEPTH;
	conn_param.flow_control = 1;
	conn_param.rnr_retry_count = 7;

	ret = rdma_accept(_ep->id, &conn_param);
	if (ret)
		return -errno;

	connreq = container_of(_ep->info->handle, struct fi_ibv_connreq, handle);
	free(connreq);

	return 0;
}

static int
fi_ibv_msg_ep_reject(struct fid_pep *pep, fid_t handle,
		  const void *param, size_t paramlen)
{
	struct fi_ibv_connreq *connreq;
	int ret;

	connreq = container_of(handle, struct fi_ibv_connreq, handle);
	ret = rdma_reject(connreq->id, param, (uint8_t) paramlen) ? -errno : 0;
	free(connreq);
	return ret;
}

static int fi_ibv_msg_ep_shutdown(struct fid_ep *ep, uint64_t flags)
{
	struct fi_ibv_msg_ep *_ep;
	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	return rdma_disconnect(_ep->id) ? -errno : 0;
}

static struct fi_ops_cm fi_ibv_msg_ep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_ibv_msg_ep_setname,
	.getname = fi_ibv_msg_ep_getname,
	.getpeer = fi_ibv_msg_ep_getpeer,
	.connect = fi_ibv_msg_ep_connect,
	.listen = fi_no_listen,
	.accept = fi_ibv_msg_ep_accept,
	.reject = fi_no_reject,
	.shutdown = fi_ibv_msg_ep_shutdown,
};

static int
fi_ibv_msg_ep_getopt(fid_t fid, int level, int optname,
		  void *optval, size_t *optlen)
{
	switch (level) {
	case FI_OPT_ENDPOINT:
		switch (optname) {
		case FI_OPT_CM_DATA_SIZE:
			if (*optlen < sizeof(size_t))
				return -FI_ETOOSMALL;
			*((size_t *) optval) = VERBS_CM_DATA_SIZE;
			*optlen = sizeof(size_t);
			return 0;
		default:
			return -FI_ENOPROTOOPT;
		}
	default:
		return -FI_ENOPROTOOPT;
	}
	return 0;
}

static int
fi_ibv_msg_ep_setopt(fid_t fid, int level, int optname,
		  const void *optval, size_t optlen)
{
	switch (level) {
	case FI_OPT_ENDPOINT:
		return -FI_ENOPROTOOPT;
	default:
		return -FI_ENOPROTOOPT;
	}
	return 0;
}

static int fi_ibv_msg_ep_enable(struct fid_ep *ep)
{
	struct fi_ibv_msg_ep *_ep;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	if (!_ep->eq)
		return -FI_ENOEQ;
	if (!_ep->scq || !_ep->rcq)
		return -FI_ENOCQ;

	return fi_ibv_msg_ep_create_qp(_ep);
}

static struct fi_ops_ep fi_ibv_msg_ep_base_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_no_cancel,
	.getopt = fi_ibv_msg_ep_getopt,
	.setopt = fi_ibv_msg_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static struct fi_ibv_msg_ep *fi_ibv_alloc_msg_ep(struct fi_info *info)
{
	struct fi_ibv_msg_ep *ep;

	ep = calloc(1, sizeof *ep);
	if (!ep)
		return NULL;

	ep->info = fi_dupinfo(info);
	if (!ep->info)
		goto err;

	return ep;
err:
	free(ep);
	return NULL;
}

static void fi_ibv_free_msg_ep(struct fi_ibv_msg_ep *ep)
{
	if (ep->id)
		rdma_destroy_ep(ep->id);
	fi_freeinfo(ep->info);
	free(ep);
}

static int fi_ibv_msg_ep_close(fid_t fid)
{
	struct fi_ibv_msg_ep *ep;

	ep = container_of(fid, struct fi_ibv_msg_ep, ep_fid.fid);
	fi_ibv_free_msg_ep(ep);

	return 0;
}

static int fi_ibv_msg_ep_control(struct fid *fid, int command, void *arg)
{
	struct fid_ep *ep;

	switch (fid->fclass) {
	case FI_CLASS_EP:
		ep = container_of(fid, struct fid_ep, fid);
		switch (command) {
		case FI_ENABLE:
			return fi_ibv_msg_ep_enable(ep);
			break;
		default:
			return -FI_ENOSYS;
		}
		break;
	default:
		return -FI_ENOSYS;
	}
}

static struct fi_ops fi_ibv_msg_ep_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_msg_ep_close,
	.bind = fi_ibv_msg_ep_bind,
	.control = fi_ibv_msg_ep_control,
	.ops_open = fi_no_ops_open,
};

static int
fi_ibv_open_ep(struct fid_domain *domain, struct fi_info *info,
	    struct fid_ep **ep, void *context)
{
	struct fi_ibv_domain *dom;
	struct fi_ibv_msg_ep *_ep;
	struct fi_ibv_connreq *connreq;
	struct fi_ibv_pep *pep;
	struct fi_info *fi;
	int ret;

	dom = container_of(domain, struct fi_ibv_domain, domain_fid);
	if (strcmp(dom->verbs->device->name, info->domain_attr->name)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_DOMAIN, "Invalid info->domain_attr->name\n");
		return -FI_EINVAL;
	}

	fi = fi_ibv_search_verbs_info(NULL, info->domain_attr->name);
	if (!fi) {
		FI_INFO(&fi_ibv_prov, FI_LOG_DOMAIN, "Unable to find matching verbs_info\n");
		return -FI_EINVAL;
	}

	if (info->ep_attr) {
		ret = fi_ibv_check_ep_attr(info->ep_attr, fi);
		if (ret)
			return ret;
	}

	if (info->tx_attr) {
		ret = fi_ibv_check_tx_attr(info->tx_attr, info, fi);
		if (ret)
			return ret;
	}

	if (info->rx_attr) {
		ret = fi_ibv_check_rx_attr(info->rx_attr, info, fi);
		if (ret)
			return ret;
	}

	_ep = fi_ibv_alloc_msg_ep(info);
	if (!_ep)
		return -FI_ENOMEM;

	if (!info->handle) {
		ret = fi_ibv_create_ep(NULL, NULL, 0, info, NULL, &_ep->id);
		if (ret)
			goto err;
	} else if (info->handle->fclass == FI_CLASS_CONNREQ) {
		connreq = container_of(info->handle, struct fi_ibv_connreq, handle);
		_ep->id = connreq->id;
        } else if (info->handle->fclass == FI_CLASS_PEP) {
		pep = container_of(info->handle, struct fi_ibv_pep, pep_fid.fid);
		_ep->id = pep->id;
		pep->id = NULL;

		if (rdma_resolve_addr(_ep->id, info->src_addr, info->dest_addr, VERBS_RESOLVE_TIMEOUT)) {
			ret = -errno;
			FI_INFO(&fi_ibv_prov, FI_LOG_DOMAIN, "Unable to rdma_resolve_addr\n");
			goto err;
		}

		if (rdma_resolve_route(_ep->id, VERBS_RESOLVE_TIMEOUT)) {
			ret = -errno;
			FI_INFO(&fi_ibv_prov, FI_LOG_DOMAIN, "Unable to rdma_resolve_route\n");
			goto err;
		}
	} else {
		ret = -FI_ENOSYS;
		goto err;
	}

	_ep->id->context = &_ep->ep_fid.fid;

	_ep->ep_fid.fid.fclass = FI_CLASS_EP;
	_ep->ep_fid.fid.context = context;
	_ep->ep_fid.fid.ops = &fi_ibv_msg_ep_ops;
	_ep->ep_fid.ops = &fi_ibv_msg_ep_base_ops;
	_ep->ep_fid.msg = &fi_ibv_msg_ep_msg_ops;
	_ep->ep_fid.cm = &fi_ibv_msg_ep_cm_ops;
	_ep->ep_fid.rma = &fi_ibv_msg_ep_rma_ops;
	_ep->ep_fid.atomic = &fi_ibv_msg_ep_atomic_ops;

	*ep = &_ep->ep_fid;

	return 0;
err:
	fi_ibv_free_msg_ep(_ep);
	return ret;
}

static ssize_t
fi_ibv_eq_readerr(struct fid_eq *eq, struct fi_eq_err_entry *entry,
		  uint64_t flags)
{
	struct fi_ibv_eq *_eq;

	_eq = container_of(eq, struct fi_ibv_eq, eq_fid.fid);
	if (!_eq->err.err)
		return 0;

	*entry = _eq->err;
	_eq->err.err = 0;
	_eq->err.prov_errno = 0;
	return sizeof(*entry);
}

/* TODO: This should copy the listening fi_info as the base */
static struct fi_info *
fi_ibv_eq_cm_getinfo(struct fi_ibv_fabric *fab, struct rdma_cm_event *event)
{
	struct fi_info *info, *fi;
	struct fi_ibv_connreq *connreq;

	if (fi_ibv_init_info()) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "Unable to initialize verbs_info\n");
		return NULL;
	}

	fi = fi_ibv_search_verbs_info(NULL, ibv_get_device_name(event->id->verbs->device));
	if (!fi)
		return NULL;

	info = fi_dupinfo(fi);
	if (!info)
		return NULL;

	info->fabric_attr->fabric = &fab->fabric_fid;
	if (!(info->fabric_attr->prov_name = strdup(VERBS_PROV_NAME)))
		goto err;

	fi_ibv_update_info(NULL, info);

	info->src_addrlen = fi_ibv_sockaddr_len(rdma_get_local_addr(event->id));
	if (!(info->src_addr = malloc(info->src_addrlen)))
		goto err;
	memcpy(info->src_addr, rdma_get_local_addr(event->id), info->src_addrlen);

	info->dest_addrlen = fi_ibv_sockaddr_len(rdma_get_peer_addr(event->id));
	if (!(info->dest_addr = malloc(info->dest_addrlen)))
		goto err;
	memcpy(info->dest_addr, rdma_get_peer_addr(event->id), info->dest_addrlen);

	FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "src_addr: %s:%d\n",
		inet_ntoa(((struct sockaddr_in *)info->src_addr)->sin_addr),
		ntohs(((struct sockaddr_in *)info->src_addr)->sin_port));

	FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "dst_addr: %s:%d\n",
		inet_ntoa(((struct sockaddr_in *)info->dest_addr)->sin_addr),
		ntohs(((struct sockaddr_in *)info->dest_addr)->sin_port));

	connreq = calloc(1, sizeof *connreq);
	if (!connreq)
		goto err;

	connreq->handle.fclass = FI_CLASS_CONNREQ;
	connreq->id = event->id;
	info->handle = &connreq->handle;
	return info;
err:
	fi_freeinfo(info);
	return NULL;
}

static ssize_t
fi_ibv_eq_cm_process_event(struct fi_ibv_eq *eq, struct rdma_cm_event *cma_event,
	uint32_t *event, struct fi_eq_cm_entry *entry, size_t len)
{
	fid_t fid;
	size_t datalen;

	fid = cma_event->id->context;
	switch (cma_event->event) {
//	case RDMA_CM_EVENT_ADDR_RESOLVED:
//		return 0;
//	case RDMA_CM_EVENT_ROUTE_RESOLVED:
//		return 0;
	case RDMA_CM_EVENT_CONNECT_REQUEST:
		*event = FI_CONNREQ;
		entry->info = fi_ibv_eq_cm_getinfo(eq->fab, cma_event);
		if (!entry->info) {
			rdma_destroy_id(cma_event->id);
			return 0;
		}
		break;
	case RDMA_CM_EVENT_ESTABLISHED:
		*event = FI_CONNECTED;
		entry->info = NULL;
		break;
	case RDMA_CM_EVENT_DISCONNECTED:
		*event = FI_SHUTDOWN;
		entry->info = NULL;
		break;
	case RDMA_CM_EVENT_ADDR_ERROR:
	case RDMA_CM_EVENT_ROUTE_ERROR:
	case RDMA_CM_EVENT_CONNECT_ERROR:
	case RDMA_CM_EVENT_UNREACHABLE:
		eq->err.fid = fid;
		eq->err.err = cma_event->status;
		return -FI_EAVAIL;
	case RDMA_CM_EVENT_REJECTED:
		eq->err.fid = fid;
		eq->err.err = ECONNREFUSED;
		eq->err.prov_errno = cma_event->status;
		return -FI_EAVAIL;
	case RDMA_CM_EVENT_DEVICE_REMOVAL:
		eq->err.fid = fid;
		eq->err.err = ENODEV;
		return -FI_EAVAIL;
	case RDMA_CM_EVENT_ADDR_CHANGE:
		eq->err.fid = fid;
		eq->err.err = EADDRNOTAVAIL;
		return -FI_EAVAIL;
	default:
		return 0;
	}

	entry->fid = fid;
	datalen = MIN(len - sizeof(*entry), cma_event->param.conn.private_data_len);
	if (datalen)
		memcpy(entry->data, cma_event->param.conn.private_data, datalen);
	return sizeof(*entry) + datalen;
}

static ssize_t fi_ibv_eq_write_event(struct fi_ibv_eq *eq, uint32_t event,
		const void *buf, size_t len)
{
	struct fi_ibv_eq_entry *entry;

	entry = calloc(1, sizeof(struct fi_ibv_eq_entry) + len);
	if (!entry)
		return -FI_ENOMEM;

	entry->event = event;
	entry->len = len;
	memcpy(entry->eq_entry, buf, len);

	fastlock_acquire(&eq->lock);
	dlistfd_insert_tail(&entry->item, &eq->list_head);
	fastlock_release(&eq->lock);

	return len;
}

ssize_t fi_ibv_eq_write(struct fid_eq *eq_fid, uint32_t event,
		const void *buf, size_t len, uint64_t flags)
{
	struct fi_ibv_eq *eq;

	eq = container_of(eq_fid, struct fi_ibv_eq, eq_fid.fid);
	if (!(eq->flags & FI_WRITE))
		return -FI_EINVAL;

	return fi_ibv_eq_write_event(eq, event, buf, len);
}

static size_t fi_ibv_eq_read_event(struct fi_ibv_eq *eq, uint32_t *event,
		void *buf, size_t len, uint64_t flags)
{
	struct fi_ibv_eq_entry *entry;
	ssize_t ret = 0;

	fastlock_acquire(&eq->lock);

	if (dlistfd_empty(&eq->list_head))
		goto out;

	entry = container_of(eq->list_head.list.next, struct fi_ibv_eq_entry, item);
	if (entry->len > len) {
		ret = -FI_ETOOSMALL;
		goto out;
	}

	ret = entry->len;
	*event = entry->event;
	memcpy(buf, entry->eq_entry, entry->len);

	if (!(flags & FI_PEEK)) {
		dlistfd_remove(eq->list_head.list.next, &eq->list_head);
		free(entry);
	}

out:
	fastlock_release(&eq->lock);
	return ret;
}

static ssize_t
fi_ibv_eq_read(struct fid_eq *eq_fid, uint32_t *event,
	       void *buf, size_t len, uint64_t flags)
{
	struct fi_ibv_eq *eq;
	struct rdma_cm_event *cma_event;
	ssize_t ret = 0;

	eq = container_of(eq_fid, struct fi_ibv_eq, eq_fid.fid);

	if (eq->err.err)
		return -FI_EAVAIL;

	if ((ret = fi_ibv_eq_read_event(eq, event, buf, len, flags)))
		return ret;

	if (eq->channel) {
		ret = rdma_get_cm_event(eq->channel, &cma_event);
		if (ret)
			return -errno;

		if (len < sizeof(struct fi_eq_cm_entry))
			return -FI_ETOOSMALL;

		ret = fi_ibv_eq_cm_process_event(eq, cma_event, event,
				(struct fi_eq_cm_entry *)buf, len);

		if (flags & FI_PEEK)
			fi_ibv_eq_write_event(eq, *event, buf, len);

		rdma_ack_cm_event(cma_event);
		return ret;
	}

	return -FI_EAGAIN;
}

static ssize_t
fi_ibv_eq_sread(struct fid_eq *eq_fid, uint32_t *event,
		void *buf, size_t len, int timeout, uint64_t flags)
{
	struct fi_ibv_eq *eq;
	struct epoll_event events[2];
	ssize_t ret;

	eq = container_of(eq_fid, struct fi_ibv_eq, eq_fid.fid);

	while (1) {
		ret = fi_ibv_eq_read(eq_fid, event, buf, len, flags);
		if (ret && (ret != -FI_EAGAIN))
			return ret;

		ret = epoll_wait(eq->epfd, events, 2, timeout);
		if (ret == 0)
			return -FI_EAGAIN;
		else if (ret < 0)
			return ret;
	};
}

static const char *
fi_ibv_eq_strerror(struct fid_eq *eq, int prov_errno, const void *err_data,
		   char *buf, size_t len)
{
	if (buf && len)
		strncpy(buf, strerror(prov_errno), len);
	return strerror(prov_errno);
}

static struct fi_ops_eq fi_ibv_eq_ops = {
	.size = sizeof(struct fi_ops_eq),
	.read = fi_ibv_eq_read,
	.readerr = fi_ibv_eq_readerr,
	.write = fi_ibv_eq_write,
	.sread = fi_ibv_eq_sread,
	.strerror = fi_ibv_eq_strerror
};

static int fi_ibv_eq_control(fid_t fid, int command, void *arg)
{
	struct fi_ibv_eq *eq;
	int ret = 0;

	eq = container_of(fid, struct fi_ibv_eq, eq_fid.fid);
	switch (command) {
	case FI_GETWAIT:
		if (!eq->epfd) {
			ret = -FI_ENODATA;
			break;
		}
		*(int *) arg = eq->epfd;
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}

	return ret;
}

static int fi_ibv_eq_close(fid_t fid)
{
	struct fi_ibv_eq *eq;
	struct fi_ibv_eq_entry *entry;

	eq = container_of(fid, struct fi_ibv_eq, eq_fid.fid);

	if (eq->channel)
		rdma_destroy_event_channel(eq->channel);

	close(eq->epfd);

	fastlock_acquire(&eq->lock);
	while(!dlistfd_empty(&eq->list_head)) {
		entry = container_of(eq->list_head.list.next, struct fi_ibv_eq_entry, item);
		dlistfd_remove(eq->list_head.list.next, &eq->list_head);
		free(entry);
	}

	dlistfd_head_free(&eq->list_head);
	fastlock_destroy(&eq->lock);
	free(eq);

	return 0;
}

static struct fi_ops fi_ibv_eq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_eq_close,
	.bind = fi_no_bind,
	.control = fi_ibv_eq_control,
	.ops_open = fi_no_ops_open,
};

static int
fi_ibv_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
	       struct fid_eq **eq, void *context)
{
	struct fi_ibv_eq *_eq;
	struct epoll_event event;
	int ret;

	_eq = calloc(1, sizeof *_eq);
	if (!_eq)
		return -ENOMEM;

	_eq->fab = container_of(fabric, struct fi_ibv_fabric, fabric_fid);

	fastlock_init(&_eq->lock);
	ret = dlistfd_head_init(&_eq->list_head);
	if (ret) {
		FI_INFO(&fi_ibv_prov, FI_LOG_EQ, "Unable to initialize dlistfd\n");
		goto err1;
	}

	_eq->epfd = epoll_create1(0);
	if (_eq->epfd < 0) {
		ret = -errno;
		goto err2;
	}

	memset(&event, 0, sizeof(event));
	event.events = EPOLLIN;

	if (epoll_ctl(_eq->epfd, EPOLL_CTL_ADD, _eq->list_head.fd[LIST_READ_FD], &event)) {
		ret = -errno;
		goto err3;
	}

	switch (attr->wait_obj) {
	case FI_WAIT_NONE:
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
		_eq->channel = rdma_create_event_channel();
		if (!_eq->channel) {
			ret = -errno;
			goto err3;
		}

		ret = fi_fd_nonblock(_eq->channel->fd);
		if (ret)
			goto err4;

		if (epoll_ctl(_eq->epfd, EPOLL_CTL_ADD, _eq->channel->fd, &event)) {
			ret = -errno;
			goto err4;
		}

		break;
	default:
		ret = -FI_ENOSYS;
		goto err1;
	}

	_eq->flags = attr->flags;
	_eq->eq_fid.fid.fclass = FI_CLASS_EQ;
	_eq->eq_fid.fid.context = context;
	_eq->eq_fid.fid.ops = &fi_ibv_eq_fi_ops;
	_eq->eq_fid.ops = &fi_ibv_eq_ops;

	*eq = &_eq->eq_fid;
	return 0;
err4:
	if (_eq->channel)
		rdma_destroy_event_channel(_eq->channel);
err3:
	close(_eq->epfd);
err2:
	dlistfd_head_free(&_eq->list_head);
err1:
	fastlock_destroy(&_eq->lock);
	free(_eq);
	return ret;
}

static ssize_t
fi_ibv_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *entry,
		  uint64_t flags)
{
	struct fi_ibv_cq *_cq;

	_cq = container_of(cq, struct fi_ibv_cq, cq_fid);
	if (!_cq->wc.status)
		return 0;

	entry->op_context = (void *) (uintptr_t) _cq->wc.wr_id;
	entry->flags = 0;
	entry->err = EIO;
	entry->prov_errno = _cq->wc.status;
	memcpy(&entry->err_data, &_cq->wc.vendor_err,
	       sizeof(_cq->wc.vendor_err));

	_cq->wc.status = 0;
	return sizeof(*entry);
}

static inline int
fi_ibv_poll_events(struct fi_ibv_cq *_cq, int timeout)
{
	int ret;
	void *context;
	struct pollfd fds[2];
	char data;

	fds[0].fd = _cq->channel->fd;
	fds[1].fd = _cq->signal_fd[0];

	fds[0].events = fds[1].events = POLLIN;

	ret = poll(fds, 2, timeout);
	if (ret == 0)
		return -FI_EAGAIN;
	else if (ret < 0)
		return -errno;

	if (fds[1].revents & POLLIN) {
		do {
			ret = read(fds[1].fd, &data, 1);
		} while (ret > 0);
		return -FI_EAGAIN;
	} else if (fds[0].revents & POLLIN) {
		ret = ibv_get_cq_event(_cq->channel, &_cq->cq, &context);
		if (ret)
			return ret;

		ibv_ack_cq_events(_cq->cq, 1);
	} else {
		FI_WARN(&fi_ibv_prov, FI_LOG_CQ, "Unknown poll error: check revents\n");
		return -FI_EOTHER;
	}

	return 0;
}

static ssize_t
fi_ibv_cq_sread(struct fid_cq *cq, void *buf, size_t count, const void *cond,
		int timeout)
{
	ssize_t ret = 0, cur;
	ssize_t  threshold;
	struct fi_ibv_cq *_cq;

	_cq = container_of(cq, struct fi_ibv_cq, cq_fid);

	if (!_cq->channel)
		return -FI_ENOSYS;

	threshold = (_cq->wait_cond == FI_CQ_COND_THRESHOLD) ?
		MIN((ssize_t) cond, count) : 1;

	for (cur = 0; cur < threshold; ) {
		ret = _cq->cq_fid.ops->read(&_cq->cq_fid, buf, count - cur);
		if (ret > 0) {
			buf += ret * _cq->entry_size;
			cur += ret;
			if (cur >= threshold)
				break;
		} else if (ret != -FI_EAGAIN) {
			break;
		}

		ret = ibv_req_notify_cq(_cq->cq, 0);
		if (ret) {
			FI_WARN(&fi_ibv_prov, FI_LOG_CQ, "ibv_req_notify_cq error: %d\n", ret);
			break;
		}

		/* Read again to fetch any completions that we might have missed
		 * while rearming */
		ret = _cq->cq_fid.ops->read(&_cq->cq_fid, buf, count - cur);
		if (ret > 0) {
			buf += ret * _cq->entry_size;
			cur += ret;
			if (cur >= threshold)
				break;
		} else if (ret != -FI_EAGAIN) {
			break;
		}

		ret = fi_ibv_poll_events(_cq, timeout);
		if (ret)
			break;
	}

	return cur ? cur : ret;
}

static uint64_t fi_ibv_comp_flags(struct ibv_wc *wc)
{
	uint64_t flags = 0;

	if (wc->wc_flags & IBV_WC_WITH_IMM)
		flags |= FI_REMOTE_CQ_DATA;

	switch (wc->opcode) {
	case IBV_WC_SEND:
		flags |= FI_SEND | FI_MSG;
		break;
	case IBV_WC_RDMA_WRITE:
		flags |= FI_RMA | FI_WRITE;
		break;
	case IBV_WC_RDMA_READ:
		flags |= FI_RMA | FI_READ;
		break;
	case IBV_WC_COMP_SWAP:
		flags |= FI_ATOMIC;
		break;
	case IBV_WC_FETCH_ADD:
		flags |= FI_ATOMIC;
		break;
	case IBV_WC_RECV:
		flags |= FI_RECV | FI_MSG;
		break;
	case IBV_WC_RECV_RDMA_WITH_IMM:
		flags |= FI_RMA | FI_REMOTE_WRITE;
		break;
	default:
		break;
	}
	return flags;
}

static ssize_t fi_ibv_cq_read_context(struct fid_cq *cq, void *buf, size_t count)
{
	struct fi_ibv_cq *_cq;
	struct fi_cq_entry *entry = buf;
	ssize_t ret = 0, i;

	_cq = container_of(cq, struct fi_ibv_cq, cq_fid);
	if (_cq->wc.status)
		return -FI_EAVAIL;

	for (i = 0; i < count; i++) {
		ret = ibv_poll_cq(_cq->cq, 1, &_cq->wc);
		if (ret <= 0)
			break;

		if (_cq->wc.status) {
			ret = -FI_EAVAIL;
			break;
		}

		entry->op_context = (void *) (uintptr_t) _cq->wc.wr_id;
		entry += 1;
	}

	return i ? i : (ret ? ret : -FI_EAGAIN);
}

static ssize_t fi_ibv_cq_read_msg(struct fid_cq *cq, void *buf, size_t count)
{
	struct fi_ibv_cq *_cq;
	struct fi_cq_msg_entry *entry = buf;
	ssize_t ret = 0, i;

	_cq = container_of(cq, struct fi_ibv_cq, cq_fid);
	if (_cq->wc.status)
		return -FI_EAVAIL;

	for (i = 0; i < count; i++) {
		ret = ibv_poll_cq(_cq->cq, 1, &_cq->wc);
		if (ret <= 0)
			break;

		if (_cq->wc.status) {
			ret = -FI_EAVAIL;
			break;
		}

		entry->op_context = (void *) (uintptr_t) _cq->wc.wr_id;
		entry->flags = fi_ibv_comp_flags(&_cq->wc);
		entry->len = (uint64_t) _cq->wc.byte_len;
		entry += 1;
	}

	return i ? i : (ret ? ret : -FI_EAGAIN);
}

static ssize_t fi_ibv_cq_read_data(struct fid_cq *cq, void *buf, size_t count)
{
	struct fi_ibv_cq *_cq;
	struct fi_cq_data_entry *entry = buf;
	ssize_t ret = 0, i;

	_cq = container_of(cq, struct fi_ibv_cq, cq_fid);
	if (_cq->wc.status)
		return -FI_EAVAIL;

	for (i = 0; i < count; i++) {
		ret = ibv_poll_cq(_cq->cq, 1, &_cq->wc);
		if (ret <= 0)
			break;

		if (_cq->wc.status) {
			ret = -FI_EAVAIL;
			break;
		}

		entry->op_context = (void *) (uintptr_t) _cq->wc.wr_id;
		entry->flags = fi_ibv_comp_flags(&_cq->wc);
		if (_cq->wc.wc_flags & IBV_WC_WITH_IMM) {
			entry->data = ntohl(_cq->wc.imm_data);
		} else {
			entry->data = 0;
		}
		if (_cq->wc.opcode & (IBV_WC_RECV | IBV_WC_RECV_RDMA_WITH_IMM))
			entry->len = _cq->wc.byte_len;
		else
			entry->len = 0;

		entry += 1;
	}

	return i ? i : (ret ? ret : -FI_EAGAIN);
}

static const char *
fi_ibv_cq_strerror(struct fid_cq *eq, int prov_errno, const void *err_data,
		   char *buf, size_t len)
{
	if (buf && len)
		strncpy(buf, ibv_wc_status_str(prov_errno), len);
	return ibv_wc_status_str(prov_errno);
}

static int
fi_ibv_cq_signal(struct fid_cq *cq)
{
	struct fi_ibv_cq *_cq;
	char data = '0';

	_cq = container_of(cq, struct fi_ibv_cq, cq_fid);

	if (write(_cq->signal_fd[1], &data, 1) != 1) {
		FI_WARN(&fi_ibv_prov, FI_LOG_CQ, "Error signalling CQ\n");
		return -errno;
	}

	return 0;
}

static struct fi_ops_cq fi_ibv_cq_context_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = fi_ibv_cq_read_context,
	.readfrom = fi_no_cq_readfrom,
	.readerr = fi_ibv_cq_readerr,
	.sread = fi_ibv_cq_sread,
	.sreadfrom = fi_no_cq_sreadfrom,
	.signal = fi_ibv_cq_signal,
	.strerror = fi_ibv_cq_strerror
};

static struct fi_ops_cq fi_ibv_cq_msg_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = fi_ibv_cq_read_msg,
	.readfrom = fi_no_cq_readfrom,
	.readerr = fi_ibv_cq_readerr,
	.sread = fi_ibv_cq_sread,
	.sreadfrom = fi_no_cq_sreadfrom,
	.signal = fi_ibv_cq_signal,
	.strerror = fi_ibv_cq_strerror
};

static struct fi_ops_cq fi_ibv_cq_data_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = fi_ibv_cq_read_data,
	.readfrom = fi_no_cq_readfrom,
	.readerr = fi_ibv_cq_readerr,
	.sread = fi_ibv_cq_sread,
	.sreadfrom = fi_no_cq_sreadfrom,
	.signal = fi_ibv_cq_signal,
	.strerror = fi_ibv_cq_strerror
};

static int fi_ibv_cq_control(fid_t fid, int command, void *arg)
{
	struct fi_ibv_cq *cq;
	int ret = 0;

	cq = container_of(fid, struct fi_ibv_cq, cq_fid.fid);
	switch(command) {
	case FI_GETWAIT:
		if (!cq->channel) {
			ret = -FI_ENODATA;
			break;
		}
		*(int *) arg = cq->channel->fd;
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}

	return ret;
}

static int fi_ibv_cq_close(fid_t fid)
{
	struct fi_ibv_cq *cq;
	int ret;

	cq = container_of(fid, struct fi_ibv_cq, cq_fid.fid);
	if (cq->cq) {
		ret = ibv_destroy_cq(cq->cq);
		if (ret)
			return -ret;
	}

	if (cq->signal_fd[0]) {
		close(cq->signal_fd[0]);
	}
	if (cq->signal_fd[1]) {
		close(cq->signal_fd[1]);
	}

	if (cq->channel)
		ibv_destroy_comp_channel(cq->channel);

	free(cq);
	return 0;
}

static struct fi_ops fi_ibv_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_cq_close,
	.bind = fi_no_bind,
	.control = fi_ibv_cq_control,
	.ops_open = fi_no_ops_open,
};

static int
fi_ibv_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
	    struct fid_cq **cq, void *context)
{
	struct fi_ibv_cq *_cq;
	int ret;

	_cq = calloc(1, sizeof *_cq);
	if (!_cq)
		return -FI_ENOMEM;

	_cq->domain = container_of(domain, struct fi_ibv_domain, domain_fid);

	switch (attr->wait_obj) {
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
		_cq->channel = ibv_create_comp_channel(_cq->domain->verbs);
		if (!_cq->channel) {
			ret = -errno;
			goto err1;
		}

		ret = fi_fd_nonblock(_cq->channel->fd);
		if (ret)
			goto err2;

		if (socketpair(AF_UNIX, SOCK_STREAM, 0, _cq->signal_fd)) {
			ret = -errno;
			goto err2;
		}

		ret = fi_fd_nonblock(_cq->signal_fd[0]);
		if (ret)
			goto err3;

		break;
	case FI_WAIT_NONE:
		break;
	default:
		ret = -FI_ENOSYS;
		goto err3;
	}

	_cq->cq = ibv_create_cq(_cq->domain->verbs, attr->size, _cq,
				_cq->channel, attr->signaling_vector);
	if (!_cq->cq) {
		ret = -errno;
		goto err3;
	}

	if (_cq->channel) {
		ret = ibv_req_notify_cq(_cq->cq, 0);
		if (ret) {
			FI_WARN(&fi_ibv_prov, FI_LOG_CQ, "ibv_req_notify_cq failed\n");
			goto err4;
		}
	}

	_cq->flags |= attr->flags;
	_cq->wait_cond = attr->wait_cond;
	_cq->cq_fid.fid.fclass = FI_CLASS_CQ;
	_cq->cq_fid.fid.context = context;
	_cq->cq_fid.fid.ops = &fi_ibv_cq_fi_ops;

	switch (attr->format) {
	case FI_CQ_FORMAT_CONTEXT:
		_cq->cq_fid.ops = &fi_ibv_cq_context_ops;
		_cq->entry_size = sizeof(struct fi_cq_entry);
		break;
	case FI_CQ_FORMAT_MSG:
		_cq->cq_fid.ops = &fi_ibv_cq_msg_ops;
		_cq->entry_size = sizeof(struct fi_cq_msg_entry);
		break;
	case FI_CQ_FORMAT_DATA:
		_cq->cq_fid.ops = &fi_ibv_cq_data_ops;
		_cq->entry_size = sizeof(struct fi_cq_data_entry);
		break;
	default:
		ret = -FI_ENOSYS;
		goto err4;
	}

	*cq = &_cq->cq_fid;
	return 0;

err4:
	ibv_destroy_cq(_cq->cq);
err3:
	close(_cq->signal_fd[0]);
	close(_cq->signal_fd[1]);
err2:
	if (_cq->channel)
		ibv_destroy_comp_channel(_cq->channel);
err1:
	free(_cq);
	return ret;
}

static int fi_ibv_mr_close(fid_t fid)
{
	struct fi_ibv_mem_desc *mr;
	int ret;

	mr = container_of(fid, struct fi_ibv_mem_desc, mr_fid.fid);
	ret = -ibv_dereg_mr(mr->mr);
	if (!ret)
		free(mr);
	return ret;
}

static struct fi_ops fi_ibv_mr_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_mr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int
fi_ibv_mr_reg(struct fid *fid, const void *buf, size_t len,
	   uint64_t access, uint64_t offset, uint64_t requested_key,
	   uint64_t flags, struct fid_mr **mr, void *context)
{
	struct fi_ibv_mem_desc *md;
	int fi_ibv_access;
	struct fid_domain *domain;

	if (flags)
		return -FI_EBADFLAGS;

	if (fid->fclass != FI_CLASS_DOMAIN) {
		return -FI_EINVAL;
	}
	domain = container_of(fid, struct fid_domain, fid);

	md = calloc(1, sizeof *md);
	if (!md)
		return -FI_ENOMEM;

	md->domain = container_of(domain, struct fi_ibv_domain, domain_fid);
	md->mr_fid.fid.fclass = FI_CLASS_MR;
	md->mr_fid.fid.context = context;
	md->mr_fid.fid.ops = &fi_ibv_mr_ops;

	fi_ibv_access = IBV_ACCESS_LOCAL_WRITE;
	if (access & FI_REMOTE_READ)
		fi_ibv_access |= IBV_ACCESS_REMOTE_READ;
	if (access & FI_REMOTE_WRITE)
		fi_ibv_access |= IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

	md->mr = ibv_reg_mr(md->domain->pd, (void *) buf, len, fi_ibv_access);
	if (!md->mr)
		goto err;

	md->mr_fid.mem_desc = (void *) (uintptr_t) md->mr->lkey;
	md->mr_fid.key = md->mr->rkey;
	*mr = &md->mr_fid;
	return 0;

err:
	free(md);
	return -errno;
}

static int fi_ibv_close(fid_t fid)
{
	struct fi_ibv_domain *domain;
	int ret;

	domain = container_of(fid, struct fi_ibv_domain, domain_fid.fid);
	if (domain->pd) {
		ret = ibv_dealloc_pd(domain->pd);
		if (ret)
			return -ret;
		domain->pd = NULL;
	}

	free(domain);
	return 0;
}

static int fi_ibv_open_device_by_name(struct fi_ibv_domain *domain, const char *name)
{
	struct ibv_context **dev_list;
	int i, ret = -FI_ENODEV;

	if (!name)
		return -FI_EINVAL;

	dev_list = rdma_get_devices(NULL);
	if (!dev_list)
		return -errno;

	for (i = 0; dev_list[i]; i++) {
		if (!strcmp(name, ibv_get_device_name(dev_list[i]->device))) {
			domain->verbs = dev_list[i];
			ret = 0;
			break;
		}
	}
	rdma_free_devices(dev_list);
	return ret;
}

static struct fi_ops fi_ibv_fid_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_mr fi_ibv_domain_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = fi_ibv_mr_reg,
	.regv = fi_no_mr_regv,
	.regattr = fi_no_mr_regattr,
};

static struct fi_ops_domain fi_ibv_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = fi_no_av_open,
	.cq_open = fi_ibv_cq_open,
	.endpoint = fi_ibv_open_ep,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = fi_no_cntr_open,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
};

static int
fi_ibv_domain(struct fid_fabric *fabric, struct fi_info *info,
	   struct fid_domain **domain, void *context)
{
	struct fi_ibv_domain *_domain;
	struct fi_info *fi;
	int ret;

	fi = fi_ibv_search_verbs_info(NULL, info->domain_attr->name);
	if (!fi)
		return -FI_EINVAL;

	ret = fi_ibv_check_domain_attr(info->domain_attr, fi);
	if (ret)
		return ret;

	_domain = calloc(1, sizeof *_domain);
	if (!_domain)
		return -FI_ENOMEM;

	ret = fi_ibv_open_device_by_name(_domain, info->domain_attr->name);
	if (ret)
		goto err;

	_domain->pd = ibv_alloc_pd(_domain->verbs);
	if (!_domain->pd) {
		ret = -errno;
		goto err;
	}

	_domain->domain_fid.fid.fclass = FI_CLASS_DOMAIN;
	_domain->domain_fid.fid.context = context;
	_domain->domain_fid.fid.ops = &fi_ibv_fid_ops;
	_domain->domain_fid.ops = &fi_ibv_domain_ops;
	_domain->domain_fid.mr = &fi_ibv_domain_mr_ops;

	*domain = &_domain->domain_fid;
	return 0;
err:
	free(_domain);
	return ret;
}

static int fi_ibv_pep_setname(fid_t pep_fid, void *addr, size_t addrlen)
{
	struct fi_ibv_pep *pep;
	int ret;

	pep = container_of(pep_fid, struct fi_ibv_pep, pep_fid);

	if (pep->src_addrlen && (addrlen != pep->src_addrlen)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_FABRIC, "addrlen expected: %d, got: %d.\n",
				pep->src_addrlen, addrlen);
		return -FI_EINVAL;
	}

	/* Re-create id if already bound */
	if (pep->bound) {
		ret = rdma_destroy_id(pep->id);
		if (ret) {
			FI_INFO(&fi_ibv_prov, FI_LOG_FABRIC, "Unable to destroy previous rdma_cm_id\n");
			return -errno;
		}
		ret = rdma_create_id(NULL, &pep->id, NULL, RDMA_PS_TCP);
		if (ret) {
			FI_INFO(&fi_ibv_prov, FI_LOG_FABRIC, "Unable to create rdma_cm_id\n");
			return -errno;
		}
	}

	ret = rdma_bind_addr(pep->id, (struct sockaddr *)addr);
	if (ret) {
		FI_INFO(&fi_ibv_prov, FI_LOG_FABRIC, "Unable to bind addres to rdma_cm_id\n");
		return -errno;
	}

	return 0;
}

static int fi_ibv_pep_getname(fid_t pep, void *addr, size_t *addrlen)
{
	struct fi_ibv_pep *_pep;
	struct sockaddr *sa;

	_pep = container_of(pep, struct fi_ibv_pep, pep_fid);
	sa = rdma_get_local_addr(_pep->id);
	return fi_ibv_copy_addr(addr, addrlen, sa);
}

static int fi_ibv_pep_listen(struct fid_pep *pep_fid)
{
	struct fi_ibv_pep *pep;
	struct sockaddr *addr;

	pep = container_of(pep_fid, struct fi_ibv_pep, pep_fid);

	addr = rdma_get_local_addr(pep->id);
	if (addr) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "Listening on %s:%d\n",
			inet_ntoa(((struct sockaddr_in *)addr)->sin_addr),
			ntohs(((struct sockaddr_in *)addr)->sin_port));
	}

	return rdma_listen(pep->id, 0) ? -errno : 0;
}

static struct fi_ops_cm fi_ibv_pep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_ibv_pep_setname,
	.getname = fi_ibv_pep_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_ibv_pep_listen,
	.accept = fi_no_accept,
	.reject = fi_ibv_msg_ep_reject,
	.shutdown = fi_no_shutdown,
};

static int fi_ibv_pep_bind(fid_t fid, struct fid *bfid, uint64_t flags)
{
	struct fi_ibv_pep *pep;
	int ret;

	pep = container_of(fid, struct fi_ibv_pep, pep_fid.fid);
	if (bfid->fclass != FI_CLASS_EQ)
		return -FI_EINVAL;

	pep->eq = container_of(bfid, struct fi_ibv_eq, eq_fid.fid);
	ret = rdma_migrate_id(pep->id, pep->eq->channel);
	if (ret)
		return -errno;

	return 0;
}

static int fi_ibv_pep_close(fid_t fid)
{
	struct fi_ibv_pep *pep;

	pep = container_of(fid, struct fi_ibv_pep, pep_fid.fid);
	if (pep->id)
		rdma_destroy_ep(pep->id);

	free(pep);
	return 0;
}

static struct fi_ops fi_ibv_pep_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_pep_close,
	.bind = fi_ibv_pep_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int
fi_ibv_passive_ep(struct fid_fabric *fabric, struct fi_info *info,
	      struct fid_pep **pep, void *context)
{
	struct fi_ibv_pep *_pep;
	int ret;

	_pep = calloc(1, sizeof *_pep);
	if (!_pep)
		return -FI_ENOMEM;

	ret = rdma_create_id(NULL, &_pep->id, NULL, RDMA_PS_TCP);
	if (ret) {
		FI_INFO(&fi_ibv_prov, FI_LOG_DOMAIN, "Unable to create rdma_cm_id\n");
		goto err1;
	}

	if (info->src_addr) {
		ret = rdma_bind_addr(_pep->id, (struct sockaddr *)info->src_addr);
		if (ret) {
			FI_INFO(&fi_ibv_prov, FI_LOG_DOMAIN, "Unable to bind addres to rdma_cm_id\n");
			goto err2;
		}
		_pep->bound = 1;
	}

	_pep->id->context = &_pep->pep_fid.fid;

	_pep->pep_fid.fid.fclass = FI_CLASS_PEP;
	_pep->pep_fid.fid.context = context;
	_pep->pep_fid.fid.ops = &fi_ibv_pep_ops;
	_pep->pep_fid.cm = &fi_ibv_pep_cm_ops;

	_pep->src_addrlen = info->src_addrlen;

	*pep = &_pep->pep_fid;
	return 0;

err2:
	rdma_destroy_id(_pep->id);
err1:
	free(_pep);
	return ret;
}

static int fi_ibv_fabric_close(fid_t fid)
{
	free(fid);
	return 0;
}

static struct fi_ops fi_ibv_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_fabric fi_ibv_ops_fabric = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = fi_ibv_domain,
	.passive_ep = fi_ibv_passive_ep,
	.eq_open = fi_ibv_eq_open,
	.wait_open = fi_no_wait_open,
};

static int fi_ibv_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
			 void *context)
{
	struct fi_ibv_fabric *fab;
	struct fi_info *info;
	int ret;

	ret = fi_ibv_init_info();
	if (ret)
		return ret;

	info = fi_ibv_search_verbs_info(attr->name, NULL);
	if (!info)
		return -FI_ENODATA;

	ret = fi_ibv_check_fabric_attr(attr, info);
	if (ret)
		return -FI_ENODATA;

	fab = calloc(1, sizeof(*fab));
	if (!fab)
		return -FI_ENOMEM;

	fab->fabric_fid.fid.fclass = FI_CLASS_FABRIC;
	fab->fabric_fid.fid.context = context;
	fab->fabric_fid.fid.ops = &fi_ibv_fi_ops;
	fab->fabric_fid.ops = &fi_ibv_ops_fabric;
	*fabric = &fab->fabric_fid;
	return 0;
}

static void fi_ibv_fini(void)
{
	fi_freeinfo(verbs_info);
}

VERBS_INI
{
	return &fi_ibv_prov;
}
