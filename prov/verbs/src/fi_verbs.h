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

#ifndef _FI_VERBS_H
#define _FI_VERBS_H

#if HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdlib.h>
#include <netinet/in.h>
#include <infiniband/verbs.h>

#include <fi_list.h>
#include <rdma/fabric.h>

#include <prov/verbs/src/verbs_utils.h>
#include <prov/verbs/src/uthash.h>

#define VERBS_PROV_NAME "verbs"
#define VERBS_PROV_VERS FI_VERSION(1,0)

#define VERBS_IB_PREFIX "IB-0x"
#define VERBS_IWARP_FABRIC "Ethernet-iWARP"
#define VERBS_ANY_FABRIC "Any RDMA fabric"
#define VERBS_CM_DATA_SIZE 56
#define VERBS_RESOLVE_TIMEOUT 2000	// ms

#define VERBS_EP_RDM_CAPS (FI_SEND | FI_RECV | FI_TAGGED)

#define VERBS_EP_MSG_CAPS (FI_MSG | FI_RMA | FI_ATOMICS | FI_READ | FI_WRITE | \
                    FI_SEND | FI_RECV | FI_REMOTE_READ | FI_REMOTE_WRITE)

#define VERBS_CAPS (VERBS_EP_RDM_CAPS | VERBS_EP_MSG_CAPS)

#define VERBS_MODE (FI_LOCAL_MR)
#define VERBS_EP_RDM_MODE (0)

#define VERBS_TX_OP_FLAGS (FI_INJECT | FI_COMPLETION | FI_TRANSMIT_COMPLETE)
#define VERBS_TX_OP_FLAGS_IWARP (FI_INJECT | FI_COMPLETION)
#define VERBS_TX_MODE VERBS_MODE
#define VERBS_RX_MODE (FI_LOCAL_MR | FI_RX_CQ_DATA)
#define VERBS_MSG_ORDER (FI_ORDER_RAR | FI_ORDER_RAW | FI_ORDER_RAS | \
		FI_ORDER_WAW | FI_ORDER_WAS | FI_ORDER_SAW | FI_ORDER_SAS )

#define VERBS_INJECT_FLAGS(ep, len, flags) (((flags & FI_INJECT) || \
		len <= ep->info->tx_attr->inject_size) ? IBV_SEND_INLINE : 0)
#define VERBS_INJECT(ep, len) VERBS_INJECT_FLAGS(ep, len, ep->info->tx_attr->op_flags)

#define VERBS_SELECTIVE_COMP(ep) (ep->ep_flags & FI_SELECTIVE_COMPLETION)

#define VERBS_COMP_FLAGS(ep, flags) ((!VERBS_SELECTIVE_COMP(ep) || \
		(flags & (FI_COMPLETION | FI_TRANSMIT_COMPLETE))) ? \
		IBV_SEND_SIGNALED : 0)

#define VERBS_COMP(ep) VERBS_COMP_FLAGS(ep, ep->info->tx_attr->op_flags)

#define VERBS_COMP_READ_FLAGS(ep, flags) ((!VERBS_SELECTIVE_COMP(ep) || \
		(flags & (FI_COMPLETION | FI_TRANSMIT_COMPLETE | FI_DELIVERY_COMPLETE))) ? \
		IBV_SEND_SIGNALED : 0)

#define VERBS_COMP_READ(ep) VERBS_COMP_READ_FLAGS(ep, ep->info->tx_attr->op_flags)

#define VERBS_DBG(subsys, ...) FI_DBG(&fi_ibv_prov, subsys, __VA_ARGS__)
#define VERBS_INFO(subsys, ...) FI_INFO(&fi_ibv_prov, subsys, __VA_ARGS__)

#define VERBS_INFO_ERRNO(subsys, fn, errno) VERBS_INFO(subsys, fn ": %s(%d)\n",	\
		strerror(errno), errno)

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
    struct fi_ibv_rdm_ep *ep;
    int format;
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

struct fi_ibv_av {
    struct fid_av av;
    struct fi_ibv_domain *domain;
    struct fi_ibv_rdm_ep *ep;
    int type;
    size_t count;
};

#define fi_ibv_memcpy fi_ibv_memcpy_impl
#define fi_ibv_memset fi_ibv_memset_impl

typedef long long int fi_ibv_memcpy_chunk_type_t;

static inline void *fi_ibv_memcpy_impl(void *dst, const void *src, size_t size)
{
    size_t i = 0;
    size_t body_offset = 0;

    size_t body_size = size / sizeof(fi_ibv_memcpy_chunk_type_t);
    size_t size_tail = size % sizeof(fi_ibv_memcpy_chunk_type_t);

    for (i = 0; i < body_size; i++) {
        *((fi_ibv_memcpy_chunk_type_t *) dst + i) =
            *((fi_ibv_memcpy_chunk_type_t *) src + i);
    }

    body_offset = body_size * sizeof(fi_ibv_memcpy_chunk_type_t);

    for (i = 0; i < size_tail; i++) {
        *((unsigned char *)dst + i + body_offset) =
            *((unsigned char *)src + i + body_offset);
    }

    return dst;
}

static inline void *fi_ibv_memset_impl(void *dst, int c, size_t size)
{
    size_t i = 0;

    for (i = 0; i < size; i++) {
        *((unsigned char *)dst + i) = (unsigned char)c;
    }

    return dst;
}

/* Common verbs */
int fi_ibv_getinfo(uint32_t version, const char *node, const char *service,
                   uint64_t flags, struct fi_info *hints,
                   struct fi_info **info);
int fi_ibv_init_info(int init_ep_type);
void fi_ibv_update_info(const struct fi_info *hints, struct fi_info *info);
struct fi_info *fi_ibv_search_verbs_info(const char *fabric_name,
                                         const char *domain_name);
int fi_ibv_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
                  void *context);
int fi_ibv_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
                   struct fid_eq **eq, void *context);
int fi_ibv_domain(struct fid_fabric *fabric, struct fi_info *info,
                  struct fid_domain **domain, void *context);
int fi_ibv_passive_ep(struct fid_fabric *fabric, struct fi_info *info,
                      struct fid_pep **pep, void *context);
int fi_ibv_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
                   struct fid_cq **cq, void *context);
int fi_ibv_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
                   struct fid_av **av, void *context);

/* EP_MSG */
int fi_ibv_set_cq_ops_ep_msg(struct fi_ibv_cq *cq);
int fi_ibv_msg_open_ep(struct fid_domain *domain, struct fi_info *info,
                       struct fid_ep **ep, void *context);

#endif /* _FI_VERBS_H */
