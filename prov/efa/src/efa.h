/*
 * Copyright (c) 2018-2022 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#ifndef EFA_H
#define EFA_H

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

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

#include <infiniband/verbs.h>
#include <infiniband/efadv.h>

#include "ofi.h"
#include "ofi_enosys.h"
#include "ofi_list.h"
#include "ofi_util.h"
#include "ofi_file.h"

#include "efa_base_ep.h"
#include "dgram/efa_dgram.h"
#include "efa_mr.h"
#include "efa_shm.h"
#include "efa_hmem.h"
#include "efa_device.h"
#include "efa_domain.h"
#include "efa_errno.h"
#include "efa_user_info.h"
#include "efa_fork_support.h"
#include "rdm/efa_rdm_peer.h"
#include "rdm/rxr.h"

#define EFA_PROV_NAME "efa"

#define EFA_ABI_VER_MAX_LEN 8

#define EFA_WCE_CNT 1024

#define EFA_EP_TYPE_IS_RDM(_info) \
	(_info && _info->ep_attr && (_info->ep_attr->type == FI_EP_RDM))

#define EFA_EP_TYPE_IS_DGRAM(_info) \
	(_info && _info->ep_attr && (_info->ep_attr->type == FI_EP_DGRAM))

extern struct fi_provider efa_prov;
extern struct util_prov efa_util_prov;

#define EFA_WARN(subsys, ...) FI_WARN(&efa_prov, subsys, __VA_ARGS__)
#define EFA_TRACE(subsys, ...) FI_TRACE(&efa_prov, subsys, __VA_ARGS__)
#define EFA_INFO(subsys, ...) FI_INFO(&efa_prov, subsys, __VA_ARGS__)
#define EFA_INFO_ERRNO(subsys, fn, errno) \
	EFA_INFO(subsys, fn ": %s(%d)\n", strerror(errno), errno)
#define EFA_WARN_ERRNO(subsys, fn, errno) \
	EFA_WARN(subsys, fn ": %s(%d)\n", strerror(errno), errno)
#define EFA_DBG(subsys, ...) FI_DBG(&efa_prov, subsys, __VA_ARGS__)

#define EFA_DGRAM_CONNID (0x0)

/*
 * Specific flags and attributes for shm provider
 */
/* maximum name length for shm endpoint */
#define EFA_SHM_NAME_MAX	   (256)

#define EFA_DEF_POOL_ALIGNMENT (8)
#define EFA_MEM_ALIGNMENT (64)

#define EFA_DEF_CQ_SIZE 1024
#define EFA_MR_IOV_LIMIT 1
#define EFA_MR_SUPPORTED_PERMISSIONS (FI_SEND | FI_RECV | FI_REMOTE_READ)

/*
 * Setting ibv_qp_attr.rnr_retry to this number when modifying qp
 * to cause firmware to retry indefininetly.
 */
#define EFA_RNR_INFINITE_RETRY 7

/*
 * Multiplier to give some room in the device memory registration limits
 * to allow processes added to a running job to bootstrap.
 */
#define EFA_MR_CACHE_LIMIT_MULT (.9)

#define EFA_MIN_AV_SIZE (16384)

#define EFA_DEFAULT_RUNT_SIZE (307200)
#define EFA_DEFAULT_INTER_MAX_MEDIUM_MESSAGE_SIZE (65536)
#define EFA_DEFAULT_INTER_MIN_READ_MESSAGE_SIZE (1048576)
#define EFA_DEFAULT_INTER_MIN_READ_WRITE_SIZE (65536)

struct efa_fabric {
	struct util_fabric	util_fabric;
	struct fid_fabric *shm_fabric;
#ifdef EFA_PERF_ENABLED
	struct ofi_perfset perf_set;
#endif
};

struct efa_ah {
	uint8_t		gid[EFA_GID_LEN]; /* efa device GID */
	struct ibv_ah	*ibv_ah; /* created by ibv_create_ah() using GID */
	uint16_t	ahn; /* adress handle number */
	int		refcnt; /* reference counter. Multiple efa_conn can share an efa_ah */
	UT_hash_handle	hh; /* hash map handle, link all efa_ah with efa_ep->ah_map */
};

static inline
int efa_str_to_ep_addr(const char *node, const char *service, struct efa_ep_addr *addr)
{
	int ret;

	if (!node)
		return -FI_EINVAL;

	memset(addr, 0, sizeof(*addr));

	ret = inet_pton(AF_INET6, node, addr->raw);
	if (ret != 1)
		return -FI_EINVAL;
	if (service)
		addr->qpn = atoi(service);

	return 0;
}

static inline
bool efa_is_same_addr(struct efa_ep_addr *lhs, struct efa_ep_addr *rhs)
{
	return !memcmp(lhs->raw, rhs->raw, sizeof(lhs->raw)) &&
	       lhs->qpn == rhs->qpn && lhs->qkey == rhs->qkey;
}

struct efa_conn {
	struct efa_ah		*ah;
	struct efa_ep_addr	*ep_addr;
	/* for FI_AV_TABLE, fi_addr is same as util_av_fi_addr,
	 * for FI_AV_MAP, fi_addr is pointer to efa_conn; */
	fi_addr_t		fi_addr;
	fi_addr_t		util_av_fi_addr;
	struct efa_rdm_peer	rdm_peer;
};

struct efa_wc {
	struct ibv_wc		ibv_wc;
	/* Source address */
	uint16_t		efa_ah;
};

struct efa_wce {
	struct slist_entry	entry;
	struct efa_wc		wc;
};

typedef void (*efa_cq_read_entry)(struct ibv_cq_ex *ibv_cqx, int index, void *buf);

struct efa_cq {
	struct util_cq		util_cq;
	struct efa_domain	*domain;
	size_t			entry_size;
	efa_cq_read_entry	read_entry;
	ofi_spin_t		lock;
	struct ofi_bufpool	*wce_pool;
	uint32_t	flags; /* User defined capability mask */

	struct ibv_cq_ex	*ibv_cq_ex;
};

struct efa_av_entry {
	uint8_t			ep_addr[EFA_EP_ADDR_LEN];
	struct efa_conn		conn;
};

struct efa_cur_reverse_av_key {
	uint16_t ahn;
	uint16_t qpn;
};

struct efa_cur_reverse_av {
	struct efa_cur_reverse_av_key key;
	struct efa_conn *conn;
	UT_hash_handle hh;
};

struct efa_prv_reverse_av_key {
	uint16_t ahn;
	uint16_t qpn;
	uint32_t connid;
};

struct efa_prv_reverse_av {
	struct efa_prv_reverse_av_key key;
	struct efa_conn *conn;
	UT_hash_handle hh;
};

static inline struct efa_av *rxr_ep_av(struct rxr_ep *ep)
{
	return container_of(ep->base_ep.util_ep.av, struct efa_av, util_av);
}

extern struct fi_ops_cm efa_ep_cm_ops;
extern struct fi_ops_msg efa_ep_msg_ops;
extern struct fi_ops_rma efa_ep_rma_ops;

ssize_t efa_rma_post_read(struct efa_ep *ep, const struct fi_msg_rma *msg,
			  uint64_t flags, bool self_comm);

const struct fi_info *efa_get_efa_info(const char *domain_name);
int efa_domain_open(struct fid_fabric *fabric_fid, struct fi_info *info,
		    struct fid_domain **domain_fid, void *context);
int efa_ep_open(struct fid_domain *domain_fid, struct fi_info *info,
		struct fid_ep **ep_fid, void *context);
int efa_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		struct fid_av **av_fid, void *context);
int efa_cq_open(struct fid_domain *domain_fid, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context);
int efa_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric_fid,
	       void *context);

/* AV sub-functions */
int efa_av_insert_one(struct efa_av *av, struct efa_ep_addr *addr,
		      fi_addr_t *fi_addr, uint64_t flags, void *context);

struct efa_conn *efa_av_addr_to_conn(struct efa_av *av, fi_addr_t fi_addr);

/* Caller must hold cq->inner_lock. */
void efa_cq_inc_ref_cnt(struct efa_cq *cq, uint8_t sub_cq_idx);
/* Caller must hold cq->inner_lock. */
void efa_cq_dec_ref_cnt(struct efa_cq *cq, uint8_t sub_cq_idx);

fi_addr_t efa_av_reverse_lookup_rdm(struct efa_av *av, uint16_t ahn, uint16_t qpn, struct rxr_pkt_entry *pkt_entry);

fi_addr_t efa_av_reverse_lookup_dgram(struct efa_av *av, uint16_t ahn, uint16_t qpn);

int efa_prov_initialize(void);

void efa_prov_finalize(void);

ssize_t efa_post_flush(struct efa_ep *ep, struct ibv_send_wr **bad_wr, bool free);

ssize_t efa_cq_readfrom(struct fid_cq *cq_fid, void *buf, size_t count, fi_addr_t *src_addr);

ssize_t efa_cq_readerr(struct fid_cq *cq_fid, struct fi_cq_err_entry *entry, uint64_t flags);

/**
 * @brief return whether this endpoint should write error cq entry for RNR.
 *
 * For an endpoint to write RNR completion, two conditions must be met:
 *
 * First, the end point must be able to receive RNR completion from rdma-core,
 * which means rnr_etry must be less then EFA_RNR_INFINITE_RETRY.
 *
 * Second, the app need to request this feature when opening endpoint
 * (by setting info->domain_attr->resource_mgmt to FI_RM_DISABLED).
 * The setting was saved as rxr_ep->handle_resource_management.
 *
 * @param[in]	ep	endpoint
 */
static inline
bool rxr_ep_should_write_rnr_completion(struct rxr_ep *ep)
{
	return (rxr_env.rnr_retry < EFA_RNR_INFINITE_RETRY) &&
		(ep->handle_resource_management == FI_RM_DISABLED);
}

#define RXR_REQ_OPT_HDR_ALIGNMENT 8
#define RXR_REQ_OPT_RAW_ADDR_HDR_SIZE (((sizeof(struct rxr_req_opt_raw_addr_hdr) + EFA_EP_ADDR_LEN - 1)/RXR_REQ_OPT_HDR_ALIGNMENT + 1) * RXR_REQ_OPT_HDR_ALIGNMENT)

/*
 * Per libfabric standard, the prefix must be a multiple of 8, hence the static assert
 */
#define RXR_MSG_PREFIX_SIZE (sizeof(struct rxr_pkt_entry) + sizeof(struct rxr_eager_msgrtm_hdr) + RXR_REQ_OPT_RAW_ADDR_HDR_SIZE)

#if defined(static_assert) && defined(__x86_64__)
static_assert(RXR_MSG_PREFIX_SIZE % 8 == 0, "message prefix size alignment check");
#endif

/* Performance counter declarations */
#ifdef EFA_PERF_ENABLED
#define EFA_PERF_FOREACH(DECL)	\
	DECL(perf_efa_tx),	\
	DECL(perf_efa_recv),	\
	DECL(efa_perf_size)	\

enum efa_perf_counters {
	EFA_PERF_FOREACH(OFI_ENUM_VAL)
};

extern const char *efa_perf_counters_str[];

static inline void efa_perfset_start(struct rxr_ep *ep, size_t index)
{
	struct efa_domain *domain = rxr_ep_domain(ep);
	struct efa_fabric *fabric = container_of(domain->util_domain.fabric,
						 struct efa_fabric,
						 util_fabric);
	ofi_perfset_start(&fabric->perf_set, index);
}

static inline void efa_perfset_end(struct rxr_ep *ep, size_t index)
{
	struct efa_domain *domain = rxr_ep_domain(ep);
	struct efa_fabric *fabric = container_of(domain->util_domain.fabric,
						 struct efa_fabric,
						 util_fabric);
	ofi_perfset_end(&fabric->perf_set, index);
}
#else
#define efa_perfset_start(ep, index) do {} while (0)
#define efa_perfset_end(ep, index) do {} while (0)
#endif

#endif /* EFA_H */
