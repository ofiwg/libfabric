/*
 * Copyright (c) 2018-2019 Amazon.com, Inc. or its affiliates. All rights reserved.
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
#include <uthash.h>

#include "infiniband/efa_arch.h"
#include "infiniband/efa_verbs.h"
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

#include "ofi.h"
#include "ofi_enosys.h"
#include "ofi_list.h"
#include "ofi_util.h"
#include "ofi_file.h"

#define EFA_PROV_NAME "efa"
#define EFA_PROV_VERS FI_VERSION(3, 0)

#define EFA_WARN(subsys, ...) FI_WARN(&efa_prov, subsys, __VA_ARGS__)
#define EFA_TRACE(subsys, ...) FI_TRACE(&efa_prov, subsys, __VA_ARGS__)
#define EFA_INFO(subsys, ...) FI_INFO(&efa_prov, subsys, __VA_ARGS__)
#define EFA_INFO_ERRNO(subsys, fn, errno) \
	EFA_INFO(subsys, fn ": %s(%d)\n", strerror(errno), errno)
#define EFA_WARN_ERRNO(subsys, fn, errno) \
	EFA_WARN(subsys, fn ": %s(%d)\n", strerror(errno), errno)
#define EFA_DBG(subsys, ...) FI_DBG(&efa_prov, subsys, __VA_ARGS__)

#define EFA_ABI_VER_MAX_LEN 8

#define EFA_WCE_CNT 1024

#define EFA_EP_TYPE_IS_RDM(_info) \
	(_info && _info->ep_attr && (_info->ep_attr->type == FI_EP_RDM))

#define EFA_MEM_ALIGNMENT (64)

#define EFA_DEF_CQ_SIZE 1024
#define EFA_MR_IOV_LIMIT 1
#define EFA_MR_SUPPORTED_PERMISSIONS (FI_SEND | FI_RECV)

#define EFA_DEF_NUM_MR_CACHE 36

extern int efa_mr_cache_enable;
extern size_t efa_mr_max_cached_count;
extern size_t efa_mr_max_cached_size;
extern int efa_mr_cache_merge_regions;

extern struct fi_provider efa_prov;
extern struct util_prov efa_util_prov;

struct efa_fabric {
	struct util_fabric	util_fabric;
};

struct efa_ep_addr {
	uint8_t			raw[16];
	uint16_t		qpn;
	struct efa_ep_addr	*next;
};

#define EFA_EP_ADDR_LEN sizeof(struct efa_ep_addr)

struct efa_conn {
	struct efa_ah		*ah;
	struct efa_ep_addr	ep_addr;
};

struct efa_domain {
	struct util_domain	util_domain;
	struct efa_context	*ctx;
	struct efa_pd		*pd;
	struct fi_info		*info;
	struct efa_fabric	*fab;
	int			rdm;
	struct ofi_mr_cache	cache;
};

struct fi_ops_mr efa_domain_mr_ops;
struct fi_ops_mr efa_domain_mr_cache_ops;
int efa_mr_cache_entry_reg(struct ofi_mr_cache *cache,
			   struct ofi_mr_entry *entry);
void efa_mr_cache_entry_dereg(struct ofi_mr_cache *cache,
			      struct ofi_mr_entry *entry);

struct efa_wc {
	uint64_t		wr_id;
	/* Completion flags */
	uint64_t		flags;
	/* Immediate data in network byte order */
	uint64_t		imm_data;
	/* Size of received data */
	uint32_t		byte_len;
	uint32_t		comp_status;
	struct efa_qp		*qp;
	/* Source address */
	uint16_t		efa_ah;
	uint16_t		src_qp;
};

struct efa_wce {
	struct slist_entry	entry;
	struct efa_wc		wc;
};

typedef void (*efa_cq_read_entry)(struct efa_wc *wc, int index, void *buf);

struct efa_sub_cq {
	uint16_t		consumed_cnt;
	int			phase;
	uint8_t			*buf;
	int			qmask;
	int			cqe_size;
	uint32_t		ref_cnt;
};

struct efa_cq {
	struct fid_cq		cq_fid;
	struct efa_domain	*domain;
	size_t			entry_size;
	efa_cq_read_entry	read_entry;
	struct slist		wcq;
	fastlock_t		outer_lock;
	struct ofi_bufpool	*wce_pool;

	struct ibv_cq		ibv_cq;
	uint8_t			*buf;
	size_t			buf_size;
	fastlock_t		inner_lock;
	uint32_t		cqn;
	int			cqe_size;
	struct efa_sub_cq	*sub_cq_arr;
	uint16_t		num_sub_cqs;
	/* Index of next sub cq idx to poll. This is used to guarantee fairness for sub cqs */
	uint16_t		next_poll_idx;
};

struct efa_device {
	struct verbs_device		verbs_dev;
	int				page_size;
	int				abi_version;
};

struct efa_context {
	struct ibv_context	ibv_ctx;
	int			efa_everbs_cmd_fd;
	struct efa_qp		**qp_table;
	pthread_mutex_t		qp_table_mutex;

	int			cqe_size;
	uint16_t		sub_cqs_per_cq;
	uint16_t		inject_size;
	uint32_t		cmds_supp_udata;
	uint32_t		max_llq_size;
	uint64_t		max_mr_size;
};

struct efa_pd {
	struct ibv_pd		ibv_pd;
	struct efa_context	*context;
	uint16_t		pdn;
};

struct efa_wq {
	uint64_t			*wrid;
	/* wrid_idx_pool: Pool of free indexes in the wrid array, used to select the
	 * wrid entry to be used to hold the next tx packet's context.
	 * At init time, entry N will hold value N, as OOO tx-completions arrive,
	 * the value stored in a given entry might not equal the entry's index.
	 */
	uint32_t			*wrid_idx_pool;
	uint32_t			wqe_cnt;
	uint32_t			wqe_posted;
	uint32_t			wqe_completed;
	uint16_t			desc_idx;
	uint16_t			desc_mask;
	/* wrid_idx_pool_next: Index of the next entry to use in wrid_idx_pool. */
	uint16_t			wrid_idx_pool_next;
	int				max_sge;
	int				phase;
};

struct efa_sq {
	struct efa_wq	wq;
	uint32_t	*db;
	uint8_t		*desc;
	uint32_t	desc_offset;
	size_t		desc_ring_mmap_size;
	size_t		max_inline_data;
	size_t		immediate_data_width;
	uint16_t	sub_cq_idx;
};

struct efa_rq {
	struct efa_wq	wq;
	uint32_t	*db;
	uint8_t		*buf;
	size_t		buf_size;
	uint16_t	sub_cq_idx;
};

struct efa_qp {
	struct ibv_qp	ibv_qp;
	struct efa_ep	*ep;
	struct efa_sq	sq;
	struct efa_rq	rq;
	uint32_t	qp_num;
	int		page_size;
};

struct efa_ah {
	struct ibv_ah	ibv_ah;
	uint16_t	efa_address_handle;
};

struct efa_mem_desc {
	struct fid_mr		mr_fid;
	struct ibv_mr		*mr;
	struct efa_domain	*domain;
	/* Used only in MR cache */
	struct ofi_mr_entry	*entry;
};

struct efa_ep {
	struct fid_ep		ep_fid;
	struct efa_domain	*domain;
	struct efa_qp		*qp;
	struct efa_cq		*rcq;
	struct efa_cq		*scq;
	struct efa_av		*av;
	struct fi_info		*info;
	void			*src_addr;
};

typedef struct efa_conn *
	(*efa_addr_to_conn_func)
	(struct efa_av *av, fi_addr_t addr);

struct efa_av {
	struct fid_av		av_fid;
	struct efa_domain	*domain;
	struct efa_ep		*ep;
	size_t			count;
	size_t			used;
	size_t			next;
	uint64_t		flags;
	enum fi_av_type		type;
	efa_addr_to_conn_func	addr_to_conn;
	struct efa_reverse_av	*reverse_av;
	/* Used only for FI_AV_TABLE */
	struct efa_conn **conn_table;
};

struct efa_ah_qpn {
	uint16_t efa_ah;
	uint16_t qpn;
};

struct efa_reverse_av {
	struct efa_ah_qpn key;
	fi_addr_t fi_addr;
	UT_hash_handle hh;
};

struct efa_ep_domain {
	char		*suffix;
	enum fi_ep_type	type;
	uint64_t	caps;
};

struct efa_device_attr {
	struct ibv_device_attr	ibv_attr;
	uint32_t		max_sq_wr;
	uint32_t		max_rq_wr;
	uint16_t		max_sq_sge;
	uint16_t		max_rq_sge;
};

static inline struct efa_device *to_efa_dev(struct ibv_device *ibdev)
{
	return container_of(ibdev, struct efa_device, verbs_dev);
}

static inline struct efa_context *to_efa_ctx(struct ibv_context *ibctx)
{
	return container_of(ibctx, struct efa_context, ibv_ctx);
}

static inline struct efa_pd *to_efa_pd(struct ibv_pd *ibpd)
{
	return container_of(ibpd, struct efa_pd, ibv_pd);
}

static inline struct efa_cq *to_efa_cq(struct ibv_cq *ibcq)
{
	return container_of(ibcq, struct efa_cq, ibv_cq);
}

static inline struct efa_qp *to_efa_qp(struct ibv_qp *ibqp)
{
	return container_of(ibqp, struct efa_qp, ibv_qp);
}

static inline struct efa_ah *to_efa_ah(struct ibv_ah *ibah)
{
	return container_of(ibah, struct efa_ah, ibv_ah);
}

static inline unsigned long align(unsigned long val, unsigned long align)
{
	return (val + align - 1) & ~(align - 1);
}

static inline uint32_t align_up_queue_size(uint32_t req)
{
	req--;
	req |= req >> 1;
	req |= req >> 2;
	req |= req >> 4;
	req |= req >> 8;
	req |= req >> 16;
	req++;
	return req;
}

#define is_power_of_2(x) (!(x == 0) && !(x & (x - 1)))
#define align_down_to_power_of_2(x)		\
	({					\
		__typeof__(x) n = (x);		\
		while (n & (n - 1))		\
			n = n & (n - 1);	\
		n;				\
	})

extern const struct efa_ep_domain efa_rdm_domain;
extern const struct efa_ep_domain efa_dgrm_domain;

struct fi_ops_cm efa_ep_cm_ops;
struct fi_ops_msg efa_ep_msg_ops;

const struct fi_info *efa_get_efa_info(const char *domain_name);
int efa_domain_open(struct fid_fabric *fabric_fid, struct fi_info *info,
		    struct fid_domain **domain_fid, void *context);
int efa_ep_open(struct fid_domain *domain_fid, struct fi_info *info,
		struct fid_ep **ep_fid, void *context);
int efa_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		struct fid_av **av_fid, void *context);
int efa_cq_open(struct fid_domain *domain_fid, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context);

/* Caller must hold cq->inner_lock. */
void efa_cq_inc_ref_cnt(struct efa_cq *cq, uint8_t sub_cq_idx);
/* Caller must hold cq->inner_lock. */
void efa_cq_dec_ref_cnt(struct efa_cq *cq, uint8_t sub_cq_idx);

fi_addr_t efa_ah_qpn_to_addr(struct efa_ep *ep, uint16_t ah, uint16_t qpn);

struct fi_provider *init_lower_efa_prov();

#endif /* EFA_H */
