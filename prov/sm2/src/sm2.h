/*
 * Copyright (c) 2015-2021 Intel Corporation, Inc.  All rights reserved.
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

#include "sm2_common.h"

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <sys/types.h>
#include <sys/statvfs.h>
#include <pthread.h>
#include <stdint.h>
#include <stddef.h>

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
#include <rdma/providers/fi_prov.h>
#include <rdma/providers/fi_peer.h>

#include <ofi.h>
#include <ofi_enosys.h>
#include <ofi_rbuf.h>
#include <ofi_list.h>
#include <ofi_signal.h>
#include <ofi_epoll.h>
#include <ofi_util.h>
#include <ofi_atomic.h>
#include <ofi_iov.h>
#include <ofi_mr.h>
#include <ofi_lock.h>

#ifndef _SM2_H_
#define _SM2_H_

struct sm2_env {
	size_t sar_threshold;
	int disable_cma;
	int use_dsa_sar;
};

extern struct sm2_env sm2_env;
extern struct fi_provider sm2_prov;
extern struct fi_info sm2_info;
extern struct util_prov sm2_util_prov;
extern int sm2_global_ep_idx; //protected by the ep_list_lock

int sm2_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		void *context);

struct sm2_av {
	struct util_av		util_av;
	struct sm2_map		*sm2_map;
	size_t			used;
};

static inline int64_t sm2_addr_lookup(struct util_av *av, fi_addr_t fiaddr)
{
	return *((int64_t *) ofi_av_get_addr(av, fiaddr));
}

int sm2_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **dom, void *context);

int sm2_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		struct fid_eq **eq, void *context);

int sm2_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		struct fid_av **av, void *context);

#define SM2_IOV_LIMIT		4

struct sm2_rx_entry {
	struct fi_peer_rx_entry	peer_entry;
	struct iovec		iov[SM2_IOV_LIMIT];
	void			*desc[SM2_IOV_LIMIT];
	int64_t			peer_id;
	uint64_t		ignore;
	int			multi_recv_ref;
	uint64_t		err;
	enum fi_hmem_iface	iface;
	uint64_t		device;
};

struct sm2_tx_entry {
	struct sm2_cmd	cmd;
	int64_t		peer_id;
	void		*context;
	struct iovec	iov[SM2_IOV_LIMIT];
	uint32_t	iov_count;
	uint64_t	op_flags;
	size_t		bytes_done;
	int		next;
	void		*map_ptr;
	struct sm2_ep_name *map_name;
	enum fi_hmem_iface	iface;
	uint64_t		device;
	int			fd;
};

struct sm2_sar_entry {
	struct dlist_entry	entry;
	struct sm2_cmd		cmd;
	struct fi_peer_rx_entry	*rx_entry;
	size_t			bytes_done;
	int			next;
	struct iovec		iov[SM2_IOV_LIMIT];
	size_t			iov_count;
	enum fi_hmem_iface	iface;
	uint64_t		device;
};

struct sm2_cq {
	struct util_cq util_cq;
	struct fid_peer_cq *peer_cq;
};

typedef int (*sm2_rx_comp_func)(struct sm2_cq *cq, void *context,
		uint64_t flags, size_t len, void *buf,
		fi_addr_t fi_addr, uint64_t tag, uint64_t data);

struct sm2_match_attr {
	fi_addr_t	id;
	uint64_t	tag;
	uint64_t	ignore;
};

static inline int sm2_match_id(fi_addr_t addr, fi_addr_t match_addr)
{
	return (addr == FI_ADDR_UNSPEC) || (match_addr == FI_ADDR_UNSPEC) ||
		(addr == match_addr);
}

static inline int sm2_match_tag(uint64_t tag, uint64_t ignore, uint64_t match_tag)
{
	return ((tag | ignore) == (match_tag | ignore));
}

static inline enum fi_hmem_iface sm2_get_mr_hmem_iface(struct util_domain *domain,
				void **desc, uint64_t *device)
{
	if (!(domain->mr_mode & FI_MR_HMEM) || !desc || !*desc) {
		*device = 0;
		return FI_HMEM_SYSTEM;
	}

	*device = ((struct ofi_mr *) *desc)->device;
	return ((struct ofi_mr *) *desc)->iface;
}

static inline uint64_t sm2_get_mr_flags(void **desc)
{
	assert(desc && *desc);
	return ((struct ofi_mr *) *desc)->flags;
}

struct sm2_cmd_ctx {
	struct dlist_entry entry;
	struct sm2_ep *ep;
	struct sm2_cmd cmd;
};

OFI_DECLARE_FREESTACK(struct sm2_rx_entry, sm2_recv_fs);
OFI_DECLARE_FREESTACK(struct sm2_cmd_ctx, sm2_cmd_ctx_fs);
OFI_DECLARE_FREESTACK(struct sm2_tx_entry, sm2_pend_fs);
OFI_DECLARE_FREESTACK(struct sm2_sar_entry, sm2_sar_fs);

struct sm2_queue {
	struct dlist_entry list;
	dlist_func_t *match_func;
};

struct sm2_fabric {
	struct util_fabric	util_fabric;
};

struct sm2_domain {
	struct util_domain	util_domain;
	int			fast_rma;
	/* cache for use with hmem ipc */
	struct ofi_mr_cache	*ipc_cache;
	struct fid_peer_srx	*srx;
};

#define SM2_PREFIX	"fi_shm://"
#define SM2_PREFIX_NS	"fi_ns://"

#define SM2_ZE_SOCK_PATH	"/dev/shm/ze_"

#define SM2_RMA_ORDER (OFI_ORDER_RAR_SET | OFI_ORDER_RAW_SET | FI_ORDER_RAS |	\
		       OFI_ORDER_WAR_SET | OFI_ORDER_WAW_SET | FI_ORDER_WAS |	\
		       FI_ORDER_SAR | FI_ORDER_SAW)
#define sm2_fast_rma_enabled(mode, order) ((mode & FI_MR_VIRT_ADDR) && \
			!(order & SM2_RMA_ORDER))

static inline uint64_t sm2_get_offset(void *base, void *addr)
{
	return (uintptr_t) ((char *) addr - (char *) base);
}

static inline void *sm2_get_ptr(void *base, uint64_t offset)
{
	return (char *) base + (uintptr_t) offset;
}

struct sm2_sock_name {
	char name[SM2_SOCK_NAME_MAX];
	struct dlist_entry entry;
};

struct sm2_sock_info {
	char			name[SM2_SOCK_NAME_MAX];
	int			listen_sock;
	ofi_epoll_t		epollfd;
	struct fd_signal	signal;
	pthread_t		listener_thread;
	int			*my_fds;
	int			nfds;
};

struct sm2_srx_ctx {
	struct fid_peer_srx	peer_srx;
	struct sm2_queue	recv_queue;
	struct sm2_queue	trecv_queue;
	bool			dir_recv;
	size_t			min_multi_recv_size;
	uint64_t		rx_op_flags;
	uint64_t		rx_msg_flags;

	struct sm2_cq		*cq;
	struct sm2_queue	unexp_msg_queue;
	struct sm2_queue	unexp_tagged_queue;
	struct sm2_recv_fs	*recv_fs;
	ofi_spin_t		lock;
};

struct sm2_rx_entry *sm2_alloc_rx_entry(struct sm2_srx_ctx *srx);

struct sm2_ep {
	struct util_ep		util_ep;
	sm2_rx_comp_func	rx_comp;
	size_t			tx_size;
	size_t			rx_size;
	const char		*name;
	uint64_t		msg_id;
	struct sm2_region	*volatile region;
	//if double locking is needed, shm region lock must
	//be acquired before any shm EP locks
	ofi_spin_t		tx_lock;

	struct fid_ep		*srx;
	struct sm2_cmd_ctx_fs	*cmd_ctx_fs;
	struct sm2_pend_fs	*pend_fs;
	struct sm2_sar_fs	*sar_fs;

	struct dlist_entry	sar_list;

	int			ep_idx;
	struct sm2_sock_info	*sock_info;
	void			*dsa_context;
};

static inline struct sm2_srx_ctx *sm2_get_sm2_srx(struct sm2_ep *ep)
{
	return (struct sm2_srx_ctx *) ep->srx->fid.context;
}

static inline struct fid_peer_srx *sm2_get_peer_srx(struct sm2_ep *ep)
{
	return container_of(ep->srx, struct fid_peer_srx, ep_fid);
}

#define sm2_ep_rx_flags(sm2_ep) ((sm2_ep)->util_ep.rx_op_flags)
#define sm2_ep_tx_flags(sm2_ep) ((sm2_ep)->util_ep.tx_op_flags)

static inline int sm2_mmap_name(char *shm_name, const char *ep_name,
				uint64_t msg_id)
{
	return snprintf(shm_name, SM2_NAME_MAX - 1, "%s_%ld",
			ep_name, msg_id);
}

int sm2_srx_context(struct fid_domain *domain, struct fi_rx_attr *attr,
		struct fid_ep **rx_ep, void *context);

int sm2_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep, void *context);
void sm2_ep_exchange_fds(struct sm2_ep *ep, int64_t id);

int sm2_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context);
int sm2_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		  struct fid_cntr **cntr_fid, void *context);

int64_t sm2_verify_peer(struct sm2_ep *ep, fi_addr_t fi_addr);

void sm2_format_pend_resp(struct sm2_tx_entry *pend, struct sm2_cmd *cmd,
			  void *context, enum fi_hmem_iface iface, uint64_t device,
			  const struct iovec *iov, uint32_t iov_count,
			  uint64_t op_flags, int64_t id, struct sm2_resp *resp);
void sm2_generic_format(struct sm2_cmd *cmd, int64_t peer_id, uint32_t op,
			uint64_t tag, uint64_t data, uint64_t op_flags);
size_t sm2_copy_to_sar(struct smr_freestack *sar_pool, struct sm2_resp *resp,
		       struct sm2_cmd *cmd, enum fi_hmem_iface, uint64_t device,
		       const struct iovec *iov, size_t count,
		       size_t *bytes_done, int *next);
size_t sm2_copy_from_sar(struct smr_freestack *sar_pool, struct sm2_resp *resp,
			 struct sm2_cmd *cmd, enum fi_hmem_iface iface,
			 uint64_t device, const struct iovec *iov, size_t count,
			 size_t *bytes_done, int *next);

int sm2_select_proto(bool use_ipc, bool cma_avail, enum fi_hmem_iface iface,
		     uint32_t op, uint64_t total_len, uint64_t op_flags);
typedef ssize_t (*sm2_proto_func)(struct sm2_ep *ep, struct sm2_region *peer_smr,
		int64_t id, int64_t peer_id, uint32_t op, uint64_t tag,
		uint64_t data, uint64_t op_flags, enum fi_hmem_iface iface,
		uint64_t device, const struct iovec *iov, size_t iov_count,
		size_t total_len, void *context);
extern sm2_proto_func sm2_proto_ops[sm2_src_max];

int sm2_write_err_comp(struct util_cq *cq, void *context,
		       uint64_t flags, uint64_t tag, uint64_t err);
int sm2_complete_tx(struct sm2_ep *ep, void *context, uint32_t op,
		    uint64_t flags);
int sm2_complete_rx(struct sm2_ep *ep, void *context, uint32_t op,
		    uint64_t flags, size_t len, void *buf, int64_t id,
		    uint64_t tag, uint64_t data);
int sm2_rx_comp(struct sm2_cq *cq, void *context, uint64_t flags, size_t len,
		void *buf, fi_addr_t fi_addr, uint64_t tag, uint64_t data);
int sm2_rx_src_comp(struct sm2_cq *cq, void *context, uint64_t flags,
		    size_t len, void *buf, fi_addr_t fi_addr, uint64_t tag,
		    uint64_t data);

static inline uint64_t sm2_rx_cq_flags(uint32_t op, uint64_t rx_flags,
				       uint16_t op_flags)
{
	rx_flags |= ofi_rx_cq_flags(op);
	if (op_flags & SM2_REMOTE_CQ_DATA)
		rx_flags |= FI_REMOTE_CQ_DATA;
	return rx_flags;
}


bool sm2_adjust_multi_recv(struct sm2_srx_ctx *srx,
			   struct fi_peer_rx_entry *rx_entry, size_t len);
void sm2_init_rx_entry(struct sm2_rx_entry *entry, const struct iovec *iov,
		       void **desc, size_t count, fi_addr_t addr,
		       void *context, uint64_t tag, uint64_t flags);
struct sm2_rx_entry *sm2_get_recv_entry(struct sm2_srx_ctx *srx,
		const struct iovec *iov, void **desc, size_t count, fi_addr_t addr,
		void *context, uint64_t tag, uint64_t ignore, uint64_t flags);

void sm2_ep_progress(struct util_ep *util_ep);


static inline bool sm2_ze_ipc_enabled(struct sm2_region *smr,
				      struct sm2_region *peer_smr)
{
	return (smr->flags & SM2_FLAG_IPC_SOCK) &&
	       (peer_smr->flags & SM2_FLAG_IPC_SOCK);
}

int sm2_unexp_start(struct fi_peer_rx_entry *rx_entry);
#endif
