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

#ifndef _SMR_H_
#define _SMR_H_

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_trigger.h>
#include <rdma/fabric.h>
#include <rdma/providers/fi_prov.h>
#include <rdma/providers/fi_peer.h>

#include "ofi.h"
#include "ofi_atom.h"
#include "ofi_atomic.h"
#include "ofi_atomic_queue.h"
#include "ofi_enosys.h"
#include "ofi_epoll.h"
#include "ofi_hmem.h"
#include "ofi_iov.h"
#include "ofi_list.h"
#include "ofi_lock.h"
#include "ofi_mb.h"
#include <ofi_mem.h>
#include "ofi_mr.h"
#include <ofi_proto.h>
#include "ofi_prov.h"
#include "ofi_rbuf.h"
#include "ofi_shm_p2p.h"
#include "ofi_signal.h"
#include "ofi_tree.h"
#include "ofi_util.h"
#include "ofi_xpmem.h"

#include "smr_fifo.h"
#include "smr_util.h"

struct smr_env {
	int disable_cma;
	int use_dsa_sar;
	size_t max_gdrcopy_size;
	int use_xpmem;
};

extern struct smr_env smr_env;
extern struct fi_provider smr_prov;
extern struct fi_info smr_info;
extern struct util_prov smr_util_prov;
extern int smr_global_ep_idx; //protected by the ep_list_lock

int smr_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		void *context);

struct smr_av {
	struct util_av		util_av;
	struct smr_map		*smr_map;
	size_t			used;
};

static inline int64_t smr_addr_lookup(struct util_av *av, fi_addr_t fiaddr)
{
	return *((int64_t *) ofi_av_get_addr(av, fiaddr));
}

int smr_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **dom, void *context);

int smr_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		struct fid_eq **eq, void *context);

int smr_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		struct fid_av **av, void *context);

int smr_query_atomic(struct fid_domain *domain, enum fi_datatype datatype,
		enum fi_op op, struct fi_atomic_attr *attr, uint64_t flags);

#define SMR_IOV_LIMIT		4

// TODO merge/simplify?
struct smr_tx_entry {
	struct		smr_ep *ep;
	int64_t		peer_id;
	void		*context;
	struct iovec	iov[SMR_IOV_LIMIT];
	uint32_t	iov_count;
	uint64_t	op_flags;
	size_t		bytes_done;
	void		*map_ptr;
	struct smr_ep_name *map_name;
	struct ofi_mr	*mr[SMR_IOV_LIMIT];
	int			fd;
};

// TODO remove cmd?
struct smr_pend_entry {
	struct			smr_ep *ep;
	struct dlist_entry	entry;
	struct smr_cmd		cmd;
	struct fi_peer_rx_entry	*rx_entry;
	struct smr_cmd_ctx	*cmd_ctx;
	void			*context; //instead of rx_entry?
	size_t			bytes_done;
	struct iovec		iov[SMR_IOV_LIMIT];
	size_t			iov_count;
	struct ofi_mr		*mr[SMR_IOV_LIMIT];
	struct ofi_mr_entry	*ipc_entry;
	ofi_hmem_async_event_t	async_event;
};

// TODO DELETE
// SHOULD NOT BE USED :)
struct smr_cmd_ctx {
	struct dlist_entry entry;
	struct smr_ep *ep;
	struct smr_cmd cmd;
	struct smr_pend_entry *sar_entry;
	struct slist buf_list;
};

OFI_DECLARE_FREESTACK(struct smr_tx_entry, smr_tx_fs);

struct smr_fabric {
	struct util_fabric	util_fabric;
};

struct smr_domain {
	struct util_domain	util_domain;
	int			fast_rma;
	/* cache for use with hmem ipc */
	struct ofi_mr_cache	*ipc_cache;
	struct fid_peer_srx	*srx;
};

#define SMR_PREFIX	"fi_shm://"
#define SMR_PREFIX_NS	"fi_ns://"

#define SMR_ZE_SOCK_PATH	"/dev/shm/ze_"

#define SMR_RMA_ORDER (OFI_ORDER_RAR_SET | OFI_ORDER_RAW_SET | FI_ORDER_RAS |	\
		       OFI_ORDER_WAR_SET | OFI_ORDER_WAW_SET | FI_ORDER_WAS |	\
		       FI_ORDER_SAR | FI_ORDER_SAW)
#define smr_fast_rma_enabled(mode, order) ((mode & FI_MR_VIRT_ADDR) && \
			!(order & SMR_RMA_ORDER))

static inline uint64_t smr_get_offset(void *base, void *addr)
{
	return (uintptr_t) ((char *) addr - (char *) base);
}

static inline void *smr_get_ptr(void *base, uint64_t offset)
{
	return (char *) base + (uintptr_t) offset;
}

struct smr_sock_name {
	char name[SMR_SOCK_NAME_MAX];
	struct dlist_entry entry;
};

enum smr_cmap_state {
	SMR_CMAP_INIT = 0,
	SMR_CMAP_SUCCESS,
	SMR_CMAP_FAILED,
};

struct smr_cmap_entry {
	enum smr_cmap_state	state;
	int			device_fds[ZE_MAX_DEVICES];
};

struct smr_sock_info {
	char			name[SMR_SOCK_NAME_MAX];
	int			listen_sock;
	ofi_epoll_t		epollfd;
	struct fd_signal	signal;
	pthread_t		listener_thread;
	int			*my_fds;
	int			nfds;
	struct smr_cmap_entry	peers[SMR_MAX_PEERS];
};

struct smr_unexp_buf {
	struct slist_entry entry;
	char buf[SMR_SAR_SIZE];
};

struct smr_ep {
	struct util_ep		util_ep;
	size_t			tx_size;
	size_t			rx_size;
	const char		*name;
	uint64_t		msg_id;
	struct smr_region	*volatile region;
	struct fid_ep		*srx;
	// TODO USE THIS OR DELETE IT
	struct ofi_bufpool	*cmd_ctx_pool;
	struct ofi_bufpool	*unexp_buf_pool;
	struct ofi_bufpool	*pend_buf_pool;

	struct smr_tx_fs	*tx_fs;
	struct dlist_entry	sar_list;
	struct dlist_entry	ipc_cpy_pend_list;

	int			ep_idx;
	enum ofi_shm_p2p_type	p2p_type;
	struct smr_sock_info	*sock_info;
	void			*dsa_context;
	void 			(*smr_progress_ipc_list)(struct smr_ep *ep);
};

static inline struct fid_peer_srx *smr_get_peer_srx(struct smr_ep *ep)
{
	return container_of(ep->srx, struct fid_peer_srx, ep_fid);
}

#define smr_ep_rx_flags(smr_ep) ((smr_ep)->util_ep.rx_op_flags)
#define smr_ep_tx_flags(smr_ep) ((smr_ep)->util_ep.tx_op_flags)

int smr_srx_context(struct fid_domain *domain, struct fi_rx_attr *attr,
		struct fid_ep **rx_ep, void *context);

int smr_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep, void *context);
void smr_ep_exchange_fds(struct smr_ep *ep, int64_t id);

int smr_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context);
int smr_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		  struct fid_cntr **cntr_fid, void *context);

int64_t smr_verify_peer(struct smr_ep *ep, fi_addr_t fi_addr);

void smr_format_pend(struct smr_ep *ep, struct smr_tx_entry *pend,
		     struct smr_cmd *cmd, void *context, struct ofi_mr **mr,
		     const struct iovec *iov, uint32_t iov_count,
		     uint64_t op_flags, int64_t id);
void smr_generic_format(struct smr_cmd *cmd, int64_t peer_id, uint32_t op,
			uint64_t tag, uint64_t data, uint64_t op_flags,
			uintptr_t rma_cmd);
size_t smr_copy_to_sar(struct smr_region *smr, struct smr_cmd *cmd,
		         struct ofi_mr **mr, const struct iovec *iov,
		         size_t count, size_t *bytes_done);
size_t smr_copy_from_sar(struct smr_region *smr, struct smr_cmd *cmd,
		         struct ofi_mr **mr, const struct iovec *iov,
		         size_t count, size_t *bytes_done);
int smr_select_proto(void **desc, size_t iov_count, bool cma_avail,
		     uint32_t op, uint64_t total_len, uint64_t op_flags);
typedef ssize_t (*smr_proto_func)(struct smr_ep *ep,
		struct smr_region *peer_smr, int64_t id, int64_t peer_id,
		uint32_t op, uint64_t tag, uint64_t data, uint64_t op_flags,
		struct ofi_mr **desc, const struct iovec *iov, size_t iov_count,
		size_t total_len, void *context, uintptr_t rma_cmd);
extern smr_proto_func smr_proto_ops[smr_src_max];

int smr_write_err_comp(struct util_cq *cq, void *context,
		       uint64_t flags, uint64_t tag, uint64_t err);
int smr_complete_tx(struct smr_ep *ep, void *context, uint32_t op,
		    uint64_t flags);
int smr_complete_rx(struct smr_ep *ep, void *context, uint32_t op,
		    uint64_t flags, size_t len, void *buf, int64_t id,
		    uint64_t tag, uint64_t data);

static inline uint64_t smr_rx_cq_flags(uint32_t op, uint64_t rx_flags,
				       uint64_t op_flags)
{
	return ofi_rx_cq_flags(op) | rx_flags | (op_flags & FI_REMOTE_CQ_DATA);
}

void smr_ep_progress(struct util_ep *util_ep);

static inline bool smr_vma_enabled(struct smr_ep *ep,
				   struct smr_region *peer_smr)
{
	if (ep->region == peer_smr)
		return (ep->region->cma_cap_self == SMR_VMA_CAP_ON ||
			ep->region->xpmem_cap_self == SMR_VMA_CAP_ON);
	else
		return (ep->region->cma_cap_peer == SMR_VMA_CAP_ON ||
			peer_smr->xpmem_cap_self == SMR_VMA_CAP_ON);
}

static inline bool smr_ze_ipc_enabled(struct smr_region *smr,
				      struct smr_region *peer_smr)
{
	return (smr->flags & SMR_FLAG_IPC_SOCK) &&
	       (peer_smr->flags & SMR_FLAG_IPC_SOCK);
}

int smr_unexp_start(struct fi_peer_rx_entry *rx_entry);

void smr_progress_ipc_list(struct smr_ep *ep);
static inline void smr_progress_ipc_list_noop(struct smr_ep *ep)
{
	// noop
}

static inline uintptr_t smr_get_peer_ptr(struct smr_region *my_smr,
			int64_t id, int64_t peer_id,
			struct smr_cmd *local_ptr)
{
	struct smr_region *peer_smr = smr_peer_region(my_smr, id);
	uint64_t offset = (uintptr_t) local_ptr - (uintptr_t) my_smr;

	return smr_peer_data(peer_smr)[peer_id].local_region + offset;
}

static inline uintptr_t smr_get_owner_ptr(struct smr_region *my_smr,
			int64_t id, int64_t peer_id,
			struct smr_cmd *local_ptr)
{
	struct smr_region *peer_smr = smr_peer_region(my_smr, id);
	uint64_t offset = (uintptr_t) local_ptr - (uintptr_t) peer_smr;

	return (uintptr_t) peer_smr->base_addr + offset;
}

static inline uintptr_t smr_get_owner_ptr_from_owner(struct smr_region *my_smr,
			struct smr_region *peer_smr, int peer_id,
			uintptr_t remote_ptr)
{
	uint64_t offset = (uintptr_t) remote_ptr - (uintptr_t) smr_peer_data(peer_smr)[peer_id].local_region;
	return (uintptr_t) my_smr->base_addr + offset;
}

//TODO change cmd/resp queues into immediate commit atomic queues
static inline ssize_t smr_commit_cmd(struct smr_region *my_smr,
			int64_t id, struct smr_cmd *cmd)
{
	struct smr_region *peer_smr = smr_peer_region(my_smr, id);
	struct smr_fifo *queue = smr_cmd_queue(peer_smr);
	uintptr_t peer_ptr = smr_get_peer_ptr(my_smr, id, cmd->msg.hdr.id, cmd);

	if (smr_fifo_commit(queue, peer_ptr)) {
		return -FI_EAGAIN;
	}
	return FI_SUCCESS;
}

//TODO squash these functions to take in id?
static inline ssize_t smr_return_cmd(struct smr_region *my_smr,
				     struct smr_cmd *cmd)
{

	struct smr_region *peer_smr = smr_peer_region(my_smr, cmd->msg.hdr.id);
	struct smr_fifo *queue = smr_return_queue(peer_smr);
	int64_t id = cmd->msg.hdr.id;
	int64_t peer_id = smr_peer_data(my_smr)[id].addr.id;
	uintptr_t peer_ptr = smr_get_owner_ptr(my_smr, id, peer_id, cmd);

	if (smr_fifo_commit(queue, peer_ptr)) {
		assert(0);
	}
	return FI_SUCCESS;
}

static inline struct smr_cmd *smr_read_cmd(struct smr_region *smr)
{
	struct smr_fifo *queue = smr_cmd_queue(smr);

	return (struct smr_cmd *) smr_fifo_read(queue);
}

static inline struct smr_cmd *smr_read_return(struct smr_region *smr)
{
	struct smr_fifo *queue = smr_return_queue(smr);

	return (struct smr_cmd *) smr_fifo_read(queue);
}

/* SMR FUNCTIONS FOR DSA SUPPORT */
void smr_dsa_init(void);
void smr_dsa_cleanup(void);
size_t smr_dsa_copy_to_sar(struct smr_ep *ep, struct smr_region *smr,
		struct smr_cmd *cmd, const struct iovec *iov, size_t count,
		size_t *bytes_done, void *entry_ptr);
size_t smr_dsa_copy_from_sar(struct smr_ep *ep, struct smr_region *smr,
		struct smr_cmd *cmd, const struct iovec *iov, size_t count,
		size_t *bytes_done, void *entry_ptr);
void smr_dsa_context_init(struct smr_ep *ep);
void smr_dsa_context_cleanup(struct smr_ep *ep);
void smr_dsa_progress(struct smr_ep *ep);

#endif
