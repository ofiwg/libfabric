/*
 * Copyright (c) Intel Corporation, Inc.  All rights reserved.
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
#include <pthread.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/un.h>
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
#include <rdma/providers/fi_peer.h>
#include <rdma/providers/fi_prov.h>

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
#include "ofi_mem.h"
#include "ofi_mr.h"
#include "ofi_proto.h"
#include "ofi_prov.h"
#include "ofi_rbuf.h"
#include "ofi_shm_p2p.h"
#include "ofi_signal.h"
#include "ofi_tree.h"
#include "ofi_util.h"
#include "ofi_xpmem.h"

#define SMR_VERSION	9

#define SMR_FLAG_ATOMIC		(1 << 0)
#define SMR_FLAG_DEBUG		(1 << 1)
#define SMR_FLAG_HMEM_ENABLED	(1 << 3)
#define SMR_FLAG_CMA_INIT	(1 << 4)

//shm region defines
#define SMR_CMD_SIZE		440	/* align with 64-byte cache line */

//reserves 0-255 for defined ops and room for new ops
//256 and beyond reserved for ctrl ops
#define SMR_OP_MAX (1 << 8)

#define SMR_REMOTE_CQ_DATA	(1 << 0)

/* SMR op_src: Specifies data source location */
enum {
	smr_proto_inline,	/* command data */
	smr_proto_inject,	/* inject buffers */
	smr_proto_iov,		/* reference iovec via CMA */
	smr_proto_sar,		/* segmentation fallback protocol */
	smr_proto_ipc,		/* device IPC handle protocol */
	smr_proto_max,
};

/*
 * Unique smr_op_hdr for smr message protocol:
 *	entry - for internal use managing commands (must be kept first)
 *	tx_ctx - source side context (unused by target side)
 *	rx_ctx - target side context (unused by source side)
 * 	id - local shm_id of peer sending msg (for shm lookup)
 * 	op - type of op (ex. ofi_op_msg, defined in ofi_proto.h)
 * 	proto - msg src (ex. smr_src_inline, defined above)
 * 	op_flags - operation flags (ex. SMR_REMOTE_CQ_DATA, defined above)
 * 	size - size of data transfer
 * 	proto_data - src of additional protocol data (inject offset)
 * 	status - returned status of operation
 * 	cq_data - remote CQ data
 */
struct  smr_cmd_hdr {
	uint64_t		entry;
	uint64_t		tx_ctx;
	uint64_t		rx_ctx;
	int64_t			id;
	uint32_t		op;
	uint16_t		proto;
	uint16_t		op_flags;

	uint64_t		size;
	uint64_t		proto_data;
	int64_t			status;
	uint64_t		cq_data;
	union {
		uint64_t	tag;
		struct {
			uint8_t	datatype;
			uint8_t	atomic_op;
		};
	};
} __attribute__ ((aligned(16)));

#define SMR_BUF_BATCH_MAX	64
#define SMR_MSG_DATA_LEN	(SMR_CMD_SIZE - \
				 (sizeof(struct smr_cmd_hdr) + \
				  sizeof(struct smr_cmd_rma)))
#define SMR_IOV_LIMIT		4

struct smr_cmd_rma {
	uint64_t		rma_count;
	union {
		struct fi_rma_iov	rma_iov[SMR_IOV_LIMIT];
		struct fi_rma_ioc	rma_ioc[SMR_IOV_LIMIT];
	};
};

struct smr_cmd_data {
	union {
		uint8_t			msg[SMR_MSG_DATA_LEN];
		struct {
			size_t		iov_count;
			struct iovec	iov[SMR_IOV_LIMIT];
		};
		struct {
			uint32_t	buf_batch_size;
			int16_t		sar[SMR_BUF_BATCH_MAX];
		};
		struct ipc_info		ipc_info;
	};
};
STATIC_ASSERT(sizeof(struct smr_cmd_data) == SMR_MSG_DATA_LEN, smr_cmd_size);

struct smr_cmd {
	struct smr_cmd_hdr	hdr;
	struct smr_cmd_data	data;
	struct smr_cmd_rma	rma;
};

#define SMR_INJECT_SIZE		(1 << 12) //4096
#define SMR_COMP_INJECT_SIZE	(SMR_INJECT_SIZE / 2)
#define SMR_SAR_SIZE		(1 << 15) //32768

#define SMR_DIR "/dev/shm/"
#define SMR_NAME_MAX	256
#define SMR_PATH_MAX	(SMR_NAME_MAX + sizeof(SMR_DIR))

struct smr_peer_data {
	int64_t			id;
	uint32_t		sar_status;
	uint16_t		name_sent;
	uint16_t		ipc_valid;
	uintptr_t		local_region;
	struct ofi_xpmem_client xpmem;
};

extern struct dlist_entry ep_name_list;
extern pthread_mutex_t ep_list_lock;

struct smr_region;

struct smr_ep_name {
	char name[SMR_NAME_MAX];
	struct smr_region *region;
	struct dlist_entry entry;
};

static inline const char *smr_no_prefix(const char *addr)
{
	char *start;

	return (start = strstr(addr, "://")) ? start + 3 : addr;
}

struct smr_peer {
	char			name[SMR_NAME_MAX];
	bool			id_assigned;
	fi_addr_t		fiaddr;
	struct smr_region	*region;
	int			pid_fd;
};

#define SMR_MAX_PEERS	256

struct smr_map {
	ofi_spin_t		lock;
	int64_t			cur_id;
	int 			num_peers;
	uint16_t		flags;
	struct ofi_rbmap	rbmap;
	struct smr_peer		peers[SMR_MAX_PEERS];
};

struct smr_region {
	uint8_t			version;
	uint8_t			resv;
	uint16_t		flags;
	uint8_t			self_vma_caps;
	uint8_t			peer_vma_caps;

	uint16_t		max_sar_buf_per_peer;
	struct ofi_xpmem_pinfo	xpmem_self;
	struct ofi_xpmem_pinfo	xpmem_peer;

	int			pid;
	int			resv2;

	void			*base_addr;

	char			name[SMR_NAME_MAX];

	size_t			total_size;

	/* offsets from start of smr_region */
	size_t			cmd_queue_offset;
	size_t			cmd_stack_offset;
	size_t			inject_pool_offset;
	size_t			ret_queue_offset;
	size_t			sar_pool_offset;
	size_t			peer_data_offset;
};

static inline void smr_set_vma_cap(uint8_t *vma_cap, uint8_t type, bool avail)
{
	(*vma_cap) &= ~(1 << type);
	(*vma_cap) |= (uint8_t) avail << type;
}

static inline uint8_t smr_get_vma_cap(uint8_t vma_cap, uint8_t type)
{
	return vma_cap & (1 << type);
}

struct smr_inject_buf {
	union {
		uint8_t		data[SMR_INJECT_SIZE];
		struct {
			uint8_t	buf[SMR_COMP_INJECT_SIZE];
			uint8_t comp[SMR_COMP_INJECT_SIZE];
		};
	};
};

struct smr_sar_buf {
	uint8_t		buf[SMR_SAR_SIZE];
};

struct smr_cmd_entry {
	uintptr_t ptr;
	struct smr_cmd cmd;
};

//temporary wrapper until I get it right
struct smr_return_entry {
	uintptr_t ptr;
};

/* Queue of offsets of the command blocks obtained from the command pool
 * freestack
 */
OFI_DECLARE_ATOMIC_Q(struct smr_cmd_entry, smr_cmd_queue);
OFI_DECLARE_ATOMIC_Q(struct smr_return_entry, smr_return_queue);

struct smr_ep {
	struct util_ep		util_ep;
	size_t			tx_size;
	size_t			rx_size;
	const char		*name;
	uint64_t		msg_id;
	struct smr_region	*volatile region;
	struct fid_peer_srx	*srx;
	struct ofi_bufpool	*cmd_ctx_pool;
	struct ofi_bufpool	*unexp_buf_pool;
	struct ofi_bufpool	*pend_buf_pool;

	struct smr_tx_fs	*tx_fs;
	struct slist		overflow_list;
	struct dlist_entry	ipc_cpy_pend_list;
	size_t			min_multi_recv_size;

	int			ep_idx;
	enum ofi_shm_p2p_type	p2p_type;
	void			*dsa_context;
	void 			(*smr_progress_ipc_list)(struct smr_ep *ep);
};

struct smr_av {
	struct util_av		util_av;
	struct smr_map		smr_map;
	size_t			used;
};

static inline struct smr_region *smr_peer_region(struct smr_ep *ep, int i)
{
	return container_of(ep->util_ep.av, struct smr_av, util_av)->
			    smr_map.peers[i].region;
}
static inline struct smr_cmd_queue *smr_cmd_queue(struct smr_region *smr)
{
	return (struct smr_cmd_queue *) ((char *) smr + smr->cmd_queue_offset);
}
static inline struct smr_freestack *smr_cmd_stack(struct smr_region *smr)
{
	return (struct smr_freestack *) ((char *) smr + smr->cmd_stack_offset);
}
static inline struct smr_freestack *smr_inject_pool(struct smr_region *smr)
{
	return (struct smr_freestack *) ((char *) smr + smr->inject_pool_offset);
}
static inline struct smr_return_queue *smr_return_queue(struct smr_region *smr)
{
	return (struct smr_return_queue *) ((char *) smr + smr->ret_queue_offset);
}
static inline struct smr_peer_data *smr_peer_data(struct smr_region *smr)
{
	return (struct smr_peer_data *) ((char *) smr + smr->peer_data_offset);
}
static inline struct smr_freestack *smr_sar_pool(struct smr_region *smr)
{
	return (struct smr_freestack *) ((char *) smr + smr->sar_pool_offset);
}

struct smr_attr {
	const char	*name;
	size_t		rx_count;
	size_t		tx_count;
	uint16_t	flags;
};
size_t smr_calculate_size_offsets(size_t tx_count, size_t rx_count,
				  size_t *cmd_offset, size_t *cs_offset,
				  size_t *inject_offset, size_t *rq_offset,
				  size_t *sar_offset, size_t *peer_offset);
void smr_cma_check(struct smr_region *region,
		   struct smr_region *peer_region);
void smr_cleanup(void);
int smr_map_to_region(const struct fi_provider *prov, struct smr_map *map,
		      int64_t id);
void smr_map_to_endpoint(struct smr_ep *ep, int64_t id);
void smr_unmap_region(const struct fi_provider *prov, struct smr_map *map,
		      int64_t id, bool found);
void smr_unmap_from_endpoint(struct smr_ep *ep, int64_t id);
void smr_exchange_all_peers(struct smr_ep *ep);
int smr_map_add(const struct fi_provider *prov, struct smr_map *map,
		const char *name, int64_t *id);
void smr_map_del(struct smr_map *map, int64_t id);

struct smr_region *smr_map_get(struct smr_map *map, int64_t id);

int smr_create(const struct fi_provider *prov, struct smr_map *map,
	       const struct smr_attr *attr, struct smr_region *volatile *smr);
void smr_free(struct smr_region *smr);

static inline uintptr_t smr_local_to_peer(struct smr_ep *ep,
			int64_t id, int64_t peer_id,
			uintptr_t local_ptr)
{
	struct smr_region *peer_smr = smr_peer_region(ep, id);
	uint64_t offset = local_ptr - (uintptr_t) ep->region;

	return smr_peer_data(peer_smr)[peer_id].local_region + offset;
}

static inline uintptr_t smr_peer_to_peer(struct smr_ep *ep,
			int64_t id, uintptr_t local_ptr)
{
	struct smr_region *peer_smr = smr_peer_region(ep, id);
	uint64_t offset = local_ptr - (uintptr_t) peer_smr;

	return (uintptr_t) peer_smr->base_addr + offset;
}

static inline uintptr_t smr_peer_to_owner(struct smr_ep *ep,
			int64_t id, uintptr_t local_ptr)
{
	struct smr_region *peer_smr = smr_peer_region(ep, id);
	uint64_t offset = local_ptr - (uintptr_t) peer_smr;

	return (uintptr_t) peer_smr->base_addr + offset;
}

static inline void smr_return_cmd(struct smr_ep *ep, struct smr_cmd *cmd)
{
	struct smr_region *peer_smr = smr_peer_region(ep, cmd->hdr.id);
	uintptr_t peer_ptr = smr_peer_to_owner(ep, cmd->hdr.id, (uintptr_t) cmd);
	int64_t pos;
	struct smr_return_entry *queue_entry;
	int ret;

	ret = smr_return_queue_next(smr_return_queue(peer_smr), &queue_entry, &pos);
	if (ret == -FI_ENOENT) {
		//return queue runs in parallel to command stack
		//ie we will never run out of space
		assert(0);
	}

	assert(peer_ptr >= (uintptr_t) peer_smr->base_addr &&
		peer_ptr < (uintptr_t) peer_smr->base_addr + peer_smr->total_size);
	queue_entry->ptr = peer_ptr;

	smr_return_queue_commit(queue_entry, pos);
}

struct smr_env {
	size_t sar_threshold;
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
		     enum fi_op op, struct fi_atomic_attr *attr,
		     uint64_t flags);

struct smr_tx_entry {
	int64_t			peer_id;
	void			*context;
	struct iovec		iov[SMR_IOV_LIMIT];
	uint32_t		iov_count;
	uint64_t		op_flags;
	size_t			bytes_done;
	void			*map_ptr;
	struct smr_ep_name 	*map_name;
	struct ofi_mr		*mr[SMR_IOV_LIMIT];
};

struct smr_pend_entry {
	struct dlist_entry	entry;
	struct smr_cmd		*cmd;
	struct fi_peer_rx_entry	*rx_entry;
	struct smr_cmd_ctx	*cmd_ctx;
	size_t			bytes_done;
	struct iovec		iov[SMR_IOV_LIMIT];
	size_t			iov_count;
	struct ofi_mr		*mr[SMR_IOV_LIMIT];
	struct ofi_mr_entry	*ipc_entry;
	ofi_hmem_async_event_t	async_event;
};

struct smr_cmd_ctx {
	struct dlist_entry entry;
	struct smr_ep *ep;
	struct smr_cmd *cmd;
	struct smr_cmd cmd_cpy;
	char msg[SMR_MSG_DATA_LEN];
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
	struct fid_ep		rx_ep;
	struct fid_peer_srx	*srx;
};

#define SMR_PREFIX	"fi_shm://"
#define SMR_PREFIX_NS	"fi_ns://"

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

struct smr_unexp_buf {
	struct slist_entry entry;
	char buf[SMR_SAR_SIZE];
};

#define smr_ep_rx_flags(smr_ep) ((smr_ep)->util_ep.rx_op_flags)
#define smr_ep_tx_flags(smr_ep) ((smr_ep)->util_ep.tx_op_flags)

static inline int smr_mmap_name(char *shm_name, const char *ep_name,
				uint64_t msg_id)
{
	return snprintf(shm_name, SMR_NAME_MAX - 1, "%s_%ld",
			ep_name, msg_id);
}

int smr_endpoint(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **ep, void *context);

int smr_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context);
int smr_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		  struct fid_cntr **cntr_fid, void *context);

int64_t smr_verify_peer(struct smr_ep *ep, fi_addr_t fi_addr);

void smr_format_pend(struct smr_tx_entry *pend, void *context,
		     struct ofi_mr **mr, const struct iovec *iov,
		     uint32_t iov_count, uint64_t op_flags, int64_t id);
void smr_generic_format(struct smr_cmd *cmd, int64_t peer_id, uint32_t op,
			uint64_t tag, uint64_t data, uint64_t op_flags);
size_t smr_copy_to_sar(struct smr_ep *ep, struct smr_freestack *sar_pool,
		       struct smr_cmd *cmd, struct ofi_mr **mr,
		       const struct iovec *iov, size_t count,
		       size_t *bytes_done);
size_t smr_copy_from_sar(struct smr_ep *ep, struct smr_freestack *sar_pool,
			 struct smr_cmd *cmd, struct ofi_mr **mr,
			 const struct iovec *iov, size_t count,
			 size_t *bytes_done);
int smr_select_proto(void **desc, size_t iov_count, bool cma_avail,
		     bool ipc_valid, uint32_t op, uint64_t total_len,
		     uint64_t op_flags);
typedef ssize_t (*smr_proto_func)(
		struct smr_ep *ep, struct smr_region *peer_smr,
		int64_t id, int64_t peer_id, uint32_t op, uint64_t tag,
		uint64_t data, uint64_t op_flags, struct ofi_mr **desc,
		const struct iovec *iov, size_t iov_count, size_t total_len,
		void *context, struct smr_cmd *cmd);
extern smr_proto_func smr_proto_ops[smr_proto_max];

int smr_write_err_comp(struct util_cq *cq, void *context,
		       uint64_t flags, uint64_t tag, int err);
int smr_complete_tx(struct smr_ep *ep, void *context, uint32_t op,
		    uint64_t flags);
int smr_complete_rx(struct smr_ep *ep, void *context, uint32_t op,
		    uint64_t flags, size_t len, void *buf, int64_t id,
		    uint64_t tag, uint64_t data);

static inline uint64_t smr_rx_cq_flags(uint64_t rx_flags, uint16_t op_flags)
{
	if (op_flags & SMR_REMOTE_CQ_DATA)
		rx_flags |= FI_REMOTE_CQ_DATA;
	return rx_flags;
}

void smr_ep_progress(struct util_ep *util_ep);

static inline bool smr_vma_enabled(struct smr_ep *ep,
				   struct smr_region *peer_smr)
{
	return ep->region == peer_smr ? ep->region->self_vma_caps :
					ep->region->peer_vma_caps;
}

static inline void smr_set_ipc_valid(struct smr_ep *ep, uint64_t id)
{
	struct smr_av *av;

	av = container_of(ep->util_ep.av, struct smr_av, util_av);

	if (ofi_hmem_is_initialized(FI_HMEM_ZE) &&
	    av->smr_map.peers[id].pid_fd == -1)
		smr_peer_data(ep->region)[id].ipc_valid = 0;
        else
		smr_peer_data(ep->region)[id].ipc_valid = 1;
}

static inline bool smr_ipc_valid(struct smr_ep *ep, struct smr_region *peer_smr,
				 int64_t id, int64_t peer_id)
{
	return (smr_peer_data(ep->region)[id].ipc_valid &&
		smr_peer_data(peer_smr)[peer_id].ipc_valid);
}

int smr_unexp_start(struct fi_peer_rx_entry *rx_entry);

void smr_progress_ipc_list(struct smr_ep *ep);
static inline void smr_progress_ipc_list_noop(struct smr_ep *ep)
{
	// noop
}

/* SMR FUNCTIONS FOR DSA SUPPORT */
void smr_dsa_init(void);
void smr_dsa_cleanup(void);
size_t smr_dsa_copy_to_sar(struct smr_ep *ep, struct smr_freestack *sar_pool,
			   struct smr_cmd *cmd, const struct iovec *iov,
			   size_t count, size_t *bytes_done);
size_t smr_dsa_copy_from_sar(struct smr_ep *ep, struct smr_freestack *sar_pool,
			     struct smr_cmd *cmd, const struct iovec *iov,
			     size_t count, size_t *bytes_done);
void smr_dsa_context_init(struct smr_ep *ep);
void smr_dsa_context_cleanup(struct smr_ep *ep);
void smr_dsa_progress(struct smr_ep *ep);

#endif
