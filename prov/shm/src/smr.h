/*
 * Copyright (c) 2015-2018 Intel Corporation, Inc.  All rights reserved.
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

#include <ofi.h>
#include <ofi_enosys.h>
#include <ofi_shm.h>
#include <ofi_rbuf.h>
#include <ofi_list.h>
#include <ofi_signal.h>
#include <ofi_util.h>
#include <ofi_atomic.h>

#ifndef _SMR_H_
#define _SMR_H_


#define SMR_MAJOR_VERSION 1
#define SMR_MINOR_VERSION 0

extern struct fi_provider smr_prov;
extern struct fi_info smr_info;
extern struct util_prov smr_util_prov;

int smr_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		void *context);

struct smr_av {
	struct util_av		util_av;
	struct smr_map		*smr_map;
};

int smr_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **dom, void *context);

int smr_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		struct fid_eq **eq, void *context);

int smr_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		struct fid_av **av, void *context);

int smr_query_atomic(struct fid_domain *domain, enum fi_datatype datatype,
		enum fi_op op, struct fi_atomic_attr *attr, uint64_t flags);

#define SMR_IOV_LIMIT		4

struct smr_ep_entry {
	struct dlist_entry	entry;
	void			*context;
	fi_addr_t		addr;
	uint64_t		tag;
	uint64_t		ignore;
	struct iovec		iov[SMR_IOV_LIMIT];
	uint32_t		iov_count;
	uint32_t		flags;
	uint64_t		err;
};

struct smr_ep;
typedef int (*smr_rx_comp_func)(struct smr_ep *ep, void *context,
		uint64_t flags, size_t len, void *buf, void *addr,
		uint64_t tag, uint64_t data, uint64_t err);
typedef int (*smr_tx_comp_func)(struct smr_ep *ep, void *context,
		uint64_t flags, uint64_t err);


struct smr_match_attr {
	fi_addr_t	addr;
	uint64_t	tag;
	uint64_t	ignore;
};

static inline int smr_match_addr(fi_addr_t addr, fi_addr_t match_addr)
{
	return (addr == FI_ADDR_UNSPEC) || (match_addr == FI_ADDR_UNSPEC) ||
		(addr == match_addr);
}

static inline int smr_match_tag(uint64_t tag, uint64_t ignore, uint64_t match_tag)
{
	return ((tag | ignore) == (match_tag | ignore));
}

struct smr_unexp_msg {
	struct dlist_entry entry;
	struct smr_cmd cmd;
};

DECLARE_FREESTACK(struct smr_ep_entry, smr_recv_fs);
DECLARE_FREESTACK(struct smr_unexp_msg, smr_unexp_fs);
DECLARE_FREESTACK(struct smr_cmd, smr_pend_fs);

struct smr_queue {
	struct dlist_entry list;
	dlist_func_t *match_func;
};

struct smr_fabric {
	struct util_fabric	util_fabric;
	int			dom_idx;
};

struct smr_domain {
	struct util_domain	util_domain;
	int			dom_idx;
	int			ep_idx;
	int			fast_rma;
};

#define SMR_PREFIX	"fi_shm://"
#define SMR_PREFIX_NS	"fi_ns://"

static inline const char *smr_no_prefix(const char *addr)
{
	char *start;

	return (start = strstr(addr, "://")) ? start + 3 : addr;
}

#define SMR_RMA_ORDER (FI_ORDER_RAR | FI_ORDER_RAW | FI_ORDER_RAS |	\
		       FI_ORDER_WAR | FI_ORDER_WAW | FI_ORDER_WAS |	\
		       FI_ORDER_SAR | FI_ORDER_SAW)
#define smr_fast_rma_enabled(mode, order) ((mode & FI_MR_VIRT_ADDR) && \
			!(order & SMR_RMA_ORDER))

struct smr_ep {
	struct util_ep		util_ep;
	smr_rx_comp_func	rx_comp;
	smr_tx_comp_func	tx_comp;
	size_t			tx_size;
	size_t			rx_size;
	size_t			min_multi_recv_size;
	const char		*name;
	struct smr_region	*region;
	struct smr_recv_fs	*recv_fs; /* protected by rx_cq lock */
	struct smr_queue	recv_queue;
	struct smr_queue	trecv_queue;
	struct smr_unexp_fs	*unexp_fs;
	struct smr_pend_fs	*pend_fs;
	struct smr_queue	unexp_queue;
};

#define smr_ep_rx_flags(smr_ep) ((smr_ep)->util_ep.rx_op_flags)
#define smr_ep_tx_flags(smr_ep) ((smr_ep)->util_ep.tx_op_flags)

int smr_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep, void *context);

int smr_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context);

int smr_verify_peer(struct smr_ep *ep, int peer_id);

void smr_post_pend_resp(struct smr_cmd *cmd, struct smr_cmd *pend,
			struct smr_resp *resp);
void smr_generic_format(struct smr_cmd *cmd, fi_addr_t peer_id,
		uint32_t op, uint64_t tag, uint8_t datatype, uint8_t atomic_op,
		uint64_t data, uint64_t op_flags);
void smr_format_inline(struct smr_cmd *cmd, fi_addr_t peer_id,
		const struct iovec *iov, size_t count,
		uint32_t op, uint64_t tag, uint64_t data, uint64_t op_flags);
void smr_format_inject(struct smr_cmd *cmd, fi_addr_t peer_id,
		const struct iovec *iov, size_t count,
		uint32_t op, uint64_t tag, uint64_t data, uint64_t op_flags,
		struct smr_region *smr, struct smr_inject_buf *tx_buf);
void smr_format_iov(struct smr_cmd *cmd, fi_addr_t peer_id,
		const struct iovec *iov, size_t count, size_t total_len,
		uint32_t op, uint64_t tag, uint64_t data, uint64_t op_flags,
		void *context, struct smr_region *smr, struct smr_resp *resp,
		struct smr_cmd *pend);

int smr_tx_comp(struct smr_ep *ep, void *context, uint64_t flags, uint64_t err);
int smr_tx_comp_signal(struct smr_ep *ep, void *context, uint64_t flags,
		       uint64_t err);
int smr_rx_comp(struct smr_ep *ep, void *context, uint64_t flags, size_t len,
		void *buf, void *addr, uint64_t tag, uint64_t data,
		uint64_t err);
int smr_rx_src_comp(struct smr_ep *ep, void *context, uint64_t flags,
		    size_t len, void *buf, void *addr, uint64_t tag,
		    uint64_t data, uint64_t err);
int smr_rx_comp_signal(struct smr_ep *ep, void *context, uint64_t flags,
		       size_t len, void *buf, void *addr, uint64_t tag,
		       uint64_t data, uint64_t err);
int smr_rx_src_comp_signal(struct smr_ep *ep, void *context, uint64_t flags,
			   size_t len, void *buf, void *addr, uint64_t tag,
			   uint64_t data, uint64_t err);

uint64_t smr_tx_comp_flags(uint32_t op);
uint64_t smr_rx_comp_flags(uint32_t op, uint16_t op_flags);
uint64_t smr_mr_reg_flags(uint32_t op, uint16_t atomic_op);

void smr_ep_progress(struct util_ep *util_ep);
int smr_progress_unexp(struct smr_ep *ep, struct smr_ep_entry *entry);

#endif
