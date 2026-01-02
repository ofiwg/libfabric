/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2017 DataDirect Networks, Inc. All rights reserved.
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_PROV_H_
#define _CXIP_PROV_H_

#include <netinet/ether.h>
#include "config.h"

#include <pthread.h>
#include <json-c/json.h>

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
#include <semaphore.h>

#include <ofi.h>
#include <ofi_atom.h>
#include <ofi_atomic.h>
#include <ofi_mr.h>
#include <ofi_enosys.h>
#include <ofi_indexer.h>
#include <ofi_rbuf.h>
#include <ofi_lock.h>
#include <ofi_list.h>
#include <ofi_file.h>
#include <ofi_osd.h>
#include <ofi_util.h>
#include <ofi_mem.h>
#include <ofi_hmem.h>
#include <unistd.h>

#include "libcxi/libcxi.h"
#include "cxip_faults.h"
#include "fi_cxi_ext.h"







/* Forward declarations for function pointer typedef parameters */
struct cxip_zbcoll_obj;
struct cxip_curl_handle;

/* Function pointer typedefs (needed by split headers) */
typedef void (*zbcomplete_t)(struct cxip_zbcoll_obj *zb, void *usrptr);
typedef void (*curlcomplete_t)(struct cxip_curl_handle *);

/* Extern declarations for global variables */
extern struct cxip_environment cxip_env;
extern struct fi_provider cxip_prov;
extern struct util_prov cxip_util_prov;
extern char cxip_prov_name[];
extern struct fi_fabric_attr cxip_fabric_attr;
extern struct fi_domain_attr cxip_domain_attr;
extern bool cxip_collectives_supported;
extern int sc_page_size;
extern struct slist cxip_if_list;

/* Coll trace globals used by inline trace functions */
extern bool cxip_coll_trace_muted;
extern bool cxip_coll_trace_append;
extern bool cxip_coll_trace_linebuf;
extern int cxip_coll_trace_rank;
extern int cxip_coll_trace_numranks;
extern FILE *cxip_coll_trace_fid;
extern bool cxip_coll_prod_trace_initialized;
extern uint64_t cxip_coll_trace_mask;

/* Split headers - types, macros, and function declarations */
#include "cxip/enums.h"
#include "cxip/env.h"
#include "cxip/cmdq.h"
#include "cxip/ptelist_buf.h"
#include "cxip/eq.h"
#include "cxip/cq.h"
#include "cxip/pte.h"
#include "cxip/req_buf.h"
#include "cxip/addr.h"
#include "cxip/coll_trace.h"
#include "cxip/log.h"
#include "cxip/portals_table.h"
#include "cxip/fabric.h"
#include "cxip/cntr.h"
#include "cxip/zbcoll.h"
#include "cxip/repsum.h"
#include "cxip/curl.h"
#include "cxip/info.h"
#include "cxip/msg_hpc.h"
#include "cxip/iomm.h"
#include "cxip/auth.h"
#include "cxip/rma.h"
#include "cxip/atomic.h"
#include "cxip/nic.h"
#include "cxip/common.h"
#include "cxip/req.h"
#include "cxip/av.h"
#include "cxip/coll.h"
#include "cxip/msg.h"
#include "cxip/if.h"
#include "cxip/telemetry.h"
#include "cxip/evtq.h"
#include "cxip/rdzv_pte.h"
#include "cxip/mr_lac_cache.h"
#include "cxip/rxc.h"
#include "cxip/txc.h"
#include "cxip/ctrl.h"
#include "cxip/ep.h"
#include "cxip/mr.h"
#include "cxip/fc.h"
#include "cxip/dom.h"

/*
 * Inline function definitions
 *
 * These are kept here (not in split headers) because they often
 * access struct members from multiple modules, requiring all types
 * to be fully defined first.
 */

static inline bool cxip_software_pte_allowed(enum cxip_ep_ptle_mode rx_match_mode)
{
	return rx_match_mode != CXIP_PTLTE_HARDWARE_MODE;
}

static inline
uint64_t cxip_adjust_remote_offset(uint64_t *addr, uint64_t key)
{
	struct cxip_mr_key cxip_key = {
		.raw = key,
	};

	if (cxip_key.cached) {
		*addr += cxip_key.lac_off;
		if (*addr & ~CXIP_MR_VALID_OFFSET_MASK)
			return -FI_EINVAL;
	}
	return FI_SUCCESS;
}

static inline bool cxip_domain_mr_cache_enabled(struct cxip_domain *dom)
{
	return dom->iomm.domain == &dom->util_domain;
}

static inline bool cxip_domain_mr_cache_iface_enabled(struct cxip_domain *dom,
						      enum fi_hmem_iface iface)
{
	return cxip_domain_mr_cache_enabled(dom) && dom->iomm.monitors[iface];
}

static inline ssize_t
cxip_copy_to_hmem_iov(struct cxip_domain *domain, enum fi_hmem_iface hmem_iface,
		      uint64_t device, const struct iovec *hmem_iov,
		      size_t hmem_iov_count, uint64_t hmem_iov_offset,
		      const void *src, size_t size)
{
	return domain->hmem_ops.copy_to_hmem_iov(hmem_iface, device, hmem_iov,
						 hmem_iov_count,
						 hmem_iov_offset, src, size);
}

static inline bool cxip_is_trig_req(struct cxip_req *req)
{
	return req->trig_cntr != NULL;
}

static inline uint16_t cxip_evtq_eqn(struct cxip_evtq *evtq)
{
	return evtq->eq->eqn;
}

static inline void cxip_cntr_progress_inc(struct cxip_cntr *cntr)
{
	ofi_genlock_lock(&cntr->progress_count_lock);
	assert(cntr->progress_count >= 0);
	cntr->progress_count++;
	ofi_genlock_unlock(&cntr->progress_count_lock);
}

static inline void cxip_cntr_progress_dec(struct cxip_cntr *cntr)
{
	ofi_genlock_lock(&cntr->progress_count_lock);
	cntr->progress_count--;
	assert(cntr->progress_count >= 0);
	ofi_genlock_unlock(&cntr->progress_count_lock);
}

static inline unsigned int cxip_cntr_progress_get(struct cxip_cntr *cntr)
{
	unsigned int count;

	ofi_genlock_lock(&cntr->progress_count_lock);
	count = cntr->progress_count;
	ofi_genlock_unlock(&cntr->progress_count_lock);

	return count;
}

static inline int fls64(uint64_t x)
{
	if (!x)
		return 0;

	return (sizeof(x) * 8) - __builtin_clzl(x);
}

static inline void cxip_msg_counters_init(struct cxip_msg_counters *cntrs)
{
	int i;
	int j;
	int k;

	for (i = 0; i < CXIP_LIST_COUNTS; i++) {
		for (j = 0; j < OFI_HMEM_MAX; j++) {
			for (k = 0; k < CXIP_COUNTER_BUCKETS; k++)
				ofi_atomic_initialize32(&cntrs->msg_count[i][j][k], 0);
		}
	}
}

static inline void
cxip_msg_counters_msg_record(struct cxip_msg_counters *cntrs,
			     enum c_ptl_list list, enum fi_hmem_iface buf_type,
			     size_t msg_size)
{
	unsigned int bucket;

	/* Buckets to bytes
	 * Bucket 0: 0 bytes
	 * Bucket 1: 1 byte
	 * Bucket 2: 2 bytes
	 * Bucket 3: 4 bytes
	 * ...
	 * Bucket CXIP_BUCKET_MAX: (1 << (CXIP_BUCKET_MAX - 1))
	 */

	/* Round size up to the nearest power of 2. */
	bucket = fls64(msg_size);
	if ((1ULL << bucket) < msg_size)
		bucket++;

	bucket = MIN(CXIP_BUCKET_MAX, bucket);

	ofi_atomic_add32(&cntrs->msg_count[list][buf_type][bucket], 1);
}

static inline void cxip_copy_to_md(struct cxip_md *md, void *dest,
				   const void *src, size_t size,
				   bool require_dev_reg_copy)
{
	ssize_t ret __attribute__((unused));
	struct iovec iov;
	bool dev_reg_copy = require_dev_reg_copy ||
		(md->handle_valid && size <= cxip_env.safe_devmem_copy_threshold);

	/* Favor dev reg access instead of relying on HMEM copy functions. */
	if (dev_reg_copy) {
		ret = ofi_hmem_dev_reg_copy_to_hmem(md->info.iface, md->handle,
						    dest, src, size);
		assert(ret == FI_SUCCESS);
	} else {
		iov.iov_base = dest;
		iov.iov_len = size;

		ret = md->dom->hmem_ops.copy_to_hmem_iov(md->info.iface,
							 md->info.device, &iov,
							 1, 0, src, size);
		assert(ret == size);
	}
}

static inline void cxip_copy_from_md(struct cxip_md *md, void *dest,
				     const void *src, size_t size,
				     bool require_dev_reg_copy)
{
	ssize_t ret __attribute__((unused));
	struct iovec iov;
	bool dev_reg_copy = require_dev_reg_copy ||
		(md->handle_valid && size <= cxip_env.safe_devmem_copy_threshold);

	/* Favor dev reg access instead of relying on HMEM copy functions. */
	if (dev_reg_copy) {
		ret = ofi_hmem_dev_reg_copy_from_hmem(md->info.iface,
						      md->handle,
						      dest, src, size);
		assert(ret == FI_SUCCESS);
	} else {
		iov.iov_base = (void *)src;
		iov.iov_len = size;


		ret = md->dom->hmem_ops.copy_from_hmem_iov(dest, size,
							   md->info.iface,
							   md->info.device,
							   &iov, 1, 0);
		assert(ret == size);
	}
}

static inline void
cxip_ep_obj_copy_to_md(struct cxip_ep_obj *ep, struct cxip_md *md, void *dest,
		       const void *src, size_t size)
{
	cxip_copy_to_md(md, dest, src, size,
			ep->require_dev_reg_copy[md->info.iface]);
}

static inline void
cxip_ep_obj_copy_from_md(struct cxip_ep_obj *ep, struct cxip_md *md, void *dest,
			 const void *src, size_t size)
{
	cxip_copy_from_md(md, dest, src, size,
			  ep->require_dev_reg_copy[md->info.iface]);
}

static inline bool cxip_ep_obj_mr_relaxed_order(struct cxip_ep_obj *ep)
{
	if (cxip_env.mr_target_ordering ==  MR_ORDER_STRICT)
		return false;

	if (cxip_env.mr_target_ordering ==  MR_ORDER_RELAXED)
		return true;

	if ((ep->rx_attr.msg_order & FI_ORDER_RMA_WAW) &&
	     ep->ep_attr.max_order_waw_size != 0)
		return false;

	if ((ep->rx_attr.msg_order & FI_ORDER_WAW) &&
	    ep->ep_attr.max_order_waw_size != 0)
		return false;

	return true;
}

static inline void cxip_txc_otx_reqs_inc(struct cxip_txc *txc)
{
	assert(ofi_genlock_held(&txc->ep_obj->lock) == 1);
	txc->otx_reqs++;
}

static inline void cxip_txc_otx_reqs_dec(struct cxip_txc *txc)
{
	assert(ofi_genlock_held(&txc->ep_obj->lock) == 1);
	txc->otx_reqs--;
	assert(txc->otx_reqs >= 0);
}

static inline int cxip_txc_otx_reqs_get(struct cxip_txc *txc)
{
	assert(ofi_genlock_held(&txc->ep_obj->lock) == 1);
	return txc->otx_reqs;
}

static inline void cxip_txc_otx_reqs_init(struct cxip_txc *txc)
{
	txc->otx_reqs = 0;
}

static inline void cxip_rxc_orx_reqs_inc(struct cxip_rxc *rxc)
{
	assert(ofi_genlock_held(&rxc->ep_obj->lock) == 1);
	rxc->orx_reqs++;
}

static inline void cxip_rxc_orx_reqs_dec(struct cxip_rxc *rxc)
{
	assert(ofi_genlock_held(&rxc->ep_obj->lock) == 1);
	rxc->orx_reqs--;
	assert(rxc->orx_reqs >= 0);
}

static inline int cxip_rxc_orx_reqs_get(struct cxip_rxc *rxc)
{
	assert(ofi_genlock_held(&rxc->ep_obj->lock) == 1);
	return rxc->orx_reqs;
}

static inline void cxip_rxc_orx_reqs_init(struct cxip_rxc *rxc)
{
	rxc->orx_reqs = 0;
}

static inline int cxip_av_entry_count(struct cxip_av *av)
{
	return ofi_atomic_get32(&av->av_entry_cnt);
}

static inline uint64_t _dbl2bits(double d)
{
#if (BYTE_ORDER == LITTLE_ENDIAN)
	union cxip_dbl_bits x = {.dval = d};
	return x.ival;
#else
#error "Unsupported processor byte ordering"
#endif
}

static inline double _bits2dbl(uint64_t i)
{
#if (BYTE_ORDER == LITTLE_ENDIAN)
	union cxip_dbl_bits x = {.ival = i};
	return x.dval;
#else
#error "Unsupported processor byte ordering"
#endif
}

static inline void _decompose_dbl(double d, int *sgn, int *exp,
				  unsigned long *man)
{
#if (BYTE_ORDER == LITTLE_ENDIAN)
	union cxip_dbl_bits x = {.dval = d};
	*sgn = (x.sign) ? -1 : 1;
	*exp = x.exponent;
	*man = x.mantissa;
#else
#error "Unsupported processor byte ordering"
#endif
}

static inline void single_to_double_quote(char *str)
{
	do {if (*str == '\'') *str = '"';} while (*(++str));
}

static inline bool cxip_cmdq_empty(struct cxip_cmdq *cmdq)
{
	return cxi_cq_empty(cmdq->dev_cmdq);
}

static inline bool cxip_cmdq_match(struct cxip_cmdq *cmdq, uint16_t vni,
				   enum cxi_traffic_class tc,
				   enum cxi_traffic_class_type tc_type)
{
	return (cmdq->cur_cp->vni == vni) && (cmdq->cur_cp->tc == tc) &&
		(cmdq->cur_cp->tc_type == tc_type);
}

static inline bool cxip_cmdq_prev_match(struct cxip_cmdq *cmdq, uint16_t vni,
					enum cxi_traffic_class tc,
					enum cxi_traffic_class_type tc_type)
{
	return (cmdq->prev_cp->vni == vni) && (cmdq->prev_cp->tc == tc) &&
		(cmdq->prev_cp->tc_type == tc_type);
}

static inline struct fid_peer_srx *cxip_get_owner_srx(struct cxip_rxc *rxc)
{
	return rxc->ep_obj->owner_srx;
}

static inline int cxip_fc_reason(const union c_event *event)
{
	if (!event->tgt_long.initiator.state_change.sc_nic_auto)
		return CXIP_FC_SOFTWARE_INITIATED;

	return event->tgt_long.initiator.state_change.sc_reason;
}

static inline void cxip_txq_ring(struct cxip_cmdq *cmdq, bool more,
				 int otx_reqs)
{
	if (!more) {
		switch (cmdq->llring_mode) {
		case CXIP_LLRING_IDLE:
			if (!otx_reqs)
				cxi_cq_ll_ring(cmdq->dev_cmdq);
			else
				cxi_cq_ring(cmdq->dev_cmdq);
			break;
		case CXIP_LLRING_ALWAYS:
			cxi_cq_ll_ring(cmdq->dev_cmdq);
			break;
		case CXIP_LLRING_NEVER:
		default:
			cxi_cq_ring(cmdq->dev_cmdq);
			break;
		}
	}
}

static inline int cxip_no_discard(struct fi_peer_rx_entry *rx_entry)
{
	return -FI_ENOSYS;
}

static inline void
cxip_domain_add_txc(struct cxip_domain *dom, struct cxip_txc *txc)
{
	ofi_spin_lock(&dom->lock);
	dlist_insert_tail(&txc->dom_entry, &dom->txc_list);
	ofi_spin_unlock(&dom->lock);
}

static inline void
cxip_domain_remove_txc(struct cxip_domain *dom, struct cxip_txc *txc)
{
	ofi_spin_lock(&dom->lock);
	dlist_remove(&txc->dom_entry);
	ofi_spin_unlock(&dom->lock);
}

static inline void
cxip_domain_add_cntr(struct cxip_domain *dom, struct cxip_cntr *cntr)
{
	ofi_spin_lock(&dom->lock);
	dlist_insert_tail(&cntr->dom_entry, &dom->cntr_list);
	ofi_atomic_inc32(&dom->ref);
	ofi_spin_unlock(&dom->lock);
}

static inline void
cxip_domain_remove_cntr(struct cxip_domain *dom, struct cxip_cntr *cntr)
{
	ofi_spin_lock(&dom->lock);
	dlist_remove(&cntr->dom_entry);
	ofi_atomic_dec32(&dom->ref);
	ofi_spin_unlock(&dom->lock);
}

static inline void
cxip_domain_add_cq(struct cxip_domain *dom, struct cxip_cq *cq)
{
	ofi_spin_lock(&dom->lock);
	dlist_insert_tail(&cq->dom_entry, &dom->cq_list);
	ofi_atomic_inc32(&dom->ref);
	ofi_spin_unlock(&dom->lock);
}

static inline void
cxip_domain_remove_cq(struct cxip_domain *dom, struct cxip_cq *cq)
{
	ofi_spin_lock(&dom->lock);
	dlist_remove(&cq->dom_entry);
	ofi_atomic_dec32(&dom->ref);
	ofi_spin_unlock(&dom->lock);
}

static inline
struct cxip_ctrl_req *cxip_domain_ctrl_id_at(struct cxip_domain *dom,
					     int buffer_id)
{
	if (ofi_idx_is_valid(&dom->req_ids, buffer_id))
		return ofi_idx_at(&dom->req_ids, buffer_id);
	return NULL;
}

static inline uint32_t cxip_mac_to_nic(struct ether_addr *mac)
{
	return mac->ether_addr_octet[5] |
			(mac->ether_addr_octet[4] << 8) |
			((mac->ether_addr_octet[3] & 0xF) << 16);
}

static inline bool is_netsim(struct cxip_ep_obj *ep_obj)
{
	return (ep_obj->domain->iface->info->device_platform ==
		CXI_PLATFORM_NETSIM);
}

static inline void cxip_coll_trace_set(int mod)
{
	cxip_coll_trace_mask |= (1L << mod);
}

static inline void cxip_coll_trace_clr(int mod)
{
	cxip_coll_trace_mask &= ~(1L << mod);
}

static inline bool cxip_coll_trace_true(int mod)
{
	return (!cxip_coll_trace_muted) && (cxip_coll_trace_mask & (1L << mod));
}

static inline bool cxip_coll_prod_trace_true(void)
{
	return cxip_coll_prod_trace_initialized;
}

static inline int cxip_cacheline_size(void)
{
	FILE *f;
	int cache_line_size;
	int ret;

	f = fopen(CXIP_SYSFS_CACHE_LINE_SIZE, "r");
	if (!f) {
		_CXIP_WARN(FI_LOG_CORE,
			   "Error %d determining cacheline size\n",
			   errno);
		cache_line_size = CXIP_DEFAULT_CACHE_LINE_SIZE;
	} else {
		ret = fscanf(f, "%d", &cache_line_size);
		if (ret != 1) {
			_CXIP_WARN(FI_LOG_CORE,
				   "Error reading cacheline size\n");
			cache_line_size = CXIP_DEFAULT_CACHE_LINE_SIZE;
		}

		fclose(f);
	}

	return cache_line_size;
}

static inline int
cxip_txc_copy_from_hmem(struct cxip_txc *txc, struct cxip_md *hmem_md,
			void *dest, const void *hmem_src, size_t size)
{
	enum fi_hmem_iface iface;
	uint64_t device;
	struct iovec hmem_iov;
	struct cxip_domain *domain = txc->domain;
	uint64_t flags;
	bool unmap_hmem_md = false;
	int ret;

	/* Default to memcpy unless FI_HMEM is set. */
	if (!txc->hmem) {
		memcpy(dest, hmem_src, size);
		return FI_SUCCESS;
	}

	/* With HMEM enabled, performing memory registration will also cause
	 * the device buffer to be registered for CPU load/store access. Being
	 * able to perform load/store instead of using the generic HMEM copy
	 * routines and/or HMEM override copy routines can significantly reduce
	 * latency. Thus, this path is favored.
	 *
	 * However, if FORK_SAFE variables are enabled, we avoid this mapping
	 * to keep from designating the entire page in which the buffer
	 * resides as don't copy, and take the performance hit.
	 *
	 * Memory registration can result in additional latency. Expectation is
	 * the MR cache can amortize the additional memory registration latency.
	 */
	if (!cxip_env.fork_safe_requested) {
		if (!hmem_md) {
			ret = cxip_ep_obj_map(txc->ep_obj, hmem_src, size,
					      CXI_MAP_READ, 0, &hmem_md);
			if (ret) {
				TXC_WARN(txc, "cxip_ep_obj_map failed: %d:%s\n",
					 ret, fi_strerror(-ret));
				return ret;
			}

			unmap_hmem_md = true;
		}

		cxip_ep_obj_copy_from_md(txc->ep_obj, hmem_md, dest, hmem_src,
					 size);
		if (unmap_hmem_md)
			cxip_unmap(hmem_md);

		return FI_SUCCESS;
	}

	/* Slow path HMEM copy path.*/
	iface = ofi_get_hmem_iface(hmem_src, &device, &flags);
	hmem_iov.iov_base = (void *)hmem_src;
	hmem_iov.iov_len = size;

	ret = domain->hmem_ops.copy_from_hmem_iov(dest, size, iface, device,
						  &hmem_iov, 1, 0);
	if (ret != size) {
		if (ret < 0) {
			TXC_WARN(txc, "copy_from_hmem_iov failed: %d:%s\n", ret,
				 fi_strerror(-ret));
			return ret;
		}

		TXC_WARN(txc,
			 "copy_from_hmem_iov short copy: expect=%ld got=%d\n",
			 size, ret);
		return -FI_EIO;
	}

	return FI_SUCCESS;
}

static inline
int cxip_set_recv_match_id(struct cxip_rxc *rxc, fi_addr_t src_addr,
			   bool auth_key, uint32_t *match_id, uint16_t *vni)
{
	struct cxip_addr caddr;
	int ret;

	/* If FI_DIRECTED_RECV and a src_addr is specified, encode the address
	 * in the LE for matching. If application AVs are symmetric, use
	 * logical FI address for matching. Otherwise, use physical address.
	 */
	if (rxc->attr.caps & FI_DIRECTED_RECV &&
	    src_addr != FI_ADDR_UNSPEC) {
		if (rxc->ep_obj->av->symmetric) {
			/* PID is not used for matching */
			*match_id = CXI_MATCH_ID(rxc->pid_bits,
						C_PID_ANY, src_addr);
			*vni = rxc->ep_obj->auth_key.vni;
		} else {
			ret = cxip_av_lookup_addr(rxc->ep_obj->av, src_addr,
						  &caddr);
			if (ret != FI_SUCCESS) {
				RXC_WARN(rxc, "Failed to look up FI addr: %d\n",
					 ret);
				return -FI_EINVAL;
			}

			*match_id = CXI_MATCH_ID(rxc->pid_bits, caddr.pid,
						 caddr.nic);
			if (auth_key)
				*vni = caddr.vni;
			else
				*vni = rxc->ep_obj->auth_key.vni;
		}
	} else {
		*match_id = CXI_MATCH_ID_ANY;
		*vni = 0;
	}

	return FI_SUCCESS;
}

static inline void cxip_set_env_rx_match_mode(void)
{
	char *param_str = NULL;

	fi_param_get_str(&cxip_prov, "rx_match_mode", &param_str);
	/* Parameters to tailor hybrid hardware to software transitions
	 * that are initiated by software.
	 */
	fi_param_define(&cxip_prov, "hybrid_preemptive", FI_PARAM_BOOL,
			"Enable/Disable low LE preemptive UX transitions.");
	fi_param_get_bool(&cxip_prov, "hybrid_preemptive",
			  &cxip_env.hybrid_preemptive);
	fi_param_define(&cxip_prov, "hybrid_recv_preemptive", FI_PARAM_BOOL,
			"Enable/Disable low LE preemptive recv transitions.");
	fi_param_get_bool(&cxip_prov, "hybrid_recv_preemptive",
			  &cxip_env.hybrid_recv_preemptive);
	fi_param_define(&cxip_prov, "hybrid_unexpected_msg_preemptive",
			FI_PARAM_BOOL,
			"Enable preemptive transition to software endpoint when number of hardware unexpected messages exceeds RX attribute size");
	fi_param_get_bool(&cxip_prov, "hybrid_unexpected_msg_preemptive",
			  &cxip_env.hybrid_unexpected_msg_preemptive);
	fi_param_define(&cxip_prov, "hybrid_posted_recv_preemptive",
			FI_PARAM_BOOL,
			"Enable preemptive transition to software endpoint when number of posted receives exceeds RX attribute size");
	fi_param_get_bool(&cxip_prov, "hybrid_posted_recv_preemptive",
			  &cxip_env.hybrid_posted_recv_preemptive);

	if (param_str) {
		if (!strcasecmp(param_str, "hardware")) {
			cxip_env.rx_match_mode = CXIP_PTLTE_HARDWARE_MODE;
			cxip_env.msg_offload = true;
		} else if (!strcmp(param_str, "software")) {
			cxip_env.rx_match_mode = CXIP_PTLTE_SOFTWARE_MODE;
			cxip_env.msg_offload = false;
		} else if (!strcmp(param_str, "hybrid")) {
			cxip_env.rx_match_mode = CXIP_PTLTE_HYBRID_MODE;
			cxip_env.msg_offload = true;
		} else {
			_CXIP_WARN(FI_LOG_FABRIC, "Unrecognized rx_match_mode: %s\n",
				  param_str);
			cxip_env.rx_match_mode = CXIP_PTLTE_HARDWARE_MODE;
			cxip_env.msg_offload = true;
		}
	}

	if (cxip_env.rx_match_mode != CXIP_PTLTE_HYBRID_MODE &&
	    cxip_env.hybrid_preemptive) {
		cxip_env.hybrid_preemptive = false;
		_CXIP_WARN(FI_LOG_FABRIC, "Not in hybrid mode, ignoring preemptive\n");
	}

	if (cxip_env.rx_match_mode != CXIP_PTLTE_HYBRID_MODE &&
	    cxip_env.hybrid_recv_preemptive) {
		_CXIP_WARN(FI_LOG_FABRIC, "Not in hybrid mode, ignore LE  recv preemptive\n");
		cxip_env.hybrid_recv_preemptive = 0;
	}

	if (cxip_env.rx_match_mode != CXIP_PTLTE_HYBRID_MODE &&
	    cxip_env.hybrid_posted_recv_preemptive) {
		_CXIP_WARN(FI_LOG_FABRIC, "Not in hybrid mode, ignore hybrid_posted_recv_preemptive\n");
		cxip_env.hybrid_posted_recv_preemptive = 0;
	}

	if (cxip_env.rx_match_mode != CXIP_PTLTE_HYBRID_MODE &&
	    cxip_env.hybrid_unexpected_msg_preemptive) {
		_CXIP_WARN(FI_LOG_FABRIC, "Not in hybrid mode, ignore hybrid_unexpected_msg_preemptive\n");
		cxip_env.hybrid_unexpected_msg_preemptive = 0;
	}
}

#endif /* _CXIP_PROV_H_ */
