/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_EP_H_
#define _CXIP_EP_H_


#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <ofi_list.h>
#include <ofi_atom.h>

/* Forward declarations */
struct cxip_av;
struct cxip_cmdq;
struct cxip_cq;
struct cxip_domain;
struct cxip_eq;
struct cxip_md;
struct cxip_portals_table;
struct cxip_rxc;
struct cxip_txc;

/* Macros */
#define CXIP_EP_MAX_CTX_BITS		0

#define CXIP_EP_MAX_TX_CNT		(1 << CXIP_EP_MAX_CTX_BITS)

#define CXIP_EP_MAX_RX_CNT		(1 << CXIP_EP_MAX_CTX_BITS)

#define CXIP_EP_MAX_MSG_SZ		((1ULL << 32) - 1)

#define CXIP_EP_MIN_MULTI_RECV		64

#define CXIP_EP_MAX_MULTI_RECV		((1 << 24) - 1)

#define CXIP_EP_PRI_CAPS \
	(FI_RMA | FI_ATOMICS | FI_TAGGED | FI_RECV | FI_SEND | \
	 FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE | \
	 FI_DIRECTED_RECV | FI_MSG | FI_NAMED_RX_CTX | FI_HMEM | \
	 FI_COLLECTIVE)

#define CXIP_EP_SEC_CAPS \
	(FI_SOURCE | FI_SOURCE_ERR | FI_LOCAL_COMM | \
	 FI_REMOTE_COMM | FI_RMA_EVENT | FI_MULTI_RECV | FI_FENCE | FI_TRIGGER)

#define CXIP_EP_CAPS (CXIP_EP_PRI_CAPS | CXIP_EP_SEC_CAPS)

#define CXIP_EP_CQ_FLAGS \
	(FI_SEND | FI_TRANSMIT | FI_RECV | FI_SELECTIVE_COMPLETION)

#define CXIP_EP_CNTR_FLAGS \
	(FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | \
	 FI_REMOTE_WRITE)

/* Type definitions */
struct cxip_ep_obj {
	/* Allow lock to be optimized out with FI_THREAD_DOMAIN */
	struct ofi_genlock lock;
	struct cxip_domain *domain;
	struct cxip_av *av;

	struct fid_peer_srx *owner_srx;

	/* Domain has been configured with FI_AV_AUTH_KEY. */
	bool av_auth_key;

	/* This is only valid if FI_AV_AUTH_KEY is false. */
	struct cxi_auth_key auth_key;

	/* Array of VNIs if FI_AV_AUTH_KEY is true. */
	uint16_t *vnis;
	size_t vni_count;

	struct cxip_addr src_addr;
	fi_addr_t fi_addr;

	bool enabled;

	/* Endpoint protocol implementations.
	 * FI_PROTO_CXI - Portals SAS protocol
	 */
	uint32_t protocol;
	struct cxip_txc *txc;
	struct cxip_rxc *rxc;

	/* Internal support for CQ wait object */
	struct cxil_wait_obj *priv_wait;
	int wait_fd;

	/* ASIC version associated with EP/Domain */
	enum cassini_version asic_ver;

	/* Information that might be owned by an EP (or a SEP
	 * when implemented). Should ultimately be a pointer
	 * to a base/specialization.
	 */
	struct cxip_ctrl ctrl;

	/* Command queues. Each EP has 1 transmit and 1 target
	 * command queue that can be shared. An optional 2nd transmit
	 * command queue may be created for RX initiated rgets.
	 */
	struct cxip_cmdq *txq;
	ofi_atomic32_t txq_ref;
	struct cxip_cmdq *tgq;
	ofi_atomic32_t tgq_ref;
	struct cxip_cmdq *rx_txq;

	/* Libfabric software EQ resource */
	struct cxip_eq *eq;
	struct dlist_entry eq_link;

	/* Values at base EP creation */
	uint64_t caps;
	struct fi_ep_attr ep_attr;
	struct fi_tx_attr tx_attr;
	struct fi_rx_attr rx_attr;

	/* Require memcpy's via the dev reg APIs. */
	bool require_dev_reg_copy[OFI_HMEM_MAX];

	/* Collectives support */
	struct cxip_ep_coll_obj coll;
	struct cxip_ep_zbcoll_obj zbcoll;

	size_t txq_size;
	size_t tgq_size;
	ofi_atomic32_t ref;
	struct cxip_portals_table *ptable;
};

struct cxip_ep {
	struct fid_ep ep;
	struct fi_tx_attr tx_attr;
	struct fi_rx_attr rx_attr;
	struct cxip_ep_obj *ep_obj;
	int is_alias;
};

/* Function declarations */
int cxip_ep_obj_map(struct cxip_ep_obj *ep, const void *buf, unsigned long len,
		    uint64_t access, uint64_t flags, struct cxip_md **md);

int cxip_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep, void *context);

int cxip_ep_cmdq(struct cxip_ep_obj *ep_obj, bool transmit, uint32_t tclass,
		 struct cxi_eq *evtq, struct cxip_cmdq **cmdq);

void cxip_ep_cmdq_put(struct cxip_ep_obj *ep_obj, bool transmit);

void cxip_ep_progress(struct fid *fid);

void cxip_ep_flush_trig_reqs(struct cxip_ep_obj *ep_obj);

void cxip_ep_ctrl_progress(struct cxip_ep_obj *ep_obj, bool internal);

void cxip_ep_ctrl_progress_locked(struct cxip_ep_obj *ep_obj, bool internal);

void cxip_ep_tx_ctrl_progress(struct cxip_ep_obj *ep_obj, bool internal);

void cxip_ep_tx_ctrl_progress_locked(struct cxip_ep_obj *ep_obj, bool internal);

void cxip_ep_tgt_ctrl_progress(struct cxip_ep_obj *ep_obj, bool internal);

void cxip_ep_tgt_ctrl_progress_locked(struct cxip_ep_obj *ep_obj,
				      bool internal);

int cxip_ep_ctrl_init(struct cxip_ep_obj *ep_obj);

void cxip_ep_ctrl_fini(struct cxip_ep_obj *ep_obj);

int cxip_ep_trywait(struct cxip_ep_obj *ep_obj, struct cxip_cq *cq);

size_t cxip_ep_get_unexp_msgs(struct fid_ep *fid_ep,
			      struct fi_cq_tagged_entry *entry, size_t count,
			      fi_addr_t *src_addr, size_t *ux_count);

#endif /* _CXIP_EP_H_ */
