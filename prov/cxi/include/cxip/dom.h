/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_DOM_H_
#define _CXIP_DOM_H_


#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <semaphore.h>
#include <ofi_list.h>
#include <ofi_atom.h>
#include <ofi_lock.h>

/* Forward declarations */
struct cxip_cmdq;
struct cxip_cntr;
struct cxip_ctrl_req;
struct cxip_eq;
struct cxip_fabric;
struct cxip_if;
struct cxip_lni;
struct cxip_mr;
struct cxip_telemetry;

/* Macros */
#define CXIP_DOM_CAPS (FI_LOCAL_COMM | FI_REMOTE_COMM | FI_AV_USER_ID | FI_PEER)

#define DOM_INFO(dom, fmt, ...) \
	_CXIP_INFO(FI_LOG_DOMAIN, "DOM (cxi%u:%u:%u:%u:%#x): " fmt "", \
		   (dom)->iface->info->dev_id, (dom)->lni->lni->id, \
		   (dom)->auth_key.svc_id, (dom)->auth_key.vni, \
		   (dom)->nic_addr, ##__VA_ARGS__)

#define DOM_WARN(dom, fmt, ...) \
	_CXIP_WARN(FI_LOG_DOMAIN, "DOM (cxi%u:%u:%u:%u:%#x): " fmt "", \
		   (dom)->iface->info->dev_id, (dom)->lni->lni->id, \
		   (dom)->auth_key.svc_id, (dom)->auth_key.vni, \
		   (dom)->nic_addr, ##__VA_ARGS__)

/* Type definitions */
struct cxip_domain_cmdq {
	struct dlist_entry entry;
	struct cxip_cmdq *cmdq;
};

struct cxip_domain {
	struct util_domain util_domain;
	struct cxip_fabric *fab;
	ofi_spin_t lock;
	ofi_atomic32_t ref;

	struct fid_ep rx_ep;
	struct fid_peer_srx *owner_srx;

	uint32_t tclass;

	struct cxip_eq *eq; //unused
	struct cxip_eq *mr_eq; //unused

	/* Assigned NIC address */
	uint32_t nic_addr;

	/* Device info */
	struct cxip_if *iface;

	/* Device partition */
	struct cxip_lni *lni;

	/* Trigger and CT support */
	struct cxip_cmdq *trig_cmdq;
	struct ofi_genlock trig_cmdq_lock;
	bool cntr_init;

	/* Provider generated RKEYs, else client */
	bool is_prov_key;

	/* Can disable caching of provider generated RKEYs */
	bool prov_key_cache;

	/* Provider generated RKEYs optimized MR disablement/enablement */
	bool optimized_mrs;

	/* Enable MR match event counting enables a more robust
	 * MR when using FI_MR_PROV_KEY. It disables hardware cached
	 * MR keys and ensures memory backing a MR cannot be
	 * remotely accessed even if that memory remains in the
	 * libfabric MR cache.
	 */
	bool mr_match_events;

	/* Domain wide MR resources.
	 *   Req IDs are control buffer IDs to map MR or MR cache to an LE.
	 *   MR IDs are used by non-cached provider key MR to decouple the
	 *   MR and Req ID, and do not map directly to the MR LE.
	 */
	ofi_spin_t ctrl_id_lock;
	struct indexer req_ids;
	struct indexer mr_ids;

	/* If FI_MR_PROV_KEY is not cached, keys include a sequence number
	 * to reduce the likelyhood of a stale key being used to access
	 * a recycled MR key.
	 */
	uint32_t prov_key_seqnum;

	/* Translation cache */
	struct ofi_mr_cache iomm;
	bool odp;
	bool ats;
	bool hmem;

	/* ATS translation support */
	struct cxip_md scalable_md;
	bool scalable_iomm;
	bool rocr_dev_mem_only;

	/* Domain state */
	bool enabled;

	/* List of allocated resources used for deferred work queue processing.
	 */
	struct dlist_entry txc_list;
	struct dlist_entry cntr_list;
	struct dlist_entry cq_list;

	struct fi_hmem_override_ops hmem_ops;
	bool hybrid_mr_desc;

	/* Container of in-use MRs against this domain. */
	struct cxip_mr_domain mr_domain;

	/* Counters collected for the duration of the domain existence. */
	struct cxip_telemetry *telemetry;

	/* NIC AMO operation which is remapped to a PCIe operation. */
	int amo_remap_to_pcie_fadd;

	/* Maximum number of triggered operations configured for the service
	 * ID.
	 */
	int max_trig_op_in_use;
	sem_t *trig_op_lock;

	/* Domain has been configured with FI_AV_AUTH_KEY. */
	bool av_auth_key;

	/* This is only valid if FI_AV_AUTH_KEY is false. */
	struct cxi_auth_key auth_key;

	/* Maximum number of auth keys requested by user. */
	size_t auth_key_entry_max;

	/* Domain has been configured with FI_AV_USER_ID. */
	bool av_user_id;

	/* Domain level TX command queues used when number of authorization
	 * keys exceeds LCID limit.
	 */
	struct dlist_entry cmdq_list;
	unsigned int cmdq_cnt;
	struct ofi_genlock cmdq_lock;
	size_t tx_size;

	/* domain level match mode override */
	enum cxip_ep_ptle_mode rx_match_mode;
	bool msg_offload;
	size_t req_buf_size;

};

/* Function declarations */
int cxip_domain_emit_idc_put(struct cxip_domain *dom, uint16_t vni,
			     enum cxi_traffic_class tc,
			     const struct c_cstate_cmd *c_state,
			     const struct c_idc_put_cmd *put, const void *buf,
			     size_t len, uint64_t flags);

int cxip_domain_emit_dma(struct cxip_domain *dom, uint16_t vni,
			 enum cxi_traffic_class tc, struct c_full_dma_cmd *dma,
			 uint64_t flags);

int cxip_domain_emit_idc_amo(struct cxip_domain *dom, uint16_t vni,
			     enum cxi_traffic_class tc,
			     const struct c_cstate_cmd *c_state,
			     const struct c_idc_amo_cmd *amo, uint64_t flags,
			     bool fetching, bool flush);

int cxip_domain_emit_dma_amo(struct cxip_domain *dom, uint16_t vni,
			     enum cxi_traffic_class tc,
			     struct c_dma_amo_cmd *amo, uint64_t flags,
			     bool fetching, bool flush);

int cxip_domain_emit_idc_msg(struct cxip_domain *dom, uint16_t vni,
			     enum cxi_traffic_class tc,
			     const struct c_cstate_cmd *c_state,
			     const struct c_idc_msg_hdr *msg, const void *buf,
			     size_t len, uint64_t flags);

int cxip_domain_valid_vni(struct cxip_domain *dom, struct cxi_auth_key *key);

int cxip_domain(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **dom, void *context);

void cxip_dom_cntr_disable(struct cxip_domain *dom);

int cxip_domain_ctrl_id_alloc(struct cxip_domain *dom,
			      struct cxip_ctrl_req *req);

void cxip_domain_ctrl_id_free(struct cxip_domain *dom,
			      struct cxip_ctrl_req *req);

int cxip_domain_prov_mr_id_alloc(struct cxip_domain *dom,
				 struct cxip_mr *mr);

void cxip_domain_prov_mr_id_free(struct cxip_domain *dom,
				 struct cxip_mr *mr);

int cxip_domain_dwq_emit_dma(struct cxip_domain *dom, uint16_t vni,
			     enum cxi_traffic_class tc,
			     enum cxi_traffic_class_type tc_type,
			     struct cxip_cntr *trig_cntr, size_t trig_thresh,
			     struct c_full_dma_cmd *dma, uint64_t flags);

int cxip_domain_dwq_emit_amo(struct cxip_domain *dom, uint16_t vni,
			     enum cxi_traffic_class tc,
			     enum cxi_traffic_class_type tc_type,
			     struct cxip_cntr *trig_cntr, size_t trig_thresh,
			     struct c_dma_amo_cmd *amo, uint64_t flags,
			     bool fetching, bool flush);

#endif /* _CXIP_DOM_H_ */
