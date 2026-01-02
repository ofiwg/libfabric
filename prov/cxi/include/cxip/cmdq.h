/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_CMDQ_H_
#define _CXIP_CMDQ_H_


#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Forward declarations */
struct cxip_lni;

/* Type definitions */
struct cxip_cmdq {
	struct cxi_cq *dev_cmdq;
	struct c_cstate_cmd c_state;
	enum cxip_llring_mode llring_mode;

	struct cxi_cp *cur_cp;
	struct cxi_cp *prev_cp;
	struct cxip_lni *lni;
};

/* Function declarations */
int cxip_cmdq_emit_idc_put(struct cxip_cmdq *cmdq,
			   const struct c_cstate_cmd *c_state,
			   const struct c_idc_put_cmd *put, const void *buf,
			   size_t len, uint64_t flags);

int cxip_cmdq_emit_dma(struct cxip_cmdq *cmdq, struct c_full_dma_cmd *dma,
		       uint64_t flags);

int cxip_cmdq_emic_idc_amo(struct cxip_cmdq *cmdq,
			   const struct c_cstate_cmd *c_state,
			   const struct c_idc_amo_cmd *amo, uint64_t flags,
			   bool fetching, bool flush);

int cxip_cmdq_emit_dma_amo(struct cxip_cmdq *cmdq, struct c_dma_amo_cmd *amo,
			   uint64_t flags, bool fetching, bool flush);

int cxip_cmdq_emit_idc_msg(struct cxip_cmdq *cmdq,
			   const struct c_cstate_cmd *c_state,
			   const struct c_idc_msg_hdr *msg, const void *buf,
			   size_t len, uint64_t flags);

enum cxi_traffic_class cxip_ofi_to_cxi_tc(uint32_t ofi_tclass);

int cxip_cmdq_cp_set(struct cxip_cmdq *cmdq, uint16_t vni,
		     enum cxi_traffic_class tc,
		     enum cxi_traffic_class_type tc_type);

int cxip_cmdq_cp_modify(struct cxip_cmdq *cmdq, uint16_t vni,
			enum cxi_traffic_class tc);

int cxip_cmdq_alloc(struct cxip_lni *lni, struct cxi_eq *evtq,
		    struct cxi_cq_alloc_opts *cq_opts, uint16_t vni,
		    enum cxi_traffic_class tc,
		    enum cxi_traffic_class_type tc_type,
		    struct cxip_cmdq **cmdq);

void cxip_cmdq_free(struct cxip_cmdq *cmdq);

int cxip_cmdq_emit_c_state(struct cxip_cmdq *cmdq,
			   const struct c_cstate_cmd *cmd);

#endif /* _CXIP_CMDQ_H_ */
