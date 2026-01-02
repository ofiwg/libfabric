/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_CTRL_H_
#define _CXIP_CTRL_H_


#include <stdint.h>
#include <ofi_list.h>

/* Forward declarations */
struct cxip_cmdq;
struct cxip_ep_obj;
struct cxip_mr;
struct cxip_pte;

/* Type definitions */
struct cxip_ctrl_req_mr {
	struct cxip_mr *mr;
};

struct cxip_ctrl_send {
	uint32_t nic_addr;
	uint32_t pid;
	uint16_t vni;
	union cxip_match_bits mb;
};

struct cxip_ctrl_req {
	struct dlist_entry ep_entry;
	struct cxip_ep_obj *ep_obj;
	int req_id;
	int (*cb)(struct cxip_ctrl_req *req, const union c_event *evt);

	union {
		struct cxip_ctrl_req_mr mr;
		struct cxip_ctrl_send send;
	};
};

struct cxip_ctrl {
	/* wait object is required to wake up CQ waiters
	 * when control progress is required.
	 */
	struct cxil_wait_obj *wait;

	struct cxi_eq *tgt_evtq;
	struct cxi_eq *tx_evtq;

	/* TX command queue is used to initiate side-band messaging
	 * and is TX credit based.
	 */
	struct cxip_cmdq *txq;
	unsigned int tx_credits;

	/* Target command queue is used for appending RX side-band
	 * messaging control LE and managing standard MR LE.
	 */
	struct cxip_cmdq *tgq;
	struct cxip_pte *pte;
	struct cxip_ctrl_req msg_req;

	/* FI_MR_PROV_KEY caching, protected with ep_obj->lock */
	struct cxip_mr_lac_cache std_mr_cache[CXIP_NUM_CACHED_KEY_LE];
	struct cxip_mr_lac_cache opt_mr_cache[CXIP_NUM_CACHED_KEY_LE];

	struct dlist_entry mr_list;

	/* Event queue buffers */
	void *tgt_evtq_buf;
	struct cxi_md *tgt_evtq_buf_md;
	void *tx_evtq_buf;
	struct cxi_md *tx_evtq_buf_md;
};

/* Function declarations */
void cxip_ctrl_mr_cache_flush(struct cxip_ep_obj *ep_obj);

int cxip_ctrl_msg_send(struct cxip_ctrl_req *req, uint64_t data);

#endif /* _CXIP_CTRL_H_ */
