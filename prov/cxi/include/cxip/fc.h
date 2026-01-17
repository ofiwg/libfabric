/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_FC_H_
#define _CXIP_FC_H_

#include <ofi_list.h>
#include <stdbool.h>
#include <stdint.h>

/* Forward declarations */
struct cxip_ep_obj;
struct cxip_rxc_hpc;
struct cxip_txc_hpc;

/* Macros */
#define CXIP_FC_SOFTWARE_INITIATED -1

/* Type definitions */
struct cxip_fc_peer {
	struct dlist_entry txc_entry;
	struct cxip_txc_hpc *txc;
	struct cxip_ctrl_req req;
	struct cxip_addr caddr;
	struct dlist_entry msg_queue;
	uint16_t pending;
	uint16_t dropped;
	uint16_t pending_acks;
	bool replayed;
	unsigned int retry_count;
};

struct cxip_fc_drops {
	struct dlist_entry rxc_entry;
	struct cxip_rxc_hpc *rxc;
	struct cxip_ctrl_req req;
	uint32_t nic_addr;
	uint32_t pid;
	uint16_t vni;
	uint16_t drops;
	unsigned int retry_count;
};

/* Function declarations */
int cxip_fc_process_drops(struct cxip_ep_obj *ep_obj, uint32_t nic_addr,
			  uint32_t pid, uint16_t vni, uint16_t drops);

int cxip_fc_resume(struct cxip_ep_obj *ep_obj, uint32_t nic_addr, uint32_t pid,
		   uint16_t vni);

#endif /* _CXIP_FC_H_ */
