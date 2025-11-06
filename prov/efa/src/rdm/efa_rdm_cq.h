/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_RDM_CQ_H
#define EFA_RDM_CQ_H

#include "efa_cq.h"
#include "efa_data_path_ops.h"
#include <ofi_util.h>

struct efa_rdm_cq {
	struct efa_cq efa_cq;
	struct fid_cq *shm_cq;
	struct dlist_entry ibv_cq_poll_list;
	bool need_to_scan_ep_list;
};

int efa_rdm_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		    struct fid_cq **cq_fid, void *context);

void efa_rdm_cq_poll_ibv_cq_closing_ep(struct efa_ibv_cq *ibv_cq, struct efa_rdm_ep *closing_ep);
int efa_rdm_cq_poll_ibv_cq(ssize_t cqe_to_process, struct efa_ibv_cq *ibv_cq);

#if ENABLE_DEBUG
static inline struct efa_rdm_pke *efa_rdm_cq_get_pke_from_wr_id(uint64_t wr_id)
{
	struct efa_rdm_pke *pkt_entry;
	uint8_t gen = wr_id & (EFA_RDM_BUFPOOL_ALIGNMENT - 1);
	wr_id &= ~(EFA_RDM_BUFPOOL_ALIGNMENT - 1);
	pkt_entry = (struct efa_rdm_pke *) wr_id;
	assert(pkt_entry->gen == gen);
	return pkt_entry;
}
#endif

#endif
