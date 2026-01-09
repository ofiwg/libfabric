/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_RDZV_PTE_H_
#define _CXIP_RDZV_PTE_H_

#include <ofi_atom.h>

/* Forward declarations */
struct cxip_pte;
struct cxip_req;
struct cxip_txc_hpc;

/* Type definitions */
struct cxip_rdzv_pte {
	struct cxip_txc_hpc *txc;
	struct cxip_pte *pte;

	/* Count of the number of buffers successfully linked on this PtlTE. */
	ofi_atomic32_t le_linked_success_count;

	/* Count of the number of buffers failed to link on this PtlTE. */
	ofi_atomic32_t le_linked_failure_count;
};

struct cxip_rdzv_match_pte {
	struct cxip_rdzv_pte base_pte;

	/* Request structure used to handle zero byte puts used for match
	 * complete.
	 */
	struct cxip_req *zbp_req;

	/* Request structures used to handle rendezvous source/data transfers.
	 * There is one request structure (and LE) for each LAC.
	 */
	struct cxip_req *src_reqs[RDZV_SRC_LES];
};

struct cxip_rdzv_nomatch_pte {
	struct cxip_rdzv_pte base_pte;
	struct cxip_req *le_req;
};

/* Function declarations */
int cxip_rdzv_match_pte_alloc(struct cxip_txc_hpc *txc,
			      struct cxip_rdzv_match_pte **rdzv_pte);

int cxip_rdzv_nomatch_pte_alloc(struct cxip_txc_hpc *txc, int lac,
				struct cxip_rdzv_nomatch_pte **rdzv_pte);

int cxip_rdzv_pte_src_req_alloc(struct cxip_rdzv_match_pte *pte, int lac);

void cxip_rdzv_match_pte_free(struct cxip_rdzv_match_pte *pte);

void cxip_rdzv_nomatch_pte_free(struct cxip_rdzv_nomatch_pte *pte);

int cxip_rdzv_pte_zbp_cb(struct cxip_req *req, const union c_event *event);

int cxip_rdzv_pte_src_cb(struct cxip_req *req, const union c_event *event);

#endif /* _CXIP_RDZV_PTE_H_ */
