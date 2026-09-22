/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_rdm_ep.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_pke.h"
#include "efa_rdm_pke_rtw.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_protocol.h"
#include "efa_rdm_pke_req.h"

/**
 * @brief allcoate an RX entry for a incoming RTW packet
 *
 * The RX entry will be allocated from endpoint's OP entry
 * pool
 * @param[in]	pkt_entry	received RTW packet
 *
 * @return
 * pointer to the newly allocated RX entry.
 * NULL when OP entry pool has been exhausted.
 */
struct efa_rdm_ope *efa_rdm_pke_alloc_rtw_rxe(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *rxe;
	struct efa_rdm_base_hdr *base_hdr;

	rxe = efa_rdm_ep_alloc_rxe(pkt_entry->ep, pkt_entry->peer, ofi_op_write);
	if (OFI_UNLIKELY(!rxe))
		return NULL;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	if (base_hdr->flags & EFA_RDM_REQ_OPT_CQ_DATA_HDR) {
		rxe->cq_entry.flags |= FI_REMOTE_CQ_DATA;
		rxe->cq_entry.data = efa_rdm_pke_get_req_cq_data(pkt_entry);
	}

	rxe->internal_flags |= EFA_RDM_OPE_INTERNAL;
	return rxe;
}
