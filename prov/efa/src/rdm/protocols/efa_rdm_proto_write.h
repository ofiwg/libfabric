/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_RDM_PROTO_WRITE_H
#define EFA_RDM_PROTO_WRITE_H

#include "efa_rdm_ope.h"
#include "efa_rdm_pke.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_protocol.h"

/**
 * @brief initialize the payload and rma_iov of a RTW packet
 *
 * Used by EAGER and LONGCTS RTW.
 * @param[in,out]	pkt_entry	RTW packet entry
 * @param[in]		txe		TX entry that has RMA write information
 * @param[in]		rma_iov		the "rma_iov" field in RTW packet header
 *
 * @returns
 * 0 on success
 * negative libfabric error code on error.
 */
static inline
ssize_t efa_rdm_proto_write_rtw_pke_init_common(struct efa_rdm_pke *pkt_entry,
				    struct efa_rdm_ope *txe,
				    struct efa_rma_iov *rma_iov)
{
	size_t hdr_size;
	size_t data_size;
	int i;

	for (i = 0; i < txe->rma_iov_count; ++i) {
		rma_iov[i].addr = txe->rma_iov[i].addr;
		rma_iov[i].len = txe->rma_iov[i].len;
		rma_iov[i].key = txe->rma_iov[i].key;
	}

	hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);
	data_size = MIN(txe->ep->mtu_size - hdr_size, txe->total_len);
	return efa_rdm_pke_init_payload_from_ope(pkt_entry, txe, hdr_size, 0, data_size);
}

#endif /* EFA_RDM_PROTO_WRITE_H */
