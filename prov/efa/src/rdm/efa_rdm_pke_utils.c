/*
 * Copyright (c) Amazon.com, Inc. or its affiliates.
 * All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <rdma/fi_rma.h>
#include "efa_mr.h"
#include "efa_hmem.h"
#include "efa_rdm_pke.h"
#include "rxr_pkt_type.h"
#include "efa_rdm_pkt_type.h"
#include "efa_rdm_protocol.h"
#include "rxr_pkt_type_req.h"

/**
 * @brief determine the payload offset of a received packet.
 *
 * For a packet receive overwire, the payload is in the wiredata,
 * immediately after header and optional user buffer information.
 * This function find the offset of payoff in respect of wiredata.
 *
 * @param[in]	pkt_entry	received paket entry
 * @return	an integer offset
 */
size_t efa_rdm_pke_get_payload_offset(struct efa_rdm_pke *pkt_entry)
{
	struct rxr_base_hdr *base_hdr;
	int pkt_type, read_iov_count;
	size_t payload_offset;

	assert(pkt_entry->alloc_type != EFA_RDM_PKE_FROM_EFA_TX_POOL);

	/* packet entry from read copy pool contains only payload
	 * in its wire data (no header, no optional user buffer
	 * information)
	 */
	if (pkt_entry->alloc_type == EFA_RDM_PKE_FROM_READ_COPY_POOL)
		return 0;

	base_hdr = rxr_get_base_hdr(pkt_entry->wiredata);
	pkt_type = base_hdr->type;
	assert(efa_rdm_pkt_type_contains_data(pkt_type));
	if (efa_rdm_pkt_type_is_req(pkt_type)) {
		payload_offset = rxr_pkt_req_hdr_size_from_pkt_entry(pkt_entry);
		assert(payload_offset > 0);

		if (pkt_type == RXR_RUNTREAD_MSGRTM_PKT ||
		    pkt_type == RXR_RUNTREAD_TAGRTM_PKT) {
			read_iov_count = rxr_get_runtread_rtm_base_hdr(pkt_entry->wiredata)->read_iov_count;
			payload_offset +=  read_iov_count * sizeof(struct fi_rma_iov);
		}

		return payload_offset;
	}

	if (pkt_type == RXR_DATA_PKT) {
		payload_offset = sizeof(struct rxr_data_hdr);
		if (base_hdr->flags & RXR_PKT_CONNID_HDR)
			payload_offset += sizeof(struct rxr_data_opt_connid_hdr);
		return payload_offset;
	}

	if (pkt_type == RXR_READRSP_PKT)
		return sizeof(struct rxr_readrsp_hdr);

	if (pkt_type == RXR_ATOMRSP_PKT)
		return sizeof(struct rxr_atomrsp_hdr);

	/* all packet types that can contain data has been processed.
	 * we should never reach here;
	 */
	assert(0);
	return -1;
}