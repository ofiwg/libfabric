/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef _EFA_RDM_PKE_UTILS_H
#define _EFA_RDM_PKE_UTILS_H

#include "efa_rdm_pke.h"
#include "efa_rdm_protocol.h"
#include "efa_rdm_pkt_type.h"
#include "efa_mr.h"

/**
 * @brief get the base header of an pke
 *
 * @param[in]	pke	packet entry
 * @returns	base header
 */
static inline
struct efa_rdm_base_hdr *efa_rdm_pke_get_base_hdr(struct efa_rdm_pke *pke)
{
	return (struct efa_rdm_base_hdr *)pke->wiredata;
}

/**
 * @brief return the segment offset of user data in packet entry
 *
 * segment_offset is the user data's offset in repect of user's
 * buffer.
 *
 * @param[in]	pkt_entry	packet entry
 * @return	the value of seg_offset in the packet
 */
static inline
size_t efa_rdm_pke_get_segment_offset(struct efa_rdm_pke *pke)
{
	int pkt_type, hdr_offset;
	static const int offset_of_seg_offset_in_header[] = {
		[EFA_RDM_CTSDATA_PKT] = offsetof(struct efa_rdm_ctsdata_hdr, seg_offset),
		[EFA_RDM_MEDIUM_MSGRTM_PKT] = offsetof(struct efa_rdm_medium_rtm_base_hdr, seg_offset),
		[EFA_RDM_MEDIUM_TAGRTM_PKT] = offsetof(struct efa_rdm_medium_rtm_base_hdr, seg_offset),
		[EFA_RDM_DC_MEDIUM_MSGRTM_PKT] = offsetof(struct efa_rdm_dc_medium_rtm_base_hdr, seg_offset),
		[EFA_RDM_DC_MEDIUM_TAGRTM_PKT] = offsetof(struct efa_rdm_dc_medium_rtm_base_hdr, seg_offset),
		[EFA_RDM_RUNTREAD_MSGRTM_PKT] = offsetof(struct efa_rdm_runtread_rtm_base_hdr, seg_offset),
		[EFA_RDM_RUNTREAD_TAGRTM_PKT] = offsetof(struct efa_rdm_runtread_rtm_base_hdr, seg_offset),
	};

	pkt_type = efa_rdm_pke_get_base_hdr(pke)->type;
	assert(efa_rdm_pkt_type_contains_data(pkt_type));

	if (efa_rdm_pkt_type_contains_seg_offset(pkt_type)) {
		/* all such packet types has been listed in the array */
		hdr_offset = offset_of_seg_offset_in_header[pkt_type];

		assert(hdr_offset);
		return *(uint64_t *)(pke->wiredata + hdr_offset);
	}

	return 0;
}

/**
 * @brief copy data from the user's buffer to the packet's wiredata.
 *
 * @param[in]		iov_mr  memory descriptors of the user's data buffer.
 * @param[in,out]	pke	    packet entry. Header must have been set when the function is called.
 * @param[in]		ope		operation entry that has user buffer information.
 * @param[in]		payload_offset	the data offset in reference to pkt_entry->wiredata.
 * @param[in]		segment_offset	the data offset in reference to user's buffer
 * @param[in]		data_size	length of the data to be set up.
 * @return   		length of data copied.
 */
static inline size_t
efa_rdm_pke_copy_from_hmem_iov(struct efa_mr *iov_mr, struct efa_rdm_pke *pke,
			       struct efa_rdm_ope *ope, size_t payload_offset,
			       size_t segment_offset, size_t data_size)
{
	size_t copied;

	if (iov_mr && (iov_mr->peer.flags & OFI_HMEM_DATA_DEV_REG_HANDLE)) {
		assert(iov_mr->peer.hmem_data);
		copied = ofi_dev_reg_copy_from_hmem_iov(pke->wiredata + payload_offset,
							data_size, iov_mr->peer.iface,
							(uint64_t)iov_mr->peer.hmem_data,
							ope->iov, ope->iov_count,
							segment_offset);
	} else {
		copied = ofi_copy_from_hmem_iov(pke->wiredata + payload_offset,
		                                data_size,
		                                iov_mr ? iov_mr->peer.iface : FI_HMEM_SYSTEM,
		                                iov_mr ? iov_mr->peer.device.reserved : 0,
		                                ope->iov, ope->iov_count, segment_offset);
	}

	return copied;
}

size_t efa_rdm_pke_get_payload_offset(struct efa_rdm_pke *pkt_entry);

ssize_t efa_rdm_pke_init_payload_from_ope(struct efa_rdm_pke *pke,
					  struct efa_rdm_ope *ope,
					  size_t payload_offset,
					  size_t segment_offset,
					  size_t data_size);

ssize_t efa_rdm_pke_copy_payload_to_ope(struct efa_rdm_pke *pke,
					struct efa_rdm_ope *ope);

uint32_t *efa_rdm_pke_connid_ptr(struct efa_rdm_pke *pkt_entry);

int efa_rdm_pke_get_available_copy_methods(struct efa_rdm_ep *ep,
					   struct efa_mr *efa_mr,
					   bool *restrict local_read_available,
					   bool *restrict cuda_memcpy_available,
					   bool *restrict gdrcopy_available);

#endif
