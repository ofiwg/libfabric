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

#include <ofi_atomic.h>
#include "efa.h"

#include "efa_rdm_rma.h"
#include "efa_rdm_msg.h"
#include "efa_rdm_pke_cmd.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_srx.h"
#include "rxr_pkt_type_req.h"
#include "efa_rdm_pke_rta.h"
#include "efa_rdm_pke_rtw.h"
#include "efa_rdm_pke_rtr.h"
#include "rxr_tp.h"

/*
 * Utility constants and functions shared by all REQ packet
 * types.
 */
struct rxr_req_inf {
	uint64_t extra_info_id;
	uint64_t base_hdr_size;
	uint64_t ex_feature_flag;
};

static const
struct rxr_req_inf REQ_INF_LIST[] = {
	/* rtm header */
	[RXR_EAGER_MSGRTM_PKT] = {0, sizeof(struct rxr_eager_msgrtm_hdr), 0},
	[RXR_EAGER_TAGRTM_PKT] = {0, sizeof(struct rxr_eager_tagrtm_hdr), 0},
	[RXR_MEDIUM_MSGRTM_PKT] = {0, sizeof(struct rxr_medium_msgrtm_hdr), 0},
	[RXR_MEDIUM_TAGRTM_PKT] = {0, sizeof(struct rxr_medium_tagrtm_hdr), 0},
	[RXR_LONGCTS_MSGRTM_PKT] = {0, sizeof(struct rxr_longcts_msgrtm_hdr), 0},
	[RXR_LONGCTS_TAGRTM_PKT] = {0, sizeof(struct rxr_longcts_tagrtm_hdr), 0},
	[RXR_LONGREAD_MSGRTM_PKT] = {0, sizeof(struct rxr_longread_msgrtm_hdr), RXR_EXTRA_FEATURE_RDMA_READ},
	[RXR_LONGREAD_TAGRTM_PKT] = {0, sizeof(struct rxr_longread_tagrtm_hdr), RXR_EXTRA_FEATURE_RDMA_READ},
	[RXR_DC_EAGER_MSGRTM_PKT] = {0, sizeof(struct rxr_dc_eager_msgrtm_hdr), RXR_EXTRA_FEATURE_DELIVERY_COMPLETE},
	[RXR_DC_EAGER_TAGRTM_PKT] = {0, sizeof(struct rxr_dc_eager_tagrtm_hdr), RXR_EXTRA_FEATURE_DELIVERY_COMPLETE},
	[RXR_DC_MEDIUM_MSGRTM_PKT] = {0, sizeof(struct rxr_dc_medium_msgrtm_hdr), RXR_EXTRA_FEATURE_DELIVERY_COMPLETE},
	[RXR_DC_MEDIUM_TAGRTM_PKT] = {0, sizeof(struct rxr_dc_medium_tagrtm_hdr), RXR_EXTRA_FEATURE_DELIVERY_COMPLETE},
	[RXR_DC_LONGCTS_MSGRTM_PKT] = {0, sizeof(struct rxr_longcts_msgrtm_hdr), RXR_EXTRA_FEATURE_DELIVERY_COMPLETE},
	[RXR_DC_LONGCTS_TAGRTM_PKT] = {0, sizeof(struct rxr_longcts_tagrtm_hdr), RXR_EXTRA_FEATURE_DELIVERY_COMPLETE},
	[RXR_RUNTCTS_MSGRTM_PKT] = {0, sizeof(struct rxr_runtcts_msgrtm_hdr), RXR_EXTRA_FEATURE_RUNT},
	[RXR_RUNTCTS_TAGRTM_PKT] = {0, sizeof(struct rxr_runtcts_tagrtm_hdr), RXR_EXTRA_FEATURE_RUNT},
	[RXR_RUNTREAD_MSGRTM_PKT] = {0, sizeof(struct rxr_runtread_msgrtm_hdr), RXR_EXTRA_FEATURE_RUNT | RXR_EXTRA_FEATURE_RDMA_READ},
	[RXR_RUNTREAD_TAGRTM_PKT] = {0, sizeof(struct rxr_runtread_tagrtm_hdr), RXR_EXTRA_FEATURE_RUNT | RXR_EXTRA_FEATURE_RDMA_READ},
	/* rtw header */
	[RXR_EAGER_RTW_PKT] = {0, sizeof(struct rxr_eager_rtw_hdr), 0},
	[RXR_DC_EAGER_RTW_PKT] = {0, sizeof(struct rxr_dc_eager_rtw_hdr), RXR_EXTRA_FEATURE_DELIVERY_COMPLETE},
	[RXR_LONGCTS_RTW_PKT] = {0, sizeof(struct rxr_longcts_rtw_hdr), 0},
	[RXR_DC_LONGCTS_RTW_PKT] = {0, sizeof(struct rxr_longcts_rtw_hdr), RXR_EXTRA_FEATURE_DELIVERY_COMPLETE},
	[RXR_LONGREAD_RTW_PKT] = {0, sizeof(struct rxr_longread_rtw_hdr), RXR_EXTRA_FEATURE_RDMA_READ},
	[RXR_RUNTCTS_RTW_PKT] = {0, sizeof(struct rxr_runtcts_rtw_hdr), RXR_EXTRA_FEATURE_RUNT},
	[RXR_RUNTREAD_RTW_PKT] = {0, sizeof(struct rxr_runtread_rtw_hdr), RXR_EXTRA_FEATURE_RUNT},
	/* rtr header */
	[RXR_SHORT_RTR_PKT] = {0, sizeof(struct rxr_rtr_hdr), 0},
	[RXR_LONGCTS_RTR_PKT] = {0, sizeof(struct rxr_rtr_hdr), 0},
	[RXR_READ_RTR_PKT] = {0, sizeof(struct rxr_base_hdr), RXR_EXTRA_FEATURE_RDMA_READ},
	/* rta header */
	[RXR_WRITE_RTA_PKT] = {0, sizeof(struct rxr_rta_hdr), 0},
	[RXR_DC_WRITE_RTA_PKT] = {0, sizeof(struct rxr_rta_hdr), RXR_EXTRA_FEATURE_DELIVERY_COMPLETE},
	[RXR_FETCH_RTA_PKT] = {0, sizeof(struct rxr_rta_hdr), 0},
	[RXR_COMPARE_RTA_PKT] = {0, sizeof(struct rxr_rta_hdr), 0},
};

bool rxr_pkt_req_supported_by_peer(int req_type, struct efa_rdm_peer *peer)
{
	assert(peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED);

	int extra_info_id = REQ_INF_LIST[req_type].extra_info_id;

	return peer->extra_info[extra_info_id] & REQ_INF_LIST[req_type].ex_feature_flag;
}

/**
 * @brief Determine which Read based protocol to use
 *
 * @param[in] peer		rdm peer
 * @param[in] op		operation type
 * @param[in] flags		the flags that the application used to call fi_* functions
 * @param[in] hmem_info	configured protocol limits
 * @return The read-based protocol to use based on inputs.
 */
int rxr_pkt_type_readbase_rtm(struct efa_rdm_peer *peer, int op, uint64_t fi_flags, struct efa_hmem_info *hmem_info)
{
	assert(op == ofi_op_tagged || op == ofi_op_msg);
	if (peer->num_read_msg_in_flight == 0 &&
	    hmem_info->runt_size > peer->num_runt_bytes_in_flight &&
	    !(fi_flags & FI_DELIVERY_COMPLETE)) {
		return (op == ofi_op_tagged) ? RXR_RUNTREAD_TAGRTM_PKT
					     : RXR_RUNTREAD_MSGRTM_PKT;
	} else {
		return (op == ofi_op_tagged) ? RXR_LONGREAD_TAGRTM_PKT
					     : RXR_LONGREAD_MSGRTM_PKT;
	}
}

void rxr_pkt_init_req_hdr(struct efa_rdm_pke *pkt_entry,
			  int pkt_type,
			  struct efa_rdm_ope *txe)
{
	char *opt_hdr;
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct rxr_base_hdr *base_hdr;

	/* init the base header */
	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	base_hdr->type = pkt_type;
	base_hdr->version = RXR_PROTOCOL_VERSION;
	base_hdr->flags = 0;

	ep = txe->ep;
	peer = efa_rdm_ep_get_peer(ep, txe->addr);
	assert(peer);

	if (efa_rdm_peer_need_raw_addr_hdr(peer)) {
		/*
		 * This is the first communication with this peer on this
		 * endpoint, so send the core's address for this EP in the REQ
		 * so the remote side can insert it into its address vector.
		 */
		base_hdr->flags |= RXR_REQ_OPT_RAW_ADDR_HDR;
	} else if (efa_rdm_peer_need_connid(peer)) {
		/*
		 * After receiving handshake packet, we will know the peer's capability.
		 *
		 * If the peer need connid, we will include the optional connid
		 * header in the req packet header.The peer will use it
		 * to verify my identity.
		 *
		 * This logic means that a req packet cannot have both
		 * the optional raw address header and the optional connid header.
		 */
		base_hdr->flags |= RXR_PKT_CONNID_HDR;
	}

	if (txe->fi_flags & FI_REMOTE_CQ_DATA) {
		base_hdr->flags |= RXR_REQ_OPT_CQ_DATA_HDR;
	}

	/* init the opt header */
	opt_hdr = (char *)base_hdr + rxr_pkt_req_base_hdr_size(pkt_entry);
	if (base_hdr->flags & RXR_REQ_OPT_RAW_ADDR_HDR) {
		struct rxr_req_opt_raw_addr_hdr *raw_addr_hdr;

		raw_addr_hdr = (struct rxr_req_opt_raw_addr_hdr *)opt_hdr;
		raw_addr_hdr->addr_len = RXR_REQ_OPT_RAW_ADDR_HDR_SIZE - sizeof(struct rxr_req_opt_raw_addr_hdr);
		assert(raw_addr_hdr->addr_len >= sizeof(ep->base_ep.src_addr));
		memcpy(raw_addr_hdr->raw_addr, &ep->base_ep.src_addr, sizeof(ep->base_ep.src_addr));
		opt_hdr += RXR_REQ_OPT_RAW_ADDR_HDR_SIZE;
	}

	if (base_hdr->flags & RXR_REQ_OPT_CQ_DATA_HDR) {
		struct rxr_req_opt_cq_data_hdr *cq_data_hdr;

		cq_data_hdr = (struct rxr_req_opt_cq_data_hdr *)opt_hdr;
		cq_data_hdr->cq_data = txe->cq_entry.data;
		opt_hdr += sizeof(*cq_data_hdr);
	}

	if (base_hdr->flags & RXR_PKT_CONNID_HDR) {
		struct rxr_req_opt_connid_hdr *connid_hdr;

		connid_hdr = (struct rxr_req_opt_connid_hdr *)opt_hdr;
		connid_hdr->connid = efa_rdm_ep_raw_addr(ep)->qkey;
		opt_hdr += sizeof(*connid_hdr);
	}

	pkt_entry->addr = txe->addr;
	assert(opt_hdr - pkt_entry->wiredata == rxr_pkt_req_hdr_size_from_pkt_entry(pkt_entry));
}

/**
 * @brief Get the value of rma_iov_count in pkt header.
 *
 * @param[in] pkt_entry the pkt entry
 * @return The rma_iov_count in the pkt header, if it is part of the header.
 * Otherwise return 0.
 */
uint32_t rxr_pkt_hdr_rma_iov_count(struct efa_rdm_pke *pkt_entry)
{
	int pkt_type = efa_rdm_pke_get_base_hdr(pkt_entry)->type;

	if (pkt_type == RXR_EAGER_RTW_PKT ||
	    pkt_type == RXR_DC_EAGER_RTW_PKT ||
	    pkt_type == RXR_LONGCTS_RTW_PKT ||
	    pkt_type == RXR_DC_LONGCTS_RTW_PKT ||
	    pkt_type == RXR_LONGREAD_RTW_PKT)
		return efa_rdm_pke_get_rtw_base_hdr(pkt_entry)->rma_iov_count;

	if (pkt_type == RXR_SHORT_RTR_PKT ||
		 pkt_type == RXR_LONGCTS_RTR_PKT)
		return efa_rdm_pke_get_rtr_hdr(pkt_entry)->rma_iov_count;

	if (pkt_type == RXR_WRITE_RTA_PKT ||
		 pkt_type == RXR_DC_WRITE_RTA_PKT ||
		 pkt_type == RXR_FETCH_RTA_PKT ||
		 pkt_type == RXR_COMPARE_RTA_PKT)
		return efa_rdm_pke_get_rta_hdr(pkt_entry)->rma_iov_count;

	return 0;
}

size_t rxr_pkt_req_base_hdr_size(struct efa_rdm_pke *pkt_entry)
{
	struct rxr_base_hdr *base_hdr;
	uint32_t rma_iov_count;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	assert(base_hdr->type >= RXR_REQ_PKT_BEGIN);

	rma_iov_count = rxr_pkt_hdr_rma_iov_count(pkt_entry);
	return REQ_INF_LIST[base_hdr->type].base_hdr_size +
	       rma_iov_count * sizeof(struct fi_rma_iov);
}

/**
 * @brief return the optional raw addr header pointer in a req packet
 *
 * @param[in]	pkt_entry	an REQ packet entry
 * @return	If the input has the optional raw addres header, return the pointer to it.
 *		Otherwise, return NULL
 */
void *rxr_pkt_req_raw_addr(struct efa_rdm_pke *pkt_entry)
{
	char *opt_hdr;
	struct rxr_base_hdr *base_hdr;
	struct rxr_req_opt_raw_addr_hdr *raw_addr_hdr;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	opt_hdr = pkt_entry->wiredata + rxr_pkt_req_base_hdr_size(pkt_entry);
	if (base_hdr->flags & RXR_REQ_OPT_RAW_ADDR_HDR) {
		/* For req packet, the optional connid header and the optional
		 * raw address header are mutually exclusive.
		 */
		assert(!(base_hdr->flags & RXR_PKT_CONNID_HDR));
		raw_addr_hdr = (struct rxr_req_opt_raw_addr_hdr *)opt_hdr;
		return raw_addr_hdr->raw_addr;
	}

	return NULL;
}

/**
 * @brief return the pointer to connid in a req packet
 *
 * @param[in]	pkt_entry	an REQ packet entry
 * @return	If the input has the optional connid header, return the pointer to connid
 * 		Otherwise, return NULL
 */
uint32_t *rxr_pkt_req_connid_ptr(struct efa_rdm_pke *pkt_entry)
{
	char *opt_hdr;
	struct rxr_base_hdr *base_hdr;
	struct rxr_req_opt_connid_hdr *connid_hdr;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	opt_hdr = pkt_entry->wiredata + rxr_pkt_req_base_hdr_size(pkt_entry);

	if (base_hdr->flags & RXR_REQ_OPT_RAW_ADDR_HDR) {
		struct rxr_req_opt_raw_addr_hdr *raw_addr_hdr;
		struct efa_ep_addr *raw_addr;

		raw_addr_hdr = (struct rxr_req_opt_raw_addr_hdr *)opt_hdr;
		raw_addr = (struct efa_ep_addr *)raw_addr_hdr->raw_addr;
		return &raw_addr->qkey;
	}

	if (base_hdr->flags & RXR_REQ_OPT_CQ_DATA_HDR)
		opt_hdr += sizeof(struct rxr_req_opt_cq_data_hdr);

	if (base_hdr->flags & RXR_PKT_CONNID_HDR) {
		connid_hdr = (struct rxr_req_opt_connid_hdr *)opt_hdr;
		return &connid_hdr->connid;
	}

	return NULL;
}

/**
 * @brief calculate the exact header size for a given REQ pkt_entry
 *
 * The difference between this function and rxr_pkt_req_hdr_size() is
 * the handling of the size of req opt raw address header.
 *
 * rxr_pkt_req_hdr_size() always use RXR_REQ_OPT_RAW_ADDR_HDR_SIZE, while
 * this function pull raw address from pkt_entry's size.
 *
 * The difference is because older version of libfabric EFA provider uses
 * a different opt header size.
 *
 * @param[in]	pkt_entry		packet entry
 * @return 	header size of the REQ packet entry
 */
size_t rxr_pkt_req_hdr_size_from_pkt_entry(struct efa_rdm_pke *pkt_entry)
{
	char *opt_hdr;
	struct rxr_base_hdr *base_hdr;
	struct rxr_req_opt_raw_addr_hdr *raw_addr_hdr;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	opt_hdr = pkt_entry->wiredata + rxr_pkt_req_base_hdr_size(pkt_entry);

	/*
	 * It is not possible to have both optional raw addr header and optional
	 * connid header in a packet header.
	 */
	if (base_hdr->flags & RXR_REQ_OPT_RAW_ADDR_HDR) {
		assert(!(base_hdr->flags & RXR_PKT_CONNID_HDR));
		raw_addr_hdr = (struct rxr_req_opt_raw_addr_hdr *)opt_hdr;
		opt_hdr += sizeof(struct rxr_req_opt_raw_addr_hdr) + raw_addr_hdr->addr_len;
	}

	if (base_hdr->flags & RXR_REQ_OPT_CQ_DATA_HDR)
		opt_hdr += sizeof(struct rxr_req_opt_cq_data_hdr);

	if (base_hdr->flags & RXR_PKT_CONNID_HDR) {
		assert(!(base_hdr->flags & RXR_REQ_OPT_RAW_ADDR_HDR));
		opt_hdr += sizeof(struct rxr_req_opt_connid_hdr);
	}

	return opt_hdr - pkt_entry->wiredata;
}

int64_t rxr_pkt_req_cq_data(struct efa_rdm_pke *pkt_entry)
{
	char *opt_hdr;
	struct rxr_base_hdr *base_hdr;
	struct rxr_req_opt_cq_data_hdr *cq_data_hdr;
	struct rxr_req_opt_raw_addr_hdr *raw_addr_hdr;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	opt_hdr = pkt_entry->wiredata + rxr_pkt_req_base_hdr_size(pkt_entry);
	if (base_hdr->flags & RXR_REQ_OPT_RAW_ADDR_HDR) {
		raw_addr_hdr = (struct rxr_req_opt_raw_addr_hdr *)opt_hdr;
		opt_hdr += sizeof(struct rxr_req_opt_raw_addr_hdr) + raw_addr_hdr->addr_len;
	}

	assert(base_hdr->flags & RXR_REQ_OPT_CQ_DATA_HDR);
	cq_data_hdr = (struct rxr_req_opt_cq_data_hdr *)opt_hdr;
	return cq_data_hdr->cq_data;
}

/**
 * @brief calculates the exact header size given a REQ packet type, flags, and IOV count.
 *
 * @param[in]	pkt_type	packet type
 * @param[in]	flags	flags from packet
 * @param[in]	rma_iov_count	number of RMA IOV structures present
 * @return	The exact size of the packet header
 */
inline
size_t rxr_pkt_req_hdr_size(int pkt_type, uint16_t flags, size_t rma_iov_count)
{
	int hdr_size = REQ_INF_LIST[pkt_type].base_hdr_size;

	if (flags & RXR_REQ_OPT_RAW_ADDR_HDR) {
		/* It is impossible to have both optional connid hdr and opt_raw_addr_hdr
		 * in the header, and length of opt raw addr hdr is larger than
		 * connid hdr (which is confirmed by the following assertion).
		 */
		assert(RXR_REQ_OPT_RAW_ADDR_HDR_SIZE >= sizeof(struct rxr_req_opt_connid_hdr));
		hdr_size += RXR_REQ_OPT_RAW_ADDR_HDR_SIZE;
	} else if (flags & RXR_PKT_CONNID_HDR) {
		hdr_size += sizeof(struct rxr_req_opt_connid_hdr);
	}

	if (flags & RXR_REQ_OPT_CQ_DATA_HDR) {
		hdr_size += sizeof(struct rxr_req_opt_cq_data_hdr);
	}

	if (efa_rdm_pkt_type_contains_rma_iov(pkt_type)) {
		hdr_size += rma_iov_count * sizeof(struct fi_rma_iov);
	}

	return hdr_size;
}

/**
 * @brief calculates the max header size given a REQ packet type
 *
 * @param[in]	pkt_type	packet type
 * @return	The max possible size of the packet header
 */
inline size_t rxr_pkt_req_max_hdr_size(int pkt_type)
{
	/* To calculate max REQ reader size, we should include all possible REQ opt header flags.
	 * However, because the optional connid header and optional raw address header cannot
	 * exist at the same time, and the raw address header is longer than connid header,
	 * we did not include the flag for CONNID header
	 */
	uint16_t header_flags = RXR_REQ_OPT_RAW_ADDR_HDR | RXR_REQ_OPT_CQ_DATA_HDR;

	return rxr_pkt_req_hdr_size(pkt_type, header_flags, RXR_IOV_LIMIT);
}

size_t rxr_pkt_max_hdr_size(void)
{
	size_t max_hdr_size = 0;
	size_t pkt_type = RXR_REQ_PKT_BEGIN;

	while (pkt_type < RXR_EXTRA_REQ_PKT_END) {
		max_hdr_size = MAX(max_hdr_size,
				rxr_pkt_req_max_hdr_size(pkt_type));
		if (pkt_type == RXR_BASELINE_REQ_PKT_END)
			pkt_type = RXR_EXTRA_REQ_PKT_BEGIN;
		else
			pkt_type += 1;
	}

	return max_hdr_size;
}
