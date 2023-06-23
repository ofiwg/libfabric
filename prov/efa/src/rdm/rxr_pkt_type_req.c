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


/*
 * REQ packet type functions
 *
 *     init() functions
 */
static inline
ssize_t rxr_pkt_init_rtm(struct efa_rdm_pke *pkt_entry,
			 int pkt_type,
			 struct efa_rdm_ope *txe,
			 size_t data_offset,
			 int data_size)
{
	struct rxr_rtm_base_hdr *rtm_hdr;
	int ret;

	rxr_pkt_init_req_hdr(pkt_entry, pkt_type, txe);

	rtm_hdr = (struct rxr_rtm_base_hdr *)pkt_entry->wiredata;
	rtm_hdr->flags |= RXR_REQ_MSG;
	rtm_hdr->msg_id = txe->msg_id;

	if (data_size == -1) {
		data_size = MIN(txe->total_len - data_offset,
				txe->ep->mtu_size - rxr_pkt_req_hdr_size_from_pkt_entry(pkt_entry));

		if (data_size + data_offset < txe->total_len) {
			if (efa_mr_is_cuda(txe->desc[0])) {
				if (txe->ep->sendrecv_in_order_aligned_128_bytes)
					data_size &= ~(EFA_RDM_IN_ORDER_ALIGNMENT - 1);
				else
					data_size &= ~(EFA_RDM_CUDA_MEMORY_ALIGNMENT -1);
			}
		}
	}

	ret = efa_rdm_pke_init_payload_from_ope(pkt_entry, txe,
						rxr_pkt_req_hdr_size_from_pkt_entry(pkt_entry),
						data_offset, data_size);
	return ret;
}

ssize_t rxr_pkt_init_eager_msgrtm(struct efa_rdm_pke *pkt_entry,
				  struct efa_rdm_ope *txe)
{
	int ret;

	ret = rxr_pkt_init_rtm(pkt_entry, RXR_EAGER_MSGRTM_PKT, txe, 0, -1);
	if (ret)
		return ret;

	assert(txe->total_len == pkt_entry->payload_size);
	return 0;
}

ssize_t rxr_pkt_init_dc_eager_msgrtm(struct efa_rdm_pke *pkt_entry,
				     struct efa_rdm_ope *txe)

{
	struct rxr_dc_eager_msgrtm_hdr *dc_eager_msgrtm_hdr;
	int ret;

	txe->rxr_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;
	ret = rxr_pkt_init_rtm(pkt_entry, RXR_DC_EAGER_MSGRTM_PKT, txe, 0, -1);
	if (ret)
		return ret;
	dc_eager_msgrtm_hdr = rxr_get_dc_eager_msgrtm_hdr(pkt_entry->wiredata);
	dc_eager_msgrtm_hdr->hdr.send_id = txe->tx_id;
	return 0;
}

ssize_t rxr_pkt_init_eager_tagrtm(struct efa_rdm_pke *pkt_entry,
				  struct efa_rdm_ope *txe)
{
	struct rxr_base_hdr *base_hdr;
	int ret;

	ret = rxr_pkt_init_rtm(pkt_entry, RXR_EAGER_TAGRTM_PKT, txe, 0, -1);
	if (ret)
		return ret;
	assert(txe->total_len == pkt_entry->payload_size);
	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	base_hdr->flags |= RXR_REQ_TAGGED;
	rxr_pkt_rtm_settag(pkt_entry, txe->tag);
	return 0;
}

ssize_t rxr_pkt_init_dc_eager_tagrtm(struct efa_rdm_pke *pkt_entry,
				     struct efa_rdm_ope *txe)
{
	struct rxr_base_hdr *base_hdr;
	struct rxr_dc_eager_tagrtm_hdr *dc_eager_tagrtm_hdr;
	int ret;

	txe->rxr_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;
	ret = rxr_pkt_init_rtm(pkt_entry, RXR_DC_EAGER_TAGRTM_PKT, txe, 0, -1);
	if (ret)
		return ret;
	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	base_hdr->flags |= RXR_REQ_TAGGED;
	rxr_pkt_rtm_settag(pkt_entry, txe->tag);

	dc_eager_tagrtm_hdr = rxr_get_dc_eager_tagrtm_hdr(pkt_entry->wiredata);
	dc_eager_tagrtm_hdr->hdr.send_id = txe->tx_id;
	return 0;
}

ssize_t rxr_pkt_init_medium_msgrtm(struct efa_rdm_pke *pkt_entry,
				   struct efa_rdm_ope *txe,
				   size_t data_offset,
				   int data_size)

{
	struct rxr_medium_rtm_base_hdr *rtm_hdr;
	int ret;

	efa_rdm_ope_try_fill_desc(txe, 0, FI_SEND);

	ret = rxr_pkt_init_rtm(pkt_entry, RXR_MEDIUM_MSGRTM_PKT,
			       txe, data_offset, data_size);
	if (ret)
		return ret;

	rtm_hdr = rxr_get_medium_rtm_base_hdr(pkt_entry->wiredata);
	rtm_hdr->msg_length = txe->total_len;
	rtm_hdr->seg_offset = data_offset;
	return 0;
}

ssize_t rxr_pkt_init_dc_medium_msgrtm(struct efa_rdm_pke *pkt_entry,
				      struct efa_rdm_ope *txe,
				      size_t data_offset,
				      int data_size)
{
	struct rxr_dc_medium_msgrtm_hdr *dc_medium_msgrtm_hdr;
	int ret;

	txe->rxr_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	efa_rdm_ope_try_fill_desc(txe, 0, FI_SEND);

	ret = rxr_pkt_init_rtm(pkt_entry, RXR_DC_MEDIUM_MSGRTM_PKT,
			       txe, data_offset, data_size);
	if (ret)
		return ret;

	dc_medium_msgrtm_hdr = rxr_get_dc_medium_msgrtm_hdr(pkt_entry->wiredata);
	dc_medium_msgrtm_hdr->hdr.msg_length = txe->total_len;
	dc_medium_msgrtm_hdr->hdr.seg_offset = data_offset;
	dc_medium_msgrtm_hdr->hdr.send_id = txe->tx_id;
	return 0;
}

ssize_t rxr_pkt_init_medium_tagrtm(struct efa_rdm_pke *pkt_entry,
				   struct efa_rdm_ope *txe,
				   size_t data_offset,
				   int data_size)
{
	struct rxr_medium_rtm_base_hdr *rtm_hdr;
	int ret;

	efa_rdm_ope_try_fill_desc(txe, 0, FI_SEND);

	ret = rxr_pkt_init_rtm(pkt_entry, RXR_MEDIUM_TAGRTM_PKT,
			       txe, data_offset, data_size);
	if (ret)
		return ret;

	rtm_hdr = rxr_get_medium_rtm_base_hdr(pkt_entry->wiredata);
	rtm_hdr->msg_length = txe->total_len;
	rtm_hdr->seg_offset = data_offset;
	rtm_hdr->hdr.flags |= RXR_REQ_TAGGED;
	rxr_pkt_rtm_settag(pkt_entry, txe->tag);
	return 0;
}

ssize_t rxr_pkt_init_dc_medium_tagrtm(struct efa_rdm_pke *pkt_entry,
				      struct efa_rdm_ope *txe,
				      size_t data_offset,
				      int data_size)
{
	struct rxr_dc_medium_tagrtm_hdr *dc_medium_tagrtm_hdr;
	int ret;

	txe->rxr_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	efa_rdm_ope_try_fill_desc(txe, 0, FI_SEND);

	ret = rxr_pkt_init_rtm(pkt_entry, RXR_DC_MEDIUM_TAGRTM_PKT,
			       txe, data_offset, data_size);
	if (ret)
		return ret;

	dc_medium_tagrtm_hdr = rxr_get_dc_medium_tagrtm_hdr(pkt_entry->wiredata);
	dc_medium_tagrtm_hdr->hdr.msg_length = txe->total_len;
	dc_medium_tagrtm_hdr->hdr.seg_offset = data_offset;
	dc_medium_tagrtm_hdr->hdr.hdr.flags |= RXR_REQ_TAGGED;
	dc_medium_tagrtm_hdr->hdr.send_id = txe->tx_id;
	rxr_pkt_rtm_settag(pkt_entry, txe->tag);
	return 0;
}

int rxr_pkt_init_longcts_rtm(struct efa_rdm_pke *pkt_entry,
			     int pkt_type,
			     struct efa_rdm_ope *txe)
{
	struct rxr_longcts_rtm_base_hdr *rtm_hdr;
	int ret;

	ret = rxr_pkt_init_rtm(pkt_entry, pkt_type, txe, 0, -1);
	if (ret)
		return ret;

	rtm_hdr = rxr_get_longcts_rtm_base_hdr(pkt_entry->wiredata);
	rtm_hdr->msg_length = txe->total_len;
	rtm_hdr->send_id = txe->tx_id;
	rtm_hdr->credit_request = efa_env.tx_min_credits;
	return 0;
}

ssize_t rxr_pkt_init_longcts_msgrtm(struct efa_rdm_pke *pkt_entry,
				    struct efa_rdm_ope *txe)
{
	return rxr_pkt_init_longcts_rtm(pkt_entry, RXR_LONGCTS_MSGRTM_PKT, txe);
}

ssize_t rxr_pkt_init_dc_longcts_msgrtm(struct efa_rdm_pke *pkt_entry,
				       struct efa_rdm_ope *txe)
{
	txe->rxr_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;
	return rxr_pkt_init_longcts_rtm(pkt_entry, RXR_DC_LONGCTS_MSGRTM_PKT, txe);
}

ssize_t rxr_pkt_init_longcts_tagrtm(struct efa_rdm_pke *pkt_entry,
				    struct efa_rdm_ope *txe)
{
	struct rxr_base_hdr *base_hdr;
	int ret;

	ret = rxr_pkt_init_longcts_rtm(pkt_entry, RXR_LONGCTS_TAGRTM_PKT, txe);
	if (ret)
		return ret;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	base_hdr->flags |= RXR_REQ_TAGGED;
	rxr_pkt_rtm_settag(pkt_entry, txe->tag);
	return 0;
}

ssize_t rxr_pkt_init_dc_longcts_tagrtm(struct efa_rdm_pke *pkt_entry,
				       struct efa_rdm_ope *txe)
{
	struct rxr_base_hdr *base_hdr;
	int ret;

	txe->rxr_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;
	ret = rxr_pkt_init_longcts_rtm(pkt_entry, RXR_DC_LONGCTS_TAGRTM_PKT, txe);
	if (ret)
		return ret;
	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	base_hdr->flags |= RXR_REQ_TAGGED;
	rxr_pkt_rtm_settag(pkt_entry, txe->tag);
	return 0;
}

ssize_t rxr_pkt_init_longread_rtm(struct efa_rdm_pke *pkt_entry,
				  int pkt_type,
				  struct efa_rdm_ope *txe)
{
	struct rxr_longread_rtm_base_hdr *rtm_hdr;
	struct fi_rma_iov *read_iov;
	size_t hdr_size;
	int err;

	rxr_pkt_init_req_hdr(pkt_entry, pkt_type, txe);

	rtm_hdr = rxr_get_longread_rtm_base_hdr(pkt_entry->wiredata);
	rtm_hdr->hdr.flags |= RXR_REQ_MSG;
	rtm_hdr->hdr.msg_id = txe->msg_id;
	rtm_hdr->msg_length = txe->total_len;
	rtm_hdr->send_id = txe->tx_id;
	rtm_hdr->read_iov_count = txe->iov_count;

	hdr_size = rxr_pkt_req_hdr_size_from_pkt_entry(pkt_entry);
	read_iov = (struct fi_rma_iov *)(pkt_entry->wiredata + hdr_size);
	err = efa_rdm_txe_prepare_to_be_read(txe, read_iov);
	if (OFI_UNLIKELY(err))
		return err;

	pkt_entry->pkt_size = hdr_size + txe->iov_count * sizeof(struct fi_rma_iov);
	pkt_entry->ope = txe;
	return 0;
}

ssize_t rxr_pkt_init_longread_msgrtm(struct efa_rdm_pke *pkt_entry,
				     struct efa_rdm_ope *txe)
{
	return rxr_pkt_init_longread_rtm(pkt_entry, RXR_LONGREAD_MSGRTM_PKT, txe);
}

ssize_t rxr_pkt_init_longread_tagrtm(struct efa_rdm_pke *pkt_entry,
				     struct efa_rdm_ope *txe)
{
	ssize_t err;
	struct rxr_base_hdr *base_hdr;

	err = rxr_pkt_init_longread_rtm(pkt_entry, RXR_LONGREAD_TAGRTM_PKT, txe);
	if (err)
		return err;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	base_hdr->flags |= RXR_REQ_TAGGED;
	rxr_pkt_rtm_settag(pkt_entry, txe->tag);
	return 0;
}

/**
 * @brief fill in the rxr_runtread_rtm_base_hdr and data of a RUNTREAD packet
 *
 * only thing left unset is tag
 *
 * 
 * @param[out]		pkt_entry	pkt_entry to be initialzied
 * @param[in]		pkt_type	RXR_RUNREAD_MSGRTM or RXR_RUNTREAD_TAGRTM
 * @param[in]		txe	contains information of the send operation
 */
static
ssize_t rxr_pkt_init_runtread_rtm(struct efa_rdm_pke *pkt_entry,
				  int pkt_type,
				  struct efa_rdm_ope *txe,
				  int64_t segment_offset,
				  int64_t data_size)
{
	struct rxr_runtread_rtm_base_hdr *rtm_hdr;
	struct fi_rma_iov *read_iov;
	size_t hdr_size, payload_offset;
	int err;

	assert(txe->bytes_runt);

	rxr_pkt_init_req_hdr(pkt_entry, pkt_type, txe);

	rtm_hdr = rxr_get_runtread_rtm_base_hdr(pkt_entry->wiredata);
	rtm_hdr->hdr.flags |= RXR_REQ_MSG;
	rtm_hdr->hdr.msg_id = txe->msg_id;
	rtm_hdr->msg_length = txe->total_len;
	rtm_hdr->send_id = txe->tx_id;
	rtm_hdr->seg_offset = segment_offset;
	rtm_hdr->runt_length = txe->bytes_runt;
	rtm_hdr->read_iov_count = txe->iov_count;

	hdr_size = rxr_pkt_req_hdr_size_from_pkt_entry(pkt_entry);
	read_iov = (struct fi_rma_iov *)(pkt_entry->wiredata + hdr_size);
	err = efa_rdm_txe_prepare_to_be_read(txe, read_iov);
	if (OFI_UNLIKELY(err))
		return err;

	payload_offset  = hdr_size + txe->iov_count * sizeof(struct fi_rma_iov);
	return efa_rdm_pke_init_payload_from_ope(pkt_entry, txe,
						payload_offset,
						segment_offset,
						data_size);
}

ssize_t rxr_pkt_init_runtread_msgrtm(struct efa_rdm_pke *pkt_entry,
				     struct efa_rdm_ope *txe,
				     size_t data_offset,
				     int data_size)
{
	return rxr_pkt_init_runtread_rtm(pkt_entry, RXR_RUNTREAD_MSGRTM_PKT, txe, data_offset, data_size);
}

ssize_t rxr_pkt_init_runtread_tagrtm(struct efa_rdm_pke *pkt_entry,
				     struct efa_rdm_ope *txe,
				     size_t data_offset,
				     int data_size)
{
	ssize_t err;
	struct rxr_base_hdr *base_hdr;

	err = rxr_pkt_init_runtread_rtm(pkt_entry, RXR_RUNTREAD_TAGRTM_PKT, txe, data_offset, data_size);
	if (err)
		return err;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	base_hdr->flags |= RXR_REQ_TAGGED;
	rxr_pkt_rtm_settag(pkt_entry, txe->tag);
	return 0;
}

/*
 *     handle_sent() functions
 */

/*
 *         rxr_pkt_handle_eager_rtm_sent() is empty and is defined in rxr_pkt_type_req.h
 */
void rxr_pkt_handle_medium_rtm_sent(struct efa_rdm_ep *ep,
				    struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	txe->bytes_sent += pkt_entry->payload_size;
}

void rxr_pkt_handle_longcts_rtm_sent(struct efa_rdm_ep *ep,
				  struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	txe->bytes_sent += pkt_entry->payload_size;
	assert(txe->bytes_sent < txe->total_len);

	if (efa_is_cache_available(efa_rdm_ep_domain(ep)))
		efa_rdm_ope_try_fill_desc(txe, 0, FI_SEND);
}


void rxr_pkt_handle_longread_rtm_sent(struct efa_rdm_ep *ep,
				      struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_peer *peer;

	peer = efa_rdm_ep_get_peer(ep, pkt_entry->addr);
	assert(peer);
	peer->num_read_msg_in_flight += 1;
}

void rxr_pkt_handle_runtread_rtm_sent(struct efa_rdm_ep *ep,
				      struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	size_t pkt_data_size = pkt_entry->payload_size;

	peer = efa_rdm_ep_get_peer(ep, pkt_entry->addr);
	assert(peer);

	txe = pkt_entry->ope;
	txe->bytes_sent += pkt_data_size;
	peer->num_runt_bytes_in_flight += pkt_data_size;

	if (rxr_get_runtread_rtm_base_hdr(pkt_entry->wiredata)->seg_offset == 0 &&
	    txe->total_len > txe->bytes_runt)
		peer->num_read_msg_in_flight += 1;
}

/*
 *     handle_send_completion() functions
 */
void rxr_pkt_handle_eager_rtm_send_completion(struct efa_rdm_ep *ep,
					      struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	assert(txe->total_len == pkt_entry->payload_size);
	efa_rdm_ope_handle_send_completed(txe);
}

void rxr_pkt_handle_medium_rtm_send_completion(struct efa_rdm_ep *ep,
					       struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	txe->bytes_acked += pkt_entry->payload_size;
	if (txe->total_len == txe->bytes_acked)
		efa_rdm_ope_handle_send_completed(txe);
}

void rxr_pkt_handle_longcts_rtm_send_completion(struct efa_rdm_ep *ep,
					     struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	txe->bytes_acked += pkt_entry->payload_size;
	if (txe->total_len == txe->bytes_acked)
		efa_rdm_ope_handle_send_completed(txe);
}

void rxr_pkt_handle_runtread_rtm_send_completion(struct efa_rdm_ep *ep,
						 struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	size_t pkt_data_size;

	txe = pkt_entry->ope;
	pkt_data_size = pkt_entry->payload_size;
	txe->bytes_acked += pkt_data_size;

	peer = efa_rdm_ep_get_peer(ep, pkt_entry->addr);
	assert(peer);
	assert(peer->num_runt_bytes_in_flight >= pkt_data_size);
	peer->num_runt_bytes_in_flight -= pkt_data_size;
	if (txe->total_len == txe->bytes_acked)
		efa_rdm_ope_handle_send_completed(txe);
}

/*
 *     proc() functions
 */
size_t rxr_pkt_rtm_total_len(struct efa_rdm_pke *pkt_entry)
{
	struct rxr_base_hdr *base_hdr;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	switch (base_hdr->type) {
	case RXR_EAGER_MSGRTM_PKT:
	case RXR_EAGER_TAGRTM_PKT:
	case RXR_DC_EAGER_MSGRTM_PKT:
	case RXR_DC_EAGER_TAGRTM_PKT:
		return pkt_entry->payload_size;
	case RXR_MEDIUM_MSGRTM_PKT:
	case RXR_MEDIUM_TAGRTM_PKT:
		return rxr_get_medium_rtm_base_hdr(pkt_entry->wiredata)->msg_length;
	case RXR_DC_MEDIUM_MSGRTM_PKT:
	case RXR_DC_MEDIUM_TAGRTM_PKT:
		return rxr_get_dc_medium_rtm_base_hdr(pkt_entry->wiredata)->msg_length;
	case RXR_LONGCTS_MSGRTM_PKT:
	case RXR_LONGCTS_TAGRTM_PKT:
	case RXR_DC_LONGCTS_MSGRTM_PKT:
	case RXR_DC_LONGCTS_TAGRTM_PKT:
		return rxr_get_longcts_rtm_base_hdr(pkt_entry->wiredata)->msg_length;
	case RXR_LONGREAD_MSGRTM_PKT:
	case RXR_LONGREAD_TAGRTM_PKT:
		return rxr_get_longread_rtm_base_hdr(pkt_entry->wiredata)->msg_length;
	case RXR_RUNTREAD_MSGRTM_PKT:
	case RXR_RUNTREAD_TAGRTM_PKT:
		return rxr_get_runtread_rtm_base_hdr(pkt_entry->wiredata)->msg_length;
	default:
		assert(0 && "Unknown REQ packet type\n");
	}

	return 0;
}

/*
 * @brief Update rxe with the following information in RTM packet entry.
 *            address:       this is necessary because original address in
 *                           rxe can be FI_ADDR_UNSPEC
 *            cq_entry.data: for FI_REMOTE_CQ_DATA
 *            msg_id:        message id
 *            total_len:     application might provide a buffer that is larger
 *                           then incoming message size.
 *            tag:           sender's tag can be different from receiver's tag
 *                           becuase match only requires
 *                           (sender_tag | ignore) == (receiver_tag | ignore)
 *        This function is applied to both unexpected rxe (when they are
 *        allocated) and expected rxe (when they are matched to a RTM)
 *
 * @param pkt_entry(input)  RTM packet entry
 * @param rxe(input)   rxe to be updated
 */
void rxr_pkt_rtm_update_rxe(struct efa_rdm_pke *pkt_entry,
				 struct efa_rdm_ope *rxe)
{
	struct rxr_base_hdr *base_hdr;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	if (base_hdr->flags & RXR_REQ_OPT_CQ_DATA_HDR) {
		rxe->cq_entry.flags |= FI_REMOTE_CQ_DATA;
		rxe->cq_entry.data = rxr_pkt_req_cq_data(pkt_entry);
	}

	rxe->addr = pkt_entry->addr;
	rxe->msg_id = rxr_pkt_msg_id(pkt_entry);
	rxe->total_len = rxr_pkt_rtm_total_len(pkt_entry);
	rxe->tag = rxr_pkt_rtm_tag(pkt_entry);
	rxe->cq_entry.tag = rxe->tag;
}

struct efa_rdm_ope *rxr_pkt_get_rtm_matched_rxe(struct efa_rdm_ep *ep,
						struct efa_rdm_pke *pkt_entry,
						struct fi_peer_rx_entry *peer_rxe,
						uint32_t op)
{
	struct efa_rdm_ope *rxe;

	rxe = efa_rdm_ep_alloc_rxe(ep, pkt_entry->addr, op);
	if (OFI_UNLIKELY(!rxe))
		return NULL;

	rxe->state = EFA_RDM_RXE_MATCHED;

	efa_rdm_srx_update_rxe(peer_rxe, rxe);
	rxr_pkt_rtm_update_rxe(pkt_entry, rxe);

	return rxe;
}

struct efa_rdm_ope *rxr_pkt_get_msgrtm_rxe(struct efa_rdm_ep *ep,
						 struct efa_rdm_pke **pkt_entry_ptr)
{
	struct fid_peer_srx *peer_srx;
	struct fi_peer_rx_entry *peer_rxe;
	struct efa_rdm_ope *rxe;
	size_t data_size;
	int ret;
	int pkt_type;

	if ((*pkt_entry_ptr)->alloc_type == EFA_RDM_PKE_FROM_USER_BUFFER) {
		/* If a pkt_entry is constructred from user supplied buffer,
		 * the endpoint must be in zero copy receive mode.
		 */
		assert(ep->use_zcpy_rx);
		/* In this mode, an rxe is always created together
		 * with this pkt_entry, and pkt_entry->ope is pointing
		 * to it. Thus we can skip the matching process, and return
		 * pkt_entry->ope right away.
		 */
		assert((*pkt_entry_ptr)->ope);
		return (*pkt_entry_ptr)->ope;
	}

	peer_srx = util_get_peer_srx(ep->peer_srx_ep);
	data_size = rxr_pkt_rtm_total_len(*pkt_entry_ptr);

	ret = peer_srx->owner_ops->get_msg(peer_srx, (*pkt_entry_ptr)->addr, data_size, &peer_rxe);

	if (ret == FI_SUCCESS) { /* A matched rxe is found */
		rxe = rxr_pkt_get_rtm_matched_rxe(ep, *pkt_entry_ptr, peer_rxe, ofi_op_msg);
		if (OFI_UNLIKELY(!rxe)) {
			efa_base_ep_write_eq_error(&ep->base_ep, FI_ENOBUFS, FI_EFA_ERR_RXE_POOL_EXHAUSTED);
			return NULL;
		}
		rxr_tracepoint(msg_match_expected_nontagged, rxe->msg_id,
			    (size_t) rxe->cq_entry.op_context, rxe->total_len);
	} else if (ret == -FI_ENOENT) { /* No matched rxe is found */
		/*
		 * efa_rdm_msg_alloc_unexp_rxe_for_rtm() might release pkt_entry,
		 * thus we have to use pkt_entry_ptr here
		 */
		rxe = efa_rdm_msg_alloc_unexp_rxe_for_rtm(ep, pkt_entry_ptr, ofi_op_msg);
		if (OFI_UNLIKELY(!rxe)) {
			efa_base_ep_write_eq_error(&ep->base_ep, FI_ENOBUFS, FI_EFA_ERR_RXE_POOL_EXHAUSTED);
			return NULL;
		}
		(*pkt_entry_ptr)->ope = rxe;
		peer_rxe->peer_context = (*pkt_entry_ptr);
		rxe->peer_rxe = peer_rxe;
		rxr_tracepoint(msg_recv_unexpected_nontagged, rxe->msg_id,
			    (size_t) rxe->cq_entry.op_context, rxe->total_len);
	} else { /* Unexpected errors */
		EFA_WARN(FI_LOG_EP_CTRL,
			"get_msg failed, error: %d\n",
			ret);
		return NULL;
	}

	pkt_type = efa_rdm_pke_get_base_hdr(*pkt_entry_ptr)->type;
	if (efa_rdm_pkt_type_is_mulreq(pkt_type))
		rxr_pkt_rx_map_insert(ep, *pkt_entry_ptr, rxe);

	return rxe;
}

struct efa_rdm_ope *rxr_pkt_get_tagrtm_rxe(struct efa_rdm_ep *ep,
						 struct efa_rdm_pke **pkt_entry_ptr)
{
	struct fid_peer_srx *peer_srx;
	struct fi_peer_rx_entry *peer_rxe;
	struct efa_rdm_ope *rxe;
	size_t data_size;
	int ret;
	int pkt_type;

	peer_srx = util_get_peer_srx(ep->peer_srx_ep);
	data_size = rxr_pkt_rtm_total_len(*pkt_entry_ptr);

	ret = peer_srx->owner_ops->get_tag(peer_srx, (*pkt_entry_ptr)->addr,
					   data_size,
					   rxr_pkt_rtm_tag(*pkt_entry_ptr),
					   &peer_rxe);

	if (ret == FI_SUCCESS) { /* A matched rxe is found */
		rxe = rxr_pkt_get_rtm_matched_rxe(ep, *pkt_entry_ptr, peer_rxe, ofi_op_tagged);
		if (OFI_UNLIKELY(!rxe)) {
			efa_base_ep_write_eq_error(&ep->base_ep, FI_ENOBUFS, FI_EFA_ERR_RXE_POOL_EXHAUSTED);
			return NULL;
		}
		rxr_tracepoint(msg_match_expected_tagged, rxe->msg_id,
			    (size_t) rxe->cq_entry.op_context, rxe->total_len);
	} else if (ret == -FI_ENOENT) { /* No matched rxe is found */
		/*
		 * efa_rdm_msg_alloc_unexp_rxe_for_rtm() might release pkt_entry,
		 * thus we have to use pkt_entry_ptr here
		 */
		rxe = efa_rdm_msg_alloc_unexp_rxe_for_rtm(ep, pkt_entry_ptr, ofi_op_tagged);
		if (OFI_UNLIKELY(!rxe)) {
			efa_base_ep_write_eq_error(&ep->base_ep, FI_ENOBUFS, FI_EFA_ERR_RXE_POOL_EXHAUSTED);
			return NULL;
		}
		(*pkt_entry_ptr)->ope = rxe;
		peer_rxe->peer_context = *pkt_entry_ptr;
		rxe->peer_rxe = peer_rxe;
		rxr_tracepoint(msg_recv_unexpected_tagged, rxe->msg_id,
			    (size_t) rxe->cq_entry.op_context, rxe->total_len);
	} else { /* Unexpected errors */
		EFA_WARN(FI_LOG_EP_CTRL,
			"get_tag failed, error: %d\n",
			ret);
		return NULL;
	}

	pkt_type = efa_rdm_pke_get_base_hdr(*pkt_entry_ptr)->type;
	if (efa_rdm_pkt_type_is_mulreq(pkt_type))
		rxr_pkt_rx_map_insert(ep, *pkt_entry_ptr, rxe);

	return rxe;
}

ssize_t rxr_pkt_proc_matched_longread_rtm(struct efa_rdm_ep *ep,
				      struct efa_rdm_ope *rxe,
				      struct efa_rdm_pke *pkt_entry)
{
	struct rxr_longread_rtm_base_hdr *rtm_hdr;
	struct fi_rma_iov *read_iov;

	rtm_hdr = rxr_get_longread_rtm_base_hdr(pkt_entry->wiredata);
	read_iov = (struct fi_rma_iov *)(pkt_entry->wiredata +
									rxr_pkt_req_hdr_size_from_pkt_entry(pkt_entry));

	rxe->tx_id = rtm_hdr->send_id;
	rxe->rma_iov_count = rtm_hdr->read_iov_count;
	memcpy(rxe->rma_iov, read_iov,
	       rxe->rma_iov_count * sizeof(struct fi_rma_iov));

	efa_rdm_pke_release_rx(ep, pkt_entry);
	rxr_tracepoint(longread_read_posted, rxe->msg_id,
		    (size_t) rxe->cq_entry.op_context, rxe->total_len);

	return efa_rdm_ope_post_remote_read_or_queue(rxe);
}

ssize_t rxr_pkt_proc_matched_mulreq_rtm(struct efa_rdm_ep *ep,
					struct efa_rdm_ope *rxe,
					struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_pke *cur, *nxt;
	int pkt_type;
	ssize_t ret, err;

	pkt_type = efa_rdm_pke_get_base_hdr(pkt_entry)->type;

	if (efa_rdm_pkt_type_is_runtread(pkt_type)) {
		struct rxr_runtread_rtm_base_hdr *runtread_rtm_hdr;

		runtread_rtm_hdr = rxr_get_runtread_rtm_base_hdr(pkt_entry->wiredata);
		rxe->bytes_runt = runtread_rtm_hdr->runt_length;
		if (rxe->total_len > rxe->bytes_runt && rxe->bytes_read_total_len == 0) {
			struct fi_rma_iov *read_iov;

			rxe->tx_id = runtread_rtm_hdr->send_id;
			read_iov = (struct fi_rma_iov *)(pkt_entry->wiredata + rxr_pkt_req_hdr_size_from_pkt_entry(pkt_entry));
			rxe->rma_iov_count = runtread_rtm_hdr->read_iov_count;
			memcpy(rxe->rma_iov, read_iov, rxe->rma_iov_count * sizeof(struct fi_rma_iov));
			rxr_tracepoint(runtread_read_posted, rxe->msg_id,
				    (size_t) rxe->cq_entry.op_context, rxe->total_len);

			err = efa_rdm_ope_post_remote_read_or_queue(rxe);
			if (err)
				return err;
		}
	}

	ret = 0;
	cur = pkt_entry;
	while (cur) {
		assert(cur->payload);
		assert(cur->payload_size);
		/* efa_rdm_pke_copy_payload_to_ope() can release rxe, so
		 * bytes_received must be calculated before it.
		 */
		rxe->bytes_received += cur->payload_size;
		rxe->bytes_received_via_mulreq += cur->payload_size;
		if (efa_rdm_ope_mulreq_total_data_size(rxe, pkt_type) == rxe->bytes_received_via_mulreq)
			rxr_pkt_rx_map_remove(ep, cur, rxe);

		/* efa_rdm_copy_payload_to_ope() will release cur, so
		 * cur->next must be copied out before it.
		 */
		nxt = cur->next;
		cur->next = NULL;

		err = efa_rdm_pke_copy_payload_to_ope(cur, rxe);
		if (err) {
			efa_rdm_pke_release_rx(ep, cur);
			ret = err;
		}

		cur = nxt;
	}

	return ret;
}

/**
 * @brief process a matched eager rtm packet entry
 *
 * For an eager message, it will write rx completion,
 * release packet entry and rxe.
 *
 * @param[in]	ep		endpoint
 * @param[in]	rxe	rxe
 * @param[in]	pkt_entry	packet entry
 * @return	On success, return 0
 * 		On failure, return libfabric error code
 */
ssize_t rxr_pkt_proc_matched_eager_rtm(struct efa_rdm_ep *ep,
				       struct efa_rdm_ope *rxe,
				       struct efa_rdm_pke *pkt_entry)
{
	int err;
	int hdr_size;

	if (pkt_entry->alloc_type != EFA_RDM_PKE_FROM_USER_BUFFER) {
		/*
		 * On success, efa_pke_copy_payload_to_ope will write rx completion,
		 * release pkt_entry and rxe
		 */
		err = efa_rdm_pke_copy_payload_to_ope(pkt_entry, rxe);
		if (err)
			efa_rdm_pke_release_rx(ep, pkt_entry);

		return err;
	}

	/* In this case, data is already in user provided buffer, so no need
	 * to copy. However, we do need to make sure the packet header length
	 * is correct. Otherwise, user will get wrong data.
	 *
	 * The expected header size is
	 * 	ep->msg_prefix_size - sizeof(struct efa_rdm_pke)
	 * because we used the first sizeof(struct efa_rdm_pke) to construct
	 * a pkt_entry.
	 */
	hdr_size = pkt_entry->payload - pkt_entry->wiredata;
	if (hdr_size != ep->msg_prefix_size - sizeof(struct efa_rdm_pke)) {
		/* if header size is wrong, the data in user buffer is not useful.
		 * setting rxe->cq_entry.len here will cause an error cq entry
		 * to be written to application.
		 */
		rxe->cq_entry.len = 0;
	} else {
		assert(rxe->cq_entry.buf == pkt_entry->wiredata - sizeof(struct efa_rdm_pke));
		rxe->cq_entry.len = pkt_entry->pkt_size + sizeof(struct efa_rdm_pke);
	}

	efa_rdm_rxe_report_completion(rxe);
	efa_rdm_rxe_release(rxe);

	/* no need to release packet entry because it is
	 * constructed using user supplied buffer */
	return 0;
}

ssize_t rxr_pkt_proc_matched_rtm(struct efa_rdm_ep *ep,
				 struct efa_rdm_ope *rxe,
				 struct efa_rdm_pke *pkt_entry)
{
	int pkt_type;
	ssize_t ret;

	assert(rxe->state == EFA_RDM_RXE_MATCHED);

	if (!rxe->peer) {
		rxe->addr = pkt_entry->addr;
		rxe->peer = efa_rdm_ep_get_peer(ep, rxe->addr);
		assert(rxe->peer);
		dlist_insert_tail(&rxe->peer_entry, &rxe->peer->rxe_list);
	}

	/* Adjust rxe->cq_entry.len as needed.
	 * Initialy rxe->cq_entry.len is total recv buffer size.
	 * rxe->total_len is from REQ packet and is total send buffer size.
	 * if send buffer size < recv buffer size, we adjust value of rxe->cq_entry.len
	 * if send buffer size > recv buffer size, we have a truncated message and will
	 * write error CQ entry.
	 */
	if (rxe->cq_entry.len > rxe->total_len)
		rxe->cq_entry.len = rxe->total_len;

	pkt_type = efa_rdm_pke_get_base_hdr(pkt_entry)->type;

	if (pkt_type > RXR_DC_REQ_PKT_BEGIN &&
	    pkt_type < RXR_DC_REQ_PKT_END)
		rxe->rxr_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	if (pkt_type == RXR_LONGCTS_MSGRTM_PKT ||
	    pkt_type == RXR_LONGCTS_TAGRTM_PKT)
		rxe->tx_id = rxr_get_longcts_rtm_base_hdr(pkt_entry->wiredata)->send_id;
	else if (pkt_type == RXR_DC_EAGER_MSGRTM_PKT ||
		 pkt_type == RXR_DC_EAGER_TAGRTM_PKT)
		rxe->tx_id = rxr_get_dc_eager_rtm_base_hdr(pkt_entry->wiredata)->send_id;
	else if (pkt_type == RXR_DC_MEDIUM_MSGRTM_PKT ||
		 pkt_type == RXR_DC_MEDIUM_TAGRTM_PKT)
		rxe->tx_id = rxr_get_dc_medium_rtm_base_hdr(pkt_entry->wiredata)->send_id;
	else if (pkt_type == RXR_DC_LONGCTS_MSGRTM_PKT ||
		 pkt_type == RXR_DC_LONGCTS_TAGRTM_PKT)
		rxe->tx_id = rxr_get_longcts_rtm_base_hdr(pkt_entry->wiredata)->send_id;

	rxe->msg_id = rxr_get_rtm_base_hdr(pkt_entry->wiredata)->msg_id;

	if (pkt_type == RXR_LONGREAD_MSGRTM_PKT || pkt_type == RXR_LONGREAD_TAGRTM_PKT)
		return rxr_pkt_proc_matched_longread_rtm(ep, rxe, pkt_entry);

	if (efa_rdm_pkt_type_is_mulreq(pkt_type))
		return rxr_pkt_proc_matched_mulreq_rtm(ep, rxe, pkt_entry);

	if (pkt_type == RXR_EAGER_MSGRTM_PKT ||
	    pkt_type == RXR_EAGER_TAGRTM_PKT ||
	    pkt_type == RXR_DC_EAGER_MSGRTM_PKT ||
	    pkt_type == RXR_DC_EAGER_TAGRTM_PKT) {
		return rxr_pkt_proc_matched_eager_rtm(ep, rxe, pkt_entry);
	}

	rxe->bytes_received += pkt_entry->payload_size;
	ret = efa_rdm_pke_copy_payload_to_ope(pkt_entry, rxe);
	if (ret) {
		return ret;
	}
#if ENABLE_DEBUG
	dlist_insert_tail(&rxe->pending_recv_entry, &ep->ope_recv_list);
	ep->pending_recv_counter++;
#endif
	rxe->state = EFA_RDM_RXE_RECV;
	ret = efa_rdm_ope_post_send_or_queue(rxe, RXR_CTS_PKT);

	return ret;
}

ssize_t rxr_pkt_proc_msgrtm(struct efa_rdm_ep *ep,
			    struct efa_rdm_pke *pkt_entry)
{
	ssize_t err;
	struct efa_rdm_ope *rxe;
	struct fid_peer_srx *peer_srx;

	rxe = rxr_pkt_get_msgrtm_rxe(ep, &pkt_entry);
	if (OFI_UNLIKELY(!rxe)) {
		efa_base_ep_write_eq_error(&ep->base_ep, FI_ENOBUFS, FI_EFA_ERR_RXE_POOL_EXHAUSTED);
		efa_rdm_pke_release_rx(ep, pkt_entry);
		return -FI_ENOBUFS;
	}

	if (rxe->state == EFA_RDM_RXE_MATCHED) {
		err = rxr_pkt_proc_matched_rtm(ep, rxe, pkt_entry);
		if (OFI_UNLIKELY(err)) {
			efa_rdm_rxe_handle_error(rxe, -err, FI_EFA_ERR_PKT_PROC_MSGRTM);
			efa_rdm_pke_release_rx(ep, pkt_entry);
			efa_rdm_rxe_release(rxe);
			return err;
		}
	} else if (rxe->state == EFA_RDM_RXE_UNEXP) {
		peer_srx = util_get_peer_srx(ep->peer_srx_ep);
		return peer_srx->owner_ops->queue_msg(rxe->peer_rxe);
	}

	return 0;
}

ssize_t rxr_pkt_proc_tagrtm(struct efa_rdm_ep *ep,
			    struct efa_rdm_pke *pkt_entry)
{
	ssize_t err;
	struct efa_rdm_ope *rxe;
	struct fid_peer_srx *peer_srx;

	rxe = rxr_pkt_get_tagrtm_rxe(ep, &pkt_entry);
	if (OFI_UNLIKELY(!rxe)) {
		efa_base_ep_write_eq_error(&ep->base_ep, FI_ENOBUFS, FI_EFA_ERR_RXE_POOL_EXHAUSTED);
		efa_rdm_pke_release_rx(ep, pkt_entry);
		return -FI_ENOBUFS;
	}

	if (rxe->state == EFA_RDM_RXE_MATCHED) {
		err = rxr_pkt_proc_matched_rtm(ep, rxe, pkt_entry);
		if (OFI_UNLIKELY(err)) {
			efa_rdm_rxe_handle_error(rxe, -err, FI_EFA_ERR_PKT_PROC_TAGRTM);
			efa_rdm_pke_release_rx(ep, pkt_entry);
			efa_rdm_rxe_release(rxe);
			return err;
		}
	} else if (rxe->state == EFA_RDM_RXE_UNEXP) {
		peer_srx = util_get_peer_srx(ep->peer_srx_ep);
		return peer_srx->owner_ops->queue_tag(rxe->peer_rxe);
	}

	return 0;
}

/*
 * proc() functions called by rxr_pkt_handle_recv_completion()
 */
ssize_t rxr_pkt_proc_rtm_rta(struct efa_rdm_ep *ep,
			     struct efa_rdm_pke *pkt_entry)
{
	struct rxr_base_hdr *base_hdr;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	assert(base_hdr->type >= RXR_BASELINE_REQ_PKT_BEGIN);

	switch (base_hdr->type) {
	case RXR_EAGER_MSGRTM_PKT:
	case RXR_MEDIUM_MSGRTM_PKT:
	case RXR_LONGCTS_MSGRTM_PKT:
	case RXR_LONGREAD_MSGRTM_PKT:
	case RXR_RUNTREAD_MSGRTM_PKT:
	case RXR_DC_EAGER_MSGRTM_PKT:
	case RXR_DC_MEDIUM_MSGRTM_PKT:
	case RXR_DC_LONGCTS_MSGRTM_PKT:
		return rxr_pkt_proc_msgrtm(ep, pkt_entry);
	case RXR_EAGER_TAGRTM_PKT:
	case RXR_MEDIUM_TAGRTM_PKT:
	case RXR_LONGCTS_TAGRTM_PKT:
	case RXR_LONGREAD_TAGRTM_PKT:
	case RXR_RUNTREAD_TAGRTM_PKT:
	case RXR_DC_EAGER_TAGRTM_PKT:
	case RXR_DC_MEDIUM_TAGRTM_PKT:
	case RXR_DC_LONGCTS_TAGRTM_PKT:
		return rxr_pkt_proc_tagrtm(ep, pkt_entry);
	case RXR_WRITE_RTA_PKT:
		return efa_rdm_pke_proc_write_rta(pkt_entry);
	case RXR_DC_WRITE_RTA_PKT:
		return efa_rdm_pke_proc_dc_write_rta(pkt_entry);
	case RXR_FETCH_RTA_PKT:
		return efa_rdm_pke_proc_fetch_rta(pkt_entry);
	case RXR_COMPARE_RTA_PKT:
		return efa_rdm_pke_proc_compare_rta(pkt_entry);
	default:
		EFA_WARN(FI_LOG_EP_CTRL,
			"Unknown packet type ID: %d\n",
		       base_hdr->type);
		efa_base_ep_write_eq_error(&ep->base_ep, FI_EINVAL, FI_EFA_ERR_UNKNOWN_PKT_TYPE);
		efa_rdm_pke_release_rx(ep, pkt_entry);
	}

	return -FI_EINVAL;
}

void rxr_pkt_handle_rtm_rta_recv(struct efa_rdm_ep *ep,
				 struct efa_rdm_pke *pkt_entry)
{
	struct rxr_base_hdr *base_hdr;
	struct efa_rdm_peer *peer;
	int ret, msg_id;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	assert(base_hdr->type >= RXR_BASELINE_REQ_PKT_BEGIN);

	if (efa_rdm_pkt_type_is_mulreq(base_hdr->type)) {
		struct efa_rdm_ope *rxe;
		struct efa_rdm_pke *unexp_pkt_entry;

		rxe = rxr_pkt_rx_map_lookup(ep, pkt_entry);
		if (rxe) {
			if (rxe->state == EFA_RDM_RXE_MATCHED) {
				rxr_pkt_proc_matched_mulreq_rtm(ep, rxe, pkt_entry);
			} else {
				assert(rxe->unexp_pkt);
				unexp_pkt_entry = rxr_pkt_get_unexp(ep, &pkt_entry);
				efa_rdm_pke_append(rxe->unexp_pkt, unexp_pkt_entry);
			}

			return;
		}
	}

	peer = efa_rdm_ep_get_peer(ep, pkt_entry->addr);
	assert(peer);
	msg_id = rxr_pkt_msg_id(pkt_entry);
	ret = efa_rdm_peer_reorder_msg(peer, ep, pkt_entry);
	if (ret == 1) {
		/* Packet was queued */
		return;
	}

	if (OFI_UNLIKELY(ret == -FI_EALREADY)) {
		/* Packet with same msg_id has been processed before */
		EFA_WARN(FI_LOG_EP_CTRL,
			"Invalid msg_id: %" PRIu32
			" robuf->exp_msg_id: %" PRIu32 "\n",
		       msg_id, peer->robuf.exp_msg_id);
		efa_base_ep_write_eq_error(&ep->base_ep, FI_EIO, FI_EFA_ERR_PKT_ALREADY_PROCESSED);
		efa_rdm_pke_release_rx(ep, pkt_entry);
		return;
	}

	if (OFI_UNLIKELY(ret == -FI_ENOMEM)) {
		/* running out of memory while copy packet */
		efa_base_ep_write_eq_error(&ep->base_ep, FI_ENOBUFS, FI_EFA_ERR_OOM);
		return;
	}

	if (OFI_UNLIKELY(ret < 0)) {
		EFA_WARN(FI_LOG_EP_CTRL,
			"Unknown error %d processing REQ packet msg_id: %"
			PRIu32 "\n", ret, msg_id);
		efa_base_ep_write_eq_error(&ep->base_ep, FI_EIO, FI_EFA_ERR_OTHER);
		return;
	}


	/*
	 * rxr_pkt_proc_rtm_rta() will write error cq entry if needed,
	 * thus we do not write error cq entry
	 */
	ret = rxr_pkt_proc_rtm_rta(ep, pkt_entry);
	if (OFI_UNLIKELY(ret))
		return;

	ofi_recvwin_slide((&peer->robuf));
	efa_rdm_peer_proc_pending_items_in_robuf(peer, ep);
}
