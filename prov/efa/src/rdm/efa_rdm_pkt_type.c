#include <stdint.h>
#include "efa_mr.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_peer.h"
#include "efa_rdm_protocol.h"
#include "efa_rdm_pkt_type.h"

struct efa_rdm_pkt_type_req_info EFA_RDM_PKT_TYPE_REQ_INFO_VEC[] = {
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

/**
 * @brief determine whether a REQ packet type is supported by the peer
 *
 * Some REQ packet type rely on an extra feature, which may not be
 * supported by peer.
 *
 * It is caller's responsibility to ensure this function is
 * called after the handshake packet has been received from the peer.
 *
 * @param[in]		req_pkt_type	REQ packet type
 * @param[in]		peer		peer, from whose the handshake has been received.
 * @returns
 * a boolean
 */
bool efa_rdm_pkt_type_is_supported_by_peer(int req_pkt_type, struct efa_rdm_peer *peer)
{
	assert(peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED);

	int extra_info_id = EFA_RDM_PKT_TYPE_REQ_INFO_VEC[req_pkt_type].extra_info_id;

	return peer->extra_info[extra_info_id] & EFA_RDM_PKT_TYPE_REQ_INFO_VEC[req_pkt_type].ex_feature_flag;
}

/**
 * @brief calculates the exact header size given a REQ packet type, flags, and IOV count.
 *
 * @param[in]	pkt_type	packet type
 * @param[in]	flags	flags from packet
 * @param[in]	rma_iov_count	number of RMA IOV structures present
 * @return	The exact size of the packet header
 */
size_t efa_rdm_pkt_type_get_req_hdr_size(int pkt_type, uint16_t flags, size_t rma_iov_count)
{
	int hdr_size = EFA_RDM_PKT_TYPE_REQ_INFO_VEC[pkt_type].base_hdr_size;

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
static inline
size_t efa_rdm_pkt_type_get_req_max_hdr_size(int pkt_type)
{
	/* To calculate max REQ reader size, we should include all possible REQ opt header flags.
	 * However, because the optional connid header and optional raw address header cannot
	 * exist at the same time, and the raw address header is longer than connid header,
	 * we did not include the flag for CONNID header
	 */
	uint16_t header_flags = RXR_REQ_OPT_RAW_ADDR_HDR | RXR_REQ_OPT_CQ_DATA_HDR;

	return efa_rdm_pkt_type_get_req_hdr_size(pkt_type, header_flags, RXR_IOV_LIMIT);
}

/**
 * @brief maximum header size of all possible packet types
 */
size_t efa_rdm_pkt_type_get_max_hdr_size(void)
{
	size_t max_hdr_size = 0;
	size_t pkt_type = RXR_REQ_PKT_BEGIN;

	while (pkt_type < RXR_EXTRA_REQ_PKT_END) {
		max_hdr_size = MAX(max_hdr_size,
				efa_rdm_pkt_type_get_req_max_hdr_size(pkt_type));
		if (pkt_type == RXR_BASELINE_REQ_PKT_END)
			pkt_type = RXR_EXTRA_REQ_PKT_BEGIN;
		else
			pkt_type += 1;
	}

	return max_hdr_size;
}