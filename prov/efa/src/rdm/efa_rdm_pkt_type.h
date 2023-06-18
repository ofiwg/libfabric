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

#ifndef _EFA_RDM_PKT_TYPE_H
#define _EFA_RDM_PKT_TYPE_H

#include "efa_rdm_protocol.h"

/**
 * @brief determine whether a packet types header cotains the "rma_iov" field
 * @returns a boolean
 */
static inline
int efa_rdm_pkt_type_contains_rma_iov(int pkt_type)
{
	switch (pkt_type) {
		case RXR_EAGER_RTW_PKT:
		case RXR_DC_EAGER_RTW_PKT:
		case RXR_LONGCTS_RTW_PKT:
		case RXR_DC_LONGCTS_RTW_PKT:
		case RXR_LONGREAD_RTW_PKT:
		case RXR_SHORT_RTR_PKT:
		case RXR_LONGCTS_RTR_PKT:
		case RXR_WRITE_RTA_PKT:
		case RXR_DC_WRITE_RTA_PKT:
		case RXR_FETCH_RTA_PKT:
		case RXR_COMPARE_RTA_PKT:
			return 1;
			break;
		default:
			return 0;
			break;
	}
}

/**
 * @brief determine whether a req pkt type is part of a runt protocol
 *
 * A runt protocol send user data into two parts. The first part
 * was sent by multiple eagerly sent packages. The rest of the
 * data is sent regularly.
 *
 * @param[in]		pkt_type		REQ packet type
 * @return		a boolean
 */
static inline
bool efa_rdm_pkt_type_is_runt(int pkt_type)
{
	return (pkt_type >= RXR_RUNT_PKT_BEGIN && pkt_type < RXR_RUNT_PKT_END);
}

/**
 * @brief determine whether a req pkt type is eager RTM or RTW
 *
 * @param[in]		pkt_type		REQ packet type
 * @return		a boolean
 */
static inline
bool efa_rdm_pkt_type_is_eager(int pkt_type)
{
	switch(pkt_type) {
	case RXR_EAGER_MSGRTM_PKT:
	case RXR_EAGER_TAGRTM_PKT:
	case RXR_EAGER_RTW_PKT:
	case RXR_DC_EAGER_MSGRTM_PKT:
	case RXR_DC_EAGER_TAGRTM_PKT:
	case RXR_DC_EAGER_RTW_PKT:
		return 1;
	default:
		return 0;
	}
}

/**
 * @brief determine whether a req pkt type is part of a medium protocol
 *
 * medium protocol send user data eagerly without CTS based flow control.
 *
 * @param[in]		pkt_type		REQ packet type
 * @return		a boolean
 */
static inline
bool efa_rdm_pkt_type_is_medium(int pkt_type)
{
	return pkt_type == RXR_MEDIUM_TAGRTM_PKT || pkt_type == RXR_MEDIUM_MSGRTM_PKT ||
	       pkt_type == RXR_DC_MEDIUM_MSGRTM_PKT ||pkt_type == RXR_DC_MEDIUM_TAGRTM_PKT;
}

/**
 * @brief determine whether a req pkt type is longcts RTM or RTW.
 *
 * @param[in]		pkt_type		REQ packet type
 * @return		a boolean
 */
static inline
bool efa_rdm_pkt_type_is_longcts_req(int pkt_type)
{
	switch(pkt_type) {
	case RXR_LONGCTS_MSGRTM_PKT:
	case RXR_LONGCTS_TAGRTM_PKT:
	case RXR_LONGCTS_RTW_PKT:
	case RXR_DC_LONGCTS_MSGRTM_PKT:
	case RXR_DC_LONGCTS_TAGRTM_PKT:
	case RXR_DC_LONGCTS_RTW_PKT:
		return 1;
	default:
		return 0;
	}
}

/**
 * @brief determine whether a req pkt type is RTA
 *
 * @param[in]		pkt_type		REQ packet type
 * @return		a boolean
 */
static inline
bool efa_rdm_pkt_type_is_rta(int pkt_type)
{
	switch(pkt_type) {
	case RXR_WRITE_RTA_PKT:
	case RXR_FETCH_RTA_PKT:
	case RXR_COMPARE_RTA_PKT:
	case RXR_DC_WRITE_RTA_PKT:
		return 1;
	default:
		return 0;
	}
}

/**
 * @brief determine whether a pkt type is runtread rtm
 *
 * @param[in]		pkt_type		REQ packet type
 * @return		a boolean
 */
static inline
bool efa_rdm_pkt_type_is_runtread(int pkt_type)
{
	return pkt_type == RXR_RUNTREAD_TAGRTM_PKT || pkt_type == RXR_RUNTREAD_MSGRTM_PKT;
}

/**
 * @brief determine whether a req pkt type is part of a multi-req protocol
 *
 * A multi-req protocol sends multiple (>=2) data containing REQ packets.
 * This function determine whether a req pkt type is part of a multi-req
 * protocol
 *
 * @param[in]		pkt_type		REQ packet type
 * @return		a boolean
 */
static inline
bool efa_rdm_pkt_type_is_mulreq(int pkt_type)
{
	return efa_rdm_pkt_type_is_medium(pkt_type) || efa_rdm_pkt_type_is_runt(pkt_type);
}

/**
 * @brief determine whether a packet for a given type has user data in it.
 *
 * @param[in]		pkt_type		packet type
 * @return		a boolean
 */
static inline
bool efa_rdm_pkt_type_contains_data(int pkt_type)
{
	return pkt_type == RXR_READRSP_PKT ||
	       pkt_type == RXR_ATOMRSP_PKT ||
	       pkt_type == RXR_DATA_PKT ||
	       efa_rdm_pkt_type_is_runt(pkt_type) ||
	       efa_rdm_pkt_type_is_eager(pkt_type) ||
	       efa_rdm_pkt_type_is_medium(pkt_type) ||
	       efa_rdm_pkt_type_is_longcts_req(pkt_type) ||
	       efa_rdm_pkt_type_is_rta(pkt_type);
}

#endif
