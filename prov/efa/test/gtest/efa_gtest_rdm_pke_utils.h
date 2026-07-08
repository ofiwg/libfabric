/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

/* C-linkage bridge for rdm pke tests. 
 * See efa_gtest_common_helpers.h for why this exists. */

#ifndef EFA_GTEST_RDM_PKE_UTILS_H
#define EFA_GTEST_RDM_PKE_UTILS_H

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calls the RTM RX handler with a LONGCTS READ_NACK packet whose msg_id
 * is missing from the peer's rxe_map.
 *
 * @param[in]	ep		RDM efa endpoint
 * @param[in]	peer_addr	pre-inserted fi_addr of the peer
 * @param[in]	tagged		non-zero drives the tagged path (proc_tagrtm)
 * @param[out]	ret		return code of the handler
 * @return	1 if the handler returned without crashing, 0 on setup failure
 */
int efa_test_rtm_read_nack_missing_rxe(struct fid_ep *ep, fi_addr_t peer_addr,
				       int tagged, ssize_t *ret);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_RDM_PKE_UTILS_H */
