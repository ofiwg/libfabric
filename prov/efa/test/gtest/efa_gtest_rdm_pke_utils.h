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
struct efa_rdm_pke;

/**
 * @brief Allocate a linked list of n unexpected packet entries from the
 * endpoint's unexpected packet pool.
 * @return head of the chain, or NULL on allocation failure.
 */
struct efa_rdm_pke *efa_test_pke_build_unexp_chain(struct fid_ep *ep, size_t n);

void efa_test_pke_release_cloned(struct efa_rdm_pke *head);

/**
 * @brief Count allocated buffers in the unexpected packet pool.
 */
size_t efa_test_ep_unexp_pool_outstanding(struct fid_ep *ep);

int efa_test_failed_reorder_msg_releases_rx_pkt(struct fid_ep *ep,
						fi_addr_t peer_addr,
						size_t *to_post_before,
						size_t *to_post_after);

int efa_test_failed_reorder_msg_overflow_releases_rx_pkt_and_entry(
	struct fid_ep *ep, fi_addr_t peer_addr, size_t *to_post_before,
	size_t *to_post_after, size_t *overflow_free_before,
	size_t *overflow_free_after);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_RDM_PKE_UTILS_H */
