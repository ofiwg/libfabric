/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_RDM_AV_UTILS_H
#define EFA_GTEST_RDM_AV_UTILS_H

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct efa_av_array;
struct efa_prv_reverse_av;
struct efa_av_entry;

/*
 * What the injected reader observed at the moment the promotion was about to
 * publish the explicit AV entry to the reverse AV. See
 * efa_test_av_publish_ordering_probe.
 */
struct efa_test_av_publish_observation {
	/* the probe ran */
	bool ran;
	/* explicit fi_addr the entry is about to be published under */
	fi_addr_t explicit_fi_addr;
	/* lookup 1, (GID, QPN) -> fi_addr, must still miss here */
	bool reverse_av_resolves;
	/* lookup 2, fi_addr -> peer, must already hit here */
	bool peer_map_resolves;
	/* the peer lookup 2 found is the one that existed before the promotion */
	bool peer_is_migrated_peer;
};

/**
 * @brief Put a fabricated peer in the implicit AV and give it a peer struct
 *
 * Leaves the endpoint's implicit peer map holding a peer for the new implicit
 * fi_addr, which is the state a promotion has to re-key. The fabricated address
 * is kept for efa_test_av_publish_ordering_promote.
 *
 * @param[in]	ep	endpoint
 * @param[in]	av	address vector
 * @return	0 on success, or a negative libfabric error code
 */
int efa_test_av_publish_ordering_setup(struct fid_ep *ep, struct fid_av *av);

/**
 * @brief fi_av_insert the address from the setup, promoting it to the explicit AV
 *
 * @param[in]	av	address vector
 * @param[out]	explicit_fi_addr	fi_addr the address was promoted to
 * @return	the fi_av_insert return value, so 1 on success
 */
int efa_test_av_publish_ordering_promote(struct fid_av *av,
					fi_addr_t *explicit_fi_addr);

/**
 * @brief Run the CQ read path's lookups from inside the publish sequence
 *
 * Call this from a wrapped efa_rdm_av_reverse_av_add, before __real_, so it runs
 * at the exact point a concurrent CQ read could first observe the entry through
 * the lock-free reverse AV. Records what it saw in the observation the test
 * reads back; does not assert, so a violation is reported by the test rather
 * than aborting the suite.
 */
void efa_test_av_publish_ordering_probe(struct efa_av_entry *entry);

/**
 * @brief Read back what the last probe observed
 */
const struct efa_test_av_publish_observation *
efa_test_av_publish_ordering_observation(void);

/**
 * @brief Insert and remove a throwaway address, freeing its util AV entry
 *
 * The util AV entry pool is indexed and does not zero recycled buffers, so the
 * next explicit insert reuses this fi_addr's buffer with the removed entry's
 * bytes still in it. Lets a test run the publish ordering checks against a
 * recycled entry rather than a fresh one.
 *
 * @param[in]	ep	endpoint, for a GID the real ibv_create_ah accepts
 * @param[in]	av	address vector
 * @param[out]	recycled_fi_addr	fi_addr whose buffer was freed
 * @return	0 on success, or a negative libfabric error code
 */
int efa_test_av_publish_ordering_recycle_slot(struct fid_ep *ep,
					      struct fid_av *av,
					      fi_addr_t *recycled_fi_addr);

/**
 * @brief Whether exactly one peer exists for the promoted address
 *
 * True when the implicit peer map slot has been vacated and the explicit slot
 * holds the same peer the implicit slot held before the promotion, which is the
 * observable state a duplicate-peer bug breaks.
 */
bool efa_test_av_publish_ordering_single_peer(fi_addr_t explicit_fi_addr);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_RDM_AV_UTILS_H */
