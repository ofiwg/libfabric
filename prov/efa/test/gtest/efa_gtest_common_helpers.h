/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/*
 * C-linkage helper functions for gtest.
 *
 * EFA internal headers (efa.h, efa_av.h, etc.) cannot be included from C++
 * because they transitively pull in unix/osd.h which uses C _Complex types.
 * Functions in this file are implemented in efa_gtest_common_helpers.c
 * (compiled as C) and provide a C++-callable interface to EFA internals that
 * the test files need but cannot access directly.
 */

#ifndef EFA_GTEST_COMMON_HELPERS_H
#define EFA_GTEST_COMMON_HELPERS_H

#include <stdbool.h>
#include <stdint.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Whether the real selected EFA device advertises both RDMA read and
 * write (the condition under which the efa-direct provider advertises FI_RMA).
 * The RMA tests require this: fi_enable creates a real QP with RDMA send-ops
 * flags that a non-RDMA device rejects with -FI_EOPNOTSUPP, so tests query this
 * and GTEST_SKIP on hosts that lack it.
 */
bool efa_test_device_supports_rma(void);

/**
 * @brief Set efa_env.track_mr and return its previous value. Toggling track_mr
 * before EP construction controls whether the direct-ope tracking pool (and
 * thus the efa_direct_ope_alloc/release branch in the post paths) is created.
 */
int efa_test_set_track_mr(int value);

/**
 * @brief Insert a fabricated peer via fi_av_insert so efa_av_addr_to_conn
 * returns a valid conn (with ah + ep_addr) for the RMA post paths.
 * @return the fi_addr on success, or FI_ADDR_NOTAVAIL on failure.
 */
fi_addr_t efa_test_rma_insert_peer(struct fid_ep *ep, struct fid_av *av);

/**
 * @brief Read the domain's zero-byte bounce buffer address and lkey for the
 * endpoint behind @p ep. The 0-byte RMA post paths build their single SGE from
 * these, so a test can assert the SGE was wired to the bounce buffer rather
 * than to caller memory.
 */
void efa_test_get_zero_byte_bounce_buf(struct fid_ep *ep, uint64_t *addr,
				       uint32_t *lkey);

/**
 * @brief Fabricate a unique address and insert it via fi_av_insert (explicit
 * path).
 * @return Number of addresses successfully inserted (0 on expected failure).
 */
int efa_test_explicit_av_insert(struct fid_ep *ep, struct fid_av *av,
				fi_addr_t *addr);

/**
 * @brief Insert a peer with a fabricated GID that won't be in the ah_map.
 * This forces efa_ah_alloc to call ibv_create_ah for a new GID.
 * @return The fi_addr, or FI_ADDR_NOTAVAIL on failure.
 */
fi_addr_t efa_test_insert_peer_new_gid(struct fid_ep *ep, struct fid_av *av);

struct ibv_ah;

/**
 * @brief Resolve an implicit-AV fi_addr to the ibv_ah backing its conn.
 * Lets a test assert which underlying AH a peer insert ended up using
 * (e.g. that an ENOMEM retry kept the newly created AH).
 */
struct ibv_ah *efa_test_implicit_addr_to_ibv_ah(struct fid_av *av,
						fi_addr_t fi_addr);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_COMMON_HELPERS_H */
