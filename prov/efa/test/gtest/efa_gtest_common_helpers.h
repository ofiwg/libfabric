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

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

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

/**
 * @brief Verify efa_rdm_ope_process_queued_ope() is a no-op when no
 * EFA_RDM_OPE_QUEUED_* flag is set on the ope.
 *
 * Exercises the flag-derivation early return: with no queued bit set the
 * function must return 0 without dispatching, clearing flags, or touching
 * the domain's ope_queued_list.
 *
 * @return 0 if the expected behavior holds, a negative code otherwise.
 */
int efa_test_ope_process_queued_no_flag(struct fid_ep *ep, struct fid_av *av);

/**
 * @brief Verify efa_rdm_ope_process_queued_ope() derives the
 * EFA_RDM_OPE_QUEUED_BEFORE_HANDSHAKE flag from the ope's internal_flags
 * and dispatches to the before-handshake repost path.
 *
 * The peer is left without EFA_RDM_PEER_HANDSHAKE_RECEIVED so the repost
 * returns -FI_EAGAIN before any device interaction. On EAGAIN the function
 * must leave the flag set, keep the ope on ope_queued_list, and not
 * decrement ope_queued_before_handshake_cnt.
 *
 * @return 0 if the expected behavior holds, a negative code otherwise.
 */
int efa_test_ope_process_queued_before_handshake_eagain(struct fid_ep *ep,
							 struct fid_av *av);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_COMMON_HELPERS_H */
