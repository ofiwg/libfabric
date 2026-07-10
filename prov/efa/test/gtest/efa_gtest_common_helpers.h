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
 * @brief Insert the ep's own address as a peer via an (explicit) fi_av_insert.
 * @return Number of addresses inserted (1 on success), or negative on error.
 */
int efa_test_insert_self_peer(struct fid_ep *ep, struct fid_av *av,
			      fi_addr_t *addr);

/**
 * @brief Insert a peer with a fabricated GID that won't be in the ah_map.
 * This forces efa_ah_alloc to call ibv_create_ah for a new GID.
 * @return The fi_addr, or FI_ADDR_NOTAVAIL on failure.
 */
fi_addr_t efa_test_insert_peer_new_gid(struct fid_ep *ep, struct fid_av *av);

/**
 * @brief Insert a peer using the EP's own GID.
 * @return the fi_addr, or FI_ADDR_NOTAVAIL on failure
 */
fi_addr_t efa_test_insert_self_gid_peer(struct fid_ep *ep, struct fid_av *av);

struct ibv_ah;

/**
 * @brief Resolve an implicit-AV fi_addr to the ibv_ah backing its conn.
 * Lets a test assert which underlying AH a peer insert ended up using
 * (e.g. that an ENOMEM retry kept the newly created AH).
 */
struct ibv_ah *efa_test_implicit_addr_to_ibv_ah(struct fid_av *av,
						fi_addr_t fi_addr);

/**
 * @brief Get/set an RDM domain's shm_domain.
 */
struct fid_domain *efa_test_get_shm_domain(struct fid_domain *domain);
void efa_test_set_shm_domain(struct fid_domain *domain,
			     struct fid_domain *shm_domain);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_COMMON_HELPERS_H */
