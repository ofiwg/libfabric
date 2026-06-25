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
 * @brief Fabricate a unique efa_ep_addr with a GID guaranteed not to be in
 * ah_map. Derives from the endpoint's real GID with a flipped byte and
 * incrementing suffix.
 *
 * @param[in]	ep	endpoint to get the base GID from
 * @param[out]	addr	populated with the fabricated address
 */
void efa_test_fabricate_addr(struct fid_ep *ep, struct efa_ep_addr *addr);

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

struct efa_ibv_cq;

/*
 * Layout-compatible copy of struct efa_context (from efa.h).
 * Guarded to avoid redefinition in .c files that include efa.h directly.
 */
#ifndef EFA_H
struct efa_context {
	uint64_t completion_flags;
	fi_addr_t addr;
};
#endif

/**
 * @brief Get the efa_ibv_cq embedded in a fid_cq.
 * Navigates: fid_cq → util_cq → efa_cq → efa_ibv_cq.
 */
struct efa_ibv_cq *efa_test_get_ibv_cq(struct fid_cq *cq_fid);

/**
 * @brief Get the QP number of the endpoint's underlying EFA QP.
 */
uint32_t efa_test_get_qp_num(struct fid_ep *ep);

/**
 * @brief Ensure the efa_cq's err_buf is allocated.
 * efa_rdm_cq_open does not allocate err_buf (only efa_cq_open does),
 * but efa_cq_poll_ibv_cq's error path requires it.
 */
void efa_test_alloc_err_buf(struct efa_ibv_cq *ibv_cq);
void efa_test_free_err_buf(struct efa_ibv_cq *ibv_cq);

/**
 * @brief Set the status and wr_id fields on the ibv_cq_ex inside efa_ibv_cq.
 * Needed because efa_ibv_cq is opaque from C++.
 */
void efa_test_set_ibv_cq_ex(struct efa_ibv_cq *ibv_cq, int status,
			    uint64_t wr_id);

int efa_cq_poll_ibv_cq(ssize_t cqe_to_process, struct efa_ibv_cq *ibv_cq);

/**
 * @brief Allocate a struct efa_direct_ope (opaque from C++) wrapping the given
 * efa_context. Used by track_mr tests that set wr_id to a direct_ope pointer.
 * The returned pointer is the value efa_direct_ope_release is expected to
 * receive. Free with efa_test_free_direct_ope.
 */
void *efa_test_alloc_direct_ope(struct efa_context *ctx);
void efa_test_free_direct_ope(void *direct_ope);

/**
 * @brief Get/set efa_env.track_mr. efa_env.h is not includable from C++ (pulls
 * in _Complex types via unix/osd.h), so tests toggle the global through these.
 */
int efa_test_get_track_mr(void);
void efa_test_set_track_mr(int val);
/**
 * @brief Read back one already-staged completion from the CQ's util_cq without
 * progressing. Wraps ofi_cq_read_entries (not fi_cq_read), so it does NOT
 * re-enter the mocked efa_cq poll path. Lets a success-path test prove a
 * completion was actually written (op_context/flags/len/data) rather than only
 * asserting the poll return value. Returns 1 on one entry, -FI_EAGAIN if none.
 */
ssize_t efa_test_cq_read_staged_data_entry(struct fid_cq *cq_fid,
					   struct fi_cq_data_entry *entry);

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
