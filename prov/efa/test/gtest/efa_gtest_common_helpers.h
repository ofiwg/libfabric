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
 * @brief Whether the real selected EFA device advertises both RDMA read and write.
 */
int efa_test_device_supports_rma(void);

/**
 * @brief Set efa_env.track_mr and return its previous value.
 */
int efa_test_set_track_mr(int value);

/**
 * @brief Read the domain's zero-byte bounce buffer address and lkey for the
 * endpoint behind @p ep.
 */
void efa_test_get_zero_byte_bounce_buf(struct fid_ep *ep, uint64_t *addr,
				       uint32_t *lkey);

/**
 * @brief Insert a peer with a fabricated GID via fi_av_insert.
 */
int efa_test_av_insert_fake_gid(struct fid_ep *ep, struct fid_av *av,
				fi_addr_t *addr);

/**
 * @brief Insert the ep's own (self) GID as a peer via fi_av_insert.
 */
int efa_test_av_insert_self(struct fid_ep *ep, struct fid_av *av,
			    fi_addr_t *addr);

/**
 * @brief Insert a fabricated-GID peer via the RDM-internal efa_av_insert_one.
 */
fi_addr_t efa_test_av_insert_new_ah(struct fid_ep *ep, struct fid_av *av);

struct efa_ibv_cq;
struct efa_context;

/**
 * @brief Allocate an efa_context
 */
struct efa_context *efa_test_alloc_context(uint64_t completion_flags,
					   fi_addr_t addr);

/**
 * @brief Get the efa_ibv_cq embedded in a fid_cq (opaque from C++).
 */
struct efa_ibv_cq *efa_test_get_ibv_cq(struct fid_cq *cq_fid);

/**
 * @brief Get a CQ's internal err_buf
 */
char *efa_test_get_cq_err_buf(struct fid_cq *cq_fid);

/**
 * @brief Capacity of the CQ's internal err_buf (EFA_ERROR_MSG_BUFFER_LENGTH),
 * the upper bound on err_data_size the fallback error path can report.
 */
extern const size_t efa_test_cq_err_buf_len;

/**
 * @brief Get the QP number of the endpoint's underlying EFA QP. Works for the
 * bare efa_base_ep endpoint used by the efa-direct fabric.
 */
uint32_t efa_test_get_qp_num(struct fid_ep *ep);

/**
 * @brief Set status and wr_id on the ibv_cq_ex inside efa_ibv_cq, to simulate a
 * polled completion.
 */
void efa_test_set_ibv_cq_ex(struct efa_ibv_cq *ibv_cq, int status,
			    uint64_t wr_id);

struct ibv_ah;

/**
 * @brief Resolve an implicit-AV fi_addr to the underlying ibv_ah.
 */
struct ibv_ah *efa_test_implicit_addr_to_ibv_ah(struct fid_av *av,
						fi_addr_t fi_addr);

size_t efa_test_ope_list_count(struct fid_ep *ep);

/**
 * @brief Get/set an RDM domain's shm_domain.
 */
struct fid_domain *efa_test_get_shm_domain(struct fid_domain *domain);
void efa_test_set_shm_domain(struct fid_domain *domain,
			     struct fid_domain *shm_domain);

/**
 * @brief Get util_domain's refcount.
 */
int efa_test_get_util_domain_ref(struct fid_domain *domain);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_COMMON_HELPERS_H */
