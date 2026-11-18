/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_XPU_ADDR_UTILS_H
#define EFA_GTEST_XPU_ADDR_UTILS_H

#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Insert a peer that has @ep's address but @qpn and @qkey of its own,
 * so a lookup that reports them can only have read them from the AV entry.
 */
int efa_test_xpu_av_insert_peer(struct fid_ep *ep, struct fid_av *av,
				uint16_t qpn, uint32_t qkey, fi_addr_t *addr);

/** @brief The address handle number of an AV entry, or -1 if it has none. */
int efa_test_xpu_av_ahn(struct fid_av *av, fi_addr_t fi_addr);

/** @brief The local key of an MR. */
uint32_t efa_test_xpu_mr_lkey(struct fid_mr *mr);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_XPU_ADDR_UTILS_H */
