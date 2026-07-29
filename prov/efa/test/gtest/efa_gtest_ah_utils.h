/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_AH_UTILS_H
#define EFA_GTEST_AH_UTILS_H

#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct efa_ah;

/**
 * @brief Call efa_ah_alloc with a fabricated GID that's not in the ah_map
 */
struct efa_ah *efa_test_ah_alloc_fabricated_gid(struct fid_ep *ep);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_AH_UTILS_H */
