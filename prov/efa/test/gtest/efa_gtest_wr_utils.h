/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

/* C-linkage bridge for the flush (efa_wr.c) tests.
 * See efa_gtest_common_helpers.h for why this exists. */

#ifndef EFA_GTEST_WR_UTILS_H
#define EFA_GTEST_WR_UTILS_H

#include <stddef.h>
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Number of receive work requests staged on the endpoint but not yet
 * handed to the device, i.e. base_ep->recv_wr_index.
 */
size_t efa_test_ep_recv_wr_index(struct fid_ep *ep_fid);

/**
 * @brief Whether the endpoint holds transmit work that has been built but not
 * initiated. On the direct data path that is a non-empty send queue batch
 * (sq.num_wqe_pending); otherwise it is an open ibv_wr_start block
 * (base_ep->is_wr_started).
 */
int efa_test_ep_tx_wr_pending(struct fid_ep *ep_fid);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_WR_UTILS_H */
