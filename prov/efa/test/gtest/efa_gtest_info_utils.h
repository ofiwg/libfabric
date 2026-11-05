/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_INFO_UTILS_H
#define EFA_GTEST_INFO_UTILS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Set FI_EFA_TX_SIZE / FI_EFA_RX_SIZE (efa_env.tx_size / rx_size).
 * Both are read on every fi_getinfo() call, so the next call reflects them.
 *
 * @param[in]	tx		new efa_env.tx_size, 0 for the default
 * @param[in]	rx		new efa_env.rx_size, 0 for the default
 * @param[out]	prev_tx		previous efa_env.tx_size, may be NULL
 * @param[out]	prev_rx		previous efa_env.rx_size, may be NULL
 */
void efa_test_set_queue_size_env(size_t tx, size_t rx, size_t *prev_tx,
				 size_t *prev_rx);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_INFO_UTILS_H */
