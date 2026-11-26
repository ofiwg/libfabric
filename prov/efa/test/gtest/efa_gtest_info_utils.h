/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_INFO_UTILS_H
#define EFA_GTEST_INFO_UTILS_H

#include <stddef.h>
#include <stdint.h>

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

int efa_test_prov_info_link_speed(uint64_t port_speed, size_t *link_speed);

size_t efa_test_device_vrb_link_speed(void);

uint64_t efa_test_device_port_speed(void);

size_t efa_test_device_rdm_info_link_speed(void);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_INFO_UTILS_H */
