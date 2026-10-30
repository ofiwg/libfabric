/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_gtest_info_utils.h"

int efa_test_set_queue_size_env(size_t tx, size_t rx, size_t *prev_tx,
				size_t *prev_rx)
{
	if (prev_tx)
		*prev_tx = efa_env.tx_size;
	if (prev_rx)
		*prev_rx = efa_env.rx_size;

	efa_env.tx_size = tx;
	efa_env.rx_size = rx;

	efa_util_prov_finalize();
	return efa_util_prov_initialize();
}
