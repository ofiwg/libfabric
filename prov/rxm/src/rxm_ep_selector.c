/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2026 Intel Corporation, Inc. All rights reserved. */

#include "rxm.h"

static uint8_t rxm_single_ep_select(struct rxm_conn *conn,
				    const struct rxm_pkt *pkt)
{
	OFI_UNUSED(conn);
	OFI_UNUSED(pkt);
	return 0;
}

const struct rxm_ep_selector rxm_selector_single_ep = {
	.select = rxm_single_ep_select,
};
