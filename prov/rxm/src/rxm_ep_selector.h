/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2026 Intel Corporation, Inc. All rights reserved. */

#ifndef RXM_EP_SELECTOR_H
#define RXM_EP_SELECTOR_H

#include <stdint.h>

struct rxm_conn;
struct rxm_pkt;

/* Maps a TX operation to an index into rxm_conn::msg_eps. select() receives
 * the rxm_pkt of framed sends, or NULL for the RMA and rendezvous RMA paths,
 * which carry no rxm header.
 */
struct rxm_ep_selector {
	uint8_t (*select)(struct rxm_conn *conn, const struct rxm_pkt *pkt);
};

extern const struct rxm_ep_selector rxm_selector_single_ep;

#endif /* RXM_EP_SELECTOR_H */
