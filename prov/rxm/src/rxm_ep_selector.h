/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2026 Intel Corporation, Inc. All rights reserved. */

#ifndef RXM_EP_SELECTOR_H
#define RXM_EP_SELECTOR_H

#include <stdint.h>

#include <ofi_indexer.h>

struct rxm_conn;
struct rxm_pkt;

/* Maps a TX operation to an index into rxm_conn::msg_eps. select() receives
 * the rxm_pkt of framed sends, or NULL for the RMA and rendezvous RMA paths,
 * which carry no rxm header. destroy() is optional and lets a stateful
 * selector free itself.
 */
struct rxm_ep_selector {
	uint8_t (*select)(struct rxm_conn *conn, const struct rxm_pkt *pkt);
	void (*destroy)(struct rxm_ep_selector *sel);
};

struct rxm_rr_selector {
	struct rxm_ep_selector base;
	/* Next ep index to hand out, in [1, num_msg_eps - 1]. */
	uint8_t rr_next;
	/* msg_id -> (ep_idx + 1) stored as void *. The +1 distinguishes a
	 * pin to ep 0 from an absent entry. */
	struct index_map sar_pins;
};

extern const struct rxm_ep_selector rxm_selector_single_ep;

struct rxm_ep_selector *rxm_rr_selector_alloc(void);

#endif /* RXM_EP_SELECTOR_H */
