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

static uint8_t rxm_rr_next(struct rxm_rr_selector *rr, struct rxm_conn *conn)
{
	uint8_t idx;

	if (OFI_UNLIKELY(conn->num_msg_eps <= 1))
		return 0;

	/* Self-heal if num_msg_eps shrank and left rr_next past the end. */
	if (OFI_UNLIKELY(rr->rr_next >= conn->num_msg_eps))
		rr->rr_next = 1;

	idx = rr->rr_next;
	if (OFI_UNLIKELY(++rr->rr_next >= conn->num_msg_eps))
		rr->rr_next = 1;

	if (OFI_LIKELY(conn->states[idx] == RXM_CM_CONNECTED))
		return idx;

	/* Advancing unconditionally means warm-up walks every slot and fires
	 * a connect on each, bringing the QPs up in parallel. Until a slot is
	 * up its share of the traffic goes to the primary; a failed connect
	 * leaves the slot idle to be retried when the cursor wraps. */
	if (OFI_UNLIKELY(conn->states[idx] == RXM_CM_IDLE))
		(void) rxm_send_connect(conn, idx);

	return 0;
}

static uint8_t rxm_rr_select(struct rxm_conn *conn, const struct rxm_pkt *pkt)
{
	struct rxm_rr_selector *rr =
		container_of(conn->selector, struct rxm_rr_selector, base);
	enum rxm_sar_seg_type seg_type;
	uint64_t msg_id;
	void *slot;
	uint8_t idx;

	if (!pkt)
		return rxm_rr_next(rr, conn);

	if (OFI_LIKELY(pkt->ctrl_hdr.type != rxm_ctrl_seg))
		return 0;

	seg_type = rxm_sar_get_seg_type((struct ofi_ctrl_hdr *) &pkt->ctrl_hdr);
	msg_id = pkt->ctrl_hdr.msg_id;

	switch (seg_type) {
	case RXM_SAR_SEG_MIDDLE:
	case RXM_SAR_SEG_LAST:
		/* Every non-FIRST segment must take the same ep so the receiver
		 * sees them in order. The pin has to outlive LAST: clearing it
		 * there would let a deferred LAST re-pin elsewhere and overtake
		 * an in-flight MIDDLE. FIRST clears it instead, which is safe
		 * because the tx-buf index keying the pin is not reused until
		 * both the first and last sends have completed. */
		slot = ofi_idm_lookup(&rr->sar_pins, (int) msg_id);
		if (slot) {
			idx = (uint8_t) ((uintptr_t) slot - 1);
			if (OFI_LIKELY(idx < conn->num_msg_eps))
				return idx;
			ofi_idm_clear(&rr->sar_pins, (int) msg_id);
		}
		idx = rxm_rr_next(rr, conn);
		/* If the map cannot grow, run the rest of the message on ep 0:
		 * later segments miss the lookup and land there too. */
		if (OFI_UNLIKELY(ofi_idm_set(&rr->sar_pins, (int) msg_id,
					     (void *) (uintptr_t) (idx + 1)) < 0))
			return 0;
		return idx;
	default:
		/* FIRST always goes out on the primary. */
		if (ofi_idm_lookup(&rr->sar_pins, (int) msg_id))
			ofi_idm_clear(&rr->sar_pins, (int) msg_id);
		return 0;
	}
}

static void rxm_rr_destroy(struct rxm_ep_selector *sel)
{
	struct rxm_rr_selector *rr =
		container_of(sel, struct rxm_rr_selector, base);

	/* Values are encoded indices, not pointers, so nothing to free. */
	ofi_idm_reset(&rr->sar_pins, NULL);
	free(rr);
}

struct rxm_ep_selector *rxm_rr_selector_alloc(void)
{
	struct rxm_rr_selector *rr;

	rr = calloc(1, sizeof(*rr));
	if (!rr)
		return NULL;

	rr->base.select = rxm_rr_select;
	rr->base.destroy = rxm_rr_destroy;
	rr->rr_next = 1;
	return &rr->base;
}
