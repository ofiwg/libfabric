#ifndef RXM_EP_SELECTOR_H
#define RXM_EP_SELECTOR_H

#include <stdint.h>

#include <ofi_indexer.h>

struct rxm_conn;
struct rxm_pkt;

struct rxm_ep_selector {
	uint8_t (*select)(struct rxm_conn *conn, const struct rxm_pkt *pkt);
	void (*destroy)(struct rxm_ep_selector *sel);
};

struct rxm_rr_selector {
	struct rxm_ep_selector base;
	/* Next RR ep index to try, in [1, num_msg_eps - 1]. Advanced with
	 * compare-and-wrap instead of a modulo to avoid a runtime integer
	 * divide in the hot path. Clamped on use so it self-heals if
	 * num_msg_eps shrinks (e.g. an ep connection fails). */
	uint8_t rr_next;
	/* msg_id -> (ep_idx + 1) encoded as void*; entry present means
	 * the SAR message is pinned to that ep. Absent (NULL) means no
	 * pin yet. +1 encoding distinguishes "pinned to ep 0" from
	 * "not present". */
	struct index_map sar_pins;
};

extern const struct rxm_ep_selector rxm_selector_single_ep;

struct rxm_ep_selector *rxm_rr_selector_alloc(void);

#endif /* RXM_EP_SELECTOR_H */
