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
	uint32_t rr_counter;
	/* msg_id -> (ep_idx + 1) encoded as void*; entry present means
	 * the SAR message is pinned to that ep. Absent (NULL) means no
	 * pin yet. +1 encoding distinguishes "pinned to ep 0" from
	 * "not present". */
	struct index_map sar_pins;
};

extern const struct rxm_ep_selector rxm_selector_single_ep;

struct rxm_ep_selector *rxm_rr_selector_alloc(void);

#endif /* RXM_EP_SELECTOR_H */
