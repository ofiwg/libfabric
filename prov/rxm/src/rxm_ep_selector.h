#ifndef RXM_EP_SELECTOR_H
#define RXM_EP_SELECTOR_H

#include <stdint.h>

struct rxm_conn;
struct rxm_pkt;

struct rxm_ep_selector {
	uint8_t (*select)(struct rxm_conn *conn, const struct rxm_pkt *pkt);
};

extern const struct rxm_ep_selector rxm_selector_single_ep;

#endif /* RXM_EP_SELECTOR_H */
