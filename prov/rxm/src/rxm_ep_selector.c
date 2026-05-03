#include "rxm.h"
#include "rxm_ep_selector.h"

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
