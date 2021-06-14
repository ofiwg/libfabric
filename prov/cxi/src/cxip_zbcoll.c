/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2021 Hewlett Packard Enterprise Development LP
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <endian.h>
#include <sys/time.h>
#include <sys/types.h>

#include <ofi_list.h>
#include <ofi.h>
#include <fenv.h>
#include <xmmintrin.h>

#include "cxip.h"

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_CTRL, __VA_ARGS__)

/**
 * @brief Send callback to manage ACK.
 *
 * The request must be retried, or freed, in this function.
 * Caller does not handle error returns gracefully, handle them here.
 *
 * @param req   : original request
 * @param event : CXI driver event
 * @return int FI_SUCCESS
 */
static int zbdata_send_cb(struct cxip_ctrl_req *req, const union c_event *event)
{
	int ret;

	switch (event->hdr.event_type) {
	case C_EVENT_ACK:
		switch (cxi_event_rc(event)) {
		case C_RC_OK:
			free(req);
			break;
		case C_RC_ENTRY_NOT_FOUND:
			/* really bad match criteria */
		case C_RC_PTLTE_NOT_FOUND:
			/* PTLTE not set up yet */
		default:
			CXIP_DBG("%s: failed connection, retry\n", __func__);
			do {
				usleep(10000);	// TODO: parametrize
				ret = cxip_ctrl_msg_send(req);
			} while (ret == -FI_EAGAIN);
			if (ret != FI_SUCCESS) {
				CXIP_WARN("log retry failure\n");
				free(req);
			}
			break;
		}
		break;
	default:
		CXIP_WARN("%s: Unexpected event type: %s\n",
		       	 __func__, cxi_event_to_str(event));
		free(req);
	}
	return FI_SUCCESS;
}

/**
 * @brief Send a zero-buffer collective packet.
 *
 * Creates a request packet that must be freed (or retried) in callback.
 *
 * @param zbcoll : collective state structure
 * @param dstnid : destination address
 * @param mb     : packet to send
 * @return int
 */
int cxip_zb_coll_send(struct cxip_zb_coll_state *zbcoll, uint32_t dstnid,
		      union cxip_match_bits mb)
{
	struct cxip_ep_obj *ep_obj;
	struct cxip_ctrl_req *req;
	int ret;

	/* container = container_of(ptr_I_have, cont_type, cont_field) */
	ep_obj = container_of(zbcoll, struct cxip_ep_obj, zb_coll);

	req = calloc(1, sizeof(*req));
	if (! req)
		return -FI_ENOMEM;

	req->ep_obj = ep_obj;
	req->cb = zbdata_send_cb;
	req->send.nic_addr = dstnid;
	req->send.pid = ep_obj->src_addr.pid;
	req->send.mb = mb;
	req->send.mb.ctrl_le_type = CXIP_CTRL_LE_TYPE_CTRL_MSG;
	req->send.mb.ctrl_msg_type = CXIP_CTRL_MSG_ZB_DATA;

	ret = cxip_ctrl_msg_send(req);

	return ret;
}

/**
 * @brief Configure the zero-buffer collective.
 *
 * @param ep       : endpoint
 * @return int     : return code, 0 on success
 */
int cxip_zb_coll_config(struct fid_ep *ep, int num_nids, uint32_t *nids,
			int radix)
{
	struct cxip_ep *cxi_ep;
	struct cxip_zb_coll_state *zbcoll;

	cxi_ep = container_of(ep, struct cxip_ep, ep.fid);
	zbcoll = &cxi_ep->ep_obj->zb_coll;
	memset(zbcoll, 0, sizeof(*zbcoll));
	/* STUB */

	return 0;
}
/**
 * @brief Progress any asynchronous zbcoll request.
 *
 * @param ep       : endpoint
 */
void cxip_zb_coll_progress(struct fid_ep *ep)
{
	struct cxip_ep *cxi_ep __attribute__((unused));

	cxi_ep = container_of(ep, struct cxip_ep, ep.fid);
	cxip_ep_ctrl_progress(cxi_ep->ep_obj);
}

int cxip_zb_coll_init(struct fid_ep *ep)
{
	struct cxip_ep *cxi_ep __attribute__((unused));

	cxi_ep = container_of(ep, struct cxip_ep, ep.fid);
	/* STUB */

	return 0;
}

int cxip_zb_coll_barrier(struct fid_ep *ep)
{
	struct cxip_ep *cxi_ep __attribute__((unused));

	cxi_ep = container_of(ep, struct cxip_ep, ep.fid);
	/* STUB */

	return 0;
}

int cxip_zb_coll_bcast(struct fid_ep *ep, uint64_t data)
{
	struct cxip_ep *cxi_ep __attribute__((unused));

	cxi_ep = container_of(ep, struct cxip_ep, ep.fid);
	/* STUB */

	return 0;
}

/* Target receive zb collective packet */
int cxip_zb_coll_recv(struct cxip_ep_obj *ep_obj, uint32_t init_nic,
		      uint32_t init_pid, const union cxip_match_bits mbv)
{
	struct cxip_zb_coll_state *zbcoll __attribute__((unused));

	/* STUB */
	printf("zbdata_recv_cb OK\n");
	printf("  init_nic = %d\n", init_nic);
	printf("  init_pid = %d\n", init_pid);
	printf("  packet   = %016lx\n", mbv.raw);

	return 0;
}

