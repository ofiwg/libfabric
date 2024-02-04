/*
 * SPDX-License-Identifier: BSD-2 Clause or GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>

#include <ofi_list.h>
#include <ofi.h>

#include "cxip.h"

#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_INFO(...) _CXIP_INFO(FI_LOG_EP_CTRL, __VA_ARGS__)

static void cxip_rxc_cs_progress(struct cxip_rxc *rxc)
{
	cxip_evtq_progress(&rxc->rx_evtq);
}

static int cxip_rxc_cs_cancel_msg_recv(struct cxip_req *req)
{
	/* Perform default */
	return cxip_recv_cancel(req);
}

/* Handle any control messaging callbacks specific to protocol */
static int cxip_rxc_cs_ctrl_msg_cb(struct cxip_ctrl_req *req,
				    const union c_event *event)
{
	/* Placeholder */
	return -FI_ENOSYS;
}

static void cxip_rxc_cs_init_struct(struct cxip_rxc *rxc_base,
				    struct cxip_ep_obj *ep_obj)
{
	/* Placeholder */
}

static void cxip_rxc_cs_fini_struct(struct cxip_rxc *rxc)
{
	/* Placeholder */
}

static int cxip_rxc_cs_msg_init(struct cxip_rxc *rxc_base)
{
	/* Placeholder */
	return FI_SUCCESS;
}

static int cxip_rxc_cs_msg_fini(struct cxip_rxc *rxc_base)
{
	/* Placeholder */
	return FI_SUCCESS;
}

static void cxip_rxc_cs_cleanup(struct cxip_rxc *rxc_base)
{
	/* Placeholder */

	/* Cancel Receives */
	cxip_rxc_recv_req_cleanup(rxc_base);
}

/*
 * cxip_recv_common() - Common message receive function. Used for tagged and
 * untagged sends of all sizes.
 */
static ssize_t
cxip_recv_common(struct cxip_rxc *rxc, void *buf, size_t len, void *desc,
		 fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
		 void *context, uint64_t flags, bool tagged,
		 struct cxip_cntr *comp_cntr)
{
	/* Placeholder */
	return -FI_ENOSYS;
}

static void cxip_txc_cs_progress(struct cxip_txc *txc)
{
	/* Placeholder - must process RNR */
}

static int cxip_txc_cs_cancel_msg_send(struct cxip_req *req)
{
	/* Placeholder CS can cancel transmits */
	return -FI_ENOENT;
}

static void cxip_txc_cs_init_struct(struct cxip_txc *txc_base,
				    struct cxip_ep_obj *ep_obj)
{
	/* Placeholder */
}

static void cxip_txc_cs_fini_struct(struct cxip_txc *txc)
{
	/* Placeholder */
}

static int cxip_txc_cs_msg_init(struct cxip_txc *txc_base)
{
	/* Placeholder */
	return FI_SUCCESS;
}

static int cxip_txc_cs_msg_fini(struct cxip_txc *txc_base)
{
	/* Placeholder */
	return FI_SUCCESS;
}

static void cxip_txc_cs_cleanup(struct cxip_txc *txc_base)
{
	/* Placeholder */
}

/*
 * cxip_send_common() - Common message send function. Used for tagged and
 * untagged sends of all sizes. This includes triggered operations.
 */
static ssize_t
cxip_send_common(struct cxip_txc *txc, uint32_t tclass, const void *buf,
		 size_t len, void *desc, uint64_t data, fi_addr_t dest_addr,
		 uint64_t tag, void *context, uint64_t flags, bool tagged,
		 bool triggered, uint64_t trig_thresh,
		 struct cxip_cntr *trig_cntr, struct cxip_cntr *comp_cntr)
{
	/* Placeholder */
	return -FI_ENOSYS;
}

struct cxip_rxc_ops cs_rxc_ops = {
	.recv_common = cxip_recv_common,
	.progress = cxip_rxc_cs_progress,
	.cancel_msg_recv = cxip_rxc_cs_cancel_msg_recv,
	.ctrl_msg_cb = cxip_rxc_cs_ctrl_msg_cb,
	.init_struct = cxip_rxc_cs_init_struct,
	.fini_struct = cxip_rxc_cs_fini_struct,
	.cleanup = cxip_rxc_cs_cleanup,
	.msg_init = cxip_rxc_cs_msg_init,
	.msg_fini = cxip_rxc_cs_msg_fini,
};

struct cxip_txc_ops cs_txc_ops = {
	.send_common = cxip_send_common,
	.progress = cxip_txc_cs_progress,
	.cancel_msg_send = cxip_txc_cs_cancel_msg_send,
	.init_struct = cxip_txc_cs_init_struct,
	.fini_struct = cxip_txc_cs_fini_struct,
	.cleanup = cxip_txc_cs_cleanup,
	.msg_init = cxip_txc_cs_msg_init,
	.msg_fini = cxip_txc_cs_msg_fini,
};
