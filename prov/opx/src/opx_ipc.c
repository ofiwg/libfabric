/*
 * Copyright (C) 2025-2025 by Cornelis Networks.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifdef OPX_HMEM

#include "rdma/opx/opx_ipc.h"
#include "rdma/opx/fi_opx_endpoint.h"

int opx_ipc_send_cts(union fi_opx_hfi1_deferred_work *work)
{
	struct opx_hfi1_rx_ipc_rts_params *params = &work->rx_ipc_rts;
	struct fi_opx_ep		  *opx_ep = params->opx_ep;
	const uint64_t			   bth_rx = ((uint64_t) params->origin_rx) << OPX_BTH_SUBCTXT_RX_SHIFT;

	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "IPC-RECV-SEND-CTS");
	uint64_t pos;
	ssize_t	 rc = fi_opx_shm_dynamic_tx_connect(OPX_SHM_TRUE, opx_ep, params->origin_rx, params->target_hfi_unit);

	if (OFI_UNLIKELY(rc)) {
		OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "IPC-RECV-SEND-CTS");
		return -FI_EAGAIN;
	}

	union opx_hfi1_packet_hdr *const hdr = opx_shm_tx_next(
		&opx_ep->tx->shm, params->target_hfi_unit, params->origin_rx, &pos, opx_ep->daos_info.hfi_rank_enabled,
		params->u32_extended_rx, opx_ep->daos_info.rank_inst, &rc);

	if (!hdr) {
		OPX_TRACER_TRACE(OPX_TRACER_END_ERROR, "IPC-RECV-SEND-CTS");
		return rc;
	}

	/* Note that we do not set stl.hdr.lrh.pktlen here (usually lrh_dws << 32),
	   because this is intranode and since it's a CTS packet, lrh.pktlen
	   isn't used/needed */
	hdr->qw_9B[0] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[0] | params->lrh_dlid;
	hdr->qw_9B[1] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[1] | bth_rx;
	hdr->qw_9B[2] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[2];
	hdr->qw_9B[3] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[3];
	hdr->qw_9B[4] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[4] | (params->niov << 48) | FI_OPX_HFI_DPUT_OPCODE_IPC;
	hdr->qw_9B[5] = params->origin_byte_counter_vaddr;
	hdr->qw_9B[6] = 0;

	opx_shm_tx_advance(&opx_ep->tx->shm, (void *) hdr, pos);
	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "IPC-RECV-SEND-CTS");
	return FI_SUCCESS;
}
#endif
