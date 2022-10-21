/*
 * Copyright (C) 2022 by Cornelis Networks.
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

#include <assert.h>
#include <string.h>
#include "rdma/opx/fi_opx_hfi1_sdma.h"

void fi_opx_hfi1_sdma_hit_zero(struct fi_opx_completion_counter *cc)
{
	struct fi_opx_hfi1_dput_params *params = (struct fi_opx_hfi1_dput_params *) cc->work_elem;
	if (params->work_elem.complete) {
		FI_WARN(&fi_opx_provider, FI_LOG_EP_DATA,
			"SDMA Work Entry hit zero more than once! cc->byte_counter = %ld\n", cc->byte_counter);
		assert(0);
		return;
	}

	assert(params->sdma_we == NULL || !fi_opx_hfi1_sdma_has_unsent_packets(params->sdma_we));

	if (params->delivery_completion && params->origin_byte_counter) {
		/* Set the sender's byte counter to 0 to notify them that the send is
		   complete. We should assume that the instant we set it to 0, the
		   pointer will become invalid, so NULL it. */
		*params->origin_byte_counter = 0;
		params->origin_byte_counter = NULL;
	}

	if (cc->next) {
		cc->next->byte_counter = 0;
		cc->next->hit_zero(cc->next);
		cc->next = NULL;
	}

	// Set the work element to complete so it can be removed from the work pending queue and freed
	params->work_elem.complete = true;
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "=================== SDMA HIT ZERO\n");
}

void fi_opx_hfi1_sdma_handle_errors(struct fi_opx_ep *opx_ep, struct fi_opx_hfi1_sdma_work_entry* we, uint8_t code)
{
	const pid_t pid = getpid();
	FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "SDMA Error, not handled\n");

	fprintf(stderr, "(%d) ERROR: SDMA Abort code %#0hhX!\n", pid, code);
	fprintf(stderr, "(%d) ===================================== SDMA_WE -- called writev rc=%ld, errno=%d  num_pkts=%u Params were: fd=%d iovecs=%p num_iovs=%d \n",
		pid, we->writev_rc, errno, we->num_packets, opx_ep->hfi->fd, we->iovecs, we->num_iovs);
	fprintf(stderr, "(%d) hfi->info.sdma.queue_size == %0hu\n", pid, opx_ep->hfi->info.sdma.queue_size);
	fprintf(stderr, "(%d) hfi->info.sdma.fill_index == %0hu\n", pid, opx_ep->hfi->info.sdma.fill_index);
	fprintf(stderr, "(%d) hfi->info.sdma.done_index == %0hu\n", pid, opx_ep->hfi->info.sdma.done_index);
	fprintf(stderr, "(%d) hfi->info.sdma.available  == %0hu\n", pid, opx_ep->hfi->info.sdma.available_counter);
	fprintf(stderr, "(%d) hfi->info.sdma.completion_queue == %p\n", pid, opx_ep->hfi->info.sdma.completion_queue);
	volatile struct hfi1_sdma_comp_entry * entry = opx_ep->hfi->info.sdma.completion_queue;

	fprintf(stderr, "(%d) we->header_vec.npkts=%hd, frag_size=%hd, cmp_idx=%hd, ctrl=%#04hX, status=%#0hX, errCode=%#0hX\n",
		pid,
		we->header_vec.req_info.npkts,
		we->header_vec.req_info.fragsize,
		we->header_vec.req_info.comp_idx,
		we->header_vec.req_info.ctrl,
		entry[we->header_vec.req_info.comp_idx].status,
		entry[we->header_vec.req_info.comp_idx].errcode);

	for (int i = 0; i < we->num_iovs; i++) {
		fprintf(stderr, "(%d) we->iovecs[%d].base = %p, len = %lu\n", pid, i, we->iovecs[i].iov_base, we->iovecs[i].iov_len);
		fprintf(stderr, "(%d) First 8 bytes of %p == %#16.16lX\n", pid, we->iovecs[i].iov_base, *((uint64_t *) we->iovecs[i].iov_base));
	}

	fprintf(stderr, "(%d) PBC: %#16.16lX\n", pid, we->header_vec.scb_qws[0]);
#ifndef NDEBUG
	fi_opx_hfi1_dump_packet_hdr((union fi_opx_hfi1_packet_hdr *) &we->header_vec.scb_qws[1], "fi_opx_hfi1_sdma_handle_errors", 92);
#endif

	abort();
}
