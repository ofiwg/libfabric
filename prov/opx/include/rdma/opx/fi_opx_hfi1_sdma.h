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
#ifndef _FI_OPX_HFI1_SDMA_H_
#define _FI_OPX_HFI1_SDMA_H_

#include <rdma/hfi/hfi1_user.h>

#include "rdma/opx/fi_opx_hfi1.h"
#include "rdma/opx/fi_opx_reliability.h"
#include "rdma/opx/fi_opx_endpoint.h"
#include "rdma/opx/fi_opx_hfi1_transport.h"

struct fi_opx_hfi1_sdma_header_vec {
	struct sdma_req_info		req_info;	// 8 bytes
	uint64_t			scb_qws[8];	// 64 bytes
}; // Length is FI_OPX_HFI1_SDMA_HDR_SIZE (72 bytes)

struct fi_opx_hfi1_sdma_packet {
	uint64_t				length;
	struct fi_opx_reliability_tx_replay	*replay;
	struct fi_opx_hfi1_sdma_header_vec	header_vec;
	struct hfi1_sdma_comp_entry		comp_entry; // 8 bytes
};

struct fi_opx_hfi1_sdma_work_entry {
	struct fi_opx_hfi1_sdma_work_entry	*next;
	struct fi_opx_completion_counter	*cc;
	union fi_opx_reliability_tx_psn		*psn_ptr;
	ssize_t					writev_rc;
	enum hfi1_sdma_comp_state		comp_state;
	uint32_t				total_payload;
	uint32_t				num_packets;
	uint16_t 				first_comp_index;
	uint16_t				dlid;
	uint8_t 				num_iovs;
	uint8_t					rs;
	uint8_t					rx;
	bool					in_use;
	struct fi_opx_hfi1_sdma_packet		packets[FI_OPX_HFI1_SDMA_MAX_REQUEST_PACKETS];
	struct iovec				iovecs[FI_OPX_HFI1_SDMA_MAX_IOV_LEN];
};

void fi_opx_hfi1_sdma_hit_zero(struct fi_opx_completion_counter *cc);
void fi_opx_hfi1_sdma_handle_errors(struct fi_opx_ep *opx_ep, struct fi_opx_hfi1_sdma_work_entry* we, uint8_t code);

__OPX_FORCE_INLINE__
bool fi_opx_hfi1_sdma_use_sdma(struct fi_opx_ep *opx_ep,
				uint64_t *origin_byte_counter,
				const uint32_t opcode,
				const bool is_intranode)
{
	return !is_intranode &&
		opcode != FI_OPX_HFI_DPUT_OPCODE_ATOMIC_COMPARE_FETCH &&
		opcode != FI_OPX_HFI_DPUT_OPCODE_ATOMIC_FETCH &&
		origin_byte_counter &&
		*origin_byte_counter >= FI_OPX_SDMA_MIN_LENGTH &&
		opx_ep->tx->use_sdma;
}

__OPX_FORCE_INLINE__
void fi_opx_hfi1_sdma_init_cc(struct fi_opx_ep *opx_ep,
			      struct fi_opx_hfi1_dput_params *params)
{
	if (!params->delivery_completion) {
		params->cc = NULL;
		return;
	}
	struct fi_opx_completion_counter *cc = ofi_buf_alloc(opx_ep->rma_counter_pool);
	assert(cc);
	cc->byte_counter = *params->origin_byte_counter;
	cc->cq = NULL;
	cc->work_elem = (void *)params;
	cc->cntr = NULL;
	cc->hit_zero = fi_opx_hfi1_sdma_hit_zero;
	params->cc = cc;
}

__OPX_FORCE_INLINE__
void fi_opx_hfi1_sdma_reset_for_newop(struct fi_opx_hfi1_sdma_work_entry* we)
{
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SDMA_WE -- reset WE %d\n", we->first_comp_index);
	assert(we->comp_state != QUEUED);

	we->comp_state = FREE;
	we->num_iovs = 0;
	we->num_packets = 0;
	we->total_payload = 0;
	we->writev_rc = 0;
	we->psn_ptr = NULL;
}

__OPX_FORCE_INLINE__
bool fi_opx_hfi1_sdma_has_unsent_packets(struct fi_opx_hfi1_sdma_work_entry* we)
{
	return we->num_packets && we->comp_state != QUEUED;
}

__OPX_FORCE_INLINE__
bool fi_opx_hfi1_sdma_can_add_packet(struct fi_opx_hfi1_sdma_work_entry *we,
				     uint64_t payload_bytes)
{
	assert(payload_bytes > 0);
	assert(payload_bytes <= FI_OPX_HFI1_PACKET_MTU);

	return (we->num_packets < FI_OPX_HFI1_SDMA_MAX_REQUEST_PACKETS) &&
		 (we->comp_state == FREE || we->comp_state == COMPLETE);
}

__OPX_FORCE_INLINE__
void fi_opx_hfi1_sdma_poll_completion(struct fi_opx_ep *opx_ep)
{

	struct fi_opx_hfi1_context *hfi = opx_ep->hfi;
	uint16_t queue_size = hfi->info.sdma.queue_size;

	while (hfi->info.sdma.available_counter < queue_size) {
		volatile struct hfi1_sdma_comp_entry * entry = &hfi->info.sdma.completion_queue[hfi->info.sdma.done_index];
		if (entry->status == QUEUED) {
			break;
		}

		// Update the status/errcode of the work entry who was using this index
		assert(hfi->info.sdma.queued_entries[hfi->info.sdma.done_index]);
		hfi->info.sdma.queued_entries[hfi->info.sdma.done_index]->status = entry->status;
		hfi->info.sdma.queued_entries[hfi->info.sdma.done_index]->errcode = entry->errcode;
		hfi->info.sdma.queued_entries[hfi->info.sdma.done_index] = NULL;

		assert(entry->status == COMPLETE || entry->status == FREE);
		++hfi->info.sdma.available_counter;
		hfi->info.sdma.done_index = (hfi->info.sdma.done_index + 1) % (queue_size);
		if (hfi->info.sdma.done_index == hfi->info.sdma.fill_index) {
			assert(hfi->info.sdma.available_counter == queue_size);
		}
	}
}

__OPX_FORCE_INLINE__
enum hfi1_sdma_comp_state fi_opx_hfi1_sdma_get_status(struct fi_opx_ep *opx_ep,
						      struct fi_opx_hfi1_sdma_work_entry *we)
{

	if (we->comp_state == FREE) {
		return FREE;
	}

	if (we->comp_state == ERROR) {
		return ERROR;
	}

	if (we->comp_state == QUEUED) {
		for (int i = 0; i < we->num_packets; ++i) {
			if (OFI_UNLIKELY(we->packets[i].comp_entry.status == ERROR)) {
				FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
					"===================================== SDMA_WE -- Found error in queued entry, status=%d, error=%d\n",
					we->packets[i].comp_entry.status, we->packets[i].comp_entry.errcode);
				fi_opx_hfi1_sdma_handle_errors(opx_ep, we, 0x11);
				we->comp_state = ERROR;
				return ERROR;
			}
			if (we->packets[i].comp_entry.status == QUEUED) {
				// Found one queued item, have to wait
				FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"===================================== SDMA_WE -- Found queued comp_entry at %d\n", i);
				return QUEUED;
			}
			assert(we->packets[i].comp_entry.status == COMPLETE);
			assert(we->packets[i].comp_entry.errcode == 0);
		}
	}

	// If we havn't returned yet, all comp_state entries we care about are either COMPLETE or FREE
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== SDMA_WE -- Detected Comp entry %d finished worked.  Resetting to allow more ops\n",  we->first_comp_index);

	// I don't think we need this, since we're going to set it to free in reset_for_newop anyway.
	// If we can remove this, also need to remove assert != QUEUED in reset function
	we->comp_state = FREE;

	fi_opx_hfi1_sdma_reset_for_newop(we);

	return COMPLETE;
}

__OPX_FORCE_INLINE__
struct fi_opx_hfi1_sdma_work_entry* fi_opx_hfi1_sdma_get_idle_we(struct fi_opx_ep *opx_ep)
{

	struct fi_opx_hfi1_sdma_work_entry *entry =
		(struct fi_opx_hfi1_sdma_work_entry *) ofi_buf_alloc(opx_ep->tx->sdma_work_pool);

	if (!entry) {
		return NULL;
	}

	entry->next = NULL;
	entry->comp_state = FREE;
	entry->num_iovs = 0;
	entry->num_packets = 0;
	entry->total_payload = 0;
	entry->writev_rc = 0;
	entry->psn_ptr = NULL;
	entry->in_use = true;

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SDMA_WE -- giving WE %d\n", entry->first_comp_index);
	return entry;
}

__OPX_FORCE_INLINE__
void fi_opx_hfi1_sdma_return_we(struct fi_opx_ep *opx_ep, struct fi_opx_hfi1_sdma_work_entry* we) {
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SDMA_WE -- returned WE %d\n", we->first_comp_index);
	assert(we->next == NULL);
	assert(we->in_use);
	we->in_use = false;

	ofi_buf_free(we);
}

__OPX_FORCE_INLINE_AND_FLATTEN__
struct fi_opx_hfi1_sdma_work_entry *opx_sdma_get_new_work_entry(struct fi_opx_ep *opx_ep,
								uint16_t *reqs_used,
								struct slist *sdma_reqs,
								struct fi_opx_hfi1_sdma_work_entry *current)
{
	// Get a new SDMA work entry. First try to get an idle one if
	// we're not already using too many.
	struct fi_opx_hfi1_sdma_work_entry *sdma_we;
	if ((*reqs_used) < 2) {
		sdma_we = fi_opx_hfi1_sdma_get_idle_we(opx_ep);
		if (sdma_we) {
			++(*reqs_used);
			assert(sdma_we->next == NULL);
			return sdma_we;
		}
	}

	// No idle entries available, or we've already been allocated the max.
	// See if one of our existing entries is available for re-use.
	sdma_we = (struct fi_opx_hfi1_sdma_work_entry *) sdma_reqs->head;

	if (sdma_we && sdma_we != current) {
		enum hfi1_sdma_comp_state sdma_status = fi_opx_hfi1_sdma_get_status(opx_ep, sdma_we);
		if (sdma_status == COMPLETE || sdma_status == FREE) {
			slist_remove_head(sdma_reqs);
			sdma_we->next = NULL;
			return sdma_we;
		}
	}
	return NULL;
}

__OPX_FORCE_INLINE__
bool fi_opx_hfi1_sdma_add_packet(struct fi_opx_hfi1_sdma_work_entry *we,
				struct fi_opx_reliability_tx_replay *replay,
				struct fi_opx_completion_counter *cc,
				void *buf,
				uint64_t payload_bytes,
				uint16_t dlid, uint8_t rs, uint8_t rx,
				bool delivery_completion,
				uint16_t frag_size)
{
	assert(payload_bytes <= FI_OPX_HFI1_PACKET_MTU);

#ifndef NDEBUG
	if (we->num_packets >= FI_OPX_HFI1_SDMA_MAX_REQUEST_PACKETS) {
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== SDMA_WE -- Can't add_packet, num_packets=%d,  payload_bytes=%ld\n", we->num_packets, payload_bytes);
		assert(0); //Why didn't they call can_add_packet before calling this?
		return false;
	}
#endif

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SDMA_WE -- Add_packet, payload_bytes=%ld\n", payload_bytes);
	
	we->packets[we->num_packets].header_vec.req_info.ctrl = FI_OPX_HFI1_SDMA_REQ_HEADER_FIXEDBITS;
	we->packets[we->num_packets].header_vec.req_info.fragsize = frag_size;
	we->packets[we->num_packets].header_vec.req_info.npkts = 1;
	we->packets[we->num_packets].header_vec.scb_qws[0] = replay->scb.qw0;  //PBC_dws
	we->packets[we->num_packets].header_vec.scb_qws[1] = replay->scb.hdr.qw[0];
	we->packets[we->num_packets].header_vec.scb_qws[2] = replay->scb.hdr.qw[1];
	we->packets[we->num_packets].header_vec.scb_qws[3] = replay->scb.hdr.qw[2];
	we->packets[we->num_packets].header_vec.scb_qws[4] = replay->scb.hdr.qw[3];
	we->packets[we->num_packets].header_vec.scb_qws[5] = replay->scb.hdr.qw[4];
	we->packets[we->num_packets].header_vec.scb_qws[6] = replay->scb.hdr.qw[5];
	we->packets[we->num_packets].header_vec.scb_qws[7] = replay->scb.hdr.qw[6];
	we->packets[we->num_packets].replay = replay;
	we->packets[we->num_packets].length = payload_bytes;
	we->packets[we->num_packets].comp_entry.status = QUEUED;
	we->packets[we->num_packets].comp_entry.errcode = 0;

	we->cc = cc;
	we->dlid = dlid;
	we->rs = rs;
	we->rx = rx;

	we->iovecs[we->num_iovs].iov_len = FI_OPX_HFI1_SDMA_HDR_SIZE;
	we->iovecs[we->num_iovs].iov_base = &we->packets[we->num_packets].header_vec;

	we->num_iovs++;
	if (delivery_completion) {
		we->iovecs[we->num_iovs].iov_len = replay->iov[0].iov_len;
		we->iovecs[we->num_iovs].iov_base = replay->iov[0].iov_base;
	} else {
		we->iovecs[we->num_iovs].iov_len = payload_bytes;
		we->iovecs[we->num_iovs].iov_base = buf;
	}
	we->num_iovs++;
	we->num_packets++;

	we->total_payload += payload_bytes;
	assert(we->num_iovs <= FI_OPX_HFI1_SDMA_MAX_IOV_LEN);
	assert(we->num_packets <= FI_OPX_HFI1_SDMA_MAX_REQUEST_PACKETS);

	return true;
}

__OPX_FORCE_INLINE__
void fi_opx_hfi1_sdma_do_sdma(struct fi_opx_ep *opx_ep,
				struct fi_opx_hfi1_sdma_work_entry *we,
				bool delivery_completion,
				enum ofi_reliability_kind reliability)
{
	assert(we->comp_state == FREE);
	assert(we->num_packets > 0);
	assert(opx_ep->hfi->info.sdma.available_counter >= we->num_packets);

	int64_t psn;

	psn = fi_opx_reliability_tx_next_psn(&opx_ep->ep_fid,
					&opx_ep->reliability->state,
					we->dlid,
					we->rx,
					we->rs,
					&we->psn_ptr,
					we->num_packets);
	if (psn == -1) {
		fprintf(stderr, "(%d) Failed to get %u PSNs!\n", getpid(), we->num_packets);
		assert(psn != -1);
	}

	/* Assign the queue indexes and PSNs for each packet, and then
	   register the replay with reliability */
	uint16_t *fill_index = &opx_ep->hfi->info.sdma.fill_index;
	uint16_t *avail = &opx_ep->hfi->info.sdma.available_counter;
	assert((*avail) >= we->num_packets);
	we->first_comp_index = *fill_index;
	for (int i = 0; i < we->num_packets; ++i) {
		assert((((*avail) < opx_ep->hfi->info.sdma.queue_size) &&
			(*fill_index) != opx_ep->hfi->info.sdma.done_index) ||
		       (((*avail) == opx_ep->hfi->info.sdma.queue_size) &&
			(*fill_index) == opx_ep->hfi->info.sdma.done_index));

		assert(opx_ep->hfi->info.sdma.completion_queue[*fill_index].status == FREE ||
		       opx_ep->hfi->info.sdma.completion_queue[*fill_index].status == COMPLETE);

		we->packets[i].header_vec.req_info.comp_idx = *fill_index;
		assert(opx_ep->hfi->info.sdma.queued_entries[*fill_index] == NULL);
		opx_ep->hfi->info.sdma.queued_entries[*fill_index] = &we->packets[i].comp_entry;
		*fill_index = ((*fill_index) + 1) % (opx_ep->hfi->info.sdma.queue_size);
		--(*avail);

		we->packets[i].header_vec.scb_qws[3] |= psn;
		we->packets[i].replay->scb.hdr.qw[2] |= psn;

		// TODO: Make two versions of this for-loop, one with register no update and one with register with update,
		//       and determine which one to use by delivery completion
		if (delivery_completion) {
			fi_opx_reliability_client_replay_register_with_update(
						&opx_ep->reliability->state, we->dlid,
						we->rs, we->rx, we->psn_ptr,
						we->packets[i].replay, we->cc,
						we->packets[i].length,
						reliability);
		} else {
			fi_opx_reliability_client_replay_register_no_update(
						&opx_ep->reliability->state, we->dlid,
						we->rs, we->rx, we->psn_ptr,
						we->packets[i].replay,
						reliability);
		}
		psn = (psn + 1) & MAX_PSN;
	}

	//rc will be the number of packets being processed concurrently by SDMA, or -1 if error
	ssize_t rc = writev(opx_ep->hfi->fd, we->iovecs, we->num_iovs);
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SDMA_WE -- called writev rc=%ld  Params were: fd=%d iovecs=%p num_iovs=%d \n", rc, opx_ep->hfi->fd, we->iovecs, we->num_iovs);

	we->writev_rc = rc;
	if (rc > 0) {
		assert (rc == we->num_packets);
		we->comp_state = QUEUED;
	} else {
		we->comp_state = ERROR;
		fi_opx_hfi1_sdma_handle_errors(opx_ep, we, 0x22);
	}
}

__OPX_FORCE_INLINE__
void fi_opx_hfi1_sdma_flush(struct fi_opx_ep *opx_ep,
			    struct fi_opx_hfi1_sdma_work_entry *we,
			    struct slist *sdma_reqs,
			    uint64_t *origin_byte_counter,
			    bool delivery_completion,
			    enum ofi_reliability_kind reliability)
{
	fi_opx_hfi1_sdma_do_sdma(opx_ep, we,
				delivery_completion,
				reliability);

	if (!delivery_completion) {
		*origin_byte_counter -= we->total_payload;
	}

	assert(we->next == NULL);
	slist_insert_tail((struct slist_entry *)we, sdma_reqs);
}

__OPX_FORCE_INLINE__
void fi_opx_hfi1_sdma_finish(struct fi_opx_hfi1_dput_params *params)
{
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SDMA FINISH Work Item %p -- (begin)\n",
		params);
	if (params->sdma_we) {
		fi_opx_hfi1_sdma_return_we(params->opx_ep, params->sdma_we);
		params->sdma_we = NULL;
	}

	struct fi_opx_hfi1_sdma_work_entry *sdma_we = 
		(struct fi_opx_hfi1_sdma_work_entry *) slist_remove_head(&params->sdma_reqs);

	// Return the inactive SDMA WEs we were using
	while (sdma_we) {
		sdma_we->next = NULL;
		fi_opx_hfi1_sdma_return_we(params->opx_ep, sdma_we);
		sdma_we = (struct fi_opx_hfi1_sdma_work_entry *) slist_remove_head(&params->sdma_reqs);
	}

	assert(slist_empty(&params->sdma_reqs));
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SDMA FINISH Work Item %p -- (end)\n",
		params);
}

#endif /* _FI_OPX_HFI1_SDMA_H_ */
