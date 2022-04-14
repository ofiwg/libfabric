/*
 * Copyright (C) 2016 by Argonne National Laboratory.
 * Copyright (C) 2021 Cornelis Networks.
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
#include <ofi.h>
#include "rdma/opx/fi_opx_atomic.h"
#include "rdma/opx/fi_opx_endpoint.h"
#include "rdma/opx/fi_opx_rma.h"
#include "rdma/opx/fi_opx.h"
#include <ofi_enosys.h>
#include <complex.h>


static inline int fi_opx_check_atomic(struct fi_opx_ep *opx_ep, enum fi_datatype dt, enum fi_op op,
				      size_t count)
{
#ifdef DEBUG
	switch ((int)op) {
	case FI_MIN:
	case FI_MAX:
	case FI_SUM:
	case FI_PROD:
	case FI_LOR:
	case FI_LAND:
	case FI_BOR:
	case FI_BAND:
	case FI_LXOR:
	case FI_ATOMIC_READ:
	case FI_ATOMIC_WRITE:
	case FI_CSWAP:
	case FI_CSWAP_NE:
	case FI_CSWAP_LE:
	case FI_CSWAP_LT:
	case FI_CSWAP_GE:
	case FI_CSWAP_GT:
	case FI_MSWAP:
		break;
	default:
		return -FI_EINVAL;
	}
	if (((int)dt >= FI_DATATYPE_LAST) || ((int)dt < 0))
		return -FI_EINVAL;

	if (!opx_ep)
		return -FI_EINVAL;
	if (opx_ep->state != FI_OPX_EP_ENABLED)
		return -FI_EINVAL;

	if (count == 0)
		return -FI_EINVAL;

	const enum fi_av_type av_type = opx_ep->av->av_type;

	if (av_type == FI_AV_UNSPEC)
		return -FI_EINVAL;
	if (av_type == FI_AV_MAP && opx_ep->av_type != FI_AV_MAP)
		return -FI_EINVAL;
	if (av_type == FI_AV_TABLE && opx_ep->av_type != FI_AV_TABLE)
		return -FI_EINVAL;
#endif
	return 0;
}

__OPX_FORCE_INLINE__
void fi_opx_atomic_fetch_internal(struct fi_opx_ep *opx_ep,
				  const void *buf,
				  const size_t len, const union fi_opx_addr opx_dst_addr,
				  const uint64_t addr_offset,
				  const uint64_t key,
				  const void *fetch_vaddr,
				  union fi_opx_context *opx_context, const uint64_t tx_op_flags,
				  const struct fi_opx_cq *opx_cq,
				  const struct fi_opx_cntr *opx_cntr,
				  struct fi_opx_completion_counter *cc,
				  enum fi_datatype dt, enum fi_op op,
				  const int lock_required, const uint64_t caps,
				  const enum ofi_reliability_kind reliability)
{
	fi_opx_ep_rx_poll(&opx_ep->ep_fid, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME);

	const unsigned is_intranode = fi_opx_rma_dput_is_intranode(caps, opx_dst_addr, opx_ep);
	const uint64_t dest_rx = opx_dst_addr.hfi1_rx;
	const uint64_t lrh_dlid = FI_OPX_ADDR_TO_HFI1_LRH_DLID(opx_dst_addr.fi);
	const uint64_t bth_rx = dest_rx << 56;

	uint8_t *sbuf = (uint8_t *)buf;
	uintptr_t rbuf = addr_offset;
	uint64_t bytes_to_send = len;

	if (tx_op_flags & FI_INJECT) {
		assert((tx_op_flags & (FI_COMPLETION | FI_TRANSMIT_COMPLETE)) !=
		       (FI_COMPLETION | FI_TRANSMIT_COMPLETE));
		assert((tx_op_flags & (FI_COMPLETION | FI_DELIVERY_COMPLETE)) !=
		       (FI_COMPLETION | FI_DELIVERY_COMPLETE));
		fprintf(stderr, "FI_INJECT flag unimplemented with rma_write internal\n");
		abort();
	}
	uint64_t max_blocks_per_packet, max_bytes_per_packet;
	if (is_intranode) {
		max_blocks_per_packet = FI_OPX_SHM_FIFO_SIZE >> 6;
		max_bytes_per_packet = FI_OPX_SHM_FIFO_SIZE;
	} else {
		max_blocks_per_packet = opx_ep->tx->pio_max_eager_tx_bytes >> 6;
		max_bytes_per_packet = opx_ep->tx->pio_max_eager_tx_bytes;
	}

	uint64_t bytes_sent = 0;
	while (bytes_to_send > 0) {
		bytes_to_send += (sizeof(struct fi_opx_hfi1_dput_iov));
		uint64_t totbytes = (bytes_to_send < max_bytes_per_packet) ? bytes_to_send : max_bytes_per_packet;
		uint64_t blocks_to_send_in_this_packet =
			bytes_to_send < max_bytes_per_packet ? bytes_to_send >> 6 : max_blocks_per_packet;
		uint64_t bytes_to_send_in_this_packet = blocks_to_send_in_this_packet << 6;
		uint64_t bytes_remain = totbytes - bytes_to_send_in_this_packet;
		// Handle the remainder case
		if (bytes_remain && blocks_to_send_in_this_packet < 128) {
			bytes_to_send_in_this_packet = bytes_to_send;
			blocks_to_send_in_this_packet += 1;
		}
		const uint64_t pbc_dws = 2 + /* pbc */
					 2 + /* lrh */
					 3 + /* bth */
					 9 + /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
					 (blocks_to_send_in_this_packet << 4);

		const uint16_t lrh_dws = htons(pbc_dws - 1);
		struct fi_opx_hfi1_dput_iov dput_iov =
			{
				(uint64_t)fetch_vaddr + bytes_sent,
				addr_offset + bytes_sent,
				bytes_to_send_in_this_packet - sizeof(struct fi_opx_hfi1_dput_iov)
			};
		uint64_t op64 = (op == FI_NOOP) ? FI_NOOP-1 : op;
		uint64_t dt64 = (dt == FI_VOID) ? FI_VOID-1 : dt;


		if (is_intranode) { /* compile-time constant expression */
			uint64_t pos;
			union fi_opx_hfi1_packet_hdr *tx_hdr =
				opx_shm_tx_next(&opx_ep->tx->shm, dest_rx, &pos);
			while(OFI_UNLIKELY(tx_hdr == NULL)) {
				fi_opx_shm_poll_many(&opx_ep->ep_fid, 0);
				tx_hdr = opx_shm_tx_next(
					&opx_ep->tx->shm, dest_rx, &pos);
			}
			tx_hdr->qw[0] = opx_ep->rx->tx.dput.hdr.qw[0] | lrh_dlid |
					((uint64_t)lrh_dws << 32);
			tx_hdr->qw[1] = opx_ep->rx->tx.dput.hdr.qw[1] | bth_rx;
			tx_hdr->qw[2] = opx_ep->rx->tx.dput.hdr.qw[2];
			tx_hdr->qw[3] = opx_ep->rx->tx.dput.hdr.qw[3];
			tx_hdr->qw[4] = opx_ep->rx->tx.dput.hdr.qw[4] | (FI_OPX_HFI_DPUT_OPCODE_ATOMIC_FETCH) | (dt64 <<32) | (op64 << 40) | (bytes_to_send_in_this_packet << 48);
			tx_hdr->qw[5] = key;
			tx_hdr->qw[6] = (uintptr_t)cc;

			union fi_opx_hfi1_packet_payload *const tx_payload =
				(union fi_opx_hfi1_packet_payload *)(tx_hdr + 1);

			memcpy((void *)tx_payload->byte, (const void *)&dput_iov,
			       sizeof(dput_iov));

			memcpy((void *)&tx_payload->byte[sizeof(dput_iov)], (const void *)sbuf,
			       bytes_to_send_in_this_packet-sizeof(dput_iov));

			opx_shm_tx_advance(&opx_ep->tx->shm, (void *)tx_hdr, pos);

		} else {
			/* compile-time constant expression */
			struct fi_opx_reliability_tx_replay *replay = NULL;
			if (reliability != OFI_RELIABILITY_KIND_NONE) {
				replay = fi_opx_reliability_client_replay_allocate(&opx_ep->reliability->state,
					true);
			}

			union fi_opx_reliability_tx_psn *psn_ptr = NULL;
			const int64_t psn =
				(reliability != OFI_RELIABILITY_KIND_NONE) ?
					fi_opx_reliability_tx_next_psn(&opx_ep->ep_fid,
									&opx_ep->reliability->state,
									opx_dst_addr.uid.lid,
									dest_rx,
									opx_dst_addr.reliability_rx,
									&psn_ptr) :
					0;
			if(OFI_UNLIKELY(psn == -1)) {
				fi_opx_reliability_client_replay_deallocate(&opx_ep->reliability->state, replay);
				// Handle eagain
				abort();
				//return -FI_EAGAIN;
			}

			/* BLOCK until enough credits become available */
			union fi_opx_hfi1_pio_state pio_state = *opx_ep->tx->pio_state;
			uint16_t total_credits_needed = blocks_to_send_in_this_packet + 1;
			uint16_t total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);
			if (OFI_UNLIKELY(total_credits_available < total_credits_needed)) {
				do {
					fi_opx_compiler_msync_writes(); // credit return
					FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
					total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);
				} while (total_credits_available < total_credits_needed);
				opx_ep->tx->pio_state->qw0 = pio_state.qw0;
			}

			replay->scb.qw0 = opx_ep->rx->tx.dput.qw0 | pbc_dws | ((opx_ep->tx->force_credit_return & FI_OPX_HFI1_PBC_CR_MASK) << FI_OPX_HFI1_PBC_CR_SHIFT);
			replay->scb.hdr.qw[0] = opx_ep->rx->tx.dput.hdr.qw[0] | lrh_dlid | ((uint64_t)lrh_dws << 32);
			replay->scb.hdr.qw[1] = opx_ep->rx->tx.dput.hdr.qw[1] | bth_rx;
			replay->scb.hdr.qw[2] = opx_ep->rx->tx.dput.hdr.qw[2] | psn;
			replay->scb.hdr.qw[3] = opx_ep->rx->tx.dput.hdr.qw[3];
			replay->scb.hdr.qw[4] = opx_ep->rx->tx.dput.hdr.qw[4] | (FI_OPX_HFI_DPUT_OPCODE_ATOMIC_FETCH) | (dt64 <<32) | (op64 << 40) | (bytes_to_send_in_this_packet << 48);
			replay->scb.hdr.qw[5] = key;
			replay->scb.hdr.qw[6] = (uintptr_t)cc;

			uint8_t *replay_payload = (uint8_t*)replay->payload;
			memcpy((void *)replay->payload, (const void *)&dput_iov, sizeof(dput_iov));
			memcpy((void *)(replay_payload + sizeof(dput_iov)),
				   (const void *)sbuf,
			       bytes_to_send_in_this_packet-sizeof(dput_iov));

			FI_OPX_HFI1_CLEAR_CREDIT_RETURN(opx_ep);
			fi_opx_reliability_service_do_replay(&opx_ep->reliability->service, replay);
			fi_opx_reliability_client_replay_register_no_update(
				&opx_ep->reliability->state, opx_dst_addr.uid.lid,
				opx_dst_addr.reliability_rx, dest_rx, psn_ptr, replay,
				reliability);
		} /* if !is_intranode */
		// actual payload bytes
		bytes_sent += dput_iov.bytes;
		rbuf += dput_iov.bytes;
		sbuf += dput_iov.bytes;
		// payload and metadata bytes in the packet
		bytes_to_send -= bytes_to_send_in_this_packet;
	} /* while bytes_to_send */

	return;
}


__OPX_FORCE_INLINE__
void fi_opx_atomic_cas_internal(struct fi_opx_ep *opx_ep,
				const void *buf,
				const size_t len, const union fi_opx_addr opx_dst_addr,
				const uint64_t addr_offset,
				const uint64_t key,
				const void *fetch_vaddr,
				const void *compare_vaddr,
				union fi_opx_context *opx_context, const uint64_t tx_op_flags,
				const struct fi_opx_cq *opx_cq,
				const struct fi_opx_cntr *opx_cntr,
				struct fi_opx_completion_counter *cc,
				enum fi_datatype dt, enum fi_op op,
				const int lock_required, const uint64_t caps,
				const enum ofi_reliability_kind reliability)
{
	fi_opx_ep_rx_poll(&opx_ep->ep_fid, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME);

	const unsigned is_intranode = fi_opx_rma_dput_is_intranode(caps, opx_dst_addr, opx_ep);
	const uint64_t dest_rx = opx_dst_addr.hfi1_rx;
	const uint64_t lrh_dlid = FI_OPX_ADDR_TO_HFI1_LRH_DLID(opx_dst_addr.fi);
	const uint64_t bth_rx = dest_rx << 56;

	uint8_t *sbuf = (uint8_t *)buf;
	uint8_t *cbuf = (uint8_t *)compare_vaddr;
	uintptr_t rbuf = addr_offset;
	uint64_t bytes_to_send = len * 2;

	if (tx_op_flags & FI_INJECT) {
		assert((tx_op_flags & (FI_COMPLETION | FI_TRANSMIT_COMPLETE)) !=
		       (FI_COMPLETION | FI_TRANSMIT_COMPLETE));
		assert((tx_op_flags & (FI_COMPLETION | FI_DELIVERY_COMPLETE)) !=
		       (FI_COMPLETION | FI_DELIVERY_COMPLETE));
		fprintf(stderr, "FI_INJECT flag unimplemented with rma_write internal\n");
		abort();
	}
	uint64_t max_blocks_per_packet, max_bytes_per_packet;
	if (is_intranode) {
		max_blocks_per_packet = FI_OPX_SHM_FIFO_SIZE >> 6;
		max_bytes_per_packet = FI_OPX_SHM_FIFO_SIZE;
	} else {
		max_blocks_per_packet = opx_ep->tx->pio_max_eager_tx_bytes >> 6;
		max_bytes_per_packet = opx_ep->tx->pio_max_eager_tx_bytes;
	}

    uint64_t bytes_sent = 0;
	while (bytes_to_send > 0) {
		bytes_to_send += (sizeof(struct fi_opx_hfi1_dput_iov));
		uint64_t totbytes = (bytes_to_send < max_bytes_per_packet) ? bytes_to_send : max_bytes_per_packet;
		uint64_t blocks_to_send_in_this_packet =
			bytes_to_send < max_bytes_per_packet ? bytes_to_send >> 6 : max_blocks_per_packet;
		uint64_t bytes_to_send_in_this_packet = blocks_to_send_in_this_packet << 6;
		uint64_t bytes_remain = totbytes - bytes_to_send_in_this_packet;
        // Handle the remainder case
		if (bytes_remain && blocks_to_send_in_this_packet < 128) {
			bytes_to_send_in_this_packet = bytes_to_send;
			blocks_to_send_in_this_packet += 1;
		}
		const uint64_t pbc_dws = 2 + /* pbc */
					 2 + /* lrh */
					 3 + /* bth */
					 9 + /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
					 (blocks_to_send_in_this_packet << 4);

		const uint16_t lrh_dws = htons(pbc_dws - 1);
		struct fi_opx_hfi1_dput_iov dput_iov =
			{
				(uint64_t)fetch_vaddr + bytes_sent,
				addr_offset + bytes_sent,
				(bytes_to_send_in_this_packet - sizeof(struct fi_opx_hfi1_dput_iov))
			};
		uint64_t op64 = (op == FI_NOOP) ? FI_NOOP-1 : op;
		uint64_t dt64 = (dt == FI_VOID) ? FI_VOID-1 : dt;

		if (is_intranode) { /* compile-time constant expression */
			uint64_t pos;
			union fi_opx_hfi1_packet_hdr *tx_hdr =
				opx_shm_tx_next(&opx_ep->tx->shm, dest_rx, &pos);

			while(OFI_UNLIKELY(tx_hdr == NULL)) {
				fi_opx_shm_poll_many(&opx_ep->ep_fid, 0);
				tx_hdr = opx_shm_tx_next(
					&opx_ep->tx->shm, dest_rx, &pos);
			}
			tx_hdr->qw[0] = opx_ep->rx->tx.dput.hdr.qw[0] | lrh_dlid |
					((uint64_t)lrh_dws << 32);
			tx_hdr->qw[1] = opx_ep->rx->tx.dput.hdr.qw[1] | bth_rx;
			tx_hdr->qw[2] = opx_ep->rx->tx.dput.hdr.qw[2];
			tx_hdr->qw[3] = opx_ep->rx->tx.dput.hdr.qw[3];

			tx_hdr->qw[4] = opx_ep->rx->tx.dput.hdr.qw[4] | (FI_OPX_HFI_DPUT_OPCODE_ATOMIC_COMPARE_FETCH) | (dt64 <<32) | (op64 << 40) | (bytes_to_send_in_this_packet << 48);
			tx_hdr->qw[5] = key;
			tx_hdr->qw[6] = (uintptr_t)cc;

			union fi_opx_hfi1_packet_payload *const tx_payload =
				(union fi_opx_hfi1_packet_payload *)(tx_hdr + 1);
			memcpy((void *)tx_payload->byte, (const void *)&dput_iov,
			       sizeof(dput_iov));


			uint64_t bytes_to_memcpy = (bytes_to_send_in_this_packet-sizeof(dput_iov))/2;
			memcpy((void *)&tx_payload->byte[sizeof(dput_iov)], (const void *)sbuf, bytes_to_memcpy);
			memcpy((void *)&tx_payload->byte[sizeof(dput_iov) + bytes_to_memcpy],
				   (const void *)cbuf,
				   bytes_to_memcpy);

			opx_shm_tx_advance(&opx_ep->tx->shm, (void *)tx_hdr, pos);

		} else {
			/* compile-time constant expression */
			struct fi_opx_reliability_tx_replay *replay = NULL;
			if (reliability != OFI_RELIABILITY_KIND_NONE) {
				replay = fi_opx_reliability_client_replay_allocate(&opx_ep->reliability->state,
					true);
			}
			union fi_opx_reliability_tx_psn *psn_ptr = NULL;
			const int64_t psn =
				(reliability != OFI_RELIABILITY_KIND_NONE) ?
					fi_opx_reliability_tx_next_psn(&opx_ep->ep_fid,
									&opx_ep->reliability->state,
									opx_dst_addr.uid.lid,
									dest_rx,
									opx_dst_addr.reliability_rx,
									&psn_ptr) :
					0;
			if(OFI_UNLIKELY(psn == -1)) {
				fi_opx_reliability_client_replay_deallocate(&opx_ep->reliability->state, replay);
				//TODO Handle eagain
				abort();
				// return -FI_EAGAIN;
			}

			/* BLOCK until enough credits become available */
			union fi_opx_hfi1_pio_state pio_state = *opx_ep->tx->pio_state;
			uint16_t total_credits_needed = blocks_to_send_in_this_packet + 1;
			uint16_t total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);
			if (OFI_UNLIKELY(total_credits_available < total_credits_needed)) {
				do {
					fi_opx_compiler_msync_writes(); // credit return
					FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
					total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);
				} while (total_credits_available < total_credits_needed);
				opx_ep->tx->pio_state->qw0 = pio_state.qw0;
			}

			replay->scb.qw0 = opx_ep->rx->tx.dput.qw0 | pbc_dws | ((opx_ep->tx->force_credit_return & FI_OPX_HFI1_PBC_CR_MASK) << FI_OPX_HFI1_PBC_CR_SHIFT);
			replay->scb.hdr.qw[0] = opx_ep->rx->tx.dput.hdr.qw[0] | lrh_dlid | ((uint64_t)lrh_dws << 32);
			replay->scb.hdr.qw[1] = opx_ep->rx->tx.dput.hdr.qw[1] | bth_rx;
			replay->scb.hdr.qw[2] = opx_ep->rx->tx.dput.hdr.qw[2] | psn;
			replay->scb.hdr.qw[3] = opx_ep->rx->tx.dput.hdr.qw[3],
			replay->scb.hdr.qw[4] = opx_ep->rx->tx.dput.hdr.qw[4] | (FI_OPX_HFI_DPUT_OPCODE_ATOMIC_COMPARE_FETCH) | (dt64 <<32) | (op64 << 40) | (bytes_to_send_in_this_packet << 48);
			replay->scb.hdr.qw[5] = key;
			replay->scb.hdr.qw[6] = (uintptr_t)cc;

			uint8_t *replay_payload = (uint8_t*)replay->payload;
			uint64_t bytes_to_memcpy = (bytes_to_send_in_this_packet-sizeof(dput_iov))/2;
			memcpy(replay_payload, (const void *)&dput_iov, sizeof(dput_iov));
			memcpy(replay_payload + sizeof(dput_iov), (const void *)sbuf, bytes_to_memcpy);
			memcpy(replay_payload + sizeof(dput_iov) + bytes_to_memcpy,
				   (const void *)cbuf, bytes_to_memcpy);

			FI_OPX_HFI1_CLEAR_CREDIT_RETURN(opx_ep);

			fi_opx_reliability_service_do_replay(&opx_ep->reliability->service, replay);
			fi_opx_reliability_client_replay_register_no_update(
				&opx_ep->reliability->state, opx_dst_addr.uid.lid,
				opx_dst_addr.reliability_rx, dest_rx, psn_ptr, replay,
				reliability);

		} /* if !is_intranode */
		// actual payload bytes
		bytes_sent += dput_iov.bytes;
		rbuf += dput_iov.bytes;
		sbuf += dput_iov.bytes;
		cbuf += dput_iov.bytes;
		// payload and metadata bytes in the packet
		bytes_to_send -= bytes_to_send_in_this_packet;
	} /* while bytes_to_send */
	return;
}




__OPX_FORCE_INLINE__
size_t fi_opx_atomic_internal(struct fi_opx_ep *opx_ep,
				const void *buf, size_t count,
				const union fi_opx_addr opx_dst_addr,
				uint64_t addr, uint64_t key,
				enum fi_datatype datatype, enum fi_op op,
				void *context, struct fi_opx_completion_counter *cc,
				const unsigned is_fetch, const void *fetch_vaddr,
				const unsigned is_compare, const void *compare_vaddr,
				const uint64_t tx_op_flags, const int lock_required,
				const enum fi_av_type av_type, const uint64_t caps,
				const enum ofi_reliability_kind reliability)
{
	assert((is_fetch == 0) || (is_fetch == 1));
	assert((is_compare == 0) || (is_compare == 1));

	if(op == FI_ATOMIC_READ) {
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
					 "===================================== ATOMIC READ (begin)\n");
		struct iovec iov = { (void*)fetch_vaddr, count * sizeofdt(datatype) };
		cc->cntr = opx_ep->read_cntr;
		fi_opx_readv_internal(opx_ep, &iov, 1, opx_dst_addr, &addr, &key,
							  (union fi_opx_context *)context, opx_ep->tx->op_flags,
							  opx_ep->rx->cq, opx_ep->read_cntr, cc,
							  datatype, op,
							  FI_OPX_HFI_DPUT_OPCODE_GET,
							  lock_required, caps, reliability);
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
					 "===================================== ATOMIC READ (end)\n");
		return count;
	}

	if (is_fetch && !is_compare) {
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
					 "===================================== ATOMIC FETCH (begin)\n");
		cc->cntr = opx_ep->read_cntr;
		fi_opx_atomic_fetch_internal(opx_ep, buf,  count * sizeofdt(datatype), opx_dst_addr,
									 addr, key,
									 fetch_vaddr,
									 (union fi_opx_context *)context, opx_ep->tx->op_flags,
									 opx_ep->rx->cq, opx_ep->read_cntr, cc,
									 datatype, op,
									 lock_required, caps, reliability);
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
					 "===================================== ATOMIC FETCH (end)\n");

	} else if (is_fetch && is_compare) {
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
					 "===================================== ATOMIC CAS (begin)\n");
		cc->cntr = opx_ep->read_cntr;
		fi_opx_atomic_cas_internal(opx_ep, buf,  count * sizeofdt(datatype), opx_dst_addr,
								   addr, key,
								   fetch_vaddr,
								   compare_vaddr,
								   (union fi_opx_context *)context, opx_ep->tx->op_flags,
								   opx_ep->rx->cq, opx_ep->read_cntr, cc,
								   datatype, op,
								   lock_required, caps, reliability);
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
					 "===================================== ATOMIC CAS (end)\n");

	} else if(!is_fetch && is_compare) {
		fprintf(stderr, "fi_opx_atomic_internal:  compare without fetch not implemented\n");
		abort();
	} else {
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
					 "===================================== ATOMIC WRITE (begin)\n");
		cc->cntr = opx_ep->write_cntr;
		fi_opx_write_internal(opx_ep, buf, count*sizeofdt(datatype), opx_dst_addr, addr, key,
							  (union fi_opx_context *)NULL, cc, datatype, op, opx_ep->tx->op_flags,
							  lock_required, caps, reliability);
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
					 "===================================== ATOMIC WRITE (end)\n");
	}

	return count;
}


ssize_t fi_opx_atomic_generic(struct fid_ep *ep, const void *buf, size_t count, fi_addr_t dst_addr,
			      uint64_t addr, uint64_t key, enum fi_datatype datatype, enum fi_op op,
			      void *context, const int lock_required, const enum fi_av_type av_type,
			      const uint64_t caps, const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep;

	opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	if (lock_required) {
		fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__);
		abort();
	}

	union fi_opx_addr opx_addr = { .fi = dst_addr };
	if (av_type == FI_AV_TABLE) {
		opx_addr = opx_ep->tx->av_addr[dst_addr];
	}

	if (OFI_UNLIKELY(!opx_reliability_ready(ep,
			&opx_ep->reliability->state,
			opx_addr.uid.lid,
			opx_addr.hfi1_rx,
			opx_addr.reliability_rx,
			reliability))) {
		fi_opx_ep_rx_poll(&opx_ep->ep_fid, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME);
		return -FI_EAGAIN;
	}

	struct fi_opx_completion_counter *cc = ofi_buf_alloc(opx_ep->rma_counter_pool);
	cc->byte_counter = sizeofdt(datatype) * count;
	cc->cq = (((opx_ep->tx->op_flags & FI_COMPLETION) == FI_COMPLETION) ||
		  ((opx_ep->tx->op_flags & FI_DELIVERY_COMPLETE) == FI_DELIVERY_COMPLETE)) ?
			 opx_ep->rx->cq :
			 NULL;
	cc->context = context;
	cc->hit_zero = fi_opx_hit_zero;

	union fi_opx_context *opx_context = (union fi_opx_context *)cc->context;
	if(opx_context && cc->cq) opx_context->flags = FI_ATOMIC | FI_WRITE;

	size_t xfer __attribute__((unused));
	xfer = fi_opx_atomic_internal(opx_ep, buf, count, opx_addr, addr, key, datatype, op,
				      context, cc, 0, NULL, 0, NULL, opx_ep->tx->op_flags,
				      lock_required, av_type, caps, reliability);
	assert(xfer == count);

	return 0;
}

__OPX_FORCE_INLINE__
ssize_t fi_opx_atomic_writemsg_generic(struct fid_ep *ep,
					const struct fi_msg_atomic *msg,
					const uint64_t flags,
					const int lock_required,
					const enum fi_av_type av_type,
					const uint64_t caps,
					const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep;
	opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	const enum fi_datatype datatype = msg->datatype;
	const enum fi_op op = msg->op;

	int ret = fi_opx_check_atomic(opx_ep, datatype, op, 1);
	if (ret)
		return ret;

	if (lock_required) {
		fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__);
		abort();
	}

	const union fi_opx_addr opx_dst_addr = { .fi = (av_type == FI_AV_TABLE) ?
							       opx_ep->tx->av_addr[msg->addr].fi :
							       msg->addr };

	if (OFI_UNLIKELY(!opx_reliability_ready(ep,
			&opx_ep->reliability->state,
			opx_dst_addr.uid.lid,
			opx_dst_addr.hfi1_rx,
			opx_dst_addr.reliability_rx,
			reliability))) {
		fi_opx_ep_rx_poll(&opx_ep->ep_fid, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME);
		return -FI_EAGAIN;
	}

	struct fi_opx_completion_counter *cc = ofi_buf_alloc(opx_ep->rma_counter_pool);
	size_t index;
	cc->byte_counter = 0;
	for (index = 0; index < msg->iov_count; index++) {
		cc->byte_counter += sizeofdt(datatype) * msg->msg_iov[index].count;
	}

	cc->cq = ((flags & FI_COMPLETION) == FI_COMPLETION) ? opx_ep->rx->cq : NULL;
	cc->context = msg->context;
	union fi_opx_context *opx_context = (union fi_opx_context *)cc->context;
	if(opx_context && cc->cq) opx_context->flags = FI_ATOMIC | FI_WRITE;

	cc->hit_zero = fi_opx_hit_zero;

	const size_t dtsize = sizeofdt(datatype);

	size_t rma_iov_index = 0;
	const size_t rma_iov_count = msg->rma_iov_count;
	uint64_t rma_iov_dtcount = msg->rma_iov[rma_iov_index].count;
	uint64_t rma_iov_addr = msg->rma_iov[rma_iov_index].addr;
	uint64_t rma_iov_key = msg->rma_iov[rma_iov_index].key;

	size_t msg_iov_index = 0;
	const size_t msg_iov_count = msg->iov_count;
	uint64_t msg_iov_dtcount = msg->msg_iov[msg_iov_index].count;
	uintptr_t msg_iov_vaddr = (uintptr_t)msg->msg_iov[msg_iov_index].addr;

	while (msg_iov_dtcount != 0 && rma_iov_dtcount != 0) {
		const size_t count_requested = MIN(msg_iov_dtcount, rma_iov_dtcount);

		const size_t count_transfered =
			fi_opx_atomic_internal(opx_ep, (void *)msg_iov_vaddr, count_requested,
					       opx_dst_addr, rma_iov_addr, rma_iov_key, datatype,
					       op, NULL, cc, 0, NULL, 0, NULL, flags, lock_required,
					       av_type, caps, reliability);

		const size_t bytes_transfered = dtsize * count_transfered;

		msg_iov_dtcount -= count_transfered;
		msg_iov_vaddr += bytes_transfered;

		if ((msg_iov_dtcount == 0) && ((msg_iov_index + 1) < msg_iov_count)) {
			++msg_iov_index;
			msg_iov_dtcount = msg->msg_iov[msg_iov_index].count;
			msg_iov_vaddr = (uintptr_t)msg->msg_iov[msg_iov_index].addr;
		}

		rma_iov_dtcount -= count_transfered;
		rma_iov_addr += bytes_transfered;

		if ((rma_iov_dtcount == 0) && ((rma_iov_index + 1) < rma_iov_count)) {
			++rma_iov_index;
			rma_iov_dtcount = msg->rma_iov[rma_iov_index].count;
			rma_iov_addr = msg->rma_iov[rma_iov_index].addr;
			rma_iov_key = msg->rma_iov[rma_iov_index].key;
		}
	}

	return 0;
}

__OPX_FORCE_INLINE__
ssize_t fi_opx_atomic_readwritemsg_generic(struct fid_ep *ep,
					   const struct fi_msg_atomic *msg,
					   struct fi_ioc *resultv,
					   const size_t result_count,
					   const uint64_t flags,
					   const int lock_required,
					   const enum fi_av_type av_type,
					   const uint64_t caps,
					   const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep;
	opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	const enum fi_datatype datatype = msg->datatype;
	const enum fi_op op = msg->op;

	int ret = fi_opx_check_atomic(opx_ep, datatype, op, 1);
	if (ret)
		return ret;

	if (lock_required) {
		fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__);
		abort();
	}

	const union fi_opx_addr opx_dst_addr = { .fi = (av_type == FI_AV_TABLE) ?
							       opx_ep->tx->av_addr[msg->addr].fi :
							       msg->addr };

	if (OFI_UNLIKELY(!opx_reliability_ready(ep,
			&opx_ep->reliability->state,
			opx_dst_addr.uid.lid,
			opx_dst_addr.hfi1_rx,
			opx_dst_addr.reliability_rx,
			reliability))) {
		fi_opx_ep_rx_poll(&opx_ep->ep_fid, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME);
		return -FI_EAGAIN;
	}

	const size_t dtsize = sizeofdt(datatype);

	size_t rma_iov_index = 0;
	const size_t rma_iov_count = msg->rma_iov_count;
	uint64_t rma_iov_dtcount = msg->rma_iov[rma_iov_index].count;
	uint64_t rma_iov_addr = msg->rma_iov[rma_iov_index].addr;
	uint64_t rma_iov_key = msg->rma_iov[rma_iov_index].key;

	size_t rst_iov_index = 0;
	const size_t rst_iov_count = result_count;
	uint64_t rst_iov_dtcount = resultv[rst_iov_index].count;
	uintptr_t rst_iov_vaddr = (uintptr_t)resultv[rst_iov_index].addr;

	struct fi_opx_completion_counter *cc = ofi_buf_alloc(opx_ep->rma_counter_pool);
	cc->byte_counter = 0;
	ssize_t index = 0;
	for (index = 0; index < msg->iov_count; index++) {
		cc->byte_counter += sizeofdt(datatype) * msg->msg_iov[index].count;
	}
	cc->cq = ((flags & FI_COMPLETION) == FI_COMPLETION) ? opx_ep->rx->cq : NULL;
	cc->context = msg->context;
	union fi_opx_context *opx_context = (union fi_opx_context *)cc->context;
	if(opx_context && cc->cq) opx_context->flags = FI_ATOMIC | FI_READ;


	cc->hit_zero = fi_opx_hit_zero;

	if (op != FI_ATOMIC_READ) { /* likely */

		size_t msg_iov_index = 0;
		const size_t msg_iov_count = msg->iov_count;
		uint64_t msg_iov_dtcount = msg->msg_iov[msg_iov_index].count;
		uintptr_t msg_iov_vaddr = (uintptr_t)msg->msg_iov[msg_iov_index].addr;

		size_t count_requested = MIN3(msg_iov_dtcount, rma_iov_dtcount, rst_iov_dtcount);

		while (count_requested > 0) {
			const size_t count_transfered =
				fi_opx_atomic_internal(opx_ep, (void *)msg_iov_vaddr,
						       count_requested, opx_dst_addr, rma_iov_addr,
						       rma_iov_key, datatype, op, NULL, cc, 1,
						       (const void *)rst_iov_vaddr, 0, NULL, flags,
						       lock_required, av_type, caps, reliability);

			const size_t bytes_transfered = dtsize * count_transfered;

			msg_iov_dtcount -= count_transfered;
			msg_iov_vaddr += bytes_transfered;

			if ((msg_iov_dtcount == 0) && ((msg_iov_index + 1) < msg_iov_count)) {
				++msg_iov_index;
				msg_iov_dtcount = msg->msg_iov[msg_iov_index].count;
				msg_iov_vaddr = (uintptr_t)msg->msg_iov[msg_iov_index].addr;
			}

			rma_iov_dtcount -= count_transfered;
			rma_iov_addr += bytes_transfered;

			if ((rma_iov_dtcount == 0) && ((rma_iov_index + 1) < rma_iov_count)) {
				++rma_iov_index;
				rma_iov_dtcount = msg->rma_iov[rma_iov_index].count;
				rma_iov_addr = msg->rma_iov[rma_iov_index].addr;
				rma_iov_key = msg->rma_iov[rma_iov_index].key;
			}

			rst_iov_dtcount -= count_transfered;
			rst_iov_vaddr += bytes_transfered;

			if ((rst_iov_dtcount == 0) && ((rst_iov_index + 1) < rst_iov_count)) {
				++rst_iov_index;
				rst_iov_dtcount = resultv[rst_iov_index].count;
				rst_iov_vaddr = (uintptr_t)resultv[rst_iov_index].addr;
			}

			count_requested = MIN3(msg_iov_dtcount, rma_iov_dtcount, rst_iov_dtcount);
		}

	} else {
		size_t count_requested = MIN(rma_iov_dtcount, rst_iov_dtcount);

		while (rma_iov_dtcount != 0 && rst_iov_dtcount != 0) {
			const size_t count_transfered = fi_opx_atomic_internal(
				opx_ep, NULL, count_requested, opx_dst_addr, rma_iov_addr,
				rma_iov_key, datatype, op, NULL, cc, 1, (const void *)rst_iov_vaddr,
				0, NULL, flags, lock_required, av_type, caps, reliability);

			const size_t bytes_transfered = dtsize * count_transfered;

			rma_iov_dtcount -= count_transfered;
			rma_iov_addr += bytes_transfered;

			if ((rma_iov_dtcount == 0) && ((rma_iov_index + 1) < rma_iov_count)) {
				++rma_iov_index;
				rma_iov_dtcount = msg->rma_iov[rma_iov_index].count;
				rma_iov_addr = msg->rma_iov[rma_iov_index].addr;
				rma_iov_key = msg->rma_iov[rma_iov_index].key;
			}

			rst_iov_dtcount -= count_transfered;
			rst_iov_vaddr += bytes_transfered;

			if ((rst_iov_dtcount == 0) && ((rst_iov_index + 1) < rst_iov_count)) {
				++rst_iov_index;
				rst_iov_dtcount = resultv[rst_iov_index].count;
				rst_iov_vaddr = (uintptr_t)resultv[rst_iov_index].addr;
			}

			count_requested = MIN(rma_iov_dtcount, rst_iov_dtcount);
		}
	}

	return 0;
}

__OPX_FORCE_INLINE__
ssize_t fi_opx_atomic_compwritemsg_generic(struct fid_ep *ep,
					   const struct fi_msg_atomic *msg,
					   const struct fi_ioc *comparev,
					   size_t compare_count,
					   struct fi_ioc *resultv,
					   size_t result_count,
					   uint64_t flags,
					   const int lock_required,
					   const enum fi_av_type av_type,
					   const uint64_t caps,
					   const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep;
	opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	const enum fi_datatype datatype = msg->datatype;
	const enum fi_op op = msg->op;

	int ret = fi_opx_check_atomic(opx_ep, datatype, op, 1);
	if (ret)
		return ret;

	if (lock_required) {
		fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__);
		abort();
	}

	const union fi_opx_addr opx_dst_addr = { .fi = (av_type == FI_AV_TABLE) ?
							       opx_ep->tx->av_addr[msg->addr].fi :
							       msg->addr };

	if (OFI_UNLIKELY(!opx_reliability_ready(ep,
			&opx_ep->reliability->state,
			opx_dst_addr.uid.lid,
			opx_dst_addr.hfi1_rx,
			opx_dst_addr.reliability_rx,
			reliability))) {
		fi_opx_ep_rx_poll(&opx_ep->ep_fid, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME);
		return -FI_EAGAIN;
	}

	const size_t dtsize = sizeofdt(datatype);

	size_t rma_iov_index = 0;
	const size_t rma_iov_count = msg->rma_iov_count;
	uint64_t rma_iov_dtcount = msg->rma_iov[rma_iov_index].count;
	uint64_t rma_iov_addr = msg->rma_iov[rma_iov_index].addr;
	uint64_t rma_iov_key = msg->rma_iov[rma_iov_index].key;

	size_t msg_iov_index = 0;
	const size_t msg_iov_count = msg->iov_count;
	uint64_t msg_iov_dtcount = msg->msg_iov[msg_iov_index].count;
	uintptr_t msg_iov_vaddr = (uintptr_t)msg->msg_iov[msg_iov_index].addr;

	size_t rst_iov_index = 0;
	const size_t rst_iov_count = result_count;
	uint64_t rst_iov_dtcount = resultv[rst_iov_index].count;
	uintptr_t rst_iov_vaddr = (uintptr_t)resultv[rst_iov_index].addr;

	size_t cmp_iov_index = 0;
	const size_t cmp_iov_count = compare_count;
	uint64_t cmp_iov_dtcount = comparev[cmp_iov_index].count;
	uintptr_t cmp_iov_vaddr = (uintptr_t)comparev[cmp_iov_index].addr;

	struct fi_opx_completion_counter *cc = ofi_buf_alloc(opx_ep->rma_counter_pool);
	cc->byte_counter = 0;
	ssize_t index;
	for (index = 0; index < msg->iov_count; index++) {
		cc->byte_counter += sizeofdt(datatype)* msg->msg_iov[index].count;
	}
	cc->cq = ((flags & FI_COMPLETION) == FI_COMPLETION) ? opx_ep->rx->cq : NULL;
	cc->context = msg->context;
	union fi_opx_context *opx_context = (union fi_opx_context *)cc->context;
	if(opx_context && cc->cq) opx_context->flags = FI_ATOMIC | FI_READ;

	cc->hit_zero = fi_opx_hit_zero;

	while (msg_iov_dtcount != 0 && rma_iov_dtcount != 0 && rst_iov_dtcount != 0 &&
	       cmp_iov_dtcount != 0) {
		const size_t count_requested =
			MIN4(msg_iov_dtcount, rma_iov_dtcount, rst_iov_dtcount, cmp_iov_dtcount);

		const size_t count_transfered =
			fi_opx_atomic_internal(opx_ep, (void *)msg_iov_vaddr, count_requested,
					       opx_dst_addr, rma_iov_addr, rma_iov_key, datatype,
					       op, NULL, cc, 1, (const void *)rst_iov_vaddr, 1,
					       (const void *)cmp_iov_vaddr, flags, lock_required,
					       av_type, caps, reliability);

		const size_t bytes_transfered = dtsize * count_transfered;

		msg_iov_dtcount -= count_transfered;
		msg_iov_vaddr += bytes_transfered;

		if ((msg_iov_dtcount == 0) && ((msg_iov_index + 1) < msg_iov_count)) {
			++msg_iov_index;
			msg_iov_dtcount = msg->msg_iov[msg_iov_index].count;
			msg_iov_vaddr = (uintptr_t)msg->msg_iov[msg_iov_index].addr;
		}

		rma_iov_dtcount -= count_transfered;
		rma_iov_addr += bytes_transfered;

		if ((rma_iov_dtcount == 0) && ((rma_iov_index + 1) < rma_iov_count)) {
			++rma_iov_index;
			rma_iov_dtcount = msg->rma_iov[rma_iov_index].count;
			rma_iov_addr = msg->rma_iov[rma_iov_index].addr;
			rma_iov_key = msg->rma_iov[rma_iov_index].key;
		}

		rst_iov_dtcount -= count_transfered;
		rst_iov_vaddr += bytes_transfered;

		if ((rst_iov_dtcount == 0) && ((rst_iov_index + 1) < rst_iov_count)) {
			++rst_iov_index;
			rst_iov_dtcount = resultv[rst_iov_index].count;
			rst_iov_vaddr = (uintptr_t)resultv[rst_iov_index].addr;
		}

		cmp_iov_dtcount -= count_transfered;
		cmp_iov_vaddr += bytes_transfered;

		if ((cmp_iov_dtcount == 0) && ((cmp_iov_index + 1) < cmp_iov_count)) {
			++cmp_iov_index;
			cmp_iov_dtcount = comparev[cmp_iov_index].count;
			cmp_iov_vaddr = (uintptr_t)comparev[cmp_iov_index].addr;
		}
	}

	return 0;
}

/*
 * Generic function to handle both fetching (1 operand) and compare
 * (2 operand) atomics.
 */

__OPX_FORCE_INLINE__
ssize_t fi_opx_fetch_compare_atomic_generic(
	struct fid_ep *ep, const void *buf, size_t count, void *desc, const void *compare,
	void *compare_desc, void *result, void *result_desc, fi_addr_t dest_addr, uint64_t addr,
	uint64_t key, enum fi_datatype datatype, enum fi_op op, void *context, int lock_required,
	const enum fi_av_type av_type, const uint64_t caps,
	const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep;

	opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	if (lock_required) {
		fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__);
		abort();
	}

	union fi_opx_addr opx_addr = { .fi = dest_addr };
	if (av_type == FI_AV_TABLE) {
		opx_addr = opx_ep->tx->av_addr[dest_addr];
	}

	if (OFI_UNLIKELY(!opx_reliability_ready(ep,
			&opx_ep->reliability->state,
			opx_addr.uid.lid,
			opx_addr.hfi1_rx,
			opx_addr.reliability_rx,
			reliability))) {
		fi_opx_ep_rx_poll(&opx_ep->ep_fid, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME);
		return -FI_EAGAIN;
	}

	struct fi_opx_completion_counter *cc = ofi_buf_alloc(opx_ep->rma_counter_pool);
	cc->byte_counter = sizeofdt(datatype) * count;
	cc->cq = (((opx_ep->tx->op_flags & FI_COMPLETION) == FI_COMPLETION) ||
		  ((opx_ep->tx->op_flags & FI_DELIVERY_COMPLETE) == FI_DELIVERY_COMPLETE)) ?
			 opx_ep->rx->cq :
			 NULL;
	cc->context = context;
	cc->hit_zero = fi_opx_hit_zero;

	union fi_opx_context *opx_context = (union fi_opx_context *)cc->context;
	if(opx_context && cc->cq) opx_context->flags = FI_ATOMIC | FI_WRITE;

	size_t xfer __attribute__((unused));
	xfer = fi_opx_atomic_internal(opx_ep, buf, count, opx_addr, addr, key, datatype, op,
				      context, cc, 1, result, compare!=NULL, compare, opx_ep->tx->op_flags,
				      lock_required, av_type, caps, reliability);
	assert(xfer == count);

	return 0;
}
ssize_t fi_opx_fetch_atomic_generic(struct fid_ep *ep, const void *buf, size_t count, void *desc,
				    void *result, void *result_desc, fi_addr_t dest_addr,
				    uint64_t addr, uint64_t key, enum fi_datatype datatype,
				    enum fi_op op, void *context, const int lock_required,
				    const enum fi_av_type av_type, const uint64_t caps,
				    const enum ofi_reliability_kind reliability)
{
	return fi_opx_fetch_compare_atomic_generic(ep, buf, count, desc, NULL, NULL, result,
						   result_desc, dest_addr, addr, key, datatype, op,
						   context, lock_required, av_type, caps,
						   reliability);
}

ssize_t fi_opx_compare_atomic_generic(struct fid_ep *ep, const void *buf, size_t count, void *desc,
				      const void *compare, void *compare_desc, void *result,
				      void *result_desc, fi_addr_t dest_addr, uint64_t addr,
				      uint64_t key, enum fi_datatype datatype, enum fi_op op,
				      void *context, const int lock_required,
				      const enum fi_av_type av_type, const uint64_t caps,
				      const enum ofi_reliability_kind reliability)
{
	return fi_opx_fetch_compare_atomic_generic(ep, buf, count, desc, compare, compare_desc,
						   result, result_desc, dest_addr, addr, key,
						   datatype, op, context, lock_required, av_type,
						   caps, reliability);
}

ssize_t fi_opx_inject_atomic_generic(struct fid_ep *ep, const void *buf, size_t count,
				     fi_addr_t dest_addr, uint64_t addr, uint64_t key,
				     enum fi_datatype datatype, enum fi_op op,
				     const int lock_required, const enum fi_av_type av_type,
				     const uint64_t caps,
				     const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	int ret = fi_opx_check_atomic(opx_ep, datatype, op, count);
	if (ret)
		return ret;

	if (lock_required) {
		fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__);
		abort();
	}
	assert(dest_addr != FI_ADDR_UNSPEC);
	const union fi_opx_addr opx_dst_addr = { .fi = (av_type == FI_AV_TABLE) ?
							       opx_ep->tx->av_addr[dest_addr].fi :
							       dest_addr };

	if (OFI_UNLIKELY(!opx_reliability_ready(ep,
			&opx_ep->reliability->state,
			opx_dst_addr.uid.lid,
			opx_dst_addr.hfi1_rx,
			opx_dst_addr.reliability_rx,
			reliability))) {
		fi_opx_ep_rx_poll(&opx_ep->ep_fid, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME);
		return -FI_EAGAIN;
	}

	struct fi_opx_completion_counter *cc = ofi_buf_alloc(opx_ep->rma_counter_pool);
	cc->byte_counter = sizeofdt(datatype) * count;
	cc->cq = NULL;
	cc->context = NULL;
	cc->hit_zero = fi_opx_hit_zero;
	cc->cntr = opx_ep->write_cntr;

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			 "===================================== ATOMIC INJECT WRITE (begin)\n");

	fi_opx_write_internal(opx_ep, buf, count*sizeofdt(datatype),
				opx_dst_addr, addr, key, NULL, cc, datatype,
				op, opx_ep->tx->op_flags | FI_INJECT,
				lock_required, caps, reliability);

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== ATOMIC INJECT WRITE (end)\n");
	return 0;
}

ssize_t fi_opx_atomic(struct fid_ep *ep, const void *buf, size_t count, void *desc,
		      fi_addr_t dst_addr, uint64_t addr, uint64_t key, enum fi_datatype datatype,
		      enum fi_op op, void *context)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	const enum fi_threading threading = opx_ep->threading;
	if (OFI_UNLIKELY(fi_opx_threading_unknown(threading))) {
		return -FI_EINVAL;
	}

	const int lock_required = fi_opx_threading_lock_required(threading);

	ssize_t rc;
	fi_opx_lock_if_required(&opx_ep->lock, lock_required);

	rc = fi_opx_atomic_generic(ep, buf, count, dst_addr, addr, key, datatype, op,
				     context, FI_OPX_LOCK_NOT_REQUIRED,
				     opx_ep->av_type, 0x0018000000000000ull,
				     OPX_RELIABILITY);

	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);

	return rc;
}

ssize_t fi_opx_fetch_atomic(struct fid_ep *ep, const void *buf, size_t count, void *desc,
			    void *result, void *result_desc, fi_addr_t dest_addr, uint64_t addr,
			    uint64_t key, enum fi_datatype datatype, enum fi_op op, void *context)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	const enum fi_threading threading = opx_ep->threading;
	if (OFI_UNLIKELY(fi_opx_threading_unknown(threading))) {
		return -FI_EINVAL;
	}

	const int lock_required = fi_opx_threading_lock_required(threading);

	ssize_t rc;
	fi_opx_lock_if_required(&opx_ep->lock, lock_required);

	if (opx_ep->av_type == FI_AV_MAP) {
		rc = fi_opx_fetch_atomic_generic(
			ep, buf, count, desc, result, result_desc, dest_addr, addr, key,
			datatype, op, context, FI_OPX_LOCK_NOT_REQUIRED,
			FI_AV_MAP, 0x0018000000000000ull, OPX_RELIABILITY);
	} else {
		rc = fi_opx_fetch_atomic_generic(
			ep, buf, count, desc, result, result_desc, dest_addr, addr, key,
			datatype, op, context, FI_OPX_LOCK_NOT_REQUIRED,
			FI_AV_TABLE, 0x0018000000000000ull, OPX_RELIABILITY);
	}

	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);

	return rc;
}

ssize_t fi_opx_compare_atomic(struct fid_ep *ep, const void *buf, size_t count, void *desc,
			      const void *compare, void *compare_desc, void *result,
			      void *result_desc, fi_addr_t dest_addr, uint64_t addr, uint64_t key,
			      enum fi_datatype datatype, enum fi_op op, void *context)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	const enum fi_threading threading = opx_ep->threading;
	if (OFI_UNLIKELY(fi_opx_threading_unknown(threading))) {
		return -FI_EINVAL;
	}

	const int lock_required = fi_opx_threading_lock_required(threading);

	ssize_t rc;
	fi_opx_lock_if_required(&opx_ep->lock, lock_required);

	if (opx_ep->av_type == FI_AV_MAP) {
		rc = fi_opx_compare_atomic_generic(
			ep, buf, count, desc, compare, compare_desc, result, result_desc,
			dest_addr, addr, key, datatype, op, context, FI_OPX_LOCK_NOT_REQUIRED,
			FI_AV_MAP, 0x0018000000000000ull, OPX_RELIABILITY);
	} else {
		rc = fi_opx_compare_atomic_generic(
			ep, buf, count, desc, compare, compare_desc, result, result_desc,
			dest_addr, addr, key, datatype, op, context, FI_OPX_LOCK_NOT_REQUIRED,
			FI_AV_TABLE, 0x0018000000000000ull, OPX_RELIABILITY);
	}

	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);

	return rc;
}

ssize_t fi_opx_inject_atomic(struct fid_ep *ep, const void *buf, size_t count, fi_addr_t dest_addr,
			     uint64_t addr, uint64_t key, enum fi_datatype datatype, enum fi_op op)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	const enum fi_threading threading = opx_ep->threading;
	if (OFI_UNLIKELY(fi_opx_threading_unknown(threading))) {
		return -FI_EINVAL;
	}

	const int lock_required = fi_opx_threading_lock_required(threading);

	ssize_t rc;
	fi_opx_lock_if_required(&opx_ep->lock, lock_required);

	if (opx_ep->av_type == FI_AV_MAP) {
		rc = fi_opx_inject_atomic_generic(ep, buf, count, dest_addr, addr, key,
						  datatype, op, FI_OPX_LOCK_NOT_REQUIRED,
						  FI_AV_MAP, 0x0018000000000000ull,
						  OPX_RELIABILITY);
	} else {
		rc = fi_opx_inject_atomic_generic(ep, buf, count, dest_addr, addr, key,
						  datatype, op, FI_OPX_LOCK_NOT_REQUIRED,
						  FI_AV_TABLE, 0x0018000000000000ull,
						  OPX_RELIABILITY);
	}

	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);
	return rc;
}

ssize_t fi_opx_atomicv(struct fid_ep *ep, const struct fi_ioc *iov, void **desc, size_t count,
		       uint64_t addr, uint64_t key, enum fi_datatype datatype, enum fi_op op,
		       void *context)
{
	errno = FI_ENOSYS;
	return -errno;
}

ssize_t fi_opx_atomic_writemsg(struct fid_ep *ep, const struct fi_msg_atomic *msg, uint64_t flags)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	const enum fi_threading threading = opx_ep->threading;
	if (OFI_UNLIKELY(fi_opx_threading_unknown(threading))) {
		return -FI_EINVAL;
	}

	const int lock_required = fi_opx_threading_lock_required(threading);

	ssize_t rc;
	fi_opx_lock_if_required(&opx_ep->lock, lock_required);

	if (opx_ep->av_type == FI_AV_MAP) {
		rc = fi_opx_atomic_writemsg_generic(ep, msg, flags, FI_OPX_LOCK_NOT_REQUIRED,
						    FI_AV_MAP,
						    0x0018000000000000ull,
						    OPX_RELIABILITY);
	} else {
		rc = fi_opx_atomic_writemsg_generic(ep, msg, flags, FI_OPX_LOCK_NOT_REQUIRED,
						    FI_AV_TABLE,
						    0x0018000000000000ull,
						    OPX_RELIABILITY);
	}

	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);
	return rc;
}

ssize_t fi_opx_atomic_readwritemsg(struct fid_ep *ep, const struct fi_msg_atomic *msg,
				   struct fi_ioc *resultv, void **result_desc, size_t result_count,
				   uint64_t flags)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	const enum fi_threading threading = opx_ep->threading;
	if (OFI_UNLIKELY(fi_opx_threading_unknown(threading))) {
		return -FI_EINVAL;
	}

	const int lock_required = fi_opx_threading_lock_required(threading);

	ssize_t rc;
	fi_opx_lock_if_required(&opx_ep->lock, lock_required);

	if (opx_ep->av_type == FI_AV_MAP) {
		rc = fi_opx_atomic_readwritemsg_generic(ep, msg, resultv, result_count,
							flags, FI_OPX_LOCK_NOT_REQUIRED,
							FI_AV_MAP,
							0x0018000000000000ull,
							OPX_RELIABILITY);
	} else {
		rc = fi_opx_atomic_readwritemsg_generic(ep, msg, resultv, result_count,
							flags, FI_OPX_LOCK_NOT_REQUIRED,
							FI_AV_TABLE,
							0x0018000000000000ull,
							OPX_RELIABILITY);
	}

	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);
	return rc;
}

ssize_t fi_opx_atomic_compwritemsg(struct fid_ep *ep, const struct fi_msg_atomic *msg,
				   const struct fi_ioc *comparev, void **compare_desc,
				   size_t compare_count, struct fi_ioc *resultv, void **result_desc,
				   size_t result_count, uint64_t flags)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	const enum fi_threading threading = opx_ep->threading;
	if (OFI_UNLIKELY(fi_opx_threading_unknown(threading))) {
		return -FI_EINVAL;
	}

	const int lock_required = fi_opx_threading_lock_required(threading);

	ssize_t rc;
	fi_opx_lock_if_required(&opx_ep->lock, lock_required);

	if (opx_ep->av_type == FI_AV_MAP) {
		rc = fi_opx_atomic_compwritemsg_generic(ep, msg, comparev, compare_count,
							resultv, result_count, flags,
							FI_OPX_LOCK_NOT_REQUIRED,
							FI_AV_MAP, 0x0018000000000000ull,
							OPX_RELIABILITY);
	} else {
		rc = fi_opx_atomic_compwritemsg_generic(ep, msg, comparev, compare_count,
							resultv, result_count, flags,
							FI_OPX_LOCK_NOT_REQUIRED,
							FI_AV_TABLE, 0x0018000000000000ull,
							OPX_RELIABILITY);
	}

	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);
	return rc;
}

int fi_opx_atomic_writevalid(struct fid_ep *ep, enum fi_datatype datatype, enum fi_op op,
			     size_t *count)
{
	static size_t sizeofdt[FI_DATATYPE_LAST] = {
		sizeof(int8_t), /* FI_INT8 */
		sizeof(uint8_t), /* FI_UINT8 */
		sizeof(int16_t), /* FI_INT16 */
		sizeof(uint16_t), /* FI_UINT16 */
		sizeof(int32_t), /* FI_INT32 */
		sizeof(uint32_t), /* FI_UINT32 */
		sizeof(int64_t), /* FI_INT64 */
		sizeof(uint64_t), /* FI_UINT64 */
		sizeof(float), /* FI_FLOAT */
		sizeof(double), /* FI_DOUBLE */
		sizeof(complex float), /* FI_FLOAT_COMPLEX */
		sizeof(complex double), /* FI_DOUBLE_COMPLEX */
		sizeof(long double), /* FI_LONG_DOUBLE */
		sizeof(complex long double) /* FI_LONG_DOUBLE_COMPLEX */
	};

	if ((op > FI_ATOMIC_WRITE) || (datatype >= FI_DATATYPE_LAST)) {
		*count = 0;
		errno = FI_EOPNOTSUPP;
		return -errno;
	}

	// *count = sizeof(union fi_opx_hfi1_packet_payload) / sizeofdt[datatype];
	*count = UINT64_MAX / sizeofdt[datatype];
	return 0;
}

int fi_opx_atomic_readwritevalid(struct fid_ep *ep, enum fi_datatype datatype, enum fi_op op,
				 size_t *count)
{
	static size_t sizeofdt[FI_DATATYPE_LAST] = {
		sizeof(int8_t), /* FI_INT8 */
		sizeof(uint8_t), /* FI_UINT8 */
		sizeof(int16_t), /* FI_INT16 */
		sizeof(uint16_t), /* FI_UINT16 */
		sizeof(int32_t), /* FI_INT32 */
		sizeof(uint32_t), /* FI_UINT32 */
		sizeof(int64_t), /* FI_INT64 */
		sizeof(uint64_t), /* FI_UINT64 */
		sizeof(float), /* FI_FLOAT */
		sizeof(double), /* FI_DOUBLE */
		sizeof(complex float), /* FI_FLOAT_COMPLEX */
		sizeof(complex double), /* FI_DOUBLE_COMPLEX */
		sizeof(long double), /* FI_LONG_DOUBLE */
		sizeof(complex long double) /* FI_LONG_DOUBLE_COMPLEX */
	};

	if ((op > FI_ATOMIC_WRITE) || (datatype >= FI_DATATYPE_LAST)) {
		*count = 0;
		errno = FI_EOPNOTSUPP;
		return -errno;
	}

	//*count = (sizeof(union fi_opx_hfi1_packet_payload) -
	//	  sizeof(struct fi_opx_hfi1_fetch_metadata)) /
	//	 sizeofdt[datatype];
	*count = UINT64_MAX /sizeofdt[datatype];
	return 0;
}

int fi_opx_atomic_compwritevalid(struct fid_ep *ep, enum fi_datatype datatype, enum fi_op op,
				 size_t *count)
{
	static size_t sizeofdt[FI_DATATYPE_LAST] = {
		sizeof(int8_t), /* FI_INT8 */
		sizeof(uint8_t), /* FI_UINT8 */
		sizeof(int16_t), /* FI_INT16 */
		sizeof(uint16_t), /* FI_UINT16 */
		sizeof(int32_t), /* FI_INT32 */
		sizeof(uint32_t), /* FI_UINT32 */
		sizeof(int64_t), /* FI_INT64 */
		sizeof(uint64_t), /* FI_UINT64 */
		sizeof(float), /* FI_FLOAT */
		sizeof(double), /* FI_DOUBLE */
		sizeof(complex float), /* FI_FLOAT_COMPLEX */
		sizeof(complex double), /* FI_DOUBLE_COMPLEX */
		sizeof(long double), /* FI_LONG_DOUBLE */
		sizeof(complex long double) /* FI_LONG_DOUBLE_COMPLEX */
	};

	if ((op < FI_CSWAP) || (op >= FI_ATOMIC_OP_LAST) || (datatype >= FI_DATATYPE_LAST)) {
		*count = 0;
		errno = FI_EOPNOTSUPP;
		return -errno;
	}

	// *count = (sizeof(union fi_opx_hfi1_packet_payload) / 2) / sizeofdt[datatype];
	*count = (UINT64_MAX / 2) / sizeofdt[datatype];
	return 0;
}

static struct fi_ops_atomic fi_opx_ops_atomic = { .size = sizeof(struct fi_ops_atomic),
						  .write = fi_opx_atomic,
						  .writev = fi_no_atomic_writev,
						  .writemsg = fi_opx_atomic_writemsg,
						  .inject = fi_opx_inject_atomic,
						  .readwrite = fi_opx_fetch_atomic,
						  .readwritev = fi_no_atomic_readwritev,
						  .readwritemsg = fi_opx_atomic_readwritemsg,
						  .compwrite = fi_opx_compare_atomic,
						  .compwritev = fi_no_atomic_compwritev,
						  .compwritemsg = fi_opx_atomic_compwritemsg,
						  .writevalid = fi_opx_atomic_writevalid,
						  .readwritevalid = fi_opx_atomic_readwritevalid,
						  .compwritevalid = fi_opx_atomic_compwritevalid };

int fi_opx_init_atomic_ops(struct fid_ep *ep, struct fi_info *info)
{
	struct fi_opx_ep *opx_ep;
	opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	if (!info || !opx_ep)
		goto err;

	if (info->caps & FI_ATOMICS || (info->tx_attr && (info->tx_attr->caps & FI_ATOMICS))) {
		opx_ep->ep_fid.atomic = &fi_opx_ops_atomic;
	}
	return 0;
err:
	errno = FI_EINVAL;
	return -errno;
}

int fi_opx_enable_atomic_ops(struct fid_ep *ep)
{
	struct fi_opx_ep *opx_ep;
	opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	if (!opx_ep || !opx_ep->domain)
		goto err;

	if (!opx_ep->ep_fid.atomic) {
		/* atomic ops not enabled on this endpoint */
		return 0;
	}
	/* fill in atomic formats */

	return 0;
err:
	errno = FI_EINVAL;
	return -errno;
	return 0;
}

int fi_opx_finalize_atomic_ops(struct fid_ep *ep)
{
	return 0;
}

FI_OPX_ATOMIC_SPECIALIZED_FUNC(OPX_LOCK, OPX_AV, 0x0018000000000000ull, OPX_RELIABILITY)

ssize_t fi_opx_atomic_FABRIC_DIRECT(struct fid_ep *ep, const void *buf, size_t count, void *desc,
				    fi_addr_t dest_addr, uint64_t addr, uint64_t key,
				    enum fi_datatype datatype, enum fi_op op, void *context)
{
	return FI_OPX_ATOMIC_SPECIALIZED_FUNC_NAME(atomic, OPX_LOCK, OPX_AV, 0x0018000000000000ull,
						   OPX_RELIABILITY)(
		ep, buf, count, desc, dest_addr, addr, key, datatype, op, context);
}

ssize_t fi_opx_inject_atomic_FABRIC_DIRECT(struct fid_ep *ep, const void *buf, size_t count,
					   fi_addr_t dest_addr, uint64_t addr, uint64_t key,
					   enum fi_datatype datatype, enum fi_op op)
{
	return FI_OPX_ATOMIC_SPECIALIZED_FUNC_NAME(inject_atomic, OPX_LOCK, OPX_AV,
						   0x0018000000000000ull, OPX_RELIABILITY)(
		ep, buf, count, dest_addr, addr, key, datatype, op);
}

ssize_t fi_opx_fetch_atomic_FABRIC_DIRECT(struct fid_ep *ep, const void *buf, size_t count,
					  void *desc, void *result, void *result_desc,
					  fi_addr_t dest_addr, uint64_t addr, uint64_t key,
					  enum fi_datatype datatype, enum fi_op op, void *context)
{
	return FI_OPX_ATOMIC_SPECIALIZED_FUNC_NAME(fetch_atomic, OPX_LOCK, OPX_AV,
						   0x0018000000000000ull,
						   OPX_RELIABILITY)(ep, buf, count, desc, result,
								    result_desc, dest_addr, addr,
								    key, datatype, op, context);
}

ssize_t fi_opx_compare_atomic_FABRIC_DIRECT(struct fid_ep *ep, const void *buf, size_t count,
					    void *desc, const void *compare, void *compare_desc,
					    void *result, void *result_desc, fi_addr_t dest_addr,
					    uint64_t addr, uint64_t key, enum fi_datatype datatype,
					    enum fi_op op, void *context)
{
	return FI_OPX_ATOMIC_SPECIALIZED_FUNC_NAME(compare_atomic, OPX_LOCK, OPX_AV,
						   0x0018000000000000ull, OPX_RELIABILITY)(
		ep, buf, count, desc, compare, compare_desc, result, result_desc, dest_addr, addr,
		key, datatype, op, context);
}
