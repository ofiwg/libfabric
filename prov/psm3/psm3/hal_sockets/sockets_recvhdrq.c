#ifdef PSM_SOCKETS
/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2021 Intel Corporation.

  This program is free software; you can redistribute it and/or modify
  it under the terms of version 2 of the GNU General Public License as
  published by the Free Software Foundation.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  Contact Information:
  Intel Corporation, www.intel.com

  BSD LICENSE

  Copyright(c) 2021 Intel Corporation.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/* Copyright (c) 2003-2021 Intel Corporation. All rights reserved. */

#include "psm_user.h"
#include "psm2_hal.h"

#include "ips_epstate.h"
#include "ips_proto.h"
#include "ips_expected_proto.h"
#include "ips_proto_help.h"
#include "ips_proto_internal.h"
#include "sockets_hal.h"
#include <sys/poll.h>

/*
 * Receive header queue initialization.
 */
psm2_error_t
psm3_sockets_recvhdrq_init(const struct ips_epstate *epstate,
		  const struct ips_proto *proto,
		  const struct ips_recvhdrq_callbacks *callbacks,
		  struct ips_recvhdrq *recvq
		)
{
	psm2_error_t err = PSM2_OK;

	memset(recvq, 0, sizeof(*recvq));
	recvq->proto = (struct ips_proto *)proto;
	pthread_spin_init(&recvq->hdrq_lock, PTHREAD_PROCESS_SHARED);

	recvq->epstate = epstate;
	recvq->recvq_callbacks = *callbacks;	/* deep copy */
	SLIST_INIT(&recvq->pending_acks);

	return err;
}

/* receive service routine for each packet opcode starting at
 * OPCODE_RESERVED (C0)
 */
ips_packet_service_fn_t
psm3_sockets_packet_service_routines[] = {
psm3_ips_proto_process_unknown_opcode,	/* 0xC0 */
psm3_ips_proto_mq_handle_tiny,		/* OPCODE_TINY */
psm3_ips_proto_mq_handle_short,		/* OPCODE_SHORT */
psm3_ips_proto_mq_handle_eager,		/* OPCODE_EAGER */
psm3_ips_proto_mq_handle_rts,		/* OPCODE_LONG_RTS */
psm3_ips_proto_mq_handle_cts,		/* OPCODE_LONG_CTS */
psm3_ips_proto_mq_handle_data,		/* OPCODE_LONG_DATA */
psm3_ips_proto_process_unknown_opcode,	/* C7 */
psm3_ips_proto_process_unknown_opcode,	/* C8 */

/* these are control packets */
psm3_ips_proto_process_ack,		/* OPCODE_ACK */
psm3_ips_proto_process_nak,		/* OPCODE_NAK */
psm3_ips_proto_process_unknown_opcode,	/* CB */
psm3_ips_proto_process_err_chk,		/* OPCODE_ERR_CHK */
psm3_ips_proto_process_unknown_opcode,	/* CD */
psm3_ips_proto_connect_disconnect,	/* OPCODE_CONNECT_REQUEST */
psm3_ips_proto_connect_disconnect,	/* OPCODE_CONNECT_REPLY */
psm3_ips_proto_connect_disconnect,	/* OPCODE_DISCONNECT__REQUEST */
psm3_ips_proto_connect_disconnect,	/* OPCODE_DISCONNECT_REPLY */

/* rest are not control packets */
psm3_ips_proto_am,			/* OPCODE_AM_REQUEST_NOREPLY */
psm3_ips_proto_am,			/* OPCODE_AM_REQUEST */
psm3_ips_proto_am			/* OPCODE_AM_REPLY */

/* D5-DF (OPCODE_FUTURE_FROM to OPCODE_FUTURE_TO) reserved for expansion */
};

/*---------------------------------------------------------------------------*/
/* TCP specific code */

#define if_incomplete(RECVLEN, EXPLEN)  \
	if ((RECVLEN > 0 && RECVLEN < EXPLEN) || (RECVLEN < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)))

#define if_complete(RECVLEN, EXPLEN) if (RECVLEN > 0 && RECVLEN >= EXPLEN)

static __inline__ ssize_t psm3_sockets_tcp_aggressive_recv(int fd, uint8_t *buf, uint32_t size) {
	uint32_t remainder  = size;
	uint8_t *rbuf = buf;
	int ret;
	while (remainder) {
#ifdef PSM_FI
		uint32_t part_len = remainder;
		if_pf(PSM3_FAULTINJ_ENABLED()) {
			PSM3_FAULTINJ_STATIC_DECL(fi_recvpart, "recvpart",
					"partial TCP recv",
					1, IPS_FAULTINJ_RECVPART);
			if_pf(PSM3_FAULTINJ_IS_FAULT(fi_recvpart, NULL, ""))
				part_len = min(remainder, 32);	// purposely less than min pkt size
		}
		ret = recv(fd, rbuf, part_len, MSG_DONTWAIT);
#else
		ret = recv(fd, rbuf, remainder, MSG_DONTWAIT);
#endif
		PSM2_LOG_MSG("recv fd=%d ret=%d", fd, ret);
		if (ret > 0) {
			remainder -= ret;
			rbuf += ret;
#ifdef PSM_FI
			if (part_len != remainder + ret)	// fault was injected
				return size - remainder;
#endif
		} else if (ret < 0 && remainder < size) {
			return size - remainder;
		} else {
			return ret;
		}
	}
	return size;
}

static __inline__ ssize_t drain_tcp_stream(int fd, uint8_t *buf, uint32_t buf_size, uint32_t drain_size) {
	uint32_t remainder  = drain_size;
	int ret;
	while (remainder) {
		ret = recv(fd, buf, buf_size < remainder ? buf_size : remainder, MSG_DONTWAIT);
		if (ret <= 0) {
			if (errno == EAGAIN || errno == EWOULDBLOCK) {
				_HFI_VDBG("No more data to read. Try to poll again.\n");
				// simple solution - try to finish the reading
				int old_errno = errno;
				struct pollfd pfd = {
					.fd = fd,
					.events = POLLIN
				};
				if (poll(&pfd, 1, TCP_POLL_TO) == 1 && pfd.revents==POLLIN) {
					continue;
				};
				errno = old_errno;
			}
			return ret;
		} else {
			remainder -= ret;
		}
	}
	return drain_size;
}

#define TCPRCV_OK 0
#define TCPRCV_PARTIAL -1
#define TCPRCV_DISCARD -2

/**
 * Receive byte stream from TCP. This approach benefits tiny and short messages
 *
 * Normal data read
 * 1) read data into sockets_ep.rbuf as much as we can
 * 2) if received length is less than ips_message_header size, records partial
 *    info, such as rbuf_cur_fd, rbuf_cur_offset etc., in sockets_ep, and returns
 *    TCPRCV_PARTIAL. Next call of psm3_sockets_recvhdrq_progress will call this function
 *    to continue the read. See "Partial data read" described below.
 * 3) if data received contains ips_message_header, get pktlen from it.
 *    3.1) if received data length is larger than pktlen, we got a whole PSM
 *         pkt, and have extra data for next pkts. Record extra data info, such
 *         as next_offset, next_len, and return TCPRCV_OK.
 *         psm3_sockets_recvhdrq_progress will go ahead process the data upon TCPRCV_OK.
 *         It will also check upd_ep.next_offset and call this function if it's
 *         not zero that means extra data exist. See "Extra data read" described
 *         below.
 *    3.2) if received data length is less than pktlen, try to read the
 *         remainder pkt body (we can return TCPRCV_PARTIAL and let the next
 *         call to continue read. But for the continue read from step 2), we
 *         only read in msg header. We want to be aggressive here to continue
 *         read in pkt body). If we got partial PSM pkt, record partial pkt info
 *         and return TCPRCV_PARTIAL. Next call of psm3_sockets_recvhdrq_progress will
 *         continue the read. See "Partial data read" described below. If we
 *         got the whole PSM pkt, return TCPRCV_OK. psm3_sockets_recvhdrq_progress will
 *         process the pkt.
 *    3.3) if received data length is pktlen, we got the whole pkt. returns
 *         TCPRCV_OK. psm3_sockets_recvhdrq_progress will process the pkt.
 *
 * Partial data read (when rbuf_cur_fd is not NULL)
 * 1) get start position and expected pkt payload size from rbuf cur fields in
 *    sockets_ep, and then read data from socket
 * 2) if received data length is expected length
 *    2.1) if start position >= ips_message_header size, we were reading
 *         remainder pkt body, and here we got the whole pkt. Returns
 *         TCPRCV_OK, and psm3_sockets_recvhdrq_progress will process the pkt.
 *    2.2) if start position is less than ips_message_header size, we were
 *         reading remainder pkt header, and here we got the header. Get pktlen
 *         from header, and try to read the pkt body. Same logic as step 3.2) in
 *         "Normal data read".
 *
 * Extra data read (when next_offset is non-zero)
 * 1) if extra data length >= ips_message_header size, get pktlen from header
 *    1.1) if extra data length >= pktlen, we have the whole pkt in
 *         sockets_ep.rbuf. No any data read need. Directly set receiver buffer
 *         to the start point of the extra data, and return TCPRCV_OK. If
 *         extra data length > pktlen, we have data for more pkts. Records
 *         next pkt info, such as next_offset, next_len etc, as well, so
 *         psm3_sockets_recvhdrq_progress will call this function again to process
 *         more extra data.
 *    1.2) if extra data length < pktlen, we only have partial pkt. Move the
 *         extra data to the beginner of sockets_ep.rbuf, so we can treat it as
 *         normal data read. Set received data length to the extra data
 *         length, and then go to step 2) in "Normal data read".
 * 2) if extra data length < ips_message_header size, we only have partial pkt.
 *    see above step 1.2).
 */
static __inline__ int
psm3_sockets_tcp_recv(psm2_ep_t ep, int fd, struct ips_recvhdrq_event *rcv_ev, uint8_t **ret_buf, int* ret_recvlen)
{
	uint8_t *buf;
	uint8_t *rbuf;
	uint32_t explen, acplen;
	uint32_t pktlen = 0;
	int recvlen, ret = TCPRCV_OK;

	if (ep->sockets_ep.rbuf_next_offset) {
		buf = ep->sockets_ep.rbuf + ep->sockets_ep.rbuf_next_offset;
		recvlen = ep->sockets_ep.rbuf_next_len;
		acplen = sizeof(struct ips_message_header);
		if (recvlen >= sizeof(struct ips_message_header)) {
			pktlen = ips_proto_lrh2_be_to_bytes(rcv_ev->proto,
				((struct ips_message_header *)buf)->lrh[2]);
			if (recvlen >= pktlen) {
				// already have a complete pkt, go ahead to process it
				if (recvlen > pktlen) {
					ep->sockets_ep.rbuf_next_offset += pktlen;
					ep->sockets_ep.rbuf_next_len -= pktlen;
					_HFI_VDBG("Next pkt for fd=%d: offset=%d len=%d\n",
						fd, ep->sockets_ep.rbuf_next_offset,
						ep->sockets_ep.rbuf_next_len);
				} else {
					ep->sockets_ep.rbuf_next_offset = 0;
					ep->sockets_ep.rbuf_next_len = 0;
				}
				recvlen = pktlen;
				rcv_ev->payload_size = pktlen - sizeof(struct ips_message_header);
				goto out;
			}
			explen = pktlen;
		} else {
			explen = acplen;
		}
		memmove(ep->sockets_ep.rbuf, buf, recvlen);
		buf = ep->sockets_ep.rbuf;
		ep->sockets_ep.rbuf_next_offset = 0;
		ep->sockets_ep.rbuf_next_len = 0;
	} else {
		buf = ep->sockets_ep.rbuf;
		if_pf (ep->sockets_ep.rbuf_cur_fd) {
			rbuf = ep->sockets_ep.rbuf + ep->sockets_ep.rbuf_cur_offset;
			explen = sizeof(struct ips_message_header)
					+ ep->sockets_ep.rbuf_cur_payload
					- ep->sockets_ep.rbuf_cur_offset;
			acplen = explen;
			_HFI_VDBG("Continue partial data: remainder=%d\n", explen);
		} else {
			rbuf = buf;
			// Try to read as much as we can
			explen = ep->sockets_ep.buf_size;
			acplen = sizeof(struct ips_message_header);
		}
		//recvlen = recv(fd, rbuf, explen, MSG_DONTWAIT);
		recvlen = psm3_sockets_tcp_aggressive_recv(fd, rbuf, explen);
		_HFI_VDBG("Got %d from fd=%d\n", recvlen, fd);
	}
	if_incomplete (recvlen, acplen) {
		// partial data
		ep->sockets_ep.rbuf_cur_fd = fd;
		if (recvlen > 0) {
			ep->sockets_ep.rbuf_cur_offset += recvlen;
		}
		_HFI_VDBG("Partial data: offset=%d, payload=%d recvlen=%d explen=%d\n",
			ep->sockets_ep.rbuf_cur_offset,
			ep->sockets_ep.rbuf_cur_payload, recvlen, explen);
		ret = TCPRCV_PARTIAL;
		goto out;
	}
	if_complete (recvlen, acplen) {
		if (ep->sockets_ep.rbuf_cur_offset < sizeof(struct ips_message_header)) {
			// got header
			pktlen = ips_proto_lrh2_be_to_bytes(rcv_ev->proto,
				((struct ips_message_header *)buf)->lrh[2]);
			if_pf (pktlen > ep->sockets_ep.buf_size) {
				// shouldn't happen
				_HFI_ERROR( "unexpected large recv fd=%d: pktlen=%u buf_size=%u on %s\n",
						fd, pktlen, ep->sockets_ep.buf_size, ep->dev_name);
				ret = drain_tcp_stream(fd, buf, ep->sockets_ep.buf_size,
						pktlen - sizeof(struct ips_message_header));
				ret = TCPRCV_DISCARD;
				goto out;
			}
			recvlen += ep->sockets_ep.rbuf_cur_offset;
			if (recvlen > pktlen) {
				ep->sockets_ep.rbuf_next_offset = pktlen;
				ep->sockets_ep.rbuf_next_len = recvlen - pktlen;
				ep->sockets_ep.rbuf_next_fd = fd;
				_HFI_VDBG("Got %d extra data from fd=%d on %s\n",
						ep->sockets_ep.rbuf_next_len, fd, ep->dev_name);
				recvlen = pktlen;
			} else if (recvlen < pktlen) {
				rbuf = buf + recvlen;
				explen = pktlen - recvlen;
				// read remainder body
				//int recvlen2 = recv(fd, rbuf, explen, MSG_DONTWAIT);
				int recvlen2 = psm3_sockets_tcp_aggressive_recv(fd, rbuf, explen);
				_HFI_VDBG("Got %d from fd=%d for remainder data\n", recvlen2, fd);
				if_incomplete (recvlen2, explen) {
					// partial data
					ep->sockets_ep.rbuf_cur_fd = fd;
					ep->sockets_ep.rbuf_cur_offset = recvlen;
					if (recvlen2 > 0) {
						ep->sockets_ep.rbuf_cur_offset += recvlen2;
					}
					ep->sockets_ep.rbuf_cur_payload =
						pktlen - sizeof(struct ips_message_header);
					_HFI_VDBG("New partial data: offset=%d, payload=%d\n",
						ep->sockets_ep.rbuf_cur_offset,
						ep->sockets_ep.rbuf_cur_payload);
					ret = TCPRCV_PARTIAL;
					goto out;
				}
				if_complete (recvlen2, explen) {
					recvlen = pktlen;
				}
			} else {
				ep->sockets_ep.rbuf_next_fd = 0;
				_HFI_VDBG("Got whole pkt len=%d from fd=%d on %s\n",
						recvlen, fd, ep->dev_name);
			}
			rcv_ev->payload_size = pktlen - sizeof(struct ips_message_header);
		} else {
			// get all remainder data
			recvlen += ep->sockets_ep.rbuf_cur_offset;
			rcv_ev->payload_size = recvlen - sizeof(struct ips_message_header);
		}
		if (ep->sockets_ep.rbuf_cur_fd && recvlen) {
			// reset rebuf cur fields
			ep->sockets_ep.rbuf_cur_fd = 0;
			ep->sockets_ep.rbuf_cur_offset = 0;
			ep->sockets_ep.rbuf_cur_payload = 0;
			ep->sockets_ep.rbuf_next_fd = 0;
			_HFI_VDBG("Got remainder data fd=%d. Total recvlen=%d payload_size=%d opcode=%x\n",
				fd, recvlen, rcv_ev->payload_size,
				_get_proto_hfi_opcode((struct ips_message_header *)buf));
		}
	}
out:
	*ret_buf = buf;
	*ret_recvlen = recvlen;
	return ret;
}
static __inline__ int
psm3_sockets_tcp_process_packet(struct ips_recvhdrq_event *rcv_ev,
               psm2_ep_t ep, uint8_t *buf, int fd,
               struct sockaddr_in6 *rem_addr,
               struct ips_recvhdrq *recvq, int flush)
{
	int ret = IPS_RECVHDRQ_CONTINUE;
	rcv_ev->p_hdr = (struct ips_message_header *)buf;
	rcv_ev->payload = (buf + sizeof(struct ips_message_header));
	uint8_t opcode = _get_proto_hfi_opcode(rcv_ev->p_hdr);

	_HFI_VDBG("TCP receive - opcode %x on %s\n", opcode, ep->dev_name);

	PSM2_LOG_MSG("Process PKT: opcode=%x payload_size=%d", opcode, rcv_ev->payload_size);
	PSM2_LOG_PKT_STRM(PSM2_LOG_RX,rcv_ev->p_hdr,"PKT_STRM:");
	struct ips_epstate_entry *epstaddr =
		ips_epstate_lookup(recvq->epstate, rcv_ev->p_hdr->connidx);

	if_pf((epstaddr == NULL) || (epstaddr->ipsaddr == NULL)) {
		rcv_ev->ipsaddr = NULL;
		ep->sockets_ep.tcp_incoming_fd = fd;
		recvq->recvq_callbacks.callback_packet_unknown(rcv_ev);
		ep->sockets_ep.tcp_incoming_fd = 0;
#if 0
	} else if_pf (psmi_sockaddr_cmp(&epstaddr->ipsaddr->sockets.remote_pri_addr, rem_addr)) {
		// TBD - we could also compare sin6_scope_id here or in
		// psmi_sockaddr_cmp to confirm interface used
		// TBD - we get some packets from unexpected sockets
		// occurs under higher stress
		_HFI_ERROR("mismatched IP %s got %s on %s\n",
				psmi_sockaddr_fmt((struct sockaddr *)&epstaddr->ipsaddr->sockets.remote_pri_addr, 0),
				psmi_sockaddr_fmt((struct sockaddr *)&rem_addr, 1),
				ep->dev_name);
		// TBD - we really want a variation of callback which will
		// not check for attempted connect REQ/DREQ, but do the rest
		rcv_ev->ipsaddr = NULL;
		recvq->recvq_callbacks.callback_packet_unknown(rcv_ev);
#endif
	} else {
		rcv_ev->ipsaddr = epstaddr->ipsaddr;
		ep->sockets_ep.tcp_incoming_fd = fd;
		psmi_assert(PSMI_HOWMANY(psm3_sockets_packet_service_routines)
				== OPCODE_FUTURE_FROM - OPCODE_RESERVED);
#ifndef PSM_TCP_ACK
		// TCP doesn't need ACK so simulate an ack for all packets sent
		// thus far. This essentially disables process_ack processing and
		// in rare cases involving TCP partial sends and remainder handling
		// this will delay ack processing until tcp_spio_transfer_frame has
		// reported send is done (such reports can be late due to partial send
		// handling and remainder).  The upside is it protects against
		// a very slow return path delivering a very stale PSN (> 2 billion
		// packets sent while return packet in flight).  OPA and verbs would
		// not have such an issue since lack of received acks would pace
		// them via PSM credit mechanism.
		if (opcode >= OPCODE_TINY && opcode <= OPCODE_LONG_DATA) {
			ips_epaddr_flow_t flowid = ips_proto_flowid(rcv_ev->p_hdr);
			psmi_assert(flowid < EP_FLOW_LAST);
			struct ips_flow *flow = &rcv_ev->ipsaddr->flows[flowid];
			// ack_seq_num is last received+1 and xmit_ack_num
			// is last acked+1, so this simulates ack up to
			// xmit_ack_num-1 (e.g. the last packet we sent).  Net result is
			// the process_ack call after handling the inbound packet is a noop
			rcv_ev->p_hdr->ack_seq_num = flow->xmit_ack_num.psn_num;
		}
#endif
		ret = ips_proto_process_packet(rcv_ev,
			psm3_sockets_packet_service_routines);
#ifdef TCP_RCV_FLUSH
#ifndef PSM_TCP_ACK
		// When running ack-less we can miss the opportunity to immediately
		// take advantage of send buffer space now available due to the
		// receipt of TCP acks carried with the packet we just received.
		// So if caller had gotten a new packet from the wire we try to
		// make some send progress here (just like process_ack would if
		// we were running with acks).
		if (flush) {
			ips_epaddr_flow_t flowid = ips_proto_flowid(rcv_ev->p_hdr);
			psmi_assert(flowid < EP_FLOW_LAST);
			struct ips_flow *flow = &rcv_ev->ipsaddr->flows[flowid];
			if (!SLIST_EMPTY(&flow->scb_pend))
				flow->flush(flow, NULL);
		}
#endif
#endif
		ep->sockets_ep.tcp_incoming_fd = 0;
		if_pf (ret == IPS_RECVHDRQ_REVISIT)
		{
			// try processing on next progress call
			_HFI_VDBG( "REVISIT returned on process_packet\n");
			// normally PSM would stop processing the header Q and
			// poll the same entry again.  We can't do that with a
			// UDP socket, so we stash the buffer and payload length
			// in our ep and will revisit it next time we are called
			ep->sockets_ep.revisit_buf = buf;
			ep->sockets_ep.revisit_fd = fd;
			ep->sockets_ep.revisit_payload_size = rcv_ev->payload_size;
			GENERIC_PERF_END(PSM_RX_SPEEDPATH_CTR); /* perf stats */
		}
	}
	return ret;
}

psm2_error_t psm3_sockets_tcp_recvhdrq_progress(struct ips_recvhdrq *recvq)
{
	GENERIC_PERF_BEGIN(PSM_RX_SPEEDPATH_CTR); /* perf stats */

	int ret = IPS_RECVHDRQ_CONTINUE;
	int recvlen = 0;
	psm2_ep_t ep = recvq->proto->ep;
	struct sockaddr_storage rem_addr;
	socklen_t len_addr = sizeof(rem_addr);
	PSMI_CACHEALIGN struct ips_recvhdrq_event rcv_ev = {
		.proto = recvq->proto,
		.recvq = recvq,
		//.ptype = RCVHQ_RCV_TYPE_ERROR
	};
	uint32_t num_done = 0;

	int cur_nfds;
	int incoming_fd;
	int rc, i;
	int to_adjust_fds = 0, idx_offset = 0;
	int rcv_state = TCPRCV_OK;
	bool is_udp = true;
	while (1) {
		uint8_t *buf;
		// TODO: optimize below logic
		if_pf (ep->sockets_ep.revisit_buf || ep->sockets_ep.rbuf_next_offset) {
			if (ep->sockets_ep.revisit_buf) {
				_HFI_VDBG("Process revisit pkt\n");
				buf = ep->sockets_ep.revisit_buf;
				ep->sockets_ep.revisit_buf = NULL;
				incoming_fd = ep->sockets_ep.revisit_fd;
				ep->sockets_ep.revisit_fd = 0;
				rcv_ev.payload_size = ep->sockets_ep.revisit_payload_size;
				ep->sockets_ep.revisit_payload_size = 0;
				ret = psm3_sockets_tcp_process_packet(&rcv_ev, ep, buf, incoming_fd, (struct sockaddr_in6 *)&rem_addr, recvq, 0);
			} else {
				// this happens when we get multiple pkts and need to revisit one of them.
				// after the revisiting, we need to continue process the remainder pkts
				rcv_state = psm3_sockets_tcp_recv(ep, ep->sockets_ep.rbuf_next_fd, &rcv_ev, &buf, &recvlen);
				_HFI_VDBG("Process remainder revisit pkt. rcv_state=%d\n", rcv_state);
				if (rcv_state == TCPRCV_PARTIAL) {
					recvq->proto->stats.partial_read_cnt++;
					break;
				}
				incoming_fd = ep->sockets_ep.rbuf_next_fd;
				ret = psm3_sockets_tcp_process_packet(&rcv_ev, ep, buf, incoming_fd, (struct sockaddr_in6 *)&rem_addr, recvq, 1);
			}
			if_pf (ret == IPS_RECVHDRQ_REVISIT)
			{
				return PSM2_OK_NO_PROGRESS;
			}
			num_done++;
			// if we can't process this now (such as an RTS we revisited and
			// ended up queueing on unexpected queue) we're told
			// to stop processing, we'll look at the rest later
			if_pf (ret == IPS_RECVHDRQ_BREAK) {
				_HFI_VDBG("stop rcvq\n");
				break;
			}
			continue;
		}

		// TODO: change to use epoll to further improve performance
		rc = poll(ep->sockets_ep.fds, ep->sockets_ep.nfds, 0);
		if (rc < 0) {
			_HFI_ERROR("failed to poll '%s' (%d) on %s epid %s\n",
				strerror(errno), errno, ep->dev_name, psm3_epid_fmt(ep->epid, 0));
			GENERIC_PERF_END(PSM_RX_SPEEDPATH_CTR); /* perf stats */
			return PSM2_INTERNAL_ERR;
		} else if (rc == 0) {
			break;
		}

		cur_nfds = ep->sockets_ep.nfds;
		for (i = 0; i < cur_nfds; i++) {
			if (ep->sockets_ep.fds[i].revents == 0 || ep->sockets_ep.fds[i].fd == -1) {
				// when process DISCONN on a different UDP socket/fd we may
				// close fd and set fds[] to -1 before we come to process this fd.
				continue;
			}
			if (ep->sockets_ep.fds[i].revents != POLLIN) {
				// POLLNVAL is expected if remote closed the socket
				if (ep->sockets_ep.fds[i].revents != POLLNVAL) {
					_HFI_VDBG("Unexpected returned events fd=%d (%d) on %s epid %s\n",
						ep->sockets_ep.fds[i].fd, ep->sockets_ep.fds[i].revents,
						ep->dev_name, psm3_epid_fmt(ep->epid, 0));
				}
				psm3_sockets_tcp_close_fd(ep, ep->sockets_ep.fds[i].fd, i, NULL);
				to_adjust_fds = 1;
				break;
			}
			// listening socket
			if (ep->sockets_ep.fds[i].fd == ep->sockets_ep.listener_fd) {
				while(1) {
					len_addr = sizeof(rem_addr);
					incoming_fd = accept(ep->sockets_ep.fds[i].fd, (struct sockaddr *)&rem_addr, &len_addr);
					if (incoming_fd < 0) {
						if (errno != EWOULDBLOCK && errno != EAGAIN) {
							_HFI_ERROR("failed accept '%s' (%d) on %s epid %s\n",
								strerror(errno), errno, ep->dev_name, psm3_epid_fmt(ep->epid, 0));
							GENERIC_PERF_END(PSM_RX_SPEEDPATH_CTR); /* perf stats */
							return PSM2_INTERNAL_ERR;
						}
						break;
					}
					// coverity[uninit_use_in_call] - rem_addr initialized in accept() call above
					_HFI_PRDBG("Accept connection (fd=%d) from %s on %s epid %s\n", incoming_fd,
						psmi_sockaddr_fmt((struct sockaddr *)&rem_addr, 0),
						ep->dev_name, psm3_epid_fmt(ep->epid, 1));
					PSM2_LOG_MSG("Accept connection (fd=%d) from %s on %s epid %s", incoming_fd,
 						psmi_sockaddr_fmt((struct sockaddr *)&rem_addr, 0),
 						ep->dev_name, psm3_epid_fmt(ep->epid, 1));
					if (psm3_sockets_tcp_add_fd(ep, incoming_fd) != PSM2_OK) {
						return PSM2_INTERNAL_ERR;
					}
				}
				continue;
			} else {
				if_pf (ep->sockets_ep.rbuf_cur_fd &&
					ep->sockets_ep.rbuf_cur_fd != ep->sockets_ep.fds[i].fd) {
					// have partial data to read and the fd is not expected, skip it
					_HFI_VDBG("Skip fd=%d, expected fd=%d\n", ep->sockets_ep.fds[i].fd,
						ep->sockets_ep.rbuf_cur_fd);
					// set rcv_state=TCPRCV_PARTIAL, so if the expected fd not ready, we will break
					// the while loop
					rcv_state = TCPRCV_PARTIAL;
					continue;
				}
				is_udp = ep->sockets_ep.fds[i].fd == ep->sockets_ep.udp_rx_fd;
process:
				if (!is_udp) {
					rcv_state = psm3_sockets_tcp_recv(ep, ep->sockets_ep.fds[i].fd, &rcv_ev, &buf, &recvlen);
					if (rcv_state == TCPRCV_PARTIAL) {
						recvq->proto->stats.partial_read_cnt++;
						break;
					} else if (rcv_state == TCPRCV_DISCARD) {
						goto processed;
					}
				} else {
					buf = ep->sockets_ep.rbuf;
					// UDP is datagram. we shall get the whole pkt.
					recvlen = recvfrom(ep->sockets_ep.udp_rx_fd, buf, ep->sockets_ep.buf_size,
								MSG_DONTWAIT|MSG_TRUNC,
								(struct sockaddr *)&rem_addr, &len_addr);
					if (recvlen > sizeof(struct ips_message_header)) {
						rcv_ev.payload_size = recvlen - sizeof(struct ips_message_header);
					}
				}
				if (recvlen < 0) {
					if (errno == EAGAIN || errno == EWOULDBLOCK) {
						break;
					} else {
						// TBD - how to best handle errors
						_HFI_ERROR("failed recv '%s' (%d) on %s epid %s\n",
							strerror(errno), errno, ep->dev_name, psm3_epid_fmt(ep->epid, 0));
						GENERIC_PERF_END(PSM_RX_SPEEDPATH_CTR); /* perf stats */
						return PSM2_INTERNAL_ERR;
					}
				} else if (!is_udp && recvlen == 0) {
					psm3_sockets_tcp_close_fd(ep, ep->sockets_ep.fds[i].fd, i, NULL);
					to_adjust_fds = 1;
					continue;
				}
				if_pf (_HFI_VDBG_ON) {
					if (!is_udp) {
						len_addr = sizeof(rem_addr);
						if (getpeername(ep->sockets_ep.fds[i].fd, (struct sockaddr *)&rem_addr, &len_addr) != 0) {
							len_addr = 0;
						}
					}
					if (len_addr) {
						// coverity[uninit_use_in_call] - intended, need to know what exactly in rem_addr
						_HFI_VDBG("got recv %u bytes from IP %s payload_size=%d opcode=%x\n", recvlen,
							psmi_sockaddr_fmt((struct sockaddr *)&rem_addr, 0),
							rcv_ev.payload_size,
							_get_proto_hfi_opcode((struct ips_message_header *)buf));
					} else {
						_HFI_VDBG("got recv %u bytes from IP n/a\n", recvlen);
					}
				}
				if_pf (_HFI_PDBG_ON)
					_HFI_PDBG_DUMP_ALWAYS(buf, recvlen);
				if_pf (recvlen < sizeof(struct ips_message_header)) {
					_HFI_ERROR( "unexpected small recv: %u on %s\n", recvlen, ep->dev_name);
					goto processed;
				}
			}
			ret = psm3_sockets_tcp_process_packet(&rcv_ev, ep, buf,
						ep->sockets_ep.fds[i].fd,
						(struct sockaddr_in6 *)&rem_addr, recvq, 1);
			if_pf (ret == IPS_RECVHDRQ_REVISIT)
			{
				return PSM2_OK_NO_PROGRESS;
			}
processed:
			num_done++;
			// if we can't process this now (such as an RTS we revisited and
			// ended up queueing on unexpected queue) we're told
			// to stop processing, we'll look at the rest later
			if_pf (ret == IPS_RECVHDRQ_BREAK) {
				_HFI_VDBG("stop rcvq\n");
				break;
			}
			if (ep->sockets_ep.rbuf_next_offset) {
				goto process;
			}
		}
		if (to_adjust_fds) {
			idx_offset = 0;
			for (i = 0; i < ep->sockets_ep.nfds; i++) {
				if (idx_offset > 0) {
					ep->sockets_ep.fds[i].fd = ep->sockets_ep.fds[i + idx_offset].fd;
				}
				if (ep->sockets_ep.fds[i].fd == -1) {
					idx_offset += 1;
					i -= 1;
					ep->sockets_ep.nfds -= 1;
				}
			}
			to_adjust_fds = 0;
		}
		if (rcv_state == TCPRCV_PARTIAL) {
			break;
		}
	}

	/* Process any pending acks before exiting */
	process_pending_acks(recvq);

	GENERIC_PERF_END(PSM_RX_SPEEDPATH_CTR); /* perf stats */

	return num_done?PSM2_OK:PSM2_OK_NO_PROGRESS;

}

/*---------------------------------------------------------------------------*/
/* UDP specific code */

static __inline__ int
psm3_sockets_udp_process_packet(struct ips_recvhdrq_event *rcv_ev,
               psm2_ep_t ep, uint8_t *buf,
               struct sockaddr_in6 *rem_addr,
               struct ips_recvhdrq *recvq)
{
	int ret = IPS_RECVHDRQ_CONTINUE;
	rcv_ev->p_hdr = (struct ips_message_header *)buf;
	rcv_ev->payload = (buf + sizeof(struct ips_message_header));

#ifdef UDP_DEBUG
	_HFI_VDBG("UDP receive - opcode %x on %s\n",
		_get_proto_hfi_opcode(rcv_ev->p_hdr), ep->dev_name);
#endif

	PSM2_LOG_MSG("Process PKT: opcode=%x payload_size=%d", _get_proto_hfi_opcode(rcv_ev->p_hdr), rcv_ev->payload_size);
	PSM2_LOG_PKT_STRM(PSM2_LOG_RX,rcv_ev->p_hdr,"PKT_STRM:");
	struct ips_epstate_entry *epstaddr =
		ips_epstate_lookup(recvq->epstate, rcv_ev->p_hdr->connidx);

	if_pf((epstaddr == NULL) || (epstaddr->ipsaddr == NULL)) {
		rcv_ev->ipsaddr = NULL;
		recvq->recvq_callbacks.callback_packet_unknown(rcv_ev);
#if 0
	} else if_pf (psmi_sockaddr_cmp(&epstaddr->ipsaddr->sockets.remote_pri_addr, rem_addr)) {
		// TBD - we could also compare sin6_scope_id here or in
		// psmi_sockaddr_cmp to confirm interface used
		// TBD - we get some packets from unexpected sockets
		// occurs under higher stress
		_HFI_ERROR("mismatched IP %s got %s on %s\n",
				psmi_sockaddr_fmt((struct sockaddr *)&epstaddr->ipsaddr->sockets.remote_pri_addr, 0),
				psmi_sockaddr_fmt((struct sockaddr *)&rem_addr, 1),
				ep->dev_name);
		// TBD - we really want a variation of callback which will
		// not check for attempted connect REQ/DREQ, but do the rest
		rcv_ev->ipsaddr = NULL;
		recvq->recvq_callbacks.callback_packet_unknown(rcv_ev);
#endif
	} else {
		rcv_ev->ipsaddr = epstaddr->ipsaddr;
		psmi_assert(PSMI_HOWMANY(psm3_sockets_packet_service_routines)
				== OPCODE_FUTURE_FROM - OPCODE_RESERVED);
		ret = ips_proto_process_packet(rcv_ev,
			psm3_sockets_packet_service_routines);
		if_pf (ret == IPS_RECVHDRQ_REVISIT)
		{
			// try processing on next progress call
			_HFI_VDBG( "REVISIT returned on process_packet\n");
			// normally PSM would stop processing the header Q and
			// poll the same entry again.  We can't do that with a
			// UDP socket, so we stash the buffer and payload length
			// in our ep and will revisit it next time we are called
			ep->sockets_ep.revisit_buf = buf;
			ep->sockets_ep.revisit_payload_size = rcv_ev->payload_size;
			GENERIC_PERF_END(PSM_RX_SPEEDPATH_CTR); /* perf stats */
		}
	}
	return ret;
}

psm2_error_t psm3_sockets_udp_recvhdrq_progress(struct ips_recvhdrq *recvq)
{
	GENERIC_PERF_BEGIN(PSM_RX_SPEEDPATH_CTR); /* perf stats */

	int ret = IPS_RECVHDRQ_CONTINUE;
	int recvlen = 0;
	psm2_ep_t ep = recvq->proto->ep;
	struct sockaddr_storage rem_addr;
	socklen_t len_addr = sizeof(rem_addr);
	PSMI_CACHEALIGN struct ips_recvhdrq_event rcv_ev = {
		.proto = recvq->proto,
		.recvq = recvq,
		//.ptype = RCVHQ_RCV_TYPE_ERROR
	};
	uint32_t num_done = 0;

	while (1) {
		uint8_t *buf;
		// TBD really only need to check this on 1st loop
		if_pf (ep->sockets_ep.revisit_buf) {
			buf = ep->sockets_ep.revisit_buf;
			ep->sockets_ep.revisit_buf = NULL;
			rcv_ev.payload_size = ep->sockets_ep.revisit_payload_size;
			ep->sockets_ep.revisit_payload_size = 0;
		} else {
			buf = ep->sockets_ep.rbuf;
			// TBD - do we need rem_addr?  if not, can use recv
			// MSG_DONTWAIT is redundant since we set O_NONBLOCK
			recvlen = recvfrom(ep->sockets_ep.udp_rx_fd, buf, ep->sockets_ep.buf_size,
								MSG_DONTWAIT|MSG_TRUNC,
								(struct sockaddr *)&rem_addr, &len_addr);
			if (recvlen < 0) {
				if (errno == EAGAIN || errno == EWOULDBLOCK) {
					break;
				} else {
					// TBD - how to best handle errors
					_HFI_ERROR("failed recv '%s' (%d) on %s epid %s\n",
						strerror(errno), errno, ep->dev_name, psm3_epid_fmt(ep->epid, 0));
					GENERIC_PERF_END(PSM_RX_SPEEDPATH_CTR); /* perf stats */
					return PSM2_INTERNAL_ERR;
				}
			}
			// coverity[uninit_use] - rem_addr initialized in recvfrom() call above
			if_pf (len_addr > sizeof(rem_addr)
				|| rem_addr.ss_family != AF_INET6) {
				// TBD - how to best handle errors
				// coverity[uninit_use_in_call] - rem_addr initialized in recvfrom() call above
				_HFI_ERROR("unexpected rem_addr type (%u) on %s epid %s\n",
					rem_addr.ss_family, ep->dev_name, psm3_epid_fmt(ep->epid, 0));
				GENERIC_PERF_END(PSM_RX_SPEEDPATH_CTR); /* perf stats */
				return PSM2_INTERNAL_ERR;
			}
			if_pf (_HFI_VDBG_ON) {
				if (len_addr) {
					_HFI_VDBG("got recv %u bytes from IP %s payload_size=%d opcode=%x\n", recvlen,
						psmi_sockaddr_fmt((struct sockaddr *)&rem_addr, 0),
						rcv_ev.payload_size,
						_get_proto_hfi_opcode((struct ips_message_header *)buf));
				} else {
					_HFI_VDBG("got recv %u bytes from IP n/a\n", recvlen);
				}
			}
			if_pf (_HFI_PDBG_ON)
				_HFI_PDBG_DUMP_ALWAYS(buf, recvlen);
			if_pf (recvlen < sizeof(struct ips_message_header)) {
				_HFI_ERROR( "unexpected small recv: %u on %s\n", recvlen, ep->dev_name);
				goto processed;
			} else if_pf (recvlen > ep->sockets_ep.buf_size) {
				_HFI_ERROR( "unexpected large recv: %u on %s\n", recvlen, ep->dev_name);
				goto processed;
			}
			rcv_ev.payload_size = recvlen - sizeof(struct ips_message_header);
		}
		ret = psm3_sockets_udp_process_packet(&rcv_ev, ep, buf,
						(struct sockaddr_in6 *)&rem_addr, recvq);
		if_pf (ret == IPS_RECVHDRQ_REVISIT)
		{
			return PSM2_OK_NO_PROGRESS;
		}
processed:
		num_done++;
		// if we can't process this now (such as an RTS we revisited and
		// ended up queueing on unexpected queue) we're told
		// to stop processing, we'll look at the rest later
		if_pf (ret == IPS_RECVHDRQ_BREAK) {
			_HFI_VDBG("stop rcvq\n");
			break;
		}
	}

	/* Process any pending acks before exiting */
	process_pending_acks(recvq);

	GENERIC_PERF_END(PSM_RX_SPEEDPATH_CTR); /* perf stats */

	return num_done?PSM2_OK:PSM2_OK_NO_PROGRESS;
}

#endif /* PSM_SOCKETS */
