#ifdef PSM_SOCKETS
/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2017 Intel Corporation.

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

  Copyright(c) 2017 Intel Corporation.

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

/* Copyright (c) 2003-2017 Intel Corporation. All rights reserved. */
#ifndef _SOCKETS_SPIO_C_
#define _SOCKETS_SPIO_C_

/* included header files  */
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <linux/sockios.h>
#include "ips_proto.h"
#include "ips_proto_internal.h"
#include "ips_proto_params.h"

/*
 * Check and process events
 * return value:
 *  PSM2_OK: normal events processing;
 *  PSM2_OK_NO_PROGRESS: no event is processed;
 */
static inline psm2_error_t
psm3_sockets_spio_process_events(const struct ptl *ptl_gen)
{
	// TODD - TBD - check link status events for UD/UDP
	return PSM2_OK;
}

/*---------------------------------------------------------------------------*/
/* TCP specific code */

static __inline__ ssize_t
psm3_sockets_tcp_aggressive_send(int fd, uint8_t *buf, size_t len)
{
	size_t remainder = len;
	uint8_t *rbuf = buf;
	ssize_t ret;

	while (remainder) {
		ret = send(fd, rbuf, remainder, MSG_DONTWAIT);
		if (ret > 0) {
			remainder -= ret;
			rbuf += ret;
		} else if (remainder < len) {
			return len - remainder;
		} else {
			return ret;
		}
	}
	return len;
}
static __inline__ bool
psm3_sockets_ips_msg_hdr_equal(struct ips_message_header* msg1,
	struct ips_message_header* msg2)
{
	return 0 == memcmp(msg1, msg2, sizeof(*msg1));
}

static __inline__ psm2_error_t
psm3_sockets_tcp_aux_send(psm2_ep_t ep, struct ips_flow *flow, uint8_t *sbuf, unsigned len)
{
	psm2_error_t ret = PSM2_OK;
#ifdef PSM_FI
	if_pf(PSM3_FAULTINJ_ENABLED_EP(ep)) {
		PSM3_FAULTINJ_STATIC_DECL(fi_sendlost, "sendlost",
				"drop packet before sending",
				1, IPS_FAULTINJ_SENDLOST);
		if_pf(PSM3_FAULTINJ_IS_FAULT(fi_sendlost, ep, " UDP DISCON"))
			return ret;
	}
#endif // PSM_FI
	if_pf (sendto(ep->sockets_ep.udp_tx_fd, sbuf, len, 0,
		&flow->ipsaddr->sockets.remote_aux_addr,
		sizeof(flow->ipsaddr->sockets.remote_aux_addr)) == -1) {
		PSM2_LOG_MSG("sendto fd=%d ret=-1 errno=%d", ep->sockets_ep.udp_tx_fd, errno);
		if (errno != EAGAIN && errno != EWOULDBLOCK) {
			_HFI_ERROR("UDP send failed on %s: %s\n", ep->dev_name, strerror(errno));
		}
		ret = PSM2_EP_NO_RESOURCES;
#ifdef PSM_LOG
	} else {
		PSM2_LOG_MSG("sendto fd=%d ret=%d", ep->sockets_ep.udp_tx_fd, ret);
#endif
	}
	return ret;
}

// when called:
//		scb->ips_lrh has fixed size PSM header including OPA LRH
//		payload, length is data after header
//		we don't do checksum, let verbs handle that for us
//		the above is in unregistered memory, perhaps even on stack
// for isCtrlMsg, scb is only partially initialized (see ips_scb.h)
// and payload and length may refer to buffers on stack
static inline psm2_error_t
psm3_sockets_tcp_spio_transfer_frame(struct ips_proto *proto, struct ips_flow *flow,
			struct ips_scb *scb, uint32_t *payload,
			uint32_t length, uint32_t isCtrlMsg,
			uint32_t cksum_valid, uint32_t cksum
#ifdef PSM_CUDA
			, uint32_t is_cuda_payload
#endif
			)
{
	psm2_error_t ret = PSM2_OK;
	psm2_ep_t ep = proto->ep;
	uint8_t *sbuf = ep->sockets_ep.sbuf;
	unsigned len;
	struct ips_message_header *ips_lrh = &scb->ips_lrh;
	uint8_t opcode = _get_proto_hfi_opcode(ips_lrh);
	static struct ips_message_header last_sent; // implicit zero init for static
	PSM2_LOG_MSG("entering with fd=%d len=%d opcode=%x",
		flow->ipsaddr->sockets.tcp_fd, length, opcode);
	if (ep->sockets_ep.sbuf_flow) {
		// have remaining data
		if (flow != ep->sockets_ep.sbuf_flow) {
			// new pkt, but we need to finish remaining data first
			_HFI_VDBG("Reject fd=%d because of remaining data. Continue fd=%d offset=%d remainder=%d\n",
					flow->ipsaddr->sockets.tcp_fd, ep->sockets_ep.sbuf_flow->ipsaddr->sockets.tcp_fd,
					ep->sockets_ep.sbuf_offset, ep->sockets_ep.sbuf_remainder);
			PSM2_LOG_MSG("coming fd=%d, send remainder fd=%d",
				flow->ipsaddr->sockets.tcp_fd,
				ep->sockets_ep.sbuf_flow->ipsaddr->sockets.tcp_fd);
			// set flow so we will send out remaining data
			flow = ep->sockets_ep.sbuf_flow;
			// set ret to PSM2_EP_NO_RESOURCES so the caller will retry this new pkt later
			ret = PSM2_EP_NO_RESOURCES;
		} else if (!psm3_sockets_ips_msg_hdr_equal((struct ips_message_header*)sbuf, ips_lrh)) {
			// new data on the same flow. send out remainder first, then return PSM2_EP_NO_RESOURCES
			// to resend the new data
			ret = PSM2_EP_NO_RESOURCES;
		}
		// continue send remainder data
		len = ep->sockets_ep.sbuf_remainder;
		goto send;
	}
	// we reach here when no remaining data. i.e. new packet
	// For packets eligible to being retried immediately via the control queue,
	// we catch duplicate sends (matching last_sent) and avoid the unnecessary
	// duplicate send by simply returning PSM2_OK. This can avoid an infinite
	// sequence of partial sends followed by a retry via control queue.
	if (proto->ctrl_msg_queue_enqueue & proto->message_type_to_mask[opcode]) {
		psmi_assert(isCtrlMsg);
		psmi_assert(length == 0);
		if (psm3_sockets_ips_msg_hdr_equal(&last_sent, ips_lrh)) {
			// this is resending of ctrl message, and we already
			// queued one, just say it's fine
			_HFI_VDBG("Duplicate Control Message sending skipped: opcode=%x fd=%d\n",
				opcode, flow->ipsaddr->sockets.tcp_fd);
			PSM2_LOG_MSG("fd=%d ctr msg already sent out", flow->ipsaddr->sockets.tcp_fd);
			return PSM2_OK;
		}
	}
	if (scb->scb_flags & IPS_SEND_FLAG_TCP_REMAINDER) {
		// revisit the scb that has partial data sending, and all data already out
		// so return PSM2_OK
		_HFI_VDBG("Data already sent out fd=%d\n", flow->ipsaddr->sockets.tcp_fd);
		PSM2_LOG_MSG("fd=%d data msg already sent out", flow->ipsaddr->sockets.tcp_fd);
#ifdef PSM_TCP_ACK
		return PSM2_OK;
#else
		return isCtrlMsg ? PSM2_OK : PSM2_TCP_DATA_SENT;
#endif
	}
	len = sizeof(*ips_lrh) + length;

#ifdef PSM_FI
	// This is a bit of a high stress cheat.  While TCP is loss-less
	// we know that our callers for control messages may be forced to
	// discard control messages when we have NO_RESOURCES for long enough
	// or the shallow control queue overflows.  So for control messages
	// we can discard the packet here to simulate those side effects of
	// extreme TCP backpressure and long duration NO_RESOURCES.
	// The "sendfull*" injectors have a similar effect but will rarely
	// simulate enough repeated backpressure for the caller to actually give up.
	// We do not simulate loss for data packets since we commit to their
	// reliable delivery here once we return PSM2_OK.  For data packets the
	// "sendfull" injector will create the need for callers to retry sending.
	// We omit connect packets since they are less likely to see NO_RESOURCES
	// long enough for their caller to timeout (and we want to be able to set
	// a high sendlost without causing job launch issues).  Disconnect only
	// has 100 quick retries, so we let sendlost apply to it.
	if_pf(PSM3_FAULTINJ_ENABLED_EP(ep) && isCtrlMsg
			&& opcode != OPCODE_CONNECT_REQUEST
			&& opcode != OPCODE_CONNECT_REPLY) {
		PSM3_FAULTINJ_STATIC_DECL(fi_sendlost, "sendlost",
				"drop packet before sending",
				1, IPS_FAULTINJ_SENDLOST);
		if_pf(PSM3_FAULTINJ_IS_FAULT(fi_sendlost, ep, " TCP ctrl"))
			return PSM2_OK;
	}
#endif // PSM_FI
	PSMI_LOCK_ASSERT(proto->mq->progress_lock);
	psmi_assert_always(! cksum_valid);	// no software checksum yet

	if (!isCtrlMsg && proto->ep->sockets_ep.snd_pace_thresh) {
		uint32_t used;
		if (ioctl(flow->ipsaddr->sockets.tcp_fd, SIOCOUTQ, &used) == 0) {
			if (flow->used_snd_buff && used > proto->ep->sockets_ep.snd_pace_thresh
				&& used >= flow->used_snd_buff) {
				_HFI_VDBG("Pre=%d Cur=%d Delta=%d len=%d opcode=%x fd=%d\n",
						flow->used_snd_buff, used,
						used - flow->used_snd_buff,
						len, opcode, flow->ipsaddr->sockets.tcp_fd);
				return PSM2_EP_NO_RESOURCES;
			}
			flow->used_snd_buff = used;
		} else {
			_HFI_DBG("ERR: %s\n", strerror(errno));
		}
	}

	// TBD - we should be able to skip sending some headers such as OPA lrh and
	// perhaps bth (does PSM use bth to hold PSNs? - yes)
	// copy scb->ips_lrh to send buffer
	_HFI_VDBG("copy lrh %p\n", ips_lrh);
	memcpy(sbuf, ips_lrh, sizeof(*ips_lrh));
#ifndef PSM_TCP_ACK
	// clear IPS_SEND_FLAG_ACKREQ in bth[2] because TCP doesn't need ack
	if (!isCtrlMsg) {
		((struct ips_message_header *)sbuf)->bth[2] &= __cpu_to_be32(~IPS_SEND_FLAG_ACKREQ);
	}
#endif
	// copy payload to send buffer, length could be zero, be safe
	_HFI_VDBG("copy payload %p %u\n",  payload, length);
#ifdef PSM_CUDA
	if (is_cuda_payload) {
		PSMI_CUDA_CALL(cuMemcpyDtoH, sbuf+sizeof(*ips_lrh),
				(CUdeviceptr)payload, length);
	} else
#endif
	{
		memcpy(sbuf+sizeof(*ips_lrh), payload, length);
	}
	_HFI_VDBG("TCP send - opcode %x len %u fd=%d\n", opcode, len, flow->ipsaddr->sockets.tcp_fd);
	// we don't support software checksum
	psmi_assert_always(! (proto->flags & IPS_PROTO_FLAG_CKSUM));

	if_pf (ips_lrh->khdr.kdeth0 & __cpu_to_le32(IPS_SEND_FLAG_INTR)) {
		_HFI_VDBG("send solicted event\n");
		// TBD - how to send so wake up rcvthread?  Separate socket?
	}

	if_pf (_HFI_PDBG_ON) {
		_HFI_PDBG_ALWAYS("sockets_tcp_spio_transfer_frame: len %u, remote IP %s payload %u\n",
			len,
			psmi_sockaddr_fmt((struct sockaddr *)&flow->ipsaddr->sockets.remote_pri_addr, 0),
			length);
		_HFI_PDBG_DUMP_ALWAYS(sbuf, len);
	}
send:
	// opcode of the data will go out. The data can be the previous data with partial sending
	// that is different from scb data
	opcode = _get_proto_hfi_opcode((struct  ips_message_header*)sbuf);
	if_pf (opcode == OPCODE_DISCONNECT_REPLY) {
		return psm3_sockets_tcp_aux_send(ep, flow, sbuf, len);
	}
	if (flow->ipsaddr->sockets.tcp_fd > 0) {
		psmi_assert((len & 3) == 0);	// must be DWORD mult
#ifdef PSM_FI
	size_t part_len = len;
	if_pf(PSM3_FAULTINJ_ENABLED_EP(ep)) {
		PSM3_FAULTINJ_STATIC_DECL(fi_sendpart, "sendpart",
				"partial TCP send",
				1, IPS_FAULTINJ_SENDPART);
		if_pf(PSM3_FAULTINJ_IS_FAULT(fi_sendpart, ep, ""))
			part_len = min(len, 32);	// purposely less than min pkt size
	}
#endif // PSM_FI
//		ssize_t res = psm3_sockets_tcp_aggressive_send(flow->ipsaddr->sockets.tcp_fd, sbuf + ep->sockets_ep.sbuf_offset, len);
		ssize_t res = send(flow->ipsaddr->sockets.tcp_fd, sbuf + ep->sockets_ep.sbuf_offset,
#ifdef PSM_FI
							part_len,
#else
							len,
#endif
							MSG_DONTWAIT);
		PSM2_LOG_MSG("send fd=%d len=%d opcode=%x ret=%d", flow->ipsaddr->sockets.tcp_fd, len, opcode, res);
		if (res == len) {
			// send out full pkt (or last chunk)
			ep->sockets_ep.sbuf_offset = 0;
			ep->sockets_ep.sbuf_remainder = 0;
			ep->sockets_ep.sbuf_flow = NULL;
			if (proto->ctrl_msg_queue_enqueue & proto->message_type_to_mask[opcode]) {
				memcpy(&last_sent, sbuf, sizeof(last_sent));
			}
			_HFI_VDBG("Sent successfully. opcode=%x fd=%d\n", opcode, flow->ipsaddr->sockets.tcp_fd);
		} else if (res > 0) {
			// send out partial pkt
			if (ep->sockets_ep.sbuf_flow == NULL) {
				ep->sockets_ep.sbuf_offset = res;
				ep->sockets_ep.sbuf_remainder = len - res;
				ep->sockets_ep.sbuf_flow = flow;
				scb->scb_flags |= IPS_SEND_FLAG_TCP_REMAINDER;
			} else {
				ep->sockets_ep.sbuf_offset += res;
				ep->sockets_ep.sbuf_remainder -= res;
			}
			proto->stats.partial_write_cnt++;
			_HFI_VDBG("Partial sending. res=%ld len=%d offset=%d remainder=%d fd=%d\n",
					res, len, ep->sockets_ep.sbuf_offset, ep->sockets_ep.sbuf_remainder,
					flow->ipsaddr->sockets.tcp_fd);
			ret = PSM2_EP_NO_RESOURCES;
		} else if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ENOTCONN) {
			// socket is not ready. Either of outgoing buffer is full or not yet connected.
			_HFI_VDBG("Partial or empty sending. errno=%d offset=%d remainder=%d fd=%d\n",
				errno, ep->sockets_ep.sbuf_offset, ep->sockets_ep.sbuf_remainder,
				flow->ipsaddr->sockets.tcp_fd);
			ret = PSM2_EP_NO_RESOURCES;
		} else {
			if (flow->ipsaddr->cstate_outgoing == CSTATE_ESTABLISHED) {
				// error
				_HFI_INFO("TCP send fd=%d opcode=%x failed on %s: %s\n",
					flow->ipsaddr->sockets.tcp_fd, opcode,
					ep->dev_name, strerror(errno));
			} else {
				// sending data under undesired state, such as WAITING, WAITING_DISC, DISCONNECTED
				_HFI_DBG("TCP send fd=%d failed on %s. cstate_outgoing=%d opcode=%x error: %s\n",
					flow->ipsaddr->sockets.tcp_fd, ep->dev_name, flow->ipsaddr->cstate_outgoing,
					opcode, strerror(errno));
			}
			if (isCtrlMsg && opcode!=OPCODE_CONNECT_REQUEST && opcode!=OPCODE_CONNECT_REPLY) {
				// send non-conn control message via aux socket (UDP)
				// it doesn't make sense to continue CONN msg with aux socket because
				// we do not support sending data msg via aux socket (UDP)
				ret = psm3_sockets_tcp_aux_send(ep, flow, sbuf + ep->sockets_ep.sbuf_offset, len);
				if (ret != PSM2_OK && ep->sockets_ep.sbuf_flow) {
					// hopefully this will not happen. TBD - how we handle this case
					_HFI_ERROR("TCP send failed in the middle of msg transition.\n");
				}
			} else {
				// TCP has lost the connection.  We can't put TCP data scb's
				// on UDP since TCP's PSM3_MTU may be too large.  We can't
				// segment since caller has assigned psn's for this and scb's
				// after it already.  Also with TCP ack-less behavior we
				// may have lost some packets previously given to the sockets
				// stack and we have no way to know which and retransmit.
				// So our only recourse is an error so caller doesn't think
				// this packet was sent.  Ultimately unless we get a remote
				// disconnect, the job may hang (which is what verbs or OPA
				// would do if network connectivity was lost), however unlike
				// OPA we will not be able to resume if connectivity is restored
				ret = PSM2_EP_NO_NETWORK;
			}
			// close the TCP fd, and we will switch to use aux socket that is UDP
			psm3_sockets_tcp_close_fd(ep, flow->ipsaddr->sockets.tcp_fd, -1, flow);
			flow->ipsaddr->sockets.tcp_fd = 0;
		}
	} else {
		if (isCtrlMsg && opcode!=OPCODE_CONNECT_REQUEST && opcode!=OPCODE_CONNECT_REPLY) {
			// send control message via aux socket (UDP)
			_HFI_VDBG("Invalid tcp_fd on %s! Try to use aux socket.\n", ep->dev_name);
			ret = psm3_sockets_tcp_aux_send(ep, flow, sbuf + ep->sockets_ep.sbuf_offset, len);
		} else {
			// unable to switch to UDP for data packets, see discussion above
			_HFI_VDBG("Invalid tcp_fd on %s!\n", ep->dev_name);
			ret = PSM2_EP_NO_NETWORK;
		}
	}

#ifndef PSM_TCP_ACK
	// return PSM2_OK for ctrl msg and PSM2_TCP_DATA_SENT for data msg
	if (ret == PSM2_OK && !isCtrlMsg) {
		return PSM2_TCP_DATA_SENT;
	}
#endif
	return ret;
}

/*---------------------------------------------------------------------------*/
/* UDP specific code */

// segment and send a eager or long_data SCB using GSO send
// caller will wait for ack before freeing scb, so we could potentially
// send directly from the payload in future and avoid some copies
static __inline__ int
psm3_sockets_udp_gso_send(int fd, struct ips_proto *proto, struct sockaddr_in6 *addr, 
		struct ips_scb *scb, uint8_t *payload, uint32_t length,
		uint32_t frag_size
#ifdef PSM_CUDA
		, uint32_t is_cuda_payload
#endif
		)
{
	char control[CMSG_SPACE(sizeof(uint16_t))] = {0};
	struct msghdr msg = {0};
	struct iovec iov = {0};
	uint8_t *sbuf_gso = proto->ep->sockets_ep.sbuf_udp_gso;
	struct ips_message_header *ips_lrh = &scb->ips_lrh;
	psm2_ep_t ep = proto->ep;
	uint32_t gso_len = 0;

	// get some fields out of header, we'll be modifying these on each
	// fragment
	uint32_t psn = __be32_to_cpu(ips_lrh->bth[2]) & proto->psn_mask;
	uint32_t bth_w2 = __be32_to_cpu(ips_lrh->bth[2]) & ~proto->psn_mask;
	uint32_t offset = ips_lrh->hdr_data.u32w0;

	psmi_assert(length > 0);
	psmi_assert(scb->nfrag > 1);
	psmi_assert(scb->nfrag_remaining <= scb->nfrag);
	psmi_assert(scb->nfrag_remaining);
	psmi_assert(scb->nfrag <= ep->chunk_max_segs);
	psmi_assert(frag_size);
	psmi_assert(frag_size <= scb->frag_size);
	psmi_assert(scb->frag_size <= ep->mtu);
	psmi_assert(0 == (frag_size & 0x3) || length <= frag_size);

	while (length) {
		uint32_t len = min(length, frag_size);
		struct ips_message_header *hdr = (struct ips_message_header*)sbuf_gso;
		_HFI_VDBG("copy lrh %p\n", ips_lrh);
		memcpy(sbuf_gso, ips_lrh, sizeof(*ips_lrh));
		// patch up hdr's psn, ACKREQ, offset and len in sbuf_gso
		hdr->bth[2] = __cpu_to_be32(bth_w2 | (psn & proto->psn_mask)
				| (length == len?IPS_SEND_FLAG_ACKREQ:0));
		hdr->hdr_data.u32w0 = offset;
		// Note length includes ICRC but pretend HW added and removed it
		// so we never actually see it
		hdr->lrh[2] = ips_proto_bytes_to_lrh2_be(proto,
                                len + sizeof(*ips_lrh) + HFI_CRC_SIZE_IN_BYTES);

		_HFI_VDBG("copy payload %p %u\n", payload, len);
#ifdef PSM_CUDA
		if (is_cuda_payload) {
			PSMI_CUDA_CALL(cuMemcpyDtoH, sbuf_gso+sizeof(*ips_lrh),
					(CUdeviceptr)payload, len);
		} else
#endif
		{
			memcpy(sbuf_gso+sizeof(*ips_lrh), payload, len);
		}

		if_pf (_HFI_PDBG_ON) {
			_HFI_PDBG_ALWAYS("udp_transfer_frame: len %u, remote IP %s, payload %u\n",
				(unsigned)(len+sizeof(*ips_lrh)),
				psmi_sockaddr_fmt((struct sockaddr *)addr, 0),
				len);
			_HFI_PDBG_DUMP_ALWAYS(sbuf_gso, sizeof(*ips_lrh)+ len);
		}
		_HFI_VDBG("UDP send - opcode %x len %u\n",
					_get_proto_hfi_opcode(ips_lrh),
					(unsigned int)(sizeof(*ips_lrh) + len));

		gso_len += sizeof(*ips_lrh) + len;
		sbuf_gso += sizeof(*ips_lrh) + len;
		psn++;
		offset += len;
		payload += len;
		length -= len;
	}

	iov.iov_base = ep->sockets_ep.sbuf_udp_gso;
	iov.iov_len = gso_len;

	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;
	msg.msg_control = control;
	msg.msg_controllen = sizeof(control);
	msg.msg_name = addr;
	msg.msg_namelen = sizeof(struct sockaddr_in6);

	// specify how to segment (segment size)
	struct cmsghdr* cm = CMSG_FIRSTHDR(&msg);
	cm->cmsg_level = SOL_UDP;
	cm->cmsg_type = UDP_SEGMENT;
	cm->cmsg_len = CMSG_LEN(sizeof(uint16_t));
	uint16_t *pcmd = (uint16_t*)CMSG_DATA(cm);
	*pcmd = sizeof(*ips_lrh) + frag_size;

	if_pf (-1 == sendmsg(fd, &msg, ep->sockets_ep.udp_gso_zerocopy ? MSG_ZEROCOPY : 0))
		return -1;
	if (ep->sockets_ep.udp_gso_zerocopy) {
		// flush zerocopy
		memset(&msg, 0, sizeof(msg));
		while (1) {
			if_pt (-1 == recvmsg(fd, &msg, MSG_ERRQUEUE))
				return (errno == EAGAIN) ? 0 : -1;
			msg.msg_flags = 0;
		}
	}
	return 0;
}

// when called:
//		scb->ips_lrh has fixed size PSM header including OPA LRH
//		payload, length is data after header
//		we don't do checksum, let verbs handle that for us
//		the above is in unregistered memory, perhaps even on stack
// for isCtrlMsg, scb is only partially initialized (see ips_scb.h)
// and payload and length may refer to buffers on stack
static inline psm2_error_t
psm3_sockets_udp_spio_transfer_frame(struct ips_proto *proto, struct ips_flow *flow,
			struct ips_scb *scb, uint32_t *payload,
			uint32_t length, uint32_t isCtrlMsg,
			uint32_t cksum_valid, uint32_t cksum
#ifdef PSM_CUDA
			, uint32_t is_cuda_payload
#endif
			)
{
	psm2_error_t ret = PSM2_OK;
	psm2_ep_t ep = proto->ep;
	uint8_t *sbuf = ep->sockets_ep.sbuf;
	unsigned len;
	struct ips_message_header *ips_lrh = &scb->ips_lrh;
	len = sizeof(*ips_lrh) + length;

#ifdef PSM_FI
	if_pf(PSM3_FAULTINJ_ENABLED_EP(ep)) {
		PSM3_FAULTINJ_STATIC_DECL(fi_sendlost, "sendlost",
				"drop packet before sending",
				1, IPS_FAULTINJ_SENDLOST);
		if_pf(PSM3_FAULTINJ_IS_FAULT(fi_sendlost, ep, " UDP"))
			return PSM2_OK;
	}
#endif // PSM_FI
	PSMI_LOCK_ASSERT(proto->mq->progress_lock);
	psmi_assert_always(! cksum_valid);	// no software checksum yet
	// TBD - we should be able to skip sending some headers such as OPA lrh and
	// perhaps bth (does PSM use bth to hold PSNs? - yes)
	// copy scb->ips_lrh to send buffer
	// when called as part of retransmission:
	//	nfrag remains original total frags in message
	//	nfrag_remaining is fragments left to do (only valid if nfrag>1)
	// if nfrag>1 but nfrag_remaining == 1, don't need to use GSO
	// could just falthrough to sendto because payload, length and ACKREQ
	// are all properly set.  But GSO may allow zerocopy option, so use it.
	psmi_assert(isCtrlMsg || ips_scb_buffer(scb) == payload);
	if (scb->nfrag > 1 /* && scb->nfrag_remaining > 1 */) {
		// when nfrag>1, length and payload_size are just 1st pkt size
		psmi_assert(!isCtrlMsg);
		psmi_assert(length);
		psmi_assert(length <= scb->chunk_size_remaining);
		psmi_assert(scb->payload_size == length);
		if_pf (-1 == psm3_sockets_udp_gso_send(ep->sockets_ep.udp_tx_fd, proto,
					&flow->ipsaddr->sockets.remote_pri_addr,
					scb, (uint8_t*)payload, scb->chunk_size_remaining,
					scb->frag_size
#ifdef PSM_CUDA
					,is_cuda_payload
#endif
					)) {
			if (errno != EAGAIN && errno != EWOULDBLOCK) {
				_HFI_ERROR("UDP GSO send failed on %s: %s\n", ep->dev_name, strerror(errno));
				ret = PSM2_EP_NO_RESOURCES;
			}
		}
		return ret;
	}

	_HFI_VDBG("copy lrh %p\n", ips_lrh);
	memcpy(sbuf, ips_lrh, sizeof(*ips_lrh));
	// copy payload to send buffer, length could be zero, be safe
	_HFI_VDBG("copy payload %p %u\n",  payload, length);
#ifdef PSM_CUDA
	if (is_cuda_payload) {
		PSMI_CUDA_CALL(cuMemcpyDtoH, sbuf+sizeof(*ips_lrh),
				(CUdeviceptr)payload, length);
	} else
#endif
	{
		memcpy(sbuf+sizeof(*ips_lrh), payload, length);
	}
	_HFI_VDBG("UDP send - opcode %x len %u\n",
				_get_proto_hfi_opcode((struct  ips_message_header*)sbuf), len);
	// we don't support software checksum
	psmi_assert_always(! (proto->flags & IPS_PROTO_FLAG_CKSUM));

	if_pf (ips_lrh->khdr.kdeth0 & __cpu_to_le32(IPS_SEND_FLAG_INTR)) {
		_HFI_VDBG("send solicted event\n");
		// TBD - how to send so wake up rcvthread?  Separate socket?
	}

	if_pf (_HFI_PDBG_ON) {
		_HFI_PDBG_ALWAYS("udp_transfer_frame: len %u, remote IP %s payload %u\n",
			len,
			psmi_sockaddr_fmt((struct sockaddr *)&flow->ipsaddr->sockets.remote_pri_addr, 0),
			length);
		_HFI_PDBG_DUMP_ALWAYS(sbuf, len);
	}
	// UDP is datagram oriented, each send is delivered as a single datagram
	// this is unlike TCP which is bytestream oriented.
	if_pf (sendto(ep->sockets_ep.udp_tx_fd, sbuf, len, 0,
		&flow->ipsaddr->sockets.remote_pri_addr,
		sizeof(flow->ipsaddr->sockets.remote_pri_addr)) == -1) {
		if (errno != EAGAIN && errno != EWOULDBLOCK) {
			_HFI_ERROR("UDP send failed on %s: %s\n", ep->dev_name, strerror(errno));
			ret = PSM2_EP_NO_RESOURCES;
		}
	}
	return ret;
}
#endif /* PSM_SOCKETS */
#endif /* _SOCKETS_SPIO_C_ */
