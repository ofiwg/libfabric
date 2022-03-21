#ifdef PSM_SOCKETS
/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2015 Intel Corporation.

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

  Copyright(c) 2015 Intel Corporation.

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

/* Copyright (c) 2003-2015 Intel Corporation. All rights reserved. */


#ifndef _PSMI_IN_USER_H
#error "hal_sockets/sockets_ep.h not meant to be included directly, include psm_user.h instead"
#endif

#ifndef PSM_HAL_SOCKETS_EP_H
#define PSM_HAL_SOCKETS_EP_H

#include <netinet/in.h>

#ifdef RNDV_MOD
#ifdef PSM_CUDA
#include <infiniband/verbs.h>
#include <psm_rndv_mod.h>
#endif
#endif

#ifndef SOL_UDP
#define SOL_UDP 17
#endif

#ifndef UDP_SEGMENT
#define UDP_SEGMENT 103
#endif

#ifndef UDP_GRO
#define UDP_GRO 104
#endif

#ifndef SO_ZEROCOPY
#define SO_ZEROCOPY 60
#endif

#ifndef MSG_ZEROCOPY
#define MSG_ZEROCOPY 0x4000000
#endif

#ifndef UDP_MAX_SEGMENTS
#define UDP_MAX_SEGMENTS (1 << 6UL)
#endif

/* mode for EP as selected by PSM3_SOCKETS */
#define PSM3_SOCKETS_TCP 0
#define PSM3_SOCKETS_UDP 1

#define MAX_PSM_HEADER 64			// sizeof(ips_lrh) == 56, round up to 64

#define NETDEV_PORT 1			// default port if not specified

#define BUFFER_HEADROOM 0		// how much extra to allocate in buffers
								// as a paranoid headroom for use of more than
								// intended.  Was 64, but seems we can do
								// without it and hence make buffers better
								// page aligned
								// value here should be a multiple of CPU
								// cache size
#define CPU_PAGE_ALIGN	PSMI_PAGESIZE	// boundary to align buffer pools for

#include <sys/poll.h>

#define TCP_PORT_AUTODETECT 0			// default is to allow system to use any port
#define TCP_PORT_HIGH	TCP_PORT_AUTODETECT	// default TCP port range high end
#define TCP_PORT_LOW	TCP_PORT_AUTODETECT	// default TCP port range low end
#define TCP_BACKLOG	1024			// backlog for socket.
#define TCP_INI_CONN	1024			// initial fds array (see psm3_sockets_ep) size
#define TCP_INC_CONN	128			// fds array grow size
#define TCP_POLL_TO	1000			// timeout for continuous poll in ms. used when no more data in
						// the middle of draining a packet.
#define TCP_MAX_PKTLEN	((64*1024-1)*4)	// pktlen in LRH is 16 bits, so the
										// max pktlen is (64k-1)*4 = 256k-4
#define TCP_MAX_MTU (TCP_MAX_PKTLEN - MAX_PSM_HEADER)
#define TCP_DEFAULT_MTU (64*1024)

// this structure can be part of psm2_ep
// one instance of this per local end point (NIC)
// we will create a single UDP socket with related resources to
// permit an eager data movement mechanism
// conceptually similar to a psmi_context_t which refers to an HFI context
// TODO - later could optimize cache hit rates by putting some of the less
// frequently used fields in a different part of psm2_ep struct
struct psm3_sockets_ep {
	unsigned sockets_mode;	// PSM3_SOCKETS_TCP or PSM3_SOCKETS_UDP
	int udp_rx_fd;	// SOCK_DGRAM
	int udp_tx_fd;	// SOCK_DGRAM
	/* fields specific to TCP */
	int listener_fd; // listening socket
	int tcp_incoming_fd; // latest incoming socket
	struct pollfd *fds; // one extra for listening socket
	int nfds;
	int max_fds;
	uint32_t snd_pace_thresh; // send pace threshold
	/* fields specific to UDP */
	int udp_gso;	// is GSO enabled for UDP
	uint8_t *sbuf_udp_gso;	// buffer to compose UDP GSO packet sequence
	int udp_gso_zerocopy;	// is UDP GSO Zero copy option enabled
	int udp_gro; // will be used later
	/* fields used for both UDP and TCP */
	uint8_t *sbuf;
	uint8_t *rbuf;
	uint32_t buf_size;
	uint32_t max_buffering;	// max send/recv side buffering below us
	uint32_t if_index;	// index of our local netdev
	in_port_t pri_socket;	// primary socket, UDP/TCP based on sockets_mode
	in_port_t aux_socket;	// for TCP only: aux UDP socket
	int if_mtu;
	short if_flags;
	// if asked to revisit a packet we save it here
	uint8_t *revisit_buf;
	int revisit_fd;
	uint32_t revisit_payload_size;

	/* remaining fields are for TCP only */
	// read in partial pkt in rbuf
	int rbuf_cur_fd; // socket to continue read
	uint32_t rbuf_cur_offset; // position in rbuf to continue read
	uint32_t rbuf_cur_payload; // expected cur pkt payload size

	// multiple pkts in rbuf, i.e. has extra data in rbuf
	int rbuf_next_fd; // socket to continue read if last pkt is partial
	uint32_t rbuf_next_offset; // position in rbuf for the next pkt
	uint32_t rbuf_next_len; // total length of the extra data

	// send out partial pkt from sbuf
	struct ips_flow *sbuf_flow; // the flow where we will continue sending data
	uint32_t sbuf_offset; // position in sbuf to continue pkt send
	uint32_t sbuf_remainder; // length of remainder data to send out
};

extern psm2_error_t psm3_ep_open_sockets(psm2_ep_t ep, int unit, int port,
			psm2_uuid_t const job_key);
extern void psm3_hfp_sockets_context_initstats(psm2_ep_t ep);
extern void psm3_ep_free_sockets(psm2_ep_t ep);
extern psm2_error_t psm3_sockets_ips_proto_init(struct ips_proto *proto,
			uint32_t cksum_sz);
extern psm2_error_t psm3_sockets_ips_proto_update_linkinfo(
			struct ips_proto *proto);
extern int psm3_sockets_poll_type(int poll_type, psm2_ep_t ep);
extern psm2_error_t psm3_tune_tcp_socket(const char *sck_name, psm2_ep_t ep, int fd);
extern unsigned psm3_sockets_parse_inet(int reload);

#endif // PSM_HAL_SOCKETS_EP_H
#endif // PSM_SOCKETS
