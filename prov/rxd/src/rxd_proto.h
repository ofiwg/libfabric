/*
 * Copyright (c) 2015-2018 Intel Corporation, Inc.  All rights reserved.
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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <ofi.h>
#include <ofi_proto.h>

#ifndef _RXD_PROTO_H_
#define _RXD_PROTO_H_

#define RXD_IOV_LIMIT		4
#define RXD_NAME_LENGTH		64

enum rxd_pkt_type {
	RXD_MSG			= ofi_op_msg,
	RXD_TAGGED		= ofi_op_tagged,
	RXD_READ_REQ		= ofi_op_read_req,
	RXD_WRITE		= ofi_op_write,
	RXD_ATOMIC		= ofi_op_atomic,
	RXD_ATOMIC_FETCH	= ofi_op_atomic_fetch,
	RXD_ATOMIC_COMPARE	= ofi_op_atomic_compare,
	RXD_RTS,
	RXD_CTS,
	RXD_ACK,
	RXD_DATA,
	RXD_DATA_READ,
	RXD_OP_INLINE,
	RXD_NO_OP,
};

struct rxd_base_hdr {
	uint8_t		version;
	uint8_t		type;
	uint16_t	flags;
	uint32_t	peer;
	uint64_t	seq_no;
};

struct rxd_ext_hdr {
	uint16_t	tx_id;
	uint16_t	rx_id;
	uint32_t	seg_no;
};

struct rxd_rts_pkt {
	struct rxd_base_hdr	base_hdr;
	uint64_t		dg_addr;
	uint8_t			source[RXD_NAME_LENGTH];
};

struct rxd_cts_pkt {
	struct	rxd_base_hdr	base_hdr;
	uint64_t		dg_addr;
	uint64_t		peer_addr;
};

struct rxd_ack_pkt {
	struct rxd_base_hdr	base_hdr;
	struct rxd_ext_hdr	ext_hdr;
	//TODO fill in more fields? Selective ack?
};

struct rxd_inline_pkt {
	struct rxd_base_hdr	base_hdr;
	char			msg[];
};

struct rxd_op_hdr {
	uint64_t		cq_data;
	uint64_t		size;
	uint32_t		num_segs;
	uint16_t		flags;
	uint16_t		tx_id;
};

struct rxd_msg_pkt {
	struct rxd_base_hdr	base_hdr;
	struct rxd_op_hdr	op_hdr;

	uint64_t		tag;

	char			msg[];
};

struct rxd_rma_pkt {
	struct rxd_base_hdr	base_hdr;
	struct rxd_op_hdr	op_hdr;

	uint64_t		iov_count;

	struct ofi_rma_iov	rma[RXD_IOV_LIMIT];

	char			msg[];
};

struct rxd_atom_pkt {
	struct rxd_base_hdr	base_hdr;
	struct rxd_op_hdr	op_hdr;

	uint32_t		datatype;
	uint32_t		atomic_op;
	uint64_t		iov_count;

	struct ofi_rma_iov	rma[RXD_IOV_LIMIT];

	char			msg[];
};

struct rxd_data_pkt {
	struct rxd_base_hdr	base_hdr;
	struct rxd_ext_hdr	ext_hdr;

	char			msg[];
};

#endif
