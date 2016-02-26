/*
 * Copyright (c) 2016 Intel Corporation. All rights reserved.
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

#ifndef _FI_PROTO_H_
#define _FI_PROTO_H_

#include "config.h"

#include <stdint.h>
#include <stddef.h>

#include <rdma/fi_rma.h>


#ifdef __cplusplus
extern "C" {
#endif


enum {
	ofi_op_msg,
	ofi_op_tagged,
	ofi_op_rma,
	ofi_op_atomic,
	ofi_op_ctrl
};

enum {
	ofi_ctrl_inline,
	ofi_ctrl_inject,
	ofi_ctrl_iov,
	ofi_ctrl_ack
};

/* Keep data items and size aligned */
struct ofi_cmd_hdr {
	uint8_t			version_rxid;
	uint8_t			op_ctrl;
	uint16_t		req_id;
	uint32_t		flags;

	uint64_t		size;
	uint64_t		data;
	union {
		uint64_t	tag;
		uint8_t		iov_len;
		struct {
			uint8_t	datatype;
			uint8_t	op;
			uint8_t ioc_len;
		} atomic;
		uint64_t	resv;
	};
};

struct ofi_iov {
	uint64_t		addr;
	uint64_t		len;
};

struct ofi_rma_iov {
	uint64_t		addr;
	uint64_t		len;
	uint64_t		key;
};

struct ofi_rma_ioc {
	uint64_t		addr;
	uint64_t		count;
	uint64_t		key;
};

#define OFI_CMD_SIZE		64
#define OFI_CMD_DATA_LEN	(OFI_CMD_SIZE - sizeof(struct ofi_cmd_hdr))

struct ofi_cmd {
	struct ofi_cmd_hdr	hdr;
	union {
		uint8_t		data[OFI_CMD_DATA_LEN];
		uint64_t	buf;
		struct iovec	iov[OFI_CMD_DATA_LEN / sizeof(struct iovec)];
	};
};

/* Align with sizeof struct ofi_cmd */
union ofi_cmd_data {
	uint8_t			msg[OFI_CMD_SIZE];
	struct iovec		iov[OFI_CMD_SIZE / sizeof(struct iovec)];
	struct ofi_rma_iov	rma_iov[OFI_CMD_SIZE / sizeof(struct ofi_rma_iov)];
	struct ofi_rma_ioc	rma_ioc[OFI_CMD_SIZE / sizeof(struct ofi_rma_ioc)];
};


#ifdef __cplusplus
}
#endif

#endif /* _FI_PROTO_H_ */
