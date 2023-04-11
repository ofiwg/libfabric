/*
 * Copyright (c) 2016-2021 Intel Corporation. All rights reserved.
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#ifndef _OFI_SM2_COMMON_H_
#define _OFI_SM2_COMMON_H_

#include "config.h"
#include "sm2_coordination.h"

#include <stddef.h>
#include <stdint.h>
#include <sys/un.h>

#include <ofi_atom.h>
#include <ofi_hmem.h>
#include <ofi_mem.h>
#include <ofi_proto.h>
#include <ofi_rbuf.h>
#include <ofi_tree.h>

#include <rdma/providers/fi_prov.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SM2_VERSION 5
#define SM2_IOV_LIMIT 4

// reserves 0-255 for defined ops and room for new ops
// 256 and beyond reserved for ctrl ops
#define SM2_REMOTE_CQ_DATA (1 << 0)
#define SM2_TX_COMPLETION  (1 << 2)
#define SM2_RX_COMPLETION  (1 << 3)

extern struct dlist_entry sm2_ep_name_list;
extern pthread_mutex_t sm2_ep_list_lock;

struct sm2_region;

/* SMR op_src: Specifies data source location */
enum {
	sm2_src_inject,
	sm2_buffer_return,
	sm2_src_max,
};

/*
 * Unique sm2_op_hdr for smr message protocol:
 * 	addr - local shm_id of peer sending msg (for shm lookup)
 * 	op - type of op (ex. ofi_op_msg, defined in ofi_proto.h)
 * 	op_src - msg src (ex. sm2_src_inline, defined above)
 * 	op_flags - operation flags (ex. SM2_REMOTE_CQ_DATA, defined above)
 * 	src_data - src of additional op data (inject offset / resp offset)
 * 	data - remote CQ data
 */
struct sm2_protocol_hdr {
	/*
	 * This is volatile for a reason, many things touch this
	 * and we do not want compiler optimization here
	 */
	volatile long int next; /* fifo linked list next ptr*/
	int64_t id; /* id of msg sender*/
	uint32_t op; /* fi operation */
	uint16_t op_src; /* sm2 operation */
	uint32_t op_flags; /* flags associated with op */
	uint64_t size; /* Holds total size of message */
	uint64_t data;
	uint64_t tag; /* used for tagged messages */
};

struct sm2_free_queue_entry {
	struct sm2_protocol_hdr protocol_hdr;
	uint8_t data[SM2_INJECT_SIZE];
};

struct sm2_addr {
	char name[SM2_NAME_MAX];
	int64_t id;
};

struct sm2_ep_name {
	char name[SM2_NAME_MAX];
	struct sm2_region *region;
	struct dlist_entry entry;
};

struct sm2_peer {
	struct sm2_addr peer;
	fi_addr_t fiaddr;
	struct sm2_region *region;
};

static inline struct sm2_fifo *
sm2_recv_queue(struct sm2_region *smr)
{
	return (struct sm2_fifo *) ((char *) smr + smr->recv_queue_offset);
}
static inline struct smr_freestack *
sm2_free_stack(struct sm2_region *smr)
{
	return (struct smr_freestack *) ((char *) smr + smr->free_stack_offset);
}

#ifdef __cplusplus
}

#endif

#endif /* _OFI_SM2_COMMON_H_ */
