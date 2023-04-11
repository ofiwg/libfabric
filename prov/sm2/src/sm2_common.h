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

#define SM2_FLAG_ATOMIC	      (1 << 0)
#define SM2_FLAG_DEBUG	      (1 << 1)
#define SM2_FLAG_IPC_SOCK     (1 << 2)
#define SM2_FLAG_HMEM_ENABLED (1 << 3)
#define SM2_CMD_SIZE	      256 /* align with 64-byte cache line */
// reserves 0-255 for defined ops and room for new ops
// 256 and beyond reserved for ctrl ops
#define SM2_REMOTE_CQ_DATA (1 << 0)
#define SM2_TX_COMPLETION  (1 << 2)
#define SM2_RX_COMPLETION  (1 << 3)

#define SM2_CMD_SIZE 256 /* align with 64-byte cache line */
extern struct dlist_entry sm2_ep_name_list;
extern pthread_mutex_t sm2_ep_list_lock;

struct sm2_region;

/* SMR op_src: Specifies data source location */
enum {
	sm2_src_inject,
	sm2_buffer_return,
	sm2_src_max,
};

// reserves 0-255 for defined ops and room for new ops
// 256 and beyond reserved for ctrl ops
#define SM2_OP_MAX (1 << 8)

#define SM2_REMOTE_CQ_DATA (1 << 0)
#define SM2_RMA_REQ	   (1 << 1)
#define SM2_TX_COMPLETION  (1 << 2)
#define SM2_RX_COMPLETION  (1 << 3)
#define SM2_MULTI_RECV	   (1 << 4)

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

/*
 * Unique smr_op_hdr for smr message protocol:
 * 	addr - local shm_id of peer sending msg (for shm lookup)
 * 	op - type of op (ex. ofi_op_msg, defined in ofi_proto.h)
 * 	op_src - msg src (ex. smr_src_inline, defined above)
 * 	op_flags - operation flags (ex. SMR_REMOTE_CQ_DATA, defined above)
 * 	src_data - src of additional op data (inject offset / resp offset)
 * 	data - remote CQ data
 */
struct sm2_msg_hdr {
	uint64_t msg_id;
	int64_t id;
	uint32_t op;
	uint16_t op_src;
	uint16_t op_flags;

	uint64_t size;
	uint64_t src_data;
	uint64_t data;
	union {
		uint64_t tag;
		struct {
			uint8_t datatype;
			uint8_t atomic_op;
		};
	};
} __attribute__((aligned(16)));

#define SM2_BUF_BATCH_MAX 64
#define SM2_MSG_DATA_LEN  (SM2_CMD_SIZE - sizeof(struct sm2_msg_hdr))

union sm2_cmd_data {
	uint8_t msg[SM2_MSG_DATA_LEN];
	struct {
		size_t iov_count;
		struct iovec iov[(SM2_MSG_DATA_LEN - sizeof(size_t)) /
				 sizeof(struct iovec)];
	};
	struct {
		uint32_t buf_batch_size;
		int16_t sar[SM2_BUF_BATCH_MAX];
	};
	struct ipc_info ipc_info;
};

struct sm2_cmd_msg {
	struct sm2_msg_hdr hdr;
	union sm2_cmd_data data;
};

#define SM2_RMA_DATA_LEN (128 - sizeof(uint64_t))
struct sm2_cmd_rma {
	uint64_t rma_count;
	union {
		struct fi_rma_iov
			rma_iov[SM2_RMA_DATA_LEN / sizeof(struct fi_rma_iov)];
		struct fi_rma_ioc
			rma_ioc[SM2_RMA_DATA_LEN / sizeof(struct fi_rma_ioc)];
	};
};

struct sm2_cmd {
	union {
		struct sm2_cmd_msg msg;
		struct sm2_cmd_rma rma;
	};
};

#define SM2_INJECT_SIZE	     4096
#define SM2_COMP_INJECT_SIZE (SM2_INJECT_SIZE / 2)
#define SM2_SAR_SIZE	     32768

#define SM2_DIR		  "/dev/shm/"
#define SM2_NAME_MAX	  256
#define SM2_PATH_MAX	  (SM2_NAME_MAX + sizeof(SM2_DIR))
#define SM2_SOCK_NAME_MAX sizeof(((struct sockaddr_un *) 0)->sun_path)

struct sm2_addr {
	char name[SM2_NAME_MAX];
	int64_t id;
};

struct sm2_peer_data {
	struct sm2_addr addr;
	uint32_t sar_status;
	uint32_t name_sent;
};

extern struct dlist_entry sm2_ep_name_list;
extern pthread_mutex_t sm2_ep_list_lock;
extern struct dlist_entry sm2_sock_name_list;
extern pthread_mutex_t sm2_sock_list_lock;

struct sm2_region;

struct sm2_ep_name {
	char name[SM2_NAME_MAX];
	struct sm2_region *region;
	struct dlist_entry entry;
};

static inline const char *
sm2_no_prefix(const char *addr)
{
	char *start;

	return (start = strstr(addr, "://")) ? start + 3 : addr;
}

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
