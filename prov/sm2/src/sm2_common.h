/*
 * Copyright (c) 2016-2021 Intel Corporation. All rights reserved.
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

#ifndef _OFI_SM2_H_
#define _OFI_SM2_H_

#include "config.h"

#include <stdint.h>
#include <stddef.h>
#include <sys/un.h>

#include <ofi_atom.h>
#include <ofi_proto.h>
#include <ofi_mem.h>
#include <ofi_rbuf.h>
#include <ofi_tree.h>
#include <ofi_hmem.h>

#include <rdma/providers/fi_prov.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SM2_VERSION	5

//reserves 0-255 for defined ops and room for new ops
//256 and beyond reserved for ctrl ops
#define SM2_OP_MAX (1 << 8)
#define SM2_REMOTE_CQ_DATA	(1 << 0)
#define SM2_TX_COMPLETION	(1 << 2)
#define SM2_RX_COMPLETION	(1 << 3)

#define SM2_INJECT_SIZE		4096
#define SM2_MAX_PEERS	256

#define SM2_DIR "/dev/shm/"
#define SM2_NAME_MAX	256
#define SM2_PATH_MAX	(SM2_NAME_MAX + sizeof(SM2_DIR))

extern struct dlist_entry sm2_ep_name_list;
extern pthread_mutex_t sm2_ep_list_lock;

struct sm2_region;

/* SMR op_src: Specifies data source location */
enum {
	sm2_src_inject,	/* inject buffers */
	sm2_src_max,
};

//reserves 0-255 for defined ops and room for new ops
//256 and beyond reserved for ctrl ops
#define SM2_OP_MAX (1 << 8)

#define SM2_REMOTE_CQ_DATA	(1 << 0)
#define SM2_TX_COMPLETION	(1 << 2)
#define SM2_RX_COMPLETION	(1 << 3)
#define SM2_MULTI_RECV		(1 << 4)

/*
 * Unique sm2_op_hdr for smr message protocol:
 * 	addr - local shm_id of peer sending msg (for shm lookup)
 * 	op - type of op (ex. ofi_op_msg, defined in ofi_proto.h)
 * 	op_src - msg src (ex. sm2_src_inline, defined above)
 * 	op_flags - operation flags (ex. SM2_REMOTE_CQ_DATA, defined above)
 * 	src_data - src of additional op data (inject offset / resp offset)
 * 	data - remote CQ data
 */
struct sm2_msg_hdr {
	uint64_t		msg_id;
	int64_t			id;
	uint32_t		op;
	uint16_t		op_src;
	uint16_t		op_flags;

	uint64_t		size;
	uint64_t		src_data;
	uint64_t		data;
	union {
		uint64_t	tag;
		struct {
			uint8_t	datatype;
			uint8_t	atomic_op;
		};
	};
};

struct sm2_cmd_msg {
	struct sm2_msg_hdr	hdr;
	union sm2_cmd_data	data;
};

struct sm2_addr {
	char		name[SM2_NAME_MAX];
	int64_t		id;
};

struct sm2_peer_data {
	struct sm2_addr		addr;
	uint32_t		name_sent;
};

struct sm2_ep_name {
	char name[SM2_NAME_MAX];
	struct sm2_region *region;
	struct dlist_entry entry;
};

struct sm2_peer {
	struct sm2_addr		peer;
	fi_addr_t		fiaddr;
	struct sm2_region	*region;
};

struct sm2_map {
	ofi_spin_t		lock;
	int64_t			cur_id;
	int 			num_peers;
	uint16_t		flags;
	struct ofi_rbmap	rbmap;
	struct sm2_peer		peers[SM2_MAX_PEERS];
};

struct sm2_region {
	uint8_t		version;
	uint8_t		resv;
	uint16_t	flags;
	int		pid;
	void		*base_addr;
	struct sm2_map	*map;

	size_t		total_size;

	/* offsets from start of sm2_region */
	size_t		cmd_queue_offset;   // Turns int our FIFO Queue offset
	size_t		inject_pool_offset; // Turns into our Free Queue Offset
	size_t		peer_data_offset;   // IDK what this is for, maybe for holding map of peers?
	size_t		name_offset;
};

struct sm2_inject_buf {
	union {
		uint8_t		data[SM2_INJECT_SIZE];
		struct {
			uint8_t	buf[SM2_COMP_INJECT_SIZE];
			uint8_t comp[SM2_COMP_INJECT_SIZE];
		};
	};
};

enum sm2_status {
	SM2_STATUS_SUCCESS = 0, 	/* success*/
	SM2_STATUS_BUSY = FI_EBUSY, 	/* busy */

	SM2_STATUS_OFFSET = 1024, 	/* Beginning of shm-specific codes */
};

OFI_DECLARE_CIRQUE(struct sm2_cmd, sm2_cmd_queue);

static inline struct sm2_region *sm2_peer_region(struct sm2_region *smr, int i)
{
	return smr->map->peers[i].region;
}
static inline struct sm2_cmd_queue *sm2_cmd_queue(struct sm2_region *smr)
{
	return (struct sm2_cmd_queue *) ((char *) smr + smr->cmd_queue_offset);
}
static inline struct smr_freestack *sm2_inject_pool(struct sm2_region *smr)
{
	return (struct smr_freestack *) ((char *) smr + smr->inject_pool_offset);
}
static inline struct sm2_peer_data *sm2_peer_data(struct sm2_region *smr)
{
	return (struct sm2_peer_data *) ((char *) smr + smr->peer_data_offset);
}

static inline const char *sm2_name(struct sm2_region *smr)
{
	return (const char *) smr + smr->name_offset;
}

static inline void sm2_set_map(struct sm2_region *smr, struct sm2_map *map)
{
	smr->map = map;
}

#ifdef __cplusplus
}
#endif

#endif /* _OFI_SM2_H_ */
