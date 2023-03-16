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

#define SM2_COORDINATION_FILE "/dev/shm/fi_sm2_mmaps"
#define SM2_COORDINATION_DIR "/dev/shm"

extern struct dlist_entry sm2_ep_name_list;
extern pthread_mutex_t sm2_ep_list_lock;

struct sm2_region;

/* SMR op_src: Specifies data source location */
enum {
	sm2_src_inject,	/* inject buffers */
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
	// This is volatile for a reason, many things touch this
    volatile long int next;
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

struct sm2_free_queue_entry {
	struct sm2_protocol_hdr protocol_hdr;
	uint8_t data[SM2_INJECT_SIZE];
};

struct sm2_addr {
	char		name[SM2_NAME_MAX];
	int64_t		id;
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

struct sm2_region {
	uint8_t		version;
	uint8_t		resv;
	uint16_t	flags;

	/* offsets from start of sm2_region */
	ptrdiff_t	recv_queue_offset;   // Turns into our FIFO Queue offset
	ptrdiff_t	free_stack_offset;
};

struct sm2_attr {
	const char	*name;
	size_t		num_fqe;
	uint16_t	flags;
};

struct sm2_mmap {
        char *base;
        size_t size;
        int fd;
};

struct sm2_private_aux {
	fi_addr_t	cqfid;   // fi_addr to report during in cqe's if av_user_data is on
	int pid;             // This is for verify peer, make sure entry didn't get replaced with someone new
};

struct sm2_ep_allocation_entry {
    int pid;                       // This is for allocation startup
    char ep_name[SM2_NAME_MAX];
    uint32_t startup_ready;        // TODO Make atomic
};

struct sm2_coord_file_header {
	int             file_version;
        pthread_mutex_t write_lock;
        ofi_atomic32_t	pid_lock_hint;
        int             ep_region_size;
        int             ep_enumerations_max;

        ptrdiff_t       ep_enumerations_offset; /* struct sm2_ep_allocation_entry */
        ptrdiff_t       ep_regions_offset;      /* struct ep_region */

};

ssize_t sm2_mmap_unmap_and_close(struct sm2_mmap *map );
void* sm2_mmap_remap(struct sm2_mmap *map, size_t at_least );
void* sm2_mmap_map(int fd, struct sm2_mmap *map );
ssize_t sm2_coordinator_open_and_lock(struct sm2_mmap *map_shared);
ssize_t sm2_coordinator_allocate_entry(const char* name, struct sm2_mmap *map, int *av_key, bool self);
int sm2_coordinator_lookup_entry(const char* name, struct sm2_mmap *map);
ssize_t sm2_coordinator_free_entry(struct sm2_mmap *map, int av_key);
ssize_t sm2_coordinator_lock(struct sm2_mmap *map);
ssize_t sm2_coordinator_unlock(struct sm2_mmap *map);
void* sm2_coordinator_extend_for_entry(struct sm2_mmap *map, int last_valid_entry);
size_t sm2_calculate_size_offsets(ptrdiff_t num_fqe, ptrdiff_t *rq_offset,
				ptrdiff_t *mp_offset);
void sm2_cleanup(void);
int sm2_create(const struct fi_provider *prov, const struct sm2_attr *attr,
				struct sm2_mmap *sm2_mmap, int *id);
void sm2_free(struct sm2_region *smr);

static inline struct sm2_ep_allocation_entry *sm2_mmap_entries(struct sm2_mmap *map)
{
	struct sm2_coord_file_header *header = (void*) map->base;
	return (struct sm2_ep_allocation_entry *) (map->base + header->ep_enumerations_offset);
}

static inline struct sm2_fifo *sm2_recv_queue(struct sm2_region *smr)
{
	return (struct sm2_fifo *) ((char *) smr + smr->recv_queue_offset);
}
static inline struct smr_freestack *sm2_free_stack(struct sm2_region *smr)
{
	return (struct smr_freestack *) ((char *) smr + smr->free_stack_offset);
}

static inline struct sm2_region *sm2_mmap_ep_region(struct sm2_mmap *map, int id)
{
	struct sm2_coord_file_header *header = (void*)map->base;
	return (struct sm2_region *) (map->base + header->ep_regions_offset + header->ep_region_size*id);
}

/* @return True if pid is still alive. */
static inline bool pid_lives(int pid)
{
	int err = kill( pid, 0 );
	return err == 0;
}

static inline bool sm2_mapping_long_enough_check( struct sm2_mmap *map, int jentry ) {
	ptrdiff_t entry_offset = (char*)sm2_mmap_ep_region(map, jentry+1) - map->base;
	return entry_offset <= map->size;
}

static inline void* sm2_relptr_to_absptr(int64_t relptr, struct sm2_mmap *map)
{
	return (void*) (map->base + relptr);
}
static inline int64_t sm2_absptr_to_relptr(void *absptr, struct sm2_mmap *map)
{
	return (int64_t) ((char*)absptr - map->base);
}

#ifdef __cplusplus
}

#endif

#endif /* _OFI_SM2_COMMON_H_ */
