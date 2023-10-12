/*
 * Copyright (c) Intel Corporation. All rights reserved.
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

#ifndef _SMR_FIFO_H_
#define _SMR_FIFO_H_

#include <string.h>
#include "ofi.h"
#include "ofi_atom.h"
#include <rdma/fi_errno.h>

/* Multiple writer, single reader queue.
 * Writes are protected with atomics.
 * Reads are not protected and assumed to be protected with user locking
 */
struct smr_fifo {
	int64_t		size;
	int64_t		size_mask;
        int64_t		read_pos;
	ofi_atomic64_t	write_pos;
	ofi_atomic64_t	free;
	uintptr_t	entries[];
};

static inline void smr_fifo_init(struct smr_fifo *queue, uint64_t size)
{
	assert(size == roundup_power_of_two(size));
	queue->size = size;
	queue->size_mask = size - 1;
	queue->read_pos = 0;
	ofi_atomic_initialize64(&queue->write_pos, 0);
	ofi_atomic_initialize64(&queue->free, size);
	memset(queue->entries, 0, sizeof(*queue->entries) * size);
}

//TODO figure out memory barriers
static inline int smr_fifo_commit(struct smr_fifo *queue, uintptr_t val)
{
	int64_t free, write;

	for (;;) {
		free = ofi_atomic_load_explicit64(&queue->free,
						  memory_order_relaxed);
		if (!free)
			return -FI_ENOENT;
		if (ofi_atomic_compare_exchange_weak64(
			&queue->free, &free, free - 1))
			break;
	}
	write = ofi_atomic_inc64(&queue->write_pos) - 1;//TODO add atomic to remove sub
	queue->entries[write & queue->size_mask] = val;
	return FI_SUCCESS;
}

/* All read calls within the same process must be protected by the same lock */
static inline uintptr_t smr_fifo_read(struct smr_fifo *queue)
{
	uintptr_t val;

	val = queue->entries[queue->read_pos & queue->size_mask];
	if (!val)
		return 0;

	queue->entries[queue->read_pos++ & queue->size_mask] = 0;
	ofi_atomic_inc64(&queue->free);
	return val;
}

#endif
