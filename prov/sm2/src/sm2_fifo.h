/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *	 Redistribution and use in source and binary forms, with or
 *	 without modification, are permitted provided that the following
 *	 conditions are met:
 *
 *	  - Redistributions of source code must retain the above
 *		copyright notice, this list of conditions and the following
 *		disclaimer.
 *
 *	  - Redistributions in binary form must reproduce the above
 *		copyright notice, this list of conditions and the following
 *		disclaimer in the documentation and/or other materials
 *		provided with the distribution.
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
#ifndef _SM2_FIFO_H_
#define _SM2_FIFO_H_

#include <stdbool.h>
#include <stdatomic.h>
#include <stdint.h>
#include "sm2_common.h"

#define SM2_FIFO_FREE -3

#define atomic_swap_ptr(addr, value) \
	atomic_exchange_explicit((_Atomic unsigned long *) addr, value, memory_order_relaxed)

#define atomic_compare_exchange(x, y, z) \
	__atomic_compare_exchange_n((int64_t *) (x), (int64_t *) (y), (int64_t)(z), \
								 false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)

// Multi Writer, Single Reader Queue (Not Thread Safe)
// This data structure must live in the SMR
// This implementation of this is a one directional linked list with head/tail pointers
// Every pointer is a relative offset into the Shared Memory Region

// TODO need to have FIFO Queue work with offsets instead of pointers

struct sm2_fifo {
	long int fifo_head;
	long int fifo_tail;
};


// TODO Remove PT2PT Hack to make Nemesis work for N writers to 1 receiver
// TODO Remove Owning Region, it is the hack
static inline long int virtual_addr_to_offset(struct sm2_region *owning_region,
        struct sm2_free_queue_entry *fqe) {
	return (long int) ((char *) fqe - (char *) owning_region);
}

// TODO Remove Owning Region, it is the hack
static inline struct sm2_free_queue_entry*
        offset_to_virtual_addr(struct sm2_region *owning_region,  long int fqe_offset)
{
	return (struct sm2_free_queue_entry *) ((char *) owning_region + fqe_offset);
}

// Initialize FIFO queue to empty state
static inline void sm2_fifo_init(struct sm2_fifo *fifo)
{
	fifo->fifo_head = SM2_FIFO_FREE;
	fifo->fifo_tail = SM2_FIFO_FREE;
}

static inline bool sm2_fifo_empty(struct sm2_fifo* fifo)
{
	if (fifo->fifo_head == SM2_FIFO_FREE)
		return true;
	return false;
}

/* Write, Enqueue */
// TODO Remove Owning Region, it is the pt2pt only hack
// TODO Add Memory Barriers Back In
// TODO Verify This is correct
static inline void sm2_fifo_write(struct sm2_fifo *fifo, struct sm2_region *owning_region,
        struct sm2_free_queue_entry *fqe)
{
	struct sm2_free_queue_entry *prev_fqe;
	long int offset = virtual_addr_to_offset(owning_region, fqe);
	long int prev;

	// Set next pointer to NULL
	fqe->nemesis_hdr.next = SM2_FIFO_FREE;

	prev = atomic_swap_ptr(&fifo->fifo_tail, offset);

	assert(prev != offset);

	if (OFI_LIKELY(SM2_FIFO_FREE != prev)) {
		prev_fqe = offset_to_virtual_addr(owning_region, prev);
		prev_fqe->nemesis_hdr.next = offset;
	} else {
		fifo->fifo_head = offset;
	}
}

/* Read, Dequeue */
// TODO Remove Owning Region, it is the pt2pt only hack
// TODO Add Memory Barriers Back In
// TODO Verify This is correct
static inline struct sm2_free_queue_entry* sm2_fifo_read(struct sm2_fifo *fifo,
        struct sm2_region *owning_region)
{
	struct sm2_free_queue_entry* fqe;
	long int prev_head;

	if (SM2_FIFO_FREE == fifo->fifo_head) {
		return NULL;
	}

	prev_head = fifo->fifo_head;
	fqe = offset_to_virtual_addr(owning_region, fifo->fifo_head);

	fifo->fifo_head = SM2_FIFO_FREE;

	assert(fqe->nemesis_hdr.next != prev_head);

	if (OFI_UNLIKELY(SM2_FIFO_FREE == fqe->nemesis_hdr.next)) {
		if (!atomic_compare_exchange(&fifo->fifo_tail, &prev_head, SM2_FIFO_FREE)) {
			while (SM2_FIFO_FREE == fqe->nemesis_hdr.next) {}
			fifo->fifo_head = fqe->nemesis_hdr.next;
		}
	} else {
		fifo->fifo_head = fqe->nemesis_hdr.next;
	}

	return fqe;
}

// TODO Remove Owning Region
// TODO use nemesis send instead of putting back on FQE
static inline void sm2_fifo_write_back(struct sm2_free_queue_entry *fqe,
        struct sm2_region *owning_region)
{
	fqe->protocol_hdr.op_src = sm2_buffer_return;

	// This is really bad b/c it will make our performance seem better than it actually is!
	// Can't take this out b/c in sm2_progress_recv(), we have an assumption that the buffer
	// that is sent to us is always from a peer.
	// Three Fixes:
	// 1. Fix Hack (long term)
	// 2. Use a separate fifo queue for returning buffers (which we only call when we are out)
	//	 BAD Hurts  performance
	//	 BAD more work for something we will just delete later
	// Do what we did here... cheap easy, except it makes us think we have better performance than we actually do

	smr_freestack_push(sm2_free_stack(owning_region), fqe);

	// sm2_fifo_write(sm2_recv_queue(owning_region), owning_region, fqe);
}

#endif /* _SM2_FIFO_H_ */
