/*
 * Copyright (c) 2022 UT-Battelle ORNL. All rights reserved
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

/* Code derived from Dmitry Vyukov */
/* see:  https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue */

/*  Multi-producer/multi-consumer bounded queue.
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *     1. Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *     2. Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY DMITRY VYUKOV "AS IS" AND ANY EXPRESS OR
 *  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 *  EVENT SHALL DMITRY VYUKOV OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 *  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 *  OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  The views and conclusions contained
 *  in the software and documentation are those of the authors and should not be
 *  interpreted as representing official policies, either expressed or implied, of
 *  Dmitry Vyukov.
 */

#ifndef OFI_ATOMIC_QUEUE_H
#define OFI_ATOMIC_QUEUE_H

#include <ofi_atom.h>

/*
 * This is an atomic queue, meaning no need for locking. One example usage
 * for this data structure is to build a command queue shared between
 * different processes. Multiple processes would post commands on a command
 * queue which belongs to another process. The receiving process would
 * read and process commands off the queue.
 *
 * Usage:
 *  . OFI_DECLARE_ATOMIC_Q() to declare the atomic queue
 *  . Allocate shared memory for the queue or call the create method
 *  . call the init() method to ready the queue for usage
 *  . To post on the queue call _next() method
 *     . This will return a buffer of entrytype
 *     . Initialize the entry
 *  . Call _commit() method to post for the reader
 *  . On failure instead of _commit() call _discard()
 *     . This will set the command as a no-op and put it on the queue
 *  . To read off the queue call _head()
 *     . This will return the next available entry on the queue or
 *       -FI_ENOENT if there are no more entries on the queue
 *     . if the entry is a no-op it will be released and another entry
 *       will be fetched off the queue.
 *  . Call _release() after reader is done with the entry
 */

#ifdef __cplusplus
extern "C" {
#endif

#define OFI_CACHE_LINE_SIZE (64)

typedef void (*ofi_aq_init_fn)(void *);
enum {
	OFI_AQ_FREE = 0,
	OFI_AQ_READY,
	OFI_AQ_NOOP,
};

/*
 * Base address of atomic queue must be cache line aligned to maximize atomic
 * value perforamnce benefits
 */
#define OFI_DECLARE_ATOMIC_Q(entrytype, name)			\
struct name ## _entry {						\
	ofi_atomic64_t	state;					\
	entrytype	buf;					\
} __attribute__((__aligned__(64)));				\
								\
struct name {							\
	ofi_atomic64_t	write_pos;				\
	uint8_t		pad0[OFI_CACHE_LINE_SIZE -		\
			     sizeof(ofi_atomic64_t)];		\
	int64_t		read_pos;				\
	ofi_aq_init_fn	init_fn;				\
	uint8_t		pad1[OFI_CACHE_LINE_SIZE -		\
			     (sizeof(int64_t) +			\
			     sizeof(ofi_aq_init_fn))];		\
	ofi_atomic64_t	claim_avail;				\
	uint8_t		pad2[OFI_CACHE_LINE_SIZE -		\
			     sizeof(ofi_atomic64_t)];		\
	ofi_atomic64_t	discard_avail;				\
	uint8_t		pad3[OFI_CACHE_LINE_SIZE -		\
			     sizeof(ofi_atomic64_t)];		\
	int		size;					\
	int		size_mask;				\
	uint8_t		pad4[OFI_CACHE_LINE_SIZE -		\
			     (sizeof(int) * 2)];		\
	struct name ## _entry entry[];				\
} __attribute__((__aligned__(64)));				\
								\
static inline void name ## _init(struct name *aq, size_t size,	\
				 ofi_aq_init_fn init_fn)	\
{								\
	size_t i;						\
	assert(size == roundup_power_of_two(size));		\
	assert(!((uintptr_t) aq % OFI_CACHE_LINE_SIZE));	\
	aq->size = size;					\
	aq->size_mask = aq->size - 1;				\
	aq->init_fn = init_fn;					\
	ofi_atomic_initialize64(&aq->write_pos, 0);		\
	aq->read_pos = 0;					\
	ofi_atomic_initialize64(&aq->discard_avail, 0);		\
	ofi_atomic_initialize64(&aq->claim_avail, size);	\
	for (i = 0; i < size; i++) {				\
		if (aq->init_fn)				\
			aq->init_fn(&aq->entry[i].buf);		\
		ofi_atomic_initialize64(&aq->entry[i].state, OFI_AQ_FREE);\
	}							\
}								\
								\
static inline struct name * name ## _create(size_t size)	\
{								\
	struct name *aq;					\
	aq = (struct name *) aligned_alloc(			\
			OFI_CACHE_LINE_SIZE, sizeof(*aq) +	\
			sizeof(struct name ## _entry) *		\
			(roundup_power_of_two(size)));		\
	if (aq)							\
		name ##_init(aq, roundup_power_of_two(size),	\
			     NULL);				\
	return aq;						\
}								\
								\
static inline void name ## _free(struct name *aq)		\
{								\
	free(aq);						\
}								\
static inline bool name ## _claim(struct name *aq)		\
{								\
	int64_t avail, discard_avail;				\
	avail = ofi_atomic_sub_explicit64(&aq->claim_avail, 1,	\
					  memory_order_relaxed);\
	if (avail > 0)						\
		return true;					\
								\
	discard_avail = ofi_atomic_load_explicit64(		\
					&aq->discard_avail,	\
					memory_order_acquire);	\
	if (discard_avail) {					\
		if (!ofi_atomic_compare_exchange_weak64(	\
					&aq->discard_avail,	\
					&discard_avail, 0))	\
			goto out;				\
		ofi_atomic_add_explicit64(&aq->claim_avail,	\
					  discard_avail,	\
					  memory_order_relaxed);\
	}							\
out:								\
	ofi_atomic_add_explicit64(&aq->claim_avail, 1,		\
				  memory_order_relaxed);	\
	return false;						\
								\
}								\
static inline entrytype *name ## _assign(struct name *aq)	\
{								\
	int64_t pos;						\
	while (1) {						\
		pos = ofi_atomic_load_explicit64(		\
					&aq->write_pos,		\
					memory_order_acquire);	\
		if (ofi_atomic_compare_exchange_weak64(		\
					&aq->write_pos, &pos,	\
					pos + 1))		\
			break;					\
	}							\
	return &aq->entry[pos & aq->size_mask].buf;		\
}								\
static inline entrytype *name ## _claim_assign(struct name *aq)	\
{								\
	if (name ## _claim(aq)) {				\
		return name ## _assign(aq);			\
	}							\
	return NULL;						\
}								\
static inline void name ## _release(struct name *aq,		\
				    entrytype *buf)		\
{								\
	int64_t state = OFI_AQ_FREE;				\
	struct name ## _entry *ce;				\
	ce = container_of(buf, struct name ## _entry, buf);	\
	if (aq->init_fn)					\
		aq->init_fn(&ce->buf);				\
	ofi_atomic_store_explicit64(&ce->state,	state,		\
				    memory_order_release);	\
	aq->read_pos++;						\
}								\
static inline void name ## _discard(struct name *aq)		\
{								\
	ofi_atomic_add_explicit64(&aq->discard_avail, 1,	\
		memory_order_relaxed);				\
}								\
static inline void name ## _release_discard(struct name *aq,	\
					    entrytype *buf)	\
{								\
	name ## _release(aq, buf);				\
	name ## _discard(aq);					\
}								\
static inline entrytype *name ## _head(struct name *aq)		\
{								\
	struct name ## _entry *ce;				\
	int64_t state;						\
again:								\
	ce = &aq->entry[aq->read_pos & aq->size_mask];		\
	state = ofi_atomic_load_explicit64(&ce->state,		\
					   memory_order_acquire);\
	if (state == OFI_AQ_FREE)				\
		return NULL;					\
	if (state == OFI_AQ_NOOP) {				\
		name ## _release_discard(aq, &ce->buf);		\
		goto again;					\
	}							\
	return &ce->buf;					\
}								\
static inline void name ## _commit(entrytype *buf)		\
{								\
	struct name ## _entry *ce;				\
	int64_t state = OFI_AQ_READY;				\
	ce = container_of(buf, struct name ## _entry, buf);	\
	ofi_atomic_store_explicit64(&ce->state, state,		\
				    memory_order_release);	\
}								\
static inline void name ## _cancel(entrytype *buf)		\
{								\
	struct name ## _entry *ce;				\
	ce = container_of(buf, struct name ## _entry, buf);	\
	ofi_atomic_store_explicit64(&ce->state, OFI_AQ_NOOP,	\
				    memory_order_release);	\
}								\
void dummy ## name (void) /* work-around global ; scope */

#ifdef __cplusplus
}
#endif

#endif /* OFI_ATOMIC_QUEUE_H */
