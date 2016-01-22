/*
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
 * Copyright (c) 2015 Cray Inc. All rights reserved.
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
#include <config.h>
#endif /* HAVE_CONFIG_H */

#ifndef _GNIX_UTIL_H_
#define _GNIX_UTIL_H_

#include <stdio.h>
#include "rdma/fi_log.h"

extern struct fi_provider gnix_prov;

/*
 * For debug logging (#undef NDEBUG)
 * Q: should this just always be available?
 */
#ifdef NDEBUG

#define GNIX_LOG_INTERNAL(FI_LOG_FN, subsystem, fmt, ...)	\
	FI_LOG_FN(&gnix_prov, subsystem, fmt, ##__VA_ARGS__)

#define GNIX_FI_PRINT(prov, subsystem, ...)

#else

/* defined in gnix_init.c */
#include <fi.h> /* for atomic_t */
extern __thread pid_t gnix_debug_pid;
extern __thread uint32_t gnix_debug_tid;
extern atomic_t gnix_debug_next_tid;

#define GNIX_FI_PRINT(prov, subsystem, ...)				\
	do {								\
		fi_log(prov, FI_LOG_WARN, subsystem,			\
				__func__, __LINE__, __VA_ARGS__);	\
	} while (0)


/* These macros are used to prepend the log message with the pid and
 * unique thread id.  Do not use them directly.  Rather use the normal
 * GNIX_* macros.
 */
#define GNIX_LOG_INTERNAL(FI_LOG_FN, subsystem, fmt, ...)	\
	do {							\
		const int fmt_len = 256;			\
		char new_fmt[fmt_len];				\
		if (gnix_debug_tid  == ~(uint32_t) 0) {		\
			gnix_debug_tid = atomic_inc(&gnix_debug_next_tid); \
		}						\
		if (gnix_debug_pid == ~(uint32_t) 0) {		\
			gnix_debug_pid = getpid();		\
		}						\
		snprintf(new_fmt, fmt_len, "[%%d:%%d] %s", fmt);	\
		FI_LOG_FN(&gnix_prov, subsystem, new_fmt,		\
			  gnix_debug_pid, gnix_debug_tid, ##__VA_ARGS__); \
	} while (0)

#endif

#define GNIX_WARN(subsystem, ...)                                              \
	GNIX_LOG_INTERNAL(FI_WARN, subsystem, __VA_ARGS__)
#define GNIX_TRACE(subsystem, ...)                                             \
	GNIX_LOG_INTERNAL(FI_TRACE, subsystem, __VA_ARGS__)
#define GNIX_INFO(subsystem, ...)                                              \
	GNIX_LOG_INTERNAL(FI_INFO, subsystem, __VA_ARGS__)
#define GNIX_DEBUG(subsystem, ...)                                             \
	GNIX_LOG_INTERNAL(FI_DBG, subsystem, __VA_ARGS__)
#define GNIX_ERR(subsystem, ...)                                               \
	GNIX_LOG_INTERNAL(GNIX_FI_PRINT, subsystem, __VA_ARGS__)
#define GNIX_FATAL(subsystem, ...)                                             \
	do { \
		GNIX_LOG_INTERNAL(GNIX_FI_PRINT, subsystem, __VA_ARGS__); \
		abort(); \
	} while (0)

/* dlist utilities */
#include "fi_list.h"

static inline void dlist_node_init(struct dlist_entry *e)
{
	e->prev = e->next = NULL;
}

static inline void dlist_remove_init(struct dlist_entry *e)
{
	dlist_remove(e);
	e->prev = e->next = e;
}

#define DLIST_HEAD(dlist)  struct dlist_entry dlist = { &(dlist), &(dlist) }

#define dlist_entry(e, type, member) container_of(e, type, member)

#define dlist_first_entry(h, type, member)				\
	(dlist_empty(h) ? NULL : dlist_entry((h)->next, type, member))

/* Iterate over the entries in the list */
#define dlist_for_each(h, e, member)					\
	for (e = dlist_first_entry(h, typeof(*e), member);		\
	     e && (&e->member != h);					\
	     e = dlist_entry((&e->member)->next, typeof(*e), member))

/* Iterate over the entries in the list, possibly deleting elements */
#define dlist_for_each_safe(h, e, n, member)				\
	for (e = dlist_first_entry(h, typeof(*e), member),		\
		     n = e ? dlist_entry((&e->member)->next,		\
					 typeof(*e), member) : NULL;	\
	     e && (&e->member != h);					\
	     e = n, n = dlist_entry((&e->member)->next, typeof(*e), member))


#define rwlock_t pthread_rwlock_t
#define rwlock_init(lock) pthread_rwlock_init(lock, NULL)
#define rwlock_destroy(lock) pthread_rwlock_destroy(lock)
#define rwlock_wrlock(lock) pthread_rwlock_wrlock(lock)
#define rwlock_rdlock(lock) pthread_rwlock_rdlock(lock)
#define rwlock_unlock(lock) pthread_rwlock_unlock(lock)

/*
 * prototypes
 */
int gnixu_get_rdma_credentials(void *addr, uint8_t *ptag, uint32_t *cookie);
int gnixu_to_fi_errno(int err);

int _gnix_task_is_not_app(void);
int _gnix_job_enable_unassigned_cpus(void);
int _gnix_job_disable_unassigned_cpus(void);
int _gnix_job_enable_affinity_apply(void);
int _gnix_job_disable_affinity_apply(void);

void _gnix_alps_cleanup(void);
int _gnix_job_fma_limit(uint32_t dev_id, uint8_t ptag, uint32_t *limit);
int _gnix_job_cq_limit(uint32_t dev_id, uint8_t ptag, uint32_t *limit);
int _gnix_pes_on_node(uint32_t *num_pes);
int _gnix_nics_per_rank(uint32_t *nics_per_rank);

struct gnix_reference {
	atomic_t references;
	void (*destruct)(void *obj);
};

/* Should not be used unless the reference counting variable has a
 * non-standard name
 */
#define __ref_get(ptr, var) \
	({ \
		struct gnix_reference *ref = &ptr->var; \
		int references_held = atomic_inc(&ref->references); \
		assert(references_held > 0); \
		references_held; })

#define __ref_put(ptr, var) \
	({ \
		struct gnix_reference *ref = &ptr->var; \
		int references_held = atomic_dec(&ref->references); \
		assert(references_held >= 0); \
		if (references_held == 0) \
			ref->destruct((void *) (ptr)); \
		references_held; })

/* by default, all of the gnix reference counting variables are
 *   named 'ref_cnt'. The macros provided below will autofill the var arg.
 */
#define _gnix_ref_get(ptr) __ref_get(ptr, ref_cnt)
#define _gnix_ref_put(ptr) __ref_put(ptr, ref_cnt)

static inline void _gnix_ref_init(
		struct gnix_reference *ref,
		int initial_value,
		void (*destruct)(void *))
{
	atomic_initialize(&ref->references, initial_value);
	ref->destruct = destruct;
}



#endif
