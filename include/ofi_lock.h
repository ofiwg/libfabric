/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
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

#ifndef _OFI_LOCK_H_
#define _OFI_LOCK_H_

#include "config.h"

#include <assert.h>
#include <pthread.h>
#include <stdlib.h>

#include <ofi_osd.h>


#ifdef __cplusplus
extern "C" {
#endif


int fi_wait_cond(pthread_cond_t *cond, pthread_mutex_t *mut, int timeout_ms);


#if PT_LOCK_SPIN == 1

#define fastlock_t_ pthread_spinlock_t
#define fastlock_init_(lock) pthread_spin_init(lock, PTHREAD_PROCESS_PRIVATE)
#define fastlock_destroy_(lock) pthread_spin_destroy(lock)
#define fastlock_acquire_(lock) pthread_spin_lock(lock)
#define fastlock_tryacquire_(lock) pthread_spin_trylock(lock)
#define fastlock_release_(lock) pthread_spin_unlock(lock)

#else

#define fastlock_t_ pthread_mutex_t
#define fastlock_init_(lock) pthread_mutex_init(lock, NULL)
#define fastlock_destroy_(lock) pthread_mutex_destroy(lock)
#define fastlock_acquire_(lock) pthread_mutex_lock(lock)
#define fastlock_tryacquire_(lock) pthread_mutex_trylock(lock)
#define fastlock_release_(lock) pthread_mutex_unlock(lock)

#endif /* PT_LOCK_SPIN */

#if ENABLE_DEBUG

typedef struct {
	fastlock_t_ impl;
	int is_initialized;
	int in_use;
} fastlock_t;

static inline int fastlock_init(fastlock_t *lock)
{
	int ret;

	ret = fastlock_init_(&lock->impl);
	lock->is_initialized = !ret;
	lock->in_use = 0;

	return ret;
}

static inline void fastlock_destroy(fastlock_t *lock)
{
	int ret;

	assert(lock->is_initialized);
	lock->is_initialized = 0;
	ret = fastlock_destroy_(&lock->impl);
	assert(!ret);
}

static inline void fastlock_acquire(fastlock_t *lock)
{
	int ret;

	assert(lock->is_initialized);
	ret = fastlock_acquire_(&lock->impl);
	assert(!ret);
	lock->in_use++;
}

static inline int fastlock_tryacquire(fastlock_t *lock)
{
	int ret;

	assert(lock->is_initialized);
	ret = fastlock_tryacquire_(&lock->impl);
	if (!ret)
		lock->in_use++;

	return ret;
}

static inline void fastlock_release(fastlock_t *lock)
{
	int ret;

	assert(lock->in_use);
	assert(lock->is_initialized);
	lock->in_use--;
	ret = fastlock_release_(&lock->impl);
	assert(!ret);
}

static inline int fastlock_held(fastlock_t *lock)
{
	return lock->in_use;
}

static inline void ofi_fastlock_acquire_noop(fastlock_t *lock)
{
	/* These non-op routines must be used only by single-threaded code*/
	assert(!lock->in_use);
	lock->in_use = 1;
}

static inline void ofi_fastlock_release_noop(fastlock_t *lock)
{
	assert(lock->in_use);
	lock->in_use = 0;
}

#else /* !ENABLE_DEBUG */

#  define fastlock_t fastlock_t_
#  define fastlock_init(lock) fastlock_init_(lock)
#  define fastlock_destroy(lock) fastlock_destroy_(lock)
#  define fastlock_acquire(lock) fastlock_acquire_(lock)
#  define fastlock_tryacquire(lock) fastlock_tryacquire_(lock)
#  define fastlock_release(lock) fastlock_release_(lock)
#  define fastlock_held(lock) true

static inline void ofi_fastlock_acquire_noop(fastlock_t *lock)
{
	(void) lock;
}

static inline void ofi_fastlock_release_noop(fastlock_t *lock)
{
	(void) lock;
}

#endif

typedef void(*ofi_fastlock_acquire_t)(fastlock_t *lock);
typedef void(*ofi_fastlock_release_t)(fastlock_t *lock);

static inline void ofi_fastlock_acquire(fastlock_t *lock)
{
	fastlock_acquire(lock);
}

static inline void ofi_fastlock_release(fastlock_t *lock)
{
	fastlock_release(lock);
}


#define ofi_mutex_t_ pthread_mutex_t
#define ofi_mutex_init_(lock) pthread_mutex_init(lock, NULL)
#define ofi_mutex_destroy_(lock) pthread_mutex_destroy(lock)
#define ofi_mutex_lock_(lock) pthread_mutex_lock(lock)
#define ofi_mutex_trylock_(lock) pthread_mutex_trylock(lock)
#define ofi_mutex_unlock_(lock) pthread_mutex_unlock(lock)

#if ENABLE_DEBUG

typedef struct {
	ofi_mutex_t_ impl;
	int is_initialized;
	int in_use;
} ofi_mutex_t;

static inline int ofi_mutex_init(ofi_mutex_t *lock)
{
	int ret;

	ret = ofi_mutex_init_(&lock->impl);
	lock->is_initialized = !ret;
	lock->in_use = 0;

	return ret;
}

static inline void ofi_mutex_destroy(ofi_mutex_t *lock)
{
	int ret;

	assert(lock->is_initialized);
	lock->is_initialized = 0;
	ret = ofi_mutex_destroy_(&lock->impl);
	assert(!ret);
}

static inline void ofi_mutex_lock(ofi_mutex_t *lock)
{
	int ret;

	assert(lock->is_initialized);
	ret = ofi_mutex_lock_(&lock->impl);
	assert(!ret);
	lock->in_use++;
}

static inline int ofi_mutex_trylock(ofi_mutex_t *lock)
{
	int ret;

	assert(lock->is_initialized);
	ret = ofi_mutex_trylock_(&lock->impl);
	if (!ret)
		lock->in_use++;

	return ret;
}

static inline void ofi_mutex_unlock(ofi_mutex_t *lock)
{
	int ret;

	assert(lock->in_use);
	assert(lock->is_initialized);
	lock->in_use--;
	ret = ofi_mutex_unlock_(&lock->impl);
	assert(!ret);
}

static inline int ofi_mutex_held(ofi_mutex_t *lock)
{
	return lock->in_use;
}

static inline void ofi_mutex_lock_noop(ofi_mutex_t *lock)
{
	/* These non-op routines must be used only by single-threaded code*/
	assert(!lock->in_use);
	lock->in_use = 1;
}

static inline void ofi_mutex_unlock_noop(ofi_mutex_t *lock)
{
	assert(lock->in_use);
	lock->in_use = 0;
}

#else /* !ENABLE_DEBUG */

#  define ofi_mutex_t ofi_mutex_t_
#  define ofi_mutex_init(lock) ofi_mutex_init_(lock)
#  define ofi_mutex_destroy(lock) ofi_mutex_destroy_(lock)
#  define ofi_mutex_lock(lock) ofi_mutex_lock_(lock)
#  define ofi_mutex_trylock(lock) ofi_mutex_trylock_(lock)
#  define ofi_mutex_unlock(lock) ofi_mutex_unlock_(lock)
#  define ofi_mutex_held(lock) true

static inline void ofi_mutex_lock_noop(ofi_mutex_t *lock)
{
	(void) lock;
}

static inline void ofi_mutex_unlock_noop(ofi_mutex_t *lock)
{
	(void) lock;
}

#endif

typedef void(*ofi_mutex_lock_t)(ofi_mutex_t *lock);
typedef void(*ofi_mutex_unlock_t)(ofi_mutex_t *lock);

static inline void ofi_mutex_lock_op(ofi_mutex_t *lock)
{
	ofi_mutex_lock(lock);
}

static inline void ofi_mutex_unlock_op(ofi_mutex_t *lock)
{
	ofi_mutex_unlock(lock);
}


#ifdef __cplusplus
}
#endif

#endif /* _OFI_LOCK_H_ */
