/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

#ifndef _FI_ATOM_H_
#define _FI_ATOM_H_

#include "config.h"

#include <assert.h>
#include <pthread.h>
#include <stdlib.h>

#include <fi_lock.h>
#include <fi_osd.h>

#ifdef HAVE_ATOMICS
#  include <stdatomic.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif


#if ENABLE_DEBUG
#define ATOMIC_IS_INITIALIZED(atomic) assert(atomic->is_initialized)
#else
#define ATOMIC_IS_INITIALIZED(atomic)
#endif

#ifdef HAVE_ATOMICS
typedef struct {
    atomic_int val;
#if ENABLE_DEBUG
    int is_initialized;
#endif
} atomic_t;

static inline int atomic_inc(atomic_t *atomic)
{
	ATOMIC_IS_INITIALIZED(atomic);
	return atomic_fetch_add_explicit(&atomic->val, 1, memory_order_acq_rel) + 1;
}

static inline int atomic_dec(atomic_t *atomic)
{
	ATOMIC_IS_INITIALIZED(atomic);
	return atomic_fetch_sub_explicit(&atomic->val, 1, memory_order_acq_rel) - 1;
}

static inline int atomic_set(atomic_t *atomic, int value)
{
	ATOMIC_IS_INITIALIZED(atomic);
	atomic_store(&atomic->val, value);
	return value;
}

static inline int atomic_get(atomic_t *atomic)
{
	ATOMIC_IS_INITIALIZED(atomic);
	return atomic_load(&atomic->val);
}

/* avoid using "atomic_init" so we don't conflict with symbol/macro from stdatomic.h */
static inline void atomic_initialize(atomic_t *atomic, int value)
{
	atomic_init(&atomic->val, value);
#if ENABLE_DEBUG
	atomic->is_initialized = 1;
#endif
}

static inline int atomic_add(atomic_t *atomic, int val)
{
	ATOMIC_IS_INITIALIZED(atomic);
	return atomic_fetch_add_explicit(&atomic->val,
			val, memory_order_acq_rel) + 1;
}

static inline int atomic_sub(atomic_t *atomic, int val)
{
	ATOMIC_IS_INITIALIZED(atomic);
	return atomic_fetch_sub_explicit(&atomic->val,
			val, memory_order_acq_rel) - 1;
}

#else

typedef struct {
	fastlock_t lock;
	int val;
#if ENABLE_DEBUG
	int is_initialized;
#endif
} atomic_t;

static inline int atomic_inc(atomic_t *atomic)
{
	int v;

	ATOMIC_IS_INITIALIZED(atomic);
	fastlock_acquire(&atomic->lock);
	v = ++(atomic->val);
	fastlock_release(&atomic->lock);
	return v;
}

static inline int atomic_dec(atomic_t *atomic)
{
	int v;

	ATOMIC_IS_INITIALIZED(atomic);
	fastlock_acquire(&atomic->lock);
	v = --(atomic->val);
	fastlock_release(&atomic->lock);
	return v;
}

static inline int atomic_set(atomic_t *atomic, int value)
{
	ATOMIC_IS_INITIALIZED(atomic);
	fastlock_acquire(&atomic->lock);
	atomic->val = value;
	fastlock_release(&atomic->lock);
	return value;
}

/* avoid using "atomic_init" so we don't conflict with symbol/macro from stdatomic.h */
static inline void atomic_initialize(atomic_t *atomic, int value)
{
	fastlock_init(&atomic->lock);
	atomic->val = value;
#if ENABLE_DEBUG
	atomic->is_initialized = 1;
#endif
}

static inline int atomic_get(atomic_t *atomic)
{
	ATOMIC_IS_INITIALIZED(atomic);
	return atomic->val;
}

static inline int atomic_add(atomic_t *atomic, int val)
{
	int v;

	ATOMIC_IS_INITIALIZED(atomic);
	fastlock_acquire(&atomic->lock);
	atomic->val += val;
	v = atomic->val;
	fastlock_release(&atomic->lock);
	return v;
}

static inline int atomic_sub(atomic_t *atomic, int val)
{
	int v;

	ATOMIC_IS_INITIALIZED(atomic);
	fastlock_acquire(&atomic->lock);
	atomic->val -= val;
	v = atomic->val;
	fastlock_release(&atomic->lock);
	return v;
}

#endif // HAVE_ATOMICS


#ifdef __cplusplus
}
#endif

#endif /* _FI_ATOM_H_ */
