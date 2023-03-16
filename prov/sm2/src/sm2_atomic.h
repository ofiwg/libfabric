/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2018      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2018      Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2019-2021 Google, LLC. All rights reserved.
 * Copyright (c) 2019      Triad National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2022      Amazon.com, Inc. or its affiliates.
 *                         All Rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/* This file provides shims between the opal atomics interface and the C11 atomics interface. It
 * is intended as the first step in moving to using C11 atomics across the entire codebase. Once
 * all officially supported compilers offer C11 atomic (GCC 4.9.0+, icc 2018+, pgi, xlc, etc) then
 * this shim will go away and the codebase will be updated to use C11's atomic support
 * directly.
 * This shim contains some functions already present in atomic_impl.h because we do not include
 * atomic_impl.h when using C11 atomics. It would require alot of #ifdefs to avoid duplicate
 * definitions to be worthwhile. */
/*
 *
 * This code was taken from OMPI, and modified to make naming neutral.
 * https://www.open-mpi.org/
 *
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


#include <stdatomic.h>
#include <stdbool.h>


/* TODO: Switch to using libfabric atomics, ofi_atom */

#define atomic_swap_ptr(addr, value) \
	atomic_exchange_explicit((_Atomic unsigned long *) addr, value, memory_order_relaxed)

#define atomic_compare_exchange(x, y, z) \
	__atomic_compare_exchange_n((int64_t *) (x), (int64_t *) (y), (int64_t)(z), \
								 false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)

static inline void atomic_mb(void)
{
    atomic_thread_fence(memory_order_seq_cst);
}

static inline void atomic_rmb(void)
{
#if defined(PLATFORM_ARCH_X86_64) && defined(PLATFORM_COMPILER_GNU) && __GNUC__ < 8
    /* work around a bug in older gcc versions (observed in gcc 6.x)
     * where acquire seems to get treated as a no-op instead of being
     * equivalent to __asm__ __volatile__("": : :"memory") on x86_64.
     * The issue has been fixed in the GCC 8 release series:
     * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80640
     */
    atomic_mb();
#else
    atomic_thread_fence(memory_order_acquire);
#endif
}

static inline void atomic_wmb(void)
{
    atomic_thread_fence(memory_order_release);
}
