/*
 * Copyright (C) 2026 Cornelis Networks.
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

#include "config.h"

#include <ofi.h>
#include <stdint.h>
#include <string.h>

#include "rdma/opx/fi_opx_atomic_clones.h"

#ifdef FI_OPX_RX_ATOMIC_USE_TARGET_CLONES
typedef int32_t	 fi_opx_unaligned_int32_t __attribute__((aligned(1), may_alias));
typedef uint32_t fi_opx_unaligned_uint32_t __attribute__((aligned(1), may_alias));
typedef int64_t	 fi_opx_unaligned_int64_t __attribute__((aligned(1), may_alias));
typedef uint64_t fi_opx_unaligned_uint64_t __attribute__((aligned(1), may_alias));
typedef float	 fi_opx_unaligned_float_t __attribute__((aligned(1), may_alias));
typedef double	 fi_opx_unaligned_double_t __attribute__((aligned(1), may_alias));

#define FI_OPX_RX_ATOMIC_DEFINE_TARGET_CLONE(op_, suffix_, type_, body_)                                    \
	FI_OPX_RX_ATOMIC_TARGET_CLONES                                                                      \
	static void fi_opx_rx_atomic_##op_##_##suffix_##_target_clones_impl(const void *buf, void *addr,    \
									    size_t nbytes)                  \
	{                                                                                                   \
		const type_ *buf__  = (const type_ *) buf;                                                  \
		type_	    *addr__ = (type_ *) addr;                                                       \
		const size_t count  = nbytes / sizeof(*addr__);                                             \
		size_t	     i;                                                                             \
		for (i = 0; i < count; ++i) {                                                               \
			body_;                                                                              \
		}                                                                                           \
	}                                                                                                   \
	void fi_opx_rx_atomic_##op_##_##suffix_##_target_clones(const void *buf, void *addr, size_t nbytes) \
	{                                                                                                   \
		fi_opx_rx_atomic_##op_##_##suffix_##_target_clones_impl(buf, addr, nbytes);                 \
	}

#define FI_OPX_RX_ATOMIC_DEFINE_NUMERIC_TARGET_CLONES(op_, body_)                           \
	FI_OPX_RX_ATOMIC_DEFINE_TARGET_CLONE(op_, int32, fi_opx_unaligned_int32_t, body_)   \
	FI_OPX_RX_ATOMIC_DEFINE_TARGET_CLONE(op_, uint32, fi_opx_unaligned_uint32_t, body_) \
	FI_OPX_RX_ATOMIC_DEFINE_TARGET_CLONE(op_, int64, fi_opx_unaligned_int64_t, body_)   \
	FI_OPX_RX_ATOMIC_DEFINE_TARGET_CLONE(op_, uint64, fi_opx_unaligned_uint64_t, body_) \
	FI_OPX_RX_ATOMIC_DEFINE_TARGET_CLONE(op_, float, fi_opx_unaligned_float_t, body_)   \
	FI_OPX_RX_ATOMIC_DEFINE_TARGET_CLONE(op_, double, fi_opx_unaligned_double_t, body_)

#define FI_OPX_RX_ATOMIC_DEFINE_ATOMIC_WRITE_TARGET_CLONE(suffix_)                                               \
	FI_OPX_RX_ATOMIC_TARGET_CLONES                                                                           \
	static void fi_opx_rx_atomic_atomic_write_##suffix_##_target_clones_impl(const void *buf, void *addr,    \
										 size_t nbytes)                  \
	{                                                                                                        \
		memcpy(addr, buf, nbytes);                                                                       \
	}                                                                                                        \
	void fi_opx_rx_atomic_atomic_write_##suffix_##_target_clones(const void *buf, void *addr, size_t nbytes) \
	{                                                                                                        \
		fi_opx_rx_atomic_atomic_write_##suffix_##_target_clones_impl(buf, addr, nbytes);                 \
	}

#define FI_OPX_RX_ATOMIC_DEFINE_ATOMIC_WRITE_TARGET_CLONES()      \
	FI_OPX_RX_ATOMIC_DEFINE_ATOMIC_WRITE_TARGET_CLONE(int32)  \
	FI_OPX_RX_ATOMIC_DEFINE_ATOMIC_WRITE_TARGET_CLONE(uint32) \
	FI_OPX_RX_ATOMIC_DEFINE_ATOMIC_WRITE_TARGET_CLONE(int64)  \
	FI_OPX_RX_ATOMIC_DEFINE_ATOMIC_WRITE_TARGET_CLONE(uint64) \
	FI_OPX_RX_ATOMIC_DEFINE_ATOMIC_WRITE_TARGET_CLONE(float)  \
	FI_OPX_RX_ATOMIC_DEFINE_ATOMIC_WRITE_TARGET_CLONE(double)

FI_OPX_RX_ATOMIC_DEFINE_NUMERIC_TARGET_CLONES(min, addr__[i] = buf__[i] < addr__[i] ? buf__[i] : addr__[i])
FI_OPX_RX_ATOMIC_DEFINE_NUMERIC_TARGET_CLONES(max, addr__[i] = buf__[i] > addr__[i] ? buf__[i] : addr__[i])
FI_OPX_RX_ATOMIC_DEFINE_NUMERIC_TARGET_CLONES(sum, addr__[i] += buf__[i])
FI_OPX_RX_ATOMIC_DEFINE_NUMERIC_TARGET_CLONES(prod, addr__[i] = addr__[i] * buf__[i])
FI_OPX_RX_ATOMIC_DEFINE_ATOMIC_WRITE_TARGET_CLONES()
#endif
