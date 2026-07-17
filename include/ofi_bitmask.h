/*
 * Copyright (c) 2019-2019 Intel Corporation. All rights reserved.
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

#ifndef _OFI_BITMASK_H_
#define _OFI_BITMASK_H_

#include <assert.h>
#include <rdma/fi_errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct ofi_bitmask {
	size_t size;
	uint8_t *bytes;
};

static inline int ofi_bitmask_create(struct ofi_bitmask *mask, size_t size)
{
	size_t byte_size = (size + 7) / 8;
	mask->bytes = calloc(byte_size, 1);
	if (!mask->bytes)
		return -FI_ENOMEM;

	mask->size = size;

	return FI_SUCCESS;
}

static inline void ofi_bitmask_free(struct ofi_bitmask *mask)
{
	free(mask->bytes);
	mask->bytes = NULL;
}

static inline size_t ofi_bitmask_bytesize(const struct ofi_bitmask *mask)
{
	return (mask->size + 7) / 8;
}

static inline void ofi_bitmask_unset(struct ofi_bitmask *mask, size_t idx)
{
	assert(idx < mask->size);
	mask->bytes[idx / 8] &= ~(0x01 << (idx % 8));
}

static inline void ofi_bitmask_set(struct ofi_bitmask *mask, size_t idx)
{
	assert(idx < mask->size);
	mask->bytes[idx / 8] |= (0x01 << (idx % 8));
}

static inline void ofi_bitmask_set_all(struct ofi_bitmask *mask)
{
	size_t bytesize = ofi_bitmask_bytesize(mask);
	unsigned int tail_bits = mask->size & 7;

	memset(mask->bytes, 0xff, bytesize);
	if (tail_bits)
		mask->bytes[bytesize - 1] >>= 8 - tail_bits;
}

#if !defined(__GNUC__) && !defined(__clang__)
#error "This file requires GCC/Clang builtins (__builtin_ctzll, __builtin_ctz)!"
#endif
static inline size_t ofi_bitmask_get_lsbset(const struct ofi_bitmask *mask)
{
	size_t ret = 0;

	// 1. Scan with 64-bit words
	assert(((uintptr_t) mask->bytes & 7) == 0);
	const uint64_t *words = (const uint64_t *) mask->bytes;
	const uint64_t *words_end = words + mask->size / 64;
	uint64_t word;

	while (words < words_end) {
		word = *words++;
		if (word == 0)
			ret += 64;
		else {
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
	(__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
			/*
			 * On Big Endian, the byte order in the 64-bit register
			 * is reversed. We must swap it to Little Endian so that
			 * mask.bytes[cur] maps to the lowest bits of 'word'.
			 */
			word = __builtin_bswap64(word);
#endif
			return ret + __builtin_ctzll(word);
		}
	} // while
	/*
	 * Note on memory safety for tail bits:
	 * The 'mask->bytes' buffer is strictly 64-bit aligned. Even if the
	 * allocation size is only byte-rounded (e.g., 9 bytes), reading the
	 * entire final 64-bit word is safe from hardware page faults. Because
	 * memory mapping occurs at page granularity, the remaining bytes of an
	 * aligned 8-byte word cannot reside on an unmapped page if the first
	 * byte is valid. Any trailing out-of-bounds garbage read this way is
	 * safely neutralized by the sentinel bit injected below.
	 */
	unsigned int tail_bits = mask->size & 63;

	if (tail_bits) {
		word = *words_end;
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
	(__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
		/*
		 * On Big Endian, the byte order in the 64-bit register
		 * is reversed. We must swap it to Little Endian so that
		 * mask.bytes[cur] maps to the lowest bits of 'word'.
		 */
		word = __builtin_bswap64(word);
#endif
		/*
		 * Inject a sentinel bit at the 'tail_bits' position using
		 * bitwise OR. This acts as a hard barrier: if no bits are set
		 * in the valid tail region, ctzll will stop exactly at the
		 * sentinel, returning 'tail_bits' and effectively ignoring any
		 * trailing garbage in the rest of the word.
		 */
		ret += __builtin_ctzll(word | (1ULL << tail_bits));
	}
	assert(ret <= (mask->size));
	return ret;
}

#endif
