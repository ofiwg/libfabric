/*
 * Copyright (C) 2025 Cornelis Networks.
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
#ifndef _FI_PROV_OPX_HFISVC_KEYSET_H_
#define _FI_PROV_OPX_HFISVC_KEYSET_H_

#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#include "rdma/opx/opx_hfisvc.h"
#include "rdma/opx/fi_opx_compiler.h"
#include "rdma/opx/fi_opx_debug_counters.h"

typedef uintptr_t opx_hfisvc_keyset_t;
typedef uint32_t  opx_hfisvc_key_t;

/*
 * The maximum number of access_keys to allow.
 */
#ifndef OPX_HFISVC_KEYSET_MAX_KEYS
#define OPX_HFISVC_KEYSET_MAX_KEYS (16 * 1024 * 1024)
#endif

/*
 * Allocate 512 QWs (4KB, or 32,768 keys) per malloc.
 */
#define OPX_HFISVC_KEYSET_CHUNK_SIZE_QWS  (512)
#define OPX_HFISVC_KEYSET_CHUNK_SIZE_KEYS (OPX_HFISVC_KEYSET_CHUNK_SIZE_QWS << 6)

OPX_COMPILE_TIME_ASSERT(OPX_HFISVC_KEYSET_CHUNK_SIZE_KEYS <= OPX_HFISVC_KEYSET_MAX_KEYS,
			"OPX_HFISVC_KEYSET_CHUNK_SIZE_KEYS must be <= OPX_HFISVC_KEYSET_MAX_KEYS!\n");

/**
 * Key set used for vending keys to use with HFI service requests.
 * This uses a bitmap of qws to vend keys with values ranging from
 * 0-((bitmap_size_qws * 8) - 1).
 */
struct opx_hfisvc_keyset {
	/* == CACHE LINE 0 == */
	size_t	 size_in_bytes;
	uint32_t bitmap_size_qws;
	uint32_t keys_total;
	int32_t	 keys_free;
	uint32_t unused_dw;
	uint64_t unused_qw[5];

	/* == CACHE LINE 1 == */
	uint64_t bitmap[];
} __attribute__((__packed__)) __attribute__((aligned(64)));
OPX_COMPILE_TIME_ASSERT(offsetof(struct opx_hfisvc_keyset, bitmap) == FI_OPX_CACHE_LINE_SIZE,
			"Offset of opx_hfisvc_keyset.bitmap should fall on first cacheline boundary!\n");

__OPX_FORCE_INLINE__
int opx_hfisvc_keyset_grow(struct opx_hfisvc_keyset **keyset)
{
	struct opx_hfisvc_keyset *current_keyset = *keyset;

	if ((current_keyset->keys_total + OPX_HFISVC_KEYSET_CHUNK_SIZE_KEYS) > OPX_HFISVC_KEYSET_MAX_KEYS) {
		OPX_HFISVC_DEBUG_LOG("HFISVC Unable to allocate additional keyspace, max keys (%u) reached.\n",
				     OPX_HFISVC_KEYSET_MAX_KEYS);
		return -FI_ENOMEM;
	}

	size_t new_size = current_keyset->size_in_bytes + (OPX_HFISVC_KEYSET_CHUNK_SIZE_QWS * sizeof(uint64_t));

	struct opx_hfisvc_keyset *new_keyset;

	if (posix_memalign((void **) &new_keyset, 64, new_size)) {
		OPX_HFISVC_DEBUG_LOG(
			"HFISVC Unable to allocate additional keyspace, memory allocation failed. Current allocation is %u total keys (%lu bytes).\n",
			current_keyset->keys_total, current_keyset->size_in_bytes);
		return -FI_ENOMEM;
	}

	new_keyset->size_in_bytes   = new_size;
	new_keyset->keys_total	    = current_keyset->keys_total + OPX_HFISVC_KEYSET_CHUNK_SIZE_KEYS;
	new_keyset->keys_free	    = current_keyset->keys_free + OPX_HFISVC_KEYSET_CHUNK_SIZE_KEYS;
	new_keyset->bitmap_size_qws = current_keyset->bitmap_size_qws + OPX_HFISVC_KEYSET_CHUNK_SIZE_QWS;

	int i;
	for (i = 0; i < current_keyset->bitmap_size_qws; ++i) {
		new_keyset->bitmap[i] = current_keyset->bitmap[i];
	}
	for (; i < new_keyset->bitmap_size_qws; ++i) {
		new_keyset->bitmap[i] = 0ul;
	}

	OPX_HFISVC_DEBUG_LOG("HFISVC Keyset grew from %u keys (%lu bytes) to %u keys (%lu bytes)\n",
			     current_keyset->keys_total, current_keyset->size_in_bytes, new_keyset->keys_total,
			     new_keyset->size_in_bytes);

	*keyset = new_keyset;

	free(current_keyset);

	return 0;
}

/**
 * Initialize the keyset
 */
__OPX_FORCE_INLINE__
int opx_hfisvc_keyset_init(opx_hfisvc_keyset_t *keyset)
{
	size_t keyset_mem_size =
		sizeof(struct opx_hfisvc_keyset) + (OPX_HFISVC_KEYSET_CHUNK_SIZE_QWS * sizeof(uint64_t));
	struct opx_hfisvc_keyset *new_keyset;

	if (posix_memalign((void **) &new_keyset, 64, keyset_mem_size)) {
		return -ENOMEM;
	}

	new_keyset->size_in_bytes   = keyset_mem_size;
	new_keyset->keys_free	    = OPX_HFISVC_KEYSET_CHUNK_SIZE_KEYS;
	new_keyset->keys_total	    = OPX_HFISVC_KEYSET_CHUNK_SIZE_KEYS;
	new_keyset->bitmap_size_qws = OPX_HFISVC_KEYSET_CHUNK_SIZE_QWS;

	for (int i = 0; i < OPX_HFISVC_KEYSET_CHUNK_SIZE_QWS; ++i) {
		new_keyset->bitmap[i] = 0ul;
	}

	*keyset = (opx_hfisvc_keyset_t) new_keyset;

	return 0;
}

/**
 * Allocate the next available key from the keyset.
 *
 * @return 0 if the key was allocated successfully, or -ENOSPC if no keys are available.
 *
 */
__OPX_FORCE_INLINE__
int opx_hfisvc_keyset_alloc_key(opx_hfisvc_keyset_t *keyset, opx_hfisvc_key_t *key,
				struct fi_opx_debug_counters *counters)
{
	struct opx_hfisvc_keyset *_keyset = (struct opx_hfisvc_keyset *) (*keyset);
	assert(_keyset);
	assert(key);

	if (_keyset->keys_free < 1) {
		if (opx_hfisvc_keyset_grow(&_keyset) == 0) {
			assert(_keyset->keys_free > 1);
			FI_OPX_DEBUG_COUNTERS_INC(counters->hfisvc.access_key.keyset_grow);
			*keyset = (opx_hfisvc_keyset_t) _keyset;
		} else {
			goto alloc_end;
		}
	}

	for (int i = 0; i < _keyset->bitmap_size_qws; i++) {
		uint64_t inv = ~_keyset->bitmap[i];
		if (inv) {
			uint64_t bit_index = __builtin_ctzl(inv);
			*key		   = (i * 64ul + bit_index);
			_keyset->bitmap[i] |= (1ul << bit_index);
			_keyset->keys_free--;

			FI_OPX_DEBUG_COUNTERS_INC(counters->hfisvc.access_key.alloc);

			return 0;
		}
	}
	FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
		"HFISVC Keyset error: Keyset has %d keys free of %u keys total, but no available keys were found in bitmap, abort.\n",
		_keyset->keys_free, _keyset->keys_total);
	abort();

alloc_end:
	FI_OPX_DEBUG_COUNTERS_INC(counters->hfisvc.access_key.alloc_enospc);
	return -FI_ENOSPC;
}

/**
 * Free/return a previously allocated key to the keyset so it can be reused.
 */
__OPX_FORCE_INLINE__
void opx_hfisvc_keyset_free_key(opx_hfisvc_keyset_t keyset, opx_hfisvc_key_t key,
				struct fi_opx_debug_counters *counters)
{
	FI_OPX_DEBUG_COUNTERS_INC(counters->hfisvc.access_key.free);

	struct opx_hfisvc_keyset *_keyset = (struct opx_hfisvc_keyset *) keyset;
	assert(_keyset);

	uint64_t key_index = key >> 6;

	OPX_HFISVC_DEBUG_LOG("Freeing key %u, key_index=%016lX, _keyset->bitmap[%lX]=%016lX\n", key, key_index,
			     key_index, _keyset->bitmap[key_index]);

	// Assert that the key being freed is currently marked as being used.
	assert(_keyset->bitmap[key_index] & (1ul << (key & 0x3Ful)));

	_keyset->bitmap[key_index] &= ~(1ul << (key & 0x3Ful));
	_keyset->keys_free++;
}

/**
 * Free a keyset.
 */
__OPX_FORCE_INLINE__
void opx_hfisvc_keyset_free(opx_hfisvc_keyset_t keyset)
{
	struct opx_hfisvc_keyset *_keyset = (struct opx_hfisvc_keyset *) keyset;
	assert(_keyset);
	if (_keyset->keys_free < _keyset->keys_total) {
		FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
			"HFISVC Keyset error: Freeing the keyset while there are still %d of %u access keys still outstanding, abort.\n",
			_keyset->keys_total - _keyset->keys_free, _keyset->keys_total);
		abort();
	}
	free(_keyset);
}

#endif
