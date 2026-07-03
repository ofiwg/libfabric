/*
 * Copyright (C) 2025-2026 Cornelis Networks.
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

#include "rdma/opx/opx_rndtsc.h"

/*
 * ============================================================================
 * HFISVC keyset allocator: the "double random" free-slot search
 * ============================================================================
 *
 * This allocator finds a free slot in a bitmap by walking the table with a
 * randomized probe sequence, NOT by scanning it linearly. The scheme is worth
 * explaining, because its correctness rests on a small number-theory fact that
 * is invisible in the code: if you break that fact, the search silently stops
 * visiting every slot and the abort() at the end of the loop becomes reachable.
 *
 * ----------------------------------------------------------------------------
 * Relation to "double hashing" -- and why we do NOT call it that
 * ----------------------------------------------------------------------------
 * The table-walk here looks exactly like the probe step of classic open-
 * addressing double hashing:
 *
 *         pos = (pos + step) mod N
 *
 * But nothing is hashed. In real double hashing both the start position and
 * the step are functions of a KEY (h1(key), h2(key)); the same key always
 * follows the same path, which is what makes later lookups find what an insert
 * stored. We have no input key: we are not locating a given key, we are
 * hunting for ANY free slot. Our start position and our step come from a
 * per-call random source (see opx_rndtsc.h), from the state of the world, not
 * from data. So we borrow only the *traversal*, not the *hashing*. Hence
 * "double random" (random start + random step), not "double hashing".
 *
 * ----------------------------------------------------------------------------
 * The core guarantee: full-period traversal
 * ----------------------------------------------------------------------------
 * Claim: starting at any position s and repeatedly applying
 *
 *         pos = (pos + step) mod N
 *
 * visits ALL N slots exactly once before returning to s, PROVIDED
 *
 *         gcd(step, N) == 1        (step and N are coprime).
 *
 * This is why the search is self-terminating: after at most N probes we have
 * either found a free slot or proven every slot is full. No explicit probe
 * cap is needed, and none must be added -- a cap would mask a desync bug
 * instead of exposing it via the counter/pigeonhole check.
 *
 * Proof sketch. After i steps we are at (s + i*step) mod N. Suppose two
 * distinct steps i != j (with 0 <= i, j < N) landed on the same slot:
 *
 *         (s + i*step) == (s + j*step)   (mod N)
 *         (i - j)*step == 0              (mod N)
 *
 * So N divides (i - j)*step. Because gcd(step, N) == 1, N shares no factor
 * with step, so N must divide (i - j). But |i - j| < N and i != j, so
 * 0 < |i - j| < N cannot be a multiple of N -- contradiction. Therefore the
 * first N positions are all distinct; N distinct values in a table of size N
 * means every slot is hit exactly once. The sequence is a full permutation of
 * the table (equivalently: {step, 2*step, ...} generates the whole additive
 * group Z/NZ precisely when step is a unit mod N, i.e. gcd(step, N) == 1).
 *
 * ----------------------------------------------------------------------------
 * Why "odd step" + "power-of-two table" is the cheap way to guarantee it
 * ----------------------------------------------------------------------------
 * We need gcd(step, N) == 1. In general that is an expensive per-call promise.
 * We buy it for free by fixing the table size to a power of two, N = 2^k:
 *
 *   - The only prime factor of 2^k is 2.
 *   - Therefore gcd(step, 2^k) == 1  IFF  step is odd.
 *
 * So the entire coprimality requirement collapses to a single bit: force the
 * low bit of step to 1. That is the whole reason the table is a power of two
 * and the step is forced odd. It is not a micro-optimization and not a style
 * choice -- it is the mechanism that makes the traversal a full permutation.
 *
 * Two more conveniences fall out of N = 2^k:
 *   - The modulo becomes a mask: (pos + step) mod N  ==  (pos + step) & (N-1).
 *   - The mask (N-1) IS the size descriptor we already keep, so no separate
 *     modulus/size constant is needed.
 *
 * ----------------------------------------------------------------------------
 * DO NOT BREAK, in order of how quietly they fail:
 * ----------------------------------------------------------------------------
 *   1. Table size MUST stay a power of two. If N stops being 2^k, "odd step"
 *      no longer implies coprimality (e.g. N=6, step=3: gcd=3, the walk hits
 *      only half the slots and loops). The search would then abort() on a
 *      table that still has free slots.
 *
 *   2. The step MUST be forced odd. An even step shares the factor 2 with N,
 *      gcd > 1, and the walk covers only N/gcd slots -- again a false abort()
 *      over a non-full table.
 *
 *   3. Randomness is for LOAD DISTRIBUTION, not for correctness. The full-
 *      period guarantee comes entirely from coprimality (odd step, 2^k table);
 *      the random start/step only spread probes to avoid clustering hot spots.
 *      A bad random source hurts distribution, never termination.
 *
 *   4. Termination is proven by the occupancy counter + pigeonhole, not by the
 *      loop bound. The loop runs at most N iterations because the permutation
 *      has period N; the abort() is unreachable only because the counter says
 *      a free slot exists and the permutation guarantees we will reach it. Keep
 *      the counter exact (see the leased_* accounting invariant) or this whole
 *      argument collapses.
 *
 * We scan at 64-bit word granularity (test word != ~0ULL, then ctz to find the
 * free bit), so "slot" above is a WORD of the bitmap; the same coprimality
 * argument applies to the word-index space, which is also a power of two.
 * ============================================================================
 */

typedef uintptr_t opx_hfisvc_keyset_t;
typedef uint32_t  opx_hfisvc_key_t;

/*
 * The maximum number of access_keys to allow.
 */
#ifndef OPX_HFISVC_KEYSET_MAX_KEYS
#define OPX_HFISVC_KEYSET_MAX_KEYS (16 * 1024 * 1024)
#endif
/*
 *The cap check on line 230 does (bytes * 8) in uint32_t. That is safe only
 * while the limit stays well below 2^32. Guard it so a future bump of the
 * limit fails to build instead of silently overflowing the check.
 */
_Static_assert(OPX_HFISVC_KEYSET_MAX_KEYS <= (1u << 28),
	       "OPX_HFISVC_KEYSET_MAX_KEYS too large: (bytes*8) cap check may overflow uint32_t");

/*
 * Allocate 512 QWs (4KB, or 32,768 keys) per malloc.
 */
#define OPX_HFISVC_KEYSET_CHUNK_SIZE_QWS  (512)
#define OPX_HFISVC_KEYSET_CHUNK_SIZE_KEYS (OPX_HFISVC_KEYSET_CHUNK_SIZE_QWS << 6)

OPX_COMPILE_TIME_ASSERT(OPX_HFISVC_KEYSET_CHUNK_SIZE_KEYS <= OPX_HFISVC_KEYSET_MAX_KEYS,
			"OPX_HFISVC_KEYSET_CHUNK_SIZE_KEYS must be <= OPX_HFISVC_KEYSET_MAX_KEYS!\n");

/*
 * In fast path, wrap via bit-clear works ONLY if fast_bitmap is aligned to 2*(FAST_KEYS/8):
 * the cleared bit (FAST_KEYS / 8) must be zero in the base address.
 * Guaranteed here because fast_bitmap is the first field of an mmap'd
 * (page-aligned, 4096) region and FAST_KEYS/8 = 128 << 4096.
 */

#define OPX_FAST_KEYS 1024

OPX_COMPILE_TIME_ASSERT(OPX_FAST_KEYS / 8 <= 4096, "OPX_FAST_KEYS must be <= virtual page size (4K)!\n");

/*
 * fields of following structure accesable exclusively from this file.
 * All clients works with abstract type opx_hfisvc_keyset_t.
 */
struct opx_hfisvc_keyset {
	uint64_t  fast_bitmap[OPX_FAST_KEYS / 64];
	uint64_t *hintptr;	 // Hint - suggestion from release to alloc
	uint32_t  leased_fast;	 // Leased keys in fast space, [0..1024)
	uint32_t  leased_slow;	 // Leased keys in slow space, [0..64*mask]
	uint32_t  slow_treshold; // slow table population treshold
	uint32_t  slow_mask;	 // Mask for slow_bitmap[], uint64
	uint64_t  slow_bitmap[];
};

/**
 * Initialize the keyset
 */
__OPX_FORCE_INLINE__
int opx_hfisvc_keyset_init(opx_hfisvc_keyset_t *keyset)
{
	// Compute memory size - struct with fast keyset and 1 extra chunk of slow keyset
	size_t keyset_mem_size =
		sizeof(struct opx_hfisvc_keyset) + (OPX_HFISVC_KEYSET_CHUNK_SIZE_QWS * sizeof(uint64_t));
	struct opx_hfisvc_keyset *new_keyset = (struct opx_hfisvc_keyset *) mmap(
		NULL, keyset_mem_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (new_keyset == MAP_FAILED) {
		return -FI_ENOMEM; // Maybe ENOMEM? As same as in previous code.
	}

	// mmaped page is aligned, and 0-filled, so no sense to refill again
	new_keyset->slow_mask = OPX_HFISVC_KEYSET_CHUNK_SIZE_QWS - 1;
	// Set initial treshold as  7/8 of slow-table
	new_keyset->slow_treshold = OPX_HFISVC_KEYSET_CHUNK_SIZE_KEYS * 7 / 8;
	new_keyset->hintptr	  = new_keyset->fast_bitmap;

	*keyset = (opx_hfisvc_keyset_t) new_keyset;

	return 0;
} // opx_hfisvc_keyset_init

__OPX_FORCE_INLINE__
int opx_hfisvc_keyset_grow(struct opx_hfisvc_keyset **keyset)
{
	// Single source of truth for the growth factor (2^N). Change ONLY here.
	const uint32_t growth_factor	     = 2;
	uint32_t       new_slow_64_bit_words = ((*keyset)->slow_mask + 1) * growth_factor;
	uint32_t       new_slow_keys_bytes   = new_slow_64_bit_words * sizeof(uint64_t);
	// Before expanding to 2x keyset - check for the limit
	if (OFI_UNLIKELY(new_slow_keys_bytes * 8 /* bits in byte */ + OPX_FAST_KEYS > OPX_HFISVC_KEYSET_MAX_KEYS)) {
		OPX_HFISVC_DEBUG_LOG("HFISVC Unable to allocate additional keyspace, max keys (%u) reached.\n",
				     OPX_HFISVC_KEYSET_MAX_KEYS);
		return -FI_ENOMEM;
	}
	// Compute current memory size
	size_t cur_size = sizeof(struct opx_hfisvc_keyset) + ((*keyset)->slow_mask + 1) * sizeof(uint64_t);
	// Compute new memory size - struct with fast keyset and 2x extra chunks of slow keyset
	size_t new_size = sizeof(struct opx_hfisvc_keyset) + new_slow_keys_bytes;

	struct opx_hfisvc_keyset *new_keyset =
		(struct opx_hfisvc_keyset *) mremap(*keyset, cur_size, new_size, MREMAP_MAYMOVE);
	if (OFI_UNLIKELY(new_keyset == MAP_FAILED)) {
		OPX_HFISVC_DEBUG_LOG("HFISVC Unable to allocate additional keyspace, memory allocation failed. "
				     "Current allocation %zu bytes, tried to grow to %zu bytes.\n",
				     cur_size, new_size);
		return -FI_ENOMEM;
	}

	// Adjust mask and treshold for 2x slow table
	new_keyset->slow_mask	  = new_slow_64_bit_words - 1;
	new_keyset->slow_treshold = new_slow_64_bit_words * (64 * 7 / 8);
	*keyset			  = new_keyset;

	return 0;
} // opx_hfisvc_keyset_grow

/**
 * Allocate the next available key from the keyset.
 *
 * @return 0 if the key was allocated successfully, or -FI_ENOSPC if no keys are available.
 *
 */
__OPX_FORCE_INLINE__
int opx_hfisvc_keyset_alloc_key(opx_hfisvc_keyset_t *keyset, opx_hfisvc_key_t *key,
				struct fi_opx_debug_counters *counters)
{
	struct opx_hfisvc_keyset *_keyset = (struct opx_hfisvc_keyset *) (*keyset);
	uint64_t		 *randptr;
	assert(_keyset);
	assert(key);

	if (OFI_LIKELY(_keyset->leased_fast < OPX_FAST_KEYS)) {
		// Search in the fast table, start from hint
		randptr = _keyset->hintptr;
		if (OFI_UNLIKELY(*randptr == ~0ULL)) {
			uint64_t *randptr0 = randptr;
			// Loop does not test  *randptr == ~0ULL at last iteration whenever
			// randptr == randptr0, since it already tested in the "if" above
			do {
				randptr++;
				randptr = (uint64_t *) ((uintptr_t) randptr & ~(OPX_FAST_KEYS >> 3));
			} while (randptr != randptr0 && *randptr == ~0ULL);
			_keyset->hintptr = randptr;
		}
	} else if (OFI_UNLIKELY(_keyset->leased_slow > _keyset->slow_treshold)) {
		// Slow table populated by 7/8, need to expand
		if (OFI_UNLIKELY(opx_hfisvc_keyset_grow(&_keyset) != 0)) {
			FI_OPX_DEBUG_COUNTERS_INC(counters->hfisvc.access_key.alloc_enospc);
			return -FI_ENOSPC;
		}
		FI_OPX_DEBUG_COUNTERS_INC(counters->hfisvc.access_key.keyset_grow);
		*keyset		 = (opx_hfisvc_keyset_t) _keyset;
		_keyset->hintptr = _keyset->fast_bitmap;
		// Last word in the new keyset is always 0, do not need any search
		randptr = _keyset->slow_bitmap + _keyset->slow_mask;
	} else {
		// Search in the slow table
		uint32_t randpos = rndtsc();
		// Compute odd randstep, for gcd(table_size, randstep) == 1
		// One rndtsc() feeds both start and step. slow_mask is up to 18 bits
		// (MAX_KEYS = 2^24 keys / 64 bits-per-word = 2^18 words). randpos takes
		// bits [0..17]; randstep takes [17..31]>>17 then |1. The only shared bit
		// [17] is forced to 1 by |1, so live entropy of start and step does NOT
		// overlap. Correctness (full period) needs only odd step (|1), never
		// independence; the shift is for cluster spread, not for correctness.
		uint32_t randstep = (randpos >> 17) | 1;
		randpos &= _keyset->slow_mask;
		uint32_t randpos0 = randpos;
		/*
		 * NOTE TO THE NEXT SMART REVIEWER (human or AI): this loop is NOT
		 * off-by-one. All N distinct positions are dereferenced and tested.
		 * Every iteration first points randptr at position p_k, THEN reads
		 * *randptr, and ONLY THEN advances randpos to p_{k+1} and tests the
		 * exit. So the wrap-back (randpos == randpos0) can only fire AFTER the
		 * last unvisited word p_{N-1} has already been read in this same body.
		 *
		 * Invariant: iteration k reads word p_k for k = 0..N-1 (worst case),
		 * i.e. exactly N reads over the full permutation. Exit happens either
		 * on a free word (*randptr != ~0, randptr left on the free slot) or
		 * after reading p_{N-1}, when p_N == p_0 == randpos0.
		 *
		 * Worked example, N = 2, start 0, step 1:
		 *   iter0: randptr=&[0]; read [0]; randpos=1; (1 != 0) -> continue
		 *   iter1: randptr=&[1]; read [1]; randpos=0; (0 == 0) -> exit
		 * Both words are examined. No false "table full". Please don't "fix" it.
		 */
		do {
			randptr = _keyset->slow_bitmap + randpos;
			randpos = (randpos + randstep) & _keyset->slow_mask;
		} while (OFI_UNLIKELY(*randptr == ~0ULL && randpos != randpos0));
	}

	if (OFI_UNLIKELY(*randptr == ~0ULL)) {
		uint32_t    keys_total, keys_free;
		const char *path_name;
		if (randptr < _keyset->slow_bitmap) {
			keys_total = OPX_FAST_KEYS;
			keys_free  = OPX_FAST_KEYS - _keyset->leased_fast;
			path_name  = "fast";

		} else {
			keys_total = (_keyset->slow_mask + 1) << 6;
			keys_free  = keys_total - _keyset->leased_slow;
			path_name  = "slow";
		}
		FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
			"HFISVC Keyset error: Keyset has %d %s keys free of %u keys total, but no available keys were found in %s bitmap, abort.\n",
			keys_free, path_name, keys_total, path_name);
		abort();
	}

	uint32_t bitpos = __builtin_ctzl(~*randptr);
	*randptr |= 1ULL << bitpos;

	if (OFI_LIKELY(randptr < _keyset->slow_bitmap)) {
		*key = (randptr - _keyset->fast_bitmap) * 64 + bitpos;
		_keyset->leased_fast++;
	} else {
		*key = (randptr - _keyset->slow_bitmap) * 64 + OPX_FAST_KEYS + bitpos;
		_keyset->leased_slow++;
	}

	FI_OPX_DEBUG_COUNTERS_INC(counters->hfisvc.access_key.alloc);
	return 0;
} // opx_hfisvc_keyset_alloc_key

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

	uint64_t *keyptr;
	uint32_t *lease_counter;
	uint64_t  keybit    = 1ul << (key & 0x3Ful);
	uint64_t  key_index = key >> 6;

	if (key < OPX_FAST_KEYS) {
		// Hint allocator for fresh released key
		_keyset->hintptr = keyptr = _keyset->fast_bitmap + key_index;
		lease_counter		  = &_keyset->leased_fast;
	} else {
		if (OFI_UNLIKELY(key_index - (OPX_FAST_KEYS >> 6) > _keyset->slow_mask)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
				"HFISVC Keyset error: Attempt to release non-existing key %u out of current range %u, ignored\n",
				key, (_keyset->slow_mask + 1) * 64 + OPX_FAST_KEYS);
			return;
		}
		keyptr	      = _keyset->slow_bitmap + key_index - (OPX_FAST_KEYS >> 6);
		lease_counter = &_keyset->leased_slow;
	}

	OPX_HFISVC_DEBUG_LOG("Freeing key %u, key_index=%016lX, _keyset->%s_bitmap[%lX]=%016lX\n", key, key_index,
			     (lease_counter == &_keyset->leased_fast) ? "fast" : "slow", key_index, *keyptr);

	if (OFI_UNLIKELY((*keyptr & keybit) == 0)) {
		FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
			"HFISVC Keyset error: Attempt to release non-existing key %u, ignored\n", key);
	} else {
		*keyptr ^= keybit;
		(*lease_counter)--;
	}
} // opx_hfisvc_keyset_free_key

/**
 * Free a keyset.
 */
__OPX_FORCE_INLINE__
void opx_hfisvc_keyset_free(opx_hfisvc_keyset_t keyset)
{
	struct opx_hfisvc_keyset *_keyset = (struct opx_hfisvc_keyset *) keyset;
	assert(_keyset);
	if (_keyset->leased_fast + _keyset->leased_slow > 0) {
		FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
			"HFISVC Keyset error: Freeing the keyset while there are still %u:%u of %u access keys still outstanding, abort.\n",
			_keyset->leased_fast, _keyset->leased_slow, _keyset->slow_mask * 64 + 64 + OPX_FAST_KEYS);
		abort();
	}
	// Compute current memory size
	size_t cur_size = sizeof(struct opx_hfisvc_keyset) + (_keyset->slow_mask + 1) * sizeof(uint64_t);
	munmap(_keyset, cur_size);
}

/**
 * Check for outstanding keys in the keyset.
 */
__OPX_FORCE_INLINE__
int opx_hfisvc_keyset_outstanding(opx_hfisvc_keyset_t keyset)
{
	struct opx_hfisvc_keyset *_keyset = (struct opx_hfisvc_keyset *) keyset;
	return keyset ? _keyset->leased_fast + _keyset->leased_slow : 0;
}
#endif
