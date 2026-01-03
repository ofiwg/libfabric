/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

/*
 * ofi_bloom.h - Bloom filter implementation for libfabric
 *
 * This is a block-based bloom filter inspired by Boost.Bloom.
 * It uses a single hash value and derives multiple bit positions through
 * multiplicative hashing, providing O(1) insertion and lookup with a
 * configurable false positive rate.
 *
 * Key design choices:
 * - Block-based: Each insert/lookup touches only one cache line (64 bytes)
 * - Branchless lookup: Consistent performance regardless of hit/miss
 * - Counter-based: Supports deletion (counting bloom filter variant)
 * - SIMD-friendly: Block size matches typical SIMD register width
 *
 * Usage:
 *   struct ofi_bloom bloom;
 *   ofi_bloom_init(&bloom, expected_elements, target_fpr);
 *   ofi_bloom_insert(&bloom, hash_value);
 *   if (ofi_bloom_may_contain(&bloom, hash_value)) { ... }
 *   ofi_bloom_remove(&bloom, hash_value);
 *   ofi_bloom_fini(&bloom);
 */

#ifndef _OFI_BLOOM_H_
#define _OFI_BLOOM_H_

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <rdma/fi_errno.h>

/*
 * Multiplier for mulx64 hash derivation (2^64 / phi, from Boost.Bloom).
 * This is used in the mulx64 function to derive additional hash values
 * via 128-bit multiplication when we exhaust the bits in the current hash.
 */
#define OFI_BLOOM_MULX_CONST 0x9e3779b97f4a7c15ULL

/*
 * Block size - 64 bytes aligned to cache line.
 * Each insert/lookup only touches one block for cache efficiency.
 *
 * For a counting bloom filter, each byte is a counter, giving us
 * 64 counters per block. We use 6 bits from the hash to select
 * which counter (2^6 = 64).
 */
#define OFI_BLOOM_BLOCK_BYTES 64
#define OFI_BLOOM_BLOCK_COUNTERS OFI_BLOOM_BLOCK_BYTES
#define OFI_BLOOM_COUNTER_BITS 6 /* log2(64) = 6 */

/*
 * Default number of bits set per insert (k' in literature).
 * Higher values reduce false positives but increase lookup time.
 * 8 provides a good balance for most use cases.
 */
#define OFI_BLOOM_DEFAULT_BITS_PER_INSERT 8

/*
 * Bloom filter block - 64 bytes aligned to cache line.
 * Uses 8-bit counters to support deletion (counting bloom filter).
 * With 64 counters per block and k=8 probes, we get good distribution.
 */
struct ofi_bloom_block {
	uint8_t counters[OFI_BLOOM_BLOCK_COUNTERS];
} __attribute__((aligned(64)));

/*
 * Bloom filter structure.
 *
 * @blocks:          Array of counter blocks
 * @num_blocks:      Number of blocks (r in Boost.Bloom notation)
 * @bits_per_insert: Number of bits set per element (k')
 * @num_elements:    Current count of inserted elements
 */
struct ofi_bloom {
	struct ofi_bloom_block *blocks;
	size_t num_blocks;
	unsigned int bits_per_insert;
	size_t num_elements;
};

/*
 * ofi_bloom_mulx64() - Derive a new hash value via 128-bit multiply.
 *
 * This is the hash derivation function from Boost.Bloom. It multiplies
 * the input by the golden ratio constant using 128-bit arithmetic,
 * then XORs the high and low 64-bit halves for maximum mixing.
 *
 * This is called when we've exhausted the usable bits from the current
 * hash value and need to generate more independent bit positions.
 */
static inline uint64_t ofi_bloom_mulx64(uint64_t h)
{
#if defined(__SIZEOF_INT128__)
	__uint128_t r = (__uint128_t)h * OFI_BLOOM_MULX_CONST;
	return (uint64_t)(r >> 64) ^ (uint64_t)r;
#else
	/* Fallback 64-bit implementation */
	uint64_t h_lo = (uint32_t)h;
	uint64_t h_hi = h >> 32;
	uint64_t c_lo = (uint32_t)OFI_BLOOM_MULX_CONST;
	uint64_t c_hi = OFI_BLOOM_MULX_CONST >> 32;

	uint64_t r3 = h_hi * c_hi;
	uint64_t r2a = h_lo * c_hi;
	uint64_t r2b = h_hi * c_lo;
	uint64_t r1 = h_lo * c_lo;

	r3 += r2a >> 32;
	r3 += r2b >> 32;

	uint64_t r2 = (r1 >> 32) + (uint32_t)r2a + (uint32_t)r2b;
	uint64_t lo = (r2 << 32) + (uint32_t)r1;
	uint64_t hi = r3 + (r2 >> 32);

	return hi ^ lo;
#endif
}

/*
 * ofi_bloom_fastrange() - Map hash to [0, range) without modulo.
 *
 * Uses Lemire's fastrange technique: multiply and take high bits.
 * This is faster than modulo and has good distribution properties.
 *
 * @h:     64-bit hash value
 * @range: Upper bound (exclusive)
 *
 * Returns: Value in [0, range)
 */
static inline size_t ofi_bloom_fastrange(uint64_t h, size_t range)
{
	/* Compute (h * range) >> 64 using 128-bit arithmetic */
#if defined(__SIZEOF_INT128__)
	return (size_t)(((__uint128_t)h * range) >> 64);
#else
	/* Fallback for platforms without 128-bit support */
	uint64_t h_lo = (uint32_t)h;
	uint64_t h_hi = h >> 32;
	uint64_t r_lo = (uint32_t)range;
	uint64_t r_hi = range >> 32;

	uint64_t cross1 = h_lo * r_hi;
	uint64_t cross2 = h_hi * r_lo;
	uint64_t high = h_hi * r_hi;

	uint64_t carry = ((h_lo * r_lo) >> 32) + (cross1 & 0xFFFFFFFF) +
			 (cross2 & 0xFFFFFFFF);
	return (size_t)(high + (cross1 >> 32) + (cross2 >> 32) +
			(carry >> 32));
#endif
}

/*
 * ofi_bloom_init() - Initialize a bloom filter.
 *
 * Allocates memory for the filter sized to achieve approximately the
 * target false positive rate for the expected number of elements.
 *
 * @bloom:             Bloom filter to initialize
 * @expected_elements: Expected number of elements to insert
 * @target_fpr:        Target false positive rate (0.0 to 1.0)
 *
 * Returns: 0 on success, -FI_ENOMEM on allocation failure
 *
 * Note: If target_fpr is 0, a reasonable default is used.
 *       The actual FPR may differ from target depending on usage.
 */
static inline int ofi_bloom_init(struct ofi_bloom *bloom,
				 size_t expected_elements,
				 double target_fpr)
{
	size_t counters_needed;
	size_t num_blocks;

	bloom->bits_per_insert = OFI_BLOOM_DEFAULT_BITS_PER_INSERT;

	/*
	 * Calculate required counters using the formula:
	 * m = -n * ln(p) / (ln(2)^2)
	 *
	 * For simplicity, we use an approximation:
	 * m ≈ n * 10 for ~1% FPR
	 * m ≈ n * 15 for ~0.1% FPR
	 * m ≈ n * 20 for ~0.01% FPR
	 */
	if (target_fpr <= 0.0001)
		counters_needed = expected_elements * 20;
	else if (target_fpr <= 0.001)
		counters_needed = expected_elements * 15;
	else if (target_fpr <= 0.01)
		counters_needed = expected_elements * 10;
	else
		counters_needed = expected_elements * 8;

	/* Round up to nearest block (each block has BLOCK_COUNTERS counters) */
	num_blocks = (counters_needed + OFI_BLOOM_BLOCK_COUNTERS - 1) /
		     OFI_BLOOM_BLOCK_COUNTERS;
	if (num_blocks < 1)
		num_blocks = 1;

	bloom->blocks = calloc(num_blocks, sizeof(struct ofi_bloom_block));
	if (!bloom->blocks)
		return -FI_ENOMEM;

	bloom->num_blocks = num_blocks;
	bloom->num_elements = 0;

	return 0;
}

/*
 * ofi_bloom_init_sized() - Initialize with explicit size.
 *
 * @bloom:           Bloom filter to initialize
 * @num_blocks:      Number of 64-byte blocks to allocate
 * @bits_per_insert: Number of bits to set per element
 *
 * Returns: 0 on success, -FI_ENOMEM on allocation failure
 */
static inline int ofi_bloom_init_sized(struct ofi_bloom *bloom,
				       size_t num_blocks,
				       unsigned int bits_per_insert)
{
	if (num_blocks < 1)
		num_blocks = 1;
	if (bits_per_insert < 1 || bits_per_insert > 16)
		bits_per_insert = OFI_BLOOM_DEFAULT_BITS_PER_INSERT;

	bloom->blocks = calloc(num_blocks, sizeof(struct ofi_bloom_block));
	if (!bloom->blocks)
		return -FI_ENOMEM;

	bloom->num_blocks = num_blocks;
	bloom->bits_per_insert = bits_per_insert;
	bloom->num_elements = 0;

	return 0;
}

/*
 * ofi_bloom_fini() - Clean up and free bloom filter resources.
 */
static inline void ofi_bloom_fini(struct ofi_bloom *bloom)
{
	free(bloom->blocks);
	bloom->blocks = NULL;
	bloom->num_blocks = 0;
	bloom->num_elements = 0;
}

/*
 * ofi_bloom_clear() - Reset all counters to zero.
 */
static inline void ofi_bloom_clear(struct ofi_bloom *bloom)
{
	memset(bloom->blocks, 0,
	       bloom->num_blocks * sizeof(struct ofi_bloom_block));
	bloom->num_elements = 0;
}

/*
 * ofi_bloom_insert() - Insert an element (by its hash) into the filter.
 *
 * Sets bits_per_insert counters in a single block determined by the hash.
 * Uses counting (increment counters) to support deletion.
 *
 * Hash derivation follows Boost.Bloom's approach:
 * - Extract OFI_BLOOM_COUNTER_BITS bits at a time by shifting right
 * - When bits are exhausted, call mulx64 to generate fresh bits
 * - This maximizes the entropy extracted from each hash computation
 *
 * @bloom: Bloom filter
 * @hash:  Hash value of the element to insert
 */
static inline void ofi_bloom_insert(struct ofi_bloom *bloom, uint64_t hash)
{
	struct ofi_bloom_block *block;
	size_t block_idx;
	uint64_t h;
	unsigned int i;
	unsigned int counter_idx;

	/*
	 * How many counter positions can we extract from a 64-bit hash?
	 * We skip the top bits used for block selection and consume
	 * COUNTER_BITS at a time. After exhausting, we call mulx64.
	 *
	 * With 6-bit counters: (64 - 6) / 6 = 9 positions per hash
	 * (We reserve top bits for block selection in first iteration)
	 */
	static const unsigned int positions_per_hash =
		(64 - OFI_BLOOM_COUNTER_BITS) / OFI_BLOOM_COUNTER_BITS;

	/* Select block using fastrange on original hash */
	block_idx = ofi_bloom_fastrange(hash, bloom->num_blocks);
	block = &bloom->blocks[block_idx];

	h = hash;
	unsigned int positions_used = 0;

	/* Set bits_per_insert counters in the selected block */
	for (i = 0; i < bloom->bits_per_insert; i++) {
		/* Shift right to consume next COUNTER_BITS */
		h >>= OFI_BLOOM_COUNTER_BITS;

		/* Select counter within block using bottom bits */
		counter_idx = (unsigned int)(h & (OFI_BLOOM_BLOCK_COUNTERS - 1));

		/* Increment counter (saturating at 255) */
		if (block->counters[counter_idx] < 255)
			block->counters[counter_idx]++;

		positions_used++;
		if (positions_used >= positions_per_hash && i + 1 < bloom->bits_per_insert) {
			/* Exhausted bits, derive fresh hash via mulx64 */
			h = ofi_bloom_mulx64(h);
			positions_used = 0;
		}
	}

	bloom->num_elements++;
}

/*
 * ofi_bloom_may_contain() - Check if an element might be in the filter.
 *
 * Returns true if the element may be present (with possible false positive),
 * or false if the element is definitely not present.
 *
 * Uses branchless implementation for consistent performance.
 * Hash derivation matches ofi_bloom_insert exactly.
 *
 * @bloom: Bloom filter
 * @hash:  Hash value of the element to check
 *
 * Returns: true if element may be present, false if definitely not
 */
static inline bool ofi_bloom_may_contain(struct ofi_bloom *bloom, uint64_t hash)
{
	struct ofi_bloom_block *block;
	size_t block_idx;
	uint64_t h;
	unsigned int i;
	unsigned int counter_idx;
	uint64_t result = ~0ULL; /* All bits set - will be ANDed */

	static const unsigned int positions_per_hash =
		(64 - OFI_BLOOM_COUNTER_BITS) / OFI_BLOOM_COUNTER_BITS;

	/* Select block using fastrange on original hash */
	block_idx = ofi_bloom_fastrange(hash, bloom->num_blocks);
	block = &bloom->blocks[block_idx];

	h = hash;
	unsigned int positions_used = 0;

	/* Check all bits_per_insert counters (branchless) */
	for (i = 0; i < bloom->bits_per_insert; i++) {
		/* Shift right to consume next COUNTER_BITS */
		h >>= OFI_BLOOM_COUNTER_BITS;

		/* Select counter within block using bottom bits */
		counter_idx = (unsigned int)(h & (OFI_BLOOM_BLOCK_COUNTERS - 1));

		/* Accumulate result (branchless: AND with counter != 0) */
		result &= -(uint64_t)(block->counters[counter_idx] != 0);

		positions_used++;
		if (positions_used >= positions_per_hash && i + 1 < bloom->bits_per_insert) {
			/* Exhausted bits, derive fresh hash via mulx64 */
			h = ofi_bloom_mulx64(h);
			positions_used = 0;
		}
	}

	return result != 0;
}

/*
 * ofi_bloom_remove() - Remove an element from the filter.
 *
 * Decrements counters for the element's bits. Only valid if the
 * element was previously inserted. Removing an element that wasn't
 * inserted leads to undefined behavior.
 *
 * Hash derivation matches ofi_bloom_insert exactly.
 *
 * @bloom: Bloom filter
 * @hash:  Hash value of the element to remove
 */
static inline void ofi_bloom_remove(struct ofi_bloom *bloom, uint64_t hash)
{
	struct ofi_bloom_block *block;
	size_t block_idx;
	uint64_t h;
	unsigned int i;
	unsigned int counter_idx;

	static const unsigned int positions_per_hash =
		(64 - OFI_BLOOM_COUNTER_BITS) / OFI_BLOOM_COUNTER_BITS;

	/* Select block using fastrange on original hash */
	block_idx = ofi_bloom_fastrange(hash, bloom->num_blocks);
	block = &bloom->blocks[block_idx];

	h = hash;
	unsigned int positions_used = 0;

	/* Decrement bits_per_insert counters in the selected block */
	for (i = 0; i < bloom->bits_per_insert; i++) {
		/* Shift right to consume next COUNTER_BITS */
		h >>= OFI_BLOOM_COUNTER_BITS;

		/* Select counter within block using bottom bits */
		counter_idx = (unsigned int)(h & (OFI_BLOOM_BLOCK_COUNTERS - 1));

		/* Decrement counter (don't underflow) */
		if (block->counters[counter_idx] > 0)
			block->counters[counter_idx]--;

		positions_used++;
		if (positions_used >= positions_per_hash && i + 1 < bloom->bits_per_insert) {
			/* Exhausted bits, derive fresh hash via mulx64 */
			h = ofi_bloom_mulx64(h);
			positions_used = 0;
		}
	}

	if (bloom->num_elements > 0)
		bloom->num_elements--;
}

/*
 * ofi_bloom_count() - Return the number of inserted elements.
 *
 * Note: This is an approximation if remove() has been used incorrectly.
 */
static inline size_t ofi_bloom_count(struct ofi_bloom *bloom)
{
	return bloom->num_elements;
}

/*
 * ofi_bloom_empty() - Check if the filter is empty.
 */
static inline bool ofi_bloom_empty(struct ofi_bloom *bloom)
{
	return bloom->num_elements == 0;
}

/*
 * ofi_bloom_fpr() - Estimate the current false positive rate.
 *
 * Uses the classical bloom filter FPR formula adjusted for block filters.
 * This is an approximation and actual FPR may vary.
 *
 * @bloom: Bloom filter
 *
 * Returns: Estimated false positive rate (0.0 to 1.0)
 */
static inline double ofi_bloom_fpr(struct ofi_bloom *bloom)
{
	/* m = total number of counters in the filter */
	double m = (double)(bloom->num_blocks * OFI_BLOOM_BLOCK_COUNTERS);
	double n = (double)bloom->num_elements;
	double k = (double)bloom->bits_per_insert;
	double exponent;
	double base;
	double result;
	int i;

	if (n == 0)
		return 0.0;

	/*
	 * Classical FPR formula: (1 - e^(-kn/m))^k
	 * We use a power approximation since we don't want to pull in libm.
	 */
	exponent = -k * n / m;

	/* Approximate e^x using (1 + x/256)^256 for |x| < 1 */
	if (exponent > -1.0 && exponent < 1.0) {
		base = 1.0 + exponent / 256.0;
		result = base;
		for (i = 0; i < 8; i++)
			result = result * result;
		result = 1.0 - result;
	} else {
		/* For large negative exponents, e^x ≈ 0 */
		result = 1.0;
	}

	/* Raise to power k */
	base = result;
	result = 1.0;
	for (i = 0; i < (int)k; i++)
		result *= base;

	return result;
}

/*
 * ============================================================================
 * Non-Counting Bloom Filter (ofi_bloom_set)
 * ============================================================================
 *
 * A standard bit-based bloom filter that does NOT support deletion.
 * Use this when:
 * - Elements are only added, never removed
 * - Memory efficiency is important (8x more bits per byte than counting)
 * - Periodic rebuilds are acceptable if the filter fills up
 *
 * Each 64-byte block contains 512 bits instead of 64 counters.
 * This means 8x better memory efficiency at the cost of no deletion.
 */

/*
 * For bit-based filter: 64 bytes = 512 bits, need 9 bits to index (2^9 = 512)
 */
#define OFI_BLOOM_SET_BLOCK_BITS (OFI_BLOOM_BLOCK_BYTES * 8)
#define OFI_BLOOM_SET_BIT_INDEX_BITS 9 /* log2(512) = 9 */

/*
 * Bit-based bloom filter block - 64 bytes aligned to cache line.
 * Contains 512 bits for maximum memory efficiency.
 */
struct ofi_bloom_set_block {
	uint64_t bits[OFI_BLOOM_BLOCK_BYTES / sizeof(uint64_t)];
} __attribute__((aligned(64)));

/*
 * Bit-based bloom filter structure (no deletion support).
 *
 * @blocks:          Array of bit blocks
 * @num_blocks:      Number of blocks
 * @bits_per_insert: Number of bits set per element (k)
 * @num_elements:    Count of inserted elements
 */
struct ofi_bloom_set {
	struct ofi_bloom_set_block *blocks;
	size_t num_blocks;
	unsigned int bits_per_insert;
	size_t num_elements;
};

/*
 * ofi_bloom_set_init() - Initialize a bit-based bloom filter.
 *
 * @bloom:             Bloom filter to initialize
 * @expected_elements: Expected number of elements to insert
 * @target_fpr:        Target false positive rate (0.0 to 1.0)
 *
 * Returns: 0 on success, -FI_ENOMEM on allocation failure
 */
static inline int ofi_bloom_set_init(struct ofi_bloom_set *bloom,
				     size_t expected_elements,
				     double target_fpr)
{
	size_t bits_needed;
	size_t num_blocks;

	bloom->bits_per_insert = OFI_BLOOM_DEFAULT_BITS_PER_INSERT;

	/*
	 * Calculate required bits using the formula:
	 * m = -n * ln(p) / (ln(2)^2)
	 *
	 * Approximations:
	 * m ≈ n * 10 for ~1% FPR
	 * m ≈ n * 15 for ~0.1% FPR
	 * m ≈ n * 20 for ~0.01% FPR
	 */
	if (target_fpr <= 0.0001)
		bits_needed = expected_elements * 20;
	else if (target_fpr <= 0.001)
		bits_needed = expected_elements * 15;
	else if (target_fpr <= 0.01)
		bits_needed = expected_elements * 10;
	else
		bits_needed = expected_elements * 8;

	/* Round up to nearest block (each block has 512 bits) */
	num_blocks = (bits_needed + OFI_BLOOM_SET_BLOCK_BITS - 1) /
		     OFI_BLOOM_SET_BLOCK_BITS;
	if (num_blocks < 1)
		num_blocks = 1;

	bloom->blocks = calloc(num_blocks, sizeof(struct ofi_bloom_set_block));
	if (!bloom->blocks)
		return -FI_ENOMEM;

	bloom->num_blocks = num_blocks;
	bloom->num_elements = 0;

	return 0;
}

/*
 * ofi_bloom_set_init_sized() - Initialize with explicit size.
 *
 * @bloom:           Bloom filter to initialize
 * @num_blocks:      Number of 64-byte blocks to allocate
 * @bits_per_insert: Number of bits to set per element
 *
 * Returns: 0 on success, -FI_ENOMEM on allocation failure
 */
static inline int ofi_bloom_set_init_sized(struct ofi_bloom_set *bloom,
					   size_t num_blocks,
					   unsigned int bits_per_insert)
{
	if (num_blocks < 1)
		num_blocks = 1;
	if (bits_per_insert < 1 || bits_per_insert > 16)
		bits_per_insert = OFI_BLOOM_DEFAULT_BITS_PER_INSERT;

	bloom->blocks = calloc(num_blocks, sizeof(struct ofi_bloom_set_block));
	if (!bloom->blocks)
		return -FI_ENOMEM;

	bloom->num_blocks = num_blocks;
	bloom->bits_per_insert = bits_per_insert;
	bloom->num_elements = 0;

	return 0;
}

/*
 * ofi_bloom_set_fini() - Clean up and free bloom filter resources.
 */
static inline void ofi_bloom_set_fini(struct ofi_bloom_set *bloom)
{
	free(bloom->blocks);
	bloom->blocks = NULL;
	bloom->num_blocks = 0;
	bloom->num_elements = 0;
}

/*
 * ofi_bloom_set_clear() - Reset all bits to zero.
 */
static inline void ofi_bloom_set_clear(struct ofi_bloom_set *bloom)
{
	memset(bloom->blocks, 0,
	       bloom->num_blocks * sizeof(struct ofi_bloom_set_block));
	bloom->num_elements = 0;
}

/*
 * ofi_bloom_set_insert() - Insert an element into the filter.
 *
 * Sets bits_per_insert bits in a single block determined by the hash.
 * This is a one-way operation - bits cannot be unset without clearing
 * the entire filter.
 *
 * @bloom: Bloom filter
 * @hash:  Hash value of the element to insert
 */
static inline void ofi_bloom_set_insert(struct ofi_bloom_set *bloom,
					uint64_t hash)
{
	struct ofi_bloom_set_block *block;
	size_t block_idx;
	uint64_t h;
	unsigned int i;
	unsigned int bit_idx;
	unsigned int word_idx;
	unsigned int bit_in_word;

	/*
	 * How many bit positions can we extract from a 64-bit hash?
	 * With 9-bit indices: (64 - 9) / 9 = 6 positions per hash
	 */
	static const unsigned int positions_per_hash =
		(64 - OFI_BLOOM_SET_BIT_INDEX_BITS) / OFI_BLOOM_SET_BIT_INDEX_BITS;

	/* Select block using fastrange on original hash */
	block_idx = ofi_bloom_fastrange(hash, bloom->num_blocks);
	block = &bloom->blocks[block_idx];

	h = hash;
	unsigned int positions_used = 0;

	/* Set bits_per_insert bits in the selected block */
	for (i = 0; i < bloom->bits_per_insert; i++) {
		/* Shift right to consume next BIT_INDEX_BITS */
		h >>= OFI_BLOOM_SET_BIT_INDEX_BITS;

		/* Select bit within block (0 to 511) */
		bit_idx = (unsigned int)(h & (OFI_BLOOM_SET_BLOCK_BITS - 1));

		/* Set the bit */
		word_idx = bit_idx / 64;
		bit_in_word = bit_idx % 64;
		block->bits[word_idx] |= (1ULL << bit_in_word);

		positions_used++;
		if (positions_used >= positions_per_hash &&
		    i + 1 < bloom->bits_per_insert) {
			h = ofi_bloom_mulx64(h);
			positions_used = 0;
		}
	}

	bloom->num_elements++;
}

/*
 * ofi_bloom_set_may_contain() - Check if an element might be in the filter.
 *
 * Returns true if the element may be present (with possible false positive),
 * or false if the element is definitely not present.
 *
 * @bloom: Bloom filter
 * @hash:  Hash value of the element to check
 *
 * Returns: true if element may be present, false if definitely not
 */
static inline bool ofi_bloom_set_may_contain(struct ofi_bloom_set *bloom,
					     uint64_t hash)
{
	struct ofi_bloom_set_block *block;
	size_t block_idx;
	uint64_t h;
	unsigned int i;
	unsigned int bit_idx;
	unsigned int word_idx;
	unsigned int bit_in_word;
	uint64_t result = ~0ULL;

	static const unsigned int positions_per_hash =
		(64 - OFI_BLOOM_SET_BIT_INDEX_BITS) / OFI_BLOOM_SET_BIT_INDEX_BITS;

	/* Select block using fastrange on original hash */
	block_idx = ofi_bloom_fastrange(hash, bloom->num_blocks);
	block = &bloom->blocks[block_idx];

	h = hash;
	unsigned int positions_used = 0;

	/* Check all bits_per_insert bits (branchless) */
	for (i = 0; i < bloom->bits_per_insert; i++) {
		/* Shift right to consume next BIT_INDEX_BITS */
		h >>= OFI_BLOOM_SET_BIT_INDEX_BITS;

		/* Select bit within block (0 to 511) */
		bit_idx = (unsigned int)(h & (OFI_BLOOM_SET_BLOCK_BITS - 1));

		/* Check the bit (branchless) */
		word_idx = bit_idx / 64;
		bit_in_word = bit_idx % 64;
		result &= -(uint64_t)((block->bits[word_idx] >> bit_in_word) & 1);

		positions_used++;
		if (positions_used >= positions_per_hash &&
		    i + 1 < bloom->bits_per_insert) {
			h = ofi_bloom_mulx64(h);
			positions_used = 0;
		}
	}

	return result != 0;
}

/*
 * ofi_bloom_set_count() - Return the number of inserted elements.
 */
static inline size_t ofi_bloom_set_count(struct ofi_bloom_set *bloom)
{
	return bloom->num_elements;
}

/*
 * ofi_bloom_set_empty() - Check if the filter is empty.
 */
static inline bool ofi_bloom_set_empty(struct ofi_bloom_set *bloom)
{
	return bloom->num_elements == 0;
}

/*
 * ofi_bloom_set_fpr() - Estimate the current false positive rate.
 *
 * @bloom: Bloom filter
 *
 * Returns: Estimated false positive rate (0.0 to 1.0)
 */
static inline double ofi_bloom_set_fpr(struct ofi_bloom_set *bloom)
{
	/* m = total number of bits in the filter */
	double m = (double)(bloom->num_blocks * OFI_BLOOM_SET_BLOCK_BITS);
	double n = (double)bloom->num_elements;
	double k = (double)bloom->bits_per_insert;
	double exponent;
	double base;
	double result;
	int i;

	if (n == 0)
		return 0.0;

	exponent = -k * n / m;

	if (exponent > -1.0 && exponent < 1.0) {
		base = 1.0 + exponent / 256.0;
		result = base;
		for (i = 0; i < 8; i++)
			result = result * result;
		result = 1.0 - result;
	} else {
		result = 1.0;
	}

	base = result;
	result = 1.0;
	for (i = 0; i < (int)k; i++)
		result *= base;

	return result;
}

#endif /* _OFI_BLOOM_H_ */
