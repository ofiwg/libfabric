/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef X86_64_BITOPS_H_
#define X86_64_BITOPS_H_

#include <stdint.h>
#include <ofi.h>

static ALWAYS_INLINE unsigned ilog2_u32(uint32_t n)
{
	uint32_t result;
	asm("bsrl %1,%0"
		: "=r" (result)
		: "r" (n));
	return result;
}

static ALWAYS_INLINE unsigned ilog2_u64(uint64_t n)
{
	uint64_t result;
	asm("bsrq %1,%0"
		: "=r" (result)
		: "r" (n));
	return result;
}

static ALWAYS_INLINE unsigned ffs64(uint64_t n)
{
    uint64_t result;
    asm("bsfq %1,%0"
        : "=r" (result)
        : "r" (n));
    return result;
}

#endif
