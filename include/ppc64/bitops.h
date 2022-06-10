/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef PPC64_BITOPS_H_
#define PPC64_BITOPS_H_

#include <stdint.h>
#include <ofi.h>

static ALWAYS_INLINE unsigned ilog2_u32(uint32_t n)
{
	int bit;
	asm ("cntlzw %0,%1" : "=r" (bit) : "r" (n));
	return 31 - bit;
}

static ALWAYS_INLINE unsigned ilog2_u64(uint64_t n)
{
	int bit;
	asm ("cntlzd %0,%1" : "=r" (bit) : "r" (n));
	return 63 - bit;
}

static ALWAYS_INLINE unsigned _ffs64(uint64_t n)
{
    return ilog2_u64(n & -n);
}

#endif
