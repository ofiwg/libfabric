/*
 * Copyright (c) 2013-2017 Intel Corporation. All rights reserved.
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

#include "ofi_atomic.h"


static const size_t ofi_datatype_size_table[] = {
	[FI_INT8]   = sizeof(int8_t),
	[FI_UINT8]  = sizeof(uint8_t),
	[FI_INT16]  = sizeof(int16_t),
	[FI_UINT16] = sizeof(uint16_t),
	[FI_INT32]  = sizeof(int32_t),
	[FI_UINT32] = sizeof(uint32_t),
	[FI_INT64]  = sizeof(int64_t),
	[FI_UINT64] = sizeof(uint64_t),
	[FI_FLOAT]  = sizeof(float),
	[FI_DOUBLE] = sizeof(double),
	[FI_FLOAT_COMPLEX]  = sizeof(ofi_complex_float),
	[FI_DOUBLE_COMPLEX] = sizeof(ofi_complex_double),
	[FI_LONG_DOUBLE]    = sizeof(long double),
	[FI_LONG_DOUBLE_COMPLEX] = sizeof(ofi_complex_long_double),
};

size_t ofi_datatype_size(enum fi_datatype datatype)
{
	if (datatype >= FI_DATATYPE_LAST) {
		errno = FI_EINVAL;
		return 0;
	}
	return ofi_datatype_size_table[datatype];
}


/*
 * Basic atomic operations
 */

#define OFI_OP_MIN(type,dst,src)   if ((dst) > (src)) (dst) = (src)
#define OFI_OP_MAX(type,dst,src)   if ((dst) < (src)) (dst) = (src)
#define OFI_OP_SUM(type,dst,src)   (dst) += (src)
#define OFI_OP_PROD(type,dst,src)  (dst) *= (src)
#define OFI_OP_LOR(type,dst,src)   (dst) = (dst) || (src)
#define OFI_OP_LAND(type,dst,src)  (dst) = (dst) && (src)
#define OFI_OP_BOR(type,dst,src)   (dst) |= (src)
#define OFI_OP_BAND(type,dst,src)  (dst) &= (src)
#define OFI_OP_LXOR(type,dst,src)  (dst) = ((dst) && !(src)) || (!(dst) && (src))
#define OFI_OP_BXOR(type,dst,src)  (dst) ^= (src)
#define OFI_OP_READ(type,dst,src)  /* src unused, dst is written to result */
#define OFI_OP_WRITE(type,dst,src) (dst) = (src)

#define OFI_OP_CSWAP_EQ(type,dst,src,cmp) if ((dst) == (cmp)) (dst) = (src)
#define OFI_OP_CSWAP_NE(type,dst,src,cmp) if ((dst) != (cmp)) (dst) = (src)
#define OFI_OP_CSWAP_LE(type,dst,src,cmp) if ((dst) <= (cmp)) (dst) = (src)
#define OFI_OP_CSWAP_LT(type,dst,src,cmp) if ((dst) <  (cmp)) (dst) = (src)
#define OFI_OP_CSWAP_GE(type,dst,src,cmp) if ((dst) >= (cmp)) (dst) = (src)
#define OFI_OP_CSWAP_GT(type,dst,src,cmp) if ((dst) >  (cmp)) (dst) = (src)
#define OFI_OP_MSWAP(type,dst,src,cmp)    (dst) = (((src) & (cmp)) | \
						   ((dst) & ~(cmp)))

/* Need special handlers for complex datatypes for portability */
#define OFI_OP_SUM_COMPLEX(type,dst,src)  (dst) = ofi_complex_sum_##type(dst,src)
#define OFI_OP_PROD_COMPLEX(type,dst,src) (dst) = ofi_complex_prod_##type(dst,src)
#define OFI_OP_LOR_COMPLEX(type,dst,src)  (dst) = ofi_complex_lor_##type(dst,src)
#define OFI_OP_LAND_COMPLEX(type,dst,src) (dst) = ofi_complex_land_##type(dst,src)
#define OFI_OP_LXOR_COMPLEX(type,dst,src) (dst) = ofi_complex_lxor_##type(dst,src)
#define OFI_OP_READ_COMPLEX		  OFI_OP_READ
#define OFI_OP_WRITE_COMPLEX		  OFI_OP_WRITE

#define OFI_OP_CSWAP_EQ_COMPLEX(type,dst,src,cmp) \
			if (ofi_complex_eq_##type(dst,cmp)) (dst) = (src)
#define OFI_OP_CSWAP_NE_COMPLEX(type,dst,src,cmp) \
			if (!ofi_complex_eq_##type(dst,cmp)) (dst) = (src)


/********************************
 * ATOMIC TYPE function templates
 ********************************/

#define OFI_DEF_NOOP_NAME NULL,
#define OFI_DEF_NOOP_FUNC

/*
 * WRITE
 */
#define OFI_DEF_WRITE_NAME(op, type) ofi_write_## op ##_## type,
#define OFI_DEF_WRITE_COMPLEX_NAME(op, type) ofi_write_## op ##_## type,

#define OFI_DEF_WRITE_FUNC(op, type)					\
	static void ofi_write_## op ##_## type				\
		(void *dst, const void *src, size_t cnt)		\
	{								\
		size_t i;						\
		type *d = (dst);					\
		const type *s = (src);					\
		for (i = 0; i < cnt; i++)				\
			op(type, d[i], s[i]);				\
	}

#define OFI_DEF_WRITE_COMPLEX_FUNC(op, type)				\
	static void ofi_write_## op ##_## type				\
		(void *dst, const void *src, size_t cnt)		\
	{								\
		size_t i;						\
		ofi_complex_##type *d = (dst);				\
		const ofi_complex_##type *s = (src);			\
		for (i = 0; i < cnt; i++)				\
			op(type, d[i], s[i]);				\
	}

/*
 * READ (fetch)
 */
#define OFI_DEF_READ_NAME(op, type) ofi_read_## op ##_## type,
#define OFI_DEF_READ_COMPLEX_NAME(op, type) ofi_read_## op ##_## type,

#define OFI_DEF_READ_FUNC(op, type)					\
	static void ofi_read_## op ##_## type				\
		(void *dst, const void *src, void *res, size_t cnt) 	\
	{								\
		size_t i;						\
		type *d = (dst);					\
		type *r = (res);					\
		OFI_UNUSED(src);					\
		for (i = 0; i < cnt; i++)				\
			r[i] = d[i];					\
	}

#define OFI_DEF_READ_COMPLEX_FUNC(op, type)				\
	static void ofi_read_## op ##_## type				\
		(void *dst, const void *src, void *res, size_t cnt)	\
	{								\
		size_t i;						\
		ofi_complex_##type *d = (dst);				\
		ofi_complex_##type *r = (res);				\
		OFI_UNUSED(src);					\
		for (i = 0; i < cnt; i++)				\
			r[i] = d[i];					\
	}

/*
 * READWRITE (fetch-write)
 */
#define OFI_DEF_READWRITE_NAME(op, type) ofi_readwrite_## op ##_## type,
#define OFI_DEF_READWRITE_COMPLEX_NAME(op, type) ofi_readwrite_## op ##_## type,

#define OFI_DEF_READWRITE_FUNC(op, type)				\
	static void ofi_readwrite_## op ##_## type			\
		(void *dst, const void *src, void *res, size_t cnt)	\
	{								\
		size_t i;						\
		type *d = (dst);					\
		const type *s = (src);					\
		type *r = (res);					\
		for (i = 0; i < cnt; i++) {				\
			r[i] = d[i];					\
			op(type, d[i], s[i]);				\
		}							\
	}

#define OFI_DEF_READWRITE_COMPLEX_FUNC(op, type)			\
	static void ofi_readwrite_## op ##_## type			\
		(void *dst, const void *src, void *res, size_t cnt)	\
	{								\
		size_t i;						\
		ofi_complex_##type *d = (dst);				\
		const ofi_complex_##type *s = (src);			\
		ofi_complex_##type *r = (res);				\
		for (i = 0; i < cnt; i++) {				\
			r[i] = d[i];					\
			op(type, d[i], s[i]);				\
		}							\
	}

/*
 * CSWAP
 */
#define OFI_DEF_CSWAP_NAME(op, type) ofi_cswap_## op ##_## type,
#define OFI_DEF_CSWAP_COMPLEX_NAME(op, type) ofi_cswap_## op ##_## type,

#define OFI_DEF_CSWAP_FUNC(op, type)					\
	static void ofi_cswap_## op ##_## type				\
		(void *dst, const void *src, const void *cmp,		\
		 void *res, size_t cnt)					\
	{								\
		size_t i;						\
		type *d = (dst);					\
		const type *s = (src);					\
		const type *c = (cmp);					\
		type *r = (res);					\
		for (i = 0; i < cnt; i++) {				\
			r[i] = d[i];					\
			op(type, d[i], s[i], c[i]);			\
		}							\
	}

#define OFI_DEF_CSWAP_COMPLEX_FUNC(op, type)				\
	static void ofi_cswap_## op ##_## type				\
		(void *dst, const void *src, const void *cmp,		\
		 void *res, size_t cnt)					\
	{								\
		size_t i;						\
		ofi_complex_##type *d = (dst);				\
		const ofi_complex_##type *s = (src);			\
		const ofi_complex_##type *c = (cmp);			\
		ofi_complex_##type *r = (res);				\
		for (i = 0; i < cnt; i++) {				\
			r[i] = d[i];					\
			op(type, d[i], s[i], c[i]);			\
		}							\
	}


/*********************************************************************
 * Macros create atomic functions for each operation for each datatype
 *********************************************************************/

/*
 * Define all handlers in order to populate the dispatch table correctly.
 *
 * ATOMICTYPE - WRITE, READ, READWRITE, CSWAP, MSWAP
 * FUNCNAME - Define function or simply generate function name
 *            The latter is needed to populate the dispatch table
 * op - OFI_OP_XXX function should perform (e.g. OFI_OP_MIN)
 */
#define OFI_DEFINE_ALL_HANDLERS(ATOMICTYPE, FUNCNAME, op)		\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, int8_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, uint8_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, int16_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, uint16_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, int32_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, uint32_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, int64_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, uint64_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, float)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, double)			\
	OFI_DEF_##ATOMICTYPE##_COMPLEX_##FUNCNAME(op ##_COMPLEX, float)	\
	OFI_DEF_##ATOMICTYPE##_COMPLEX_##FUNCNAME(op ##_COMPLEX, double)\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, long_double)		\
	OFI_DEF_##ATOMICTYPE##_COMPLEX_##FUNCNAME(op ##_COMPLEX, long_double)

#define OFI_DEFINE_REALNO_HANDLERS(ATOMICTYPE, FUNCNAME, op)		\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, int8_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, uint8_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, int16_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, uint16_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, int32_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, uint32_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, int64_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, uint64_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, float)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, double)			\
	OFI_DEF_NOOP_##FUNCNAME						\
	OFI_DEF_NOOP_##FUNCNAME						\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, long_double)		\
	OFI_DEF_NOOP_##FUNCNAME

#define OFI_DEFINE_INT_HANDLERS(ATOMICTYPE, FUNCNAME, op)		\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, int8_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, uint8_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, int16_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, uint16_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, int32_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, uint32_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, int64_t)			\
	OFI_DEF_##ATOMICTYPE##_##FUNCNAME(op, uint64_t)			\
	OFI_DEF_NOOP_##FUNCNAME						\
	OFI_DEF_NOOP_##FUNCNAME						\
	OFI_DEF_NOOP_##FUNCNAME						\
	OFI_DEF_NOOP_##FUNCNAME						\
	OFI_DEF_NOOP_##FUNCNAME						\
	OFI_DEF_NOOP_##FUNCNAME


/**********************
 * Write dispatch table
 **********************/

OFI_DEFINE_REALNO_HANDLERS(WRITE, FUNC, OFI_OP_MIN)
OFI_DEFINE_REALNO_HANDLERS(WRITE, FUNC, OFI_OP_MAX)
OFI_DEFINE_ALL_HANDLERS(WRITE, FUNC, OFI_OP_SUM)
OFI_DEFINE_ALL_HANDLERS(WRITE, FUNC, OFI_OP_PROD)
OFI_DEFINE_ALL_HANDLERS(WRITE, FUNC, OFI_OP_LOR)
OFI_DEFINE_ALL_HANDLERS(WRITE, FUNC, OFI_OP_LAND)
OFI_DEFINE_INT_HANDLERS(WRITE, FUNC, OFI_OP_BOR)
OFI_DEFINE_INT_HANDLERS(WRITE, FUNC, OFI_OP_BAND)
OFI_DEFINE_ALL_HANDLERS(WRITE, FUNC, OFI_OP_LXOR)
OFI_DEFINE_INT_HANDLERS(WRITE, FUNC, OFI_OP_BXOR)
OFI_DEFINE_ALL_HANDLERS(WRITE, FUNC, OFI_OP_WRITE)

void (*ofi_atomic_write_handlers[OFI_WRITE_OP_LAST][FI_DATATYPE_LAST])
	(void *dst, const void *src, size_t cnt) =
{
	{ OFI_DEFINE_REALNO_HANDLERS(WRITE, NAME, OFI_OP_MIN) },
	{ OFI_DEFINE_REALNO_HANDLERS(WRITE, NAME, OFI_OP_MAX) },
	{ OFI_DEFINE_ALL_HANDLERS(WRITE, NAME, OFI_OP_SUM) },
	{ OFI_DEFINE_ALL_HANDLERS(WRITE, NAME, OFI_OP_PROD) },
	{ OFI_DEFINE_ALL_HANDLERS(WRITE, NAME, OFI_OP_LOR) },
	{ OFI_DEFINE_ALL_HANDLERS(WRITE, NAME, OFI_OP_LAND) },
	{ OFI_DEFINE_INT_HANDLERS(WRITE, NAME, OFI_OP_BOR) },
	{ OFI_DEFINE_INT_HANDLERS(WRITE, NAME, OFI_OP_BAND) },
	{ OFI_DEFINE_ALL_HANDLERS(WRITE, NAME, OFI_OP_LXOR) },
	{ OFI_DEFINE_INT_HANDLERS(WRITE, NAME, OFI_OP_BXOR) },
	 /* no-op: FI_ATOMIC_READ */
	{ NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL},
	{ OFI_DEFINE_ALL_HANDLERS(WRITE, NAME, OFI_OP_WRITE) },
};


/***************************
 * Read-write dispatch table
 ***************************/

OFI_DEFINE_REALNO_HANDLERS(READWRITE, FUNC, OFI_OP_MIN)
OFI_DEFINE_REALNO_HANDLERS(READWRITE, FUNC, OFI_OP_MAX)
OFI_DEFINE_ALL_HANDLERS(READWRITE, FUNC, OFI_OP_SUM)
OFI_DEFINE_ALL_HANDLERS(READWRITE, FUNC, OFI_OP_PROD)
OFI_DEFINE_ALL_HANDLERS(READWRITE, FUNC, OFI_OP_LOR)
OFI_DEFINE_ALL_HANDLERS(READWRITE, FUNC, OFI_OP_LAND)
OFI_DEFINE_INT_HANDLERS(READWRITE, FUNC, OFI_OP_BOR)
OFI_DEFINE_INT_HANDLERS(READWRITE, FUNC, OFI_OP_BAND)
OFI_DEFINE_ALL_HANDLERS(READWRITE, FUNC, OFI_OP_LXOR)
OFI_DEFINE_INT_HANDLERS(READWRITE, FUNC, OFI_OP_BXOR)
OFI_DEFINE_ALL_HANDLERS(READ, FUNC, OFI_OP_READ)
OFI_DEFINE_ALL_HANDLERS(READWRITE, FUNC, OFI_OP_WRITE)

void (*ofi_atomic_readwrite_handlers[OFI_READWRITE_OP_LAST][FI_DATATYPE_LAST])
	(void *dst, const void *src, void *res, size_t cnt) =
{
	{ OFI_DEFINE_REALNO_HANDLERS(READWRITE, NAME, OFI_OP_MIN) },
	{ OFI_DEFINE_REALNO_HANDLERS(READWRITE, NAME, OFI_OP_MAX) },
	{ OFI_DEFINE_ALL_HANDLERS(READWRITE, NAME, OFI_OP_SUM) },
	{ OFI_DEFINE_ALL_HANDLERS(READWRITE, NAME, OFI_OP_PROD) },
	{ OFI_DEFINE_ALL_HANDLERS(READWRITE, NAME, OFI_OP_LOR) },
	{ OFI_DEFINE_ALL_HANDLERS(READWRITE, NAME, OFI_OP_LAND) },
	{ OFI_DEFINE_INT_HANDLERS(READWRITE, NAME, OFI_OP_BOR) },
	{ OFI_DEFINE_INT_HANDLERS(READWRITE, NAME, OFI_OP_BAND) },
	{ OFI_DEFINE_ALL_HANDLERS(READWRITE, NAME, OFI_OP_LXOR) },
	{ OFI_DEFINE_INT_HANDLERS(READWRITE, NAME, OFI_OP_BXOR) },
	{ OFI_DEFINE_ALL_HANDLERS(READ, NAME, OFI_OP_READ) },
	{ OFI_DEFINE_ALL_HANDLERS(READWRITE, NAME, OFI_OP_WRITE) },
};


/*****************************
 * Compare-swap dispatch table
 *****************************/

OFI_DEFINE_ALL_HANDLERS(CSWAP, FUNC, OFI_OP_CSWAP_EQ)
OFI_DEFINE_ALL_HANDLERS(CSWAP, FUNC, OFI_OP_CSWAP_NE)
OFI_DEFINE_REALNO_HANDLERS(CSWAP, FUNC, OFI_OP_CSWAP_LE)
OFI_DEFINE_REALNO_HANDLERS(CSWAP, FUNC, OFI_OP_CSWAP_LT)
OFI_DEFINE_REALNO_HANDLERS(CSWAP, FUNC, OFI_OP_CSWAP_GE)
OFI_DEFINE_REALNO_HANDLERS(CSWAP, FUNC, OFI_OP_CSWAP_GT)
OFI_DEFINE_INT_HANDLERS(CSWAP, FUNC, OFI_OP_MSWAP)

void (*ofi_atomic_swap_handlers[OFI_SWAP_OP_LAST][FI_DATATYPE_LAST])
	(void *dst, const void *src, const void *cmp, void *res, size_t cnt) =
{
	{ OFI_DEFINE_ALL_HANDLERS(CSWAP, NAME, OFI_OP_CSWAP_EQ) },
	{ OFI_DEFINE_ALL_HANDLERS(CSWAP, NAME, OFI_OP_CSWAP_NE) },
	{ OFI_DEFINE_REALNO_HANDLERS(CSWAP, NAME, OFI_OP_CSWAP_LE) },
	{ OFI_DEFINE_REALNO_HANDLERS(CSWAP, NAME, OFI_OP_CSWAP_LT) },
	{ OFI_DEFINE_REALNO_HANDLERS(CSWAP, NAME, OFI_OP_CSWAP_GE) },
	{ OFI_DEFINE_REALNO_HANDLERS(CSWAP, NAME, OFI_OP_CSWAP_GT) },
	{ OFI_DEFINE_INT_HANDLERS(CSWAP, NAME, OFI_OP_MSWAP) },
};


int ofi_atomic_valid(const struct fi_provider *prov,
		     enum fi_datatype datatype, enum fi_op op, uint64_t flags)
{
	int have_func;

	if (flags & FI_TAGGED) {
		/* Only tagged atomic write operations currently supported */
		if (flags & (FI_FETCH_ATOMIC | FI_COMPARE_ATOMIC)) {
			FI_INFO(prov, FI_LOG_DOMAIN,
				"Only tagged atomic writes supported\n");
			return -FI_ENOSYS;
		}
	} else if (flags & ~(FI_FETCH_ATOMIC | FI_COMPARE_ATOMIC)) {
		FI_INFO(prov, FI_LOG_DOMAIN, "Unknown flag specified\n");
		 return -FI_EBADFLAGS;
	} else if ((flags & FI_FETCH_ATOMIC) && (flags & FI_COMPARE_ATOMIC)) {
		FI_INFO(prov, FI_LOG_DOMAIN, "Invalid flag combination\n");
		return -FI_EBADFLAGS;
	}

	if (datatype >= FI_DATATYPE_LAST) {
		FI_INFO(prov, FI_LOG_DOMAIN, "Invalid datatype\n");
		return -FI_EOPNOTSUPP;
	}

	if (flags & FI_FETCH_ATOMIC) {
		if (op >= OFI_READWRITE_OP_LAST) {
			FI_INFO(prov, FI_LOG_DOMAIN, "Invalid fetch operation\n");
			return -FI_EOPNOTSUPP;
		}
		have_func = ofi_atomic_readwrite_handlers[op][datatype] != NULL;
	} else if (flags & FI_COMPARE_ATOMIC) {
		if (op < FI_CSWAP || op > FI_MSWAP) {
			FI_INFO(prov, FI_LOG_DOMAIN, "Invalid swap operation\n");
			return -FI_EOPNOTSUPP;
		}
		have_func = ofi_atomic_swap_handlers[op - FI_CSWAP][datatype] != NULL;
	} else {
		if (op >= OFI_WRITE_OP_LAST) {
			FI_INFO(prov, FI_LOG_DOMAIN, "Invalid write operation\n");
			return -FI_EOPNOTSUPP;
		}
		have_func = ofi_atomic_write_handlers[op][datatype] != NULL;
	}

	if (!have_func) {
		FI_INFO(prov, FI_LOG_DOMAIN, "Datatype/op combo not supported\n");
		return -FI_EOPNOTSUPP;
	}

	return 0;
}
