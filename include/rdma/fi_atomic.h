/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
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

#ifndef _FI_ATOMIC_H_
#define _FI_ATOMIC_H_

#include <assert.h>
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>


#ifdef __cplusplus
extern "C" {
#endif

#ifndef FABRIC_DIRECT

enum fi_datatype {
	FI_INT8,
	FI_UINT8,
	FI_INT16,
	FI_UINT16,
	FI_INT32,
	FI_UINT32,
	FI_INT64,
	FI_UINT64,
	FI_FLOAT,
	FI_DOUBLE,
	FI_FLOAT_COMPLEX,
	FI_DOUBLE_COMPLEX,
	FI_LONG_DOUBLE,
	FI_LONG_DOUBLE_COMPLEX,
};

enum fi_op {
	FI_MIN,
	FI_MAX,
	FI_SUM,
	FI_PROD,
	FI_LOR,
	FI_LAND,
	FI_BOR,
	FI_BAND,
	FI_LXOR,
	FI_BXOR,
	FI_ATOMIC_READ,
	FI_ATOMIC_WRITE,
	FI_CSWAP,
	FI_CSWAP_NE,
	FI_CSWAP_LE,
	FI_CSWAP_LT,
	FI_CSWAP_GE,
	FI_CSWAP_GT,
	FI_MSWAP,
};

#endif /* FABRIC_DIRECT */

struct fi_msg_atomic {
	const void		*msg_iov;
	size_t			iov_count;
	const void		*addr;
	const struct fi_rma_iov *rma_iov;
	size_t			rma_iov_count;
	int			datatype;
	int			op;
	void			*context;
	uint64_t		data;
};

struct fi_ops_atomic {
	size_t	size;
	int	(*write)(fid_t fid,
			const void *buf, size_t len,
			uint64_t addr, uint64_t key,
			int datatype, int op, void *context);
	int	(*writemem)(fid_t fid,
			const void *buf, size_t len, uint64_t mem_desc,
			uint64_t addr, uint64_t key,
			int datatype, int op, void *context);
	int	(*writev)(fid_t fid,
			const void *iov, size_t count,
			uint64_t addr, uint64_t key,
			int datatype, int op, void *context);
	int	(*writeto)(fid_t fid,
			const void *buf, size_t len,
			const void *dest_addr,
			uint64_t addr, uint64_t key,
			int datatype, int op, void *context);
	int	(*writememto)(fid_t fid,
			const void *buf, size_t len, uint64_t mem_desc,
			const void *dest_addr,
			uint64_t addr, uint64_t key,
			int datatype, int op, void *context);
	int	(*writemsg)(fid_t fid,
			const struct fi_msg_atomic *msg, uint64_t flags);

	int	(*readwrite)(fid_t fid,
			const void *buf, size_t len,
			void *result,
			uint64_t addr, uint64_t key,
			int datatype, int op, void *context);
	int	(*readwritemem)(fid_t fid,
			const void *buf, size_t len, uint64_t mem_desc,
			void *result, uint64_t result_mem_desc,
			uint64_t addr, uint64_t key,
			int datatype, int op, void *context);
	int	(*readwritev)(fid_t fid,
			const void *iov, size_t count,
			void *resultv, size_t result_count,
			uint64_t addr, uint64_t key,
			int datatype, int op, void *context);
	int	(*readwriteto)(fid_t fid,
			const void *buf, size_t len,
			void *result,
			const void *dest_addr,
			uint64_t addr, uint64_t key,
			int datatype, int op, void *context);
	int	(*readwritememto)(fid_t fid,
			const void *buf, size_t len, uint64_t mem_desc,
			void *result, uint64_t result_mem_desc,
			const void *dest_addr,
			uint64_t addr, uint64_t key,
			int datatype, int op, void *context);
	int	(*readwritemsg)(fid_t fid,
			const struct fi_msg_atomic *msg,
			void *resultv, size_t result_count,
			uint64_t flags);

	int	(*compwrite)(fid_t fid,
			const void *buf, size_t len,
			const void *compare,
			void *result,
			uint64_t addr, uint64_t key,
			int datatype, int op, void *context);
	int	(*compwritemem)(fid_t fid,
			const void *buf, size_t len, uint64_t mem_desc,
			const void *compare, uint64_t compare_mem_desc,
			void *result, uint64_t result_mem_desc,
			uint64_t addr, uint64_t key,
			int datatype, int op, void *context);
	int	(*compwritev)(fid_t fid,
			const void *iov, size_t count,
			const void *comparev, size_t compare_count,
			void *resultv, size_t result_count,
			uint64_t addr, uint64_t key,
			int datatype, int op, void *context);
	int	(*compwriteto)(fid_t fid,
			const void *buf, size_t len,
			const void *compare,
			void *result,
			const void *dest_addr,
			uint64_t addr, uint64_t key,
			int datatype, int op, void *context);
	int	(*compwritememto)(fid_t fid,
			const void *buf, size_t len, uint64_t mem_desc,
			const void *compare, uint64_t compare_mem_desc,
			void *result, uint64_t result_mem_desc,
			const void *dest_addr,
			uint64_t addr, uint64_t key,
			int datatype, int op, void *context);
	int	(*compwritemsg)(fid_t fid,
			const struct fi_msg_atomic *msg,
			const void *comparev, size_t compare_count,
			void *resultv, size_t result_count,
			uint64_t flags);

	int	(*writevalid)(fid_t fid, int datatype, int op, size_t *count);
	int	(*readwritevalid)(fid_t fid, int datatype, int op, size_t *count);
	int	(*compwritevalid)(fid_t fid, int datatype, int op, size_t *count);
};


#ifndef FABRIC_DIRECT

static inline int
fi_atomic(fid_t fid, const void *buf, size_t len, uint64_t addr,
	  uint64_t key, int datatype, int op, void *context)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, atomic);
	FI_ASSERT_OP(ep->atomic, struct fi_ops_atomic, write);
	return ep->atomic->write(fid, buf, len, addr, key, datatype, op, context);
}

static inline int
fi_atomicmem(fid_t fid, const void *buf, size_t len, uint64_t mem_desc,
	     uint64_t addr, uint64_t key, int datatype, int op, void *context)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, atomic);
	FI_ASSERT_OP(ep->atomic, struct fi_ops_atomic, writemem);
	return ep->atomic->writemem(fid, buf, len, mem_desc, addr, key,
				    datatype, op, context);
}

static inline int
fi_atomicto(fid_t fid, const void *buf, size_t len, const void *dest_addr,
	    uint64_t addr, uint64_t key, int datatype, int op, void *context)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, atomic);
	FI_ASSERT_OP(ep->atomic, struct fi_ops_atomic, writeto);
	return ep->atomic->writeto(fid, buf, len, dest_addr, addr, key,
				   datatype, op, context);
}

static inline int
fi_atomicmemto(fid_t fid, const void *buf, size_t len, uint64_t mem_desc,
	       const void *dest_addr, uint64_t addr, uint64_t key,
	       int datatype, int op, void *context)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, atomic);
	FI_ASSERT_OP(ep->atomic, struct fi_ops_atomic, writememto);
	return ep->atomic->writememto(fid, buf, len, mem_desc, dest_addr,
				      addr, key, datatype, op, context);
}

static inline int
fi_fetch_atomic(fid_t fid, const void *buf, size_t len, void *result,
		uint64_t addr, uint64_t key, int datatype, int op, void *context)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, atomic);
	FI_ASSERT_OP(ep->atomic, struct fi_ops_atomic, readwrite);
	return ep->atomic->readwrite(fid, buf, len, result, addr, key,
				     datatype, op, context);
}

static inline int
fi_fetch_atomicto(fid_t fid, const void *buf, size_t len, void *result,
		  const void *dest_addr, uint64_t addr, uint64_t key,
		  int datatype, int op, void *context)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, atomic);
	FI_ASSERT_OP(ep->atomic, struct fi_ops_atomic, readwriteto);
	return ep->atomic->readwriteto(fid, buf, len, result, dest_addr,
				       addr, key, datatype, op, context);
}

static inline int
fi_fetch_atomicmemto(fid_t fid, const void *buf, size_t len,
		     uint64_t mem_desc, void *result, uint64_t result_mem_desc,
		     const void *dest_addr, uint64_t addr, uint64_t key,
		     int datatype, int op, void *context)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, atomic);
	FI_ASSERT_OP(ep->atomic, struct fi_ops_atomic, readwritememto);
	return ep->atomic->readwritememto(fid, buf, len, mem_desc, result,
					  result_mem_desc, dest_addr, addr,
					  key, datatype, op, context);
}

static inline int
fi_atomicvalid(fid_t fid, int datatype, int op, size_t *count)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, atomic);
	FI_ASSERT_OP(ep->atomic, struct fi_ops_atomic, writevalid);
	return ep->atomic->writevalid(fid, datatype, op, count);
}

static inline int
fi_fetch_atomicvalid(fid_t fid, int datatype, int op, size_t *count)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, atomic);
	FI_ASSERT_OP(ep->atomic, struct fi_ops_atomic, readwritevalid);
	return ep->atomic->readwritevalid(fid, datatype, op, count);
}

static inline int
fi_compare_atomicvalid(fid_t fid, int datatype, int op, size_t *count)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, atomic);
	FI_ASSERT_OP(ep->atomic, struct fi_ops_atomic, compwritevalid);
	return ep->atomic->compwritevalid(fid, datatype, op, count);
}

#else // FABRIC_DIRECT
#include <rdma/fi_direct_atomic.h>
#endif

#ifdef __cplusplus
}
#endif

#endif /* _FI_ATOMIC_H_ */
