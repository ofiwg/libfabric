/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

#include "psmx2.h"

/* Atomics protocol:
 *
 * Atomics REQ:
 *	args[0].u32w0	cmd, dst_vl
 *	args[0].u32w1	count
 *	args[1].u64	req
 *	args[2].u64	addr
 *	args[3].u64	key
 *	args[4].u32w0	datatype
 *	args[4].u32w1	op
 *
 * Atomics REP:
 *	args[0].u32w0	cmd
 *	args[0].u32w1	error
 *	args[1].u64	req
 */

static fastlock_t psmx2_atomic_lock;

void psmx2_atomic_init(void)
{
	fastlock_init(&psmx2_atomic_lock);
}

void psmx2_atomic_fini(void)
{
	fastlock_destroy(&psmx2_atomic_lock);
}

static inline void psmx2_ioc_read(const struct fi_ioc *ioc, size_t count,
				  int datatype, uint8_t *buf, size_t len)
{
	int i;
	size_t copy_len;

	for (i=0; i<count && len; i++) {
		copy_len = fi_datatype_size(datatype) * ioc[i].count;
		if (copy_len > len)
			copy_len = len;
		memcpy(buf, ioc[i].addr, copy_len);
		buf += copy_len;
		len -= copy_len;
	}
}

static inline void psmx2_ioc_write(struct fi_ioc *ioc, size_t count,
				   int datatype, const uint8_t *buf, size_t len)
{
	int i;
	size_t copy_len;

	for (i=0; i<count && len; i++) {
		copy_len = fi_datatype_size(datatype) * ioc[i].count;
		if (copy_len > len)
			copy_len = len;
		memcpy(ioc[i].addr, buf, copy_len);
		buf += copy_len;
		len -= copy_len;
	}
}

static inline size_t psmx2_ioc_size(const struct fi_ioc *ioc, size_t count,
				    int datatype)
{
	int i;
	size_t len = 0;

	for (i=0; i<count; i++)
		len += fi_datatype_size(datatype) * ioc[i].count;

	return len;
}

#define CASE_INT_TYPE(FUNC,...) \
		case FI_INT8:	FUNC(__VA_ARGS__,int8_t); break; \
		case FI_UINT8:	FUNC(__VA_ARGS__,uint8_t); break; \
		case FI_INT16:	FUNC(__VA_ARGS__,int16_t); break; \
		case FI_UINT16: FUNC(__VA_ARGS__,uint16_t); break; \
		case FI_INT32:	FUNC(__VA_ARGS__,int32_t); break; \
		case FI_UINT32: FUNC(__VA_ARGS__,uint32_t); break; \
		case FI_INT64:	FUNC(__VA_ARGS__,int64_t); break; \
		case FI_UINT64: FUNC(__VA_ARGS__,uint64_t); break;

#define CASE_FP_TYPE(FUNC,...) \
		case FI_FLOAT:	FUNC(__VA_ARGS__,float); break; \
		case FI_DOUBLE:	FUNC(__VA_ARGS__,double); break; \
		case FI_LONG_DOUBLE: FUNC(__VA_ARGS__,long double); break;

#define CASE_COMPLEX_TYPE(FUNC,...) \
		case FI_FLOAT_COMPLEX:	FUNC(__VA_ARGS__,float complex); break; \
		case FI_DOUBLE_COMPLEX:	FUNC(__VA_ARGS__,double complex); break; \
		case FI_LONG_DOUBLE_COMPLEX: FUNC(__VA_ARGS__,long double complex); break;

#define SWITCH_INT_TYPE(type,...) \
		switch (type) { \
		CASE_INT_TYPE(__VA_ARGS__) \
		default: return -FI_EOPNOTSUPP; \
		}

#define SWITCH_ORD_TYPE(type,...) \
		switch (type) { \
		CASE_INT_TYPE(__VA_ARGS__) \
		CASE_FP_TYPE(__VA_ARGS__) \
		default: return -FI_EOPNOTSUPP; \
		}

#define SWITCH_ALL_TYPE(type,...) \
		switch (type) { \
		CASE_INT_TYPE(__VA_ARGS__) \
		CASE_FP_TYPE(__VA_ARGS__) \
		CASE_COMPLEX_TYPE(__VA_ARGS__) \
		default: return -FI_EOPNOTSUPP; \
		}

#define PSMX2_MIN(dst,src)	if ((dst) > (src)) (dst) = (src)
#define PSMX2_MAX(dst,src)	if ((dst) < (src)) (dst) = (src)
#define PSMX2_SUM(dst,src)	(dst) += (src)
#define PSMX2_PROD(dst,src)	(dst) *= (src)
#define PSMX2_LOR(dst,src)	(dst) = (dst) || (src)
#define PSMX2_LAND(dst,src)	(dst) = (dst) && (src)
#define PSMX2_BOR(dst,src)	(dst) |= (src)
#define PSMX2_BAND(dst,src)	(dst) &= (src)
#define PSMX2_LXOR(dst,src)	(dst) = ((dst) && !(src)) || (!(dst) && (src))
#define PSMX2_BXOR(dst,src)	(dst) ^= (src)
#define PSMX2_COPY(dst,src)	(dst) = (src)

#define PSMX2_ATOMIC_READ(dst,res,cnt,TYPE) \
		do { \
			int i; \
			TYPE *d = (dst); \
			TYPE *r = (res); \
			fastlock_acquire(&psmx2_atomic_lock); \
			for (i=0; i<(cnt); i++) \
				r[i] = d[i]; \
			fastlock_release(&psmx2_atomic_lock); \
		} while (0)

#define PSMX2_ATOMIC_WRITE(dst,src,cnt,OP,TYPE) \
		do { \
			int i; \
			TYPE *d = (dst); \
			TYPE *s = (src); \
			fastlock_acquire(&psmx2_atomic_lock); \
			for (i=0; i<cnt; i++) \
				OP(d[i],s[i]); \
			fastlock_release(&psmx2_atomic_lock); \
		} while (0)

#define PSMX2_ATOMIC_READWRITE(dst,src,res,cnt,OP,TYPE) \
		do { \
			int i; \
			TYPE *d = (dst); \
			TYPE *s = (src); \
			TYPE *r = (res); \
			fastlock_acquire(&psmx2_atomic_lock); \
			for (i=0; i<(cnt); i++) {\
				r[i] = d[i]; \
				OP(d[i],s[i]); \
			} \
			fastlock_release(&psmx2_atomic_lock); \
		} while (0)

#define PSMX2_ATOMIC_CSWAP(dst,src,cmp,res,cnt,CMP_OP,TYPE) \
		do { \
			int i; \
			TYPE *d = (dst); \
			TYPE *s = (src); \
			TYPE *c = (cmp); \
			TYPE *r = (res); \
			fastlock_acquire(&psmx2_atomic_lock); \
			for (i=0; i<(cnt); i++) { \
				r[i] = d[i]; \
				if (c[i] CMP_OP d[i]) \
					d[i] = s[i]; \
			} \
			fastlock_release(&psmx2_atomic_lock); \
		} while (0)

#define PSMX2_ATOMIC_MSWAP(dst,src,cmp,res,cnt,TYPE) \
		do { \
			int i; \
			TYPE *d = (dst); \
			TYPE *s = (src); \
			TYPE *c = (cmp); \
			TYPE *r = (res); \
			fastlock_acquire(&psmx2_atomic_lock); \
			for (i=0; i<(cnt); i++) { \
				r[i] = d[i]; \
				d[i] = (s[i] & c[i]) | (d[i] & ~c[i]); \
			} \
			fastlock_release(&psmx2_atomic_lock); \
		} while (0)

static int psmx2_atomic_do_write(void *dest, void *src,
				 int datatype, int op, int count)
{
	switch (op) {
	case FI_MIN:
		SWITCH_ORD_TYPE(datatype,PSMX2_ATOMIC_WRITE,
				dest,src,count,PSMX2_MIN);
		break;

	case FI_MAX:
		SWITCH_ORD_TYPE(datatype,PSMX2_ATOMIC_WRITE,
				dest,src,count,PSMX2_MAX);
		break;

	case FI_SUM:
		SWITCH_ALL_TYPE(datatype,PSMX2_ATOMIC_WRITE,
				dest,src,count,PSMX2_SUM);
		break;

	case FI_PROD:
		SWITCH_ALL_TYPE(datatype,PSMX2_ATOMIC_WRITE,
				dest,src,count,PSMX2_PROD);
		break;

	case FI_LOR:
		SWITCH_INT_TYPE(datatype,PSMX2_ATOMIC_WRITE,
				dest,src,count,PSMX2_LOR);
		break;

	case FI_LAND:
		SWITCH_INT_TYPE(datatype,PSMX2_ATOMIC_WRITE,
				dest,src,count,PSMX2_LAND);
		break;

	case FI_BOR:
		SWITCH_INT_TYPE(datatype,PSMX2_ATOMIC_WRITE,
				dest,src,count,PSMX2_BOR);
		break;

	case FI_BAND:
		SWITCH_INT_TYPE(datatype,PSMX2_ATOMIC_WRITE,
				dest,src,count,PSMX2_BAND);
		break;

	case FI_LXOR:
		SWITCH_INT_TYPE(datatype,PSMX2_ATOMIC_WRITE,
				dest,src,count,PSMX2_LXOR);
		break;

	case FI_BXOR:
		SWITCH_INT_TYPE(datatype,PSMX2_ATOMIC_WRITE,
				dest,src,count,PSMX2_BXOR);
		break;

	case FI_ATOMIC_WRITE:
		SWITCH_ALL_TYPE(datatype,PSMX2_ATOMIC_WRITE,
				dest,src,count,PSMX2_COPY);
		break;

	default:
		return -FI_EOPNOTSUPP;
	}

	return 0;
}

static int psmx2_atomic_do_readwrite(void *dest, void *src, void *result,
				     int datatype, int op, int count)
{
	switch (op) {
	case FI_MIN:
		SWITCH_ORD_TYPE(datatype,PSMX2_ATOMIC_READWRITE,
				dest,src,result,count,PSMX2_MIN);
		break;

	case FI_MAX:
		SWITCH_ORD_TYPE(datatype,PSMX2_ATOMIC_READWRITE,
				dest,src,result,count,PSMX2_MAX);
		break;

	case FI_SUM:
		SWITCH_ALL_TYPE(datatype,PSMX2_ATOMIC_READWRITE,
				dest,src,result,count,PSMX2_SUM);
		break;

	case FI_PROD:
		SWITCH_ALL_TYPE(datatype,PSMX2_ATOMIC_READWRITE,
				dest,src,result,count,PSMX2_PROD);
		break;

	case FI_LOR:
		SWITCH_INT_TYPE(datatype,PSMX2_ATOMIC_READWRITE,
				dest,src,result,count,PSMX2_LOR);
		break;

	case FI_LAND:
		SWITCH_INT_TYPE(datatype,PSMX2_ATOMIC_READWRITE,
				dest,src,result,count,PSMX2_LAND);
		break;

	case FI_BOR:
		SWITCH_INT_TYPE(datatype,PSMX2_ATOMIC_READWRITE,
				dest,src,result,count,PSMX2_BOR);
		break;

	case FI_BAND:
		SWITCH_INT_TYPE(datatype,PSMX2_ATOMIC_READWRITE,
				dest,src,result,count,PSMX2_BAND);
		break;

	case FI_LXOR:
		SWITCH_INT_TYPE(datatype,PSMX2_ATOMIC_READWRITE,
				dest,src,result,count,PSMX2_LXOR);
		break;

	case FI_BXOR:
		SWITCH_INT_TYPE(datatype,PSMX2_ATOMIC_READWRITE,
				dest,src,result,count,PSMX2_BXOR);
		break;

	case FI_ATOMIC_READ:
		SWITCH_ALL_TYPE(datatype,PSMX2_ATOMIC_READ,
				dest,result,count);
		break;

	case FI_ATOMIC_WRITE:
		SWITCH_ALL_TYPE(datatype,PSMX2_ATOMIC_READWRITE,
				dest,src,result,count,PSMX2_COPY);
		break;

	default:
		return -FI_EOPNOTSUPP;
	}

	return 0;
}

static int psmx2_atomic_do_compwrite(void *dest, void *src, void *compare,
				     void *result, int datatype, int op,
				     int count)
{
	switch (op) {
	case FI_CSWAP:
		SWITCH_ALL_TYPE(datatype,PSMX2_ATOMIC_CSWAP,
				dest,src,compare,result,count,==);
		break;

	case FI_CSWAP_NE:
		SWITCH_ALL_TYPE(datatype,PSMX2_ATOMIC_CSWAP,
				dest,src,compare,result,count,!=);
		break;

	case FI_CSWAP_LE:
		SWITCH_ORD_TYPE(datatype,PSMX2_ATOMIC_CSWAP,
				dest,src,compare,result,count,<=);
		break;

	case FI_CSWAP_LT:
		SWITCH_ORD_TYPE(datatype,PSMX2_ATOMIC_CSWAP,
				dest,src,compare,result,count,<);
		break;

	case FI_CSWAP_GE:
		SWITCH_ORD_TYPE(datatype,PSMX2_ATOMIC_CSWAP,
				dest,src,compare,result,count,>=);
		break;

	case FI_CSWAP_GT:
		SWITCH_ORD_TYPE(datatype,PSMX2_ATOMIC_CSWAP,
				dest,src,compare,result,count,>);
		break;

	case FI_MSWAP:
		SWITCH_INT_TYPE(datatype,PSMX2_ATOMIC_MSWAP,
				dest,src,compare,result,count);
		break;

	default:
		return -FI_EOPNOTSUPP;
	}

	return 0;
}

static void psmx2_am_atomic_completion(void *buf)
{
	if (buf)
		free(buf);
}

int psmx2_am_atomic_handler(psm2_am_token_t token,
			    psm2_amarg_t *args, int nargs, void *src,
			    uint32_t len)
{
	psm2_amarg_t rep_args[8];
	int count;
	uint8_t *addr;
	uint64_t key;
	int datatype, op;
	int err = 0;
	int op_error = 0;
	struct psmx2_am_request *req;
	struct psmx2_cq_event *event;
	struct psmx2_fid_mr *mr;
	struct psmx2_fid_domain *domain;
	struct psmx2_fid_ep *target_ep;
	struct psmx2_fid_cntr *cntr = NULL;
	struct psmx2_fid_cntr *mr_cntr = NULL;
	void *tmp_buf;
	psm2_epaddr_t epaddr;
	uint8_t vlane;
	int cmd;

	psm2_am_get_source(token, &epaddr);

	cmd = PSMX2_AM_GET_OP(args[0].u32w0);
	vlane = PSMX2_AM_GET_DST(args[0].u32w0);

	domain = psmx2_active_fabric->active_domain;
	target_ep = domain->eps[vlane];

	switch (cmd) {
	case PSMX2_AM_REQ_ATOMIC_WRITE:
		count = args[0].u32w1;
		addr = (uint8_t *)(uintptr_t)args[2].u64;
		key = args[3].u64;
		datatype = args[4].u32w0;
		op = args[4].u32w1;
		assert(len == fi_datatype_size(datatype) * count);

		mr = psmx2_mr_get(domain, key);
		op_error = mr ?
			psmx2_mr_validate(mr, (uint64_t)addr, len, FI_REMOTE_WRITE) :
			-FI_EINVAL;

		if (!op_error) {
			addr += mr->offset;
			psmx2_atomic_do_write(addr, src, datatype, op, count);

			cntr = target_ep->remote_write_cntr;
			mr_cntr = mr->cntr;

			if (cntr)
				psmx2_cntr_inc(cntr);

			if (mr_cntr && mr_cntr != cntr)
				psmx2_cntr_inc(mr_cntr);
		}

		rep_args[0].u32w0 = PSMX2_AM_REP_ATOMIC_WRITE;
		rep_args[0].u32w1 = op_error;
		rep_args[1].u64 = args[1].u64;
		err = psm2_am_reply_short(token, PSMX2_AM_ATOMIC_HANDLER,
					  rep_args, 2, NULL, 0, 0,
					  NULL, NULL );
		break;

	case PSMX2_AM_REQ_ATOMIC_READWRITE:
		count = args[0].u32w1;
		addr = (uint8_t *)(uintptr_t)args[2].u64;
		key = args[3].u64;
		datatype = args[4].u32w0;
		op = args[4].u32w1;

		if (op == FI_ATOMIC_READ)
			len = fi_datatype_size(datatype) * count;

		assert(len == fi_datatype_size(datatype) * count);

		mr = psmx2_mr_get(domain, key);
		op_error = mr ?
			psmx2_mr_validate(mr, (uint64_t)addr, len,
					  FI_REMOTE_READ|FI_REMOTE_WRITE) :
			-FI_EINVAL;

		if (!op_error) {
			addr += mr->offset;
			tmp_buf = malloc(len);
			if (tmp_buf)
				psmx2_atomic_do_readwrite(addr, src, tmp_buf,
							  datatype, op, count);
			else
				op_error = -FI_ENOMEM;

			if (op == FI_ATOMIC_READ) {
				cntr = target_ep->remote_read_cntr;
			} else {
				cntr = target_ep->remote_write_cntr;
				mr_cntr = mr->cntr;
			}

			if (cntr)
				psmx2_cntr_inc(cntr);

			if (mr_cntr && mr_cntr != cntr)
				psmx2_cntr_inc(mr_cntr);
		} else {
			tmp_buf = NULL;
		}

		rep_args[0].u32w0 = PSMX2_AM_REP_ATOMIC_READWRITE;
		rep_args[0].u32w1 = op_error;
		rep_args[1].u64 = args[1].u64;
		err = psm2_am_reply_short(token, PSMX2_AM_ATOMIC_HANDLER,
					  rep_args, 2, tmp_buf, (tmp_buf?len:0), 0,
					  psmx2_am_atomic_completion, tmp_buf );
		break;

	case PSMX2_AM_REQ_ATOMIC_COMPWRITE:
		count = args[0].u32w1;
		addr = (uint8_t *)(uintptr_t)args[2].u64;
		key = args[3].u64;
		datatype = args[4].u32w0;
		op = args[4].u32w1;
		len /= 2;
		assert(len == fi_datatype_size(datatype) * count);

		mr = psmx2_mr_get(domain, key);
		op_error = mr ?
			psmx2_mr_validate(mr, (uint64_t)addr, len,
					  FI_REMOTE_READ|FI_REMOTE_WRITE) :
			-FI_EINVAL;

		if (!op_error) {
			addr += mr->offset;
			tmp_buf = malloc(len);
			if (tmp_buf)
				psmx2_atomic_do_compwrite(addr, src, (uint8_t *)src + len,
							  tmp_buf, datatype,
							  op, count);
			else
				op_error = -FI_ENOMEM;

			cntr = target_ep->remote_write_cntr;
			mr_cntr = mr->cntr;

			if (cntr)
				psmx2_cntr_inc(cntr);

			if (mr_cntr && mr_cntr != cntr)
				psmx2_cntr_inc(mr_cntr);
		} else {
			tmp_buf = NULL;
		}

		rep_args[0].u32w0 = PSMX2_AM_REP_ATOMIC_READWRITE;
		rep_args[0].u32w1 = op_error;
		rep_args[1].u64 = args[1].u64;
		err = psm2_am_reply_short(token, PSMX2_AM_ATOMIC_HANDLER,
					  rep_args, 2, tmp_buf,
					  (tmp_buf?len:0), 0,
					  psmx2_am_atomic_completion,
					  tmp_buf );
		break;

	case PSMX2_AM_REP_ATOMIC_WRITE:
		req = (struct psmx2_am_request *)(uintptr_t)args[1].u64;
		op_error = (int)args[0].u32w1;
		assert(req->op == PSMX2_AM_REQ_ATOMIC_WRITE);
		if (req->ep->send_cq && !req->no_event) {
			event = psmx2_cq_create_event(
					req->ep->send_cq,
					req->atomic.context,
					req->atomic.buf,
					req->cq_flags,
					req->atomic.len,
					0, /* data */
					0, /* tag */
					0, /* olen */
					op_error);
			if (event)
				psmx2_cq_enqueue_event(req->ep->send_cq, event);
			else
				err = -FI_ENOMEM;
		}

		if (req->ep->write_cntr)
			psmx2_cntr_inc(req->ep->write_cntr);

		free(req);
		break;

	case PSMX2_AM_REP_ATOMIC_READWRITE:
	case PSMX2_AM_REP_ATOMIC_COMPWRITE:
		req = (struct psmx2_am_request *)(uintptr_t)args[1].u64;
		op_error = (int)args[0].u32w1;
		assert(op_error || req->atomic.len == len);

		if (!op_error) {
			if (req->atomic.result)
				memcpy(req->atomic.result, src, len);
			else
				psmx2_ioc_write(req->ioc, req->atomic.iov_count,
						req->atomic.datatype, src, len);
		}

		if (req->ep->send_cq && !req->no_event) {
			event = psmx2_cq_create_event(
					req->ep->send_cq,
					req->atomic.context,
					req->atomic.buf,
					req->cq_flags,
					req->atomic.len,
					0, /* data */
					0, /* tag */
					0, /* olen */
					op_error);
			if (event)
				psmx2_cq_enqueue_event(req->ep->send_cq, event);
			else
				err = -FI_ENOMEM;
		}

		if (req->ep->read_cntr)
			psmx2_cntr_inc(req->ep->read_cntr);

		free(req);
		break;

	default:
		err = -FI_EINVAL;
	}
	return err;
}

static int psmx2_atomic_self(int am_cmd,
			     struct psmx2_fid_ep *ep,
			     struct psmx2_fid_ep *target_ep,
			     const void *buf,
			     size_t count, void *desc,
			     const void *compare, void *compare_desc,
			     void *result, void *result_desc,
			     uint64_t addr, uint64_t key,
			     enum fi_datatype datatype,
			     enum fi_op op, void *context,
			     uint64_t flags)
{
	struct psmx2_fid_mr *mr;
	struct psmx2_cq_event *event;
	struct psmx2_fid_cntr *cntr = NULL;
	struct psmx2_fid_cntr *mr_cntr = NULL;
	void *tmp_buf;
	size_t len;
	int no_event;
	int err = 0;
	int op_error;
	int access;
	uint64_t cq_flags = 0;

	if (am_cmd == PSMX2_AM_REQ_ATOMIC_WRITE)
		access = FI_REMOTE_WRITE;
	else
		access = FI_REMOTE_READ | FI_REMOTE_WRITE;

	len = fi_datatype_size(datatype) * count;
	mr = psmx2_mr_get(psmx2_active_fabric->active_domain, key);
	op_error = mr ?  psmx2_mr_validate(mr, addr, len, access) : -FI_EINVAL;

	if (op_error)
		goto gen_local_event;

	addr += mr->offset;

	switch (am_cmd) {
	case PSMX2_AM_REQ_ATOMIC_WRITE:
		err = psmx2_atomic_do_write((void *)addr, (void *)buf,
					    (int)datatype, (int)op, (int)count);
		cq_flags = FI_WRITE | FI_ATOMIC;
		break;

	case PSMX2_AM_REQ_ATOMIC_READWRITE:
		if (result != buf) {
			err = psmx2_atomic_do_readwrite((void *)addr, (void *)buf,
							(void *)result, (int)datatype,
							(int)op, (int)count);
		} else {
			tmp_buf = malloc(len);
			if (tmp_buf) {
				memcpy(tmp_buf, result, len);
				err = psmx2_atomic_do_readwrite((void *)addr, (void *)buf,
								tmp_buf, (int)datatype,
								(int)op, (int)count);
				memcpy(result, tmp_buf, len);
				free(tmp_buf);
			} else {
				err = -FI_ENOMEM;
			}
			
		}
		if (op == FI_ATOMIC_READ)
			cq_flags = FI_READ | FI_ATOMIC;
		else
			cq_flags = FI_WRITE | FI_ATOMIC;
		break;

	case PSMX2_AM_REQ_ATOMIC_COMPWRITE:
		if (result != buf && result != compare) {
			err = psmx2_atomic_do_compwrite((void *)addr, (void *)buf,
							(void *)compare, (void *)result,
							(int)datatype, (int)op, (int)count);
		} else {
			tmp_buf = malloc(len);
			if (tmp_buf) {
				memcpy(tmp_buf, result, len);
				err = psmx2_atomic_do_compwrite((void *)addr, (void *)buf,
								(void *)compare, tmp_buf,
								(int)datatype, (int)op, (int)count);
				memcpy(result, tmp_buf, len);
				free(tmp_buf);
			} else {
				err = -FI_ENOMEM;
			}
		}
		cq_flags = FI_WRITE | FI_ATOMIC;
		break;
	}

	if (op == FI_ATOMIC_READ) {
		cntr = target_ep->remote_read_cntr;
	} else {
		cntr = target_ep->remote_write_cntr;
		mr_cntr = mr->cntr;
	}

	if (cntr)
		psmx2_cntr_inc(cntr);

	if (mr_cntr && mr_cntr != cntr)
		psmx2_cntr_inc(mr_cntr);

gen_local_event:
	no_event = ((flags & PSMX2_NO_COMPLETION) ||
		    (ep->send_selective_completion && !(flags & FI_COMPLETION)));
	if (ep->send_cq && !no_event) {
		event = psmx2_cq_create_event(
				ep->send_cq,
				context,
				(void *)buf,
				cq_flags,
				len,
				0, /* data */
				0, /* tag */
				0, /* olen */
				op_error);
		if (event)
			psmx2_cq_enqueue_event(ep->send_cq, event);
		else
			err = -FI_ENOMEM;
	}

	switch (am_cmd) {
	case PSMX2_AM_REQ_ATOMIC_WRITE:
		if (ep->write_cntr)
			psmx2_cntr_inc(ep->write_cntr);
		break;
	case PSMX2_AM_REQ_ATOMIC_READWRITE:
	case PSMX2_AM_REQ_ATOMIC_COMPWRITE:
		if (ep->read_cntr)
			psmx2_cntr_inc(ep->read_cntr);
		break;
	}

	return err;
}

ssize_t psmx2_atomic_write_generic(struct fid_ep *ep,
				   const void *buf,
				   size_t count, void *desc,
				   fi_addr_t dest_addr,
				   uint64_t addr, uint64_t key,
				   enum fi_datatype datatype,
				   enum fi_op op, void *context,
				   uint64_t flags)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	struct psmx2_epaddr_context *epaddr_context;
	struct psmx2_am_request *req;
	psm2_amarg_t args[8];
	psm2_epaddr_t psm2_epaddr;
	uint8_t vlane;
	int am_flags = PSM2_AM_FLAG_ASYNC;
	int chunk_size, len;
	size_t idx;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (flags & FI_TRIGGER) {
		struct psmx2_trigger *trigger;
		struct fi_triggered_context *ctxt = context;

		trigger = calloc(1, sizeof(*trigger));
		if (!trigger)
			return -FI_ENOMEM;

		trigger->op = PSMX2_TRIGGERED_ATOMIC_WRITE;
		trigger->cntr = container_of(ctxt->trigger.threshold.cntr,
					     struct psmx2_fid_cntr, cntr);
		trigger->threshold = ctxt->trigger.threshold.threshold;
		trigger->atomic_write.ep = ep;
		trigger->atomic_write.buf = buf;
		trigger->atomic_write.count = count;
		trigger->atomic_write.desc = desc;
		trigger->atomic_write.dest_addr = dest_addr;
		trigger->atomic_write.addr = addr;
		trigger->atomic_write.key = key;
		trigger->atomic_write.datatype = datatype;
		trigger->atomic_write.atomic_op = op;
		trigger->atomic_write.context = context;
		trigger->atomic_write.flags = flags & ~FI_TRIGGER;

		psmx2_cntr_add_trigger(trigger->cntr, trigger);
		return 0;
	}

	if (!buf)
		return -FI_EINVAL;

	if (datatype < 0 || datatype >= FI_DATATYPE_LAST)
		return -FI_EINVAL;

	if (op < 0 || op >= FI_ATOMIC_OP_LAST)
		return -FI_EINVAL;

	av = ep_priv->av;
	if (av && av->type == FI_AV_TABLE) {
		idx = dest_addr;
		if (idx >= av->last)
			return -FI_EINVAL;

		psm2_epaddr = av->epaddrs[idx];
		vlane = av->vlanes[idx];
	} else {
		 if (!dest_addr)
			return -FI_EINVAL;

		psm2_epaddr = PSMX2_ADDR_TO_EP(dest_addr);
		vlane = PSMX2_ADDR_TO_VL(dest_addr);
	}

	epaddr_context = psm2_epaddr_getctxt((void *)psm2_epaddr);
	if (epaddr_context->epid == ep_priv->domain->psm2_epid)
		return psmx2_atomic_self(PSMX2_AM_REQ_ATOMIC_WRITE, ep_priv,
					 ep_priv->domain->eps[vlane],
					 buf, count, desc, NULL, NULL, NULL,
					 NULL, addr, key, datatype, op,
					 context, flags);

	chunk_size = psmx2_am_param.max_request_short;
	len = fi_datatype_size(datatype)* count;
	if (len > chunk_size)
		return -FI_EMSGSIZE;

	if (flags & FI_INJECT) {
		req = malloc(sizeof(*req) + len);
		if (!req)
			return -FI_ENOMEM;
		memset(req, 0, sizeof(*req));
		memcpy((uint8_t *)req+sizeof(*req), (void *)buf, len);
		buf = (uint8_t *)req + sizeof(*req);
	} else {
		req = calloc(1, sizeof(*req));
		if (!req)
			return -FI_ENOMEM;
	}

	req->no_event = (flags & PSMX2_NO_COMPLETION) ||
			(ep_priv->send_selective_completion && !(flags & FI_COMPLETION));

	req->op = PSMX2_AM_REQ_ATOMIC_WRITE;
	req->atomic.buf = (void *)buf;
	req->atomic.len = len;
	req->atomic.addr = addr;
	req->atomic.key = key;
	req->atomic.context = context;
	req->atomic.datatype = datatype;
	req->ep = ep_priv;
	req->cq_flags = FI_WRITE | FI_ATOMIC;

	args[0].u32w0 = PSMX2_AM_REQ_ATOMIC_WRITE;
	PSMX2_AM_SET_DST(args[0].u32w0, vlane);
	args[0].u32w1 = count;
	args[1].u64 = (uint64_t)(uintptr_t)req;
	args[2].u64 = addr;
	args[3].u64 = key;
	args[4].u32w0 = datatype;
	args[4].u32w1 = op;
	psm2_am_request_short(psm2_epaddr,
			      PSMX2_AM_ATOMIC_HANDLER, args, 5,
			      (void *)buf, len, am_flags, NULL, NULL);

	return 0;
}

ssize_t psmx2_atomic_writev_generic(struct fid_ep *ep,
				   const struct fi_ioc *iov,
				   void **desc, size_t count,
				   fi_addr_t dest_addr,
				   uint64_t addr, uint64_t key,
				   enum fi_datatype datatype,
				   enum fi_op op, void *context,
				   uint64_t flags)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	struct psmx2_epaddr_context *epaddr_context;
	struct psmx2_am_request *req;
	psm2_amarg_t args[8];
	psm2_epaddr_t psm2_epaddr;
	uint8_t vlane;
	int am_flags = PSM2_AM_FLAG_ASYNC;
	int chunk_size;
	size_t idx;
	size_t len;
	uint8_t *buf;
	int err;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (flags & FI_TRIGGER) {
		struct psmx2_trigger *trigger;
		struct fi_triggered_context *ctxt = context;

		trigger = calloc(1, sizeof(*trigger));
		if (!trigger)
			return -FI_ENOMEM;

		trigger->op = PSMX2_TRIGGERED_ATOMIC_WRITEV;
		trigger->cntr = container_of(ctxt->trigger.threshold.cntr,
					     struct psmx2_fid_cntr, cntr);
		trigger->threshold = ctxt->trigger.threshold.threshold;
		trigger->atomic_writev.ep = ep;
		trigger->atomic_writev.iov = iov;
		trigger->atomic_writev.count = count;
		trigger->atomic_writev.desc = desc;
		trigger->atomic_writev.dest_addr = dest_addr;
		trigger->atomic_writev.addr = addr;
		trigger->atomic_writev.key = key;
		trigger->atomic_writev.datatype = datatype;
		trigger->atomic_writev.atomic_op = op;
		trigger->atomic_writev.context = context;
		trigger->atomic_writev.flags = flags & ~FI_TRIGGER;

		psmx2_cntr_add_trigger(trigger->cntr, trigger);
		return 0;
	}

	if (!iov || !count)
		return -FI_EINVAL;

	while (count && !iov[count-1].count)
		count--;

	if (datatype < 0 || datatype >= FI_DATATYPE_LAST)
		return -FI_EINVAL;

	if (op < 0 || op >= FI_ATOMIC_OP_LAST)
		return -FI_EINVAL;

	av = ep_priv->av;
	if (av && av->type == FI_AV_TABLE) {
		idx = dest_addr;
		if (idx >= av->last)
			return -FI_EINVAL;

		psm2_epaddr = av->epaddrs[idx];
		vlane = av->vlanes[idx];
	} else {
		 if (!dest_addr)
			return -FI_EINVAL;

		psm2_epaddr = PSMX2_ADDR_TO_EP(dest_addr);
		vlane = PSMX2_ADDR_TO_VL(dest_addr);
	}

	len = psmx2_ioc_size(iov, count, datatype);

	epaddr_context = psm2_epaddr_getctxt((void *)psm2_epaddr);
	if (epaddr_context->epid == ep_priv->domain->psm2_epid) {
		buf = malloc(len);
		if (!buf)
			return -FI_ENOMEM;

		psmx2_ioc_read(iov, count, datatype, buf, len);

		err = psmx2_atomic_self(PSMX2_AM_REQ_ATOMIC_WRITE, ep_priv,
					ep_priv->domain->eps[vlane],
					buf, count, NULL, NULL, NULL, NULL,
					NULL, addr, key, datatype, op,
					context, flags);

		free(buf);
		return err;
	}

	chunk_size = psmx2_am_param.max_request_short;
	if (len > chunk_size)
		return -FI_EMSGSIZE;

	if (count > 1) {
		req = malloc(sizeof(*req) + len);
		if (!req)
			return -FI_ENOMEM;

		buf = (uint8_t *)req + sizeof(*req);
		memset(req, 0, sizeof(*req));
		psmx2_ioc_read(iov, count, datatype, buf, len);
	} else {
		req = calloc(1, sizeof(*req));
		if (!req)
			return -FI_ENOMEM;
		buf = iov[0].addr;
	}

	req->no_event = (flags & PSMX2_NO_COMPLETION) ||
			(ep_priv->send_selective_completion && !(flags & FI_COMPLETION));

	req->op = PSMX2_AM_REQ_ATOMIC_WRITE;
	req->atomic.buf = (void *)buf;
	req->atomic.len = len;
	req->atomic.addr = addr;
	req->atomic.key = key;
	req->atomic.context = context;
	req->atomic.datatype = datatype;
	req->ep = ep_priv;
	req->cq_flags = FI_WRITE | FI_ATOMIC;

	args[0].u32w0 = PSMX2_AM_REQ_ATOMIC_WRITE;
	PSMX2_AM_SET_DST(args[0].u32w0, vlane);
	args[0].u32w1 = len / fi_datatype_size(datatype);
	args[1].u64 = (uint64_t)(uintptr_t)req;
	args[2].u64 = addr;
	args[3].u64 = key;
	args[4].u32w0 = datatype;
	args[4].u32w1 = op;
	psm2_am_request_short(psm2_epaddr,
			      PSMX2_AM_ATOMIC_HANDLER, args, 5,
			      (void *)buf, len, am_flags, NULL, NULL);

	return 0;
}

static ssize_t psmx2_atomic_write(struct fid_ep *ep,
				  const void *buf,
				  size_t count, void *desc,
				  fi_addr_t dest_addr,
				  uint64_t addr, uint64_t key,
				  enum fi_datatype datatype,
				  enum fi_op op, void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);
	return psmx2_atomic_write_generic(ep, buf, count, desc, dest_addr,
					  addr, key, datatype, op, context,
					  ep_priv->tx_flags);
}

static ssize_t psmx2_atomic_writemsg(struct fid_ep *ep,
				const struct fi_msg_atomic *msg,
				uint64_t flags)
{
	if (!msg || !msg->iov_count || !msg->msg_iov || !msg->rma_iov ||
	    msg->rma_iov_count != 1)
		return -FI_EINVAL;

	if (msg->iov_count > 1)
		return psmx2_atomic_writev_generic(ep, msg->msg_iov, msg->desc,
						   msg->iov_count, msg->addr,
						   msg->rma_iov[0].addr,
						   msg->rma_iov[0].key,
						   msg->datatype, msg->op,
						   msg->context, flags);

	return psmx2_atomic_write_generic(ep, msg->msg_iov[0].addr,
					  msg->msg_iov[0].count,
					  msg->desc ? msg->desc[0] : NULL,
					  msg->addr, msg->rma_iov[0].addr,
					  msg->rma_iov[0].key, msg->datatype,
					  msg->op, msg->context, flags);
}

static ssize_t psmx2_atomic_writev(struct fid_ep *ep,
				   const struct fi_ioc *iov,
				   void **desc, size_t count,
				   fi_addr_t dest_addr,
				   uint64_t addr, uint64_t key,
				   enum fi_datatype datatype,
				   enum fi_op op, void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (!iov || !count)
		return -FI_EINVAL;

	if (count > 1)
		return psmx2_atomic_writev_generic(ep, iov, desc, count,
						   dest_addr, addr, key,
						   datatype, op, context,
						   ep_priv->tx_flags);

	return psmx2_atomic_write_generic(ep, iov->addr, iov->count,
					  desc ? desc[0] : NULL, dest_addr,
					  addr, key, datatype, op, context,
					  ep_priv->tx_flags);
}

static ssize_t psmx2_atomic_inject(struct fid_ep *ep,
				   const void *buf,
				   size_t count, /*void *desc,*/
				   fi_addr_t dest_addr,
				   uint64_t addr, uint64_t key,
				   enum fi_datatype datatype,
				   enum fi_op op)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);
	return psmx2_atomic_write_generic(ep, buf, count, NULL/*desc*/,
					  dest_addr, addr, key,
					  datatype, op, NULL,
					  ep_priv->tx_flags | FI_INJECT | PSMX2_NO_COMPLETION);
}

ssize_t psmx2_atomic_readwrite_generic(struct fid_ep *ep,
				       const void *buf,
				       size_t count, void *desc,
				       void *result, void *result_desc,
				       fi_addr_t dest_addr,
				       uint64_t addr, uint64_t key,
				       enum fi_datatype datatype,
				       enum fi_op op, void *context,
				       uint64_t flags)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	struct psmx2_epaddr_context *epaddr_context;
	struct psmx2_am_request *req;
	psm2_amarg_t args[8];
	psm2_epaddr_t psm2_epaddr;
	uint8_t vlane;
	int am_flags = PSM2_AM_FLAG_ASYNC;
	int chunk_size, len;
	size_t idx;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (flags & FI_TRIGGER) {
		struct psmx2_trigger *trigger;
		struct fi_triggered_context *ctxt = context;

		trigger = calloc(1, sizeof(*trigger));
		if (!trigger)
			return -FI_ENOMEM;

		trigger->op = PSMX2_TRIGGERED_ATOMIC_READWRITE;
		trigger->cntr = container_of(ctxt->trigger.threshold.cntr,
					     struct psmx2_fid_cntr, cntr);
		trigger->threshold = ctxt->trigger.threshold.threshold;
		trigger->atomic_readwrite.ep = ep;
		trigger->atomic_readwrite.buf = buf;
		trigger->atomic_readwrite.count = count;
		trigger->atomic_readwrite.desc = desc;
		trigger->atomic_readwrite.result = result;
		trigger->atomic_readwrite.result_desc = result_desc;
		trigger->atomic_readwrite.dest_addr = dest_addr;
		trigger->atomic_readwrite.addr = addr;
		trigger->atomic_readwrite.key = key;
		trigger->atomic_readwrite.datatype = datatype;
		trigger->atomic_readwrite.atomic_op = op;
		trigger->atomic_readwrite.context = context;
		trigger->atomic_readwrite.flags = flags & ~FI_TRIGGER;

		psmx2_cntr_add_trigger(trigger->cntr, trigger);
		return 0;
	}

	if (!buf && op != FI_ATOMIC_READ)
		return -FI_EINVAL;

	if (datatype < 0 || datatype >= FI_DATATYPE_LAST)
		return -FI_EINVAL;

	if (op < 0 || op >= FI_ATOMIC_OP_LAST)
		return -FI_EINVAL;

	av = ep_priv->av;
	if (av && av->type == FI_AV_TABLE) {
		idx = dest_addr;
		if (idx >= av->last)
			return -FI_EINVAL;

		psm2_epaddr = av->epaddrs[idx];
		vlane = av->vlanes[idx];
	} else {
		if (!dest_addr)
			return -FI_EINVAL;

		psm2_epaddr = PSMX2_ADDR_TO_EP(dest_addr);
		vlane = PSMX2_ADDR_TO_VL(dest_addr);
	}

	epaddr_context = psm2_epaddr_getctxt((void *)psm2_epaddr);
	if (epaddr_context->epid == ep_priv->domain->psm2_epid)
		return psmx2_atomic_self(PSMX2_AM_REQ_ATOMIC_READWRITE,
					 ep_priv, ep_priv->domain->eps[vlane],
					 buf, count, desc, NULL, NULL, result,
					 result_desc, addr, key, datatype, op,
					 context, flags);

	chunk_size = psmx2_am_param.max_request_short;
	len = fi_datatype_size(datatype) * count;
	if (len > chunk_size)
		return -FI_EMSGSIZE;

	if ((flags & FI_INJECT) && op != FI_ATOMIC_READ) {
		req = malloc(sizeof(*req) + len);
		if (!req)
			return -FI_ENOMEM;
		memset(req, 0, sizeof(*req));
		memcpy((uint8_t *)req+sizeof(*req), (void *)buf, len);
		buf = (uint8_t *)req + sizeof(*req);
	} else {
		req = calloc(1, sizeof(*req));
		if (!req)
			return -FI_ENOMEM;
	}

	req->no_event = (flags & PSMX2_NO_COMPLETION) ||
			(ep_priv->send_selective_completion && !(flags & FI_COMPLETION));

	req->op = PSMX2_AM_REQ_ATOMIC_READWRITE;
	req->atomic.buf = (void *)buf;
	req->atomic.len = len;
	req->atomic.addr = addr;
	req->atomic.key = key;
	req->atomic.context = context;
	req->atomic.result = result;
	req->atomic.datatype = datatype;
	req->ep = ep_priv;
	if (op == FI_ATOMIC_READ)
		req->cq_flags = FI_READ | FI_ATOMIC;
	else
		req->cq_flags = FI_WRITE | FI_ATOMIC;

	args[0].u32w0 = PSMX2_AM_REQ_ATOMIC_READWRITE;
	PSMX2_AM_SET_DST(args[0].u32w0, vlane);
	args[0].u32w1 = count;
	args[1].u64 = (uint64_t)(uintptr_t)req;
	args[2].u64 = addr;
	args[3].u64 = key;
	args[4].u32w0 = datatype;
	args[4].u32w1 = op;
	psm2_am_request_short(psm2_epaddr,
			      PSMX2_AM_ATOMIC_HANDLER, args, 5,
			      (void *)buf, (buf?len:0), am_flags, NULL, NULL);

	return 0;
}

ssize_t psmx2_atomic_readwritev_generic(struct fid_ep *ep,
					const struct fi_ioc *iov,
					void **desc, size_t count,
					struct fi_ioc *resultv,
					void **result_desc,
					size_t result_count,
					fi_addr_t dest_addr,
					uint64_t addr, uint64_t key,
					enum fi_datatype datatype,
					enum fi_op op, void *context,
					uint64_t flags)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	struct psmx2_epaddr_context *epaddr_context;
	struct psmx2_am_request *req;
	psm2_amarg_t args[8];
	psm2_epaddr_t psm2_epaddr;
	uint8_t vlane;
	int am_flags = PSM2_AM_FLAG_ASYNC;
	int chunk_size;
	size_t idx;
	size_t len, result_len, iov_size;
	uint8_t *buf, *result;
	void *desc0, *result_desc0;
	int err;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (flags & FI_TRIGGER) {
		struct psmx2_trigger *trigger;
		struct fi_triggered_context *ctxt = context;

		trigger = calloc(1, sizeof(*trigger));
		if (!trigger)
			return -FI_ENOMEM;

		trigger->op = PSMX2_TRIGGERED_ATOMIC_READWRITEV;
		trigger->cntr = container_of(ctxt->trigger.threshold.cntr,
					     struct psmx2_fid_cntr, cntr);
		trigger->threshold = ctxt->trigger.threshold.threshold;
		trigger->atomic_readwritev.ep = ep;
		trigger->atomic_readwritev.iov = iov;
		trigger->atomic_readwritev.count = count;
		trigger->atomic_readwritev.desc = desc;
		trigger->atomic_readwritev.resultv = resultv;
		trigger->atomic_readwritev.result_desc = result_desc;
		trigger->atomic_readwritev.result_count = result_count;
		trigger->atomic_readwritev.dest_addr = dest_addr;
		trigger->atomic_readwritev.addr = addr;
		trigger->atomic_readwritev.key = key;
		trigger->atomic_readwritev.datatype = datatype;
		trigger->atomic_readwritev.atomic_op = op;
		trigger->atomic_readwritev.context = context;
		trigger->atomic_readwritev.flags = flags & ~FI_TRIGGER;

		psmx2_cntr_add_trigger(trigger->cntr, trigger);
		return 0;
	}

	if (((!iov || !count) && op != FI_ATOMIC_READ) || !resultv ||
	    !result_count)
		return -FI_EINVAL;

	while (count && !iov[count-1].count)
		count--;

	while (result_count && !resultv[result_count-1].count)
		result_count--;

	if (datatype < 0 || datatype >= FI_DATATYPE_LAST)
		return -FI_EINVAL;

	if (op < 0 || op >= FI_ATOMIC_OP_LAST)
		return -FI_EINVAL;

	result_len = psmx2_ioc_size(resultv, result_count, datatype);

	if (op != FI_ATOMIC_READ) {
		buf = iov[0].addr; /* as default for count == 1 */
		len = psmx2_ioc_size(iov, count, datatype);
		desc0 = desc ? desc[0] : NULL;
	} else {
		buf = NULL;
		len = result_len;
		desc0 = NULL;
	}

	if (result_len < len)
		return -FI_EINVAL;
	
	av = ep_priv->av;
	if (av && av->type == FI_AV_TABLE) {
		idx = dest_addr;
		if (idx >= av->last)
			return -FI_EINVAL;

		psm2_epaddr = av->epaddrs[idx];
		vlane = av->vlanes[idx];
	} else {
		if (!dest_addr)
			return -FI_EINVAL;

		psm2_epaddr = PSMX2_ADDR_TO_EP(dest_addr);
		vlane = PSMX2_ADDR_TO_VL(dest_addr);
	}

	epaddr_context = psm2_epaddr_getctxt((void *)psm2_epaddr);
	if (epaddr_context->epid == ep_priv->domain->psm2_epid) {
		if (buf && count > 1) {
			buf = malloc(len);
			psmx2_ioc_read(iov, count, datatype, buf, len);
			desc0 = NULL;
		}

		if (result_count > 1) {
			result = malloc(len);
			if (!result) {
				if (buf && count > 1)
					free(buf);
				return -FI_ENOMEM;
			}
			result_desc0 = result_desc ? result_desc[0] : NULL;
		} else {
			result = resultv[0].addr;
			result_desc0 = NULL;
		}

		err = psmx2_atomic_self(PSMX2_AM_REQ_ATOMIC_READWRITE,
					ep_priv, ep_priv->domain->eps[vlane],
					buf, count, desc0, NULL, NULL, result,
					result_desc0, addr, key, datatype, op,
					context, flags);

		if (result_count > 1) {
			psmx2_ioc_write(resultv, result_count, datatype, result, len);
			free(result);
		}

		if (buf && count > 1)
			free(buf);

		return err;
	}

	chunk_size = psmx2_am_param.max_request_short;
	if (len > chunk_size)
		return -FI_EMSGSIZE;

	iov_size = result_count > 1 ? result_count * sizeof(struct fi_ioc) : 0;

	if (((flags & FI_INJECT) || count > 1) && op != FI_ATOMIC_READ) {
		req = malloc(sizeof(*req) + iov_size + len);
		if (!req)
			return -FI_ENOMEM;
		buf = (uint8_t *)req +  sizeof(*req) + iov_size;
		memset(req, 0, sizeof(*req));
		psmx2_ioc_read(iov, count, datatype, buf, len);
	} else {
		req = calloc(1, sizeof(*req) + iov_size);
		if (!req)
			return -FI_ENOMEM;
	}

	if (iov_size) {
		memcpy(req->ioc, resultv, iov_size);
		req->atomic.iov_count = result_count;
		req->atomic.result = NULL;
	} else {
		req->atomic.buf = buf;
		req->atomic.result = resultv[0].addr;
	}

	req->no_event = (flags & PSMX2_NO_COMPLETION) ||
			(ep_priv->send_selective_completion && !(flags & FI_COMPLETION));

	req->op = PSMX2_AM_REQ_ATOMIC_READWRITE;
	req->atomic.buf = (void *)buf;
	req->atomic.len = len;
	req->atomic.addr = addr;
	req->atomic.key = key;
	req->atomic.context = context;
	req->atomic.datatype = datatype;
	req->ep = ep_priv;
	if (op == FI_ATOMIC_READ)
		req->cq_flags = FI_READ | FI_ATOMIC;
	else
		req->cq_flags = FI_WRITE | FI_ATOMIC;

	args[0].u32w0 = PSMX2_AM_REQ_ATOMIC_READWRITE;
	PSMX2_AM_SET_DST(args[0].u32w0, vlane);
	args[0].u32w1 = len / fi_datatype_size(datatype);
	args[1].u64 = (uint64_t)(uintptr_t)req;
	args[2].u64 = addr;
	args[3].u64 = key;
	args[4].u32w0 = datatype;
	args[4].u32w1 = op;
	psm2_am_request_short(psm2_epaddr,
			      PSMX2_AM_ATOMIC_HANDLER, args, 5,
			      (void *)buf, (buf?len:0), am_flags, NULL, NULL);

	return 0;
}

static ssize_t psmx2_atomic_readwrite(struct fid_ep *ep,
				      const void *buf,
				      size_t count, void *desc,
				      void *result, void *result_desc,
				      fi_addr_t dest_addr,
				      uint64_t addr, uint64_t key,
				      enum fi_datatype datatype,
				      enum fi_op op, void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);
	return psmx2_atomic_readwrite_generic(ep, buf, count, desc,
					      result, result_desc, dest_addr,
					      addr, key, datatype, op,
					      context, ep_priv->tx_flags);
}

static ssize_t psmx2_atomic_readwritemsg(struct fid_ep *ep,
					 const struct fi_msg_atomic *msg,
					 struct fi_ioc *resultv,
					 void **result_desc,
					 size_t result_count,
					 uint64_t flags)
{
	void *buf;
	size_t count;
	void *desc;

	if (!msg || !msg->rma_iov || msg->rma_iov_count !=1 || !resultv ||
	    !result_count)
		return -FI_EINVAL;
 
	if ((msg->op != FI_ATOMIC_READ) && (!msg->msg_iov || !msg->iov_count))
		return -FI_EINVAL;

	if ((msg->op != FI_ATOMIC_READ && msg->iov_count > 1) ||
	    result_count > 1)
		return psmx2_atomic_readwritev_generic(ep, msg->msg_iov, msg->desc,
						       msg->iov_count, resultv,
						       result_desc, result_count,
						       msg->addr,
						       msg->rma_iov[0].addr,
						       msg->rma_iov[0].key,
						       msg->datatype, msg->op,
						       msg->context, flags);

	if (msg->op == FI_ATOMIC_READ) {
		buf = NULL;
		count = resultv[0].count;
		desc = result_desc ? result_desc[0] : NULL;
	} else {
		buf = msg->msg_iov[0].addr;
		count = msg->msg_iov[0].count;
		desc = msg->desc ? msg->desc[0] : NULL;
	}

	return psmx2_atomic_readwrite_generic(ep, buf, count, desc, resultv[0].addr,
					      result_desc ? result_desc[0] : NULL,
					      msg->addr, msg->rma_iov[0].addr,
					      msg->rma_iov[0].key, msg->datatype,
					      msg->op, msg->context, flags);
}

static ssize_t psmx2_atomic_readwritev(struct fid_ep *ep,
				       const struct fi_ioc *iov,
				       void **desc, size_t count,
				       struct fi_ioc *resultv,
				       void **result_desc, size_t result_count,
				       fi_addr_t dest_addr,
				       uint64_t addr, uint64_t key,
				       enum fi_datatype datatype,
				       enum fi_op op, void *context)
{
	struct psmx2_fid_ep *ep_priv;
	void *buf;
	void *src_desc;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (!resultv || !result_count)
		return -FI_EINVAL;

	if ((op != FI_ATOMIC_READ) && (!iov || !count))
		return -FI_EINVAL;
	
	if ((op != FI_ATOMIC_READ && count > 1) || result_count > 1)
		return psmx2_atomic_readwritev_generic(ep, iov, desc, count,
					      resultv, result_desc, result_count,
					      dest_addr, addr, key, datatype, op,
					      context, ep_priv->tx_flags);

	if (FI_ATOMIC_READ) {
		buf = NULL;
		count = resultv[0].count;
		src_desc = result_desc ? result_desc[0] : NULL;
	} else {
		buf = iov[0].addr;
		count = iov[0].count;
		src_desc = desc ? desc[0] : NULL;
	}

	return psmx2_atomic_readwrite_generic(ep, buf, count, src_desc, resultv[0].addr,
					      result_desc ? result_desc[0] : NULL,
					      dest_addr, addr, key, datatype, op,
					      context, ep_priv->tx_flags);
}

ssize_t psmx2_atomic_compwrite_generic(struct fid_ep *ep,
				       const void *buf,
				       size_t count, void *desc,
				       const void *compare, void *compare_desc,
				       void *result, void *result_desc,
				       fi_addr_t dest_addr,
				       uint64_t addr, uint64_t key,
				       enum fi_datatype datatype,
				       enum fi_op op, void *context,
				       uint64_t flags)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	struct psmx2_epaddr_context *epaddr_context;
	struct psmx2_am_request *req;
	psm2_amarg_t args[8];
	psm2_epaddr_t psm2_epaddr;
	uint8_t vlane;
	int am_flags = PSM2_AM_FLAG_ASYNC;
	int chunk_size, len;
	void *tmp_buf = NULL;
	size_t idx;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (flags & FI_TRIGGER) {
		struct psmx2_trigger *trigger;
		struct fi_triggered_context *ctxt = context;

		trigger = calloc(1, sizeof(*trigger));
		if (!trigger)
			return -FI_ENOMEM;

		trigger->op = PSMX2_TRIGGERED_ATOMIC_COMPWRITE;
		trigger->cntr = container_of(ctxt->trigger.threshold.cntr,
					     struct psmx2_fid_cntr, cntr);
		trigger->threshold = ctxt->trigger.threshold.threshold;
		trigger->atomic_compwrite.ep = ep;
		trigger->atomic_compwrite.buf = buf;
		trigger->atomic_compwrite.count = count;
		trigger->atomic_compwrite.desc = desc;
		trigger->atomic_compwrite.compare = compare;
		trigger->atomic_compwrite.compare_desc = compare_desc;
		trigger->atomic_compwrite.result = result;
		trigger->atomic_compwrite.result_desc = result_desc;
		trigger->atomic_compwrite.dest_addr = dest_addr;
		trigger->atomic_compwrite.addr = addr;
		trigger->atomic_compwrite.key = key;
		trigger->atomic_compwrite.datatype = datatype;
		trigger->atomic_compwrite.atomic_op = op;
		trigger->atomic_compwrite.context = context;
		trigger->atomic_compwrite.flags = flags & ~FI_TRIGGER;

		psmx2_cntr_add_trigger(trigger->cntr, trigger);
		return 0;
	}

	if (!buf)
		return -FI_EINVAL;

	if (datatype < 0 || datatype >= FI_DATATYPE_LAST)
		return -FI_EINVAL;

	if (op < 0 || op >= FI_ATOMIC_OP_LAST)
		return -FI_EINVAL;

	av = ep_priv->av;
	if (av && av->type == FI_AV_TABLE) {
		idx = dest_addr;
		if (idx >= av->last)
			return -FI_EINVAL;

		psm2_epaddr = av->epaddrs[idx];
		vlane = av->vlanes[idx];
	} else {
		if (!dest_addr)
			return -FI_EINVAL;

		psm2_epaddr = PSMX2_ADDR_TO_EP(dest_addr);
		vlane = PSMX2_ADDR_TO_VL(dest_addr);
	}

	epaddr_context = psm2_epaddr_getctxt((void *)psm2_epaddr);
	if (epaddr_context->epid == ep_priv->domain->psm2_epid)
		return psmx2_atomic_self(PSMX2_AM_REQ_ATOMIC_COMPWRITE,
					 ep_priv, ep_priv->domain->eps[vlane],
					 buf, count, desc, compare,
					 compare_desc, result, result_desc,
					 addr, key, datatype, op,
					 context, flags);

	chunk_size = psmx2_am_param.max_request_short;
	len = fi_datatype_size(datatype) * count;
	if (len * 2 > chunk_size)
		return -FI_EMSGSIZE;

	if (flags & FI_INJECT) {
		req = malloc(sizeof(*req) + len + len);
		if (!req)
			return -FI_ENOMEM;
		memset(req, 0, sizeof(*req));
		memcpy((uint8_t *)req + sizeof(*req), (void *)buf, len);
		memcpy((uint8_t *)req + sizeof(*req) + len, (void *)compare, len);
		buf = (uint8_t *)req + sizeof(*req);
		compare = (uint8_t *)buf + len;
	} else {
		req = calloc(1, sizeof(*req));
		if (!req)
			return -FI_ENOMEM;

		if ((uintptr_t)compare != (uintptr_t)buf + len) {
			tmp_buf = malloc(len * 2);
			if (!tmp_buf) {
				free(req);
				return -FI_ENOMEM;
			}

			memcpy(tmp_buf, buf, len);
			memcpy((uint8_t *)tmp_buf + len, compare, len);
		}
	}

	req->no_event = (flags & PSMX2_NO_COMPLETION) ||
			(ep_priv->send_selective_completion && !(flags & FI_COMPLETION));

	req->op = PSMX2_AM_REQ_ATOMIC_COMPWRITE;
	req->atomic.buf = (void *)buf;
	req->atomic.len = len;
	req->atomic.addr = addr;
	req->atomic.key = key;
	req->atomic.context = context;
	req->atomic.result = result;
	req->atomic.datatype = datatype;
	req->ep = ep_priv;
	req->cq_flags = FI_WRITE | FI_ATOMIC;

	args[0].u32w0 = PSMX2_AM_REQ_ATOMIC_COMPWRITE;
	PSMX2_AM_SET_DST(args[0].u32w0, vlane);
	args[0].u32w1 = count;
	args[1].u64 = (uint64_t)(uintptr_t)req;
	args[2].u64 = addr;
	args[3].u64 = key;
	args[4].u32w0 = datatype;
	args[4].u32w1 = op;
	psm2_am_request_short(psm2_epaddr,
			      PSMX2_AM_ATOMIC_HANDLER, args, 5,
			      tmp_buf ? tmp_buf : (void *)buf,
			      len * 2, am_flags,
			      psmx2_am_atomic_completion, tmp_buf);

	return 0;
}

ssize_t psmx2_atomic_compwritev_generic(struct fid_ep *ep,
					const struct fi_ioc *iov,
					void **desc, size_t count,
					const struct fi_ioc *comparev,
					void **compare_desc,
					size_t compare_count,
					struct fi_ioc *resultv,
					void **result_desc,
					size_t result_count,
					fi_addr_t dest_addr,
					uint64_t addr, uint64_t key,
					enum fi_datatype datatype,
					enum fi_op op, void *context,
					uint64_t flags)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	struct psmx2_epaddr_context *epaddr_context;
	struct psmx2_am_request *req;
	psm2_amarg_t args[8];
	psm2_epaddr_t psm2_epaddr;
	uint8_t vlane;
	int am_flags = PSM2_AM_FLAG_ASYNC;
	int chunk_size;
	size_t idx;
	size_t len, compare_len, result_len, iov_size;
	uint8_t *buf, *compare, *result;
	void *desc0, *compare_desc0, *result_desc0;
	int err;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (flags & FI_TRIGGER) {
		struct psmx2_trigger *trigger;
		struct fi_triggered_context *ctxt = context;

		trigger = calloc(1, sizeof(*trigger));
		if (!trigger)
			return -FI_ENOMEM;

		trigger->op = PSMX2_TRIGGERED_ATOMIC_COMPWRITEV;
		trigger->cntr = container_of(ctxt->trigger.threshold.cntr,
					     struct psmx2_fid_cntr, cntr);
		trigger->threshold = ctxt->trigger.threshold.threshold;
		trigger->atomic_compwritev.ep = ep;
		trigger->atomic_compwritev.iov = iov;
		trigger->atomic_compwritev.desc = desc;
		trigger->atomic_compwritev.count = count;
		trigger->atomic_compwritev.comparev = comparev;
		trigger->atomic_compwritev.compare_desc = compare_desc;
		trigger->atomic_compwritev.compare_count = compare_count;
		trigger->atomic_compwritev.resultv = resultv;
		trigger->atomic_compwritev.result_desc = result_desc;
		trigger->atomic_compwritev.result_count = result_count;
		trigger->atomic_compwritev.dest_addr = dest_addr;
		trigger->atomic_compwritev.addr = addr;
		trigger->atomic_compwritev.key = key;
		trigger->atomic_compwritev.datatype = datatype;
		trigger->atomic_compwritev.atomic_op = op;
		trigger->atomic_compwritev.context = context;
		trigger->atomic_compwritev.flags = flags & ~FI_TRIGGER;

		psmx2_cntr_add_trigger(trigger->cntr, trigger);
		return 0;
	}

	if (!iov || !count || !comparev || !compare_count || !resultv ||
	    !result_count)
		return -FI_EINVAL;

	while (count && !iov[count-1].count)
		count--;

	while (compare_count && !comparev[compare_count-1].count)
		compare_count--;

	while (result_count && !resultv[result_count-1].count)
		result_count--;

	if (datatype < 0 || datatype >= FI_DATATYPE_LAST)
		return -FI_EINVAL;

	if (op < 0 || op >= FI_ATOMIC_OP_LAST)
		return -FI_EINVAL;

	len = psmx2_ioc_size(iov, count, datatype);
	compare_len = psmx2_ioc_size(comparev, compare_count, datatype);
	result_len = psmx2_ioc_size(resultv, result_count, datatype);

	if (compare_len < len || result_len < len)
		return -FI_EINVAL;

	av = ep_priv->av;
	if (av && av->type == FI_AV_TABLE) {
		idx = dest_addr;
		if (idx >= av->last)
			return -FI_EINVAL;

		psm2_epaddr = av->epaddrs[idx];
		vlane = av->vlanes[idx];
	} else {
		if (!dest_addr)
			return -FI_EINVAL;

		psm2_epaddr = PSMX2_ADDR_TO_EP(dest_addr);
		vlane = PSMX2_ADDR_TO_VL(dest_addr);
	}

	epaddr_context = psm2_epaddr_getctxt((void *)psm2_epaddr);
	if (epaddr_context->epid == ep_priv->domain->psm2_epid) {
		if (count > 1) {
			buf = malloc(len);
			if (!buf)
				return -FI_ENOMEM;
			psmx2_ioc_read(iov, count, datatype, buf, len);
			desc0 = NULL;
		} else {
			buf = iov[0].addr;
			desc0 = desc ? desc[0] : NULL;
		}

		if (compare_count > 1) {
			compare = malloc(len);
			if (!compare) {
				if (count > 1)
					free(buf);
				return -FI_ENOMEM;
			}
			psmx2_ioc_read(comparev, compare_count, datatype, compare, len);
			compare_desc0 = NULL;
		} else {
			compare = comparev[0].addr;
			compare_desc0 = compare_desc ? compare_desc[0] : NULL;
		}

		if (result_count > 1) {
			result = malloc(len);
			if (!result) {
				if (compare_count > 1)
					free(compare);
				if (count > 1)
					free(buf);
				return -FI_ENOMEM;
			}
			result_desc0 = NULL;
		} else {
			result = resultv[0].addr;
			result_desc0 = result_desc ? result_desc[0] : NULL;
		}

		err = psmx2_atomic_self(PSMX2_AM_REQ_ATOMIC_COMPWRITE,
					ep_priv, ep_priv->domain->eps[vlane],
					buf, count, desc0, compare, compare_desc0,
					result, result_desc0, addr, key, datatype, op,
					context, flags);

		if (result_count > 1) {
			psmx2_ioc_write(resultv, result_count, datatype, result, len);
			free(result);
		}

		if (compare_count > 1)
			free(compare);

		if (count > 1)
			free(buf);

		return err;
	}

	chunk_size = psmx2_am_param.max_request_short;
	if (len * 2 > chunk_size)
		return -FI_EMSGSIZE;

	iov_size = result_count > 1 ? result_count * sizeof(struct fi_ioc) : 0;

	if ((flags & FI_INJECT) || count > 1 || compare_count > 1 ||
	    (uintptr_t)comparev[0].addr != (uintptr_t)iov[0].addr + len) {
		req = malloc(sizeof(*req) + iov_size + len + len);
		if (!req)
			return -FI_ENOMEM;
		buf = (uint8_t *)req + sizeof(*req) + iov_size;
		memset(req, 0, sizeof(*req));
		psmx2_ioc_read(iov, count, datatype, buf, len);
		psmx2_ioc_read(comparev, compare_count, datatype, buf + len, len);
	} else {
		req = calloc(1, sizeof(*req) + iov_size);
		if (!req)
			return -FI_ENOMEM;
		buf = iov[0].addr;
	}

	if (iov_size) {
		memcpy(req->ioc, resultv, iov_size);
		req->atomic.iov_count = result_count;
		req->atomic.result = NULL;
	} else {
		req->atomic.buf = buf;
		req->atomic.result = resultv[0].addr;
	}

	req->no_event = (flags & PSMX2_NO_COMPLETION) ||
			(ep_priv->send_selective_completion && !(flags & FI_COMPLETION));

	req->op = PSMX2_AM_REQ_ATOMIC_COMPWRITE;
	req->atomic.len = len;
	req->atomic.addr = addr;
	req->atomic.key = key;
	req->atomic.context = context;
	req->atomic.datatype = datatype;
	req->ep = ep_priv;
	req->cq_flags = FI_WRITE | FI_ATOMIC;

	args[0].u32w0 = PSMX2_AM_REQ_ATOMIC_COMPWRITE;
	PSMX2_AM_SET_DST(args[0].u32w0, vlane);
	args[0].u32w1 = len / fi_datatype_size(datatype);
	args[1].u64 = (uint64_t)(uintptr_t)req;
	args[2].u64 = addr;
	args[3].u64 = key;
	args[4].u32w0 = datatype;
	args[4].u32w1 = op;
	psm2_am_request_short(psm2_epaddr,
			      PSMX2_AM_ATOMIC_HANDLER, args, 5,
			      buf, len * 2, am_flags, NULL, NULL);

	return 0;
}

static ssize_t psmx2_atomic_compwrite(struct fid_ep *ep,
				      const void *buf,
				      size_t count, void *desc,
				      const void *compare, void *compare_desc,
				      void *result, void *result_desc,
				      fi_addr_t dest_addr,
				      uint64_t addr, uint64_t key,
				      enum fi_datatype datatype,
				      enum fi_op op, void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);
	return psmx2_atomic_compwrite_generic(ep, buf, count, desc,
					      compare, compare_desc,
					      result, result_desc,
					      dest_addr, addr, key,
			        	      datatype, op, context, ep_priv->tx_flags);
}

static ssize_t psmx2_atomic_compwritemsg(struct fid_ep *ep,
					 const struct fi_msg_atomic *msg,
					 const struct fi_ioc *comparev,
					 void **compare_desc,
					 size_t compare_count,
					 struct fi_ioc *resultv,
					 void **result_desc,
					 size_t result_count,
					 uint64_t flags)
{
	if (!msg || !msg->msg_iov || msg->iov_count != 1 ||
	    !msg->rma_iov || msg->rma_iov_count != 1 ||
	    !comparev || !compare_count || !resultv || !result_count)
		return -FI_EINVAL;

	if (msg->iov_count > 1 || compare_count > 1 || result_count > 1)
		return psmx2_atomic_compwritev_generic(ep, msg->msg_iov, msg->desc,
						       msg->iov_count, comparev,
						       compare_desc, compare_count,
						       resultv, result_desc, result_count,
						       msg->addr, msg->rma_iov[0].addr,
						       msg->rma_iov[0].key, msg->datatype,
						       msg->op, msg->context, flags);

	return psmx2_atomic_compwrite_generic(ep, msg->msg_iov[0].addr,
					      msg->msg_iov[0].count,
					      msg->desc ? msg->desc[0] : NULL,
					      comparev[0].addr,
					      compare_desc ? compare_desc[0] : NULL,
					      resultv[0].addr,
					      result_desc ? result_desc[0] : NULL,
					      msg->addr, msg->rma_iov[0].addr,
					      msg->rma_iov[0].key, msg->datatype,
					      msg->op, msg->context, flags);
}

static ssize_t psmx2_atomic_compwritev(struct fid_ep *ep,
				       const struct fi_ioc *iov,
				       void **desc, size_t count,
				       const struct fi_ioc *comparev,
				       void **compare_desc,
				       size_t compare_count,
				       struct fi_ioc *resultv,
				       void **result_desc,
				       size_t result_count,
				       fi_addr_t dest_addr,
				       uint64_t addr, uint64_t key,
				       enum fi_datatype datatype,
				       enum fi_op op, void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (!iov || !count || !comparev || !compare_count || !resultv || !result_count)
		return -FI_EINVAL;

	if (count > 1 || compare_count > 1 || result_count > 1)
		return psmx2_atomic_compwritev_generic(ep, iov, desc, count,
						       comparev, compare_desc,
						       compare_count, resultv,
						       result_desc, result_count,
						       dest_addr, addr, key,
						       datatype, op, context,
						       ep_priv->tx_flags);

	return psmx2_atomic_compwrite_generic(ep, iov->addr, iov->count,
					      desc ? desc[0] : NULL,
					      comparev[0].addr,
					      compare_desc ? compare_desc[0] : NULL,
					      resultv[0].addr,
					      result_desc ? result_desc[0] : NULL,
					      dest_addr, addr, key, datatype, op,
					      context, ep_priv->tx_flags);
}

static int psmx2_atomic_writevalid(struct fid_ep *ep,
				   enum fi_datatype datatype,
				   enum fi_op op, size_t *count)
{
	int chunk_size;

	if (datatype < 0 || datatype >= FI_DATATYPE_LAST)
		return -FI_EOPNOTSUPP;

	switch (op) {
	case FI_MIN:
	case FI_MAX:
	case FI_SUM:
	case FI_PROD:
	case FI_LOR:
	case FI_LAND:
	case FI_BOR:
	case FI_BAND:
	case FI_LXOR:
	case FI_BXOR:
	case FI_ATOMIC_WRITE:
		break;

	default:
		return -FI_EOPNOTSUPP;
	}

	if (count) {
		chunk_size = psmx2_am_param.max_request_short;
		*count = chunk_size / fi_datatype_size(datatype);
	}
	return 0;
}

static int psmx2_atomic_readwritevalid(struct fid_ep *ep,
				       enum fi_datatype datatype,
				       enum fi_op op, size_t *count)
{
	int chunk_size;

	if (datatype < 0 || datatype >= FI_DATATYPE_LAST)
		return -FI_EOPNOTSUPP;

	switch (op) {
	case FI_MIN:
	case FI_MAX:
	case FI_SUM:
	case FI_PROD:
	case FI_LOR:
	case FI_LAND:
	case FI_BOR:
	case FI_BAND:
	case FI_LXOR:
	case FI_BXOR:
	case FI_ATOMIC_READ:
	case FI_ATOMIC_WRITE:
		break;

	default:
		return -FI_EOPNOTSUPP;
	}

	if (count) {
		chunk_size = psmx2_am_param.max_request_short;
		*count = chunk_size / fi_datatype_size(datatype);
	}
	return 0;
}

static int psmx2_atomic_compwritevalid(struct fid_ep *ep,
				       enum fi_datatype datatype,
				       enum fi_op op, size_t *count)
{
	int chunk_size;

	if (datatype < 0 || datatype >= FI_DATATYPE_LAST)
		return -FI_EOPNOTSUPP;

	switch (op) {
	case FI_CSWAP:
	case FI_CSWAP_NE:
		break;

	case FI_CSWAP_LE:
	case FI_CSWAP_LT:
	case FI_CSWAP_GE:
	case FI_CSWAP_GT:
		if (datatype == FI_FLOAT_COMPLEX ||
		    datatype == FI_DOUBLE_COMPLEX ||
		    datatype == FI_LONG_DOUBLE_COMPLEX)
			return -FI_EOPNOTSUPP;
		break;

	case FI_MSWAP:
		if (datatype == FI_FLOAT_COMPLEX ||
		    datatype == FI_DOUBLE_COMPLEX ||
		    datatype == FI_LONG_DOUBLE_COMPLEX ||
		    datatype == FI_FLOAT ||
		    datatype == FI_DOUBLE ||
		    datatype == FI_LONG_DOUBLE)
			return -FI_EOPNOTSUPP;
		break;

	default:
		return -FI_EOPNOTSUPP;
	}

	if (count) {
		chunk_size = psmx2_am_param.max_request_short;
		*count = chunk_size / (2 * fi_datatype_size(datatype));
	}
	return 0;
}

struct fi_ops_atomic psmx2_atomic_ops = {
	.size = sizeof(struct fi_ops_atomic),
	.write = psmx2_atomic_write,
	.writev = psmx2_atomic_writev,
	.writemsg = psmx2_atomic_writemsg,
	.inject = psmx2_atomic_inject,
	.readwrite = psmx2_atomic_readwrite,
	.readwritev = psmx2_atomic_readwritev,
	.readwritemsg = psmx2_atomic_readwritemsg,
	.compwrite = psmx2_atomic_compwrite,
	.compwritev = psmx2_atomic_compwritev,
	.compwritemsg = psmx2_atomic_compwritemsg,
	.writevalid = psmx2_atomic_writevalid,
	.readwritevalid = psmx2_atomic_readwritevalid,
	.compwritevalid = psmx2_atomic_compwritevalid,
};

