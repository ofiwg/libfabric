/*
 * Copyright (c) 2018-2023 Intel Corporation. All rights reserved.
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
#include "ofi_hook.h"
#include "ofi_prov.h"
#include "ofi_iov.h"

#define TRACE_EP(ret, ep, buf, len, addr, tagkey, ignoredata, context) \
	if (!(ret)) { \
		FI_TRACE((ep)->domain->fabric->hprov, FI_LOG_EP_DATA, \
			"buf %p len %zu src %zu tag/key 0x%zx " \
			"ignore/data 0x%zx ctx %p\n", \
			buf, len, addr, (uint64_t) tagkey, \
			(uint64_t) ignoredata, context); \
	}

#define TRACE_EP_IOV(ret, ep, iov, count, addr, tagkey, ignoredata, context) \
	TRACE_EP(ret, ep, count ? iov[0].iov_base : NULL, \
		 ofi_total_iov_len(iov, count), addr, tagkey, ignoredata, context)

typedef void (*trace_cq_entry_fn)(const struct fi_provider *prov,
				const char *func, int line,
				int count, void *buf, uint64_t data);

static void
trace_cq_unknown(const struct fi_provider *prov, const char *func,
                 int line, int count, void *buf, uint64_t data)
{
	// do nothing
}

static void
trace_cq_context_entry(const struct fi_provider *prov, const char *func,
                       int line, int count, void *buf, uint64_t data)
{
	struct fi_cq_entry *entry = (struct fi_cq_entry *)buf;

	for (int i = 0; i < count; i++) {
		fi_log(prov, FI_LOG_TRACE, FI_LOG_CQ, func, line,
		       "ctx %p\n", entry[i].op_context);
	}
}

static void
trace_cq_msg_entry(const struct fi_provider *prov, const char *func,
                   int line, int count, void *buf, uint64_t data)
{
	struct fi_cq_msg_entry *entry = (struct fi_cq_msg_entry *)buf;

	for (int i = 0; i < count; i++) {
		if (entry[i].flags & FI_RECV) {
			fi_log(prov, FI_LOG_TRACE, FI_LOG_CQ, func, line,
			       "ctx %p flags 0x%lx len %zu\n",
			       entry[i].op_context, entry[i].flags,  entry[i].len);
		} else {
			fi_log(prov, FI_LOG_TRACE, FI_LOG_CQ, func, line,
			       "ctx %p flags 0x%lx\n",
			       entry[i].op_context, entry[i].flags);
		}
	}
}

static void
trace_cq_data_entry(const struct fi_provider *prov, const char *func,
                    int line, int count, void *buf, uint64_t data)
{
	struct fi_cq_data_entry *entry = (struct fi_cq_data_entry *)buf;

	for (int i = 0; i < count; i++) {
		if (entry[i].flags & FI_RECV) {
			fi_log(prov, FI_LOG_TRACE, FI_LOG_CQ, func, line,
			       "ctx %p flags 0x%lx len %zu buf %p, data %lu\n",
			       entry[i].op_context, entry[i].flags,
			       entry[i].len, entry[i].buf,
			       (entry[i].flags & FI_REMOTE_CQ_DATA) ? entry[i].data : 0);
		} else {
			fi_log(prov, FI_LOG_TRACE, FI_LOG_CQ, func, line,
			       "ctx %p flags 0x%lx\n",
			       entry[i].op_context, entry[i].flags);
		}
	}
}

static void
trace_cq_tagged_entry(const struct fi_provider *prov, const char *func,
                      int line, int count, void *buf, uint64_t data)
{
	struct fi_cq_tagged_entry *entry = (struct fi_cq_tagged_entry *)buf;

	for (int i = 0; i < count; i++) {
		if (entry[i].flags & FI_RECV) {
			fi_log(prov, FI_LOG_TRACE, FI_LOG_CQ, func, line,
			       "ctx %p flags 0x%lx len %zu buf %p, data %lu tag 0x%lx\n",
			       entry[i].op_context, entry[i].flags,
			       entry[i].len, entry[i].buf,
			       (entry[i].flags & FI_REMOTE_CQ_DATA) ? entry[i].data : 0,
			       entry[i].tag);
		} else {
			fi_log(prov, FI_LOG_TRACE, FI_LOG_CQ, func, line,
			       "ctx %p flags 0x%lx\n",
			       entry[i].op_context, entry[i].flags);
		}
	}
}

static trace_cq_entry_fn trace_cq_entry[] = {
	trace_cq_unknown,
	trace_cq_context_entry,
	trace_cq_msg_entry,
	trace_cq_data_entry,
	trace_cq_tagged_entry
};

static inline void
trace_cq(struct hook_cq *cq, const char *func, int line,
         int count, void *buf, uint64_t data)
{
	if ((count > 0) &&
	    fi_log_enabled(cq->domain->fabric->hprov, FI_LOG_TRACE, FI_LOG_CQ)) {
		trace_cq_entry[cq->format](cq->domain->fabric->hprov, func,
		                           line, count, buf, data);
	}
}

static ssize_t
trace_atomic_write(struct fid_ep *ep,
		   const void *buf, size_t count, void *desc,
		   fi_addr_t dest_addr, uint64_t addr, uint64_t key,
		   enum fi_datatype datatype, enum fi_op op, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_atomic(myep->hep, buf, count, desc, dest_addr,
			addr, key, datatype, op, context);
	return ret;
}

static ssize_t
trace_atomic_writev(struct fid_ep *ep,
		    const struct fi_ioc *iov, void **desc, size_t count,
		    fi_addr_t dest_addr, uint64_t addr, uint64_t key,
		    enum fi_datatype datatype, enum fi_op op, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_atomicv(myep->hep, iov, desc, count, dest_addr,
			 addr, key, datatype, op, context);
	return ret;
}

static ssize_t
trace_atomic_writemsg(struct fid_ep *ep,
		      const struct fi_msg_atomic *msg, uint64_t flags)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_atomicmsg(myep->hep, msg, flags);
	return ret;
}

static ssize_t
trace_atomic_inject(struct fid_ep *ep, const void *buf, size_t count,
		    fi_addr_t dest_addr, uint64_t addr, uint64_t key,
		    enum fi_datatype datatype, enum fi_op op)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_inject_atomic(myep->hep, buf, count, dest_addr,
			       addr, key, datatype, op);
	return ret;
}

static ssize_t
trace_atomic_readwrite(struct fid_ep *ep,
		       const void *buf, size_t count, void *desc,
		       void *result, void *result_desc,
		       fi_addr_t dest_addr, uint64_t addr, uint64_t key,
		       enum fi_datatype datatype, enum fi_op op, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_fetch_atomic(myep->hep, buf, count, desc,
			      result, result_desc, dest_addr,
			      addr, key, datatype, op, context);
	return ret;
}

static ssize_t
trace_atomic_readwritev(struct fid_ep *ep,
			const struct fi_ioc *iov, void **desc, size_t count,
			struct fi_ioc *resultv, void **result_desc,
			size_t result_count, fi_addr_t dest_addr,
			uint64_t addr, uint64_t key, enum fi_datatype datatype,
			enum fi_op op, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_fetch_atomicv(myep->hep, iov, desc, count,
			       resultv, result_desc, result_count,
			       dest_addr, addr, key, datatype, op, context);
	return ret;
}

static ssize_t
trace_atomic_readwritemsg(struct fid_ep *ep, const struct fi_msg_atomic *msg,
			  struct fi_ioc *resultv, void **result_desc,
			  size_t result_count, uint64_t flags)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_fetch_atomicmsg(myep->hep, msg, resultv, result_desc,
				 result_count, flags);
	return ret;
}

static ssize_t
trace_atomic_compwrite(struct fid_ep *ep,
		       const void *buf, size_t count, void *desc,
		       const void *compare, void *compare_desc,
		       void *result, void *result_desc,
		       fi_addr_t dest_addr, uint64_t addr, uint64_t key,
		       enum fi_datatype datatype, enum fi_op op, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_compare_atomic(myep->hep, buf, count, desc,
				compare, compare_desc, result, result_desc,
				dest_addr, addr, key, datatype, op, context);
	return ret;
}

static ssize_t
trace_atomic_compwritev(struct fid_ep *ep,
			const struct fi_ioc *iov, void **desc, size_t count,
			const struct fi_ioc *comparev, void **compare_desc,
			size_t compare_count, struct fi_ioc *resultv,
			void **result_desc, size_t result_count,
			fi_addr_t dest_addr, uint64_t addr, uint64_t key,
			enum fi_datatype datatype, enum fi_op op, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_compare_atomicv(myep->hep, iov, desc, count,
				 comparev, compare_desc, compare_count,
				 resultv, result_desc, result_count, dest_addr,
				 addr, key, datatype, op, context);
	return ret;
}

static ssize_t
trace_atomic_compwritemsg(struct fid_ep *ep,
			  const struct fi_msg_atomic *msg,
			  const struct fi_ioc *comparev, void **compare_desc,
			  size_t compare_count, struct fi_ioc *resultv,
			  void **result_desc, size_t result_count,
			  uint64_t flags)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_compare_atomicmsg(myep->hep, msg,
				   comparev, compare_desc, compare_count,
				   resultv, result_desc, result_count, flags);
	return ret;
}

static int
trace_atomic_writevalid(struct fid_ep *ep, enum fi_datatype datatype,
			enum fi_op op, size_t *count)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_atomicvalid(myep->hep, datatype, op, count);
	return ret;
}

static int
trace_atomic_readwritevalid(struct fid_ep *ep, enum fi_datatype datatype,
			    enum fi_op op, size_t *count)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_fetch_atomicvalid(myep->hep, datatype, op, count);
	return ret;
}

static int
trace_atomic_compwritevalid(struct fid_ep *ep, enum fi_datatype datatype,
			    enum fi_op op, size_t *count)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_compare_atomicvalid(myep->hep, datatype, op, count);
	return ret;
}

struct fi_ops_atomic trace_atomic_ops = {
	.size = sizeof(struct fi_ops_atomic),
	.write = trace_atomic_write,
	.writev = trace_atomic_writev,
	.writemsg = trace_atomic_writemsg,
	.inject = trace_atomic_inject,
	.readwrite = trace_atomic_readwrite,
	.readwritev = trace_atomic_readwritev,
	.readwritemsg = trace_atomic_readwritemsg,
	.compwrite = trace_atomic_compwrite,
	.compwritev = trace_atomic_compwritev,
	.compwritemsg = trace_atomic_compwritemsg,
	.writevalid = trace_atomic_writevalid,
	.readwritevalid = trace_atomic_readwritevalid,
	.compwritevalid = trace_atomic_compwritevalid,
};


static ssize_t
trace_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
	   fi_addr_t src_addr, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_recv(myep->hep, buf, len, desc, src_addr, context);
	TRACE_EP(ret, myep, buf, len, src_addr, 0, 0, context);

	return ret;
}

static ssize_t
trace_recvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
	    size_t count, fi_addr_t src_addr, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_recvv(myep->hep, iov, desc, count, src_addr, context);
	TRACE_EP_IOV(ret, myep, iov, count, src_addr, 0, 0, context);

	return ret;
}

static ssize_t
trace_recvmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_recvmsg(myep->hep, msg, flags);
	TRACE_EP_IOV(ret, myep, msg->msg_iov, msg->iov_count, msg->addr,
		     0, 0, msg->context);

	return ret;
}

static ssize_t
trace_send(struct fid_ep *ep, const void *buf, size_t len, void *desc,
	   fi_addr_t dest_addr, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_send(myep->hep, buf, len, desc, dest_addr, context);
	TRACE_EP(ret, myep, buf, len, dest_addr, 0, 0, context);

	return ret;
}

static ssize_t
trace_sendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
	    size_t count, fi_addr_t dest_addr, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_sendv(myep->hep, iov, desc, count, dest_addr, context);
	TRACE_EP_IOV(ret, myep, iov, count, dest_addr, 0, 0, context);

	return ret;
}

static ssize_t
trace_sendmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_sendmsg(myep->hep, msg, flags);
	TRACE_EP_IOV(ret, myep, msg->msg_iov, msg->iov_count, msg->addr,
		     0, flags & FI_REMOTE_CQ_DATA ? msg->data : 0, msg->context);

	return ret;
}

static ssize_t
trace_inject(struct fid_ep *ep, const void *buf, size_t len,
	     fi_addr_t dest_addr)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_inject(myep->hep, buf, len, dest_addr);
	TRACE_EP(ret, myep, buf, len, dest_addr, 0, 0, NULL);

	return ret;
}

static ssize_t
trace_senddata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
	       uint64_t data, fi_addr_t dest_addr, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_senddata(myep->hep, buf, len, desc, data, dest_addr, context);
	TRACE_EP(ret, myep, buf, len, dest_addr, 0, data, context);

	return ret;
}

static ssize_t
trace_injectdata(struct fid_ep *ep, const void *buf, size_t len,
		 uint64_t data, fi_addr_t dest_addr)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_injectdata(myep->hep, buf, len, data, dest_addr);
	TRACE_EP(ret, myep, buf, len, dest_addr, 0, data, NULL);

	return ret;
}

static struct fi_ops_msg trace_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = trace_recv,
	.recvv = trace_recvv,
	.recvmsg = trace_recvmsg,
	.send = trace_send,
	.sendv = trace_sendv,
	.sendmsg = trace_sendmsg,
	.inject = trace_inject,
	.senddata = trace_senddata,
	.injectdata = trace_injectdata,
};


static ssize_t
trace_read(struct fid_ep *ep, void *buf, size_t len, void *desc,
	   fi_addr_t src_addr, uint64_t addr, uint64_t key, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_read(myep->hep, buf, len, desc, src_addr, addr, key, context);
	TRACE_EP(ret, myep, buf, len, src_addr, key, 0, context);

	return ret;
}

static ssize_t
trace_readv(struct fid_ep *ep, const struct iovec *iov, void **desc,
	    size_t count, fi_addr_t src_addr, uint64_t addr, uint64_t key,
	    void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_readv(myep->hep, iov, desc, count, src_addr,
		       addr, key, context);
	TRACE_EP_IOV(ret, myep, iov, count, src_addr, key, 0, context);

	return ret;
}

static ssize_t
trace_readmsg(struct fid_ep *ep, const struct fi_msg_rma *msg, uint64_t flags)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_readmsg(myep->hep, msg, flags);
	TRACE_EP_IOV(ret, myep, msg->msg_iov, msg->iov_count, msg->addr,
		     msg->rma_iov_count ? msg->rma_iov[0].key : 0, 0,
		     msg->context);

	return ret;
}

static ssize_t
trace_write(struct fid_ep *ep, const void *buf, size_t len, void *desc,
	    fi_addr_t dest_addr, uint64_t addr, uint64_t key, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_write(myep->hep, buf, len, desc, dest_addr,
		       addr, key, context);
	TRACE_EP(ret, myep, buf, len, dest_addr, key, 0, context);

	return ret;
}

static ssize_t
trace_writev(struct fid_ep *ep, const struct iovec *iov, void **desc,
	     size_t count, fi_addr_t dest_addr, uint64_t addr, uint64_t key,
	     void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_writev(myep->hep, iov, desc, count, dest_addr,
			addr, key, context);
	TRACE_EP_IOV(ret, myep, iov, count, dest_addr, key, 0, context);

	return ret;
}

static ssize_t
trace_writemsg(struct fid_ep *ep, const struct fi_msg_rma *msg, uint64_t flags)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_writemsg(myep->hep, msg, flags);
	TRACE_EP_IOV(ret, myep, msg->msg_iov, msg->iov_count, msg->addr,
		     msg->rma_iov_count ? msg->rma_iov[0].key : 0,
		     flags & FI_REMOTE_CQ_DATA ? msg->data : 0,
		     msg->context);

	return ret;
}

static ssize_t
trace_inject_write(struct fid_ep *ep, const void *buf, size_t len,
		   fi_addr_t dest_addr, uint64_t addr, uint64_t key)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_inject_write(myep->hep, buf, len, dest_addr, addr, key);
	TRACE_EP(ret, myep, buf, len, dest_addr, key, 0, NULL);

	return ret;
}

static ssize_t
trace_writedata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		uint64_t data, fi_addr_t dest_addr, uint64_t addr,
		uint64_t key, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_writedata(myep->hep, buf, len, desc, data,
			   dest_addr, addr, key, context);
	TRACE_EP(ret, myep, buf, len, dest_addr, key, data, context);

	return ret;
}

static ssize_t
trace_inject_writedata(struct fid_ep *ep, const void *buf, size_t len,
		       uint64_t data, fi_addr_t dest_addr, uint64_t addr,
		       uint64_t key)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_inject_writedata(myep->hep, buf, len, data, dest_addr,
				  addr, key);
	TRACE_EP(ret, myep, buf, len, dest_addr, key, data, NULL);

	return ret;
}

static struct fi_ops_rma trace_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = trace_read,
	.readv = trace_readv,
	.readmsg = trace_readmsg,
	.write = trace_write,
	.writev = trace_writev,
	.writemsg = trace_writemsg,
	.inject = trace_inject_write,
	.writedata = trace_writedata,
	.injectdata = trace_inject_writedata,
};


static ssize_t
trace_trecv(struct fid_ep *ep, void *buf, size_t len, void *desc,
	    fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
	    void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_trecv(myep->hep, buf, len, desc, src_addr,
		       tag, ignore, context);
	TRACE_EP(ret, myep, buf, len, src_addr, tag, ignore, context);

	return ret;
}

static ssize_t
trace_trecvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
	     size_t count, fi_addr_t src_addr, uint64_t tag,
	     uint64_t ignore, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_trecvv(myep->hep, iov, desc, count, src_addr,
			tag, ignore, context);
	TRACE_EP_IOV(ret, myep, iov, count, src_addr, tag, ignore, context);

	return ret;
}

static ssize_t
trace_trecvmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
	       uint64_t flags)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_trecvmsg(myep->hep, msg, flags);
	TRACE_EP_IOV(ret, myep, msg->msg_iov, msg->iov_count, msg->addr,
		     msg->tag, msg->ignore, msg->context);

	return ret;
}

static ssize_t
trace_tsend(struct fid_ep *ep, const void *buf, size_t len, void *desc,
	    fi_addr_t dest_addr, uint64_t tag, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_tsend(myep->hep, buf, len, desc, dest_addr, tag, context);
	TRACE_EP(ret, myep, buf, len, dest_addr, tag, 0, context);

	return ret;
}

static ssize_t
trace_tsendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
	     size_t count, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_tsendv(myep->hep, iov, desc, count, dest_addr, tag, context);
	TRACE_EP_IOV(ret, myep, iov, count, dest_addr, tag, 0, context);

	return ret;
}

static ssize_t
trace_tsendmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
	       uint64_t flags)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_tsendmsg(myep->hep, msg, flags);
	TRACE_EP_IOV(ret, myep, msg->msg_iov, msg->iov_count, msg->addr, 0,
		     msg->tag, msg->context);

	return ret;
}

static ssize_t
trace_tinject(struct fid_ep *ep, const void *buf, size_t len,
	      fi_addr_t dest_addr, uint64_t tag)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_tinject(myep->hep, buf, len, dest_addr, tag);
	TRACE_EP(ret, myep, buf, len, dest_addr, tag, 0, NULL);

	return ret;
}

static ssize_t
trace_tsenddata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		uint64_t data, fi_addr_t dest_addr, uint64_t tag,
		void *context)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_tsenddata(myep->hep, buf, len, desc, data,
			   dest_addr, tag, context);
	TRACE_EP(ret, myep, buf, len, dest_addr, tag, data, context);

	return ret;
}

static ssize_t
trace_tinjectdata(struct fid_ep *ep, const void *buf, size_t len,
		  uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	struct hook_ep *myep = container_of(ep, struct hook_ep, ep);
	ssize_t ret;

	ret = fi_tinjectdata(myep->hep, buf, len, data, dest_addr, tag);
	TRACE_EP(ret, myep, buf, len, dest_addr, tag, data, NULL);

	return ret;
}

static struct fi_ops_tagged trace_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = trace_trecv,
	.recvv = trace_trecvv,
	.recvmsg = trace_trecvmsg,
	.send = trace_tsend,
	.sendv = trace_tsendv,
	.sendmsg = trace_tsendmsg,
	.inject = trace_tinject,
	.senddata = trace_tsenddata,
	.injectdata = trace_tinjectdata,
};


static int trace_ep_init(struct fid *fid)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);

	/* ep->atomic = &trace_atomic_ops; */
	ep->msg = &trace_msg_ops;
	ep->rma = &trace_rma_ops;
	ep->tagged = &trace_tagged_ops;
	return 0;
}

static ssize_t trace_cq_read_op(struct fid_cq *cq, void *buf, size_t count)
{
	struct hook_cq *mycq = container_of(cq, struct hook_cq, cq);
	ssize_t ret;

	ret = fi_cq_read(mycq->hcq, buf, count);
	trace_cq(mycq, __func__, __LINE__, ret, buf, 0);
	return ret;
}

static ssize_t
trace_cq_readerr_op(struct fid_cq *cq, struct fi_cq_err_entry *buf, uint64_t flags)
{
	struct hook_cq *mycq = container_of(cq, struct hook_cq, cq);
	ssize_t ret;

	ret = fi_cq_readerr(mycq->hcq, buf, flags);
	return ret;
}

static ssize_t
trace_cq_readfrom_op(struct fid_cq *cq, void *buf, size_t count, fi_addr_t *src_addr)
{
	struct hook_cq *mycq = container_of(cq, struct hook_cq, cq);
	ssize_t ret;

	ret = fi_cq_readfrom(mycq->hcq, buf, count, src_addr);
	trace_cq(mycq, __func__, __LINE__, ret, buf, src_addr ? *src_addr : 0);
	return ret;
}

static ssize_t
trace_cq_sread_op(struct fid_cq *cq, void *buf, size_t count,
	      const void *cond, int timeout)
{
	struct hook_cq *mycq = container_of(cq, struct hook_cq, cq);
	ssize_t ret;

	ret = fi_cq_sread(mycq->hcq, buf, count, cond, timeout);
	trace_cq(mycq, __func__, __LINE__, ret, buf, 0);
	return ret;
}

static ssize_t
trace_cq_sreadfrom_op(struct fid_cq *cq, void *buf, size_t count,
		  fi_addr_t *src_addr, const void *cond, int timeout)
{
	struct hook_cq *mycq = container_of(cq, struct hook_cq, cq);
	ssize_t ret;

	ret = fi_cq_sreadfrom(mycq->hcq, buf, count, src_addr, cond, timeout);
	trace_cq(mycq, __func__, __LINE__, ret, buf, src_addr ? *src_addr : 0);
	return ret;
}

static int trace_cq_signal_op(struct fid_cq *cq)
{
	struct hook_cq *mycq = container_of(cq, struct hook_cq, cq);
	int ret;

	ret = fi_cq_signal(mycq->hcq);
	return ret;
}

struct fi_ops_cq trace_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = trace_cq_read_op,
	.readfrom = trace_cq_readfrom_op,
	.readerr = trace_cq_readerr_op,
	.sread = trace_cq_sread_op,
	.sreadfrom = trace_cq_sreadfrom_op,
	.signal = trace_cq_signal_op,
	.strerror = hook_cq_strerror,
};


static int trace_cq_init(struct fid *fid)
{
	struct fid_cq *cq = container_of(fid, struct fid_cq, fid);

	cq->ops = &trace_cq_ops;
	return 0;
}


static struct fi_ops trace_fabric_fid_ops = {
	.size = sizeof(struct fi_ops),
	.close = hook_close,
	.bind = hook_bind,
	.control = hook_control,
	.ops_open = hook_ops_open,
};


struct hook_prov_ctx hook_trace_ctx;

static int hook_trace_fabric(struct fi_fabric_attr *attr,
			     struct fid_fabric **fabric, void *context)
{
	struct fi_provider *hprov = context;
	struct hook_fabric *fab;

	FI_TRACE(hprov, FI_LOG_FABRIC, "Installing trace hook\n");
	fab = calloc(1, sizeof *fab);
	if (!fab)
		return -FI_ENOMEM;

	hook_fabric_init(fab, HOOK_TRACE, attr->fabric, hprov,
			 &trace_fabric_fid_ops, &hook_trace_ctx);
	*fabric = &fab->fabric;
	return 0;
}

struct hook_prov_ctx hook_trace_ctx = {
	.prov = {
		.version = OFI_VERSION_DEF_PROV,
		/* We're a pass-through provider, so the fi_version is always the latest */
		.fi_version = OFI_VERSION_LATEST,
		.name = "ofi_hook_trace",
		.getinfo = NULL,
		.fabric = hook_trace_fabric,
		.cleanup = NULL,
	},
};

HOOK_TRACE_INI
{
	hook_trace_ctx.ini_fid[FI_CLASS_CQ] = trace_cq_init;
	hook_trace_ctx.ini_fid[FI_CLASS_EP] = trace_ep_init;

	return &hook_trace_ctx.prov;
}
