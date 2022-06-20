/*
 * Copyright (c) 2022 Intel Corporation. All rights reserved.
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

#include <sys/types.h>
#include <errno.h>

#include <ofi_prov.h>
#include "tcp2.h"


static ssize_t
tcp2_rdm_recv(struct fid_ep *ep_fid, void *buf, size_t len,
	      void *desc, fi_addr_t src_addr, void *context)
{
	struct tcp2_rdm *rdm;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	return fi_recv(&rdm->srx->rx_fid, buf, len, desc, src_addr, context);
}

static ssize_t
tcp2_rdm_recvv(struct fid_ep *ep_fid, const struct iovec *iov,
	  void **desc, size_t count, fi_addr_t src_addr, void *context)
{
	struct tcp2_rdm *rdm;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	return fi_recvv(&rdm->srx->rx_fid, iov, desc, count, src_addr, context);
}

static ssize_t
tcp2_rdm_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		 uint64_t flags)
{
	struct tcp2_rdm *rdm;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	return fi_recvmsg(&rdm->srx->rx_fid, msg, flags);
}

static ssize_t
tcp2_rdm_send(struct fid_ep *ep_fid, const void *buf, size_t len,
	      void *desc, fi_addr_t dest_addr, void *context)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_send(&conn->ep->util_ep.ep_fid, buf, len, desc, 0, context);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_sendv(struct fid_ep *ep_fid, const struct iovec *iov,
	       void **desc, size_t count, fi_addr_t dest_addr, void *context)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_sendv(&conn->ep->util_ep.ep_fid, iov, desc, count, 0, context);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		 uint64_t flags)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, msg->addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_sendmsg(&conn->ep->util_ep.ep_fid, msg, flags);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_inject(struct fid_ep *ep_fid, const void *buf,
		size_t len, fi_addr_t dest_addr)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_inject(&conn->ep->util_ep.ep_fid, buf, len, 0);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_senddata(struct fid_ep *ep_fid, const void *buf, size_t len,
		  void *desc, uint64_t data, fi_addr_t dest_addr, void *context)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_senddata(&conn->ep->util_ep.ep_fid, buf, len, desc, data, 0,
			  context);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_injectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
		    uint64_t data, fi_addr_t dest_addr)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_injectdata(&conn->ep->util_ep.ep_fid, buf, len, data, 0);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static struct fi_ops_msg tcp2_rdm_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = tcp2_rdm_recv,
	.recvv = tcp2_rdm_recvv,
	.recvmsg = tcp2_rdm_recvmsg,
	.send = tcp2_rdm_send,
	.sendv = tcp2_rdm_sendv,
	.sendmsg = tcp2_rdm_sendmsg,
	.inject = tcp2_rdm_inject,
	.senddata = tcp2_rdm_senddata,
	.injectdata = tcp2_rdm_injectdata,
};

static ssize_t
tcp2_rdm_trecv(struct fid_ep *ep_fid, void *buf, size_t len,
	       void *desc, fi_addr_t src_addr,
	       uint64_t tag, uint64_t ignore, void *context)
{
	struct tcp2_rdm *rdm;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	return fi_trecv(&rdm->srx->rx_fid, buf, len, desc, src_addr,
			tag, ignore, context);
}

static ssize_t
tcp2_rdm_trecvv(struct fid_ep *ep_fid, const struct iovec *iov,
		void **desc, size_t count, fi_addr_t src_addr,
		uint64_t tag, uint64_t ignore, void *context)
{
	struct tcp2_rdm *rdm;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	return fi_trecvv(&rdm->srx->rx_fid, iov, desc, count, src_addr,
			 tag, ignore, context);
}

static ssize_t
tcp2_rdm_trecvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
		  uint64_t flags)
{
	struct tcp2_rdm *rdm;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	return fi_trecvmsg(&rdm->srx->rx_fid, msg, flags);
}

static ssize_t
tcp2_rdm_tsend(struct fid_ep *ep_fid, const void *buf, size_t len,
	       void *desc, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_tsend(&conn->ep->util_ep.ep_fid, buf, len, desc, 0, tag,
		       context);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_tsendv(struct fid_ep *ep_fid, const struct iovec *iov,
		void **desc, size_t count, fi_addr_t dest_addr,
		uint64_t tag, void *context)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_tsendv(&conn->ep->util_ep.ep_fid, iov, desc, count, 0, tag,
			context);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_tsendmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
		  uint64_t flags)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, msg->addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_tsendmsg(&conn->ep->util_ep.ep_fid, msg, flags);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_tinject(struct fid_ep *ep_fid, const void *buf,
		 size_t len, fi_addr_t dest_addr, uint64_t tag)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_tinject(&conn->ep->util_ep.ep_fid, buf, len, 0, tag);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_tsenddata(struct fid_ep *ep_fid, const void *buf, size_t len,
		   void *desc, uint64_t data, fi_addr_t dest_addr,
		   uint64_t tag, void *context)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_tsenddata(&conn->ep->util_ep.ep_fid, buf, len, desc, data, 0,
			   tag, context);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_tinjectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
		    uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_tinjectdata(&conn->ep->util_ep.ep_fid, buf, len, data, 0, tag);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static struct fi_ops_tagged tcp2_rdm_tagged_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = tcp2_rdm_trecv,
	.recvv = tcp2_rdm_trecvv,
	.recvmsg = tcp2_rdm_trecvmsg,
	.send = tcp2_rdm_tsend,
	.sendv = tcp2_rdm_tsendv,
	.sendmsg = tcp2_rdm_tsendmsg,
	.inject = tcp2_rdm_tinject,
	.senddata = tcp2_rdm_tsenddata,
	.injectdata = tcp2_rdm_tinjectdata,
};

static ssize_t
tcp2_rdm_read(struct fid_ep *ep_fid, void *buf, size_t len,
	      void *desc, fi_addr_t src_addr, uint64_t addr,
	      uint64_t key, void *context)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, src_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_read(&conn->ep->util_ep.ep_fid, buf, len, desc, src_addr, addr,
		      key, context);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_readv(struct fid_ep *ep_fid, const struct iovec *iov,
	       void **desc, size_t count, fi_addr_t src_addr,
	       uint64_t addr, uint64_t key, void *context)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, src_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_readv(&conn->ep->util_ep.ep_fid, iov, desc, count, src_addr, addr,
		       key, context);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_readmsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
		 uint64_t flags)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, msg->addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_readmsg(&conn->ep->util_ep.ep_fid, msg, flags);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_write(struct fid_ep *ep_fid, const void *buf,
	       size_t len, void *desc, fi_addr_t dest_addr,
	       uint64_t addr, uint64_t key, void *context)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_write(&conn->ep->util_ep.ep_fid, buf, len, desc, dest_addr,
		       addr, key, context);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_writev(struct fid_ep *ep_fid, const struct iovec *iov,
		void **desc, size_t count, fi_addr_t dest_addr,
		uint64_t addr, uint64_t key, void *context)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_writev(&conn->ep->util_ep.ep_fid, iov, desc, count, dest_addr,
			addr, key, context);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_writemsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
		  uint64_t flags)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, msg->addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_writemsg(&conn->ep->util_ep.ep_fid, msg, flags);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_inject_write(struct fid_ep *ep_fid, const void *buf,
		      size_t len, fi_addr_t dest_addr,
		      uint64_t addr, uint64_t key)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_inject_write(&conn->ep->util_ep.ep_fid, buf, len, dest_addr,
			      addr, key);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_writedata(struct fid_ep *ep_fid, const void *buf,
		   size_t len, void *desc, uint64_t data,
		   fi_addr_t dest_addr, uint64_t addr,
		   uint64_t key, void *context)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_writedata(&conn->ep->util_ep.ep_fid, buf, len, desc, data,
			   dest_addr, addr, key, context);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

static ssize_t
tcp2_rdm_inject_writedata(struct fid_ep *ep_fid, const void *buf,
			  size_t len, uint64_t data, fi_addr_t dest_addr,
			  uint64_t addr, uint64_t key)
{
	struct tcp2_rdm *rdm;
	struct tcp2_conn *conn;
	ssize_t ret;

	rdm = container_of(ep_fid, struct tcp2_rdm, util_ep.ep_fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = tcp2_get_conn(rdm, dest_addr, &conn);
	if (ret)
		goto unlock;

	ret = fi_inject_writedata(&conn->ep->util_ep.ep_fid, buf, len, data,
				  dest_addr, addr, key);
unlock:
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	return ret;
}

struct fi_ops_rma tcp2_rdm_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = tcp2_rdm_read,
	.readv = tcp2_rdm_readv,
	.readmsg = tcp2_rdm_readmsg,
	.write = tcp2_rdm_write,
	.writev = tcp2_rdm_writev,
	.writemsg = tcp2_rdm_writemsg,
	.inject = tcp2_rdm_inject_write,
	.writedata = tcp2_rdm_writedata,
	.injectdata = tcp2_rdm_inject_writedata,
};

static int tcp2_rdm_setname(fid_t fid, void *addr, size_t addrlen)
{
	struct tcp2_rdm *rdm;

	rdm = container_of(fid, struct tcp2_rdm, util_ep.ep_fid.fid);
	return fi_setname(&rdm->pep->util_pep.pep_fid.fid, addr, addrlen);
}

static int tcp2_rdm_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct tcp2_rdm *rdm;

	rdm = container_of(fid, struct tcp2_rdm, util_ep.ep_fid.fid);
	return fi_getname(&rdm->pep->util_pep.pep_fid.fid, addr, addrlen);
}

static struct fi_ops_cm tcp2_rdm_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = tcp2_rdm_setname,
	.getname = tcp2_rdm_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

static ssize_t tcp2_rdm_cancel(struct fid *fid, void *context)
{
	struct tcp2_rdm *rdm;

	rdm = container_of(fid, struct tcp2_rdm, util_ep.ep_fid.fid);
	return fi_cancel(&rdm->srx->rx_fid.fid, context);
}

static int tcp2_rdm_getopt(struct fid *fid, int level, int optname,
			   void *optval, size_t *optlen)
{
	return -FI_ENOPROTOOPT;
}

static int tcp2_rdm_setopt(struct fid *fid, int level, int optname,
			   const void *optval, size_t optlen)
{
	return -FI_ENOPROTOOPT;
}

static struct fi_ops_ep tcp2_rdm_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = tcp2_rdm_cancel,
	.getopt = tcp2_rdm_getopt,
	.setopt = tcp2_rdm_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static int tcp2_enable_rdm(struct tcp2_rdm *rdm)
{
	struct tcp2_progress *progress;
	struct fi_info *info;
	size_t len;
	int ret;

	progress = tcp2_rdm2_progress(rdm);
	ofi_genlock_lock(&progress->rdm_lock);

	ret = tcp2_listen(rdm->pep, progress);
	if (ret)
		return ret;

	/* TODO: Move updating the src_addr to pep_listen(). */
	len = sizeof(rdm->addr);
	ret = fi_getname(&rdm->pep->util_pep.pep_fid.fid, &rdm->addr, &len);
	if (ret) {
		TCP2_WARN_ERR(FI_LOG_EP_CTRL, "fi_getname", ret);
		goto unlock;
	}

	/* Update src_addr that will be used for active endpoints.
	 * Zero out the port to avoid address conflicts, as we will
	 * create multiple msg ep's for a single rdm ep.
	 */
	info = rdm->pep->info;
	free(info->src_addr);
	info->src_addr = NULL;
	info->src_addrlen = 0;

	info->src_addr = mem_dup(&rdm->addr, len);
	if (!info->src_addr) {
		ret = -FI_ENOMEM;
		goto unlock;
	}

	info->src_addrlen = len;
	ofi_addr_set_port(info->src_addr, 0);
	dlist_insert_tail(&rdm->progress_entry, &progress->event_list);

unlock:
	ofi_genlock_unlock(&progress->rdm_lock);
	return ret;
}

static int tcp2_rdm_ctrl(struct fid *fid, int command, void *arg)
{
	struct tcp2_rdm *rdm;

	rdm = container_of(fid, struct tcp2_rdm, util_ep.ep_fid.fid);
	switch (command) {
	case FI_ENABLE:
		if (!rdm->util_ep.av)
			return -FI_EOPBADSTATE;

		if (!rdm->util_ep.tx_cq || !rdm->util_ep.rx_cq)
			return -FI_ENOCQ;

		return tcp2_enable_rdm(rdm);
	default:
		return -FI_ENOSYS;
	}
	return 0;
}

static int tcp2_rdm_close(struct fid *fid)
{
	struct tcp2_rdm *rdm;
	int ret;

	rdm = container_of(fid, struct tcp2_rdm, util_ep.ep_fid.fid);
	ofi_genlock_lock(&tcp2_rdm2_progress(rdm)->rdm_lock);
	ret = fi_close(&rdm->pep->util_pep.pep_fid.fid);
	if (ret) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL, \
			"Unable to close passive endpoint\n");
		return ret;
	}

	tcp2_freeall_conns(rdm);
	ofi_genlock_unlock(&tcp2_rdm2_progress(rdm)->rdm_lock);

	ret = fi_close(&rdm->srx->rx_fid.fid);
	if (ret) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL, \
			"Unable to close msg shared ctx\n");
		return ret;
	}

	ofi_endpoint_close(&rdm->util_ep);
	free(rdm);
	return 0;
}

static struct fi_ops tcp2_rdm_fid_ops = {
	.size = sizeof(struct fi_ops),
	.close = tcp2_rdm_close,
	.bind = ofi_ep_fid_bind,
	.control = tcp2_rdm_ctrl,
	.ops_open = fi_no_ops_open,
};

static int tcp2_init_rdm(struct tcp2_rdm *rdm, struct fi_info *info)
{
	struct fi_info *msg_info;
	struct fid_ep *srx;
	struct fid_pep *pep;
	int ret;

	msg_info = fi_dupinfo(info);
	if (!msg_info)
		return -FI_ENOMEM;

	msg_info->ep_attr->type = FI_EP_MSG;
	msg_info->ep_attr->rx_ctx_cnt = FI_SHARED_CONTEXT;

	ret = fi_srx_context(&rdm->util_ep.domain->domain_fid, info->rx_attr,
			     &srx, rdm);
	if (ret) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"Unable to open shared receive context\n");
		goto err1;
	}

	ret = fi_passive_ep(&rdm->util_ep.domain->fabric->fabric_fid, msg_info,
			    &pep, rdm);
	if (ret) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"Unable to open passive ep\n");
		goto err2;
	}

	dlist_init(&rdm->loopback_list);
	rdm->srx = container_of(srx, struct tcp2_srx, rx_fid);
	rdm->pep = container_of(pep, struct tcp2_pep, util_pep);
	return 0;

err2:
	fi_close(&srx->fid);
err1:
	fi_freeinfo(msg_info);
	return ret;
}

int tcp2_rdm_ep(struct fid_domain *domain, struct fi_info *info,
		struct fid_ep **ep_fid, void *context)
{
	struct tcp2_rdm *rdm;
	int ret;

	rdm = calloc(1, sizeof(*rdm));
	if (!rdm)
		return -FI_ENOMEM;

	slist_init(&rdm->event_list);
	dlist_init(&rdm->progress_entry);
	ret = ofi_endpoint_init(domain, &tcp2_util_prov, info, &rdm->util_ep,
				context, NULL);
	if (ret)
		goto err1;

	ret = tcp2_init_rdm(rdm, info);
	if (ret)
		goto err2;

	*ep_fid = &rdm->util_ep.ep_fid;
	(*ep_fid)->fid.ops = &tcp2_rdm_fid_ops;
	(*ep_fid)->ops = &tcp2_rdm_ep_ops;
	(*ep_fid)->cm = &tcp2_rdm_cm_ops;
	(*ep_fid)->msg = &tcp2_rdm_msg_ops;
	(*ep_fid)->rma = &tcp2_rdm_rma_ops;
	(*ep_fid)->tagged = &tcp2_rdm_tagged_ops;

	return 0;

err2:
	ofi_endpoint_close(&rdm->util_ep);
err1:
	free(rdm);
	return ret;
}
