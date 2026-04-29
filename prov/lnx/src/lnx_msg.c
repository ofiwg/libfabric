/*
 * Copyright (c) 2022 ORNL. All rights reserved.
 * Copyright (c) Intel Corporation. All rights reserved.
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

#include "lnx.h"

static ssize_t lnx_trecv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			 fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
			 void *context)
{
	struct lnx_ep *lep;
	const struct iovec iov = {.iov_base = buf, .iov_len = len};

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	return lnx_process_recv(lep, &iov, &desc, src_addr, 1, tag, ignore,
				context, lnx_ep_rx_flags(lep), true);
}

static ssize_t lnx_trecvv(struct fid_ep *ep, const struct iovec *iov,
			  void **desc, size_t count, fi_addr_t src_addr,
			  uint64_t tag, uint64_t ignore, void *context)
{
	struct lnx_ep *lep;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	return lnx_process_recv(lep, iov, desc, src_addr, count, tag, ignore,
				context, lnx_ep_rx_flags(lep), true);
}

static ssize_t lnx_trecvmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			    uint64_t flags)
{
	struct lnx_ep *lep;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	return lnx_process_recv(lep, msg->msg_iov, msg->desc, msg->addr,
				msg->iov_count, msg->tag, msg->ignore,
				msg->context, flags | lep->le_ep.rx_msg_flags,
				true);
}

static int lnx_send_lookup(struct fid_ep *ep, fi_addr_t dest_addr, void **desc,
			   size_t count, void **core_desc, fi_addr_t *core_addr,
			   struct lnx_core_ep **cep)
{
	struct lnx_ep *lep;
	int rc;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, dest_addr, cep,
						   core_addr);
	if (rc)
		return rc;

	FI_DBG(&lnx_prov, FI_LOG_CORE, "sending to %" PRIx64 "\n", *core_addr);
	if (desc) {
		rc = lnx_mr_regattr_core((*cep)->cep_domain, desc, count,
					 core_desc);
		if (rc)
			return rc;
	}
	return 0;
}

static ssize_t lnx_tsend(struct fid_ep *ep, const void *buf, size_t len,
			 void *desc, fi_addr_t dest_addr, uint64_t tag,
			 void *context)
{
	int rc;
	void *core_desc = NULL;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;

	rc = lnx_send_lookup(ep, dest_addr, &desc, 1, &core_desc, &core_addr,
			     &cep);
	if (rc)
		return rc;

	rc = fi_tsend(cep->cep_ep, buf, len, core_desc, core_addr, tag,
		      context);
	if (!rc)
		cep->cep_t_stats.st_num_tsend++;

	return rc;
}

static ssize_t lnx_tsendv(struct fid_ep *ep, const struct iovec *iov,
			  void **desc, size_t count, fi_addr_t dest_addr,
			  uint64_t tag, void *context)
{
	int rc;
	void *core_desc[LNX_IOV_LIMIT] = {0};
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;

	rc = lnx_send_lookup(ep, dest_addr, desc, count, core_desc, &core_addr,
			     &cep);
	if (rc)
		return rc;

	rc = fi_tsendv(cep->cep_ep, iov, core_desc, count, core_addr, tag,
		       context);
	if (!rc)
		cep->cep_t_stats.st_num_tsendv++;

	return rc;
}

static ssize_t lnx_tsendmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			    uint64_t flags)
{
	int rc;
	void *core_desc[LNX_IOV_LIMIT] = {0};
	struct lnx_core_ep *cep;
	struct fi_msg_tagged core_msg;

	memcpy(&core_msg, msg, sizeof(*msg));

	rc = lnx_send_lookup(ep, msg->addr, msg->desc, msg->iov_count,
			     core_desc, &core_msg.addr, &cep);
	if (rc)
		return rc;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %" PRIx64 " tag %" PRIx64 "\n",
	       core_msg.addr, core_msg.tag);

	core_msg.desc = core_desc;

	rc = fi_tsendmsg(cep->cep_ep, &core_msg, flags);
	if (!rc)
		cep->cep_t_stats.st_num_tsendmsg++;

	return rc;
}

static ssize_t lnx_tinject(struct fid_ep *ep, const void *buf, size_t len,
			   fi_addr_t dest_addr, uint64_t tag)
{
	int rc;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;

	rc = lnx_send_lookup(ep, dest_addr, NULL, 0, NULL, &core_addr, &cep);
	if (rc)
		return rc;

	rc = fi_tinject(cep->cep_ep, buf, len, core_addr, tag);
	if (!rc)
		cep->cep_t_stats.st_num_tinject++;

	return rc;
}

static ssize_t lnx_tsenddata(struct fid_ep *ep, const void *buf, size_t len,
			     void *desc, uint64_t data, fi_addr_t dest_addr,
			     uint64_t tag, void *context)
{
	int rc;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;
	void *core_desc = NULL;

	rc = lnx_send_lookup(ep, dest_addr, &desc, 1, &core_desc, &core_addr,
			     &cep);
	if (rc)
		return rc;

	rc = fi_tsenddata(cep->cep_ep, buf, len, core_desc, data, core_addr,
			  tag, context);
	if (!rc)
		cep->cep_t_stats.st_num_tsenddata++;

	return rc;
}

static ssize_t lnx_tinjectdata(struct fid_ep *ep, const void *buf, size_t len,
			       uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	int rc;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;

	rc = lnx_send_lookup(ep, dest_addr, NULL, 0, NULL, &core_addr, &cep);
	if (rc)
		return rc;

	rc = fi_tinjectdata(cep->cep_ep, buf, len, data, core_addr, tag);
	if (!rc)
		cep->cep_t_stats.st_num_tinjectdata++;

	return rc;
}

static ssize_t lnx_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			fi_addr_t src_addr, void *context)
{
	struct lnx_ep *lep;
	const struct iovec iov = {.iov_base = buf, .iov_len = len};

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	return lnx_process_recv(lep, &iov, &desc, src_addr, 1, 0, 0, context,
				lnx_ep_rx_flags(lep), false);
}

static ssize_t lnx_recvv(struct fid_ep *ep, const struct iovec *iov,
			 void **desc, size_t count, fi_addr_t src_addr,
			 void *context)
{
	struct lnx_ep *lep;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	return lnx_process_recv(lep, iov, desc, src_addr, count, 0, 0,
				context, lnx_ep_rx_flags(lep), false);
}

static ssize_t lnx_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
			   uint64_t flags)
{
	struct lnx_ep *lep;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	return lnx_process_recv(lep, msg->msg_iov, msg->desc, msg->addr,
				msg->iov_count, 0, 0, msg->context,
				flags | lep->le_ep.rx_msg_flags, false);
}

static ssize_t lnx_send(struct fid_ep *ep, const void *buf, size_t len,
			void *desc, fi_addr_t dest_addr, void *context)
{
	int rc;
	void *core_desc = NULL;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;

	rc = lnx_send_lookup(ep, dest_addr, &desc, 1, &core_desc, &core_addr,
			     &cep);
	if (rc)
		return rc;

	rc = fi_send(cep->cep_ep, buf, len, core_desc, core_addr, context);
	if (!rc)
		cep->cep_t_stats.st_num_send++;

	return rc;
}

static ssize_t lnx_sendv(struct fid_ep *ep, const struct iovec *iov,
			 void **desc, size_t count, fi_addr_t dest_addr,
			 void *context)
{
	int rc;
	void *core_desc[LNX_IOV_LIMIT] = {0};
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;

	rc = lnx_send_lookup(ep, dest_addr, desc, count, core_desc, &core_addr,
			     &cep);
	if (rc)
		return rc;

	rc = fi_sendv(cep->cep_ep, iov, core_desc, count, core_addr, context);
	if (!rc)
		cep->cep_t_stats.st_num_sendv++;

	return rc;
}

static ssize_t lnx_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			   uint64_t flags)
{
	int rc;
	void *core_desc[LNX_IOV_LIMIT] = {0};
	struct lnx_core_ep *cep;
	struct fi_msg core_msg;

	memcpy(&core_msg, msg, sizeof(*msg));

	rc = lnx_send_lookup(ep, msg->addr, msg->desc, msg->iov_count,
			     core_desc, &core_msg.addr, &cep);
	if (rc)
		return rc;

	core_msg.desc = core_desc;

	rc = fi_sendmsg(cep->cep_ep, &core_msg, flags);
	if (!rc)
		cep->cep_t_stats.st_num_sendmsg++;

	return rc;
}

static ssize_t lnx_inject(struct fid_ep *ep, const void *buf, size_t len,
			  fi_addr_t dest_addr)
{
	int rc;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;

	rc = lnx_send_lookup(ep, dest_addr, NULL, 0, NULL, &core_addr, &cep);
	if (rc)
		return rc;

	rc = fi_inject(cep->cep_ep, buf, len, core_addr);
	if (!rc)
		cep->cep_t_stats.st_num_inject++;

	return rc;
}

static ssize_t lnx_senddata(struct fid_ep *ep, const void *buf, size_t len,
			    void *desc, uint64_t data, fi_addr_t dest_addr,
			    void *context)
{
	int rc;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;
	void *core_desc = NULL;

	rc = lnx_send_lookup(ep, dest_addr, &desc, 1, &core_desc, &core_addr,
			     &cep);
	if (rc)
		return rc;

	rc = fi_senddata(cep->cep_ep, buf, len, core_desc, data, core_addr,
			 context);
	if (!rc)
		cep->cep_t_stats.st_num_senddata++;

	return rc;
}

static ssize_t lnx_injectdata(struct fid_ep *ep, const void *buf, size_t len,
			      uint64_t data, fi_addr_t dest_addr)
{
	int rc;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;

	rc = lnx_send_lookup(ep, dest_addr, NULL, 0, NULL, &core_addr, &cep);
	if (rc)
		return rc;

	rc = fi_injectdata(cep->cep_ep, buf, len, data, core_addr);
	if (!rc)
		cep->cep_t_stats.st_num_injectdata++;

	return rc;
}

struct fi_ops_tagged lnx_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = lnx_trecv,
	.recvv = lnx_trecvv,
	.recvmsg = lnx_trecvmsg,
	.send = lnx_tsend,
	.sendv = lnx_tsendv,
	.sendmsg = lnx_tsendmsg,
	.inject = lnx_tinject,
	.senddata = lnx_tsenddata,
	.injectdata = lnx_tinjectdata,
};

struct fi_ops_msg lnx_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = lnx_recv,
	.recvv = lnx_recvv,
	.recvmsg = lnx_recvmsg,
	.send = lnx_send,
	.sendv = lnx_sendv,
	.sendmsg = lnx_sendmsg,
	.inject = lnx_inject,
	.senddata = lnx_senddata,
	.injectdata = lnx_injectdata,
};
