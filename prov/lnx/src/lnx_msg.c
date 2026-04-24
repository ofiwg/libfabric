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

	return lnx_process_recv(lep, &iov, desc, src_addr, 1, tag, ignore,
				context, lnx_ep_rx_flags(lep), true);
}

static ssize_t lnx_trecvv(struct fid_ep *ep, const struct iovec *iov,
			  void **desc, size_t count, fi_addr_t src_addr,
			  uint64_t tag, uint64_t ignore, void *context)
{
	void *mr_desc;
	struct lnx_ep *lep;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;


	if (count == 0) {
		mr_desc = NULL;
	} else if (iov && count >= 1 &&
		   count <= lep->le_domain->ld_iov_limit) {
		mr_desc = desc ? desc[0] : NULL;
	} else {
		FI_WARN(&lnx_prov, FI_LOG_CORE, "Invalid IOV\n");
		return -FI_EINVAL;
	}

	return lnx_process_recv(lep, iov, mr_desc, src_addr, count, tag, ignore,
				context, lnx_ep_rx_flags(lep), true);
}

static ssize_t lnx_trecvmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			    uint64_t flags)
{
	void *mr_desc;
	struct lnx_ep *lep;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	if (msg->iov_count == 0) {
		mr_desc = NULL;
	} else if (msg->msg_iov && msg->iov_count >= 1 &&
		   msg->iov_count <= lep->le_domain->ld_iov_limit) {
		mr_desc = msg->desc ? msg->desc[0] : NULL;
	} else {
		FI_WARN(&lnx_prov, FI_LOG_CORE, "Invalid IOV\n");
		return -FI_EINVAL;
	}

	return lnx_process_recv(lep, msg->msg_iov, mr_desc, msg->addr,
				msg->iov_count, msg->tag, msg->ignore,
				msg->context, flags | lep->le_ep.rx_msg_flags,
				true);
}

static ssize_t lnx_tsend(struct fid_ep *ep, const void *buf, size_t len,
			 void *desc, fi_addr_t dest_addr, uint64_t tag,
			 void *context)
{
	int rc;
	struct lnx_ep *lep;
	void *core_desc = NULL;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, dest_addr, &cep,
						   &core_addr);
	if (rc)
		return rc;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %" PRIx64 " tag %" PRIx64 " buf %p len %zu\n",
	       core_addr, tag, buf, len);

	if (desc) {
		rc = lnx_mr_regattr_core(cep->cep_domain, desc, &core_desc);
		if (rc)
			return rc;
	}

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
	struct lnx_ep *lep;
	void *core_desc = NULL;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, dest_addr, &cep,
						   &core_addr);
	if (rc)
		return rc;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %" PRIx64 " tag %" PRIx64 " buf %p len %zu\n",
	       core_addr, tag, iov->iov_base, iov->iov_len);

	if (desc && *desc) {
		rc = lnx_mr_regattr_core(cep->cep_domain, *desc, &core_desc);
		if (rc)
			return rc;
	}

	rc = fi_tsendv(cep->cep_ep, iov, (core_desc) ? &core_desc : NULL,
		       count, core_addr, tag, context);

	if (!rc)
		cep->cep_t_stats.st_num_tsendv++;

	return rc;
}

static ssize_t lnx_tsendmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			    uint64_t flags)
{
	int rc;
	struct lnx_ep *lep;
	void *core_desc = NULL;
	struct lnx_core_ep *cep;
	struct fi_msg_tagged core_msg;

	memcpy(&core_msg, msg, sizeof(*msg));

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, core_msg.addr, &cep,
						   &core_msg.addr);
	if (rc)
		return rc;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %" PRIx64 " tag %" PRIx64 "\n",
	       core_msg.addr, core_msg.tag);

	if (core_msg.desc && *core_msg.desc) {
		rc = lnx_mr_regattr_core(cep->cep_domain, *core_msg.desc,
					 &core_desc);
		if (rc)
			return rc;
		core_msg.desc = &core_desc;
	}

	rc = fi_tsendmsg(cep->cep_ep, &core_msg, flags);

	if (!rc)
		cep->cep_t_stats.st_num_tsendmsg++;

	return rc;
}

static ssize_t lnx_tinject(struct fid_ep *ep, const void *buf, size_t len,
			   fi_addr_t dest_addr, uint64_t tag)
{
	int rc;
	struct lnx_ep *lep;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, dest_addr, &cep,
						   &core_addr);
	if (rc)
		return rc;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %" PRIx64 " tag %" PRIx64 " buf %p len %zu\n",
	       core_addr, tag, buf, len);

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
	struct lnx_ep *lep;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;
	void *core_desc = desc;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, dest_addr, &cep,
						   &core_addr);
	if (rc)
		return rc;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %" PRIx64 " tag %" PRIx64 " buf %p len %zu\n",
	       core_addr, tag, buf, len);

	if (desc) {
		rc = lnx_mr_regattr_core(cep->cep_domain, desc, &core_desc);
		if (rc)
			return rc;
	}

	rc = fi_tsenddata(cep->cep_ep, buf, len, core_desc,
			  data, core_addr, tag, context);

	if (!rc)
		cep->cep_t_stats.st_num_tsenddata++;

	return rc;
}

static ssize_t lnx_tinjectdata(struct fid_ep *ep, const void *buf, size_t len,
			       uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	int rc;
	struct lnx_ep *lep;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, dest_addr, &cep,
						   &core_addr);
	if (rc)
		return rc;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %" PRIx64 " tag %" PRIx64 " buf %p len %zu\n",
	       core_addr, tag, buf, len);

	rc = fi_tinjectdata(cep->cep_ep, buf, len, data, core_addr, tag);

	if (!rc)
		cep->cep_t_stats.st_num_tinjectdata++;

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
	.recv = fi_no_msg_recv,
	.recvv = fi_no_msg_recvv,
	.recvmsg = fi_no_msg_recvmsg,
	.send = fi_no_msg_send,
	.sendv = fi_no_msg_sendv,
	.sendmsg = fi_no_msg_sendmsg,
	.inject = fi_no_msg_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};
