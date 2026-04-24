/*
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

static ssize_t lnx_read(struct fid_ep *ep_fid, void *buf, size_t len,
			void *desc, fi_addr_t src_addr, uint64_t addr,
			uint64_t key, void *context)
{
	int rc;
	struct lnx_ep *lep;
	void *core_desc = NULL;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;
	struct lnx_mr_key *mr_key = (struct lnx_mr_key *) key;

	lep = container_of(ep_fid, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, src_addr, &cep,
						   &core_addr);
	if (rc)
		return rc;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "reading from %" PRIx64 " buf %p len %zu\n",
	       core_addr, buf, len);

	if (desc) {
		rc = lnx_mr_regattr_core(cep->cep_domain, &desc, 1, &core_desc);
		if (rc)
			return rc;
	}

	rc = fi_read(cep->cep_ep, buf, len, core_desc, core_addr, addr,
		     mr_key->prov_keys[cep->cep_domain->idx], context);
	if (!rc)
		cep->cep_t_stats.st_num_read++;

	return rc;
}

static ssize_t lnx_readv(struct fid_ep *ep_fid, const struct iovec *iov,
			 void **desc, size_t count, fi_addr_t src_addr,
			 uint64_t addr, uint64_t key, void *context)
{
	int rc;
	struct lnx_ep *lep;
	void *core_desc[LNX_IOV_LIMIT] = {0};
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;
	struct lnx_mr_key *mr_key = (struct lnx_mr_key *) key;

	lep = container_of(ep_fid, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, src_addr, &cep,
						   &core_addr);

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "reading from %" PRIx64 " buf %p len %zu\n",
	       core_addr, iov[0].iov_base, iov[0].iov_len);

	if (desc) {
		rc = lnx_mr_regattr_core(cep->cep_domain, desc, count,
					 core_desc);
		if (rc)
			return rc;
	}

	rc = fi_readv(cep->cep_ep, iov, core_desc, count, core_addr, addr,
		      mr_key->prov_keys[cep->cep_domain->idx], context);
	if (!rc)
		cep->cep_t_stats.st_num_readv++;

	return rc;}

static ssize_t lnx_readmsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
			   uint64_t flags)
{
	int rc;
	struct lnx_ep *lep;
	void *core_desc[LNX_IOV_LIMIT] = {0};
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;
	struct lnx_mr_key *mr_key;
	struct fi_msg_rma core_msg = *msg;
	struct fi_rma_iov core_rma_iov[LNX_IOV_LIMIT] = {0};
	int i;

	lep = container_of(ep_fid, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, msg->addr, &cep,
						   &core_addr);

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "reading from %" PRIx64 " buf %p len %zu\n",
	       core_addr, msg->msg_iov[0].iov_base, msg->msg_iov[0].iov_len);

	if (msg->desc) {
		rc = lnx_mr_regattr_core(cep->cep_domain, msg->desc,
					 msg->iov_count, core_desc);
		if (rc)
			return rc;
	}

	for (i = 0; i < msg->rma_iov_count; i++) {
		mr_key = (struct lnx_mr_key *) msg->rma_iov[i].key;
		core_rma_iov[i] = msg->rma_iov[i];
		core_rma_iov[i].key = mr_key->prov_keys[cep->cep_domain->idx];
	}
	core_msg.rma_iov = core_rma_iov;

	rc = fi_readmsg(cep->cep_ep, &core_msg, flags);
	if (!rc)
		cep->cep_t_stats.st_num_readmsg++;

	return rc;}

static ssize_t lnx_write(struct fid_ep *ep_fid, const void *buf, size_t len,
			 void *desc, fi_addr_t dest_addr, uint64_t addr,
			 uint64_t key, void *context)
{
	int rc;
	struct lnx_ep *lep;
	void *core_desc = NULL;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;
	struct lnx_mr_key *mr_key = (struct lnx_mr_key *) key;

	lep = container_of(ep_fid, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, dest_addr, &cep,
						   &core_addr);

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "writing to %" PRIx64 " buf %p len %zu\n",
	       core_addr, buf, len);

	if (desc) {
		rc = lnx_mr_regattr_core(cep->cep_domain, &desc, 1, &core_desc);
		if (rc)
			return rc;
	}

	rc = fi_write(cep->cep_ep, buf, len, core_desc, core_addr, addr,
		      mr_key->prov_keys[cep->cep_domain->idx], context);
	if (!rc)
		cep->cep_t_stats.st_num_write++;

	return rc;
}

static ssize_t lnx_writev(struct fid_ep *ep_fid, const struct iovec *iov,
			  void **desc, size_t count, fi_addr_t dest_addr,
			  uint64_t addr, uint64_t key, void *context)
{
	int rc;
	struct lnx_ep *lep;
	void *core_desc[LNX_IOV_LIMIT] = {0};
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;
	struct lnx_mr_key *mr_key = (struct lnx_mr_key *) key;

	lep = container_of(ep_fid, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, dest_addr, &cep,
						   &core_addr);

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "writing to %" PRIx64 " buf %p len %zu\n",
	       core_addr, iov[0].iov_base, iov[0].iov_len);

	if (desc) {
		rc = lnx_mr_regattr_core(cep->cep_domain, desc, count,
					 core_desc);
		if (rc)
			return rc;
	}

	rc = fi_writev(cep->cep_ep, iov, core_desc, count, core_addr, addr,
		       mr_key->prov_keys[cep->cep_domain->idx], context);
	if (!rc)
		cep->cep_t_stats.st_num_writev++;

	return rc;
}


static ssize_t lnx_writemsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
			    uint64_t flags)
{
	int rc;
	struct lnx_ep *lep;
	void *core_desc[LNX_IOV_LIMIT] = {0};
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;
	struct lnx_mr_key *mr_key;
	struct fi_msg_rma core_msg = *msg;
	struct fi_rma_iov core_rma_iov[LNX_IOV_LIMIT] = {0};
	int i;

	lep = container_of(ep_fid, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, msg->addr, &cep,
						   &core_addr);

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "writing to %" PRIx64 " buf %p len %zu\n",
	       core_addr, msg->msg_iov[0].iov_base, msg->msg_iov[0].iov_len);

	if (msg->desc) {
		rc = lnx_mr_regattr_core(cep->cep_domain, msg->desc,
					 msg->iov_count, core_desc);
		if (rc)
			return rc;
	}
	for (i = 0; i < msg->rma_iov_count; i++) {
		mr_key = (struct lnx_mr_key *) msg->rma_iov[i].key;
		core_rma_iov[i] = msg->rma_iov[i];
		core_rma_iov[i].key = mr_key->prov_keys[cep->cep_domain->idx];
	}
	core_msg.rma_iov = core_rma_iov;

	rc = fi_writemsg(cep->cep_ep, &core_msg, flags);
	if (!rc)
		cep->cep_t_stats.st_num_writemsg++;

	return rc;
}

static ssize_t lnx_writedata(struct fid_ep *ep_fid, const void *buf, size_t len,
			     void *desc, uint64_t data, fi_addr_t dest_addr,
			     uint64_t addr, uint64_t key, void *context)
{
	int rc;
	struct lnx_ep *lep;
	void *core_desc = NULL;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;
	struct lnx_mr_key *mr_key = (struct lnx_mr_key *) key;

	lep = container_of(ep_fid, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, dest_addr, &cep,
						   &core_addr);

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "writing to %" PRIx64 " buf %p len %zu\n",
	       core_addr, buf, len);

	if (desc) {
		rc = lnx_mr_regattr_core(cep->cep_domain, &desc, 1, &core_desc);
		if (rc)
			return rc;
	}

	rc = fi_writedata(cep->cep_ep, buf, len, core_desc, data, core_addr,
			  addr, mr_key->prov_keys[cep->cep_domain->idx],
			  context);
	if (!rc)
		cep->cep_t_stats.st_num_writedata++;

	return rc;}

static ssize_t lnx_rma_inject(struct fid_ep *ep_fid, const void *buf,
			      size_t len, fi_addr_t dest_addr, uint64_t addr,
			      uint64_t key)
{
	int rc;
	struct lnx_ep *lep;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;
	struct lnx_mr_key *mr_key = (struct lnx_mr_key *) key;

	lep = container_of(ep_fid, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, dest_addr, &cep,
						   &core_addr);

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "writing to %" PRIx64 " buf %p len %zu\n",
	       core_addr, buf, len);

	rc = fi_inject_write(cep->cep_ep, buf, len, core_addr, addr,
			     mr_key->prov_keys[cep->cep_domain->idx]);
	if (!rc)
		cep->cep_t_stats.st_num_inject_write++;

	return rc;
}

static ssize_t lnx_inject_writedata(struct fid_ep *ep_fid, const void *buf,
				    size_t len, uint64_t data,
				    fi_addr_t dest_addr, uint64_t addr,
				    uint64_t key)
{
	int rc;
	struct lnx_ep *lep;
	struct lnx_core_ep *cep;
	fi_addr_t core_addr;
	struct lnx_mr_key *mr_key = (struct lnx_mr_key *) key;

	lep = container_of(ep_fid, struct lnx_ep, le_ep.ep_fid.fid);
	if (!lep)
		return -FI_ENOSYS;

	rc = lnx_select_send_endpoints[lep->le_mr](lep, dest_addr, &cep,
						   &core_addr);

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "writing to %" PRIx64 " buf %p len %zu\n",
	       core_addr, buf, len);

	rc = fi_inject_writedata(cep->cep_ep, buf, len, data, core_addr, addr,
				 mr_key->prov_keys[cep->cep_domain->idx]);
	if (!rc)
		cep->cep_t_stats.st_num_inject_writedata++;

	return rc;
}

struct fi_ops_rma lnx_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = lnx_read,
	.readv = lnx_readv,
	.readmsg = lnx_readmsg,
	.write = lnx_write,
	.writev = lnx_writev,
	.writemsg = lnx_writemsg,
	.inject = lnx_rma_inject,
	.writedata = lnx_writedata,
	.injectdata = lnx_inject_writedata,
};
