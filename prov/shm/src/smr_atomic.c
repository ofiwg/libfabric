/*
 * Copyright (c) 2013-2018 Intel Corporation. All rights reserved
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

#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>

#include "ofi_iov.h"
#include "smr.h"


static void smr_format_rma_ioc(struct smr_cmd *cmd, const struct fi_rma_ioc *rma_ioc,
			       size_t ioc_count)
{
	cmd->rma.rma_count = ioc_count;
	memcpy(cmd->rma.rma_ioc, rma_ioc, sizeof(*rma_ioc) * ioc_count);
}

static void smr_format_inline_atomic(struct smr_cmd *cmd, fi_addr_t peer_id,
				     const struct iovec *iov, size_t count,
				     const struct iovec *compv,
				     size_t comp_count,  uint32_t op,
				     enum fi_datatype datatype,
				     enum fi_op atomic_op)
{
	size_t comp_size;

	smr_generic_format(cmd, peer_id, op, 0, datatype,
			   atomic_op, 0, 0);
	cmd->msg.hdr.op_src = smr_src_inline;
	switch (op) {
	case ofi_op_atomic:
	case ofi_op_atomic_fetch:
		cmd->msg.hdr.size = ofi_copy_from_iov(cmd->msg.data.msg,
						SMR_MSG_DATA_LEN, iov, count, 0);
		break;
	case ofi_op_atomic_compare:
		cmd->msg.hdr.size = ofi_copy_from_iov(cmd->msg.data.buf,
						SMR_MSG_DATA_LEN, iov, count, 0);
		comp_size = ofi_copy_from_iov(cmd->msg.data.comp,
					      SMR_MSG_DATA_LEN, compv,
					      comp_count, 0);
		if (comp_size != cmd->msg.hdr.size)
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"atomic and compare buffer size mimatch\n");
		break;
	default:
		break;
	}
}

static void smr_format_inject_atomic(struct smr_cmd *cmd, fi_addr_t peer_id,
				     const struct iovec *iov, size_t count,
				     const struct iovec *resultv,
				     size_t result_count,
				     const struct iovec *compv,
				     size_t comp_count,
				     uint32_t op, enum fi_datatype datatype,
				     enum fi_op atomic_op,
				     struct smr_region *smr,
				     struct smr_inject_buf *tx_buf)
{
	size_t comp_size;

	smr_generic_format(cmd, peer_id, op, 0, datatype,
			   atomic_op, 0, 0);
	cmd->msg.hdr.op_src = smr_src_inject;
	cmd->msg.hdr.src_data = (char **) tx_buf - (char **) smr;

	switch (op) {
	case ofi_op_atomic:
	case ofi_op_atomic_fetch:
		if (atomic_op == FI_ATOMIC_READ)
			cmd->msg.hdr.size = ofi_total_iov_len(resultv, result_count);
		else
			cmd->msg.hdr.size = ofi_copy_from_iov(tx_buf->data,
						SMR_INJECT_SIZE, iov, count, 0);
		break;
	case ofi_op_atomic_compare:
		cmd->msg.hdr.size = ofi_copy_from_iov(tx_buf->buf,
						SMR_COMP_INJECT_SIZE, iov, count, 0);
		comp_size = ofi_copy_from_iov(tx_buf->comp, SMR_COMP_INJECT_SIZE,
					      compv, comp_count, 0);
		if (comp_size != cmd->msg.hdr.size)
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"atomic and compare buffer size mimatch\n");
		break;
	default:
		break;
	}
}

static void smr_ioc_to_iov(const struct fi_ioc *ioc, struct iovec *iov,
			   size_t count, size_t size)
{
	int i;
	for (i = 0; i < count; i++) {
		iov[i].iov_base = ioc[i].addr;
		iov[i].iov_len = ioc[i].count * size;
	}
}

static int smr_fetch_result(struct smr_ep *ep, struct smr_region *peer_smr,
			    struct iovec *iov, size_t iov_count,
			    const struct fi_rma_ioc *rma_ioc, size_t rma_count,
			    enum fi_datatype datatype, size_t total_len)
{
	int ret, i;
	struct iovec rma_iov[SMR_IOV_LIMIT];

	for (i = 0; i < rma_count; i++) {
		rma_iov[i].iov_base = (void *) rma_ioc[i].addr;
		rma_iov[i].iov_len = rma_ioc[i].count * ofi_datatype_size(datatype);
	}

	ret = process_vm_readv(peer_smr->pid, iov, iov_count,
			       rma_iov, rma_count, 0);
	if (ret != total_len) {
		if (ret < 0) {
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"CMA write error\n");
			return -errno;
		} else { 
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"partial read occurred\n");
			return -FI_EIO;
		}
	}

	return 0; 
}

static void smr_post_fetch_resp(struct smr_ep *ep, struct smr_cmd *cmd,
				const struct iovec *result_iov, size_t count)
{
	struct smr_cmd *pend;
	struct smr_resp *resp;

	assert(!ofi_cirque_isfull(smr_resp_queue(ep->region)));
	resp = ofi_cirque_tail(smr_resp_queue(ep->region));

	cmd->msg.hdr.data = (uint64_t) ((char **) resp -
			    (char **) ep->region);

	pend = freestack_pop(ep->pend_fs);
	smr_post_pend_resp(cmd, pend, resp);
	memcpy(pend->msg.data.iov, result_iov,
	       sizeof(*result_iov) * count);
	pend->msg.data.iov_count = count;

	ofi_cirque_commit(smr_resp_queue(ep->region));
}

static ssize_t smr_generic_atomic(struct fid_ep *ep_fid,
			const struct fi_ioc *ioc, void **desc, size_t count,
			const struct fi_ioc *compare_ioc, void **compare_desc,
			size_t compare_count, struct fi_ioc *result_ioc,
			void **result_desc, size_t result_count,
			fi_addr_t addr, const struct fi_rma_ioc *rma_ioc,
			size_t rma_count, enum fi_datatype datatype,
			enum fi_op atomic_op, void *context, uint32_t op)
{
	struct smr_ep *ep;
	struct smr_domain *domain;
	struct smr_region *peer_smr;
	struct smr_inject_buf *tx_buf;
	struct smr_cmd *cmd;
	struct iovec iov[SMR_IOV_LIMIT];
	struct iovec compare_iov[SMR_IOV_LIMIT];
	struct iovec result_iov[SMR_IOV_LIMIT];
	int peer_id, err = 0;
	uint16_t flags = 0;
	ssize_t ret = 0;
	size_t msg_len, total_len;

	assert(count <= SMR_IOV_LIMIT);
	assert(result_count <= SMR_IOV_LIMIT);
	assert(compare_count <= SMR_IOV_LIMIT);
	assert(rma_count <= SMR_IOV_LIMIT);

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
	domain = container_of(ep->util_ep.domain, struct smr_domain, util_domain);

	peer_id = (int) addr;
	ret = smr_verify_peer(ep, peer_id);
	if(ret)
		return ret;

	peer_smr = smr_peer_region(ep->region, peer_id);
	fastlock_acquire(&peer_smr->lock);
	if (peer_smr->cmd_cnt < 2) {
		ret = -FI_EAGAIN;
		goto unlock_region;
	}

	fastlock_acquire(&ep->util_ep.tx_cq->cq_lock);
	if (ofi_cirque_isfull(ep->util_ep.tx_cq->cirq)) {
		ret = -FI_EAGAIN;
		goto unlock_cq;
	}

	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));
	msg_len = total_len = ofi_datatype_size(datatype) *
			      ofi_total_ioc_cnt(ioc, count);
	
	switch (op) {
	case ofi_op_atomic_compare:
		assert(compare_ioc);
		smr_ioc_to_iov(compare_ioc, compare_iov, compare_count,
			       ofi_datatype_size(datatype));
		total_len *= 2;
		/* fall through */
	case ofi_op_atomic_fetch:
		assert(result_ioc);
		smr_ioc_to_iov(result_ioc, result_iov, result_count,
			       ofi_datatype_size(datatype));
		if (!domain->fast_rma)
			flags |= SMR_RMA_REQ;
		/* fall through */
	case ofi_op_atomic:
		if (atomic_op != FI_ATOMIC_READ) {
			assert(ioc);
			smr_ioc_to_iov(ioc, iov, count, ofi_datatype_size(datatype));
		} else {
			count = 0;
		}
		break;
	default:
		break;
	}

	if (total_len <= SMR_MSG_DATA_LEN && !(flags & SMR_RMA_REQ)) {
		smr_format_inline_atomic(cmd, smr_peer_addr(ep->region)[peer_id].addr,
					 iov, count, compare_iov, compare_count,
					 op, datatype, atomic_op);
	} else if (total_len <= SMR_INJECT_SIZE) {
		tx_buf = smr_freestack_pop(smr_inject_pool(peer_smr));
		smr_format_inject_atomic(cmd, smr_peer_addr(ep->region)[peer_id].addr,
					 iov, count, result_iov, result_count,
					 compare_iov, compare_count, op, datatype,
					 atomic_op, peer_smr, tx_buf);
	} else {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"message too large\n");
		ret = -FI_EINVAL;
		goto unlock_cq;
	}
	cmd->msg.hdr.op_flags |= flags;

	ofi_cirque_commit(smr_cmd_queue(peer_smr));
	peer_smr->cmd_cnt--;

	if (op != ofi_op_atomic) {
		if (flags & SMR_RMA_REQ) {
			smr_post_fetch_resp(ep, cmd,
				(const struct iovec *) result_iov,
				result_count);
			goto format_rma;
		}
		err = smr_fetch_result(ep, peer_smr, result_iov, result_count,
				       rma_ioc, rma_count, datatype, msg_len);
		if (err)
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"unable to fetch results");
	}

	ret = ep->tx_comp(ep, context, smr_tx_comp_flags(op), err);
	if (ret) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unable to process tx completion\n");
	}

format_rma:
	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));
	smr_format_rma_ioc(cmd, rma_ioc, rma_count);
	ofi_cirque_commit(smr_cmd_queue(peer_smr));
	peer_smr->cmd_cnt--;
unlock_cq:
	fastlock_release(&ep->util_ep.tx_cq->cq_lock);
unlock_region:
	fastlock_release(&peer_smr->lock);
	return ret;
}

static ssize_t smr_atomic_writemsg(struct fid_ep *ep,
			const struct fi_msg_atomic *msg, uint64_t flags)
{
	return smr_generic_atomic(ep, msg->msg_iov, msg->desc, msg->iov_count,
				  NULL, NULL, 0, NULL, NULL, 0, msg->addr,
				  msg->rma_iov, msg->rma_iov_count,
				  msg->datatype, msg->op, msg->context,
				  ofi_op_atomic);
}

static ssize_t smr_atomic_writev(struct fid_ep *ep,
			const struct fi_ioc *iov, void **desc, size_t count,
			fi_addr_t dest_addr, uint64_t addr, uint64_t key,
			enum fi_datatype datatype, enum fi_op op, void *context)
{
	struct fi_rma_ioc rma_iov;

	rma_iov.addr = addr;
	rma_iov.count = ofi_total_ioc_cnt(iov, count);
	rma_iov.key = key;

	return smr_generic_atomic(ep, iov, desc, count, NULL, NULL, 0, NULL,
				  NULL, 0, dest_addr, &rma_iov, 1, datatype,
				  op, context, ofi_op_atomic);
}

static ssize_t smr_atomic_write(struct fid_ep *ep, const void *buf, size_t count,
			void *desc, fi_addr_t dest_addr, uint64_t addr,
			uint64_t key, enum fi_datatype datatype, enum fi_op op,
			void *context)
{
	struct fi_ioc iov;
	struct fi_rma_ioc rma_iov;

	iov.addr = (void *) buf;
	iov.count = count;

	rma_iov.addr = addr;
	rma_iov.count = count;
	rma_iov.key = key;

	return smr_generic_atomic(ep, &iov, &desc, 1, NULL, NULL, 0, NULL, NULL, 0,
				  dest_addr, &rma_iov, 1, datatype, op, context,
				  ofi_op_atomic);
}

static ssize_t smr_atomic_inject(struct fid_ep *ep_fid, const void *buf,
			size_t count, fi_addr_t dest_addr, uint64_t addr,
			uint64_t key, enum fi_datatype datatype, enum fi_op op)
{
	struct smr_ep *ep;
	struct smr_region *peer_smr;
	struct smr_inject_buf *tx_buf;
	struct smr_cmd *cmd;
	struct iovec iov;
	struct fi_rma_ioc rma_ioc;
	int peer_id;
	ssize_t ret = 0;
	size_t total_len;

	assert(count <= SMR_INJECT_SIZE);

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	peer_id = (int) dest_addr;
	ret = smr_verify_peer(ep, peer_id);
	if(ret)
		return ret;

	peer_smr = smr_peer_region(ep->region, peer_id);
	fastlock_acquire(&peer_smr->lock);
	if (peer_smr->cmd_cnt < 2) {
		ret = -FI_EAGAIN;
		goto unlock_region;
	}

	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));
	total_len = count * ofi_datatype_size(datatype);
	
	iov.iov_base = (void *) buf;
	iov.iov_len = total_len;

	rma_ioc.addr = addr;
	rma_ioc.count = count;
	rma_ioc.key = key;

	if (total_len <= SMR_MSG_DATA_LEN) {
		smr_format_inline_atomic(cmd, smr_peer_addr(ep->region)[peer_id].addr,
					 &iov, 1, NULL, 0, ofi_op_atomic,
					 datatype, op);
	} else if (total_len <= SMR_INJECT_SIZE) {
		tx_buf = smr_freestack_pop(smr_inject_pool(peer_smr));
		smr_format_inject_atomic(cmd, smr_peer_addr(ep->region)[peer_id].addr,
					 &iov, 1, NULL, 0, NULL, 0, ofi_op_atomic,
					 datatype, op, peer_smr, tx_buf);
	}

	ofi_cirque_commit(smr_cmd_queue(peer_smr));
	peer_smr->cmd_cnt--;
	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));
	smr_format_rma_ioc(cmd, &rma_ioc, 1);
	ofi_cirque_commit(smr_cmd_queue(peer_smr));
	peer_smr->cmd_cnt--;

unlock_region:
	fastlock_release(&peer_smr->lock);
	return ret;
}

static ssize_t smr_atomic_readwritemsg(struct fid_ep *ep,
			const struct fi_msg_atomic *msg, struct fi_ioc *resultv,
			void **result_desc, size_t result_count, uint64_t flags)
{
	return smr_generic_atomic(ep, msg->msg_iov, msg->desc, msg->iov_count,
				  NULL, NULL, 0, resultv, result_desc,
				  result_count, msg->addr,
				  msg->rma_iov, msg->rma_iov_count,
				  msg->datatype, msg->op, msg->context,
				  ofi_op_atomic_fetch);
}

static ssize_t smr_atomic_readwritev(struct fid_ep *ep,
			const struct fi_ioc *iov, void **desc, size_t count,
			struct fi_ioc *resultv, void **result_desc,
			size_t result_count, fi_addr_t dest_addr, uint64_t addr,
			uint64_t key, enum fi_datatype datatype, enum fi_op op,
			void *context)
{
	struct fi_rma_ioc rma_iov;

	rma_iov.addr = addr;
	rma_iov.count = ofi_total_ioc_cnt(iov, count);
	rma_iov.key = key;

	return smr_generic_atomic(ep, iov, desc, count, NULL, NULL, 0, resultv,
				  result_desc, result_count, dest_addr,
				  &rma_iov, 1, datatype, op, context,
				  ofi_op_atomic_fetch);
}

static ssize_t smr_atomic_readwrite(struct fid_ep *ep, const void *buf,
			size_t count, void *desc, void *result,
			void *result_desc, fi_addr_t dest_addr, uint64_t addr,
			uint64_t key, enum fi_datatype datatype, enum fi_op op,
			void *context)
{
	struct fi_ioc iov, resultv;
	struct fi_rma_ioc rma_iov;

	iov.addr = (void *) buf;
	iov.count = count;

	resultv.addr = result;
	resultv.count = count;

	rma_iov.addr = addr;
	rma_iov.count = count;
	rma_iov.key = key;

	return smr_generic_atomic(ep, &iov, &desc, 1, NULL, NULL, 0, &resultv,
				  &result_desc, 1, dest_addr, &rma_iov, 1,
				  datatype, op, context, ofi_op_atomic_fetch);
}

static ssize_t smr_atomic_compwritemsg(struct fid_ep *ep,
			const struct fi_msg_atomic *msg,
			const struct fi_ioc *comparev, void **compare_desc,
			size_t compare_count, struct fi_ioc *resultv,
			void **result_desc, size_t result_count, uint64_t flags)
{
	return smr_generic_atomic(ep, msg->msg_iov, msg->desc, msg->iov_count,
				  comparev, compare_desc, compare_count,
				  resultv, result_desc,
				  result_count, msg->addr,
				  msg->rma_iov, msg->rma_iov_count,
				  msg->datatype, msg->op, msg->context,
				  ofi_op_atomic_compare);
}

static ssize_t smr_atomic_compwritev(struct fid_ep *ep,
			const struct fi_ioc *iov, void **desc, size_t count,
			const struct fi_ioc *comparev, void **compare_desc,
			size_t compare_count, struct fi_ioc *resultv,
			void **result_desc, size_t result_count,
			fi_addr_t dest_addr, uint64_t addr, uint64_t key,
			enum fi_datatype datatype, enum fi_op op, void *context)
{
	struct fi_rma_ioc rma_iov;

	rma_iov.addr = addr;
	rma_iov.count = ofi_total_ioc_cnt(iov, count);
	rma_iov.key = key;

	return smr_generic_atomic(ep, iov, desc, count, comparev, compare_desc,
				  compare_count, resultv, result_desc,
				  result_count, dest_addr, &rma_iov, 1,
				  datatype, op, context, ofi_op_atomic_compare);
}

static ssize_t smr_atomic_compwrite(struct fid_ep *ep, const void *buf,
			size_t count, void *desc, const void *compare,
			void *compare_desc, void *result, void *result_desc,
			fi_addr_t dest_addr, uint64_t addr, uint64_t key,
			enum fi_datatype datatype, enum fi_op op, void *context)
{
	struct fi_ioc iov, resultv, comparev;
	struct fi_rma_ioc rma_iov;

	iov.addr = (void *) buf;
	iov.count = count;

	resultv.addr = result;
	resultv.count = count;

	comparev.addr = (void *) compare;
	comparev.count = count;

	rma_iov.addr = addr;
	rma_iov.count = count;
	rma_iov.key = key;

	return smr_generic_atomic(ep, &iov, &desc, 1, &comparev, &compare_desc,
				  1, &resultv, &result_desc, 1, dest_addr,
				  &rma_iov, 1, datatype, op, context,
				  ofi_op_atomic_compare);
}

int smr_query_atomic(struct fid_domain *domain, enum fi_datatype datatype,
		     enum fi_op op, struct fi_atomic_attr *attr, uint64_t flags)
{
	int ret;
	size_t total_size;

	if (flags & FI_TAGGED) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"tagged atomic op not supported\n");
		return -FI_EINVAL;
	}

	ret = ofi_atomic_valid(&smr_prov, datatype, op, flags);
	if (ret || !attr)
		return ret;

	attr->size = ofi_datatype_size(datatype);

	total_size = (flags & FI_COMPARE_ATOMIC) ? SMR_COMP_INJECT_SIZE :
		      SMR_INJECT_SIZE;
	attr->count = total_size / attr->size;

	return ret;
}

static int smr_atomic_valid(struct fid_ep *ep, enum fi_datatype datatype,
			    enum fi_op op, size_t *count)
{
	struct fi_atomic_attr attr;
	int ret;

	ret = smr_query_atomic(NULL, datatype, op, &attr, 0);

	if (!ret)
		*count = attr.count;

	return ret;
}

static int smr_atomic_fetch_valid(struct fid_ep *ep, enum fi_datatype datatype,
				  enum fi_op op, size_t *count)
{
	struct fi_atomic_attr attr;
	int ret;

	ret = smr_query_atomic(NULL, datatype, op, &attr, FI_FETCH_ATOMIC);

	if (!ret)
		*count = attr.count;

	return ret;
}

static int smr_atomic_comp_valid(struct fid_ep *ep, enum fi_datatype datatype,
				 enum fi_op op, size_t *count)
{
	struct fi_atomic_attr attr;
	int ret;

	ret = smr_query_atomic(NULL, datatype, op, &attr, FI_COMPARE_ATOMIC);

	if (!ret)
		*count = attr.count;

	return ret;
}

struct fi_ops_atomic smr_atomic_ops = {
	.size = sizeof(struct fi_ops_atomic),
	.write = smr_atomic_write,
	.writev = smr_atomic_writev,
	.writemsg = smr_atomic_writemsg,
	.inject = smr_atomic_inject,
	.readwrite = smr_atomic_readwrite,
	.readwritev = smr_atomic_readwritev,
	.readwritemsg = smr_atomic_readwritemsg,
	.compwrite = smr_atomic_compwrite,
	.compwritev = smr_atomic_compwritev,
	.compwritemsg = smr_atomic_compwritemsg,
	.writevalid = smr_atomic_valid,
	.readwritevalid = smr_atomic_fetch_valid,
	.compwritevalid = smr_atomic_comp_valid,
};
