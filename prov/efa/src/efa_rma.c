/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <ofi_mem.h>
#include <ofi_iov.h>
#include "efa.h"
#include "efa_av.h"
#include "efa_data_path_ops.h"
#include "efa_data_path_direct.h"

/*
 * struct fi_efa_msg_rma is handed to fi_writemsg() as a struct fi_msg_rma
 * pointer, and the EFA metadata is read from the bytes that follow it. That
 * only holds while the core descriptor sits at offset 0 and the metadata starts
 * exactly where it ends, so pin both: if core ever changes struct fi_msg_rma,
 * this fails the build instead of silently reading the wrong offsets.
 */
#ifdef static_assert
static_assert(offsetof(struct fi_efa_msg_rma, msg) == 0,
	      "fi_msg_rma must be the first member of fi_efa_msg_rma");
static_assert(offsetof(struct fi_efa_msg_rma, feature_bits) ==
		      sizeof(struct fi_msg_rma),
	      "fi_efa_msg_rma metadata must directly follow fi_msg_rma");
/*
 * The action descriptors are append-only: their members keep their offsets for
 * the life of the interface, and new per-WR metadata appends to
 * fi_efa_msg_rma instead. Pin the shape so growing either one fails the build.
 */
static_assert(sizeof(struct fi_efa_comp_action_desc) == 2 * sizeof(uint32_t),
	      "fi_efa_comp_action_desc must stay id/value");
static_assert(offsetof(struct fi_efa_msg_rma, local) ==
		      sizeof(struct fi_msg_rma) + sizeof(uint64_t),
	      "fi_efa_msg_rma.local must directly follow feature_bits");
static_assert(offsetof(struct fi_efa_msg_rma, remote) ==
		      offsetof(struct fi_efa_msg_rma, local) +
			      sizeof(struct fi_efa_comp_action_desc),
	      "fi_efa_msg_rma.remote must directly follow local");
#endif

/**
 * @brief check whether endpoint was configured with FI_RMA capability
 * @return -FI_EOPNOTSUPP if FI_RMA wasn't requested, 0 if it was.
 */
static inline int efa_rma_check_cap(struct efa_base_ep *base_ep) {
	if ((base_ep->info->caps & FI_RMA) == FI_RMA)
		return 0;
	EFA_WARN_ONCE(FI_LOG_EP_DATA, "Operation requires FI_RMA capability, which was not requested.\n");
	return -FI_EOPNOTSUPP;
}

/*
 * efa_rma_post_read() will post a read request.
 *
 * Input:
 *     base_ep: endpoint
 *     msg: read operation information
 *     flags: currently no flags is taken
 *
 * On success return 0,
 * If read failed, return the error of read operation
 */
static inline ssize_t efa_rma_post_read(struct efa_base_ep *base_ep,
					const struct fi_msg_rma *msg,
					uint64_t flags)
{
	struct efa_domain *domain = base_ep->domain;
	struct efa_mr *efa_mr;
	struct efa_av_entry *entry;
	size_t iov_count = msg->iov_count;
	struct ibv_sge sge_list[2];  /* efa device support up to 2 iov */
	uintptr_t wr_id;
	int i, err = 0;
	size_t total_len;
	struct efa_context *efa_ctx;
	struct efa_direct_ope *direct_ope = NULL;

	efa_tracepoint(read_begin_msg_context, (size_t) msg->context, (size_t) msg->addr);

	total_len = ofi_total_iov_len(msg->msg_iov, msg->iov_count);

	EFA_DBG(FI_LOG_EP_DATA,
		"total len: %zu, addr: %lu, context: %lx, flags: %lx\n",
		total_len, msg->addr, (size_t) msg->context, flags);

	assert(msg->iov_count <= base_ep->domain->info->tx_attr->iov_limit);
	assert(msg->rma_iov_count > 0 &&
	       msg->rma_iov_count <= base_ep->domain->info->tx_attr->rma_iov_limit);
	assert(total_len <= base_ep->domain->device->max_rdma_size);

	ofi_genlock_lock(&base_ep->util_ep.lock);

	/* Prepare work request ID */
	if (base_ep->context_mode != USE_CONTEXT2) {
		/* No FI_CONTEXT2: echo raw context pointer, never dereference. */
		wr_id = (uintptr_t) msg->context;
	} else {
		efa_ctx = efa_fill_context(msg->context, msg->addr, flags,
							       FI_RMA | FI_READ);
		if (efa_env.track_mr && efa_ctx) {
			direct_ope = efa_direct_txe_alloc(
				base_ep, efa_ctx, NULL, msg);
			if (!direct_ope) {
				EFA_WARN(FI_LOG_EP_DATA,
					 "Failed to allocate direct TX operation entry for MR tracking\n");
				err = -FI_EAGAIN;
				goto out_err;
			}
			wr_id = (uintptr_t) direct_ope;
		} else {
			wr_id = (uintptr_t) efa_ctx;
		}
	}

	/* Handle 0-byte read with bounce buffer */
	if (total_len == 0) {
		sge_list[0].addr = (uint64_t)domain->zero_byte_bounce_buf;
		sge_list[0].length = 0;
		sge_list[0].lkey = domain->zero_byte_bounce_buf_mr->lkey;
		iov_count = 1;
	} else {
		/* Prepare SGE list */
		for (i = 0; i < msg->iov_count; ++i) {
			sge_list[i].addr = (uint64_t)msg->msg_iov[i].iov_base;
			sge_list[i].length = msg->msg_iov[i].iov_len;
			if (OFI_UNLIKELY(!msg->desc || !msg->desc[i])) {
				EFA_WARN(FI_LOG_EP_CTRL,
					 "EFA direct requires FI_MR_LOCAL but "
					 "application does not provide a valid desc\n");
				err = -FI_EINVAL;
				goto out_err;
			}
			efa_mr = (struct efa_mr *)msg->desc[i];
			sge_list[i].lkey = efa_mr->lkey;
		}
	}

	entry = efa_av_addr_to_entry(base_ep->av, msg->addr);
	assert(entry);

	/* Use consolidated RDMA read function */
	/* ep->domain->info->tx_attr->rma_iov_limit is set to 1 */
	err = efa_qp_post_read(base_ep->qp, sge_list, iov_count,
			       msg->rma_iov[0].key, msg->rma_iov[0].addr,
			       wr_id, flags,
			       entry->ah, efa_av_entry_ep_addr(entry)->qpn, efa_av_entry_ep_addr(entry)->qkey);
	if (OFI_UNLIKELY(err)) {
		err = (err == ENOMEM) ? -FI_EAGAIN : -err;
		goto out_err;
	}

	efa_tracepoint(post_read, wr_id, (uintptr_t)msg->context);

	ofi_genlock_unlock(&base_ep->util_ep.lock);

	return err;

out_err:
	if (direct_ope)
		efa_direct_ope_release(base_ep, direct_ope);
	ofi_genlock_unlock(&base_ep->util_ep.lock);

	return err;
}

static
ssize_t efa_rma_readmsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg, uint64_t flags)
{
	struct efa_base_ep *base_ep;
	int err;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	err = efa_rma_check_cap(base_ep);
	if (err)
		return err;

	return efa_rma_post_read(base_ep, msg, flags | base_ep->util_ep.tx_msg_flags);
}

static
ssize_t efa_rma_readv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		      size_t iov_count, fi_addr_t src_addr, uint64_t addr,
		      uint64_t key, void *context)
{
	struct fi_rma_iov rma_iov;
	struct fi_msg_rma msg;
	struct efa_base_ep *base_ep;
	size_t len;
	int err;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	err = efa_rma_check_cap(base_ep);
	if (err)
		return err;

	len = ofi_total_iov_len(iov, iov_count);
	EFA_SETUP_RMA_IOV(rma_iov, addr, len, key);
	EFA_SETUP_MSG_RMA(msg, iov, desc, iov_count, src_addr, &rma_iov, 1,
			  context, 0);

	return efa_rma_post_read(base_ep, &msg, efa_tx_flags(base_ep));
}

static
ssize_t efa_rma_read(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
		     fi_addr_t src_addr, uint64_t addr, uint64_t key,
		     void *context)
{
	struct iovec iov;
	struct fi_rma_iov rma_iov;
	struct fi_msg_rma msg;
	struct efa_base_ep *base_ep;
	int err;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	assert(len <= base_ep->max_rma_size);
	err = efa_rma_check_cap(base_ep);
	if (err)
		return err;

	EFA_SETUP_IOV(iov, buf, len);
	EFA_SETUP_RMA_IOV(rma_iov, addr, len, key);
	EFA_SETUP_MSG_RMA(msg, &iov, &desc, 1, src_addr, &rma_iov, 1, context, 0);

	return efa_rma_post_read(base_ep, &msg, efa_tx_flags(base_ep));
}

/**
 * @brief Validate the completion-action metadata of a WR and pack it for the
 *        WQE builders
 *
 * Reads the local and remote action descriptors selected by @emsg's
 * feature_bits and fills @sig with the action ids and their packed device
 * operands. A descriptor member whose feature bit is unset is not read and its
 * operand stays 0, which is what the device treats as "no data".
 *
 * An out-of-range entry index is not caught here: an action id is a device
 * handle, and a remote id belongs to a peer, so the provider cannot resolve
 * either back to the action's num_entries. The device rejects it, surfacing as
 * a completion with error.
 *
 * @param base_ep endpoint the WR is posted on
 * @param emsg    the caller's extended message descriptor
 * @param sig[out] per-WR action descriptor handed to the WQE builders
 * @return 0 on success, otherwise a negative libfabric error code.
 */
static inline int
efa_rma_init_comp_action_wr(struct efa_base_ep *base_ep,
			    const struct fi_efa_msg_rma *emsg,
			    struct efa_comp_action_wr *sig)
{
	uint64_t fb = emsg->feature_bits;
	uint64_t unknown =
		fb & ~(uint64_t) FI_EFA_MSG_RMA_SUPPORTED_FEATURE_BITS;

	/*
	 * Checked before the endpoint state so a descriptor built against a
	 * newer header is rejected on its own terms, not silently treated as if
	 * the unknown fields were absent.
	 */
	if (unknown) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "Unsupported feature_bits in fi_efa_msg_rma: 0x%lx\n",
			 (unsigned long) unknown);
		return -FI_EOPNOTSUPP;
	}

	if (!base_ep->comp_action_enabled) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "FI_EFA_EXTENDED_MSG used but action support is not "
			 "enabled on the endpoint\n");
		return -FI_EINVAL;
	}

	/* An action's value bit requires its ID bit. */
	if (((fb & FI_EFA_LOCAL_ACTION_VALUE) &&
	     !(fb & FI_EFA_LOCAL_ACTION_ID)) ||
	    ((fb & FI_EFA_REMOTE_ACTION_VALUE) &&
	     !(fb & FI_EFA_REMOTE_ACTION_ID))) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "FI_EFA_*_ACTION_VALUE set without the corresponding "
			 "FI_EFA_*_ACTION_ID\n");
		return -FI_EINVAL;
	}

	sig->feature_bits = fb;

	if (fb & FI_EFA_LOCAL_ACTION_ID) {
		sig->local_action_id = emsg->local.id;
		sig->local_action_data =
			(fb & FI_EFA_LOCAL_ACTION_VALUE) ? emsg->local.value : 0;
	}

	if (fb & FI_EFA_REMOTE_ACTION_ID) {
		sig->remote_action_id = emsg->remote.id;
		sig->remote_action_data =
			(fb & FI_EFA_REMOTE_ACTION_VALUE) ? emsg->remote.value : 0;
	}

	return 0;
}

/**
 * @brief Post a WRITE request
 *
 * Input:
 *     base_ep: endpoint
 *     msg: read operation information
 *     flags: flags passed
 * @return On success return 0, otherwise return a negative libfabric error code.
 */
static inline ssize_t efa_rma_post_write(struct efa_base_ep *base_ep,
					 const struct fi_msg_rma *msg,
					 uint64_t flags)
{
	struct efa_domain *domain = base_ep->domain;
	struct efa_av_entry *entry;
	size_t iov_count = msg->iov_count;
	struct ibv_sge sge_list[2];  /* efa device support up to 2 iov */
	struct ibv_data_buf inline_data_list[2];
	uintptr_t wr_id;
	bool use_inline, len_fits_inline, is_hmem;
	int err = 0;
	size_t total_len = ofi_total_iov_len(msg->msg_iov, msg->iov_count);
	struct efa_context *efa_ctx;
	struct efa_direct_ope *direct_ope = NULL;
	struct efa_comp_action_wr sig = {0};
	const struct efa_comp_action_wr *sig_ptr = NULL;

	efa_tracepoint(write_begin_msg_context, (size_t) msg->context, (size_t) msg->addr);
	EFA_DBG(FI_LOG_EP_DATA,
		"total len: %zu, addr: %lu, context: %lx, flags: %lx\n",
		total_len, msg->addr, (size_t) msg->context, flags);

	ofi_genlock_lock(&base_ep->util_ep.lock);

	if (flags & FI_EFA_EXTENDED_MSG) {
		const struct fi_efa_msg_rma *emsg =
			(const struct fi_efa_msg_rma *) msg;

		err = efa_rma_init_comp_action_wr(base_ep, emsg, &sig);
		if (err)
			goto out_err;
		sig_ptr = &sig;
	}

	/* Prepare work request ID */
	if (base_ep->context_mode != USE_CONTEXT2) {
		/* No FI_CONTEXT2: echo raw context pointer, never dereference. */
		wr_id = (uintptr_t) msg->context;
	} else {
		efa_ctx = efa_fill_context(msg->context, msg->addr, flags,
							       FI_RMA | FI_WRITE);
		if (efa_env.track_mr && efa_ctx) {
			direct_ope = efa_direct_txe_alloc(
				base_ep, efa_ctx, NULL, msg);
			if (!direct_ope) {
				EFA_WARN(FI_LOG_EP_DATA,
					 "Failed to allocate direct TX operation entry for MR tracking\n");
				err = -FI_EAGAIN;
				goto out_err;
			}
			wr_id = (uintptr_t) direct_ope;
		} else {
			wr_id = (uintptr_t) efa_ctx;
		}
	}

	len_fits_inline = total_len <= base_ep->inject_rma_size;
	is_hmem = false;
	if (msg->desc) {
		for (size_t i = 0; i < msg->iov_count; i++) {
			if (efa_mr_is_hmem(msg->desc[i])) {
				is_hmem = true;
				break;
			}
		}
	}
	use_inline = len_fits_inline && !is_hmem;

	if (!use_inline && (flags & FI_INJECT)) {
		err = -FI_EOPNOTSUPP;
		if (!len_fits_inline) {
			EFA_WARN(FI_LOG_EP_DATA,
				 "FI_INJECT is requested but message "
				 "size of %zu exceeds inject_rma_size "
				 "of %zu.\n", total_len,
				 base_ep->inject_rma_size);
			err = -FI_EINVAL;
		} else {
			assert(is_hmem);
			EFA_WARN(FI_LOG_EP_DATA,
				 "FI_INJECT is not supported for "
				 "FI_HMEM memory.\n");
			err = -FI_ENOSYS;
		}
		goto out_err;
	}

	/* Handle 0-byte write with bounce buffer */
	if (total_len == 0) {
		use_inline = false;
		sge_list[0].addr = (uint64_t)domain->zero_byte_bounce_buf;
		sge_list[0].length = 0;
		sge_list[0].lkey = domain->zero_byte_bounce_buf_mr->lkey;
		iov_count = 1;
	} else if (use_inline) {
		for (size_t i = 0; i < msg->iov_count; i++) {
			inline_data_list[i].addr = msg->msg_iov[i].iov_base;
			inline_data_list[i].length = msg->msg_iov[i].iov_len;
		}
	} else {
		/* Prepare SGE list */
		for (size_t i = 0; i < msg->iov_count; ++i) {
			sge_list[i].addr = (uint64_t)msg->msg_iov[i].iov_base;
			sge_list[i].length = msg->msg_iov[i].iov_len;
			if (OFI_UNLIKELY(!msg->desc || !msg->desc[i])) {
				EFA_WARN(FI_LOG_EP_CTRL,
					 "EFA direct requires FI_MR_LOCAL but "
					 "application does not provide a valid desc\n");
				err = -FI_EINVAL;
				goto out_err;
			}
			sge_list[i].lkey = ((struct efa_mr *)msg->desc[i])->lkey;
		}
	}

	entry = efa_av_addr_to_entry(base_ep->av, msg->addr);
	assert(entry);

	/* Use consolidated RDMA write function */
	err = efa_qp_post_write(base_ep->qp, sge_list, iov_count,
				inline_data_list, use_inline,
				msg->rma_iov[0].key, msg->rma_iov[0].addr,
				wr_id, msg->data, flags,
				entry->ah, efa_av_entry_ep_addr(entry)->qpn,
				efa_av_entry_ep_addr(entry)->qkey,
				sig_ptr);
	if (OFI_UNLIKELY(err)) {
		err = (err == ENOMEM) ? -FI_EAGAIN : -err;
		goto out_err;
	}

	efa_tracepoint(post_write, wr_id, (uintptr_t)msg->context);

	ofi_genlock_unlock(&base_ep->util_ep.lock);

	return err;

out_err:
	if (direct_ope)
		efa_direct_ope_release(base_ep, direct_ope);
	ofi_genlock_unlock(&base_ep->util_ep.lock);

	return err;
}

ssize_t efa_rma_writemsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
			 uint64_t flags)
{
	struct efa_base_ep *base_ep;
	int err;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	err = efa_rma_check_cap(base_ep);
	if (err)
		return err;

	return efa_rma_post_write(base_ep, msg, flags | base_ep->util_ep.tx_msg_flags);
}

ssize_t efa_rma_writev(struct fid_ep *ep_fid, const struct iovec *iov,
		       void **desc, size_t iov_count, fi_addr_t dest_addr,
		       uint64_t addr, uint64_t key, void *context)
{
	struct fi_rma_iov rma_iov;
	struct fi_msg_rma msg;
	struct efa_base_ep *base_ep;
	size_t len;
	int err;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	err = efa_rma_check_cap(base_ep);
	if (err)
		return err;

	len = ofi_total_iov_len(iov, iov_count);
	EFA_SETUP_RMA_IOV(rma_iov, addr, len, key);
	EFA_SETUP_MSG_RMA(msg, iov, desc, iov_count, dest_addr, &rma_iov, 1,
			  context, 0);

	return efa_rma_post_write(base_ep, &msg, efa_tx_flags(base_ep));
}

ssize_t efa_rma_write(struct fid_ep *ep_fid, const void *buf, size_t len,
		      void *desc, fi_addr_t dest_addr, uint64_t addr,
		      uint64_t key, void *context)
{
	struct iovec iov;
	struct fi_rma_iov rma_iov;
	struct fi_msg_rma msg;
	struct efa_base_ep *base_ep;
	int err;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	assert(len <= base_ep->max_rma_size);
	err = efa_rma_check_cap(base_ep);
	if (err)
		return err;

	EFA_SETUP_IOV(iov, buf, len);
	EFA_SETUP_RMA_IOV(rma_iov, addr, len, key);
	EFA_SETUP_MSG_RMA(msg, &iov, &desc, 1, dest_addr, &rma_iov, 1, context, 0);

	return efa_rma_post_write(base_ep, &msg, efa_tx_flags(base_ep));
}

ssize_t efa_rma_writedata(struct fid_ep *ep_fid, const void *buf, size_t len,
			  void *desc, uint64_t data, fi_addr_t dest_addr,
			  uint64_t addr, uint64_t key, void *context)
{
	struct iovec iov;
	struct fi_rma_iov rma_iov;
	struct fi_msg_rma msg;
	struct efa_base_ep *base_ep;
	int err;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	assert(len <= base_ep->max_rma_size);
	err = efa_rma_check_cap(base_ep);
	if (err)
		return err;

	EFA_SETUP_IOV(iov, buf, len);
	EFA_SETUP_RMA_IOV(rma_iov, addr, len, key);
	EFA_SETUP_MSG_RMA(msg, &iov, &desc, 1, dest_addr, &rma_iov, 1, context, data);

	return efa_rma_post_write(base_ep, &msg, FI_REMOTE_CQ_DATA | efa_tx_flags(base_ep));
}

ssize_t efa_rma_inject_write(struct fid_ep *ep_fid, const void *buf, size_t len,
			     fi_addr_t dest_addr, uint64_t addr, uint64_t key)
{
	struct fi_msg_rma msg;
	struct iovec iov;
	struct fi_rma_iov rma_iov;
	struct efa_base_ep *base_ep;
	int err;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	err = efa_rma_check_cap(base_ep);
	if (err)
		return err;
	if (base_ep->context_mode != USE_CONTEXT2 || len > base_ep->inject_rma_size)
		return -FI_ENOSYS;

	EFA_SETUP_IOV(iov, buf, len);
	EFA_SETUP_RMA_IOV(rma_iov, addr, len, key);
	EFA_SETUP_MSG_RMA(msg, &iov, NULL, 1, dest_addr, &rma_iov, 1, NULL, 0);

	return efa_rma_post_write(base_ep, &msg, FI_INJECT);
}

ssize_t efa_rma_inject_writedata(struct fid_ep *ep_fid, const void *buf,
				 size_t len, uint64_t data, fi_addr_t dest_addr,
				 uint64_t addr, uint64_t key)
{
	struct fi_msg_rma msg;
	struct iovec iov;
	struct fi_rma_iov rma_iov;
	struct efa_base_ep *base_ep;
	int err;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	err = efa_rma_check_cap(base_ep);
	if (err)
		return err;
	if (base_ep->context_mode != USE_CONTEXT2 || len > base_ep->inject_rma_size)
		return -FI_ENOSYS;

	EFA_SETUP_IOV(iov, buf, len);
	EFA_SETUP_RMA_IOV(rma_iov, addr, len, key);
	EFA_SETUP_MSG_RMA(msg, &iov, NULL, 1, dest_addr, &rma_iov, 1, NULL, data);

	return efa_rma_post_write(base_ep, &msg, FI_INJECT | FI_REMOTE_CQ_DATA);
}



struct fi_ops_rma efa_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = efa_rma_read,
	.readv = efa_rma_readv,
	.readmsg = efa_rma_readmsg,
	.write = efa_rma_write,
	.writev = efa_rma_writev,
	.writemsg = efa_rma_writemsg,
	.inject = efa_rma_inject_write,
	.writedata = efa_rma_writedata,
	.injectdata = efa_rma_inject_writedata,
};
