/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved. */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <errno.h>
#include <string.h>
#include "config.h"
#include <ofi_mem.h>
#include "efa.h"
#include "efa_av.h"
#include "efa_cntr.h"
#include "efa_cq.h"
#include <infiniband/verbs.h>


static inline uint64_t efa_cq_opcode_to_fi_flags(enum ibv_wc_opcode opcode) {
	switch (opcode) {
	case IBV_WC_SEND:
		return FI_SEND | FI_MSG;
	case IBV_WC_RECV:
		return FI_RECV | FI_MSG;
	case IBV_WC_RDMA_WRITE:
		return FI_RMA | FI_WRITE;
	case IBV_WC_RECV_RDMA_WITH_IMM:
		return FI_REMOTE_CQ_DATA | FI_RMA | FI_REMOTE_WRITE;
	case IBV_WC_RDMA_READ:
		return FI_RMA | FI_READ;
	default:
		assert(0);
		return 0;
	}
}

static void efa_cq_construct_cq_entry(struct ibv_cq_ex *ibv_cqx,
				      struct fi_cq_tagged_entry *entry)
{
	entry->op_context = (void *)ibv_cqx->wr_id;
	entry->flags = efa_cq_opcode_to_fi_flags(ibv_wc_read_opcode(ibv_cqx));
	entry->len = ibv_wc_read_byte_len(ibv_cqx);
	entry->buf = NULL;
	entry->data = 0;
	entry->tag = 0;

	if (ibv_wc_read_wc_flags(ibv_cqx) & IBV_WC_WITH_IMM) {
		entry->flags |= FI_REMOTE_CQ_DATA;
		entry->data = ibv_wc_read_imm_data(ibv_cqx);
	}
}

/**
 * @brief handle the situation that a TX/RX operation encountered error
 *
 * This function does the following to handle error:
 *
 * 1. write an error cq entry for the operation, if writing
 *    CQ error entry failed, it will write eq entry.
 *
 * 2. increase error counter.
 *
 * 3. print warning message with self and peer's raw address
 *
 * @param[in]	base_ep     efa_base_ep
 * @param[in]	ibv_cq_ex   extended ibv cq
 * @param[in]	err         positive libfabric error code
 * @param[in]	prov_errno  positive EFA provider specific error code
 * @param[in]	is_tx       if the error is for TX or RX operation
 */
static void efa_cq_handle_error(struct efa_base_ep *base_ep,
				struct ibv_cq_ex *ibv_cq_ex, int err,
				int prov_errno, bool is_tx)
{
	struct fi_cq_err_entry err_entry;
	fi_addr_t addr;
	char err_msg[EFA_ERROR_MSG_BUFFER_LENGTH] = {0};
	int write_cq_err;

	memset(&err_entry, 0, sizeof(err_entry));
	efa_cq_construct_cq_entry(ibv_cq_ex, (struct fi_cq_tagged_entry *) &err_entry);
	err_entry.err = err;
	err_entry.prov_errno = prov_errno;

	if (is_tx)
		// TODO: get correct peer addr for TX operation
		addr = FI_ADDR_NOTAVAIL;
	else
		addr = efa_av_reverse_lookup(base_ep->av,
					     ibv_wc_read_slid(ibv_cq_ex),
					     ibv_wc_read_src_qp(ibv_cq_ex));

	if (OFI_UNLIKELY(efa_write_error_msg(base_ep, addr, prov_errno,
					     err_msg,
					     &err_entry.err_data_size))) {
		err_entry.err_data_size = 0;
	} else {
		err_entry.err_data = err_msg;
	}

	EFA_WARN(FI_LOG_CQ, "err: %d, message: %s (%d)\n",
		err_entry.err,
		err_entry.err_data
			? (const char *) err_entry.err_data
			: efa_strerror(err_entry.prov_errno),
		err_entry.prov_errno);

	efa_show_help(err_entry.prov_errno);

	efa_cntr_report_error(&base_ep->util_ep, err_entry.flags);
	write_cq_err = ofi_cq_write_error(is_tx ? base_ep->util_ep.tx_cq :
						  base_ep->util_ep.rx_cq,
					  &err_entry);
	if (write_cq_err) {
		EFA_WARN(
			FI_LOG_CQ,
			"Error writing error cq entry when handling %s error\n",
			is_tx ? "TX" : "RX");
		efa_base_ep_write_eq_error(base_ep, err, prov_errno);
	}
}

/**
 * @brief handle the event that a TX request has been completed
 *
 * @param[in]		base_ep     efa_base_ep
 * @param[in]		ibv_cq_ex   extended ibv cq
 * @param[in]		cq_entry    fi_cq_tagged_entry
 */
static void efa_cq_handle_tx_completion(struct efa_base_ep *base_ep,
					struct ibv_cq_ex *ibv_cq_ex,
					struct fi_cq_tagged_entry *cq_entry)
{
	struct util_cq *tx_cq = base_ep->util_ep.tx_cq;
	int ret = 0;

	/* NULL wr_id means no FI_COMPLETION flag */
	if (!ibv_cq_ex->wr_id)
		return;

	/* TX completions should not send peer address to util_cq */
	if (base_ep->util_ep.caps & FI_SOURCE)
		ret = ofi_cq_write_src(tx_cq, cq_entry->op_context,
				       cq_entry->flags, cq_entry->len,
				       cq_entry->buf, cq_entry->data,
				       cq_entry->tag, FI_ADDR_NOTAVAIL);
	else
		ret = ofi_cq_write(tx_cq, cq_entry->op_context, cq_entry->flags,
				   cq_entry->len, cq_entry->buf, cq_entry->data,
				   cq_entry->tag);

	if (OFI_UNLIKELY(ret)) {
		EFA_WARN(FI_LOG_CQ, "Unable to write send completion: %s\n",
			 fi_strerror(-ret));
		efa_cq_handle_error(base_ep, ibv_cq_ex, -ret,
				    FI_EFA_ERR_WRITE_SEND_COMP, true);
	}
}

/**
 * @brief handle the event that a RX request has been completed
 *
 * @param[in]		base_ep     efa_base_ep
 * @param[in]		ibv_cq_ex   extended ibv cq
 * @param[in]		cq_entry    fi_cq_tagged_entry
 */
static void efa_cq_handle_rx_completion(struct efa_base_ep *base_ep,
					struct ibv_cq_ex *ibv_cq_ex,
					struct fi_cq_tagged_entry *cq_entry)
{
	struct util_cq *rx_cq = base_ep->util_ep.rx_cq;
	fi_addr_t src_addr;
	int ret = 0;

	/* NULL wr_id means no FI_COMPLETION flag */
	if (!ibv_cq_ex->wr_id)
		return;

	if (base_ep->util_ep.caps & FI_SOURCE) {
		src_addr = efa_av_reverse_lookup(base_ep->av,
						 ibv_wc_read_slid(ibv_cq_ex),
						 ibv_wc_read_src_qp(ibv_cq_ex));
		ret = ofi_cq_write_src(rx_cq, cq_entry->op_context,
				       cq_entry->flags, cq_entry->len,
				       cq_entry->buf, cq_entry->data,
				       cq_entry->tag, src_addr);
	} else {
		ret = ofi_cq_write(rx_cq, cq_entry->op_context, cq_entry->flags,
				   cq_entry->len, cq_entry->buf, cq_entry->data,
				   cq_entry->tag);
	}

	if (OFI_UNLIKELY(ret)) {
		EFA_WARN(FI_LOG_CQ, "Unable to write recv completion: %s\n",
			 fi_strerror(-ret));
		efa_cq_handle_error(base_ep, ibv_cq_ex, -ret,
				    FI_EFA_ERR_WRITE_RECV_COMP, false);
	}
}

/**
 * @brief handle rdma-core CQ completion resulted from IBV_WRITE_WITH_IMM
 *
 * This function handles hardware-assisted RDMA writes with immediate data at
 * remote endpoint.  These do not have a packet context, nor do they have a
 * connid available.
 * 
 * @param[in]		base_ep     efa_base_ep
 * @param[in]		ibv_cq_ex   extended ibv cq
 */
static void
efa_cq_proc_ibv_recv_rdma_with_imm_completion(struct efa_base_ep *base_ep,
					      struct ibv_cq_ex *ibv_cq_ex)
{
	struct util_cq *rx_cq = base_ep->util_ep.rx_cq;
	int ret;
	fi_addr_t src_addr;
	uint32_t imm_data = ibv_wc_read_imm_data(ibv_cq_ex);
	uint32_t len = ibv_wc_read_byte_len(ibv_cq_ex);
	uint64_t flags = FI_REMOTE_CQ_DATA | FI_RMA | FI_REMOTE_WRITE;

	if (base_ep->util_ep.caps & FI_SOURCE) {
		src_addr = efa_av_reverse_lookup(base_ep->av,
						 ibv_wc_read_slid(ibv_cq_ex),
						 ibv_wc_read_src_qp(ibv_cq_ex));
		ret = ofi_cq_write_src(rx_cq, NULL, flags, len, NULL, imm_data,
				       0, src_addr);
	} else {
		ret = ofi_cq_write(rx_cq, NULL, flags, len, NULL, imm_data, 0);
	}

	if (OFI_UNLIKELY(ret)) {
		EFA_WARN(FI_LOG_CQ,
			 "Unable to write a cq entry for remote for RECV_RDMA "
			 "operation: %s\n",
			 fi_strerror(-ret));
		efa_base_ep_write_eq_error(base_ep, -ret,
					   FI_EFA_ERR_WRITE_RECV_COMP);
	}
}

/**
 * @brief poll rdma-core cq and process the cq entry
 *
 * @param[in]	cqe_to_process    Max number of cq entry to poll and process. 
 * A negative number means to poll until cq empty.
 * @param[in]   util_cq           util_cq
 */
void efa_cq_poll_ibv_cq(ssize_t cqe_to_process, struct util_cq *util_cq)
{
	bool should_end_poll = false;
	struct efa_base_ep *base_ep;
	struct efa_cq *cq;
	struct efa_domain *efa_domain;
	struct fi_cq_tagged_entry cq_entry = {0};
	struct fi_cq_err_entry err_entry;
	ssize_t err = 0;
	size_t num_cqe = 0; /* Count of read entries */
	int prov_errno, opcode;

	/* Initialize an empty ibv_poll_cq_attr struct for ibv_start_poll.
	 * EFA expects .comp_mask = 0, or otherwise returns EINVAL.
	 */
	struct ibv_poll_cq_attr poll_cq_attr = {.comp_mask = 0};

	cq = container_of(util_cq, struct efa_cq, util_cq);
	efa_domain = container_of(cq->util_cq.domain, struct efa_domain, util_domain);

	/* Call ibv_start_poll only once */
	err = ibv_start_poll(cq->ibv_cq.ibv_cq_ex, &poll_cq_attr);
	should_end_poll = !err;

	while (!err) {
		base_ep = efa_domain->qp_table[ibv_wc_read_qp_num(cq->ibv_cq.ibv_cq_ex) & efa_domain->qp_table_sz_m1]->base_ep;
		opcode = ibv_wc_read_opcode(cq->ibv_cq.ibv_cq_ex);
		if (cq->ibv_cq.ibv_cq_ex->status) {
			prov_errno = ibv_wc_read_vendor_err(cq->ibv_cq.ibv_cq_ex);
			switch (opcode) {
			case IBV_WC_SEND: /* fall through */
			case IBV_WC_RDMA_WRITE: /* fall through */
			case IBV_WC_RDMA_READ:
				efa_cq_handle_error(base_ep, cq->ibv_cq.ibv_cq_ex,
						    to_fi_errno(prov_errno),
						    prov_errno, true);
				break;
			case IBV_WC_RECV: /* fall through */
			case IBV_WC_RECV_RDMA_WITH_IMM:
				if (efa_cq_wc_is_unsolicited(cq->ibv_cq.ibv_cq_ex)) {
					EFA_WARN(FI_LOG_CQ,
						 "Receive error %s (%d) for "
						 "unsolicited write recv",
						 efa_strerror(prov_errno),
						 prov_errno);
					efa_base_ep_write_eq_error(
						base_ep,
						to_fi_errno(prov_errno),
						prov_errno);
					break;
				}
				efa_cq_handle_error(base_ep, cq->ibv_cq.ibv_cq_ex,
						    to_fi_errno(prov_errno),
						    prov_errno, false);
				break;
			default:
				EFA_WARN(FI_LOG_EP_CTRL, "Unhandled op code %d\n", opcode);
				assert(0 && "Unhandled op code");
			}
			break;
		}

		efa_cq_construct_cq_entry(cq->ibv_cq.ibv_cq_ex, &cq_entry);

		switch (opcode) {
		case IBV_WC_SEND: /* fall through */
		case IBV_WC_RDMA_WRITE: /* fall through */
		case IBV_WC_RDMA_READ:
			efa_cq_handle_tx_completion(base_ep, cq->ibv_cq.ibv_cq_ex, &cq_entry);
			efa_cntr_report_tx_completion(&base_ep->util_ep, cq_entry.flags);
			break;
		case IBV_WC_RECV:
			efa_cq_handle_rx_completion(base_ep, cq->ibv_cq.ibv_cq_ex, &cq_entry);
			efa_cntr_report_rx_completion(&base_ep->util_ep, cq_entry.flags);
			break;
		case IBV_WC_RECV_RDMA_WITH_IMM:
			efa_cq_proc_ibv_recv_rdma_with_imm_completion(
				base_ep, cq->ibv_cq.ibv_cq_ex);
			efa_cntr_report_rx_completion(&base_ep->util_ep, cq_entry.flags);
			break;
		default:
			EFA_WARN(FI_LOG_EP_CTRL,
				"Unhandled cq type\n");
			assert(0 && "Unhandled cq type");
		}

		num_cqe++;
		if (num_cqe == cqe_to_process) {
			break;
		}

		err = ibv_next_poll(cq->ibv_cq.ibv_cq_ex);
	}

	if (err && err != ENOENT) {
		err = err > 0 ? err : -err;
		prov_errno = ibv_wc_read_vendor_err(cq->ibv_cq.ibv_cq_ex);
		EFA_WARN(FI_LOG_CQ,
			 "Unexpected error when polling ibv cq, err: %s (%zd) "
			 "prov_errno: %s (%d)\n",
			 fi_strerror(err), err, efa_strerror(prov_errno),
			 prov_errno);
		efa_show_help(prov_errno);
		err_entry = (struct fi_cq_err_entry) {
			.err = err,
			.prov_errno = prov_errno,
			.op_context = NULL,
		};
		ofi_cq_write_error(&cq->util_cq, &err_entry);
	}

	if (should_end_poll)
		ibv_end_poll(cq->ibv_cq.ibv_cq_ex);
}

static const char *efa_cq_strerror(struct fid_cq *cq_fid,
				   int prov_errno,
				   const void *err_data,
				   char *buf, size_t len)
{
	return err_data
		? (const char *) err_data
		: efa_strerror(prov_errno);
}

static struct fi_ops_cq efa_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = ofi_cq_read,
	.readfrom = ofi_cq_readfrom,
	.readerr = ofi_cq_readerr,
	.sread = fi_no_cq_sread,
	.sreadfrom = fi_no_cq_sreadfrom,
	.signal = fi_no_cq_signal,
	.strerror = efa_cq_strerror
};

void efa_cq_progress(struct util_cq *cq)
{
	efa_cq_poll_ibv_cq(efa_env.efa_cq_read_size, cq);
}

static int efa_cq_close(fid_t fid)
{
	struct efa_cq *cq;
	int ret;

	cq = container_of(fid, struct efa_cq, util_cq.cq_fid.fid);

	if (cq->ibv_cq.ibv_cq_ex) {
		ret = -ibv_destroy_cq(ibv_cq_ex_to_cq(cq->ibv_cq.ibv_cq_ex));
		if (ret) {
			EFA_WARN(FI_LOG_CQ, "Unable to close ibv cq: %s\n",
				fi_strerror(-ret));
			return ret;
		}
		cq->ibv_cq.ibv_cq_ex = NULL;
	}

	ret = ofi_cq_cleanup(&cq->util_cq);
	if (ret)
		return ret;

	free(cq);

	return 0;
}

static struct fi_ops efa_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};


int efa_cq_open(struct fid_domain *domain_fid, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context)
{
	struct efa_cq *cq;
	struct efa_domain *efa_domain;
	int err, retv;

	if (attr->wait_obj != FI_WAIT_NONE)
		return -FI_ENOSYS;

	cq = calloc(1, sizeof(*cq));
	if (!cq)
		return -FI_ENOMEM;

	err = ofi_cq_init(&efa_prov, domain_fid, attr, &cq->util_cq,
			  &efa_cq_progress, context);
	if (err) {
		EFA_WARN(FI_LOG_CQ, "Unable to create UTIL_CQ\n");
		goto err_free_cq;
	}

	efa_domain = container_of(cq->util_cq.domain, struct efa_domain,
				  util_domain);
	err = efa_cq_ibv_cq_ex_open(attr, efa_domain->device->ibv_ctx,
				    &cq->ibv_cq.ibv_cq_ex,
				    &cq->ibv_cq.ibv_cq_ex_type);
	if (err) {
		EFA_WARN(FI_LOG_CQ, "Unable to create extended CQ: %s\n", fi_strerror(err));
		goto err_free_util_cq;
	}

	*cq_fid = &cq->util_cq.cq_fid;
	(*cq_fid)->fid.fclass = FI_CLASS_CQ;
	(*cq_fid)->fid.context = context;
	(*cq_fid)->fid.ops = &efa_cq_fi_ops;
	(*cq_fid)->ops = &efa_cq_ops;

	return 0;

err_free_util_cq:
	retv = ofi_cq_cleanup(&cq->util_cq);
	if (retv)
		EFA_WARN(FI_LOG_CQ, "Unable to close util cq: %s\n",
			 fi_strerror(-retv));
err_free_cq:
	free(cq);
	return err;
}
