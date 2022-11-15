/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 * Copyright (c) 2021-2022 Hewlett Packard Enterprise Development LP
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>

#include <ofi_list.h>
#include <ofi.h>

#include "cxip.h"

#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_CTRL, __VA_ARGS__)

/*
 * cxip_rma_selective_completion_cb() - RMA selective completion callback.
 */
int cxip_rma_selective_completion_cb(struct cxip_req *req,
				     const union c_event *event)
{
	/* When errors happen, send events can occur before the put/get event.
	 * These events should just be dropped.
	 */
	if (event->hdr.event_type == C_EVENT_SEND) {
		CXIP_WARN("Unexpected %s event: rc=%s\n",
			  cxi_event_to_str(event),
			  cxi_rc_to_str(cxi_event_rc(event)));
		return FI_SUCCESS;
	}

	return cxip_cq_req_error(req, 0, FI_EIO, cxi_event_rc(event), NULL, 0);
}

/*
 * cxip_rma_write_selective_completion_req() - Return request state associated
 * with all RMA write with selective completion transactions on the transmit
 * context.
 *
 * The request is freed when the TXC send CQ is closed.
 */
static struct cxip_req *cxip_rma_write_selective_completion_req(struct cxip_txc *txc)
{
	if (!txc->rma_write_selective_completion_req) {
		struct cxip_req *req;

		req = cxip_cq_req_alloc(txc->send_cq, 0, txc);
		if (!req)
			return NULL;

		req->cb = cxip_rma_selective_completion_cb;
		req->context = (uint64_t)txc->fid.ctx.fid.context;
		req->flags = FI_RMA | FI_WRITE;
		req->addr = FI_ADDR_UNSPEC;

		txc->rma_write_selective_completion_req = req;
	}

	return txc->rma_write_selective_completion_req;
}

/*
 * cxip_rma_read_selective_completion_req() - Return request state associated
 * with all RMA read with selective completion transactions on the transmit
 * context.
 *
 * The request is freed when the TXC send CQ is closed.
 */
static struct cxip_req *cxip_rma_read_selective_completion_req(struct cxip_txc *txc)
{
	if (!txc->rma_read_selective_completion_req) {
		struct cxip_req *req;

		req = cxip_cq_req_alloc(txc->send_cq, 0, txc);
		if (!req)
			return NULL;

		req->cb = cxip_rma_selective_completion_cb;
		req->context = (uint64_t)txc->fid.ctx.fid.context;
		req->flags = FI_RMA | FI_READ;
		req->addr = FI_ADDR_UNSPEC;

		txc->rma_read_selective_completion_req = req;
	}

	return txc->rma_read_selective_completion_req;
}

/*
 * cxip_rma_cb() - RMA event callback.
 */
static int cxip_rma_cb(struct cxip_req *req, const union c_event *event)
{
	int ret;
	int event_rc;
	bool success_event = !!(req->flags & FI_COMPLETION);
	struct cxip_txc *txc = req->rma.txc;

	/* When errors happen, send events can occur before the put/get event.
	 * These events should just be dropped.
	 */
	if (event->hdr.event_type == C_EVENT_SEND) {
		TXC_WARN(txc, "Unexpected %s event: rc=%s\n",
			 cxi_event_to_str(event),
			 cxi_rc_to_str(cxi_event_rc(event)));
		return FI_SUCCESS;
	}

	req->flags &= (FI_RMA | FI_READ | FI_WRITE);

	if (req->rma.local_md)
		cxip_unmap(req->rma.local_md);

	if (req->rma.ibuf)
		cxip_cq_ibuf_free(req->cq, req->rma.ibuf);

	event_rc = cxi_init_event_rc(event);
	if (event_rc == C_RC_OK) {
		if (success_event) {
			ret = cxip_cq_req_complete(req);
			if (ret != FI_SUCCESS)
				TXC_WARN(txc,
					 "Failed to report completion: %d\n",
					 ret);
		}
	} else {
		ret = cxip_cq_req_error(req, 0, FI_EIO, event_rc,
					NULL, 0);
		if (ret != FI_SUCCESS)
			TXC_WARN(txc, "Failed to report error: %d\n", ret);
	}

	ofi_atomic_dec32(&req->rma.txc->otx_reqs);
	cxip_cq_req_free(req);

	return FI_SUCCESS;
}

static int cxip_rma_emit_dma(struct cxip_txc *txc, const void *buf, size_t len,
			     struct cxip_mr *mr, union c_fab_addr *dfa,
			     uint8_t *idx_ext, uint64_t addr, uint64_t key,
			     uint64_t data, uint64_t flags, void *context,
			     bool write, bool unr, uint32_t tclass,
			     enum cxi_traffic_class_type tc_type,
			     bool triggered, uint64_t trig_thresh,
			     struct cxip_cntr *trig_cntr,
			     struct cxip_cntr *comp_cntr)
{
	struct cxip_req *req = NULL;
	struct cxip_md *dma_md = NULL;
	void *dma_buf;
	struct c_full_dma_cmd dma_cmd = {};
	struct c_ct_cmd ct_cmd;
	int ret;
	struct cxip_domain *dom = txc->domain;
	struct cxip_cmdq *cmdq = triggered ? dom->trig_cmdq : txc->tx_cmdq;
	struct cxip_cntr *cntr;
	void *inject_req;

	/* MR desc cannot be value unless hybrid MR desc is enabled. */
	if (!dom->hybrid_mr_desc)
		mr = NULL;

	/* DMA commands always require a request structure regardless if
	 * FI_COMPLETION is set. This is due to the provider doing internally
	 * memory registration and having to clean up the registration on DMA
	 * operation completion.
	 */
	if ((len && (flags & FI_INJECT)) || (flags & FI_COMPLETION) || !mr) {
		req = cxip_cq_req_alloc(txc->send_cq, 0, txc);
		if (!req) {
			ret = -FI_EAGAIN;
			TXC_WARN(txc, "Failed to allocate request: %d:%s\n",
					ret, fi_strerror(-ret));
			goto err;
		}

		req->context = (uint64_t)context;
		req->cb = cxip_rma_cb;
		req->flags = FI_RMA | (write ? FI_WRITE : FI_READ) |
			(flags & FI_COMPLETION);
		req->rma.txc = txc;
		req->type = CXIP_REQ_RMA;
		req->trig_cntr = trig_cntr;
	}

	if (len) {
		/* If the operation is an DMA inject operation (which can occur
		 * when doing RMA commands to unoptimized MRs), a provider
		 * bounce buffer is always needed to store the user payload.
		 *
		 * Always prefer user provider MR over internally mapping the
		 * buffer.
		 */
		if (flags & FI_INJECT) {
			assert(req != NULL);

			req->rma.ibuf = cxip_cq_ibuf_alloc(txc->send_cq);
			if (!req->rma.ibuf) {
				ret = -FI_ENOMEM;
				TXC_WARN(txc,
					"Failed to allocate bounce buffer: %d:%s\n",
					ret, fi_strerror(-ret));
				goto err_free_cq_req;
			}

			ret = cxip_txc_copy_from_hmem(txc, NULL, req->rma.ibuf,
						      buf, len);
			if (ret){
				TXC_WARN(txc,
					 "cxip_txc_copy_from_hmem failed: %d:%s\n",
					 ret, fi_strerror(-ret));
				goto err_free_rma_buf;
			}

			dma_buf = (void *)req->rma.ibuf;
			dma_md = cxip_cq_ibuf_md(req->rma.ibuf);
		} else if (mr) {
			dma_buf = (void *)buf;
			dma_md = mr->md;
		} else {
			assert(req != NULL);

			ret = cxip_map(dom, buf, len, 0, &req->rma.local_md);
			if (ret) {
				TXC_WARN(txc, "Failed to map buffer: %d:%s\n",
					ret, fi_strerror(-ret));
				goto err_free_cq_req;
			}

			dma_buf = (void *)buf;
			dma_md = req->rma.local_md;
		}
	}

	/* Build the DMA command before taking the command queue lock. */
	dma_cmd.command.cmd_type = C_CMD_TYPE_DMA;
	dma_cmd.index_ext = *idx_ext;
	dma_cmd.event_send_disable = 1;
	dma_cmd.dfa = *dfa;

	ret = cxip_adjust_remote_offset(&addr, key);
	if (ret) {
		TXC_WARN(txc, "Remote offset overflow\n");
		goto err_free_cq_req;
	}
	dma_cmd.remote_offset = addr;
	dma_cmd.eq = cxip_cq_tx_eqn(txc->send_cq);
	dma_cmd.match_bits = key;

	if (req) {
		dma_cmd.user_ptr = (uint64_t)req;
	} else {
		if (write)
			inject_req = cxip_rma_write_selective_completion_req(txc);
		else
			inject_req = cxip_rma_read_selective_completion_req(txc);

		if (!inject_req) {
			ret = -FI_EAGAIN;
			TXC_WARN(txc,
				 "Failed to allocate inject request: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err_free_rma_buf;
		}

		dma_cmd.user_ptr = (uint64_t)inject_req;
		dma_cmd.event_success_disable = 1;
	}

	if (!unr)
		dma_cmd.restricted = 1;

	if (write) {
		dma_cmd.command.opcode = C_CMD_PUT;

		/* Triggered DMA operations have their own completion counter
		 * and the one associated with the TXC cannot be used.
		 */
		cntr = triggered ? comp_cntr : txc->write_cntr;
		if (cntr) {
			dma_cmd.event_ct_ack = 1;
			dma_cmd.ct = cntr->ct->ctn;
		}

		if (flags & (FI_DELIVERY_COMPLETE | FI_MATCH_COMPLETE))
			dma_cmd.flush = 1;
	} else {
		dma_cmd.command.opcode = C_CMD_GET;

		/* Triggered DMA operations have their own completion counter
		 * and the one associated with the TXC cannot be used.
		 */
		cntr = triggered ? comp_cntr : txc->read_cntr;
		if (cntr) {
			dma_cmd.event_ct_reply = 1;
			dma_cmd.ct = cntr->ct->ctn;
		}
	}

	/* Only need to fill if DMA command address fields if MD is valid. */
	if (dma_md) {
		dma_cmd.lac = dma_md->md->lac;
		dma_cmd.local_addr = CXI_VA_TO_IOVA(dma_md->md, dma_buf);
		dma_cmd.request_len = len;
	}

	/* Triggered operations do not support changing of traffic classes.
	 * Thus, communication profile cannot be changed.
	 */
	ofi_spin_lock(&cmdq->lock);

	if (triggered) {
		memset(&ct_cmd, 0, sizeof(ct_cmd));
		ct_cmd.trig_ct = trig_cntr->ct->ctn,
		ct_cmd.threshold = trig_thresh,

		ret = cxi_cq_emit_trig_full_dma(cmdq->dev_cmdq, &ct_cmd,
						&dma_cmd);
		if (ret) {
			TXC_WARN(txc,
				 "Failed to emit trigger dma command: %d:%s\n",
				 ret, fi_strerror(-ret));

			/* Always return -FI_EAGAIN for this failure. */
			ret = -FI_EAGAIN;
			goto err_cq_unlock;
		}
	} else {
		/* Ensure correct traffic class is used. */
		ret = cxip_txq_cp_set(cmdq, txc->ep_obj->auth_key.vni,
				      cxip_ofi_to_cxi_tc(tclass), tc_type);
		if (ret) {
			TXC_WARN(txc, "Failed to set traffic class: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err_cq_unlock;
		}

		/* Honor fence if requested. */
		if (flags & (FI_FENCE | FI_CXI_WEAK_FENCE)) {
			ret = cxi_cq_emit_cq_cmd(cmdq->dev_cmdq,
						 C_CMD_CQ_FENCE);
			if (ret) {
				TXC_WARN(txc,
					 "Failed to issue fence command: %d:%s\n",
					 ret, fi_strerror(-ret));

				/* Always return -FI_EAGAIN for this failure. */
				ret = -FI_EAGAIN;
				goto err_cq_unlock;
			}
		}

		/* Finally emit the RMA DMA command. */
		ret = cxi_cq_emit_dma(cmdq->dev_cmdq, &dma_cmd);
		if (ret) {
			TXC_WARN(txc, "Failed to emit idc_put command: %d:%s\n",
				 ret, fi_strerror(-ret));

			/* Always return -FI_EAGAIN for this failure. */
			ret = -FI_EAGAIN;
			goto err_cq_unlock;
		}
	}

	/* Kick the command queue. */
	cxip_txq_ring(cmdq, !!(flags & FI_MORE),
		      ofi_atomic_get32(&txc->otx_reqs));

	if (req)
		ofi_atomic_inc32(&txc->otx_reqs);

	ofi_spin_unlock(&cmdq->lock);

	return FI_SUCCESS;

err_cq_unlock:
	ofi_spin_unlock(&cmdq->lock);
err_free_rma_buf:
	if (req && req->rma.ibuf)
		cxip_cq_ibuf_free(txc->send_cq, req->rma.ibuf);
err_free_cq_req:
	if (req)
		cxip_cq_req_free(req);
err:
	return ret;
}

static int cxip_rma_emit_idc(struct cxip_txc *txc, const void *buf, size_t len,
			     union c_fab_addr *dfa, uint8_t *idx_ext,
			     uint64_t addr, uint64_t key, uint64_t data,
			     uint64_t flags, void *context, bool unr,
			     uint32_t tclass,
			     enum cxi_traffic_class_type tc_type)
{
	int ret;
	struct cxip_req *req = NULL;
	void *hmem_buf = NULL;
	void *idc_buf;
	struct c_cstate_cmd cstate_cmd = {};
	struct c_idc_put_cmd idc_put = {};
	void *inject_req;
	struct cxip_cmdq *cmdq = txc->tx_cmdq;

	/* IDCs must be traffic if the user requests a completion event. */
	if (flags & FI_COMPLETION) {
		req = cxip_cq_req_alloc(txc->send_cq, 0, txc);
		if (!req) {
			ret = -FI_EAGAIN;
			TXC_WARN(txc, "Failed to allocate request: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err;
		}

		req->context = (uint64_t)context;
		req->cb = cxip_rma_cb;
		req->flags = FI_RMA | FI_WRITE | (flags & FI_COMPLETION);
		req->rma.txc = txc;
		req->type = CXIP_REQ_RMA;
	}

	/* If HMEM is request and since the buffer type may not be host memory,
	 * doing a memcpy could result in a segfault. Thus, an HMEM bounce
	 * buffer is required to ensure IDC payload is in host memory.
	 */
	if (txc->hmem && len) {
		hmem_buf = cxip_cq_ibuf_alloc(txc->send_cq);
		if (!hmem_buf) {
			ret = -FI_ENOMEM;
			TXC_WARN(txc,
				 "Failed to allocate bounce buffer: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err_free_cq_req;
		}

		ret = cxip_txc_copy_from_hmem(txc, NULL, hmem_buf, buf, len);
		if (ret) {
			TXC_WARN(txc,
				 "cxip_txc_copy_from_hmem failed: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err_free_hmem_buf;
		}

		idc_buf = hmem_buf;
	} else {
		idc_buf = (void *)buf;
	}

	/* Build up the c-state command before taking the command queue lock. */
	cstate_cmd.event_send_disable = 1;
	cstate_cmd.index_ext = *idx_ext;
	cstate_cmd.eq = cxip_cq_tx_eqn(txc->send_cq);

	if (flags & (FI_DELIVERY_COMPLETE | FI_MATCH_COMPLETE))
		cstate_cmd.flush = 1;

	if (!unr)
		cstate_cmd.restricted = 1;

	if (txc->write_cntr) {
		cstate_cmd.event_ct_ack = 1;
		cstate_cmd.ct = txc->write_cntr->ct->ctn;
	}

	/* If the user has not request a completion, success events will be
	 * disabled. But, if for some reason the operation completes with an
	 * error, an event will occur. For this case, a TXC inject request is
	 * allocated. This request enables the reporting of failed operation to
	 *  the completion queue. This request is freed when the TXC is closed.
	 */
	if (req) {
		cstate_cmd.user_ptr = (uint64_t)req;
	} else {
		inject_req = cxip_rma_write_selective_completion_req(txc);
		if (!inject_req) {
			ret = -FI_EAGAIN;
			TXC_WARN(txc,
				 "Failed to allocate inject request: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err_free_hmem_buf;
		}

		cstate_cmd.user_ptr = (uint64_t)inject_req;
		cstate_cmd.event_success_disable = 1;
	}

	/* Build up the IDC command before taking the command lock queue. */
	idc_put.idc_header.dfa = *dfa;

	ret = cxip_adjust_remote_offset(&addr, key);
	if (ret) {
		TXC_WARN(txc, "Remote offset overflow\n");
		goto err_free_hmem_buf;
	}
	idc_put.idc_header.remote_offset = addr;

	/* Emit all commands. Note that if any of the operations do not take,
	 * no cleaning up of the command queue is needed.
	 */
	ofi_spin_lock(&cmdq->lock);

	/* Ensure correct traffic class is used. */
	ret = cxip_txq_cp_set(cmdq, txc->ep_obj->auth_key.vni,
			      cxip_ofi_to_cxi_tc(tclass), tc_type);
	if (ret) {
		TXC_WARN(txc, "Failed to set traffic class: %d:%s\n", ret,
			 fi_strerror(-ret));
		goto err_cq_unlock;
	}

	/* Honor fence if requested. */
	if (flags & (FI_FENCE | FI_CXI_WEAK_FENCE)) {
		ret = cxi_cq_emit_cq_cmd(cmdq->dev_cmdq, C_CMD_CQ_FENCE);
		if (ret) {
			TXC_WARN(txc, "Failed to issue fence command: %d:%s\n",
				 ret, fi_strerror(-ret));

			/* Always return -FI_EAGAIN for this failure. */
			ret = -FI_EAGAIN;
			goto err_cq_unlock;
		}
	}

	/* Update the hardware command queue state to ensure correct fields are
	 * associated with the IDC command.
	 */
	ret = cxip_cmdq_emit_c_state(cmdq, &cstate_cmd);
	if (ret) {
		TXC_WARN(txc, "Failed to emit c_state command: %d:%s\n", ret,
			 fi_strerror(-ret));
		goto err_cq_unlock;
	}

	/* Update the IDC put command and payload in the command queue. */
	ret = cxi_cq_emit_idc_put(cmdq->dev_cmdq, &idc_put, idc_buf, len);
	if (ret) {
		TXC_WARN(txc, "Failed to emit idc_put command: %d:%s\n", ret,
			 fi_strerror(-ret));

		/* Always return -FI_EAGAIN for this failure. */
		ret = -FI_EAGAIN;
		goto err_cq_unlock;
	}

	/* Kick the command queue. */
	cxip_txq_ring(cmdq, !!(flags & FI_MORE),
		      ofi_atomic_get32(&txc->otx_reqs));
	if (req)
		ofi_atomic_inc32(&txc->otx_reqs);

	ofi_spin_unlock(&cmdq->lock);

	if (hmem_buf)
		cxip_cq_ibuf_free(txc->send_cq, hmem_buf);

	return FI_SUCCESS;

err_cq_unlock:
	ofi_spin_unlock(&cmdq->lock);
err_free_hmem_buf:
	if (hmem_buf)
		cxip_cq_ibuf_free(txc->send_cq, hmem_buf);
err_free_cq_req:
	if (req)
		cxip_cq_req_free(req);
err:
	return ret;
}

static bool cxip_rma_is_unrestricted(struct cxip_txc *txc, uint64_t key,
				     uint64_t msg_order, bool write)
{
	/* Unoptimized keys are implemented with match bits and must always be
	 * unrestricted.
	 */
	if (!txc->domain->mr_util->key_is_opt(key))
		return true;

	/* If FI_RMA_EVENTS are requested, it is assumed that the user will bind
	 * remote MRs to a counter or support remote RMA events. If this is the
	 * case, unrestircted must always been used.
	 */
	if (txc->ep_obj->caps & FI_RMA_EVENT)
		return true;

	/* If the operation is an RMA write and the user has requested fabric
	 * write after write ordering, unrestricted must be used.
	 */
	if (write && msg_order & (FI_ORDER_WAW | FI_ORDER_RMA_WAW))
		return true;

	return false;
}

static bool cxip_rma_is_idc(struct cxip_txc *txc, uint64_t key, size_t len,
			    bool write, bool triggered, bool unr)
{
	size_t max_idc_size = unr ? CXIP_INJECT_SIZE : C_MAX_IDC_PAYLOAD_RES;

	/* IDC commands are not supported for unoptimized MR since the IDC
	 * small message format does not support remote offset which is needed
	 * for RMA commands.
	 */
	if (!txc->domain->mr_util->key_is_opt(key))
		return false;

	/* IDC commands are only support with RMA writes. */
	if (!write)
		return false;

	/* IDC commands only support a limited payload size. */
	if (len > max_idc_size)
		return false;

	/* Triggered operations never can be issued with an IDC. */
	if (triggered)
		return false;

	return true;
}

/*
 * cxip_rma_common() - Perform an RMA operation.
 *
 * Common RMA function. Performs RMA reads and writes of all kinds.
 *
 * Generally, operations are supported by Cassini DMA commands. IDC commands
 * are used instead for Write operations smaller than the maximum IDC payload
 * size.
 *
 * If the FI_COMPLETION flag is specified, the operation will generate a
 * libfabric completion event. If an event is not requested and an IDC command
 * is used, hardware success events will be suppressed. If a completion is
 * required but an IDC can't be used, the provider tracks the request
 * internally, but will suppress the libfabric event. The provider must track
 * DMA commands in order to clean up the source buffer mapping on completion.
 */
ssize_t cxip_rma_common(enum fi_op_type op, struct cxip_txc *txc,
			const void *buf, size_t len, void *desc,
			fi_addr_t tgt_addr, uint64_t addr, uint64_t key,
			uint64_t data, uint64_t flags, uint32_t tclass,
			uint64_t msg_order, void *context,
			bool triggered, uint64_t trig_thresh,
			struct cxip_cntr *trig_cntr,
			struct cxip_cntr *comp_cntr)
{
	struct cxip_addr caddr;
	union c_fab_addr dfa;
	uint8_t idx_ext;
	uint32_t pid_idx;
	enum cxi_traffic_class_type tc_type;
	bool write = op == FI_OP_WRITE;
	bool unr;
	bool idc;
	int ret;

	if (!txc->enabled) {
		TXC_WARN(txc, "TXC not enabled\n");
		return -FI_EOPBADSTATE;
	}

	if (!ofi_rma_initiate_allowed(txc->attr.caps & ~FI_ATOMIC)) {
		TXC_WARN(txc, "TXC caps do not support RMA initiator\n");
		return -FI_ENOPROTOOPT;
	}

	if (len && !buf) {
		TXC_WARN(txc, "Invalid buffer\n");
		return -FI_EINVAL;
	}

	if ((flags & FI_INJECT) && len > CXIP_INJECT_SIZE) {
		TXC_WARN(txc, "RMA inject size exceeds limit\n");
		return -FI_EMSGSIZE;
	}

	if (len > CXIP_EP_MAX_MSG_SZ) {
		TXC_WARN(txc, "RMA length exceeds limit\n");
		return -FI_EMSGSIZE;
	}

	if (!txc->domain->mr_util->key_is_valid(key)) {
		TXC_WARN(txc, "Invalid remote key: 0x%lx\n", key);
		return -FI_EKEYREJECTED;
	}

	unr = cxip_rma_is_unrestricted(txc, key, msg_order, write);
	idc = cxip_rma_is_idc(txc, key, len, write, triggered, unr);

	/* To prevent CQ overrun, perform a sanity check to ensure there is
	 * enough space for a potential event.
	 */
	if (cxip_cq_saturated(txc->send_cq)) {
		TXC_DBG(txc, "CQ saturated\n");
		return -FI_EAGAIN;
	}

	/* Build target network address. */
	ret = _cxip_av_lookup(txc->ep_obj->av, tgt_addr, &caddr);
	if (ret) {
		TXC_WARN(txc, "Failed to look up FI addr: %d:%s\n",
			 ret, fi_strerror(-ret));
		return ret;
	}

	pid_idx = txc->domain->mr_util->key_to_ptl_idx(txc->domain, key, write);
	cxi_build_dfa(caddr.nic, caddr.pid, txc->pid_bits, pid_idx, &dfa,
		      &idx_ext);

	/* Select the correct traffic class type within a traffic class. */
	if (!unr && (flags & FI_CXI_HRP))
		tc_type = CXI_TC_TYPE_HRP;
	else if (!unr)
		tc_type = CXI_TC_TYPE_RESTRICTED;
	else
		tc_type = CXI_TC_TYPE_DEFAULT;

	/* IDC commands are preferred wherever possible since the payload is
	 * written with the command thus avoiding all memory registration. In
	 * addition, this allows for success events to be surpressed if
	 * FI_COMPLETION is not requested.
	 */
	if (idc)
		ret = cxip_rma_emit_idc(txc, buf, len, &dfa, &idx_ext, addr,
					key, data, flags, context, unr,
					tclass, tc_type);
	else
		ret = cxip_rma_emit_dma(txc, buf, len, desc, &dfa, &idx_ext,
					addr, key, data, flags, context, write,
					unr, tclass, tc_type,
					triggered, trig_thresh,
					trig_cntr, comp_cntr);

	if (ret)
		TXC_WARN(txc,
			 "%s RMA %s failed: buf=%p len=%lu rkey=%#lx roffset=%#lx nic=%#x pid=%u pid_idx=%u\n",
			 idc ? "IDC" : "DMA", write ? "write" : "read",
			 buf, len, key, addr, caddr.nic, caddr.pid, pid_idx);
	else
		TXC_DBG(txc,
			"%s RMA %s emitted: buf=%p len=%lu rkey=%#lx roffset=%#lx nic=%#x pid=%u pid_idx=%u\n",
			idc ? "IDC" : "DMA", write ? "write" : "read",
			buf, len, key, addr, caddr.nic, caddr.pid, pid_idx);

	return ret;
}

/*
 * Libfabric APIs
 */

static ssize_t cxip_rma_write(struct fid_ep *ep, const void *buf, size_t len,
			      void *desc, fi_addr_t dest_addr, uint64_t addr,
			      uint64_t key, void *context)
{
	struct cxip_txc *txc;
	struct fi_tx_attr *attr;

	if (cxip_fid_to_tx_info(ep, &txc, &attr) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_rma_common(FI_OP_WRITE, txc, buf, len, desc, dest_addr,
			       addr, key, 0, attr->op_flags, attr->tclass,
			       attr->msg_order, context, false, 0, NULL, NULL);
}

static ssize_t cxip_rma_writev(struct fid_ep *ep, const struct iovec *iov,
			       void **desc, size_t count, fi_addr_t dest_addr,
			       uint64_t addr, uint64_t key, void *context)
{
	struct cxip_txc *txc;
	struct fi_tx_attr *attr;

	if (cxip_fid_to_tx_info(ep, &txc, &attr) != FI_SUCCESS)
		return -FI_EINVAL;

	if (!iov || count != 1)
		return -FI_EINVAL;

	return cxip_rma_common(FI_OP_WRITE, txc, iov[0].iov_base,
			       iov[0].iov_len, desc ? desc[0] : NULL, dest_addr,
			       addr, key, 0, attr->op_flags, attr->tclass,
			       attr->msg_order, context, false, 0, NULL, NULL);
}

static ssize_t cxip_rma_writemsg(struct fid_ep *ep,
				 const struct fi_msg_rma *msg, uint64_t flags)
{
	struct cxip_txc *txc;
	struct fi_tx_attr *attr;

	if (!msg || !msg->msg_iov || !msg->rma_iov ||
	    msg->iov_count != 1 || msg->rma_iov_count != 1)
		return -FI_EINVAL;

	if (flags & ~(CXIP_WRITEMSG_ALLOWED_FLAGS | FI_CXI_HRP |
		      FI_CXI_WEAK_FENCE))
		return -FI_EBADFLAGS;

	if (cxip_fid_to_tx_info(ep, &txc, &attr) != FI_SUCCESS)
		return -FI_EINVAL;

	if (flags & FI_FENCE && !(txc->attr.caps & FI_FENCE))
		return -FI_EINVAL;

	/* If selective completion is not requested, always generate
	 * completions.
	 */
	if (!txc->selective_completion)
		flags |= FI_COMPLETION;

	return cxip_rma_common(FI_OP_WRITE, txc, msg->msg_iov[0].iov_base,
			       msg->msg_iov[0].iov_len,
			       msg->desc ? msg->desc[0] : NULL, msg->addr,
			       msg->rma_iov[0].addr, msg->rma_iov[0].key,
			       msg->data, flags, attr->tclass, attr->msg_order,
			       msg->context, false, 0, NULL, NULL);
}

ssize_t cxip_rma_inject(struct fid_ep *ep, const void *buf, size_t len,
			fi_addr_t dest_addr, uint64_t addr, uint64_t key)
{
	struct cxip_txc *txc;
	struct fi_tx_attr *attr;

	if (cxip_fid_to_tx_info(ep, &txc, &attr) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_rma_common(FI_OP_WRITE, txc, buf, len, NULL, dest_addr,
			       addr, key, 0, FI_INJECT, attr->tclass,
			       attr->msg_order, NULL, false, 0, NULL, NULL);
}

static ssize_t cxip_rma_read(struct fid_ep *ep, void *buf, size_t len,
			     void *desc, fi_addr_t src_addr, uint64_t addr,
			     uint64_t key, void *context)
{
	struct cxip_txc *txc;
	struct fi_tx_attr *attr;

	if (cxip_fid_to_tx_info(ep, &txc, &attr) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_rma_common(FI_OP_READ, txc, buf, len, desc, src_addr,
			       addr, key, 0, attr->op_flags, attr->tclass,
			       attr->msg_order, context, false, 0, NULL, NULL);
}

static ssize_t cxip_rma_readv(struct fid_ep *ep, const struct iovec *iov,
			      void **desc, size_t count, fi_addr_t src_addr,
			      uint64_t addr, uint64_t key, void *context)
{
	struct cxip_txc *txc;
	struct fi_tx_attr *attr;

	if (cxip_fid_to_tx_info(ep, &txc, &attr) != FI_SUCCESS)
		return -FI_EINVAL;

	if (!iov || count != 1)
		return -FI_EINVAL;

	return cxip_rma_common(FI_OP_READ, txc, iov[0].iov_base, iov[0].iov_len,
			       desc ? desc[0] : NULL, src_addr, addr, key, 0,
			       attr->op_flags, attr->tclass, attr->msg_order,
			       context, false, 0, NULL, NULL);
}

static ssize_t cxip_rma_readmsg(struct fid_ep *ep,
				const struct fi_msg_rma *msg, uint64_t flags)
{
	struct cxip_txc *txc;
	struct fi_tx_attr *attr;

	if (!msg || !msg->msg_iov || !msg->rma_iov ||
	    msg->iov_count != 1 || msg->rma_iov_count != 1)
		return -FI_EINVAL;

	if (flags & ~CXIP_READMSG_ALLOWED_FLAGS)
		return -FI_EBADFLAGS;

	if (cxip_fid_to_tx_info(ep, &txc, &attr) != FI_SUCCESS)
		return -FI_EINVAL;

	if (flags & FI_FENCE && !(txc->attr.caps & FI_FENCE))
		return -FI_EINVAL;

	/* If selective completion is not requested, always generate
	 * completions.
	 */
	if (!txc->selective_completion)
		flags |= FI_COMPLETION;

	return cxip_rma_common(FI_OP_READ, txc, msg->msg_iov[0].iov_base,
			       msg->msg_iov[0].iov_len,
			       msg->desc ? msg->desc[0] : NULL, msg->addr,
			       msg->rma_iov[0].addr, msg->rma_iov[0].key,
			       msg->data, flags, attr->tclass, attr->msg_order,
			       msg->context, false, 0, NULL, NULL);
}

struct fi_ops_rma cxip_ep_rma = {
	.size = sizeof(struct fi_ops_rma),
	.read = cxip_rma_read,
	.readv = cxip_rma_readv,
	.readmsg = cxip_rma_readmsg,
	.write = cxip_rma_write,
	.writev = cxip_rma_writev,
	.writemsg = cxip_rma_writemsg,
	.inject = cxip_rma_inject,
	.injectdata = fi_no_rma_injectdata,
	.writedata = fi_no_rma_writedata,
};
