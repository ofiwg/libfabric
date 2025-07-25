/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include <infiniband/efadv.h>

#include "efa.h"
#include "efa_av.h"
#include "efa_cqdirect.h"
#include "efa_cqdirect_efa_io_defs.h"
#include "efa_cqdirect_structs.h"

#if HAVE_EFA_CQ_DIRECT

#include "efa_cqdirect_entry.h"

static void efa_cq_fill_cqdirect_pfns(struct efa_ibv_cq *ibv_cq)
{
	ibv_cq->start_poll = efa_cqdirect_start_poll;
	ibv_cq->next_poll = efa_cqdirect_next_poll;
	ibv_cq->end_poll = efa_cqdirect_end_poll;
	ibv_cq->wc_is_unsolicited = efa_cqdirect_wc_is_unsolicited;
	ibv_cq->read_byte_len = efa_cqdirect_wc_read_byte_len;
	ibv_cq->read_imm_data = efa_cqdirect_wc_read_imm_data;
	ibv_cq->read_opcode = efa_cqdirect_wc_read_opcode;
	ibv_cq->read_slid = efa_cqdirect_wc_read_slid;
	ibv_cq->read_src_qp = efa_cqdirect_wc_read_src_qp;
	ibv_cq->read_qp_num = efa_cqdirect_wc_read_qp_num;
	ibv_cq->read_vendor_err = efa_cqdirect_wc_read_vendor_err;
	ibv_cq->read_wc_flags = efa_cqdirect_wc_read_wc_flags;
	ibv_cq->read_sgid = efa_cqdirect_wc_read_sgid;
};

static void efa_qp_fill_cqdirect_pfns(struct efa_qp *qp)
{
	qp->post_recv = efa_cqdirect_post_recv;
	qp->wr_complete = efa_cqdirect_wr_complete;
	qp->wr_rdma_read = efa_cqdirect_wr_rdma_read;
	qp->wr_rdma_write = efa_cqdirect_wr_rdma_write;
	qp->wr_rdma_write_imm = efa_cqdirect_wr_rdma_write_imm;
	qp->wr_send = efa_cqdirect_wr_send;
	qp->wr_send_imm = efa_cqdirect_wr_send_imm;
	qp->wr_set_inline_data_list = efa_cqdirect_wr_set_inline_data_list;
	qp->wr_set_sge_list = efa_cqdirect_wr_set_sge_list;
	qp->wr_set_ud_addr = efa_cqdirect_wr_set_ud_addr;
	qp->wr_start = efa_cqdirect_wr_start;
};

int efa_cqdirect_qp_initialize(struct efa_qp *efa_qp)
{
	/* Called during efa_base_ep_create_qp.
	 * See also rdma-core/providers/efa/verbs.c: efa_setup_qp
	 */
	struct efa_cqdirect_qp *direct_qp = &efa_qp->cqdirect_qp;
	struct efa_base_ep *base_ep = efa_qp->base_ep;

	struct efadv_wq_attr sq_attr;
	struct efadv_wq_attr rq_attr;
	int ret = 0;

	memset(&efa_qp->cqdirect_qp, 0, sizeof(efa_qp->cqdirect_qp));

	ret = efadv_query_qp_wqs(efa_qp->ibv_qp, &sq_attr, &rq_attr,
				 sizeof(rq_attr));
	if (ret != FI_SUCCESS)
		return ret;

	direct_qp->rq.buf = rq_attr.buffer;
	direct_qp->rq.wq.db = rq_attr.doorbell;
	direct_qp->rq.wq.wqe_size = rq_attr.entry_size;
	efa_cqdirect_wq_initialize(&direct_qp->rq.wq, rq_attr.num_entries,
				   &base_ep->util_ep.lock);

	direct_qp->sq.desc = sq_attr.buffer;
	direct_qp->sq.wq.phase = 0;
	direct_qp->sq.wq.db = sq_attr.doorbell;
	direct_qp->sq.num_wqe_pending = 0;

	direct_qp->sq.wq.wqe_size = sq_attr.entry_size;
	efa_cqdirect_wq_initialize(&direct_qp->sq.wq, sq_attr.num_entries,
				   &base_ep->util_ep.lock);

	/* see efa_qp_init_indices */

	efa_qp->cqdirect_enabled = 1;
	efa_qp_fill_cqdirect_pfns(efa_qp);
	return ret;
}

/**
 * @brief Clean up the resources created for direct qp
 * called during efa_qp_destruct
 * @param efa_qp ptr to efa_qp
 */
void efa_cqdirect_qp_finalize(struct efa_qp *efa_qp)
{
	struct efa_cqdirect_qp *direct_qp = &efa_qp->cqdirect_qp;

	efa_cqdirect_wq_finalize(&direct_qp->sq.wq);
	efa_cqdirect_wq_finalize(&direct_qp->rq.wq);
}

int efa_cqdirect_cq_initialize(struct efa_cq *efa_cq)
{
	struct efadv_cq_attr attr = {0};
	struct efa_cqdirect_cq *cqdirect = &efa_cq->ibv_cq.cqdirect;
	int ret;

	memset(cqdirect, 0, sizeof(*cqdirect));

	/**
	 * We cannot use direct cq when hardware is still using sub cq.
	 * Also disable direct cq when it's specified by environment
	 */
	if (!efa_env.use_direct_cq_ops || efa_device_use_sub_cq()) {
		/* nothing to do.  Not using directcq.*/
		return FI_SUCCESS;
	}

	ret = efadv_query_cq(ibv_cq_ex_to_cq(efa_cq->ibv_cq.ibv_cq_ex), &attr,
			     sizeof(attr));
	if (ret != FI_SUCCESS) {
		return ret;
	}

	efa_cq->ibv_cq.cqdirect_enabled = 1;
	efa_cq_fill_cqdirect_pfns(&efa_cq->ibv_cq);
	cqdirect->buffer = attr.buffer;
	cqdirect->entry_size = attr.entry_size;
	cqdirect->num_entries = attr.num_entries;

	cqdirect->phase = 1;
	cqdirect->consumed_cnt = 0;
	cqdirect->qmask = cqdirect->num_entries - 1;

	return FI_SUCCESS;
}

#endif /* end of HAVE_EFA_CQ_DIRECT */
