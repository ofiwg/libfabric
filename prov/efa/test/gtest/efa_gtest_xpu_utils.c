/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_av.h"
#include "efa_cntr.h"
#include "efa_cq.h"
#include "efa_mr.h"
#include "efa_xpu.h"
#include "efa_gtest_xpu_utils.h"

int efa_test_xpu_device_support(void)
{
	/* What the provider advertises FI_XPU on: see efa_prov.c. */
	return efa_device_support_data_path_direct(&g_efa_selected_device_list[0]);
}

int efa_test_xpu_set_use_data_path_direct(int value)
{
	int prev = efa_env.use_data_path_direct;

	efa_env.use_data_path_direct = value;
	return prev;
}

int efa_test_xpu_set_ep_data_path_direct(struct fid_ep *ep, int value)
{
	struct efa_base_ep *base_ep =
		container_of(ep, struct efa_base_ep, util_ep.ep_fid);
	int prev;

	if (!base_ep->qp)
		return -1;

	prev = base_ep->qp->data_path_direct_enabled;
	base_ep->qp->data_path_direct_enabled = value;
	return prev;
}

int efa_test_xpu_fill_wq_attrs(void *sq_attr, void *rq_attr,
			       const struct efa_test_xpu_wq_geometry *sq,
			       const struct efa_test_xpu_wq_geometry *rq)
{
#if HAVE_EFADV_QUERY_QP_WQS
	struct efadv_wq_attr *attrs[2] = { (struct efadv_wq_attr *) sq_attr,
					   (struct efadv_wq_attr *) rq_attr };
	const struct efa_test_xpu_wq_geometry *geos[2] = { sq, rq };
	int i;

	for (i = 0; i < 2; i++) {
		memset(attrs[i], 0, sizeof(*attrs[i]));
		attrs[i]->num_entries = geos[i]->num_entries;
		attrs[i]->entry_size = geos[i]->entry_size;
		attrs[i]->max_batch = geos[i]->max_batch;
		attrs[i]->buffer = (uint8_t *) geos[i]->buffer;
		attrs[i]->doorbell = (uint32_t *) geos[i]->doorbell;
#if HAVE_EFADV_WQ_ATTR_CAPS
		if (geos[i]->req_id_64_bit)
			attrs[i]->caps |= EFADV_WQ_CAPS_64_BIT_REQ_ID;
#endif
	}

	return 0;
#else
	return -FI_EOPNOTSUPP;
#endif
}

int efa_test_xpu_fill_cq_attr(void *cq_attr, uint32_t num_entries,
			      uint32_t entry_size, void *buffer)
{
#if HAVE_EFADV_QUERY_CQ
	struct efadv_cq_attr *attr = (struct efadv_cq_attr *) cq_attr;

	memset(attr, 0, sizeof(*attr));
	attr->num_entries = num_entries;
	attr->entry_size = entry_size;
	attr->buffer = (uint8_t *) buffer;

	return 0;
#else
	return -FI_EOPNOTSUPP;
#endif
}

int efa_test_xpu_install_cq_state(struct fid_cq *cq_fid,
				  struct fid_xpu_ctx *ctx, void *cq_buf_dev)
{
	struct efa_cq *cq = container_of(cq_fid, struct efa_cq, util_cq.cq_fid);

	if (cq->xpu_state)
		return -FI_EALREADY;

	cq->xpu_state = efa_xpu_cq_state_create(ctx);
	if (!cq->xpu_state)
		return -FI_ENOMEM;

	cq->xpu_state->cq_buf_dev = cq_buf_dev;
	return 0;
}

int efa_test_xpu_cq_publishers(struct fid_cq *cq_fid)
{
	return container_of(cq_fid, struct efa_cq, util_cq.cq_fid)
		->xpu_publishers;
}

const void *efa_test_xpu_qp_ctx(struct fid_ep *ep_fid, bool rq)
{
	struct efa_base_ep *base_ep =
		container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);

#if HAVE_EFA_DATA_PATH_DIRECT
	return rq ? base_ep->qp->data_path_direct_qp.rq.xpu_ctx :
		    base_ep->qp->data_path_direct_qp.sq.xpu_ctx;
#else
	(void) base_ep;
	(void) rq;
	return NULL;
#endif
}

void efa_test_xpu_ep_note_done(struct fid_ep *ep_fid, bool rq,
			       uint16_t req_id)
{
	struct efa_base_ep *base_ep =
		container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);

	efa_xpu_ep_note_done(base_ep->xpu_state,
			     rq ? EFA_XPU_RQ_DONE : EFA_XPU_SQ_DONE, req_id);
}

/* End a CQ read: publish the done cursors it advanced. */
void efa_test_xpu_cq_flush(struct fid_cq *cq_fid)
{
	struct efa_cq *cq = container_of(cq_fid, struct efa_cq, util_cq.cq_fid);

	efa_cq_xpu_flush(cq);
}

int efa_test_xpu_install_cntr_state(struct fid_cntr *cntr_fid,
				    struct fid_xpu_ctx *ctx, void *value_dev,
				    void *err_dev)
{
	struct efa_cntr *cntr =
		container_of(cntr_fid, struct efa_cntr, util_cntr.cntr_fid);

	if (cntr->xpu_state)
		return -FI_EALREADY;

	cntr->xpu_state = efa_xpu_cntr_state_create(ctx);
	if (!cntr->xpu_state)
		return -FI_ENOMEM;

	cntr->xpu_state->cntr_value_dev = value_dev;
	cntr->xpu_state->cntr_alloc_addr = value_dev;
	cntr->xpu_state->cntr_err_dev = err_dev;
	cntr->xpu_state->cntr_err_alloc_addr = err_dev;
	return 0;
}

int efa_test_xpu_av_insert_peer(struct fid_ep *ep, struct fid_av *av,
				uint16_t qpn, uint32_t qkey, fi_addr_t *addr)
{
	struct efa_ep_addr raw_addr = { 0 };
	size_t raw_addr_len = sizeof(raw_addr);
	int ret;

	ret = fi_getname(&ep->fid, &raw_addr, &raw_addr_len);
	if (ret)
		return ret;

	raw_addr.qpn = qpn;
	raw_addr.qkey = qkey;

	return fi_av_insert(av, &raw_addr, 1, addr, 0, NULL);
}

int efa_test_xpu_av_ahn(struct fid_av *av_fid, fi_addr_t fi_addr)
{
	struct efa_av *av = container_of(av_fid, struct efa_av, util_av.av_fid);
	struct efa_av_entry *av_entry;

	av_entry = (struct efa_av_entry *) ofi_av_get_addr(&av->util_av,
							  fi_addr);
	if (!av_entry || !av_entry->ah)
		return -1;

	return (int) av_entry->ah->ahn;
}

uint32_t efa_test_xpu_mr_lkey(struct fid_mr *mr)
{
	struct efa_mr *efa_mr =
		container_of(&mr->fid, struct efa_mr, mr_fid.fid);

	return efa_mr->lkey;
}
