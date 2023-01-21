/*
 * (C) Copyright 2022 Hewlett Packard Enterprise Development LP
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
#include "config.h"
#include "cxip.h"

#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_CTRL, __VA_ARGS__)

static void cxip_rdzv_pte_cb(struct cxip_pte *pte, const union c_event *event)
{
	switch (pte->state) {
	case C_PTLTE_ENABLED:
		break;
	default:
		CXIP_FATAL("Unexpected state received: %u\n", pte->state);
	}
}

static bool cxip_rdzv_pte_append_done(struct cxip_rdzv_pte *pte,
				      int expected_success_count)
{
	if (ofi_atomic_get32(&pte->le_linked_success_count) ==
	    expected_success_count)
		return true;

	if (ofi_atomic_get32(&pte->le_linked_failure_count) != 0)
		return true;

	return false;
}

static void cxip_rdzv_pte_src_reqs_free(struct cxip_rdzv_pte *pte)
{
	int i;

	/* The corresponding LE is not freed using an unlink command. Instead,
	 * this logic relies on the freeing of the hardware PtlTE to release the
	 * LEs.
	 */
	for (i = 0; i < RDZV_SRC_LES; i++) {
		if (pte->src_reqs[i])
			cxip_evtq_req_free(pte->src_reqs[i]);
	}
}

int cxip_rdzv_pte_src_req_alloc(struct cxip_rdzv_pte *pte, int lac)
{
	int ret;
	union cxip_match_bits mb;
	union cxip_match_bits ib;
	uint32_t le_flags;
	int expected_success_count;
	struct cxip_req *req;

	/* Reuse a previously allocated request whenever possible. */
	if (pte->src_reqs[lac])
		return FI_SUCCESS;

	ofi_spin_lock(&pte->src_reqs_lock);
	if (pte->src_reqs[lac]) {
		ofi_spin_unlock(&pte->src_reqs_lock);
		return FI_SUCCESS;
	}

	mb.raw = 0;
	mb.rdzv_lac = lac;
	ib.raw = ~0;
	ib.rdzv_lac = 0;
	le_flags = C_LE_UNRESTRICTED_BODY_RO | C_LE_UNRESTRICTED_END_RO |
		C_LE_OP_GET | C_LE_EVENT_UNLINK_DISABLE;

	req = cxip_evtq_req_alloc(&pte->txc->tx_evtq, 1, pte);
	if (!req) {
		ret = -FI_EAGAIN;
		CXIP_WARN("Failed to allocate %d rendezvous source request: %d:%s\n",
			  lac, ret, fi_strerror(-ret));
		goto err_unlock;
	}
	req->cb = cxip_rdzv_pte_src_cb;

	expected_success_count =
		ofi_atomic_get32(&pte->le_linked_success_count) + 1;

	ret = cxip_pte_append(pte->pte, 0, -1ULL, lac, C_PTL_LIST_PRIORITY,
			      req->req_id, mb.raw, ib.raw,
			      CXI_MATCH_ID_ANY, 0, le_flags, NULL,
			      pte->txc->rx_cmdq, true);
	if (ret) {
		CXIP_WARN("Failed to issue %d rendezvous source request LE append: %d:%s\n",
			  lac, ret, fi_strerror(-ret));
		goto err_free_req;
	}

	/* Poll until the LE is linked or a failure occurs. */
	do {
		cxip_cq_progress(pte->txc->send_cq);
		sched_yield();
	} while (!cxip_rdzv_pte_append_done(pte, expected_success_count));

	if (ofi_atomic_get32(&pte->le_linked_failure_count)) {
		ret = -FI_EIO;
		CXIP_WARN("Failed to append %d rendezvous source request LE: %d:%s\n",
			  lac, ret, fi_strerror(-ret));
		goto err_free_req;
	}

	pte->src_reqs[lac] = req;

	ofi_spin_unlock(&pte->src_reqs_lock);

	return FI_SUCCESS;

err_free_req:
	cxip_evtq_req_free(req);
err_unlock:
	ofi_spin_unlock(&pte->src_reqs_lock);

	return ret;
}

static void cxip_rdzv_pte_zbp_req_free(struct cxip_rdzv_pte *pte)
{
	/* The corresponding LE is not freed using an unlink command. Instead,
	 * this logic relies on the freeing of the hardware PtlTE to release the
	 * LEs.
	 */
	cxip_evtq_req_free(pte->zbp_req);
}

static int cxip_rdzv_pte_zbp_req_alloc(struct cxip_rdzv_pte *pte)
{
	uint32_t le_flags = C_LE_UNRESTRICTED_BODY_RO |
		C_LE_UNRESTRICTED_END_RO | C_LE_OP_PUT |
		C_LE_EVENT_UNLINK_DISABLE;
	union cxip_match_bits mb = {
		.le_type = CXIP_LE_TYPE_ZBP,
	};
	union cxip_match_bits ib = {
		.tag = ~0,
		.tx_id = ~0,
		.cq_data = 1,
		.tagged = 1,
		.match_comp = 1,
	};
	int ret;
	int expected_success_count;

	pte->zbp_req = cxip_evtq_req_alloc(&pte->txc->tx_evtq, 1, pte);
	if (!pte->zbp_req) {
		ret = -FI_ENOMEM;
		CXIP_WARN("Failed to allocate zero byte put request: %d:%s\n",
			  ret, fi_strerror(-ret));
		goto err;
	}

	pte->zbp_req->cb = cxip_rdzv_pte_zbp_cb;

	expected_success_count =
		ofi_atomic_get32(&pte->le_linked_success_count) + 1;

	ret = cxip_pte_append(pte->pte, 0, 0, 0, C_PTL_LIST_PRIORITY,
			      pte->zbp_req->req_id, mb.raw, ib.raw,
			      CXI_MATCH_ID_ANY, 0, le_flags, NULL,
			      pte->txc->rx_cmdq, true);
	if (ret) {
		CXIP_WARN("Failed to issue zero byte put LE append: %d:%s\n",
			  ret, fi_strerror(-ret));
		goto err_free_req;
	}

	/* Poll until the LE is linked or a failure occurs. */
	do {
		cxip_cq_progress(pte->txc->send_cq);
		sched_yield();
	} while (!cxip_rdzv_pte_append_done(pte, expected_success_count));

	if (ofi_atomic_get32(&pte->le_linked_failure_count)) {
		ret = -FI_EIO;
		CXIP_WARN("Failed to append zero byte put LE: %d:%s\n", ret,
			  fi_strerror(-ret));
		goto err_free_req;
	}

	return FI_SUCCESS;

err_free_req:
	cxip_evtq_req_free(pte->zbp_req);
err:
	return ret;
}

void cxip_rdzv_pte_free(struct cxip_rdzv_pte *pte)
{
	/* Freeing the PtlTE causes the PtlTE to be reset and all LEs to be
	 * freed. Thus, no need to issue disable and/or unlink commands.
	 */
	cxip_pte_free(pte->pte);

	/* Flush the CQ to ensure any events referencing the rendezvous requests
	 * are processed.
	 */
	cxip_cq_progress(pte->txc->send_cq);

	/* Release all the rendezvous requests. */
	cxip_rdzv_pte_src_reqs_free(pte);
	cxip_rdzv_pte_zbp_req_free(pte);

	free(pte);
}

int cxip_rdzv_pte_alloc(struct cxip_txc *txc, struct cxip_rdzv_pte **rdzv_pte)
{
	int ret;
	struct cxi_pt_alloc_opts pt_opts = {
		.is_matching = 1,
	};
	struct cxip_rdzv_pte *pte;

	pte = calloc(1, sizeof(*pte));
	if (!pte) {
		ret = -ENOMEM;
		CXIP_WARN("Failed to allocate memory for rendezvous PtlTE: %d:%s\n",
			  ret, fi_strerror(-ret));
		goto err;
	}

	pte->txc = txc;
	ofi_atomic_initialize32(&pte->le_linked_success_count, 0);
	ofi_atomic_initialize32(&pte->le_linked_failure_count, 0);
	ofi_spin_init(&pte->src_reqs_lock);

	if (txc->ep_obj->av->attr.flags & FI_SYMMETRIC)
		pt_opts.use_logical = 1;

	/* Reserve the Rendezvous Send PTE */
	ret = cxip_pte_alloc(txc->ep_obj->if_dom, txc->tx_evtq.eq,
			     txc->domain->iface->dev->info.rdzv_get_idx,
			     false, &pt_opts, cxip_rdzv_pte_cb, txc,
			     &pte->pte);
	if (ret) {
		CXIP_WARN("Failed to allocate rendezvous PtlTE: %d:%s\n", ret,
			  fi_strerror(-ret));
		goto err_free_rdzv_pte_mem;
	}

	ret = cxip_rdzv_pte_zbp_req_alloc(pte);
	if (ret) {
		CXIP_WARN("Failed to allocate zero byte put request: %d:%s\n",
			  ret, fi_strerror(-ret));
		goto err_free_rdzv_pte;
	}

	ret = cxip_pte_set_state_wait(pte->pte, txc->rx_cmdq, txc->send_cq,
				      C_PTLTE_ENABLED, 0);
	if (ret) {
		CXIP_WARN("Failed to enqueue command: %d:%s\n", ret,
			  fi_strerror(-ret));
		goto err_free_rdzv_pte_zbp_req;
	}

	*rdzv_pte = pte;
	return FI_SUCCESS;

err_free_rdzv_pte_zbp_req:
	cxip_rdzv_pte_zbp_req_free(pte);
err_free_rdzv_pte:
	cxip_pte_free(pte->pte);
err_free_rdzv_pte_mem:
	free(pte);
err:
	return ret;
}
