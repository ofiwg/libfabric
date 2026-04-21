/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2013-2018 Intel Corporation, Inc.  All rights reserved. */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "ofi_util.h"
#include "efa.h"
#include "efa_cntr.h"
#include "efa_cq.h"
#include "efa_domain.h"
#include "rdm/efa_rdm_cntr.h"
#include "rdm/efa_rdm_cq.h"
#include "rdm/efa_rdm_ep.h"

static uint64_t efa_rdm_cntr_read(struct fid_cntr *cntr_fid)
{
	struct efa_rdm_cntr *efa_rdm_cntr;
	struct efa_domain *domain;

	efa_rdm_cntr = container_of(cntr_fid, struct efa_rdm_cntr, efa_cntr.util_cntr.cntr_fid);
	domain = container_of(efa_rdm_cntr->efa_cntr.util_cntr.domain, struct efa_domain, util_domain);

	/* Flush SHM completions into the peer counter before reading */
	ofi_genlock_lock(&domain->srx_lock);
	if (efa_rdm_cntr->shm_cntr)
		fi_cntr_read(efa_rdm_cntr->shm_cntr);
	ofi_genlock_unlock(&domain->srx_lock);

	return efa_cntr_read(cntr_fid);
}

static uint64_t efa_rdm_cntr_readerr(struct fid_cntr *cntr_fid)
{
	struct efa_rdm_cntr *efa_rdm_cntr;
	struct efa_domain *domain;

	efa_rdm_cntr = container_of(cntr_fid, struct efa_rdm_cntr, efa_cntr.util_cntr.cntr_fid);
	domain = container_of(efa_rdm_cntr->efa_cntr.util_cntr.domain, struct efa_domain, util_domain);

	ofi_genlock_lock(&domain->srx_lock);
	if (efa_rdm_cntr->shm_cntr)
		fi_cntr_read(efa_rdm_cntr->shm_cntr);
	ofi_genlock_unlock(&domain->srx_lock);

	return efa_cntr_readerr(cntr_fid);
}

static struct fi_ops_cntr efa_rdm_cntr_ops = {
	.size = sizeof(struct fi_ops_cntr),
	.read = efa_rdm_cntr_read,
	.readerr = efa_rdm_cntr_readerr,
	.add = ofi_cntr_add,
	.adderr = ofi_cntr_adderr,
	.set = ofi_cntr_set,
	.seterr = ofi_cntr_seterr,
	.wait = efa_cntr_wait
};

static int efa_rdm_cntr_close(struct fid *fid)
{
	struct efa_rdm_cntr *cntr;
	int ret, retv;

	retv = 0;
	cntr = container_of(fid, struct efa_rdm_cntr, efa_cntr.util_cntr.cntr_fid.fid);

	if (cntr->shm_cntr) {
		ret = fi_close(&cntr->shm_cntr->fid);
		if (ret) {
			EFA_WARN(FI_LOG_CNTR, "Unable to close shm cntr: %s\n", fi_strerror(-ret));
			retv = ret;
		}
	}

	efa_cntr_destruct(&cntr->efa_cntr);
	free(cntr);
	return retv;
}

static struct fi_ops efa_rdm_cntr_fi_ops = {
	.size = sizeof(efa_rdm_cntr_fi_ops),
	.close = efa_rdm_cntr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static void efa_rdm_cntr_progress(struct util_cntr *cntr)
{
	struct dlist_entry *item;
	struct efa_rdm_cntr *efa_rdm_cntr;
	struct efa_domain *efa_domain;
	struct efa_rdm_ep *efa_rdm_ep;
	struct fid_list_entry *fid_entry;

	ofi_genlock_lock(&cntr->ep_list_lock);
	efa_rdm_cntr = container_of(cntr, struct efa_rdm_cntr, efa_cntr.util_cntr);
	efa_domain = container_of(efa_rdm_cntr->efa_cntr.util_cntr.domain, struct efa_domain, util_domain);

	/**
	 * TODO: It's better to just post the initial batch of internal rx pkts during ep enable
	 * so we don't have to iterate cntr->ep_list here.
	 * However, it is observed that doing that will hurt performance if application opens
	 * some idle endpoints and never poll completions for them. Move these initial posts to
	 * the first polling before having a long term fix.
	 */
	if (efa_rdm_cntr->need_to_scan_ep_list) {
		dlist_foreach(&cntr->ep_list, item) {
			fid_entry = container_of(item, struct fid_list_entry, entry);
			efa_rdm_ep = container_of(fid_entry->fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);
			if (efa_rdm_ep->base_ep.efa_qp_enabled)
				efa_rdm_ep_post_internal_rx_pkts(efa_rdm_ep);
		}
		efa_rdm_cntr->need_to_scan_ep_list = false;
	}

	efa_cntr_progress_ibv_cq_poll_list(&efa_rdm_cntr->efa_cntr);
	efa_domain_progress_rdm_peers_and_queues(efa_domain);
	ofi_genlock_unlock(&cntr->ep_list_lock);
}

int efa_rdm_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		      struct fid_cntr **cntr_fid, void *context)
{
	int ret;
	struct efa_rdm_cntr *cntr;
	struct efa_domain *efa_domain;
	struct fi_cntr_attr shm_cntr_attr = {0};
	struct fi_peer_cntr_context peer_cntr_context = {0};

	cntr = calloc(1, sizeof(*cntr));
	if (!cntr)
		return -FI_ENOMEM;

	cntr->need_to_scan_ep_list = false;
	efa_domain = container_of(domain, struct efa_domain,
				  util_domain.domain_fid);

	ret = efa_cntr_construct(&cntr->efa_cntr, domain, attr,
				 efa_rdm_cntr_progress, context);
	if (ret)
		goto free;

	*cntr_fid = &cntr->efa_cntr.util_cntr.cntr_fid;
	cntr->efa_cntr.util_cntr.cntr_fid.ops = &efa_rdm_cntr_ops;
	cntr->efa_cntr.util_cntr.cntr_fid.fid.ops = &efa_rdm_cntr_fi_ops;

	/* open shm cntr as peer cntr */
	if (efa_domain->shm_domain) {
		memcpy(&shm_cntr_attr, attr, sizeof(*attr));
		shm_cntr_attr.flags |= FI_PEER;
		peer_cntr_context.size = sizeof(peer_cntr_context);
		peer_cntr_context.cntr = cntr->efa_cntr.util_cntr.peer_cntr;
		ret = fi_cntr_open(efa_domain->shm_domain, &shm_cntr_attr,
				   &cntr->shm_cntr, &peer_cntr_context);
		if (ret) {
			EFA_WARN(FI_LOG_CNTR, "Unable to open shm cntr, err: %s\n", fi_strerror(-ret));
			goto free;
		}
	}

	return FI_SUCCESS;

free:
	free(cntr);
	return ret;
}

/**
 * @brief Insert tx/rx cq into the cntr ibv_cq_poll_list for efa-rdm EP
 *
 * Reuses efa_base_ep_insert_cntr_ibv_cq_poll_list for the common logic,
 * then sets need_to_scan_ep_list on each bound efa_rdm_cntr.
 *
 * @param ep efa_base_ep
 * @return int 0 on success, negative integer on failure
 */
int efa_rdm_ep_insert_cntr_ibv_cq_poll_list(struct efa_base_ep *ep)
{
	int i, ret;
	struct efa_rdm_cntr *efa_rdm_cntr;
	struct util_cntr *util_cntr;

	ret = efa_base_ep_insert_cntr_ibv_cq_poll_list(ep);
	if (ret)
		return ret;

	for (i = 0; i < CNTR_CNT; i++) {
		util_cntr = ep->util_ep.cntrs[i];
		if (util_cntr) {
			efa_rdm_cntr = container_of(util_cntr, struct efa_rdm_cntr, efa_cntr.util_cntr);
			ofi_genlock_lock(&efa_rdm_cntr->efa_cntr.util_cntr.ep_list_lock);
			efa_rdm_cntr->need_to_scan_ep_list = true;
			ofi_genlock_unlock(&efa_rdm_cntr->efa_cntr.util_cntr.ep_list_lock);
		}
	}

	return FI_SUCCESS;
}

/**
 * @brief Remove tx/rx cq from the cntr ibv_cq_poll_list for efa-rdm EP
 *
 * @param ep efa_base_ep
 */
void efa_rdm_ep_remove_cntr_ibv_cq_poll_list(struct efa_base_ep *ep)
{
	efa_base_ep_remove_cntr_ibv_cq_poll_list(ep);
}
