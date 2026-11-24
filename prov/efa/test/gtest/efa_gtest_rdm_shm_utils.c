/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_av.h"
#include "rdm/efa_rdm_av.h"
#include "rdm/efa_rdm_cntr.h"
#include "rdm/efa_rdm_cq.h"
#include "rdm/efa_rdm_ep.h"
#include "efa_gtest_rdm_shm_utils.h"

int efa_test_rdm_ep_has_shm_ep(struct fid_ep *ep)
{
	return container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid)
		       ->shm_ep != NULL;
}

int efa_test_rdm_av_has_shm_av(struct fid_av *av)
{
	struct efa_av *efa_av = container_of(av, struct efa_av, util_av.av_fid);

	return ((struct efa_rdm_av *) efa_av)->shm_rdm_av != NULL;
}

int efa_test_rdm_cq_shm_cq_ref(struct fid_cq *cq)
{
	struct efa_rdm_cq *efa_rdm_cq =
		container_of(cq, struct efa_rdm_cq, efa_cq.util_cq.cq_fid);

	if (!efa_rdm_cq->shm_cq)
		return -1;

	return ofi_atomic_get32(
		&container_of(efa_rdm_cq->shm_cq, struct util_cq, cq_fid)->ref);
}

int efa_test_rdm_cntr_shm_cntr_ref(struct fid_cntr *cntr)
{
	struct efa_rdm_cntr *efa_rdm_cntr = container_of(
		cntr, struct efa_rdm_cntr, efa_cntr.util_cntr.cntr_fid);

	if (!efa_rdm_cntr->shm_cntr)
		return -1;

	return ofi_atomic_get32(&container_of(efa_rdm_cntr->shm_cntr,
					      struct util_cntr, cntr_fid)
					 ->ref);
}
