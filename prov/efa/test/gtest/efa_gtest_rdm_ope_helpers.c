/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_rdm_ope_helpers.h"
#include "efa_gtest_common_helpers.h"
#include "efa.h"
#include "efa_av.h"
#include "rdm/efa_rdm_ep.h"

int efa_test_drive_rxe_unexp_handle_error(struct fid_ep *ep, void *op_context,
					  int err, int *prov_errno_out)
{
	struct efa_rdm_ep *efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	fi_addr_t peer_addr = 0;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *rxe;
	int prov_errno = EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE;
	int ret;

	ret = efa_test_av_insert_self(
		ep, &efa_rdm_ep->base_ep.util_ep.av->av_fid, &peer_addr);
	if (ret != 1)
		return -FI_EINVAL;

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	if (!peer)
		return -FI_EINVAL;

	rxe = efa_rdm_ep_alloc_rxe(efa_rdm_ep, peer, ofi_op_tagged);
	if (!rxe)
		return -FI_ENOMEM;

	rxe->state = EFA_RDM_RXE_UNEXP;
	rxe->cq_entry.op_context = op_context;

	efa_rdm_rxe_handle_error(rxe, err, prov_errno);
	efa_rdm_rxe_release(rxe);

	if (prov_errno_out)
		*prov_errno_out = prov_errno;

	return 0;
}
