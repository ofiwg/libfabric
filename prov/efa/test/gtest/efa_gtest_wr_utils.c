/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_base_ep.h"
#include "efa_gtest_wr_utils.h"

static struct efa_base_ep *efa_test_wr_base_ep(struct fid_ep *ep_fid)
{
	return container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
}

size_t efa_test_ep_recv_wr_index(struct fid_ep *ep_fid)
{
	return efa_test_wr_base_ep(ep_fid)->recv_wr_index;
}

int efa_test_ep_tx_wr_pending(struct fid_ep *ep_fid)
{
	struct efa_base_ep *base_ep = efa_test_wr_base_ep(ep_fid);

#if HAVE_EFA_DATA_PATH_DIRECT
	if (base_ep->qp->data_path_direct_enabled)
		return base_ep->qp->data_path_direct_qp.sq.num_wqe_pending != 0;
#endif
	return base_ep->is_wr_started;
}
