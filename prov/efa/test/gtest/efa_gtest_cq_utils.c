/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_base_ep.h"
#include "efa_cq.h"
#include "efa_data_path_direct_entry.h"
#include "efa_io_defs.h"
#include "efa_gtest_cq_utils.h"

void efa_test_dropped_completion_advances_cc(
	struct fid_cq *cq_fid, struct fid_ep *ep_fid,
	struct efa_test_dropped_cqe_result *result)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	struct efa_base_ep *base_ep =
		container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	struct efa_cq cq =
		*container_of(cq_fid, struct efa_cq, util_cq.cq_fid);
	struct efa_data_path_direct_cq *direct_cq =
		&cq.ibv_cq.data_path_direct;
	struct efa_io_rx_cdesc_ex cqes[2] = {0};
	struct efa_io_cdesc_common *cqe = &cqes[0].base.common;

	cqe->qp_num = base_ep->qp->qp_num + 1;
	EFA_SET(&cqe->flags, EFA_IO_CDESC_COMMON_Q_TYPE, EFA_IO_SEND_QUEUE);
	EFA_SET(&cqe->flags, EFA_IO_CDESC_COMMON_PHASE, 1);

	memset(direct_cq, 0, sizeof(*direct_cq));
	direct_cq->buffer = (uint8_t *) cqes;
	direct_cq->entry_size = sizeof(cqes[0]);
	direct_cq->num_entries = 2;
	direct_cq->qmask = 1;
	direct_cq->phase = 1;

	result->supported = 1;
	result->first_ret =
		efa_data_path_direct_start_poll(&cq.ibv_cq, NULL);
	result->first_consumed_cnt = direct_cq->consumed_cnt;
	result->first_cc = direct_cq->cc;
	result->second_ret =
		efa_data_path_direct_start_poll(&cq.ibv_cq, NULL);
	result->second_consumed_cnt = direct_cq->consumed_cnt;
	result->second_cc = direct_cq->cc;
#else
	memset(result, 0, sizeof(*result));
#endif
}
