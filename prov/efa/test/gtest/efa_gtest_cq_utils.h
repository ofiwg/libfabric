/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_CQ_UTILS_H
#define EFA_GTEST_CQ_UTILS_H

#include <stdint.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct efa_test_dropped_cqe_result {
	int supported;
	int first_ret;
	uint16_t first_consumed_cnt;
	uint16_t first_cc;
	int second_ret;
	uint16_t second_consumed_cnt;
	uint16_t second_cc;
};

void efa_test_dropped_completion_advances_cc(
	struct fid_cq *cq_fid, struct fid_ep *ep_fid,
	struct efa_test_dropped_cqe_result *result);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_CQ_UTILS_H */
