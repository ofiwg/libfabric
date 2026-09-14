/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_DOMAIN_UTILS_H
#define EFA_GTEST_DOMAIN_UTILS_H

#include <stdint.h>
#include <stdbool.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

struct ibv_qp;
struct efadv_wq_attr;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Stand in for efadv_query_qp_wqs(), reporting a capability on the send
 * queue so that a caller which must not be shown caps can be told apart from
 * one which must. struct efadv_wq_attr does not always have a caps member, so
 * the attributes are filled from C.
 */
int efa_test_mock_efadv_query_qp_wqs(struct ibv_qp *ibvqp,
				     struct efadv_wq_attr *sq_attr,
				     struct efadv_wq_attr *rq_attr,
				     uint32_t inlen);

/**
 * @brief Offset the mock above reports for the completion action block, chosen
 * to match no plausible derivation from the queue entry layout so that a
 * provider computing one instead of passing the device's value through fails.
 */
#define EFA_TEST_MOCK_ACTION_BLOCK_OFFSET 104

/**
 * @brief The fi_efa_wq_caps bits the mock above makes the device report on its
 * send queue, which is none on a build whose efadv_wq_attr has no caps member.
 */
uint16_t efa_test_mock_efadv_sq_caps(void);

/**
 * @brief The completion action fi_efa_wq_caps bit the mock above makes the
 * device report, which is none on a build without efadv completion actions.
 */
uint16_t efa_test_mock_efadv_sq_comp_action_caps(void);

/**
 * @brief EFA_TEST_MOCK_ACTION_BLOCK_OFFSET, or 0 on a build without efadv
 * completion actions, where the mock reports no action block.
 */
uint16_t efa_test_mock_efadv_action_block_offset(void);

/**
 * @brief Read the QKEY the provider recorded on the endpoint's EFA QP. The
 * endpoint must be enabled.
 */
uint32_t efa_test_get_qp_qkey(struct fid_ep *ep);

/**
 * @brief Read the QKEY fi_getname() reports for @p ep, i.e. the one peers
 * insert into their AV. struct efa_ep_addr is opaque from C++.
 *
 * @return the fi_getname() return code; @p qkey is only set on success.
 */
int efa_test_getname_qkey(struct fid_ep *ep, uint32_t *qkey);

/**
 * @brief Read the endpoint's comp_action_enabled flag (set via
 * FI_OPT_EFA_COMP_ACTION). struct efa_base_ep is opaque from C++.
 */
/**
 * @brief Whether a per-work-request action descriptor carries exactly these
 * feature bits, ids and operands. struct efa_comp_action_wr is opaque from C++.
 */
struct efa_comp_action_wr;

bool efa_test_comp_action_wr_matches(const struct efa_comp_action_wr *sig,
				     uint64_t feature_bits, uint32_t local_id,
				     uint64_t local_data, uint32_t remote_id,
				     uint64_t remote_data);

bool efa_test_get_comp_action_enabled(struct fid_ep *ep);

/**
 * @brief Whether the selected EFA device (and this build) support
 * completion-action. Mirrors efa_device_support_comp_action(); lets gtest
 * skip action tests on hardware/builds without support.
 */
bool efa_test_device_supports_comp_action(void);

/**
 * @brief Whether the provider was compiled with efadv completion-action
 * support (HAVE_EFADV_COMP_ACTION). When false, the action ops are stubs that
 * return -FI_ENOSYS. Distinct from device capability: a build may support the
 * feature while a specific device does not advertise it.
 */
bool efa_test_build_has_comp_action(void);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_DOMAIN_UTILS_H */
