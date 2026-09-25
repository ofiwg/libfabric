/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_DOMAIN_UTILS_H
#define EFA_GTEST_DOMAIN_UTILS_H

#include <stdint.h>
#include <stdbool.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
