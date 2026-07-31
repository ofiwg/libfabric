/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_TCLASS_UTILS_H
#define EFA_GTEST_TCLASS_UTILS_H

#include <stdint.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct efadv_qp_init_attr;

/**
 * @brief Whether this build detected efadv_qp_init_attr::sl (HAVE_EFADV_SL).
 * When 0 the provider cannot request a service level at all, so EFA reports
 * FI_TC_LOW_LATENCY as unsupported and every QP is created with the default SL.
 */
int efa_test_have_efadv_sl(void);

/**
 * @brief Read efadv_qp_init_attr::sl, the QP's requested service level.
 * Returns EFA_QP_DEFAULT_SERVICE_LEVEL when the field is not available.
 */
uint8_t efa_test_efadv_attr_sl(const struct efadv_qp_init_attr *attr);

extern const uint8_t efa_test_qp_default_sl;
extern const uint8_t efa_test_qp_low_latency_sl;

/**
 * @brief Read the tclass the domain stored from info->domain_attr->tclass.
 */
uint32_t efa_test_get_domain_tclass(struct fid_domain *domain);

/**
 * @brief Read the tclass the endpoint stored from info->tx_attr->tclass.
 */
uint32_t efa_test_get_base_ep_tclass(struct fid_ep *ep);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_TCLASS_UTILS_H */
