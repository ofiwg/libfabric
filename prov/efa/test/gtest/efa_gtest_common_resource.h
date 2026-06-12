/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_COMMON_RESOURCE_H
#define EFA_GTEST_COMMON_RESOURCE_H

#include <gtest/gtest.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_eq.h>

#define EFA_FABRIC_NAME		"efa"
#define EFA_DIRECT_FABRIC_NAME	"efa-direct"

#define MR_MODE_BITS (FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_LOCAL)

struct efa_resource {
	struct fi_info *hints;
	struct fi_info *info;
	struct fid_fabric *fabric;
	struct fid_domain *domain;
	struct fid_ep *ep;
	struct fid_eq *eq;
	struct fid_av *av;
	struct fid_cq *cq;
};

/**
 * @brief Construct a full set of OFI resources for testing.
 * Creates fabric, domain, endpoint, EQ, AV, and CQ, then enables the endpoint.
 * Uses gtest ASSERT macros — test is aborted on any failure.
 *
 * @param[out]	resource	struct to populate with created resources
 * @param[in]	ep_type		endpoint type (FI_EP_RDM or FI_EP_DGRAM)
 * @param[in]	fabric_name	fabric name (EFA_FABRIC_NAME or EFA_DIRECT_FABRIC_NAME)
 */
void efa_test_resource_construct(struct efa_resource *resource,
				 enum fi_ep_type ep_type, const char *fabric_name);

/**
 * @brief Destroy all OFI resources in the resource struct.
 * Closes resources in correct order. Uses gtest EXPECT macros so that
 * all cleanup is attempted even if one close fails.
 *
 * @param[in,out]	resource	struct with resources to close
 */
void efa_test_resource_destruct(struct efa_resource *resource);

#endif /* EFA_GTEST_COMMON_RESOURCE_H */
