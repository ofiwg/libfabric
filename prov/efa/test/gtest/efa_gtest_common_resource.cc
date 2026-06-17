/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_resource.h"

struct fi_info *efa_test_alloc_default_hints(enum fi_ep_type ep_type,
					     const char *fabric_name)
{
	struct fi_info *hints;

	hints = fi_allocinfo();
	if (!hints)
		return NULL;

	if (fabric_name)
		hints->fabric_attr->name = strdup(fabric_name);
	hints->ep_attr->type = ep_type;
	hints->domain_attr->mr_mode = MR_MODE_BITS;

	if (!fabric_name || !strcasecmp(fabric_name, EFA_DIRECT_FABRIC_NAME))
		hints->mode |= FI_CONTEXT2;

	if (ep_type == FI_EP_DGRAM)
		hints->mode |= FI_MSG_PREFIX | FI_CONTEXT2;

	return hints;
}

void efa_test_resource_construct(struct efa_resource *resource,
				 struct fi_info *hints)
{
	int ret;
	struct fi_av_attr av_attr = {};
	struct fi_cq_attr cq_attr = {};
	struct fi_eq_attr eq_attr = {};
	const char *fabric_name;
	uint32_t fi_version;

	cq_attr.format = FI_CQ_FORMAT_DATA;

	ASSERT_NE(hints, nullptr);
	resource->hints = hints;

	/* The fabric and ep type are already encoded in hints; derive the API
	 * version from the fabric name rather than taking it as a separate
	 * argument that could disagree with hints. */
	fabric_name = hints->fabric_attr ? hints->fabric_attr->name : NULL;
	fi_version =
		(fabric_name && !strcmp(EFA_DIRECT_FABRIC_NAME, fabric_name)) ?
			FI_VERSION(2, 0) :
			FI_VERSION(1, 14);

	ret = fi_getinfo(fi_version, NULL, NULL, 0ULL, resource->hints,
			 &resource->info);
	ASSERT_EQ(ret, 0) << "fi_getinfo failed: " << fi_strerror(-ret);

	ret = fi_fabric(resource->info->fabric_attr, &resource->fabric, NULL);
	ASSERT_EQ(ret, 0) << "fi_fabric failed: " << fi_strerror(-ret);

	ret = fi_domain(resource->fabric, resource->info, &resource->domain,
			NULL);
	ASSERT_EQ(ret, 0) << "fi_domain failed: " << fi_strerror(-ret);

	ret = fi_endpoint(resource->domain, resource->info, &resource->ep,
			  NULL);
	ASSERT_EQ(ret, 0) << "fi_endpoint failed: " << fi_strerror(-ret);

	ret = fi_eq_open(resource->fabric, &eq_attr, &resource->eq, NULL);
	ASSERT_EQ(ret, 0) << "fi_eq_open failed: " << fi_strerror(-ret);

	fi_ep_bind(resource->ep, &resource->eq->fid, 0);

	ret = fi_av_open(resource->domain, &av_attr, &resource->av, NULL);
	ASSERT_EQ(ret, 0) << "fi_av_open failed: " << fi_strerror(-ret);

	fi_ep_bind(resource->ep, &resource->av->fid, 0);

	ret = fi_cq_open(resource->domain, &cq_attr, &resource->cq, NULL);
	ASSERT_EQ(ret, 0) << "fi_cq_open failed: " << fi_strerror(-ret);

	fi_ep_bind(resource->ep, &resource->cq->fid, FI_SEND | FI_RECV);

	ret = fi_enable(resource->ep);
	ASSERT_EQ(ret, 0) << "fi_enable failed: " << fi_strerror(-ret);
}

void efa_test_resource_destruct(struct efa_resource *resource)
{
	if (resource->ep) {
		EXPECT_EQ(fi_close(&resource->ep->fid), 0);
		resource->ep = NULL;
	}

	if (resource->eq) {
		EXPECT_EQ(fi_close(&resource->eq->fid), 0);
		resource->eq = NULL;
	}

	if (resource->cq) {
		EXPECT_EQ(fi_close(&resource->cq->fid), 0);
		resource->cq = NULL;
	}

	if (resource->av) {
		EXPECT_EQ(fi_close(&resource->av->fid), 0);
		resource->av = NULL;
	}

	if (resource->domain) {
		EXPECT_EQ(fi_close(&resource->domain->fid), 0);
		resource->domain = NULL;
	}

	if (resource->fabric) {
		EXPECT_EQ(fi_close(&resource->fabric->fid), 0);
		resource->fabric = NULL;
	}

	if (resource->info) {
		fi_freeinfo(resource->info);
		resource->info = NULL;
	}

	if (resource->hints) {
		fi_freeinfo(resource->hints);
		resource->hints = NULL;
	}
}
