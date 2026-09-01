/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_ah.h"
#include "efa_base_ep.h"
#include "efa_gtest_ah_utils.h"
#include "efa_gtest_common_helpers.h"

struct efa_ah *efa_test_ah_alloc_fabricated_gid(struct fid_ep *ep)
{
	struct efa_base_ep *base_ep =
		container_of(ep, struct efa_base_ep, util_ep.ep_fid);
	struct efa_ep_addr raw_addr;

	efa_test_fabricate_addr(ep, &raw_addr);
	return efa_ah_alloc(base_ep->domain, raw_addr.raw, sizeof(struct efa_ah));
}
