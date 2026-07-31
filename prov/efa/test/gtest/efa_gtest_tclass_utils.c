/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_gtest_tclass_utils.h"

const uint8_t efa_test_qp_default_sl = EFA_QP_DEFAULT_SERVICE_LEVEL;
const uint8_t efa_test_qp_low_latency_sl = EFA_QP_LOW_LATENCY_SERVICE_LEVEL;

int efa_test_have_efadv_sl(void)
{
#if HAVE_EFADV_SL
	return 1;
#else
	return 0;
#endif
}

uint8_t efa_test_efadv_attr_sl(const struct efadv_qp_init_attr *attr)
{
#if HAVE_EFADV_SL
	return attr->sl;
#else
	(void) attr;
	return EFA_QP_DEFAULT_SERVICE_LEVEL;
#endif
}

uint32_t efa_test_get_domain_tclass(struct fid_domain *domain)
{
	struct efa_domain *efa_domain =
		container_of(domain, struct efa_domain, util_domain.domain_fid);

	return efa_domain->info->domain_attr->tclass;
}

uint32_t efa_test_get_base_ep_tclass(struct fid_ep *ep)
{
	struct efa_base_ep *base_ep =
		container_of(ep, struct efa_base_ep, util_ep.ep_fid);

	return base_ep->info->tx_attr->tclass;
}
