/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_DIRECT_EP_H
#define EFA_DIRECT_EP_H

#include "efa_base_ep.h"

/**
 * @brief EFA direct endpoint structure
 *
 * Wraps efa_base_ep as first member (for castability) and adds
 * fields that are only used by the efa-direct path.
 */
struct efa_direct_ep {
	struct efa_base_ep base_ep;

	struct ofi_bufpool *ope_pool;	/**< pool for efa_direct_ope */
	struct dlist_entry ope_list;	/**< list of outstanding ops */
};

static_assert(offsetof(struct efa_direct_ep, base_ep) == 0,
	      "efa_base_ep must be first member of efa_direct_ep for container_of safety");

#endif /* EFA_DIRECT_EP_H */
