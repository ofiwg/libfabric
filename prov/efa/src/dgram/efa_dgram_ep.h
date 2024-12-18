/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_base_ep.h"

#ifndef EFA_DGRAM_H
#define EFA_DGRAM_H

struct efa_dgram_ep {
	struct efa_base_ep base_ep;
};

int efa_dgram_ep_open(struct fid_domain *domain_fid, struct fi_info *info,
		      struct fid_ep **ep_fid, void *context);

extern struct fi_ops_msg efa_dgram_ep_msg_ops;
extern struct fi_ops_rma efa_dgram_ep_rma_ops;
#endif
