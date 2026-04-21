/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef _EFA_RDM_CNTR_H_
#define _EFA_RDM_CNTR_H_

#include "efa_cntr.h"

struct efa_rdm_cntr {
	struct efa_cntr efa_cntr;
	struct fid_cntr *shm_cntr;
	bool need_to_scan_ep_list;
};

_Static_assert(offsetof(struct efa_rdm_cntr, efa_cntr) == 0,
	       "efa_rdm_cntr::efa_cntr must be the first member");

int efa_rdm_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		      struct fid_cntr **cntr_fid, void *context);

int efa_rdm_ep_insert_cntr_ibv_cq_poll_list(struct efa_base_ep *ep);

void efa_rdm_ep_remove_cntr_ibv_cq_poll_list(struct efa_base_ep *ep);

#endif
