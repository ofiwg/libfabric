/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_RDM_CQ_H
#define EFA_RDM_CQ_H

#include <ofi_util.h>

struct efa_rdm_cq {
	struct util_cq util_cq;
	struct fid_cq *shm_cq;
};

/*
 * Control header with completion data. CQ data length is static.
 */
#define EFA_RDM_CQ_DATA_SIZE (4)

int efa_rdm_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		    struct fid_cq **cq_fid, void *context);

#endif