/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#ifndef _EFA_CQDIRECT_H
#define _EFA_CQDIRECT_H

#include "efa.h"
#include "efa_base_ep.h"
#include "efa_cq.h"
#include <infiniband/verbs.h>

int efa_cqdirect_qp_initialize(struct efa_qp *efa_qp);
int efa_cqdirect_cq_initialize(struct efa_cq *efa_cq);
void efa_cqdirect_qp_finalize(struct efa_qp *efa_qp);
#endif