/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_FABRIC_UTIL_H
#define EFA_FABRIC_UTIL_H

#include "efa.h"

int efa_fabric_init_base(struct efa_fabric *efa_fabric,
			 struct fi_fabric_attr *attr,
			 void *context);

int efa_fabric_destruct_base(struct efa_fabric *efa_fabric);

int efa_non_cq_trywait(struct fid *fid);

bool efa_feature_in(const char * const *list, size_t n, const char *feature);

#endif /* EFA_FABRIC_UTIL_H */
