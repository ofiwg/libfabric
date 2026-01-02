/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_FABRIC_H_
#define _CXIP_FABRIC_H_


#include <ofi_atom.h>

/* Type definitions */
struct cxip_fabric {
	struct util_fabric util_fabric;
	ofi_atomic32_t ref;
};

/* Function declarations */
int cxip_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		void *context);

#endif /* _CXIP_FABRIC_H_ */
