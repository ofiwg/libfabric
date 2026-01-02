/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_EQ_H_
#define _CXIP_EQ_H_

#include <ofi_list.h>
#include <ofi_lock.h>

/* Macros */
#define CXIP_EQ_DEF_SZ (1 << 8)

#define CXIP_EQ_MAP_FLAGS (CXI_MAP_WRITE | CXI_MAP_PIN)

/* Type definitions */
struct cxip_eq {
	struct util_eq util_eq;
	struct fi_eq_attr attr;
	struct dlist_entry ep_list;
	ofi_mutex_t list_lock;
};

/* Function declarations */
int cxip_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		 struct fid_eq **eq, void *context);

#endif /* _CXIP_EQ_H_ */
