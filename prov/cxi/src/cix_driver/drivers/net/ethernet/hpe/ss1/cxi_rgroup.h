/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2024 Hewlett Packard Enterprise Development LP */

/* TODO: are these really part of the uapi? */

#ifndef _CXI_RGROUP_H_
#define _CXI_RGROUP_H_

#include <linux/xarray.h>
#include "cass_core.h"

/* Resource Groups contigurables */

#define RGROUP_GFP_OPTS              (GFP_KERNEL)
#define RGROUP_ID_MIN                (CXI_DEFAULT_SVC_ID)
#define RGROUP_ID_MAX                (INT_MAX)
#define RGROUP_ID_LIMITS             (XA_LIMIT(RGROUP_ID_MIN, \
					       RGROUP_ID_MAX))
#define RGROUP_XARRAY_FLAGS          (XA_FLAGS_ALLOC1)

/* Resource Entries */

#define RESOURCE_ENTRY_GFP_OPTS      (GFP_KERNEL)
#define RESOURCE_ENTRY_ID_MIN        (0)
#define RESOURCE_ENTRY_ID_MAX        (INT_MAX)
#define RESOURCE_ENTRY_ID_LIMITS     (XA_LIMIT(RESOURCE_ENTRY_ID_MIN, \
					       RESOURCE_ENTRY_ID_MAX))
#define RESOURCE_ENTRY_XARRAY_FLAGS  (XA_FLAGS_ALLOC)

struct cxi_rgroup_attr {
	unsigned int   cntr_pool_id;
	unsigned int   lnis_per_rgid;
	bool           system_service;
	char           name[50];
};

struct cxi_rgroup_state {
	bool           enabled;
	refcount_t     refcount;
};

struct cxi_rgroup {
	unsigned int                   id;
	struct cass_dev                *hw;
	struct cxi_rgroup_attr         attr;
	struct cxi_rgroup_state        state;
	struct cxi_resource_entry_list resource_entry_list;
	struct cxi_ac_entry_list       ac_entry_list;
	struct cxi_rgroup_pools        pools;
};

void cxi_dev_rgroup_init(struct cxi_dev *dev);
void cxi_dev_rgroup_fini(struct cxi_dev *dev);

/**
 * for_each_rgroup() - Iterate over rgroup_list
 *
 * @list: rgroup list
 * @index: index of @entry
 * @entry: rgroup retrieved from array
 *
 * Return: first non-zero return value of operator or 0
 */
#define for_each_rgroup(index, entry) \
	xa_for_each(&hw->rgroup_list.xarray, index, entry)

#endif /* _CXI_RGROUP_H_ */
