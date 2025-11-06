/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2024 Hewlett Packard Enterprise Development LP */

#ifndef _CASS_RGROUP_H_
#define _CASS_RGROUP_H_

#include <linux/xarray.h>

#include "cxi_rgroup.h"

void cass_dev_rgroup_init(struct cass_dev *hw);

void cass_dev_rgroup_fini(struct cass_dev *hw);

int cass_rgroup_add_resource(struct cxi_rgroup *rgroup,
			     struct cxi_resource_entry *resource);

int cass_rgroup_remove_resource(struct cxi_rgroup *rgroup,
				struct cxi_resource_entry *resource);

void cass_free_resource(struct cxi_rgroup *rgroup,
			struct cxi_resource_entry *entry);

int cass_alloc_resource(struct cxi_rgroup *rgroup,
			struct cxi_resource_entry *entry);

void cass_get_tle_in_use(struct cxi_rgroup *rgroup,
			 struct cxi_resource_entry *entry);

#endif /* _CASS_RGROUP_H_ */
