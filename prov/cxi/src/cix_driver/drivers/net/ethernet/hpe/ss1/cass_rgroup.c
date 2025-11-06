// SPDX-License-Identifier: GPL-2.0
/* Copyright 2024 Hewlett Packard Enterprise Development LP */

/* Resource Group implementation */

#include "cass_core.h"

/**
 * get_cxi_dev() - type-safe function to get CXI device pointer
 *                 from Cassini device
 *
 * @hw: cass_dev pointer
 *
 * Return: Embedded CXI device pointer
 */
static struct cxi_dev *get_cxi_dev(struct cass_dev *hw)
{
	return &hw->cdev;
}

void cass_dev_rgroup_init(struct cass_dev *hw)
{
	int i;

	spin_lock_init(&hw->rgrp_lock);

	hw->resource_use[CXI_RESOURCE_PTLTE].max = C_NUM_PTLTES;
	hw->resource_use[CXI_RESOURCE_TXQ].max = C_NUM_TRANSMIT_CQS;
	hw->resource_use[CXI_RESOURCE_TGQ].max = C_NUM_TARGET_CQS;
	hw->resource_use[CXI_RESOURCE_EQ].max = EQS_AVAIL;
	hw->resource_use[CXI_RESOURCE_CT].max = CTS_AVAIL;
	hw->resource_use[CXI_RESOURCE_PE0_LE].max = pe_total_les;
	hw->resource_use[CXI_RESOURCE_PE1_LE].max = pe_total_les;
	hw->resource_use[CXI_RESOURCE_PE2_LE].max = pe_total_les;
	hw->resource_use[CXI_RESOURCE_PE3_LE].max = pe_total_les;
	hw->resource_use[CXI_RESOURCE_TLE].max = C_NUM_TLES;
	/* AC 0 invalid and one reserved for shared physical mappings */
	hw->resource_use[CXI_RESOURCE_AC].max = ACS_AVAIL;

	/* start with all resources shared */
	for (i = 0; i < CXI_RESOURCE_MAX; i++) {
		hw->resource_use[i].in_use = 0;
		hw->resource_use[i].reserved = 0;
		hw->resource_use[i].shared_use = 0;
		hw->resource_use[i].shared = hw->resource_use[i].max;
	}

	cxi_dev_rgroup_init(get_cxi_dev(hw));
}

void cass_dev_rgroup_fini(struct cass_dev *hw)
{
	cxi_dev_rgroup_fini(get_cxi_dev(hw));
}

int cass_rgroup_add_resource(struct cxi_rgroup *rgroup,
			     struct cxi_resource_entry *resource)
{
	int rc = 0;
	struct cass_dev *hw = rgroup->hw;
	struct cxi_resource_use *r_use = &hw->resource_use[resource->type];

	if (!resource->limits.max && !resource->limits.reserved)
		return 0;

	spin_lock(&hw->rgrp_lock);

	if (resource->type == CXI_RESOURCE_TLE) {
		/* Enforce minimum TLEs */
		if (resource->limits.reserved) {
			if (resource->limits.reserved < CASS_MIN_POOL_TLES) {
				pr_debug("%s reserved minimum must be >= %d\n",
					 cxi_resource_type_to_str(CXI_RESOURCE_TLE),
					 CASS_MIN_POOL_TLES);
				rc = -EINVAL;
				goto unlock;
			}
		}

		/* Enforce max and reserved TLEs equal */
		if (resource->limits.max != resource->limits.reserved) {
			pr_debug("%s max must equal reserved\n",
				 cxi_resource_type_to_str(CXI_RESOURCE_TLE));
			rc = -EINVAL;
			goto unlock;
		}
	}

	if (resource->limits.reserved > r_use->shared) {
		pr_debug("Error - %s reserved requested (%lu) > shared available (%lu)\n",
			 cxi_resource_type_to_str(resource->type),
			 resource->limits.reserved, r_use->shared);
		rc = -ENOSPC;
		goto unlock;
	}

	if (resource->limits.max > r_use->max) {
		pr_debug("Error - %s max requested (%lu) > max available (%lu)\n",
			 cxi_resource_type_to_str(resource->type),
			 resource->limits.max, r_use->max);
		rc = -ENOSPC;
		goto unlock;
	}

	/* resources that need extra configuring */
	if (resource->type >= CXI_RESOURCE_PE0_LE &&
	    resource->type <= CXI_RESOURCE_PE3_LE) {
		int le_pool_id;
		int pe = resource->type - CXI_RESOURCE_PE0_LE;

		le_pool_id = ida_simple_get(&hw->le_pool_ids[pe],
					    DEFAULT_LE_POOL_ID,
					    CASS_NUM_LE_POOLS, GFP_NOWAIT);
		if (le_pool_id < 0) {
			pr_debug("%s pool unavailable.\n",
				 cxi_resource_type_to_str(resource->type));
			rc = -EBADR;
			goto unlock;
		}

		rgroup->pools.le_pool_id[pe] = le_pool_id;
	} else if (resource->type == CXI_RESOURCE_TLE) {
		int tle_pool_id;

		tle_pool_id = ida_simple_get(&hw->tle_pool_ids,
					     DEFAULT_TLE_POOL_ID,
					     C_CQ_CFG_TLE_POOL_ENTRIES,
					     GFP_NOWAIT);
		if (tle_pool_id < 0) {
			pr_debug("%s pool unavailable.\n",
				 cxi_resource_type_to_str(resource->type));
			rc = -EBADR;
			goto unlock;
		}

		rgroup->pools.tle_pool_id = tle_pool_id;
	}

	r_use->reserved += resource->limits.reserved;
	r_use->shared -= resource->limits.reserved;

	pr_debug("type:%s limits.reserved:%ld limits.max:%ld reserved:%ld shared:%ld\n",
		 cxi_resource_type_to_str(resource->type),
		 resource->limits.reserved, resource->limits.max,
		 r_use->reserved, r_use->shared);

	if (resource->type >= CXI_RESOURCE_PE0_LE &&
	    resource->type <= CXI_RESOURCE_PE3_LE) {
		int pe = resource->type - CXI_RESOURCE_PE0_LE;
		struct cxi_limits les = {
			.max = resource->limits.max,
			.res = resource->limits.reserved,
		};

		cass_cfg_le_pools(hw, rgroup->pools.le_pool_id[pe], pe, &les,
				  false);
	} else if (resource->type == CXI_RESOURCE_TLE) {
		struct cxi_limits tles = {
			.max = resource->limits.max,
			.res = resource->limits.reserved,
		};

		cass_cfg_tle_pool(hw, rgroup->pools.tle_pool_id, &tles, false);
	}

unlock:
	spin_unlock(&hw->rgrp_lock);

	return rc;
}

int cass_rgroup_remove_resource(struct cxi_rgroup *rgroup,
				struct cxi_resource_entry *resource)
{
	struct cass_dev *hw = rgroup->hw;
	struct cxi_resource_use *r_use;

	r_use = &hw->resource_use[resource->type];

	spin_lock(&hw->rgrp_lock);

	r_use->reserved -= resource->limits.reserved;
	r_use->shared += resource->limits.reserved;

	pr_debug("type:%s limits.reserved:%ld limits.max:%ld reserved:%ld shared:%ld\n",
		 cxi_resource_type_to_str(resource->type),
		 resource->limits.reserved, resource->limits.max,
		 r_use->reserved, r_use->shared);

	if (resource->type >= CXI_RESOURCE_PE0_LE &&
	    resource->type <= CXI_RESOURCE_PE3_LE) {
		int pe = resource->type - CXI_RESOURCE_PE0_LE;
		struct cxi_limits les = {
			.max = resource->limits.max,
			.res = resource->limits.reserved,
		};

		cass_cfg_le_pools(hw, rgroup->pools.le_pool_id[pe], pe, &les,
				  true);
		ida_simple_remove(&hw->le_pool_ids[pe],
				  rgroup->pools.le_pool_id[pe]);
		rgroup->pools.le_pool_id[pe] = -1;
	} else if (resource->type == CXI_RESOURCE_TLE) {
		struct cxi_limits tles = {
			.max = resource->limits.max,
			.res = resource->limits.reserved,
		};

		cass_cfg_tle_pool(hw, rgroup->pools.tle_pool_id, &tles, true);
		ida_simple_remove(&hw->tle_pool_ids, rgroup->pools.tle_pool_id);
		rgroup->pools.tle_pool_id = -1;
	}

	spin_unlock(&hw->rgrp_lock);

	return 0;
}

void cass_free_resource(struct cxi_rgroup *rgroup,
			struct cxi_resource_entry *entry)
{
	struct cass_dev *hw = rgroup->hw;
	struct cxi_resource_use *r_use = &hw->resource_use[entry->type];

	spin_lock(&hw->rgrp_lock);

	/* Free from shared space if applicable */
	if (entry->limits.in_use > entry->limits.reserved)
		r_use->shared_use--;

	r_use->in_use--;
	entry->limits.in_use--;

	spin_unlock(&hw->rgrp_lock);
}

int cass_alloc_resource(struct cxi_rgroup *rgroup,
			struct cxi_resource_entry *entry)
{
	int rc = 0;
	size_t available;
	struct cass_dev *hw = rgroup->hw;
	struct cxi_resource_use *r_use = &hw->resource_use[entry->type];

	spin_lock(&hw->rgrp_lock);

	available = r_use->max - r_use->shared_use;

	if (entry->limits.in_use < entry->limits.reserved) {
		r_use->in_use++;
		entry->limits.in_use++;
	} else if (entry->limits.in_use < entry->limits.max && available) {
		entry->limits.in_use++;
		r_use->in_use++;
		r_use->shared_use++;
	} else {
		pr_debug("%s unavailable use:%ld reserved:%ld max:%ld shared_use:%ld\n",
			 cxi_resource_type_to_str(entry->type),
			 entry->limits.in_use, entry->limits.reserved,
			 entry->limits.max, r_use->shared_use);
		rc = -ENOSPC;
	}

	spin_unlock(&hw->rgrp_lock);

	return rc;
}

void cass_get_tle_in_use(struct cxi_rgroup *rgroup,
			 struct cxi_resource_entry *entry)
{
	union c_cq_sts_tle_in_use tle_in_use;

	if (cxi_rgroup_tle_pool_id(rgroup) == -1)
		return;

	cass_read(rgroup->hw,
		  C_CQ_STS_TLE_IN_USE(cxi_rgroup_tle_pool_id(rgroup)),
		  &tle_in_use, sizeof(tle_in_use));

	entry->limits.in_use = tle_in_use.count;
}
