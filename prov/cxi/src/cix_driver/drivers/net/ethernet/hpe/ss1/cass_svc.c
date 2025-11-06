// SPDX-License-Identifier: GPL-2.0
/* Copyright 2020 Hewlett Packard Enterprise Development LP */

/* Service Management */

#include <linux/cred.h>

#include "cass_core.h"
#include "cxi_rxtx_profile.h"
#include "cxi_rxtx_profile_list.h"
#include "cass_ss1_debugfs.h"

static bool disable_default_svc = true;
module_param(disable_default_svc, bool, 0444);
MODULE_PARM_DESC(disable_default_svc, "Disable the default service.");

static bool default_svc_test_mode;
module_param(default_svc_test_mode, bool, 0444);
MODULE_PARM_DESC(default_svc_test_mode,
		 "Remove all safety rails for default service.");

static int default_svc_num_tles = 512;
module_param(default_svc_num_tles, int, 0644);
MODULE_PARM_DESC(default_svc_num_tles,
		 "Number of reserved TLEs for default service");

static unsigned int default_vnis[CXI_SVC_MAX_VNIS] = {1, 10, 0, 0};
module_param_array(default_vnis, uint, NULL, 0444);
MODULE_PARM_DESC(default_vnis,
		 "Default VNIS. Should be consistent at the fabric level");

static void svc_destroy(struct cass_dev *hw, struct cxi_svc_priv *svc_priv);

static enum cxi_resource_type stype_to_rtype(enum cxi_rsrc_type type, int pe)
{
	switch (type) {
	case CXI_RSRC_TYPE_PTE:
		return CXI_RESOURCE_PTLTE;
	case CXI_RSRC_TYPE_TXQ:
		return CXI_RESOURCE_TXQ;
	case CXI_RSRC_TYPE_TGQ:
		return CXI_RESOURCE_TGQ;
	case CXI_RSRC_TYPE_EQ:
		return CXI_RESOURCE_EQ;
	case CXI_RSRC_TYPE_CT:
		return CXI_RESOURCE_CT;
	case CXI_RSRC_TYPE_LE:
		return CXI_RESOURCE_PE0_LE + pe;
	case CXI_RSRC_TYPE_TLE:
		return CXI_RESOURCE_TLE;
	case CXI_RSRC_TYPE_AC:
		return CXI_RESOURCE_AC;
	default:
		return CXI_RESOURCE_MAX;
	}
}

static void copy_rsrc_use(struct cxi_dev *dev, struct cxi_rsrc_use *rsrcs,
			  struct cxi_rgroup *rgroup)
{
	int rc;
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	union c_cq_sts_tle_in_use tle_in_use;
	int type;
	enum cxi_resource_type rtype;
	struct cxi_resource_entry *entry;

	for (type = 0; type < CXI_RSRC_TYPE_MAX; type++) {
		rtype = stype_to_rtype(type, 0);
		if (type == CXI_RSRC_TYPE_TLE) {
			if (cxi_rgroup_tle_pool_id(rgroup) == -1)
				continue;

			cass_read(hw,
				  C_CQ_STS_TLE_IN_USE(cxi_rgroup_tle_pool_id(rgroup)),
				  &tle_in_use, sizeof(tle_in_use));
			rsrcs->in_use[type] = tle_in_use.count;
			rsrcs->tle_pool_id = cxi_rgroup_tle_pool_id(rgroup);
		} else {
			rc = cxi_rgroup_get_resource_entry(rgroup,
							   rtype, &entry);
			if (rc) {
				rsrcs->in_use[type] = 0;
				continue;
			}

			rsrcs->in_use[type] = entry->limits.in_use;
		}
	}
}

void cass_cfg_tle_pool(struct cass_dev *hw, int pool_id,
		       const struct cxi_limits *tles, bool release)
{
	union c_cq_cfg_tle_pool tle_pool;

	if (release) {
		tle_pool.max_alloc = 0;
		tle_pool.num_reserved = 0;
	} else {
		tle_pool.max_alloc = tles->max;
		tle_pool.num_reserved = tles->res;
	}
	cass_write(hw, C_CQ_CFG_TLE_POOL(pool_id), &tle_pool,
			   sizeof(tle_pool));
}

void cass_tle_init(struct cass_dev *hw)
{
	int i;
	union c_cq_cfg_sts_tle_shared tle_shared;
	union c_cq_cfg_tle_pool tle_pool_cfg = {
		.max_alloc = 0,
		.num_reserved = 0,
	};

	/* Ensure there is no shared space for TLEs */
	tle_shared.num_shared = 0;
	cass_write(hw, C_CQ_CFG_STS_TLE_SHARED, &tle_shared,
		   sizeof(tle_shared));

	/* Disable all pools */
	for (i = 0; i < C_CQ_CFG_TLE_POOL_ENTRIES; i++)
		cass_write(hw, C_CQ_CFG_TLE_POOL(i), &tle_pool_cfg,
			   sizeof(tle_pool_cfg));
}

static void default_rsrc_limits(struct cxi_rsrc_limits *limits)
{
	memset(limits, 0, sizeof(*limits));

	limits->type[CXI_RSRC_TYPE_PTE].max = C_NUM_PTLTES;
	limits->type[CXI_RSRC_TYPE_TXQ].max = C_NUM_TRANSMIT_CQS;
	limits->type[CXI_RSRC_TYPE_TGQ].max = C_NUM_TARGET_CQS;
	limits->type[CXI_RSRC_TYPE_EQ].max = EQS_AVAIL;
	limits->type[CXI_RSRC_TYPE_CT].max = CTS_AVAIL;
	limits->type[CXI_RSRC_TYPE_LE].max = pe_total_les;
	limits->type[CXI_RSRC_TYPE_TLE].max = C_NUM_TLES;
	limits->type[CXI_RSRC_TYPE_AC].max = ACS_AVAIL;
}

int cass_svc_init(struct cass_dev *hw)
{
	static struct cxi_svc_desc svc_desc = {
		.resource_limits = true,
	};
	int i, svc_id;
	struct cxi_rsrc_limits limits;

	/* TODO differentiate PF/VF */
	if (!hw->cdev.is_physfn)
		return 0;

	default_rsrc_limits(&hw->cdev.prop.rsrcs);

	for (i = 0; i < CXI_RSRC_TYPE_MAX; i++) {
		/* Set up resource limits for default service */
		if (i != CXI_RSRC_TYPE_TLE) {
			limits.type[i].max = hw->cdev.prop.rsrcs.type[i].max;
			limits.type[i].res = 0;
		} else {
			limits.type[i].res = max(CASS_MIN_POOL_TLES,
						 default_svc_num_tles);
			limits.type[i].max = limits.type[i].res;
		}
	}
	svc_desc.limits = limits;

	for (i = CXI_TC_DEDICATED_ACCESS; i <= CXI_TC_BEST_EFFORT; ++i)
		svc_desc.tcs[i] = true;

	if (!default_svc_test_mode) {
		svc_desc.restricted_vnis = true;
		svc_desc.num_vld_vnis = 0;

		for (i = 0; i < CXI_SVC_MAX_VNIS; i++) {
			if (!is_vni_valid(default_vnis[i]))
				break;
			svc_desc.num_vld_vnis++;
			svc_desc.vnis[i] = default_vnis[i];
		}

		if (!svc_desc.num_vld_vnis)
			return -EINVAL;
	}

	mutex_init(&hw->svc_lock);
	idr_init(&hw->svc_ids);
	INIT_LIST_HEAD(&hw->svc_list);
	hw->svc_count = 0;

	/* Create default service. It will get the default ID of
	 * CXI_DEFAULT_SVC_ID
	 */
	svc_id = cxi_svc_alloc(&hw->cdev, &svc_desc, NULL, "default");
	if (svc_id < 0)
		return svc_id;

	svc_service_debugfs_create(hw);

	return 0;
}

/* Check if a there are enough unused instances of a particular resource */
static bool rsrc_available(struct cass_dev *hw,
			   struct cxi_rsrc_limits *limits,
			   enum cxi_rsrc_type type, int pe,
			   struct cxi_svc_fail_info *fail_info)
{
	u16 shared_avail;
	enum cxi_resource_type rtype;

	rtype = stype_to_rtype(type, pe);
	if (rtype >= CXI_RESOURCE_MAX)
		return false;

	shared_avail = hw->resource_use[rtype].shared;

	/* Always fill out fail_info if a resource was requested so it
	 * accurately reflects how many of resource X was available.
	 * If all resources are in fact available, it won't be sent
	 * back to the user.
	 */
	if (fail_info)
		fail_info->rsrc_avail[type] = shared_avail;

	if (limits->type[type].res > shared_avail)
		return false;

	return true;
}

/* Return resource reservations upon destruction of a service
 * Caller must hold hw->svc_lock.
 */
static void free_rsrc(struct cxi_svc_priv *svc_priv,
		      enum cxi_rsrc_type type)
{
	int rc;
	int pe;
	enum cxi_resource_type rtype = stype_to_rtype(type, 0);

	if (type == CXI_RSRC_TYPE_LE) {
		for (pe = 0; pe < C_PE_COUNT; pe++) {
			rc = cxi_rgroup_delete_resource(svc_priv->rgroup,
							rtype + pe);
			if (rc)
				pr_debug("delete resource %s failed %d\n",
					 cxi_resource_type_to_str(type + pe),
					 rc);
		}

		return;
	}

	rc = cxi_rgroup_delete_resource(svc_priv->rgroup, rtype);
	if (rc)
		pr_debug("delete resource %s failed %d\n",
			 cxi_rsrc_type_to_str(type), rc);
}

static void free_rsrcs(struct cxi_svc_priv *svc_priv)
{
	int i;

	for (i = 0; i < CXI_RSRC_TYPE_MAX; i++)
		free_rsrc(svc_priv, i);
}

static int add_resource(struct cxi_rgroup *rgroup, enum cxi_rsrc_type type,
			struct cxi_resource_limits *limits)
{
	int rc;
	int pe;

	if (type == CXI_RSRC_TYPE_LE) {
		for (pe = 0; pe < C_PE_COUNT; pe++) {
			rc = cxi_rgroup_add_resource(rgroup,
						     stype_to_rtype(type, pe),
						     limits);
			if (rc) {
				pr_debug("add resource %s PE %d failed\n",
					 cxi_rsrc_type_to_str(type), pe);
				return rc;
			}
		}

		return rc;
	}

	rc = cxi_rgroup_add_resource(rgroup,
				     stype_to_rtype(type, 0), limits);
	if (rc)
		pr_debug("add resource %s failed\n",
			 cxi_rsrc_type_to_str(type));

	return rc;
}

/* For each resource requested, check if enough of that resource is available.
 * If they are all available, update the reserved values in the device.
 * Caller must hold hw->svc_lock.
 */
static int reserve_rsrcs(struct cass_dev *hw,
			 struct cxi_svc_priv *svc_priv,
			 struct cxi_svc_fail_info *fail_info)
{
	int i;
	int pe;
	int rc = 0;
	struct cxi_rgroup *rgroup = svc_priv->rgroup;
	struct cxi_rsrc_limits *limits = &svc_priv->svc_desc.limits;

	/* Default pool for default svc or when there are no LE limits */
	if (!svc_priv->svc_desc.resource_limits)
		default_rsrc_limits(limits);

	for (i = CXI_RSRC_TYPE_PTE; i < CXI_RSRC_TYPE_MAX; i++) {
		if (!limits->type[i].res && !limits->type[i].max)
			continue;

		if (i == CXI_RSRC_TYPE_TLE) {
			if (svc_priv->svc_desc.svc_id != CXI_DEFAULT_SVC_ID) {
				/* Ensure TLE max/res are at least CASS_MIN_POOL_TLES */
				if (limits->type[i].res < CASS_MIN_POOL_TLES)
					limits->type[i].res = CASS_MIN_POOL_TLES;
				/* Force TLE max/res to be equal */
				limits->type[i].max = limits->type[i].res;
			}
		} else if (i == CXI_RSRC_TYPE_LE) {
			for (pe = 0; pe < C_PE_COUNT; pe++) {
				if (!rsrc_available(hw, limits, i, pe,
						    fail_info)) {
					pr_debug("resource %s PE %d unavailable\n",
						 cxi_rsrc_type_to_str(i), pe);
					rc = -ENOSPC;
					goto nospace;
				}
			}
		} else if (!rsrc_available(hw, limits, i, 0, fail_info)) {
			pr_debug("resource %s unavailable\n",
				 cxi_rsrc_type_to_str(i));
			rc = -ENOSPC;
			goto nospace;
		}
	}

nospace:
	if (rc)
		return rc;

	/* Now reserve resources since needed ones are available */
	for (i = CXI_RSRC_TYPE_PTE; i < CXI_RSRC_TYPE_MAX; i++) {
		struct cxi_resource_limits lim = {
			.reserved = limits->type[i].res,
			.max = limits->type[i].max
		};

		if (!lim.reserved && !lim.max)
			continue;

		rc = add_resource(rgroup, i, &lim);
		if (rc) {
			pr_debug("resource %s add_resource failed %d\n",
				 cxi_rsrc_type_to_str(i), rc);

			if (rc == -EBADR) {
				if (i == CXI_RSRC_TYPE_TLE)
					fail_info->no_tle_pools = true;

				if (i == CXI_RSRC_TYPE_LE)
					fail_info->no_le_pools = true;
				rc = -ENOSPC;
			}

			goto err;
		}
	}

	return 0;

err:
	/* Remove any resources we already allocated */
	for (--i; i >= CXI_RSRC_TYPE_PTE; i--) {
		if (!limits->type[i].res)
			continue;

		free_rsrc(svc_priv, i);
	}

	return rc;
}

/* Basic sanity checks for user provided service descriptor */
static int validate_descriptor(struct cass_dev *hw,
			       const struct cxi_svc_desc *svc_desc)
{
	int i;

	if (svc_desc->restricted_vnis) {
		if (svc_desc->num_vld_vnis > CXI_SVC_MAX_VNIS)
			return -EINVAL;
		for (i = 0; i < svc_desc->num_vld_vnis; i++) {
			if (!is_vni_valid(svc_desc->vnis[i]))
				return -EINVAL;
		}
	}

	if (svc_desc->restricted_members) {
		for (i = 0; i < CXI_SVC_MAX_MEMBERS; i++) {
			if (svc_desc->members[i].type < 0 ||
			    svc_desc->members[i].type >= CXI_SVC_MEMBER_MAX)
				return -EINVAL;
		}
	}

	if (svc_desc->resource_limits) {
		for (i = 0; i < CXI_RSRC_TYPE_MAX; i++) {
			if (svc_desc->limits.type[i].max <
			    svc_desc->limits.type[i].res)
				return -EINVAL;
			if (svc_desc->limits.type[i].max >
			    hw->cdev.prop.rsrcs.type[i].max)
				return -EINVAL;
		}
	}

	return 0;
}

static enum cxi_ac_type svc_mbr_to_ac_type(enum cxi_svc_member_type type,
					   bool restricted_members)
{
	if (!restricted_members)
		return CXI_AC_OPEN;

	switch (type) {
	case CXI_SVC_MEMBER_UID:
		return CXI_AC_UID;
	case CXI_SVC_MEMBER_GID:
		return CXI_AC_GID;
	case CXI_SVC_MEMBER_IGNORE:
		fallthrough;
	default:
		return 0;
	}
}

static void set_tcs(struct cxi_dev *dev, struct cxi_svc_priv *svc_priv)
{
	int i, j;
	struct cxi_tx_profile *tx_profile;
	struct cxi_svc_desc *svc_desc = &svc_priv->svc_desc;

	for (i = 0; i < svc_priv->svc_desc.num_vld_vnis; i++) {
		tx_profile = svc_priv->tx_profile[i];

		for (j = 0; j < CXI_TC_MAX; j++)
			if (!svc_desc->restricted_tcs || svc_desc->tcs[j])
				cxi_tx_profile_set_tc(tx_profile, j, true);
	}
}

static int alloc_rgroup_ac_entries(struct cxi_dev *dev,
				   struct cxi_svc_priv *svc_priv)
{
	int i;
	int rc;
	enum cxi_ac_type type;
	unsigned int ac_entry_id;
	union cxi_ac_data ac_data = {};
	struct cxi_svc_desc *svc_desc = &svc_priv->svc_desc;

	if (!svc_desc->restricted_members)
		return cxi_rgroup_add_ac_entry(svc_priv->rgroup, CXI_AC_OPEN,
					       &ac_data, &ac_entry_id);

	for (i = 0; i < CXI_SVC_MAX_MEMBERS; i++) {
		/* No AC entry for member[].type of CXI_SVC_MEMBER_IGNORE */
		type = svc_mbr_to_ac_type(svc_desc->members[i].type,
					  svc_desc->restricted_members);
		if (!type)
			continue;

		if (type == CXI_AC_UID)
			ac_data.uid = svc_desc->members[i].svc_member.uid;
		else if (type == CXI_AC_GID)
			ac_data.gid = svc_desc->members[i].svc_member.gid;

		rc = cxi_rgroup_add_ac_entry(svc_priv->rgroup, type, &ac_data,
					     &ac_entry_id);
		if (rc)
			goto cleanup;
	}

	return 0;

cleanup:
	cxi_ac_entry_list_destroy(&svc_priv->rgroup->ac_entry_list);

	return rc;
}

static void remove_profile_ac_entries(struct cxi_dev *dev,
				      struct cxi_svc_priv *svc_priv)
{
	int i;
	struct cxi_svc_desc *svc_desc = &svc_priv->svc_desc;

	for (i = 0; i < svc_desc->num_vld_vnis; i++) {
		cxi_tx_profile_remove_ac_entries(svc_priv->tx_profile[i]);
		cxi_rx_profile_remove_ac_entries(svc_priv->rx_profile[i]);
	}
}

/* No AC entry will be allocated for a member[].type of
 * CXI_SVC_MEMBER_IGNORE.
 */
static int alloc_profile_ac_entries(struct cxi_dev *dev,
				    struct cxi_svc_priv *svc_priv)
{
	int i;
	int j;
	int rc;
	enum cxi_ac_type type;
	unsigned int ac_entry_id;
	struct cxi_svc_desc *svc_desc = &svc_priv->svc_desc;

	if (!svc_desc->restricted_members) {
		for (j = 0; j < svc_priv->svc_desc.num_vld_vnis; j++) {
			rc = cxi_rx_profile_add_ac_entry(svc_priv->rx_profile[j],
							 CXI_AC_OPEN, 0, 0,
							 &ac_entry_id);
			if (rc)
				goto cleanup;

			rc = cxi_tx_profile_add_ac_entry(svc_priv->tx_profile[j],
							 CXI_AC_OPEN, 0, 0,
							 &ac_entry_id);
			if (rc)
				goto cleanup;
		}

		return 0;
	}

	for (j = 0; j < svc_priv->svc_desc.num_vld_vnis; j++) {
		for (i = 0; i < CXI_SVC_MAX_MEMBERS; i++) {
			type = svc_mbr_to_ac_type(svc_desc->members[i].type,
						  svc_desc->restricted_members);
			if (!type)
				continue;

			rc = cxi_rx_profile_add_ac_entry(
					svc_priv->rx_profile[j], type,
					svc_desc->members[i].svc_member.uid,
					svc_desc->members[i].svc_member.gid,
					&ac_entry_id);
			if (rc)
				goto cleanup;

			rc = cxi_tx_profile_add_ac_entry(
					svc_priv->tx_profile[j], type,
					svc_desc->members[i].svc_member.uid,
					svc_desc->members[i].svc_member.gid,
					&ac_entry_id);
			if (rc)
				goto cleanup;
		}
	}

	return 0;

cleanup:
	remove_profile_ac_entries(dev, svc_priv);

	return rc;
}

static void release_rxtx_profiles(struct cxi_dev *dev,
				  struct cxi_svc_priv *svc_priv)
{
	int i;

	remove_profile_ac_entries(dev, svc_priv);

	for (i = 0; i < svc_priv->svc_desc.num_vld_vnis; i++) {
		cxi_rx_profile_dec_refcount(dev, svc_priv->rx_profile[i]);
		cxi_tx_profile_dec_refcount(dev, svc_priv->tx_profile[i],
					    true);
	}
}

/* Setup up to 4 RX/TX Profiles if restricted_vnis = 1
 * Otherwise set up a single RX/TX profile for a requested VNI range
 */
static int alloc_rxtx_profiles(struct cxi_dev *dev,
			       struct cxi_svc_priv *svc_priv,
			       const struct cxi_rxtx_vni_attr *vni_range_attr)
{
	int i;
	int rc;
	const struct cxi_rxtx_vni_attr *vni_attr;
	struct cxi_svc_desc *svc_desc = &svc_priv->svc_desc;

	if (!svc_desc->restricted_vnis) {
		if (!vni_range_attr) {
			cxidev_err(dev, "vni_range_attr NULL for vni_range\n");
			return -EINVAL;
		}
		svc_priv->svc_desc.num_vld_vnis = 1;
	}

	for (i = 0; i < svc_priv->svc_desc.num_vld_vnis; i++) {
		struct cxi_tx_attr tx_attr = {};
		struct cxi_rx_attr rx_attr = {};
		struct cxi_rxtx_vni_attr restricted_vni_attr = {
			.ignore = 0,
			.match = svc_desc->vnis[i],
			.name = "",
		};

		vni_attr = &restricted_vni_attr;
		if (!svc_desc->restricted_vnis)
			vni_attr = vni_range_attr;

		tx_attr.vni_attr = *vni_attr;
		rx_attr.vni_attr = *vni_attr;

		svc_priv->tx_profile[i] = cxi_dev_alloc_tx_profile(dev,
								   &tx_attr);
		if (IS_ERR(svc_priv->tx_profile[i])) {
			rc = PTR_ERR(svc_priv->tx_profile[i]);
			svc_priv->tx_profile[i] = NULL;
			goto release_profiles;
		}

		svc_priv->rx_profile[i] = cxi_dev_alloc_rx_profile(dev,
								   &rx_attr);
		if (IS_ERR(svc_priv->rx_profile[i])) {
			rc = PTR_ERR(svc_priv->rx_profile[i]);
			svc_priv->rx_profile[i] = NULL;
			goto release_profiles;
		}
	}

	rc = alloc_profile_ac_entries(dev, svc_priv);
	if (rc)
		goto release_profiles;

	set_tcs(dev, svc_priv);

	return 0;

release_profiles:
	release_rxtx_profiles(dev, svc_priv);
	return rc;
}

static int svc_enable(struct cxi_dev *dev, struct cxi_svc_priv *svc_priv,
		      bool enable)
	__must_hold(&hw->svc_lock)
{
	int i;
	int rc = 0;

	if (enable) {
		cxi_rgroup_enable(svc_priv->rgroup);
		svc_priv->svc_desc.enable = 1;

		for (i = 0; i < svc_priv->svc_desc.num_vld_vnis; i++) {
			rc = cxi_tx_profile_enable(dev,
						   svc_priv->tx_profile[i]);
			if (rc)
				goto disable;

			rc = cxi_rx_profile_enable(dev,
						   svc_priv->rx_profile[i]);
			if (rc)
				goto disable;
		}

		return 0;
	}

disable:
	cxi_rgroup_disable(svc_priv->rgroup);
	svc_priv->svc_desc.enable = 0;

	for (i = 0; i < svc_priv->svc_desc.num_vld_vnis; i++) {
		cxi_tx_profile_disable(dev, svc_priv->tx_profile[i]);
		cxi_rx_profile_disable(dev, svc_priv->rx_profile[i]);
	}

	return rc;
}

/**
 * cxi_svc_alloc() - Allocate a service
 *
 * @dev: Cassini Device
 * @svc_desc: A service descriptor that contains requests for various resources,
 *            and optionally identifies member processes, tcs, vnis, etc. see
 *            cxi_svc_desc.
 * @fail_info: extra information when a failure occurs
 * @name: name for service
 *
 * Return: Service ID on success. Else, negative errno value.
 */
int cxi_svc_alloc(struct cxi_dev *dev, const struct cxi_svc_desc *svc_desc,
		  struct cxi_svc_fail_info *fail_info, char *name)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_svc_priv *svc_priv;
	int rc;
	struct cxi_rgroup *rgroup;
	struct cxi_rgroup_attr attr = {
		.cntr_pool_id = svc_desc->cntr_pool_id,
		.system_service = svc_desc->is_system_svc,
		.lnis_per_rgid = CXI_DEFAULT_LNIS_PER_RGID,
	};

	rc = validate_descriptor(hw, svc_desc);
	if (rc)
		return rc;

	svc_priv = kzalloc(sizeof(*svc_priv), GFP_KERNEL);
	if (!svc_priv)
		return -ENOMEM;
	svc_priv->svc_desc = *svc_desc;

	rgroup = cxi_dev_alloc_rgroup(dev, &attr);
	if (IS_ERR(rgroup)) {
		rc = PTR_ERR(rgroup);
		goto free_svc;
	}

	svc_priv->rgroup = rgroup;
	svc_priv->svc_desc.svc_id = cxi_rgroup_id(rgroup);

	rc = idr_alloc(&hw->svc_ids, svc_priv, cxi_rgroup_id(rgroup),
		       cxi_rgroup_id(rgroup) + 1,
		       GFP_NOWAIT);
	if (rc < 0) {
		cxidev_err(&hw->cdev, "%s Service idr could not be obtained for rgroup ID %d rc:%d\n",
			   hw->cdev.name, cxi_rgroup_id(rgroup), rc);
		goto release_rgroup;
	}

	cxi_rgroup_set_name(rgroup, name);

	rc = alloc_rgroup_ac_entries(dev, svc_priv);
	if (rc)
		goto remove_idr;

	/* If restricted_vnis is set setup profiles now. Otherwise they will
	 * set up later when a vni range is requested
	 */
	if (svc_desc->restricted_vnis) {
		rc = alloc_rxtx_profiles(dev, svc_priv, NULL);
		if (rc)
			goto remove_rgrp_ac_entries;
	}

	mutex_lock(&hw->svc_lock);
	rc = reserve_rsrcs(hw, svc_priv, fail_info);
	if (rc)
		goto unlock;

	/* SVC is enabled by default for backwards compatibility.
	 * If disable_default_svc is true, the default service
	 * will be disabled.
	 * Setting restricted_vnis = 0 now indicates that a
	 * VNI range will be set up after the service is enabled.
	 * Do not enable the svc/rgroup/profiles until then.
	 */
	if (((cxi_rgroup_id(rgroup) == CXI_DEFAULT_SVC_ID) &&
	     disable_default_svc) ||
	    !svc_desc->restricted_vnis) {
		svc_priv->svc_desc.enable = 0;
	} else {
		rc = svc_enable(dev, svc_priv, true);
		if (rc)
			goto free_resources;
	}

	list_add_tail(&svc_priv->list, &hw->svc_list);
	hw->svc_count++;
	mutex_unlock(&hw->svc_lock);
	refcount_inc(&hw->refcount);

	return cxi_rgroup_id(rgroup);

free_resources:
	free_rsrcs(svc_priv);
unlock:
	mutex_unlock(&hw->svc_lock);
	if (svc_desc->restricted_vnis)
		release_rxtx_profiles(dev, svc_priv);
remove_rgrp_ac_entries:
	cxi_ac_entry_list_destroy(&svc_priv->rgroup->ac_entry_list);
remove_idr:
	idr_remove(&hw->svc_ids, cxi_rgroup_id(rgroup));
release_rgroup:
	cxi_rgroup_dec_refcount(rgroup);
	return rc;
free_svc:
	kfree(svc_priv);

	return rc;
}
EXPORT_SYMBOL(cxi_svc_alloc);

static void svc_destroy(struct cass_dev *hw, struct cxi_svc_priv *svc_priv)
{
	int rc;
	int svc_id = cxi_rgroup_id(svc_priv->rgroup);

	free_rsrcs(svc_priv);

	release_rxtx_profiles(&hw->cdev, svc_priv);

	rc = cxi_rgroup_dec_refcount(svc_priv->rgroup);
	if (rc)
		pr_err("cxi_dev_release_rgroup_by_id failed %d\n", rc);

	idr_remove(&hw->svc_ids, svc_id);
	list_del(&svc_priv->list);
	hw->svc_count--;

	refcount_dec(&hw->refcount);
	kfree(svc_priv);
}

/**
 * cxi_svc_destroy() - Destroy a service
 *
 * @dev: Cassini Device
 * @svc_id: Service ID of service to be destroyed.
 *
 * Return: 0 on success. Else a negative errno value.
 */
int cxi_svc_destroy(struct cxi_dev *dev, u32 svc_id)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_svc_priv *svc_priv;

	/* Don't destroy default svc */
	if (svc_id == CXI_DEFAULT_SVC_ID)
		return -EINVAL;

	mutex_lock(&hw->svc_lock);

	svc_priv = idr_find(&hw->svc_ids, svc_id);
	if (!svc_priv) {
		mutex_unlock(&hw->svc_lock);
		return -EINVAL;
	}

	/* Don't delete if an LNI is still using this SVC */
	if (cxi_rgroup_refcount(svc_priv->rgroup) > 1) {
		mutex_unlock(&hw->svc_lock);
		return -EBUSY;
	}

	svc_destroy(hw, svc_priv);

	mutex_unlock(&hw->svc_lock);

	return 0;
}
EXPORT_SYMBOL(cxi_svc_destroy);

/*
 * cxi_svc_rsrc_list_get - Get per service information on resource usage.
 *
 * @dev: Cassini Device
 * @count: number of services descriptors for which space
 *         has been allocated. 0 initially, to determine count.
 * @rsrc_list: destination to land service descriptors
 *
 * Return: number of service descriptors
 * If the specified count is equal to (or greater than) the number of
 * active service descriptors, they are copied to the provided user
 * buffer.
 */
int cxi_svc_rsrc_list_get(struct cxi_dev *dev, int count,
			  struct cxi_rsrc_use *rsrc_list)
{

	int i = 0;
	struct cxi_svc_priv *svc_priv;
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);

	mutex_lock(&hw->svc_lock);

	if (count < hw->svc_count) {
		mutex_unlock(&hw->svc_lock);
		return hw->svc_count;
	}

	list_for_each_entry(svc_priv, &hw->svc_list, list) {
		copy_rsrc_use(dev, &rsrc_list[i], svc_priv->rgroup);
		rsrc_list[i].svc_id = svc_priv->svc_desc.svc_id;
		i++;
	}

	mutex_unlock(&hw->svc_lock);

	return i;
}
EXPORT_SYMBOL(cxi_svc_rsrc_list_get);

/*
 * cxi_svc_rsrc_get - Get rsrc_use from svc_id
 *
 * @dev: Cassini Device
 * @svc_id: svc_id of the descriptor to find which is equivalent to the
 *          rgroup ID.
 * @rsrc_use: destination to land resource usage
 *
 * Return: 0 on success or a negative errno
 */
int cxi_svc_rsrc_get(struct cxi_dev *dev, unsigned int svc_id,
		     struct cxi_rsrc_use *rsrc_use)
{
	int rc;
	struct cxi_rgroup *rgroup;

	rc = cxi_dev_find_rgroup_inc_refcount(dev, svc_id, &rgroup);
	if (rc)
		return -EINVAL;

	copy_rsrc_use(dev, rsrc_use, rgroup);
	cxi_rgroup_dec_refcount(rgroup);

	return 0;
}
EXPORT_SYMBOL(cxi_svc_rsrc_get);

/*
 * cxi_svc_list_get - Assemble list of active services descriptors
 *
 * @dev: Cassini Device
 * @count: number of services descriptors for which space
 *         has been allocated. 0 initially, to determine count.
 * @svc_list: destination to land service descriptors
 *
 * Return: number of service descriptors
 * If the specified count is equal to (or greater than) the number of
 * active service descriptors, they are copied to the provided user
 * buffer.
 */
int cxi_svc_list_get(struct cxi_dev *dev, int count,
		     struct cxi_svc_desc *svc_list)
{

	int i = 0;
	struct cxi_svc_priv *svc_priv;
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);

	mutex_lock(&hw->svc_lock);

	if (count < hw->svc_count) {
		mutex_unlock(&hw->svc_lock);
		return hw->svc_count;
	}

	list_for_each_entry(svc_priv, &hw->svc_list, list) {
		svc_list[i] = svc_priv->svc_desc;
		i++;
	}
	mutex_unlock(&hw->svc_lock);

	return i;
}
EXPORT_SYMBOL(cxi_svc_list_get);

/*
 * cxi_svc_get - Get svc_desc from svc_id
 *
 * @dev: Cassini Device
 * @svc_id: svc_id of the descriptor to find which is equivalent to the
 *          rgroup ID.
 * @svc_desc: destination to land service descriptor
 *
 * Return: 0 on success or a negative errno
 */
int cxi_svc_get(struct cxi_dev *dev, unsigned int svc_id,
		struct cxi_svc_desc *svc_desc)
{
	struct cxi_svc_priv *svc_priv;
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);

	mutex_lock(&hw->svc_lock);

	/* Find priv descriptor */
	svc_priv = idr_find(&hw->svc_ids, svc_id);
	if (!svc_priv) {
		mutex_unlock(&hw->svc_lock);
		return -EINVAL;
	}

	*svc_desc = svc_priv->svc_desc;
	mutex_unlock(&hw->svc_lock);

	return 0;
}
EXPORT_SYMBOL(cxi_svc_get);

void cxi_free_resource(struct cxi_dev *dev, struct cxi_svc_priv *svc_priv,
		      enum cxi_rsrc_type type)
{
	return cxi_rgroup_free_resource(svc_priv->rgroup,
					stype_to_rtype(type, 0));
}

/* used to allocate ACs, etc. */
int cxi_alloc_resource(struct cxi_dev *dev, struct cxi_svc_priv *svc_priv,
		       enum cxi_rsrc_type type)
{
	return cxi_rgroup_alloc_resource(svc_priv->rgroup,
					 stype_to_rtype(type, 0));
}

/**
 * cxi_svc_enable() - Enable or Disable a service.
 *
 * @dev: Cassini Device
 * @svc_id: Service ID of the service to be enabled.
 * @enable: Boolean value indicating whether to enable or disable the service.
 *
 * Return: 0 on success or negative errno value.
 */
int cxi_svc_enable(struct cxi_dev *dev, unsigned int svc_id, bool enable)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_svc_priv *svc_priv;
	int rc = 0;

	mutex_lock(&hw->svc_lock);

	svc_priv = idr_find(&hw->svc_ids, svc_id);
	if (!svc_priv) {
		rc = -EINVAL;
		cxidev_err(dev, "Invalid service ID: %u\n", svc_id);
		goto unlock;
	}

	/* Service must be unused for it to be enabled/disabled. */
	if (refcount_read(&svc_priv->rgroup->state.refcount) > 1) {
		rc = -EBUSY;
		goto unlock;
	}

	rc = svc_enable(dev, svc_priv, enable);
unlock:
	mutex_unlock(&hw->svc_lock);
	return rc;
}
EXPORT_SYMBOL(cxi_svc_enable);

/**
 * cxi_svc_update() - Modify an existing service.
 *
 * @dev: Cassini Device
 * @svc_desc: A service descriptor that contains requests for various resources,
 *            and optionally identifies member processes, tcs, vnis, etc. see
 *            cxi_svc_desc.
 *
 * Currently does not honor changes to resource limits in a svc_desc.
 *
 * Return: 0 on success. Else, negative errno value.
 */
int cxi_svc_update(struct cxi_dev *dev, const struct cxi_svc_desc *svc_desc)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_svc_priv *svc_priv;
	int rc;

	rc = validate_descriptor(hw, svc_desc);
	if (rc)
		return rc;

	mutex_lock(&hw->svc_lock);

	/* Find priv descriptor */
	svc_priv = idr_find(&hw->svc_ids, svc_desc->svc_id);
	if (!svc_priv) {
		rc = -EINVAL;
		goto error;
	}

	/* Service must be unused for it to be updated. */
	if (refcount_read(&svc_priv->rgroup->state.refcount) > 1) {
		rc = -EBUSY;
		goto error;
	}

	/* TODO Handle Resource Reservation Changes */
	if (svc_priv->svc_desc.resource_limits != svc_desc->resource_limits) {
		rc = -EINVAL;
		goto error;
	}

	rc = svc_enable(dev, svc_priv, svc_desc->enable);
	if (rc)
		goto error;

	/* Update TCs, VNIs, Members */
	svc_priv->svc_desc.restricted_members = svc_desc->restricted_members;
	svc_priv->svc_desc.restricted_vnis = svc_desc->restricted_vnis;
	svc_priv->svc_desc.num_vld_vnis = svc_desc->num_vld_vnis;
	svc_priv->svc_desc.restricted_tcs = svc_desc->restricted_tcs;
	svc_priv->svc_desc.cntr_pool_id = svc_desc->cntr_pool_id;
	svc_priv->svc_desc.enable = svc_desc->enable;

	memcpy(svc_priv->svc_desc.tcs, svc_desc->tcs, sizeof(svc_desc->tcs));
	memcpy(svc_priv->svc_desc.vnis, svc_desc->vnis, sizeof(svc_desc->vnis));
	memcpy(svc_priv->svc_desc.members, svc_desc->members, sizeof(svc_desc->members));
	// TODO: update TX profile?
error:
	mutex_unlock(&hw->svc_lock);
	return rc;
}
EXPORT_SYMBOL(cxi_svc_update);

/**
 * cxi_svc_set_lpr() - Update an existing service to set the LNIs per RGID
 *
 * For backwards compatibility, check if service is in use instead of
 * checking if rgroup is enabled.
 *
 * @dev: Cassini Device
 * @svc_id: Service ID of service to be updated.
 * @lnis_per_rgid: New value of lnis_per_rgid
 *
 * Return: 0 on success or negative errno value.
 */
int cxi_svc_set_lpr(struct cxi_dev *dev, unsigned int svc_id,
		    unsigned int lnis_per_rgid)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_svc_priv *svc_priv;

	if (lnis_per_rgid > C_NUM_LACS)
		return -EINVAL;

	mutex_lock(&hw->svc_lock);

	svc_priv = idr_find(&hw->svc_ids, svc_id);
	if (!svc_priv) {
		mutex_unlock(&hw->svc_lock);
		return -EINVAL;
	}

	/* Service must be unused for it to be updated. */
	if (cxi_rgroup_refcount(svc_priv->rgroup) > 1) {
		mutex_unlock(&hw->svc_lock);
		return -EBUSY;
	}

	cxi_rgroup_set_lnis_per_rgid_compat(svc_priv->rgroup, lnis_per_rgid);

	mutex_unlock(&hw->svc_lock);

	return 0;
}
EXPORT_SYMBOL(cxi_svc_set_lpr);

/**
 * cxi_svc_get_lpr() - Get the LNIs per RGID of the indicated service
 *
 * @dev: Cassini Device
 * @svc_id: Service ID of service to be updated.
 *
 * Return: lnis_per_rgid on success or negative errno value.
 */
int cxi_svc_get_lpr(struct cxi_dev *dev, unsigned int svc_id)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_svc_priv *svc_priv;

	mutex_lock(&hw->svc_lock);

	svc_priv = idr_find(&hw->svc_ids, svc_id);
	if (!svc_priv) {
		mutex_unlock(&hw->svc_lock);
		return -EINVAL;
	}

	mutex_unlock(&hw->svc_lock);

	return cxi_rgroup_lnis_per_rgid(svc_priv->rgroup);
}
EXPORT_SYMBOL(cxi_svc_get_lpr);

/**
 * cxi_svc_set_exclusive_cp() - Set the exclusive_cp bit for a service
 *
 * @dev: Cassini Device
 * @svc_id: Service ID of service to be updated.
 * @exclusive_cp: New value for exclusive_cp (true or false)
 *
 * Return: 0 on success or negative errno value.
 */
int cxi_svc_set_exclusive_cp(struct cxi_dev *dev, unsigned int svc_id,
			     bool exclusive_cp)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_svc_priv *svc_priv;
	int rc;

	mutex_lock(&hw->svc_lock);

	svc_priv = idr_find(&hw->svc_ids, svc_id);
	if (!svc_priv) {
		rc = -EINVAL;
		goto unlock;
	}

	if (svc_priv->svc_desc.restricted_vnis) {
		cxidev_err(dev, "Exclusive CP not allowed with restricted VNIs\n");
		rc = -EINVAL;
		goto unlock;
	}

	/* One VNI Range will be allowed, tied to tx_profile 0.
	 * This call will fail if the svc/tx_profile is already enabled
	 */
	if (!svc_priv->tx_profile[0]) {
		cxidev_err(dev, "tx_profile[0] not initialized for svc_id: %d\n",
			   svc_id);
		rc = -EINVAL;
		goto unlock;
	}

	rc = cxi_tx_profile_set_exclusive_cp(svc_priv->tx_profile[0],
					     exclusive_cp);
	if (rc)
		cxidev_err(dev, "Failed to set exclusive CP for svc_id: %d rc:%d\n",
			   svc_id, rc);

unlock:
	mutex_unlock(&hw->svc_lock);
	return rc;
}
EXPORT_SYMBOL(cxi_svc_set_exclusive_cp);

/**
 * cxi_svc_get_exclusive_cp() - Get the exclusive_cp bit for a service
 *
 * @dev: Cassini Device
 * @svc_id: Service ID of service to query.
 *
 * Return: 1 if exclusive_cp is set, 0 if not, or negative errno value.
 */
int cxi_svc_get_exclusive_cp(struct cxi_dev *dev, unsigned int svc_id)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_svc_priv *svc_priv;
	int rc;

	mutex_lock(&hw->svc_lock);

	svc_priv = idr_find(&hw->svc_ids, svc_id);
	if (!svc_priv) {
		rc = -EINVAL;
		goto unlock;
	}

	if (!svc_priv->tx_profile[0]) {
		rc = -ENOENT;
		goto unlock;
	}

	/* Grab the first TX Profile and report its exclusive_cp value */
	rc = cxi_tx_profile_exclusive_cp(svc_priv->tx_profile[0]);
unlock:
	mutex_unlock(&hw->svc_lock);

	return rc;
}
EXPORT_SYMBOL(cxi_svc_get_exclusive_cp);

/**
 * cxi_svc_set_vni_range() - Add TX/RX profiles for a contiguous range of VNIs to a service
 *
 * The provided range must be exactly representable as a mask/match pair.
 * Requirements:
 *   - The number of values in the range must be a power of two (1, 2, 4, 8, 16, ...).
 *   - The first value in the range (vni_min) must be a multiple of the range size.
 *   - The svc must not have the restricted_vnis bit set.
 *
 * For example:
 *   64–127: 64 values, starting value (64) is a multiple of the
 *           range size (64), so the range is valid.
 *   32–95 : 64 values, starting value (32) is not a multiple of the
 *           range size (64), so the range is invalid.
 *
 * @dev: Cassini Device
 * @svc_id: Service ID of service to be updated.
 * @vni_min: Minimum VNI value (inclusive)
 * @vni_max: Maximum VNI value (inclusive)
 *
 * Return: 0 on success, or negative errno value on failure.
 */
int cxi_svc_set_vni_range(struct cxi_dev *dev, unsigned int svc_id,
			  unsigned int vni_min, unsigned int vni_max)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_svc_priv *svc_priv;
	struct cxi_rxtx_vni_attr vni_attr = {
		.name = "",
	};
	unsigned int range;
	int rc = 0;

	mutex_lock(&hw->svc_lock);

	svc_priv = idr_find(&hw->svc_ids, svc_id);
	if (!svc_priv) {
		rc = -EINVAL;
		goto unlock_return;
	}

	/* VNI range is incompatible with distinct VNIs */
	if (svc_priv->svc_desc.restricted_vnis) {
		cxidev_err(dev, "Cannot specify distinct VNIs and a VNI range");
		rc = -EINVAL;
		goto unlock_return;
	}
	if (!is_vni_valid(vni_min)) {
		cxidev_err(dev, "vni_min %u invalid", vni_min);
		rc = -EINVAL;
		goto unlock_return;
	}
	if (!is_vni_valid(vni_max)) {
		cxidev_err(dev, "vni_max %u invalid", vni_max);
		rc = -EINVAL;
		goto unlock_return;
	}
	if (vni_max < vni_min) {
		cxidev_err(dev, "vni_max %u is less than vni_min %u",
			   vni_max, vni_min);
		rc = -EINVAL;
		goto unlock_return;
	}

	range = vni_max - vni_min + 1;

	if (!is_power_of_2(range)) {
		cxidev_err(dev, "VNI range [%u, %u] is not a power-of-two size",
			   vni_min, vni_max);
		rc = -EINVAL;
		goto unlock_return;
	}
	/* Validate the range is aligned */
	if (vni_min & (range - 1)) {
		cxidev_err(dev, "VNI range [%u, %u] is not aligned. min (%u) must be a multiple of range size (%u)",
			   vni_min, vni_max, vni_min, range);
		rc = -EINVAL;
		goto unlock_return;
	}

	vni_attr.match = vni_min;
	vni_attr.ignore = range - 1;

	rc = alloc_rxtx_profiles(dev, svc_priv, &vni_attr);

unlock_return:
	mutex_unlock(&hw->svc_lock);
	return rc;
}
EXPORT_SYMBOL(cxi_svc_set_vni_range);

/**
 * cxi_svc_get_vni_range() - Get the VNI range associated with a service
 *
 * @dev: Cassini Device
 * @svc_id: Service ID of service to query.
 * @vni_min: Pointer to store minimum VNI value (inclusive)
 * @vni_max: Pointer to store maximum VNI value (inclusive)
 *
 * Return: 0 on success, or negative errno value.
 */
int cxi_svc_get_vni_range(struct cxi_dev *dev, unsigned int svc_id,
			  unsigned int *vni_min, unsigned int *vni_max)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_svc_priv *svc_priv;
	struct cxi_tx_attr tx_attr;
	int rc = 0;

	mutex_lock(&hw->svc_lock);

	svc_priv = idr_find(&hw->svc_ids, svc_id);
	if (!svc_priv) {
		cxidev_err(dev, "svc_id %u not found", svc_id);
		rc = -EINVAL;
		goto unlock_return;
	}
	if (svc_priv->svc_desc.restricted_vnis) {
		cxidev_err(dev, "svc_id %u does not have a vni range", svc_id);
		rc = -EINVAL;
		goto unlock_return;
	}
	if (!svc_priv->svc_desc.num_vld_vnis) {
		cxidev_err(dev, "svc_id %u has no valid TX/RX profiles", svc_id);
		rc = -EINVAL;
		goto unlock_return;
	}

	if (!svc_priv->tx_profile[0]) {
		rc = -ENOENT;
		goto unlock_return;
	}

	rc = cxi_tx_profile_get_info(dev, svc_priv->tx_profile[0], &tx_attr,
				     NULL);
	if (rc) {
		cxidev_err(dev, "Failed to get TX profile info for svc_id %u",
			   svc_id);
		goto unlock_return;
	}

	*vni_min = tx_attr.vni_attr.match;
	*vni_max = tx_attr.vni_attr.match + tx_attr.vni_attr.ignore;

unlock_return:
	mutex_unlock(&hw->svc_lock);
	return rc;
}
EXPORT_SYMBOL(cxi_svc_get_vni_range);

void cass_svc_fini(struct cass_dev *hw)
{
	struct cxi_svc_priv *svc_priv;
	struct cxi_svc_priv *tmp;

	if (!hw->cdev.is_physfn)
		return;

	debugfs_remove(hw->svc_debug);
	list_for_each_entry_safe(svc_priv, tmp, &hw->svc_list, list)
		svc_destroy(hw, svc_priv);

	idr_destroy(&hw->svc_ids);
}
