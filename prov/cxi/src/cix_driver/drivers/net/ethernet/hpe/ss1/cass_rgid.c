// SPDX-License-Identifier: GPL-2.0
/* Copyright 2024 Hewlett Packard Enterprise Development LP */

/* Cassini RGID management */

#include <linux/debugfs.h>
#include <linux/types.h>

#include "cass_core.h"

struct cass_rgid_priv {
	int lnis;
	int svc_id;
	struct ida lac_table;
	struct idr lcid_table;
	refcount_t refcount;
};

/**
 * cass_rgid_init() - Initialize the RGID array
 *
 * @hw: Cassini device
 */
void cass_rgid_init(struct cass_dev *hw)
{
	xa_init_flags(&hw->rgid_array, XA_FLAGS_ALLOC1);
	refcount_set(&hw->rgids_refcount, 1);
}

/**
 * cass_rgid_fini() - Clean up the RGID array
 *
 * @hw: Cassini device
 */
void cass_rgid_fini(struct cass_dev *hw)
{
	unsigned long id;
	struct cass_rgid_priv *rgid_priv;

	xa_for_each(&hw->rgid_array, id, rgid_priv)
		kfree(rgid_priv);

	xa_destroy(&hw->rgid_array);
}

/**
 * cass_lac_get() - Get an LAC
 *
 * @hw: Cassini device
 * @id: Value of RGID to get an LAC from
 * @return: 0 on success or negative error
 */
int cass_lac_get(struct cass_dev *hw, int id)
{
	struct cass_rgid_priv *rgid_priv = xa_load(&hw->rgid_array, id);

	if (!rgid_priv)
		return 0;

	return ida_alloc_max(&rgid_priv->lac_table, C_NUM_LACS - 1, GFP_KERNEL);
}

/**
 * cass_lac_put() - Free an LAC
 *
 * @hw: Cassini device
 * @id: Value of RGID
 * @lac: LAC to free
 */
void cass_lac_put(struct cass_dev *hw, int id, int lac)
{
	struct cass_rgid_priv *rgid_priv = xa_load(&hw->rgid_array, id);

	if (!rgid_priv)
		return;

	ida_free(&rgid_priv->lac_table, lac);
}

/**
 * cass_lcid_get() - Get an LCID
 *
 * @hw: Cassini device
 * @cp_priv: Communication Profile object
 * @rgid: Value of RGID to get an LCID from
 * @return: 0 or positive value on success or negative error
 */
int cass_lcid_get(struct cass_dev *hw, struct cxi_cp_priv *cp_priv, int rgid)
	__must_hold(&hw->cp_lock)
{
	struct cass_rgid_priv *rgid_priv = xa_load(&hw->rgid_array, rgid);

	if (!rgid_priv)
		return 0;

	return idr_alloc(&rgid_priv->lcid_table, cp_priv, 0, C_COMM_PROF_PER_CQ,
			 GFP_KERNEL);
}

/**
 * cass_lcid_put() - Free an LCID
 *
 * @hw: Cassini device
 * @rgid: Value of RGID
 * @lcid: LCID to free
 */
void cass_lcid_put(struct cass_dev *hw, int rgid, int lcid)
	__must_hold(&hw->cp_lock)
{
	struct cass_rgid_priv *rgid_priv = xa_load(&hw->rgid_array, rgid);

	if (!rgid_priv)
		return;

	idr_remove(&rgid_priv->lcid_table, lcid);
}

/**
 * cass_cp_find() - Get the CP accociated with an RGID and LCID
 *
 * @hw: Cassini device
 * @rgid: Value of RGID
 * @lcid: Value of LCID
 * @return: CP object on success or NULL on error
 */
struct cxi_cp_priv *cass_cp_find(struct cass_dev *hw, int rgid, int lcid)
{
	struct cxi_cp_priv *cp_priv;
	struct cass_rgid_priv *rgid_priv = xa_load(&hw->rgid_array, rgid);

	if (!rgid_priv)
		return NULL;

	mutex_lock(&hw->cp_lock);
	cp_priv = idr_find(&rgid_priv->lcid_table, lcid);
	mutex_unlock(&hw->cp_lock);

	return cp_priv;
}

/**
 * cass_cp_rgid_find() - Get the CP associated with an RGID that matches
 *                       the {vni_pcp,tc,tc_type} tupple
 *
 * @hw: Cassini device
 * @rgid: Value of RGID
 * @vni_pcp: VNI of communication profile. If using an ethernet tc
 *           this argument is treated as PCP.
 * @tc: Traffic class of the communication profile
 * @tc_type: Traffic class type of the communication profile
 * @return: CP object on success or NULL on error
 */
struct cxi_cp_priv *cass_cp_rgid_find(struct cass_dev *hw, int rgid,
				      unsigned int vni_pcp, unsigned int tc,
				      enum cxi_traffic_class_type tc_type)
{
	int lcid;
	struct cxi_cp_priv *cp_priv;
	struct cass_rgid_priv *rgid_priv = xa_load(&hw->rgid_array, rgid);

	if (!rgid_priv)
		return NULL;

	for (lcid = 0; lcid < C_COMM_PROF_PER_CQ; lcid++) {
		cp_priv = idr_find(&rgid_priv->lcid_table, lcid);
		if (!cp_priv)
			continue;

		if (vni_pcp == cp_priv->cp.vni_pcp && tc == cp_priv->cp.tc &&
		    tc_type == cp_priv->cass_cp->tc_type) {
			refcount_inc(&cp_priv->refcount);

			return cp_priv;
		}
	}

	return NULL;
}

/**
 * cass_rgid_get() - Get an RGID from the pool
 *
 * @hw: Cassini device
 * @rgroup: Resource group container
 * @return: rgid on success or negative error
 */
int cass_rgid_get(struct cass_dev *hw, struct cxi_rgroup *rgroup)
{
	int ret;
	u32 id;
	unsigned long idx;
	struct cass_rgid_priv *rgidp;
	struct cass_rgid_priv *rgid_priv;
	unsigned int lnis_per_rgid = rgroup->attr.lnis_per_rgid;

	rgid_priv = kzalloc(sizeof(*rgid_priv), GFP_KERNEL);
	if (!rgid_priv)
		return -ENOMEM;

	xa_lock(&hw->rgid_array);

	xa_for_each(&hw->rgid_array, idx, rgidp) {
		if ((rgidp->svc_id == rgroup->id) &&
		    (rgidp->lnis == lnis_per_rgid) &&
		    (refcount_read(&rgidp->refcount) < rgidp->lnis)) {
			refcount_inc(&rgidp->refcount);
			kfree(rgid_priv);
			id = idx;
			goto done;
		}
	}

	ret = __xa_alloc(&hw->rgid_array, &id, rgid_priv,
			 XA_LIMIT(1, C_NUM_RGIDS - 1), GFP_KERNEL);

	/* -EBUSY is returned when there is no room */
	if (ret == -EBUSY)
		ret = -ENOSPC;

	if (ret)
		goto unlock_free;

	rgid_priv->svc_id = rgroup->id;
	rgid_priv->lnis = lnis_per_rgid;
	refcount_set(&rgid_priv->refcount, 1);
	ida_init(&rgid_priv->lac_table);
	idr_init(&rgid_priv->lcid_table);
	refcount_inc(&hw->rgids_refcount);

done:
	xa_unlock(&hw->rgid_array);

	return id;

unlock_free:
	xa_unlock(&hw->rgid_array);
	kfree(rgid_priv);

	if (ret == -ENOSPC)
		pr_debug("RGID space exhausted\n");
	else
		pr_err("Failed to store id %d ret:%d\n", id, ret);

	return ret;
}

/**
 * cass_rgid_put() - Return RGID to the pool
 *
 * @hw: Cassini device
 * @id: Value of RGID
 */
void cass_rgid_put(struct cass_dev *hw, int id)
{
	struct cass_rgid_priv *rgid_priv;

	xa_lock(&hw->rgid_array);

	rgid_priv = xa_load(&hw->rgid_array, id);
	if (!rgid_priv) {
		pr_err("rgid_priv NULL for id:%d\n", id);
		goto unlock;
	}

	if (refcount_dec_and_test(&rgid_priv->refcount)) {
		WARN_ON(__xa_erase(&hw->rgid_array, id) == NULL);
		refcount_dec(&hw->rgids_refcount);
		kfree(rgid_priv);
	}
unlock:
	xa_unlock(&hw->rgid_array);
}
