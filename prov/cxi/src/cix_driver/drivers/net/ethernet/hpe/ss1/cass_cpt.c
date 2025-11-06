// SPDX-License-Identifier: GPL-2.0
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

/* Communication Profile Table (CPT) management */

#include <linux/iopoll.h>
#include "cass_core.h"

#define ENABLED_PFQ 0

static int cass_cp_clear_table(int id, void *ptr, void *data)
{
	struct cass_cp *cp = ptr;
	struct cass_dev *hw = data;
	const union c_cq_cfg_cp_table cp_cfg = {};

	cass_set_cp_table(hw, cp->id, &cp_cfg);

	return 0;
}

/**
 * cass_clear_cps() - Walk list of Communication profiles and clear
 *                    the entries.
 *
 * @tx_profile: TX profile to clean
 */
void cass_clear_cps(struct cxi_tx_profile *tx_profile)
	__must_hold(&hw->cp_lock)
{
	struct cass_dev *hw = tx_profile->config.hw;

	idr_for_each(&tx_profile->config.cass_cp_table, &cass_cp_clear_table,
		     hw);
}

struct vni_tc_data {
	unsigned int vni_pcp;
	unsigned int tc;
	enum cxi_traffic_class_type tc_type;
};

/**
 * cass_cp_free() - Free a communication profile.
 *
 * @cp: Communication profile to be freed.
 */
static void cass_cp_free(struct cass_cp *cp)
{
	struct cass_dev *hw = cp->hw;
	const union c_cq_cfg_cp_table cp_cfg = {};

	cxidev_dbg(&hw->cdev, "%s cp freed: cp=%u vni_pcp=%u label=%u\n",
		   hw->cdev.name, cp->id, cp->vni_pcp, cp->tc);

	cass_set_cp_table(hw, cp->id, &cp_cfg);

	if (!cp->tx_profile->config.exclusive_cp)
		idr_remove(&cp->tx_profile->config.cass_cp_table, cp->list_id);

	ida_simple_remove(&hw->cp_table, cp->id);

	kfree(cp);
}

/**
 * cass_cp_alloc() - Allocate a new communication profile.
 *
 * @tx_profile: TX profile object
 * @vni_pcp: VNI for communication profile. If using an ethernet tc,
 * this argument is treated as PCP.
 * @tc: Traffic class for communication profile.
 * @tc_type: Traffic class type for communication profile.
 *
 * @return: Valid pointer on success. Else, negative errno value.
 */
static struct cass_cp *cass_cp_alloc(struct cxi_tx_profile *tx_profile,
				     unsigned int vni_pcp,
				     unsigned int tc,
				     enum cxi_traffic_class_type tc_type)
	__must_hold(&hw->cp_lock)
{
	struct cass_cp *cp;
	unsigned int cp_id;
	int rc;
	struct cass_dev *hw = tx_profile->config.hw;
	union c_cq_cfg_cp_table cp_cfg = {
		.valid = 1,
		.hrp_vld = tc_type == CXI_TC_TYPE_HRP,
	};
	union c_cq_cfg_cp_fl_table flow_cfg = {};
	struct cass_tc_cfg tc_cfg;

	rc = cass_tc_find(hw, tc, tc_type, vni_pcp, &tc_cfg);
	if (rc) {
		cxidev_dbg(&hw->cdev, "%s TC mapping does not exist, TC: %s TYPE: %s",
			   hw->cdev.name, cxi_tc_to_str(tc),
			   cxi_tc_type_to_str(tc_type));
		return ERR_PTR(-EINVAL);
	}

	cp = kzalloc(sizeof(*cp), GFP_KERNEL);
	if (!cp)
		return ERR_PTR(-ENOMEM);

	rc = ida_simple_get(&hw->cp_table, 1, C_CQ_CFG_CP_TABLE_ENTRIES,
			    GFP_KERNEL);
	if (rc < 0) {
		cxidev_dbg(&hw->cdev,
			   "%s hardware communication profiles exhausted\n",
			   hw->cdev.name);
		goto error_free_cp;
	}

	cp_id = rc;

	if (!tx_profile->config.exclusive_cp) {
		rc = idr_alloc(&tx_profile->config.cass_cp_table, cp, 1, 0, GFP_KERNEL);
		if (rc < 0)
			goto error_remove_ida;
	}

	refcount_set(&cp->ref, 1);

	cp->list_id = rc;
	cp->id = cp_id;
	cp->vni_pcp = vni_pcp;
	cp->tc = tc;
	cp->tc_type = tc_type;
	cp->hw = hw;
	cp->tx_profile = tx_profile;

	cp_cfg.vni = vni_pcp;
	cp_cfg.dscp_unrsto = tc_cfg.unres_req_dscp;
	cp_cfg.dscp_rstuno = tc_cfg.res_req_dscp;

	if (is_eth_tc(tc)) {
		cp_cfg.srb_disabled = 1;
		cp_cfg.sct_disabled = 1;
		cp_cfg.smt_disabled = 1;
	}

	cass_set_cp_table(hw, cp_id, &cp_cfg);

	flow_cfg.pfq = tc_cfg.ocuset;
	flow_cfg.tc = tc_cfg.cq_tc;

	cass_set_cp_fl_table(hw, cp_id, &flow_cfg);

	cxidev_dbg(&hw->cdev,
		   "%s cp allocated: tc=%s tc_type=%s cp=%u vni_pcp=%u label=%u tc=%u pfq=%u exclusive=%u\n",
		   hw->cdev.name, cxi_tc_to_str(tc),
		   cxi_tc_type_to_str(tc_type), cp->id, cp->vni_pcp, cp->tc,
		   flow_cfg.tc, flow_cfg.pfq, cp->tx_profile->config.exclusive_cp);

	return cp;

error_remove_ida:
	ida_simple_remove(&hw->cp_table, cp_id);
error_free_cp:
	kfree(cp);

	return ERR_PTR(rc);
}

/**
 * cass_cp_put() - Put a communication profile.
 *
 * @cp: Communication profile where reference should be put.
 */
static void cass_cp_put(struct cass_cp *cp)
	__must_hold(&hw->cp_lock)
{
	if (refcount_dec_and_test(&cp->ref))
		cass_cp_free(cp);
}

static int cp_match(int id, void *ptr, void *data)
{
	struct cass_cp *cp = ptr;
	struct vni_tc_data *vni_tc = data;

	if (vni_tc->vni_pcp == cp->vni_pcp &&
	    vni_tc->tc == cp->tc &&
	    vni_tc->tc_type == cp->tc_type)
		return cp->list_id;

	return 0;
}

/**
 * cass_cp_get() - Get a communication profile.
 *
 * @tx_profile: TX profile
 * @vni_pcp: VNI for communication profile. If using an ethernet tc,
 *           this argument is treated as PCP.
 * @tc: Traffic class for communication profile.
 * @tc_type: Traffic class type for communication profile.
 *
 * Reusing communication profiles is preferred over allocating a new one.
 *
 * @return: Valid pointer on success. Else, negative errno value.
 */
static struct cass_cp *cass_cp_get(struct cxi_tx_profile *tx_profile,
				   unsigned int vni_pcp,
				   unsigned int tc,
				   enum cxi_traffic_class_type tc_type)
	__must_hold(&hw->cp_lock)
{
	int id;
	struct cass_cp *cp;
	struct vni_tc_data data = {
		.vni_pcp = vni_pcp,
		.tc = tc,
		.tc_type = tc_type
	};

	if (tx_profile->config.exclusive_cp)
		return cass_cp_alloc(tx_profile, vni_pcp, tc, tc_type);

	id = idr_for_each(&tx_profile->config.cass_cp_table, &cp_match, &data);
	if (id) {
		cp = idr_find(&tx_profile->config.cass_cp_table, id);
		if (!cp) {
			pr_err("Could not find id %d\n", id);
			return ERR_PTR(-ENOENT);
		}

		refcount_inc(&cp->ref);

		return cp;
	}

	return cass_cp_alloc(tx_profile, vni_pcp, tc, tc_type);
}

/**
 * cxi_cp_alloc() - Allocate a communication profile
 *
 * @lni: LNI the communication profile should be mapped into
 * @vni_pcp: VNI of communication profile. If using an ethernet tc
 * this argument is treated as PCP.
 * @tc: Traffic class of the communication profile
 * @tc_type: Traffic class type of the communication profile
 *
 * Note: Allocating communication profiles with the same LNI, VNI, TC, and TC
 * type will result in a redundant communication profile being mapped into the
 * LNI.
 *
 * @return: Valid pointer on success. Else, negative errno value.
 */
struct cxi_cp *cxi_cp_alloc(struct cxi_lni *lni, unsigned int vni_pcp,
			    unsigned int tc,
			    enum cxi_traffic_class_type tc_type)
{
	struct cxi_lni_priv *lni_priv = container_of(lni, struct cxi_lni_priv,
						     lni);
	struct cxi_dev *dev = lni_priv->dev;
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cass_cp *cass_cp;
	struct cxi_cp_priv *cp_priv;
	struct cxi_tx_profile *tx_profile = NULL;
	int rc;
	int lcid;

	if (tc < 0 || tc >= CXI_ETH_TC_MAX || tc_type < 0 ||
	    tc_type >= CXI_TC_TYPE_MAX)
		return ERR_PTR(-EINVAL);

	/* Specific traffic class checks. */
	if (is_eth_tc(tc)) {
		/* Only 8 PCPs (0-7) are supported. */
		if (vni_pcp > 7)
			return ERR_PTR(-EINVAL);
		else if (!capable(CAP_NET_RAW))
			return ERR_PTR(-EPERM);

		tx_profile = cxi_dev_get_eth_tx_profile(&hw->cdev);
	} else {
		if (!is_vni_valid(vni_pcp)) {
			pr_debug("Invalid vni:%d\n", vni_pcp);
			return ERR_PTR(-EINVAL);
		}

		tx_profile = cxi_dev_get_tx_profile(&hw->cdev, vni_pcp);
		if (IS_ERR(tx_profile)) {
			rc = PTR_ERR(tx_profile);
			pr_debug("tx_profile not found for vni:%d rc:%d\n",
				 vni_pcp, rc);
			return ERR_PTR(rc);
		}

		if (!cxi_tx_profile_valid_tc(tx_profile, tc)) {
			pr_debug("Invalid tc:%d for tx_profile ID:%d\n", tc,
				 tx_profile->profile_common.id);
			rc = -EINVAL;
			goto free_tx_profile;
		}
	}

	mutex_lock(&hw->cp_lock);

	cp_priv = cass_cp_rgid_find(hw, lni_priv->lni.rgid, vni_pcp, tc,
				    tc_type);
	if (cp_priv) {
		mutex_unlock(&hw->cp_lock);
		pr_debug("Reuse cp:%u rgid:%u lcid:%u refcount:%d\n",
			 cp_priv->cass_cp->id, cp_priv->rgid,
			 cp_priv->cp.lcid, refcount_read(&cp_priv->refcount));
		return &cp_priv->cp;
	}

	cp_priv = kzalloc(sizeof(*cp_priv), GFP_KERNEL);
	if (!cp_priv) {
		mutex_unlock(&hw->cp_lock);
		return ERR_PTR(-ENOMEM);
	}

	cp_priv->rgid = lni_priv->lni.rgid;
	cp_priv->cp.vni_pcp = vni_pcp;
	cp_priv->cp.tc = tc;
	cp_priv->cp.tc_type = tc_type;
	refcount_set(&cp_priv->refcount, 1);

	/* Allocate the communication profile. */
	cass_cp = cass_cp_get(tx_profile, vni_pcp, tc, tc_type);
	if (IS_ERR(cass_cp)) {
		rc = PTR_ERR(cass_cp);
		goto free_cp_priv;
	}

	cp_priv->cass_cp = cass_cp;

	/* Map the communication profile to LCID. */
	lcid = cass_lcid_get(hw, cp_priv, lni_priv->lni.rgid);
	if (lcid < 0) {
		cxidev_dbg(dev, "%s rgid=%u lcids exhausted\n", dev->name,
			   lni_priv->lni.rgid);
		rc = lcid;
		goto free_cp;
	}
	cp_priv->cp.lcid = lcid;

	mutex_unlock(&hw->cp_lock);

	cass_set_cid(hw, lni_priv->lni.rgid, lcid, cass_cp->id);

	cxidev_dbg(dev, "%s lcid allocated: rgid=%u lcid=%u cp=%u\n", dev->name,
		   lni_priv->lni.rgid, cp_priv->cp.lcid, cass_cp->id);

	cass_flush_pci(hw);

	return &cp_priv->cp;

free_cp:
	cass_cp_put(cass_cp);
free_cp_priv:
	mutex_unlock(&hw->cp_lock);
	kfree(cp_priv);
free_tx_profile:
	cxi_tx_profile_dec_refcount(dev, tx_profile, true);

	return ERR_PTR(rc);
}
EXPORT_SYMBOL(cxi_cp_alloc);

/**
 * cxi_cp_free() - Free the communication profile
 *
 * @cp: Communication profile to be freed
 */
void cxi_cp_free(struct cxi_cp *cp)
{
	struct cxi_cp_priv *cp_priv = container_of(cp, struct cxi_cp_priv, cp);
	struct cass_dev *hw = cp_priv->cass_cp->hw;
	struct cxi_tx_profile *tx_profile = cp_priv->cass_cp->tx_profile;

	cxi_tx_profile_dec_refcount(&hw->cdev, tx_profile, true);

	mutex_lock(&hw->cp_lock);

	if (!refcount_dec_and_test(&cp_priv->refcount)) {
		mutex_unlock(&hw->cp_lock);
		pr_debug("Return cp:%u rgid:%u lcid:%u refcount:%d\n",
			 cp_priv->cass_cp->id, cp_priv->rgid,
			 cp_priv->cp.lcid, refcount_read(&cp_priv->refcount));
		return;
	}

	cxidev_dbg(&hw->cdev, "%s lcid freed: rgid=%u lcid=%u cp=%u\n",
		   hw->cdev.name, cp_priv->rgid, cp_priv->cp.lcid,
		   cp_priv->cass_cp->id);

	cass_set_cid(hw, cp_priv->rgid, cp_priv->cp.lcid, 0);

	cass_lcid_put(hw, cp_priv->rgid, cp_priv->cp.lcid);
	cass_cp_put(cp_priv->cass_cp);

	mutex_unlock(&hw->cp_lock);

	kfree(cp_priv);

	cass_flush_pci(hw);
}
EXPORT_SYMBOL(cxi_cp_free);

int cxi_cp_modify(struct cxi_cp *cp, unsigned int vni_pcp)
{
	struct cxi_cp_priv *cp_priv = container_of(cp, struct cxi_cp_priv, cp);
	struct cass_cp *cass_cp = cp_priv->cass_cp;
	struct cass_dev *hw = cp_priv->cass_cp->hw;
	int rc = 0;
	struct cass_tc_cfg tc_cfg;
	union c_cq_cfg_cp_table cp_cfg = {
	      .valid = 1,
	      .hrp_vld = cp->tc_type == CXI_TC_TYPE_HRP,
	};

	if (!cass_cp->tx_profile->config.exclusive_cp)
		return -EINVAL;

	if (is_eth_tc(cp->tc)) {
		pr_debug("eth tc:%s for cp:%u is invalid for cp modify\n",
			 cxi_tc_to_str(cp->tc), cass_cp->id);
		return -EINVAL;
	}

	if (!cxi_valid_vni(&hw->cdev, CXI_PROF_TX, vni_pcp)) {
		pr_debug("cp modify received invalid vni: %d for cp:%u\n",
			 vni_pcp, cass_cp->id);
		return -EINVAL;
	}

	rc = cass_tc_find(hw, cp->tc, cp->tc_type, vni_pcp, &tc_cfg);
	if (rc) {
		cxidev_dbg(&hw->cdev,
			   "%s tc mapping does not exist for cp:%u vni_pcp:%u tc:%s tc_type:%s",
			   hw->cdev.name, cass_cp->id, vni_pcp,
			   cxi_tc_to_str(cp->tc),
			   cxi_tc_type_to_str(cp->tc_type));
		return -EINVAL;
	}

	cp->vni_pcp = vni_pcp;
	cp_cfg.vni = vni_pcp;
	cp_cfg.dscp_unrsto = tc_cfg.unres_req_dscp;
	cp_cfg.dscp_rstuno = tc_cfg.res_req_dscp;

	cass_set_cp_table(hw, cass_cp->id, &cp_cfg);

	cxidev_dbg(&hw->cdev,
		   "%s cp modified: tc=%s tc_type=%s cp=%u vni_pcp=%u\n",
		   hw->cdev.name, cxi_tc_to_str(cp->tc),
		   cxi_tc_type_to_str(cp->tc_type), cass_cp->id, cp->vni_pcp);

	cass_flush_pci(hw);

	return rc;
}
EXPORT_SYMBOL(cxi_cp_modify);
