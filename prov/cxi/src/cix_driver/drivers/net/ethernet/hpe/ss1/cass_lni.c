// SPDX-License-Identifier: GPL-2.0
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

/* Cassini NI management */

#include <linux/types.h>

#include "cass_core.h"
#include "cass_ss1_debugfs.h"

unsigned int cxi_lni_get_pe_num(struct cxi_lni_priv *lni)
{
	int pe;
	int xchg;

	/* Round robin across all PEs. */
	do {
		pe = atomic_read(&lni->lpe_pe_num);
		xchg = pe + 1;
		if (xchg == C_PE_COUNT)
			xchg = 0;
	} while (atomic_cmpxchg(&lni->lpe_pe_num, pe, xchg) != pe);

	return pe;
}

/**
 * cxi_lni_alloc() - Allocate a logical NI (LNI) on a device.
 *
 * @dev: CXI device
 * @svc_id: Service ID of service to be used. May be CXI_DEFAULT_SVC_ID.
 */
struct cxi_lni *cxi_lni_alloc(struct cxi_dev *dev, unsigned int svc_id)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_lni_priv *lni_priv;
	struct cxi_rgroup *rgroup;
	int rc;
	int id;
	int rgid;
	void *err;

	rc = cxi_dev_find_rgroup_inc_refcount(dev, svc_id, &rgroup);
	if (rc)
		return ERR_PTR(-EINVAL);

	/* Ensure service is enabled */
	if (!cxi_rgroup_is_enabled(rgroup)) {
		err = ERR_PTR(-EKEYREVOKED);
		goto dec_svc;
	}

	/* Verify calling user/group has permission to use this service */
	if (!cxi_rgroup_valid_user(rgroup)) {
		err = ERR_PTR(-EPERM);
		goto dec_svc;
	}

	rgid = cass_rgid_get(hw, rgroup);
	if (rgid < 0) {
		err = ERR_PTR(rgid);
		goto dec_svc;
	}

	id = ida_alloc_min(&hw->lni_table, 1, GFP_KERNEL);
	if (id < 0) {
		err = ERR_PTR(id);
		goto put_rgid;
	}

	lni_priv = kzalloc(sizeof(*lni_priv), GFP_KERNEL);
	if (!lni_priv) {
		err = ERR_PTR(-ENOMEM);
		goto free_lni_id;
	}

	lni_priv->rgroup = rgroup;
	lni_priv->dev = dev;
	lni_priv->lni.id = id;
	lni_priv->lni.rgid = rgid;
	lni_priv->pid = current->pid;
	atomic_set(&lni_priv->lpe_pe_num, rgid % C_PE_COUNT);

	spin_lock_init(&lni_priv->res_lock);
	refcount_set(&lni_priv->refcount, 1);
	mutex_init(&lni_priv->ac_list_mutex);

	INIT_LIST_HEAD(&lni_priv->domain_list);
	INIT_LIST_HEAD(&lni_priv->eq_list);
	INIT_LIST_HEAD(&lni_priv->cq_list);
	INIT_LIST_HEAD(&lni_priv->pt_list);
	INIT_LIST_HEAD(&lni_priv->ac_list);
	INIT_LIST_HEAD(&lni_priv->ct_list);
	INIT_LIST_HEAD(&lni_priv->reserved_pids);
	INIT_LIST_HEAD(&lni_priv->ct_cleanups_list);
	INIT_LIST_HEAD(&lni_priv->pt_cleanups_list);
	INIT_LIST_HEAD(&lni_priv->cq_cleanups_list);
	INIT_LIST_HEAD(&lni_priv->eq_cleanups_list);

	spin_lock(&hw->lni_lock);
	list_add_tail(&lni_priv->list, &hw->lni_list);
	atomic_inc(&hw->stats.lni);
	spin_unlock(&hw->lni_lock);

	refcount_inc(&hw->refcount);

	lni_debugfs_create(lni_priv->lni.id, hw, lni_priv);

	return &lni_priv->lni;

free_lni_id:
	ida_free(&hw->lni_table, id);
put_rgid:
	cass_rgid_put(hw, rgid);
dec_svc:
	cxi_rgroup_dec_refcount(rgroup);
	return err;
}
EXPORT_SYMBOL(cxi_lni_alloc);

/* Cleanup of an LNI. Return true if cleanup succeeded.
 *
 * If force is true, the LNI will be freed even if the CQs are still
 * busy. This is only intended for driver unloading.
 */
static bool try_cleanup_lni(struct cxi_lni_priv *lni, bool force)
{
	struct cxi_dev *dev = lni->dev;
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);

	/* Cleanup cannot progress until all the CQs have been freed */
	finalize_cq_cleanups(lni, force);
	if (!force && !list_empty(&lni->cq_cleanups_list))
		return false;

	/* Same with the portals */
	finalize_pt_cleanups(lni, force);
	if (!force && !list_empty(&lni->pt_cleanups_list))
		return false;

	finalize_eq_cleanups(lni);
	finalize_ct_cleanups(lni);

	cass_acs_free(lni);

	cxidev_WARN_ONCE(dev,
			 !refcount_dec_and_test(&lni->refcount),
			 "Resource leaks - LNI refcount not zero: %d\n",
			 refcount_read(&lni->refcount));

	debugfs_remove_recursive(lni->debug_dir);

	cxi_rgroup_dec_refcount(lni->rgroup);
	refcount_dec(&hw->refcount);
	atomic_dec(&hw->stats.lni);
	ida_free(&hw->lni_table, lni->lni.id);
	cass_rgid_put(hw, lni->lni.rgid);

	return true;
}

/* Try to cleanup all the pending LNIs.
 * Returns false if at least one LNI is still pending.
 *
 * If force is true, the LNIs will be cleaned up.
 */
bool lni_cleanups(struct cass_dev *hw, bool force)
{
	struct cxi_lni_priv *lni;
	struct cxi_lni_priv *tmp;
	bool done = true;

	mutex_lock(&hw->lni_cleanups_lock);

	list_for_each_entry_safe(lni, tmp, &hw->lni_cleanups_list, list) {
		if (try_cleanup_lni(lni, force)) {
			list_del(&lni->list);
			kfree(lni);
		} else {
			done = false;
		}
	}

	mutex_unlock(&hw->lni_cleanups_lock);

	return done;
}

void lni_cleanups_work(struct work_struct *work)
{
	struct cass_dev *hw =
		container_of(work, struct cass_dev, lni_cleanups_work.work);

	/* If cleanup is not complete, reschedule the work. */
	if (!lni_cleanups(hw, false))
		mod_delayed_work(system_wq, &hw->lni_cleanups_work, HZ);
}

/**
 * cxi_lni_free() - Shutdown and release an LNI.
 *
 * @lni: LNI to release
 */
int cxi_lni_free(struct cxi_lni *lni)
{
	struct cxi_lni_priv *lni_priv =
		container_of(lni, struct cxi_lni_priv, lni);
	struct cxi_dev *dev = lni_priv->dev;
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);

	spin_lock(&hw->lni_lock);
	list_del(&lni_priv->list);
	spin_unlock(&hw->lni_lock);

	cxi_domain_lni_cleanup(lni_priv);

	if (hw->cdev.is_physfn) {
		cass_acs_disable(lni_priv);
		cxi_inbound_wait(dev);
	}

	if (try_cleanup_lni(lni_priv, false)) {
		kfree(lni_priv);
	} else {
		mutex_lock(&hw->lni_cleanups_lock);
		list_add_tail(&lni_priv->list, &hw->lni_cleanups_list);
		mutex_unlock(&hw->lni_cleanups_lock);

		mod_delayed_work(system_wq, &hw->lni_cleanups_work, HZ);
	}

	return 0;
}
EXPORT_SYMBOL(cxi_lni_free);
