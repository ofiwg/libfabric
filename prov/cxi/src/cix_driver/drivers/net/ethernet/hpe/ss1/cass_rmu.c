// SPDX-License-Identifier: GPL-2.0
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

/* Create and destroy Cassini command queues */

#include <linux/hpe/cxi/cxi.h>
#include <linux/idr.h>
#include <linux/kernel.h>
#include <linux/rbtree.h>
#include <linux/types.h>

#include "cass_core.h"
#include "cass_ss1_debugfs.h"

/* Add a new domain into a device domain tree. */
static int insert_domain(struct rb_root *root,
			 struct cxi_domain_priv *domain_priv)
{
	struct rb_node **new = &(root->rb_node), *parent = NULL;

	/* Figure out where to put new node */
	while (*new) {
		struct cxi_domain_priv *this =
			container_of(*new, struct cxi_domain_priv, node);

		parent = *new;
		if (domain_priv->domain.vni < this->domain.vni)
			new = &((*new)->rb_left);
		else if (domain_priv->domain.vni > this->domain.vni)
			new = &((*new)->rb_right);
		else if (domain_priv->domain.pid < this->domain.pid)
			new = &((*new)->rb_left);
		else if (domain_priv->domain.pid > this->domain.pid)
			new = &((*new)->rb_right);
		else
			return -EEXIST;
	}

	/* Add new node and rebalance tree. */
	rb_link_node(&domain_priv->node, parent, new);
	rb_insert_color(&domain_priv->node, root);

	return 0;
}

/* Atomically reserve a contiguous range of VNI PIDs. On success, PIDs are
 * reserved to the LNI. Reserved PIDs are released when the LNI is destroyed.
 * cxi_domain_alloc() must be used to create a Domain using a reserved PID.
 */
int cxi_domain_reserve(struct cxi_lni *lni, unsigned int vni, unsigned int pid,
		       unsigned int count)
{
	struct cxi_lni_priv *lni_priv =
		container_of(lni, struct cxi_lni_priv, lni);
	struct cxi_dev *cdev = lni_priv->dev;
	struct cxi_rx_profile *rx_profile;
	struct cxi_reserved_pids *pids;
	int rc;
	int i;

	/* Sanity checks. */
	if (!is_vni_valid(vni))
		return -EINVAL;

	if (!cxi_valid_vni(cdev, CXI_PROF_RX, vni))
		return -EINVAL;

	if (pid >= cdev->prop.pid_count && pid != C_PID_ANY)
		return -EINVAL;

	if (!count || count > cdev->prop.pid_count)
		return -EINVAL;

	rx_profile = cxi_dev_find_rx_profile(cdev, vni);
	if (!rx_profile)
		return -ENOENT;

	rc = cxi_rx_profile_alloc_pid(lni_priv, rx_profile, pid, vni, count, true);
	if (rc < 0)
		goto put_rx_profile;
	pid = rc;

	pids = kzalloc(sizeof(*pids), GFP_KERNEL);
	if (!pids) {
		rc = -ENOMEM;
		goto clear_pids;
	}

	pids->rx_profile = rx_profile;

	for (i = 0; i < count; i++)
		set_bit(pid + i, pids->table);

	spin_lock(&lni_priv->res_lock);
	list_add_tail(&pids->entry, &lni_priv->reserved_pids);
	spin_unlock(&lni_priv->res_lock);

	return pid;

clear_pids:
	cxi_rx_profile_update_pid_table(rx_profile, pid, count, false);
put_rx_profile:
	cxi_rx_profile_dec_refcount(cdev, rx_profile);

	return rc;
}
EXPORT_SYMBOL(cxi_domain_reserve);

/* Clean up PIDs reserved to the LNI. */
void cxi_domain_lni_cleanup(struct cxi_lni_priv *lni_priv)
{
	struct cxi_reserved_pids *pids;

	while ((pids = list_first_entry_or_null(&lni_priv->reserved_pids,
						struct cxi_reserved_pids,
						entry))) {
		list_del(&pids->entry);
		cxi_rx_profile_andnot_pid_table(pids,
						lni_priv->dev->prop.pid_count);
		cxi_rx_profile_dec_refcount(lni_priv->dev, pids->rx_profile);
		kfree(pids);
	}
}

/* Allocate a new domain, with a unique per-device VNI+PID. The VNI is
 * reserved in the RMU table if it doesn't already exist.
 */
struct cxi_domain *cxi_domain_alloc(struct cxi_lni *lni, unsigned int vni,
				    unsigned int pid)
{
	struct cxi_lni_priv *lni_priv =
		container_of(lni, struct cxi_lni_priv, lni);
	struct cxi_dev *cdev = lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	struct cxi_domain_priv *domain_priv;
	int rc;
	int domain_pid = pid;
	struct cxi_rx_profile *rx_profile;

	/* Sanity checks. */
	if (!is_vni_valid(vni))
		return ERR_PTR(-EINVAL);

	if (domain_pid >= cdev->prop.pid_count && domain_pid != C_PID_ANY)
		return ERR_PTR(-EINVAL);

	rx_profile = cxi_dev_get_rx_profile(cdev, vni);
	if (IS_ERR(rx_profile)) {
		rc = PTR_ERR(rx_profile);
		pr_debug("rx_profile not found for vni:%d rc:%d\n",
			 vni, rc);
		return ERR_PTR(rc);
	}

	domain_pid = cxi_rx_profile_alloc_pid(lni_priv, rx_profile, pid, vni,
					      1, false);
	if (domain_pid < 0) {
		rc = domain_pid;
		goto put_rx_profile;
	}

	domain_priv = kzalloc(sizeof(*domain_priv), GFP_KERNEL);
	if (domain_priv == NULL) {
		rc = -ENOMEM;
		goto free_pid;
	}

	/* Get a domain ID */
	rc = ida_simple_get(&hw->domain_table, 1, 0, GFP_KERNEL);
	if (rc < 0) {
		cxidev_err(cdev, "ida_simple_get failed %d\n", rc);
		goto free_domain;
	}
	domain_priv->domain.id = rc;

	refcount_set(&domain_priv->refcount, 1);

	domain_priv->lni_priv = lni_priv;
	domain_priv->rx_profile = rx_profile;
	domain_priv->domain.vni = vni;
	domain_priv->domain.pid = domain_pid;

	spin_lock(&hw->domain_lock);
	rc = insert_domain(&hw->domain_tree, domain_priv);
	spin_unlock(&hw->domain_lock);
	if (rc)
		goto free_dom_id;

	domain_debugfs_create(vni, domain_pid, hw, domain_priv);

	spin_lock(&lni_priv->res_lock);
	atomic_inc(&hw->stats.domain);
	list_add_tail(&domain_priv->list, &lni_priv->domain_list);
	spin_unlock(&lni_priv->res_lock);

	refcount_inc(&lni_priv->refcount);
	return &domain_priv->domain;

free_dom_id:
	ida_simple_remove(&hw->domain_table, domain_priv->domain.id);
free_domain:
	kfree(domain_priv);
free_pid:
	cxi_rx_profile_update_pid_table(rx_profile, pid, 1, false);
put_rx_profile:
	cxi_rx_profile_dec_refcount(cdev, rx_profile);

	return ERR_PTR(rc);
}
EXPORT_SYMBOL(cxi_domain_alloc);

/* Free an allocated domain. */
void cxi_domain_free(struct cxi_domain *domain)
{
	struct cxi_domain_priv *domain_priv =
			container_of(domain, struct cxi_domain_priv, domain);
	struct cxi_lni_priv *lni_priv = domain_priv->lni_priv;
	struct cxi_dev *cdev = lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);

	cxidev_WARN_ONCE(cdev, !refcount_dec_and_test(&domain_priv->refcount),
			 "Resource leaks - Domain refcount not zero: %d\n",
			 refcount_read(&domain_priv->refcount));

	spin_lock(&lni_priv->res_lock);
	list_del(&domain_priv->list);
	atomic_dec(&hw->stats.domain);
	spin_unlock(&lni_priv->res_lock);

	debugfs_remove_recursive(domain_priv->debug_dir);

	spin_lock(&hw->domain_lock);
	rb_erase(&domain_priv->node, &hw->domain_tree);
	spin_unlock(&hw->domain_lock);

	cxi_rx_profile_update_pid_table(domain_priv->rx_profile,
					domain->pid, 1, false);
	cxi_rx_profile_dec_refcount(cdev, domain_priv->rx_profile);

	ida_simple_remove(&hw->domain_table, domain_priv->domain.id);
	kfree(domain_priv);

	refcount_dec(&lni_priv->refcount);
}
EXPORT_SYMBOL(cxi_domain_free);
