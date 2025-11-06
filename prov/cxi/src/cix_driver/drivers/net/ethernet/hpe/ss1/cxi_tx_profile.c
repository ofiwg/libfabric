// SPDX-License-Identifier: GPL-2.0
/* Copyright (C) 2024 Hewlett Packard Enterprise Development LP */

/* TX Profile Implementation */

#include "cass_core.h"
#include "cxi_rxtx_profile.h"
#include "cxi_rxtx_profile_list.h"

#define TX_PROFILE_GFP_OPTS  (GFP_KERNEL)

static struct cass_dev *get_cass_dev(struct cxi_dev *dev)
{
	return container_of(dev, struct cass_dev, cdev);
}

/* initialize common profile and cassini config members */
static void cass_dev_init_tx_profile(struct cass_dev *hw,
				     struct cxi_tx_profile *tx_profile,
				     const struct cxi_tx_attr *tx_attr)
{
	cxi_rxtx_profile_init(&tx_profile->profile_common,
			      hw, &tx_attr->vni_attr);
	cass_tx_profile_init(hw, tx_profile);
}

void cxi_dev_init_eth_tx_profile(struct cass_dev *hw)
{
	struct cxi_tx_attr tx_attr = {};

	cass_dev_init_tx_profile(hw, &hw->eth_tx_profile, &tx_attr);
	refcount_inc(&hw->refcount);
}
EXPORT_SYMBOL(cxi_dev_init_eth_tx_profile);

/**
 * cxi_tx_profile_list_destroy() - remove all Profile references from
 *                                 the TX profile list and free the
 *                                 underlying resources.
 *
 * Take the cp_lock so that the cass_cp_table can be iterated
 * over to disable the CPs associated with each TX profile.
 *
 * @hw: Cassini Device ptr
 */
void cxi_tx_profile_list_destroy(struct cass_dev *hw)
{
	struct cxi_rxtx_profile *profile;
	struct cxi_rxtx_profile_list *list = &hw->tx_profile_list;
	unsigned long index = list->limits->min;

	cass_cp_lock(hw);
	xa_lock(&list->xarray);

	xa_for_each(&list->xarray, index, profile) {
		__xa_erase(&list->xarray, index);

		refcount_dec(&profile->state.refcount);
		cxi_tx_profile_dec_refcount(&hw->cdev,
					    co_tx_profile(profile),
					    false);
	}

	xa_unlock(&list->xarray);
	cass_cp_unlock(hw);

	xa_destroy(&list->xarray);
}

/**
 * cxi_dev_alloc_tx_profile() - Allocate a TX Profile
 *
 * @dev: Cassini Device
 * @tx_attr: TX attributes for the Profile
 *
 * Return: tx_profile ptr on success, or a negative errno value.
 */
struct cxi_tx_profile *cxi_dev_alloc_tx_profile(struct cxi_dev *dev,
					const struct cxi_tx_attr *tx_attr)
{
	int                    ret = 0;
	struct cass_dev        *hw = get_cass_dev(dev);
	struct cxi_tx_profile  *tx_profile;

	if (!zero_vni(&tx_attr->vni_attr) &&
	    !vni_well_formed(&tx_attr->vni_attr))
		return ERR_PTR(-EDOM);

	/* Allocate memory */
	tx_profile = kzalloc(sizeof(*tx_profile), TX_PROFILE_GFP_OPTS);
	if (!tx_profile)
		return ERR_PTR(-ENOMEM);

	cass_dev_init_tx_profile(hw, tx_profile, tx_attr);

	cxi_rxtx_profile_list_lock(&hw->tx_profile_list);

	/* make sure the VNI space is unique */
	ret = cxi_rxtx_profile_list_iterate(&hw->tx_profile_list,
					    vni_overlap_test,
					    &tx_profile->profile_common);
	if (ret)
		goto unlock_free_return;

	/* Insert into device list if unique */
	ret = cxi_rxtx_profile_list_insert(&hw->tx_profile_list,
					   &tx_profile->profile_common,
					   &tx_profile->profile_common.id);

	cxi_rxtx_profile_list_unlock(&hw->tx_profile_list);

	if (ret)
		goto free_return;

	refcount_inc(&hw->refcount);

	return tx_profile;

unlock_free_return:
	cxi_rxtx_profile_list_unlock(&hw->tx_profile_list);

free_return:
	kfree(tx_profile);
	return ERR_PTR(ret);
}
EXPORT_SYMBOL(cxi_dev_alloc_tx_profile);

/**
 * cxi_dev_find_tx_profile() - Find the TX profile containing a vni
 *
 * @dev: Cassini Device
 * @vni: Match criteria to find TX profile
 *
 * Return: tx_profile ptr on success or NULL
 */
struct cxi_tx_profile *cxi_dev_find_tx_profile(struct cxi_dev *dev,
					       uint16_t vni)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_rxtx_profile_list *list = &hw->tx_profile_list;
	struct cxi_tx_profile *tx_profile = NULL;
	struct cxi_rxtx_profile *profile;
	unsigned long index = list->limits->min;
	uid_t uid = __kuid_val(current_euid());
	gid_t gid = __kgid_val(current_egid());
	unsigned int ac_entry_id;
	int rc;

	cxi_rxtx_profile_list_lock(list);

	xa_for_each(&list->xarray, index, profile) {
		if (!profile->state.enable)
			continue;

		if ((vni & ~profile->vni_attr.ignore) == profile->vni_attr.match) {
			tx_profile = container_of(profile,
						  struct cxi_tx_profile,
						  profile_common);

			rc = cxi_rxtx_profile_get_ac_entry_id_by_user(
						&tx_profile->profile_common,
						uid, gid, CXI_AC_ANY,
						&ac_entry_id);
			if (!rc) {
				refcount_inc(&profile->state.refcount);
				break;
			}

			tx_profile = ERR_PTR(rc);
		}
	}

	cxi_rxtx_profile_list_unlock(list);

	return tx_profile;
}

/**
 * cxi_dev_get_tx_profile - Get a TX profile
 *
 * Check if one is already allocated and return it.
 * For kernel users (i.e. kcxi), allocate one if not found, add
 * an AC entry, set the TCs and enable the profile.
 *
 * @dev: Cassini Device
 * @vni: VNI to find
 *
 * Return: tx_profile ptr on success, or a negative errno value.
 */
struct cxi_tx_profile *cxi_dev_get_tx_profile(struct cxi_dev *dev,
					      unsigned int vni)
{
	int i;
	int rc;
	unsigned int ac_entry_id;
	struct cass_dev *hw = get_cass_dev(dev);
	struct cxi_tx_profile *tx_profile;
	struct cxi_tx_attr tx_attr = {
		.vni_attr = {
			.match = vni,
			.ignore = 0
		}
	};

	mutex_lock(&hw->tx_profile_get_lock);

	tx_profile = cxi_dev_find_tx_profile(dev, vni);
	if (tx_profile)
		goto done;

	/* For a non-root user, just report no entry found.*/
	if (__kuid_val(current_euid())) {
		tx_profile = ERR_PTR(-ENOENT);
		goto done;
	}

	tx_profile = cxi_dev_alloc_tx_profile(dev, &tx_attr);
	if (IS_ERR(tx_profile))
		goto done;

	rc = cxi_tx_profile_add_ac_entry(tx_profile, CXI_AC_UID,
					 __kuid_val(current_euid()), 0,
					 &ac_entry_id);
	if (rc) {
		tx_profile = ERR_PTR(rc);
		goto done;
	}

	/* For backward compatibility to the old service api,
	 * enable all the TCs.
	 */
	for (i = CXI_TC_DEDICATED_ACCESS; i <= CXI_TC_BEST_EFFORT; i++)
		cxi_tx_profile_set_tc(tx_profile, i, true);

	rc = cxi_tx_profile_enable(dev, tx_profile);
	if (rc)
		tx_profile = ERR_PTR(rc);

done:
	mutex_unlock(&hw->tx_profile_get_lock);

	return tx_profile;
}
EXPORT_SYMBOL(cxi_dev_get_tx_profile);

/**
 * cxi_tx_profile_valid_tc() - Check if a TC is valid for this TX profile
 *
 * @tx_profile: Profile to check against
 * @tc: Traffic Class
 *
 * Return:
 * * true if valid
 */
bool cxi_tx_profile_valid_tc(struct cxi_tx_profile *tx_profile, unsigned int tc)
{
	if (tc >= CXI_TC_MAX)
		return false;

	return test_bit(tc, tx_profile->config.tc_table);
}

/**
 * cxi_dev_set_tx_profile_attr() - Set the TX profile attributes
 *
 * @dev: Cassini Device
 * @tx_profile: TX profile to update
 * @tx_attr: Attributes of the TX Profile
 *
 * Return: 0 on success, or a negative errno value.
 */
int cxi_dev_set_tx_profile_attr(struct cxi_dev *dev,
				struct cxi_tx_profile *tx_profile,
				const struct cxi_tx_attr *tx_attr)
{
	struct cass_dev *hw = get_cass_dev(dev);

	if (tx_profile->profile_common.state.enable)
		return -EBUSY;

	if (!vni_well_formed(&tx_attr->vni_attr)) {
		pr_debug("VNI not well formed match:%d ignore:%d\n",
			 tx_attr->vni_attr.match, tx_attr->vni_attr.ignore);
		return -EINVAL;
	}

	if (!unique_vni_space(hw, &hw->tx_profile_list, &tx_attr->vni_attr)) {
		pr_debug("VNI not unique match:%d ignore:%d\n",
			 tx_attr->vni_attr.match, tx_attr->vni_attr.ignore);
		return -EINVAL;
	}

	tx_profile->profile_common.vni_attr.match = tx_attr->vni_attr.match;
	tx_profile->profile_common.vni_attr.ignore = tx_attr->vni_attr.ignore;

	if (!tx_attr->vni_attr.name[0])
		strscpy(tx_profile->profile_common.vni_attr.name,
			tx_attr->vni_attr.name,
			ARRAY_SIZE(tx_attr->vni_attr.name));

	return 0;
}
EXPORT_SYMBOL(cxi_dev_set_tx_profile_attr);

/**
 * cxi_tx_profile_enable() - Enable a Profile
 *
 * @dev: Cassini Device
 * @tx_profile: Profile to be enabled.
 *
 * Return:
 * * 0       - success
 */
int cxi_tx_profile_enable(struct cxi_dev *dev,
			   struct cxi_tx_profile *tx_profile)
{
	if (!tx_profile)
		return -EINVAL;

	if (cxi_tx_profile_is_enabled(tx_profile))
		return 0;

	if (zero_vni(&tx_profile->profile_common.vni_attr)) {
		pr_debug("Cannot enable profile with invalid VNI\n");
		return -EINVAL;
	}

	// TODO: more hw setup here?
	tx_profile->profile_common.state.enable = true;

	return 0;
}
EXPORT_SYMBOL(cxi_tx_profile_enable);

/**
 * cxi_tx_profile_disable() - Disable a Profile
 *
 * @dev: Cassini Device
 * @tx_profile: Profile to be disabled.
 */
void cxi_tx_profile_disable(struct cxi_dev *dev,
			    struct cxi_tx_profile *tx_profile)
{
	if (!tx_profile)
		return;

	if (!cxi_tx_profile_is_enabled(tx_profile))
		return;

	tx_profile->profile_common.state.enable = false;
	cass_clear_cps(tx_profile);
}
EXPORT_SYMBOL(cxi_tx_profile_disable);

/**
 * cxi_tx_profile_is_enabled() - Report TX profile is enabled
 *
 * @tx_profile: Profile object
 */
bool cxi_tx_profile_is_enabled(const struct cxi_tx_profile *tx_profile)
{
	if (!tx_profile)
		return false;

	return tx_profile->profile_common.state.enable;
}
EXPORT_SYMBOL(cxi_tx_profile_is_enabled);

/**
 * cxi_tx_profile_dec_refcount() - Decrement refcount and cleanup
 *                                 if last reference
 *
 * @dev: Cassini device pointer
 * @tx_profile: pointer to Profile
 * @list_remove: don't remove from tx_profile_list
 *
 */
int cxi_tx_profile_dec_refcount(struct cxi_dev *dev,
				struct cxi_tx_profile *tx_profile,
				bool list_remove)
{
	struct cass_dev *hw = get_cass_dev(dev);
	int    ret;

	if (!tx_profile)
		return 0;

	ret = refcount_dec_and_test(&tx_profile->profile_common.state.refcount);
	if (!ret)
		return -EBUSY;

	cxi_tx_profile_disable(dev, tx_profile);
	cxi_tx_profile_remove_ac_entries(tx_profile);

	refcount_dec(&hw->refcount);

	if (list_remove)
		cxi_rxtx_profile_list_remove(&hw->tx_profile_list,
					     tx_profile->profile_common.id);

	kfree(tx_profile);
	return 0;
}
EXPORT_SYMBOL(cxi_tx_profile_dec_refcount);

/**
 * cxi_dev_get_eth_tx_profile() - Get ethernet TX profile
 *
 * @dev: Cassini Device
 *
 * Return: tx_profile ethernet ptr
 */
struct cxi_tx_profile *cxi_dev_get_eth_tx_profile(struct cxi_dev *dev)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);

	refcount_inc(&hw->eth_tx_profile.profile_common.state.refcount);
	return &hw->eth_tx_profile;
}

/**
 * cxi_eth_tx_profile_cleanup() - Clean up the TX profile for ethernet
 *
 * @hw: the device
 */
void cxi_eth_tx_profile_cleanup(struct cass_dev *hw)
{
	cxi_tx_profile_disable(&hw->cdev, &hw->eth_tx_profile);
	refcount_dec(&hw->refcount);
}
EXPORT_SYMBOL(cxi_eth_tx_profile_cleanup);

/**
 * cxi_tx_profile_get_info() - Retrieve the attributes and state associated
 *                             with this Profile
 *
 * @dev: Cassini Device
 * @tx_profile: the Profile
 * @tx_attr: location to place attributes
 * @state: location to put state
 *
 * Note: vni_attr and/or state may be NULL.  If both are NULL,
 * this return value indicates whether the Profile exists
 * with the given Id value.
 *
 * Return:
 * * 0      - success
 * * -ENOENT - No tx_profile
 * * -EBADR - tx_profile_id unknown
 */
int cxi_tx_profile_get_info(struct cxi_dev *dev,
			    struct cxi_tx_profile *tx_profile,
			    struct cxi_tx_attr *tx_attr,
			    struct cxi_rxtx_profile_state *state)
{
	if (!tx_profile)
		return -ENOENT;

	cxi_rxtx_profile_get_info(&tx_profile->profile_common,
				  &tx_attr->vni_attr, state);

	/* TODO: gather other TX attributes */

	return 0;
}
EXPORT_SYMBOL(cxi_tx_profile_get_info);

/**
 * cxi_tx_profile_set_tc() - Set/clear a traffic class in the TX profile
 *
 * @tx_profile: pointer to Profile
 * @tc: traffic class to add/clear
 * @set: operation - set true / clear false
 *
 * Return:
 * * 0       - success
 * * -EINVAL - tc outside of allowed range
 */
int cxi_tx_profile_set_tc(struct cxi_tx_profile *tx_profile, int tc, bool set)
{
	if (tc < CXI_TC_DEDICATED_ACCESS || tc > CXI_TC_MAX)
		return -EINVAL;

	spin_lock(&tx_profile->config.lock);

	if (set)
		set_bit(tc, tx_profile->config.tc_table);
	else
		clear_bit(tc, tx_profile->config.tc_table);

	spin_unlock(&tx_profile->config.lock);

	return 0;
}
EXPORT_SYMBOL(cxi_tx_profile_set_tc);

static void print_tx_profile_ac_entry_info(struct cxi_tx_profile *tx_profile,
					   struct seq_file *s)
{
	int i;
	int rc;
	size_t num_ids;
	size_t max_ids;
	unsigned int *ac_entry_ids = NULL;
	enum cxi_ac_type ac_type;
	union cxi_ac_data ac_data;

	rc = cxi_tx_profile_get_ac_entry_ids(tx_profile, 0, ac_entry_ids,
					     &num_ids);
	if (rc && rc != -ENOSPC)
		goto done;

	ac_entry_ids = kmalloc_array(num_ids, sizeof(*ac_entry_ids),
				     GFP_ATOMIC);
	if (!ac_entry_ids) {
		rc = -ENOMEM;
		goto done;
	}

	rc = cxi_tx_profile_get_ac_entry_ids(tx_profile, num_ids, ac_entry_ids,
					     &max_ids);
	if (rc)
		goto freemem;

	seq_puts(s, "        AC-entries: ");
	for (i = 0; i < num_ids; i++) {
		rc = cxi_tx_profile_get_ac_entry_data(tx_profile,
						      ac_entry_ids[i],
						      &ac_type, &ac_data);
		if (rc)
			break;

		seq_printf(s, "ID:%d type:%s uid/gid:%d%s",
			   ac_entry_ids[i], AC_TYPE(ac_type),
			   ac_type == CXI_AC_OPEN ? 0 : ac_data.uid,
			   i < (num_ids - 1) ? ", " : "");
	}

freemem:
	kfree(ac_entry_ids);
done:
	if (rc)
		seq_puts(s, "\n");
}

static void print_profile(struct cxi_tx_profile *tx_profile, struct seq_file *s)
{
	struct cxi_rxtx_vni_attr vni_attr;
	struct cxi_rxtx_profile_state state;
	struct cxi_rxtx_profile *profile = &tx_profile->profile_common;

	cxi_rxtx_profile_get_info(profile, &vni_attr, &state);
	seq_printf(s, "  ID:%-2d Name:%s VNI:match:%u ignore:%u State:%s refcount:%d Exclusive-CP:%d\n",
		   profile->id,
		   vni_attr.name[0] ? vni_attr.name : "none",
		   vni_attr.match, vni_attr.ignore,
		   state.enable ? "enabled" : "disabled",
		   refcount_read(&state.refcount),
		   tx_profile->config.exclusive_cp);

	print_tx_profile_ac_entry_info(tx_profile, s);
}

/**
 * cxi_tx_profile_print() - Print the TX profile info
 *
 * @s: file ptr
 *
 */
void cxi_tx_profile_print(struct seq_file *s)
{
	struct cass_dev *hw = s->private;
	struct cxi_rxtx_profile_list *list = &hw->tx_profile_list;
	struct cxi_tx_profile *tx_profile;
	struct cxi_rxtx_profile *profile;
	unsigned long index;
	int tcs;
	int i;

	seq_puts(s, "\nTX profiles:\n");

	cxi_rxtx_profile_list_lock(list);

	xa_for_each(&list->xarray, index, profile) {
		tx_profile = container_of(profile, struct cxi_tx_profile,
					  profile_common);
		print_profile(tx_profile, s);

		seq_puts(s, "\n        TCs:");
		for (i = 0, tcs = 0; i < CXI_TC_MAX; i++) {
			if (test_bit(i, tx_profile->config.tc_table)) {
				seq_printf(s, "%s%s", tcs ? ", " : "",
					   cxi_tc_strs[i]);
				tcs++;
			}
		}

		seq_puts(s, "\n");
	}

	cxi_rxtx_profile_list_unlock(list);
}

/**
 * cxi_tx_profile_remove_ac_entry() - disable access control to a Profile
 *                                    by access control id.
 *
 * @tx_profile: pointer to Profile
 * @ac_entry_id: id of AC entry
 *
 * Return:
 * * 0       - success
 * * -EBADR  - ac entry id unknown
 */
int cxi_tx_profile_remove_ac_entry(struct cxi_tx_profile *tx_profile,
				   unsigned int ac_entry_id)
{
	return cxi_rxtx_profile_remove_ac_entry(&tx_profile->profile_common,
						ac_entry_id);
}
EXPORT_SYMBOL(cxi_tx_profile_remove_ac_entry);

/**
 * cxi_tx_profile_get_ac_entry_ids() - get the list of AC entry ids
 *                                     associated with a Profile
 *
 * @tx_profile: pointer to Profile
 * @max_ids: size of the ac_entry_ids array
 * @ac_entry_ids: location to store ids
 * @num_ids: number of valid ids in ac_entry_ids array on success
 *
 * Return:
 * * 0       - success
 * * -ENOSPC - max_ids is not large enough.  num_ids holds value required.
 */
int cxi_tx_profile_get_ac_entry_ids(struct cxi_tx_profile *tx_profile,
				    size_t max_ids,
				    unsigned int *ac_entry_ids,
				    size_t *num_ids)
{
	return cxi_rxtx_profile_get_ac_entry_ids(&tx_profile->profile_common,
						 max_ids, ac_entry_ids, num_ids);
}
EXPORT_SYMBOL(cxi_tx_profile_get_ac_entry_ids);

/**
 * cxi_tx_profile_get_ac_entry_data() - retrieve the type and data for a
 *                                      AC entry associated with a Profile
 *
 * @tx_profile: pointer to Profile
 * @ac_entry_id: id of AC entry
 * @ac_type: location to store AC entry type
 * @ac_data: location to store AC data
 *
 * Return:
 * * 0       - success
 * * -EBADR  - AC entry id unknown
 */
int cxi_tx_profile_get_ac_entry_data(struct cxi_tx_profile *tx_profile,
				     unsigned int ac_entry_id,
				     enum cxi_ac_type *ac_type,
				     union cxi_ac_data *ac_data)
{
	return cxi_rxtx_profile_get_ac_entry_data(&tx_profile->profile_common,
						  ac_entry_id, ac_type, ac_data);
}
EXPORT_SYMBOL(cxi_tx_profile_get_ac_entry_data);

/**
 * cxi_tx_profile_get_ac_entry_id_by_data() - get the AC entry id associated
 *                                            with a given VNI entry type and data
 *
 * @tx_profile: pointer to Profile
 * @ac_type: type of AC entry to look for
 * @ac_data: AC entry data to look for
 * @ac_entry_id: location to store AC entry id on success
 *
 * Return:
 * * 0        - success
 * * -ENODATA - AC entry with given type&data not found
 * * -EBADR   - invalid ac_type
 */
int cxi_tx_profile_get_ac_entry_id_by_data(struct cxi_tx_profile *tx_profile,
					   enum cxi_ac_type ac_type,
					   const union cxi_ac_data *ac_data,
					   unsigned int *ac_entry_id)
{
	return cxi_rxtx_profile_get_ac_entry_id_by_data(&tx_profile->profile_common,
							ac_type, ac_data, ac_entry_id);
}
EXPORT_SYMBOL(cxi_tx_profile_get_ac_entry_id_by_data);

/**
 * cxi_tx_profile_get_ac_entry_id_by_user() - retrieve the AC entry associated
 *                                            with a Profile by user and group
 *
 * @tx_profile: pointer to Profile
 * @uid: user id
 * @gid: group id
 * @desired_types: OR'd list of enum cxi_ac_type values
 * @ac_entry_id: location to store AC entry id on success
 *
 * Return:
 * * 0       - success
 * * -EPERM  - no AC entries found for given uid and gid
 * * -EBADR  - invalid desired_types
 *
 * Note: multiple AC entries may apply.  The priority of return is
 * CXI_AC_UID, CXI_AC_GID, CXI_AC_OPEN.
 */
int cxi_tx_profile_get_ac_entry_id_by_user(struct cxi_tx_profile *tx_profile,
					   uid_t uid,
					   gid_t gid,
					   cxi_ac_typeset_t desired_types,
					   unsigned int *ac_entry_id)
{
	return cxi_rxtx_profile_get_ac_entry_id_by_user(&tx_profile->profile_common,
							uid, gid, desired_types,
							ac_entry_id);
}
EXPORT_SYMBOL(cxi_tx_profile_get_ac_entry_id_by_user);

/**
 * cxi_tx_profile_add_ac_entry() - add an Access Control entry to
 *                                 an existing Profile
 *
 * @tx_profile: TX profile to add AC Entry
 * @type: type of AC Entry to add
 * @uid: UID for AC Entry
 * @gid: GID for AC Entry
 * @ac_entry_id: Location to store resulting id
 *
 * Return:
 * * 0       - success
 * * -EEXIST - AC Entry already exists
 */
int cxi_tx_profile_add_ac_entry(struct cxi_tx_profile *tx_profile,
				enum cxi_ac_type type, uid_t uid, gid_t gid,
				unsigned int *ac_entry_id)
{
	union cxi_ac_data data = {};

	switch (type) {
	case CXI_AC_UID:
		data.uid = uid;
		break;
	case CXI_AC_GID:
		data.gid = gid;
		break;
	case CXI_AC_OPEN:
		break;
	default:
		return -EDOM;
	}

	return cxi_rxtx_profile_add_ac_entry(&tx_profile->profile_common,
					     type, &data, ac_entry_id);
}
EXPORT_SYMBOL(cxi_tx_profile_add_ac_entry);

/**
 * cxi_tx_profile_remove_ac_entries() - remove Access Control entries
 *                                          from profile
 *
 * @tx_profile: TX profile from which to remove AC entries
 */
void cxi_tx_profile_remove_ac_entries(struct cxi_tx_profile *tx_profile)
{
	if (!tx_profile)
		return;

	cxi_ac_entry_list_destroy(&tx_profile->profile_common.ac_entry_list);
}
EXPORT_SYMBOL(cxi_tx_profile_remove_ac_entries);

/**
 * cxi_tx_profile_exclusive_cp() - Get tx_profile exclusive CP value
 *
 * @tx_profile: pointer to Profile
 *
 * Return: exclusive_cp value
 */
bool cxi_tx_profile_exclusive_cp(struct cxi_tx_profile *tx_profile)
{
	if (!tx_profile)
		return false;

	return tx_profile->config.exclusive_cp;
}
EXPORT_SYMBOL(cxi_tx_profile_exclusive_cp);

/**
 * cxi_tx_profile_set_exclusive_cp() - Set tx_profile exclusive CP value
 *
 * @tx_profile: pointer to Profile
 * @exclusive_cp: Value to set
 *
 * Return: 0 - success, -EBUSY if not enabled
 */
int cxi_tx_profile_set_exclusive_cp(struct cxi_tx_profile *tx_profile,
				    bool exclusive_cp)
{
	if (cxi_tx_profile_is_enabled(tx_profile))
		return -EBUSY;

	tx_profile->config.exclusive_cp = exclusive_cp;

	return 0;
}
EXPORT_SYMBOL(cxi_tx_profile_set_exclusive_cp);
