// SPDX-License-Identifier: GPL-2.0
/* Copyright (C) 2024 Hewlett Packard Enterprise Development LP */

/* Common parts of RX and TX Profile */

#include "cass_core.h"
#include "cxi_rxtx_profile.h"
#include "cxi_rxtx_profile_list.h"

/**
 * cxi_rxtx_profile_init() - Initialize the common portions
 *
 * @rxtx_profile: profile pointer
 * @hw: Cassini device
 * @vni_attr: VNI attributes for profile
 *
 */
void cxi_rxtx_profile_init(struct cxi_rxtx_profile *rxtx_profile,
			   struct cass_dev *hw,
			   const struct cxi_rxtx_vni_attr *vni_attr)
{
	rxtx_profile->vni_attr.match  = vni_attr->match;
	rxtx_profile->vni_attr.ignore = vni_attr->ignore;
	strscpy(rxtx_profile->vni_attr.name, vni_attr->name,
		ARRAY_SIZE(rxtx_profile->vni_attr.name));

	/* Profiles are long lived.  We indicate this by an
	 * initial reference count of 1, and a release status of false.
	 *
	 * The release operation is atomic: it sets the value of
	 * 'released' to true, and decrements the reference count.
	 *
	 * When no references remain, the profile is deleted.
	 *
	 * TODO: is revoke still needed?
	 */

	refcount_set(&rxtx_profile->state.refcount, 1);
	rxtx_profile->state.enable = false;

	cxi_ac_entry_list_init(&rxtx_profile->ac_entry_list);
}

/**
 * cxi_rxtx_profile_destroy() - tear down common portion
 *
 * @rxtx_profile: profile pointer
 */
void cxi_rxtx_profile_destroy(struct cxi_rxtx_profile *rxtx_profile)
{
	/* TODO: make sure these commented functions get */
	/* implemented in rx and tx profiles... */

	/* cass_vni_entry_disable_hw(vni_entry); */
	cxi_ac_entry_list_destroy(&rxtx_profile->ac_entry_list);
}

/**
 * cxi_rxtx_profile_find_inc_refcount() - Look up RX Profile in the
 *                                        device list by ID and increment
 *                                        its refcount.
 *
 * @list: profile list
 * @profile_id: the ID to use for the lookup
 * @rxtx_profile: location to place rxtx_profile pointer
 *
 * Return: 0 on success or error code.
 *
 * Note: this function is used internally for cases where we need to
 * examine a profile even if it is released or revoked.
 *
 * Refcount must be decremented when usage is done.
 */
int cxi_rxtx_profile_find_inc_refcount(struct cxi_rxtx_profile_list *list,
				       unsigned int profile_id,
				       struct cxi_rxtx_profile **rxtx_profile)
	__must_hold(&list->xarray.xa_lock)
{
	struct cxi_rxtx_profile    *profile;
	int    ret = 0;

	ret = cxi_rxtx_profile_list_retrieve(list, profile_id, &profile);
	if (ret)
		goto out;

	if (!refcount_inc_not_zero(&profile->state.refcount)) {
		ret = -EBADR;
		goto out;
	}

	*rxtx_profile = profile;

out:
	return ret;
}

/**
 * cxi_rxtx_profile_release() - Mark a Profile as released.
 *
 * No new references can be taken.
 *
 * @rxtx_profile: pointer to profile
 */
void cxi_rxtx_profile_release(struct cxi_rxtx_profile *rxtx_profile)
{
	/* Reduce the reference count.  Note that this function is
	 * called via pointer to profile, indicating that the caller
	 * also has a reference to this profile.  Thus the reference
	 * count after this function returns will be at least 1.
	 * (Eating the return from cxi_rxtx_profile_dec_refcount() is ok).
	 */
	refcount_dec(&rxtx_profile->state.refcount);
}

/**
 * cxi_rxtx_profile_revoke() - Revoke the VNI associated with profile.
 *
 * RDMA operations for this VNI will fail.  Since the Profile is
 * essentially dead at this point, 'revoke' implies 'release' as well.
 *
 * @rxtx_profile: pointer to profile
 */
void cxi_rxtx_profile_revoke(struct cxi_rxtx_profile *rxtx_profile)
{
	/* Remove all the Access Control entries associated with
	 * this Profile.  This ensures it can't be used again.
	 */

	cxi_ac_entry_list_purge(&rxtx_profile->ac_entry_list);

	/* TODO: this has to be done at the RX or TX layer */
	/* cass_rx_profile_revoke(rx_profile); */

	cxi_rxtx_profile_release(rxtx_profile);
}

/**
 * cxi_rxtx_profile_get_info() - Retrieve the attributes and state associated
 *                               with this Profile.
 *
 * @rxtx_profile: pointer to Profile
 * @vni_attr: location to place attributes
 * @state: location to put state
 *
 * Note: attr and/or state may be NULL.
 */
void cxi_rxtx_profile_get_info(struct cxi_rxtx_profile *rxtx_profile,
			       struct cxi_rxtx_vni_attr *vni_attr,
			       struct cxi_rxtx_profile_state *state)
{
	if (vni_attr) {
		vni_attr->ignore   = rxtx_profile->vni_attr.ignore;
		vni_attr->match    = rxtx_profile->vni_attr.match;
		strscpy(vni_attr->name, rxtx_profile->vni_attr.name,
			ARRAY_SIZE(vni_attr->name));
	}

	if (state) {
		state->enable = rxtx_profile->state.enable;
		state->refcount = rxtx_profile->state.refcount;
	}
}

/**
 * cxi_rxtx_profile_add_ac_entry() - add an Access Control entry to
 *                                   an existing Profile
 *
 * @rxtx_profile: pointer to Profile
 * @ac_type: type of AC Entry to add
 * @ac_data: UID/GID for AC Entry
 * @ac_entry_id: location to put AC Entry id on success
 *
 * Return:
 * * 0       - success
 * * -EBUSY  - AC Entry already exists
 */
int cxi_rxtx_profile_add_ac_entry(struct cxi_rxtx_profile *rxtx_profile,
				  enum cxi_ac_type ac_type,
				  union cxi_ac_data *ac_data,
				  unsigned int *ac_entry_id)
{
	/* insert will fail with error if data is already present */

	return cxi_ac_entry_list_insert(&rxtx_profile->ac_entry_list, ac_type,
					ac_data, ac_entry_id);
}

/**
 * cxi_rxtx_profile_remove_ac_entry() - disable access control to a Profile
 *                                      by access control id.
 *
 * @rxtx_profile: pointer to profile
 * @ac_entry_id: id of AC entry
 *
 * Return:
 * * 0       - success
 * * -EBADR  - ac entry id unknown
 */
int cxi_rxtx_profile_remove_ac_entry(struct cxi_rxtx_profile *rxtx_profile,
				     unsigned int ac_entry_id)
{
	return cxi_ac_entry_list_delete(&rxtx_profile->ac_entry_list,
					ac_entry_id);
}

/**
 * cxi_rxtx_profile_get_ac_entry_ids() - get the list of AC entry ids
 *                                       associated with a Profile
 *
 * @rxtx_profile: pointer to Profile
 * @max_ids: size of the ac_entry_ids array
 * @ac_entry_ids: location to store ids
 * @num_ids: number of valid ids in ac_entry_ids array on success
 *
 * Return:
 * * 0       - success
 * * -ENOSPC - max_ids is not large enough.  num_ids holds value required.
 */
int cxi_rxtx_profile_get_ac_entry_ids(struct cxi_rxtx_profile *rxtx_profile,
				      size_t max_ids,
				      unsigned int *ac_entry_ids,
				      size_t *num_ids)
{
	return cxi_ac_entry_list_get_ids(&rxtx_profile->ac_entry_list, max_ids,
					 ac_entry_ids, num_ids);
}

/**
 * cxi_rxtx_profile_get_ac_entry_data() - retrieve the type and data for a
 *                                        AC entry associated with a Profile
 *
 * @rxtx_profile: pointer to Profile
 * @ac_entry_id: id of AC entry
 * @ac_type: location to store AC entry type
 * @ac_data: location to store AC data
 *
 * Return:
 * * 0       - success
 * * -EBADR  - AC entry id unknown
 */
int cxi_rxtx_profile_get_ac_entry_data(struct cxi_rxtx_profile *rxtx_profile,
				       unsigned int ac_entry_id,
				       enum cxi_ac_type *ac_type,
				       union cxi_ac_data *ac_data)
{
	return cxi_ac_entry_list_retrieve_by_id(&rxtx_profile->ac_entry_list,
						ac_entry_id, ac_type, ac_data);
}

/**
 * cxi_rxtx_profile_get_ac_entry_id_by_data() - get the AC entry id associated
 *                                              with a Profile for a given
 *                                              type and data
 *
 * @rxtx_profile: pointer to Profile
 * @ac_type: type of AC entry to look for
 * @ac_data: AC entry data to look for
 * @ac_entry_id: location to store AC entry id on success
 *
 * Return:
 * * 0        - success
 * * -ENODATA - AC entry with given type&data not found
 * * -EBADR   - invalid ac_type
 */
int cxi_rxtx_profile_get_ac_entry_id_by_data(struct cxi_rxtx_profile *rxtx_profile,
					     enum cxi_ac_type ac_type,
					     const union cxi_ac_data *ac_data,
					     unsigned int *ac_entry_id)
{
	return cxi_ac_entry_list_retrieve_by_data(&rxtx_profile->ac_entry_list,
						  ac_type, ac_data, ac_entry_id);
}

/**
 * cxi_rxtx_profile_get_ac_entry_id_by_user() - retrieve the AC entry associated
 *                                              with a Profile by user and group
 *
 * @rxtx_profile: pointer to Profile
 * @uid: user id
 * @gid: group id
 * @desired_types: one or more of the enum cxi_ac_type values OR's together
 * @ac_entry_id: location to store AC entry id on success
 *
 * Return:
 * * 0       - success
 * * -EPERM  - no AC entries found for given uid and gid
 *
 * Note: multiple AC entries may apply.  The priority of return is
 * CXI_AC_UID, CXI_AC_GID, CXI_AC_OPEN.
 */
int cxi_rxtx_profile_get_ac_entry_id_by_user(struct cxi_rxtx_profile *rxtx_profile,
					     uid_t uid,
					     gid_t gid,
					     cxi_ac_typeset_t desired_types,
					     unsigned int *ac_entry_id)
{
	return cxi_ac_entry_list_retrieve_by_user(&rxtx_profile->ac_entry_list,
						  uid, gid, desired_types, ac_entry_id);
}

struct valid_ac_data {
	uid_t uid;
	gid_t gid;
};

struct valid_user_data {
	unsigned int vni;
	struct valid_ac_data ac_data;
};

int vni_overlap_test(struct cxi_rxtx_profile *profile1,
		     void *user_arg)
{
	struct cxi_rxtx_profile  *profile2 = user_arg;
	bool overlap = vni_overlap(&profile1->vni_attr, &profile2->vni_attr);

	/* ignore profiles with 0 VNI */
	if (zero_vni(&profile1->vni_attr))
		return 0;

	return overlap ? -EEXIST : 0;
}

/* Make sure the VNI space is unique */
bool unique_vni_space(struct cass_dev *hw,
		      struct cxi_rxtx_profile_list *list,
		      const struct cxi_rxtx_vni_attr *attr)
{
	int rc;
	struct cxi_rxtx_profile rxtx_prof = {
		.vni_attr.match = attr->match,
		.vni_attr.ignore = attr->ignore,
	};

	cxi_rxtx_profile_list_lock(list);

	rc = cxi_rxtx_profile_list_iterate(list,
					   vni_overlap_test,
					   &rxtx_prof);

	cxi_rxtx_profile_list_unlock(list);

	return !rc;
}

static int valid_vni_operator(struct cxi_rxtx_profile *rxtx_profile,
			       void *user_data)
{
	int rc;
	bool valid = false;
	unsigned int ac_entry_id;
	struct cxi_rxtx_vni_attr vni_attr;
	struct cxi_rxtx_profile_state state;
	struct valid_user_data *data = user_data;
	struct cxi_rxtx_vni_attr user_vni = {
		.match = data->vni,
	};
	struct valid_ac_data ac_data = data->ac_data;

	cxi_rxtx_profile_get_info(rxtx_profile, &vni_attr, &state);

	if (!state.enable)
		return false;

	/* If there is a vni match, check if user is authorized */
	if (vni_overlap(&user_vni, &vni_attr)) {
		rc = cxi_rxtx_profile_get_ac_entry_id_by_user(rxtx_profile,
							      ac_data.uid,
							      ac_data.gid,
							      CXI_AC_ANY,
							      &ac_entry_id);
		if (!rc)
			valid = true;
	}

	return valid;
}

/**
 * cxi_valid_vni() - Check if user has permission to use a VNI by searching
 *                   for a profile containing the VNI.
 *
 * @dev: Cassini Device
 * @type: TX or RX profile
 * @vni: VNI to verify
 *
 * Return:
 * * true if VNI is valid
 */
bool cxi_valid_vni(struct cxi_dev *dev, enum cxi_profile_type type,
		   unsigned int vni)
{
	struct cxi_rxtx_profile_list *list;
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct valid_user_data user_data = {
		.vni = vni,
		.ac_data = {
			.uid = __kuid_val(current_euid()),
			.gid = __kgid_val(current_egid()),
		},
	};

	if (type == CXI_PROF_RX)
		list = &hw->rx_profile_list;
	else
		list = &hw->tx_profile_list;

	return cxi_rxtx_profile_list_iterate(list, valid_vni_operator,
					     &user_data);
}
