/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright (C) 2024 Hewlett Packard Enterprise Development LP */

/** Common parts of RX and TX Profiles
 *
 * These functions should only be used to implement RX and TX profile
 * functions.
 */

#ifndef _CXI_RXTX_PROFILE_H_
#define _CXI_RXTX_PROFILE_H_

#define DEF_RMU_INDEX (-1)

void cxi_rxtx_profile_init(struct cxi_rxtx_profile *rxtx_profile,
			   struct cass_dev *hw,
			   const struct cxi_rxtx_vni_attr *vni_attr);

void cxi_rxtx_profile_destroy(struct cxi_rxtx_profile *rxtx_profile);

int cxi_rxtx_profile_find_inc_refcount(struct cxi_rxtx_profile_list *list,
				       unsigned int profile_id,
				       struct cxi_rxtx_profile **rxtx_profile);

int cxi_rxtx_profile_test_insert(struct cxi_rxtx_profile_list *list,
				 struct cxi_rxtx_profile *rxtx_profile,
				 unsigned int *id,
				 int (*test_func)(const struct cxi_rxtx_profile *p1,
						  const struct cxi_rxtx_profile *p2));

void cxi_rxtx_profile_release(struct cxi_rxtx_profile *rxtx_profile);
void cxi_rxtx_profile_revoke(struct cxi_rxtx_profile *rxtx_profile);

void cxi_rxtx_profile_get_info(struct cxi_rxtx_profile *rxtx_profile,
			       struct cxi_rxtx_vni_attr *vni_attr,
			       struct cxi_rxtx_profile_state *state);

int cxi_rxtx_profile_add_ac_entry(struct cxi_rxtx_profile *rxtx_profile,
				  enum cxi_ac_type ac_type,
				  union cxi_ac_data *ac_data,
				  unsigned int *ac_entry_id);

int cxi_rxtx_profile_remove_ac_entry(struct cxi_rxtx_profile *rxtx_profile,
				     unsigned int ac_entry_id);

int cxi_rxtx_profile_get_ac_entry_ids(struct cxi_rxtx_profile *rxtx_profile,
				      size_t max_ids,
				      unsigned int *ac_entry_ids,
				      size_t *num_ids);

int cxi_rxtx_profile_get_ac_entry_data(struct cxi_rxtx_profile *rxtx_profile,
				       unsigned int ac_entry_id,
				       enum cxi_ac_type *ac_type,
				       union cxi_ac_data *ac_data);

int cxi_rxtx_profile_get_ac_entry_id_by_data(struct cxi_rxtx_profile *rxtx_profile,
					     enum cxi_ac_type ac_type,
					     const union cxi_ac_data *ac_data,
					     unsigned int *ac_entry_id);

int cxi_rxtx_profile_get_ac_entry_id_by_user(struct cxi_rxtx_profile *rxtx_profile,
					     uid_t uid,
					     gid_t gid,
					     cxi_ac_typeset_t desired_types,
					     unsigned int *ac_entry_id);

void cxi_tx_profile_list_destroy(struct cass_dev *cdev);

int vni_overlap_test(struct cxi_rxtx_profile *profile1,
		     void *user_arg);
bool unique_vni_space(struct cass_dev *hw,
		      struct cxi_rxtx_profile_list *list,
		      const struct cxi_rxtx_vni_attr *attr);

__maybe_unused
static struct cxi_rx_profile *co_rx_profile(struct cxi_rxtx_profile *profile)
{
	return container_of(profile, struct cxi_rx_profile, profile_common);
}

__maybe_unused
static struct cxi_tx_profile *co_tx_profile(struct cxi_rxtx_profile *profile)
{
	return container_of(profile, struct cxi_tx_profile, profile_common);
}

/**
 * vni_well_formed() - determine if any of the match bits
 *                     are also in the ignore mask.
 *
 * @attr: pointer to the VNI attributes
 *
 * Return: whether we consider this set of attributes good.
 */
static inline bool vni_well_formed(const struct cxi_rxtx_vni_attr *attr)
{
	return !(attr->match & attr->ignore);
}

/**
 * zero_vni() - Check for the zero vni which is invalid
 *              Used for allocating an rx profile with unknown attributes
 *              which will be updated later.
 *
 * @attr: pointer to the VNI attributes
 *
 * Return: The match value is the default (invalid) value of 0.
 */
static inline bool zero_vni(const struct cxi_rxtx_vni_attr *attr)
{
	return !attr->match;
}

/**
 * vni_overlap() - determine if 2 vni specifications share
 *                 any VNI space.
 *
 * @attr1: pointer to a VNI attribute
 * @attr2: pointer to second VNI attribute
 *
 * Return: true if any VNI satisfies both specifications.
 */
static inline bool vni_overlap(const struct cxi_rxtx_vni_attr *attr1,
			       const struct cxi_rxtx_vni_attr *attr2)
{
	uint16_t   ignore = attr1->ignore | attr2->ignore;

	if ((attr1->match & ~ignore) != (attr2->match & ~ignore))
		return false;

	return true;
}

#endif /* _CXI_RXTX_PROFILE_H_ */
