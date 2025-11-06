// SPDX-License-Identifier: GPL-2.0-only
/* Copyright (C) 2024 Hewlett Packard Enterprise Development LP */

/* RX Profile and TX Profile List Implementations */

#include "cass_core.h"
#include "cxi_rxtx_profile_list.h"

/* **************************************************************** */
/* RXTX Profile List implementation                                 */
/* **************************************************************** */

/**
 * cxi_rxtx_profile_list_init() - initialize the Profile list object
 *
 * @list: address of list object
 * @limits: address of index limits struct
 * @flags: optional flags for xarray
 * @gfp_opts: memory allocation options for xarray
 */
void cxi_rxtx_profile_list_init(struct cxi_rxtx_profile_list *list,
				struct xa_limit *limits,
				gfp_t flags,
				gfp_t gfp_opts)
{
	xa_init_flags(&list->xarray, flags);

	list->limits   = limits;
	list->flags    = flags;
	list->gfp_opts = gfp_opts;
}

/**
 * cxi_rxtx_profile_list_destroy() - remove all Profile references
 * from the list and free the underlying resources.
 *
 * @list: address of list
 * @cleanup: function to call for additional cleanup processing of each
 *           profile in the list
 * @user_arg: second arg to cleanup function
 */
void cxi_rxtx_profile_list_destroy(struct cxi_rxtx_profile_list *list,
				   void (*cleanup)(struct cxi_rxtx_profile *profile,
						   void *user_arg),
				   void *user_arg)
{
	unsigned long              index;
	struct cxi_rxtx_profile    *profile;

	index = list->limits->min;

	xa_lock(&list->xarray);

	xa_for_each(&list->xarray, index, profile) {
		__xa_erase(&list->xarray, index);
		if (cleanup)
			cleanup(profile, user_arg);
	}

	xa_unlock(&list->xarray);
	xa_destroy(&list->xarray);
}

/**
 * cxi_rxtx_profile_list_lock() - lock the list
 *
 * @list: address of list
 */
void cxi_rxtx_profile_list_lock(struct cxi_rxtx_profile_list *list)
	__acquires(&list->xarray.xa_lock)
{
	xa_lock(&list->xarray);
}

/**
 * cxi_rxtx_profile_list_unlock() - unlock the list
 *
 * @list: address of list
 */
void cxi_rxtx_profile_list_unlock(struct cxi_rxtx_profile_list *list)
	__releases(&list->xarray.xa_lock)
{
	xa_unlock(&list->xarray);
}

/**
 * cxi_rxtx_profile_list_insert() - Allocate a free Id and
 *                                  add a Profile to the list.
 *
 * @list: address of list
 * @profile: address of Profile to add
 * @profile_id: location to store newly allocated Id on success
 *
 * Return:
 * * 0 - success
 *
 * Errors related to memory allocation failures are also possible.
 */
int cxi_rxtx_profile_list_insert(struct cxi_rxtx_profile_list *list,
				 struct cxi_rxtx_profile *profile,
				 unsigned int *profile_id)
	__must_hold(&list->xarray.xa_lock)
{
	int    ret;

	ret = __xa_alloc(&list->xarray, &profile->id, profile,
			 *list->limits, list->gfp_opts);

	if (!ret)
		*profile_id = profile->id;

	return ret;
}

/**
 * cxi_rxtx_profile_list_remove() - remove the Profile from the list
 *
 * @list: address of list struct
 * @rxtx_profile_id: index of Profile
 *
 * Return: 0 on success
 * -EBADR: Profile struct is not in list
 */
int cxi_rxtx_profile_list_remove(struct cxi_rxtx_profile_list *list,
				 unsigned int rxtx_profile_id)
{
	struct cxi_rxtx_profile   *previous_entry;

	previous_entry = xa_erase(&list->xarray, rxtx_profile_id);

	return previous_entry ? 0 : -EBADR;
}

/**
 * cxi_rxtx_profile_list_retrieve() - get the pointer to a Profile by Id
 *
 * @list: address of list
 * @profile_id: Id of desired Profile
 * @profile: location to place Profile pointer
 *
 * Return: 0 on success
 * -EBADR: Profile with desired Id not found.
 */
int cxi_rxtx_profile_list_retrieve(struct cxi_rxtx_profile_list *list,
				   unsigned int profile_id,
				   struct cxi_rxtx_profile **profile)
{
	struct cxi_rxtx_profile  *my_profile;
	int    ret;

	my_profile = xa_load(&list->xarray, profile_id);
	ret = xa_err(my_profile);

	if (ret)
		return ret;

	*profile = my_profile;

	return my_profile ? 0 : -EBADR;
}

/**
 * cxi_rxtx_profile_list_iterate() - walk list and call function on each member.
 *
 * @list: address of list
 * @operator: function to call (see below).
 * @user_args: other data to pass to operator (see below).
 *
 * This iterator calls the function 'operator(profile, user_args)' for each
 * profile in the list.
 * Iteration continues as long as the return value of 'operator' is 0.
 *
 * The return value of the iterator function is the final return value
 * of the operator.
 *
 * The caller may lock the list during iteration but it is not required.
 */
int cxi_rxtx_profile_list_iterate(struct cxi_rxtx_profile_list *list,
				  int (*operator)(struct cxi_rxtx_profile *profile,
						  void *user_args),
				  void *user_args)
{
	struct cxi_rxtx_profile    *profile;
	unsigned long              index;

	index = list->limits->min;

	xa_for_each(&list->xarray, index, profile) {
		int ret = operator(profile, user_args);

		if (ret)
			return ret;
	}

	return 0;
}

struct get_ids_data {
	size_t         max_ids;
	unsigned int   *ids;
	size_t         num_ids;
};

static int get_ids_operator(struct cxi_rxtx_profile *rxtx_profile,
			    void *user_data)
{
	struct get_ids_data   *data = user_data;

	if (data->num_ids < data->max_ids) {
		*data->ids = rxtx_profile->id;
		data->ids++;
	}

	data->num_ids++;
	return 0;
}

/**
 * cxi_rxtx_profile_list_get_ids() - Get a list of the active IDs
 *
 * @list: address of list
 * @max_ids: size of the ids array
 * @ids: array to put ids in
 * @num_ids: number of valid IDs in ids
 *
 * Return:
 * * 0       - success
 * * -ENOSPC - max_ids is not large enough, num_ids hold required value
 */
int cxi_rxtx_profile_list_get_ids(struct cxi_rxtx_profile_list *list,
				  size_t max_ids,
				  unsigned int *ids,
				  size_t *num_ids)
{
	struct get_ids_data   data = {
		.max_ids = max_ids,
		.ids     = ids,
		.num_ids = 0,
	};
	int    ret;

	ret = cxi_rxtx_profile_list_iterate(list, get_ids_operator, &data);
	if (ret)
		return ret;

	*num_ids = data.num_ids;

	return (data.num_ids > max_ids) ? -ENOSPC : 0;
}
