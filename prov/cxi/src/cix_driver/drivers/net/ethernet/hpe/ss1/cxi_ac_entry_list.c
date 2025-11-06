// SPDX-License-Identifier: GPL-2.0
/* Copyright 2020 Hewlett Packard Enterprise Development LP */

/* Access Control Entry lists */

#include "cass_core.h"
#include "cxi_ac_entry_list.h"

/* **************************************************************** */
/* Access Control List operations                                   */
/* **************************************************************** */

struct cxi_ac_entry {
	unsigned int       id;
	enum cxi_ac_type   type;
	union cxi_ac_data  data;
};

static const struct xa_limit ac_entry_id_limits  = AC_ENTRY_ID_LIMITS;
static const uid_t  invalid_uid = (uid_t) -1;
static const gid_t  invalid_gid = (gid_t) -1;

static bool validate_ac_data(cxi_ac_typeset_t ac_type,
			     const union cxi_ac_data *data)
{
	switch (ac_type) {
	case CXI_AC_UID:
		return (data->uid != invalid_uid);
	case CXI_AC_GID:
		return (data->gid != invalid_gid);
	case CXI_AC_OPEN:
		return true;
	default:
		return false;
	}
}

static int ac_entry_list_insert_entry(struct cxi_ac_entry_list *list,
				      enum cxi_ac_type type,
				      const union cxi_ac_data *data,
				      unsigned int *ac_entry_id)
{
	struct cxi_ac_entry   *ac_entry;
	int    ret;

	ac_entry = kzalloc(sizeof(*ac_entry), AC_ENTRY_GFP_OPTS);
	if (!ac_entry)
		return -ENOMEM;

	/* allocate an Id for this entry */

	ret = xa_alloc(&list->id.xarray, &ac_entry->id, ac_entry,
		       ac_entry_id_limits, AC_ENTRY_GFP_OPTS);

	if (ret)
		goto free_entry;

	ac_entry->type = type;
	ac_entry->data = *data;
	*ac_entry_id   = ac_entry->id;

	/* store the entry in the appropriate xarray */

	switch (ac_entry->type) {
	case CXI_AC_UID:
		ret = xa_insert(&list->uid.xarray, ac_entry->data.uid,
			       ac_entry, AC_ENTRY_GFP_OPTS);
		break;
	case CXI_AC_GID:
		ret = xa_insert(&list->gid.xarray, ac_entry->data.gid,
				ac_entry, AC_ENTRY_GFP_OPTS);
		break;
	case CXI_AC_OPEN:
		xa_lock(&list->id.xarray);
		if (list->open_entry)
			ret = -EBUSY;
		else
			list->open_entry = ac_entry;
		xa_unlock(&list->id.xarray);
		break;
	default:
		ret = -EBADR;
	}

	if (!ret)
		return 0;

	/* if we failed, we need to deallocate the id */

	xa_erase(&list->id.xarray, ac_entry->id);

free_entry:
	kfree(ac_entry);
	return ret;
}

static int ac_entry_list_delete_entry(struct cxi_ac_entry_list *list,
				      unsigned int ac_entry_id)
{
	struct cxi_ac_entry   *ac_entry;
	int    ret;

	ac_entry = xa_load(&list->id.xarray, ac_entry_id);
	ret = xa_err(ac_entry);
	if (ret)
		return ret;
	if (!ac_entry)
		return -ENODATA;

	switch (ac_entry->type) {
	case CXI_AC_UID:
		xa_erase(&list->uid.xarray, ac_entry->data.uid);
		break;
	case CXI_AC_GID:
		xa_erase(&list->gid.xarray, ac_entry->data.gid);
		break;
	case CXI_AC_OPEN:
		list->open_entry = NULL;
		break;
	default:
		ret = -EIO;
		break;
	}

	xa_erase(&list->id.xarray, ac_entry_id);

	kfree(ac_entry);
	return ret;
}

/**
 * ac_entry_list_iterate() - call a function on all entries in the list
 *
 * @list: list pointer
 * @operator: function to call with each AC Entry and user_data
 * @user_data: second argument to operator function
 *
 * Return: the first non-zero return value of 'operator' or 0.
 */
static int ac_entry_list_iterate(struct cxi_ac_entry_list *list,
				 int (*operator)(struct cxi_ac_entry *ac_entry,
						 void *user_data),
				 void *user_data)
{
	struct cxi_ac_entry     *ac_entry;
	unsigned long           index = ac_entry_id_limits.min;

	xa_for_each(&list->id.xarray, index, ac_entry) {
		int ret = operator(ac_entry, user_data);

		if (ret)
			return ret;
	}

	return 0;
}

/**
 * cxi_ac_entry_list_init() - initialize a list of AC Entries
 *
 * @list: address of list
 */
void cxi_ac_entry_list_init(struct cxi_ac_entry_list *list)
{
	xa_init_flags(&list->id.xarray, AC_ENTRY_ID_XARRAY_FLAGS);
	xa_init_flags(&list->gid.xarray, AC_ENTRY_GID_XARRAY_FLAGS);
	xa_init_flags(&list->uid.xarray, AC_ENTRY_UID_XARRAY_FLAGS);
}

/**
 * cxi_ac_entry_list_purge() - Remove all AC Entries from a list
 *
 * @list: address of list
 */
void cxi_ac_entry_list_purge(struct cxi_ac_entry_list *list)
{
	unsigned long          index = ac_entry_id_limits.min;
	struct cxi_ac_entry    *ac_entry;

	xa_for_each(&list->id.xarray, index, ac_entry) {
		ac_entry_list_delete_entry(list, index);
	}
}

/**
 * cxi_ac_entry_list_destroy() - tear down a list of AC Entries.  The
 *                               list infrastructure is destroyed.
 *
 * @list: pointer to list
 */
void cxi_ac_entry_list_destroy(struct cxi_ac_entry_list *list)
{
	cxi_ac_entry_list_purge(list);

	xa_destroy(&list->id.xarray);
	xa_destroy(&list->gid.xarray);
	xa_destroy(&list->uid.xarray);
}

/**
 * cxi_ac_entry_list_empty() - determine if any AC Entries are present
 *
 * @list: list pointer
 *
 * Return: true if no entries present, false otherwise
 */
bool cxi_ac_entry_list_empty(struct cxi_ac_entry_list *list)
{
	return xa_empty(&list->id.xarray);
}

/**
 * cxi_ac_entry_list_insert() - allocate an id and add an AC Entry
 *                              to the list
 *
 * @list: list pointer
 * @ac_type: one of the cxi_ac_type enums
 * @ac_data: pointer to cxi_ac_data struct
 * @id: location to store new Id
 *
 * Return:
 * * 0       - success
 * * -ENOMEM - unable to allocate memory
 * * -EBUSY  - no free entries available
 * * -EBADR  - invalid type or data
 * * -EEXIST - entry with same data exists
 */
int cxi_ac_entry_list_insert(struct cxi_ac_entry_list *list,
			     enum cxi_ac_type ac_type,
			     const union cxi_ac_data *ac_data,
			     unsigned int *id)
{
	int   ret;

	if (!validate_ac_data(ac_type, ac_data))
		return -EBADR;

	ret = ac_entry_list_insert_entry(list, ac_type, ac_data, id);
	switch (ret) {
	case -EBUSY:   /* translate xarray EBUSY to EEXIST */
		return -EEXIST;
	default:
		return ret;
	}
}

/**
 * cxi_ac_entry_list_delete() - remove an AC Entry from the
 *                              list by Id
 *
 * @list: list pointer
 * @id: desired AC Entry Id to remove
 *
 * Return:
 * * 0        - success
 * * -EBADR   - not found with given id
 */
int cxi_ac_entry_list_delete(struct cxi_ac_entry_list *list,
			     unsigned int id)
{
	return ac_entry_list_delete_entry(list, id);
}

/**
 * cxi_ac_entry_list_retrieve_by_id() - get AC type and data by id
 *
 * @list: list pointer
 * @id: id of ac entry to retrieve
 * @type: location to store type
 * @data: location to store data
 *
 * Return:
 * * 0       - success
 * * -EBADR  - AC Entry with given id not found
 */
int cxi_ac_entry_list_retrieve_by_id(struct cxi_ac_entry_list *list,
				     unsigned int id,
				     enum cxi_ac_type *type,
				     union cxi_ac_data *data)
{
	struct cxi_ac_entry   *ac_entry;

	ac_entry = xa_load(&list->id.xarray, id);
	if (!ac_entry)
		return -EBADR;

	switch (ac_entry->type) {
	case CXI_AC_UID:
		data->uid = ac_entry->data.uid;
		break;
	case CXI_AC_GID:
		data->gid = ac_entry->data.gid;
		break;
	case CXI_AC_OPEN:
		break;
	default:
		return -EIO;   /* yikes */
	}
	*type = ac_entry->type;
	return 0;
}

/**
 * cxi_ac_entry_list_retrieve_by_data() - get an AC Entry ID by type and
 *                                        data value
 *
 * @list: list pointer
 * @ac_type: type of entry
 * @ac_data: uid/gid data
 * @id: location to store id value
 *
 * Return:
 * * 0        - success
 * * -EBADR   - AC Entry with given type&data not found
 * * -EINVAL  - invalid ac_type or data
 */
int
cxi_ac_entry_list_retrieve_by_data(struct cxi_ac_entry_list *list,
				   enum cxi_ac_type ac_type,
				   const union cxi_ac_data *ac_data,
				   unsigned int *id)
{
	struct cxi_ac_entry   *ac_entry = NULL;

	if (!validate_ac_data(ac_type, ac_data))
		return -EINVAL;

	switch (ac_type) {
	case CXI_AC_OPEN:
		ac_entry = list->open_entry;
		break;
	case CXI_AC_UID:
		ac_entry = xa_load(&list->uid.xarray, ac_data->uid);
		break;
	case CXI_AC_GID:
		ac_entry = xa_load(&list->gid.xarray, ac_data->gid);
		break;
	default:
		break;
	}

	if (!ac_entry)
		return -EBADR;

	*id = ac_entry->id;
	return 0;
}

/**
 * cxi_ac_entry_list_retrieve_by_user() - get an access control entry Id
 *           which covers the given uid/gid, if any.  There may be more than
 *           one possible, the priority order is: UID, GID, OPEN.
 *
 * @list: the ac_entry_list to search
 * @uid: user id
 * @gid: group id
 * @desired_types: one of more of the enum cxi_ac_type values (OR'd together)
 * @id: location to store the id
 *
 * Return:
 * * 0        - success
 * * -EPERM   - no access control found for uid/gid
 * * -EBADR   - no valid desired types specified
 */
int cxi_ac_entry_list_retrieve_by_user(struct cxi_ac_entry_list *list,
				       uid_t uid,
				       gid_t gid,
				       cxi_ac_typeset_t desired_types,
				       unsigned int *id)
{
	union cxi_ac_data           data = {};

	if ((desired_types & CXI_AC_ANY) == 0)
		return -EBADR;

	if (desired_types & CXI_AC_UID) {
		data.uid = uid;
		if (!cxi_ac_entry_list_retrieve_by_data(list, CXI_AC_UID,
							&data, id))
			return 0;
	}

	if (desired_types & CXI_AC_GID) {
		data.gid = gid;
		if (!cxi_ac_entry_list_retrieve_by_data(list, CXI_AC_GID,
							&data, id))
			return 0;
	}

	if (desired_types & CXI_AC_OPEN) {
		if (!cxi_ac_entry_list_retrieve_by_data(list, CXI_AC_OPEN,
							&data, id))
			return 0;
	}

	return -EPERM;
}

struct get_ids_data {
	size_t         max_ids;
	unsigned int   *ids;
	size_t         num_ids;
};

static int get_ids_operator(struct cxi_ac_entry *ac_entry,
			    void *user_data)
{
	struct get_ids_data   *data = user_data;

	if (data->num_ids < data->max_ids) {
		*data->ids = ac_entry->id;
		data->ids++;
	}

	data->num_ids++;
	return 0;
}

/**
 * cxi_ac_entry_list_get_ids() - retrieve the ids of all entries in the list
 *
 * @list: the list pointer
 * @max_ids: the size of the 'ids' array
 * @ids: location to store ids
 * @num_ids: location to store actual number of ids copied to list
 *
 * Return:
 * * 0       - success
 * * -ENOSPC - max_ids is not large enough (num_ids holds required size)
 */
int cxi_ac_entry_list_get_ids(struct cxi_ac_entry_list *list,
			      size_t max_ids,
			      unsigned int *ids,
			      size_t *num_ids)
{
	struct get_ids_data    data = {
		.max_ids = max_ids,
		.ids     = ids,
		.num_ids = 0,
	};
	int    ret;

	ret = ac_entry_list_iterate(list, get_ids_operator, &data);
	if (ret)
		return ret;

	*num_ids = data.num_ids;

	return (data.num_ids <= max_ids) ? 0 : -ENOSPC;
}
