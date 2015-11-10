/*
 * Copyright (c) 2015 Cray Inc. All rights reserved.
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

//
// Address vector common code
//
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <assert.h>

#include "gnix.h"
#include "gnix_util.h"
#include "gnix_hashtable.h"
#include "gnix_av.h"

/*
 * local variables and structs
 */

#define GNIX_AV_ENTRY_VALID		(1ULL)
#define GNIX_AV_ENTRY_CM_NIC_ID		(1ULL << 2)

struct gnix_av_block {
	struct slist_entry        slist;
	struct gnix_av_addr_entry *base;
};

/*******************************************************************************
 * Forward declarations of ops structures.
 ******************************************************************************/
static struct fi_ops_av gnix_av_ops;
static struct fi_ops gnix_fi_av_ops;

/*******************************************************************************
 * Helper functions.
 ******************************************************************************/
/*
 * TODO: Check for named AV creation and check RX CTX bits.
 */
static int gnix_verify_av_attr(struct fi_av_attr *attr)
{
	int ret = FI_SUCCESS;

	switch (attr->type) {
	case FI_AV_TABLE:
	case FI_AV_MAP:
		break;
	default:
		ret = -FI_EINVAL;
		break;
	}

	return ret;
}

/*
 * Check the capacity of the internal table used to represent FI_AV_TABLE type
 * address vectors. Initially the table starts with a capacity and count of 0
 * and the capacity increases by roughly double each time a resize is necessary.
 */
static int gnix_check_capacity(struct gnix_fid_av *av, size_t count)
{
	struct gnix_av_addr_entry *addrs = NULL;
	int *valid_entry_vec = NULL;
	size_t capacity = av->capacity;
	size_t prev_capacity;

	/*
	 * av->count + count is the amount of used indices after adding the
	 * count items.
	 */
	prev_capacity = capacity;
	while (capacity < av->count + count) {
		/*
		 * Handle initial capacity of 0, by adding 1.
		 */
		capacity = capacity * 2 + 1;
	}

	/*
	 * Don't need to grow the table.
	 */
	if (capacity == av->capacity) {
		return FI_SUCCESS;
	}

	addrs = realloc(av->table, capacity * sizeof(*addrs));
	if (!addrs) {
		return -FI_ENOMEM;
	}

	memset(&addrs[prev_capacity], 0, (capacity - prev_capacity) *
	       sizeof(*addrs));

	valid_entry_vec = realloc(av->valid_entry_vec, capacity * sizeof(int));
	if (!valid_entry_vec) {
		return -FI_ENOMEM;
	}

	memset(&valid_entry_vec[prev_capacity], 0, (capacity - prev_capacity) *
	       sizeof(int));

	/*
	 * Update table and capacity to reflect new values.
	 */
	av->table = addrs;
	av->valid_entry_vec = valid_entry_vec;
	av->capacity = capacity;

	return FI_SUCCESS;
}

/*******************************************************************************
 * FI_AV_TABLE specific implementations.
 ******************************************************************************/
static int table_insert(struct gnix_fid_av *int_av, const void *addr,
			size_t count, fi_addr_t *fi_addr, uint64_t flags,
			void *context)
{
	struct gnix_ep_name *temp = NULL;
	int ret = count;
	size_t index;
	size_t i;

	if (gnix_check_capacity(int_av, count)) {
		ret = -FI_ENOMEM;
		goto err;
	}

	assert(int_av->table);
	for (index = int_av->count, i = 0; i < count; index++, i++) {
		temp = &((struct gnix_ep_name *)addr)[i];
		int_av->table[index].gnix_addr = temp->gnix_addr;
		int_av->valid_entry_vec[index] = 1;
		int_av->table[index].name_type = temp->name_type;
		int_av->table[index].cookie = temp->cookie;
		int_av->table[index].cm_nic_cdm_id =
				temp->cm_nic_cdm_id;
		if (fi_addr)
			fi_addr[i] = index;
	}

	int_av->count += count;

err:
	return ret;
}

/*
 * Currently only marks as 'not valid'. Should actually free memory.
 * If any of the given address fail to be removed (are already marked removed)
 * then the return value will be -FI_EINVAL.
 */
static int table_remove(struct gnix_fid_av *int_av, fi_addr_t *fi_addr,
			size_t count, uint64_t flags)
{
	int ret = FI_SUCCESS;
	size_t index;
	size_t i;

	for (i = 0; i < count; i++) {
		index = (size_t) fi_addr[i];
		if (index < int_av->count) {
			if (int_av->valid_entry_vec[index] == 0) {
				ret = -FI_EINVAL;
				break;
			} else {
				int_av->valid_entry_vec[index] = 0;
			}
		} else {
			ret = -FI_EINVAL;
			break;
		}
	}

	return ret;
}

/*
 * table_lookup(): Translate fi_addr_t to struct gnix_address.
 */
static int table_lookup(struct gnix_fid_av *int_av, fi_addr_t fi_addr,
			struct gnix_av_addr_entry **entry_ptr)
{
	size_t index;
	struct gnix_av_addr_entry *entry = NULL;

	index = (size_t)fi_addr;
	if (index >= int_av->count)
		return -FI_EINVAL;

	assert(int_av->table);
	entry = &int_av->table[index];

	if (entry == NULL)
		return -FI_EINVAL;

	if (int_av->valid_entry_vec[index] == 0)
		return -FI_EINVAL;

	*entry_ptr = entry;

	return FI_SUCCESS;
}


/*******************************************************************************
 * FI_AV_MAP specific implementations.
 ******************************************************************************/

static int map_insert(struct gnix_fid_av *int_av, const void *addr,
		      size_t count, fi_addr_t *fi_addr, uint64_t flags,
		      void *context)
{
	int ret;
	struct gnix_ep_name *temp = NULL;
	struct gnix_av_addr_entry *the_entry;
	gnix_ht_key_t key;
	size_t i;
	struct gnix_av_block *blk = NULL;

	assert(int_av->map_ht != NULL);

	if (count == 0)
		return 0;

	blk = calloc(1, sizeof(struct gnix_av_block));
	if (blk == NULL)
		return -FI_ENOMEM;

	blk->base =  calloc(count, sizeof(struct gnix_av_addr_entry));
	if (blk->base == NULL) {
		free(blk);
		return -FI_ENOMEM;
	}

	slist_insert_tail(&blk->slist, &int_av->block_list);

	for (i = 0; i < count; i++) {
		temp = &((struct gnix_ep_name *)addr)[i];
		((struct gnix_address *)fi_addr)[i] = temp->gnix_addr;
		the_entry =  &blk->base[i];
		memcpy(&the_entry->gnix_addr, &temp->gnix_addr,
		       sizeof(struct gnix_address));
		the_entry->name_type = temp->name_type;
		the_entry->cm_nic_cdm_id = temp->cm_nic_cdm_id;
		the_entry->cookie = temp->cookie;
		memcpy(&key, &temp->gnix_addr, sizeof(gnix_ht_key_t));
		ret = _gnix_ht_insert(int_av->map_ht,
				      key,
				      the_entry);
		/*
		 * we are okay with user trying to add more
		 * entries with same key.
		 */
		if ((ret != FI_SUCCESS) && (ret != -FI_ENOSPC)) {
			GNIX_WARN(FI_LOG_AV,
				  "_gnix_ht_insert failed %d\n",
				  ret);
			return ret;
		}

	}

	return count;
}

/*
 * TODO: slab should be freed when entries in the slab drop to zero,
 * or as an alternative, have a free list for slabs so they can be
 * reused if new fi_av_insert operations are performed.
 */
static int map_remove(struct gnix_fid_av *int_av, fi_addr_t *fi_addr,
		      size_t count, uint64_t flags)
{
	int i,ret = FI_SUCCESS;
	struct gnix_av_addr_entry *the_entry = NULL;
	gnix_ht_key_t key;

	for (i = 0; i < count; i++) {

		key = *(gnix_ht_key_t *)&fi_addr[i];

		/*
		 * first see if we have this entry in the hash
		 * TODO: is there a race condition here for multi-threaded?
		 */

		the_entry = _gnix_ht_lookup(int_av->map_ht, key);
		if (the_entry == NULL)
			return -FI_ENOENT;

		ret = _gnix_ht_remove(int_av->map_ht, key);

	}

	return ret;
}

static int map_lookup(struct gnix_fid_av *int_av, fi_addr_t fi_addr,
		      struct gnix_av_addr_entry **entry_ptr)
{
	gnix_ht_key_t *key = (gnix_ht_key_t *)&fi_addr;
	struct gnix_av_addr_entry *entry;

	entry = _gnix_ht_lookup(int_av->map_ht, *key);
	if (entry == NULL)
		return -FI_ENOENT;

	*entry_ptr = entry;

	return FI_SUCCESS;
}

/*******************************************************************************
 * FI_AV API implementations.
 ******************************************************************************/

int _gnix_av_lookup(struct gnix_fid_av *gnix_av, fi_addr_t fi_addr,
		    struct gnix_av_addr_entry **entry_ptr)
{
	int ret = FI_SUCCESS;

	GNIX_TRACE(FI_LOG_AV, "\n");

	if (!gnix_av) {
		ret = -FI_EINVAL;
		goto err;
	}

	switch (gnix_av->type) {
	case FI_AV_TABLE:
		ret = table_lookup(gnix_av, fi_addr, entry_ptr);
		break;
	case FI_AV_MAP:
		ret = map_lookup(gnix_av, fi_addr, entry_ptr);
		break;
	default:
		ret = -FI_EINVAL;
		break;
	}

err:
	return ret;
}

/*
 * Note: this function (according to WG), is not intended to
 * typically be used in the critical path for messaging/rma/amo
 * requests
 */
static int gnix_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr,
			  size_t *addrlen)
{
	struct gnix_fid_av *gnix_av;
	struct gnix_ep_name ep_name = { {0} };
	struct gnix_av_addr_entry *entry = NULL;
	int rc;

	GNIX_TRACE(FI_LOG_AV, "\n");

	if (!av || !addrlen)
		return -FI_EINVAL;

	if (*addrlen < sizeof(ep_name)) {
		*addrlen = sizeof(ep_name);
		return -FI_ETOOSMALL;
	}

	/*
	 * user better have provided a buffer since the
	 * value stored in addrlen is big enough to return ep_name
	 */

	if (!addr)
		return -FI_EINVAL;

	gnix_av = container_of(av, struct gnix_fid_av, av_fid);

	rc = _gnix_av_lookup(gnix_av, fi_addr, &entry);
	if (rc != FI_SUCCESS) {
		GNIX_WARN(FI_LOG_AV, "_gnix_av_lookup failed: %d\n", rc);
		return rc;
	}

	memcpy(&ep_name.gnix_addr, &entry->gnix_addr,
	       sizeof(struct gnix_address));
	ep_name.name_type = entry->name_type;
	ep_name.cm_nic_cdm_id = entry->cm_nic_cdm_id;
	ep_name.cookie = entry->cookie;

	memcpy(addr, (void *)&ep_name, MIN(*addrlen, sizeof(ep_name)));
	*addrlen = sizeof(ep_name);

	return FI_SUCCESS;
}

static int gnix_av_insert(struct fid_av *av, const void *addr, size_t count,
			  fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	struct gnix_fid_av *int_av = NULL;
	int ret = FI_SUCCESS;

	GNIX_TRACE(FI_LOG_AV, "\n");

	if (!av) {
		ret = -FI_EINVAL;
		goto err;
	}

	int_av = container_of(av, struct gnix_fid_av, av_fid);

	if (!int_av) {
		ret = -FI_EINVAL;
		goto err;
	}

	switch (int_av->type) {
	case FI_AV_TABLE:
		ret =
		    table_insert(int_av, addr, count, fi_addr, flags, context);
		break;
	case FI_AV_MAP:
		ret = map_insert(int_av, addr, count, fi_addr, flags, context);
		break;
	default:
		ret = -FI_EINVAL;
		break;
	}

err:
	return ret;
}

static int gnix_av_remove(struct fid_av *av, fi_addr_t *fi_addr, size_t count,
			  uint64_t flags)
{
	struct gnix_fid_av *int_av = NULL;
	int ret = FI_SUCCESS;

	GNIX_TRACE(FI_LOG_AV, "\n");

	if (!av) {
		ret = -FI_EINVAL;
		goto err;
	}

	int_av = container_of(av, struct gnix_fid_av, av_fid);

	if (!int_av) {
		ret = -FI_EINVAL;
		goto err;
	}

	switch (int_av->type) {
	case FI_AV_TABLE:
		ret = table_remove(int_av, fi_addr, count, flags);
		break;
	case FI_AV_MAP:
		ret = map_remove(int_av, fi_addr, count, flags);
		break;
	default:
		ret = -FI_EINVAL;
		break;
	}

err:
	return ret;
}

/*
 * Given an address pointed to by addr, stuff a string into buf representing:
 * device_addr:cdm_id
 * where device_addr and cdm_id are represented in hexadecimal.
 */
static const char *gnix_av_straddr(struct fid_av *av, const void *addr,
				   char *buf, size_t *len)
{
	char int_buf[64];
	int size;

	const struct gnix_address *gnix_addr = addr;

	size =
	    snprintf(int_buf, sizeof(int_buf), "0x%08" PRIx32 ":0x%08" PRIx32,
		     gnix_addr->device_addr, gnix_addr->cdm_id);

	snprintf(buf, *len, "%s", int_buf);
	*len = size + 1;

	return buf;
}

static void __av_destruct(void *obj)
{
	int ret;
	struct gnix_fid_av *av = (struct gnix_fid_av *) obj;
	struct slist_entry *blk_entry;
	struct gnix_av_block *temp;


	if (av->type == FI_AV_TABLE) {
		if (av->table) {
			free(av->table);
		}
	} else if (av->type == FI_AV_MAP) {

		while (!slist_empty(&av->block_list)) {
			blk_entry = slist_remove_head(&av->block_list);
			temp = container_of(blk_entry,
					struct gnix_av_block, slist);
			free(temp->base);
			free(temp);
		}

		ret = _gnix_ht_destroy(av->map_ht);
		if (ret != FI_SUCCESS)
			GNIX_WARN(FI_LOG_AV,
				  "_gnix_ht_destroy failed %d\n",
				  ret);
		free(av->map_ht);
	}
	if (av->valid_entry_vec) {
		free(av->valid_entry_vec);
	} else {
		GNIX_WARN(FI_LOG_AV, "valid_entry_vec is NULL\n");
	}

	free(av);
}

static int gnix_av_close(fid_t fid)
{
	struct gnix_fid_av *av = NULL;
	int ret = FI_SUCCESS;
	int references_held;

	GNIX_TRACE(FI_LOG_AV, "\n");

	if (!fid) {
		ret = -FI_EINVAL;
		goto err;
	}
	av = container_of(fid, struct gnix_fid_av, av_fid.fid);

	references_held = _gnix_ref_put(av);
	if (references_held) {
		GNIX_INFO(FI_LOG_AV, "failed to fully close av due to lingering "
				"references. references=%i av=%p\n",
				references_held, av);
	}

err:
	return ret;
}


/*
 * TODO: Support shared named AVs.
 */
int gnix_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		 struct fid_av **av, void *context)
{
	struct gnix_fid_domain *int_dom = NULL;
	struct gnix_fid_av *int_av = NULL;
	struct gnix_hashtable_attr ht_attr;

	enum fi_av_type type = FI_AV_TABLE;
	size_t count = 128;
	int ret = FI_SUCCESS;

	GNIX_TRACE(FI_LOG_AV, "\n");

	if (!domain) {
		ret = -FI_EINVAL;
		goto err;
	}

	int_dom = container_of(domain, struct gnix_fid_domain, domain_fid);
	if (!int_dom) {
		ret = -FI_EINVAL;
		goto err;
	}

	int_av = calloc(1, sizeof(*int_av));
	if (!int_av) {
		ret = -FI_ENOMEM;
		goto err;
	}

	if (attr) {
		if (gnix_verify_av_attr(attr)) {
			ret = -FI_EINVAL;
			goto cleanup;
		}

		type = attr->type;
		count = attr->count;
	}

	int_av->domain = int_dom;
	int_av->type = type;
	int_av->addrlen = sizeof(struct gnix_address);

	int_av->capacity = count;
	if (type == FI_AV_TABLE) {
		int_av->table = calloc(count,
				       sizeof(struct gnix_av_addr_entry));
		if (!int_av->table) {
			ret = -FI_ENOMEM;
			goto cleanup;
		}
	}

	int_av->valid_entry_vec = calloc(count, sizeof(int));
	if (!int_av->valid_entry_vec) {
		ret = -FI_ENOMEM;
		goto cleanup;
	}

	int_av->av_fid.fid.fclass = FI_CLASS_AV;
	int_av->av_fid.fid.context = context;
	int_av->av_fid.fid.ops = &gnix_fi_av_ops;
	int_av->av_fid.ops = &gnix_av_ops;

	if (type == FI_AV_MAP) {
		int_av->map_ht = calloc(1, sizeof(struct gnix_hashtable));
		if (int_av->map_ht == NULL)
			goto cleanup;

		/*
		 * use same parameters as used for ep vc hash
		 */

		ht_attr.ht_initial_size = int_dom->params.ct_init_size;
		ht_attr.ht_maximum_size = int_dom->params.ct_max_size;
		ht_attr.ht_increase_step = int_dom->params.ct_step;
		ht_attr.ht_increase_type = GNIX_HT_INCREASE_MULT;
		ht_attr.ht_collision_thresh = 500;
		ht_attr.ht_hash_seed = 0xdeadbeefbeefdead;
		ht_attr.ht_internal_locking = 1;
		ht_attr.destructor = NULL;

		ret = _gnix_ht_init(int_av->map_ht,
				    &ht_attr);
		 slist_init(&int_av->block_list);
	}
	_gnix_ref_init(&int_av->ref_cnt, 1, __av_destruct);

	*av = &int_av->av_fid;

	return ret;

cleanup:
	if (int_av->table != NULL)
		free(int_av->table);
	if (int_av->valid_entry_vec != NULL)
		free(int_av->valid_entry_vec);
	free(int_av);
err:
	return ret;
}

/*******************************************************************************
 * FI_OPS_* data structures.
 ******************************************************************************/
static struct fi_ops_av gnix_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = gnix_av_insert,
	.remove = gnix_av_remove,
	.lookup = gnix_av_lookup,
	.straddr = gnix_av_straddr
};

static struct fi_ops gnix_fi_av_ops = {
	.size = sizeof(struct fi_ops),
	.close = gnix_av_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open
};
