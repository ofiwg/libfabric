/*
 * Copyright (c) 2017 Cray Inc. All rights reserved.
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

#include "rdma/fabric.h"
#include "rdma/fi_errno.h"
#include "fi_ext_gni.h"

#include "gnix_auth_key.h"
#include "gnix_hashtable.h"
#include "gnix.h"

#define GNIX_AUTH_KEY_HASHSEED 0xdeadbeef

/* Global data storage for authorization key information */
gnix_hashtable_t __gnix_auth_key_ht;

int _gnix_get_next_reserved_key(struct gnix_auth_key *info)
{
	int reserved_key;
	int offset = info->attr.user_key_limit;
	int retry_limit = 10; /* randomly picked */
	int ret;

	if (!info) {
		GNIX_WARN(FI_LOG_MR, "bad authorization key, key=%p\n",
			info);
		return -FI_EINVAL;
	}

	do {
		reserved_key = _gnix_find_first_zero_bit(&info->prov);
		if (reserved_key >= 0) {
			ret = _gnix_test_and_set_bit(&info->prov, reserved_key);
			if (ret)
				reserved_key = -FI_EAGAIN;
		}
		retry_limit--;
	} while (reserved_key < 0 && retry_limit > 0);

	ret = (reserved_key < 0) ? reserved_key : (offset + reserved_key);

	GNIX_INFO(FI_LOG_DOMAIN, "returning key=%d offset=%d\n", ret, offset);

	return ret;
}

int _gnix_release_reserved_key(struct gnix_auth_key *info, int reserved_key)
{
	int offset = info->attr.user_key_limit;
	int ret;

	if (!info || reserved_key < 0) {
		GNIX_WARN(FI_LOG_MR, "bad authorization key or reserved key,"
			" auth_key=%p requested_key=%d\n",
			info, reserved_key);
		return -FI_EINVAL;
	}

	ret = _gnix_test_and_clear_bit(&info->prov, reserved_key - offset);
	assert(ret == 1);

	return (ret == 1) ? FI_SUCCESS : -FI_EBUSY;
}

int _gnix_auth_key_enable(struct gnix_auth_key *info)
{
	int ret = -FI_EBUSY;

	if (!info) {
		GNIX_WARN(FI_LOG_MR, "bad authorization key, key=%p\n",
			info);
		return -FI_EINVAL;
	}

	fastlock_acquire(&info->lock);
	if (!info->enabled) {
		info->enabled = 1;

		ret = _gnix_alloc_bitmap(&info->prov,
			info->attr.prov_key_limit);
		assert(ret == FI_SUCCESS);

		ret = _gnix_alloc_bitmap(&info->user,
			info->attr.user_key_limit);
		assert(ret == FI_SUCCESS);

		GNIX_INFO(FI_LOG_DOMAIN,
				"set resource limits: pkey=%08x ptag=%d "
				"reserved=%d registration_limit=%d "
				"reserved_keys=%d-%d\n",
				info->cookie,
				info->ptag,
				info->attr.prov_key_limit,
				info->attr.user_key_limit,
				info->attr.user_key_limit,
				(info->attr.prov_key_limit +
				info->attr.user_key_limit - 1));
		ret = FI_SUCCESS;
	}
	fastlock_release(&info->lock);

	if (ret == -FI_EBUSY) {
		GNIX_DEBUG(FI_LOG_MR, "authorization key already enabled, "
			"auth_key=%p\n", info);
	}

	return ret;
}

struct gnix_auth_key *_gnix_auth_key_alloc()
{
	struct gnix_auth_key *auth_key = NULL;

	auth_key = calloc(1, sizeof(*auth_key));
	if (auth_key) {
		fastlock_init(&auth_key->lock);
	} else {
		GNIX_WARN(FI_LOG_MR, "failed to allocate memory for "
			"authorization key\n");
	}

	return auth_key;
}

int _gnix_auth_key_insert(
		uint8_t *auth_key,
		size_t auth_key_size,
		struct gnix_auth_key *to_insert)
{
	int ret;
	gnix_ht_key_t key;
	struct fi_gni_auth_key *gni_auth_key =
		(struct fi_gni_auth_key *) auth_key;

	if (!to_insert) {
		GNIX_WARN(FI_LOG_MR, "bad parameters, to_insert=%p\n",
			to_insert);
		return -FI_EINVAL;
	}

	if (auth_key_size == GNIX_PROV_DEFAULT_AUTH_KEYLEN)
		key = 0;
	else {
		if (!auth_key) {
			GNIX_INFO(FI_LOG_FABRIC, "auth key is null\n");
			return -FI_EINVAL;
		}

		switch (gni_auth_key->type) {
		case GNIX_AKT_RAW:
			key = (gnix_ht_key_t) gni_auth_key->raw.protection_key;
			break;
		default:
			GNIX_INFO(FI_LOG_FABRIC, "unrecognized auth key "
				"type, type=%d\n",
				gni_auth_key->type);
			return -FI_EINVAL;
		}
	}

	ret = _gnix_ht_insert(&__gnix_auth_key_ht, key, to_insert);
	if (ret) {
		GNIX_WARN(FI_LOG_MR, "failed to insert entry, ret=%d\n",
			ret);
	}

	return ret;
}

int _gnix_auth_key_free(struct gnix_auth_key *key)
{
	int ret;

	if (!key) {
		GNIX_WARN(FI_LOG_MR, "bad parameters, key=%p\n", key);
		return -FI_EINVAL;
	}

	fastlock_destroy(&key->lock);

	if (key->enabled) {
		ret = _gnix_free_bitmap(&key->user);
		assert(ret == FI_SUCCESS);
		if (ret) {
			GNIX_ERR(FI_LOG_MR, "failed to free bitmap, bitmap=%p\n",
				&key->user);
		}

		ret = _gnix_free_bitmap(&key->prov);
		assert(ret == FI_SUCCESS);
		if (ret) {
			GNIX_ERR(FI_LOG_MR, "failed to free bitmap, bitmap=%p\n",
				&key->prov);
		}
	}
	key->enabled = 0;

	free(key);

	return FI_SUCCESS;
}

struct gnix_auth_key *
_gnix_auth_key_lookup(uint8_t *auth_key, size_t auth_key_size)
{
	gnix_ht_key_t key;
	struct gnix_auth_key *ptr = NULL;
	struct fi_gni_auth_key *gni_auth_key;

	if (auth_key_size == GNIX_PROV_DEFAULT_AUTH_KEYLEN) {
		key = 0;
	} else {
		if (!auth_key) {
			GNIX_INFO(FI_LOG_FABRIC,
				"null auth key provided, cannot find entry\n");
			return NULL;
		}

		gni_auth_key = (struct fi_gni_auth_key *) auth_key;
		switch (gni_auth_key->type) {
		case GNIX_AKT_RAW:
			key = (gnix_ht_key_t) gni_auth_key->raw.protection_key;
			break;
		default:
			GNIX_INFO(FI_LOG_FABRIC, "unrecognized auth key type, "
				"type=%d\n", gni_auth_key->type);
			return NULL;
		}

	}

	ptr = (struct gnix_auth_key *) _gnix_ht_lookup(
		&__gnix_auth_key_ht, key);

	return ptr;
}

int _gnix_auth_key_subsys_init(void)
{
	int ret = FI_SUCCESS;

	gnix_hashtable_attr_t attr = {
			.ht_initial_size     = 8,
			.ht_maximum_size     = 256,
			.ht_increase_step    = 2,
			.ht_increase_type    = GNIX_HT_INCREASE_MULT,
			.ht_collision_thresh = 400,
			.ht_hash_seed        = 0xcafed00d,
			.ht_internal_locking = 1,
			.destructor          = NULL
	};

	ret = _gnix_ht_init(&__gnix_auth_key_ht, &attr);
	assert(ret == FI_SUCCESS);

	return ret;
}

int _gnix_auth_key_subsys_fini(void)
{
	return FI_SUCCESS;
}

struct gnix_auth_key *_gnix_auth_key_create(
		uint8_t *auth_key,
		size_t auth_key_size)
{
	struct gnix_auth_key *to_insert;
	struct fi_gni_auth_key *gni_auth_key;
	int ret;
	gni_return_t grc;
	uint8_t ptag;
	uint32_t cookie;

	if (auth_key_size == GNIX_PROV_DEFAULT_AUTH_KEYLEN) {
		gnixu_get_rdma_credentials(NULL, &ptag, &cookie);
	} else {
		gni_auth_key = (struct fi_gni_auth_key *) auth_key;
		switch (gni_auth_key->type) {
		case GNIX_AKT_RAW:
			cookie = gni_auth_key->raw.protection_key;
			break;
		default:
			GNIX_WARN(FI_LOG_FABRIC,
				"unrecognized auth key type, type=%d\n",
				gni_auth_key->type);
			return NULL;
		}

		grc = GNI_GetPtag(0, cookie, &ptag);
		if (grc) {
			GNIX_WARN(FI_LOG_FABRIC,
				"could not retrieve ptag, "
				"cookie=%d ret=%d\n", cookie, grc);
			return NULL;
		}
	}

	to_insert = _gnix_auth_key_alloc();
	if (!to_insert) {
		GNIX_WARN(FI_LOG_MR, "failed to allocate memory for "
			"auth key\n");
		return NULL;
	}

	to_insert->attr.prov_key_limit = gnix_default_prov_registration_limit;
	to_insert->attr.user_key_limit = gnix_default_user_registration_limit;
	to_insert->ptag = ptag;
	to_insert->cookie = cookie;

	ret = _gnix_auth_key_insert(auth_key, auth_key_size, to_insert);
	if (ret) {
		GNIX_INFO(FI_LOG_MR, "failed to insert authorization key, "
			"key=%p len=%d to_insert=%p ret=%d\n",
			auth_key, auth_key_size, to_insert, ret);
		_gnix_auth_key_free(to_insert);
		to_insert = NULL;
	}

	return to_insert;
}
