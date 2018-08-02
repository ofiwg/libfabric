/*
 * Copyright (c) 2018 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include "mrail.h"

static char **mrail_addr_strv = NULL;
struct fi_info *mrail_info_vec[MRAIL_MAX_INFO] = {0};
size_t mrail_num_info = 0;
fastlock_t mrail_info_vec_lock;

static inline char **mrail_split_addr_strc(const char *addr_strc)
{
	char **addr_strv = ofi_split_and_alloc(addr_strc, ",", NULL);
	if (!addr_strv) {
		FI_WARN(&mrail_prov, FI_LOG_CORE,
			"Unable to split a FI_ADDR_STRV string\n");
		return NULL;
	}
	return addr_strv;
}

static int mrail_parse_env_vars(void)
{
	char *addr_strc;
	int ret;

	fi_param_define(&mrail_prov, "addr_strc", FI_PARAM_STRING, "List of rail"
			" addresses of format FI_ADDR_STR delimited by comma");
	ret = fi_param_get_str(&mrail_prov, "addr_strc", &addr_strc);
	if (ret) {
		FI_WARN(&mrail_prov, FI_LOG_CORE, "Unable to read "
			"OFI_MRAIL_ADDR_STRC env variable\n");
		return ret;
	}
	mrail_addr_strv = mrail_split_addr_strc(addr_strc);
	if (!mrail_addr_strv)
		return -FI_ENOMEM;
	return 0;
}

int mrail_clear_cache()
{
	size_t i;
	// mrail_info_vec only caches the latest getinfo
	for (i = 0; mrail_info_vec[i]; i++)
		fi_freeinfo(mrail_info_vec[i]);
	memset(mrail_info_vec, 0, sizeof(mrail_info_vec));
	mrail_num_info = 0;
	return 0;
}

int mrail_populate_info_attr(struct fi_info *info, const struct fi_info *hints)
{
	size_t mr_key_size;
	uint32_t num_rails;
	for (num_rails = 0; mrail_addr_strv[num_rails]; ++num_rails)
		;

	mr_key_size = num_rails * sizeof(struct mrail_addr_key);

	free(info->fabric_attr->name);
	free(info->domain_attr->name);

	info->fabric_attr->name = NULL;
	info->domain_attr->name = NULL;

	info->fabric_attr->name = strdup(mrail_info.fabric_attr->name);
	if (!info->fabric_attr->name) {
		return -FI_ENOMEM;
	}
	info->domain_attr->name = strdup(mrail_info.domain_attr->name);
	if (!info->domain_attr->name) {
		return -FI_ENOMEM;
	}
	info->ep_attr->protocol		= mrail_info.ep_attr->protocol;
	info->ep_attr->protocol_version	= mrail_info.ep_attr->protocol_version;
	info->fabric_attr->prov_version	= FI_VERSION(MRAIL_MAJOR_VERSION,
						     MRAIL_MINOR_VERSION);
	info->domain_attr->mr_key_size	= mr_key_size;
	info->domain_attr->mr_mode	|= FI_MR_RAW;

	/* Account for one iovec buffer used for mrail header */
	assert(info->tx_attr->iov_limit);
	info->tx_attr->iov_limit--;

	/* Claiming messages larger than FI_OPT_BUFFERED_LIMIT would consume
	 * a scatter/gather entry for mrail_hdr
	 */
	info->rx_attr->iov_limit--;

	if (info->tx_attr->inject_size < sizeof(struct mrail_hdr))
		info->tx_attr->inject_size = 0;
	else
		info->tx_attr->inject_size -= sizeof(struct mrail_hdr);

	if (hints && hints->tx_attr && (hints->tx_attr->op_flags & FI_COMPLETION))
		info->tx_attr->op_flags |= FI_COMPLETION;

	// TODO set src_addr to FI_ADDR_STRC address

	return 0;
}

// Populates mrail info vec with given list
// mrail_info_vec:
//	[0]: mrail_hdr_prov0 -> info_rail0 -> info_rail1 -> ... // sockets
//	[1]: mrail_hdr_prov1 -> info_rail0 -> info_rail1 -> ... // tcp
//      ...
int mrail_populate_info_vec(struct fi_info *core_info, const struct fi_info *hints)
{
	struct fi_info *tmp_info = NULL;
	int ret = 0;
	size_t i;

	fastlock_acquire(&mrail_info_vec_lock);

	if (mrail_info_vec[0]){
		FI_WARN(&mrail_prov, FI_LOG_CORE, "Overwriting mrail_info_vec cache\n"
                        "Do not use previously givien info structs\n");
		mrail_clear_cache();
	}
	while (core_info) {
		tmp_info = core_info;
		for (i = 0; i < mrail_num_info; i++) {
			if (!strcmp(core_info->fabric_attr->prov_name,
			    mrail_info_vec[i]->fabric_attr->prov_name)) {
				tmp_info = mrail_info_vec[i];
				while (tmp_info->next)
					tmp_info = tmp_info->next;
				tmp_info->next = core_info;
				tmp_info = tmp_info->next;
				core_info = core_info->next;
				tmp_info->next = NULL;
				break;
			}
		}
		if (!mrail_info_vec[i]) {
			assert(mrail_num_info < MRAIL_MAX_INFO);
			mrail_info_vec[i] = core_info;
			core_info = core_info->next;
			mrail_info_vec[i]->next = NULL;
			mrail_num_info++;
		}
	}

	for (i = 0; i < mrail_num_info; i++) {
		tmp_info = fi_dupinfo(mrail_info_vec[i]);
		ret = mrail_populate_info_attr(tmp_info, hints);
		if (ret){
			fi_freeinfo(tmp_info);
			goto out;
		}
		tmp_info->next = mrail_info_vec[i];
		mrail_info_vec[i] = tmp_info;
	}

out:
	fastlock_release(&mrail_info_vec_lock);
	return ret;
}

// Returns duplicated header elements for the info vec
// mrail_info_hdr_prov0 -> mrail_info_hdr_prov1 -> ...
int mrail_get_prov_list(struct fi_info **info)
{
	struct fi_info *tail = NULL;
	size_t i;
	int ret = 0;

	fastlock_acquire(&mrail_info_vec_lock);

	for (i = 0; i < mrail_num_info; i++) {
		if (i == 0) {
			tail = fi_dupinfo(mrail_info_vec[i]);
			*info = tail;
		} else {
			tail->next = fi_dupinfo(mrail_info_vec[i]);
			tail = tail->next;
		}
		if (!tail) {
			fi_freeinfo(*info);
			ret = -FI_ENOMEM;
			goto out;
		}
	}
out:
	fastlock_release(&mrail_info_vec_lock);
	return ret;
}

int mrail_get_core_info(uint32_t version, const char *node, const char *service,
			uint64_t flags, const struct fi_info *hints,
			struct fi_info **core_info)
{
	struct fi_info *core_hints, *info, *fi = NULL;
	size_t i;
	int ret = 0;

	if (!mrail_addr_strv) {
		FI_WARN(&mrail_prov, FI_LOG_FABRIC,
			"OFI_MRAIL_ADDR_STRC env variable not set!\n");
		return -FI_ENODATA;
	}

	core_hints = fi_dupinfo(hints);
	if (!core_hints)
		return -FI_ENOMEM;

	if (!hints) {
		core_hints->mode = MRAIL_PASSTHROUGH_MODES;
		assert(core_hints->domain_attr);
		core_hints->domain_attr->mr_mode = MRAIL_PASSTHROUGH_MR_MODES;
	} else {
		if (hints->tx_attr) {
			if (hints->tx_attr->iov_limit)
				core_hints->tx_attr->iov_limit =
					hints->tx_attr->iov_limit + 1;
			if (hints->rx_attr->iov_limit)
				core_hints->rx_attr->iov_limit =
					hints->rx_attr->iov_limit + 1;
			core_hints->tx_attr->op_flags &= ~FI_COMPLETION;
		}
	}

	core_hints->mode |= FI_BUFFERED_RECV;

	if (!core_hints->fabric_attr)
		core_hints->fabric_attr = calloc(1, sizeof(*core_hints->fabric_attr));

	ret = ofi_exclude_prov_name(&core_hints->fabric_attr->prov_name,
				    mrail_prov.name);
	if (ret)
		goto out;

	for (i = 0; mrail_addr_strv[i]; i++) {
		free(core_hints->src_addr);
		ret = ofi_str_toaddr(mrail_addr_strv[i],
				     &core_hints->addr_format,
				     &core_hints->src_addr,
				     &core_hints->src_addrlen);
		if (ret) {
			FI_WARN(&mrail_prov, FI_LOG_FABRIC,
				"Unable to convert FI_ADDR_STR to device "
				"specific address\n");
			goto err;
		}

		FI_DBG(&mrail_prov, FI_LOG_CORE,
		       "--- Begin fi_getinfo for rail: %zd ---\n", i);

		ret = fi_getinfo(version, node, service, flags,
				 core_hints, &info);

		FI_DBG(&mrail_prov, FI_LOG_CORE,
		       "--- End fi_getinfo for rail: %zd ---\n", i);
		if (ret)
			goto err;
		if (!fi) {
			*core_info = info;
			fi = info;
		} else {
			while (fi->next)
				fi = fi->next;
			fi->next = info;
		}
	}

	goto out;
err:
	if (fi)
		fi_freeinfo(*core_info);
out:
	fi_freeinfo(core_hints);
	return ret;
}

static int mrail_getinfo(uint32_t version, const char *node, const char *service,
			 uint64_t flags, const struct fi_info *hints,
			 struct fi_info **info)
{
	struct fi_info *core_info;
	int ret;

	ret = mrail_get_core_info(version, node, service, flags, hints, &core_info);
	if (ret)
		goto err;

	ret = mrail_populate_info_vec(core_info, hints);
	if (ret)
		goto err;

	ret = mrail_get_prov_list(&core_info);
	if (ret)
		goto err;

	*info = core_info;

	return 0;

err:
	fi_freeinfo(core_info);
	fi_freeinfo(*info);
	return ret;
}

static void mrail_fini(void)
{
	size_t i;
	for (i = 0; i < mrail_num_info; i++)
		fi_freeinfo(mrail_info_vec[i]);
	fastlock_destroy(&mrail_info_vec_lock);
}

struct fi_provider mrail_prov = {
	.name = OFI_UTIL_PREFIX "mrail",
	.version = FI_VERSION(MRAIL_MAJOR_VERSION, MRAIL_MINOR_VERSION),
	.fi_version = FI_VERSION(1, 6),
	.getinfo = mrail_getinfo,
	.fabric = mrail_fabric_open,
	.cleanup = mrail_fini
};

struct util_prov mrail_util_prov = {
	.prov = &mrail_prov,
	.info = &mrail_info,
	.flags = 0,
};

MRAIL_INI
{
	fastlock_init(&mrail_info_vec_lock);
	mrail_parse_env_vars();
	return &mrail_prov;
}

