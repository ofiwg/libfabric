/*
 * Copyright (c) 2018 Intel Corporation, Inc.  All rights reserved.
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
/* Not thread safe */
struct fi_info *mrail_info_vec[MRAIL_MAX_INFO] = {0};
size_t mrail_num_info = 0;

static inline char **mrail_split_addr_strc(const char *addr_strc)
{
	char **addr_strv = ofi_split_and_alloc(addr_strc, ",");
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
	if (!(mrail_addr_strv = mrail_split_addr_strc(addr_strc)))
		return -FI_ENOMEM;
	return 0;
}

int mrail_get_core_info(uint32_t version, const char *node, const char *service,
			uint64_t flags, const struct fi_info *hints,
			struct fi_info **core_info)
{
	struct fi_info *core_hints, *info, *fi = NULL;
	const char *core_name;
	size_t len, i;
	int ret = 0;

	if (!mrail_addr_strv) {
		FI_WARN(&mrail_prov, FI_LOG_FABRIC,
			"OFI_MRAIL_ADDR_STRC env variable not set!\n");
		return -FI_ENODATA;
	}

	if (!(core_hints = fi_dupinfo(hints)))
		return -FI_ENOMEM;

	if (core_hints->fabric_attr)
		free(core_hints->fabric_attr->prov_name);

	if (hints && hints->fabric_attr && hints->fabric_attr->prov_name) {
		core_name = ofi_core_name(hints->fabric_attr->prov_name, &len);
		if (core_name) {
			core_hints->fabric_attr->prov_name = strndup(core_name,
								     len);
			if (!core_hints->fabric_attr->prov_name) {
				FI_WARN(&mrail_prov, FI_LOG_FABRIC,
					"Unable to alloc prov name\n");
				ret = -FI_ENOMEM;
				goto out;
			}
		}
	} else {
		core_hints->fabric_attr->prov_name = NULL;
	}

	flags |= OFI_CORE_PROV_ONLY;

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

		ret = fi_getinfo(version, node, service, flags,
				 core_hints, &info);
		if (ret)
			goto err;

		assert(!info->next);

		if (!fi)
			*core_info = info;
		else
			fi->next = info;
		fi = info;
	}
	goto out;
err:
	if (fi)
		fi_freeinfo(*core_info);
out:
	fi_freeinfo(core_hints);
	return ret;
}

static struct fi_info *mrail_dupinfo(const struct fi_info *info)
{
	struct fi_info *dup, *fi, *head = NULL;

	while (info) {
		if (!(dup = fi_dupinfo(info)))
			goto err;
		if (!head)
			head = fi = dup;
		else
			fi->next = dup;
		fi = dup;
		info = info->next;
	}
	return head;
err:
	fi_freeinfo(head);
	return NULL;
}

static int mrail_getinfo(uint32_t version, const char *node, const char *service,
			 uint64_t flags, const struct fi_info *hints,
			 struct fi_info **info)
{
	struct fi_info *fi;
	int ret;

	if (mrail_num_info >= MRAIL_MAX_INFO) {
		FI_WARN(&mrail_prov, FI_LOG_CORE,
			"Max mrail_num_info reached\n");
		assert(0);
		return -FI_ENODATA;
	}

	ret = mrail_get_core_info(version, node, service, flags, hints, info);
	if (ret)
		return ret;

	if (!(fi = fi_dupinfo(*info))) {
		ret = -FI_ENOMEM;
		goto err1;
	}

	free(fi->fabric_attr->name);
	free(fi->fabric_attr->prov_name);
	free(fi->domain_attr->name);

	fi->fabric_attr->name = NULL;
	fi->fabric_attr->prov_name = NULL;
	fi->domain_attr->name = NULL;

	if (!(fi->fabric_attr->name = strdup(mrail_info.fabric_attr->name))) {
		ret = -FI_ENOMEM;
		goto err2;
	}
	if (!(fi->domain_attr->name = strdup(mrail_info.domain_attr->name))) {
		ret = -FI_ENOMEM;
		goto err2;
	}
	fi->ep_attr->protocol 		= mrail_info.ep_attr->protocol;
	fi->ep_attr->protocol_version 	= mrail_info.ep_attr->protocol_version;
	fi->fabric_attr->prov_version	= FI_VERSION(MRAIL_MAJOR_VERSION,
						     MRAIL_MINOR_VERSION);
	// TODO set src_addr to FI_ADDR_STRC address
	fi->next = *info;
	*info = fi;

	if (!(mrail_info_vec[mrail_num_info] = mrail_dupinfo(*info)))
		goto err2;

	mrail_num_info++;

	return 0;
err2:
	fi_freeinfo(fi);
err1:
	fi_freeinfo(*info);
	return ret;
}

static void mrail_fini(void)
{
	size_t i;
	for (i = 0; i < mrail_num_info; i++)
		fi_freeinfo(mrail_info_vec[i]);
}

struct fi_provider mrail_prov = {
	.name = "ofi_mrail",
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
	mrail_parse_env_vars();
	return &mrail_prov;
}

