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
 *	- Redistributions of source code must retain the above
 *	  copyright notice, this list of conditions and the following
 *	  disclaimer.
 *
 *	- Redistributions in binary form must reproduce the above
 *	  copyright notice, this list of conditions and the following
 *	  disclaimer in the documentation and/or other materials
 *	  provided with the distribution.
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

#include <shared/ofi_str.h>

#include "mrail.h"

static char **mrail_addr_strv = NULL;
/* Not thread safe */
struct fi_info *mrail_info_vec[MRAIL_MAX_INFO] = {0};
size_t mrail_num_info = 0;

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
		core_hints->mode = MRAIL_PASSTHRU_MODES;
		assert(core_hints->domain_attr);
		core_hints->domain_attr->mr_mode = MRAIL_PASSTHRU_MR_MODES;
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
	core_hints->caps |= FI_SOURCE;

	if (!core_hints->fabric_attr) {
		core_hints->fabric_attr = calloc(1, sizeof(*core_hints->fabric_attr));
		if (!core_hints->fabric_attr) {
			ret = -FI_ENOMEM;
			goto out;
		}
	}

	if (!core_hints->domain_attr) {
		core_hints->domain_attr = calloc(1, sizeof(*core_hints->domain_attr));
		if (!core_hints->domain_attr) {
			ret = -FI_ENOMEM;
			goto out;
		}
	}
	core_hints->domain_attr->av_type = FI_AV_TABLE;

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

		ret = fi_getinfo(version, NULL, NULL, 0, core_hints, &info);

		FI_DBG(&mrail_prov, FI_LOG_CORE,
		       "--- End fi_getinfo for rail: %zd ---\n", i);
		if (ret)
			goto err;

		if (!fi)
			*core_info = info;
		else
			fi->next = info;
		fi = info;

		/* We only want the first fi_info entry per rail */
		if (info->next) {
			fi_freeinfo(info->next);
			info->next = NULL;
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

static void mrail_adjust_info(struct fi_info *info, const struct fi_info *hints)
{
	info->mode &= ~FI_BUFFERED_RECV;

	if (!hints)
		return;

	if (hints->domain_attr) {
		if (hints->domain_attr->av_type)
			info->domain_attr->av_type = hints->domain_attr->av_type;
	}

	if (hints->tx_attr) {
		if (hints->tx_attr->op_flags & FI_COMPLETION)
			info->tx_attr->op_flags |= FI_COMPLETION;
	}
}

static struct fi_info *mrail_get_prefix_info(struct fi_info *core_info)
{
	struct fi_info *fi;
	uint32_t num_rails;

	for (fi = core_info, num_rails = 0; fi; fi = fi->next, ++num_rails)
		;

	fi = fi_dupinfo(core_info);
	if (!fi)
		return NULL;

	free(fi->fabric_attr->name);
	free(fi->domain_attr->name);

	fi->fabric_attr->name = NULL;
	fi->domain_attr->name = NULL;

	fi->fabric_attr->name = strdup(mrail_info.fabric_attr->name);
	if (!fi->fabric_attr->name)
		goto err;

	fi->domain_attr->name = strdup(mrail_info.domain_attr->name);
	if (!fi->domain_attr->name)
		goto err;

	fi->ep_attr->protocol		= mrail_info.ep_attr->protocol;
	fi->ep_attr->protocol_version	= mrail_info.ep_attr->protocol_version;
	fi->fabric_attr->prov_version	= FI_VERSION(MRAIL_MAJOR_VERSION,
						     MRAIL_MINOR_VERSION);
	fi->domain_attr->mr_key_size	= (num_rails *
					   sizeof(struct mrail_addr_key));
	fi->domain_attr->mr_mode	|= FI_MR_RAW;

	/* Account for one iovec buffer used for mrail header */
	assert(fi->tx_attr->iov_limit);
	fi->tx_attr->iov_limit--;

	/* Claiming messages larger than FI_OPT_BUFFERED_LIMIT would consume
	 * a scatter/gather entry for mrail_hdr */
	fi->rx_attr->iov_limit--;

	if (fi->tx_attr->inject_size < sizeof(struct mrail_hdr))
		fi->tx_attr->inject_size = 0;
	else
		fi->tx_attr->inject_size -= sizeof(struct mrail_hdr);
	return fi;
err:
	fi_freeinfo(fi);
	return NULL;
}

static int mrail_check_modes(const struct fi_info *hints)
{
	if (!hints)
		return 0;

	if (hints->mode & ~MRAIL_PASSTHRU_MODES) {
		FI_INFO(&mrail_prov, FI_LOG_CORE,
			"Unable to pass through given modes: %s\n",
			fi_tostr(&hints->mode, FI_TYPE_MODE));
		return -FI_ENODATA;
	}

	if (hints->domain_attr &&
	    (hints->domain_attr->mr_mode & ~MRAIL_PASSTHRU_MR_MODES)) {
		FI_INFO(&mrail_prov, FI_LOG_CORE,
			"Unable to pass through given MR modes: %s\n",
			fi_tostr(&hints->domain_attr->mr_mode, FI_TYPE_MR_MODE));
		return -FI_ENODATA;
	}
	return 0;
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

	ret = mrail_check_modes(hints);
	if (ret)
		return ret;

	ret = mrail_get_core_info(version, node, service, flags, hints, info);
	if (ret)
		return ret;

	fi = mrail_get_prefix_info(*info);
	if (!fi) {
		ret = -FI_ENOMEM;
		goto err1;
	}

	mrail_adjust_info(fi, hints);

	// TODO set src_addr to FI_ADDR_STRC address
	fi->next = *info;
	*info = fi;

	mrail_info_vec[mrail_num_info] = mrail_dupinfo(*info);
	if (!mrail_info_vec[mrail_num_info])
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
	.name = OFI_UTIL_PREFIX "mrail",
	.version = FI_VERSION(MRAIL_MAJOR_VERSION, MRAIL_MINOR_VERSION),
	.fi_version = FI_VERSION(1, 7),
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

