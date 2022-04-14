/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "efa.h"
#include "efa_prov_info.h"

static int efa_check_hints(uint32_t version, const struct fi_info *hints,
			   const struct fi_info *info)
{
	uint64_t prov_mode;
	size_t size;
	int ret;

	if (hints->caps & ~(info->caps)) {
		EFA_INFO(FI_LOG_CORE, "Unsupported capabilities\n");
		OFI_INFO_CHECK(&efa_prov, info, hints, caps, FI_TYPE_CAPS);
		return -FI_ENODATA;
	}

	prov_mode = ofi_mr_get_prov_mode(version, hints, info);

	if ((hints->mode & prov_mode) != prov_mode) {
		EFA_INFO(FI_LOG_CORE, "Required hints mode bits not set\n");
		OFI_INFO_MODE(&efa_prov, prov_mode, hints->mode);
		return -FI_ENODATA;
	}

	if (hints->fabric_attr) {
		ret = ofi_check_fabric_attr(&efa_prov, info->fabric_attr,
					    hints->fabric_attr);

		if (ret)
			return ret;
	}

	switch (hints->addr_format) {
	case FI_FORMAT_UNSPEC:
	case FI_ADDR_EFA:
		size = EFA_EP_ADDR_LEN;
		break;
	default:
		EFA_INFO(FI_LOG_CORE,
			 "Address format not supported: hints[%u], supported[%u,%u]\n",
			 hints->addr_format, FI_FORMAT_UNSPEC, FI_ADDR_EFA);
		return -FI_ENODATA;
	}

	if (hints->src_addr && hints->src_addrlen < size)
		return -FI_ENODATA;

	if (hints->dest_addr && hints->dest_addrlen < size)
		return -FI_ENODATA;

	if (hints->domain_attr) {
		ret = ofi_check_domain_attr(&efa_prov, version, info->domain_attr, hints);
		if (ret)
			return ret;
	}

	if (hints->ep_attr) {
		ret = ofi_check_ep_attr(&efa_util_prov, info->fabric_attr->api_version, info, hints);
		if (ret)
			return ret;
	}

	if (hints->rx_attr) {
		ret = ofi_check_rx_attr(&efa_prov, info, hints->rx_attr, hints->mode);
		if (ret)
			return ret;
	}

	if (hints->tx_attr) {
		ret = ofi_check_tx_attr(&efa_prov, info->tx_attr, hints->tx_attr, hints->mode);
		if (ret)
			return ret;
	}

	return 0;
}

/**
 * @brief set the desc_addr field of user info
 *
 * @param	node[in]	node from user's call to fi_getinfo()
 * @param	service[in]	service from user's call to fi_getinfo()
 * @param	flags[in]	flags from user's call to fi_getinfo()
 * @param	hints[in]	hints from user's call to fi_getinfo()
 * @param	fi[out]		user_info object to be updated, can be a list of infos
 *
 * @return	0 on success
 * 		negative libfabric error code on failure
 */
int efa_user_info_set_dest_addr(const char *node, const char *service, uint64_t flags,
				const struct fi_info *hints, struct fi_info *fi)
{
	struct efa_ep_addr tmp_addr;
	void *dest_addr = NULL;
	int ret = FI_SUCCESS;
	struct fi_info *cur;

	if (flags & FI_SOURCE) {
		if (hints && hints->dest_addr)
			dest_addr = hints->dest_addr;
	} else {
		if (node || service) {
			ret = efa_str_to_ep_addr(node, service, &tmp_addr);
			if (ret)
				return ret;
			dest_addr = &tmp_addr;
		} else if (hints && hints->dest_addr) {
			dest_addr = hints->dest_addr;
		}
	}

	if (dest_addr) {
		for (cur = fi; cur; cur = cur->next) {
			cur->dest_addr = malloc(EFA_EP_ADDR_LEN);
			if (!cur->dest_addr) {
				for (; fi->dest_addr; fi = fi->next)
					free(fi->dest_addr);
				return -FI_ENOMEM;
			}
			memcpy(cur->dest_addr, dest_addr, EFA_EP_ADDR_LEN);
			cur->dest_addrlen = EFA_EP_ADDR_LEN;
		}
	}
	return ret;
}

/**
 * @brief check the src_addr and desc_addr field of user provided hints
 *
 * @param	node[in]	node from user's call to fi_getinfo()
 * @param	service[in]	service from user's call to fi_getinfo()
 * @param	flags[in]	flags from user's call to fi_getinfo()
 * @param	hints[in]	hints from user's call to fi_getinfo()
 * @param	fi[out]		user_info object to be updated, can be a list of infos
 *
 * @return	0, if hints matches EFA provider's capability
 * 		-FI_ENODATA, otherwise
 */
int efa_user_info_check_hints_addr(const char *node, const char *service,
				   uint64_t flags, const struct fi_info *hints)
{
	if (!(flags & FI_SOURCE) && hints && hints->src_addr &&
	    hints->src_addrlen != EFA_EP_ADDR_LEN)
		return -FI_ENODATA;

	if (((!node && !service) || (flags & FI_SOURCE)) &&
	    hints && hints->dest_addr &&
	    hints->dest_addrlen != EFA_EP_ADDR_LEN)
		return -FI_ENODATA;

	return 0;
}

static int efa_get_matching_info(uint32_t version, const char *node, uint64_t flags,
				 const struct fi_info *hints, struct fi_info **info)
{
	const struct fi_info *check_info;
	struct fi_info *fi, *tail;
	int ret;

	*info = tail = NULL;

	for (check_info = efa_util_prov.info; check_info; check_info = check_info->next) {
		ret = efa_prov_info_compare_src_addr(node, flags, hints, check_info);
		if (ret)
			continue;

		EFA_INFO(FI_LOG_FABRIC, "found match for interface %s %s\n", node, check_info->fabric_attr->name);
		if (hints) {
			ret = efa_check_hints(version, hints, check_info);
			if (ret)
				continue;
		}

		fi = fi_dupinfo(check_info);
		if (!fi) {
			ret = -FI_ENOMEM;
			goto err_free_info;
		}

		fi->fabric_attr->api_version = version;

		if (!*info)
			*info = fi;
		else
			tail->next = fi;
		tail = fi;
	}

	if (!*info)
		return -FI_ENODATA;

	return 0;

err_free_info:
	fi_freeinfo(*info);
	*info = NULL;
	return ret;
}

int efa_getinfo(uint32_t version, const char *node, const char *service,
		uint64_t flags, const struct fi_info *hints, struct fi_info **info)
{
	int ret;

	ret = efa_user_info_check_hints_addr(node, service, flags, hints);
	if (ret)
		goto out;

	ret = efa_get_matching_info(version, node, flags, hints, info);
	if (ret)
		goto out;

	ret = efa_user_info_set_dest_addr(node, service, flags, hints, *info);
	if (ret)
		goto out;

	ofi_alter_info(*info, hints, version);

out:
	if (!ret || ret == -FI_ENOMEM || ret == -FI_ENODEV) {
		return ret;
	} else {
		fi_freeinfo(*info);
		*info = NULL;
		return -FI_ENODATA;
	}
}

