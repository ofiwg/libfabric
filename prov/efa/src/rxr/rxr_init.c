/*
 * Copyright (c) 2019 Amazon.com, Inc. or its affiliates.
 * All rights reserved.
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

#include <rdma/fi_errno.h>

#include <ofi_prov.h>
#include "rxr.h"
#include "efa.h"

struct fi_info *shm_info;

struct fi_provider *lower_efa_prov;
struct efa_ep_addr *local_efa_addr;


struct rxr_env rxr_env = {
	.rx_window_size	= RXR_DEF_MAX_RX_WINDOW,
	.tx_max_credits = RXR_DEF_MAX_TX_CREDITS,
	.tx_min_credits = RXR_DEF_MIN_TX_CREDITS,
	.tx_queue_size = 0,
	.enable_sas_ordering = 1,
	.enable_shm_transfer = 1,
	.shm_av_size = 128,
	.shm_max_medium_size = 4096,
	.recvwin_size = RXR_RECVWIN_SIZE,
	.cq_size = RXR_DEF_CQ_SIZE,
	.max_memcpy_size = 4096,
	.mtu_size = 0,
	.tx_size = 0,
	.rx_size = 0,
	.tx_iov_limit = 0,
	.rx_iov_limit = 0,
	.rx_copy_unexp = 1,
	.rx_copy_ooo = 1,
	.max_timeout = RXR_DEF_RNR_MAX_TIMEOUT,
	.timeout_interval = 0, /* 0 is random timeout */
};

static void rxr_init_env(void)
{
	fi_param_get_int(&rxr_prov, "rx_window_size", &rxr_env.rx_window_size);
	fi_param_get_int(&rxr_prov, "tx_max_credits", &rxr_env.tx_max_credits);
	fi_param_get_int(&rxr_prov, "tx_min_credits", &rxr_env.tx_min_credits);
	fi_param_get_int(&rxr_prov, "tx_queue_size", &rxr_env.tx_queue_size);
	fi_param_get_int(&rxr_prov, "enable_sas_ordering", &rxr_env.enable_sas_ordering);
	fi_param_get_int(&rxr_prov, "enable_shm_transfer", &rxr_env.enable_shm_transfer);
	fi_param_get_int(&rxr_prov, "shm_av_size", &rxr_env.shm_av_size);
	fi_param_get_int(&rxr_prov, "shm_max_medium_size", &rxr_env.shm_max_medium_size);
	fi_param_get_int(&rxr_prov, "recvwin_size", &rxr_env.recvwin_size);
	fi_param_get_int(&rxr_prov, "cq_size", &rxr_env.cq_size);
	fi_param_get_size_t(&rxr_prov, "max_memcpy_size",
			    &rxr_env.max_memcpy_size);
	fi_param_get_bool(&rxr_prov, "mr_cache_enable",
			  &efa_mr_cache_enable);
	fi_param_get_bool(&rxr_prov, "mr_cache_merge_regions",
			  &efa_mr_cache_merge_regions);
	fi_param_get_size_t(&rxr_prov, "mr_max_cached_count",
			    &efa_mr_max_cached_count);
	fi_param_get_size_t(&rxr_prov, "mr_max_cached_size",
			    &efa_mr_max_cached_size);
	fi_param_get_size_t(&rxr_prov, "mtu_size",
			    &rxr_env.mtu_size);
	fi_param_get_size_t(&rxr_prov, "tx_size", &rxr_env.tx_size);
	fi_param_get_size_t(&rxr_prov, "rx_size", &rxr_env.rx_size);
	fi_param_get_size_t(&rxr_prov, "tx_iov_limit", &rxr_env.tx_iov_limit);
	fi_param_get_size_t(&rxr_prov, "rx_iov_limit", &rxr_env.rx_iov_limit);
	fi_param_get_bool(&rxr_prov, "rx_copy_unexp",
			  &rxr_env.rx_copy_unexp);
	fi_param_get_bool(&rxr_prov, "rx_copy_ooo",
			  &rxr_env.rx_copy_ooo);
	fi_param_get_int(&rxr_prov, "max_timeout", &rxr_env.max_timeout);
	fi_param_get_int(&rxr_prov, "timeout_interval",
			 &rxr_env.timeout_interval);
}

/*
 * Stringify the void *addr to a string smr_name formatted as `gid_qpn`, which
 * will be used to insert into shm provider's AV. Then shm uses smr_name as
 * ep_name to create the shared memory region.
 *
 * The IPv6 address length is 46, but the max supported name length for shm is 32.
 * The string `gid_qpn` could be truncated during snprintf.
 * The current way works because the IPv6 addresses starting with FE in hexadecimals represent
 * link local IPv6 addresses, which has reserved first 64 bits (FE80::/64).
 * e.g., fe80:0000:0000:0000:0436:29ff:fe8e:ceaa -> fe80::436:29ff:fe8e:ceaa
 * And the length of string `gid_qpn` (fe80::436:29ff:fe8e:ceaa_***) will not exceed 32.
 * If the address is NOT link local, we need to think another reasonable way to
 * generate the string.
 */
int rxr_ep_efa_addr_to_str(const void *addr, char *smr_name)
{
	char gid[INET6_ADDRSTRLEN] = { 0 };
	uint16_t qpn;
	int ret;

	if (!inet_ntop(AF_INET6, ((struct efa_ep_addr *)addr)->raw, gid, INET6_ADDRSTRLEN)) {
		printf("Failed to get current EFA's GID, errno: %d\n", errno);
		return 0;
	}
	qpn = ((struct efa_ep_addr *)addr)->qpn;

	ret = snprintf(smr_name, RXR_MAX_NAME_LENGTH, "%s_%d", gid, qpn);

	return (ret <= 0) ? ret : FI_SUCCESS;
}

void rxr_info_to_core_mr_modes(uint32_t version,
			       const struct fi_info *hints,
			       struct fi_info *core_info)
{
	if (hints && hints->domain_attr &&
	    (hints->domain_attr->mr_mode & (FI_MR_SCALABLE | FI_MR_BASIC))) {
		core_info->mode = FI_LOCAL_MR | FI_MR_ALLOCATED;
		core_info->domain_attr->mr_mode = hints->domain_attr->mr_mode;
	} else if (FI_VERSION_LT(version, FI_VERSION(1, 5))) {
		core_info->mode |= FI_LOCAL_MR | FI_MR_ALLOCATED;
		core_info->domain_attr->mr_mode = FI_MR_UNSPEC;
	} else {
		core_info->domain_attr->mr_mode |=
			FI_MR_LOCAL | FI_MR_ALLOCATED;
		if (!hints)
			core_info->domain_attr->mr_mode |= OFI_MR_BASIC_MAP;
		else if (hints->domain_attr)
			core_info->domain_attr->mr_mode |=
				hints->domain_attr->mr_mode & OFI_MR_BASIC_MAP;
	}
}

static int rxr_copy_attr(const struct fi_info *info, struct fi_info *dup)
{
	if (info->src_addr) {
		dup->src_addrlen = info->src_addrlen;
		dup->src_addr = mem_dup(info->src_addr,
					info->src_addrlen);
		if (!dup->src_addr)
			return -FI_ENOMEM;
	}
	if (info->dest_addr) {
		dup->dest_addrlen = info->dest_addrlen;
		dup->dest_addr = mem_dup(info->dest_addr,
					 info->dest_addrlen);
		if (!dup->dest_addr)
			return -FI_ENOMEM;
	}
	if (info->fabric_attr) {
		if (info->fabric_attr->name) {
			dup->fabric_attr->name =
				strdup(info->fabric_attr->name);
			if (!dup->fabric_attr->name)
				return -FI_ENOMEM;
		}
	}
	if (info->domain_attr) {
		if (info->domain_attr->name) {
			dup->domain_attr->name =
				strdup(info->domain_attr->name);
			if (!dup->domain_attr->name)
				return -FI_ENOMEM;
		}
	}
	if (info->nic) {
		dup->nic = ofi_nic_dup(info->nic);
		if (!dup->nic)
			return -FI_ENOMEM;
	}
	return 0;
}

static int rxr_info_to_core(uint32_t version, const struct fi_info *rxr_info,
			    struct fi_info **core_info)
{
	int ret = 0;
	*core_info = fi_allocinfo();
	if (!*core_info)
		return -FI_ENOMEM;

	rxr_info_to_core_mr_modes(version, rxr_info, *core_info);
	(*core_info)->caps = FI_MSG;
	(*core_info)->ep_attr->type = FI_EP_RDM;
	(*core_info)->tx_attr->op_flags = FI_TRANSMIT_COMPLETE;

	(*core_info)->addr_format = FI_ADDR_EFA;

	/*
	 * Skip copying address, domain, fabric info.
	 */
	if (!rxr_info)
		return 0;

	ret = rxr_copy_attr(rxr_info, *core_info);
	if (ret)
		fi_freeinfo(*core_info);
	return ret;
}

/* Explicitly set all necessary bits before calling shm provider's getinfo function */
void rxr_set_shm_hints(struct fi_info *shm_hints)
{
	shm_hints->caps = FI_MSG | FI_TAGGED | FI_RECV | FI_SEND | FI_READ
			   | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE
			   | FI_MULTI_RECV | FI_RMA;
	shm_hints->domain_attr->av_type = FI_AV_TABLE;
	shm_hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR;
	shm_hints->domain_attr->caps |= FI_LOCAL_COMM;
	shm_hints->tx_attr->msg_order = FI_ORDER_SAS;
	shm_hints->rx_attr->msg_order = FI_ORDER_SAS;
	shm_hints->fabric_attr->name = strdup("shm");
	shm_hints->fabric_attr->prov_name = strdup("shm");
	shm_hints->ep_attr->type = FI_EP_RDM;
}

/* Pass tx/rx attr that user specifies down to core provider */
void rxr_reset_rx_tx_to_core(const struct fi_info *user_info,
			     struct fi_info *core_info)
{
	/* rx attr */
	core_info->rx_attr->total_buffered_recv =
		user_info->rx_attr->total_buffered_recv < core_info->rx_attr->total_buffered_recv ?
		user_info->rx_attr->total_buffered_recv : core_info->rx_attr->total_buffered_recv;
	core_info->rx_attr->size =
		user_info->rx_attr->size < core_info->rx_attr->size ?
		user_info->rx_attr->size : core_info->rx_attr->size;
	core_info->rx_attr->iov_limit =
		user_info->rx_attr->iov_limit < core_info->rx_attr->iov_limit ?
		user_info->rx_attr->iov_limit : core_info->rx_attr->iov_limit;
	/* tx attr */
	core_info->tx_attr->inject_size =
		user_info->tx_attr->inject_size < core_info->tx_attr->inject_size ?
		user_info->tx_attr->inject_size : core_info->tx_attr->inject_size;
	core_info->tx_attr->size =
		user_info->tx_attr->size < core_info->tx_attr->size ?
		user_info->tx_attr->size : core_info->tx_attr->size;
	core_info->tx_attr->iov_limit =
		user_info->tx_attr->iov_limit < core_info->tx_attr->iov_limit ?
		user_info->tx_attr->iov_limit : core_info->tx_attr->iov_limit;
}

void rxr_set_rx_tx_size(struct fi_info *info,
			const struct fi_info *core_info)
{
	if (rxr_env.tx_size > 0)
		info->tx_attr->size = rxr_env.tx_size;
	else
		info->tx_attr->size = core_info->tx_attr->size;

	if (rxr_env.rx_size > 0)
		info->rx_attr->size = rxr_env.rx_size;
	else
		info->rx_attr->size = core_info->rx_attr->size;

	if (rxr_env.tx_iov_limit > 0)
		info->tx_attr->iov_limit = rxr_env.tx_iov_limit;

	if (rxr_env.rx_iov_limit > 0)
		info->rx_attr->iov_limit = rxr_env.rx_iov_limit;
}

static int rxr_dgram_info_to_rxr(uint32_t version,
				 const struct fi_info *core_info,
				 struct fi_info *info) {
	rxr_set_rx_tx_size(info, core_info);
	return 0;
}

static int rxr_info_to_rxr(uint32_t version, const struct fi_info *core_info,
			   struct fi_info *info, const struct fi_info *hints)
{
	info->caps = rxr_info.caps;
	info->mode = rxr_info.mode;

	*info->tx_attr = *rxr_info.tx_attr;
	*info->rx_attr = *rxr_info.rx_attr;
	*info->ep_attr = *rxr_info.ep_attr;
	*info->domain_attr = *rxr_info.domain_attr;

	info->tx_attr->inject_size =
		core_info->tx_attr->inject_size > RXR_CTRL_HDR_SIZE_NO_CQ ?
		core_info->tx_attr->inject_size - RXR_CTRL_HDR_SIZE_NO_CQ
		: 0;
	rxr_info.tx_attr->inject_size = info->tx_attr->inject_size;

	info->addr_format = core_info->addr_format;
	info->domain_attr->ep_cnt = core_info->domain_attr->ep_cnt;
	info->domain_attr->cq_cnt = core_info->domain_attr->cq_cnt;
	info->domain_attr->mr_key_size = core_info->domain_attr->mr_key_size;

	/*
	 * Handle user-provided hints and adapt the info object passed back up
	 * based on EFA-specific constraints.
	 */
	if (hints) {
		/* Disable packet reordering if the app doesn't need it */
		if (hints->tx_attr)
			if (!(hints->tx_attr->msg_order & FI_ORDER_SAS))
				rxr_env.enable_sas_ordering = 0;

		/* We only support manual progress for RMA operations */
		if (hints->caps & FI_RMA) {
			info->domain_attr->control_progress = FI_PROGRESS_MANUAL;
			info->domain_attr->data_progress = FI_PROGRESS_MANUAL;
		}

		/* Use a table for AV if the app has no strong requirement */
		if (!hints->domain_attr || hints->domain_attr->av_type == FI_AV_UNSPEC)
			info->domain_attr->av_type = FI_AV_TABLE;
	}

	rxr_set_rx_tx_size(info, core_info);
	return 0;
}

/*
 * For the RDM endpoint, translate user hints to hints for the lower layer and
 * call getinfo on the lower layer.
 */
int rxr_get_lower_rdm_info(uint32_t version, const char *node,
			   const char *service, uint64_t flags,
			   const struct util_prov *util_prov,
			   const struct fi_info *util_hints,
			   struct fi_info **core_info)
{
	struct fi_info *core_hints = NULL;
	int ret;

	ret = ofi_prov_check_info(&rxr_util_prov, version, util_hints);
	if (ret)
		return ret;

	ret = rxr_info_to_core(version, util_hints, &core_hints);
	if (ret)
		return ret;

	ret = lower_efa_prov->getinfo(version, node, service, flags,
				      core_hints, core_info);
	fi_freeinfo(core_hints);
	return ret;
}

/*
 * Call getinfo on lower efa provider to get all locally qualified fi_info
 * structure, then store the corresponding efa nic GIDs
 */
int rxr_get_local_gids(struct fi_provider *lower_efa_prov)
{
	struct fi_info *core_info, *cur;
	struct efa_ep_addr *cur_efa_addr;
	int ret;

	cur_efa_addr = local_efa_addr = NULL;
	core_info = cur = NULL;

	ret = lower_efa_prov->getinfo(rxr_prov.fi_version, NULL, NULL, 0, NULL, &core_info);
	if (ret)
		return ret;

	local_efa_addr = (struct efa_ep_addr *)malloc(sizeof(struct efa_ep_addr));
	if (!local_efa_addr) {
		ret = -FI_ENOMEM;
		goto out;
	}
	local_efa_addr->next = NULL;

	cur_efa_addr = local_efa_addr;
	for (cur = core_info; cur; cur = cur->next) {
		memcpy(cur_efa_addr->raw, ((struct efa_ep_addr *)cur->src_addr)->raw, 16);
		if (cur->next) {
			cur_efa_addr->next = (struct efa_ep_addr *)malloc(sizeof(struct efa_ep_addr));
			if (!cur_efa_addr->next) {
				ret = -FI_ENOMEM;
				goto out;
			}
			cur_efa_addr = cur_efa_addr->next;
			cur_efa_addr->next = NULL;
		}
	}

out:
	fi_freeinfo(core_info);
	return ret;
}

static int rxr_dgram_getinfo(uint32_t version, const char *node,
			     const char *service, uint64_t flags,
			     const struct fi_info *hints, struct fi_info **info,
			     struct fi_info **tail)
{
	struct fi_info *core_info, *util_info, *cur;
	int ret;

	core_info = NULL;

	ret = lower_efa_prov->getinfo(version, node, service,
				      flags, hints, &core_info);

	if (ret)
		return ret;

	ret = -FI_ENODATA;

	for (cur = core_info; cur; cur = cur->next) {
		/* Skip non DGRAM info structs */
		if (cur->ep_attr->type != FI_EP_DGRAM)
			continue;

		ret = 0;

		util_info = fi_dupinfo(cur);
		if (!util_info) {
			ret = -FI_ENOMEM;
			fi_freeinfo(*info);
			goto out;
		}

		rxr_dgram_info_to_rxr(version, cur, util_info);

		if (!*info)
			*info = util_info;
		else
			(*tail)->next = util_info;
		*tail = util_info;
	}

out:
	fi_freeinfo(core_info);
	return ret;
}

static int rxr_getinfo(uint32_t version, const char *node,
		       const char *service, uint64_t flags,
		       const struct fi_info *hints, struct fi_info **info)
{
	struct fi_info *core_info, *util_info, *cur, *tail;
	struct fi_info *shm_hints;
	int ret;

	*info = tail = core_info = NULL;

	if (hints && hints->ep_attr && hints->ep_attr->type == FI_EP_DGRAM)
		goto dgram_info;

	ret = rxr_get_lower_rdm_info(version, node, service, flags,
				     &rxr_util_prov, hints, &core_info);

	if (ret == -FI_ENODATA)
		goto dgram_info;

	if (ret)
		return ret;

	for (cur = core_info; cur; cur = cur->next) {
		util_info = fi_allocinfo();
		if (!util_info) {
			ret = -FI_ENOMEM;
			fi_freeinfo(*info);
			goto out;
		}

		rxr_info_to_rxr(version, cur, util_info, hints);

		ret = rxr_copy_attr(cur, util_info);
		if (ret) {
			fi_freeinfo(util_info);
			fi_freeinfo(*info);
			goto out;
		}


		ofi_alter_info(util_info, hints, version);
		if (!*info)
			*info = util_info;
		else
			tail->next = util_info;
		tail = util_info;
	}

dgram_info:
	ret = rxr_dgram_getinfo(version, node, service, flags, hints, info,
				&tail);
	/*
	 * Ignore dgram getinfo return code if rdm getinfo was successful.
	 */
	if (ret == -FI_ENODATA && *info)
		ret = 0;

	if (rxr_env.enable_shm_transfer && !shm_info) {
		shm_info = fi_allocinfo();
		shm_hints = fi_allocinfo();
		rxr_set_shm_hints(shm_hints);
		ret = fi_getinfo(FI_VERSION(1, 8), NULL, NULL, 0, shm_hints, &shm_info);
		fi_freeinfo(shm_hints);
		if (ret) {
			FI_WARN(&rxr_prov, FI_LOG_CORE, "Failed to get shm provider's info.\n");
			goto out;
		}
		assert(!strcmp(shm_info->fabric_attr->name, "shm"));
	}
out:
	fi_freeinfo(core_info);
	return ret;
}

static void rxr_fini(void)
{
	struct efa_ep_addr *cur;

	if (lower_efa_prov)
		lower_efa_prov->cleanup();

	if (rxr_env.enable_shm_transfer) {
		/* Cleanup all local efa nic GIDs */
		while (local_efa_addr) {
			cur = local_efa_addr;
			local_efa_addr = local_efa_addr->next;
			free(cur);
		}
		if (shm_info)
			fi_freeinfo(shm_info);
	}
}

struct fi_provider rxr_prov = {
	.name = "efa",
	.version = FI_VERSION(RXR_MAJOR_VERSION, RXR_MINOR_VERSION),
	.fi_version = RXR_FI_VERSION,
	.getinfo = rxr_getinfo,
	.fabric = rxr_fabric,
	.cleanup = rxr_fini
};

EFA_INI
{
	fi_param_define(&rxr_prov, "rx_window_size", FI_PARAM_INT,
			"Defines the maximum window size that a receiver will return for matched large messages. (Default: 128).");
	fi_param_define(&rxr_prov, "tx_max_credits", FI_PARAM_INT,
			"Defines the maximum number of credits a sender requests from a receiver (Default: 64).");
	fi_param_define(&rxr_prov, "tx_min_credits", FI_PARAM_INT,
			"Defines the minimum number of credits a sender requests from a receiver (Default: 32).");
	fi_param_define(&rxr_prov, "tx_queue_size", FI_PARAM_INT,
			"Defines the maximum number of unacknowledged sends with the NIC.");
	fi_param_define(&rxr_prov, "enable_sas_ordering", FI_PARAM_INT,
			"Enable packet reordering for the RDM endpoint. This is always enabled when FI_ORDER_SAS is requested by the application. (Default: 1)");
	fi_param_define(&rxr_prov, "enable_shm_transfer", FI_PARAM_INT,
			"Enable using SHM provider to provide the communication between processes on the same system. (Default: 1)");
	fi_param_define(&rxr_prov, "shm_av_size", FI_PARAM_INT,
			"Defines the maximum number of entries in SHM provider's address vector (Default 128).");
	fi_param_define(&rxr_prov, "shm_max_medium_size", FI_PARAM_INT,
			"Defines the switch point between small/medium message and large message. The message larger than this switch point will be transferred with large message protocol (Default 4096).");
	fi_param_define(&rxr_prov, "recvwin_size", FI_PARAM_INT,
			"Defines the size of sliding receive window. (Default: 16384)");
	fi_param_define(&rxr_prov, "cq_size", FI_PARAM_INT,
			"Define the size of completion queue. (Default: 8192)");
	fi_param_define(&rxr_prov, "mr_cache_enable", FI_PARAM_BOOL,
			"Enables using the mr cache and in-line registration instead of a bounce buffer for iov's larger than max_memcpy_size. Defaults to true. When disabled, only uses a bounce buffer.");
	fi_param_define(&rxr_prov, "mr_cache_merge_regions", FI_PARAM_BOOL,
			"Enables merging overlapping and adjacent memory registration regions. Defaults to true.");
	fi_param_define(&rxr_prov, "mr_max_cached_count", FI_PARAM_INT,
			"Sets the maximum number of memory registrations that can be cached at any time.");
	fi_param_define(&rxr_prov, "mr_max_cached_size", FI_PARAM_INT,
			"Sets the maximum amount of memory that cached memory registrations can hold onto at any time.");
	fi_param_define(&rxr_prov, "max_memcpy_size", FI_PARAM_INT,
			"Threshold size switch between using memory copy into a pre-registered bounce buffer and memory registration on the user buffer. (Default: 4096)");
	fi_param_define(&rxr_prov, "mtu_size", FI_PARAM_INT,
			"Override the MTU size of the device.");
	fi_param_define(&rxr_prov, "tx_size", FI_PARAM_INT,
			"Set the maximum number of transmit operations before the provider returns -FI_EAGAIN. For only the RDM endpoint, this parameter will cause transmit operations to be queued when this value is set higher than the default and the transmit queue is full.");
	fi_param_define(&rxr_prov, "rx_size", FI_PARAM_INT,
			"Set the maximum number of receive operations before the provider returns -FI_EAGAIN.");
	fi_param_define(&rxr_prov, "tx_iov_limit", FI_PARAM_INT,
			"Maximum transmit iov_limit.");
	fi_param_define(&rxr_prov, "rx_iov_limit", FI_PARAM_INT,
			"Maximum receive iov_limit.");
	fi_param_define(&rxr_prov, "rx_copy_unexp", FI_PARAM_BOOL,
			"Enables the use of a separate pool of bounce-buffers to copy unexpected messages out of the pre-posted receive buffers. (Default: 1)");
	fi_param_define(&rxr_prov, "rx_copy_ooo", FI_PARAM_BOOL,
			"Enables the use of a separate pool of bounce-buffers to copy out-of-order RTS packets out of the pre-posted receive buffers. (Default: 1)");
	fi_param_define(&rxr_prov, "max_timeout", FI_PARAM_INT,
			"Set the maximum timeout (us) for backoff to a peer after a receiver not ready error. (Default: 1000000)");
	fi_param_define(&rxr_prov, "timeout_interval", FI_PARAM_INT,
			"Set the time interval (us) for the base timeout to use for exponential backoff to a peer after a receiver not ready error. (Default: 0 [random])");
	rxr_init_env();

	lower_efa_prov = init_lower_efa_prov();
	if (!lower_efa_prov)
		return NULL;

	if (rxr_env.enable_shm_transfer && rxr_get_local_gids(lower_efa_prov))
		return NULL;

	return &rxr_prov;
}
