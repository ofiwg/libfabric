/*
 * Copyright (c) 2015-2016 Intel Corporation. All rights reserved.
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

#include "udpx.h"
#include "ofi_osd.h"

#define UDPX_TX_CAPS (OFI_TX_MSG_CAPS | FI_MULTICAST)
#define UDPX_RX_CAPS (FI_SOURCE | OFI_RX_MSG_CAPS)
#define UDPX_DOMAIN_CAPS (FI_LOCAL_COMM | FI_REMOTE_COMM)

struct fi_tx_attr udpx_tx_attr = {
	.caps = UDPX_TX_CAPS,
	.inject_size = UDPX_MAX_MSG_SIZE(UDPX_MTU),
	.size = 1024,
	.iov_limit = UDPX_IOV_LIMIT
};

struct fi_rx_attr udpx_rx_attr = {
	.caps = UDPX_RX_CAPS,
	.size = 1024,
	.iov_limit = UDPX_IOV_LIMIT
};

struct fi_ep_attr udpx_ep_attr = {
	.type = FI_EP_DGRAM,
	.protocol = FI_PROTO_UDP,
	.protocol_version = 0,
	.max_msg_size = UDPX_MAX_MSG_SIZE(UDPX_MTU),
	.tx_ctx_cnt = 1,
	.rx_ctx_cnt = 1
};

struct fi_domain_attr udpx_domain_attr = {
	.caps = UDPX_DOMAIN_CAPS,
	.name = "udp",
	.threading = FI_THREAD_SAFE,
	.control_progress = FI_PROGRESS_AUTO,
	.data_progress = FI_PROGRESS_AUTO,
	.resource_mgmt = FI_RM_ENABLED,
	.av_type = FI_AV_UNSPEC,
	.mr_mode = OFI_MR_BASIC | OFI_MR_SCALABLE,
	.mr_key_size = sizeof(uint64_t),
	.cq_cnt = 256,
	.ep_cnt = 256,
	.tx_ctx_cnt = 256,
	.rx_ctx_cnt = 256,
	.max_ep_tx_ctx = 1,
	.max_ep_rx_ctx = 1
};

struct fi_fabric_attr udpx_fabric_attr = {
	.name = "UDP-IP",
	.prov_version = OFI_VERSION_DEF_PROV
};

struct fi_info udpx_info = {
	.caps = UDPX_DOMAIN_CAPS | UDPX_TX_CAPS | UDPX_RX_CAPS,
	.addr_format = FI_SOCKADDR,
	.tx_attr = &udpx_tx_attr,
	.rx_attr = &udpx_rx_attr,
	.ep_attr = &udpx_ep_attr,
	.domain_attr = &udpx_domain_attr,
	.fabric_attr = &udpx_fabric_attr
};

struct util_prov udpx_util_prov = {
	.prov = &udpx_prov,
	.info = NULL,
        .flags = 0,
};


static int match_interface(struct slist_entry *entry, const void *infop)
{
	struct ofi_addr_list_entry *addr_entry;
	const struct fi_info* info = infop;

	addr_entry = container_of(entry, struct ofi_addr_list_entry, entry);
	return strcmp(addr_entry->net_name, info->fabric_attr->name) == 0 &&
	       strcmp(addr_entry->ifa_name, info->domain_attr->name) == 0;
}

static void set_mtu_from_addr_list(struct fi_info* info,
				   struct slist *addr_list)
{
	struct ofi_addr_list_entry *addr_entry;
	struct slist_entry *entry;
	int max_msg_size;

	entry = slist_find_first_match(addr_list, match_interface, info);
	if (entry) {
		addr_entry = container_of(entry,
					  struct ofi_addr_list_entry,
					  entry);
		max_msg_size = UDPX_MAX_MSG_SIZE(addr_entry->mtu);
		if (max_msg_size > 0) {
			info->tx_attr->inject_size = max_msg_size;
			info->ep_attr->max_msg_size = max_msg_size;
		}
	} else {
		FI_DBG(&udpx_prov, FI_LOG_CORE,
		       "Failed to match interface (%s, %s) to "
		       "address for MTU size\n",
		       info->fabric_attr->name, info->domain_attr->name);
	}
}

void udpx_util_prov_init(uint32_t version)
{

	struct slist addr_list;
	struct fi_info* cur;
	struct fi_info* info;

        if (udpx_util_prov.info == NULL) {
		udpx_util_prov.info = &udpx_info;
		info = fi_allocinfo();
		ofi_ip_getinfo(&udpx_util_prov, version, NULL, NULL, 0, NULL,
			       &info);
		slist_init(&addr_list);
		ofi_get_list_of_addr(&udpx_prov, "iface", &addr_list);
		for (cur = info; cur; cur = cur->next)
			set_mtu_from_addr_list(cur, &addr_list);
		*(struct fi_info**)&udpx_util_prov.info = info;
		ofi_free_list_of_addr(&addr_list);
	}
}

void udpx_util_prov_fini()
{
	if (udpx_util_prov.info != NULL)
		fi_freeinfo((struct fi_info*)udpx_util_prov.info);
}
