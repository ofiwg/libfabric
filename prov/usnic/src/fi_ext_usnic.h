/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
 * Copyright (c) 2015-2016 Cisco Systems, Inc.  All rights reserved.
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

#ifndef _FI_EXT_USNIC_H_
#define _FI_EXT_USNIC_H_

/*
 * See the fi_usnic.7 man page for information about the usnic provider
 * extensions provided in this header.
 */

#include <stdint.h>
#include <net/if.h>

#define FI_PROTO_RUDP (100U | FI_PROV_SPECIFIC)

#define FI_EXT_USNIC_INFO_VERSION 2

#define FI_EXT_USNIC_MAX_DEVNAME 16

/*
 * usNIC specific info
 */
struct fi_usnic_cap {
	const char *uc_capability;
	int uc_present;
};

struct fi_usnic_info_v1 {
	uint32_t ui_link_speed;
	uint32_t ui_netmask_be;
	char ui_ifname[IFNAMSIZ];

	uint32_t ui_num_vf;
	uint32_t ui_qp_per_vf;
	uint32_t ui_cq_per_vf;
};

struct fi_usnic_info_v2 {
	/* Put all of the v1 fields at the start to provide some backward
	 * compatibility.
	 */
	uint32_t		ui_link_speed;
	uint32_t		ui_netmask_be;
	char			ui_ifname[IFNAMSIZ];
	unsigned		ui_num_vf;
	unsigned		ui_qp_per_vf;
	unsigned		ui_cq_per_vf;

	char			ui_devname[FI_EXT_USNIC_MAX_DEVNAME];
	uint8_t			ui_mac_addr[6];

	uint32_t		ui_ipaddr_be;
	uint32_t		ui_prefixlen;
	uint32_t		ui_mtu;
	uint8_t			ui_link_up;

	uint32_t		ui_vendor_id;
	uint32_t		ui_vendor_part_id;
	uint32_t		ui_device_id;
	char			ui_firmware[64];

	unsigned		ui_intr_per_vf;
	unsigned		ui_max_cq;
	unsigned		ui_max_qp;

	unsigned		ui_max_cqe;
	unsigned		ui_max_send_credits;
	unsigned		ui_max_recv_credits;

	const char		*ui_nicname;
	const char		*ui_pid;

	struct fi_usnic_cap	**ui_caps;
};

struct fi_usnic_info {
	uint32_t ui_version;
	union {
		struct fi_usnic_info_v1 v1;
		struct fi_usnic_info_v2 v2;
	} ui;
};

/*
 * usNIC-specific fabric ops
 */
#define FI_USNIC_FABRIC_OPS_1 "fabric_ops 1"
struct fi_usnic_ops_fabric {
	size_t size;
	int (*getinfo)(uint32_t version, struct fid_fabric *fabric,
				struct fi_usnic_info *info);
};

/*
 * usNIC-specific AV ops
 */
#define FI_USNIC_AV_OPS_1 "av_ops 1"
struct fi_usnic_ops_av {
	size_t size;
	int (*get_distance)(struct fid_av *av, void *addr, int *metric);
};

int usdf_fabric_ops_open(struct fid *fid, const char *ops_name, uint64_t flags,
		void **ops, void *context);
int usdf_av_ops_open(struct fid *fid, const char *ops_name, uint64_t flags,
		void **ops, void *context);

#endif /* _FI_EXT_USNIC_H_ */
