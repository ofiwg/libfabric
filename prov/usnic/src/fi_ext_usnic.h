/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
 * Copyright (c) 2015 Cisco Systems, Inc.  All rights reserved.
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
 *	  copyright notice, this list of conditions and the following
 *	  disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
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

#ifndef _FI_EXT_USNIC_H_
#define _FI_EXT_USNIC_H_

/*
 * See the fi_usnic.7 man page for information about the usnic provider
 * extensions provided in this header.
 */

#include <stdint.h>
#include <net/if.h>

#define FI_PROTO_RUDP 100

#define FI_EXT_USNIC_INFO_VERSION 1

/*
 * usNIC specific info
 */
struct fi_usnic_info_v1 {
	uint32_t ui_link_speed;
	uint32_t ui_netmask_be;
	char ui_ifname[IFNAMSIZ];

	uint32_t ui_num_vf;
	uint32_t ui_qp_per_vf;
	uint32_t ui_cq_per_vf;
};

struct fi_usnic_info {
	uint32_t ui_version;
	union {
		struct fi_usnic_info_v1 v1;
	} ui;
};

struct fi_usnic_shdom {
	uint32_t handle;
};

/*
 * usNIC-specific fabric ops
 */
#define FI_USNIC_FABRIC_OPS_1 "fabric_ops 1"
struct fi_usnic_ops_fabric {
	size_t size;
	int (*getinfo)(uint32_t version, struct fid_fabric *fabric,
				struct fi_usnic_info *info);
	int (*verbs_compat)(uint8_t op, uint8_t sub_op, void *context,
				void *out);
	int (*share_domain)(struct fid_fabric *fabric, struct fi_info *info,
				struct fi_usnic_shdom *shdom, uint64_t share_key,
				struct fid_domain **domain, void *context);
};

enum verbs_compat_op {
	VERBS_COMPAT_OP_GET_DATA_STRUCTURE = 0,
	__VERBS_COMPAT_OP_MAX,
};
#define VERBS_COMPAT_OP_MAX (__VERBS_COMPAT_OP_MAX - 1)


enum verbs_data_structure {
	VERBS_DATA_IBV_DEVICE_ATTR = 0,
	VERBS_DATA_IBV_PORT_ATTR,
	__VERBS_DATA_MAX,
};
#define VERBS_DATA_MAX (__VERBS_DATA_MAX -1)


/*
 * usNIC-specific AV ops
 */
#define FI_USNIC_AV_OPS_1 "av_ops 1"
struct fi_usnic_ops_av {
	size_t size;
	int (*get_distance)(struct fid_av *av, void *addr, int *metric);
};

/*
 * usNIC-specific domain ops
 */
#define FI_USNIC_DOMAIN_OPS_1 "domain_ops 1"
struct fi_usnic_ops_domain {
	size_t size;
	int (*alloc_shdom)(struct fid_domain *domain, uint64_t share_key,
				struct fi_usnic_shdom *shdom);
};

#endif /* _FI_EXT_USNIC_H_ */
