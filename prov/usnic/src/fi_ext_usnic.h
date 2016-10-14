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
} __attribute__((packed));

/* In API version 1.2 and below, the v1 structure did not contain any 64-bit
 * data types and therefore had a 4-byte alignment. Once v2 of the extension API
 * was introduced in version 1.3, the extra pointers mandated an 8-byte
 * alignment thus changing the offset of the v1 structure. This means that the
 * alignment difference manifests when an application using v1 of the extension
 * is compiled with Libfabric v1.1.x or v1.2.x, but then runs with libfabric.so
 * that is v1.3.x or higher (and vice versa). Make the alignment explicit and
 * consistent by adding an extra 32-bit padding (4 uint8_t). The padding and
 * packed attribute fix the compatibility issues with 1.3 and above.
 *
 * Fixed on the 1.1.x branch after the 1.1.1 release (backported from 1.4rc1).
 */
struct fi_usnic_info {
	uint32_t ui_version;
	uint8_t ui_pad0[4];
	union {
		struct fi_usnic_info_v1 v1;
	} ui;
} __attribute__((packed));

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

#endif /* _FI_EXT_USNIC_H_ */
