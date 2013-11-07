/*
 * Copyright (c) 2004 Topspin Communications.  All rights reserved.
 * Copyright (c) 2005 Voltaire, Inc. All rights reserved.
 * Copyright (c) 2013 Intel Corp., Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
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

#ifndef _FI_UMAD_H_
#define _FI_UMAD_H_

#include <linux/types.h>
#include <linux/ioctl.h>


#ifdef __cplusplus
extern "C" {
#endif


/*
 * This file must be kept in sync with the kernel's version of ib_user_mad.h
 */

#define UMAD_MIN_ABI_VERSION	5
#define UMAD_MAX_ABI_VERSION	5


struct umad_hdr {
	__u32	id;
	__u32	status;
	__u32	timeout_ms;
	__u32	retries;
	__u32	length;
	__be32	qpn;
	__be32  qkey;
	__be16	lid;
	__u8	sl;
	__u8	path_bits;
	__u8	grh_present;
	__u8	gid_index;
	__u8	hop_limit;
	__u8	traffic_class;
	__u8	gid[16];
	__be32	flow_label;
	__u16	pkey_index;
	__u8	reserved[6];
};

struct umad_data {
	struct umad_hdr hdr;
	__u64	data[0];
};

typedef unsigned long __attribute__((aligned(4))) packed_ulong;
#define UMAD_LONGS_PER_METHOD_MASK (128 / (8 * sizeof (long)))

struct umad_reg_req {
	__u32	id;
	packed_ulong method_mask[UMAD_LONGS_PER_METHOD_MASK];
	__u8	qpn;
	__u8	mgmt_class;
	__u8	mgmt_class_version;
	__u8    oui[3];
	__u8	rmpp_version;
};

#define UMAD_IOCTL_MAGIC	0x1b
#define UMAD_REGISTER_AGENT	_IOWR(UMAD_IOCTL_MAGIC, 1, struct umad_reg_req)
#define UMAD_UNREGISTER_AGENT	_IOW(UMAD_IOCTL_MAGIC, 2, __u32)
#define UMAD_ENABLE_PKEY	_IO(UMAD_IOCTL_MAGIC, 3)


#define FI_UVERBS_CLASS_NAME	"umad"
#define FI_UMAD_OPS		(4ULL << FI_OPS_LIB_SHIFT)

struct fi_umad_ops {
	size_t	size;
	int	(*get_abi)(void);
};

#ifdef __cplusplus
}
#endif

#endif /* _FI_UMAD_H_ */
