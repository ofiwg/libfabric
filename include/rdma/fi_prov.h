/*
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2005, 2006 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2005 PathScale, Inc.  All rights reserved.
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
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

#ifndef _FI_PROV_H_
#define _FI_PROV_H_

#include <rdma/fabric.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Extension that low-level drivers should add to their .so filename
 * (probably via libtool "-release" option).  For example a low-level
 * driver named "libfoo" should build a plug-in named "libfoo-fi.so".
 */
#define FI_LIB_EXTENSION fi

struct fi_ops_prov {
	size_t	size;
	int	(*getinfo)(int version, const char *node, const char *service,
			uint64_t flags, struct fi_info *hints, struct fi_info **info);
	int	(*freeinfo)(struct fi_info *info);
	int	(*domain)(struct fid_fabric *fabric, struct fi_info *info,
			struct fid_domain **dom, void *context);
};

int fi_version_register(int version, struct fi_ops_prov *ops);
static inline int fi_register(struct fi_ops_prov *ops)
{
	return fi_version_register(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), ops);
}


#define FI_LIB_CLASS_NAME	"libfabric"

struct fi_ops_lib {
	size_t		size;
	size_t		(*context_size)(void);
	const char *	(*sysfs_path)(void);
	int		(*read_file)(const char *dir, const char *file,
				     char *buf, size_t size);
};

#ifdef __cplusplus
}
#endif

#endif /* _FI_PROV_H_ */
