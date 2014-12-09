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
#define FI_LIB_EXTENSION "fi"
#define FI_LIB_SUFFIX FI_LIB_EXTENSION ".so"

#define FI_LIB_CLASS_NAME	"libfabric"

struct fi_provider {
	uint32_t version;
	uint32_t fi_version;
	int	(*getinfo)(uint32_t version, const char *node, const char *service,
			uint64_t flags, struct fi_info *hints, struct fi_info **info);
	int	(*fabric)(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
			void *context);
	const char *name;
};

/* ctor and dtor function signatures */
#define INI_SIG(name) struct fi_provider* name(void)
#define FINI_SIG(name) void name(void)

/* dl providers ctor and dtors are called when loaded and unloaded */
#define EXT_INI \
	__attribute__((visibility ("default"))) \
	struct fi_provider* fi_prov_ini(void)

#define EXT_FINI \
	__attribute__((visibility ("default"))) \
	void fi_prov_fini(void)

#ifdef __cplusplus
}
#endif

#endif /* _FI_PROV_H_ */
