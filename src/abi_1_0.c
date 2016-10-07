/*
 * Copyright (c) 2016 Intel Corporation. All rights reserved.
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

#include "config.h"

#include <assert.h>
#include <stdlib.h>
#include <stddef.h>

#include <rdma/fabric.h>
#include <fi_abi.h>


/*
 * TODO: Add structures that change between 1.0 and 1.1
 */
struct fi_fabric_attr_1_0 {
	struct fid_fabric	*fabric;
	char			*name;
	char			*prov_name;
	uint32_t		prov_version;
};

struct fi_info_1_0 {
	struct fi_info		*next;
	uint64_t		caps;
	uint64_t		mode;
	uint32_t		addr_format;
	size_t			src_addrlen;
	size_t			dest_addrlen;
	void			*src_addr;
	void			*dest_addr;
	fid_t			handle;
	struct fi_tx_attr	*tx_attr;
	struct fi_rx_attr	*rx_attr;
	struct fi_ep_attr	*ep_attr;
	struct fi_domain_attr	*domain_attr;
	struct fi_fabric_attr	*fabric_attr;
};


/*
 * TODO: translate from 1.0 structures to 1.1 where needed on all calls below.
 */
__attribute__((visibility ("default")))
int fi_getinfo_1_0(uint32_t version, const char *node, const char *service,
		    uint64_t flags, struct fi_info_1_0 *hints,
		    struct fi_info_1_0 **info)
{
	return fi_getinfo(version, node, service, flags,
			  (struct fi_info *) hints,
			  (struct fi_info **) info);
}
COMPAT_SYMVER(fi_getinfo_1_0, fi_getinfo, FABRIC_1.0);

__attribute__((visibility ("default")))
void fi_freeinfo_1_0(struct fi_info_1_0 *info)
{
	fi_freeinfo((struct fi_info *) info);
}
COMPAT_SYMVER(fi_freeinfo_1_0, fi_freeinfo, FABRIC_1.0);

__attribute__((visibility ("default")))
struct fi_info_1_0 *fi_dupinfo_1_0(const struct fi_info_1_0 *info)
{
	return (struct fi_info_1_0 *) fi_dupinfo((const struct fi_info *) info);
}
COMPAT_SYMVER(fi_dupinfo_1_0, fi_dupinfo, FABRIC_1.0);
