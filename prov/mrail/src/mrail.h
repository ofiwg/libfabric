/*
 * Copyright (c) 2018 Intel Corporation, Inc.  All rights reserved.
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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <string.h>

#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include "rdma/providers/fi_log.h"

#include <ofi.h>
#include <ofi_util.h>
#include <ofi_proto.h>
#include <ofi_prov.h>
#include <ofi_enosys.h>

#define MRAIL_MAJOR_VERSION 1
#define MRAIL_MINOR_VERSION 0

#define MRAIL_MAX_INFO 100

#define MRAIL_PASSTHROUGH_MODES		0
#define MRAIL_PASSTHROUGH_MR_MODES	(FI_MR_LOCAL | OFI_MR_BASIC_MAP)

extern struct fi_info mrail_info;
extern struct fi_provider mrail_prov;
extern struct util_prov mrail_util_prov;
extern struct fi_fabric_attr mrail_fabric_attr;

extern struct fi_info *mrail_info_vec[MRAIL_MAX_INFO];
extern size_t mrail_num_info;

struct mrail_fabric {
	struct util_fabric util_fabric;
	struct fi_info *info;
	struct fid_fabric **fabrics;
	size_t num_fabrics;
};

struct mrail_domain {
	struct util_domain util_domain;
	struct fi_info *info;
	struct fid_domain **domains;
	size_t num_domains;
	size_t addrlen;
};

struct mrail_av {
	struct util_av util_av;
	struct fid_av **avs;
	size_t *rail_addrlen;
	size_t num_avs;
	ofi_atomic32_t index;
};

struct mrail_cq {
	struct util_cq util_cq;
	struct fid_cq **cqs;
	size_t num_cqs;
};

struct mrail_ep {
	struct util_ep util_ep;
	struct fi_info *info;
	struct fid_ep **eps;
	size_t num_eps;
	ofi_atomic32_t tx_rail;
	ofi_atomic32_t rx_rail;
};

int mrail_get_core_info(uint32_t version, const char *node, const char *service,
			uint64_t flags, const struct fi_info *hints,
			struct fi_info **core_info);
int mrail_fabric_open(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		       void *context);
int mrail_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		       struct fid_domain **domain, void *context);
int mrail_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		   struct fid_cq **cq_fid, void *context);
int mrail_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		   struct fid_av **av_fid, void *context);
int mrail_ep_open(struct fid_domain *domain, struct fi_info *info,
		   struct fid_ep **ep_fid, void *context);

static inline struct fi_info *mrail_get_info_cached(char *name)
{
	struct fi_info *info;
	size_t i;

	for (i = 0; i < mrail_num_info; i++) {
		info = mrail_info_vec[i];
		if (!strcmp(info->fabric_attr->name, name))
			return info;
	}

	FI_WARN(&mrail_prov, FI_LOG_CORE, "Unable to find matching "
		"fi_info in mrail_info_vec for given fabric name\n");
	return NULL;
}

static inline int mrail_close_fids(struct fid **fids, size_t count)
{
	int ret, retv = 0;
	size_t i;

	for (i = 0; i < count; i++) {
		if (fids[i]) {
			ret = fi_close(fids[i]);
			if (ret)
				retv = ret;
		}
	}
	return retv;
}
