/*
 * Copyright (c) 2016 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
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

#include <pthread.h>
#include <stdio.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>

#include <fi.h>
#include <fi_enosys.h>
#include <fi_util.h>

#ifndef _RXM_H_
#define _RXM_H_

#endif

#define RXM_MAJOR_VERSION 1
#define RXM_MINOR_VERSION 0

extern struct fi_provider rxm_prov;
extern struct util_prov rxm_util_prov;

struct rxm_fabric {
	struct util_fabric util_fabric;
	struct fid_fabric *msg_fabric;
};

struct rxm_domain {
	struct util_domain util_domain;
	struct fid_domain *msg_domain;
};

struct rxm_mr {
	struct fid_mr mr_fid;
	struct fid_mr *msg_mr;
};

struct rxm_cq {
	struct util_cq util_cq;
	struct fid_cq *msg_cq;
};

extern struct fi_provider rxm_prov;
extern struct fi_info rxm_info;
extern struct fi_fabric_attr rxm_fabric_attr;
extern struct fi_domain_attr rxm_domain_attr;

int rxm_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
			void *context);
int rxm_alter_layer_info(struct fi_info *layer_info, struct fi_info *base_info);
int rxm_alter_base_info(struct fi_info *base_info, struct fi_info *layer_info);
int rxm_domain_open(struct fid_fabric *fabric, struct fi_info *info,
			     struct fid_domain **dom, void *context);
int rxm_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
			 struct fid_cq **cq_fid, void *context);
