/*
 * Copyright (c) 2015-2016 Intel Corporation, Inc.  All rights reserved.
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
#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_trigger.h>

#include <fi.h>
#include <fi_proto.h>
#include <fi_enosys.h>
#include <fi_rbuf.h>
#include <fi_list.h>
#include <fi_util.h>

#ifndef _RXD_H_
#define _RXD_H_

#define RXD_MAJOR_VERSION 	(1)
#define RXD_MINOR_VERSION 	(0)
#define RXD_PROTOCOL_VERSION 	(1)
#define RXD_FI_VERSION 		FI_VERSION(1,3)

#define RXD_IOV_LIMIT		(4)
#define RXD_DEF_CQ_CNT		(8)
#define RXD_DEF_EP_CNT 		(8)
#define RXD_AV_DEF_COUNT	(128)

#define RXD_MAX_TX_BITS 	(10)
#define RXD_MAX_RX_BITS 	(10)
#define RXD_TX_ID(seq, id)	(((seq) << RXD_MAX_TX_BITS) | id)
#define RXD_RX_ID(seq, id)	(((seq) << RXD_MAX_RX_BITS) | id)
#define RXD_TX_IDX_BITS		((1ULL << RXD_MAX_TX_BITS) - 1)
#define RXD_RX_IDX_BITS		((1ULL << RXD_MAX_RX_BITS) - 1)

extern struct fi_provider rxd_prov;
extern struct fi_info rxd_info;
extern struct fi_fabric_attr rxd_fabric_attr;
extern struct util_prov rxd_util_prov;

struct rxd_fabric {
	struct util_fabric util_fabric;
	struct fid_fabric *dg_fabric;
};

struct rxd_domain {
	struct util_domain util_domain;
	struct fid_domain *dg_domain;

	size_t addrlen;
	ssize_t max_mtu_sz;
	uint64_t dg_mode;
	int do_progress;
	pthread_t progress_thread;
	fastlock_t lock;

	struct dlist_entry ep_list;
	struct dlist_entry cq_list;
};

struct rxd_cq {
	struct util_cq util_cq;
	struct dlist_entry dom_entry;
};

struct rxd_ep {
	struct dlist_entry dom_entry;
};

int rxd_alter_layer_info(struct fi_info *layer_info, struct fi_info *base_info);
int rxd_alter_base_info(struct fi_info *base_info, struct fi_info *layer_info);

int rxd_fabric(struct fi_fabric_attr *attr,
	       struct fid_fabric **fabric, void *context);
int rxd_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		    struct fid_domain **dom, void *context);

#endif
