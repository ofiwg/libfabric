/*
 * Copyright (c) 2015 Cray Inc.  All rights reserved.
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

#ifndef _FI_EXT_GNI_H_
#define _FI_EXT_GNI_H_

#ifdef __cplusplus
extern "C" {
#endif

#if HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */

#define FI_GNI_DOMAIN_OPS_1 "domain ops 1"
typedef enum dom_ops_val { GNI_MSG_RENDEZVOUS_THRESHOLD,
			   GNI_RMA_RDMA_THRESHOLD,
			   GNI_CONN_TABLE_INITIAL_SIZE,
			   GNI_CONN_TABLE_MAX_SIZE,
			   GNI_CONN_TABLE_STEP_SIZE,
			   GNI_VC_ID_TABLE_CAPACITY,
			   GNI_MBOX_PAGE_SIZE,
			   GNI_MBOX_NUM_PER_SLAB,
			   GNI_MBOX_MAX_CREDIT,
			   GNI_MBOX_MSG_MAX_SIZE,
			   GNI_RX_CQ_SIZE,
			   GNI_TX_CQ_SIZE,
			   GNI_MAX_RETRANSMITS,
			   GNI_ERR_INJECT_COUNT,
			   GNI_MR_CACHE_LAZY_DEREG,
			   GNI_NUM_DOM_OPS
} dom_ops_val_t;

/* per domain gni provider specific ops */
struct fi_gni_ops_domain {
	int (*set_val)(struct fid *fid, dom_ops_val_t t, void *val);
	int (*get_val)(struct fid *fid, dom_ops_val_t t, void *val);
};

/* per domain parameters */
struct gnix_ops_domain {
	uint32_t msg_rendezvous_thresh;
	uint32_t rma_rdma_thresh;
	uint32_t ct_init_size;
	uint32_t ct_max_size;
	uint32_t ct_step;
	uint32_t vc_id_table_capacity;
	uint32_t mbox_page_size;
	uint32_t mbox_num_per_slab;
	uint32_t mbox_maxcredit;
	uint32_t mbox_msg_maxsize;
	uint32_t rx_cq_size;
	uint32_t tx_cq_size;
	uint32_t max_retransmits;
	int32_t err_inject_count;
};

#endif /* _FI_EXT_GNI_H_ */
