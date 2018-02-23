/*
 * Copyright (c) 2018 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL); Version 2, available from the file
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

#ifndef _OFI_HOOK_H_
#define _OFI_HOOK_H_

#include <assert.h>

#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>


/*
 * Define hook structs so we can cast from fid to parent using simple cast.
 * This lets us have a single close() call.
 */

extern struct fi_ops hook_fid_ops;
struct fid *hook_to_hfid(const struct fid *fid);

struct hook_fabric {
	struct fid_fabric fabric;
	struct fid_fabric *hfabric;
};

int hook_fabric(struct fid_fabric *hfabric, struct fid_fabric **fabric);


struct hook_domain {
	struct fid_domain domain;
	struct fid_domain *hdomain;
};

int hook_domain(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **domain, void *context);


struct hook_av {
	struct fid_av av;
	struct fid_av *hav;
};

int hook_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		 struct fid_av **av, void *context);


struct hook_wait {
	struct fid_wait wait;
	struct fid_wait *hwait;
};

int hook_wait_open(struct fid_fabric *fabric, struct fi_wait_attr *attr,
		   struct fid_wait **waitset);
int hook_trywait(struct fid_fabric *fabric, struct fid **fids, int count);


struct hook_poll {
	struct fid_poll poll;
	struct fid_poll *hpoll;
};

int hook_poll_open(struct fid_domain *domain, struct fi_poll_attr *attr,
		   struct fid_poll **pollset);


struct hook_eq {
	struct fid_eq eq;
	struct fid_eq *heq;
};

int hook_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		 struct fid_eq **eq, void *context);


struct hook_cq {
	struct fid_cq cq;
	struct fid_cq *hcq;
};

int hook_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq, void *context);


struct hook_cntr {
	struct fid_cntr cntr;
	struct fid_cntr *hcntr;
};

int hook_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		   struct fid_cntr **cntr, void *context);


struct hook_ep {
	struct fid_ep ep;
	struct fid_ep *hep;
};

int hook_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep, void *context);
int hook_scalable_ep(struct fid_domain *domain, struct fi_info *info,
		     struct fid_ep **sep, void *context);
int hook_srx_ctx(struct fid_domain *domain,
		 struct fi_rx_attr *attr, struct fid_ep **rx_ep,
		 void *context);

extern struct fi_ops_cm hook_cm_ops;
extern struct fi_ops_msg hook_msg_ops;
extern struct fi_ops_rma hook_rma_ops;
extern struct fi_ops_tagged hook_tagged_ops;
extern struct fi_ops_atomic hook_atomic_ops;


struct hook_pep {
	struct fid_pep pep;
	struct fid_pep *hpep;
};

int hook_passive_ep(struct fid_fabric *fabric, struct fi_info *info,
		    struct fid_pep **pep, void *context);


struct hook_stx {
	struct fid_stx stx;
	struct fid_stx *hstx;
};

int hook_stx_ctx(struct fid_domain *domain,
		 struct fi_tx_attr *attr, struct fid_stx **stx,
		 void *context);


struct hook_mr {
	struct fid_mr mr;
	struct fid_mr *hmr;
};


#endif /* _OFI_HOOK_H_ */
