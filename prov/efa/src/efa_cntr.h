/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#if HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */

#ifndef _EFA_CNTR_H_
#define _EFA_CNTR_H_

struct efa_cntr {
	struct util_cntr util_cntr;
	struct dlist_entry ibv_cq_poll_list;
};

int efa_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		  struct fid_cntr **cntr_fid, void *context);

int efa_cntr_construct(struct efa_cntr *cntr, struct fid_domain *domain,
		       struct fi_cntr_attr *attr,
		       ofi_cntr_progress_func progress, void *context);

void efa_cntr_destruct(struct efa_cntr *cntr);

void efa_cntr_progress_ibv_cq_poll_list(struct efa_cntr *efa_cntr);

uint64_t efa_cntr_read(struct fid_cntr *cntr_fid);

uint64_t efa_cntr_readerr(struct fid_cntr *cntr_fid);

int efa_cntr_wait(struct fid_cntr *cntr_fid, uint64_t threshold, int timeout);

void efa_cntr_report_tx_completion(struct util_ep *ep, uint64_t flags);

void efa_cntr_report_rx_completion(struct util_ep *ep, uint64_t flags);

void efa_cntr_report_error(struct util_ep *ep, uint64_t flags);

#endif
