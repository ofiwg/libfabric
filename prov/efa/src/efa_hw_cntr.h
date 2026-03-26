/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef _EFA_HW_CNTR_H_
#define _EFA_HW_CNTR_H_

#include "efa_cntr.h"

#if HAVE_EFADV_CREATE_COMP_CNTR

extern struct fi_ops efa_hw_cntr_fi_ops;
extern struct fi_ops_cntr efa_hw_cntr_ops;

int efa_hw_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		     struct efa_cntr *cntr, struct fid_cntr **cntr_fid,
		     void *context,
		     struct efadv_comp_cntr_init_attr *cc_attr);

int efa_hw_cntr_close(struct fid *fid);

/* Increment the hardware completion counter by value */
int efa_hw_cntr_add(struct fid_cntr *cntr_fid, uint64_t value);

/* Increment the hardware error counter by value */
int efa_hw_cntr_adderr(struct fid_cntr *cntr_fid, uint64_t value);

/* Set the hardware completion counter to value */
int efa_hw_cntr_set(struct fid_cntr *cntr_fid, uint64_t value);

/* Set the hardware error counter to value */
int efa_hw_cntr_seterr(struct fid_cntr *cntr_fid, uint64_t value);

/* Read the current hardware completion counter value */
uint64_t efa_hw_cntr_read(struct fid_cntr *cntr_fid);

/* Read the current hardware error counter value */
uint64_t efa_hw_cntr_readerr(struct fid_cntr *cntr_fid);

/* Wait until the hardware completion counter reaches threshold or timeout.
 * Polls with exponential backoff (up to 5 attempts or infinite when timeout is -1).
 * Returns FI_SUCCESS when counter >= threshold, -FI_EAVAIL if the error
 * counter changes, -FI_ETIMEDOUT on timeout, -FI_EINVAL if the counter
 * was configured with FI_WAIT_NONE, or -FI_EOPNOTSUPP if the counter
 * memory is on device (DMABUF) without a host mapping.
 */
int efa_hw_cntr_wait(struct fid_cntr *cntr_fid, uint64_t threshold,
		     int timeout);
#endif /* HAVE_EFADV_CREATE_COMP_CNTR */

#endif /* _EFA_HW_CNTR_H_ */
