/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_MR_LAC_CACHE_H_
#define _CXIP_MR_LAC_CACHE_H_

/* cxip_mr_lac_cache type definition */
/* This is in a separate header to break the circular dependency between mr.h
 * and ctrl.h */

/* Forward declarations */
struct cxip_ctrl_req;

struct cxip_mr_lac_cache {
	/* MR referencing the associated MR cache LE, can only
	 * be flushed if reference count is 0.
	 */
	ofi_atomic32_t ref;
	union cxip_match_bits mb;
	struct cxip_ctrl_req *ctrl_req;
};

#endif /* _CXIP_MR_LAC_CACHE_H_ */
