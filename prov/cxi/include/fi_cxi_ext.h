/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2020 Cray Inc. All rights reserved.
 */

#ifndef _FI_CXI_EXT_H_
#define _FI_CXI_EXT_H_


#define FI_CXI_DOM_OPS_1 "dom_ops_v1"

struct fi_cxi_dom_ops {
	int (*cntr_read)(struct fid *fid, unsigned int cntr, uint64_t *value,
		      struct timespec *ts);
};

#endif /* _FI_CXI_EXT_H_ */
