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

/*
 * CXI Authorization Key
 */
struct cxi_auth_key {
	/* The CXI service assigned to the Domain and Endpoints. A CXI service
	 * is associated with a set of local resource limits, VNIs, and Traffic
	 * Classes.
	 *
	 * The svc_id used by an OFI Domain must match all Endpoints belonging
	 * to the Domain.
	 */
	uint32_t svc_id;

	/* The Virtual Network ID (VNI) assigned to the Endpoint. Two Endpoints
	 * must use the same VNI in order to communicate.
	 *
	 * Note that while the CXI service may define one or more VNIs which a
	 * process can access, an Endpoint is assigned to only one.
	 */
	uint16_t vni;
};

#endif /* _FI_CXI_EXT_H_ */
