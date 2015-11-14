/*
 * Copyright (c) 2015 Cray Inc. All rights reserved.
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
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

#ifndef _GNIX_AV_H_
#define _GNIX_AV_H_

#include "gnix.h"

/*
 * this structure should ideally be as compact
 * as possible, since its looked up in the critical
 * path for FI_EP_RDM EPs that use FI_AV_MAP.  It
 * needs to hold sufficient content that the gnix_ep_name
 * can be regnerated in full for fi_av_lookup.
 */

/**
 * Av addr entry struct
 *
 * @var gnix_addr            gnix address for this entry
 * @var name_type            the endpoint type associated with this
 *                           address (GNIX_EPN_TYPE_UNBOUND/BOUND)
 * @var cm_nic_cdm_id        for GNIX_EPN_TYPE_UNBOUND endpoint types
 *                           the cdm id of the cm_nic with which the endpoint
 *                           is associated
 * @var cookie               RDMA cookie credential for the endpoint
 *                           this entry corresponds to
 */
struct gnix_av_addr_entry {
	struct gnix_address gnix_addr;
	struct {
		uint32_t name_type : 8;
		uint32_t cm_nic_cdm_id : 24;
		uint32_t cookie;
	};
};

/*
 * Prototypes for GNI AV helper functions for managing the AV system.
 */

/**
 * @brief  Return pointer to an AV table internal gnix_av_addr_entry for
 *         a given fi_addr address
 *
 * @param[in]     gnix_av   pointer to a previously allocated gnix_fid_av
 * @param[in]     fi_addr   address to be translated
 * @param[out]    addr      pointer to address entry in AV table
 * @return  FI_SUCCESS on success, -FI_EINVAL on error
 */
int _gnix_av_lookup(struct gnix_fid_av *gnix_av, fi_addr_t fi_addr,
		    struct gnix_av_addr_entry **addr);


#endif /* _GNIX_AV_H_ */
