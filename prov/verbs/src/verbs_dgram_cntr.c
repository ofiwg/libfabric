/*
 * Copyright (c) 2017 Intel Corporation, Inc.  All rights reserved.
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

#include "verbs_dgram.h"

int fi_ibv_dgram_cntr_open(struct fid_domain *domain_fid,
			   struct fi_cntr_attr *attr,
			   struct fid_cntr **cntr_fid,
			   void *context)
{
	int ret;
	struct fi_ibv_dgram_cntr *cntr;
	struct fi_ibv_domain *domain;

	cntr = calloc(1, sizeof(*cntr));
	if (!cntr)
		return -FI_ENOMEM;

	domain = container_of(domain_fid, struct fi_ibv_domain,
			      util_domain.domain_fid);
	if (!domain) {
		ret = -FI_EINVAL;
 		goto free;
	}

	assert(domain->ep_type == FI_EP_DGRAM);

	ret = ofi_cntr_init(&fi_ibv_prov, domain_fid, attr, &cntr->util_cntr,
			    &ofi_cntr_progress, context);
	if (ret)
		goto free;

	*cntr_fid = &cntr->util_cntr.cntr_fid;
	return FI_SUCCESS;

free:
	free(cntr);
	return ret;
}
