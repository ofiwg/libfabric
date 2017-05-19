/*
 * Copyright (c) 2016 Intel Corporation, Inc.  All rights reserved.
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

#include <stdlib.h>
#include <string.h>

#include <fi_enosys.h>
#include <fi_util.h>

int ofi_ep_bind_av(struct util_ep *util_ep, struct util_av *av)
{
	if (util_ep->av) {
		FI_WARN(util_ep->av->prov, FI_LOG_EP_CTRL,
				"duplicate AV binding\n");
		return -FI_EINVAL;
	}
	util_ep->av = av;
	ofi_atomic_inc32(&av->ref);

	fastlock_acquire(&av->lock);
	dlist_insert_tail(&util_ep->av_entry, &av->ep_list);
	fastlock_release(&av->lock);

	return 0;
}

int ofi_endpoint_init(struct fid_domain *domain, const struct util_prov *util_prov,
		      struct fi_info *info, struct util_ep *ep, void *context,
		      ofi_ep_progress_func progress)
{
	struct util_domain *util_domain;
	int ret;

	util_domain = container_of(domain, struct util_domain, domain_fid);

	if (!info || !info->ep_attr || !info->rx_attr || !info->tx_attr)
		return -FI_EINVAL;

	ret = ofi_check_info(util_prov,
			     util_domain->fabric->fabric_fid.api_version, info);
	if (ret)
		return ret;

	ep->ep_fid.fid.fclass = FI_CLASS_EP;
	ep->ep_fid.fid.context = context;
	ep->domain = util_domain;
	ep->progress = progress;
	ofi_atomic_inc32(&util_domain->ref);
	fastlock_init(&ep->lock);
	return 0;
}

int ofi_endpoint_close(struct util_ep *util_ep)
{
	fastlock_destroy(&util_ep->lock);
	if (util_ep->av) {
		fastlock_acquire(&util_ep->av->lock);
		dlist_remove(&util_ep->av_entry);
		fastlock_release(&util_ep->av->lock);

		ofi_atomic_dec32(&util_ep->av->ref);
	}

	ofi_atomic_dec32(&util_ep->domain->ref);
	return 0;
}
