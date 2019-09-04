/*
 * Copyright (c) 2019 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
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
#include "mlx.h"
#include "ofi_util.h"

static int mlx_cntr_wait(struct fid_cntr *cntr_fid, uint64_t threshold, int timeout)
{
	struct fid_list_entry *fid_entry;
	struct util_cntr *cntr;
	struct mlx_ep *ep;
	uint64_t start, errcnt;
	int ret, ep_retry;

	cntr = container_of(cntr_fid, struct util_cntr, cntr_fid);
	assert(cntr->wait);
	errcnt = ofi_atomic_get64(&cntr->err);
	start = (timeout >= 0) ? fi_gettime_ms() : 0;

	do {
		cntr->progress(cntr);
		if (threshold <= ofi_atomic_get64(&cntr->cnt))
			return FI_SUCCESS;

		if (errcnt != ofi_atomic_get64(&cntr->err))
			return -FI_EAVAIL;

		if (timeout >= 0) {
			timeout -= (int) (fi_gettime_ms() - start);
			if (timeout <= 0)
				return -FI_ETIMEDOUT;
		}

		ep_retry = -1;
		fastlock_acquire(&cntr->ep_list_lock);
		dlist_foreach_container(&cntr->ep_list, struct fid_list_entry,
					fid_entry, entry) {
			ep = container_of(fid_entry->fid, struct mlx_ep,
					  ep.ep_fid.fid);
		}
		fastlock_release(&cntr->ep_list_lock);
		/*Add wait call here*/
	} while (!ret);

	return ret;
}


int mlx_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		  struct fid_cntr **cntr_fid, void *context)
{
	int ret;
	struct util_cntr *cntr;

	cntr = calloc(1, sizeof(*cntr));
	if (!cntr)
		return -FI_ENOMEM;

	ret = ofi_cntr_init(&mlx_prov, domain, attr, cntr,
			    &ofi_cntr_progress, context);
	if (ret)
		goto free;

	*cntr_fid = &cntr->cntr_fid;
	cntr->cntr_fid.ops->wait = mlx_cntr_wait;
	return FI_SUCCESS;

free:
	free(cntr);
	return ret;
}
