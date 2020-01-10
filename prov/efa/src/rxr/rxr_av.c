/*
 * Copyright (c) 2019 Amazon.com, Inc. or its affiliates.
 * All rights reserved.
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

#include "rxr.h"
#include "efa.h"
#include <inttypes.h>

static int rxr_av_insertsvc(struct fid_av *av, const char *node,
			    const char *service, fi_addr_t *fi_addr,
			    uint64_t flags, void *context)
{
	return -FI_ENOSYS;
}

static int rxr_av_insertsym(struct fid_av *av_fid, const char *node,
			    size_t nodecnt, const char *service, size_t svccnt,
			    fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	return -FI_ENOSYS;
}

static int rxr_av_remove(struct fid_av *av_fid, fi_addr_t *fi_addr,
			 size_t count, uint64_t flags)
{
	int ret = 0;
	size_t i;
	struct rxr_av *av;
	struct efa_av_entry *av_entry;
	void *addr;

	av = container_of(av_fid, struct rxr_av, util_av.av_fid);
	addr = calloc(1, av->rdm_addrlen);
	if (!addr) {
		FI_WARN(&rxr_prov, FI_LOG_AV,
			"Failed to allocate memory for av addr\n");
		return -FI_ENOMEM;
	}

	fastlock_acquire(&av->util_av.lock);
	for (i = 0; i < count; i++) {
		ret = fi_av_lookup(av->rdm_av, fi_addr[i],
				   addr, &av->rdm_addrlen);
		if (ret)
			break;

		ret = fi_av_remove(av->rdm_av, &fi_addr[i], 1, flags);
		if (ret)
			break;

		HASH_FIND(hh, av->av_map, addr, av->rdm_addrlen, av_entry);

		/* remove an address from shm provider's av */
		if (rxr_env.enable_shm_transfer && av_entry->local_mapping) {
			ret = fi_av_remove(av->shm_rdm_av, &av_entry->shm_rdm_addr, 1, flags);
			if (ret)
				break;
		}

		if (av_entry) {
			HASH_DEL(av->av_map, av_entry);
			free(av_entry);
		}

		av->used--;
	}
	fastlock_release(&av->util_av.lock);
	free(addr);
	return ret;
}

static const char *rxr_av_straddr(struct fid_av *av, const void *addr,
				  char *buf, size_t *len)
{
	struct rxr_av *rxr_av;

	rxr_av = container_of(av, struct rxr_av, util_av.av_fid);
	return rxr_av->rdm_av->ops->straddr(rxr_av->rdm_av, addr, buf, len);
}

static int rxr_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr,
			 size_t *addrlen)
{
	struct rxr_av *rxr_av;

	rxr_av = container_of(av, struct rxr_av, util_av.av_fid);
	return fi_av_lookup(rxr_av->rdm_av, fi_addr, addr, addrlen);
}

static struct fi_ops_av rxr_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insertsvc = rxr_av_insertsvc,
	.insertsym = rxr_av_insertsym,
	.remove = rxr_av_remove,
	.lookup = rxr_av_lookup,
	.straddr = rxr_av_straddr,
};

static int rxr_av_close(struct fid *fid)
{
	struct rxr_av *av;
	struct efa_av_entry *curr_av_entry, *tmp;
	int ret = 0;

	av = container_of(fid, struct rxr_av, util_av.av_fid);
	ret = fi_close(&av->rdm_av->fid);
	if (ret)
		goto err;
	if (rxr_env.enable_shm_transfer) {
		ret = fi_close(&av->shm_rdm_av->fid);
		if (ret) {
			FI_WARN(&rxr_prov, FI_LOG_AV, "Failed to close shm av\n");
			goto err;
		}
	}

	ret = ofi_av_close(&av->util_av);
	if (ret)
		goto err;

err:
	HASH_ITER(hh, av->av_map, curr_av_entry, tmp) {
		HASH_DEL(av->av_map, curr_av_entry);
		free(curr_av_entry);
	}
	free(av);
	return ret;
}

static int rxr_av_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	return ofi_av_bind(fid, bfid, flags);
}

static struct fi_ops rxr_av_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxr_av_close,
	.bind = rxr_av_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int rxr_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		struct fid_av **av_fid, void *context)
{
	struct rxr_av *av;
	struct rxr_domain *domain;
	struct fi_av_attr av_attr;
	struct util_av_attr util_attr;
	size_t universe_size;
	int ret, retv;

	if (!attr)
		return -FI_EINVAL;

	if (attr->name)
		return -FI_ENOSYS;

	/* FI_EVENT, FI_READ, and FI_SYMMETRIC are not supported */
	if (attr->flags)
		return -FI_ENOSYS;

	domain = container_of(domain_fid, struct rxr_domain,
			      util_domain.domain_fid);
	av = calloc(1, sizeof(*av));
	if (!av)
		return -FI_ENOMEM;

	/*
	 * TODO: remove me once RxR supports resizing members tied to the AV
	 * size.
	 */
	if (!attr->count)
		attr->count = EFA_MIN_AV_SIZE;
	else
		attr->count = MAX(attr->count, EFA_MIN_AV_SIZE);

	if (fi_param_get_size_t(NULL, "universe_size",
				&universe_size) == FI_SUCCESS)
		attr->count = MAX(attr->count, universe_size);

	util_attr.addrlen = sizeof(fi_addr_t);
	util_attr.flags = 0;
	if (attr->type == FI_AV_UNSPEC){
		if (domain->util_domain.av_type != FI_AV_UNSPEC)
			attr->type = domain->util_domain.av_type;
		else
			attr->type = FI_AV_TABLE;
	}
	ret = ofi_av_init(&domain->util_domain, attr, &util_attr,
			  &av->util_av, context);
	if (ret)
		goto err;

	av_attr = *attr;

	FI_DBG(&rxr_prov, FI_LOG_AV, "fi_av_attr:%" PRId64 "\n",
	       av_attr.flags);

	av_attr.type = FI_AV_TABLE;

	ret = fi_av_open(domain->rdm_domain, &av_attr, &av->rdm_av, context);
	if (ret)
		goto err;

	if (rxr_env.enable_shm_transfer) {
		/*
		 * shm av supports maximum 256 entries
		 * Reset the count to 128 to reduce memory footprint and satisfy
		 * the need of the instances with more CPUs.
		 */
		assert(rxr_env.shm_av_size <= EFA_SHM_MAX_AV_COUNT);
		av_attr.count = rxr_env.shm_av_size;
		ret = fi_av_open(domain->shm_domain, &av_attr, &av->shm_rdm_av, context);
		if (ret)
			goto err_close_rdm_av;
	}

	av->rdm_addrlen = domain->addrlen;

	*av_fid = &av->util_av.av_fid;
	(*av_fid)->fid.fclass = FI_CLASS_AV;
	(*av_fid)->fid.ops = &rxr_av_fi_ops;
	(*av_fid)->ops = &rxr_av_ops;
	return 0;

err_close_rdm_av:
	retv = fi_close(&av->rdm_av->fid);
	if (retv)
		FI_WARN(&rxr_prov, FI_LOG_AV,
				"Unable to close rdm av: %s\n", fi_strerror(-retv));
err:
	free(av);
	return ret;
}
