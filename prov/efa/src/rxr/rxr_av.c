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

/*
 * Local/remote peer detection by comparing peer GID with stored local GIDs
 */
static bool rxr_is_local_peer(struct rxr_av *av, const void *addr)
{
	struct efa_ep_addr *cur_efa_addr = local_efa_addr;

#if ENABLE_DEBUG
	char peer_gid[INET6_ADDRSTRLEN] = { 0 };

	if (!inet_ntop(AF_INET6, ((struct efa_ep_addr *)addr)->raw, peer_gid, INET6_ADDRSTRLEN)) {
		FI_WARN(&rxr_prov, FI_LOG_AV, "Failed to get current EFA's GID, errno: %d\n", errno);
		return 0;
	}
	FI_DBG(&rxr_prov, FI_LOG_AV, "The peer's GID is %s.\n", peer_gid);
#endif
	while (cur_efa_addr) {
		if (!memcmp(((struct efa_ep_addr *)addr)->raw, cur_efa_addr->raw, 16)) {
			FI_DBG(&rxr_prov, FI_LOG_AV, "The peer is local.\n");
			return 1;
		}
		cur_efa_addr = cur_efa_addr->next;
	}

	return 0;
}

/*
 * Insert address translation in core av & in hash. Return 1 on successful
 * insertion regardless of whether it is in the hash table or not, 0 if the
 * lower layer av insert fails.
 *
 * If shm transfer is enabled and the addr comes from local peer,
 * 1. convert addr to format 'gid_qpn', which will be set as shm's ep name later.
 * 2. insert gid_qpn into shm's av
 * 3. store returned fi_addr from shm into the hash table
 */
int rxr_av_insert_rdm_addr(struct rxr_av *av, const void *addr,
			   fi_addr_t *rdm_fiaddr, uint64_t flags,
			   void *context)
{
	struct rxr_av_entry *av_entry;
	fi_addr_t shm_fiaddr;
	struct rxr_peer *peer;
	struct rxr_ep *rxr_ep;
	struct util_ep *util_ep;
	struct dlist_entry *ep_list_entry;
	char smr_name[NAME_MAX];
	int ret = 1;

	fastlock_acquire(&av->util_av.lock);

	HASH_FIND(hh, av->av_map, addr, av->rdm_addrlen, av_entry);

	if (av_entry) {
		*rdm_fiaddr = (fi_addr_t)av_entry->rdm_addr;
		goto find_out;
	}
	ret = fi_av_insert(av->rdm_av, addr, 1, rdm_fiaddr, flags, context);
	if (OFI_UNLIKELY(ret != 1)) {
		FI_DBG(&rxr_prov, FI_LOG_AV,
		       "Error in inserting address: %s\n", fi_strerror(-ret));
		goto out;
	}
	av_entry = calloc(1, sizeof(*av_entry));
	if (OFI_UNLIKELY(!av_entry)) {
		ret = -FI_ENOMEM;
		FI_WARN(&rxr_prov, FI_LOG_AV,
			"Failed to allocate memory for av_entry\n");
		goto out;
	}
	memcpy(av_entry->addr, addr, av->rdm_addrlen);
	av_entry->rdm_addr = *(uint64_t *)rdm_fiaddr;

	/* If peer is local, insert the address into shm provider's av */
	if (rxr_env.enable_shm_transfer && rxr_is_local_peer(av, addr)) {
		ret = rxr_ep_efa_addr_to_str(addr, smr_name);
		if (ret != FI_SUCCESS)
			goto out;

		ret = fi_av_insert(av->shm_rdm_av, smr_name, 1, &shm_fiaddr, flags, context);
		if (OFI_UNLIKELY(ret != 1)) {
			FI_DBG(&rxr_prov, FI_LOG_AV, "Failed to insert address to shm provider's av: %s\n",
			       fi_strerror(-ret));
			goto out;
		}
		FI_DBG(&rxr_prov, FI_LOG_AV,
			"Insert %s to shm provider's av. addr = %" PRIu64 " rdm_fiaddr = %" PRIu64
			" shm_rdm_fiaddr = %" PRIu64 "\n", smr_name, *(uint64_t *)addr, *rdm_fiaddr, shm_fiaddr);
		av_entry->local_mapping = 1;
		av_entry->shm_rdm_addr = shm_fiaddr;

		/*
		 * Walk through all the EPs that bound to the AV,
		 * update is_local flag and shm fi_addr_t in corresponding peer structure
		 */
		dlist_foreach(&av->util_av.ep_list, ep_list_entry) {
			util_ep = container_of(ep_list_entry, struct util_ep, av_entry);
			rxr_ep = container_of(util_ep, struct rxr_ep, util_ep);
			peer = rxr_ep_get_peer(rxr_ep, *rdm_fiaddr);
			peer->shm_fiaddr = shm_fiaddr;
			peer->is_local = 1;
		}
	}

	HASH_ADD(hh, av->av_map, addr, av->rdm_addrlen, av_entry);

find_out:
	FI_DBG(&rxr_prov, FI_LOG_AV,
	       "addr = %" PRIu64 " rdm_fiaddr =  %" PRIu64 "\n",
	       *(uint64_t *)addr, *rdm_fiaddr);
out:
	fastlock_release(&av->util_av.lock);
	return ret;
}

static int rxr_av_insert(struct fid_av *av_fid, const void *addr,
			 size_t count, fi_addr_t *fi_addr, uint64_t flags,
			 void *context)
{
	struct rxr_av *av;
	fi_addr_t fi_addr_res;
	int i = 0, ret = 0, success_cnt = 0;

	/*
	 * Providers are allowed to ignore FI_MORE. FI_SYNC_ERR is not
	 * supported.
	 */
	flags &= ~FI_MORE;

	if (flags)
		return -FI_ENOSYS;

	av = container_of(av_fid, struct rxr_av, util_av.av_fid);

	if (av->util_av.count < av->rdm_av_used + count) {
		FI_WARN(&rxr_prov, FI_LOG_AV,
			"AV insert failed. Expect inserting %zu AV entries, but only %zu available\n",
			count, av->util_av.count - av->rdm_av_used);
		if (av->util_av.eq)
			ofi_av_write_event(&av->util_av, i, FI_ENOMEM, context);
		goto out;
	}

	for (; i < count; i++, addr = (uint8_t *)addr + av->rdm_addrlen) {
		ret = rxr_av_insert_rdm_addr(av, addr, &fi_addr_res,
					     flags, context);
		if (ret != 1)
			break;

		if (fi_addr)
			fi_addr[i] = fi_addr_res;

		success_cnt++;
	}

	av->rdm_av_used += success_cnt;

out:
	/* cancel remaining request and log to event queue */
	for (; i < count ; i++) {
		if (av->util_av.eq)
			ofi_av_write_event(&av->util_av, i, FI_ECANCELED,
					   context);
		if (fi_addr)
			fi_addr[i] = FI_ADDR_NOTAVAIL;
	}

	/* update success to event queue */
	if (av->util_av.eq)
		ofi_av_write_event(&av->util_av, success_cnt, 0, context);

	return success_cnt;
}

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
	struct rxr_av_entry *av_entry;
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

		av->rdm_av_used--;
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
	.insert = rxr_av_insert,
	.insertsvc = rxr_av_insertsvc,
	.insertsym = rxr_av_insertsym,
	.remove = rxr_av_remove,
	.lookup = rxr_av_lookup,
	.straddr = rxr_av_straddr,
};

static int rxr_av_close(struct fid *fid)
{
	struct rxr_av *av;
	struct rxr_av_entry *curr_av_entry, *tmp;
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
		attr->count = RXR_MIN_AV_SIZE;
	else
		attr->count = MAX(attr->count, RXR_MIN_AV_SIZE);

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
		assert(rxr_env.shm_av_size <= RXR_SHM_MAX_AV_COUNT);
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
