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

const size_t fi_ibv_dgram_av_entry_size = sizeof(struct fi_ibv_dgram_av_entry);

static inline
int fi_ibv_dgram_av_fi_addr_cmp(void *addr1, void *addr2)
{
    return (*(int *)addr1 < *(int *)addr2) ?
		-1 :
		(*(int *)addr1 > *(int *)addr2);
}

/* Jenknin's one-at-a-time hash */
static inline
int fi_ibv_dgram_one_at_a_time_hash(const int8_t *key, size_t len)
{
	size_t i = 0;
	int hash = 0;

	while (i != len) {
		hash += key[i++];
		hash += hash << 10;
		hash ^= hash >> 6;
	}
	hash += hash << 3;
	hash ^= hash >> 11;
	hash += hash << 15;

	return hash;	
}

static inline
int fi_ibv_dgram_av_slot(const struct fi_ibv_dgram_av_entry *av_entry)
{
	return fi_ibv_dgram_one_at_a_time_hash(
			(int8_t *)av_entry, fi_ibv_dgram_av_entry_size);
}

static inline int fi_ibv_dgram_av_is_addr_valid(struct fi_ibv_dgram_av *av,
						const void *addr)
{
	const struct ofi_ib_ud_ep_name *check_name = addr;
	return (check_name->lid > 0);
}

static inline
int fi_ibv_dgram_verify_av_insert(struct util_av *av, uint64_t flags)
{
	if ((av->flags & FI_EVENT) && !av->eq) {
		VERBS_WARN(FI_LOG_AV, "No EQ bound to AV\n");
		return -FI_ENOEQ;
	}

	if (flags & ~(FI_MORE)) {
		VERBS_WARN(FI_LOG_AV, "Unsupported flags\n");
		return -FI_ENOEQ;
	}

	return FI_SUCCESS;
}

static int fi_ibv_dgram_av_insert_addr(struct fi_ibv_dgram_av *av,
				       const void *addr,
				       fi_addr_t *fi_addr,
				       void *context)
{
	int ret, index = -1;
	int *fi_addr_index;
	struct fi_ibv_domain *domain;
	struct fi_ibv_dgram_av_entry *av_entry;
	struct ibv_ah *ah;

	domain = container_of(&av->util_av.domain->domain_fid,
			      struct fi_ibv_domain,
			      util_domain.domain_fid);
	if (!domain) {
		ret = -FI_EINVAL;
		goto fn1;
	}

	if (fi_ibv_dgram_av_is_addr_valid(av, addr)) {
		struct ofi_ib_ud_ep_name *ep_name =
			(struct ofi_ib_ud_ep_name *)addr;
		struct ibv_ah_attr ah_attr = {
			.is_global     = 0,
			.dlid          = ep_name->lid,
			.sl            = ep_name->sl,
			.src_path_bits = 0,
			.port_num      = 1,
		};
		if (ep_name->gid.global.interface_id) {
			ah_attr.is_global = 1;
			ah_attr.grh.hop_limit = 64;
			ah_attr.grh.dgid = ep_name->gid;
			ah_attr.grh.sgid_index = 0;
		}
		ah = ibv_create_ah(domain->pd, &ah_attr);
		if (!ah) {
			ret = -errno;
			VERBS_WARN(FI_LOG_AV, "Unable to create "
				   "Address Handle, errno - %d\n", errno);
			goto fn1;
		}

		av_entry = calloc(1, sizeof(*av_entry));
		if (!av_entry) {
			ret = -FI_ENOMEM;
			goto fn2;
		}
		av_entry->ah = ah;
		av_entry->addr = ep_name;

		ret = ofi_av_insert_addr(&av->util_av, av_entry,
					 fi_ibv_dgram_av_slot(av_entry),
					 &index);
		if (!ret) {
			if (fi_addr)
				*fi_addr = index;
			if (rbtFind(av->addr_map, &index)) {
				ret = -FI_EADDRINUSE;
				goto fn3;
			}

			fi_addr_index = calloc(1, sizeof(*fi_addr_index));
			if (!fi_addr_index) {
				ret = -FI_ENOMEM;
				goto fn3;
			}
			*fi_addr_index = index;
			if (rbtInsert(av->addr_map, fi_addr_index, av_entry)) {
				ret = -FI_ENOMEM;
				goto fn4;
			}
		} else {
			goto fn3;
		}
	} else {
		ret = -FI_EADDRNOTAVAIL;
		VERBS_WARN(FI_LOG_AV, "Invalid address\n");
		goto fn1;
	}

	return ret;
fn4:
	free(fi_addr_index);
fn3:
	free(av_entry);
fn2:
	ibv_destroy_ah(ah);
fn1:
	if (fi_addr)
		*fi_addr = FI_ADDR_NOTAVAIL;
	return ret;
}

static int fi_ibv_dgram_av_insert(struct fid_av *av_fid, const void *addr,
				  size_t count, fi_addr_t *fi_addr,
				  uint64_t flags, void *context)
{
	struct fi_ibv_dgram_av *av;
	int ret, success_cnt = 0;
	size_t i;

	assert(av_fid->fid.fclass == FI_CLASS_AV);
	if (av_fid->fid.fclass != FI_CLASS_AV)
		return -FI_EINVAL;

	av = container_of(av_fid, struct fi_ibv_dgram_av, util_av.av_fid);
	if (!av)
		return -FI_EINVAL;
	ret = fi_ibv_dgram_verify_av_insert(&av->util_av, flags);
	if (ret)
		return ret;

	VERBS_DBG(FI_LOG_AV, "Inserting %"PRIu64" addresses\n", count);
	for (i = 0; i < count; i++) {
		ret = fi_ibv_dgram_av_insert_addr(
				av, (struct ofi_ib_ud_ep_name *)addr + i,
				fi_addr ? &fi_addr[i] : NULL, context);
		if (!ret)
			success_cnt++;
		else if (av->util_av.eq)
			ofi_av_write_event(&av->util_av, i, -ret, context);
	}

	VERBS_DBG(FI_LOG_AV, "%"PRIu64" addresses successful\n", count);
	if (av->util_av.eq) {
		ofi_av_write_event(&av->util_av, success_cnt, 0, context);
		ret = 0;
	} else {
		ret = success_cnt;
	}
	return ret;
}

static int fi_ibv_dgram_av_remove(struct fid_av *av_fid, fi_addr_t *fi_addr,
				  size_t count, uint64_t flags)
{
	struct fi_ibv_dgram_av *av;
	int ret, slot, i, index;
	RbtIterator it;

	assert(av_fid->fid.fclass == FI_CLASS_AV);
	if (av_fid->fid.fclass != FI_CLASS_AV)
		return -FI_EINVAL;

	av = container_of(av_fid, struct fi_ibv_dgram_av, util_av.av_fid);
	if (!av)
		return -FI_EINVAL;
	ret = fi_ibv_dgram_verify_av_insert(&av->util_av, flags);
	if (ret)
		return ret;

	for (i = count - 1; i >= 0; i--) {
	    struct fi_ibv_dgram_av_entry *av_entry;
		index = (int)fi_addr[i];
		av_entry = ofi_av_get_addr(&av->util_av, index);
		if (!av_entry) {
			VERBS_WARN(FI_LOG_AV, "Unable to find address\n");
			return -FI_ENOENT;
		}
		ret = ibv_destroy_ah(av_entry->ah);
		if (ret)
			VERBS_WARN(FI_LOG_AV,
				   "AH Destroying of fi_addr %d failed "
				   "with status - %d\n", index, ret);
			
		slot = fi_ibv_dgram_av_slot(av_entry);
		fastlock_acquire(&av->util_av.lock);
		ret = ofi_av_remove_addr(&av->util_av, slot, index);
		fastlock_release(&av->util_av.lock);
		if (ret)
			VERBS_WARN(FI_LOG_AV,
				   "Removal of fi_addr %d failed\n",
				   index);

		it = rbtFind(av->addr_map, &index);
		if (it) {
			struct fi_ibv_dgram_av_entry *av_entry_elem = NULL;
			int *fi_addr_index;
			rbtKeyValue(av->addr_map, it,
				    (void **)&fi_addr_index,
				    (void **)&av_entry_elem);
			free(fi_addr_index);
			free(av_entry_elem);
			rbtErase(av->addr_map, it);
		}
	}
	return FI_SUCCESS;
}

static inline
int fi_ibv_dgram_av_lookup(struct fid_av *av_fid, fi_addr_t fi_addr,
			   void *addr, size_t *addrlen)
{
	struct fi_ibv_dgram_av *av;
	struct fi_ibv_dgram_av_entry *av_entry;

	assert(av_fid->fid.fclass == FI_CLASS_AV);
	if (av_fid->fid.fclass != FI_CLASS_AV)
		return -FI_EINVAL;

	av = container_of(av_fid, struct fi_ibv_dgram_av, util_av.av_fid);
	if (!av)
		return -FI_EINVAL;

	av_entry = fi_ibv_dgram_av_lookup_av_entry(av, (int)fi_addr);
	if (!av_entry)
		return -FI_ENOENT;

	memcpy(addr, av_entry->addr, MIN(*addrlen, av->util_av.addrlen));
	*addrlen = av->util_av.addrlen;
	return FI_SUCCESS;
}

static inline
const char *fi_ibv_dgram_av_straddr(struct fid_av *av, const void *addr,
				    char *buf, size_t *len)
{
	return ofi_straddr(buf, len, FI_ADDR_IB_UD, addr);
}

static int fi_ibv_dgram_av_close(struct fid *av_fid)
{
	int ret;
	struct fi_ibv_dgram_av *av;
	RbtIterator iter;
	fi_addr_t *fi_addr;
	struct fi_ibv_dgram_av_entry *av_entry;

	assert(av_fid->fclass == FI_CLASS_AV);
	if (av_fid->fclass != FI_CLASS_AV)
		return -FI_EINVAL;

	av = container_of(av_fid, struct fi_ibv_dgram_av, util_av.av_fid.fid);
	if (!av)
		return -FI_EINVAL;

	for (iter = rbtBegin(av->addr_map);
	     iter != rbtEnd(av->addr_map);
	     iter = rbtNext(av->addr_map, iter)) {
		rbtKeyValue(av->addr_map, iter, (void **)&fi_addr,
			    (void **)&av_entry);
		fi_ibv_dgram_av_remove(&av->util_av.av_fid, fi_addr,
				       1, 0);
	}

	ret = ofi_av_close(&av->util_av);
	if (ret)
		return ret;

	free(av);

	return FI_SUCCESS;
}

static struct fi_ops fi_ibv_dgram_fi_ops = {
	.size		= sizeof(fi_ibv_dgram_fi_ops),
	.close		= fi_ibv_dgram_av_close,
	.bind		= ofi_av_bind,
	.control	= fi_no_control,
	.ops_open	= fi_no_ops_open,
};

static struct fi_ops_av fi_ibv_dgram_av_ops = {
	.size		= sizeof(fi_ibv_dgram_av_ops),
	.insert		= fi_ibv_dgram_av_insert,
	.insertsvc	= fi_no_av_insertsvc,
	.insertsym	= fi_no_av_insertsym,
	.remove		= fi_ibv_dgram_av_remove,
	.lookup		= fi_ibv_dgram_av_lookup,
	.straddr	= fi_ibv_dgram_av_straddr,
};

int fi_ibv_dgram_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr,
			 struct fid_av **av_fid, void *context)
{
	struct fi_ibv_domain *domain;
	struct fi_ibv_dgram_av *av;
	int ret;

	if (!attr || domain_fid->fid.fclass != FI_CLASS_DOMAIN)
		return -FI_EINVAL;

	av = calloc(1, sizeof(*av));
	if (!av)
		return -FI_ENOMEM;

	domain = container_of(domain_fid, struct fi_ibv_domain,
			      util_domain.domain_fid);
	if (!domain) {
		ret = -FI_EINVAL;
		goto err1;
	}

	assert(domain->ep_type == FI_EP_DGRAM);
	
	struct util_av_attr util_attr = {
		.overhead = attr->count >> 1,
		.flags = ((domain->util_domain.info_domain_caps & FI_SOURCE) ?
			  OFI_AV_HASH : 0),
		.addrlen = sizeof(struct ofi_ib_ud_ep_name),
	};

	if (attr->type == FI_AV_UNSPEC)
		attr->type = FI_AV_MAP;

	ret = ofi_av_init(&domain->util_domain, attr, &util_attr,
			  &av->util_av, context);
	if (ret)
		goto err1;

	av->addr_map = rbtNew(fi_ibv_dgram_av_fi_addr_cmp);
	if (!av->addr_map) {
		ret = -FI_ENOMEM;
		goto err2;
	}

	*av_fid = &av->util_av.av_fid;
	(*av_fid)->fid.ops = &fi_ibv_dgram_fi_ops;
	(*av_fid)->ops = &fi_ibv_dgram_av_ops;

	return FI_SUCCESS;
err2:
	(void) ofi_av_close(&av->util_av);
err1:
	free(av);
	return ret;
}
