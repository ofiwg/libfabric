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

/* TODO: find more deterministic hash function */
static inline int
fi_ibv_dgram_av_slot(struct util_av *av, const struct ofi_ib_ud_ep_name *ep_name)
{
	return (ep_name->gid.global.subnet_prefix + ep_name->lid + ep_name->qpn)
		% av->hash.slots;
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
	struct fi_ibv_domain *domain;
	struct ofi_ib_ud_ep_name *ep_name;
	struct ibv_ah *ah;

	domain = container_of(&av->util_av.domain->domain_fid,
			      struct fi_ibv_domain,
			      util_domain.domain_fid);
	if (!domain) {
		ret = -FI_EINVAL;
		goto fn1;
	}

	if (fi_ibv_dgram_av_is_addr_valid(av, addr)) {
		struct fi_ibv_dgram_av_entry av_entry;
		struct ibv_ah_attr ah_attr = {
			.is_global     = 0,
			.dlid          = ((struct ofi_ib_ud_ep_name *)addr)->lid,
			.sl            = ((struct ofi_ib_ud_ep_name *)addr)->sl,
			.src_path_bits = 0,
			.port_num      = 1,
		};
		ep_name = calloc(1, sizeof(*ep_name));
		if (OFI_UNLIKELY(!ep_name)) {
			ret = -FI_ENOMEM;
			goto fn1;
		}
		memcpy(ep_name, addr, sizeof(*ep_name));
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
			goto fn2;
		}

		av_entry.ah = ah;
		av_entry.addr = ep_name;

		ret = ofi_av_insert_addr(&av->util_av, &av_entry,
					 fi_ibv_dgram_av_slot(&av->util_av, ep_name),
					 &index);
		if (ret)
			goto fn3;
		if (fi_addr)
			*fi_addr = index;
	} else {
		ret = -FI_EADDRNOTAVAIL;
		VERBS_WARN(FI_LOG_AV, "Invalid address\n");
		goto fn1;
	}

	return ret;
fn3:
	ibv_destroy_ah(ah);
fn2:
	free(ep_name);
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
		struct ofi_ib_ud_ep_name *ep_name;

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
		ep_name = av_entry->addr;
		slot = fi_ibv_dgram_av_slot(&av->util_av, av_entry->addr);
		fastlock_acquire(&av->util_av.lock);
		ret = ofi_av_remove_addr(&av->util_av, slot, index);
		fastlock_release(&av->util_av.lock);
		if (ret)
			VERBS_WARN(FI_LOG_AV,
				   "Removal of fi_addr %d failed\n",
				   index);
		free(ep_name);
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

	assert(av_fid->fclass == FI_CLASS_AV);
	if (av_fid->fclass != FI_CLASS_AV)
		return -FI_EINVAL;

	av = container_of(av_fid, struct fi_ibv_dgram_av, util_av.av_fid.fid);
	if (!av)
		return -FI_EINVAL;

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
		.flags = OFI_AV_HASH,
		.addrlen = sizeof(struct fi_ibv_dgram_av_entry),
	};

	if (attr->type == FI_AV_UNSPEC)
		attr->type = FI_AV_MAP;

	ret = ofi_av_init(&domain->util_domain, attr, &util_attr,
			  &av->util_av, context);
	if (ret)
		goto err1;

	*av_fid = &av->util_av.av_fid;
	(*av_fid)->fid.ops = &fi_ibv_dgram_fi_ops;
	(*av_fid)->ops = &fi_ibv_dgram_av_ops;

	return FI_SUCCESS;
err1:
	free(av);
	return ret;
}
