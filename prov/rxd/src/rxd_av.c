/*
 * Copyright (c) 2015-2016 Intel Corporation. All rights reserved.
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

#include "rxd.h"

fi_addr_t rxd_av_get_dg_addr(struct rxd_av *av, fi_addr_t fi_addr)
{
	uint64_t *dg_idx;
	dg_idx = ofi_av_get_addr(&av->util_av, (int) fi_addr);
	return *dg_idx;
}

fi_addr_t rxd_av_get_fi_addr(struct rxd_av *av, fi_addr_t dg_addr)
{
	return (fi_addr_t) ofi_av_lookup_index(&av->util_av,
					      &dg_addr, (int) dg_addr);
}

int rxd_av_insert_dg_av(struct rxd_av *av, const void *addr)
{
	int ret;
	fastlock_acquire(&av->lock);
	ret = fi_av_insert(av->dg_av, addr, 1, NULL, 0, NULL);
	if (ret != 1)
		goto out;
	av->dg_av_used++;
out:
	fastlock_release(&av->lock);
	return ret;
}

int rxd_av_dg_reverse_lookup(struct rxd_av *av, uint64_t start_idx,
			      const void *addr, size_t addrlen, uint64_t *idx)
{
	int ret;
	size_t i, len;
	void *curr_addr;

	len = addrlen;
	curr_addr = calloc(1, av->addrlen);
	if (!curr_addr)
		return -FI_ENOMEM;

	for (i = 0; i < av->dg_av_used; i++) {
		ret = fi_av_lookup(av->dg_av, (i + start_idx) % av->dg_av_used,
				   curr_addr, &len);
		if (ret)
			continue;
		if (len == addrlen && memcmp(curr_addr, addr, len) == 0) {
			*idx = (i + start_idx) % av->dg_av_used;
			goto out;
		}
	}
	ret = -FI_ENODATA;
out:
	free(curr_addr);
	return ret;
}

size_t rxd_av_insert_check(struct rxd_av *av, const void *addr, size_t count,
			   fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	size_t i, success_cnt = 0;
	int ret, index;
	void *curr_addr;
	uint64_t dg_av_idx;

	for (i = 0; i < count; i++) {
		curr_addr = (char *) addr + av->addrlen * i;
		ret = rxd_av_dg_reverse_lookup(av, i, curr_addr, av->addrlen, &dg_av_idx);
		if (ret == -FI_ENODATA) {
			ret = fi_av_insert(av->dg_av, curr_addr, 1, &dg_av_idx,
					   flags, context);
			if (ret <= 0) {
				if (av->util_av.eq)
					ofi_av_write_event(&av->util_av, i,
							   (ret == 0) ? FI_EINVAL : -ret,
							   context);
				if (fi_addr)
					fi_addr[i] = FI_ADDR_NOTAVAIL;
				continue;
			}
		}

		ret = ofi_av_insert_addr(&av->util_av, &dg_av_idx, dg_av_idx, &index);
		if (ret) {
			if (av->util_av.eq)
				ofi_av_write_event(&av->util_av, i, -ret, context);
		} else {
			success_cnt++;
		}
		
		if (fi_addr)
			fi_addr[i] = (ret == 0) ? index : FI_ADDR_NOTAVAIL;
	}
	av->dg_av_used += success_cnt;
	if (av->util_av.eq) {
		ofi_av_write_event(&av->util_av, success_cnt, 0, context);
		ret = 0;
	} else {
		ret = success_cnt;
	}
	return ret;
}

size_t rxd_av_insert_fast(struct rxd_av *av, const void *addr, size_t count,
			   fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	size_t i, num, ret, success_cnt = 0;
	int index;
	fi_addr_t *fi_addrs;

	fi_addrs = calloc(count, sizeof(fi_addr_t));
	if (!fi_addrs)
		return -FI_ENOMEM;

	num = fi_av_insert(av->dg_av, addr, count, fi_addrs, flags, context);
	if (num != count) {
		free(fi_addrs);
		return rxd_av_insert_check(av, addr, count, fi_addr,
					    flags, context);
	}

	for (i = 0; i < num; i++) {
		ret = ofi_av_insert_addr(&av->util_av, &fi_addrs[i],
					 fi_addrs[i], &index);
		if (ret) {
			if (av->util_av.eq)
				ofi_av_write_event(&av->util_av, i, -ret, context);
		} else {
			success_cnt++;
		}

		if (fi_addr)
			fi_addr[i] = (ret == 0) ? index : FI_ADDR_NOTAVAIL;
	}

	free(fi_addrs);
	av->dg_av_used += success_cnt;

	if (av->util_av.eq) {
		ofi_av_write_event(&av->util_av, success_cnt, 0, context);
		ret = 0;
	} else {
		ret = success_cnt;
	}

	return ret;
}

static int rxd_av_insert(struct fid_av *av_fid, const void *addr, size_t count,
			fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	struct rxd_av *av;
	av = container_of(av_fid, struct rxd_av, util_av.av_fid);
	return av->dg_av_used ?
		rxd_av_insert_check(av, addr, count, fi_addr, flags, context) :
		rxd_av_insert_fast(av, addr, count, fi_addr, flags, context);
}

static int rxd_av_insertsvc(struct fid_av *av, const char *node,
			   const char *service, fi_addr_t *fi_addr,
			   uint64_t flags, void *context)
{
	return -FI_ENOSYS;
}

static int rxd_av_insertsym(struct fid_av *av_fid, const char *node, size_t nodecnt,
			   const char *service, size_t svccnt, fi_addr_t *fi_addr,
			   uint64_t flags, void *context)
{
	return -FI_ENOSYS;
}

static int rxd_av_remove(struct fid_av *av_fid, fi_addr_t *fi_addr, size_t count,
			uint64_t flags)
{
	int ret = 0;
	size_t i;
	fi_addr_t dg_idx;
	struct rxd_av *av;

	av = container_of(av_fid, struct rxd_av, util_av.av_fid);
	for (i = 0; i < count; i++) {
		dg_idx = rxd_av_get_dg_addr(av, fi_addr[i]);
		ret = fi_av_remove(av->dg_av, &dg_idx, 1, flags);
		if (ret)
			break;
		av->dg_av_used--;
	}
	return ret;
}

static const char * rxd_av_straddr(struct fid_av *av, const void *addr, char *buf, size_t *len)
{
	struct rxd_av *rxd_av;
	rxd_av = container_of(av, struct rxd_av, util_av.av_fid);
	return rxd_av->dg_av->ops->straddr(rxd_av->dg_av, addr, buf, len);
}

int rxd_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr, size_t *addrlen)
{
	struct rxd_av *rxd_av;
	fi_addr_t dg_addr;
	rxd_av = container_of(av, struct rxd_av, util_av.av_fid);
	dg_addr = rxd_av_get_dg_addr(rxd_av, fi_addr);
	return fi_av_lookup(rxd_av->dg_av, dg_addr, addr, addrlen);
}

static struct fi_ops_av rxd_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = rxd_av_insert,
	.insertsvc = rxd_av_insertsvc,
	.insertsym = rxd_av_insertsym,
	.remove = rxd_av_remove,
	.lookup = rxd_av_lookup,
	.straddr = rxd_av_straddr,
};

int rxd_av_close(struct fid *fid)
{
	int ret;
	struct rxd_av *av;
	av = container_of(fid, struct rxd_av, util_av.av_fid);
	ret = fi_close(&av->dg_av->fid);
	if (ret)
		return ret;

	ret = ofi_av_close(&av->util_av);
	if (ret)
		return ret;

	fastlock_destroy(&av->lock);
	free(av);
	return 0;
}

int rxd_av_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	return ofi_av_bind(fid, bfid, flags);
}

static struct fi_ops rxd_av_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxd_av_close,
	.bind = rxd_av_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int rxd_av_create(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		   struct fid_av **av_fid, void *context)
{
	int ret;
	struct rxd_av *av;
	struct rxd_domain *domain;
	struct util_av_attr util_attr;
	struct fi_av_attr av_attr;

	if (attr && attr->name)
		return -FI_ENOSYS;

	domain = container_of(domain_fid, struct rxd_domain, util_domain.domain_fid);
	av = calloc(1, sizeof(*av));
	if (!av)
		return -FI_ENOMEM;

	util_attr.addrlen = sizeof(fi_addr_t);
	util_attr.overhead = attr ? attr->count : 0;
	util_attr.flags = FI_SOURCE;
	av->size = attr ? attr->count : RXD_AV_DEF_COUNT;
	ret = ofi_av_init(&domain->util_domain, attr, &util_attr,
			 &av->util_av, context);
	if (ret)
		goto err1;

	av->size = av->util_av.count;
	if (attr)
		av_attr = *attr;
	av_attr.type = FI_AV_TABLE;
	av_attr.count = 0;
	av_attr.flags = 0;
	ret = fi_av_open(domain->dg_domain, &av_attr, &av->dg_av, context);
	if (ret)
		goto err2;

	fastlock_init(&av->lock);
	av->addrlen = domain->addrlen;

	*av_fid = &av->util_av.av_fid;
	(*av_fid)->fid.fclass = FI_CLASS_AV;
	(*av_fid)->fid.ops = &rxd_av_fi_ops;
	(*av_fid)->ops = &rxd_av_ops;
	return 0;

err2:
	ofi_av_close(&av->util_av);
err1:
	free(av);
	return ret;
}
