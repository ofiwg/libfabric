/*
 * Copyright (c) 2015-2017 Intel Corporation. All rights reserved.
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
#include <inttypes.h>


/*
 * The RXD code is agnostic wrt the datagram address format, but we need
 * to know the size of the address in order to iterate over them.  Because
 * the datagram AV may be configured for asynchronous operation, open a
 * temporary one to insert/lookup the address to get the size.  I agree it's
 * goofy.
 */
static int rxd_av_set_addrlen(struct rxd_av *av, const void *addr)
{
	struct rxd_domain *domain;
	struct fid_av *tmp_av;
	struct fi_av_attr attr;
	uint8_t tmp_addr[RXD_MAX_DGRAM_ADDR];
	size_t len;
	int ret;

	FI_INFO(&rxd_prov, FI_LOG_AV, "determine dgram address len\n");
	memset(&attr, 0, sizeof attr);
	attr.type = FI_AV_TABLE;
	attr.count = 1;

	domain = container_of(av->util_av.domain, struct rxd_domain, util_domain);
	ret = fi_av_open(domain->dg_domain, &attr, &tmp_av, NULL);
	if (ret) {
		FI_WARN(&rxd_prov, FI_LOG_AV, "failed to open av: %d (%s)\n",
			-ret, fi_strerror(-ret));
		return ret;
	}

	ret = fi_av_insert(tmp_av, addr, 1, NULL, 0, NULL);
	if (ret != 1) {
		FI_WARN(&rxd_prov, FI_LOG_AV, "addr insert failed: %d (%s)\n",
			-ret, fi_strerror(-ret));
		goto close;
	}

	len = sizeof tmp_addr;
	ret = fi_av_lookup(tmp_av, 0, tmp_addr, &len);
	if (ret) {
		FI_WARN(&rxd_prov, FI_LOG_AV, "addr lookup failed: %d (%s)\n",
			-ret, fi_strerror(-ret));
		goto close;
	}

	FI_INFO(&rxd_prov, FI_LOG_AV, "set dgram address len: %zu\n", len);
	av->dg_addrlen = len;
close:
	fi_close(&tmp_av->fid);
	return ret;
}

fi_addr_t rxd_av_dg_addr(struct rxd_av *av, fi_addr_t fi_addr)
{
	uint64_t *dg_idx;

	dg_idx = ofi_av_get_addr(&av->util_av, (int) fi_addr);
	return *dg_idx;
}

fi_addr_t rxd_av_fi_addr(struct rxd_av *av, fi_addr_t dg_fiaddr)
{
	int ret;

	ret = ofi_av_lookup_index(&av->util_av, &dg_fiaddr, (int) dg_fiaddr);
	return (ret < 0) ? FI_ADDR_UNSPEC : ret;
}

int rxd_av_dg_reverse_lookup(struct rxd_av *av, uint64_t start_idx,
			      const void *addr, fi_addr_t *dg_fiaddr)
{
	uint8_t curr_addr[RXD_MAX_DGRAM_ADDR];
	size_t i, len;
	int ret;

	for (i = 0; i < (size_t) av->dg_av_used; i++) {
		len = sizeof curr_addr;
		ret = fi_av_lookup(av->dg_av, (i + start_idx) % av->dg_av_used,
				   curr_addr, &len);
		if (!ret) {
			*dg_fiaddr = (i + start_idx) % av->dg_av_used;
			FI_DBG(&rxd_prov, FI_LOG_AV, "found: %" PRIu64 "\n",
				*dg_fiaddr);
			return 0;
		}
	}
	FI_DBG(&rxd_prov, FI_LOG_AV, "addr not found\n");
	return -FI_ENODATA;
}

int rxd_av_insert_dg_addr(struct rxd_av *av, uint64_t hint_index,
			  const void *addr, fi_addr_t *dg_fiaddr)
{
	int ret;

	fastlock_acquire(&av->util_av.lock);
	if (!av->dg_addrlen) {
		ret = rxd_av_set_addrlen(av, addr);
		if (ret)
			goto out;
		ret = -FI_ENODATA;
	} else {
		ret = rxd_av_dg_reverse_lookup(av, hint_index, addr, dg_fiaddr);
	}

	if (ret == -FI_ENODATA) {
		ret = fi_av_insert(av->dg_av, addr, 1, dg_fiaddr, 0, NULL);
		if (ret == 1) {
			av->dg_av_used++;
			ret = 0;
		}
	}
out:
	fastlock_release(&av->util_av.lock);
	return ret;
}

static int rxd_av_insert(struct fid_av *av_fid, const void *addr, size_t count,
			fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	struct rxd_av *av;
	int i = 0, index, ret = 0, success_cnt = 0, lookup = 1;
	uint64_t dg_fiaddr;

	av = container_of(av_fid, struct rxd_av, util_av.av_fid);
	fastlock_acquire(&av->util_av.lock);
	if (!av->dg_addrlen) {
		ret = rxd_av_set_addrlen(av, addr);
		if (ret)
			goto out;
		/* Skip lookups if this is the first insertion call.  */
		lookup = 0;
	}

	for (; i < count; i++, addr = (uint8_t *) addr + av->dg_addrlen) {
		ret = lookup ? rxd_av_dg_reverse_lookup(av, i, addr, &dg_fiaddr) :
				-FI_ENODATA;
		if (ret) {
			ret = fi_av_insert(av->dg_av, addr, 1, &dg_fiaddr,
					   flags, context);
			if (ret != 1)
				break;
		}

		ret = ofi_av_insert_addr(&av->util_av, &dg_fiaddr, dg_fiaddr, &index);
		if (ret)
			break;

		success_cnt++;
		if (fi_addr)
			fi_addr[i] = index;
	}

	if (ret) {
		FI_WARN(&rxd_prov, FI_LOG_AV,
			"failed to insert address %d: %d (%s)\n",
			i, -ret, fi_strerror(-ret));
		if (av->util_av.eq)
			ofi_av_write_event(&av->util_av, i, -ret, context);
		if (fi_addr)
			fi_addr[i] = FI_ADDR_NOTAVAIL;
		i++;
	}
out:
	av->dg_av_used += success_cnt;
	fastlock_release(&av->util_av.lock);

	for (; i < count; i++) {
		if (av->util_av.eq)
			ofi_av_write_event(&av->util_av, i, FI_ECANCELED, context);
		if (fi_addr)
			fi_addr[i] = FI_ADDR_NOTAVAIL;
	}

	if (av->util_av.eq) {
		ofi_av_write_event(&av->util_av, success_cnt, 0, context);
		return 0;
	}

	return success_cnt;
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
	fi_addr_t dg_fiaddr;
	struct rxd_av *av;

	av = container_of(av_fid, struct rxd_av, util_av.av_fid);
	fastlock_acquire(&av->util_av.lock);
	for (i = 0; i < count; i++) {
		dg_fiaddr = rxd_av_dg_addr(av, fi_addr[i]);
		ret = fi_av_remove(av->dg_av, &dg_fiaddr, 1, flags);
		if (ret)
			break;
		av->dg_av_used--;
	}
	fastlock_release(&av->util_av.lock);
	return ret;
}

static const char *rxd_av_straddr(struct fid_av *av, const void *addr,
				  char *buf, size_t *len)
{
	struct rxd_av *rxd_av;
	rxd_av = container_of(av, struct rxd_av, util_av.av_fid);
	return rxd_av->dg_av->ops->straddr(rxd_av->dg_av, addr, buf, len);
}

static int rxd_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr,
			 size_t *addrlen)
{
	struct rxd_av *rxd_av;
	fi_addr_t dg_addr;

	rxd_av = container_of(av, struct rxd_av, util_av.av_fid);
	dg_addr = rxd_av_dg_addr(rxd_av, fi_addr);
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

static int rxd_av_close(struct fid *fid)
{
	struct rxd_av *av;
	int ret;

	av = container_of(fid, struct rxd_av, util_av.av_fid);
	ret = fi_close(&av->dg_av->fid);
	if (ret)
		return ret;

	ret = ofi_av_close(&av->util_av);
	if (ret)
		return ret;

	free(av);
	return 0;
}

static int rxd_av_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
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

	if (!attr)
		return -FI_EINVAL;

	if (attr->name)
		return -FI_ENOSYS;

	domain = container_of(domain_fid, struct rxd_domain, util_domain.domain_fid);
	av = calloc(1, sizeof(*av));
	if (!av)
		return -FI_ENOMEM;

	util_attr.addrlen = sizeof(fi_addr_t);
	util_attr.overhead = attr->count;
	util_attr.flags = OFI_AV_HASH;
	if (attr->type == FI_AV_UNSPEC)
		attr->type = FI_AV_TABLE;

	ret = ofi_av_init(&domain->util_domain, attr, &util_attr,
			 &av->util_av, context);
	if (ret)
		goto err1;

	av_attr = *attr;
	av_attr.type = FI_AV_TABLE;
	av_attr.count = 0;
	av_attr.flags = 0;
	ret = fi_av_open(domain->dg_domain, &av_attr, &av->dg_av, context);
	if (ret)
		goto err2;

	av->util_av.av_fid.fid.ops = &rxd_av_fi_ops;
	av->util_av.av_fid.ops = &rxd_av_ops;
	*av_fid = &av->util_av.av_fid;
	return 0;

err2:
	ofi_av_close(&av->util_av);
err1:
	free(av);
	return ret;
}
