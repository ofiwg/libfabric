/*
 * Copyright (c) 2015-2018 Intel Corporation. All rights reserved.
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


static int rxd_tree_compare(struct ofi_rbmap *map, void *key, void *data)
{
	struct rxd_av *av;
	uint8_t addr[RXD_NAME_LENGTH];
	size_t len = RXD_NAME_LENGTH;
	int ret;

	memset(addr, 0, len);
	av = container_of(map, struct rxd_av, rbmap);
	ret = fi_av_lookup(av->dg_av, (fi_addr_t) data, addr, &len);
	if (ret)
		return -1;

	return memcmp(key, addr, len);
}

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
	uint8_t tmp_addr[RXD_NAME_LENGTH];
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
	return (fi_addr >= av->util_av.count || fi_addr == FI_ADDR_UNSPEC) ?
		FI_ADDR_UNSPEC : av->tx_map[fi_addr];
}

fi_addr_t rxd_av_fi_addr(struct rxd_av *av, fi_addr_t dg_fiaddr)
{
	//TODO define behavior for duplicate av_insert calls
	fi_addr_t fi_addr = 0;

	for (fi_addr = 0; fi_addr < av->util_av.count; fi_addr++) {
		if (av->tx_map[fi_addr] == dg_fiaddr)
			return fi_addr;
	}
	return FI_ADDR_UNSPEC;
}

static fi_addr_t rxd_set_tx_addr(struct rxd_av *av, fi_addr_t addr)
{
	int tries = 0;

	while (av->tx_map[av->tx_idx] != FI_ADDR_UNSPEC &&
	       tries < av->util_av.count) {
		if (++av->tx_idx == av->util_av.count)
			av->tx_idx = 0;
		tries++;
	}
	assert(av->tx_idx < av->util_av.count && tries < av->util_av.count);
	av->tx_map[av->tx_idx] = addr;

	return av->tx_idx;
}

int rxd_av_insert_dg_addr(struct rxd_av *av, const void *addr,
			  fi_addr_t *dg_fiaddr, uint64_t flags,
			  void *context)
{
	int ret;

	ret = fi_av_insert(av->dg_av, addr, 1, dg_fiaddr,
			     flags, context);
	if (ret != 1)
		return -errno;

	ret = ofi_rbmap_insert(&av->rbmap, (void *) addr, (void *) (*dg_fiaddr));

	if (ret && ret != -FI_EALREADY) {
		fi_av_remove(av->dg_av, dg_fiaddr, 1, flags);
		return ret;
	}

	return 0;
}

static int rxd_av_insert(struct fid_av *av_fid, const void *addr, size_t count,
			fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	struct rxd_av *av;
	int i = 0, ret = 0, success_cnt = 0;
	fi_addr_t dg_fiaddr, tx_addr;

	av = container_of(av_fid, struct rxd_av, util_av.av_fid);
	fastlock_acquire(&av->util_av.lock);
	if (!av->dg_addrlen) {
		ret = rxd_av_set_addrlen(av, addr);
		if (ret)
			goto out;
	}

	for (; i < count; i++, addr = (uint8_t *) addr + av->dg_addrlen) {
		ret = rxd_av_insert_dg_addr(av, addr, &dg_fiaddr,
					    flags, context);
		if (ret && ret != -FI_EALREADY)
			break;
		if (!ret) {
			tx_addr = rxd_set_tx_addr(av, dg_fiaddr);
		} else {
			tx_addr = rxd_av_fi_addr(av, dg_fiaddr);
			ret = 0;
		}

		if (fi_addr)
			fi_addr[i] = tx_addr;
		success_cnt++;
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
	size_t i, addrlen;
	fi_addr_t dg_fiaddr;
	struct rxd_av *av;
	struct ofi_rbnode *node;
	uint8_t addr[RXD_NAME_LENGTH];

	av = container_of(av_fid, struct rxd_av, util_av.av_fid);
	fastlock_acquire(&av->util_av.lock);
	for (i = 0; i < count; i++) {
		dg_fiaddr = av->tx_map[fi_addr[i]];
		ret = fi_av_lookup(av->dg_av, dg_fiaddr, addr, &addrlen);
		if (ret)
			continue;

		node = ofi_rbmap_find(&av->rbmap, addr);
		if (!node)
			continue;

		ret = fi_av_remove(av->dg_av, &dg_fiaddr, 1, flags);
		if (ret)
			break;

		ofi_rbmap_delete(&av->rbmap, node);
		av->tx_map[fi_addr[i]] = FI_ADDR_UNSPEC;
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
	fi_addr_t dg_fiaddr;

	rxd_av = container_of(av, struct rxd_av, util_av.av_fid);
	dg_fiaddr = rxd_av_dg_addr(rxd_av, fi_addr);
	if (dg_fiaddr == FI_ADDR_UNSPEC)
		return -FI_ENODATA;

	return fi_av_lookup(rxd_av->dg_av, dg_fiaddr, addr, addrlen);
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
	int ret, i;
	struct rxd_av *av;
	struct rxd_domain *domain;
	struct util_av_attr util_attr;
	struct fi_av_attr av_attr;

	if (!attr)
		return -FI_EINVAL;

	if (attr->name)
		return -FI_ENOSYS;

	domain = container_of(domain_fid, struct rxd_domain, util_domain.domain_fid);
	av = calloc(1, sizeof(*av) + (attr->count * sizeof(fi_addr_t)));
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


	av->rbmap.compare = &rxd_tree_compare;
	ofi_rbmap_init(&av->rbmap);
	for (i = 0; i < attr->count; av->tx_map[i++] = FI_ADDR_UNSPEC)
		;

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
