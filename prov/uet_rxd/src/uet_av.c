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

#include "uet.h"
#include <inttypes.h>


static int uet_tree_compare(struct ofi_rbmap *map, void *key, void *data)
{
	struct uet_av *av;
	uint8_t addr[UET_NAME_LENGTH];
	size_t len = UET_NAME_LENGTH;
	int ret;
	fi_addr_t dg_addr;

	memset(addr, 0, len);
	av = container_of(map, struct uet_av, rbmap);
	dg_addr = (intptr_t)ofi_idx_lookup(&av->rxdaddr_dg_idx,
					   (int)(uintptr_t) data);

	ret = fi_av_lookup(av->dg_av, dg_addr,addr, &len);
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
static int uet_av_set_addrlen(struct uet_av *av, const void *addr)
{
	struct uet_domain *domain;
	struct fid_av *tmp_av;
	struct fi_av_attr attr;
	uint8_t tmp_addr[UET_NAME_LENGTH];
	fi_addr_t fiaddr;
	size_t len;
	int ret;

	FI_INFO(&uet_prov, FI_LOG_AV, "determine dgram address len\n");
	memset(&attr, 0, sizeof attr);
	attr.count = 1;

	domain = container_of(av->util_av.domain, struct uet_domain, util_domain);
	ret = fi_av_open(domain->dg_domain, &attr, &tmp_av, NULL);
	if (ret) {
		FI_WARN(&uet_prov, FI_LOG_AV, "failed to open av: %d (%s)\n",
			-ret, fi_strerror(-ret));
		return ret;
	}

	ret = fi_av_insert(tmp_av, addr, 1, &fiaddr, 0, NULL);
	if (ret != 1) {
		FI_WARN(&uet_prov, FI_LOG_AV, "addr insert failed: %d (%s)\n",
			-ret, fi_strerror(-ret));
		ret = -FI_EINVAL;
		goto close;
	}

	len = sizeof tmp_addr;
	ret = fi_av_lookup(tmp_av, fiaddr, tmp_addr, &len);
	if (ret) {
		FI_WARN(&uet_prov, FI_LOG_AV, "addr lookup failed: %d (%s)\n",
			-ret, fi_strerror(-ret));
		goto close;
	}

	FI_INFO(&uet_prov, FI_LOG_AV, "set dgram address len: %zu\n", len);
	av->dg_addrlen = len;
close:
	fi_close(&tmp_av->fid);
	return ret;
}

static fi_addr_t uet_av_dg_addr(struct uet_av *av, fi_addr_t fi_addr)
{
	fi_addr_t dg_addr;
	fi_addr_t uet_addr = (intptr_t) ofi_idx_lookup(&av->fi_addr_idx,
					     UET_IDX_OFFSET((int)fi_addr));
	if (!uet_addr)
		goto err;
	dg_addr = (intptr_t) ofi_idx_lookup(&av->rxdaddr_dg_idx, (int)uet_addr);
	if (!dg_addr)
		goto err;

	return dg_addr;
err:
	return FI_ADDR_UNSPEC;
}

static int uet_set_rxd_addr(struct uet_av *av, fi_addr_t dg_addr, fi_addr_t *addr)
{
	int rxdaddr;
	rxdaddr = ofi_idx_insert(&(av->rxdaddr_dg_idx), (void*)(uintptr_t)dg_addr);
	if (rxdaddr < 0)
		return -FI_ENOMEM;
	*addr = rxdaddr;
	return 0;

}

static int uet_set_fi_addr(struct uet_av *av, fi_addr_t uet_addr)
{
	int idx;
	fi_addr_t dg_addr;

	idx = ofi_idx_insert(&(av->fi_addr_idx), (void*)(uintptr_t)uet_addr);
	if (idx < 0)
		goto nomem1;

	if (ofi_idm_set(&(av->rxdaddr_fi_idm), (int)uet_addr,
		        (void*)(uintptr_t) idx) < 0)
		goto nomem2;

	return idx;

nomem2:
	ofi_idx_remove_ordered(&(av->fi_addr_idx), idx);
nomem1:
	dg_addr = (intptr_t) ofi_idx_remove_ordered(&(av->rxdaddr_dg_idx),
						    (int) uet_addr);
	fi_av_remove(av->dg_av, &dg_addr, 1, 0);

	return -FI_ENOMEM;
}

int uet_av_insert_dg_addr(struct uet_av *av, const void *addr,
			  fi_addr_t *uet_addr, uint64_t flags,
			  void *context)
{
	fi_addr_t dg_addr;
	int ret;

	ret = fi_av_insert(av->dg_av, addr, 1, &dg_addr,
			     flags, context);
	if (ret != 1)
		return -FI_EINVAL;

	ret = uet_set_rxd_addr(av, dg_addr, uet_addr);
	if (ret < 0) {
		goto nomem;
	}

	ret = ofi_rbmap_insert(&av->rbmap, (void *)addr, (void *)(*uet_addr),
			       NULL);
	if (ret) {
		assert(ret != -FI_EALREADY);
		ofi_idx_remove_ordered(&(av->rxdaddr_dg_idx), (int)(*uet_addr));
		goto nomem;
	}

	return ret;
nomem:
	fi_av_remove(av->dg_av, &dg_addr, 1, flags);
	return ret;

}

static int uet_av_insert(struct fid_av *av_fid, const void *addr, size_t count,
			fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	struct uet_av *av;
	int i = 0, ret = 0, success_cnt = 0;
	fi_addr_t uet_addr;
	int util_addr, *sync_err = NULL;
	struct ofi_rbnode *node;

	av = container_of(av_fid, struct uet_av, util_av.av_fid);
	ret = ofi_verify_av_insert(&av->util_av, flags, context);
	if (ret)
		return ret;

	if (flags & FI_SYNC_ERR) {
		sync_err = context;
		memset(sync_err, 0, sizeof(*sync_err) * count);
	}

	ofi_genlock_lock(&av->util_av.lock);
	if (!av->dg_addrlen) {
		ret = uet_av_set_addrlen(av, addr);
		if (ret)
			goto out;
	}

	for (; i < count; i++, addr = (uint8_t *) addr + av->dg_addrlen) {
		node = ofi_rbmap_find(&av->rbmap, (void *) addr);
		if (node) {
			uet_addr = (fi_addr_t) node->data;
		} else {
			ret = uet_av_insert_dg_addr(av, addr, &uet_addr,
						    flags, sync_err ?
						    &sync_err[i] : context);
			if (ret)
				break;
		}

		util_addr = (int)(intptr_t) ofi_idm_lookup(&av->rxdaddr_fi_idm,
							   (int) uet_addr);
		if (!util_addr) {
			util_addr = uet_set_fi_addr(av, uet_addr);
			if (util_addr < 0) {
				ret = util_addr;
				break;
			}
		}
		if (fi_addr)
			fi_addr[i] = (util_addr - 1);

		success_cnt++;
	}

	if (ret) {
		FI_WARN(&uet_prov, FI_LOG_AV,
			"failed to insert address %d: %d (%s)\n",
			i, -ret, fi_strerror(-ret));
		if (fi_addr)
			fi_addr[i] = FI_ADDR_NOTAVAIL;
		else if (sync_err)
			sync_err[i] = -ret;
		i++;
	}
out:
	av->dg_av_used += success_cnt;
	ofi_genlock_unlock(&av->util_av.lock);

	for (; i < count; i++) {
		if (fi_addr)
			fi_addr[i] = FI_ADDR_NOTAVAIL;
		else if (sync_err)
			sync_err[i] = FI_ECANCELED;
	}


	return success_cnt;
}

static int uet_av_insertsvc(struct fid_av *av, const char *node,
			   const char *service, fi_addr_t *fi_addr,
			   uint64_t flags, void *context)
{
	return -FI_ENOSYS;
}

static int uet_av_insertsym(struct fid_av *av_fid, const char *node, size_t nodecnt,
			   const char *service, size_t svccnt, fi_addr_t *fi_addr,
			   uint64_t flags, void *context)
{
	return -FI_ENOSYS;
}

static int uet_av_remove(struct fid_av *av_fid, fi_addr_t *fi_addr, size_t count,
			uint64_t flags)
{
	int ret = 0;
	size_t i;
	fi_addr_t uet_addr;
	struct uet_av *av;

	av = container_of(av_fid, struct uet_av, util_av.av_fid);
	ofi_genlock_lock(&av->util_av.lock);
	for (i = 0; i < count; i++) {
		uet_addr = (intptr_t)ofi_idx_lookup(&av->fi_addr_idx,
						    (int) UET_IDX_OFFSET(fi_addr[i]));
		if (!uet_addr)
			goto err;

		ofi_idx_remove_ordered(&(av->fi_addr_idx),
				       (int) UET_IDX_OFFSET(fi_addr[i]));
		ofi_idm_clear(&(av->rxdaddr_fi_idm), (int) uet_addr);
	}

err:
	if (ret)
		FI_WARN(&uet_prov, FI_LOG_AV, "Unable to remove address from AV\n");

	ofi_genlock_unlock(&av->util_av.lock);
	return ret;
}

static const char *uet_av_straddr(struct fid_av *av, const void *addr,
				  char *buf, size_t *len)
{
	struct uet_av *uet_av;
	uet_av = container_of(av, struct uet_av, util_av.av_fid);
	return uet_av->dg_av->ops->straddr(uet_av->dg_av, addr, buf, len);
}

static int uet_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr,
			 size_t *addrlen)
{
	struct uet_av *uet_av;
	fi_addr_t dg_fiaddr;

	uet_av = container_of(av, struct uet_av, util_av.av_fid);
	dg_fiaddr = uet_av_dg_addr(uet_av, fi_addr);
	if (dg_fiaddr == FI_ADDR_UNSPEC)
		return -FI_ENODATA;

	return fi_av_lookup(uet_av->dg_av, dg_fiaddr, addr, addrlen);
}

static struct fi_ops_av uet_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = uet_av_insert,
	.insertsvc = uet_av_insertsvc,
	.insertsym = uet_av_insertsym,
	.remove = uet_av_remove,
	.lookup = uet_av_lookup,
	.straddr = uet_av_straddr,
};

static int uet_av_close(struct fid *fid)
{
	struct uet_av *av;
	struct ofi_rbnode *node;
	fi_addr_t dg_addr, uet_addr;
	int ret;

	av = container_of(fid, struct uet_av, util_av.av_fid);

	ret = ofi_av_close(&av->util_av);
	if (ret)
		return ret;

	while ((node = ofi_rbmap_get_root(&av->rbmap))) {
		uet_addr = (fi_addr_t) node->data;
		dg_addr = (intptr_t)ofi_idx_lookup(&av->rxdaddr_dg_idx,
						   (int) uet_addr);

		ret = fi_av_remove(av->dg_av, &dg_addr, 1, 0);
		if (ret)
			FI_WARN(&uet_prov, FI_LOG_AV,
				"failed to remove dg addr: %d (%s)\n",
				-ret, fi_strerror(-ret));

		ofi_idx_remove_ordered(&(av->rxdaddr_dg_idx), (int) uet_addr);
		ofi_rbmap_delete(&av->rbmap, node);
	}
	ofi_rbmap_cleanup(&av->rbmap);

	ret = fi_close(&av->dg_av->fid);
	if (ret)
		return ret;

	ofi_idx_reset(&(av->fi_addr_idx));
	ofi_idx_reset(&(av->rxdaddr_dg_idx));
	ofi_idm_reset(&(av->rxdaddr_fi_idm), NULL);

	free(av);
	return 0;
}

static struct fi_ops uet_av_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = uet_av_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int uet_av_create(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		   struct fid_av **av_fid, void *context)
{
	int ret;
	struct uet_av *av;
	struct uet_domain *domain;
	struct util_av_attr util_attr;
	struct fi_av_attr av_attr;

	if (!attr)
		return -FI_EINVAL;

	if (attr->name)
		return -FI_ENOSYS;

	//TODO implement dynamic AV sizing
	attr->count = roundup_power_of_two(attr->count ?
					   attr->count : uet_env.max_peers);
	domain = container_of(domain_fid, struct uet_domain, util_domain.domain_fid);
	av = calloc(1, sizeof(*av));
	if (!av)
		return -FI_ENOMEM;
	memset(&(av->fi_addr_idx), 0, sizeof(av->fi_addr_idx));
	memset(&(av->rxdaddr_dg_idx), 0, sizeof(av->rxdaddr_dg_idx));
	memset(&(av->rxdaddr_fi_idm), 0, sizeof(av->rxdaddr_fi_idm));

	util_attr.addrlen = sizeof(fi_addr_t);
	util_attr.context_len = 0;
	util_attr.flags = 0;
	attr->type = domain->util_domain.av_type != FI_AV_UNSPEC ?
		     domain->util_domain.av_type : FI_AV_TABLE;

	ret = ofi_av_init(&domain->util_domain, attr, &util_attr,
			  &av->util_av, context);
	if (ret)
		goto err1;

	ofi_rbmap_init(&av->rbmap, uet_tree_compare);

	av_attr = *attr;
	av_attr.count = 0;
	av_attr.flags = 0;
	ret = fi_av_open(domain->dg_domain, &av_attr, &av->dg_av, context);
	if (ret)
		goto err2;

	av->util_av.av_fid.fid.ops = &uet_av_fi_ops;
	av->util_av.av_fid.ops = &uet_av_ops;
	*av_fid = &av->util_av.av_fid;
	return 0;

err2:
	(void) ofi_av_close(&av->util_av);
err1:
	free(av);
	return ret;
}
