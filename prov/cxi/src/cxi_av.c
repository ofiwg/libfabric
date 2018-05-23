/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2017 Los Alamos National Security, LLC.
 *                    All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <sys/types.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "cxi_prov.h"

#include "ofi_osd.h"
#include "ofi_util.h"

#define CXI_LOG_DBG(...) _CXI_LOG_DBG(FI_LOG_AV, __VA_ARGS__)
#define CXI_LOG_ERROR(...) _CXI_LOG_ERROR(FI_LOG_AV, __VA_ARGS__)

#define CXI_AV_TABLE_SZ(count, av_name)					\
		(sizeof(struct cxi_av_table_hdr) +			\
		 (CXI_IS_SHARED_AV(av_name) * count *			\
		  sizeof(uint64_t)) +					\
		 (count * sizeof(struct cxi_addr)))
#define CXI_IS_SHARED_AV(av_name) ((av_name) ? 1 : 0)

static void cxi_update_av_table(struct cxi_av *_av, size_t count)
{
	_av->table = (struct cxi_addr *)
		((char *)_av->table_hdr +
		CXI_IS_SHARED_AV(_av->attr.name) * count * sizeof(uint64_t) +
		sizeof(struct cxi_av_table_hdr));
}

static int cxi_resize_av_table(struct cxi_av *av)
{
	void *new_addr;
	size_t new_count, table_sz, old_sz;

	new_count = av->table_hdr->size * 2;
	table_sz = CXI_AV_TABLE_SZ(new_count, av->attr.name);
	old_sz = CXI_AV_TABLE_SZ(av->table_hdr->size, av->attr.name);

	if (av->attr.name) {
		new_addr = mremap(av->table_hdr, old_sz, table_sz, 0);
		if (new_addr == MAP_FAILED)
			return -1;

		av->idx_arr[av->table_hdr->stored] = av->table_hdr->stored;
	} else {
		new_addr = realloc(av->table_hdr, table_sz);
		if (!new_addr)
			return -1;
	}

	av->table_hdr = new_addr;
	av->table_hdr->size = new_count;
	cxi_update_av_table(av, new_count);

	return 0;
}

static int cxi_av_get_next_index(struct cxi_av *av)
{
	uint64_t i;

	for (i = 0; i < av->table_hdr->size; i++) {
		if (!CXI_ADDR_AV_ENTRY_VALID(&(av->table[i])))
			return i;
	}

	return -1;
}

static int cxi_check_table_in(struct cxi_av *_av, struct cxi_addr *addr,
			      fi_addr_t *fi_addr, int count, uint64_t flags,
			      void *context)
{
	int i, ret = 0;
	struct cxi_addr *av_addr;
	int index;

	if (_av->attr.flags & FI_READ)
		return -FI_EINVAL;

	for (i = 0, ret = 0; i < count; i++) {
		if (_av->table_hdr->stored == _av->table_hdr->size) {
			index = cxi_av_get_next_index(_av);
			if (index < 0) {
				if (cxi_resize_av_table(_av)) {
					if (fi_addr)
						fi_addr[i] = FI_ADDR_NOTAVAIL;
					continue;
				}
				index = _av->table_hdr->stored++;
			}
		} else {
			index = _av->table_hdr->stored++;
		}

		av_addr = &_av->table[index];
		memcpy(av_addr, &addr[i], sizeof(struct cxi_addr));
		CXI_LOG_DBG("inserted 0x%x:%u:%u\n",
			    av_addr->nic, av_addr->domain, av_addr->port);

		if (fi_addr)
			fi_addr[i] = (fi_addr_t)index;

		CXI_ADDR_AV_ENTRY_SET_VALID(av_addr);
		ret++;
	}

	return ret;
}

static int cxi_av_insert(struct fid_av *av, const void *addr, size_t count,
			 fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	struct cxi_av *_av;

	_av = container_of(av, struct cxi_av, av_fid);
	return cxi_check_table_in(_av, (struct cxi_addr *)addr,
				   fi_addr, count, flags, context);
}

static int cxi_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr,
			 size_t *addrlen)
{
	int index;
	struct cxi_av *_av;
	struct cxi_addr *av_addr;

	_av = container_of(av, struct cxi_av, av_fid);
	index = ((uint64_t)fi_addr & _av->mask);
	if (index >= (int)_av->table_hdr->size || index < 0) {
		CXI_LOG_ERROR("requested address not inserted\n");
		return -EINVAL;
	}

	av_addr = &_av->table[index];
	memcpy(addr, av_addr, MIN(*addrlen, (size_t)_av->addrlen));
	*addrlen = _av->addrlen;

	return 0;
}

static int _cxi_av_insertsvc(struct fid_av *av, const char *node,
			     const char *service, fi_addr_t *fi_addr,
			     uint64_t flags, void *context)
{
	int ret;
	struct cxi_av *_av;
	struct cxi_addr addr;

	_av = container_of(av, struct cxi_av, av_fid);

	ret = cxi_parse_addr(node, service, &addr);
	if (ret)
		return ret;

	ret = cxi_check_table_in(_av, &addr, fi_addr, 1, flags, context);

	return ret;
}

static int cxi_av_insertsvc(struct fid_av *av, const char *node,
			    const char *service, fi_addr_t *fi_addr,
			    uint64_t flags, void *context)
{
	if (!service) {
		CXI_LOG_ERROR("Port not provided\n");
		return -FI_EINVAL;
	}

	return _cxi_av_insertsvc(av, node, service, fi_addr, flags, context);
}

static int cxi_av_remove(struct fid_av *av, fi_addr_t *fi_addr, size_t count,
			 uint64_t flags)
{
	size_t i;
	struct cxi_av *_av;
	struct cxi_addr *av_addr;

	_av = container_of(av, struct cxi_av, av_fid);

	for (i = 0; i < count; i++) {
		av_addr = &_av->table[fi_addr[i]];
		CXI_ADDR_AV_ENTRY_CLR_VALID(av_addr);
	}

	return 0;
}

static const char *cxi_av_straddr(struct fid_av *av, const void *addr,
				  char *buf, size_t *len)
{
	return ofi_straddr(buf, len, FI_ADDR_CXI, addr);
}

static int cxi_av_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct cxi_av *av;
	struct cxi_eq *eq;

	if (bfid->fclass != FI_CLASS_EQ)
		return -FI_EINVAL;

	av = container_of(fid, struct cxi_av, av_fid.fid);
	eq = container_of(bfid, struct cxi_eq, eq.fid);
	av->eq = eq;

	return 0;
}

static int cxi_av_close(struct fid *fid)
{
	struct cxi_av *av;
	int ret = 0;

	av = container_of(fid, struct cxi_av, av_fid.fid);
	if (ofi_atomic_get32(&av->ref))
		return -FI_EBUSY;

	if (!av->shared) {
		free(av->table_hdr);
	} else {
		ret = ofi_shm_unmap(&av->shm);
		if (ret)
			CXI_LOG_ERROR("unmap failed: %s\n",
				      strerror(ofi_syserr()));
	}

	ofi_atomic_dec32(&av->domain->ref);
	fastlock_destroy(&av->list_lock);
	free(av);

	return 0;
}

static struct fi_ops cxi_av_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxi_av_close,
	.bind = cxi_av_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_av cxi_am_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = cxi_av_insert,
	.insertsvc = cxi_av_insertsvc,
	.insertsym = fi_no_av_insertsym,
	.remove = cxi_av_remove,
	.lookup = cxi_av_lookup,
	.straddr = cxi_av_straddr
};

static struct fi_ops_av cxi_at_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = cxi_av_insert,
	.insertsvc = cxi_av_insertsvc,
	.insertsym = fi_no_av_insertsym,
	.remove = cxi_av_remove,
	.lookup = cxi_av_lookup,
	.straddr = cxi_av_straddr
};

static int cxi_verify_av_attr(struct fi_av_attr *attr)
{
	switch (attr->type) {
	case FI_AV_MAP:
	case FI_AV_TABLE:
	case FI_AV_UNSPEC:
		break;
	default:
		return -FI_EINVAL;
	}

	if (attr->flags & FI_READ && !attr->name)
		return -FI_EINVAL;

	if (attr->rx_ctx_bits > CXI_EP_MAX_CTX_BITS) {
		CXI_LOG_ERROR("Invalid rx_ctx_bits\n");
		return -FI_EINVAL;
	}

	return 0;
}

int cxi_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		struct fid_av **av, void *context)
{
	int ret = 0;
	struct cxi_domain *dom;
	struct cxi_av *_av;
	size_t table_sz;

	if (!attr || cxi_verify_av_attr(attr))
		return -FI_EINVAL;

	if (attr->type == FI_AV_UNSPEC)
		attr->type = FI_AV_TABLE;

	dom = container_of(domain, struct cxi_domain, dom_fid);
	if (dom->attr.av_type != FI_AV_UNSPEC &&
	    dom->attr.av_type != attr->type)
		return -FI_EINVAL;

	_av = calloc(1, sizeof(*_av));
	if (!_av)
		return -FI_ENOMEM;

	_av->attr = *attr;
	_av->attr.count = (attr->count) ? attr->count : cxi_av_def_sz;
	table_sz = CXI_AV_TABLE_SZ(_av->attr.count, attr->name);

	if (attr->name) {
		ret = ofi_shm_map(&_av->shm, attr->name, table_sz,
				  attr->flags & FI_READ,
				  (void **)&_av->table_hdr);

		if (ret || _av->table_hdr == MAP_FAILED) {
			CXI_LOG_ERROR("map failed\n");
			ret = -FI_EINVAL;
			goto err;
		}

		_av->idx_arr = (uint64_t *)(_av->table_hdr + 1);
		_av->attr.map_addr = _av->idx_arr;
		attr->map_addr = _av->attr.map_addr;
		CXI_LOG_DBG("Updating map_addr: %p\n", _av->attr.map_addr);

		if (attr->flags & FI_READ) {
			if (_av->table_hdr->size != _av->attr.count) {
				ret = -FI_EINVAL;
				goto err2;
			}
		} else {
			_av->table_hdr->size = _av->attr.count;
			_av->table_hdr->stored = 0;
		}
		_av->shared = 1;
	} else {
		_av->table_hdr = calloc(1, table_sz);
		if (!_av->table_hdr) {
			ret = -FI_ENOMEM;
			goto err;
		}
		_av->table_hdr->size = _av->attr.count;
	}
	cxi_update_av_table(_av, _av->attr.count);

	_av->av_fid.fid.fclass = FI_CLASS_AV;
	_av->av_fid.fid.context = context;
	_av->av_fid.fid.ops = &cxi_av_fi_ops;

	switch (attr->type) {
	case FI_AV_MAP:
		_av->av_fid.ops = &cxi_am_ops;
		break;
	case FI_AV_TABLE:
		_av->av_fid.ops = &cxi_at_ops;
		break;
	default:
		ret = -FI_EINVAL;
		goto err2;
	}

	ofi_atomic_initialize32(&_av->ref, 0);
	ofi_atomic_inc32(&dom->ref);
	_av->domain = dom;
	switch (dom->info.addr_format) {
	case FI_ADDR_CXI:
		_av->addrlen = sizeof(struct cxi_addr);
		break;
	default:
		CXI_LOG_ERROR("Invalid address format\n");
		ret = -FI_EINVAL;
		goto err2;
	}
	dlist_init(&_av->ep_list);
	fastlock_init(&_av->list_lock);
	_av->rx_ctx_bits = attr->rx_ctx_bits;
	_av->mask = attr->rx_ctx_bits ?
		((uint64_t)1 << (64 - attr->rx_ctx_bits)) - 1 : ~0;
	*av = &_av->av_fid;
	return 0;

err2:
	if (attr->name) {
		ofi_shm_unmap(&_av->shm);
	} else {
		if (_av->table_hdr)
			free(_av->table_hdr);
	}
err:
	free(_av);
	return ret;
}

