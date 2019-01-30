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

#include "config.h"

#include <stdlib.h>
#include <string.h>

#include <ofi_util.h>

#include "sock.h"
#include "sock_util.h"

#define SOCK_LOG_DBG(...) _SOCK_LOG_DBG(FI_LOG_MR, __VA_ARGS__)
#define SOCK_LOG_ERROR(...) _SOCK_LOG_ERROR(FI_LOG_MR, __VA_ARGS__)

static int sock_mr_close(struct fid *fid)
{
	struct sock_domain *dom;
	struct sock_mr *mr;
	int err = 0;

	mr = container_of(fid, struct sock_mr, mr_fid.fid);
	dom = mr->domain;

	fastlock_acquire(&dom->lock);
	err = ofi_mr_map_remove(&dom->mr_map, mr->map_key);
	if (err != 0)
		SOCK_LOG_ERROR("MR Erase error %d \n", err);

	fastlock_release(&dom->lock);
	ofi_atomic_dec32(&dom->ref);
	free(mr->raw_key);
	free(mr);
	return 0;
}

static int sock_mr_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct sock_cntr *cntr;
	struct sock_cq *cq;
	struct sock_mr *mr;

	mr = container_of(fid, struct sock_mr, mr_fid.fid);
	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct sock_cq, cq_fid.fid);
		if (mr->domain != cq->domain)
			return -FI_EINVAL;

		if (flags & FI_REMOTE_WRITE)
			mr->cq = cq;
		break;

	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct sock_cntr, cntr_fid.fid);
		if (mr->domain != cntr->domain)
			return -FI_EINVAL;

		if (flags & FI_REMOTE_WRITE)
			mr->cntr = cntr;
		break;

	default:
		return -FI_EINVAL;
	}
	return 0;
}

static int sock_mr_get_raw(struct sock_mr *mr, struct fi_mr_raw_attr *attr) {
	size_t copy_size;
	void *copy_src;
	int ret;

	if (!attr) return -FI_EINVAL;
	if (!attr->base_addr) return -FI_EINVAL;
	if (!attr->raw_key) return -FI_EINVAL;
	if (!attr->key_size) return -FI_EINVAL;
	if (attr->flags != 0)
		SOCK_LOG_DBG("Ignoring unknown flags in FI_GET_RAW_MR: %lu\n", attr->flags);

	if (mr->domain->attr.mr_mode & FI_MR_RAW) {
		copy_size = mr->raw_key_len;
		copy_src = mr->raw_key;
	} else {
		copy_size = sizeof(mr->map_key);
		copy_src = &mr->map_key;
	}
	memcpy(attr->raw_key, copy_src, MIN(copy_size, *attr->key_size));
	ret = FI_SUCCESS;
	if (*attr->key_size < copy_size) ret = -FI_ETOOSMALL;
	*attr->key_size = copy_size;
	*attr->base_addr = 0; /* TODO: this assumes FI_MR_VIRT_ADDR... */
	return ret;
}

static int sock_mr_control(struct fid *fid, int command, void *arg)
{
	struct sock_mr *mr;

	mr = container_of(fid, struct sock_mr, mr_fid.fid);
	switch (command) {
	case FI_GET_RAW_MR:
		return sock_mr_get_raw(mr, arg);
	default:
		return -FI_ENOSYS;
	}
	return 0;
}

static struct fi_ops sock_mr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = sock_mr_close,
	.bind = sock_mr_bind,
	.control = sock_mr_control,
	.ops_open = fi_no_ops_open,
};

struct sock_mr *sock_mr_verify_key(struct sock_domain *domain, uint64_t key,
	uintptr_t *buf, size_t len, void* raw_key, size_t raw_key_len, uint64_t access)
{
	int err = 0;
	struct sock_mr *mr;

	if (domain->attr.mr_mode & FI_MR_RAW) {
		if (raw_key_len != domain->attr.mr_key_size || !raw_key ||
		    key != FI_KEY_NOTAVAIL) {
			SOCK_LOG_ERROR("MR check failed - invalid raw key input\n");
			return NULL;
		}
		/* In raw mode, provided key is meaningless. Read real key from raw_key. */
		key = *(uint64_t*)raw_key;
	} else if (raw_key_len) {
		SOCK_LOG_ERROR("MR check failed - unexpected raw key\n");
		return NULL;
	}

	fastlock_acquire(&domain->lock);

	err = ofi_mr_map_verify(&domain->mr_map, buf, len, key, access, (void **)&mr);
	if (err != 0) {
		SOCK_LOG_ERROR("MR check failed\n");
		mr = NULL;
	}

	fastlock_release(&domain->lock);

	if (mr && domain->attr.mr_mode & FI_MR_RAW) {
		if (mr->raw_key_len != raw_key_len) {
			SOCK_LOG_ERROR("MR check failed - wrong raw key length\n");
			mr = NULL;
		} else if (0 != memcmp(mr->raw_key, raw_key, mr->raw_key_len)) {
			SOCK_LOG_ERROR("MR check failed - raw key mismatch\n");
			mr = NULL;
		}
	}

	return mr;
}

struct sock_mr *sock_mr_verify_desc(struct sock_domain *domain, void *desc,
	void *buf, size_t len, uint64_t access)
{
	uint64_t key = (uintptr_t)desc;
	return sock_mr_verify_key(domain, key, buf, len, NULL, 0, access);
}

static int sock_regattr(struct fid *fid, const struct fi_mr_attr *attr,
	uint64_t flags, struct fid_mr **mr)
{
	struct fi_eq_entry eq_entry;
	struct sock_domain *dom;
	struct sock_mr *_mr;
	uint64_t key;
	struct fid_domain *domain;
	int ret = 0;

	if (fid->fclass != FI_CLASS_DOMAIN || !attr || attr->iov_count <= 0) {
		return -FI_EINVAL;
	}

	domain = container_of(fid, struct fid_domain, fid);
	dom = container_of(domain, struct sock_domain, dom_fid);

	_mr = calloc(1, sizeof(*_mr));
	if (!_mr)
		return -FI_ENOMEM;

	if (dom->attr.mr_mode & FI_MR_RAW) {
		_mr->raw_key_len = dom->attr.mr_key_size;
		assert(_mr->raw_key_len > sizeof(uint64_t));
		_mr->raw_key = calloc(1, _mr->raw_key_len);
		if (!_mr->raw_key) {
			free(_mr);
			return -FI_ENOMEM;
		}
	}

	fastlock_acquire(&dom->lock);

	_mr->mr_fid.fid.fclass = FI_CLASS_MR;
	_mr->mr_fid.fid.context = attr->context;
	_mr->mr_fid.fid.ops = &sock_mr_fi_ops;

	_mr->domain = dom;
	_mr->flags = flags;

	ret = ofi_mr_map_insert(&dom->mr_map, attr, &key, _mr);
	if (ret != 0)
		goto err;

	_mr->map_key = key;
	if (dom->attr.mr_mode & FI_MR_RAW) {
		int i;
		/* Hide the map key from the user in raw mode */
		_mr->mr_fid.key = FI_KEY_NOTAVAIL;
		/* Raw key is the rbtree key plus a pseudo-random suffix.
		 * This is not cryptographically secure, but no worse than the rbtree key
		 * implementation which is just an incremented index. */
		*(uint64_t*)(_mr->raw_key) = key;
		for (i = sizeof(key); i < _mr->raw_key_len; ++i)
			((uint8_t*)_mr->raw_key)[i] = (uint8_t)rand();
	} else {
		_mr->mr_fid.key = key;
	}
	_mr->mr_fid.mem_desc = (void *)(uintptr_t)key;
	fastlock_release(&dom->lock);

	*mr = &_mr->mr_fid;
	ofi_atomic_inc32(&dom->ref);

	if (dom->mr_eq) {
		eq_entry.fid = &domain->fid;
		eq_entry.context = attr->context;
		return sock_eq_report_event(dom->mr_eq, FI_MR_COMPLETE,
			&eq_entry, sizeof(eq_entry), 0);
	}

	return 0;

err:
	fastlock_release(&dom->lock);
	free(_mr->raw_key);
	free(_mr);
	return ret;
}

static int sock_regv(struct fid *fid, const struct iovec *iov,
	size_t count, uint64_t access,
	uint64_t offset, uint64_t requested_key,
	uint64_t flags, struct fid_mr **mr, void *context)
{
	struct fi_mr_attr attr;

	attr.mr_iov = iov;
	attr.iov_count = count;
	attr.access = access;
	attr.offset = offset;
	attr.requested_key = requested_key;
	attr.context = context;
	return sock_regattr(fid, &attr, flags, mr);
}

static int sock_reg(struct fid *fid, const void *buf, size_t len,
	uint64_t access, uint64_t offset, uint64_t requested_key,
	uint64_t flags, struct fid_mr **mr, void *context)
{
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	return sock_regv(fid, &iov, 1, access, offset, requested_key,
		flags, mr, context);
}

struct fi_ops_mr sock_dom_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = sock_reg,
	.regv = sock_regv,
	.regattr = sock_regattr,
};
