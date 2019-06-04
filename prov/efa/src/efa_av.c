/*
 * Copyright (c) 2016, Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2017-2019 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include <malloc.h>
#include <stdio.h>

#include <ofi_enosys.h>
#include "efa.h"
#include "efa_verbs.h"

static inline struct efa_conn *efa_av_tbl_idx_to_conn(struct efa_av *av, fi_addr_t addr)
{
	if (OFI_UNLIKELY(addr == FI_ADDR_UNSPEC))
		return NULL;
	return av->conn_table[addr];
}

static inline struct efa_conn *efa_av_map_addr_to_conn(struct efa_av *av, fi_addr_t addr)
{
	if (OFI_UNLIKELY(addr == FI_ADDR_UNSPEC))
		return NULL;
	return (struct efa_conn *)(void *)addr;
}

fi_addr_t efa_ah_qpn_to_addr(struct efa_ep *ep, uint16_t ah, uint16_t qpn)
{
	struct efa_reverse_av *reverse_av;
	struct efa_av *av = ep->av;
	struct efa_ah_qpn key = {
		.efa_ah = ah,
		.qpn = qpn,
	};

	HASH_FIND(hh, av->reverse_av, &key, sizeof(key), reverse_av);

	return OFI_LIKELY(!!reverse_av) ? reverse_av->fi_addr : FI_ADDR_NOTAVAIL;
}

static inline int efa_av_is_valid_address(struct efa_ep_addr *addr)
{
	struct efa_ep_addr all_zeros = {};

	return memcmp(addr->raw, all_zeros.raw, sizeof(addr->raw));
}

/* Returns the first NULL index in av connection table, starting from @hint */
static size_t efa_av_tbl_find_first_empty(struct efa_av *av, size_t hint)
{
	struct efa_conn **conn_table;

	assert(av->type == FI_AV_TABLE);

	conn_table = av->conn_table;
	for (; hint < av->count; hint++) {
		if (!conn_table[hint])
			return hint;
	}

	return -1;
}

static int efa_av_resize(struct efa_av *av, size_t new_av_count)
{
	if (av->type == FI_AV_TABLE) {
		void *p = realloc(av->conn_table,
				  (new_av_count *
				  sizeof(*av->conn_table)));

		if (p)
			av->conn_table = p;
		else
			return -FI_ENOMEM;

		memset(av->conn_table + av->count, 0,
		       (new_av_count - av->count) * sizeof(*av->conn_table));
	}

	av->count = new_av_count;

	return 0;
}

/* Inserts a single AH to AV. */
static int efa_av_insert_ah(struct efa_av *av, struct efa_ep_addr *addr, fi_addr_t *fi_addr)
{
	struct efa_pd *pd = container_of(av->domain->pd, struct efa_pd, ibv_pd);
	struct ibv_ah_attr ah_attr;
	char str[INET6_ADDRSTRLEN] = { 0 };
	struct efa_reverse_av *reverse_av;
	struct efa_ah_qpn key;
	struct efa_conn *conn;
	int err;

	memset(&ah_attr, 0, sizeof(struct ibv_ah_attr));
	inet_ntop(AF_INET6, addr->raw, str, INET6_ADDRSTRLEN);
	EFA_INFO(FI_LOG_AV, "Insert address: GID[%s] QP[%u]\n", str, addr->qpn);
	if (!efa_av_is_valid_address(addr)) {
		EFA_INFO(FI_LOG_AV, "Failed to insert bad addr");
		err = -FI_EADDRNOTAVAIL;
		goto err_invalid;
	}

	err = ofi_memalign((void **)&conn, EFA_MEM_ALIGNMENT, sizeof(*conn));
	if (err) {
		err = -FI_ENOMEM;
		goto err_invalid;
	}

	ah_attr.port_num = 1;
	memcpy(ah_attr.grh.dgid.raw, addr->raw, sizeof(addr->raw));
	conn->ah = efa_cmd_create_ah(pd, &ah_attr);
	if (!conn->ah) {
		err = -FI_EINVAL;
		goto err_free_conn;
	}
	memcpy((void *)&conn->ep_addr, addr, sizeof(*addr));

	switch (av->type) {
	case FI_AV_MAP:
		*fi_addr = (uintptr_t)(void *)conn;

		break;
	case FI_AV_TABLE:
		av->next = efa_av_tbl_find_first_empty(av, av->next);
		assert(av->next != -1);
		*fi_addr = av->next;

		av->conn_table[av->next] = conn;
		av->next++;
		break;
	default:
		assert(0);
		break;
	}

	key.efa_ah = conn->ah->efa_address_handle;
	key.qpn = addr->qpn;
	/* This is correct since the same address should be mapped to the same ah. */
	HASH_FIND(hh, av->reverse_av, &key, sizeof(key), reverse_av);
	if (!reverse_av) {
		reverse_av = malloc(sizeof(*reverse_av));
		if (!reverse_av) {
			err = -FI_ENOMEM;
			goto err_destroy_ah;
		}

		memcpy(&reverse_av->key, &key, sizeof(key));
		reverse_av->fi_addr = *fi_addr;
		HASH_ADD(hh, av->reverse_av, key,
			 sizeof(reverse_av->key), reverse_av);
	}

	EFA_INFO(FI_LOG_AV, "av successfully inserted conn[%p] fi_addr[%" PRIu64 "]\n",
		 conn, *fi_addr);

	av->used++;
	return FI_SUCCESS;

err_destroy_ah:
	efa_cmd_destroy_ah(conn->ah);
err_free_conn:
	free(conn);
err_invalid:
	*fi_addr = FI_ADDR_NOTAVAIL;
	return err;
}

static int efa_av_insert(struct fid_av *av_fid, const void *addr,
			 size_t count, fi_addr_t *fi_addr,
			 uint64_t flags, void *context)
{
	struct efa_av *av = container_of(av_fid, struct efa_av, av_fid);
	struct efa_ep_addr *addr_i;
	int *fi_errors = context;
	fi_addr_t fi_addr_res = FI_ADDR_UNSPEC;
	int failed;
	size_t i;
	int err;

	if (av->flags & FI_EVENT)
		return -FI_ENOEQ;

	if ((flags & FI_SYNC_ERR) && (!context || (flags & FI_EVENT)))
		return -FI_EINVAL;
	else if (flags & FI_SYNC_ERR)
		memset(context, 0, sizeof(int) * count);

	if (av->used + count > av->count) {
		err = efa_av_resize(av, av->used + count);
		if (err)
			return err;
	}

	failed = 0;
	for (i = 0; i < count; i++) {
		addr_i = (struct efa_ep_addr *)((uint8_t *)addr + i * EFA_EP_ADDR_LEN);
		err = efa_av_insert_ah(av, addr_i, &fi_addr_res);
		if (err)
			failed++;
		if (flags & FI_SYNC_ERR)
			fi_errors[i] = err;
		if (fi_addr)
			fi_addr[i] = fi_addr_res;
	}

	return count - failed;
}

static int efa_av_remove(struct fid_av *av_fid, fi_addr_t *fi_addr,
			 size_t count, uint64_t flags)
{
	struct efa_av *av = container_of(av_fid, struct efa_av, av_fid);
	struct efa_conn *conn = NULL;
	char str[INET6_ADDRSTRLEN];
	int ret = 0;
	int i;

	if (!fi_addr || (av->type != FI_AV_MAP && av->type != FI_AV_TABLE))
		return -FI_EINVAL;

	for (i = 0; i < count; i++) {
		struct efa_reverse_av *reverse_av;
		struct efa_ah_qpn key;

		if (fi_addr[i] == FI_ADDR_NOTAVAIL)
			continue;

		if (av->type == FI_AV_MAP) {
			conn = (struct efa_conn *)fi_addr[i];
		} else { /* (av->type == FI_AV_TABLE) */
			conn = av->conn_table[fi_addr[i]];
			av->conn_table[fi_addr[i]] = NULL;
			av->next = MIN(av->next, fi_addr[i]);
		}
		if (!conn)
			continue;

		key.efa_ah = conn->ah->efa_address_handle;
		key.qpn = conn->ep_addr.qpn;
		HASH_FIND(hh, av->reverse_av, &key, sizeof(key), reverse_av);
		if (OFI_LIKELY(!!reverse_av)) {
			HASH_DEL(av->reverse_av, reverse_av);
			free(reverse_av);
		}

		ret = efa_cmd_destroy_ah(conn->ah);
		if (ret)
			return ret;

		memset(str, 0, sizeof(str));
		inet_ntop(AF_INET6, conn->ep_addr.raw, str, INET6_ADDRSTRLEN);
		EFA_INFO(FI_LOG_AV, "av_remove conn[%p] with GID[%s] QP[%u]\n", conn,
			 str, conn->ep_addr.qpn);

		free(conn);
		av->used--;
	}
	return ret;
}

static int efa_av_lookup(struct fid_av *av_fid, fi_addr_t fi_addr,
			 void *addr, size_t *addrlen)
{
	struct efa_av *av = container_of(av_fid, struct efa_av, av_fid);
	struct efa_conn *conn = NULL;

	if (av->type != FI_AV_MAP && av->type != FI_AV_TABLE)
		return -FI_EINVAL;

	if (fi_addr == FI_ADDR_NOTAVAIL)
		return -FI_EINVAL;

	if (av->type == FI_AV_MAP) {
		conn = (struct efa_conn *)fi_addr;
	} else { /* (av->type == FI_AV_TABLE) */
		if (fi_addr >= av->count)
			return -EINVAL;

		conn = av->conn_table[fi_addr];
	}
	if (!conn)
		return -EINVAL;

	memcpy(addr, (void *)&conn->ep_addr, MIN(sizeof(conn->ep_addr), *addrlen));
	*addrlen = sizeof(conn->ep_addr);
	return 0;
}

static const char *efa_av_straddr(struct fid_av *av_fid, const void *addr,
				  char *buf, size_t *len)
{
	return ofi_straddr(buf, len, FI_ADDR_EFA, addr);
}

static struct fi_ops_av efa_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = efa_av_insert,
	.insertsvc = fi_no_av_insertsvc,
	.insertsym = fi_no_av_insertsym,
	.remove = efa_av_remove,
	.lookup = efa_av_lookup,
	.straddr = efa_av_straddr
};

static int efa_av_close(struct fid *fid)
{
	struct efa_av *av;
	int ret = 0;
	int i;

	av = container_of(fid, struct efa_av, av_fid.fid);
	for (i = 0; i < av->count; i++) {
		fi_addr_t addr = i;

		ret = efa_av_remove(&av->av_fid, &addr, 1, 0);
		if (ret)
			return ret;
	}
	free(av->conn_table);
	free(av);
	return 0;
}

static struct fi_ops efa_av_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_av_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int efa_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		struct fid_av **av_fid, void *context)
{
	struct efa_domain *domain;
	struct efa_av *av;
	size_t count = 64;
	int err;

	domain = container_of(domain_fid, struct efa_domain,
			      util_domain.domain_fid);

	if (!attr)
		return -FI_EINVAL;

	if (attr->flags)
		return -FI_EBADFLAGS;

	switch (attr->type) {
	case FI_AV_UNSPEC:
	case FI_AV_TABLE:
		attr->type = FI_AV_TABLE;
		break;
	case FI_AV_MAP:
	default:
		return -EINVAL;
	}

	if (attr->count)
		count = attr->count;

	av = calloc(1, sizeof(*av));
	if (!av)
		return -ENOMEM;

	av->domain = domain;
	av->type = attr->type;
	av->count = count;
	av->used = 0;
	av->next = 0;

	if (av->type == FI_AV_TABLE && av->count > 0) {
		av->conn_table = calloc(av->count, sizeof(*av->conn_table));
		if (!av->conn_table) {
			err = -ENOMEM;
			goto err_free_av;
		}
	}

	if (av->type == FI_AV_MAP)
		av->addr_to_conn = efa_av_map_addr_to_conn;
	else /* if (av->type == FI_AV_TABLE) */
		av->addr_to_conn = efa_av_tbl_idx_to_conn;

	av->av_fid.fid.fclass = FI_CLASS_AV;
	av->av_fid.fid.context = context;
	av->av_fid.fid.ops = &efa_av_fi_ops;

	av->av_fid.ops = &efa_av_ops;

	*av_fid = &av->av_fid;
	return 0;

err_free_av:
	free(av);
	return err;
}
