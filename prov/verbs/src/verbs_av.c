/*
 * Copyright (c) 2016, Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#include <pthread.h>
#include <stdio.h>

#include <fi_enosys.h>
#include "fi_verbs.h"
#include "ep_rdm/verbs_rdm.h"

static int fi_ibv_av_close(fid_t fid)
{
	struct fi_ibv_av *av = container_of(fid, struct fi_ibv_av, av_fid.fid);
	free(av);
	return 0;
}

static struct fi_ops fi_ibv_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_av_close,
	.bind = fi_no_bind,
};

static inline struct fi_ibv_rdm_conn *
fi_ibv_rdm_av_tbl_idx_to_conn(struct fi_ibv_rdm_ep *ep, fi_addr_t addr)
{
	return (addr == FI_ADDR_UNSPEC) ? NULL : ep->domain->rdm_cm->conn_table[addr];
}

static inline struct fi_ibv_rdm_conn *
fi_ibv_rdm_av_map_addr_to_conn(struct fi_ibv_rdm_ep *ep, fi_addr_t addr)
{
	return (struct fi_ibv_rdm_conn *)
		(addr == FI_ADDR_UNSPEC ? NULL : (void *)addr);
}

static inline fi_addr_t
fi_ibv_rdm_to_conn_to_av_tbl_idx(struct fi_ibv_rdm_ep *ep, struct fi_ibv_rdm_conn *conn)
{
	size_t i;
	if (conn == NULL)
		return FI_ADDR_UNSPEC;

	for (i = 0; i < ep->av->used; i++) {
		if (ep->domain->rdm_cm->conn_table[i] == conn) {
			return i;
		}
	}

	return FI_ADDR_UNSPEC;
}

static inline fi_addr_t
fi_ibv_rdm_conn_to_av_map_addr(struct fi_ibv_rdm_ep *ep, struct fi_ibv_rdm_conn *conn)
{
	return (conn == NULL) ? FI_ADDR_UNSPEC : (fi_addr_t)(uintptr_t)conn;
}

/* TODO: match rest of verbs code for variable naming */
int fi_ibv_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		   struct fid_av **av_fid, void *context)
{
	struct fi_ibv_domain *fid_domain;
	struct fi_ibv_av *av;
	size_t count = 64;

	fid_domain = container_of(domain, struct fi_ibv_domain, domain_fid);

	if (!attr)
		return -FI_EINVAL;

	switch (attr->type) {
	case FI_AV_UNSPEC:
		attr->type = FI_AV_MAP;
	case FI_AV_MAP:
	case FI_AV_TABLE:
		break;
	default:
		return -EINVAL;
	}

	if (attr->count)
		count = attr->count;

	av = calloc(1, sizeof *av);
	if (!av)
		return -ENOMEM;

	assert(fid_domain->rdm);
	av->domain = fid_domain;
	av->type = attr->type;
	av->count = count;
	av->used = 0;

	if (av->type == FI_AV_TABLE && av->count > 0) {
		av->domain->rdm_cm->conn_table =
			calloc(av->count, sizeof(*av->domain->rdm_cm->conn_table));
		if (!av->domain->rdm_cm->conn_table) {
			free(av);
			return -ENOMEM;
		}
	}

	if (av->type == FI_AV_MAP) {
		av->addr_to_conn = fi_ibv_rdm_av_map_addr_to_conn;
		av->conn_to_addr = fi_ibv_rdm_conn_to_av_map_addr;
	} else /* if (av->type == FI_AV_TABLE) */ {
		av->addr_to_conn = fi_ibv_rdm_av_tbl_idx_to_conn;
		av->conn_to_addr = fi_ibv_rdm_to_conn_to_av_tbl_idx;
	}

	av->av_fid.fid.fclass = FI_CLASS_AV;
	av->av_fid.fid.context = context;
	av->av_fid.fid.ops = &fi_ibv_fi_ops;

	av->av_fid.ops = fi_ibv_rdm_set_av_ops();

	*av_fid = &av->av_fid;
	return 0;
}
