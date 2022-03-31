/*
 * Copyright (c) 2022 ORNL. All rights reserved.
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>

#include <rdma/fi_errno.h>
#include "ofi_util.h"
#include "ofi.h"
#include "shared/ofi_str.h"
#include "ofi_prov.h"
#include "ofi_perf.h"
#include "ofi_hmem.h"
#include "rdma/fi_ext.h"
#include "linkx.h"

static inline struct lnx_peer *
get_peer(struct lnx_peer **peers, fi_addr_t addr)
{
	/* TODO: need to support FI_ADDR_UNSPEC */
	if (!peers || addr == FI_ADDR_UNSPEC)
		return NULL;

	return peers[addr];
}

ssize_t lnx_trecv(struct fid_ep *ep, void *buf, size_t len, void *desc,
		fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context)
{
	/* TODO: To properly handle recv for multi-rail, we'll need to have
	 * shared queues. As you can be receiving from any of the underlying
	 * core endpoints
	 */
	int rc;
	struct lnx_ep *lep;
	struct fid_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_recv_pathway(get_peer(peer_tbl->lpt_entries, src_addr),
								 &cep, &core_addr);
	if (rc)
		return rc;

	rc = fi_trecv(cep, buf, len, desc, core_addr, tag, ignore, context);

	return rc;
}

ssize_t lnx_trecvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
		void *context)
{
	int rc;
	struct lnx_ep *lep;
	struct fid_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_recv_pathway(get_peer(peer_tbl->lpt_entries, src_addr),
								 &cep, &core_addr);
	if (rc)
		return rc;

	rc = fi_trecvv(cep, iov, desc, count, core_addr, tag, ignore, context);

	return rc;
}

ssize_t lnx_trecvmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
		uint64_t flags)
{
	int rc;
	struct lnx_ep *lep;
	struct fid_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_recv_pathway(get_peer(peer_tbl->lpt_entries, msg->addr),
								 &cep, &core_addr);
	if (rc)
		return rc;

	rc = fi_trecvmsg(cep, msg, flags);

	return rc;
}

ssize_t lnx_tsend(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		fi_addr_t dest_addr, uint64_t tag, void *context)
{
	int rc;
	struct lnx_ep *lep;
	struct fid_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[dest_addr], &cep, &core_addr);
	if (rc)
		return rc;

	rc = fi_tsend(cep, buf, len, desc, core_addr, tag, context);

	return rc;
}

ssize_t lnx_tsendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	int rc;
	struct lnx_ep *lep;
	struct fid_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[dest_addr], &cep, &core_addr);
	if (rc)
		return rc;

	rc = fi_tsendv(cep, iov, desc, count, core_addr, tag, context);

	return rc;
}

ssize_t lnx_tsendmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
		uint64_t flags)
{
	int rc;
	struct lnx_ep *lep;
	struct fid_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[msg->addr], &cep, &core_addr);
	if (rc)
		return rc;

	rc = fi_tsendmsg(cep, msg, flags);

	return rc;
}

ssize_t lnx_tinject(struct fid_ep *ep, const void *buf, size_t len,
		fi_addr_t dest_addr, uint64_t tag)
{
	int rc;
	struct lnx_ep *lep;
	struct fid_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[dest_addr], &cep, &core_addr);
	if (rc)
		return rc;

	rc = fi_tinject(cep, buf, len, core_addr, tag);

	return rc;
}

ssize_t lnx_tsenddata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		uint64_t data, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	int rc;
	struct lnx_ep *lep;
	struct fid_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[dest_addr], &cep, &core_addr);
	if (rc)
		return rc;

	rc = fi_tsenddata(cep, buf, len, desc, data, core_addr, tag, context);

	return rc;
}

ssize_t lnx_tinjectdata(struct fid_ep *ep, const void *buf, size_t len,
		uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	int rc;
	struct lnx_ep *lep;
	struct fid_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;

	lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[dest_addr], &cep, &core_addr);
	if (rc)
		return rc;

	rc = fi_tinjectdata(cep, buf, len, data, core_addr, tag);

	return rc;
}

struct fi_ops_tagged lnx_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = lnx_trecv,
	.recvv = lnx_trecvv,
	.recvmsg = lnx_trecvmsg,
	.send = lnx_tsend,
	.sendv = lnx_tsendv,
	.sendmsg = lnx_tsendmsg,
	.inject = lnx_tinject,
	.senddata = lnx_tsenddata,
	.injectdata = lnx_tinjectdata,
};

struct fi_ops_msg lnx_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = fi_no_msg_recv,
	.recvv = fi_no_msg_recvv,
	.recvmsg = fi_no_msg_recvmsg,
	.send = fi_no_msg_send,
	.sendv = fi_no_msg_sendv,
	.sendmsg = fi_no_msg_sendmsg,
	.inject = fi_no_msg_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};

struct fi_ops_rma lnx_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = fi_no_rma_read,
	.readv = fi_no_rma_readv,
	.readmsg = fi_no_rma_readmsg,
	.write = fi_no_rma_write,
	.writev = fi_no_rma_writev,
	.writemsg = fi_no_rma_writemsg,
	.inject = fi_no_rma_inject,
	.writedata = fi_no_rma_writedata,
	.injectdata = fi_no_rma_injectdata,
};

struct fi_ops_atomic lnx_atomic_ops = {
	.size = sizeof(struct fi_ops_atomic),
	.write = fi_no_atomic_write,
	.writev = fi_no_atomic_writev,
	.writemsg = fi_no_atomic_writemsg,
	.inject = fi_no_atomic_inject,
	.readwrite = fi_no_atomic_readwrite,
	.readwritev = fi_no_atomic_readwritev,
	.readwritemsg = fi_no_atomic_readwritemsg,
	.compwrite = fi_no_atomic_compwrite,
	.compwritev = fi_no_atomic_compwritev,
	.compwritemsg = fi_no_atomic_compwritemsg,
	.writevalid = fi_no_atomic_writevalid,
	.readwritevalid = fi_no_atomic_readwritevalid,
	.compwritevalid = fi_no_atomic_compwritevalid,
};


