/*
 * Copyright (c) 2022 Intel Corporation. All rights reserved.
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

#include "peer.h"

static int peer_cq_close(struct fid *fid)
{
	struct peer_cq *cq;
	int ret;

	cq = container_of(fid, struct peer_cq, util_cq.cq_fid.fid);

	ret = ofi_cq_cleanup(&cq->util_cq);
	if (ret)
		return ret;

	free(cq);
	return 0;
}

static struct fi_ops peer_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = peer_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_cq peer_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = fi_no_cq_read,
	.readfrom = fi_no_cq_readfrom,
	.readerr = fi_no_cq_readerr,
	.sread = fi_no_cq_sread,
	.sreadfrom = fi_no_cq_sreadfrom,
	.signal = fi_no_cq_signal,
	.strerror = fi_no_cq_strerror,
};

static int peer_cq_init(const struct fi_provider* provider, 
	struct fid_domain *domain, struct fi_cq_attr *attr,
	struct fid_cq **cq_fid, struct fi_peer_cq_context *peer_context)
{
	struct peer_cq *cq;
	int ret;

	if (!attr || !(attr->flags & FI_PEER)) {
		FI_WARN(provider, FI_LOG_CORE, "FI_PEER flag required\n");
                return -FI_EINVAL;
	}

	if (!peer_context || peer_context->size < sizeof(*peer_context)) {
		FI_WARN(provider, FI_LOG_CORE, "invalid peer CQ context\n");
                return -FI_EINVAL;
	}

	cq = calloc(1, sizeof(*cq));
	if (!cq)
		return -FI_ENOMEM;

	cq->peer_cq = peer_context->cq;

	ret = ofi_cq_init(provider, domain, attr, &cq->util_cq, 
		&ofi_cq_progress, NULL);
	if (ret)
		goto err;

	*cq_fid = &cq->util_cq.cq_fid;
	(*cq_fid)->fid.ops = &peer_cq_fi_ops;
	(*cq_fid)->ops = &peer_cq_ops;
	return 0;

err:
	free(cq);
	return ret;
}

int ofi_peer_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context)
{
	struct util_domain *util_domain;
	struct fi_peer_cq_context *peer_context;

	util_domain = container_of(domain, struct util_domain, domain_fid.fid);
	peer_context = context;

	return peer_cq_init(util_domain->fabric->prov, domain, attr, 
		cq_fid, peer_context);
}

ssize_t ofi_peer_cq_write(struct fid_cq *cq_fid, void *context, uint64_t flags,
			size_t len, void *buf, uint64_t data, uint64_t tag,
			fi_addr_t src)
{
	struct peer_cq *cq;

	cq = container_of(cq_fid, struct peer_cq, util_cq.cq_fid);

	return cq->peer_cq->owner_ops->write(cq->peer_cq, context,
					  flags, len, buf, data, tag, src);
}
