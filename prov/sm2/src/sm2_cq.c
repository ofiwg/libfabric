/*
 * Copyright (c) 2013-2017 Intel Corporation. All rights reserved.
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include <stdlib.h>
#include <string.h>

#include "sm2.h"

static int sm2_peer_cq_close(struct fid *fid)
{
	free(container_of(fid, struct fid_peer_cq, fid));
	return 0;
}

static int sm2_cq_close(struct fid *fid)
{
	int ret;
	struct sm2_cq *sm2_cq;

	sm2_cq = container_of(fid, struct sm2_cq, util_cq.cq_fid.fid);

	ret = ofi_cq_cleanup(&sm2_cq->util_cq);
	if (ret)
		return ret;

	if (!(sm2_cq->util_cq.flags & FI_PEER))
		fi_close(&sm2_cq->peer_cq->fid);

	free(sm2_cq);
	return 0;
}

static struct fi_ops sm2_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = sm2_cq_close,
	.bind = fi_no_bind,
	.control = ofi_cq_control,
	.ops_open = fi_no_ops_open,
};

static ssize_t sm2_peer_cq_write(struct fid_peer_cq *cq, void *context, uint64_t flags,
		size_t len, void *buf, uint64_t data, uint64_t tag,
		fi_addr_t src)
{
	struct sm2_cq *sm2_cq;
	int ret;

	sm2_cq = cq->fid.context;

	if (src == FI_ADDR_NOTAVAIL)
		ret = ofi_cq_write(&sm2_cq->util_cq, context, flags, len,
				   buf, data, tag);
	else
		ret = ofi_cq_write_src(&sm2_cq->util_cq, context, flags, len,
				       buf, data, tag, src);

	if (sm2_cq->util_cq.wait)
		sm2_cq->util_cq.wait->signal(sm2_cq->util_cq.wait);

	return ret;
}

static ssize_t sm2_peer_cq_writeerr(struct fid_peer_cq *cq,
				    const struct fi_cq_err_entry *err_entry)
{
	return ofi_cq_write_error(&((struct sm2_cq *)
				  (cq->fid.context))->util_cq, err_entry);
}

static struct fi_ops_cq_owner sm2_peer_cq_owner_ops = {
	.size = sizeof(struct fi_ops_cq_owner),
	.write = &sm2_peer_cq_write,
	.writeerr = &sm2_peer_cq_writeerr,
};

static struct fi_ops sm2_peer_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = sm2_peer_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int sm2_init_peer_cq(struct sm2_cq *sm2_cq)
{
	sm2_cq->peer_cq = calloc(1, sizeof(*sm2_cq->peer_cq));
	if (!sm2_cq->peer_cq)
		return -FI_ENOMEM;

	sm2_cq->peer_cq->fid.fclass = FI_CLASS_PEER_CQ;
	sm2_cq->peer_cq->fid.context = sm2_cq;
	sm2_cq->peer_cq->fid.ops = &sm2_peer_cq_fi_ops;
	sm2_cq->peer_cq->owner_ops = &sm2_peer_cq_owner_ops;

	return 0;
}

static ssize_t sm2_cq_read(struct fid_cq *cq_fid, void *buf, size_t count)
{
	return ofi_cq_readfrom(cq_fid, buf, count, NULL);
}

static struct fi_ops_cq sm2_peer_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = sm2_cq_read,
	.readfrom = fi_no_cq_readfrom,
	.readerr = fi_no_cq_readerr,
	.sread = fi_no_cq_sread,
	.sreadfrom = fi_no_cq_sreadfrom,
	.signal = fi_no_cq_signal,
	.strerror = fi_no_cq_strerror,
};

int sm2_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context)
{
	struct sm2_cq *sm2_cq;
	int ret;

	switch (attr->wait_obj) {
	case FI_WAIT_UNSPEC:
		attr->wait_obj = FI_WAIT_YIELD;
		/* fall through */
	case FI_WAIT_NONE:
	case FI_WAIT_YIELD:
		break;
	default:
		FI_INFO(&sm2_prov, FI_LOG_CQ, "CQ wait not yet supported\n");
		return -FI_ENOSYS;
	}

	sm2_cq = calloc(1, sizeof(*sm2_cq));
	if (!sm2_cq)
		return -FI_ENOMEM;

	ret = ofi_cq_init(&sm2_prov, domain, attr, &sm2_cq->util_cq,
			  &ofi_cq_progress, context);
	if (ret)
		goto free;

	if (attr->flags & FI_PEER) {
		sm2_cq->peer_cq = ((struct fi_peer_cq_context *) context)->cq;
		sm2_cq->util_cq.cq_fid.ops = &sm2_peer_cq_ops;
	} else {
		ret = sm2_init_peer_cq(sm2_cq);
		if (ret)
			goto cleanup;
	}

	sm2_cq->util_cq.cq_fid.fid.ops = &sm2_cq_fi_ops;
	(*cq_fid) = &sm2_cq->util_cq.cq_fid;
	return 0;

cleanup:
	(void) ofi_cq_cleanup(&sm2_cq->util_cq);
free:
	free(sm2_cq);
	return ret;
}
