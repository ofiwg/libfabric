/*
 * Copyright (c) 2019-2023 Amazon.com, Inc. or its affiliates.
 * All rights reserved.
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
#include "efa.h"
#include "efa_rdm_cq.h"
#include "ofi_util.h"

static
const char *efa_rdm_cq_strerror(struct fid_cq *cq_fid, int prov_errno,
				const void *err_data, char *buf, size_t len)
{
	return efa_strerror(prov_errno, err_data);
}

/**
 * @brief close a CQ of EFA RDM endpoint
 *
 * @param[in,out]	fid	fid of the CQ to be closed
 * @returns		0 on sucesss,
 * 			negative libfabric error code on error
 * @relates efa_rdm_cq
 */
static
int efa_rdm_cq_close(struct fid *fid)
{
	int ret, retv;
	struct efa_rdm_cq *cq;

	retv = 0;

	cq = container_of(fid, struct efa_rdm_cq, util_cq.cq_fid.fid);

	if (cq->shm_cq) {
		ret = fi_close(&cq->shm_cq->fid);
		if (ret) {
			EFA_WARN(FI_LOG_CQ, "Unable to close shm cq: %s\n", fi_strerror(-ret));
			retv = ret;
		}
	}

	ret = ofi_cq_cleanup(&cq->util_cq);
	if (ret)
		return ret;
	free(cq);
	return retv;
}

static struct fi_ops efa_rdm_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_rdm_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static ssize_t efa_rdm_cq_readfrom(struct fid_cq *cq_fid, void *buf, size_t count, fi_addr_t *src_addr)
{
	struct efa_rdm_cq *cq;
	ssize_t ret;
	struct util_srx_ctx *srx_ctx;

	cq = container_of(cq_fid, struct efa_rdm_cq, util_cq.cq_fid.fid);

	srx_ctx = cq->util_cq.domain->srx->ep_fid.fid.context;

	ofi_genlock_lock(srx_ctx->lock);

	if (cq->shm_cq)
		fi_cq_read(cq->shm_cq, NULL, 0);

	ret = ofi_cq_read_entries(&cq->util_cq, buf, count, src_addr);

	if (ret > 0)
		goto out;

	ret = ofi_cq_readfrom(&cq->util_cq.cq_fid, buf, count, src_addr);

out:
	ofi_genlock_unlock(srx_ctx->lock);

	return ret;
}

static struct fi_ops_cq efa_rdm_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = ofi_cq_read,
	.readfrom = efa_rdm_cq_readfrom,
	.readerr = ofi_cq_readerr,
	.sread = fi_no_cq_sread,
	.sreadfrom = fi_no_cq_sreadfrom,
	.signal = fi_no_cq_signal,
	.strerror = efa_rdm_cq_strerror,
};

/**
 * @brief create a CQ for EFA RDM provider
 *
 * Note that EFA RDM provider used the util_cq as its CQ
 *
 * @param[in]		domain		efa domain
 * @param[in]		attr		cq attribuite
 * @param[out]		cq_fid 		fid of the created cq
 * @param[in]		context 	currently EFA provider does not accept any context
 * @returns		0 on success
 * 			negative libfabric error code on error
 * @relates efa_rdm_cq
 */
int efa_rdm_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		    struct fid_cq **cq_fid, void *context)
{
	int ret;
	struct efa_rdm_cq *cq;
	struct efa_domain *efa_domain;
	struct fi_cq_attr shm_cq_attr = {0};
	struct fi_peer_cq_context peer_cq_context = {0};

	if (attr->wait_obj != FI_WAIT_NONE)
		return -FI_ENOSYS;

	cq = calloc(1, sizeof(*cq));
	if (!cq)
		return -FI_ENOMEM;

	efa_domain = container_of(domain, struct efa_domain,
				  util_domain.domain_fid);
	/* Override user cq size if it's less than recommended cq size */
	attr->size = MAX(efa_domain->rdm_cq_size, attr->size);

	ret = ofi_cq_init(&efa_prov, domain, attr, &cq->util_cq,
			  &ofi_cq_progress, context);

	if (ret)
		goto free;

	*cq_fid = &cq->util_cq.cq_fid;
	(*cq_fid)->fid.ops = &efa_rdm_cq_fi_ops;
	(*cq_fid)->ops = &efa_rdm_cq_ops;

	/* open shm cq as peer cq */
	if (efa_domain->shm_domain) {
		memcpy(&shm_cq_attr, attr, sizeof(*attr));
		/* Bind ep with shm provider's cq */
		shm_cq_attr.flags |= FI_PEER;
		peer_cq_context.size = sizeof(peer_cq_context);
		peer_cq_context.cq = cq->util_cq.peer_cq;
		ret = fi_cq_open(efa_domain->shm_domain, &shm_cq_attr,
				 &cq->shm_cq, &peer_cq_context);
		if (ret) {
			EFA_WARN(FI_LOG_CQ, "Unable to open shm cq: %s\n", fi_strerror(-ret));
			goto free;
		}
	}

	return 0;
free:
	free(cq);
	return ret;
}
