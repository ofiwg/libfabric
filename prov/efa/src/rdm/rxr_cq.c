/*
 * Copyright (c) 2019-2022 Amazon.com, Inc. or its affiliates.
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

#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <ofi_iov.h>
#include <ofi_recvwin.h>
#include "efa.h"
#include "rxr_rma.h"
#include "rxr_msg.h"
#include "rxr_cntr.h"
#include "rxr_read.h"
#include "rxr_atomic.h"
#include "rxr_pkt_cmd.h"

#include "rxr_tp.h"

static const char *rxr_cq_strerror(struct fid_cq *cq_fid, int prov_errno,
				   const void *err_data, char *buf, size_t len)
{
	return efa_strerror(prov_errno);
}





static int rxr_cq_close(struct fid *fid)
{
	int ret;
	struct util_cq *cq;

	cq = container_of(fid, struct util_cq, cq_fid.fid);
	ret = ofi_cq_cleanup(cq);
	if (ret)
		return ret;
	free(cq);
	return 0;
}

static struct fi_ops rxr_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxr_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_cq rxr_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = ofi_cq_read,
	.readfrom = ofi_cq_readfrom,
	.readerr = ofi_cq_readerr,
	.sread = fi_no_cq_sread,
	.sreadfrom = fi_no_cq_sreadfrom,
	.signal = fi_no_cq_signal,
	.strerror = rxr_cq_strerror,
};

int rxr_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context)
{
	int ret;
	struct util_cq *cq;
	struct efa_domain *efa_domain;

	if (attr->wait_obj != FI_WAIT_NONE)
		return -FI_ENOSYS;

	cq = calloc(1, sizeof(*cq));
	if (!cq)
		return -FI_ENOMEM;

	efa_domain = container_of(domain, struct efa_domain,
				  util_domain.domain_fid);
	/* Override user cq size if it's less than recommended cq size */
	attr->size = MAX(efa_domain->rdm_cq_size, attr->size);

	ret = ofi_cq_init(&rxr_prov, domain, attr, cq,
			  &ofi_cq_progress, context);

	if (ret)
		goto free;

	*cq_fid = &cq->cq_fid;
	(*cq_fid)->fid.ops = &rxr_cq_fi_ops;
	(*cq_fid)->ops = &rxr_cq_ops;
	return 0;
free:
	free(cq);
	return ret;
}
