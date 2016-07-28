/*
 * Copyright (c) 2013-2016 Intel Corporation. All rights reserved.
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

#include "rxm.h"

static int rxm_msg_cq_read(struct util_cq *util_cq, struct fid_cq *cq,
		struct fi_cq_tagged_entry *comp)
{
	struct util_cq_err_entry *entry;
	int ret;

	ret = fi_cq_read(cq, comp, 1);
	if (ret == -FI_EAVAIL) {
		entry = calloc(1, sizeof(*entry));
		if (!entry) {
			FI_WARN(&rxm_prov, FI_LOG_CQ,
					"Unable to allocate util_cq_err_entry\n");
			return -FI_ENOMEM;
		}
		ret = fi_cq_readerr(cq, &entry->err_entry, 0);
		if (ret < 0) {
			FI_WARN(&rxm_prov, FI_LOG_CQ, "Unable to fi_cq_readerr\n");
			free(entry);
			return ret;
		} else {
			FI_WARN(&rxm_prov, FI_LOG_CQ, "fi_cq_readerr: %s\n",
					fi_cq_strerror(cq, entry->err_entry.prov_errno,
						entry->err_entry.err_data, NULL, 0));
		}
		slist_insert_tail(&entry->list_entry, &util_cq->err_list);
		comp->flags = UTIL_FLAG_ERROR;
	}

	return ret;
}

void rxm_cq_progress(struct util_cq *util_cq)
{
	ssize_t ret = 0;
	struct rxm_cq *rxm_cq;
	struct fi_cq_tagged_entry *comp;

	rxm_cq = container_of(util_cq, struct rxm_cq, util_cq);

	fastlock_acquire(&util_cq->cq_lock);
	do {
		if (cirque_isfull(util_cq->cirq))
			goto out;

		comp = cirque_tail(util_cq->cirq);
		ret = rxm_msg_cq_read(util_cq, rxm_cq->msg_cq, comp);
		if (ret < 0)
			goto out;
		cirque_commit(util_cq->cirq);
	} while (ret > 0);
out:
	fastlock_release(&util_cq->cq_lock);
}

static int rxm_cq_close(struct fid *fid)
{
	struct rxm_cq *rxm_cq;
	int ret, retv = 0;

	rxm_cq = container_of(fid, struct rxm_cq, util_cq.cq_fid.fid);

	ret = ofi_cq_cleanup(&rxm_cq->util_cq);
	if (ret)
		retv = ret;

	ret = fi_close(&rxm_cq->msg_cq->fid);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Unable to close MSG CQ\n");
		retv = ret;
	}
	free(rxm_cq);
	return retv;
}

static struct fi_ops rxm_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxm_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int rxm_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq_fid, void *context)
{
	struct rxm_domain *rxm_domain;
	struct rxm_cq *rxm_cq;
	int ret;

	rxm_cq = calloc(1, sizeof(*rxm_cq));
	if (!rxm_cq)
		return -FI_ENOMEM;

	rxm_domain = container_of(domain, struct rxm_domain, util_domain.domain_fid);

	ret = fi_cq_open(rxm_domain->msg_domain, attr, &rxm_cq->msg_cq, context);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Unable to open MSG CQ\n");
		goto err1;
	}

	ret = ofi_cq_init(&rxm_prov, domain, attr, &rxm_cq->util_cq,
			&rxm_cq_progress, context);
	if (ret)
		goto err2;

	*cq_fid = &rxm_cq->util_cq.cq_fid;
	/* Override util_cq_fi_ops */
	(*cq_fid)->fid.ops = &rxm_cq_fi_ops;
	return 0;
err2:
	fi_close(&rxm_cq->msg_cq->fid);
err1:
	free(rxm_cq);
	return ret;
}
