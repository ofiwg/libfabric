/*
 * Copyright (c) 2018 Intel Corporation. All rights reserved.
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

#include "tcpx.h"

static ssize_t tcpx_eq_read(struct fid_eq *eq_fid, uint32_t *event,
			    void *buf, size_t len, uint64_t flags)
{
	struct util_eq *eq;
	struct util_event *entry;
	ssize_t ret;

	eq = container_of(eq_fid, struct util_eq, eq_fid);

	fastlock_acquire(&eq->lock);
	if (slist_empty(&eq->list)) {
		fastlock_release(&eq->lock);
		tcpx_conn_mgr_run(eq);
		fastlock_acquire(&eq->lock);
		if (slist_empty(&eq->list)) {
			ret = -FI_EAGAIN;
			FI_DBG(&tcpx_prov, FI_LOG_EQ, "eq empty\n");
			goto out;
		}
	}

	entry = container_of(eq->list.head, struct util_event, entry);
	if (entry->err && !(flags & UTIL_FLAG_ERROR)) {
		ret = -FI_EAVAIL;
		goto out;
	} else if (!entry->err && (flags & UTIL_FLAG_ERROR)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	if (event)
		*event = entry->event;
	if (buf) {
		ret = MIN(len, (size_t)entry->size);
		memcpy(buf, entry->data, ret);
	}  else {
		ret = 0;
	}

	if (!(flags & FI_PEEK)) {
		slist_remove_head(&eq->list);
		free(entry);
	}
out:
	fastlock_release(&eq->lock);
	return ret;
}

static ssize_t tcpx_eq_readerr(struct fid_eq *eq_fid, struct fi_eq_err_entry *buf,
			       uint64_t flags)
{
	return tcpx_eq_read(eq_fid, NULL, buf, sizeof(*buf),
			    flags | UTIL_FLAG_ERROR);
}

static ssize_t tcpx_eq_sread(struct fid_eq *eq_fid, uint32_t *event, void *buf,
			     size_t len, int timeout, uint64_t flags)
{
	struct util_eq *eq;

	eq = container_of(eq_fid, struct util_eq, eq_fid);
	if (!eq->internal_wait) {
		FI_WARN(eq->prov, FI_LOG_EQ, "EQ not configured for sread\n");
		return -FI_ENOSYS;
	}

	fi_wait(&eq->wait->wait_fid, timeout);
	return fi_eq_read(eq_fid, event, buf, len, flags);
}

int tcpx_eq_create(struct fid_fabric *fabric_fid, struct fi_eq_attr *attr,
		   struct fid_eq **eq_fid, void *context)
{
	struct util_eq *eq;
	struct fi_wait_attr wait_attr;
	struct fid_wait *wait;
	int ret;

	ret = ofi_eq_create(fabric_fid, attr, eq_fid, context);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_EQ,
			"EQ creation failed\n");
		return ret;
	}

	eq = container_of(*eq_fid, struct util_eq, eq_fid);

	eq->eq_fid.ops->read	= tcpx_eq_read;
	eq->eq_fid.ops->readerr	= tcpx_eq_readerr;
	eq->eq_fid.ops->sread	= tcpx_eq_sread;

	if (!eq->wait) {
		memset(&wait_attr, 0, sizeof wait_attr);
		wait_attr.wait_obj = FI_WAIT_FD;
		ret = fi_wait_open(fabric_fid, &wait_attr, &wait);
		if (ret) {
			FI_WARN(&tcpx_prov, FI_LOG_EQ,
				"opening wait failed\n");
			goto err;
		}
		eq->internal_wait = 1;
		eq->wait = container_of(wait, struct util_wait,
					wait_fid);
	}
	return 0;
err:
	fi_close(&eq->eq_fid.fid);
	return ret;
}
