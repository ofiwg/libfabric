/*
 * Copyright (c) 2018-2022 Intel Corporation. All rights reserved.
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

#include "xnet.h"


/* If we don't have an EQ, then we're writing an event for an rdm ep.
 * That goes directly on the rdm event list.
 */
int xnet_eq_write(struct util_eq *eq, uint32_t event,
		  const void *buf, size_t len, uint64_t flags)
{
	struct xnet_event *entry;
	const struct fi_eq_entry *eq_event;
	struct xnet_rdm *rdm;

	if (eq)
		return (int) fi_eq_write(&eq->eq_fid, event, buf, len, flags);

	eq_event = buf;
	if (eq_event->fid->fclass == FI_CLASS_EP) {
		rdm = ((struct xnet_conn *) eq_event->fid->context)->rdm;
	} else {
		assert(eq_event->fid->fclass == FI_CLASS_PEP);
		rdm = eq_event->fid->context;
	}

	assert(rdm->util_ep.ep_fid.fid.fclass == FI_CLASS_EP);
	assert(xnet_progress_locked(xnet_rdm2_progress(rdm)));
	entry = malloc(sizeof(*entry));
	if (!entry)
		return -FI_ENOMEM;

	entry->rdm = rdm;
	entry->event = event;
	assert(len >= sizeof(entry->cm_entry));
	memcpy(&entry->cm_entry, buf, sizeof(entry->cm_entry));
	slist_insert_tail(&entry->list_entry,
			  &xnet_rdm2_progress(rdm)->event_list);
	return 0;
}

static ssize_t xnet_eq_read(struct fid_eq *eq_fid, uint32_t *event,
			    void *buf, size_t len, uint64_t flags)
{
	struct xnet_fabric *fabric;
	struct xnet_eq *eq;

	eq = container_of(eq_fid, struct xnet_eq, util_eq.eq_fid);
	fabric = container_of(eq->util_eq.fabric, struct xnet_fabric,
			      util_fabric);
	xnet_progress_all(fabric);

	return ofi_eq_read(eq_fid, event, buf, len, flags);
}

static int xnet_eq_close(struct fid *fid)
{
	struct xnet_eq *eq;
	int ret;

	ret = ofi_eq_cleanup(fid);
	if (ret)
		return ret;

	eq = container_of(fid, struct xnet_eq, util_eq.eq_fid.fid);

	ofi_mutex_destroy(&eq->close_lock);
	free(eq);
	return 0;
}

static struct fi_ops_eq xnet_eq_ops = {
	.size = sizeof(struct fi_ops_eq),
	.read = xnet_eq_read,
	.readerr = ofi_eq_readerr,
	.sread = ofi_eq_sread,
	.write = ofi_eq_write,
	.strerror = ofi_eq_strerror,
};

static struct fi_ops xnet_eq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = xnet_eq_close,
	.bind = fi_no_bind,
	.control = ofi_eq_control,
	.ops_open = fi_no_ops_open,
};

int xnet_eq_create(struct fid_fabric *fabric_fid, struct fi_eq_attr *attr,
		   struct fid_eq **eq_fid, void *context)
{
	struct xnet_fabric *fabric;
	struct xnet_eq *eq;
	int ret;

	eq = calloc(1, sizeof(*eq));
	if (!eq)
		return -FI_ENOMEM;

	ret = ofi_eq_init(fabric_fid, attr, &eq->util_eq.eq_fid, context);
	if (ret) {
		FI_WARN(&xnet_prov, FI_LOG_EQ,
			"EQ creation failed\n");
		goto err1;
	}

	ret = ofi_mutex_init(&eq->close_lock);
	if (ret)
		goto err2;

	eq->util_eq.eq_fid.ops	= &xnet_eq_ops;
	eq->util_eq.eq_fid.fid.ops = &xnet_eq_fi_ops;

	if (attr->wait_obj != FI_WAIT_NONE) {
		fabric = container_of(fabric_fid, struct xnet_fabric,
				      util_fabric.fabric_fid);
		ret = xnet_start_all(fabric);
		if (ret)
			goto err3;
	}

	*eq_fid = &eq->util_eq.eq_fid;
	return 0;

err3:
	ofi_mutex_destroy(&eq->close_lock);
err2:
	ofi_eq_cleanup(&eq->util_eq.eq_fid.fid);
err1:
	free(eq);
	return ret;
}
