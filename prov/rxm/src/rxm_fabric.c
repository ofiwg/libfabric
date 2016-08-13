/*
 * Copyright (c) 2016 Intel Corporation, Inc.  All rights reserved.
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

#include <fi_util.h>

#include "rxm.h"

static struct fi_ops_fabric rxm_fabric_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = rxm_domain_open,
	.passive_ep = fi_no_passive_ep,
	.eq_open = ofi_eq_create,
	.wait_open = ofi_wait_fd_open,
	.trywait = ofi_trywait
};

static int rxm_fabric_close(fid_t fid)
{
	struct rxm_fabric *rxm_fabric;
	struct fi_eq_entry entry = {0};
	ssize_t rd;
	int ret;

	rxm_fabric = container_of(fid, struct rxm_fabric, util_fabric.fabric_fid.fid);

	rd = fi_eq_write(rxm_fabric->msg_eq, FI_NOTIFY, &entry, sizeof(entry), 0);
	if (rd != sizeof(entry)) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Unable to notify listener thread\n");
		return rd;
	}

	pthread_join(rxm_fabric->msg_listener_thread, NULL);

	ret = fi_close(&rxm_fabric->msg_eq->fid);
	if (ret)
		return ret;

	ret = fi_close(&rxm_fabric->msg_fabric->fid);
	if (ret)
		return ret;

	ret = ofi_fabric_close(&rxm_fabric->util_fabric);
	if (ret)
		return ret;

	free(rxm_fabric);
	return 0;
}

static struct fi_ops rxm_fabric_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxm_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int rxm_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		void *context)
{
	struct rxm_fabric *rxm_fabric;
	struct fi_info hints, *msg_info;
	struct fi_eq_attr eq_attr;
	int ret;

	rxm_fabric = calloc(1, sizeof(*rxm_fabric));
	if (!rxm_fabric)
		return -FI_ENOMEM;

	ret = ofi_fabric_init(&rxm_prov, &rxm_fabric_attr, attr,
			     &rxm_fabric->util_fabric, context, FI_MATCH_PREFIX);
	if (ret)
		goto err1;

	memset(&hints, 0, sizeof hints);
	if (!(hints.fabric_attr = calloc(1, sizeof(*hints.fabric_attr)))) {
		ret = -FI_ENOMEM;
		goto err2;
	}
	hints.fabric_attr->name = attr->name;
	hints.mode = rxm_info.mode;

	ret = ofix_getinfo(rxm_prov.version, NULL, NULL, 0, &rxm_util_prov,
			&hints, rxm_alter_layer_info,
			rxm_alter_base_info, 1, &msg_info);
	if (ret) {
		ret = -FI_EINVAL;
		goto err3;
	}

	ret = fi_fabric(msg_info->fabric_attr, &rxm_fabric->msg_fabric, context);
	if (ret) {
		goto err4;
	}

	eq_attr.wait_obj = FI_WAIT_UNSPEC;
	eq_attr.flags = FI_WRITE;

	ret = fi_eq_open(rxm_fabric->msg_fabric, &eq_attr, &rxm_fabric->msg_eq, NULL);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Unable to open msg EQ\n");
		goto err5;
	}

	if (pthread_create(&rxm_fabric->msg_listener_thread, 0,
				rxm_msg_listener, rxm_fabric)) {
		ret = -errno;
		FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Unable to create msg_cm_listener_thread\n");
		goto err6;
	}

	*fabric = &rxm_fabric->util_fabric.fabric_fid;
	(*fabric)->fid.ops = &rxm_fabric_fi_ops;
	(*fabric)->ops = &rxm_fabric_ops;

	free(hints.fabric_attr);
	fi_freeinfo(msg_info);
	return 0;
err6:
	fi_close(&rxm_fabric->msg_eq->fid);
err5:
	fi_close(&rxm_fabric->msg_fabric->fid);
err4:
	fi_freeinfo(msg_info);
err3:
	free(hints.fabric_attr);
err2:
	ofi_fabric_close(&rxm_fabric->util_fabric);
err1:
	free(rxm_fabric);
	return ret;
}
