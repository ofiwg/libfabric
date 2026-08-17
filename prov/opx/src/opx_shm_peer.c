/*
 * Copyright (C) 2026 Cornelis Networks.
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

#include "rdma/opx/fi_opx.h"
#include "opx_shm_peer.h"

void opx_shm_info_create(const struct fi_info *app_info, struct fi_info **shm_info)
{
	struct fi_info *shm_hints;
	const char     *shm_provider = "shm";
	int		ret;

	*shm_info = NULL;

	/*
	 * The shm provider only handles node-local peers.  If the application
	 * explicitly requires remote communication but not local, there is no
	 * reason to stand up an intra-node peer; OPX loops such traffic back
	 * over its own fabric path.
	 */
	if ((app_info->caps & FI_REMOTE_COMM) && !(app_info->caps & FI_LOCAL_COMM)) {
		FI_INFO(fi_opx_global.prov, FI_LOG_CORE,
			"OPX will not use shm for intra-node communication because the application does not require local communication\n");
		return;
	}

	shm_hints = fi_allocinfo();
	if (!shm_hints) {
		FI_WARN(fi_opx_global.prov, FI_LOG_CORE, "Disabling OPX shm peer; fi_allocinfo() failed\n");
		return;
	}

	shm_hints->caps = app_info->caps & ~FI_REMOTE_COMM;

	/* Match OPX's offset-based RMA addressing (OPX advertises mr_mode
	 * without FI_MR_VIRT_ADDR): a node-local RMA carries a 0-based offset
	 * that the shm peer resolves against the target MR, so do not request
	 * FI_MR_VIRT_ADDR (that would put shm in fast-rma/absolute-VA mode). */
	shm_hints->domain_attr->mr_mode = 0;
	if (app_info->caps & FI_HMEM) {
		shm_hints->domain_attr->mr_mode |= FI_MR_HMEM;
	}

	shm_hints->domain_attr->threading = app_info->domain_attr->threading;
	shm_hints->domain_attr->av_type	  = FI_AV_TABLE;
	shm_hints->domain_attr->caps |= FI_LOCAL_COMM | FI_PEER | FI_AV_USER_ID;
	shm_hints->tx_attr->msg_order	  = FI_ORDER_SAS;
	shm_hints->rx_attr->msg_order	  = FI_ORDER_SAS;
	shm_hints->tx_attr->op_flags	  = app_info->tx_attr->op_flags;
	shm_hints->rx_attr->op_flags	  = app_info->rx_attr->op_flags;
	shm_hints->fabric_attr->name	  = strdup(shm_provider);
	shm_hints->fabric_attr->prov_name = strdup(shm_provider);
	shm_hints->ep_attr->type	  = FI_EP_RDM;

	if (!shm_hints->fabric_attr->name || !shm_hints->fabric_attr->prov_name) {
		FI_WARN(fi_opx_global.prov, FI_LOG_CORE, "Disabling OPX shm peer; strdup() failed\n");
		fi_freeinfo(shm_hints);
		return;
	}

	ret = fi_getinfo(FI_VERSION(1, 19), NULL, NULL, OFI_GETINFO_HIDDEN, shm_hints, shm_info);
	fi_freeinfo(shm_hints);
	if (ret) {
		FI_WARN(fi_opx_global.prov, FI_LOG_CORE,
			"Disabling OPX shm peer; failed to get shm provider info: %s\n", fi_strerror(-ret));
		*shm_info = NULL;
		return;
	}

	assert(!strncmp((*shm_info)->fabric_attr->name, shm_provider, strlen(shm_provider)));
}
