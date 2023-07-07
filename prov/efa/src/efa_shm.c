/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
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
#include <assert.h>
#include "efa.h"
#include "efa_shm.h"


/**
 * @brief construct an unique shm endpoint name (smr_name) from EFA raw address
 *
 * Note even though all shm endpoints are on same instance. But because
 * one instance can have multiple EFA device, it is still necessary
 * to include GID on the name.
 *
 * a smr name consist of the following 4 parts:
 *
 *    GID:   ipv6 address from inet_ntop
 *    QPN:   %04x format
 *    QKEY:  %08x format
 *    UID:   %04x format
 *
 * each part is connected via an underscore.
 *
 * The following is an example:
 *
 *    fe80::4a5:28ff:fe98:e500_0001_12918366_03e8
 *
 * @param[in]		ptr		pointer to raw address (struct efa_ep_addr)
 * @param[out]		smr_name	an unique name for shm ep
 * @param[in,out]	smr_name_len    As input, specify size of the "smr_name" buffer.
 *					As output, specify number of bytes written to the buffer.
 *
 * @return	0 on success.
 * 		negative error code on failure.
 */
int efa_shm_ep_name_construct(char *smr_name, size_t *smr_name_len, struct efa_ep_addr *raw_addr)
{
	char gidstr[INET6_ADDRSTRLEN] = { 0 };
	int ret;

	if (!inet_ntop(AF_INET6, raw_addr->raw, gidstr, INET6_ADDRSTRLEN)) {
		EFA_WARN(FI_LOG_CQ, "Failed to convert GID to string errno: %d\n", errno);
		return -errno;
	}

	ret = snprintf(smr_name, *smr_name_len, "%s_%04x_%08x_%04x",
		       gidstr, raw_addr->qpn, raw_addr->qkey, getuid());
	if (ret < 0)
		return ret;

	if (ret == 0 || ret >= *smr_name_len)
		return -FI_EINVAL;

	/* plus 1 here for the ending '\0' character, which was not
	 * included in ret of snprintf
	 */
	*smr_name_len = ret + 1;
	return FI_SUCCESS;
}

/**
 * @brief Create a shm info object based on application info
 *
 * @param[in] app_info the application info
 * @param[out] shm_info the shm info
 */
void efa_shm_info_create(const struct fi_info *app_info, struct fi_info **shm_info)
{
	int ret;
	struct fi_info *shm_hints;

	shm_hints = fi_allocinfo();
	shm_hints->caps = app_info->caps;
	shm_hints->caps &= ~FI_REMOTE_COMM;

	/*
	 * If application requests FI_HMEM and efa supports it,
	 * make this request to shm as well.
	 */
	shm_hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR;
	if (app_info && (app_info->caps & FI_HMEM)) {
		shm_hints->domain_attr->mr_mode |= FI_MR_HMEM;
	}

	shm_hints->domain_attr->av_type = FI_AV_TABLE;
	shm_hints->domain_attr->caps |= FI_LOCAL_COMM;
	shm_hints->tx_attr->msg_order = FI_ORDER_SAS;
	shm_hints->rx_attr->msg_order = FI_ORDER_SAS;
	/*
	 * Unlike efa, shm does not have FI_COMPLETION in tx/rx_op_flags unless user request
	 * it via hints. That means if user does not request FI_COMPLETION in the hints, and bind
	 * shm cq to shm ep with FI_SELECTIVE_COMPLETION flags,
	 * shm will not write cqe for fi_send* (fi_sendmsg is an exception, as user can specify flags),
	 * similarly for the recv ops. It is common for application like ompi to
	 * bind cq with FI_SELECTIVE_COMPLETION, and call fi_senddata in which it expects libfabric to
	 * write cqe. We should follow this pattern and request FI_COMPLETION to shm as default tx/rx_op_flags.
	 */
	shm_hints->tx_attr->op_flags  = FI_COMPLETION;
	shm_hints->rx_attr->op_flags  = FI_COMPLETION;
	shm_hints->fabric_attr->name = strdup(efa_env.intranode_provider);
	shm_hints->fabric_attr->prov_name = strdup(efa_env.intranode_provider);
	shm_hints->ep_attr->type = FI_EP_RDM;

	ret = fi_getinfo(FI_VERSION(1, 19), NULL, NULL,
	                 OFI_GETINFO_HIDDEN, shm_hints, shm_info);
	fi_freeinfo(shm_hints);
	if (ret) {
		EFA_WARN(FI_LOG_CORE, "Disabling EFA's shared memory support; "
		         "Failed to get info struct for provider %s: %s\n",
		         efa_env.intranode_provider, fi_strerror(-ret));
		*shm_info = NULL;
	} else {
		assert(!strcmp((*shm_info)->fabric_attr->name, efa_env.intranode_provider));
	}
}
