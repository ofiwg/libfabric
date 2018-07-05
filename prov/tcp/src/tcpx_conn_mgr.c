/*
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *	   Redistribution and use in source and binary forms, with or
 *	   without modification, are permitted provided that the following
 *	   conditions are met:
 *
 *		- Redistributions of source code must retain the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer.
 *
 *		- Redistributions in binary form must reproduce the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer in the documentation and/or other materials
 *		  provided with the distribution.
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

#include <rdma/fi_errno.h>

#include <ofi_prov.h>
#include "tcpx.h"
#include <poll.h>
#include <sys/types.h>
#include <ofi_util.h>

static int rx_cm_data(SOCKET fd, struct ofi_ctrl_hdr *hdr,
		      int type, struct poll_fd_info *poll_info)
{
	ssize_t ret;

	ret = ofi_recv_socket(fd, hdr,
			      sizeof(*hdr), MSG_WAITALL);
	if (ret != sizeof(*hdr))
		return -FI_EIO;

	if (hdr->type != type)
		return -FI_ECONNREFUSED;

	if (hdr->version != OFI_CTRL_VERSION)
		return -FI_ENOPROTOOPT;

	poll_info->cm_data_sz = ntohs(hdr->seg_size);
	if (poll_info->cm_data_sz) {
		if (poll_info->cm_data_sz > TCPX_MAX_CM_DATA_SIZE)
			return -FI_EINVAL;

		ret = ofi_recv_socket(fd, poll_info->cm_data,
				      poll_info->cm_data_sz, MSG_WAITALL);
		if ((size_t) ret != poll_info->cm_data_sz)
			return -FI_EIO;
	}
	return FI_SUCCESS;
}

static int tx_cm_data(SOCKET fd, uint8_t type, struct poll_fd_info *poll_info)
{
	struct ofi_ctrl_hdr hdr;
	ssize_t ret;

	memset(&hdr, 0, sizeof(hdr));
	hdr.version = OFI_CTRL_VERSION;
	hdr.type = type;
	hdr.seg_size = htons((uint16_t) poll_info->cm_data_sz);

	ret = ofi_send_socket(fd, &hdr, sizeof(hdr), MSG_NOSIGNAL);
	if (ret != sizeof(hdr))
		return -FI_EIO;

	if (poll_info->cm_data_sz) {
		ret = ofi_send_socket(fd, poll_info->cm_data,
				      poll_info->cm_data_sz, MSG_NOSIGNAL);
		if ((size_t) ret != poll_info->cm_data_sz)
			return -FI_EIO;
	}
	return FI_SUCCESS;
}

static int tcpx_ep_msg_xfer_enable(struct tcpx_ep *ep)
{
	int ret;

	fastlock_acquire(&ep->lock);
	if (ep->cm_state != TCPX_EP_CONNECTING) {
		fastlock_release(&ep->lock);
		return -FI_EINVAL;
	}
	ep->progress_func = tcpx_ep_progress;
	ret = fi_fd_nonblock(ep->conn_fd);
	if (ret)
		goto err;

	ret = tcpx_cq_wait_ep_add(ep);
	if (ret)
		goto err;

	ep->cm_state = TCPX_EP_CONNECTED;
err:
	fastlock_release(&ep->lock);
	return ret;
}

void tcpx_conn_mgr_run(struct util_eq *eq)
{
}
