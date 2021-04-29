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
#include <sys/types.h>
#include <ofi_util.h>
#include <ofi_iov.h>
#include "tcpx.h"

int tcpx_send_msg(struct tcpx_ep *ep)
{
	struct tcpx_xfer_entry *tx_entry;
	ssize_t ret;

	assert(ep->cur_tx.entry);
	tx_entry = ep->cur_tx.entry;
	ret = ofi_bsock_sendv(&ep->bsock, tx_entry->iov, tx_entry->iov_cnt);
	if (ret < 0)
		return ret;

	ep->cur_tx.data_left -= ret;
	if (ep->cur_tx.data_left) {
		ofi_consume_iov(tx_entry->iov, &tx_entry->iov_cnt, ret);
		return -FI_EAGAIN;
	}
	return FI_SUCCESS;
}

ssize_t tcpx_recv_hdr(struct tcpx_ep *ep)
{
	size_t len;
	void *buf;

	assert(!ep->cur_rx.entry);
	buf = (uint8_t *) &ep->cur_rx.hdr + ep->cur_rx.hdr_done;
	len = ep->cur_rx.hdr_len - ep->cur_rx.hdr_done;

	return ofi_bsock_recv(&ep->bsock, buf, len);
}

int tcpx_recv_msg_data(struct tcpx_ep *ep)
{
	struct tcpx_xfer_entry *rx_entry;
	ssize_t ret;

	if (!ep->cur_rx.data_left)
		return FI_SUCCESS;

	rx_entry = ep->cur_rx.entry;
	assert(ofi_total_iov_len(rx_entry->iov, rx_entry->iov_cnt) >=
	       ep->cur_rx.data_left);
	ret = ofi_bsock_recvv(&ep->bsock, rx_entry->iov, rx_entry->iov_cnt);
	if (ret < 0)
		return ret;

	ep->cur_rx.data_left -= ret;
	if (!ep->cur_rx.data_left)
		return FI_SUCCESS;

	ofi_consume_iov(rx_entry->iov, &rx_entry->iov_cnt, ret);
	if (!rx_entry->iov_cnt || !rx_entry->iov[0].iov_len)
		return -FI_ETRUNC;

	return -FI_EAGAIN;
}
