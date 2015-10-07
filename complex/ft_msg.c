/*
 * Copyright (c) 2013-2015 Intel Corporation.  All rights reserved.
 *
 * This software is available to you under the BSD license below:
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AWV
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <string.h>
#include <time.h>

#include "fabtest.h"


static int ft_post_recv(void)
{
	struct fi_msg msg;
	int ret;

	switch (test_info.class_function) {
	case FT_FUNC_SENDV:
		ft_format_iov(ft_rx_ctrl.iov, ft_ctrl.iov_array[ft_rx_ctrl.iov_iter],
				ft_rx_ctrl.buf, ft_rx_ctrl.msg_size);
		ret = fi_recvv(ft_rx_ctrl.ep, ft_rx_ctrl.iov, ft_rx_ctrl.iov_desc,
				ft_ctrl.iov_array[ft_rx_ctrl.iov_iter], ft_rx_ctrl.addr, NULL);
		ft_next_iov_cnt(&ft_rx_ctrl, fabric_info->rx_attr->iov_limit);
		break;
	case FT_FUNC_SENDMSG:
		ft_format_iov(ft_rx_ctrl.iov, ft_ctrl.iov_array[ft_rx_ctrl.iov_iter],
				ft_rx_ctrl.buf, ft_rx_ctrl.msg_size);
		msg.msg_iov = ft_rx_ctrl.iov;
		msg.desc = ft_rx_ctrl.iov_desc;
		msg.iov_count = ft_ctrl.iov_array[ft_rx_ctrl.iov_iter];
		msg.addr = ft_rx_ctrl.addr;
		msg.context = NULL;
		msg.data = 0;
		ret = fi_recvmsg(ft_rx_ctrl.ep, &msg, 0);
		ft_next_iov_cnt(&ft_rx_ctrl, fabric_info->rx_attr->iov_limit);
		break;
	default:
		ret = fi_recv(ft_rx_ctrl.ep, ft_rx_ctrl.buf, ft_rx_ctrl.msg_size,
				ft_rx_ctrl.memdesc, ft_rx_ctrl.addr, NULL);
		break;
	}

	return ret;
}

static int ft_post_trecv(void)
{
	struct fi_msg_tagged msg;
	int ret;

	switch (test_info.class_function) {
	case FT_FUNC_SENDV:
		ft_format_iov(ft_rx_ctrl.iov, ft_ctrl.iov_array[ft_rx_ctrl.iov_iter],
				ft_rx_ctrl.buf, ft_rx_ctrl.msg_size);
		ret = fi_trecvv(ft_rx_ctrl.ep, ft_rx_ctrl.iov, ft_rx_ctrl.iov_desc,
				ft_ctrl.iov_array[ft_rx_ctrl.iov_iter], ft_rx_ctrl.addr,
				ft_rx_ctrl.tag, 0, NULL);
		ft_next_iov_cnt(&ft_rx_ctrl, fabric_info->rx_attr->iov_limit);
		break;
	case FT_FUNC_SENDMSG:
		ft_format_iov(ft_rx_ctrl.iov, ft_ctrl.iov_array[ft_rx_ctrl.iov_iter],
				ft_rx_ctrl.buf, ft_rx_ctrl.msg_size);
		msg.msg_iov = ft_rx_ctrl.iov;
		msg.desc = ft_rx_ctrl.iov_desc;
		msg.iov_count = ft_ctrl.iov_array[ft_rx_ctrl.iov_iter];
		msg.addr = ft_rx_ctrl.addr;
		msg.tag = ft_rx_ctrl.tag;
		msg.ignore = 0;
		msg.context = NULL;
		ret = fi_trecvmsg(ft_rx_ctrl.ep, &msg, 0);
		ft_next_iov_cnt(&ft_rx_ctrl, fabric_info->rx_attr->iov_limit);
		break;
	default:
		ret = fi_trecv(ft_rx_ctrl.ep, ft_rx_ctrl.buf, ft_rx_ctrl.msg_size,
				ft_rx_ctrl.memdesc, ft_rx_ctrl.addr, ft_rx_ctrl.tag, 0, NULL);
		break;
	}
	return ret;
}

static int ft_post_send(void)
{
	struct fi_msg msg;
	int ret;

	switch (test_info.class_function) {
	case FT_FUNC_SENDV:
		ft_format_iov(ft_tx_ctrl.iov, ft_ctrl.iov_array[ft_tx_ctrl.iov_iter],
				ft_tx_ctrl.buf, ft_tx_ctrl.msg_size);
		ret = fi_sendv(ft_tx_ctrl.ep, ft_tx_ctrl.iov, ft_tx_ctrl.iov_desc,
				ft_ctrl.iov_array[ft_tx_ctrl.iov_iter], ft_tx_ctrl.addr, NULL);
		ft_next_iov_cnt(&ft_tx_ctrl, fabric_info->tx_attr->iov_limit);
		break;
	case FT_FUNC_SENDMSG:
		ft_format_iov(ft_tx_ctrl.iov, ft_ctrl.iov_array[ft_tx_ctrl.iov_iter],
				ft_tx_ctrl.buf, ft_tx_ctrl.msg_size);
		msg.msg_iov = ft_tx_ctrl.iov;
		msg.desc = ft_tx_ctrl.iov_desc;
		msg.iov_count = ft_ctrl.iov_array[ft_tx_ctrl.iov_iter];
		msg.addr = ft_tx_ctrl.addr;
		msg.context = NULL;
		msg.data = 0;
		ret = fi_sendmsg(ft_tx_ctrl.ep, &msg, 0);
		ft_next_iov_cnt(&ft_tx_ctrl, fabric_info->tx_attr->iov_limit);
		break;
	default:
		ret = fi_send(ft_tx_ctrl.ep, ft_tx_ctrl.buf, ft_tx_ctrl.msg_size,
				ft_tx_ctrl.memdesc, ft_tx_ctrl.addr, NULL);
		break;
	}

	return ret;
}

static int ft_post_tsend(void)
{
	struct fi_msg_tagged msg;
	int ret;

	switch (test_info.class_function) {
	case FT_FUNC_SENDV:
		ft_format_iov(ft_tx_ctrl.iov, ft_ctrl.iov_array[ft_tx_ctrl.iov_iter],
				ft_tx_ctrl.buf, ft_tx_ctrl.msg_size);
		ret = fi_tsendv(ft_tx_ctrl.ep, ft_tx_ctrl.iov, ft_tx_ctrl.iov_desc,
				ft_ctrl.iov_array[ft_tx_ctrl.iov_iter], ft_tx_ctrl.addr,
				ft_tx_ctrl.tag, NULL);
		ft_next_iov_cnt(&ft_tx_ctrl, fabric_info->tx_attr->iov_limit);
		break;
	case FT_FUNC_SENDMSG:
		ft_format_iov(ft_tx_ctrl.iov, ft_ctrl.iov_array[ft_tx_ctrl.iov_iter],
				ft_tx_ctrl.buf, ft_tx_ctrl.msg_size);
		msg.msg_iov = ft_tx_ctrl.iov;
		msg.desc = ft_tx_ctrl.iov_desc;
		msg.iov_count = ft_ctrl.iov_array[ft_tx_ctrl.iov_iter];
		msg.addr = ft_tx_ctrl.addr;
		msg.tag = ft_tx_ctrl.tag;
		msg.context = NULL;
		msg.data = 0;
		ret = fi_tsendmsg(ft_tx_ctrl.ep, &msg, 0);
		ft_next_iov_cnt(&ft_tx_ctrl, fabric_info->tx_attr->iov_limit);
		break;
	default:
		ret = fi_tsend(ft_tx_ctrl.ep, ft_tx_ctrl.buf, ft_tx_ctrl.msg_size,
				ft_tx_ctrl.memdesc, ft_tx_ctrl.addr, ft_tx_ctrl.tag, NULL);
		break;
	}
	return ret;
}

int ft_post_recv_bufs(void)
{
	int ret;

	for (; ft_rx_ctrl.credits; ft_rx_ctrl.credits--) {
		if (test_info.caps & FI_MSG) {
			ret = ft_post_recv();
		} else {
			ret = ft_post_trecv();
			if (!ret)
				ft_rx_ctrl.tag++;
		}
		if (ret) {
			if (ret == -FI_EAGAIN)
				break;
			FT_PRINTERR("recv", ret);
			return ret;
		}
	}
	return 0;
}

int ft_recv_msg(void)
{
	int credits, ret;

	if (ft_rx_ctrl.credits > (ft_rx_ctrl.max_credits >> 1)) {
		ret = ft_post_recv_bufs();
		if (ret)
			return ret;
	}

	credits = ft_rx_ctrl.credits;
	do {
		ret = ft_comp_rx(FT_COMP_TO);
		if (ret)
			return ret;
	} while (credits == ft_rx_ctrl.credits);

	return 0;
}

int ft_send_msg(void)
{
	int ret;

	while (!ft_tx_ctrl.credits) {
		ret = ft_comp_tx(FT_COMP_TO);
		if (ret)
			return ret;
	}

	ft_tx_ctrl.credits--;
	if (test_info.caps & FI_MSG) {
		ret = ft_post_send();
	} else {
		ret = ft_post_tsend();
		if (!ret)
			ft_tx_ctrl.tag++;
	}
	if (ret) {
		FT_PRINTERR("send", ret);
		return ret;
	}

	if (!ft_tx_ctrl.credits) {
		ret = ft_comp_tx(0);
		if (ret)
			return ret;
	}

	return 0;
}

int ft_send_dgram(void)
{
	int ret;

	*(uint8_t*) ft_tx_ctrl.buf = ft_tx_ctrl.seqno++;
	ret = ft_send_msg();
	return ret;
}

int ft_send_dgram_flood(void)
{
	int i, ret = 0;

	ft_tx_ctrl.seqno = 0;
	*(uint8_t*) ft_tx_ctrl.buf = 0;
	for (i = 0; i < ft_ctrl.xfer_iter - 1; i++) {
		ret = ft_send_msg();
		if (ret)
			break;
	}

	return ret;
}

int ft_recv_dgram(void)
{
	int credits, ret;

	do {
		if (ft_rx_ctrl.credits > (ft_rx_ctrl.max_credits >> 1)) {
			ret = ft_post_recv_bufs();
			if (ret)
				return ret;
		}

		credits = ft_rx_ctrl.credits;

		ret = ft_comp_rx(FT_DGRAM_POLL_TO);
		if ((credits != ft_rx_ctrl.credits) &&
		    (*(uint8_t *) ft_rx_ctrl.buf == ft_rx_ctrl.seqno)) {
			ft_rx_ctrl.seqno++;
			return 0;
		}
	} while (!ret);

	return (ret == -FI_EAGAIN) ? -FI_ETIMEDOUT : ret;
}

int ft_recv_dgram_flood(size_t *recv_cnt)
{
	int ret;
	size_t cnt = 0;

	do {
		ret = ft_post_recv_bufs();
		if (ret)
			break;

		ret = ft_comp_rx(0);
		cnt += ft_rx_ctrl.credits;

	} while (!ret && (*(uint8_t *) ft_rx_ctrl.buf != (uint8_t) ~0));

	*recv_cnt = cnt;
	return ret;
}

int ft_sendrecv_dgram(void)
{
	int ret, try;

	for (try = 0; try < 1000; try++) {
		ret = ft_send_dgram();
		if (ret)
			return ret;

		ret = ft_recv_dgram();
		if (ret != -FI_ETIMEDOUT)
			break;

		/* resend */
		if (test_info.caps & FI_TAGGED)
			ft_tx_ctrl.tag--;
		ft_tx_ctrl.seqno--;
	}

	return ret;
}
