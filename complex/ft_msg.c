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
		ft_format_iov(ft_rx.iov, ft.iov_array[ft_rx.iov_iter],
				ft_rx.buf, ft_rx.msg_size);
		ret = fi_recvv(ft_rx.ep, ft_rx.iov, ft_rx.iov_desc,
				ft.iov_array[ft_rx.iov_iter], ft_rx.addr, NULL);
		ft_next_iov_cnt(&ft_rx, fabric_info->rx_attr->iov_limit);
		break;
	case FT_FUNC_SENDMSG:
		ft_format_iov(ft_rx.iov, ft.iov_array[ft_rx.iov_iter],
				ft_rx.buf, ft_rx.msg_size);
		msg.msg_iov = ft_rx.iov;
		msg.desc = ft_rx.iov_desc;
		msg.iov_count = ft.iov_array[ft_rx.iov_iter];
		msg.addr = ft_rx.addr;
		msg.context = NULL;
		msg.data = 0;
		ret = fi_recvmsg(ft_rx.ep, &msg, 0);
		ft_next_iov_cnt(&ft_rx, fabric_info->rx_attr->iov_limit);
		break;
	default:
		ret = fi_recv(ft_rx.ep, ft_rx.buf, ft_rx.msg_size,
				ft_rx.memdesc, ft_rx.addr, NULL);
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
		ft_format_iov(ft_rx.iov, ft.iov_array[ft_rx.iov_iter],
				ft_rx.buf, ft_rx.msg_size);
		ret = fi_trecvv(ft_rx.ep, ft_rx.iov, ft_rx.iov_desc,
				ft.iov_array[ft_rx.iov_iter], ft_rx.addr,
				ft_rx.tag++, 0, NULL);
		ft_next_iov_cnt(&ft_rx, fabric_info->rx_attr->iov_limit);
		break;
	case FT_FUNC_SENDMSG:
		ft_format_iov(ft_rx.iov, ft.iov_array[ft_rx.iov_iter],
				ft_rx.buf, ft_rx.msg_size);
		msg.msg_iov = ft_rx.iov;
		msg.desc = ft_rx.iov_desc;
		msg.iov_count = ft.iov_array[ft_rx.iov_iter];
		msg.addr = ft_rx.addr;
		msg.tag = ft_rx.tag++;
		msg.ignore = 0;
		msg.context = NULL;
		ret = fi_trecvmsg(ft_rx.ep, &msg, 0);
		ft_next_iov_cnt(&ft_rx, fabric_info->rx_attr->iov_limit);
		break;
	default:
		ret = fi_trecv(ft_rx.ep, ft_rx.buf, ft_rx.msg_size,
				ft_rx.memdesc, ft_rx.addr, ft_rx.tag++, 0, NULL);
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
		ft_format_iov(ft_tx.iov, ft.iov_array[ft_tx.iov_iter],
				ft_tx.buf, ft_tx.msg_size);
		ret = fi_sendv(ft_tx.ep, ft_tx.iov, ft_tx.iov_desc,
				ft.iov_array[ft_tx.iov_iter], ft_tx.addr, NULL);
		ft_next_iov_cnt(&ft_tx, fabric_info->tx_attr->iov_limit);
		break;
	case FT_FUNC_SENDMSG:
		ft_format_iov(ft_tx.iov, ft.iov_array[ft_tx.iov_iter],
				ft_tx.buf, ft_tx.msg_size);
		msg.msg_iov = ft_tx.iov;
		msg.desc = ft_tx.iov_desc;
		msg.iov_count = ft.iov_array[ft_tx.iov_iter];
		msg.addr = ft_tx.addr;
		msg.context = NULL;
		msg.data = 0;
		ret = fi_sendmsg(ft_tx.ep, &msg, 0);
		ft_next_iov_cnt(&ft_tx, fabric_info->tx_attr->iov_limit);
		break;
	default:
		ret = fi_send(ft_tx.ep, ft_tx.buf, ft_tx.msg_size,
				ft_tx.memdesc, ft_tx.addr, NULL);
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
		ft_format_iov(ft_tx.iov, ft.iov_array[ft_tx.iov_iter],
				ft_tx.buf, ft_tx.msg_size);
		ret = fi_tsendv(ft_tx.ep, ft_tx.iov, ft_tx.iov_desc,
				ft.iov_array[ft_tx.iov_iter], ft_tx.addr,
				ft_tx.tag++, NULL);
		ft_next_iov_cnt(&ft_tx, fabric_info->tx_attr->iov_limit);
		break;
	case FT_FUNC_SENDMSG:
		ft_format_iov(ft_tx.iov, ft.iov_array[ft_tx.iov_iter],
				ft_tx.buf, ft_tx.msg_size);
		msg.msg_iov = ft_tx.iov;
		msg.desc = ft_tx.iov_desc;
		msg.iov_count = ft.iov_array[ft_tx.iov_iter];
		msg.addr = ft_tx.addr;
		msg.tag = ft_tx.tag++;
		msg.context = NULL;
		msg.data = 0;
		ret = fi_tsendmsg(ft_tx.ep, &msg, 0);
		ft_next_iov_cnt(&ft_tx, fabric_info->tx_attr->iov_limit);
		break;
	default:
		ret = fi_tsend(ft_tx.ep, ft_tx.buf, ft_tx.msg_size,
				ft_tx.memdesc, ft_tx.addr, ft_tx.tag++, NULL);
		break;
	}
	return ret;
}

int ft_post_recv_bufs(void)
{
	int ret;

	for (; ft_rx.credits; ft_rx.credits--) {
		ret = (test_info.caps & FI_TAGGED) ?
			ft_post_trecv() : ft_post_recv();
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

	if (ft_rx.credits > (ft_rx.max_credits >> 1)) {
		ret = ft_post_recv_bufs();
		if (ret)
			return ret;
	}

	credits = ft_rx.credits;
	do {
		ret = ft_comp_rx();
		if (ret)
			return ret;
	} while (credits == ft_rx.credits);

	return 0;
}

int ft_send_msg(void)
{
	int ret;

	while (!ft_tx.credits) {
		ret = ft_comp_tx();
		if (ret)
			return ret;
	}

	ft_tx.credits--;
	ret = (test_info.caps & FI_MSG) ?
		ft_post_send() : ft_post_tsend();
	if (ret) {
		FT_PRINTERR("send", ret);
		return ret;
	}

	if (!ft_tx.credits) {
		ret = ft_comp_tx();
		if (ret)
			return ret;
	}

	return 0;
}

int ft_recv_dgram(void)
{
	struct timespec s, e;
	int credits, ret;
	int64_t poll_time = 0;

	if (ft_rx.credits > (ft_rx.max_credits >> 1)) {
		ret = ft_post_recv_bufs();
		if (ret)
			return ret;
	}

	credits = ft_rx.credits;
	do {
		ret = ft_comp_rx();
		if ((credits != ft_rx.credits) || ret)
			return ret;

		if (!poll_time)
			clock_gettime(CLOCK_MONOTONIC, &s);

		clock_gettime(CLOCK_MONOTONIC,&e);
		poll_time = get_elapsed(&s, &e, MILLI);

	} while (poll_time < 1);

	return -FI_ETIMEDOUT;
}

int ft_sendrecv_dgram(void)
{
	int ret, try;

	for (try = 0; try < 1000; try++) {
		ret = ft_send_msg();
		if (ret)
			return ret;

		ret = ft_recv_dgram();
		if (ret != -FI_ETIMEDOUT)
			break;

		/* resend last tag */
		if (test_info.caps & FI_TAGGED)
			ft_tx.tag--;
	}

	return ret;
}
