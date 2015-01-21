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

static struct timespec start, end;


#define FT_CLOSE_FID(fd) \
	do { \
		if (fd) { \
			fi_close(&fd->fid); \
			fd = NULL; \
		} \
	} while (0)


void ft_record_error(int error)
{
	if (!ft.error) {
		fprintf(stderr, "ERROR [%s], continuing with test",
			fi_strerror(error));
		ft.error = error;
	}
}

static int ft_init_xcontrol(struct ft_xcontrol *ctrl)
{
	memset(ctrl, 0, sizeof *ctrl);
	ctrl->credits = FT_DEFAULT_CREDITS;
	ctrl->max_credits =  FT_DEFAULT_CREDITS;
	ft_rx.comp_wait = FI_WAIT_NONE;

	ctrl->iov = calloc(ft.iov_array[ft.iov_cnt - 1], sizeof *ctrl->iov);
	ctrl->iov_desc = calloc(ft.iov_array[ft.iov_cnt - 1],
				sizeof *ctrl->iov_desc);
	if (!ctrl->iov || !ctrl->iov_desc)
		return -FI_ENOMEM;

	return 0;
}

static int ft_init_rx_control(void)
{
	int ret;

	ret= ft_init_xcontrol(&ft_rx);
	if (ret)
		return ret;

	ft_rx.cq_format = FI_CQ_FORMAT_MSG;
	ft_rx.addr = FI_ADDR_UNSPEC;

	ft_rx.msg_size = med_size_array[med_size_cnt - 1];
	if (fabric_info && fabric_info->ep_attr &&
	    fabric_info->ep_attr->max_msg_size &&
	    fabric_info->ep_attr->max_msg_size < ft_rx.msg_size)
		ft_rx.msg_size = fabric_info->ep_attr->max_msg_size;

	return 0;
}

static int ft_init_tx_control(void)
{
	int ret;

	ret = ft_init_xcontrol(&ft_tx);
	if (ret)
		return ret;

	ft_tx.cq_format = FI_CQ_FORMAT_CONTEXT;
	return 0;
}

static int ft_init_control(void)
{
	int ret;

	memset(&ft, 0, sizeof ft);
	ft.xfer_iter = FT_DEFAULT_CREDITS;
	ft.inc_step = test_info.test_flags & FT_FLAG_QUICKTEST ? 4 : 1;

	ft.iov_array = sm_size_array;
	ft.iov_cnt = sm_size_cnt;

	if (test_info.caps & FI_INJECT) {
		ft.size_array = sm_size_array;
		ft.size_cnt = sm_size_cnt;
	} else if (test_info.caps & FI_RMA) {
		ft.size_array = lg_size_array;
		ft.size_cnt = lg_size_cnt;
	} else {
		ft.size_array = med_size_array;
		ft.size_cnt = med_size_cnt;
	}

	ret = ft_init_rx_control();
	if (!ret)
		ret = ft_init_tx_control();
	return ret;
}

static void ft_cleanup_xcontrol(struct ft_xcontrol *ctrl)
{
	FT_CLOSE_FID(ctrl->mr);
	free(ctrl->buf);
	free(ctrl->iov);
	free(ctrl->iov_desc);
	memset(ctrl, 0, sizeof *ctrl);
}

void ft_format_iov(struct iovec *iov, size_t cnt, char *buf, size_t len)
{
	size_t offset;
	int i;

	for (i = 0, offset = 0; i < cnt - 1; i++) {
		iov[i].iov_base = buf + offset;
		iov[i].iov_len = len / cnt;
		offset += iov[i].iov_len;
	}
	iov[i].iov_base = buf + offset;
	iov[i].iov_len = len - offset;
}

void ft_next_iov_cnt(struct ft_xcontrol *ctrl, size_t max_iov_cnt)
{
	ctrl->iov_iter++;
	if (ctrl->iov_iter > ft.iov_cnt ||
	    ft.iov_array[ctrl->iov_iter] > max_iov_cnt)
		ctrl->iov_iter = 0;
}

static int ft_sync_test(int value)
{
	int ret, result = -FI_EOTHER;

	ret = ft_reset_ep();
	if (ret)
		return ret;

	if (listen_sock < 0) {
		ft_fw_send(sock, &value,  sizeof value);
		ft_fw_recv(sock, &result, sizeof result);
	} else {
		ft_fw_recv(sock, &result, sizeof result);
		ft_fw_send(sock, &value,  sizeof value);
	}

	return result;
}

static int ft_pingpong(void)
{
	int ret, i;

	// TODO: current flow will not handle manual progress mode
	// it can get stuck with both sides receiving
	if (listen_sock < 0) {
		for (i = 0; i < ft.xfer_iter; i++) {
			ret = ft_send_msg();
			if (ret)
				return ret;

			ret = ft_recv_msg();
			if (ret)
				return ret;
		}
	} else {
		for (i = 0; i < ft.xfer_iter; i++) {
			ret = ft_recv_msg();
			if (ret)
				return ret;

			ret = ft_send_msg();
			if (ret)
				return ret;
		}
	}

	return 0;
}

static int ft_pingpong_dgram(void)
{
	int ret, i;

	if (listen_sock < 0) {
		for (i = 0; i < ft.xfer_iter; i++) {
			ret = ft_sendrecv_dgram();
			if (ret)
				return ret;
		}
	} else {
		ret = ft_recv_dgram();
		if (ret)
			return ret;

		for (i = 0; i < ft.xfer_iter - 1; i++) {
			ret = ft_sendrecv_dgram();
			if (ret)
				return ret;
		}

		ret = ft_send_msg();
		if (ret)
			return ret;
	}

	return 0;
}

static int ft_run_latency(void)
{
	int ret, i;

	for (i = 0; i < ft.size_cnt; i += ft.inc_step) {
		ft_tx.msg_size = ft.size_array[i];
		if (ft_tx.msg_size > fabric_info->ep_attr->max_msg_size)
			break;

		ft.xfer_iter = size_to_count(ft_tx.msg_size);
		if (test_info.test_flags & FT_FLAG_QUICKTEST)
			ft.xfer_iter >>= 2;

		ret = ft_sync_test(0);
		if (ret)
			return ret;

		clock_gettime(CLOCK_MONOTONIC, &start);
		ret = (test_info.ep_type == FI_EP_DGRAM) ?
			ft_pingpong_dgram() : ft_pingpong();
		clock_gettime(CLOCK_MONOTONIC, &end);
		if (ret)
			return ret;

		show_perf("lat", ft_tx.msg_size, ft.xfer_iter, &start, &end, 2);
	}

	return 0;
}

static void ft_cleanup(void)
{
	FT_CLOSE_FID(ep);
	FT_CLOSE_FID(pep);
	FT_CLOSE_FID(rxcq);
	FT_CLOSE_FID(txcq);
	FT_CLOSE_FID(av);
	FT_CLOSE_FID(eq);
	FT_CLOSE_FID(domain);
	FT_CLOSE_FID(fabric);
	ft_cleanup_xcontrol(&ft_rx);
	ft_cleanup_xcontrol(&ft_tx);
	memset(&ft, 0, sizeof ft);
}

int ft_run_test()
{
	int ret;

	ret = ft_init_control();
	if (ret)
		return ret;

	ret = ft_open_control();
	if (ret)
		return ret;

	if (test_info.ep_type == FI_EP_MSG && listen_sock >= 0)
		ret = ft_open_passive();
	else
		ret = ft_open_active();
	if (ret)
		return ret;

	ret = ft_enable_comm();
	if (ret)
		return ret;

	switch (test_info.test_type) {
	case FT_TEST_LATENCY:
		ret = ft_run_latency();
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}

	ft_cleanup();

	return ret ? ret : -ft.error;
}
