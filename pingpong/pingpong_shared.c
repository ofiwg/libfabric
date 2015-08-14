/*
 * Copyright (c) 2015 Cisco Systems, Inc.  All rights reserved.
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AWV
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include "shared.h"
#include "pingpong_shared.h"

fi_addr_t remote_fi_addr;
int max_credits = 128;
size_t prefix_len;
int credits = 128;
int verify_data;
void *send_buf;
void *recv_buf;

void ft_parsepongopts(int op)
{
	switch (op) {
	case 'v':
		verify_data = 1;
		break;
	case 'P':
		hints->mode |= FI_MSG_PREFIX;
		break;
	default:
		break;
	}
}

void ft_pongusage(void)
{
	FT_PRINT_OPTS_USAGE("-v", "enables data_integrity checks");
	FT_PRINT_OPTS_USAGE("-P", "enable prefix mode");
}

int send_xfer(int size)
{
	int ret;

	if (!credits) {
		ret = wait_for_completion(txcq, 1);
		if (ret)
			return ret;
	} else {
		credits--;
	}

	if (verify_data)
		ft_fill_buf((char *) send_buf + prefix_len, size);

	ret = fi_send(ep, send_buf, (size_t) size + prefix_len, fi_mr_desc(mr),
			remote_fi_addr, NULL);
	if (ret)
		FT_PRINTERR("fi_send", ret);

	return ret;
}

int send_msg(int size)
{
	int ret;

	ret = fi_send(ep, send_buf, (size_t) size + prefix_len, fi_mr_desc(mr),
			remote_fi_addr, NULL);
	if (ret) {
		FT_PRINTERR("fi_send", ret);
		return ret;
	}

	ret = wait_for_completion(txcq, 1);

	return ret;
}

int recv_xfer(int size)
{
	int ret;

	ret = wait_for_completion(rxcq, 1);
	if (ret)
		return ret;

	if (verify_data) {
		ret = ft_check_buf((char *) recv_buf + prefix_len, size);
		if (ret)
			return ret;
	}

	ret = fi_recv(ep, recv_buf, buffer_size, fi_mr_desc(mr), remote_fi_addr,
			NULL);
	if (ret)
		FT_PRINTERR("fi_recv", ret);

	return ret;
}

int recv_msg(void)
{
	int ret;

	ret = fi_recv(ep, recv_buf, buffer_size, fi_mr_desc(mr), 0, NULL);
	if (ret) {
		FT_PRINTERR("fi_recv", ret);
		return ret;
	}

	ret = wait_for_completion(rxcq, 1);

	return ret;
}

int sync_test(void)
{
	int ret;

	ret = wait_for_completion(txcq, max_credits - credits);
	if (ret)
		return ret;
	credits = max_credits;

	ret = opts.dst_addr ? send_xfer(16) : recv_xfer(16);
	if (ret)
		return ret;

	return opts.dst_addr ? recv_xfer(16) : send_xfer(16);
}
