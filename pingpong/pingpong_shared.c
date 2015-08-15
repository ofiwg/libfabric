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
#include <stdlib.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include "shared.h"
#include "pingpong_shared.h"

fi_addr_t remote_fi_addr;
int max_credits = 128;
int credits = 128;
int verify_data;
void *send_buf;
void *recv_buf;
int timeout;

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

int wait_for_completion_timeout(struct fid_cq *cq, int num_completions)
{
	int ret;
	struct timespec a, b;
	struct fi_cq_entry comp;

	clock_gettime(CLOCK_MONOTONIC, &a);
	while (num_completions > 0) {
		ret = fi_cq_read(cq, &comp, 1);
		if (ret > 0) {
			clock_gettime(CLOCK_MONOTONIC, &a);
			num_completions--;
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			if (ret == -FI_EAVAIL) {
				cq_readerr(cq, "cq");
			} else {
				FT_PRINTERR("fi_cq_read", ret);
			}
			return ret;
		} else if (timeout > 0) {
			clock_gettime(CLOCK_MONOTONIC, &b);
			if ((b.tv_sec - a.tv_sec) > timeout) {
				fprintf(stderr,
					"%ds timeout expired waiting to receive message, exiting\n",
					timeout);
				exit(FI_ENODATA);
			}
		}
	}

	return 0;
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
		ft_fill_buf((char *) send_buf + fi->ep_attr->msg_prefix_size, size);

	ret = fi_send(ep, send_buf, (size_t) size + fi->ep_attr->msg_prefix_size,
			fi_mr_desc(mr), remote_fi_addr, NULL);
	if (ret)
		FT_PRINTERR("fi_send", ret);

	return ret;
}

int send_msg(int size)
{
	int ret;

	/* TODO: Prefix mode may differ for send/recv */
	ret = fi_send(ep, send_buf, (size_t) size + fi->ep_attr->msg_prefix_size,
			fi_mr_desc(mr), remote_fi_addr, NULL);
	if (ret) {
		FT_PRINTERR("fi_send", ret);
		return ret;
	}

	ret = wait_for_completion(txcq, 1);

	return ret;
}

int recv_xfer(int size, bool enable_timeout)
{
	int ret;

	if (enable_timeout)
		ret = wait_for_completion_timeout(rxcq, 1);
	else
		ret = wait_for_completion(rxcq, 1);

	if (ret)
		return ret;

	/* TODO: Prefix mode may differ for send/recv */
	if (verify_data) {
		ret = ft_check_buf((char *) recv_buf + fi->ep_attr->msg_prefix_size,
				   size);
		if (ret)
			return ret;
	}

	ret = fi_recv(ep, recv_buf, buffer_size, fi_mr_desc(mr), remote_fi_addr,
			NULL);
	if (ret)
		FT_PRINTERR("fi_recv", ret);

	return ret;
}

int recv_msg(int size, bool enable_timeout)
{
	int ret;

	ret = fi_recv(ep, recv_buf, buffer_size, fi_mr_desc(mr), 0, NULL);
	if (ret) {
		FT_PRINTERR("fi_recv", ret);
		return ret;
	}

	if (enable_timeout)
		ret = wait_for_completion_timeout(rxcq, 1);
	else
		ret = wait_for_completion(rxcq, 1);

	return ret;
}

int sync_test(bool enable_timeout)
{
	int ret;

	ret = wait_for_completion(txcq, max_credits - credits);
	if (ret)
		return ret;
	credits = max_credits;

	ret = opts.dst_addr ? send_xfer(16) : recv_xfer(16, enable_timeout);
	if (ret)
		return ret;

	return opts.dst_addr ? recv_xfer(16, enable_timeout) : send_xfer(16);
}
