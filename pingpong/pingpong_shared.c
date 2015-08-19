/*
 * Copyright (c) 2015 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013-2015 Intel Corporation.  All rights reserved.
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

int verify_data;
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

int send_xfer(size_t size)
{
	int ret;

	if (!tx_credits) {
		ret = ft_wait_for_comp(txcq, 1);
		if (ret)
			return ret;
	} else {
		tx_credits--;
	}

	if (verify_data)
		ft_fill_buf((char *) tx_buf + ft_tx_prefix_size(), size);

	ret = fi_send(ep, tx_buf, size + ft_tx_prefix_size(),
			fi_mr_desc(mr), remote_fi_addr, NULL);
	if (ret)
		FT_PRINTERR("fi_send", ret);

	return ret;
}

int recv_xfer(size_t size, bool enable_timeout)
{
	int ret;

	if (enable_timeout)
		ret = wait_for_completion_timeout(rxcq, 1);
	else
		ret = ft_wait_for_comp(rxcq, 1);

	if (ret)
		return ret;

	if (verify_data) {
		ret = ft_check_buf((char *) rx_buf + ft_rx_prefix_size(), size);
		if (ret)
			return ret;
	}

	ret = fi_recv(ep, rx_buf, rx_size, fi_mr_desc(mr), remote_fi_addr, NULL);
	if (ret)
		FT_PRINTERR("fi_recv", ret);

	return ret;
}

int sync_test(bool enable_timeout)
{
	int ret;

	ret = ft_wait_for_comp(txcq, fi->tx_attr->size - tx_credits);
	if (ret)
		return ret;
	tx_credits = fi->tx_attr->size;

	ret = opts.dst_addr ? send_xfer(16) : recv_xfer(16, enable_timeout);
	if (ret)
		return ret;

	return opts.dst_addr ? recv_xfer(16, enable_timeout) : send_xfer(16);
}
