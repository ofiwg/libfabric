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

#include "fabtest.h"


static size_t comp_entry_cnt[] = {
	[FI_CQ_FORMAT_UNSPEC] = 0,
	[FI_CQ_FORMAT_CONTEXT] = FT_COMP_BUF_SIZE / sizeof(struct fi_cq_entry),
	[FI_CQ_FORMAT_MSG] = FT_COMP_BUF_SIZE / sizeof(struct fi_cq_msg_entry),
	[FI_CQ_FORMAT_DATA] = FT_COMP_BUF_SIZE / sizeof(struct fi_cq_data_entry),
	[FI_CQ_FORMAT_TAGGED] = FT_COMP_BUF_SIZE / sizeof(struct fi_cq_tagged_entry)
};

/*
static size_t comp_entry_size[] = {
	[FI_CQ_FORMAT_UNSPEC] = 0,
	[FI_CQ_FORMAT_CONTEXT] = sizeof(struct fi_cq_entry),
	[FI_CQ_FORMAT_MSG] = sizeof(struct fi_cq_msg_entry),
	[FI_CQ_FORMAT_DATA] = sizeof(struct fi_cq_data_entry),
	[FI_CQ_FORMAT_TAGGED] = sizeof(struct fi_cq_tagged_entry)
};
*/


static int ft_open_cqs(void)
{
	struct fi_cq_attr attr;
	int ret;

	if (!txcq) {
		memset(&attr, 0, sizeof attr);
		attr.format = ft_tx_ctrl.cq_format;
		attr.wait_obj = ft_tx_ctrl.comp_wait;
		attr.size = ft_tx_ctrl.max_credits;

		ret = fi_cq_open(domain, &attr, &txcq, NULL);
		if (ret) {
			FT_PRINTERR("fi_cq_open", ret);
			return ret;
		}
	}

	if (!rxcq) {
		memset(&attr, 0, sizeof attr);
		attr.format = ft_rx_ctrl.cq_format;
		attr.wait_obj = ft_rx_ctrl.comp_wait;
		attr.size = ft_rx_ctrl.max_credits;

		ret = fi_cq_open(domain, &attr, &rxcq, NULL);
		if (ret) {
			FT_PRINTERR("fi_cq_open", ret);
			return ret;
		}
	}

	return 0;
}

int ft_open_comp(void)
{
	int ret;

	ret = (test_info.comp_type == FT_COMP_QUEUE) ?
		ft_open_cqs() : -FI_ENOSYS;

	return ret;
}

int ft_bind_comp(struct fid_ep *ep, uint64_t flags)
{
	int ret;

	if (flags & FI_SEND) {
		ret = fi_ep_bind(ep, &txcq->fid, flags & ~FI_RECV);
		if (ret) {
			FT_PRINTERR("fi_ep_bind", ret);
			return ret;
		}
	}

	if (flags & FI_RECV) {
		ret = fi_ep_bind(ep, &rxcq->fid, flags & ~FI_SEND);
		if (ret) {
			FT_PRINTERR("fi_ep_bind", ret);
			return ret;
		}
	}

	return 0;
}

/* Read CQ until there are no more completions */
#define ft_cq_read(cq_read, cq, buf, count, completions, str, ret, ...)	\
	do {							\
		ret = cq_read(cq, buf, count, ##__VA_ARGS__);	\
		if (ret < 0) {					\
			if (ret == -FI_EAGAIN)			\
				break;				\
			if (ret == -FI_EAVAIL) {		\
				ret = ft_cq_readerr(cq);	\
			} else {				\
				FT_PRINTERR(#cq_read, ret);	\
			}					\
			return ret;				\
		} else {					\
			completions += ret;			\
		}						\
	} while (ret == count)

static int ft_comp_x(struct fid_cq *cq, struct ft_xcontrol *ft_x,
		const char *x_str, int timeout)
{
	uint8_t buf[FT_COMP_BUF_SIZE];
	struct timespec s, e;
	int poll_time = 0;
	int ret;

	switch(test_info.cq_wait_obj) {
	case FI_WAIT_NONE:
		do {
			if (!poll_time)
				clock_gettime(CLOCK_MONOTONIC, &s);

			ft_cq_read(fi_cq_read, cq, buf, comp_entry_cnt[ft_x->cq_format],
					ft_x->credits, x_str, ret);

			clock_gettime(CLOCK_MONOTONIC, &e);
			poll_time = get_elapsed(&s, &e, MILLI);
		} while (ret == -FI_EAGAIN && poll_time < timeout);

		break;
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
	case FI_WAIT_MUTEX_COND:
		ft_cq_read(fi_cq_sread, cq, buf, comp_entry_cnt[ft_x->cq_format],
			ft_x->credits, x_str, ret, NULL, timeout);
		break;
	case FI_WAIT_SET:
		FT_ERR("fi_ubertest: Unsupported cq wait object\n");
		return -1;
	default:
		FT_ERR("Unknown cq wait object\n");
		return -1;
	}

	return (ret == -FI_EAGAIN && timeout) ? ret : 0;
}

int ft_comp_rx(int timeout)
{
	return ft_comp_x(rxcq, &ft_rx_ctrl, "rxcq", timeout);
}


int ft_comp_tx(int timeout)
{
	return ft_comp_x(txcq, &ft_tx_ctrl, "txcq", timeout);
}
