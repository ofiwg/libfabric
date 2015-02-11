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


struct fid_cq *txcq, *rxcq;
//struct fid_cntr *txcntr, *rxcntr;

static size_t comp_entry_cnt[] = {
	[FI_CQ_FORMAT_UNSPEC] = 0,
	[FI_CQ_FORMAT_CONTEXT] = FT_COMP_BUF_SIZE / sizeof(struct fi_cq_entry),
	[FI_CQ_FORMAT_MSG] = FT_COMP_BUF_SIZE / sizeof(struct fi_cq_msg_entry),
	[FI_CQ_FORMAT_DATA] = FT_COMP_BUF_SIZE / sizeof(struct fi_cq_data_entry),
	[FI_CQ_FORMAT_TAGGED] = FT_COMP_BUF_SIZE / sizeof(struct fi_cq_tagged_entry)
};

static size_t comp_entry_size[] = {
	[FI_CQ_FORMAT_UNSPEC] = 0,
	[FI_CQ_FORMAT_CONTEXT] = sizeof(struct fi_cq_entry),
	[FI_CQ_FORMAT_MSG] = sizeof(struct fi_cq_msg_entry),
	[FI_CQ_FORMAT_DATA] = sizeof(struct fi_cq_data_entry),
	[FI_CQ_FORMAT_TAGGED] = sizeof(struct fi_cq_tagged_entry)
};


static int ft_open_cqs(void)
{
	struct fi_cq_attr attr;
	int ret;

	if (!txcq) {
		memset(&attr, 0, sizeof attr);
		attr.format = ft_tx.cq_format;
		attr.wait_obj = ft_tx.comp_wait;
		attr.size = ft_tx.max_credits;

		ret = fi_cq_open(domain, &attr, &txcq, NULL);
		if (ret) {
			FT_PRINTERR("fi_cq_open", ret);
			return ret;
		}
	}

	if (!rxcq) {
		memset(&attr, 0, sizeof attr);
		attr.format = ft_rx.cq_format;
		attr.wait_obj = ft_rx.comp_wait;
		attr.size = ft_rx.max_credits;

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

	return 0;
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

static void ft_check_rx_comp(void *buf)
{
	struct fi_cq_err_entry *entry;

	entry = buf;
	switch (ft_rx.cq_format) {
	case FI_CQ_FORMAT_TAGGED:
	case FI_CQ_FORMAT_DATA:
	case FI_CQ_FORMAT_MSG:
		if (entry->len != ft_tx.msg_size)
			ft_record_error(FI_EMSGSIZE);
		/* fall through */
	default:
		if (entry->op_context != NULL)
			ft_record_error(FI_EOTHER);
		break;
	}
}

int ft_comp_rx(void)
{
	uint8_t buf[FT_COMP_BUF_SIZE];
	int i, ret;

	do {
		ret = fi_cq_read(rxcq, buf, comp_entry_cnt[ft_rx.cq_format]);
		if (ret < 0) {
			if (ret == -FI_EAVAIL) {
				cq_readerr(rxcq, "rxcq");
			} else {
				FT_PRINTERR("fi_cq_read", ret);
			}
			return ret;
		} else if (ret) {
			for (i = 0; i < ret; i++) {
				ft_check_rx_comp(&buf[comp_entry_size[ft_rx.cq_format] * i]);
			}
			ft_rx.credits += ret;
		}
	} while (ret == comp_entry_cnt[ft_rx.cq_format]);

	return 0;
}


int ft_comp_tx(void)
{
	uint8_t buf[FT_COMP_BUF_SIZE];
	int ret;

	do {
		ret = fi_cq_read(txcq, buf, comp_entry_cnt[ft_tx.cq_format]);
		if (ret < 0) {
			if (ret == -FI_EAVAIL) {
				cq_readerr(txcq, "txcq");
			} else {
				FT_PRINTERR("fi_cq_read", ret);
			}
			return ret;
		} else if (ret) {
			ft_tx.credits += ret;
		}
	} while (ret == comp_entry_cnt[ft_tx.cq_format]);

	return 0;
}
