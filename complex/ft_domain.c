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


struct fid_fabric *fabric;
struct fid_domain *domain;
struct fid_eq *eq;
struct fid_av *av;


static int ft_open_fabric(void)
{
	int ret;

	if (fabric) {
		if (fabric_info->fabric_attr->fabric &&
		    fabric_info->fabric_attr->fabric != fabric) {
			printf("Opened fabric / fabric_attr mismatch\n");
			return -FI_EOTHER;
		}
		return 0;
	}

	ret = fi_fabric(fabric_info->fabric_attr, &fabric, NULL);
	if (ret)
		FI_PRINTERR("fi_fabric", ret);

	return ret;
}

static int ft_open_eq(void)
{
	struct fi_eq_attr attr;
	int ret;

	if (eq)
		return 0;

	memset(&attr, 0, sizeof attr);
	attr.wait_obj = FI_WAIT_FD;
	ret = fi_eq_open(fabric, &attr, &eq, NULL);
	if (ret)
		FI_PRINTERR("fi_eq_open", ret);

	return ret;
}

int ft_eq_readerr(void)
{
	struct fi_eq_err_entry err;
	ssize_t ret;

	ret = fi_eq_readerr(eq, &err, 0);
	if (ret != sizeof(err)) {
		FI_PRINTERR("fi_eq_readerr", ret);
		return ret;
	} else {
		fprintf(stderr, "Error event %d %s\n",
			err.err, fi_strerror(err.err));
		return err.err;
	}
}

ssize_t ft_get_event(uint32_t *event, void *buf, size_t len,
		     uint32_t event_check, size_t len_check)
{
	ssize_t ret;

	ret = fi_eq_sread(eq, event, buf, len, FT_TIMEOUT, 0);
	if (ret == -FI_EAVAIL) {
		return ft_eq_readerr();
	} else if (ret < 0) {
		FI_PRINTERR("fi_eq_sread", ret);
		return ret;
	}

	if (event_check && event_check != *event) {
		fprintf(stderr, "Unexpected event %d, wanted %d\n",
			*event, event_check);
		return -FI_ENOMSG;

	}

	if (ret < len_check) {
		fprintf(stderr, "Reported event too small\n");
		return -FI_ETOOSMALL;
	}

	return ret;
}

static int ft_open_domain(void)
{
	int ret;

	if (domain) {
		if (fabric_info->domain_attr->domain &&
		    fabric_info->domain_attr->domain != domain) {
			fprintf(stderr, "Opened domain / domain_attr mismatch\n");
			return -FI_EDOMAIN;
		}
		return 0;
	}

	ret = fi_domain(fabric, fabric_info, &domain, NULL);
	if (ret)
		FI_PRINTERR("fi_domain", ret);

	return ret;
}

static int ft_open_av(void)
{
	struct fi_av_attr attr;
	int ret;

	if (av)
		return 0;

	memset(&attr, 0, sizeof attr);
	attr.type = test_info.av_type;
	attr.count = 2;
	ret = fi_av_open(domain, &attr, &av, NULL);
	if (ret) {
		FI_PRINTERR("fi_av_open", ret);
		return ret;
	}

	return ret;
}

static int ft_setup_xcontrol_bufs(struct ft_xcontrol *ctrl)
{
	size_t size;
	int i, ret;

	size = ft.size_array[ft.size_cnt - 1];
	if (!ctrl->buf) {
		ctrl->buf = calloc(1, size);
		if (!ctrl->buf)
			return -FI_ENOMEM;
	}

	if ((fabric_info->mode & FI_LOCAL_MR) && !ctrl->mr) {
		ret = fi_mr_reg(domain, ctrl->buf, size,
				0, 0, 0, 0, &ctrl->mr, NULL);
		if (ret) {
			FI_PRINTERR("fi_mr_reg", ret);
			return ret;
		}
		ctrl->memdesc = fi_mr_desc(ctrl->mr);
	}

	for (i = 0; i < ft.iov_cnt; i++)
		ctrl->iov_desc[i] = ctrl->memdesc;

	return 0;
}

static int ft_setup_bufs(void)
{
	int ret;

	ret = ft_setup_xcontrol_bufs(&ft_rx);
	if (ret)
		return ret;

	ret = ft_setup_xcontrol_bufs(&ft_tx);
	if (ret)
		return ret;

	return 0;
}

int ft_open_control(void)
{
	int ret;

	ret = ft_open_fabric();
	if (ret)
		return ret;

	ret = ft_open_eq();
	if (ret)
		return ret;

	ret = ft_open_domain();
	if (ret)
		return ret;

	ret = ft_setup_bufs();
	if (ret)
		return ret;

	if (test_info.ep_type != FI_EP_MSG) {
		ret = ft_open_av();
		if (ret)
			return ret;
	}

	return 0;
}
