/*
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2006, 2007 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2017-2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

#include <alloca.h>
#include <errno.h>

#include <rdma/fi_errno.h>

#include "efa_ib.h"
#include "efa_io_defs.h"
#include "efa_cmd.h"

static struct efa_context **ctx_list;
static int dev_cnt;

#define EFA_UVERBS_DEV_PATH "/dev/infiniband/"
#define EFA_EVERBS_DEV_NAME "efa_everbs"

static int efa_everbs_init_cmd_file(struct efa_context *context, int devnum)
{
	int exp_mask = (EFA_USER_CMDS_SUPP_UDATA_CREATE_AH |
			EFA_USER_CMDS_SUPP_UDATA_QUERY_DEVICE);
	char *efa_everbs_dev_path;
	int efa_everbs_cmd_fd;

	/* everbs cmd file is not created/needed on newer kernels */
	if ((context->cmds_supp_udata & exp_mask) == exp_mask)
		return 0;

	if (asprintf(&efa_everbs_dev_path, EFA_UVERBS_DEV_PATH EFA_EVERBS_DEV_NAME "%d", devnum) < 0)
		return -errno;

	efa_everbs_cmd_fd = open(efa_everbs_dev_path, O_RDWR | O_CLOEXEC);
	free(efa_everbs_dev_path);
	if (efa_everbs_cmd_fd < 0) {
		EFA_WARN(FI_LOG_FABRIC, "fail to open efa_everbs cmd file [%d]\n",
			 efa_everbs_cmd_fd);
		return -errno;
	}
	context->efa_everbs_cmd_fd = efa_everbs_cmd_fd;

	return 0;
}

static inline int efa_device_parse_everbs_idx(struct ibv_device *device)
{
	int devnum;

	if (sscanf(device->dev_name, "uverbs%d", &devnum) != 1)
		return -EINVAL;

	return devnum;
}

static struct efa_context *efa_device_open(struct ibv_device *device)
{
	struct efa_context *ctx;
	char *devpath;
	int cmd_fd;
	int devnum;
	int ret;

	if (asprintf(&devpath, EFA_UVERBS_DEV_PATH "%s", device->dev_name) < 0)
		return NULL;

	/*
	 * We'll only be doing writes, but we need O_RDWR in case the
	 * provider needs to mmap() the file.
	 */
	cmd_fd = open(devpath, O_RDWR | O_CLOEXEC);
	free(devpath);

	if (cmd_fd < 0)
		return NULL;

	ctx = calloc(1, sizeof(struct efa_context));
	if (!ctx) {
		errno = ENOMEM;
		goto err_close_fd;
	}

	ret = efa_cmd_alloc_ucontext(device, ctx, cmd_fd);
	if (ret)
		goto err_free_ctx;

	ctx->cqe_size = sizeof(struct efa_io_rx_cdesc);
	if (ctx->cqe_size <= 0)
		goto err_free_ctx;

	devnum = efa_device_parse_everbs_idx(device);
	if (efa_everbs_init_cmd_file(ctx, devnum))
		goto err_free_ctx;

	pthread_mutex_init(&ctx->ibv_ctx.mutex, NULL);

	return ctx;

err_free_ctx:
	free(ctx);
err_close_fd:
	close(cmd_fd);
	return NULL;
}

static int efa_device_close(struct efa_context *ctx)
{
	int cmd_fd;

	pthread_mutex_destroy(&ctx->ibv_ctx.mutex);
	cmd_fd = ctx->ibv_ctx.cmd_fd;
	if (ctx->efa_everbs_cmd_fd)
		close(ctx->efa_everbs_cmd_fd);
	free(ctx->ibv_ctx.device);
	free(ctx);
	close(cmd_fd);

	return 0;
}

int efa_device_init(void)
{
	struct ibv_device **device_list;
	int ctx_idx;
	int ret;

	dev_cnt = efa_ib_init(&device_list);
	if (dev_cnt <= 0)
		return -ENODEV;

	ctx_list = calloc(dev_cnt, sizeof(*ctx_list));
	if (!ctx_list) {
		ret = -ENOMEM;
		goto err_free_dev_list;
	}

	for (ctx_idx = 0; ctx_idx < dev_cnt; ctx_idx++) {
		ctx_list[ctx_idx] = efa_device_open(device_list[ctx_idx]);
		if (!ctx_list[ctx_idx]) {
			ret = -ENODEV;
			goto err_close_devs;
		}
	}

	free(device_list);

	return 0;

err_close_devs:
	for (ctx_idx--; ctx_idx >= 0; ctx_idx--)
		efa_device_close(ctx_list[ctx_idx]);
	free(ctx_list);
err_free_dev_list:
	free(device_list);
	dev_cnt = 0;
	return ret;
}

void efa_device_free(void)
{
	int i;

	for (i = 0; i < dev_cnt; i++)
		efa_device_close(ctx_list[i]);

	free(ctx_list);
	dev_cnt = 0;
}

struct efa_context **efa_device_get_context_list(int *num_ctx)
{
	struct efa_context **devs = NULL;
	int i;

	devs = calloc(dev_cnt, sizeof(*devs));
	if (!devs)
		goto out;

	for (i = 0; i < dev_cnt; i++)
		devs[i] = ctx_list[i];
out:
	*num_ctx = devs ? dev_cnt : 0;
	return devs;
}

void efa_device_free_context_list(struct efa_context **list)
{
	free(list);
}

