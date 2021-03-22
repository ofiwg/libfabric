/*
 * Copyright (c) 2020 Intel Corporation. All rights reserved.
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "hmem.h"

#if HAVE_LIBZE

#include <level_zero/ze_api.h>

#define ZE_MAX_DEVICES 4

static ze_context_handle_t context;
static ze_device_handle_t devices[ZE_MAX_DEVICES];
static ze_command_queue_handle_t cmd_queue[ZE_MAX_DEVICES];
static int num_devices = 0;

static const ze_command_queue_desc_t cq_desc = {
	.stype		= ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
	.pNext		= NULL,
	.ordinal	= 0,
	.index		= 0,
	.flags		= 0,
	.mode		= ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
	.priority	= ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
};

static const ze_command_list_desc_t cl_desc = {
	.stype				= ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
	.pNext				= NULL,
	.commandQueueGroupOrdinal	= 0,
	.flags				= 0,
};

static const ze_device_mem_alloc_desc_t device_desc = {
	.stype		= ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
	.pNext		= NULL,
	.flags		= 0,
	.ordinal	= 0,
};

int ft_ze_init(void)
{
	ze_driver_handle_t driver;
	ze_context_desc_t context_desc = {0};
	ze_result_t ze_ret;
	uint32_t count;

	ze_ret = zeInit(ZE_INIT_FLAG_GPU_ONLY);
	if (ze_ret)
		return -FI_EIO;

	count = 1;
	ze_ret = zeDriverGet(&count, &driver);
	if (ze_ret)
		return -FI_EIO;

	ze_ret = zeContextCreate(driver, &context_desc, &context);
	if (ze_ret)
		return -FI_EIO;

	count = 0;
	ze_ret = zeDeviceGet(driver, &count, NULL);
	if (ze_ret || count > ZE_MAX_DEVICES)
		goto err;;

	ze_ret = zeDeviceGet(driver, &count, devices);
	if (ze_ret)
		goto err;

	for (num_devices = 0; num_devices < count; num_devices++) {
		ze_ret = zeCommandQueueCreate(context, devices[num_devices], &cq_desc,
					      &cmd_queue[num_devices]);
		if (ze_ret)
			goto err;
	}

	return FI_SUCCESS;

err:
	(void) ft_ze_cleanup();
	return -FI_EIO;
}

int ft_ze_cleanup(void)
{
	int i, ret = FI_SUCCESS;

	for (i = 0; i < num_devices; i++) {
		if (cmd_queue[i] && zeCommandQueueDestroy(cmd_queue[i]))
			ret = -FI_EINVAL;
	}

	if (zeContextDestroy(context))
		return -FI_EINVAL;

	return ret;
}

int ft_ze_alloc(uint64_t device, void **buf, size_t size)
{
	return zeMemAllocDevice(context, &device_desc, size, 16,
				devices[device], buf) ? -FI_EINVAL : 0;
}

int ft_ze_free(void *buf)
{
	return zeMemFree(context, buf) ? -FI_EINVAL : FI_SUCCESS;
}

int ft_ze_memset(uint64_t device, void *buf, int value, size_t size)
{
	ze_command_list_handle_t cmd_list;
	ze_result_t ze_ret;

	ze_ret = zeCommandListCreate(context, devices[device], &cl_desc, &cmd_list);
	if (ze_ret)
		return -FI_EIO;

	ze_ret = zeCommandListAppendMemoryFill(cmd_list, buf, &value,
					       sizeof(value), size, NULL, 0, NULL);
	if (ze_ret)
		goto free;

	ze_ret = zeCommandListClose(cmd_list);
	if (ze_ret)
		goto free;

	ze_ret = zeCommandQueueExecuteCommandLists(cmd_queue[device], 1,
						   &cmd_list, NULL);

free:
	if (!zeCommandListDestroy(cmd_list) && !ze_ret)
		return FI_SUCCESS;

	return -FI_EINVAL;
}

int ft_ze_copy(uint64_t device, void *dst, const void *src, size_t size)
{
	ze_command_list_handle_t cmd_list;
	ze_result_t ze_ret;

	ze_ret = zeCommandListCreate(context, devices[device], &cl_desc, &cmd_list);
	if (ze_ret)
		return -FI_EIO;

	ze_ret = zeCommandListAppendMemoryCopy(cmd_list, dst, src, size, NULL, 0, NULL);
	if (ze_ret)
		goto free;

	ze_ret = zeCommandListClose(cmd_list);
	if (ze_ret)
		goto free;

	ze_ret = zeCommandQueueExecuteCommandLists(cmd_queue[device], 1,
						   &cmd_list, NULL);

free:
	if (!zeCommandListDestroy(cmd_list) && !ze_ret)
		return FI_SUCCESS;

	return -FI_EINVAL;
}

#else

int ft_ze_init(void)
{
	return -FI_ENOSYS;
}

int ft_ze_cleanup(void)
{
	return -FI_ENOSYS;
}

int ft_ze_alloc(uint64_t device, void **buf, size_t size)
{
	return -FI_ENOSYS;
}

int ft_ze_free(void *buf)
{
	return -FI_ENOSYS;
}

int ft_ze_memset(uint64_t device, void *buf, int value, size_t size)
{
	return -FI_ENOSYS;
}

int ft_ze_copy(uint64_t device, void *dst, const void *src, size_t size)
{
	return -FI_ENOSYS;
}


#endif /* HAVE_LIBZE */
