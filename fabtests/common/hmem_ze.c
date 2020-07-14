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

#ifdef HAVE_LIBZE

#include <level_zero/ze_api.h>

#define ZE_MAX_DEVICES 4

static ze_driver_handle_t driver;
static ze_device_handle_t devices[ZE_MAX_DEVICES];
static ze_command_queue_handle_t cmd_queue[ZE_MAX_DEVICES];
static int num_devices = 0;

static inline int _ze_cmd_queue_create(ze_device_handle_t device,
				       ze_command_queue_handle_t *cmd_queue)
{
	ze_command_queue_desc_t desc = {
		.version	= ZE_COMMAND_QUEUE_DESC_VERSION_CURRENT,
		.flags		= ZE_COMMAND_QUEUE_FLAG_NONE,
		.mode		= ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
		.priority	= ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
		.ordinal	= 0,
	};

	return zeCommandQueueCreate(device, &desc, cmd_queue);
}

static inline int _ze_cmd_list_create(ze_device_handle_t device,
				      ze_command_list_handle_t *cmd_list)
{
	ze_command_list_desc_t desc = {
		.version	= ZE_COMMAND_LIST_DESC_VERSION_CURRENT,
		.flags		= ZE_COMMAND_LIST_FLAG_NONE,
	};
	return zeCommandListCreate(device, &desc, cmd_list);
}

int ft_ze_init(void)
{
	ze_result_t ze_ret;
	uint32_t count;

	ze_ret = zeInit(ZE_INIT_FLAG_NONE);
	if (ze_ret)
		return -FI_EIO;

	count = 1;
	ze_ret = zeDriverGet(&count, &driver);
	if (ze_ret)
		return -FI_EIO;

	count = 0;
	ze_ret = zeDeviceGet(driver, &count, NULL);
	if (ze_ret || count > ZE_MAX_DEVICES)
		return -FI_EIO;

	ze_ret = zeDeviceGet(driver, &count, devices);
	if (ze_ret)
		return -FI_EIO;

	for (num_devices = 0; num_devices < count; num_devices++) {
		ze_ret = _ze_cmd_queue_create(devices[num_devices],
					      &cmd_queue[num_devices]);
		if (ze_ret) {
			(void) ft_ze_cleanup();
			return -FI_EIO;
		}
	}

	return FI_SUCCESS;
}

int ft_ze_cleanup(void)
{
	int i, ret = FI_SUCCESS;

	for (i = 0; i < num_devices; i++) {
		if (cmd_queue[i] && zeCommandQueueDestroy(cmd_queue[i]))
			ret = -FI_EINVAL;
	}

	return ret;
}

int ft_ze_alloc(uint64_t device, void **buf, size_t size)
{
	ze_device_mem_alloc_desc_t device_desc;
	ze_host_mem_alloc_desc_t host_desc;
	ze_result_t ze_ret;

	device_desc.version = ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT;
	device_desc.ordinal = 0;
	device_desc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT;

	host_desc.version = ZE_HOST_MEM_ALLOC_DESC_VERSION_CURRENT;
	host_desc.flags = ZE_HOST_MEM_ALLOC_FLAG_DEFAULT;

	ze_ret = zeDriverAllocSharedMem(driver, &device_desc, &host_desc,
					size, 16, devices[device], buf);
	return !ze_ret ? ze_ret : -FI_EINVAL;
}

int ft_ze_free(void *buf)
{
	return zeDriverFreeMem(driver, buf) ? -FI_EINVAL : FI_SUCCESS;
}

int ft_ze_memset(uint64_t device, void *buf, int value, size_t size)
{
	ze_command_list_handle_t cmd_list;
	ze_result_t ze_ret;

	ze_ret = _ze_cmd_list_create(devices[device], &cmd_list);
	if (ze_ret)
		return -FI_EIO;

	ze_ret = zeCommandListAppendMemoryFill(cmd_list, buf, &value,
					       sizeof(value), size, NULL);
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

	ze_ret = _ze_cmd_list_create(devices[device], &cmd_list);
	if (ze_ret)
		return -FI_EIO;

	ze_ret = zeCommandListAppendMemoryCopy(cmd_list, dst, src, size, NULL);
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
