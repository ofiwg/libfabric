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

#include "ofi_hmem.h"
#include "ofi.h"

#ifdef HAVE_LIBZE

#include <level_zero/ze_api.h>

#define ZE_MAX_DEVICES 4

static ze_driver_handle_t driver;
static ze_device_handle_t devices[ZE_MAX_DEVICES];
static ze_command_queue_handle_t cmd_queue[ZE_MAX_DEVICES];
static int num_devices = 0;

static const ze_command_queue_desc_t cq_desc = {
	.version	= ZE_COMMAND_QUEUE_DESC_VERSION_CURRENT,
	.flags		= ZE_COMMAND_QUEUE_FLAG_NONE,
	.mode		= ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
	.priority	= ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
	.ordinal	= 0,
};

static const ze_command_list_desc_t cl_desc = {
	.version	= ZE_COMMAND_LIST_DESC_VERSION_CURRENT,
	.flags		= ZE_COMMAND_LIST_FLAG_NONE,
};

int ze_hmem_init(void)
{
	ze_result_t ze_ret;
	uint32_t count;

	ze_ret = zeInit(ZE_INIT_FLAG_NONE);
	if (ze_ret)
		goto err;

	count = 1;
	ze_ret = zeDriverGet(&count, &driver);
	if (ze_ret)
		goto err;

	count = 0;
	ze_ret = zeDeviceGet(driver, &count, NULL);
	if (ze_ret || count > ZE_MAX_DEVICES)
		goto err;

	ze_ret = zeDeviceGet(driver, &count, devices);
	if (ze_ret)
		goto err;

	for (num_devices = 0; num_devices < count; num_devices++) {
		ze_ret = zeCommandQueueCreate(devices[num_devices], &cq_desc,
					      &cmd_queue[num_devices]);
		if (ze_ret)
			goto free;
	}

	return FI_SUCCESS;

free:
	(void) ze_hmem_cleanup();

err:
	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to initialize ZE driver resources\n");

	return -FI_EIO;
}

int ze_hmem_cleanup(void)
{
	int i, ret = FI_SUCCESS;

	for (i = 0; i < num_devices; i++) {
		if (cmd_queue[i] && zeCommandQueueDestroy(cmd_queue[i])) {
			FI_WARN(&core_prov, FI_LOG_CORE,
				"Failed to destroy ZE cmd_queue\n");
			ret = -FI_EINVAL;
		}
	}

	return ret;
}

int ze_hmem_copy(uint64_t device, void *dst, const void *src, size_t size)
{
	ze_command_list_handle_t cmd_list;
	ze_result_t ze_ret;
	int dev_id = (int) device;

	ze_ret = zeCommandListCreate(devices[dev_id], &cl_desc, &cmd_list);
	if (ze_ret)
		goto err;

	ze_ret = zeCommandListAppendMemoryCopy(cmd_list, dst, src, size, NULL);
	if (ze_ret)
		goto free;

	ze_ret = zeCommandListClose(cmd_list);
	if (ze_ret)
		goto free;

	ze_ret = zeCommandQueueExecuteCommandLists(cmd_queue[dev_id], 1,
						   &cmd_list, NULL);

free:
	if (!zeCommandListDestroy(cmd_list) && !ze_ret)
		return FI_SUCCESS;
err:
	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform ze copy (%d)\n", ze_ret);

	return -FI_EIO;
}

bool ze_is_addr_valid(const void *addr)
{
	ze_result_t ze_ret;
	ze_memory_allocation_properties_t mem_prop;
	int i;

	for (i = 0; i < num_devices; i++) {
		ze_ret = zeDriverGetMemAllocProperties(driver, addr, &mem_prop,
						       &devices[i]);
		if (!ze_ret && mem_prop.type == ZE_MEMORY_TYPE_DEVICE)
			return true;
	}
	return false;
}

int ze_hmem_get_handle(void *dev_buf, void **handle)
{
	ze_result_t ze_ret;

	ze_ret = zeDriverGetMemIpcHandle(driver, dev_buf,
                                 (ze_ipc_mem_handle_t *) handle);
	if (ze_ret) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Unable to get handle\n");
		return -FI_EINVAL;
	}

	return FI_SUCCESS;
}

int ze_hmem_open_handle(void **handle, uint64_t device, void **ipc_ptr)
{
	ze_result_t ze_ret;

	ze_ret = zeDriverOpenMemIpcHandle(driver, devices[device],
					  *((ze_ipc_mem_handle_t *) handle),
					  ZE_IPC_MEMORY_FLAG_NONE, ipc_ptr);
	if (ze_ret) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Unable to open memory handle\n");
		return -FI_EINVAL;
	}

	return FI_SUCCESS;
}

int ze_hmem_close_handle(void *ipc_ptr)
{
	ze_result_t ze_ret;

	ze_ret = zeDriverCloseMemIpcHandle(driver, ipc_ptr);
	if (ze_ret) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Unable to close memory handle\n");
		return -FI_EINVAL;
	}

	return FI_SUCCESS;
}

#else

int ze_hmem_init(void)
{
	return -FI_ENOSYS;
}

int ze_hmem_cleanup(void)
{
	return -FI_ENOSYS;
}

int ze_hmem_copy(uint64_t device, void *dst, const void *src, size_t size)
{
	return -FI_ENOSYS;
}

bool ze_is_addr_valid(const void *addr)
{
	return false;
}

int ze_hmem_get_handle(void *dev_buf, void **handle)
{
	return -FI_ENOSYS;
}

int ze_hmem_open_handle(void **handle, uint64_t device, void **ipc_ptr)
{
	return -FI_ENOSYS;
}

int ze_hmem_close_handle(void *ipc_ptr)
{
	return -FI_ENOSYS;
}

#endif /* HAVE_LIBZE */
