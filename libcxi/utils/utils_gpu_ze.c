/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2023 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#include "utils_common.h"

#ifdef HAVE_ZE_SUPPORT

#include <dlfcn.h>
#include <level_zero/ze_api.h>

#define ZE_MAX_DEVICES 6
#define ZE_ERR(fmt, ...) \
	fprintf(stderr, "%s:%d: " fmt "", __func__, __LINE__, ##__VA_ARGS__)

static ze_context_handle_t context;
static ze_device_handle_t devices[ZE_MAX_DEVICES];
static ze_command_queue_handle_t cmd_queue[ZE_MAX_DEVICES];
static int num_devices;
static int current_device;

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

static void *libze_handle;
static ze_result_t (*cxi_zeMemAllocDevice)(
			ze_context_handle_t hContext,
			const ze_device_mem_alloc_desc_t *device_desc,
			size_t size, size_t alignment,
			ze_device_handle_t hDevice, void **pptr);
static ze_result_t (*cxi_zeMemAllocHost)(
			ze_context_handle_t hContext,
			const ze_host_mem_alloc_desc_t *host_desc,
			size_t size, size_t alignment, void **pptr);
static ze_result_t (*cxi_zeMemFree)(ze_context_handle_t hContext, void *ptr);
static ze_result_t (*cxi_zeCommandListCreate)(
			ze_context_handle_t hContext,
			ze_device_handle_t hDevice,
			const ze_command_list_desc_t *desc,
			ze_command_list_handle_t *phCommandList);
static ze_result_t (*cxi_zeCommandListAppendMemoryFill)(
			ze_command_list_handle_t hCommandList,
			void *ptr, const void *pattern,
			size_t pattern_size, size_t size,
			ze_event_handle_t hSignalEvent,
			uint32_t numWaitEvents,
			ze_event_handle_t *phWaitEvents);
static ze_result_t (*cxi_zeCommandListClose)(ze_command_list_handle_t hCommandList);
static ze_result_t (*cxi_zeCommandQueueExecuteCommandLists)(
			ze_command_queue_handle_t hCommandQueue,
			uint32_t numCommandLists,
			ze_command_list_handle_t *phCommandLists,
			ze_fence_handle_t hFence);
static ze_result_t (*cxi_zeCommandListDestroy)(ze_command_list_handle_t hCommandList);
static ze_result_t (*cxi_zeCommandListAppendMemoryCopy)(
			ze_command_list_handle_t hCommandList,
			void *dstptr, const void *srcptr,
			size_t size,
			ze_event_handle_t hSignalEvent,
			uint32_t numWaitEvents,
			ze_event_handle_t *phWaitEvents);
static ze_result_t (*cxi_zeInit)(ze_init_flags_t flags);
static ze_result_t (*cxi_zeDriverGet)(uint32_t *pCount, ze_driver_handle_t *phDrivers);
static ze_result_t (*cxi_zeContextCreate)(
			ze_driver_handle_t hDriver,
			const ze_context_desc_t *desc,
			ze_context_handle_t *phContext);
static ze_result_t (*cxi_zeDeviceGet)(
			ze_driver_handle_t hDriver,
			uint32_t *pCount,
			ze_device_handle_t *phDevices);
static ze_result_t (*cxi_zeCommandQueueCreate)(
			ze_context_handle_t hContext,
			ze_device_handle_t hDevice,
			const ze_command_queue_desc_t *desc,
			ze_command_queue_handle_t *phCommandQueue);
static ze_result_t (*cxi_zeCommandQueueDestroy)(ze_command_queue_handle_t hCommandQueue);
static ze_result_t (*cxi_zeContextDestroy)(ze_context_handle_t hContext);
static ze_result_t (*cxi_zeMemGetAddressRange)(
			ze_context_handle_t hContext,
			const void *ptr,
			void **pBase, size_t *pSize);
static ze_result_t (*cxi_zeMemGetIpcHandle)(
			ze_context_handle_t hContext,
			const void *ptr,
			ze_ipc_mem_handle_t *pIpcHandle);
static ze_result_t (*cxi_zeMemGetAllocProperties)(
			ze_context_handle_t hContext,
			const void *ptr,
			ze_memory_allocation_properties_t *pMemAllocProperties,
			ze_device_handle_t *phDevice);
void ze_lib_fini(void);

static void ze_dlclose(void)
{
	dlclose(libze_handle);
}

static int ze_dlopen(void)
{

	libze_handle = dlopen("libze_loader.so", RTLD_NOW);
	if (!libze_handle) {
		ZE_ERR("Failed to dlopen libze_loader.so\n");
		goto err;
	}

	cxi_zeMemAllocDevice = dlsym(libze_handle, "zeMemAllocDevice");
	if (!cxi_zeMemAllocDevice) {
		ZE_ERR("Failed to find zeMemAllocDevice\n");
		goto err_dlclose;
	}

	cxi_zeMemAllocHost = dlsym(libze_handle, "zeMemAllocHost");
	if (!cxi_zeMemAllocHost) {
		ZE_ERR("Failed to find cxi_zeMemAllocHost\n");
		goto err_dlclose;
	}

	cxi_zeMemFree = dlsym(libze_handle, "zeMemFree");
	if (!cxi_zeMemFree) {
		ZE_ERR("Failed to find zeMemFree\n");
		goto err_dlclose;
	}

	cxi_zeCommandListCreate = dlsym(libze_handle, "zeCommandListCreate");
	if (!cxi_zeCommandListCreate) {
		ZE_ERR("Failed to find zeCommandListCreate\n");
		goto err_dlclose;
	}

	cxi_zeCommandListAppendMemoryFill =
		dlsym(libze_handle, "zeCommandListAppendMemoryFill");
	if (!cxi_zeCommandListAppendMemoryFill) {
		ZE_ERR("Failed to find zeCommandListAppendMemoryFill\n");
		goto err_dlclose;
	}

	cxi_zeCommandListClose =
		dlsym(libze_handle, "zeCommandListClose");
	if (!cxi_zeCommandListClose) {
		ZE_ERR("Failed to find zeCommandListClose\n");
		goto err_dlclose;
	}

	cxi_zeCommandQueueExecuteCommandLists =
		dlsym(libze_handle, "zeCommandQueueExecuteCommandLists");
	if (!cxi_zeCommandQueueExecuteCommandLists) {
		ZE_ERR("Failed to find zeCommandQueueExecuteCommandLists\n");
		goto err_dlclose;
	}

	cxi_zeCommandListDestroy =
		dlsym(libze_handle, "zeCommandListDestroy");
	if (!cxi_zeCommandListDestroy) {
		ZE_ERR("Failed to find zeCommandListDestroy\n");
		goto err_dlclose;
	}

	cxi_zeCommandListAppendMemoryCopy =
		dlsym(libze_handle, "zeCommandListAppendMemoryCopy");
	if (!cxi_zeCommandListAppendMemoryCopy) {
		ZE_ERR("Failed to find zeCommandListAppendMemoryCopy\n");
		goto err_dlclose;
	}

	cxi_zeInit = dlsym(libze_handle, "zeInit");
	if (!cxi_zeInit) {
		ZE_ERR("Failed to find zeInit\n");
		goto err_dlclose;
	}

	cxi_zeDriverGet = dlsym(libze_handle, "zeDriverGet");
	if (!cxi_zeDriverGet) {
		ZE_ERR("Failed to find zeDriverGet\n");
		goto err_dlclose;
	}

	cxi_zeContextCreate = dlsym(libze_handle, "zeContextCreate");
	if (!cxi_zeContextCreate) {
		ZE_ERR("Failed to find zeContextCreate\n");
		goto err_dlclose;
	}

	cxi_zeDeviceGet = dlsym(libze_handle, "zeDeviceGet");
	if (!cxi_zeDeviceGet) {
		ZE_ERR("Failed to find zeDeviceGet\n");
		goto err_dlclose;
	}

	cxi_zeCommandQueueCreate = dlsym(libze_handle, "zeCommandQueueCreate");
	if (!cxi_zeCommandQueueCreate) {
		ZE_ERR("Failed to find zeCommandQueueCreate\n");
		goto err_dlclose;
	}

	cxi_zeCommandQueueDestroy =
		dlsym(libze_handle, "zeCommandQueueDestroy");
	if (!cxi_zeCommandQueueDestroy) {
		ZE_ERR("Failed to find zeCommandQueueDestroy\n");
		goto err_dlclose;
	}

	cxi_zeContextDestroy = dlsym(libze_handle, "zeContextDestroy");
	if (!cxi_zeContextDestroy) {
		ZE_ERR("Failed to find zeContextDestroy\n");
		goto err_dlclose;
	}

	cxi_zeMemGetAddressRange = dlsym(libze_handle, "zeMemGetAddressRange");
	if (!cxi_zeMemGetAddressRange) {
		ZE_ERR("Failed to find zeMemGetAddressRange\n");
		goto err_dlclose;
	}

	cxi_zeMemGetIpcHandle = dlsym(libze_handle, "zeMemGetIpcHandle");
	if (!cxi_zeMemGetIpcHandle) {
		ZE_ERR("Failed to find zeMemGetIpcHandle\n");
		goto err_dlclose;
	}
	cxi_zeMemGetAllocProperties = dlsym(libze_handle,
					    "zeMemGetAllocProperties");
	if (!cxi_zeMemGetAllocProperties) {
		ZE_ERR("Failed to find zeMemGetAllocProperties\n");
		goto err_dlclose;
	}

	return 0;

err_dlclose:
	dlclose(libze_handle);
err:
	return -ENODATA;
}

static int ze_malloc(void **devPtr, size_t size)
{
	ze_result_t ze_ret;

	ze_ret = cxi_zeMemAllocDevice(context, &device_desc, size, 16,
				      devices[current_device], devPtr);
	if (ze_ret != ZE_RESULT_SUCCESS) {
		ZE_ERR("Failed to allocate memory: rc=%#x", ze_ret);
		return -ENOMEM;
	}

	return 0;
}

static int ze_free(void *devPtr)
{
	ze_result_t ze_ret;

	ze_ret = cxi_zeMemFree(context, devPtr);
	if (ze_ret != ZE_RESULT_SUCCESS) {
		ZE_ERR("Failed to free memory: rc=%#x", ze_ret);
		return -ENOMEM;
	}

	return 0;
}

static int ze_memset(void *devPtr, int value, size_t size)
{
	ze_command_list_handle_t cmd_list;
	ze_result_t ze_ret;

	ze_ret = cxi_zeCommandListCreate(context, devices[current_device], &cl_desc,
					 &cmd_list);
	if (ze_ret != ZE_RESULT_SUCCESS) {
		ZE_ERR("Failed to create command list: %#x\n", ze_ret);
		return -EIO;
	}

	ze_ret = cxi_zeCommandListAppendMemoryFill(cmd_list, devPtr, &value,
						   sizeof(value), size, NULL, 0,
						   NULL);
	if (ze_ret != ZE_RESULT_SUCCESS) {
		ZE_ERR("Failed to append memory fill command: %#x\n", ze_ret);
		goto free;
	}

	ze_ret = cxi_zeCommandListClose(cmd_list);
	if (ze_ret != ZE_RESULT_SUCCESS) {
		ZE_ERR("Failed to close command: %#x\n", ze_ret);
		goto free;
	}

	ze_ret = cxi_zeCommandQueueExecuteCommandLists(cmd_queue[current_device], 1,
						       &cmd_list, NULL);
	if (ze_ret != ZE_RESULT_SUCCESS)
		ZE_ERR("Failed to execute command: %#x\n", ze_ret);

free:
	if (ze_ret == ZE_RESULT_SUCCESS) {
		ze_ret = cxi_zeCommandListDestroy(cmd_list);
		if (ze_ret == ZE_RESULT_SUCCESS)
			return 0;

		ZE_ERR("Failed to destroy command list: %#x\n", ze_ret);
	}

	return -EINVAL;
}

static int ze_memcpy(void *dst, const void *src, size_t size, int kind)
{
	ze_command_list_handle_t cmd_list;
	ze_result_t ze_ret;

	ze_ret = cxi_zeCommandListCreate(context, devices[current_device], &cl_desc,
					 &cmd_list);
	if (ze_ret != ZE_RESULT_SUCCESS) {
		ZE_ERR("Failed to create command list: %#x\n", ze_ret);
		return -EIO;
	}

	ze_ret = cxi_zeCommandListAppendMemoryCopy(cmd_list, dst, src, size,
						   NULL, 0, NULL);
	if (ze_ret != ZE_RESULT_SUCCESS) {
		ZE_ERR("Failed to append memory copy command: %#x\n", ze_ret);
		goto free;
	}

	ze_ret = cxi_zeCommandListClose(cmd_list);
	if (ze_ret != ZE_RESULT_SUCCESS) {
		ZE_ERR("Failed to close command: %#x\n", ze_ret);
		goto free;
	}

	ze_ret = cxi_zeCommandQueueExecuteCommandLists(cmd_queue[current_device], 1,
						       &cmd_list, NULL);
	if (ze_ret != ZE_RESULT_SUCCESS)
		ZE_ERR("Failed to execute command: %#x\n", ze_ret);

free:
	if (ze_ret == ZE_RESULT_SUCCESS) {
		ze_ret = cxi_zeCommandListDestroy(cmd_list);
		if (ze_ret == ZE_RESULT_SUCCESS)
			return 0;

		ZE_ERR("Failed to destroy command list: %#x\n", ze_ret);
	}

	return -EINVAL;
}

static const char *const alloc_types[] = {
	[ZE_MEMORY_TYPE_UNKNOWN] = "unknown",
	[ZE_MEMORY_TYPE_HOST] = "host",
	[ZE_MEMORY_TYPE_DEVICE] = "device",
	[ZE_MEMORY_TYPE_SHARED] = "shared",
};

static int ze_mem_props(const void *addr, void **base, size_t *size,
			int *dma_buf_fd)
{
	void *handle;
	ze_result_t ze_ret;
	ze_device_handle_t device_ptr;
	ze_memory_allocation_properties_t alloc_props = {};

	ze_ret = cxi_zeMemGetIpcHandle(context, addr,
				       (ze_ipc_mem_handle_t *)&handle);
	if (ze_ret != ZE_RESULT_SUCCESS) {
		ZE_ERR("Failed to get IPC handle: %#x\n", ze_ret);
		return -EIO;
	}

	*dma_buf_fd = (int)(uintptr_t)handle;

	ze_ret = cxi_zeMemGetAddressRange(context, addr, base, size);
	if (ze_ret != ZE_RESULT_SUCCESS) {
		ZE_ERR("Failed to get address range: %#x\n", ze_ret);
		return -EIO;
	}

	ze_ret = cxi_zeMemGetAllocProperties(context, addr, &alloc_props,
					     &device_ptr);
	if (ze_ret != ZE_RESULT_SUCCESS)
		ZE_ERR("Failed to get AllocProperties:%#x\n", ze_ret);
	else if (0)
		printf("alloc type:%d (%s)\n", alloc_props.type,
		       alloc_types[alloc_props.type]);

	return 0;
}

int ze_device_count(int *count)
{
	if (!count)
		return -EINVAL;
	*count = num_devices;
	return 0;
}

int ze_get_device(int *dev_id)
{
	if (!dev_id)
		return -EINVAL;
	*dev_id = current_device;
	return 0;
}

int ze_set_device(int dev_id)
{
	if (dev_id < 0 || dev_id > num_devices)
		return -EINVAL;
	current_device = dev_id;
	return 0;
}

int ze_lib_init(void)
{
	int ret;
	int d;
	ze_driver_handle_t driver;
	ze_context_desc_t context_desc = {};
	ze_result_t ze_ret;
	uint32_t count;

	ret = ze_dlopen();
	if (ret)
		return ret;

	ze_ret = cxi_zeInit(ZE_INIT_FLAG_GPU_ONLY);
	if (ze_ret != ZE_RESULT_SUCCESS) {
		ZE_ERR("zeInit failed: %#x\n", ze_ret);
		return -EIO;
	}

	count = 1;
	ze_ret = cxi_zeDriverGet(&count, &driver);
	if (ze_ret != ZE_RESULT_SUCCESS) {
		ZE_ERR("zeDriverGet failed: %#x\n", ze_ret);
		return -EIO;
	}

	ze_ret = cxi_zeContextCreate(driver, &context_desc, &context);
	if (ze_ret != ZE_RESULT_SUCCESS) {
		ZE_ERR("zeContextCreate failed: %#x\n", ze_ret);
		return -EIO;
	}

	count = 0;
	ze_ret = cxi_zeDeviceGet(driver, &count, NULL);
	if (ze_ret != ZE_RESULT_SUCCESS) {
		ZE_ERR("zeDeviceGet failed: %#x\n", ze_ret);
		return -EIO;
	} else if (count > ZE_MAX_DEVICES) {
		num_devices = ZE_MAX_DEVICES;
	} else {
		num_devices = count;
	}

	ze_ret = cxi_zeDeviceGet(driver, &count, devices);
	if (ze_ret != ZE_RESULT_SUCCESS) {
		ZE_ERR("zeDeviceGet failed: %#x\n", ze_ret);
		return -EIO;
	}

	for (d = 0; d < num_devices; d++) {
		ze_ret = cxi_zeCommandQueueCreate(context, devices[d],
						  &cq_desc,
						  &cmd_queue[d]);
		if (ze_ret != ZE_RESULT_SUCCESS) {
			ZE_ERR("zeCommandQueueCreate failed: %#x\n", ze_ret);
			return -EIO;
		}
	}

	g_malloc = ze_malloc;
	g_free = ze_free;
	g_memset = ze_memset;
	g_memcpy = ze_memcpy;
	g_mem_properties = ze_mem_props;
	g_device_count = ze_device_count;
	g_get_device = ze_get_device;
	g_set_device = ze_set_device;

	return 0;
}

void ze_lib_fini(void)
{
	int i;

	if (!libze_handle)
		return;

	for (i = 0; i < num_devices; i++) {
                if (cmd_queue[i]) {
			cxi_zeCommandQueueDestroy(cmd_queue[i]);
			cmd_queue[i] = NULL;
		}
	}

	if (context) {
		cxi_zeContextDestroy(context);
		context = NULL;
	}

	ze_dlclose();

	num_devices = 0;
	current_device = 0;

	g_malloc = NULL;
	g_free = NULL;
	g_memset = NULL;
	g_memcpy = NULL;
	g_mem_properties = NULL;
	g_device_count = NULL;
	g_get_device = NULL;
	g_set_device = NULL;
}

#endif /* HAVE_ZE_SUPPORT */
