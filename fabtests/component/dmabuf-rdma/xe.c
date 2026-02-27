/*
 * Copyright (c) 2021-2022 Intel Corporation.  All rights reserved.
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <level_zero/ze_api.h>
#include "util.h"
#include "xe.h"

/*
 * Memory allocation & copy routines using oneAPI L0
 */

extern int buf_location;

static ze_driver_handle_t gpu_driver;
static ze_context_handle_t gpu_context;

static int init_gpu(struct gpu *gpu)
{
	uint32_t count;
	ze_device_handle_t *devices;
	ze_command_queue_desc_t cq_desc = {
	    .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
	    .ordinal = 0,
	    .index = 0,
	    .flags = 0,
	    .mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
	    .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
	};

	count = 0;
	EXIT_ON_ERROR(libze_ops.zeDeviceGet(gpu_driver, &count, NULL));

	if (count < gpu->dev_num + 1) {
		fprintf(stderr, "GPU device %d does't exist\n", gpu->dev_num);
		goto err_out;
	}

	devices = calloc(count, sizeof(*devices));
	if (!devices) {
		perror("calloc");
		goto err_out;
	}

	EXIT_ON_ERROR(libze_ops.zeDeviceGet(gpu_driver, &count, devices));
	gpu->device = devices[gpu->dev_num];
	free(devices);

	if (gpu->subdev_num >= 0) {
		count = 0;
		EXIT_ON_ERROR(libze_ops.zeDeviceGetSubDevices(gpu->device,
							      &count, NULL));

		if (count < gpu->subdev_num + 1) {
			fprintf(stderr, "GPU subdevice %d.%d does't exist\n",
				gpu->dev_num, gpu->subdev_num);
			goto err_out;
		}

		devices = calloc(count, sizeof(*devices));
		if (!devices) {
			perror("calloc");
			goto err_out;
		}

		EXIT_ON_ERROR(libze_ops.zeDeviceGetSubDevices(gpu->device,
			      &count, devices));
		gpu->device = devices[gpu->subdev_num];
		free(devices);

		printf("using GPU subdevice %d.%d: %p (total %d)\n",
			gpu->dev_num, gpu->subdev_num, gpu->device, count);
	} else {
		printf("using GPU device %d: %p (total %d)\n", gpu->dev_num,
			gpu->device, count);
	}

	EXIT_ON_ERROR(libze_ops.zeCommandListCreateImmediate(
		      gpu_context, gpu->device, &cq_desc, &gpu->cmdl));
	return 0;

err_out:
	return -1;
}

int xe_init(struct gpu *gpu)
{
	uint32_t count;
	ze_context_desc_t ctxt_desc = {};

	EXIT_ON_ERROR(init_libze_ops());
	EXIT_ON_ERROR(libze_ops.zeInit(ZE_INIT_FLAG_GPU_ONLY));

	count = 1;
	EXIT_ON_ERROR(libze_ops.zeDriverGet(&count, &gpu_driver));
	printf("Using first driver: %p (total >= %d)\n", gpu_driver, count);

	EXIT_ON_ERROR(libze_ops.zeContextCreate(gpu_driver, &ctxt_desc,
						&gpu_context));

	return init_gpu(gpu);
}

void xe_show_buf(struct xe_buf *buf)
{
	char *type_str;
	switch(buf->type) {
		case ZE_MEMORY_TYPE_HOST:
			type_str = "HOST";
			break;
		case ZE_MEMORY_TYPE_DEVICE:
			type_str = "DEVICE";
			break;
		case ZE_MEMORY_TYPE_SHARED:
			type_str = "SHARED";
			break;
		case ZE_MEMORY_TYPE_UNKNOWN:
		default:
			type_str = "MALLOC";
			break;
	}
	printf("Allocation %s:\n"
	       "\tbuf %p\n"
	       "\talloc_base %p\n"
	       "\talloc_size %ld\n"
	       "\toffset 0x%lx\n"
	       "\tdevice %p\n",
		type_str, buf->buf, buf->base, buf->size, buf->offset,
		buf->dev);
}

int xe_get_buf_fd(void *buf)
{
	ze_ipc_mem_handle_t ipc;

	memset(&ipc, 0, sizeof(ipc));
	CHECK_ERROR((libze_ops.zeMemGetIpcHandle(gpu_context, buf, &ipc)));

	return (int)*(uint64_t *)&ipc;
err_out:
	return -1;
}

void *xe_alloc_buf(size_t page_size, size_t size, int where, struct gpu *gpu,
		   struct xe_buf *xe_buf)
{
	void *buf = NULL;
	ze_device_mem_alloc_desc_t dev_desc = {};
	ze_host_mem_alloc_desc_t host_desc = {};
	ze_memory_allocation_properties_t alloc_props = { .type = -1};
	ze_device_handle_t alloc_dev = gpu->device;
	void *alloc_base;
	size_t alloc_size;
	ze_external_memory_export_fd_t export_fd = {
		.stype	= ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD,
		.pNext	= NULL,
		.flags	= ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
		.fd	= -1,
	};

	if (where != MALLOC)
		alloc_props.pNext = &export_fd;

	switch (where) {
	  case MALLOC:
		posix_memalign(&buf, page_size, size);
		alloc_base = buf;
		alloc_size = size;
		break;
	  case HOST:
		EXIT_ON_ERROR(libze_ops.zeMemAllocHost(gpu_context, &host_desc,
						       size, page_size, &buf));
		break;
	  case DEVICE:
		EXIT_ON_ERROR(libze_ops.zeMemAllocDevice(gpu_context, &dev_desc,
							 size, page_size,
							 gpu->device,
							 &buf));
		break;
	  default:
		EXIT_ON_ERROR(libze_ops.zeMemAllocShared(gpu_context, &dev_desc,
							 &host_desc, size,
							 page_size,
							 gpu->device,
							 &buf));
		break;
	}

	if (where != MALLOC) {
		EXIT_ON_ERROR(libze_ops.zeMemGetAllocProperties(gpu_context,
								buf,
								&alloc_props,
								&alloc_dev));
		EXIT_ON_ERROR(libze_ops.zeMemGetAddressRange(gpu_context, buf,
							     &alloc_base,
							     &alloc_size));
		xe_buf->fd = export_fd.fd;
	}

	if (xe_buf) {
		xe_buf->buf = buf;
		xe_buf->base = alloc_base;
		xe_buf->size = alloc_size;
		xe_buf->offset = (char *)buf - (char *)alloc_base;
		xe_buf->type = alloc_props.type;
		xe_buf->dev = alloc_dev;
		xe_buf->location = where;
		xe_show_buf(xe_buf);
	}
	return buf;
}

void xe_free_buf(struct xe_buf *xe_buf)
{
	if (!xe_buf->buf)
		return;

	if (xe_buf->location == MALLOC)
		free(xe_buf->buf);
	else
		CHECK_ERROR(libze_ops.zeMemFree(gpu_context, xe_buf->buf));
err_out:
	return;
}

void xe_set_buf(void *buf, char c, size_t size, int location, struct gpu *gpu)
{
	if (location == MALLOC) {
		memset(buf, c, size);
	} else {
		EXIT_ON_ERROR(libze_ops.zeCommandListAppendMemoryFill(
				gpu->cmdl, buf, &c, 1, size, NULL, 0,
				NULL));
		EXIT_ON_ERROR(libze_ops.zeCommandListReset(gpu->cmdl));
	}
}

void xe_copy_buf(void *dst, void *src, size_t size, struct gpu *gpu)
{
	EXIT_ON_ERROR(libze_ops.zeCommandListAppendMemoryCopy(
				  gpu->cmdl, dst, src, size, NULL, 0,
				  NULL));
	EXIT_ON_ERROR(libze_ops.zeCommandListReset(gpu->cmdl));
}

void xe_cleanup_gpu(struct gpu *gpu)
{
	if (gpu->cmdl) {
		libze_ops.zeCommandListDestroy(gpu->cmdl);
		gpu->cmdl = NULL;
	}
	gpu->device = NULL;
	gpu->dev_num = gpu->subdev_num = -1;
}

void xe_cleanup(void)
{
	CHECK_ERROR(libze_ops.zeContextDestroy(gpu_context));
	gpu_context = NULL;
err_out:
	return;
}

void show_xe_resources(struct gpu *gpu)
{
	printf("Resources for GPU[%d]:\n"
		"\t\tdevice %d\n"
		"\t\tsubdevice %d\n"
		"\t\tdevice handle %p\n"
		"\t\tcmdl %p\n",
		gpu->dev_num, gpu->dev_num, gpu->subdev_num, gpu->device,
		gpu->cmdl);
}
