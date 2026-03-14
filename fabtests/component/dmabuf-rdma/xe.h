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

#ifndef _DMABUF_RDMA_TESTS_XE_H_
#define _DMABUF_RDMA_TESTS_XE_H_

#include <stdint.h>
#include <stdbool.h>
#include "hmem.h"
#include "level_zero/ze_api.h"

#define MAX_GPUS	(8)

/*
 * Buffer location and method of allocation
 */
enum {
	MALLOC,	/* Host memory allocated via malloc and alike */
	HOST,	/* Host memory allocated via zeMemAllocHost */
	DEVICE,	/* Device memory allocated via zeMemAllocDevice */
	SHARED	/* Shared memory allocated via zeMemAllocShared */
};

/*
 * All information related to a buffer allocated via oneAPI L0 API.
 */
struct xe_buf {
	void			*buf;
	void			*base;
	uint64_t		offset;
	size_t			size;
	ze_device_handle_t	dev;
	ze_memory_type_t	type;
	int			location;
	int			fd;
	ze_ipc_mem_handle_t	ipc_handle;
};

struct gpu {
	int dev_num;
	int subdev_num;
	ze_device_handle_t device;
	ze_command_list_handle_t cmdl;
};

/*
 * Initialize GPU device.
 */
int	xe_init(struct gpu *gpu);

/*
 * Cleanup GPU resources
 */
void xe_cleanup_gpu(struct gpu *gpu);

/*
 * Cleanup generic XE resources
 */
void	xe_cleanup(void);

/*
 * Get the dma-buf fd associated with the buffer allocated with the oneAPI L0
 * functions. Return -1 if it's not a dma-buf object.
 */
int	xe_get_buf_fd(void *buf);

/*
 * Alloctaed a buffer from specified location, on the speficied GPU if
 * applicable. The xe_buf output is optional, can pass in NULL if the
 * information is not needed.
 */
void	*xe_alloc_buf(size_t page_size, size_t size, int where, struct gpu *gpu,
		      struct xe_buf *xe_buf);

/*
 * Show the fields of the xe_buf structure.
 */
void	xe_show_buf(struct xe_buf *buf);

/*
 * Free the buffer allocated with xe_alloc_buf.
 */
void	xe_free_buf(struct xe_buf *xe_buf);

/*
 * Like memset(). Use oneAPI L0 to access device memory.
 */
void	xe_set_buf(void *buf, char c, size_t size, int location,
		   struct gpu *gpu);

/*
 * Like memcpy(). Use oneAPI L0 to access device memory.
 */
void	xe_copy_buf(void *dst, void *src, size_t size, struct gpu *gpu);

/*
 * Show the initialized GPU resources.
 */
void	show_xe_resources(struct gpu *gpu);

#endif /* _DMABUF_RDMA_TESTS_XE_H_ */

