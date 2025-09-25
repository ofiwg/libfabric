/*
 * Copyright (c) 2024, Amazon.com, Inc.  All rights reserved.
 *
 * This software is available to you under the BSD license
 * below:
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
 *
 * This test returns whether or not dmabuf is viable and supported
 * based on aws-ofi-nccl logic
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <shared.h>
#include <hmem.h>

static cuda_memory_support_e dmabuf_viable_and_supported( void )
{
	cuda_memory_support_e cuda_memory_support = ft_cuda_memory_support();

	return cuda_memory_support;
}

int main(int argc, char **argv)
{
    int ret;

    /* Make sure default CUDA device is sane for ft_cuda_init() */
    opts = INIT_OPTS;
    opts.device = 0;   /* cuda device 0 */

    /* Initialize CUDA side only; avoid ft_init_fabric() */
    ret = ft_cuda_init();
    if (ret != FI_SUCCESS) {
        FT_ERR("ft_cuda_init failed: %d", ret);
        return CUDA_MEMORY_SUPPORT__UNKNOWN;
    }

    cuda_memory_support_e cuda_memory_support = dmabuf_viable_and_supported();
    FT_INFO("dmabuf: ft_cuda_memory_support() -> %d", cuda_memory_support);

    ft_cuda_cleanup();
    return cuda_memory_support;
}