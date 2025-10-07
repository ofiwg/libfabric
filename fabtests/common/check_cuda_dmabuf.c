/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 * SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved.
*/

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <shared.h>
#include <hmem.h>

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
        return FT_CUDA_NOT_SUPPORTED;
    }

    enum ft_cuda_memory_support cuda_memory_support = ft_cuda_memory_support();
    FT_INFO("dmabuf: ft_cuda_memory_support() -> %d", cuda_memory_support);

    ft_cuda_cleanup();
    return cuda_memory_support;
}