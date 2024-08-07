/*
 * Copyright (c) 2018-2024 GigaIO, Inc. All Rights Reserved.
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
#include "test_util.h"

int hmem_cuda_init(void);
void hmem_cuda_cleanup(void);
int hmem_cuda_alloc(void *uaddr, size_t len);
void hmem_cuda_free(void *uaddr);
int hmem_cuda_memcpy_h2d(void *dst, void *src, size_t len);
int hmem_cuda_memcpy_d2h(void *dst, void *src, size_t len);

int hmem_rocm_init(void);
void hmem_rocm_cleanup(void);
int hmem_rocm_alloc(void *uaddr, size_t len);
void hmem_rocm_free(void *uaddr);
int hmem_rocm_memcpy_h2d(void *dst, const void *src, size_t len);
int hmem_rocm_memcpy_d2h(void *dst, const void *src, size_t len);
