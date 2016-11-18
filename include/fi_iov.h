/*
 * Copyright (c) 2016 Intel Corporation.  All rights reserved.
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
 *
 */

#if !defined(IOV_H)
#define IOV_H

#include "config.h"

#include <stdlib.h>
#include <sys/uio.h>
#include <inttypes.h>

#define OFI_COPY_IOV_TO_BUF 0
#define OFI_COPY_BUF_TO_IOV 1

static inline size_t ofi_get_iov_len(const struct iovec *iov, size_t iov_count)
{
	size_t i, len = 0;
	for (i = 0; i < iov_count; i++)
		len += iov[i].iov_len;
	return len;
}

uint64_t ofi_copy_iov_buf(const struct iovec *iov, size_t iov_count,
		void *buf, uint64_t rem, uint64_t skip, int dir);

#endif /* IOV_H */

