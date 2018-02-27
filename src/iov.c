/*
 * Copyright (c) 2016 Intel Corp., Inc.  All rights reserved.
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

#include "config.h"

#include <string.h>

#include <ofi.h>
#include <ofi_iov.h>

uint64_t ofi_copy_iov_buf(const struct iovec *iov, size_t iov_count, uint64_t iov_offset,
			  void *buf, uint64_t bufsize, int dir)
{
	uint64_t done = 0, len;
	char *iov_buf;
	size_t i;

	for (i = 0; i < iov_count && bufsize; i++) {
		len = iov[i].iov_len;

		if (iov_offset > len) {
			iov_offset -= len;
			continue;
		}

		iov_buf = (char *)iov[i].iov_base + iov_offset;
		len -= iov_offset;

		len = MIN(len, bufsize);
		if (dir == OFI_COPY_BUF_TO_IOV)
			memcpy(iov_buf, (char *) buf + done, len);
		else if (dir == OFI_COPY_IOV_TO_BUF)
			memcpy((char *) buf + done, iov_buf, len);

		iov_offset = 0;
		bufsize -= len;
		done += len;
	}
	return done;
}

void ofi_consume_iov(struct iovec *iov, size_t *iov_count, size_t consumed)
{
	size_t i;

	if (*iov_count == 1)
		goto out;

	for (i = 0; i < *iov_count; i++) {
		if (consumed < iov[i].iov_len)
			break;
		consumed -= iov[i].iov_len;
	}
	memmove(iov, &iov[i], sizeof(*iov) * (*iov_count - i));
	*iov_count -= i;
out:
	iov[0].iov_base = (uint8_t *)iov[0].iov_base + consumed;
	iov[0].iov_len -= consumed;
}

int ofi_truncate_iov(struct iovec *iov, size_t *iov_count, size_t trim_size)
{
	size_t i;

	for (i = 0; i < *iov_count; i++) {
		if (trim_size <= iov[i].iov_len) {
			iov[i].iov_len = trim_size;
			*iov_count = i + 1;
			return FI_SUCCESS;
		}
		trim_size -= iov[i].iov_len;
	}
	return -FI_ETRUNC;
}
