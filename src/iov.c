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

#include <fi.h>
#include <fi_iov.h>

uint64_t ofi_copy_iov_buf(const struct iovec *iov, size_t iov_count,
		void *buf, uint64_t rem, uint64_t skip, int dir)
{
	int i;
	uint64_t done = 0, len;
	char *iov_base;

	for (i = 0; i < iov_count && rem; i++) {
		len = iov[i].iov_len;

		if (skip > len) {
			skip -= len;
			continue;
		}

		iov_base = (char *)iov[i].iov_base + skip;
		len -= skip;

		len = MIN(len, rem);
		if (dir == OFI_COPY_BUF_TO_IOV)
			memcpy(iov_base, (char *) buf + done, len);
		else if (dir == OFI_COPY_IOV_TO_BUF)
			memcpy((char *) buf + done, iov_base, len);
		skip = 0, rem -= len, done += len;
	}
	return done;
}
