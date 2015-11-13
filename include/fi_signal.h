/*
 * Copyright (c) 2011-2015 Intel Corporation.  All rights reserved.
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

#if !defined(FI_SIGNAL_H)
#define FI_SIGNAL_H

#include "config.h"

#include <unistd.h>
#include <errno.h>
#include <fcntl.h>

#include <rdma/fi_errno.h>

#include "fi.h"


enum {
	FI_READ_FD,
	FI_WRITE_FD
};

struct fd_signal {
	int		rcnt;
	int		wcnt;
	int		fd[2];
};

static inline int fd_signal_init(struct fd_signal *signal)
{
	int ret;

	ret = socketpair(AF_UNIX, SOCK_STREAM, 0, signal->fd);
	if (ret < 0)
		return -errno;

	ret = fcntl(signal->fd[FI_READ_FD], F_SETFL, O_NONBLOCK);
	if (ret < 0)
		goto err;

	return 0;

err:
	close(signal->fd[0]);
	close(signal->fd[1]);
	return -errno;
}

static inline void fd_signal_free(struct fd_signal *signal)
{
	close(signal->fd[0]);
	close(signal->fd[1]);
}

static inline void fd_signal_set(struct fd_signal *signal)
{
	char c = 0;
	if (signal->wcnt == signal->rcnt) {
		if (write(signal->fd[FI_WRITE_FD], &c, sizeof c) == sizeof c)
			signal->wcnt++;
	}
}

static inline void fd_signal_reset(struct fd_signal *signal)
{
	char c;
	if (signal->rcnt != signal->wcnt) {
		if (read(signal->fd[FI_READ_FD], &c, sizeof c) == sizeof c)
			signal->rcnt++;
	}
}

static inline int fd_signal_poll(struct fd_signal *signal, int timeout)
{
	int ret;

	ret = fi_poll_fd(signal->fd[FI_READ_FD], timeout);
	if (ret < 0)
		return ret;

	return (ret == 0) ? -FI_ETIMEDOUT : 0;
}

#endif /* FI_SIGNAL_H */

