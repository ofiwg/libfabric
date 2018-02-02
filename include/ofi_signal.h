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

#ifndef _OFI_SIGNAL_H_
#define _OFI_SIGNAL_H_

#include "config.h"

#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>

#include <ofi_file.h>
#include <ofi_osd.h>
#include <rdma/fi_errno.h>


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
		return -ofi_sockerr();

	ret = fi_fd_nonblock(signal->fd[FI_READ_FD]);
	if (ret)
		goto err;

	return 0;

err:
	ofi_close_socket(signal->fd[0]);
	ofi_close_socket(signal->fd[1]);
	return ret;
}

static inline void fd_signal_free(struct fd_signal *signal)
{
	ofi_close_socket(signal->fd[0]);
	ofi_close_socket(signal->fd[1]);
}

static inline void fd_signal_set(struct fd_signal *signal)
{
	char c = 0;
	if (signal->wcnt == signal->rcnt) {
		if (ofi_write_socket(signal->fd[FI_WRITE_FD], &c, sizeof c) == sizeof c)
			signal->wcnt++;
	}
}

static inline void fd_signal_reset(struct fd_signal *signal)
{
	char c;
	if (signal->rcnt != signal->wcnt) {
		if (ofi_read_socket(signal->fd[FI_READ_FD], &c, sizeof c) == sizeof c)
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

#ifdef HAVE_EPOLL
#include <sys/epoll.h>

typedef int fi_epoll_t;

static inline int fi_epoll_create(int *ep)
{
	*ep = epoll_create(4);
	return *ep < 0 ? -ofi_syserr() : 0;
}

static inline int fi_epoll_add(int ep, int fd, void *context)
{
	struct epoll_event event;
	int ret;

	event.data.ptr = context;
	event.events = EPOLLIN;
	ret = epoll_ctl(ep, EPOLL_CTL_ADD, fd, &event);
	if ((ret == -1) && (ofi_syserr() != EEXIST))
		return -ofi_syserr();
	return 0;
}

static inline int fi_epoll_del(int ep, int fd)
{
	return epoll_ctl(ep, EPOLL_CTL_DEL, fd, NULL) ? -ofi_syserr() : 0;
}

static inline int fi_epoll_wait(int ep, void **contexts, int max_contexts,
                                int timeout)
{
	struct epoll_event events[max_contexts];
	int ret;
	int i;

	ret = epoll_wait(ep, events, max_contexts, timeout);
	if (ret == -1)
		return -ofi_syserr();

	for (i = 0; i < ret; i++)
		contexts[i] = events[i].data.ptr;
	return ret;
}

static inline void fi_epoll_close(int ep)
{
	close(ep);
}

#else
#include <poll.h>

typedef struct fi_epoll {
	int		size;
	int		nfds;
	struct pollfd	*fds;
	void		**context;
	int		index;
} *fi_epoll_t;

int fi_epoll_create(struct fi_epoll **ep);
int fi_epoll_add(struct fi_epoll *ep, int fd, void *context);
int fi_epoll_del(struct fi_epoll *ep, int fd);
int fi_epoll_wait(struct fi_epoll *ep, void **contexts, int max_contexts,
                  int timeout);
void fi_epoll_close(struct fi_epoll *ep);

#endif /* HAVE_EPOLL */

#endif /* _OFI_SIGNAL_H_ */
