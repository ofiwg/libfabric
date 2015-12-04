/*
 * Copyright (c) 2014 Intel Corporation, Inc.  All rights reserved.
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

#include <stdlib.h>
#include <stdio.h>

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <limits.h>
#include "sock.h"
#include "sock_util.h"

#ifdef HAVE_EPOLL
#include <sys/epoll.h>
#else
#include <poll.h>
#endif

#define SOCK_LOG_DBG(...) _SOCK_LOG_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define SOCK_LOG_ERROR(...) _SOCK_LOG_ERROR(FI_LOG_EP_CTRL, __VA_ARGS__)

#ifdef HAVE_EPOLL
int sock_epoll_create(struct sock_epoll_set *set, int size)
{
	set->size = size;
	set->used = 0;
	set->events = calloc(size, sizeof(struct epoll_event));
	return set->fd = epoll_create(size);
}

int sock_epoll_add(struct sock_epoll_set *set, int fd)
{
	struct epoll_event event;
	event.data.fd = fd;
	event.events = EPOLLIN;

	if (set->used == set->size)
		return -1;

	set->used++;
	return epoll_ctl(set->fd, EPOLL_CTL_ADD, fd, &event);
}

int sock_epoll_del(struct sock_epoll_set *set, int fd)
{
	set->used--;
	return epoll_ctl(set->fd, EPOLL_CTL_DEL, fd, NULL);
}

int sock_epoll_wait(struct sock_epoll_set *set, int timeout)
{
	if (!set->used)
		return 0;
	return epoll_wait(set->fd, set->events, set->used, timeout);
}

int sock_epoll_get_fd_at_index(struct sock_epoll_set *set, int index)
{
	return set->events[index].data.fd;
}

void sock_epoll_close(struct sock_epoll_set *set)
{
	free(set->events);
	close(set->fd);
}

#else

int sock_epoll_create(struct sock_epoll_set *set, int size)
{
	set->size = size;
	set->used = 0;
	set->pollfds = calloc(size, sizeof(struct pollfd));
	return set->pollfds ? 0 : -1;
}

int sock_epoll_add(struct sock_epoll_set *set, int fd)
{
	if (set->used == set->size)
		return -1;

	set->pollfds[set->used].fd = fd;
	set->pollfds[set->used].events = POLLIN;

	set->used++;
	return 0;
}

int sock_epoll_del(struct sock_epoll_set *set, int fd)
{
	int i;

	for (i = 0; i < set->used; i++) {
		if (set->pollfds[i].fd == fd)
      			break;
  	}

	if (i == set->used)
    		return -1;

	set->pollfds[i].fd = set->pollfds[set->used - 1].fd;
	set->used--;
	return 0;
}

int sock_epoll_wait(struct sock_epoll_set *set, int timeout)
{
	if (!set->used)
		return 0;

	return poll(set->pollfds, set->used, timeout);
}

int sock_epoll_get_fd_at_index(struct sock_epoll_set *set, int index)
{
	int i;

	for (i = 0; i < set->used; i++) {
		if ((set->pollfds[i].revents & POLLIN)) {
			if (index) {
				index--;
				continue;
      			} else {
				return set->pollfds[i].fd;
      			}
    		}
  	}
	return -1;
}

void sock_epoll_close(struct sock_epoll_set *set)
{
	free(set->pollfds);
}

#endif
