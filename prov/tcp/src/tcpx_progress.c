/*
 * Copyright (c) 2017 Intel Corporation, Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *	   Redistribution and use in source and binary forms, with or
 *	   without modification, are permitted provided that the following
 *	   conditions are met:
 *
 *		- Redistributions of source code must retain the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer.
 *
 *		- Redistributions in binary form must reproduce the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer in the documentation and/or other materials
 *		  provided with the distribution.
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

#include <rdma/fi_errno.h>

#include <prov.h>
#include "tcpx.h"

int tcpx_progress_close(struct tcpx_domain *domain)
{
	struct tcpx_progress *progress = domain->progress;


	progress->do_progress = 0;
	if (progress->progress_thread &&
	    pthread_join(progress->progress_thread, NULL)) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"progress thread failed to join\n");
	}
	fd_signal_free(&progress->signal);
	fastlock_destroy(&progress->signal_lock);
	fi_epoll_close(progress->epoll_set);
	return FI_SUCCESS;
}

void tcpx_progress_signal(struct tcpx_progress *progress)
{
	fastlock_acquire(&progress->signal_lock);
	fd_signal_set(&progress->signal);
	fastlock_release(&progress->signal_lock);
}

static int tcpx_progress_wait(struct tcpx_progress *progress)
{
	int ret;
	void *ep_context;

	ret = fi_epoll_wait(progress->epoll_set,
			    &ep_context,1,-1);
	if (ret < 0) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"Poll failed\n");
		return -errno;
	}

	fastlock_acquire(&progress->signal_lock);
	fd_signal_reset(&progress->signal);
	fastlock_release(&progress->signal_lock);

	return FI_SUCCESS;
}

static int tcpx_progress_wait_ok(struct tcpx_progress *progress)
{
	return -FI_ENOSYS;
}

void *tcpx_progress_thread(void *data)
{
	struct tcpx_progress *progress;
	progress = (struct tcpx_progress *) data;
	while (progress->do_progress) {
		if (tcpx_progress_wait_ok(progress)) {
			tcpx_progress_wait(progress);
		}
	}
	return NULL;
}

int tcpx_progress_init(struct tcpx_domain *domain,
		       struct tcpx_progress *progress)
{
	int ret;
	if (!progress)
		return -FI_EINVAL;

	progress->domain = domain;

	fastlock_init(&progress->signal_lock);
	ret = fd_signal_init(&progress->signal);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,"signal init failed\n");
		return ret;
	}

	ret = fi_epoll_create(&progress->epoll_set);
	if (ret < 0)
	{
                FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,"failed to create epoll set\n");
		ret = -errno;
                goto err1;
	}

	ret = fi_epoll_add(progress->epoll_set,
			   progress->signal.fd[FI_READ_FD], NULL);
	if (ret)
		goto err2;

	progress->do_progress = 1;
	if (pthread_create(&progress->progress_thread, NULL,
			   tcpx_progress_thread, (void *)progress)) {
		goto err2;
	}
	return -FI_SUCCESS;
err2:
	fi_epoll_close(progress->epoll_set);
err1:
	fastlock_destroy(&progress->signal_lock);
	fd_signal_free(&progress->signal);
	return ret;
}
