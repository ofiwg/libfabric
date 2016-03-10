/*
 * Copyright (c) 2014-2016 Intel Corporation, Inc.  All rights reserved.
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
#include <string.h>
#include <sys/time.h>

#include <fi_enosys.h>
#include <fi_util.h>


int util_trywait(struct fid_fabric *fabric, struct fid **fids, int count)
{
	struct util_cq *cq;
	struct util_eq *eq;
	struct util_wait *wait;
	int i, ret;

	for (i = 0; i < count; i++) {
		switch (fids[i]->fclass) {
		case FI_CLASS_CQ:
			cq = container_of(fids[i], struct util_cq, cq_fid.fid);
			wait = cq->wait;
			break;
		case FI_CLASS_EQ:
			eq = container_of(fids[i], struct util_eq, eq_fid.fid);
			wait = eq->wait;
			break;
		case FI_CLASS_CNTR:
			return -FI_ENOSYS;
		case FI_CLASS_WAIT:
			wait = container_of(fids[i], struct util_wait, wait_fid.fid);
			break;
		default:
			return -FI_EINVAL;
		}

		ret = wait->try(wait);
		if (ret)
			return ret;
	}
	return 0;
}

int fi_check_wait_attr(const struct fi_provider *prov,
		       const struct fi_wait_attr *attr)
{
	switch (attr->wait_obj) {
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
	case FI_WAIT_MUTEX_COND:
		break;
	default:
		FI_WARN(prov, FI_LOG_FABRIC, "invalid wait object type\n");
		return -FI_EINVAL;
	}

	if (attr->flags) {
		FI_WARN(prov, FI_LOG_FABRIC, "invalid flags\n");
		return -FI_EINVAL;
	}

	return 0;
}

int fi_wait_cleanup(struct util_wait *wait)
{
	int ret;

	if (atomic_get(&wait->ref))
		return -FI_EBUSY;

	ret = fi_close(&wait->pollset->poll_fid.fid);
	if (ret)
		return ret;

	atomic_dec(&wait->fabric->ref);
	return 0;
}

int fi_wait_init(struct util_fabric *fabric, struct fi_wait_attr *attr,
		 struct util_wait *wait)
{
	struct fid_poll *poll_fid;
	struct fi_poll_attr poll_attr;
	int ret;

	wait->prov = fabric->prov;
	atomic_initialize(&wait->ref, 0);
	wait->wait_fid.fid.fclass = FI_CLASS_WAIT;

	switch (attr->wait_obj) {
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
		wait->wait_obj = FI_WAIT_FD;
		break;
	case FI_WAIT_MUTEX_COND:
		wait->wait_obj = FI_WAIT_MUTEX_COND;
		break;
	default:
		assert(0);
		return -FI_EINVAL;
	}

	memset(&poll_attr, 0, sizeof poll_attr);
	ret = fi_poll_create_(fabric->prov, NULL, &poll_attr, &poll_fid);
	if (ret)
		return ret;

	wait->pollset = container_of(poll_fid, struct util_poll, poll_fid);
	wait->fabric = fabric;
	atomic_inc(&fabric->ref);
	return 0;
}

static void util_wait_fd_signal(struct util_wait *util_wait)
{
	struct util_wait_fd *wait;
	wait = container_of(util_wait, struct util_wait_fd, util_wait);
	fd_signal_set(&wait->signal);
}

static int util_wait_fd_try(struct util_wait *wait)
{
	struct util_wait_fd *wait_fd;
	void *context;
	int ret;

	wait_fd = container_of(wait, struct util_wait_fd, util_wait);
	fd_signal_reset(&wait_fd->signal);
	ret = fi_poll(&wait->pollset->poll_fid, &context, 1);
	return (ret > 0) ? -FI_EAGAIN : ret;
}

static int util_wait_fd_run(struct fid_wait *wait_fid, int timeout)
{
	struct util_wait_fd *wait;
	uint64_t start;
	int ret;

	wait = container_of(wait_fid, struct util_wait_fd, util_wait.wait_fid);
	start = (timeout >= 0) ? fi_gettime_ms() : 0;

	while (1) {
		ret = wait->util_wait.try(&wait->util_wait);
		if (ret)
			return ret == -FI_EAGAIN ? 0 : ret;

		if (timeout >= 0) {
			timeout -= (int) (fi_gettime_ms() - start);
			if (timeout <= 0)
				return -FI_ETIMEDOUT;
		}

		fi_epoll_wait(wait->epoll_fd, timeout);
	}
}

static int util_wait_fd_control(struct fid *fid, int command, void *arg)
{
	struct util_wait_fd *wait;
	int ret;

	wait = container_of(fid, struct util_wait_fd, util_wait.wait_fid.fid);
	switch (command) {
	case FI_GETWAIT:
#ifdef HAVE_EPOLL
		*(int *) arg = wait->epoll_fd;
		ret = 0;
#else
		ret = -FI_ENOSYS;
#endif
		break;
	default:
		FI_INFO(wait->util_wait.prov, FI_LOG_FABRIC,
			"unsupported command\n");
		ret = -FI_ENOSYS;
		break;
	}
	return ret;
}

static int util_wait_fd_close(struct fid *fid)
{
	struct util_wait_fd *wait;
	int ret;

	wait = container_of(fid, struct util_wait_fd, util_wait.wait_fid.fid);
	ret = fi_wait_cleanup(&wait->util_wait);
	if (ret)
		return ret;

	fi_epoll_del(wait->epoll_fd, wait->signal.fd[FI_READ_FD]);
	fd_signal_free(&wait->signal);
	fi_epoll_close(wait->epoll_fd);
	free(wait);
	return 0;
}

static struct fi_ops_wait util_wait_fd_ops = {
	.size = sizeof(struct fi_ops_wait),
	.wait = util_wait_fd_run,
};

static struct fi_ops util_wait_fd_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = util_wait_fd_close,
	.bind = fi_no_bind,
	.control = util_wait_fd_control,
	.ops_open = fi_no_ops_open,
};

static int util_verify_wait_fd_attr(const struct fi_provider *prov,
				    const struct fi_wait_attr *attr)
{
	int ret;

	ret = fi_check_wait_attr(prov, attr);
	if (ret)
		return ret;

	switch (attr->wait_obj) {
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
		break;
	default:
		FI_WARN(prov, FI_LOG_FABRIC, "unsupported wait object\n");
		return -FI_EINVAL;
	}

	return 0;
}

int fi_wait_fd_open(struct fid_fabric *fabric_fid, struct fi_wait_attr *attr,
		    struct fid_wait **waitset)
{
	struct util_fabric *fabric;
	struct util_wait_fd *wait;
	int ret;

	fabric = container_of(fabric_fid, struct util_fabric, fabric_fid);
	ret = util_verify_wait_fd_attr(fabric->prov, attr);
	if (ret)
		return ret;

	wait = calloc(1, sizeof(*wait));
	if (!wait)
		return -FI_ENOMEM;

	ret = fi_wait_init(fabric, attr, &wait->util_wait);
	if (ret)
		goto err1;

	wait->util_wait.signal = util_wait_fd_signal;
	wait->util_wait.try = util_wait_fd_try;
	ret = fd_signal_init(&wait->signal);
	if (ret)
		goto err2;

	ret = fi_epoll_create(&wait->epoll_fd);
	if (ret)
		goto err3;

	ret = fi_epoll_add(wait->epoll_fd, wait->signal.fd[FI_READ_FD],
			   &wait->util_wait.wait_fid.fid);
	if (ret)
		goto err4;

	wait->util_wait.wait_fid.fid.ops = &util_wait_fd_fi_ops;
	wait->util_wait.wait_fid.ops = &util_wait_fd_ops;

	*waitset = &wait->util_wait.wait_fid;
	return 0;

err4:
	fi_epoll_close(wait->epoll_fd);
err3:
	fd_signal_free(&wait->signal);
err2:
	fi_wait_cleanup(&wait->util_wait);
err1:
	free(wait);
	return ret;
}
