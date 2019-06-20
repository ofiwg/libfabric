/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>

#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_CORE, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_CORE, __VA_ARGS__)

enum {
	WAIT_READ_FD = 0,
	WAIT_WRITE_FD,
};

int cxip_wait_get_obj(struct fid_wait *fid, void *arg)
{
	struct fi_mutex_cond mut_cond;
	struct cxip_wait *wait;

	wait = container_of(fid, struct cxip_wait, wait_fid.fid);

	switch (wait->type) {
	case FI_WAIT_FD:
		memcpy(arg, &wait->wobj.fd[WAIT_READ_FD], sizeof(int));
		break;

	case FI_WAIT_MUTEX_COND:
		mut_cond.mutex = &wait->wobj.mutex_cond.mutex;
		mut_cond.cond = &wait->wobj.mutex_cond.cond;
		memcpy(arg, &mut_cond, sizeof(mut_cond));
		break;
	default:
		CXIP_LOG_ERROR("Invalid wait obj type\n");
		return -FI_EINVAL;
	}

	return 0;
}

static int cxip_wait_init(struct cxip_wait *wait, enum fi_wait_obj type)
{
	int ret;

	wait->type = type;

	switch (type) {
	case FI_WAIT_FD:
		if (socketpair(AF_UNIX, SOCK_STREAM, 0, wait->wobj.fd))
			return -ofi_sockerr();

		ret = fi_fd_nonblock(wait->wobj.fd[WAIT_READ_FD]);
		if (ret) {
			CXIP_LOG_ERROR("fi_fd_nonblock failed, errno: %d\n",
				       ret);
			ofi_close_socket(wait->wobj.fd[WAIT_READ_FD]);
			ofi_close_socket(wait->wobj.fd[WAIT_WRITE_FD]);
			return ret;
		}
		break;

	case FI_WAIT_MUTEX_COND:
		pthread_mutex_init(&wait->wobj.mutex_cond.mutex, NULL);
		pthread_cond_init(&wait->wobj.mutex_cond.cond, NULL);
		break;

	default:
		CXIP_LOG_ERROR("Invalid wait object type\n");
		return -FI_EINVAL;
	}
	return 0;
}

static int cxip_wait_wait(struct fid_wait *wait_fid, int timeout)
{
	struct cxip_cq *cq;
	struct cxip_wait *wait;
	uint64_t start_ms = 0, end_ms = 0;
	struct dlist_entry *p, *head;
	struct cxip_fid_list *list_item;
	int err = 0;
	ssize_t ret;
	char c;

	wait = container_of(wait_fid, struct cxip_wait, wait_fid);
	if (timeout > 0)
		start_ms = fi_gettime_ms();

	head = &wait->fid_list;
	for (p = head->next; p != head; p = p->next) {
		list_item = container_of(p, struct cxip_fid_list, entry);
		switch (list_item->fid->fclass) {
		case FI_CLASS_CQ:
			cq = container_of(list_item->fid, struct cxip_cq,
					  util_cq.cq_fid);
			cxip_cq_progress(cq);
			break;
		}
	}
	if (timeout > 0) {
		end_ms = fi_gettime_ms();
		timeout -= (int)(end_ms - start_ms);
		timeout = timeout < 0 ? 0 : timeout;
	}

	switch (wait->type) {
	case FI_WAIT_FD:
		err = fi_poll_fd(wait->wobj.fd[WAIT_READ_FD], timeout);
		if (err == 0) {
			err = -FI_ETIMEDOUT;
		} else {
			while (err > 0) {
				ret = ofi_read_socket(
					wait->wobj.fd[WAIT_READ_FD], &c, 1);
				if (ret != 1) {
					CXIP_LOG_ERROR(
						"failed to read wait_fd\n");
					err = 0;
					break;
				} else
					err--;
			}
		}
		break;

	case FI_WAIT_MUTEX_COND:
		err = fi_wait_cond(&wait->wobj.mutex_cond.cond,
				   &wait->wobj.mutex_cond.mutex, timeout);
		break;

	default:
		CXIP_LOG_ERROR("Invalid wait object type\n");
		return -FI_EINVAL;
	}
	return err;
}

void cxip_wait_signal(struct fid_wait *wait_fid)
{
	struct cxip_wait *wait;
	static char c = 'a';
	ssize_t ret;

	wait = container_of(wait_fid, struct cxip_wait, wait_fid);

	switch (wait->type) {
	case FI_WAIT_FD:
		ret = ofi_write_socket(wait->wobj.fd[WAIT_WRITE_FD], &c, 1);
		if (ret != 1)
			CXIP_LOG_ERROR("failed to signal\n");
		break;

	case FI_WAIT_MUTEX_COND:
		pthread_cond_signal(&wait->wobj.mutex_cond.cond);
		break;
	default:
		CXIP_LOG_ERROR("Invalid wait object type\n");
		return;
	}
}

static struct fi_ops_wait cxip_wait_ops = {
	.size = sizeof(struct fi_ops_wait),
	.wait = cxip_wait_wait,
};

static int cxip_wait_control(struct fid *fid, int command, void *arg)
{
	struct cxip_wait *wait;
	int ret = 0;

	wait = container_of(fid, struct cxip_wait, wait_fid.fid);
	switch (command) {
	case FI_GETWAIT:
		ret = cxip_wait_get_obj(&wait->wait_fid, arg);
		break;
	default:
		ret = -FI_EINVAL;
		break;
	}
	return ret;
}

int cxip_wait_close(fid_t fid)
{
	struct cxip_fid_list *list_item;
	struct dlist_entry *p, *head;
	struct cxip_wait *wait;

	wait = container_of(fid, struct cxip_wait, wait_fid.fid);
	head = &wait->fid_list;

	for (p = head->next; p != head;) {
		list_item = container_of(p, struct cxip_fid_list, entry);
		p = p->next;
		free(list_item);
	}

	if (wait->type == FI_WAIT_FD) {
		ofi_close_socket(wait->wobj.fd[WAIT_READ_FD]);
		ofi_close_socket(wait->wobj.fd[WAIT_WRITE_FD]);
	}

	ofi_atomic_dec32(&wait->fab->ref);
	free(wait);
	return 0;
}

static struct fi_ops cxip_wait_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_wait_close,
	.bind = fi_no_bind,
	.control = cxip_wait_control,
	.ops_open = fi_no_ops_open,
};

static int cxip_verify_wait_attr(struct fi_wait_attr *attr)
{
	switch (attr->wait_obj) {
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
	case FI_WAIT_MUTEX_COND:
		break;

	default:
		CXIP_LOG_ERROR("Invalid wait object type\n");
		return -FI_EINVAL;
	}
	if (attr->flags)
		return -FI_EINVAL;
	return 0;
}

int cxip_wait_open(struct fid_fabric *fabric, struct fi_wait_attr *attr,
		   struct fid_wait **waitset)
{
	int err;
	struct cxip_wait *wait;
	struct cxip_fabric *fab;
	enum fi_wait_obj wait_obj_type;

	if (attr && cxip_verify_wait_attr(attr))
		return -FI_EINVAL;

	fab = container_of(fabric, struct cxip_fabric, util_fabric.fabric_fid);
	if (!attr || attr->wait_obj == FI_WAIT_UNSPEC)
		wait_obj_type = FI_WAIT_FD;
	else
		wait_obj_type = attr->wait_obj;

	wait = calloc(1, sizeof(*wait));
	if (!wait)
		return -FI_ENOMEM;

	err = cxip_wait_init(wait, wait_obj_type);
	if (err) {
		free(wait);
		return err;
	}

	wait->wait_fid.fid.fclass = FI_CLASS_WAIT;
	wait->wait_fid.fid.context = 0;
	wait->wait_fid.fid.ops = &cxip_wait_fi_ops;
	wait->wait_fid.ops = &cxip_wait_ops;
	wait->fab = fab;
	wait->type = wait_obj_type;
	ofi_atomic_inc32(&fab->ref);
	dlist_init(&wait->fid_list);

	*waitset = &wait->wait_fid;
	return 0;
}
