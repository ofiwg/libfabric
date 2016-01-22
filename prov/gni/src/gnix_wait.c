/*
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
 * Copyright (c) 2015 Cray Inc. All rights reserved.
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

#include "gnix.h"
#include "gnix_wait.h"

/*******************************************************************************
 * Forward declarations for FI_OPS_* structures.
 ******************************************************************************/
static struct fi_ops gnix_fi_ops;
static struct fi_ops_wait gnix_wait_ops;

/*******************************************************************************
 * List match functions.
 ******************************************************************************/
static int gnix_match_fid(struct slist_entry *item, const void *fid)
{
	struct gnix_wait_entry *entry;

	entry = container_of(item, struct gnix_wait_entry, entry);

	return (entry->wait_obj == (struct fid *) fid);
}

/*******************************************************************************
 * Exposed helper functions.
 ******************************************************************************/
int _gnix_wait_set_add(struct fid_wait *wait, struct fid *wait_obj)
{
	struct gnix_fid_wait *wait_priv;
	struct gnix_wait_entry *wait_entry;

	GNIX_TRACE(WAIT_SUB, "\n");

	wait_entry = calloc(1, sizeof(*wait_entry));
	if (!wait_entry) {
		GNIX_WARN(WAIT_SUB,
			  "failed to allocate memory for wait entry.\n");
		return -FI_ENOMEM;
	}

	wait_priv = container_of(wait, struct gnix_fid_wait, wait.fid);

	wait_entry->wait_obj = wait_obj;

	gnix_slist_insert_tail(&wait_entry->entry, &wait_priv->set);

	return FI_SUCCESS;
}

int _gnix_wait_set_remove(struct fid_wait *wait, struct fid *wait_obj)
{
	struct gnix_fid_wait *wait_priv;
	struct gnix_wait_entry *wait_entry;
	struct slist_entry *found;

	GNIX_TRACE(WAIT_SUB, "\n");

	wait_priv = container_of(wait, struct gnix_fid_wait, wait.fid);

	found = slist_remove_first_match(&wait_priv->set, gnix_match_fid,
					 wait_obj);

	if (found) {
		wait_entry = container_of(found, struct gnix_wait_entry,
					  entry);
		free(wait_entry);

		return FI_SUCCESS;
	}

	return -FI_EINVAL;
}

int _gnix_get_wait_obj(struct fid_wait *wait, void *arg)
{
	struct fi_mutex_cond mutex_cond;
	struct gnix_fid_wait *wait_priv;
	size_t copy_size;
	const void *src;

	GNIX_TRACE(WAIT_SUB, "\n");

	if (!wait || !arg)
		return -FI_EINVAL;

	wait_priv = container_of(wait, struct gnix_fid_wait, wait);

	switch (wait_priv->type) {
	case FI_WAIT_FD:
		copy_size = sizeof(wait_priv->fd[WAIT_READ]);
		src = &wait_priv->fd[WAIT_READ];
		break;
	case FI_WAIT_MUTEX_COND:
		mutex_cond.mutex = &wait_priv->mutex;
		mutex_cond.cond = &wait_priv->cond;

		copy_size = sizeof(mutex_cond);
		src = &mutex_cond;
		break;
	default:
		GNIX_WARN(WAIT_SUB, "wait type: %d not supported.\n",
			  wait_priv->type);
		return -FI_EINVAL;
	}

	memcpy(arg, src, copy_size);

	return FI_SUCCESS;
}

void _gnix_signal_wait_obj(struct fid_wait *wait)
{
	static char msg = 'g';
	size_t len = sizeof(msg);
	struct gnix_fid_wait *wait_priv;

	wait_priv = container_of(wait, struct gnix_fid_wait, wait);

	switch (wait_priv->type) {
	case FI_WAIT_FD:
		if (write(wait_priv->fd[WAIT_WRITE], &msg, len) != len)
			GNIX_WARN(WAIT_SUB, "failed to signal wait object.\n");
		break;
	case FI_WAIT_MUTEX_COND:
		pthread_cond_signal(&wait_priv->cond);
		break;
	default:
		GNIX_WARN(WAIT_SUB,
			 "error signaling wait object: type: %d not supported.\n",
			 wait_priv->type);
		return;
	}
}

/*******************************************************************************
 * Internal helper functions.
 ******************************************************************************/
static int gnix_verify_wait_attr(struct fi_wait_attr *attr)
{
	GNIX_TRACE(WAIT_SUB, "\n");

	if (!attr || attr->flags)
		return -FI_EINVAL;

	switch (attr->wait_obj) {
	case FI_WAIT_UNSPEC:
		attr->wait_obj = FI_WAIT_FD;
	case FI_WAIT_FD:
	case FI_WAIT_MUTEX_COND:
		break;
	default:
		GNIX_WARN(WAIT_SUB, "wait type: %d not supported.\n",
			  attr->wait_obj);
		return -FI_EINVAL;
	}

	return FI_SUCCESS;
}

static int gnix_init_wait_obj(struct gnix_fid_wait *wait, enum fi_wait_obj type)
{
	long flags = 0;

	GNIX_TRACE(WAIT_SUB, "\n");

	wait->type = type;

	switch (type) {
	case FI_WAIT_FD:
		if (socketpair(AF_LOCAL, SOCK_STREAM, 0, wait->fd))
			goto err;

		fcntl(wait->fd[WAIT_READ], F_GETFL, &flags);
		if (fcntl(wait->fd[WAIT_READ], F_SETFL, flags | O_NONBLOCK))
			goto cleanup;
		break;
	case FI_WAIT_MUTEX_COND:
		pthread_mutex_init(&wait->mutex, NULL);
		pthread_cond_init(&wait->cond, NULL);
		break;
	default:
		GNIX_WARN(WAIT_SUB, "Invalid wait type: %d\n",
			 type);
		return -FI_EINVAL;
	}

	return FI_SUCCESS;

cleanup:
	close(wait->fd[WAIT_READ]);
	close(wait->fd[WAIT_WRITE]);
err:
	GNIX_WARN(WAIT_SUB, "%s\n", strerror(errno));
	return -FI_EOTHER;
}

/*******************************************************************************
 * API Functionality.
 ******************************************************************************/
static int gnix_wait_control(struct fid *wait, int command, void *arg)
{
	struct fid_wait *wait_fid_priv;

	GNIX_TRACE(WAIT_SUB, "\n");

	wait_fid_priv = container_of(wait, struct fid_wait, fid);

	switch (command) {
	case FI_GETWAIT:
		return _gnix_get_wait_obj(wait_fid_priv, arg);
	default:
		return -FI_EINVAL;
	}
}

int gnix_wait_wait(struct fid_wait *wait, int timeout)
{
	return -FI_ENOSYS;
}

int gnix_wait_close(struct fid *wait)
{
	struct gnix_fid_wait *wait_priv;

	GNIX_TRACE(WAIT_SUB, "\n");

	wait_priv = container_of(wait, struct gnix_fid_wait, wait.fid);

	if (!slist_empty(&wait_priv->set)) {
		GNIX_WARN(WAIT_SUB,
			  "resources still connected to wait set.\n");
		return -FI_EBUSY;
	}

	if (wait_priv->type == FI_WAIT_FD) {
		close(wait_priv->fd[WAIT_READ]);
		close(wait_priv->fd[WAIT_WRITE]);
	}

	_gnix_ref_put(wait_priv->fabric);

	free(wait_priv);

	return FI_SUCCESS;
}

int gnix_wait_open(struct fid_fabric *fabric, struct fi_wait_attr *attr,
		   struct fid_wait **waitset)
{
	struct gnix_fid_fabric *fab_priv;
	struct gnix_fid_wait *wait_priv;
	int ret = FI_SUCCESS;

	GNIX_TRACE(WAIT_SUB, "\n");

	ret = gnix_verify_wait_attr(attr);
	if (ret)
		goto err;

	fab_priv = container_of(fabric, struct gnix_fid_fabric, fab_fid);

	wait_priv = calloc(1, sizeof(*wait_priv));
	if (!wait_priv) {
		GNIX_WARN(WAIT_SUB,
			 "failed to allocate memory for wait set.\n");
		ret = -FI_ENOMEM;
		goto err;
	}

	ret = gnix_init_wait_obj(wait_priv, attr->wait_obj);
	if (ret)
		goto cleanup;

	slist_init(&wait_priv->set);

	wait_priv->wait.fid.fclass = FI_CLASS_WAIT;
	wait_priv->wait.fid.ops = &gnix_fi_ops;
	wait_priv->wait.ops = &gnix_wait_ops;

	wait_priv->fabric = fab_priv;

	_gnix_ref_get(fab_priv);
	*waitset = &wait_priv->wait;

	return ret;

cleanup:
	free(wait_priv);
err:
	return ret;
}

/*******************************************************************************
 * FI_OPS_* data structures.
 ******************************************************************************/
static struct fi_ops gnix_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = gnix_wait_close,
	.bind = fi_no_bind,
	.control = gnix_wait_control,
	.ops_open = fi_no_ops_open
};

static struct fi_ops_wait gnix_wait_ops = {
	.size = sizeof(struct fi_ops_wait),
	.wait = gnix_wait_wait
};
