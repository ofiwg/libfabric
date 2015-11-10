/*
 * Copyright (c) 2015 Cray Inc.  All rights reserved.
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
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

#include <assert.h>

#include <stdlib.h>

#include "gnix.h"
#include "gnix_eq.h"
#include "gnix_util.h"

/*******************************************************************************
 * Forward declaration for ops structures.
 ******************************************************************************/
static struct fi_ops_eq gnix_eq_ops;
static struct fi_ops gnix_fi_eq_ops;


/*******************************************************************************
 * Helper functions.
 ******************************************************************************/
static int gnix_eq_set_wait(struct gnix_fid_eq *eq)
{
	int ret = FI_SUCCESS;

	GNIX_TRACE(FI_LOG_EQ, "\n");

	struct fi_wait_attr requested = {
		.wait_obj = eq->attr.wait_obj,
		.flags = 0
	};

	switch (eq->attr.wait_obj) {
	case FI_WAIT_FD:
	case FI_WAIT_MUTEX_COND:
		ret = gnix_wait_open(&eq->fabric->fab_fid, &requested,
				     &eq->wait);
		break;
	case FI_WAIT_SET:
		ret = _gnix_wait_set_add(eq->attr.wait_set, &eq->eq_fid.fid);

		if (ret)
			return ret;

		eq->wait = eq->attr.wait_set;
		break;
	default:
		break;
	}

	return ret;
}

static int gnix_verify_eq_attr(struct fi_eq_attr *attr)
{

	GNIX_TRACE(FI_LOG_EQ, "\n");

	if (!attr)
		return -FI_EINVAL;

	if (!attr->size)
		attr->size = GNIX_EQ_DEFAULT_SIZE;

	/*
	 * Initial implementation doesn't support any type of wait object.
	 */
	switch (attr->wait_obj) {
	case FI_WAIT_NONE:
		break;
	case FI_WAIT_SET:
		if (!attr->wait_set) {
			GNIX_WARN(FI_LOG_EQ,
				  "FI_WAIT_SET is set, but wait_set field doesn't reference a wait object.\n");
			return -FI_EINVAL;
		}
		break;
	case FI_WAIT_UNSPEC:
		attr->wait_obj = FI_WAIT_FD;
		break;
	case FI_WAIT_FD:
	case FI_WAIT_MUTEX_COND:
		break;
	default:
		GNIX_WARN(FI_LOG_EQ, "wait type: %d unsupported.\n",
			  attr->wait_obj);
		return -FI_EINVAL;
	}

	return FI_SUCCESS;
}

static void free_eq_entry(struct slist_entry *item)
{
	struct gnix_eq_entry *entry;

	entry = container_of(item, struct gnix_eq_entry, item);

	free(entry->the_entry);
	free(entry);
}

static struct slist_entry *alloc_eq_entry(size_t size)
{
	struct gnix_eq_entry *entry = calloc(1, sizeof(*entry));

	if (!entry) {
		GNIX_ERR(FI_LOG_EQ, "out of memory\n");
		goto err;
	}

	if (size) {
		entry->the_entry = malloc(size);
		if (!entry->the_entry) {
			GNIX_ERR(FI_LOG_EQ, "out of memory\n");
			goto cleanup;
		}
	}

	return &entry->item;

cleanup:
	free(entry);
err:
	return NULL;
}

/* Temporarily mark as unused to avoid build warnings. */
static ssize_t gnix_eq_write_error(struct fid_eq*, fid_t, void*, uint64_t, int,
				   int, void*, size_t) __attribute__((unused));

static ssize_t gnix_eq_write_error(struct fid_eq *eq, fid_t fid,
				   void *context, uint64_t index, int err,
				   int prov_errno, void *err_data,
				   size_t err_size)
{
	struct fi_eq_err_entry *error;
	struct gnix_eq_entry *event;
	struct gnix_fid_eq *eq_priv;
	struct slist_entry *item;

	ssize_t ret = FI_SUCCESS;

	if (!eq)
		return -FI_EINVAL;

	eq_priv = container_of(eq, struct gnix_fid_eq, eq_fid);

	fastlock_acquire(&eq_priv->lock);

	item = _gnix_queue_get_free(eq_priv->errors);
	if (!item) {
		GNIX_ERR(FI_LOG_EQ, "error creating error entry\n");
		ret = -FI_ENOMEM;
		goto err;
	}

	event = container_of(item, struct gnix_eq_entry, item);

	error = event->the_entry;

	error->fid = fid;
	error->context = context;
	error->data = index;
	error->err = err;
	error->prov_errno = prov_errno;
	error->err_data = err_data;
	error->err_data_size = err_size;

	_gnix_queue_enqueue(eq_priv->errors, &event->item);

err:
	fastlock_release(&eq_priv->lock);

	return ret;
}

static void __eq_destruct(void *obj)
{
	struct gnix_fid_eq *eq = (struct gnix_fid_eq *) obj;

	_gnix_ref_put(eq->fabric);

	fastlock_destroy(&eq->lock);

	switch (eq->attr.wait_obj) {
	case FI_WAIT_NONE:
		break;
	case FI_WAIT_SET:
		_gnix_wait_set_remove(eq->wait, &eq->eq_fid.fid);
		break;
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
	case FI_WAIT_MUTEX_COND:
		assert(eq->wait);
		gnix_wait_close(&eq->wait->fid);
		break;
	default:
		GNIX_WARN(FI_LOG_EQ, "format: %d unsupported\n.",
			  eq->attr.wait_obj);
		break;
	}

	_gnix_queue_destroy(eq->events);
	_gnix_queue_destroy(eq->errors);

	free(eq);
}

/*******************************************************************************
 * API function implementations.
 ******************************************************************************/
/*
 * - Handle FI_WRITE flag. When not included, replace write function with
 *   fi_no_eq_write.
 */
int gnix_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		 struct fid_eq **eq, void *context)
{
	struct gnix_fid_eq *eq_priv;

	int ret = FI_SUCCESS;

	GNIX_TRACE(FI_LOG_EQ, "\n");

	if (!fabric)
		return -FI_EINVAL;

	eq_priv = calloc(1, sizeof(*eq_priv));
	if (!eq_priv)
		return -FI_ENOMEM;

	ret = gnix_verify_eq_attr(attr);
	if (ret)
		goto err;

	eq_priv->fabric = container_of(fabric, struct gnix_fid_fabric,
					  fab_fid);

	_gnix_ref_init(&eq_priv->ref_cnt, 1, __eq_destruct);

	_gnix_ref_get(eq_priv->fabric);

	eq_priv->eq_fid.fid.fclass = FI_CLASS_EQ;
	eq_priv->eq_fid.fid.context = context;
	eq_priv->eq_fid.fid.ops = &gnix_fi_eq_ops;
	eq_priv->eq_fid.ops = &gnix_eq_ops;
	eq_priv->attr = *attr;

	fastlock_init(&eq_priv->lock);

	ret = gnix_eq_set_wait(eq_priv);
	if (ret)
		goto err1;

	ret = _gnix_queue_create(&eq_priv->events, alloc_eq_entry,
				 free_eq_entry, 0, eq_priv->attr.size);
	if (ret)
		goto err1;

	ret = _gnix_queue_create(&eq_priv->errors, alloc_eq_entry,
				 free_eq_entry, sizeof(struct fi_eq_err_entry),
				 0);
	if (ret)
		goto err2;

	*eq = &eq_priv->eq_fid;
	return ret;

err2:
	_gnix_queue_destroy(eq_priv->events);
err1:
	_gnix_ref_put(eq_priv->fabric);
	fastlock_destroy(&eq_priv->lock);
err:
	free(eq_priv);
	return ret;
}

static int gnix_eq_close(struct fid *fid)
{
	struct gnix_fid_eq *eq;
	int references_held;

	GNIX_TRACE(FI_LOG_EQ, "\n");

	if (!fid)
		return -FI_EINVAL;

	eq = container_of(fid, struct gnix_fid_eq, eq_fid);

	references_held = _gnix_ref_put(eq);
	if (references_held) {
		GNIX_INFO(FI_LOG_EQ, "failed to fully close eq due "
				"to lingering references. references=%i eq=%p\n",
				references_held, eq);
	}

	return FI_SUCCESS;
}

static ssize_t gnix_eq_sread(struct fid_eq *eq, uint32_t *event, void *buf,
			     size_t len, int timeout, uint64_t flags)
{
	return -FI_ENOSYS;
}

static ssize_t gnix_eq_read(struct fid_eq *eq, uint32_t *event, void *buf,
			    size_t len, uint64_t flags)
{
	struct gnix_fid_eq *eq_priv;
	struct gnix_eq_entry *entry;
	struct slist_entry *item;

	ssize_t read_size = len;

	eq_priv = container_of(eq, struct gnix_fid_eq, eq_fid);

	fastlock_acquire(&eq_priv->lock);

	item = _gnix_queue_peek(eq_priv->events);

	if (!item) {
		read_size = -FI_EAGAIN;
		goto err;
	}

	entry = container_of(item, struct gnix_eq_entry, item);

	if (read_size < entry->len) {
		read_size = -FI_ETOOSMALL;
		goto err;
	}

	*event = entry->type;

	memcpy(buf, entry->the_entry, read_size);

	if (!(flags & FI_PEEK)) {
		item = _gnix_queue_dequeue(eq_priv->events);

		free(entry->the_entry);
		entry->the_entry = NULL;

		_gnix_queue_enqueue_free(eq_priv->events, &entry->item);
	}

err:
	fastlock_release(&eq_priv->lock);

	return read_size;
}

static int gnix_eq_control(struct fid *eq, int command, void *arg)
{
	struct gnix_fid_eq *eq_priv;

	eq_priv = container_of(eq, struct gnix_fid_eq, eq_fid);

	switch (command) {
	case FI_GETWAIT:
		return _gnix_get_wait_obj(eq_priv->wait, arg);
	default:
		return -FI_EINVAL;
	}
}

static ssize_t gnix_eq_readerr(struct fid_eq *eq, struct fi_eq_err_entry *buf,
			       uint64_t flags)
{
	struct gnix_fid_eq *eq_priv;
	struct gnix_eq_entry *entry;
	struct slist_entry *item;

	ssize_t read_size = sizeof(*buf);

	eq_priv = container_of(eq, struct gnix_fid_eq, eq_fid);

	fastlock_acquire(&eq_priv->lock);

	if (flags & FI_PEEK)
		item = _gnix_queue_peek(eq_priv->errors);
	else
		item = _gnix_queue_dequeue(eq_priv->errors);

	if (!item) {
		read_size = -FI_EAGAIN;
		goto err;
	}

	entry = container_of(item, struct gnix_eq_entry, item);

	memcpy(buf, entry->the_entry, read_size);

	_gnix_queue_enqueue_free(eq_priv->errors, &entry->item);

err:
	fastlock_release(&eq_priv->lock);

	return read_size;
}

static ssize_t gnix_eq_write(struct fid_eq *eq, uint32_t event,
			     const void *buf, size_t len, uint64_t flags)
{
	struct gnix_fid_eq *eq_priv;
	struct slist_entry *item;
	struct gnix_eq_entry *entry;

	ssize_t ret = len;

	eq_priv = container_of(eq, struct gnix_fid_eq, eq_fid);

	fastlock_acquire(&eq_priv->lock);

	item = _gnix_queue_get_free(eq_priv->events);
	if (!item) {
		GNIX_ERR(FI_LOG_EQ, "error creating eq_entry\n");
		ret = -FI_ENOMEM;
		goto err;
	}

	entry = container_of(item, struct gnix_eq_entry, item);

	entry->the_entry = calloc(1, len);
	if (!entry->the_entry) {
		GNIX_ERR(FI_LOG_EQ, "error allocating buffer\n");
		ret = -FI_ENOMEM;
		goto err;
	}

	memcpy(entry->the_entry, buf, len);

	entry->len = len;
	entry->type = event;
	entry->flags = flags;

	_gnix_queue_enqueue(eq_priv->events, &entry->item);

	if (eq_priv->wait)
		_gnix_signal_wait_obj(eq_priv->wait);

err:
	fastlock_release(&eq_priv->lock);

	return ret;
}

static const char *gnix_eq_strerror(struct fid_eq *eq, int prov_errno,
				    const void *err_data, char *buf, size_t len)
{
	return NULL;
}

/*******************************************************************************
 * FI_OPS_* data structures.
 ******************************************************************************/
static struct fi_ops_eq gnix_eq_ops = {
	.size = sizeof(struct fi_ops_eq),
	.read = gnix_eq_read,
	.readerr = gnix_eq_readerr,
	.write = gnix_eq_write,
	.sread = gnix_eq_sread,
	.strerror = gnix_eq_strerror
};

static struct fi_ops gnix_fi_eq_ops = {
	.size = sizeof(struct fi_ops),
	.close = gnix_eq_close,
	.bind = fi_no_bind,
	.control = gnix_eq_control,
	.ops_open = fi_no_ops_open
};
