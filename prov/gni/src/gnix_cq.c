/*
 * Copyright (c) 2015 Cray Inc. All rights reserved.
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

/*
 * CQ common code
 */
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "gnix.h"
#include "gnix_cq.h"
#include "gnix_nic.h"

/*******************************************************************************
 * Function pointer for filling specific entry format type.
 ******************************************************************************/
typedef void (*fill_entry)(void *cq_entry, void *op_context, uint64_t flags,
			   size_t len, void *buf, uint64_t data, uint64_t tag);

/*******************************************************************************
 * Forward declarations for filling functions.
 ******************************************************************************/
static void fill_cq_entry(void *cq_entry, void *op_context, uint64_t flags,
			  size_t len, void *buf, uint64_t data, uint64_t tag);
static void fill_cq_msg(void *cq_entry, void *op_context, uint64_t flags,
			size_t len, void *buf, uint64_t data, uint64_t tag);
static void fill_cq_data(void *cq_entry, void *op_context, uint64_t flags,
			size_t len, void *buf, uint64_t data, uint64_t tag);
static void fill_cq_tagged(void *cq_entry, void *op_context, uint64_t flags,
			size_t len, void *buf, uint64_t data, uint64_t tag);

/*******************************************************************************
 * Forward declarations for ops structures.
 ******************************************************************************/
static const struct fi_ops gnix_cq_fi_ops;
static const struct fi_ops_cq gnix_cq_ops;

/*******************************************************************************
 * Size array corresponding format type to format size.
 ******************************************************************************/
static const size_t const format_sizes[] = {
	[FI_CQ_FORMAT_UNSPEC]  = sizeof(GNIX_CQ_DEFAULT_FORMAT),
	[FI_CQ_FORMAT_CONTEXT] = sizeof(struct fi_cq_entry),
	[FI_CQ_FORMAT_MSG]     = sizeof(struct fi_cq_msg_entry),
	[FI_CQ_FORMAT_DATA]    = sizeof(struct fi_cq_data_entry),
	[FI_CQ_FORMAT_TAGGED]  = sizeof(struct fi_cq_tagged_entry)
};

static const fill_entry const fill_function[] = {
	[FI_CQ_FORMAT_UNSPEC]  = fill_cq_entry,
	[FI_CQ_FORMAT_CONTEXT] = fill_cq_entry,
	[FI_CQ_FORMAT_MSG]     = fill_cq_msg,
	[FI_CQ_FORMAT_DATA]    = fill_cq_data,
	[FI_CQ_FORMAT_TAGGED]  = fill_cq_tagged
};

/*******************************************************************************
 * Internal helper functions
 ******************************************************************************/
static void fill_cq_entry(void *cq_entry, void *op_context, uint64_t flags,
			  size_t len, void *buf, uint64_t data, uint64_t tag)
{
	struct fi_cq_entry *entry = cq_entry;

	entry->op_context = op_context;
}

static void fill_cq_msg(void *cq_entry, void *op_context, uint64_t flags,
			size_t len, void *buf, uint64_t data, uint64_t tag)
{
	struct fi_cq_msg_entry *entry = cq_entry;

	entry->op_context = op_context;
	entry->flags = flags;
	entry->len = len;
}

static void fill_cq_data(void *cq_entry, void *op_context, uint64_t flags,
			 size_t len, void *buf, uint64_t data, uint64_t tag)
{
	struct fi_cq_data_entry *entry = cq_entry;

	entry->op_context = op_context;
	entry->flags = flags;
	entry->len = len;
	entry->buf = buf;
	entry->data = data;
}

static void fill_cq_tagged(void *cq_entry, void *op_context, uint64_t flags,
			   size_t len, void *buf, uint64_t data, uint64_t tag)
{
	struct fi_cq_tagged_entry *entry = cq_entry;

	entry->op_context = op_context;
	entry->flags = flags;
	entry->buf = buf;
	entry->data = data;
	entry->tag = tag;
	entry->len = len;
}

static int verify_cq_attr(struct fi_cq_attr *attr, struct fi_ops_cq *ops,
			  struct fi_ops *fi_cq_ops)
{
	GNIX_TRACE(FI_LOG_CQ, "\n");

	if (!attr || !ops || !fi_cq_ops)
		return -FI_EINVAL;

	if (!attr->size)
		attr->size = GNIX_CQ_DEFAULT_SIZE;

	switch (attr->format) {
	case FI_CQ_FORMAT_UNSPEC:
		attr->format = FI_CQ_FORMAT_CONTEXT;
	case FI_CQ_FORMAT_CONTEXT:
	case FI_CQ_FORMAT_MSG:
	case FI_CQ_FORMAT_DATA:
	case FI_CQ_FORMAT_TAGGED:
		break;
	default:
		GNIX_WARN(FI_LOG_CQ, "format: %d unsupported\n.",
			  attr->format);
		return -FI_EINVAL;
	}

	switch (attr->wait_obj) {
	case FI_WAIT_NONE:
		ops->sread = fi_no_cq_sread;
		ops->signal = fi_no_cq_signal;
		ops->sreadfrom = fi_no_cq_sreadfrom;
		fi_cq_ops->control = fi_no_control;
		break;
	case FI_WAIT_SET:
		if (!attr->wait_set) {
			GNIX_WARN(FI_LOG_CQ,
				  "FI_WAIT_SET is set, but wait_set field doesn't reference a wait object.\n");
			return -FI_EINVAL;
		}
		break;
	case FI_WAIT_UNSPEC:
		attr->wait_obj = FI_WAIT_FD;
	case FI_WAIT_FD:
	case FI_WAIT_MUTEX_COND:
		break;
	default:
		GNIX_WARN(FI_LOG_CQ, "wait type: %d unsupported.\n",
			  attr->wait_obj);
		return -FI_EINVAL;
	}

	return FI_SUCCESS;
}

static int gnix_cq_set_wait(struct gnix_fid_cq *cq)
{
	int ret = FI_SUCCESS;

	GNIX_TRACE(FI_LOG_CQ, "\n");

	struct fi_wait_attr requested = {
		.wait_obj = cq->attr.wait_obj,
		.flags = 0
	};

	switch (cq->attr.wait_obj) {
	case FI_WAIT_FD:
	case FI_WAIT_MUTEX_COND:
		ret = gnix_wait_open(&cq->domain->fabric->fab_fid,
				&requested, &cq->wait);
		break;
	case FI_WAIT_SET:
		cq->wait = cq->attr.wait_set;
		ret = _gnix_wait_set_add(cq->wait, &cq->cq_fid.fid);

		if (!ret)
			cq->wait = cq->attr.wait_set;

		break;
	default:
		break;
	}

	return ret;
}

static void free_cq_entry(struct slist_entry *item)
{
	struct gnix_cq_entry *entry;

	entry = container_of(item, struct gnix_cq_entry, item);

	free(entry->the_entry);
	free(entry);
}

static struct slist_entry *alloc_cq_entry(size_t size)
{
	struct gnix_cq_entry *entry = malloc(sizeof(*entry));

	if (!entry) {
		GNIX_WARN(FI_LOG_CQ, "out of memory\n");
		goto err;
	}

	entry->the_entry = malloc(size);
	if (!entry->the_entry) {
		GNIX_WARN(FI_LOG_CQ, "out of memory\n");
		goto cleanup;
	}

	return &entry->item;

cleanup:
	free(entry);
err:
	return NULL;
}

extern int _gnix_cm_nic_progress(struct gnix_cm_nic *cm_nic);

static int __gnix_cq_progress(struct gnix_fid_cq *cq)
{
	struct gnix_cq_poll_nic *pnic, *tmp;
	int rc;

	rwlock_rdlock(&cq->nic_lock);

	dlist_for_each_safe(&cq->poll_nics, pnic, tmp, list) {
		rc = _gnix_nic_progress(pnic->nic);
		if (rc) {
			GNIX_WARN(FI_LOG_CQ,
				  "_gnix_nic_progress failed: %d\n", rc);
		}
	}

	rwlock_unlock(&cq->nic_lock);

	if (unlikely(cq->domain->control_progress != FI_PROGRESS_AUTO)) {
		if (cq->domain->cm_nic != NULL) {
			rc = _gnix_cm_nic_progress(cq->domain->cm_nic);
			if (rc)
				GNIX_WARN(FI_LOG_CQ,
				  "_gnix_cm_nic_progress returned: %d\n", rc);
		}
	}

	return FI_SUCCESS;
}


/*******************************************************************************
 * Exposed helper functions
 ******************************************************************************/
ssize_t _gnix_cq_add_event(struct gnix_fid_cq *cq, void *op_context,
			   uint64_t flags, size_t len, void *buf,
			   uint64_t data, uint64_t tag, fi_addr_t src_addr)
{
	struct gnix_cq_entry *event;
	struct slist_entry *item;

	fastlock_acquire(&cq->lock);

	item = _gnix_queue_get_free(cq->events);
	if (!item) {
		GNIX_WARN(FI_LOG_CQ, "error creating cq_entry\n");
		return -FI_ENOMEM;
	}

	event = container_of(item, struct gnix_cq_entry, item);

	assert(event->the_entry);

	fill_function[cq->attr.format](event->the_entry, op_context, flags,
			len, buf, data, tag);
	event->src_addr = src_addr;

	_gnix_queue_enqueue(cq->events, &event->item);
	GNIX_INFO(FI_LOG_CQ, "Added event: %lx\n", op_context);

	if (cq->wait)
		_gnix_signal_wait_obj(cq->wait);

	fastlock_release(&cq->lock);

	return FI_SUCCESS;
}

ssize_t _gnix_cq_add_error(struct gnix_fid_cq *cq, void *op_context,
			   uint64_t flags, size_t len, void *buf,
			   uint64_t data, uint64_t tag, size_t olen,
			   int err, int prov_errno, void *err_data)
{
	struct fi_cq_err_entry *error;
	struct gnix_cq_entry *event;
	struct slist_entry *item;

	ssize_t ret = FI_SUCCESS;

	GNIX_INFO(FI_LOG_CQ, "creating error event entry\n");


	fastlock_acquire(&cq->lock);

	item = _gnix_queue_get_free(cq->errors);
	if (!item) {
		GNIX_WARN(FI_LOG_CQ, "error creating error entry\n");
		ret = -FI_ENOMEM;
		goto err;
	}

	event = container_of(item, struct gnix_cq_entry, item);

	error = event->the_entry;

	error->op_context = op_context;
	error->flags = flags;
	error->len = len;
	error->buf = buf;
	error->data = data;
	error->tag = tag;
	error->olen = olen;
	error->err = err;
	error->prov_errno = prov_errno;
	error->err_data = err_data;

	_gnix_queue_enqueue(cq->errors, &event->item);

err:
	fastlock_release(&cq->lock);

	return ret;
}

int _gnix_cq_poll_nic_add(struct gnix_fid_cq *cq, struct gnix_nic *nic)
{
	struct gnix_cq_poll_nic *pnic, *tmp;

	rwlock_wrlock(&cq->nic_lock);

	dlist_for_each_safe(&cq->poll_nics, pnic, tmp, list) {
		if (pnic->nic == nic) {
			pnic->ref_cnt++;
			rwlock_unlock(&cq->nic_lock);
			return FI_SUCCESS;
		}
	}

	pnic = malloc(sizeof(struct gnix_cq_poll_nic));
	if (!pnic) {
		GNIX_WARN(FI_LOG_CQ, "Failed to add NIC to CQ poll list.\n");
		rwlock_unlock(&cq->nic_lock);
		return -FI_ENOMEM;
	}

	/* EP holds a ref count on the NIC */
	pnic->nic = nic;
	pnic->ref_cnt = 1;
	dlist_init(&pnic->list);
	dlist_insert_tail(&pnic->list, &cq->poll_nics);

	rwlock_unlock(&cq->nic_lock);

	GNIX_INFO(FI_LOG_CQ, "Added NIC(%p) to CQ(%p) poll list\n",
		  nic, cq);

	return FI_SUCCESS;
}

int _gnix_cq_poll_nic_rem(struct gnix_fid_cq *cq, struct gnix_nic *nic)
{
	struct gnix_cq_poll_nic *pnic, *tmp;

	rwlock_wrlock(&cq->nic_lock);

	dlist_for_each_safe(&cq->poll_nics, pnic, tmp, list) {
		if (pnic->nic == nic) {
			if (!--pnic->ref_cnt) {
				dlist_remove(&pnic->list);
				free(pnic);
				GNIX_INFO(FI_LOG_CQ,
					  "Removed NIC(%p) from CQ(%p) poll list\n",
					  nic, cq);
			}
			rwlock_unlock(&cq->nic_lock);
			return FI_SUCCESS;
		}
	}

	rwlock_unlock(&cq->nic_lock);

	GNIX_WARN(FI_LOG_CQ, "NIC not found on CQ poll list.\n");
	return -FI_EINVAL;
}

static void __cq_destruct(void *obj)
{
	struct gnix_fid_cq *cq = (struct gnix_fid_cq *) obj;

	_gnix_ref_put(cq->domain);

	switch (cq->attr.wait_obj) {
	case FI_WAIT_NONE:
		break;
	case FI_WAIT_SET:
		_gnix_wait_set_remove(cq->wait, &cq->cq_fid.fid);
		break;
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
	case FI_WAIT_MUTEX_COND:
		assert(cq->wait);
		gnix_wait_close(&cq->wait->fid);
		break;
	default:
		GNIX_WARN(FI_LOG_CQ, "format: %d unsupported\n.",
			  cq->attr.wait_obj);
		break;
	}

	_gnix_queue_destroy(cq->events);
	_gnix_queue_destroy(cq->errors);

	fastlock_destroy(&cq->lock);
	free(cq->cq_fid.ops);
	free(cq->cq_fid.fid.ops);
	free(cq);
}

/*******************************************************************************
 * API functions.
 ******************************************************************************/
static int gnix_cq_close(fid_t fid)
{
	struct gnix_fid_cq *cq;
	int references_held;

	GNIX_TRACE(FI_LOG_CQ, "\n");

	cq = container_of(fid, struct gnix_fid_cq, cq_fid);

	references_held = _gnix_ref_put(cq);

	if (references_held) {
		GNIX_INFO(FI_LOG_CQ, "failed to fully close cq due to lingering "
				"references. references=%i cq=%p\n",
				references_held, cq);
	}

	return FI_SUCCESS;
}

static ssize_t gnix_cq_readfrom(struct fid_cq *cq, void *buf, size_t count,
				fi_addr_t *src_addr)
{
	struct gnix_fid_cq *cq_priv;
	struct gnix_cq_entry *event;
	struct slist_entry *temp;

	ssize_t read_count = 0;

	if (!cq || !buf || !count)
		return -FI_EINVAL;

	cq_priv = container_of(cq, struct gnix_fid_cq, cq_fid);

	__gnix_cq_progress(cq_priv);

	if (_gnix_queue_peek(cq_priv->errors))
		return -FI_EAVAIL;

	assert(buf);

	fastlock_acquire(&cq_priv->lock);

	while (_gnix_queue_peek(cq_priv->events) && count--) {
		temp = _gnix_queue_dequeue(cq_priv->events);
		event = container_of(temp, struct gnix_cq_entry, item);

		assert(event->the_entry);
		memcpy(buf, event->the_entry, cq_priv->entry_size);
		if (src_addr)
			memcpy(src_addr, &event->src_addr, sizeof(fi_addr_t));

		_gnix_queue_enqueue_free(cq_priv->events, &event->item);

		buf += cq_priv->entry_size;

		read_count++;
	}

	fastlock_release(&cq_priv->lock);

	return read_count ?: -FI_EAGAIN;
}

static ssize_t gnix_cq_read(struct fid_cq *cq, void *buf, size_t count)
{
	return gnix_cq_readfrom(cq, buf, count, NULL);
}

static ssize_t gnix_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *buf,
			       uint64_t flags)
{
	struct gnix_fid_cq *cq_priv;
	struct gnix_cq_entry *event;
	struct slist_entry *entry;

	ssize_t read_count = 0;

	if (!cq || !buf)
		return -FI_EINVAL;

	cq_priv = container_of(cq, struct gnix_fid_cq, cq_fid);

	fastlock_acquire(&cq_priv->lock);

	entry = _gnix_queue_dequeue(cq_priv->errors);
	if (!entry) {
		read_count = -FI_EAGAIN;
		goto err;
	}

	event = container_of(entry, struct gnix_cq_entry, item);

	memcpy(buf, event->the_entry, sizeof(struct fi_cq_err_entry));

	_gnix_queue_enqueue_free(cq_priv->errors, &event->item);

	read_count++;

err:
	fastlock_release(&cq_priv->lock);

	return read_count;
}

static const char *gnix_cq_strerror(struct fid_cq *cq, int prov_errno,
				    const void *prov_data, char *buf,
				    size_t len)
{
	return NULL;
}

static int gnix_cq_signal(struct fid_cq *cq)
{
	struct gnix_fid_cq *cq_priv;

	cq_priv = container_of(cq, struct gnix_fid_cq, cq_fid);

	if (cq_priv->wait)
		_gnix_signal_wait_obj(cq_priv->wait);

	return FI_SUCCESS;
}

static int gnix_cq_control(struct fid *cq, int command, void *arg)
{
	struct gnix_fid_cq *cq_priv;

	cq_priv = container_of(cq, struct gnix_fid_cq, cq_fid);

	switch (command) {
	case FI_GETWAIT:
		return _gnix_get_wait_obj(cq_priv->wait, arg);
	default:
		return -FI_EINVAL;
	}
}


int gnix_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq, void *context)
{
	struct gnix_fid_domain *domain_priv;
	struct gnix_fid_cq *cq_priv;
	struct fi_ops_cq *cq_ops;
	struct fi_ops *fi_cq_ops;

	int ret = FI_SUCCESS;

	GNIX_TRACE(FI_LOG_CQ, "\n");

	cq_ops = calloc(1, sizeof(*cq_ops));
	if (!cq_ops) {
		ret = -FI_ENOMEM;
		goto err;
	}

	fi_cq_ops = calloc(1, sizeof(*fi_cq_ops));
	if (!fi_cq_ops) {
		ret = -FI_ENOMEM;
		goto err1;
	}

	*cq_ops = gnix_cq_ops;
	*fi_cq_ops = gnix_cq_fi_ops;

	ret = verify_cq_attr(attr, cq_ops, fi_cq_ops);
	if (ret)
		goto err2;

	domain_priv = container_of(domain, struct gnix_fid_domain, domain_fid);
	if (!domain_priv) {
		ret = -FI_EINVAL;
		goto err2;
	}

	cq_priv = calloc(1, sizeof(*cq_priv));
	if (!cq_priv) {
		ret = -FI_ENOMEM;
		goto err2;
	}

	cq_priv->domain = domain_priv;
	cq_priv->attr = *attr;

	_gnix_ref_init(&cq_priv->ref_cnt, 1, __cq_destruct);
	_gnix_ref_get(cq_priv->domain);
	dlist_init(&cq_priv->poll_nics);
	rwlock_init(&cq_priv->nic_lock);

	cq_priv->cq_fid.fid.fclass = FI_CLASS_CQ;
	cq_priv->cq_fid.fid.context = context;
	cq_priv->cq_fid.fid.ops = fi_cq_ops;
	cq_priv->cq_fid.ops = cq_ops;

	/*
	 * Although we don't need to store entry_size since we're already
	 * storing the format, this might provide a performance benefit
	 * when allocating storage.
	 */
	cq_priv->entry_size = format_sizes[cq_priv->attr.format];


	ret = gnix_cq_set_wait(cq_priv);
	if (ret)
		goto err3;

	fastlock_init(&cq_priv->lock);

	ret = _gnix_queue_create(&cq_priv->events, alloc_cq_entry,
				 free_cq_entry, cq_priv->entry_size,
				 cq_priv->attr.size);
	if (ret)
		goto err4;

	ret = _gnix_queue_create(&cq_priv->errors, alloc_cq_entry,
				 free_cq_entry, sizeof(struct fi_cq_err_entry),
				 0);
	if (ret)
		goto err5;

	*cq = &cq_priv->cq_fid;
	return ret;

err5:
	_gnix_queue_destroy(cq_priv->events);
err4:
	_gnix_ref_put(cq_priv->domain);
	fastlock_destroy(&cq_priv->lock);
err3:
	free(cq_priv);
err2:
	free(fi_cq_ops);
err1:
	free(cq_ops);
err:
	return ret;
}


/*******************************************************************************
 * FI_OPS_* data structures.
 ******************************************************************************/
static const struct fi_ops gnix_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = gnix_cq_close,
	.bind = fi_no_bind,
	.control = gnix_cq_control,
	.ops_open = fi_no_ops_open
};

static const struct fi_ops_cq gnix_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = gnix_cq_read,
	.readfrom = gnix_cq_readfrom,
	.readerr = gnix_cq_readerr,
	.sread = fi_no_cq_sread,
	.sreadfrom = fi_no_cq_sreadfrom,
	.signal = gnix_cq_signal,
	.strerror = gnix_cq_strerror
};
