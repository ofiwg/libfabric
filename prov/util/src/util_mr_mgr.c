/*
 * Copyright (c) 2017 Intel Corporation, Inc. All rights reserved.
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

#include <fi_util.h>

int ofi_mr_mgr_init(struct ofi_mr_mgr *mgr, struct ofi_mr_mgr_attr *attr)
{
	int ret;
	assert(mgr && !mgr->domain);

	mgr->monitor.subscribe = attr->monitor_attr.subscribe;
	mgr->monitor.unsubscribe = attr->monitor_attr.unsubscribe;
	mgr->monitor.get_event = attr->monitor_attr.get_event;
	ofi_monitor_init(&mgr->monitor);

	mgr->cache.domain = mgr->domain = attr->domain;
	mgr->cache.reg_size = attr->cache_attr.reg_size;
	mgr->cache.stale_size = attr->cache_attr.stale_size;
	mgr->cache.elem_size = attr->cache_attr.elem_size;
	mgr->cache.reg_callback = attr->cache_attr.reg_callback;
	mgr->cache.dereg_callback = attr->cache_attr.dereg_callback;

	ret = ofi_mr_cache_init(&mgr->cache, mgr->domain, &mgr->monitor);
	if (ret)
		FI_WARN(mgr->domain->prov,
			FI_LOG_MR,
			"Unable to init MR cache, status - %d", ret);

	return ret;
}

struct ofi_mr_mgr_entry *
ofi_mr_mgr_insert(struct ofi_mr_mgr *mgr, const struct fi_mr_attr *mr_attr)
{
	int ret, i;
	struct ofi_mr_mgr_entry *entry;
	struct ofi_mr_reg_attr reg_attr = {
		.access		= mr_attr->access,
		.offset		= mr_attr->offset,
		.requested_key	= mr_attr->requested_key,
		.context	= mr_attr->context,
		.auth_key_size	= mr_attr->auth_key_size,
		.auth_key	= mr_attr->auth_key,
	};

	entry = calloc(1, sizeof(*entry));
	if (!entry)
		goto err1;

	entry->count = mr_attr->iov_count;
	entry->cache_entry = calloc(mr_attr->iov_count,
				    sizeof(*entry->cache_entry));
	if (!entry->cache_entry)
		goto err2;

	for (i = 0; i < mr_attr->iov_count; i++) {
		ret = ofi_mr_cache_register(&mgr->cache,
					    (uint64_t)(uintptr_t)mr_attr->mr_iov[i].iov_base,
					    mr_attr->mr_iov[i].iov_len, &reg_attr,
					    &entry->cache_entry[i]);
		if (ret)
			goto err3;
	}

	return entry;
err3:
	while (--i >= 0)
		(void)ofi_mr_cache_deregister(&mgr->cache,
					      entry->cache_entry[i]);
	free(entry->cache_entry);
err2:
	free(entry);
	entry = NULL;
err1:
	return entry;
}

void ofi_mr_mgr_remove(struct ofi_mr_mgr *mgr, struct ofi_mr_mgr_entry *entry)
{
	int i;

	for (i = 0; i < entry->count; i++)
		(void)ofi_mr_cache_deregister(&mgr->cache,
					      entry->cache_entry[i]);
	free(entry->cache_entry);
	free(entry);
}

void ofi_mr_mgr_cleanup(struct ofi_mr_mgr *mgr)
{
	ofi_mr_cache_cleanup(&mgr->cache);
	ofi_monitor_cleanup(&mgr->monitor);
}
