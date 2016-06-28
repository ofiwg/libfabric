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

#include <fi_util.h>
#include <fi.h>


static DEFINE_LIST(fabric_list);
static fastlock_t lock;


void fi_util_init(void)
{
	fastlock_init(&lock);
}

void fi_util_fini(void)
{
	fastlock_destroy(&lock);
}

void fi_fabric_insert(struct util_fabric *fabric)
{
	fastlock_acquire(&lock);
	dlist_insert_tail(&fabric->list_entry, &fabric_list);
	fastlock_release(&lock);
}

static int fabric_match_name(struct dlist_entry *item, const void *arg)
{
	struct util_fabric *fabric;

	fabric = container_of(item, struct util_fabric, list_entry);
	return !strcmp(fabric->name, arg);
}

struct util_fabric *fi_fabric_find(const char *name)
{
	struct dlist_entry *item;

	fastlock_acquire(&lock);
	item = dlist_find_first_match(&fabric_list, fabric_match_name, name);
	fastlock_release(&lock);

	return item ? container_of(item, struct util_fabric, list_entry) : NULL;
}

void fi_fabric_remove(struct util_fabric *fabric)
{
	assert(fi_fabric_find(fabric->name));
	fastlock_acquire(&lock);
	dlist_remove(&fabric->list_entry);
	fastlock_release(&lock);
}


static int fi_fid_match(struct dlist_entry *entry, const void *fid)
{
	struct fid_list_entry *item;
	item = container_of(entry, struct fid_list_entry, entry);
	return (item->fid == fid);
}

int fid_list_insert(struct dlist_entry *fid_list, fastlock_t *lock,
		    struct fid *fid)
{
	int ret = 0;
	struct dlist_entry *entry;
	struct fid_list_entry *item;

	fastlock_acquire(lock);
	entry = dlist_find_first_match(fid_list, fi_fid_match, fid);
	if (entry)
		goto out;

	item = calloc(1, sizeof(*item));
	if (!item) {
		ret = -FI_ENOMEM;
		goto out;
	}

	item->fid = fid;
	dlist_insert_tail(&item->entry, fid_list);
out:
	fastlock_release(lock);
	return ret;
}

void fid_list_remove(struct dlist_entry *fid_list, fastlock_t *lock,
		     struct fid *fid)
{
	struct fid_list_entry *item;
	struct dlist_entry *entry;

	fastlock_acquire(lock);
	entry = dlist_remove_first_match(fid_list, fi_fid_match, fid);
	fastlock_release(lock);

	if (entry) {
		item = container_of(entry, struct fid_list_entry, entry);
		free(item);
	}
}

int util_find_domain(struct dlist_entry *item, const void *arg)
{
	const struct util_domain *domain;
	const struct fi_info *info = arg;

	domain = container_of(item, struct util_domain, list_entry);

	return !strcmp(domain->name, info->domain_attr->name) &&
		!(info->caps & ~domain->caps) &&
		 ((info->mode & domain->mode) == domain->mode);
}

int util_getinfo(const struct util_prov *util_prov, uint32_t version,
		 const char *node, const char *service, uint64_t flags,
		 struct fi_info *hints, struct fi_info **info)
{
	struct util_fabric *fabric;
	struct util_domain *domain;
	struct dlist_entry *item;
	const struct fi_provider *prov = util_prov->prov;
	int ret, copy_dest;

	FI_DBG(prov, FI_LOG_CORE, "checking info\n");

	if ((flags & FI_SOURCE) && !node && !service) {
		FI_INFO(prov, FI_LOG_CORE,
			"FI_SOURCE set, but no node or service\n");
		return -FI_EINVAL;
	}

	ret = fi_check_info(util_prov, hints, FI_MATCH_EXACT);
	if (ret)
		return ret;

	*info = fi_dupinfo(util_prov->info);
	if (!*info) {
		FI_INFO(prov, FI_LOG_CORE, "cannot copy info\n");
		return -FI_ENOMEM;
	}

	ofi_alter_info(*info, hints);

	fabric = fi_fabric_find((*info)->fabric_attr->name);
	if (fabric) {
		FI_DBG(prov, FI_LOG_CORE, "Found opened fabric\n");
		(*info)->fabric_attr->fabric = &fabric->fabric_fid;

		fastlock_acquire(&fabric->lock);
		item = dlist_find_first_match(&fabric->domain_list,
					      util_find_domain, *info);
		if (item) {
			FI_DBG(prov, FI_LOG_CORE, "Found open domain\n");
			domain = container_of(item, struct util_domain,
					      list_entry);
			(*info)->domain_attr->domain = &domain->domain_fid;
		}
		fastlock_release(&fabric->lock);

	}

	if (flags & FI_SOURCE) {
		ret = ofi_get_addr((*info)->addr_format, flags,
				  node, service, &(*info)->src_addr,
				  &(*info)->src_addrlen);
		if (ret) {
			FI_INFO(prov, FI_LOG_CORE,
				"source address not available\n");
			goto err;
		}
		copy_dest = (hints && hints->dest_addr);
	} else {
		if (node || service) {
			copy_dest = 0;
			ret = ofi_get_addr((*info)->addr_format, flags,
					  node, service, &(*info)->dest_addr,
					  &(*info)->dest_addrlen);
			if (ret) {
				FI_INFO(prov, FI_LOG_CORE,
					"cannot resolve dest address\n");
				goto err;
			}
		} else {
			copy_dest = (hints && hints->dest_addr);
		}

		if (hints && hints->src_addr) {
			(*info)->src_addr = mem_dup(hints->src_addr,
						    hints->src_addrlen);
			if (!(*info)->src_addr) {
				ret = -FI_ENOMEM;
				goto err;
			}
			(*info)->src_addrlen = hints->src_addrlen;
		}
	}

	if (copy_dest) {
		(*info)->dest_addr = mem_dup(hints->dest_addr,
					     hints->dest_addrlen);
		if (!(*info)->dest_addr) {
			ret = -FI_ENOMEM;
			goto err;
		}
		(*info)->dest_addrlen = hints->dest_addrlen;
	}

	if ((*info)->dest_addr && !(*info)->src_addr) {
		ret = ofi_get_src_addr((*info)->addr_format, (*info)->dest_addr,
				      (*info)->dest_addrlen, &(*info)->src_addr,
				      &(*info)->src_addrlen);
		if (ret) {
			FI_INFO(prov, FI_LOG_CORE,
				"cannot resolve source address\n");
		}
	}
	return 0;

err:
	fi_freeinfo(*info);
	return ret;
}
