/*
 * Copyright (c) 2015-2020 Intel Corporation. All rights reserved.
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include "sm2.h"

static int
sm2_av_close(struct fid *fid)
{
	int ret;
	struct util_av *av;
	struct sm2_av *sm2_av;

	av = container_of(fid, struct util_av, av_fid);
	sm2_av = container_of(av, struct sm2_av, util_av);

	ret = ofi_av_close(av);
	if (ret)
		return ret;

	sm2_mmap_unmap_and_close(&sm2_av->sm2_mmap);
	free(sm2_av->sm2_aux);
	free(av);
	return 0;
}

/*
 * Input address: smr name (string)
 * output address: index (fi_addr_t) of the address
 * @return the number of successful insertions
 */
static int
sm2_av_insert(struct fid_av *av_fid, const void *addr, size_t count,
	      fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	struct util_av *util_av;
	struct sm2_av *sm2_av;
	int util_addr;
	int i, ret;
	int succ_count = 0;

	util_av = container_of(av_fid, struct util_av, av_fid);
	sm2_av = container_of(util_av, struct sm2_av, util_av);

	/* acquire the lock */
	sm2_coordinator_lock(&sm2_av->sm2_mmap);

	for (i = 0; i < count; i++, addr = (char *) addr + SM2_NAME_MAX) {
		FI_WARN(&sm2_prov, FI_LOG_AV,
			"fi_av_insert(): About to call "
			"sm2_coordinator_allocate_entry() "
			"on addr %s, my_pid: %d, count: %zu,  \n",
			(char *) addr, getpid(), count);
		ret = sm2_coordinator_allocate_entry(addr, &sm2_av->sm2_mmap,
						     &util_addr, false);
		FI_WARN(&sm2_prov, FI_LOG_AV,
			"fi_av_insert(): finished "
			"sm2_coordinator_allocate_entry() "
			"resulting AV Found = %d \n",
			util_addr);
		if (ret && util_av->eq) {
			ofi_av_write_event(util_av, i, -ret, context);
		}
		if (ret)
			continue;

		if (flags & FI_AV_USER_ID) {
			sm2_av->sm2_aux[util_addr].cqfid = fi_addr[i];
			FI_INFO(&sm2_prov, FI_LOG_AV,
				"fi_addr: %d, USER_ID: %" PRIu64 "\n",
				util_addr, fi_addr[i]);
		} else {
			sm2_av->sm2_aux[util_addr].cqfid = util_addr;
			FI_INFO(&sm2_prov, FI_LOG_AV, "fi_addr: %d\n",
				util_addr);
		}

		if (fi_addr) {
			fi_addr[i] = util_addr;
		}

		succ_count++;
	}
	/* release the lock */
	sm2_coordinator_unlock(&sm2_av->sm2_mmap);

	if (flags & FI_EVENT) {
		ofi_av_write_event(util_av, succ_count, 0, context);
	}
	return succ_count;
}

static int
sm2_av_remove(struct fid_av *av_fid, fi_addr_t *fi_addr, size_t count,
	      uint64_t flags)
{
	/* No-OP */
	return 0;
}

static int
sm2_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr, size_t *addrlen)
{
	struct util_av *util_av;
	struct sm2_av *sm2_av;
	struct sm2_coord_file_header *header;
	struct sm2_ep_allocation_entry *entries;

	FI_WARN(&sm2_prov, FI_LOG_AV, "sm2_av_lookup: %s\n", (char *) addr);

	*addrlen = MIN(SM2_NAME_MAX, *addrlen);

	util_av = container_of(av, struct util_av, av_fid);
	sm2_av = container_of(util_av, struct sm2_av, util_av);

	header = (void *) sm2_av->sm2_mmap.base;
	entries = (void *) (sm2_av->sm2_mmap.base +
			    header->ep_enumerations_offset);

	strncpy(addr, entries[fi_addr].ep_name, *addrlen - 1);
	((char *) addr)[*addrlen] = '\0';
	*addrlen = strnlen(entries[fi_addr].ep_name, SM2_NAME_MAX - 1) + 1;

	return 0;
}

static const char *
sm2_av_straddr(struct fid_av *av, const void *addr, char *buf, size_t *len)
{
	/* the input address is a string format */
	if (buf)
		strncpy(buf, (const char *) addr, *len);

	*len = strlen((const char *) addr) + 1;
	return buf;
}

static struct fi_ops sm2_av_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = sm2_av_close,
	.bind = ofi_av_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_av sm2_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = sm2_av_insert,
	.insertsvc = fi_no_av_insertsvc,
	.insertsym = fi_no_av_insertsym,
	.remove = sm2_av_remove,
	.lookup = sm2_av_lookup,
	.straddr = sm2_av_straddr,
};

/**
 * @brief create and open an AV
 *
 * Allocate space for the AV
 * Ensure attr->type is FI_AV_TABLE or FI_AV_UNSPEC
 *
 * @param domain
 * @param attr
 * @param[out] **av
 * @param context
 */
int
sm2_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
	    struct fid_av **av, void *context)
{
	struct util_domain *util_domain;
	struct util_av_attr util_attr;
	struct sm2_av *sm2_av;
	struct sm2_coord_file_header *header;
	int ret;

	if (!attr) {
		FI_INFO(&sm2_prov, FI_LOG_AV, "invalid attr\n");
		return -FI_EINVAL;
	}

	if (attr->name) {
		FI_INFO(&sm2_prov, FI_LOG_AV, "shared AV not supported\n");
		return -FI_ENOSYS;
	}

	if (attr->type == FI_AV_UNSPEC)
		attr->type = FI_AV_MAP;

	if (attr->type != FI_AV_MAP) {
		FI_WARN(&sm2_prov, FI_LOG_AV, "Only FI_AV_MAP Supported\n");
		return -FI_EINVAL;
	}

	util_domain = container_of(domain, struct util_domain, domain_fid);

	sm2_av = calloc(1, sizeof *sm2_av);
	if (!sm2_av)
		return -FI_ENOMEM;

	util_attr.addrlen = 0;
	util_attr.context_len = 0;
	util_attr.flags = 0;
	if (attr->count > SM2_MAX_PEERS) {
		FI_INFO(&sm2_prov, FI_LOG_AV, "count %d exceeds max peers\n",
			(int) attr->count);
		ret = -FI_ENOSYS;
		goto out;
	}

	ret = ofi_av_init(util_domain, attr, &util_attr, &sm2_av->util_av,
			  context);
	if (ret)
		goto out;

	*av = &sm2_av->util_av.av_fid;
	(*av)->fid.ops = &sm2_av_fi_ops;
	(*av)->ops = &sm2_av_ops;

	sm2_coordinator_open_and_lock(&sm2_av->sm2_mmap);
	/* TODO Add logic to shrink the file if all PIDS in it are dead*/
	sm2_coordinator_unlock(&sm2_av->sm2_mmap);

	header = (void *) sm2_av->sm2_mmap.base;
	sm2_av->sm2_aux = calloc(header->ep_enumerations_max,
				 sizeof(struct sm2_private_aux));

	// Initialize all addresses to FI_ADDR_NOTAVAIL
	for (int i = 0; i < header->ep_enumerations_max; i++) {
		sm2_av->sm2_aux[i].cqfid = FI_ADDR_NOTAVAIL;
	}

	if (ret)
		goto close;

	return 0;

close:
	ofi_av_close(&sm2_av->util_av);
out:
	free(sm2_av);
	return ret;
}
