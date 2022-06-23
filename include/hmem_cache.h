/*
 * Copyright (c) 2022 UT-Battelle, LLC. All rights reserved.
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
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

#ifndef HMEM_CACHE_H
#define HMEM_CACHE_H

#include <pgtable.h>
#include <rdma/fi_errno.h>
#include "ofi_util.h"
#include "ofi.h"
#include "shared/ofi_str.h"
#include "ofi_prov.h"
#include "ofi_perf.h"
#include "ofi_hmem.h"
#include "rdma/fi_ext.h"

#define IPC_HANDLE_SIZE		64
struct ipc_info {
	uint64_t	iface;
	uintptr_t	address;
	size_t		length;
	size_t		base_length;
	uint64_t	base_offset;
	int			dev_num;
	union {
		uint8_t		ipc_handle[IPC_HANDLE_SIZE];
		struct {
			uint64_t	device;
			uint64_t	offset;
			uint64_t	fd_handle;
		};
	};
};

struct ipc_cache_region {
	pgt_region_t		super;
	struct dlist_entry	list;
	struct ipc_info		key;
	void				*mapped_addr;
};

typedef int (*map_cb_t)(enum fi_hmem_iface iface, void **handle,
			 size_t len, uint64_t device, void **ipc_ptr);
typedef int (*unmap_cb_t)(enum fi_hmem_iface iface, void *ipc_ptr);

struct hmem_cache {
	pthread_rwlock_t lock;
	pgtable_t pgtable;
	char *name;
	/* Callbacks for mapping and unmapping*/
	map_cb_t map_cb;
	unmap_cb_t unmap_cb;
	struct hmem_cache *make_cache;
};

int ipc_create_hmem_cache(struct hmem_cache **cache,
						  const char *name,
						  map_cb_t map_cb, unmap_cb_t unmap_cb);
void ipc_destroy_hmem_cache(struct hmem_cache *cache);
int ipc_cache_map_memhandle(struct hmem_cache *cache, struct ipc_info *key,
							void **mapped_addr);
void ipc_cache_invalidate(struct hmem_cache *cache, void *address);

#endif /* HMEM_CACHE_H */
