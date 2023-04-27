/*
 * Copyright (c) Intel Corporation. All rights reserved.
 * Copyright (c) Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include "config.h"
#include "sm2_fifo.h"

#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

pthread_mutex_t sm2_ep_list_lock = PTHREAD_MUTEX_INITIALIZER;

size_t sm2_calculate_size_offsets(ptrdiff_t *rq_offset, ptrdiff_t *fs_offset)
{
	size_t total_size;

	total_size = sizeof(struct sm2_region);

	if (rq_offset)
		*rq_offset = total_size;
	total_size += sizeof(struct sm2_fifo);

	if (fs_offset)
		*fs_offset = total_size;
	total_size += freestack_size(sizeof(struct sm2_xfer_entry),
				     SM2_NUM_XFER_ENTRY_PER_PEER);

	return total_size;
}

int sm2_create(const struct fi_provider *prov, const struct sm2_attr *attr,
	       struct sm2_mmap *sm2_mmap, sm2_gid_t *gid)
{
	struct sm2_ep_name *ep_name;
	ptrdiff_t recv_queue_offset, freestack_offset;
	int ret;
	void *mapped_addr;
	struct sm2_region *smr;

	sm2_calculate_size_offsets(&recv_queue_offset, &freestack_offset);

	FI_WARN(prov, FI_LOG_EP_CTRL, "Claiming an entry for (%s)\n",
		attr->name);
	sm2_file_lock(sm2_mmap);
	ret = sm2_entry_allocate(attr->name, sm2_mmap, gid, true);

	if (ret) {
		FI_WARN(prov, FI_LOG_EP_CTRL,
			"Failed to allocate an entry in the SHM file for "
			"ourselves\n");
		sm2_file_unlock(sm2_mmap);
		return ret;
	}

	ep_name = calloc(1, sizeof(*ep_name));
	if (!ep_name) {
		FI_WARN(prov, FI_LOG_EP_CTRL, "calloc error\n");
		return -FI_ENOMEM;
	}
	strncpy(ep_name->name, (char *) attr->name, FI_NAME_MAX - 1);
	ep_name->name[FI_NAME_MAX - 1] = '\0';

	if (ret < 0) {
		FI_WARN(prov, FI_LOG_EP_CTRL, "ftruncate error\n");
		ret = -errno;
		goto remove;
	}

	mapped_addr = sm2_mmap_ep_region(sm2_mmap, *gid);

	if (mapped_addr == MAP_FAILED) {
		FI_WARN(prov, FI_LOG_EP_CTRL, "mmap error\n");
		ret = -errno;
		goto remove;
	}

	smr = mapped_addr;

	smr->version = SM2_VERSION;
	smr->flags = attr->flags;
	smr->recv_queue_offset = recv_queue_offset;
	smr->freestack_offset = freestack_offset;

	sm2_fifo_init(sm2_recv_queue(smr));
	smr_freestack_init(sm2_freestack(smr), SM2_NUM_XFER_ENTRY_PER_PEER,
			   sizeof(struct sm2_xfer_entry));

	/*
	 * Need to set PID in header here...
	 * this will unblock other processes trying to send to us
	 */
	assert(sm2_mmap_entries(sm2_mmap)[*gid].pid == getpid());
	sm2_mmap_entries(sm2_mmap)[*gid].startup_ready = true;
	atomic_wmb();

	/* Need to unlock coordinator so that others can add themselves to
	 * header */
	sm2_file_unlock(sm2_mmap);
	return 0;

remove:
	sm2_file_unlock(sm2_mmap);
	free(ep_name);
	return ret;
}
