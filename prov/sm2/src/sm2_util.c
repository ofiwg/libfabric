/*
 * Copyright (c) 2016-2021 Intel Corporation. All rights reserved.
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

#include "config.h"
#include "sm2_common.h"
#include "sm2_fifo.h"

#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

struct dlist_entry sm2_ep_name_list;
DEFINE_LIST(sm2_ep_name_list);
pthread_mutex_t sm2_ep_list_lock = PTHREAD_MUTEX_INITIALIZER;

void
sm2_cleanup(void)
{
	struct sm2_ep_name *ep_name;
	struct dlist_entry *tmp;

	pthread_mutex_lock(&sm2_ep_list_lock);
	dlist_foreach_container_safe(&sm2_ep_name_list, struct sm2_ep_name,
				     ep_name, entry, tmp) free(ep_name);
	pthread_mutex_unlock(&sm2_ep_list_lock);
}

size_t
sm2_calculate_size_offsets(ptrdiff_t num_fqe, ptrdiff_t *rq_offset,
			   ptrdiff_t *fq_offset)
{
	size_t total_size;

	/* First memory block.  The header of an sm2_region */
	total_size = sizeof(struct sm2_region);

	/* Second memory block: the recv_queue_fifo, an sm2_fifo */
	if (rq_offset)
		*rq_offset = total_size;
	total_size += sizeof(struct sm2_fifo);

	/* Third memory block: the message objects in a free queue */
	if (fq_offset)
		*fq_offset = total_size;
	total_size +=
		freestack_size(sizeof(struct sm2_free_queue_entry), num_fqe);

	/*
	 * Revisit later to see if we really need the size adjustment, or
	 * at most align to a multiple of a page size.
	 */
	total_size = roundup_power_of_two(total_size);

	return total_size;
}

int
sm2_create(const struct fi_provider *prov, const struct sm2_attr *attr,
	   struct sm2_mmap *sm2_mmap, int *id)
{
	struct sm2_ep_name *ep_name;
	ptrdiff_t recv_queue_offset, free_stack_offset;
	int ret;
	void *mapped_addr;
	struct sm2_region *smr;

	sm2_calculate_size_offsets(attr->num_fqe, &recv_queue_offset,
				   &free_stack_offset);

	FI_WARN(prov, FI_LOG_EP_CTRL, "Claiming an entry for (%s)\n",
		attr->name);
	sm2_coordinator_lock(sm2_mmap);
	ret = sm2_coordinator_allocate_entry(attr->name, sm2_mmap, id, true);

	/* TODO: handle address-in-use error (FI_EBUSY?)*/
	/* TODO: handle no available space on device error */
	/* TODO: handle no available slots left error */

	ep_name = calloc(1, sizeof(*ep_name));
	if (!ep_name) {
		FI_WARN(prov, FI_LOG_EP_CTRL, "calloc error\n");
		ret = -FI_ENOMEM;
		goto close;
	}
	strncpy(ep_name->name, (char *) attr->name, SM2_NAME_MAX - 1);
	ep_name->name[SM2_NAME_MAX - 1] = '\0';

	pthread_mutex_lock(&sm2_ep_list_lock);
	dlist_insert_tail(&ep_name->entry, &sm2_ep_name_list);

	if (ret < 0) {
		FI_WARN(prov, FI_LOG_EP_CTRL, "ftruncate error\n");
		ret = -errno;
		goto remove;
	}

	mapped_addr = sm2_mmap_ep_region(sm2_mmap, *id);

	if (mapped_addr == MAP_FAILED) {
		FI_WARN(prov, FI_LOG_EP_CTRL, "mmap error\n");
		ret = -errno;
		goto remove;
	}

	/* TODO: Handle SM2_FLAG_HMEM_ENABLED */

	pthread_mutex_unlock(&sm2_ep_list_lock);
	smr = mapped_addr;

	smr->version = SM2_VERSION;
	smr->flags = attr->flags;
	smr->recv_queue_offset = recv_queue_offset;
	smr->free_stack_offset = free_stack_offset;

	sm2_fifo_init(sm2_recv_queue(smr));
	smr_freestack_init(sm2_free_stack(smr), attr->num_fqe,
			   sizeof(struct sm2_free_queue_entry));

	/*
	 * Need to set PID in header here...
	 * this will unblock other processes trying to send to us
	 */
	assert(sm2_mmap_entries(sm2_mmap)[*id].pid == getpid());
	sm2_mmap_entries(sm2_mmap)[*id].startup_ready = 1;
	atomic_wmb();

	/* Need to unlock coordinator so that others can add themselves to
	 * header */
	sm2_coordinator_unlock(sm2_mmap);
	return 0;

remove:
	dlist_remove(&ep_name->entry);
	pthread_mutex_unlock(&sm2_ep_list_lock);
	free(ep_name);
close:
	return ret;
}
