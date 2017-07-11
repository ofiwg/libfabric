/*
 * Copyright (c) 2016 Intel Corporation, Inc.  All rights reserved.
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

#include "rxm.h"

#define rxm_entry_pop(queue, entry)			\
	do {						\
		fastlock_acquire(&queue->lock);		\
		entry = freestack_isempty(queue->fs) ?	\
			NULL : freestack_pop(queue->fs);\
		fastlock_release(&queue->lock);		\
	} while (0)

#define rxm_entry_push(queue, entry)			\
	do {						\
		fastlock_acquire(&queue->lock);		\
		freestack_push(queue->fs, entry);	\
		fastlock_release(&queue->lock);		\
	} while (0)

char *rxm_proto_state_str[] = {
	RXM_PROTO_STATES(OFI_STR)
};

struct rxm_tx_entry *rxm_tx_entry_get(struct rxm_send_queue *queue)
{
	struct rxm_tx_entry *entry;
	rxm_entry_pop(queue, entry);
	if (!entry) {
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Exhausted tx_entry freestack\n");
		return NULL;
	}
	entry->state = RXM_NONE;
	return entry;
}

void rxm_tx_entry_release(struct rxm_send_queue *queue, struct rxm_tx_entry *entry)
{
	rxm_entry_push(queue, entry);
}

struct rxm_recv_entry *rxm_recv_entry_get(struct rxm_recv_queue *queue)
{
	struct rxm_recv_entry *entry;
	rxm_entry_pop(queue, entry);
	if (!entry) {
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Exhausted recv_entry freestack\n");
		return NULL;
	}
	return entry;
}

void rxm_recv_entry_release(struct rxm_recv_queue *queue, struct rxm_recv_entry *entry)
{
	rxm_entry_push(queue, entry);
}
