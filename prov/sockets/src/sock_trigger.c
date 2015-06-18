/*
 * Copyright (c) 2014-2015 Intel Corporation, Inc.  All rights reserved.
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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sock.h"
#include "sock_util.h"

#define SOCK_LOG_DBG(...) _SOCK_LOG_DBG(FI_LOG_EP_DATA, __VA_ARGS__)
#define SOCK_LOG_ERROR(...) _SOCK_LOG_ERROR(FI_LOG_EP_DATA, __VA_ARGS__)

ssize_t sock_queue_rma_op(struct fid_ep *ep, const struct fi_msg_rma *msg,
			  uint64_t flags, uint8_t op_type)
{
	struct sock_cntr *cntr;
	struct sock_trigger *trigger;
	struct fi_triggered_context *trigger_context;
	struct fi_trigger_threshold *threshold;

	trigger_context = (struct fi_triggered_context*)msg->context;
	if ((trigger_context->event_type != FI_TRIGGER_THRESHOLD) ||
	    (flags & FI_INJECT))
		return -FI_EINVAL;

	threshold = &trigger_context->trigger.threshold;
	cntr = container_of(threshold->cntr, struct sock_cntr, cntr_fid);
	if (atomic_get(&cntr->value) >= threshold->threshold)
		return 1;

	trigger = calloc(1, sizeof *trigger);
	if (!trigger)
		return -FI_ENOMEM;
	
	trigger->threshold = threshold->threshold;
	memcpy(&trigger->op.rma.msg, msg, sizeof *msg);
	trigger->op.rma.msg.msg_iov = &trigger->op.rma.msg_iov[0];
	trigger->op.rma.msg.rma_iov = &trigger->op.rma.rma_iov[0];
	
	memcpy(&trigger->op.rma.msg_iov[0], &msg->msg_iov[0],
	       msg->iov_count * sizeof (struct iovec));
	memcpy(&trigger->op.rma.rma_iov[0], &msg->rma_iov[0],
	       msg->rma_iov_count * sizeof (struct fi_rma_iov));

	trigger->op_type = op_type;
	trigger->ep = ep;
	trigger->flags = flags;
	
	fastlock_acquire(&cntr->trigger_lock);
	dlist_insert_tail(&trigger->entry, &cntr->trigger_list);
	fastlock_release(&cntr->trigger_lock);
	return 0;
}

ssize_t sock_queue_msg_op(struct fid_ep *ep, const struct fi_msg *msg, 
			  uint64_t flags, uint8_t op_type)
{
	struct sock_cntr *cntr;
	struct sock_trigger *trigger;
	struct fi_triggered_context *trigger_context;
	struct fi_trigger_threshold *threshold;

	trigger_context = (struct fi_triggered_context*)msg->context;
	if ((trigger_context->event_type != FI_TRIGGER_THRESHOLD) ||
	    (flags & FI_INJECT))
		return -FI_EINVAL;

	threshold = &trigger_context->trigger.threshold;
	cntr = container_of(threshold->cntr, struct sock_cntr, cntr_fid);
	if (atomic_get(&cntr->value) >= threshold->threshold)
		return 1;

	trigger = calloc(1, sizeof *trigger);
	if (!trigger)
		return -FI_ENOMEM;
	
	trigger->threshold = threshold->threshold;

	memcpy(&trigger->op.msg.msg, msg, sizeof *msg);
	trigger->op.msg.msg.msg_iov = &trigger->op.msg.msg.msg_iov[0];
	memcpy((void*)&trigger->op.msg.msg.msg_iov[0], &msg->msg_iov[0],
	       msg->iov_count * sizeof (struct iovec));

	trigger->op_type = op_type;
	trigger->ep = ep;
	trigger->flags = flags;
	
	fastlock_acquire(&cntr->trigger_lock);
	dlist_insert_tail(&trigger->entry, &cntr->trigger_list);
	fastlock_release(&cntr->trigger_lock);
	return 0;
}

ssize_t sock_queue_tmsg_op(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			   uint64_t flags, uint8_t op_type)
{
	struct sock_cntr *cntr;
	struct sock_trigger *trigger;
	struct fi_triggered_context *trigger_context;
	struct fi_trigger_threshold *threshold;

	trigger_context = (struct fi_triggered_context*)msg->context;
	if ((trigger_context->event_type != FI_TRIGGER_THRESHOLD) ||
	    (flags & FI_INJECT))
		return -FI_EINVAL;

	threshold = &trigger_context->trigger.threshold;
	cntr = container_of(threshold->cntr, struct sock_cntr, cntr_fid);
	if (atomic_get(&cntr->value) >= threshold->threshold)
		return 1;

	trigger = calloc(1, sizeof *trigger);
	if (!trigger)
		return -FI_ENOMEM;
	
	trigger->threshold = threshold->threshold;
	
	memcpy(&trigger->op.tmsg.msg, msg, sizeof *msg);
	trigger->op.tmsg.msg.msg_iov = &trigger->op.tmsg.msg.msg_iov[0];
	memcpy((void*)&trigger->op.tmsg.msg.msg_iov[0], &msg->msg_iov[0],
	       msg->iov_count * sizeof (struct iovec));

	trigger->op_type = op_type;
	trigger->ep = ep;
	trigger->flags = flags;
	
	fastlock_acquire(&cntr->trigger_lock);
	dlist_insert_tail(&trigger->entry, &cntr->trigger_list);
	fastlock_release(&cntr->trigger_lock);
	return 0;
}

ssize_t sock_queue_atomic_op(struct fid_ep *ep, const struct fi_msg_atomic *msg, 
			     const struct fi_ioc *comparev, size_t compare_count, 
			     struct fi_ioc *resultv, size_t result_count, 
			     uint64_t flags, uint8_t op_type)
{
	struct sock_cntr *cntr;
	struct sock_trigger *trigger;
	struct fi_triggered_context *trigger_context;
	struct fi_trigger_threshold *threshold;

	trigger_context = (struct fi_triggered_context*)msg->context;
	if ((trigger_context->event_type != FI_TRIGGER_THRESHOLD) ||
	    (flags & FI_INJECT))
		return -FI_EINVAL;

	threshold = &trigger_context->trigger.threshold;
	cntr = container_of(threshold->cntr, struct sock_cntr, cntr_fid);
	if (atomic_get(&cntr->value) >= threshold->threshold)
		return 1;

	trigger = calloc(1, sizeof *trigger);
	if (!trigger)
		return -FI_ENOMEM;
	
	trigger->threshold = threshold->threshold;
	memcpy(&trigger->op.atomic.msg, msg, sizeof *msg);
	trigger->op.atomic.msg.msg_iov = &trigger->op.atomic.msg_iov[0];
	trigger->op.atomic.msg.rma_iov = &trigger->op.atomic.rma_iov[0];

	memcpy(&trigger->op.atomic.msg_iov[0], &msg->msg_iov[0],
	       msg->iov_count * sizeof (struct fi_ioc));
	memcpy(&trigger->op.atomic.rma_iov[0], &msg->rma_iov[0],
	       msg->iov_count * sizeof (struct fi_rma_ioc));

	if (comparev) {
		memcpy(&trigger->op.atomic.comparev[0], &comparev[0],
		       compare_count * sizeof (struct fi_ioc));
	}

	if (resultv) {
		memcpy(&trigger->op.atomic.resultv[0], &resultv[0],
		       result_count * sizeof (struct fi_ioc));
	}	
	
	trigger->op_type = op_type;
	trigger->ep = ep;
	trigger->flags = flags;
	
	fastlock_acquire(&cntr->trigger_lock);
	dlist_insert_tail(&trigger->entry, &cntr->trigger_list);
	fastlock_release(&cntr->trigger_lock);
	return 0;
}
