/*
 * Copyright (c) Amazon.com, Inc. or its affiliates.
 * All rights reserved.
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

static inline
void efa_rdm_msg_construct(struct fi_msg *msg, const struct iovec *iov, void **desc,
		       size_t count, fi_addr_t addr, void *context, uint64_t data)
{
	msg->msg_iov = iov;
	msg->desc = desc;
	msg->iov_count = count;
	msg->addr = addr;
	msg->context = context;
	msg->data = data;
}

/**
 * functions to allocate rxe for two sided operations
 */
struct efa_rdm_ope *efa_rdm_msg_alloc_rxe(struct efa_rdm_ep *ep,
					    const struct fi_msg *msg,
					    uint32_t op, uint64_t flags,
					    uint64_t tag, uint64_t ignore);

struct efa_rdm_ope *efa_rdm_msg_alloc_rxe_for_msgrtm(struct efa_rdm_ep *ep,
						     struct efa_rdm_pke **pkt_entry_ptr);

struct efa_rdm_ope *efa_rdm_msg_alloc_rxe_for_tagrtm(struct efa_rdm_ep *ep,
						     struct efa_rdm_pke **pkt_entry_ptr);

struct efa_rdm_ope *efa_rdm_msg_split_rxe(struct efa_rdm_ep *ep,
					    struct efa_rdm_ope *posted_entry,
					    struct efa_rdm_ope *consumer_entry,
					    struct efa_rdm_pke *pkt_entry);
/*
 * The following 2 OP structures are defined in efa_rdm_msg.c and is
 * used by #efa_rdm_ep_open()
 */
extern struct fi_ops_msg efa_rdm_msg_ops;

extern struct fi_ops_tagged efa_rdm_msg_tagged_ops;