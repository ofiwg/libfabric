/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#ifndef _VERBS_UTILS_H
#define _VERBS_UTILS_H

#include <alloca.h>
#include <stddef.h>
#include <sys/types.h>

#include <infiniband/verbs.h>

struct iovec;
struct sockaddr;
struct fi_ibv_msg_ep;

#define fi_ibv_set_sge(sge, buf, len, desc)		\
	do {						\
		sge.addr = (uintptr_t)buf;		\
		sge.length = (uint32_t)len;		\
		sge.lkey = (uint32_t)(uintptr_t)desc;	\
	} while (0)

#define fi_ibv_set_sge_iov(sg_list, iov, count, desc, len)		\
	do {								\
		int i;							\
		if (count) {						\
			sg_list = alloca(sizeof(*sg_list) * count);	\
			for (i = 0; i < count; i++) {			\
				fi_ibv_set_sge(sg_list[i],		\
						iov[i].iov_base,	\
						iov[i].iov_len,		\
						desc[i]);		\
				len += iov[i].iov_len;			\
			}						\
		}							\
	} while (0)

#define fi_ibv_set_sge_inline(sge, buf, len)	\
	do {					\
		sge.addr = (uintptr_t)buf;	\
		sge.length = (uint32_t)len;	\
	} while (0)

#define fi_ibv_set_sge_iov_inline(sg_list, iov, count, len)		\
	do {								\
		int i;							\
		if (count) {						\
			sg_list = alloca(sizeof(*sg_list) * count);	\
			for (i = 0; i < count; i++) {			\
				fi_ibv_set_sge_inline(sg_list[i],	\
						iov[i].iov_base,	\
						iov[i].iov_len);	\
				len += iov[i].iov_len;			\
			}						\
		}							\
	} while (0)

ssize_t fi_ibv_send(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr, size_t len,
		int count, void *context);

inline ssize_t
fi_ibv_send_buf_inline(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
		const void *buf, size_t len)
{
	struct ibv_sge sge;

	fi_ibv_set_sge_inline(sge, buf, len);
	wr->sg_list = &sge;

	return fi_ibv_send(ep, wr, len, 1, NULL);
}

int fi_ibv_sockaddr_len(struct sockaddr *addr);
int fi_ibv_copy_addr(void *dst_addr, size_t *dst_addrlen, void *src_addr);
ssize_t fi_ibv_send_buf(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
		const void *buf, size_t len, void *desc, void *context);
ssize_t fi_ibv_send_iov_flags(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
		const struct iovec *iov, void **desc, int count, void *context,
		uint64_t flags);

#define fi_ibv_send_iov(ep, wr, iov, desc, count, context)	\
	fi_ibv_send_iov_flags(ep, wr, iov, desc, count, context,\
			ep->info->tx_attr->op_flags)

#define fi_ibv_send_msg(ep, wr, msg, flags)					\
	fi_ibv_send_iov_flags(ep, wr, msg->msg_iov, msg->desc, msg->iov_count,	\
			msg->context, flags)

#endif /* _VERBS_UTILS_H */
