/*
 * Copyright (c) 2018-2022 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#ifndef EFA_DGRAM_CQ_H
#define EFA_DGRAM_CQ_H

typedef void (*efa_dgram_cq_read_entry)(struct ibv_cq_ex *ibv_cqx, int index, void *buf);

struct efa_dgram_cq {
	struct util_cq		util_cq;
	struct efa_domain	*domain;
	size_t			entry_size;
	efa_dgram_cq_read_entry	read_entry;
	ofi_spin_t		lock;
	struct ofi_bufpool	*wce_pool;
	uint32_t	flags; /* User defined capability mask */

	struct ibv_cq_ex	*ibv_cq_ex;
};

int efa_dgram_cq_open(struct fid_domain *domain_fid, struct fi_cq_attr *attr,
		      struct fid_cq **cq_fid, void *context);

ssize_t efa_dgram_cq_readfrom(struct fid_cq *cq_fid, void *buf, size_t count, fi_addr_t *src_addr);

ssize_t efa_dgram_cq_readerr(struct fid_cq *cq_fid, struct fi_cq_err_entry *entry, uint64_t flags);

#endif