/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_CQ_H_
#define _CXIP_CQ_H_


#include <stddef.h>
#include <stdbool.h>
#include <ofi_list.h>

/* Forward declarations */
struct cxip_domain;
struct cxip_req;

/* Macros */
#define CXIP_CQ_DEF_SZ			131072U

/* Type definitions */
struct cxip_cq_eq {
	struct cxi_eq *eq;
	void *buf;
	size_t len;
	struct cxi_md *md;
	bool mmap;
	unsigned int unacked_events;
	struct c_eq_status prev_eq_status;
	bool eq_saturated;
};

struct cxip_cq {
	struct util_cq util_cq;
	struct fi_cq_attr attr;

	/* Implement our own CQ ep_list_lock since common code util_cq
	 * implementation is a mutex and can not be optimized. This lock
	 * is always taken walking the CQ EP, but can be optimized to no-op.
	 */
	struct ofi_genlock ep_list_lock;

	/* CXI CQ wait object EPs are maintained in epoll FD */
	int ep_fd;

	/* CXI specific fields. */
	struct cxip_domain *domain;
	unsigned int ack_batch_size;
	struct dlist_entry dom_entry;
};

struct cxip_fid_list {
	struct dlist_entry entry;
	struct fid *fid;
};

/* Function declarations */
const char *cxip_strerror(int prov_errno);

int cxip_cq_req_complete(struct cxip_req *req);

int cxip_cq_req_complete_addr(struct cxip_req *req, fi_addr_t src);

int cxip_cq_req_error(struct cxip_req *req, size_t olen,
		      int err, int prov_errno, void *err_data,
		      size_t err_data_size, fi_addr_t src_addr);

int cxip_cq_add_wait_fd(struct cxip_cq *cq, int wait_fd, int events);

void cxip_cq_del_wait_fd(struct cxip_cq *cq, int wait_fd);

int proverr2errno(int err);

int cxip_cq_trywait(struct cxip_cq *cq);

void cxip_cq_progress(struct cxip_cq *cq);

void cxip_util_cq_progress(struct util_cq *util_cq);

int cxip_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq, void *context);

void cxip_cq_flush_trig_reqs(struct cxip_cq *cq);

#endif /* _CXIP_CQ_H_ */
