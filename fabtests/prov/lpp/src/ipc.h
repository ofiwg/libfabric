/*
 * Copyright (c) 2018-2024 GigaIO, Inc. All Rights Reserved.
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

#pragma once

#include <stdint.h>
#include <assert.h>
#include <pthread.h>

#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_atomic.h>
#include <rdma/fabric.h>

#include "error.h"

#define MAX_RANK 1024
#define MAX_MR_INFO 16
#define MAX_EP_INFO 16

struct mr_info {
	int valid;
	int skip_reg;
	uint64_t key;
	struct fid_mr *fid;
	void *uaddr;
	uint64_t length;
	unsigned int orig_seed;
	enum fi_hmem_iface hmem_iface;
};

struct ep_info {
	int valid;
	struct fid_ep *fid;
	struct fid_stx *stx;
	struct fid_cq *tx_cq_fid;
	struct fid_cq *rx_cq_fid;

	struct fid_cntr *tx_cntr_fid;
	struct fid_cntr *rx_cntr_fid;

	char name[128];
	fi_addr_t fi_addr;

	struct fid_av *av;
};

struct context {
	struct fi_context fi_context;
	uint64_t context_val;
	uint64_t mr_idx;
};

struct rank_info {
	int valid;
	uint64_t rank;
	int64_t iteration;
	pthread_mutex_t lock;
	_Atomic int peer_comm_sock;

	const char *cur_test_name;
	int cur_test_num;

	struct fid_fabric *fabric;
	struct fid_domain *domain;
	struct fi_info *fi;

	struct context sync_op_context;
	void *context_tree_root;

	int n_mr_info;
	struct mr_info mr_info[MAX_MR_INFO];

	int n_ep_info;
	struct ep_info ep_info[MAX_EP_INFO];

	int tracei;
	int trace_lines[MAXTRACE];
	const char *trace_files[MAXTRACE];
	const char *trace_funcs[MAXTRACE];
};

void server_init(const char *peerhostname, int server_port);
struct rank_info *get_rank_info(uint64_t rank, int64_t iteration);
void put_rank_info(struct rank_info *ri);
struct rank_info *exchange_rank_info(struct rank_info *ri);
void put_peer_rank_info(struct rank_info *pri);
void wait_for_peer_up(struct rank_info *ri);
void announce_peer_up(struct rank_info *ri, int64_t iteration);
void peer_barrier(struct rank_info *ri);
