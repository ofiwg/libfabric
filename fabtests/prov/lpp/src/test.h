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

#include "ipc.h"

enum node_id {
	NODE_A,
	NODE_B,
};

extern enum node_id my_node;

typedef int (*testfn)(struct rank_info *ri);

struct test {
	testfn run;
	const char *name;
};

int run_simple_rma_write(struct rank_info *ri);
int run_offset_rma_write(struct rank_info *ri);
int run_inject_rma_write(struct rank_info *ri);
int run_simple_rma_read(struct rank_info *ri);
int run_simple_msg(struct rank_info *ri);
int run_simple_small_msg(struct rank_info *ri);
int run_inject_msg(struct rank_info *ri);
int run_tagged_msg(struct rank_info *ri);
int run_directed_recv_msg(struct rank_info *ri);
int run_large_rma_write(struct rank_info *ri);
int run_large_rma_read(struct rank_info *ri);
int run_multi_recv_msg(struct rank_info *ri);
int run_multi_recv_small_msg(struct rank_info *ri);
int run_unexpected_msg(struct rank_info *ri);
int run_unexpected_multi_recv_msg(struct rank_info *ri);
int run_os_bypass_rma(struct rank_info *ri);
int run_os_bypass_offset_rma(struct rank_info *ri);
int run_os_bypass_outofbounds_rma(struct rank_info *ri);
int run_selective_completion(struct rank_info *ri);
int run_selective_completion2(struct rank_info *ri);
int run_selective_completion_error(struct rank_info *ri);
int run_selective_completion_osbypass_error(struct rank_info *ri);
int run_rsrc_mgmt_cq_overrun(struct rank_info *ri);
int run_rma_write_auto_reg_mr(struct rank_info *ri);
int run_msg_auto_reg_mr(struct rank_info *ri);
int run_small_msg_auto_reg_mr(struct rank_info *ri);
int run_rma_read_auto_reg_mr(struct rank_info *ri);
int run_zero_length(struct rank_info *ri);
int run_loopback_msg(struct rank_info *ri);
int run_loopback_small_msg(struct rank_info *ri);
int run_loopback_write(struct rank_info *ri);
int run_loopback_small_write(struct rank_info *ri);
int run_loopback_read(struct rank_info *ri);
int run_loopback_small_read(struct rank_info *ri);
int run_cq_sread(struct rank_info *ri);
int run_simple_atomic_write(struct rank_info *ri);
int run_simple_atomic_write2(struct rank_info *ri);
int run_simple_atomic_fetch_read(struct rank_info *ri);
int run_simple_atomic_fetch_read2(struct rank_info *ri);
int run_simple_atomic_fetch_write(struct rank_info *ri);
int run_simple_atomic_fetch_write2(struct rank_info *ri);
int run_simple_atomic_cswap(struct rank_info *ri);
int run_simple_atomic_cswap2(struct rank_info *ri);
int run_fi_tsenddata(struct rank_info *ri);
int run_fi_tinjectdata(struct rank_info *ri);
#ifdef USE_CUDA
int run_fi_hmem_cuda_tag_d2d(struct rank_info *ri);
int run_fi_hmem_cuda_sendrecv_d2d(struct rank_info *ri);
#endif
#ifdef USE_ROCM
int run_fi_hmem_rocm_tag_d2d(struct rank_info *ri);
#endif
