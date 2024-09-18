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

#include <sys/uio.h>
#include <stdbool.h>
#include <stddef.h>
#include <dlfcn.h>

#include "test.h"
#include "ipc.h"
#include "error.h"

#ifndef FI_OS_BYPASS
#define FI_OS_BYPASS               (1ULL << 63)
#endif

#define SEND_AND_INSIST_EQ(_ri, _tmp, _func, _ret) do {		\
	do {							\
		_tmp = _func;					\
	} while (_tmp == -FI_EAGAIN);				\
	INSIST_FI_EQ(_ri, _tmp, _ret);				\
} while (0)

struct mr_params {
	uint64_t idx;
	uint64_t length;
	uint64_t access;
	unsigned int seed;
	bool page_align;
	bool skip_reg;
	enum fi_hmem_iface hmem_iface;
};

struct ep_params {
	uint64_t idx;
	uint64_t cq_bind_flags;
	uint64_t cq_size;
	struct {
		uint64_t additional_caps;
		uint64_t op_flags;
	} tx_attr;
	struct {
		uint64_t additional_caps;
		uint64_t op_flags;
	} rx_attr;
};

struct wait_tx_cq_params {
	uint64_t ep_idx;
	uint64_t context_val;
	uint64_t flags;
	uint64_t data;
	bool expect_error;
	bool expect_empty;
};

struct wait_rx_cq_params {
	uint64_t ep_idx;
	uint64_t context_val;
	uint64_t flags;
	uint64_t data;
	bool expect_error;
	bool expect_empty;
	bool multi_recv;
	size_t buf_offset;
};

enum wait_cntr_which {
	WAIT_CNTR_INVALID = 0,
	WAIT_CNTR_TX,
	WAIT_CNTR_RX,
};

struct wait_cntr_params {
	uint64_t ep_idx;
	uint64_t val;
	enum wait_cntr_which which;
};

struct verify_buf_params {
	uint64_t mr_idx;
	uint64_t offset;
	uint64_t length;
	uint64_t expected_seed;
	uint64_t expected_seed_offset;
};

void util_global_init();
void util_init(struct rank_info *ri);
void util_teardown(struct rank_info *ri, struct rank_info *pri);
void util_sync(struct rank_info *ri, struct rank_info **pri);
void util_barrier(struct rank_info *ri);
void util_create_mr(struct rank_info *ri, struct mr_params *params);
void util_create_ep(struct rank_info *ri, struct ep_params *params);
void util_wait_tx_cq(struct rank_info *ri, struct wait_tx_cq_params *params);
void util_wait_rx_cq(struct rank_info *ri, struct wait_rx_cq_params *params);
void util_verify_buf(struct rank_info *ri, struct verify_buf_params *params);
char *util_verify_buf2(struct rank_info *ri, struct verify_buf_params *params);
void util_simple_setup(struct rank_info *ri, struct rank_info **pri,
		       size_t length, uint64_t lcl_access, uint64_t rem_access);
uint64_t util_read_tx_cntr(struct rank_info *ri, uint64_t ep_idx);
uint64_t util_read_rx_cntr(struct rank_info *ri, uint64_t ep_idx);
void util_av_insert_all(struct rank_info *ri, struct rank_info *tgt_ri);
void util_wait_cntr_many(struct rank_info *ri,
			 struct wait_cntr_params *paramslist, size_t n_params);
void util_wait_cntr(struct rank_info *ri, struct wait_cntr_params *params);
void util_get_time(struct timespec *ts);
uint64_t util_time_delta_ms(struct timespec *start, struct timespec *end);
void util_validate_cq_entry(struct rank_info *ri,
			    struct fi_cq_tagged_entry *tentry,
			    struct fi_cq_err_entry *errentry, uint64_t flags,
			    uint64_t data, uint64_t context_val, bool multi_recv,
			    uint64_t buf_offset);
static inline bool util_is_write_only() {
	char *str_value = getenv("FI_LPP_WRITE_ONLY");

	if (!str_value)
		return true;

	if (strcmp(str_value, "0") == 0 ||
	    strcasecmp(str_value, "false") == 0 ||
	    strcasecmp(str_value, "no") == 0 ||
	    strcasecmp(str_value, "off") == 0)
		return false;

	return true;
}

#define CTX_NOMR ((uint64_t)(-1))
struct context *get_ctx(struct rank_info *ri, uint64_t context_val,
			uint64_t mr_idx);
static inline struct fi_context *get_ctx_simple(struct rank_info *ri,
						uint64_t context_val)
{
	struct context *context = get_ctx(ri, context_val, CTX_NOMR);
	return &context->fi_context;
}
void free_ctx_tree(struct rank_info *ri);

static const unsigned int seed_node_a = 1234;
static const unsigned int seed_node_b = 9876;

#ifdef USE_HMEM
void hmem_init(void);
void hmem_cleanup(void);
int hmem_alloc(enum fi_hmem_iface iface, void *uaddr, size_t len);
void hmem_free(enum fi_hmem_iface iface, void *uaddr);
int hmem_memcpy_h2d(enum fi_hmem_iface iface, void *dst, void *src, size_t len);
int hmem_memcpy_d2h(enum fi_hmem_iface iface, void *dst, void *src, size_t len);
#else
static inline void hmem_init(void){}
static inline void hmem_cleanup(void){}
static inline int hmem_alloc(enum fi_hmem_iface iface, void *uaddr, size_t len){return -ENOSYS;}
static inline void hmem_free(enum fi_hmem_iface iface, void *uaddr){}
static inline int hmem_memcpy_h2d(enum fi_hmem_iface iface, void *dst, void *src, size_t len){return -ENOSYS;}
static inline int hmem_memcpy_d2h(enum fi_hmem_iface iface, void *dst, void *src, size_t len){return -ENOSYS;}
#endif
