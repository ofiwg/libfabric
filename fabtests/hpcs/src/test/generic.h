/*
 * Copyright (c) 2017-2018 Intel Corporation. All rights reserved.
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

#include <inttypes.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_eq.h>

#include <test/user.h>

struct test_arguments;

void generic_free_arguments(struct test_arguments *arguments);

void generic_tx_init_buffer(const struct test_arguments *arguments,
		uint8_t *buffer, size_t len);

void generic_rx_init_buffer(const struct test_arguments *arguments,
		uint8_t *buffer, size_t len);

struct fid_mr *generic_create_mr(const struct test_arguments *arguments,
		struct fid_domain *domain,
		const uint64_t key, uint8_t *buffer,
		size_t len, uint64_t access);

int generic_destroy_mr(const struct test_arguments *arguments,
		struct fid_mr *mr);

size_t generic_tx_window_usage(const struct test_arguments *arguments,
		const size_t transfer_id,
		const size_t transfer_count);

size_t generic_rx_window_usage(const struct test_arguments *arguments,
		const size_t transfer_id,
		const size_t transfer_count);

int generic_tx_transfer(const struct test_arguments *arguments,
		const size_t transfer_id, const size_t transfer_count,
		const fi_addr_t rx_address, struct fid_ep *endpoint,
		struct op_context *op_context, uint8_t *buffer,
		void *desc, uint64_t key, int rank, uint64_t flags);

int generic_rx_transfer(const struct test_arguments *arguments,
		const size_t transfer_id, const size_t transfer_count,
		const fi_addr_t tx_address, struct fid_ep *endpoint,
		struct op_context *op_context, uint8_t *buffer,
		void *desc, uint64_t flags);

int generic_tx_cntr_completion(const struct test_arguments *arguments,
		const size_t completion_count, struct fid_cntr *cntr);

int generic_rx_cntr_completion(const struct test_arguments *arguments,
		const size_t completion_count, struct fid_cntr *cntr);

int generic_tx_cq_completion(const struct test_arguments *args,
		struct op_context **context, struct fid_cq *cq);

int generic_rx_cq_completion(const struct test_arguments *args,
		struct op_context **context, struct fid_cq *cq);

int generic_tx_datacheck(const struct test_arguments *arguments,
		const uint8_t *buffer, size_t len);

int generic_rx_datacheck(const struct test_arguments *args,
		const uint8_t *buffer, size_t rx_peers, size_t len);

static int generic_fini_buffer(const struct test_arguments *arguments,
		uint8_t *buffer);
