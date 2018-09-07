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


#pragma once

#include <stdlib.h>
#include <stdbool.h>

#include <rdma/fabric.h>


/*
 * -----------------------------------------------------------------------------
 * TEST API
 * -----------------------------------------------------------------------------
 *
 * The test callbacks define test code to be run a certain number of times and
 * with certain ordering guarantees.  These details are described in the
 * documentation comments for each callback.  A test is therefore written by
 * providing implementations of all the callbacks described here.  The ordering
 * guarantees ensure that test code can be written in a way to avoid deadlock
 * while placing as few constraints on the ordering and completion of callbacks
 * as possible, which allows for callbacks to be executed in a multithreaded
 * way. Users should structure their test callback code with the ordering
 * guarantees in mind to avoid unexpected behavior and deadlocks.
 *
 * A test implementation also defines a configuration that is processed by the
 * caller.  For example, how to size and align buffers.
 */

#define TEST_API_VERSION_MAJOR 0
#define TEST_API_VERSION_MINOR 0

/*
 * USER ARGUMENTS
 *
 * Each test is allowed to take arbitrary user arguments that it defines.  The
 * caller does not do anything with these arguments other than store a pointer
 * to an implementation-defined region of memory that each test privately knows
 * how to interpret.  This pointer is provided in every callback, allowing each
 * test to change its behavior depending on any user-provided arguments.
 *
 * The arguments are provided to a test in the standard C style of command line
 * arguments, although they may not have necessarily come from a command line.
 * The caller will provide each test an opportunity to parse and internally
 * store these arguments prior to calling any other callbacks.
 *
 * Each test is only responsible for allocating the memory it needs to store the
 * parsed version of its arguments, and the caller will free this memory at the
 * end of the test by calling test_free_arguments (when no more callbacks will
 * be called).
 *
 * argc and argv: user arguments specific to this test, after parsing will be
 * passed as "user_arguments" to each callback.
 *
 * arguments: the parsed version of the arguments.
 *
 * buffer_size: output variable to tell core the size of the per-operation
 *		buffer it needs to allocate.
 */

struct test_arguments;

typedef int (*test_parse_arguments)(
	const int argc,
	char * const *argv,
	struct test_arguments **arguments,
	size_t *buffer_size
);

typedef void (*test_free_arguments)(
	struct test_arguments *arguments
);


/*
 * CONFIGURATION
 *
 * Functions and data structures for describing and generating the test
 * configuration.  This configuration is read by the caller to perform correct
 * initialization and to provide the correct resources to each callback.
 *
 * Data object sharing refers to how to distribute unique data objects,
 * depending on the test's requirements.  For tests which wish to really stress
 * high volumes of concurrent send/recv with a very large workload, unique TX
 * and RX data objects probably aren't needed for every unique transfer.  By
 * contrast, for tests which want to do data validity checking, reusing RX data
 * objects isn't feasible, so each transfer might have to have its own unique
 * buffer. The same applies for fetching atomics and FI_READ operations, which
 * may require unique TX data objects.
 */

enum data_object_sharing {
	DATA_OBJECT_PER_TRANSFER,
	DATA_OBJECT_PER_EP_PAIR,
	DATA_OBJECT_PER_ENDPOINT,
	DATA_OBJECT_PER_DOMAIN
};

struct test_config {
	uint64_t minimum_caps;

	bool tx_use_cntr;
	bool rx_use_cntr;
	bool tx_use_cq;
	bool rx_use_cq;

	bool rx_use_mr;

	size_t tx_buffer_size;
	size_t rx_buffer_size;
	size_t tx_buffer_alignment;
	size_t rx_buffer_alignment;

	enum data_object_sharing tx_data_object_sharing;
	enum data_object_sharing rx_data_object_sharing;

	/* number of fi_contexts to allocate for per-iteration state */
	size_t tx_context_count;
	size_t rx_context_count;

	uint64_t mr_rx_flags;
};

typedef struct test_config (*test_config)(
	const struct test_arguments *arguments
);

enum op_state {
	DONE = 0,
	PENDING
};

struct context_info {
	struct fi_context	fi_context;
	struct op_context	*op_context;
};

struct op_context {
	struct context_info	*ctxinfo;
	enum op_state		state;
	uint8_t			*buf;
	uint64_t		core_context;
	uint64_t		test_context;
	struct fid_mr		*tx_mr;
	uint64_t		test_state; /* reserved for test internal accounting */
};

/*
 * DATA OBJECT SETUP
 *
 * Setup any required data objects, including filling the buffers with initial
 * data, creating memory regions, and associating buffers with the memory
 * regions.  The memory backing these data objects is allocated and freed by the
 * caller.
 *
 * If the test configuration specifies that a memory region should be used for
 * RX, then in addition to test_rx_init_buffer, a corresponding
 * test_rx_create_mr will also be called.  test_tx_create_mr will be called if
 * the provider sets the FI_MR_LOCAL mode bit.
 *
 * arguments: the parsed version of the arguments.
 *
 * buffer: the buffer to initialize (and associate with a memory region).
 *
 * key: the key with which to create the memory region (decided automatically by
 * the caller).
 *
 * buffer: the buffer to initialize and possibly associate with a memory region
 * (allocated automatically by the caller).
 */

typedef void (*test_tx_init_buffer)(
	const struct test_arguments *arguments,
	uint8_t *buffer
);

typedef void (*test_rx_init_buffer)(
	const struct test_arguments *arguments,
	uint8_t *buffer
);

typedef struct fid_mr *(*test_tx_create_mr)(
	const struct test_arguments *arguments,
	struct fid_domain *domain,
	const uint64_t key,
	uint8_t *buffer,
	size_t len,
	uint64_t access
);

typedef struct fid_mr *(*test_rx_create_mr)(
	const struct test_arguments *arguments,
	struct fid_domain *domain,
	const uint64_t key,
	uint8_t *buffer,
	size_t len,
	uint64_t access
);


/*
 * DATA TRANSFER WINDOW USAGE
 *
 * Returns a number corresponding to the amount of a window that a given
 * transfer callback is expected to consume locally.  Effectively, this maps to
 * the number of contexts and completion queue slots that will be consumed.
 * Users who wish to use selective completion may return 0 here instead of 1,
 * which will result in no context memory to be provided and an expectation that
 * no completion queue slots will be used.  The result of these are used to
 * determine the start and end of each transfer window.  Completion queue slot
 * resource exhaustion can be achieved by causing more completions to actually
 * generate than what are reported by these calls, which would require
 * requesting a context size sufficient for multiple contexts per transfer.
 *
 * arguments: the parsed version of the arguments.
 *
 * transfer_id: uniquely identifies the transfer between a pair of endpoints,
 * useful for creating triggered chains and using as messaging tags.
 *
 * transfer_count: the total number of transfers that will occur between a pair
 * of endpoints.
 */

typedef size_t (*test_tx_window_usage)(
	const struct test_arguments *arguments,
	const size_t transfer_id,
	const size_t transfer_count
);

typedef size_t (*test_rx_window_usage)(
	const struct test_arguments *arguments,
	const size_t transfer_id,
	const size_t transfer_count
);


/*
 * DATA TRANSFER
 *
 * Execute a single unit of the TX or RX functionality under test.  If the test
 * is being run with unexpected messaging, then each TX call is guaranteed to
 * finish before the corresponding RX call starts, and vice-versa for running
 * with expected messaging.  The expected completion use case is that each call
 * will cause a single completion to be generated for each kind of completion
 * object that exists.  Users who wish to do multiple operations or multiple
 * transfers as a single "unit" should map these to individual transfer
 * callbacks and make use of the provided transfer_id to differentiate between
 * them.
 *
 * arguments: the parsed version of the arguments.
 *
 * transfer_id: uniquely identifies the transfer between a pair of endpoints,
 * useful for creating triggered chains and using as messaging tags.
 *
 * transfer_count: the total number of transfers that will occur between a pair
 * of endpoints.
 *
 * rx_address: the fabric address of the RX endpoint.
 *
 * endpoint: the local endpoint from which to initiate or receive the transfer.
 *
 * context: context requested for the operation, will be set to NULL if no
 * contexts were requested.
 *
 * buffer: the buffer from which to send or into which data will be received.
 *
 * desc: the memory descriptor associated with the data buffer.
 *
 * rank: sender's rank (used to compute offset in one-sided communication).
 *
 * offset: memory write offset, if needed.
 *
 * tx_cntr: the local endpoint's TX counter, provided for setting
 * up triggered chains.
 *
 * rx_cntr: the local endpoint's RX counter, provided for setting up
 * triggered chains.
 */

typedef int (*test_tx_transfer)(
	const struct test_arguments *arguments,
	const size_t transfer_id,
	const size_t transfer_count,
	const fi_addr_t rx_address,
	struct fid_ep *endpoint,
	struct op_context *op_context,
	uint8_t *buffer,
	void *desc,
	uint64_t key,
	int rank,
	struct fid_cntr *tx_cntr,
	struct fid_cntr *rx_cntr
);

typedef int (*test_rx_transfer)(
	const struct test_arguments *arguments,
	const size_t transfer_id,
	const size_t transfer_count,
	const fi_addr_t tx_address,
	struct fid_ep *endpoint,
	struct op_context *op_context,
	uint8_t *buffer,
	void *desc,
	struct fid_cntr *tx_cntr,
	struct fid_cntr *rx_cntr
);


/*
 * DATA TRANSFER COMPLETION
 *
 * Called after every transfer window to process completions generated during
 * that window.  The frequency at which these are called are determined by
 * options which affect the window size.  Called once per completion object.
 *
 * arguments: the parsed version of the arguments.
 *
 * completion_count: the number of completions that occurred during the last
 * transfer window.
 *
 * cntr: the TX or RX counter from which to consume counting completions.
 *
 * cq: the TX or RX completion queue from which to consume completion events.
 */

typedef int (*test_tx_cntr_completion)(
	const struct test_arguments *arguments,
	const size_t completion_count,
	struct fid_cntr *cntr
);

typedef int (*test_rx_cntr_completion)(
	const struct test_arguments *arguments,
	const size_t completion_count,
	struct fid_cntr *cntr
);

typedef int (*test_tx_cq_completion)(
	const struct test_arguments *arguments,
	struct op_context **op_context,
	struct fid_cq *cq
);

typedef int (*test_rx_cq_completion)(
	const struct test_arguments *arguments,
	struct op_context **op_context,
	struct fid_cq *cq
);


/*
 * DATA OBJECT PROCESSING
 *
 * Called once per data object on each process after all data transfers have
 * completed.  Performs RX data validity checking, and TX data validity checking
 * in the case of a fetching atomic or an RMA read.
 *
 * arguments: the parsed version of the arguments.
 *
 * transfer_count: the total number of transfers that occurred between each pair
 * of endpoints.
 *
 * buffer: the buffer into which data was received or from which data was sent.
 *
 * rx_peers: number of peers we're receiving from in this pattern (useful for
 * validating atomic add).
 */

typedef int (*test_tx_datacheck)(
	const struct test_arguments *arguments,
	const uint8_t *buffer
);

typedef int (*test_rx_datacheck)(
	const struct test_arguments *arguments,
	const uint8_t *buffer,
	size_t rx_peers
);


/*
 * DATA OBJECT TEARDOWN
 *
 * Undo any setup done during the setup_buffer or setup_mr calls.  These are
 * called after the data object in question is done being used by the data
 * transfer calls.  The memory backing the data objects will be freed
 * automatically.
 *
 * arguments: the parsed version of the arguments.
 *
 * buffer: pointer to the memory backing the buffer or memory region to
 * finialize.
 *
 * mr: the memory region to destroy.
 */

typedef int (*test_tx_fini_buffer)(
	const struct test_arguments *arguments,
	uint8_t *buffer
);

typedef int (*test_rx_fini_buffer)(
	const struct test_arguments *arguments,
	uint8_t *buffer
);

typedef int (*test_tx_destroy_mr)(
	const struct test_arguments *arguments,
	struct fid_mr *mr
);

typedef int (*test_rx_destroy_mr)(
	const struct test_arguments *arguments,
	struct fid_mr *mr
);


/*
 * CALLER INTERFACE
 *
 * The caller receives the callbacks as function pointers in the struct
 * test_api.  The caller will call the test_api function, which must be defined
 * by the test as a global symbol.  A declaration of it is provided here that
 * the test must define.  If a test's configuration is such that certain
 * callbacks will not be called, then setting those fields to a NULL pointer is
 * acceptable.
 */

struct test_api {

	test_parse_arguments parse_arguments;
	test_free_arguments free_arguments;

	test_config config;

	test_tx_init_buffer tx_init_buffer;
	test_rx_init_buffer rx_init_buffer;
	test_tx_create_mr tx_create_mr;
	test_rx_create_mr rx_create_mr;

	test_tx_window_usage tx_window_usage;
	test_rx_window_usage rx_window_usage;

	test_tx_transfer tx_transfer;
	test_rx_transfer rx_transfer;

	test_tx_cntr_completion tx_cntr_completion;
	test_rx_cntr_completion rx_cntr_completion;
	test_tx_cq_completion tx_cq_completion;
	test_rx_cq_completion rx_cq_completion;

	test_tx_datacheck tx_datacheck;
	test_rx_datacheck rx_datacheck;

	test_tx_fini_buffer tx_fini_buffer;
	test_rx_fini_buffer rx_fini_buffer;
	test_tx_destroy_mr tx_destroy_mr;
	test_rx_destroy_mr rx_destroy_mr;
};

struct test_api test_api (void);
