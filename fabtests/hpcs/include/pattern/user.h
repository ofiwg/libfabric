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
#include <errno.h>

/*
 * -----------------------------------------------------------------------------
 * TRAFFIC PATTERN API
 * -----------------------------------------------------------------------------
 *
 * A pattern can be described as a connectivity matrix between TX and RX
 * endpoints, with TX endpoints as rows and RX endpoints as columns.  Example:
 *
 *                         receivers
 *
 *                        EP0 EP1 EP2
 *                    EP0  1   0   1
 *        senders     EP1  0   1   0
 *                    EP2  0   0   1
 *
 * The above pattern shows every endpoint sending to itself, with endpoint 0
 * additionaly sending to endpoint 2.
 */

/* Initial value for iterator position. */
#define PATTERN_NO_CURRENT (-1)

/*
 * USER ARGUMENTS
 *
 * Each pattern is allowed to take arbitrary user arguments that it defines.
 * The caller does not do anything with these arguments other than store a
 * pointer to an implementation-defined region of memory that each pattern
 * privately knows how to interpret.  This pointer is provided in every
 * callback, allowing each pattern to change its behavior depending on any
 * user-provided arguments.
 *
 * The arguments are provided to a pattern in the standard C style of command
 * line arguments, although they may not have necessarily come from a command
 * line. The caller will provide each pattern an opportunity to parse and
 * internally store these arguments prior to calling any other callbacks.
 *
 * Each pattern is responsible for allocating and freeing the memory it needs to
 * store the parsed version of its arguments.  The caller will call
 * pattern_free_arguments when the pattern is done being used (when no more
 * callbacks will be called).
 *
 * argc and argv: user arguments specific to this pattern, after parsing will be
 * passed as "arguments" to each callback.
 *
 * arguments: the parsed version of the arguments.
 */

struct pattern_arguments;

typedef int(*pattern_parse_arguments)(
		const int argc,
		char * const *argv,
		struct pattern_arguments **arguments);

typedef void (*pattern_free_arguments)(
		struct pattern_arguments *arguments);


/*
 * TRAFFIC PATTERN GENERATION
 *
 * Patterns are generated using a pair of iterators.  One is used by the
 * sender logic to iterate across all receivers it will be sending something
 * to, and the other is used by the receiver logic to iterate across all the
 * senders that it expects something from.
 *
 * arguments: the parsed version of the arguments.
 *
 * my_rank: the rank of the local process.
 *
 * num_ranks: the number of ranks in the job.
 *
 * cur_[sender,receiver]: the pair from which to search for the next valid pair.
 *
 * threshold: triggered op threshold (we don't need to trigger receives, but
 *	patterns may nevertheless want to use the threshold as an internal
 *	counter).
 *
 * return value: 0 in the normal, non-error case, -ENODATA if iterator is done,
 *      some other negative error number if an error occurred.
 */

typedef int (*pattern_next_sender)(
		const struct pattern_arguments *arguments,
		int my_rank,
		int num_ranks,
		int *cur_sender,
		int *threshold);

typedef int (*pattern_next_receiver) (
		const struct pattern_arguments *arguments,
		int my_rank,
		int num_ranks,
		int *cur_receiver,
		int *threshold);

/*
 * CALLER INTERFACE
 *
 * The caller receives the callbacks as function pointers in the struct
 * pattern_api.  The caller will call the pattern_api function, which must be
 * defined by the pattern as a global symbol.  A declaration of it is provided
 * here that the pattern must define.  The pattern may set either tx_major or
 * rx_major to NULL if it only has an implementation for one of them.
 */

struct pattern_api {
	pattern_parse_arguments parse_arguments;
	pattern_free_arguments free_arguments;
	pattern_next_sender next_sender;
	pattern_next_receiver next_receiver;
	bool enable_triggered;
};


/* List of patterns defined in source files under pattern directory. */

struct pattern_api a2a_pattern_api(void);
struct pattern_api self_pattern_api(void);
struct pattern_api alltoone_pattern_api(void);
struct pattern_api ring_pattern_api(void);
