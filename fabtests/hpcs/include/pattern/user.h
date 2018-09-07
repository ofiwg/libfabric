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


/*
 * -----------------------------------------------------------------------------
 * TRAFFIC PATTERN API
 * -----------------------------------------------------------------------------
 *
 * A pattern can be described as a connectivity matrix between TX and RX
 * endpoints, with TX endpoints as rows and RX endpoints as columns.  Example:
 *
 *     EP0 EP1 EP2
 * EP0  1   0   1
 * EP1  0   1   0
 * EP2  0   0   1
 *
 * The above pattern shows every endpoint sending to itself, with endpoint 0
 * additionaly sending to endpoint 2.
 */

#define PATTERN_API_VERSION_MAJOR 0
#define PATTERN_API_VERSION_MINOR 0

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
 * All calculations required for using a pattern happen on-demand, as buffering
 * the pattern would not scale for large numbers of endpoints.
 *
 * Due to limitations of the C programming language, patterns are generated
 * rather than specified.  The caller provides a TX and RX index pair, and the
 * pattern generator returns the next TX and RX index pair as outputs.
 *
 * There are two kinds of pattern generators that may be implemented: TX (row)
 * major, and RX (column) major.  A pattern implementation must have at least
 * one of them implemented, and implementing both may provide efficiency
 * benefits.
 *
 * The caller may wish to get the first valid index pair in some range.  This
 * can be done by passing in the index pair immediately before the start of the
 * range of interest.  For example, if the caller is interested in TX = [6] and
 * RX = [10-20], the caller may invoke the TX major pattern generator with TX =
 * 6 and RX = 9.  If the caller is interested in TX = [3-4] and RX = [0], they
 * may invoke the TX major pattern generator with TX = 2 and RX = <rx_count> -
 * 1.  Generators will automatically wrap around at the end of the row or
 * column, but may overshoot the submatrix of interest, so callers must be
 * careful to detect if the generator has traversed outside the bounds of
 * interest.
 *
 * arguments: the parsed version of the arguments.
 *
 * size: the number of endpoints in the job (size of the square pattern matrix).
 *
 * current: the pair from which to search for the next valid pair.
 *
 * next: the next valid pair found in the pattern.
 */

typedef int (*pattern_next_sender)(
		const struct pattern_arguments *arguments,
		int my_rank,
		int num_ranks,
		int *cur_sender);

typedef int (*pattern_next_receiver) (
		const struct pattern_arguments *arguments,
		int my_rank,
		int num_ranks,
		int *cur_receiver);

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
};


/* List of patterns defined in source files under pattern directory. */

struct pattern_api a2a_pattern_api(void);
struct pattern_api self_pattern_api(void);
struct pattern_api alltoone_pattern_api(void);
