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

#include <pattern/user.h>
#include "util.h"

#define PATTERN_API_VERSION_MAJOR 0
#define PATTERN_API_VERSION_MINOR 0

struct pattern_arguments {};

static int parse_arguments(
		const int argc,
		char * const *argv,
		struct pattern_arguments **arguments)
{
	*arguments = NULL;
	return 0;
}

static void free_arguments(struct pattern_arguments *arguments)
{
	return;
}

static int pattern_next(
		const struct pattern_arguments *arguments,
		int my_rank,
		int num_ranks,
		int *cur,
		int *threshold)
{
	int next = *cur + 1;

	if (next >= num_ranks)
		return -ENODATA;

	*cur = next;
	return 0;
}


struct pattern_api a2a_pattern_api(void)
{
	struct pattern_api pattern_api = {
		.parse_arguments = &parse_arguments,
		.free_arguments = &free_arguments,
		.next_sender = &pattern_next,
		.next_receiver = &pattern_next
	};
	return pattern_api;
}
