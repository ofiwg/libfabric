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

#include <stdio.h>
#include <stdint.h>
#include <getopt.h>
#include <rdma/fi_errno.h>

#include <pattern/user.h>
#include "util.h"

#define PATTERN_API_VERSION_MAJOR 0
#define PATTERN_API_VERSION_MINOR 0

struct pattern_arguments {
	uint64_t target_rank;
};

static int ato_parse_arguments(
		const int argc,
		char * const *argv,
		struct pattern_arguments **arguments)
{
	int longopt_idx=0, op;
	static struct option longopt[] = {
		{"target-rank", required_argument, 0, 't'},
		{"help", no_argument, 0, 'h'},
		{0}
	};

	struct pattern_arguments *args = calloc(sizeof(struct pattern_arguments), 1);
	if (args == NULL)
		return -FI_ENOMEM;

	*args = (struct pattern_arguments) {.target_rank = 0};

	if (argc > 0 && argv != NULL) {
		while ((op = getopt_long(argc, argv, "t:h", longopt, &longopt_idx)) != -1) {
			switch (op) {
			case 't':
				if (sscanf(optarg, "%zu", &args->target_rank) != 1)
					return -FI_EINVAL;
				break;
			case 'h':
			default:
				fprintf(stderr, "<pattern arguments> :=\n"
						"\t[-t | --target-rank=<rank>]\n"
						"\t[-h | --help]\n");
				return -FI_EINVAL;
				break;
			}
		}
	}

	*arguments = args;
	return 0;
}

static void ato_free_arguments(struct pattern_arguments *arguments)
{
	return;
}

static int ato_pattern_next_sender(
		const struct pattern_arguments *arguments,
		int my_rank,
		int num_ranks,
		int *cur)
{
	if (my_rank == arguments->target_rank){
		int next = *cur + 1;

		if (next >= num_ranks)
			return -1;

		*cur = next;
		return 0;
	} else {
		return -1;
	}
}

static int ato_pattern_next_receiver(
		const struct pattern_arguments *arguments,
		int my_rank,
		int num_ranks,
		int *cur)
{
	if (*cur == PATTERN_NO_CURRENT) {
		*cur = arguments->target_rank;
		return 0;
	}

	return -1;
}


struct pattern_api alltoone_pattern_api(void)
{
	struct pattern_api pattern_api = {
		.parse_arguments = &ato_parse_arguments,
		.free_arguments = &ato_free_arguments,
		.next_sender = &ato_pattern_next_sender,
		.next_receiver = &ato_pattern_next_receiver
	};
	return pattern_api;
}
