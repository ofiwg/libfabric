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

/*
 * The ring pattern creates a chain of rx-triggered sends, such that rank
 * R sends a message to rank (R+1)%N after receiving from rank (N-1)%N.
 *
 * The pattern assigns one rank as the leader (rank 0 by default), and the
 * final message in the chain is receied by the leader, forming a ring.
 *
 * (Explicitly setting the leader as -1 is allowed, in which case there
 * is no leader.  This is expected to deadlock, which is a way to verify
 * that triggered ops aren't firing prematurely.)
 *
 * The pattern may execute up to N rings simultaneously.  Each ring has a
 * different leader, chosen sequentially.  The --multi-ring argument
 * simply runs N rings, with every rank being the leader of a single ring.
 *
 * The pattern does not allow any rank to be the leader of more than one
 * ring.
 */

#include <stdio.h>
#include <stdint.h>
#include <getopt.h>
#include <rdma/fi_errno.h>

#include <pattern/user.h>
#include <core/user.h>
#include "util.h"

#define PATTERN_API_VERSION_MAJOR 0
#define PATTERN_API_VERSION_MINOR 0

/*
 * Leader is the node that sends first in a triggered ops case.  If leader
 * is -1, there is no leader.
 *
 * Rings is the number of concurrent rings (the leader of each subsequent ring
 * is the leader's rank plus one mod N).  If rings is -1, the number of rings
 * will equal the number of ranks.
 */
struct pattern_arguments {
	int leader;
	int rings;
	int verbose;
};

static int ring_parse_arguments(
		const int argc,
		char * const *argv,
		struct pattern_arguments **arguments)
{
	int longopt_idx=0, op;
	static struct option longopt[] = {
		{"leader", required_argument, 0, 'l'},
		{"rings", required_argument, 0, 'r'},
		{"multi-ring", no_argument, 0, 'm'},
		{"verbose", no_argument, 0, 'v'},
		{"help", no_argument, 0, 'h'},
		{0}
	};

	int have_rings=0, have_multi_ring=0;

	struct pattern_arguments *args = calloc(sizeof(struct pattern_arguments), 1);
	if (args == NULL)
		return -FI_ENOMEM;

	*args = (struct pattern_arguments) {.leader = 0, .rings = 1};

	if (argc > 0 && argv != NULL) {
		while ((op = getopt_long(argc, argv, "l:r:mh", longopt, &longopt_idx)) != -1) {
			switch (op) {
			case 'l':
				if (sscanf(optarg, "%d", &args->leader) != 1) {
					hpcs_error("unable to parse --leader argument\n");
					return -EINVAL;
				}
				if (args->leader < 0)
					hpcs_error("ring pattern has no leader; deadlock expected if triggered ops are working correctly\n");
				break;
			case 'r':
				if (sscanf(optarg, "%u", &args->rings) != 1) {
					hpcs_error("unable to parse --rings argument\n");
					return -EINVAL;
				}
				have_rings = 1;
				break;
			case 'm':
				args->rings = -1;
				have_multi_ring = 1;
				break;
			case 'v':
				args->verbose = 1;
				break;
			case 'h':
			default:
				fprintf(stderr, "<pattern arguments> :=\n"
						"\t[-l | --leader=<rank>]\n"
						"\t[-r | --rings=<rings> | -m | --multi-ring]\n"
						"\t[-v | --verbose]\n"
						"\t[-h | --help]\n");
				return -EINVAL;
				break;
			}
		}
	}

	if (have_rings && have_multi_ring) {
		hpcs_error("--rings and --multi-ring arguments are mutually exclusive\n");
		return -EINVAL;
	}

	*arguments = args;
	return 0;
}

static void ring_free_arguments(struct pattern_arguments *arguments)
{
	free(arguments);
}

/*
 * Note: this does not handle the case where rings > num_ranks.
 */
static inline int ring_pattern_next(int is_sender,
		const struct pattern_arguments *arguments,
		int my_rank,
		int num_ranks,
		int *cur,
		int *threshold)
{
	int leader = arguments->leader;
	int rings = arguments->rings < 0
			? num_ranks
			: arguments->rings;
	int is_leader, max_threshold;

	/*
	 * Current node is a leader of some ring if it's within the
	 * first N ranks beginning with the leader (wrapping around),
	 * where N is the total number of rings.
	 */

	if (leader < 0)
		is_leader = 0;
	else if (leader + rings <= num_ranks)
		is_leader = my_rank >= leader && my_rank < leader + rings;
	else
		is_leader = my_rank >= leader || my_rank < (leader + rings) - num_ranks;

	max_threshold = is_leader ? rings - 1 : rings;

	if (rings > num_ranks) {
		hpcs_error("Ring pattern does not support a number of rings that exceeds number of ranks.\n");
		return -EINVAL;
	}

	if (arguments->verbose) {
		printf("%d ring_pattern_next_%s(... %d, %d) leader:%d rings:%d is_leader:%d max_threshold:%d\n",
				my_rank,
				is_sender ? "sender" : "receiver",
				*cur, *threshold,
				leader, rings, is_leader, max_threshold);
	}

	if (*cur == PATTERN_NO_CURRENT) {
		int peer_rank = (my_rank + (is_sender ? num_ranks - 1 : 1)) % num_ranks;

		*cur = peer_rank % num_ranks;
		*threshold = is_leader ? 0 : 1;
	} else if (*threshold < max_threshold) {
		*threshold = *threshold + 1;
	} else {
		if (arguments->verbose)
			printf("\t-> done\n");
		return -ENODATA;
	}

	if (arguments->verbose)
		printf("\t-> cur:%d, threshold:%d\n", *cur, *threshold);
	return 0;
}

static int ring_pattern_next_sender(
		const struct pattern_arguments *arguments,
		int my_rank,
		int num_ranks,
		int *cur,
		int *threshold)
{
	return ring_pattern_next(1, arguments, my_rank, num_ranks, cur, threshold);
}


static int ring_pattern_next_receiver(
		const struct pattern_arguments *arguments,
		int my_rank,
		int num_ranks,
		int *cur,
		int *threshold)
{
	return ring_pattern_next(0, arguments, my_rank, num_ranks, cur, threshold);
}


struct pattern_api ring_pattern_api(void)
{
	struct pattern_api pattern_api = {
		.parse_arguments = &ring_parse_arguments,
		.free_arguments = &ring_free_arguments,
		.next_sender = &ring_pattern_next_sender,
		.next_receiver = &ring_pattern_next_receiver,
		.enable_triggered = 1
	};
	return pattern_api;
}
