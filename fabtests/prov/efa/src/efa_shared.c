/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include <stdbool.h>
#include <stdlib.h>
#include <getopt.h>

#include <shared.h>
#include "efa_shared.h"

static struct option efa_extra_opts[] = {
	{"high-pps", no_argument, NULL, OPT_HIGH_PPS},
	{"post-list", required_argument, NULL, OPT_POST_LIST},
	{"num-eps", required_argument, NULL, OPT_NUM_EPS},
	{0, 0, 0, 0}
};

struct option *efa_long_opts;

void build_efa_long_opts(void)
{
	static bool initialized = false;
	int shared_cnt, i;
	int extra_cnt = sizeof(efa_extra_opts) / sizeof(efa_extra_opts[0]) - 1;

	if (initialized)
		return;

	for (shared_cnt = 0; long_opts[shared_cnt].name; shared_cnt++)
		;
	efa_long_opts = calloc(shared_cnt + extra_cnt + 1, sizeof(struct option));
	for (i = 0; i < extra_cnt; i++)
		efa_long_opts[i] = efa_extra_opts[i];
	for (i = 0; i < shared_cnt; i++)
		efa_long_opts[extra_cnt + i] = long_opts[i];

	initialized = true;
}

void efa_longopts_usage(void)
{
	FT_PRINT_OPTS_USAGE("--high-pps",
		"Enable FI_EFA_WR_HIGH_PPS flag on writes");
	FT_PRINT_OPTS_USAGE("--post-list <n>",
		"Batch n posts per doorbell using FI_MORE (default: 1)");
	FT_PRINT_OPTS_USAGE("-q <n>, --num-eps <n>",
		"Number of endpoints/QPs (default: 1)");
	ft_longopts_usage();
}
