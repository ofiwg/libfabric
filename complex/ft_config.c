/*
 * Copyright (c) 2015 Intel Corporation.  All rights reserved.
 *
 * This software is available to you under the BSD license below:
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AWV
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <string.h>

#include "fabtest.h"


#define FT_CAP_MSG	FI_MSG | FI_SEND | FI_RECV
#define FT_CAP_TAGGED	FI_TAGGED | FI_SEND | FI_RECV
#define FT_CAP_RMA	FI_RMA | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE
#define FT_CAP_ATOMIC	FI_ATOMICS | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE

#define FT_MODE_ALL	/*FI_CONTEXT |*/ FI_LOCAL_MR | FI_PROV_MR_ATTR /*| FI_MSG_PREFIX*/
#define FT_MODE_NONE	~0ULL


static struct ft_set test_sets[] = {
	{
		.node = "127.0.0.1",
		.service = "2224",
		.prov_name = "sockets",
		.test_type = {
			FT_TEST_LATENCY
		},
		.class_function = {
			FT_FUNC_SEND,
			FT_FUNC_SENDV,
			FT_FUNC_SENDMSG
		},
		.ep_type = {
			FI_EP_MSG,
			FI_EP_DGRAM,
			FI_EP_RDM
		},
		.comp_type = {
			FT_COMP_QUEUE
		},
		.mode = {
			FT_MODE_ALL
		},
		.caps = {
			FT_CAP_MSG,
			FT_CAP_TAGGED,
//			FT_CAP_RMA,
//			FT_CAP_ATOMIC
		},
		.test_flags = FT_FLAG_QUICKTEST
	},
};

static struct ft_series test_series;

size_t sm_size_array[] = {
		1 << 0,
		1 << 1,
		1 << 2, (1 << 2) + (1 << 1),
		1 << 3, (1 << 3) + (1 << 2),
		1 << 4, (1 << 4) + (1 << 3),
		1 << 5, (1 << 5) + (1 << 4),
		1 << 6, (1 << 6) + (1 << 5),
		1 << 7, (1 << 7) + (1 << 6),
		1 << 8
};
const unsigned int sm_size_cnt = (sizeof sm_size_array / sizeof sm_size_array[0]);

size_t med_size_array[] = {
		1 <<  4,
		1 <<  5,
		1 <<  6,
		1 <<  7, (1 <<  7) + (1 <<  6),
		1 <<  8, (1 <<  8) + (1 <<  7),
		1 <<  9, (1 <<  9) + (1 <<  8),
		1 << 10, (1 << 10) + (1 <<  9),
		1 << 11, (1 << 11) + (1 << 10),
		1 << 12, (1 << 12) + (1 << 11),
		1 << 13, (1 << 13) + (1 << 12),
		1 << 14
};
const unsigned int med_size_cnt = (sizeof med_size_array / sizeof med_size_array[0]);

size_t lg_size_array[] = {
		1 <<  4,
		1 <<  5,
		1 <<  6,
		1 <<  7, (1 <<  7) + (1 <<  6),
		1 <<  8, (1 <<  8) + (1 <<  7),
		1 <<  9, (1 <<  9) + (1 <<  8),
		1 << 10, (1 << 10) + (1 <<  9),
		1 << 11, (1 << 11) + (1 << 10),
		1 << 12, (1 << 12) + (1 << 11),
		1 << 13, (1 << 13) + (1 << 12),
		1 << 14, (1 << 14) + (1 << 13),
		1 << 15, (1 << 15) + (1 << 14),
		1 << 16, (1 << 16) + (1 << 15),
		1 << 17, (1 << 17) + (1 << 16),
		1 << 18, (1 << 18) + (1 << 17),
		1 << 19, (1 << 19) + (1 << 18),
		1 << 20, (1 << 20) + (1 << 19),
		1 << 21, (1 << 21) + (1 << 20),
		1 << 22, (1 << 22) + (1 << 21),
};
const unsigned int lg_size_cnt = (sizeof lg_size_array / sizeof lg_size_array[0]);


/*
 * TODO: Parse configuration file.
 */
struct ft_series *fts_load(char *filename)
{
	if (filename)
		printf("Using static tests. Ignoring config file %s\n", filename);

	test_series.sets = test_sets;
	test_series.nsets = sizeof(test_sets) / sizeof(test_sets[0]);

	for (fts_start(&test_series, 0); !fts_end(&test_series, 0);
	     fts_next(&test_series))
		test_series.test_count++;
	fts_start(&test_series, 0);

	printf("Test configurations loaded: %d\n", test_series.test_count);
	return &test_series;
}

void fts_close(struct ft_series *series)
{
}

void fts_start(struct ft_series *series, int index)
{
	series->cur_set = 0;
	series->cur_type = 0;
	series->cur_ep = 0;
	series->cur_comp = 0;
	series->cur_mode = 0;
	series->cur_caps = 0;

	series->test_index = 1;
	if (index > 1) {
		for (; !fts_end(series, index - 1); fts_next(series))
			;
	}
}

void fts_next(struct ft_series *series)
{
	struct ft_set *set;

	if (fts_end(series, 0))
		return;

	series->test_index++;
	set = &series->sets[series->cur_set];

	if (set->caps[++series->cur_caps])
		return;
	series->cur_caps = 0;

	if (set->mode[++series->cur_mode])
		return;
	series->cur_mode = 0;

	if (set->class_function[++series->cur_func])
		return;
	series->cur_func = 0;

	if (set->comp_type[++series->cur_comp])
		return;
	series->cur_comp = 0;

	if (set->ep_type[++series->cur_ep])
		return;
	series->cur_ep = 0;

	if (set->test_type[++series->cur_type])
		return;

	series->cur_set++;
}

int fts_end(struct ft_series *series, int index)
{
	return (series->cur_set >= series->nsets) ||
		((index > 0) && (series->test_index > index));
}

void fts_cur_info(struct ft_series *series, struct ft_info *info)
{
	static struct ft_set *set;

	memset(info, 0, sizeof *info);
	if (series->cur_set >= series->nsets)
		return;

	set = &series->sets[series->cur_set];
	info->test_type = set->test_type[series->cur_type];
	info->test_index = series->test_index;
	info->class_function = set->class_function[series->cur_func];
	info->test_flags = set->test_flags;
	info->caps = set->caps[series->cur_caps];
	info->mode = (set->mode[series->cur_mode] == FT_MODE_NONE) ?
			0 : set->mode[series->cur_mode];
	info->ep_type = set->ep_type[series->cur_ep];
	info->comp_type = set->comp_type[series->cur_comp];

	memcpy(info->node, set->node, FI_NAME_MAX);
	memcpy(info->service, set->service, FI_NAME_MAX);
	memcpy(info->prov_name, set->prov_name, FI_NAME_MAX);
}
