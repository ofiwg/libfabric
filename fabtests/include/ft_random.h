/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _FT_RANDOM_H_
#define _FT_RANDOM_H_

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include "shared.h"

#ifdef __GNUC__
    #define likely(x)   __builtin_expect(!!(x), 1)
    #define unlikely(x) __builtin_expect(!!(x), 0)
#else
    #define likely(x)   (x)
    #define unlikely(x) (x)
#endif

#define FABTESTS_RANDOM_STATE_SIZE (8)

#if !HAVE_RANDOM_R
struct random_data {
    unsigned int seed;
};
#endif

static inline void ft_random_init_data(struct random_data *restrict random_data, int seed, int salt)
{
#if !HAVE_RANDOM_R
	random_data->seed = (unsigned int)seed ^ (unsigned int)salt;
#else
	static _Thread_local char random_state[FABTESTS_RANDOM_STATE_SIZE];
	random_data->state = NULL;
	int ret = initstate_r(seed ^ salt, random_state, sizeof random_state, random_data);
	if (unlikely(ret)) {
		FT_PRINTERR("initstate_r", ret);
		abort();
	}
#endif
}

static inline int32_t ft_random_get_int32(struct random_data *restrict random_data)
{
#if !HAVE_RANDOM_R
	return rand_r(&random_data->seed);
#else
	int32_t num;
	int ret = random_r(random_data, &num);
	if (unlikely(ret)) {
		FT_PRINTERR("random_r", ret);
		abort();
	}
	return num;
#endif
}

static inline bool ft_random_get_bool(struct random_data *restrict random_data)
{
	return (bool)(ft_random_get_int32(random_data) & 1);
}

static inline int32_t ft_random_get_int32_in_range(struct random_data *restrict random_data, int32_t min_int32, int32_t max_int32)
{
	/* Max range is INT32_MAX - 1*/
	const int32_t modulo = max_int32 - min_int32 + 1;
	const int32_t rejection_limit = INT32_MAX - (INT32_MAX % modulo);

	int32_t result;
	do {
		result = ft_random_get_int32(random_data);
	} while (unlikely(result >= rejection_limit));
	return result % modulo;
}

static inline int32_t ft_random_sleep_ms(struct random_data *restrict random_data, int max_sleep_time_ms)
{
	const int32_t sleep_time_ms = ft_random_get_int32_in_range(random_data, 0, max_sleep_time_ms);
	struct timespec rem, ts = {
		.tv_sec = sleep_time_ms / 1000,
		.tv_nsec = sleep_time_ms % 1000
	};
	while(true) {
		int ret = nanosleep(&ts, &rem);
		if (likely(ret == 0))
			return sleep_time_ms;
		if (errno != EINTR) {
			FT_PRINTERR("nanosleep", ret);
			abort();
		}
		ts = rem;
	}
}

#endif /* _FT_RANDOM_H_ */
