/*
 * Copyright (c) 2015 Los Alamos Nat. Security, LLC. All rights reserved.
 * Copyright (c) Intel Corporation. All rights reserved.
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

#ifndef _FABTESTS_OSX_OSD_H_
#define _FABTESTS_OSX_OSD_H_

#include <config.h>

#include <sys/time.h>
#include <time.h>

#include <pthread.h>

#if !HAVE_CLOCK_GETTIME
#define CLOCK_REALTIME 0
#define CLOCK_REALTIME_COARSE 0
#define CLOCK_MONOTONIC 0

typedef int clockid_t;

#ifdef __cplusplus
extern "C" {
#endif

int clock_gettime(clockid_t clk_id, struct timespec *tp);

#ifdef __cplusplus
}
#endif
#endif // !HAVE_CLOCK_GETTIME

#if !defined(_POSIX_BARRIERS) || (_POSIX_BARRIERS < 0)

typedef int pthread_barrierattr_t;

typedef struct {
	int count;
	int called;
	pthread_mutex_t mutex;
	pthread_cond_t cond;
} pthread_barrier_t;

static inline int pthread_barrier_init(pthread_barrier_t *barrier,
                        	       const pthread_barrierattr_t *attr,
                        	       unsigned count)
{
	barrier->count = count;
	barrier->called = 0;
	pthread_mutex_init(&barrier->mutex, NULL);
	pthread_cond_init(&barrier->cond, NULL);
	return 0;
}

static inline int pthread_barrier_destroy(pthread_barrier_t *barrier)
{
	pthread_mutex_destroy(&barrier->mutex);
	pthread_cond_destroy(&barrier->cond);
	return 0;
}

static inline int pthread_barrier_wait(pthread_barrier_t *barrier)
{
	pthread_mutex_lock(&barrier->mutex);
	barrier->called++;
	if (barrier->called == barrier->count) {
		barrier->called = 0;
		pthread_cond_broadcast(&barrier->cond);
	} else {
		pthread_cond_wait(&barrier->cond, &barrier->mutex);
	}

	pthread_mutex_unlock(&barrier->mutex);
	return 0;
}

#endif // _POSIX_BARRIERS

#endif // FABTESTS_OSX_OSD_H
