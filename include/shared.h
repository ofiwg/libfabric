/*
 * Copyright (c) 2013,2014 Intel Corporation.  All rights reserved.
 *
 * This software is available to you under the OpenIB.org BSD license
 * below:
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

#ifndef _SHARED_H_
#define _SHARED_H_

#include <sys/socket.h>
#include <sys/types.h>

#include <rdma/fabric.h>
#include <rdma/fi_eq.h>

#ifdef __cplusplus
extern "C" {
#endif

struct test_size_param {
	int size;
	int option;
};

enum precision {
	NANO = 1,
	MICRO = 1000,
	MILLI = 1000000,
};

extern struct test_size_param test_size[];
const unsigned int test_cnt;
#define TEST_CNT test_cnt
#define FI_STR_LEN 32

int getaddr(char *node, char *service, struct sockaddr **addr, socklen_t *len);
char *size_str(char str[FI_STR_LEN], long long size);
char *cnt_str(char str[FI_STR_LEN], long long cnt);
int size_to_count(int size);
void init_test(int size, char *test_name, int *transfer_size, int *iterations);
int wait_for_completion(struct fid_cq *cq, int num_completions);
void cq_readerr(struct fid_cq *cq, char *cq_str);
int64_t get_elapsed(const struct timespec *b, const struct timespec *a, enum precision p);
void show_perf(char *name, int tsize, int iters, struct timespec *start, 
		struct timespec *end, int xfers_per_iter);
void show_perf_mr(int tsize, int iters, struct timespec *start,
		  struct timespec *end, int xfers_per_iter, int argc, char *argv[]);

#define FI_PRINTERR(call, retv) \
	do { fprintf(stderr, call ": %d (%s)\n", retv, fi_strerror(-retv)); } while (0)

#define FI_DEBUG(fmt, ...) \
	do { fprintf(stderr, "%s:%d: " fmt, __FILE__, __LINE__, ##__VA_ARGS__); } while (0)

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define ARRAY_SIZE(A) (sizeof(A)/sizeof(*A))

#ifdef __cplusplus
}
#endif

#endif /* _SHARED_H_ */
