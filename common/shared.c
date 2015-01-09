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

#include <netdb.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <rdma/fi_errno.h>

#include <shared.h>

struct test_size_param test_size[] = {
	{ 1 <<  1, 1 }, { (1 <<  1) + (1 <<  0), 2},
	{ 1 <<  2, 2 }, { (1 <<  2) + (1 <<  1), 2},
	{ 1 <<  3, 1 }, { (1 <<  3) + (1 <<  2), 2},
	{ 1 <<  4, 2 }, { (1 <<  4) + (1 <<  3), 2},
	{ 1 <<  5, 1 }, { (1 <<  5) + (1 <<  4), 2},
	{ 1 <<  6, 0 }, { (1 <<  6) + (1 <<  5), 0},
	{ 1 <<  7, 1 }, { (1 <<  7) + (1 <<  6), 0},
	{ 1 <<  8, 1 }, { (1 <<  8) + (1 <<  7), 1},
	{ 1 <<  9, 1 }, { (1 <<  9) + (1 <<  8), 1},
	{ 1 << 10, 1 }, { (1 << 10) + (1 <<  9), 1},
	{ 1 << 11, 1 }, { (1 << 11) + (1 << 10), 1},
	{ 1 << 12, 0 }, { (1 << 12) + (1 << 11), 1},
	{ 1 << 13, 1 }, { (1 << 13) + (1 << 12), 1},
	{ 1 << 14, 1 }, { (1 << 14) + (1 << 13), 1},
	{ 1 << 15, 1 }, { (1 << 15) + (1 << 14), 1},
	{ 1 << 16, 0 }, { (1 << 16) + (1 << 15), 1},
	{ 1 << 17, 1 }, { (1 << 17) + (1 << 16), 1},
	{ 1 << 18, 1 }, { (1 << 18) + (1 << 17), 1},
	{ 1 << 19, 1 }, { (1 << 19) + (1 << 18), 1},
	{ 1 << 20, 0 }, { (1 << 20) + (1 << 19), 1},
	{ 1 << 21, 1 }, { (1 << 21) + (1 << 20), 1},
	{ 1 << 22, 1 }, { (1 << 22) + (1 << 21), 1},
	{ 1 << 23, 1 },
};

const unsigned int test_cnt = (sizeof test_size / sizeof test_size[0]);

int getaddr(char *node, char *service, struct sockaddr **addr, socklen_t *len)
{
	struct addrinfo *ai;
	int ret;

	ret = getaddrinfo(node, service, NULL, &ai);
	if (ret)
		return ret;

	if ((*addr = malloc(ai->ai_addrlen))) {
		memcpy(*addr, ai->ai_addr, ai->ai_addrlen);
		*len = ai->ai_addrlen;
	} else {
		ret = EAI_MEMORY;
	}

	freeaddrinfo(ai);
	return ret;
}

char *size_str(char str[FI_STR_LEN], long long size)
{
	long long base, fraction = 0;
	char mag;

	memset(str, '\0', FI_STR_LEN);

	if (size >= (1 << 30)) {
		base = 1 << 30;
		mag = 'g';
	} else if (size >= (1 << 20)) {
		base = 1 << 20;
		mag = 'm';
	} else if (size >= (1 << 10)) {
		base = 1 << 10;
		mag = 'k';
	} else {
		base = 1;
		mag = '\0';
	}

	if (size / base < 10)
		fraction = (size % base) * 10 / base;

	if (fraction)
		snprintf(str, FI_STR_LEN, "%lld.%lld%c", size / base, fraction, mag);
	else
		snprintf(str, FI_STR_LEN, "%lld%c", size / base, mag);

	return str;
}

char *cnt_str(char str[FI_STR_LEN], long long cnt)
{
	if (cnt >= 1000000000)
		snprintf(str, FI_STR_LEN, "%lldb", cnt / 1000000000);
	else if (cnt >= 1000000)
		snprintf(str, FI_STR_LEN, "%lldm", cnt / 1000000);
	else if (cnt >= 1000)
		snprintf(str, FI_STR_LEN, "%lldk", cnt / 1000);
	else
		snprintf(str, FI_STR_LEN, "%lld", cnt);

	return str;
}

int size_to_count(int size)
{
	if (size >= (1 << 20))
		return 100;
	else if (size >= (1 << 16))
		return 1000;
	else if (size >= (1 << 10))
		return 10000;
	else
		return 100000;
}

void init_test(int size, char *test_name, size_t test_name_len,
	int *transfer_size, int *iterations)
{
	char sstr[FI_STR_LEN];

	size_str(sstr, size);
	snprintf(test_name, test_name_len, "%s_lat", sstr);
	*transfer_size = size;
	*iterations = size_to_count(*transfer_size);
}

int bind_fid( fid_t ep, fid_t res, uint64_t flags)
{
	int ret;

	ret = fi_bind(ep, res, flags);
	if (ret)
		FI_PRINTERR("fi_bind", ret);
	return ret;
}

int wait_for_completion(struct fid_cq *cq, int num_completions)
{
	int ret;
	struct fi_cq_entry comp;

	while (num_completions > 0) {
		ret = fi_cq_read(cq, &comp, 1);
		if (ret > 0) {
			num_completions--;
		} else if (ret < 0) {
			if (ret == -FI_EAVAIL) {
				cq_readerr(cq, "cq");
			} else {
				FI_PRINTERR("fi_cq_read", ret);
			}
			return ret;
		}
	}
	return 0;
}

void cq_readerr(struct fid_cq *cq, char *cq_str)
{ 
	struct fi_cq_err_entry cq_err;
	const char *err_str;
	int ret;

	ret = fi_cq_readerr(cq, &cq_err, 0);
	if (ret < 0)
		FI_PRINTERR("fi_cq_readerr", ret);

	err_str = fi_cq_strerror(cq, cq_err.prov_errno, cq_err.err_data, NULL, 0);
	FI_DEBUG("%s %s (%d)\n", cq_str, err_str, cq_err.prov_errno);
}

int64_t get_elapsed(const struct timespec *b, const struct timespec *a,
		    enum precision p)
{
    int64_t elapsed;

    elapsed = (a->tv_sec - b->tv_sec) * 1000 * 1000 * 1000;
    elapsed += a->tv_nsec - b->tv_nsec;
    return elapsed / p;
}

void show_perf(char *name, int tsize, int iters, struct timespec *start, 
		struct timespec *end, int xfers_per_iter)
{
	static int header = 1;
	char str[FI_STR_LEN];
	int64_t elapsed = get_elapsed(start, end, MICRO);
	long long bytes = (long long) iters * tsize * xfers_per_iter;

	if (header) {
		printf("%-10s%-8s%-8s%-8s%8s %10s%13s\n",
			"name", "bytes", "iters", "total", "time", "Gb/sec", "usec/xfer");
		header = 0;
	}

	printf("%-10s", name);

	printf("%-8s", size_str(str, tsize));

	printf("%-8s", cnt_str(str, iters));

	printf("%-8s", size_str(str, bytes));

	printf("%8.2fs%10.2f%11.2f\n",
		elapsed / 1000000.0, (bytes * 8) / (1000.0 * elapsed),
		((float)elapsed / iters / xfers_per_iter));
}

void show_perf_mr(int tsize, int iters, struct timespec *start,
		  struct timespec *end, int xfers_per_iter, int argc, char *argv[])
{
	static int header = 1;
	int64_t elapsed = get_elapsed(start, end, MICRO);
	long long total = (long long) iters * tsize * xfers_per_iter;
	int i;

	if (header) {
		printf("--- # ");

		for (i = 0; i < argc; ++i)
			printf("%s ", argv[i]);

		printf("\n");
		header = 0;
	}

	printf("- { ");
	printf("xfer_size: %d, ", tsize);
	printf("iterations: %d, ", iters);
	printf("total: %lld, ", total);
	printf("time: %f, ", elapsed / 1000000.0);
	printf("Gb/sec: %f, ", (total * 8) / (1000.0 * elapsed));
	printf("usec/xfer: %f", ((float)elapsed / iters / xfers_per_iter));
	printf(" }\n");
}
