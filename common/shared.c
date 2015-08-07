/*
 * Copyright (c) 2013,2014 Intel Corporation.  All rights reserved.
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

#include <netdb.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>

#include <shared.h>

struct fi_info *fi, *hints;
struct fid_fabric *fabric;
struct fid_domain *domain;
struct fid_pep *pep;
struct fid_ep *ep;
struct fid_cq *txcq, *rxcq;
struct fid_cntr *txcntr, *rxcntr;
struct fid_mr *mr;
struct fid_av *av;
struct fid_eq *eq;

struct cs_opts opts;


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

#define INTEG_SEED 7
static const char integ_alphabet[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
static const int integ_alphabet_length = (sizeof(integ_alphabet)/sizeof(*integ_alphabet)) - 1;

static int dupaddr(void **dst_addr, size_t *dst_addrlen,
		void *src_addr, size_t src_addrlen)
{
	*dst_addr = malloc(src_addrlen);
	if (!*dst_addr) {
		FT_ERR("address allocation failed\n");
		return EAI_MEMORY;
	}
	*dst_addrlen = src_addrlen;
	memcpy(*dst_addr, src_addr, src_addrlen);
	return 0;
}

static int getaddr(char *node, char *service,
			struct fi_info *hints, uint64_t flags)
{
	int ret;
	struct fi_info *fi;

	if (!node) {
		if (flags & FI_SOURCE) {
			hints->src_addr = NULL;
			hints->src_addrlen = 0;
		} else {
			hints->dest_addr = NULL;
			hints->dest_addrlen = 0;
		}
		return 0;
	}

	ret = fi_getinfo(FT_FIVERSION, node, service, flags, hints, &fi);
	if (ret) {
		FT_ERR("fi_getinfo error %s\n", fi_strerror(ret));
		return ret;
	}
	hints->addr_format = fi->addr_format;

	if (flags & FI_SOURCE) {
		ret = dupaddr(&hints->src_addr, &hints->src_addrlen,
				fi->src_addr, fi->src_addrlen);
	} else {
		ret = dupaddr(&hints->dest_addr, &hints->dest_addrlen,
				fi->dest_addr, fi->dest_addrlen);
	}

	return ret;
}

int ft_getsrcaddr(char *node, char *service, struct fi_info *hints)
{
	return getaddr(node, service, hints, FI_SOURCE);
}

int ft_getdestaddr(char *node, char *service, struct fi_info *hints)
{
	return getaddr(node, service, hints, 0);
}

int ft_read_addr_opts(char **node, char **service, struct fi_info *hints,
		uint64_t *flags, struct cs_opts *opts)
{
	int ret;

	if (opts->dst_addr) {
		ret = ft_getsrcaddr(opts->src_addr, opts->src_port, hints);
		if (ret)
			return ret;
		*node = opts->dst_addr;
		*service = opts->dst_port;
	} else {
		*node = opts->src_addr;
		*service = opts->src_port;
		*flags = FI_SOURCE;
	}

	return 0;
}

char *size_str(char str[FT_STR_LEN], long long size)
{
	long long base, fraction = 0;
	char mag;

	memset(str, '\0', FT_STR_LEN);

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
		snprintf(str, FT_STR_LEN, "%lld.%lld%c", size / base, fraction, mag);
	else
		snprintf(str, FT_STR_LEN, "%lld%c", size / base, mag);

	return str;
}

char *cnt_str(char str[FT_STR_LEN], long long cnt)
{
	if (cnt >= 1000000000)
		snprintf(str, FT_STR_LEN, "%lldb", cnt / 1000000000);
	else if (cnt >= 1000000)
		snprintf(str, FT_STR_LEN, "%lldm", cnt / 1000000);
	else if (cnt >= 1000)
		snprintf(str, FT_STR_LEN, "%lldk", cnt / 1000);
	else
		snprintf(str, FT_STR_LEN, "%lld", cnt);

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

void init_test(struct cs_opts *opts, char *test_name, size_t test_name_len)
{
	char sstr[FT_STR_LEN];

	size_str(sstr, opts->transfer_size);
	snprintf(test_name, test_name_len, "%s_lat", sstr);
	if (!(opts->user_options & FT_OPT_ITER))
		opts->iterations = size_to_count(opts->transfer_size);
}

int wait_for_data_completion(struct fid_cq *cq, int num_completions)
{
	int ret;
	struct fi_cq_data_entry comp;

	while (num_completions > 0) {
		ret = fi_cq_read(cq, &comp, 1);
		if (ret > 0) {
			num_completions--;
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			if (ret == -FI_EAVAIL) {
				cq_readerr(cq, "cq");
			} else {
				FT_PRINTERR("fi_cq_read", ret);
			}
			return ret;
		}
	}
	return 0;
}

int wait_for_completion(struct fid_cq *cq, int num_completions)
{
	int ret;
	struct fi_cq_entry comp;

	while (num_completions > 0) {
		ret = fi_cq_read(cq, &comp, 1);
		if (ret > 0) {
			num_completions--;
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			if (ret == -FI_EAVAIL) {
				cq_readerr(cq, "cq");
			} else {
				FT_PRINTERR("fi_cq_read", ret);
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
	if (ret < 0) {
		FT_PRINTERR("fi_cq_readerr", ret);
	} else {
		err_str = fi_cq_strerror(cq, cq_err.prov_errno, cq_err.err_data, NULL, 0);
		fprintf(stderr, "%s: %d %s\n", cq_str, cq_err.err,
				fi_strerror(cq_err.err));
		fprintf(stderr, "%s: prov_err: %s (%d)\n", cq_str, err_str,
				cq_err.prov_errno);
	}
}

void eq_readerr(struct fid_eq *eq, char *eq_str)
{
	struct fi_eq_err_entry eq_err;
	const char *err_str;
	int rd;

	rd = fi_eq_readerr(eq, &eq_err, 0);
	if (rd != sizeof(eq_err)) {
		FT_PRINTERR("fi_eq_readerr", rd);
	} else {
		err_str = fi_eq_strerror(eq, eq_err.prov_errno, eq_err.err_data, NULL, 0);
		fprintf(stderr, "%s: %d %s\n", eq_str, eq_err.err,
				fi_strerror(eq_err.err));
		fprintf(stderr, "%s: prov_err: %s (%d)\n", eq_str, err_str,
				eq_err.prov_errno);
	}
}

int ft_finalize(
	struct fi_info *fi,
	struct fid_ep *tx_ep,
	struct fid_cq *txcq,
	struct fid_cq *rxcq,
	fi_addr_t addr)
{
	struct fi_msg msg;
	struct iovec iov;
	struct fi_context tx_ctx;
	char message[4] = "fin";
	size_t buf_size;
	size_t prefix_size = 0;
	char *buf;
	int ret;

	if (fi && fi->ep_attr)
		prefix_size = fi->ep_attr->msg_prefix_size;

	buf_size = sizeof(message) + prefix_size;

	buf = calloc(1, buf_size);
	if (!buf) {
		perror("calloc");
		return -1;
	}

	sprintf(buf + prefix_size, "%s", message);

	iov.iov_base = buf;
	iov.iov_len = buf_size;
	msg.msg_iov = &iov;
	msg.desc = NULL;
	msg.iov_count = 1;
	msg.addr = addr;
	msg.context = &tx_ctx;
	msg.data = 0;

	ret = fi_sendmsg(tx_ep, &msg, FI_INJECT | FI_TRANSMIT_COMPLETE);
	if (ret) {
		FT_PRINTERR("fi_sendmsg", ret);
		goto err;
	}

	wait_for_data_completion(txcq, 1);
	wait_for_data_completion(rxcq, 1);

err:
	free(buf);
	return ret;
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
	char str[FT_STR_LEN];
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
		printf("---\n");

		for (i = 0; i < argc; ++i)
			printf("%s ", argv[i]);

		printf(":\n");
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

void ft_usage(char *name, char *desc)
{
	fprintf(stderr, "Usage:\n");
	fprintf(stderr, "  %s [OPTIONS]\t\tstart server\n", name);
	fprintf(stderr, "  %s [OPTIONS] <host>\tconnect to server\n", name);

	if (desc)
		fprintf(stderr, "\n%s\n", desc);

	fprintf(stderr, "\nOptions:\n");
	FT_PRINT_OPTS_USAGE("-n <domain>", "domain name");
	FT_PRINT_OPTS_USAGE("-b <src_port>", "non default source port number");
	FT_PRINT_OPTS_USAGE("-p <dst_port>", "non default destination port number");
	FT_PRINT_OPTS_USAGE("-f <provider>", "specific provider name eg sockets, verbs");
	FT_PRINT_OPTS_USAGE("-s <address>", "source address");
	FT_PRINT_OPTS_USAGE("-h", "display this help output");

	return;
}

void ft_csusage(char *name, char *desc)
{
	fprintf(stderr, "Usage:\n");
	fprintf(stderr, "  %s [OPTIONS]\t\tstart server\n", name);
	fprintf(stderr, "  %s [OPTIONS] <host>\tconnect to server\n", name);

	if (desc)
		fprintf(stderr, "\n%s\n", desc);

	fprintf(stderr, "\nOptions:\n");
	FT_PRINT_OPTS_USAGE("-n <domain>", "domain name");
	FT_PRINT_OPTS_USAGE("-b <src_port>", "non default source port number");
	FT_PRINT_OPTS_USAGE("-p <dst_port>", "non default destination port number");
	FT_PRINT_OPTS_USAGE("-f <provider>", "specific provider name eg sockets, verbs");
	FT_PRINT_OPTS_USAGE("-s <address>", "source address");
	FT_PRINT_OPTS_USAGE("-I <number>", "number of iterations");
	FT_PRINT_OPTS_USAGE("-S <size>", "specific transfer size or 'all'");
	FT_PRINT_OPTS_USAGE("-m", "machine readable output");
	FT_PRINT_OPTS_USAGE("-v", "display versions and exit");
	FT_PRINT_OPTS_USAGE("-h", "display this help output");

	return;
}

void ft_parseinfo(int op, char *optarg, struct fi_info *hints)
{
	switch (op) {
	case 'n':
		if (!hints->domain_attr) {
			hints->domain_attr = malloc(sizeof *(hints->domain_attr));
			if (!hints->domain_attr) {
				perror("malloc");
				exit(EXIT_FAILURE);
			}
		}
		hints->domain_attr->name = strdup(optarg);
		break;
	case 'f':
		if (!hints->fabric_attr) {
			hints->fabric_attr = malloc(sizeof *(hints->fabric_attr));
			if (!hints->fabric_attr) {
				perror("malloc");
				exit(EXIT_FAILURE);
			}
		}
		hints->fabric_attr->prov_name = strdup(optarg);
		break;
	default:
		/* let getopt handle unknown opts*/
		break;

	}
}

void ft_parse_addr_opts(int op, char *optarg, struct cs_opts *opts)
{
	switch (op) {
	case 's':
		opts->src_addr = optarg;
		break;
	case 'b':
		opts->src_port = optarg;
		break;
	case 'p':
		opts->dst_port = optarg;
		break;
	default:
		/* let getopt handle unknown opts*/
		break;
	}
}

void ft_parsecsopts(int op, char *optarg, struct cs_opts *opts)
{
	ft_parse_addr_opts(op, optarg, opts);

	switch (op) {
	case 'I':
		opts->user_options |= FT_OPT_ITER;
		opts->iterations = atoi(optarg);
		break;
	case 'S':
		if (!strncasecmp("all", optarg, 3)) {
			opts->size_option = 1;
		} else {
			opts->user_options |= FT_OPT_SIZE;
			opts->transfer_size = atoi(optarg);
		}
		break;
	case 'm':
		opts->machr = 1;
		break;
	default:
		/* let getopt handle unknown opts*/
		break;
	}
}

void ft_fill_buf(void *buf, int size)
{
	char *msg_buf;
	int msg_index;
	static int iter = 0;
	int i;

	msg_index = ((iter++)*INTEG_SEED) % integ_alphabet_length;
	msg_buf = (char *)buf;
	for (i = 0; i < size; i++) {
		msg_buf[i] = integ_alphabet[msg_index++];
		if (msg_index >= integ_alphabet_length)
			msg_index = 0;
	}
}

int ft_check_buf(void *buf, int size)
{
	char *recv_data;
	char c;
	static int iter = 0;
	int msg_index;
	int i;

	msg_index = ((iter++)*INTEG_SEED) % integ_alphabet_length;
	recv_data = (char *)buf;

	for (i = 0; i < size; i++) {
		c = integ_alphabet[msg_index++];
		if (msg_index >= integ_alphabet_length)
			msg_index = 0;
		if (c != recv_data[i])
			break;
	}
	if (i != size) {
		printf("Error at iteration=%d size=%d byte=%d\n",
			iter, size, i);
		return 1;
	}

	return 0;
}
