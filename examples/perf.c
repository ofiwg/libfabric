/*
 * Copyright (c) 2013 Intel Corporation.  All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <errno.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <netdb.h>
#include <fcntl.h>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_socket.h>
#include <rdma/fi_cm.h>
#include "shared.h"


struct test_size_param {
	int size;
	int option;
};

static struct test_size_param test_size[] = {
	{ 1 <<  6, 0 },
	{ 1 <<  7, 1 }, { (1 <<  7) + (1 <<  6), 1},
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
};
#define TEST_CNT (sizeof test_size / sizeof test_size[0])

enum perf_optimization {
	opt_latency,
	opt_bandwidth
};

#define SEND_CONTEXT	NULL

static int custom;
static enum perf_optimization optimization;
static int size_option;
static int iterations = 1;
static int transfer_size = 1000;
static int transfer_count = 1000;
/* TODO: make max_credits dynamic based on user input or socket size */
static int max_credits = 128;
static int credits = 128;
static char test_name[10] = "custom";
static struct timeval start, end;
static void *buf;
static size_t buffer_size;

static struct fi_info hints;
static char *dst_addr, *src_addr;
static char *port = "9228";
static fid_t lfs, ldom, lcm;
static fid_t fs, dom, mr, cq;


static void show_perf(void)
{
	char str[32];
	float usec;
	long long bytes;

	usec = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
	bytes = (long long) iterations * transfer_count * transfer_size * 2;

	/* name size transfers iterations bytes seconds Gb/sec usec/xfer */
	printf("%-10s", test_name);
	size_str(str, sizeof str, transfer_size);
	printf("%-8s", str);
	cnt_str(str, sizeof str, transfer_count);
	printf("%-8s", str);
	cnt_str(str, sizeof str, iterations);
	printf("%-8s", str);
	size_str(str, sizeof str, bytes);
	printf("%-8s", str);
	printf("%8.2fs%10.2f%11.2f\n",
		usec / 1000000., (bytes * 8) / (1000. * usec),
		(usec / iterations) / (transfer_count * 2));
}

static void init_latency_test(int size)
{
	char sstr[5];

	size_str(sstr, sizeof sstr, size);
	snprintf(test_name, sizeof test_name, "%s_lat", sstr);
	transfer_count = 1;
	transfer_size = size;
	iterations = size_to_count(transfer_size);
}

static void init_bandwidth_test(int size)
{
	char sstr[5];

	size_str(sstr, sizeof sstr, size);
	snprintf(test_name, sizeof test_name, "%s_bw", sstr);
	iterations = 1;
	transfer_size = size;
	transfer_count = size_to_count(transfer_size);
}

static int poll_all(void)
{
	struct fi_ec_entry comp;
	int ret;

	do {
		ret = fi_ec_read(cq, &comp, sizeof comp);
		if (ret > 0) {
			if (comp.op_context == SEND_CONTEXT)
				credits++;
		} else if (ret < 0) {
			printf("Completion queue read %d (%s)\n", ret, fi_strerror(-ret));
			return ret;
		}
	} while (ret);
	return 0;
}

static int send_xfer(int size)
{
	struct fi_ec_entry comp;
	int ret;

	while (!credits) {
		ret = fi_ec_read(cq, &comp, sizeof comp);
		if (ret > 0) {
			if (comp.op_context == SEND_CONTEXT)
				goto post;
		} else if (ret < 0) {
			printf("Completion queue read %d (%s)\n", ret, fi_strerror(-ret));
			return ret;
		}
	}

	credits--;
post:
	ret = fi_sendmem(fs, buf, size, fi_mr_desc(mr), SEND_CONTEXT);
	if (ret)
		printf("fi_write %d (%s)\n", ret, fi_strerror(-ret));

	return ret;
}

static int recv_xfer(int size)
{
	struct fi_ec_entry comp;
	int ret;

	while (1) {
		ret = fi_ec_read(cq, &comp, sizeof comp);
		if (ret > 0) {
			if (comp.op_context == SEND_CONTEXT)
				credits++;
			else
				break;
		} else if (ret < 0) {
			printf("Completion queue read %d (%s)\n", ret, fi_strerror(-ret));
			return ret;
		}
	}

	ret = fi_recvmem(fs, buf, buffer_size, fi_mr_desc(mr), buf);
	if (ret)
		printf("fi_recvmem %d (%s)\n", ret, fi_strerror(-ret));

	return ret;
}

static int sync_test(void)
{
	int ret;

	while (credits < max_credits)
		poll_all();

	ret = dst_addr ? send_xfer(16) : recv_xfer(16);
	if (ret)
		return ret;

	return dst_addr ? recv_xfer(16) : send_xfer(16);
}

static int run_test(void)
{
	int ret, i, t;

	ret = sync_test();
	if (ret)
		goto out;

	gettimeofday(&start, NULL);
	for (i = 0; i < iterations; i++) {
		for (t = 0; t < transfer_count; t++) {
			ret = dst_addr ? send_xfer(transfer_size) :
					 recv_xfer(transfer_size);
			if (ret)
				goto out;
		}

		for (t = 0; t < transfer_count; t++) {
			ret = dst_addr ? recv_xfer(transfer_size) :
					 send_xfer(transfer_size);
			if (ret)
				goto out;
		}
	}
	gettimeofday(&end, NULL);
	show_perf();
	ret = 0;

out:
	return ret;
}

static int alloc_cm_ec(fid_t dom, fid_t *cm_ec)
{
	struct fi_ec_attr cm_attr;
	int ret;

	memset(&cm_attr, 0, sizeof cm_attr);
	cm_attr.ec_mask = FI_EC_ATTR_MASK_V1;
	cm_attr.domain = FI_EC_DOMAIN_CM;
	cm_attr.type = FI_EC_QUEUE;
	cm_attr.format = FI_EC_FORMAT_CM;
	cm_attr.wait_obj = FI_EC_WAIT_FD;
	cm_attr.flags = FI_AUTO_RESET;
	ret = fi_ec_open(dom, &cm_attr, cm_ec, NULL);
	if (ret)
		printf("fi_ec_open cm %s\n", fi_strerror(-ret));

	return ret;
}

static void free_lres(void)
{
	fi_close(lcm);
	fi_close(ldom);
}

static int alloc_lres(struct fi_info *fi)
{
	int ret;

	ret = fi_open(NULL, fi, 0, &ldom, NULL);
	if (ret) {
		printf("fi_open %s %s\n", fi->domain_name, fi_strerror(-ret));
		return ret;
	}

	ret = alloc_cm_ec(ldom, &lcm);
	if (ret)
		fi_close(ldom);

	return ret;
}

static void free_res(void)
{
	fi_mr_unreg(mr);
	fi_close(cq);
	fi_close(dom);
	free(buf);
}

static int alloc_res(struct fi_info *fi)
{
	struct fi_ec_attr cq_attr;
	int ret;

	buffer_size = !custom ? test_size[TEST_CNT - 1].size : transfer_size;
	buf = malloc(buffer_size);
	if (!buf) {
		perror("malloc");
		return -1;
	}

	ret = fi_open(NULL, fi, 0, &dom, NULL);
	if (ret) {
		printf("fi_open %s %s\n", fi->domain_name, fi_strerror(-ret));
		goto err1;
	}

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.ec_mask = FI_EC_ATTR_MASK_V1;
	cq_attr.domain = FI_EC_DOMAIN_COMP;
	cq_attr.type = FI_EC_QUEUE;
	cq_attr.format = FI_EC_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_EC_WAIT_NONE;
	cq_attr.size = max_credits << 1;
	ret = fi_ec_open(dom, &cq_attr, &cq, NULL);
	if (ret) {
		printf("fi_eq_open comp %s\n", fi_strerror(-ret));
		goto err2;
	}

	ret = fi_mr_reg(dom, buf, buffer_size, &mr, 0, NULL);
	if (ret) {
		printf("fi_mr_reg %s\n", fi_strerror(-ret));
		goto err3;
	}
	return 0;

err3:
	fi_close(cq);
err2:
	fi_close(dom);
err1:
	free(buf);
	return ret;
}

static int bind_fid(fid_t sock, fid_t res, uint64_t flags)
{
	struct fi_resource fr;
	int ret;

	fr.fid = res;
	fr.flags = flags;
	ret = fi_bind(sock, &fr, 1);
	if (ret)
		printf("fi_bind %s\n", fi_strerror(-ret));
	return ret;
}

static int bind_lres(void)
{
	return bind_fid(lfs, lcm, 0);
}

static int bind_res(void)
{
	int ret;

	ret = bind_fid(fs, cq, FI_SEND | FI_RECV);
	if (!ret) {
		ret = fi_recvmem(fs, buf, buffer_size, fi_mr_desc(mr), buf);
		if (ret)
			printf("fi_read %d (%s)\n", ret, fi_strerror(-ret));
	}
	return ret;
}

static int server_listen(void)
{
	struct fi_info *fi;
	int ret;

	hints.flags = FI_PASSIVE;
	ret = fi_getinfo(src_addr, port, &hints, &fi);
	if (ret) {
		printf("fi_getinfo %s\n", strerror(-ret));
		return ret;
	}

	ret = fi_socket(fi, &lfs, NULL);
	if (ret) {
		printf("fi_socket %s\n", fi_strerror(-ret));
		goto err1;
	}

	ret = alloc_lres(fi);
	if (ret)
		goto err2;

	ret = bind_lres();
	if (ret)
		goto err3;

	ret = fi_listen(lfs);
	if (ret) {
		printf("fi_listen %s\n", fi_strerror(-ret));
		goto err3;
	}

	fi_freeinfo(fi);
	return 0;
err3:
	free_lres();
err2:
	fi_close(lfs);
err1:
	fi_freeinfo(fi);
	return ret;
}

static int server_connect(void)
{
	struct fi_ec_cm_entry entry;
	ssize_t rd;
	int ret;

	rd = fi_ec_read(lcm, &entry, sizeof entry);
	if (rd != sizeof entry) {
		printf("fi_ec_read %zd %s\n", rd, fi_strerror((int) -rd));
		return (int) rd;
	}

	if (entry.event != FI_CONNREQ) {
		printf("Unexpected CM event %d\n", entry.event);
		ret = -FI_EOTHER;
		goto err1;
	}

	ret = fi_socket(entry.info, &fs, NULL);
	if (ret) {
		printf("fi_socket for req %s\n", fi_strerror(-ret));
		goto err1;
	}

	ret = alloc_res(entry.info);
	if (ret)
		 goto err2;

	ret = bind_res();
	if (ret)
		goto err3;

	ret = fi_accept(fs, NULL, 0);
	if (ret) {
		printf("fi_accept %s\n", fi_strerror(-ret));
		goto err3;
	}

	fi_freeinfo(entry.info);
	return 0;

err3:
	free_res();
err2:
	fi_close(fs);
err1:
	fi_freeinfo(entry.info);
	return ret;
}

static int client_connect(void)
{
	struct fi_info *fi;
	int ret;

	if (src_addr) {
		ret = getaddr(src_addr, NULL, (struct sockaddr **) &hints.src_addr,
			      (socklen_t *) &hints.src_addrlen);
		if (ret)
			printf("source address error %s\n", gai_strerror(ret));
	}

	ret = fi_getinfo(dst_addr, port, &hints, &fi);
	if (ret) {
		printf("fi_getinfo %s\n", strerror(-ret));
		goto err1;
	}

	ret = fi_socket(fi, &fs, NULL);
	if (ret) {
		printf("fi_socket %s\n", fi_strerror(-ret));
		goto err2;
	}

	ret = alloc_res(fi);
	if (ret)
		goto err3;

	ret = bind_res();
	if (ret)
		goto err4;

	ret = fi_connect(fs, NULL, 0);
	if (ret) {
		printf("fi_connect %s\n", fi_strerror(-ret));
		goto err4;
	}

	if (hints.src_addr)
		free(hints.src_addr);
	fi_freeinfo(fi);
	return 0;

err4:
	free_res();
err3:
	fi_close(fs);
err2:
	fi_freeinfo(fi);
err1:
	if (hints.src_addr)
		free(hints.src_addr);
	return ret;
}

static int run(void)
{
	int i, ret = 0;

	if (!dst_addr) {
		ret = server_listen();
		if (ret)
			return ret;
	}

	printf("%-10s%-8s%-8s%-8s%-8s%8s %10s%13s\n",
	       "name", "bytes", "xfers", "iters", "total", "time", "Gb/sec", "usec/xfer");
	if (!custom) {
		optimization = opt_latency;
		ret = dst_addr ? client_connect() : server_connect();
		if (ret)
			return ret;

		for (i = 0; i < TEST_CNT; i++) {
			if (test_size[i].option > size_option)
				continue;
			init_latency_test(test_size[i].size);
			run_test();
		}

		/*
		 * disable bandwidth test until we have a correct flooding
		 * message protocol
		fi_shutdown(fs, 0);
		poll_all();
		fi_close(fs);
		free_res();

		optimization = opt_bandwidth;
		ret = dst_addr ? client_connect() : server_connect();
		if (ret)
			return ret;

		for (i = 0; i < TEST_CNT; i++) {
			if (test_size[i].option > size_option)
				continue;
			init_bandwidth_test(test_size[i].size);
			run_test();
		}
		*/
	} else {
		ret = dst_addr ? client_connect() : server_connect();
		if (ret)
			return ret;

		ret = run_test();
	}

	while (credits < max_credits)
		poll_all();
	fi_shutdown(fs, 0);
	fi_close(fs);
	free_res();
	if (!dst_addr)
		free_lres();
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;

	while ((op = getopt(argc, argv, "d:n:p:s:C:I:S:")) != -1) {
		switch (op) {
		case 'd':
			dst_addr = optarg;
			break;
		case 'n':
			hints.domain_name = optarg;
			break;
		case 'p':
			port = optarg;
			break;
		case 's':
			src_addr = optarg;
			break;
		case 'C':
			custom = 1;
			transfer_count = atoi(optarg);
			break;
		case 'I':
			custom = 1;
			iterations = atoi(optarg);
			break;
		case 'S':
			if (!strncasecmp("all", optarg, 3)) {
				size_option = 1;
			} else {
				custom = 1;
				transfer_size = atoi(optarg);
			}
			break;
		default:
			printf("usage: %s\n", argv[0]);
			printf("\t[-d destination_address]\n");
			printf("\t[-n domain_name]\n");
			printf("\t[-p port_number]\n");
			printf("\t[-s source_address]\n");
			printf("\t[-C transfer_count]\n");
			printf("\t[-I iterations]\n");
			printf("\t[-S transfer_size or 'all']\n");
			exit(1);
		}
	}

	hints.type = FID_MSG;
	ret = run();
	return ret;
}
