/*
 * Copyright (c) 2013-2015 Intel Corporation.  All rights reserved.
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
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <netdb.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#include <limits.h>
#include <shared.h>

#include "fabtest.h"


int listen_sock = -1;
int sock = -1;
static int persistent = 1;

//static struct timespec start, end;

static struct ft_series *series;
static int test_start_index, test_end_index = INT_MAX;
struct ft_info test_info;
struct fi_info *fabric_info;
struct ft_xcontrol ft_rx_ctrl, ft_tx_ctrl;
struct ft_control ft_ctrl;

size_t recv_size, send_size;

enum {
	FT_SUCCESS,
	FT_ENODATA,
	FT_ENOSYS,
	FT_ERROR,
	FT_MAX_RESULT
};

static int results[FT_MAX_RESULT];


static int ft_nullstr(char *str)
{
	return (!str || str[0] == '\0');
}

static char *ft_strptr(char *str)
{
	return ft_nullstr(str) ? NULL : str;
}

static void ft_show_test_info(void)
{
	printf("[%s] ", test_info.prov_name);

	switch (test_info.test_type) {
	case FT_TEST_LATENCY:
		printf("latency");
		break;
	case FT_TEST_BANDWIDTH:
		printf("bandwidth");
		break;
	default:
		break;
	}

	printf( " %s", fi_tostr(&test_info.ep_type, FI_TYPE_EP_TYPE));
	printf( " [%s]", fi_tostr(&test_info.caps, FI_TYPE_CAPS));

	switch (test_info.class_function) {
	case FT_FUNC_SEND:
		printf(" send");
		break;
	case FT_FUNC_SENDV:
		printf(" sendv");
		break;
	case FT_FUNC_SENDMSG:
		printf(" sendmsg");
		break;
	default:
		break;
	}
	printf("\n");
}

static int ft_check_info(struct fi_info *hints, struct fi_info *info)
{
	if (info->mode & ~hints->mode) {
		fprintf(stderr, "fi_getinfo unsupported mode returned\n");
		return -FI_EINVAL;
	}
	if (hints->caps != (hints->caps & info->caps)) {
		fprintf(stderr, "fi_getinfo missing caps\n");
		return -FI_EINVAL;
	}

	return 0;
}

static int ft_fw_listen(char *service)
{
	struct addrinfo *ai, hints;
	int val, ret;

	memset(&hints, 0, sizeof hints);
	hints.ai_flags = AI_PASSIVE;

	ret = getaddrinfo(NULL, service, &hints, &ai);
	if (ret) {
		fprintf(stderr, "getaddrinfo() %s\n", gai_strerror(ret));
		return ret;
	}

	listen_sock = socket(ai->ai_family, SOCK_STREAM, 0);
	if (listen_sock < 0) {
		perror("socket");
		ret = listen_sock;
		goto free;
	}

	val = 1;
	ret = setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &val, sizeof val);
	if (ret) {
		perror("setsockopt SO_REUSEADDR");
		goto close;
	}

	ret = bind(listen_sock, ai->ai_addr, ai->ai_addrlen);
	if (ret) {
		perror("bind");
		goto close;
	}

	ret = listen(listen_sock, 0);
	if (ret)
		perror("listen");

	return 0;

close:
	close(listen_sock);
free:
	freeaddrinfo(ai);
	return ret;
}

static int ft_fw_connect(char *node, char *service)
{
	struct addrinfo *ai;
	int ret;

	ret = getaddrinfo(node, service, NULL, &ai);
	if (ret) {
		perror("getaddrinfo");
		return ret;
	}

	sock = socket(ai->ai_family, SOCK_STREAM, 0);
	if (sock < 0) {
		perror("socket");
		ret = sock;
		goto free;
	}

	ret = 1;
	setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (void *) &ret, sizeof(ret));

	ret = connect(sock, ai->ai_addr, ai->ai_addrlen);
	if (ret) {
		perror("connect");
		close(sock);
	}

free:
	freeaddrinfo(ai);
	return ret;
}

static void ft_fw_shutdown(int fd)
{
	shutdown(fd, SHUT_RDWR);
	close(fd);
}

int ft_fw_send(int fd, void *msg, size_t len)
{
	int ret;

	ret = send(fd, msg, len, 0);
	if (ret == len) {
		return 0;
	} else if (ret < 0) {
		perror("send");
		return -errno;
	} else {
		perror("send aborted");
		return -FI_ECONNABORTED;
	}
}

int ft_fw_recv(int fd, void *msg, size_t len)
{
	int ret;

	ret = recv(fd, msg, len, MSG_WAITALL);
	if (ret == len) {
		return 0;
	} else if (ret == 0) {
		return -FI_ENOTCONN;
	} else if (ret < 0) {
		perror("recv");
		return -errno;
	} else {
		perror("recv aborted");
		return -FI_ECONNABORTED;
	}
}

static void ft_fw_convert_info(struct fi_info *info, struct ft_info *test_info)
{
	info->caps = test_info->caps;
	info->mode = test_info->mode;

	info->ep_attr->type = test_info->ep_type;
	info->ep_attr->protocol = test_info->protocol;
	info->ep_attr->protocol_version = test_info->protocol_version;

	if (!ft_nullstr(test_info->prov_name)) {
		info->fabric_attr->prov_name = strndup(test_info->prov_name,
					sizeof test_info->prov_name - 1);
	}
	if (!ft_nullstr(test_info->fabric_name)) {
		info->fabric_attr->name = strndup(test_info->fabric_name,
					sizeof test_info->fabric_name - 1);
	}
}

static void
ft_fw_update_info(struct ft_info *test_info, struct fi_info *info, int subindex)
{
	test_info->test_subindex = subindex;

	if (info->ep_attr) {
		test_info->protocol = info->ep_attr->protocol;
		test_info->protocol_version = info->ep_attr->protocol_version;
	}

	if (info->fabric_attr) {
		if (info->fabric_attr->prov_name) {
			strncpy(test_info->prov_name, info->fabric_attr->prov_name,
				sizeof test_info->prov_name);
		}
		if (info->fabric_attr->name) {
			strncpy(test_info->fabric_name, info->fabric_attr->name,
				sizeof test_info->fabric_name);
		}
	}
}

static int ft_fw_result_index(int fi_errno)
{
	switch (fi_errno) {
	case 0:
		return FT_SUCCESS;
	case FI_ENODATA:
		return FT_ENODATA;
	case FI_ENOSYS:
		return FT_ENOSYS;
	default:
		return FT_ERROR;
	}
}

static int ft_fw_server(void)
{
	struct fi_info *hints, *info;
	int ret;

	hints = fi_allocinfo();
	if (!hints)
		return -FI_ENOMEM;

	do {
		ret = ft_fw_recv(sock, &test_info, sizeof test_info);
		if (ret) {
			if (ret == -FI_ENOTCONN)
				ret = 0;
			break;
		}

		ft_fw_convert_info(hints, &test_info);
		printf("Starting test %d-%d: ", test_info.test_index,
			test_info.test_subindex);
		ft_show_test_info();
		ret = fi_getinfo(FT_FIVERSION, ft_strptr(test_info.node),
				 ft_strptr(test_info.service), FI_SOURCE,
				 hints, &info);
		if (ret) {
			FT_PRINTERR("fi_getinfo", ret);
		} else {
			if (info->next) {
				printf("fi_getinfo returned multiple matches\n");
				ret = -FI_E2BIG;
			} else {
				/* fabric_info is replaced when connecting */
				fabric_info = info;

				ret = ft_run_test();

				if (fabric_info != info)
					fi_freeinfo(fabric_info);
			}
			fi_freeinfo(info);
		}

		if (ret) {
			printf("Node: %s\nService: %s\n",
				test_info.node, test_info.service);
			printf("%s\n", fi_tostr(hints, FI_TYPE_INFO));
		}

		printf("Ending test %d-%d, result: %s\n", test_info.test_index,
			test_info.test_subindex, fi_strerror(-ret));
		results[ft_fw_result_index(-ret)]++;
		ret = ft_fw_send(sock, &ret, sizeof ret);
	} while (!ret);

	fi_freeinfo(hints);
	return ret;
}

static int ft_fw_process_list(struct fi_info *hints, struct fi_info *info)
{
	int ret, subindex, result, sresult;

	for (subindex = 1, fabric_info = info; fabric_info;
	     fabric_info = fabric_info->next, subindex++) {

		printf("Starting test %d-%d: ", series->test_index, subindex);
		ft_show_test_info();
		ret = ft_check_info(hints, fabric_info);
		if (ret)
			return ret;

		ft_fw_update_info(&test_info, fabric_info, subindex);
		ret = ft_fw_send(sock, &test_info, sizeof test_info);
		if (ret)
			return ret;

		result = ft_run_test();

		ret = ft_fw_recv(sock, &sresult, sizeof sresult);
		if (result)
			return result;
		else if (ret)
			return ret;
		else if (sresult)
			return sresult;
	}

	return 0;
}

static int ft_fw_client(void)
{
	struct fi_info *hints, *info;
	int ret;

	hints = fi_allocinfo();
	if (!hints)
		return -FI_ENOMEM;

	for (fts_start(series, test_start_index);
	     !fts_end(series, test_end_index);
	     fts_next(series)) {

		fts_cur_info(series, &test_info);
		ft_fw_convert_info(hints, &test_info);

		printf("Starting test %d / %d\n", test_info.test_index, series->test_count);
		ret = fi_getinfo(FT_FIVERSION, ft_strptr(test_info.node),
				 ft_strptr(test_info.service), 0, hints, &info);
		if (ret) {
			FT_PRINTERR("fi_getinfo", ret);
		} else {
			ret = ft_fw_process_list(hints, info);
			fi_freeinfo(info);
		}

		if (ret) {
			fprintf(stderr, "Node: %s\nService: %s\n",
				test_info.node, test_info.service);
			fprintf(stderr, "%s\n", fi_tostr(hints, FI_TYPE_INFO));
		}

		printf("Ending test %d / %d, result: %s\n",
			test_info.test_index, series->test_count, fi_strerror(-ret));
		results[ft_fw_result_index(-ret)]++;
	}

	fi_freeinfo(hints);
	return 0;
}

static void ft_fw_show_results(void)
{
	printf("Success: %d\n", results[FT_SUCCESS]);
	printf("ENODATA: %d\n", results[FT_ENODATA]);
	printf("ENOSYS : %d\n", results[FT_ENOSYS]);
	printf("ERROR  : %d\n", results[FT_ERROR]);
}

static void ft_fw_usage(char *program)
{
	fprintf(stderr, "usage: %s [server_node]\n", program);
	fprintf(stderr, "\nOptions:\n");
	FT_PRINT_OPTS_USAGE("-f <test_config_file>", "");
	FT_PRINT_OPTS_USAGE("-q <service_port>", "Management port for test");
	FT_PRINT_OPTS_USAGE("-p <dst_port>", "destination port number");
	FT_PRINT_OPTS_USAGE("-x", "exit after test run");
	FT_PRINT_OPTS_USAGE("-y <start_test_index>", "");
	FT_PRINT_OPTS_USAGE("-z <end_test_index>", "");
	FT_PRINT_OPTS_USAGE("-h", "display this help output");
}

int main(int argc, char **argv)
{
	char *service = "2710";
	char *filename = NULL;
	opts = INIT_OPTS;
	int ret, op;

	while ((op = getopt(argc, argv, "f:q:p:xy:z:h")) != -1) {
		switch (op) {
		case 'f':
			filename = optarg;
			break;
		case 'q':
			service = optarg;
			break;
		case 'x':
			persistent = 0;
			break;
		case 'y':
			test_start_index = atoi(optarg);
			break;
		case 'z':
			test_end_index = atoi(optarg);
			break;
		default:
			ft_parse_addr_opts(op, optarg, &opts);
		case '?':
		case 'h':
			ft_fw_usage(argv[0]);
			exit(1);
		}
	}

	if (optind < argc - 1) {
		ft_fw_usage(argv[0]);
		exit(1);
	}

	opts.dst_addr = (optind == argc - 1) ? argv[optind] : NULL;

	if (opts.dst_addr) {
		series = fts_load(filename);
		if (!series)
			exit(1);

		ret = ft_fw_connect(opts.dst_addr, service);
		if (ret)
			goto out;

		ret = ft_fw_client();
		ft_fw_shutdown(sock);
	} else {
		ret = ft_fw_listen(service);
		if (ret)
			goto out;

		do {
			sock = accept(listen_sock, NULL, 0);
			if (sock < 0) {
				ret = sock;
				perror("accept");
			}

			op = 1;
			setsockopt(sock, IPPROTO_TCP, TCP_NODELAY,
				   (void *) &op, sizeof(op));

			ret = ft_fw_server();
			ft_fw_shutdown(sock);
		} while (persistent);
	}

	ft_fw_show_results();
out:
	if (opts.dst_addr)
		fts_close(series);
	return ret;
}
