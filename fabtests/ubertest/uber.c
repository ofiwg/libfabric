/*
 * Copyright (c) 2013-2017 Intel Corporation.  All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <netdb.h>
#include <unistd.h>
#include <sys/wait.h>

#include <limits.h>
#include <shared.h>

#include "fabtest.h"

static int persistent = 1;
static int do_fork = 0;

//static struct timespec start, end;

static struct ft_series *series;
static int test_start_index, test_end_index = INT_MAX;
struct ft_info test_info;
struct fi_info *fabric_info;
struct ft_xcontrol ft_rx_ctrl, ft_tx_ctrl;
struct ft_mr_control ft_mr_ctrl;
struct ft_atomic_control ft_atom_ctrl;
struct ft_control ft_ctrl;

size_t recv_size, send_size;

enum {
	FT_SUCCESS,
	FT_SKIP,
	FT_ENODATA,
	FT_ENOSYS,
	FT_ERROR,
	FT_EIO,
	FT_MAX_RESULT
};

static int results[FT_MAX_RESULT];
static char *filename = NULL;
static char *provname = NULL;
static char *testname = NULL;


static int ft_nullstr(char *str)
{
	return (!str || str[0] == '\0');
}

static char *ft_strptr(char *str)
{
	return ft_nullstr(str) ? NULL : str;
}

static char *ft_test_type_str(enum ft_test_type enum_str)
{
	switch (enum_str) {
	case FT_TEST_LATENCY:
		return "latency";
	case FT_TEST_BANDWIDTH:
		return "bandwidth";
	case FT_TEST_UNIT:
		return "unit";
	default:
		return "test_unspec";
	}
}

static char *ft_class_func_str(enum ft_class_function enum_str)
{
	switch (enum_str) {
	case FT_FUNC_SEND:
		return (test_info.test_class & FI_TAGGED) ? "tsend" : "send";
	case FT_FUNC_SENDV:
		return (test_info.test_class & FI_TAGGED) ? "tsendv" : "sendv";
	case FT_FUNC_SENDMSG:
		return (test_info.test_class & FI_TAGGED) ? "tsendmsg" : "sendmsg";
	case FT_FUNC_INJECT:
		return (test_info.test_class & FI_TAGGED) ? "tinject" : "inject";
	case FT_FUNC_INJECTDATA:
		return (test_info.test_class & FI_TAGGED) ? "tinjectdata" : "injectdata";
	case FT_FUNC_SENDDATA:
		return (test_info.test_class & FI_TAGGED) ? "tsenddata" : "senddata";
	case FT_FUNC_READ:
		return "read";
	case FT_FUNC_READV:
		return "readv";
	case FT_FUNC_READMSG:
		return "readmsg";
	case FT_FUNC_WRITE:
		return "write";
	case FT_FUNC_WRITEV:
		return "writev";
	case FT_FUNC_WRITEMSG:
		return "writemsg";
	case FT_FUNC_INJECT_WRITE:
		return "inject_write";
	case FT_FUNC_WRITEDATA:
		return "writedata";
	case FT_FUNC_INJECT_WRITEDATA:
		return "inject_writedata";
	case FT_FUNC_ATOMIC:
		return "atomic";
	case FT_FUNC_ATOMICV:
		return "atomicv";
	case FT_FUNC_ATOMICMSG:
		return "atomic_msg";
	case FT_FUNC_INJECT_ATOMIC:
		return "inject_atomic";
	case FT_FUNC_FETCH_ATOMIC:
		return "fetch_atomic";
	case FT_FUNC_FETCH_ATOMICV:
		return "fetch_atomicv";
	case FT_FUNC_FETCH_ATOMICMSG:
		return "fetch_atomicmsg";
	case FT_FUNC_COMPARE_ATOMIC:
		return "compare_atomic";
	case FT_FUNC_COMPARE_ATOMICV:
		return "compare_atomicv";
	case FT_FUNC_COMPARE_ATOMICMSG:
		return "compare_atomicmsg";
	default:
		return "func_unspec";
	}
}

static char *ft_wait_obj_str(enum fi_wait_obj enum_str)
{
	switch (enum_str) {
	case FI_WAIT_NONE:
		return "wait_none";
	case FI_WAIT_UNSPEC:
		return "wait_unspec";
	case FI_WAIT_SET:
		return "wait_set";
	case FI_WAIT_FD:
		return "wait_fd";
	case FI_WAIT_MUTEX_COND:
		return "wait_mutex_cond";
	default:
		return "";
	}
}

static void ft_print_comp_flag(uint64_t bind, uint64_t op)
{
	printf("(bind-%s, op-%s)", (bind & FI_SELECTIVE_COMPLETION) ?
		"FI_SELECTIVE_COMPLETION" : "NONE",
		op ? fi_tostr(&op, FI_TYPE_OP_FLAGS) : "NONE");
}

static void ft_print_comp(struct ft_info *test)
{
	if (ft_use_comp_cq(test->comp_type))
		printf("comp_queue -- tx ");
	if (ft_use_comp_cntr(test->comp_type))
		printf("comp_cntr -- tx ");

	ft_print_comp_flag(test->tx_cq_bind_flags, test->tx_op_flags);
	printf(", rx: ");
	ft_print_comp_flag(test->rx_cq_bind_flags, test->rx_op_flags);
	printf(", ");
} 

static void ft_show_test_info(void)
{
	printf("[%s,", test_info.prov_name);
	printf(" %s,", ft_test_type_str(test_info.test_type));
	if (test_info.test_class & FI_ATOMIC) {
		printf(" %s ", ft_class_func_str(test_info.class_function));
		printf("(%s, ", fi_tostr(&test_info.op, FI_TYPE_ATOMIC_OP));
		printf("%s)--", fi_tostr(&test_info.datatype, FI_TYPE_ATOMIC_TYPE));
	} else {
		printf(" %s--", ft_class_func_str(test_info.class_function));
	}
	printf("%s,", fi_tostr(&test_info.msg_flags, FI_TYPE_OP_FLAGS));
	printf(" %s,", fi_tostr(&test_info.ep_type, FI_TYPE_EP_TYPE));
	printf(" %s,", fi_tostr(&test_info.av_type, FI_TYPE_AV_TYPE));
	printf(" eq_%s,", ft_wait_obj_str(test_info.eq_wait_obj));
	printf(" cq_%s,", ft_wait_obj_str(test_info.cq_wait_obj));
	printf(" cntr_%s, ", ft_wait_obj_str(test_info.cq_wait_obj));
	ft_print_comp(&test_info);
	printf(" %s,", fi_tostr(&test_info.progress, FI_TYPE_PROGRESS));
	printf(" %s (only hints),", fi_tostr(&test_info.threading, FI_TYPE_THREADING));
	printf(" [%s],", fi_tostr(&test_info.mr_mode, FI_TYPE_MR_MODE));
	printf(" [%s],", fi_tostr(&test_info.mode, FI_TYPE_MODE));
	printf(" [%s]]\n", fi_tostr(&test_info.caps, FI_TYPE_CAPS));
}

static int ft_check_info(struct fi_info *hints, struct fi_info *info)
{
	if (info->mode & ~hints->mode) {
		fprintf(stderr, "fi_getinfo unsupported mode returned:\n");
		fprintf(stderr, "hints mode: %s\n",
			fi_tostr(&hints->mode, FI_TYPE_MODE));
		fprintf(stderr, "info mode: %s\n",
			fi_tostr(&info->mode, FI_TYPE_MODE));
		return -FI_EINVAL;
	}
	if (hints->caps != (hints->caps & info->caps)) {
		fprintf(stderr, "fi_getinfo missing caps:\n");
		fprintf(stderr, "hints caps: %s\n",
			fi_tostr(&hints->caps, FI_TYPE_CAPS));
		fprintf(stderr, "info caps: %s\n",
			fi_tostr(&info->caps, FI_TYPE_CAPS));
		return -FI_EINVAL;
	}

	return 0;
}

static void ft_fw_convert_info(struct fi_info *info, struct ft_info *test_info)
{
	info->caps = test_info->caps;

	info->mode = test_info->mode;

	info->domain_attr->mr_mode = test_info->mr_mode;
	info->domain_attr->av_type = test_info->av_type;
	info->domain_attr->data_progress = test_info->progress;
	info->domain_attr->threading = test_info->threading;

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

	info->tx_attr->op_flags = test_info->tx_op_flags;
	info->rx_attr->op_flags = test_info->rx_op_flags;

	if (is_data_func(test_info->class_function) ||
	    (is_msg_func(test_info->class_function) &&
	     test_info->msg_flags & FI_REMOTE_CQ_DATA))
		info->domain_attr->cq_data_size = 4;
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
				sizeof test_info->prov_name - 1);
		}
		if (info->fabric_attr->name) {
			strncpy(test_info->fabric_name, info->fabric_attr->name,
				sizeof test_info->fabric_name - 1);
		}
	}

	if (info->domain_attr) {
		test_info->progress = info->domain_attr->data_progress;
		test_info->threading = info->domain_attr->threading;
	}

	test_info->mode = info->mode;
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
	case FI_EIO:
		return FT_EIO;
	default:
		return FT_ERROR;
	}
}

static int ft_recv_test_info(void)
{
	int ret;

	ret = ft_sock_recv(sock, &test_info, sizeof test_info);
	if (ret)
		return ret;

	test_info.node[sizeof(test_info.node) - 1] = '\0';
	test_info.service[sizeof(test_info.service) - 1] = '\0';
	test_info.prov_name[sizeof(test_info.prov_name) - 1] = '\0';
	test_info.fabric_name[sizeof(test_info.fabric_name) - 1] = '\0';
	return 0;
}

static int ft_exchange_uint32(uint32_t local, uint32_t *remote)
{
	uint32_t local_net = htonl(local);
	int ret;

	ret = ft_sock_send(sock, &local_net, sizeof local);
	if (ret) {
		FT_PRINTERR("ft_sock_send", ret);
		return ret;
	}

	ret = ft_sock_recv(sock, remote, sizeof *remote);
	if (ret) {
		FT_PRINTERR("ft_sock_recv", ret);
		return ret;
	}

	*remote = ntohl(*remote);

	return 0;
}

static int ft_skip_info(struct fi_info *hints, struct fi_info *info)
{
	uint32_t remote_protocol, skip, remote_skip;
	size_t len;
	int ret;

	//make sure remote side is using the same protocol
	ret = ft_exchange_uint32(info->ep_attr->protocol, &remote_protocol);
	if (ret)
		return ret;

	if (info->ep_attr->protocol != remote_protocol)
		return 1;

	//check needed to skip utility providers, unless requested
	skip = (!ft_util_name(hints->fabric_attr->prov_name, &len) &&
		strcmp(hints->fabric_attr->prov_name,
		info->fabric_attr->prov_name));

	ret = ft_exchange_uint32(skip, &remote_skip);
	if (ret)
		return ret;

	return skip || remote_skip;
}

static int ft_transfer_subindex(int subindex, int *remote_idx)
{
	int ret;

	ret = ft_sock_send(sock, &subindex, sizeof subindex);
	if (ret) {
		FT_PRINTERR("ft_sock_send", ret);
		return ret;
	}

	ret = ft_sock_recv(sock, remote_idx, sizeof *remote_idx);
	if (ret) {
		FT_PRINTERR("ft_sock_recv", ret);
		return ret;
	}

	return 0;
}

static int ft_fw_process_list_server(struct fi_info *hints, struct fi_info *info)
{
	int ret, subindex, remote_idx = 0, result = -FI_ENODATA, end_test = 0;
	int server_ready = 0;
	struct fi_info *open_res_info;

	ret = ft_sock_send(sock, &test_info, sizeof test_info);
	if (ret) {
		FT_PRINTERR("ft_sock_send", ret);
		return ret;
	}

	for (subindex = 1, fabric_info = info; fabric_info;
	     fabric_info = fabric_info->next, subindex++) {

		ret = ft_check_info(hints, fabric_info);
		if (ret)
			return ret;

		/* Stores the fabric_info into a tmp variable, resolves an issue caused
		*  by ft_accept with FI_EP_MSG which overwrites the fabric_info.
		*/
		open_res_info = fabric_info;
		while (1) {
			fabric_info = open_res_info;
			ret = ft_open_res();
			if (ret) {
				FT_PRINTERR("ft_open_res", ret);
				return ret;
			}

			if (!server_ready) {
				server_ready = 1;
				ret = ft_sock_send(sock, &server_ready, sizeof server_ready);
				if (ret) {
					FT_PRINTERR("ft_sock_send", ret);
					return ret;
				}
			}

			ret = ft_sock_recv(sock, &end_test, sizeof end_test);
			if (ret) {
				FT_PRINTERR("ft_sock_recv", ret);
				return ret;
			}
			if (end_test) {
				ft_cleanup();
				break;
			}

			if (ft_skip_info(hints, fabric_info)) {
				ft_cleanup();
				continue;
			}

			ret = ft_transfer_subindex(subindex, &remote_idx);
			if (ret)
				return ret;

			ft_fw_update_info(&test_info, fabric_info, subindex);

			printf("Starting test %d-%d-%d: ", test_info.test_index,
				subindex, remote_idx);
			ft_show_test_info();

			result = ft_init_test();
			if (result)
				continue;

			result = ft_run_test();

			ret = ft_sock_send(sock, &result, sizeof result);
			if (result) {
				FT_PRINTERR("ft_run_test", result);
			} else if (ret) {
				FT_PRINTERR("ft_sock_send", ret);
				return ret;
			}
		}

		end_test = (fabric_info->next == NULL);
		ret = ft_sock_send(sock, &end_test, sizeof end_test);
		if (ret) {
			FT_PRINTERR("ft_sock_send", ret);
			return ret;
		}
	}

	test_info.prov_name[0] = '\0';
	ret = ft_sock_send(sock, &test_info, sizeof test_info);
	if (ret) {
		FT_PRINTERR("ft_sock_send", ret);
		return ret;
	}

	if (subindex == 1)
		return -FI_ENODATA;

	return result;
}

static int ft_fw_process_list_client(struct fi_info *hints, struct fi_info *info)
{
	int ret, subindex, remote_idx = 0, result = -FI_ENODATA, sresult, end_test = 0;

	while (!end_test) {
		for (subindex = 1, fabric_info = info; fabric_info;
			 fabric_info = fabric_info->next, subindex++) {

			end_test = 0;
			ret = ft_sock_send(sock, &end_test, sizeof end_test);
			if (ret) {
				FT_PRINTERR("ft_sock_send", ret);
				return ret;
			}

			if (ft_skip_info(hints, fabric_info))
				continue;

			ret = ft_transfer_subindex(subindex, &remote_idx);
			if (ret)
				return ret;

			ret = ft_check_info(hints, fabric_info);
			if (ret)
				return ret;

			ft_fw_update_info(&test_info, fabric_info, subindex);
			printf("Starting test %d-%d-%d: ", test_info.test_index,
				subindex, remote_idx);
			ft_show_test_info();

			ret = ft_open_res();
			if (ret) {
				FT_PRINTERR("ft_open_res", ret);
				return ret;
			}

			result = ft_init_test();
			if (result)
				continue;

			result = ft_run_test();

			ret = ft_sock_recv(sock, &sresult, sizeof sresult);
			if (result && result != -FI_EIO) {
				FT_PRINTERR("ft_run_test", result);
				fprintf(stderr, "Node: %s\nService: %s \n",
					test_info.node, test_info.service);
				fprintf(stderr, "%s\n", fi_tostr(hints, FI_TYPE_INFO));
				return -FI_EOTHER;
			} else if (ret) {
				FT_PRINTERR("ft_sock_recv", ret);
				result = ret;
				return -FI_EOTHER;
			} else if (sresult) {
				result = sresult;
				if (sresult != -FI_EIO)
					return -FI_EOTHER;
			}
		}
		end_test = 1;
		ret = ft_sock_send(sock, &end_test, sizeof end_test);
		if (ret) {
			FT_PRINTERR("ft_sock_send", ret);
			return ret;
		}

		ret = ft_sock_recv(sock, &end_test, sizeof end_test);
		if (ret) {
			FT_PRINTERR("ft_sock_recv", ret);
			return ret;
		}
	}

	if (subindex == 1)
		return -FI_ENODATA;

	return result;
}

static int ft_server_child()
{
	struct fi_info *hints, *info;
	int ret;

	hints = fi_allocinfo();
	if (!hints)
		return -FI_ENOMEM;

	ft_fw_convert_info(hints, &test_info);
	printf("Starting test %d:\n", test_info.test_index);

	ret = fi_getinfo(FT_FIVERSION, ft_strptr(test_info.node),
			 ft_strptr(test_info.service), FI_SOURCE,
			 hints, &info);
	if (ret && ret != -FI_ENODATA) {
		FT_PRINTERR("fi_getinfo", ret);
	} else {
		ret = ft_fw_process_list_server(hints, info);
		if (ret != -FI_ENODATA)
			fi_freeinfo(info);

		if (ret && ret != -FI_EIO) {
			FT_PRINTERR("ft_fw_process_list", ret);
			printf("Node: %s\nService: %s\n",
				test_info.node, test_info.service);
			printf("%s\n", fi_tostr(hints, FI_TYPE_INFO));
		}
	}
	fi_freeinfo(hints);

	printf("Ending test %d, result: %s\n", test_info.test_index,
		fi_strerror(-ret));

	return ret;
}

static int ft_fw_server(void)
{
	int ret;
	pid_t pid;

	do {
		ret = ft_recv_test_info();
		if (ret) {
			if (ret == -FI_ENOTCONN)
				ret = 0;
			break;
		}

		if (do_fork) {
			pid = fork();
			if (!pid) {
				ret = ft_server_child();
				_exit(-ret);
			} else {
				waitpid(pid, &ret, 0);
				ret = WEXITSTATUS(ret);
			}
		} else {
			ret = ft_server_child();
		}

		results[ft_fw_result_index(ret)]++;

	} while (!ret || ret == FI_EIO || ret == FI_ENODATA);

	return ret;
}

static int ft_client_child(void)
{
	struct fi_info *hints, *info;
	int ret, result, server_ready = 0;

	result = -FI_ENODATA;
	hints = fi_allocinfo();
	if (!hints)
		return -FI_ENOMEM;

	ret = ft_getsrcaddr(opts.src_addr, opts.src_port, hints);
	if (ret)
		return ret;

	ft_fw_convert_info(hints, &test_info);

	printf("Starting test %d / %d:\n", test_info.test_index,
		series->test_count);
	while (!ft_nullstr(test_info.prov_name)) {
		printf("Starting test %d-%d: ", test_info.test_index,
			test_info.test_subindex);
		ft_show_test_info();

		ret = ft_sock_recv(sock, &server_ready, sizeof server_ready);
		if (ret)
			return ret;

		if (!server_ready)
			return -FI_EOTHER;

		result = fi_getinfo(FT_FIVERSION, ft_strptr(test_info.node),
				 ft_strptr(test_info.service), 0, hints, &info);
		if (result) {
			FT_PRINTERR("fi_getinfo", result);
		}

		ret = ft_fw_process_list_client(hints, info);
		if (ret != -FI_ENODATA)
			fi_freeinfo(info);
		else
			goto out;

		ret = ft_recv_test_info();
		if (ret) {
			FT_PRINTERR("ft_recv_test_info", ret);
			goto out;
		}
		ft_fw_convert_info(hints, &test_info);
	}

	printf("Ending test %d / %d, result: %s\n", test_info.test_index,
		series->test_count, fi_strerror(-result));
out:
	fi_freeinfo(hints);
	return result;
}

static int ft_fw_client(void)
{
	int ret, result;
	pid_t pid;


	for (fts_start(series, test_start_index);
	     !fts_end(series, test_end_index);
	     fts_next(series)) {

		fts_cur_info(series, &test_info);
		if (!fts_info_is_valid()) {
			printf("Skipping test %d (invalid):\n", test_info.test_index);
			ft_show_test_info();
			results[FT_SKIP]++;
			continue;
		}

		ret = ft_sock_send(sock, &test_info, sizeof test_info);
		if (ret) {
			FT_PRINTERR("ft_sock_send", ret);
			return ret;
		}

		ret = ft_recv_test_info();
		if (ret) {
			FT_PRINTERR("ft_recv_test_info", ret);
			return ret;
		}

		if (do_fork) {
			pid = fork();
			if (!pid) {
				result = ft_client_child();
				_exit(-result);
			} else {
				waitpid(pid, &result, 0);
				result = WEXITSTATUS(result);
			}
		} else {
			result = ft_client_child();
		}

		results[ft_fw_result_index(result)]++;
	}
	return 0;
}

static void ft_fw_show_results(void)
{
	printf("Success: %d\n", results[FT_SUCCESS]);
	printf("Skipped: %d\n", results[FT_SKIP]);
	printf("ENODATA: %d\n", results[FT_ENODATA]);
	printf("ENOSYS : %d\n", results[FT_ENOSYS]);
	printf("EIO    : %d\n", results[FT_EIO]);
	printf("ERROR  : %d\n", results[FT_ERROR]);
}

static void ft_fw_usage(char *program)
{
	fprintf(stderr, "Usage:\n");
	fprintf(stderr, "  %s [OPTIONS] \t\t\tstart server\n", program);
	fprintf(stderr, "  %s [OPTIONS] <server_node> \tconnect to server\n", program);
	fprintf(stderr, "\nOptions:\n");
	FT_PRINT_OPTS_USAGE("-q <service_port>", "Management port for test");
	FT_PRINT_OPTS_USAGE("-h", "display this help output");
	fprintf(stderr, "\nServer only options:\n");
	FT_PRINT_OPTS_USAGE("-x", "exit after test run");
	fprintf(stderr, "\nClient only options:\n");
	FT_PRINT_OPTS_USAGE("-u <test_config_file>", "test configuration file "
		"(Either config file or both provider and test name are required)");
	FT_PRINT_OPTS_USAGE("-p <provider_name>", " provider name");
	FT_PRINT_OPTS_USAGE("-t <test_name>", "test name");
	FT_PRINT_OPTS_USAGE("-y <start_test_index>", "");
	FT_PRINT_OPTS_USAGE("-z <end_test_index>", "");
	FT_PRINT_OPTS_USAGE("-s <address>", "source address");
	FT_PRINT_OPTS_USAGE("-B <src_port>", "non default source port number");
	FT_PRINT_OPTS_USAGE("-P <dst_port>", "non default destination port number "
		"(config file service parameter will override this)");
}

void ft_free()
{
	if (filename)
		free(filename);
	if (testname)
		free(testname);
	if (provname)
		free(provname);
}

static int ft_get_config_file(char *provname, char *testname, char **filename)
{
	char **prov_vec, **path_vec, *str;
	size_t i, prov_count, path_count, len;
	int ret = -FI_ENOMEM;

	// TODO use macro for ";"
	prov_vec = ft_split_and_alloc(provname, ";", &prov_count);
	if (!prov_vec) {
		FT_ERR("Unable to split provname\n");
		return -FI_EINVAL;
	}

	/* prov_count + count_of(CONFIG_PATH, "test_configs", "testname", ".test") */
	path_count = prov_count + 4;
	path_vec = calloc(path_count, sizeof(*path_vec));
	if (!path_vec)
		goto err1;

	path_vec[0] = CONFIG_PATH;
	path_vec[1] = "test_configs";

	/* Path for "prov1;prov2;prov3;..." is ".../prov3/prov2/prov1" */
	for (i = 0; i < prov_count; i++)
		path_vec[i + 2] = prov_vec[prov_count - i - 1];

	path_vec[prov_count + 2] = testname;
	path_vec[prov_count + 3] = "test";

	for (i = 0, len = 0; i < path_count; i++)
		len += strlen(path_vec[i]) + 1;

	// NULL char at the end
	len++;

	*filename = calloc(1, len);
	if (!*filename)
		goto err2;

	for (i = 0, str = *filename; i < path_count; i++) {
		if (i < path_count - 1)
			ret = snprintf(str, len, "/%s", path_vec[i]);
		else
			ret = snprintf(str, len, ".%s", path_vec[i]);
		if (ret < 0)
			goto err3;

		if (ret >= (int)len) {
			ret = -FI_ETRUNC;
			goto err3;
		}
		str += ret;
		len -= ret;
	}
	free(path_vec);
	ft_free_string_array(prov_vec);
	return 0;
err3:
	free(*filename);
	*filename = NULL;
err2:
	free(path_vec);
err1:
	ft_free_string_array(prov_vec);
	return ret;
}

int main(int argc, char **argv)
{
	char *service = "2710";
	opts = INIT_OPTS;
	int ret, op;

	while ((op = getopt(argc, argv, "p:u:t:q:xy:z:hf" ADDR_OPTS)) != -1) {
		switch (op) {
		case 'u':
			filename = strdup(optarg);
			break;
		case 'p':
			provname = strdup(optarg);
			break;
		case 't':
			testname = strdup(optarg);
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
		case 'f':
			do_fork = 1;
			break;
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			ft_fw_usage(argv[0]);
			ft_free();
			exit(1);
		}
	}

	if (optind < argc - 1) {
		ft_fw_usage(argv[0]);
		ft_free();
		exit(1);
	}

	opts.dst_addr = (optind == argc - 1) ? argv[optind] : NULL;
	if (opts.dst_addr) {
		if (!opts.dst_port)
			opts.dst_port = default_port;
		if (!filename) {
			if (!testname || !provname) {
				ft_fw_usage(argv[0]);
				ft_free();
				exit(1);
			} else {
				ret = ft_get_config_file(provname, testname,
							 &filename);
				if (ret < 0) {
					ft_free();
					exit(1);
				}
			}
		} else {
			testname = NULL;
			provname = NULL;
		}
		series = fts_load(filename);
		if (!series) {
			ft_free();
			exit(1);
		}

		ret = ft_sock_connect(opts.dst_addr, service);
		if (ret)
			goto out;

		ret = ft_fw_client();
		if (ret)
			FT_PRINTERR("ft_fw_client", ret);
		ft_sock_shutdown(sock);
	} else {
		ret = ft_sock_listen(opts.src_addr, service);
		if (ret)
			goto out;

		do {
			ret = ft_sock_accept();
			if (ret)
				goto out;

			ret = ft_fw_server();
			if (ret)
				FT_PRINTERR("ft_fw_server", ret);
			ft_sock_shutdown(sock);
		} while (persistent);
	}

	ft_fw_show_results();
out:
	if (opts.dst_addr)
		fts_close(series);
	ft_free();
	return ft_exit_code(ret);
}
