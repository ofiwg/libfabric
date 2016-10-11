/*
 * Copyright (c) 2013-2015 Intel Corporation.  All rights reserved.
 * Copyright (c) 2014 Cisco Systems, Inc.  All rights reserved.
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
 *	- Redistributions of source code must retain the above
 *	  copyright notice, this list of conditions and the following
 *	  disclaimer.
 *
 *	- Redistributions in binary form must reproduce the above
 *	  copyright notice, this list of conditions and the following
 *	  disclaimer in the documentation and/or other materials
 *	  provided with the distribution.
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
#include <errno.h>
#include <getopt.h>
#include <poll.h>
#include <time.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <inttypes.h>
#include <limits.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include "shared.h"
#include "unit_common.h"

typedef int (*ft_getinfo_init)(struct fi_info *);
typedef int (*ft_getinfo_check)(void *);

static char err_buf[512];

static int check_addr(void *addr, size_t addrlen, char *str)
{
	if (!addrlen) {
		sprintf(err_buf, "%s addrlen not set", str);
		return EXIT_FAILURE;
	}
	if (!addr) {
		sprintf(err_buf, "%s address not set", str);
		return EXIT_FAILURE;
	}
	return 0;
}

static int check_srcaddr(void *arg)
{
	struct fi_info *info = (struct fi_info *)arg;
	return check_addr(info->src_addr, info->src_addrlen, "source");
}

static int check_src_dest_addr(void *arg)
{
	struct fi_info *info = (struct fi_info *)arg;
	int ret;

	ret = check_addr(info->src_addr, info->src_addrlen, "source");
	if (ret)
		return ret;

	return check_addr(info->dest_addr, info->dest_addrlen, "destination");
}

int invalid_dom(struct fi_info *hints)
{
	if (hints->domain_attr->name)
		free(hints->domain_attr->name);
	hints->domain_attr->name = strdup("invalid_domain");
	if (!hints->domain_attr->name)
		return -FI_ENOMEM;
	return 0;
}

static int getinfo_unit_test(char *node, char *service, uint64_t flags,
		struct fi_info *hints, ft_getinfo_init init,
		ft_getinfo_check check, int ret_exp)
{
	struct fi_info *info, *fi;
	int ret;

	if (init) {
		ret = init(hints);
		if (ret)
			return ret;
	}

	ret = fi_getinfo(FT_FIVERSION, node, service, flags, hints, &info);
	if (ret) {
		if (ret == ret_exp)
			return 0;
		sprintf(err_buf, "fi_getinfo failed %s(%d)", fi_strerror(-ret), -ret);
		return ret;
	}

	if (!check)
		goto out;

	ft_foreach_info(fi, info) {
		FT_DEBUG("\nTesting for fabric: %s, domain: %s, endpoint type: %d",
				fi->fabric_attr->name, fi->domain_attr->name,
				fi->ep_attr->type);
		ret = check(info);
		if (ret)
			break;
	}
out:
	fi_freeinfo(info);
	return ret;
}

#define getinfo_test(num, desc, node, service, flags, hints, init, check,	\
		ret_exp)							\
char *getinfo ## num ## _desc = desc;						\
static int getinfo ## num(void)							\
{										\
	int ret, testret = FAIL;						\
	ret = getinfo_unit_test(node, service, flags, hints, init, check,	\
			ret_exp);						\
	if (ret)								\
		goto fail;							\
	testret = PASS;								\
fail:										\
	return TEST_RET_VAL(ret, testret);					\
}

/*
 * Tests:
 */

getinfo_test(1, "Test with no node, service or flags",
		NULL, NULL, 0, hints, NULL, check_srcaddr, 0)
getinfo_test(2, "Test with no node, valid service and FI_SOURCE flag",
		NULL, opts.src_port, FI_SOURCE, hints, NULL, check_srcaddr, 0)
getinfo_test(3, "Test with node, valid service and FI_SOURCE flag",
		opts.src_addr ? opts.src_addr : "localhost", opts.src_port,
		FI_SOURCE, hints, NULL, check_srcaddr, 0)
getinfo_test(4, "Test with node, no service and FI_SOURCE flag",
		opts.src_addr ? opts.src_addr : "localhost", NULL,
		FI_SOURCE, hints, NULL, check_srcaddr, 0)
getinfo_test(5, "Test with node, service and no flags",
		opts.dst_addr ? opts.dst_addr : "localhost", opts.dst_port,
		0, hints, NULL, check_src_dest_addr, 0)
getinfo_test(6, "Test with non-existent domain name",
		NULL, NULL, 0, hints, invalid_dom, NULL, -FI_ENODATA)

static void usage(void)
{
	ft_unit_usage("getinfo_test", "Unit tests for fi_getinfo");
}

int main(int argc, char **argv)
{
	int failed;
	int op;

	struct test_entry test_array[] = {
		TEST_ENTRY(getinfo1, getinfo1_desc),
		TEST_ENTRY(getinfo2, getinfo2_desc),
		TEST_ENTRY(getinfo3, getinfo3_desc),
		TEST_ENTRY(getinfo4, getinfo4_desc),
		TEST_ENTRY(getinfo5, getinfo5_desc),
		TEST_ENTRY(getinfo6, getinfo6_desc),
		{ NULL, "" }
	};

	opts = INIT_OPTS;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, ADDR_OPTS "p:h")) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints);
			break;
		case 'h':
			usage();
			return EXIT_SUCCESS;
		case '?':
			usage();
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];
	if (!opts.dst_port)
		opts.dst_port = "9228";
	if (!opts.src_port)
		opts.src_port = "9228";

	hints->mode = ~0;

	failed = run_tests(test_array, err_buf);
	if (failed > 0) {
		printf("\nSummary: %d tests failed\n", failed);
	} else {
		printf("\nSummary: all tests passed\n");
	}

	ft_free_res();
	return (failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
