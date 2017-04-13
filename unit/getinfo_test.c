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
#include <getopt.h>
#include <string.h>

#include <rdma/fi_errno.h>

#include "shared.h"
#include "unit_common.h"

#define TEST_ENTRY_GETINFO(name) TEST_ENTRY(getinfo_ ## name,\
					    getinfo_ ## name ## _desc)

typedef int (*ft_getinfo_init)(struct fi_info *);
typedef int (*ft_getinfo_check)(void *);

static char err_buf[512];
static char old_prov_var[128];
static char new_prov_var[128];

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
	struct fi_info *info = arg;
	return check_addr(info->src_addr, info->src_addrlen, "source");
}

static int check_src_dest_addr(void *arg)
{
	struct fi_info *info = arg;
	int ret;

	ret = check_addr(info->src_addr, info->src_addrlen, "source");
	if (ret)
		return ret;

	return check_addr(info->dest_addr, info->dest_addrlen, "destination");
}

static int check_util_prov(void *arg)
{
	struct fi_info *info = arg;
	const char *util_name;
	size_t len;

	util_name = ft_util_name(info->fabric_attr->prov_name, &len);
	if (!util_name) {
		sprintf(err_buf, "Util provider name not appended to core "
			"provider name: %s", info->fabric_attr->prov_name);
		return EXIT_FAILURE;
	}
	return 0;
}

static int check_api_version(void *arg)
{
	struct fi_info *info = arg;
	return info->fabric_attr->api_version != FT_FIVERSION;
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

#define getinfo_test(name, num, desc, node, service, flags, hints, init, check,	\
		ret_exp)							\
char *getinfo_ ## name ## num ## _desc = desc;					\
static int getinfo_ ## name ## num(void)					\
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


/* 1. No hints tests
 * These tests do not receive hints. If a particular provider has been
 * requested, the environment variable FI_PROVIDER will be set to restrict
 * the provider used. Otherwise, test failures may occur for any provider.
 */

/* 1.1 Source address only tests */
getinfo_test(no_hints, 1, "Test with no node, service, flags or hints",
		NULL, NULL, 0, NULL, NULL, check_srcaddr, 0)
getinfo_test(no_hints, 2, "Test with node, no service, FI_SOURCE flag and no hints",
		opts.src_addr ? opts.src_addr : "localhost", NULL, FI_SOURCE,
		NULL, NULL, check_srcaddr, 0)
getinfo_test(no_hints, 3, "Test with service, FI_SOURCE flag and no node or hints",
		 NULL, opts.src_port, FI_SOURCE, NULL, NULL,
		 check_srcaddr, 0)	// TODO should we check for wildcard addr?
getinfo_test(no_hints, 4, "Test with node, service, FI_SOURCE flags and no hints",
		opts.src_addr ? opts.src_addr : "localhost", opts.src_port,
		FI_SOURCE, NULL, NULL, check_srcaddr, 0)

/* 1.2 Source and destination address tests */
getinfo_test(no_hints, 5, "Test with node, service and no hints",
		opts.dst_addr ? opts.dst_addr : "localhost", opts.dst_port,
		0, NULL, NULL, check_src_dest_addr, 0)

/* 2. Test with hints */
/* 2.1 Source address only tests */
getinfo_test(src, 1, "Test with no node, service, or flags",
		NULL, NULL, 0, hints, NULL, check_srcaddr, 0)
getinfo_test(src, 2, "Test with node, no service, FI_SOURCE flag",
		opts.src_addr ? opts.src_addr : "localhost", NULL, FI_SOURCE,
		hints, NULL, check_srcaddr, 0)
getinfo_test(src, 3, "Test with service, FI_SOURCE flag and no node",
		 NULL, opts.src_port, FI_SOURCE, hints, NULL,
		 check_srcaddr, 0)	// TODO should we check for wildcard addr?
getinfo_test(src, 4, "Test with node, service, FI_SOURCE flags",
		opts.src_addr ? opts.src_addr : "localhost", opts.src_port,
		FI_SOURCE, hints, NULL, check_srcaddr, 0)

/* 2.2 Source and destination address tests */
getinfo_test(src_dest, 1, "Test with node, service",
		opts.dst_addr ? opts.dst_addr : "localhost", opts.dst_port,
		0, hints, NULL, check_src_dest_addr, 0)

getinfo_test(src_dest, 2, "Test API version",
		NULL, NULL, 0, hints, NULL, check_api_version ,0)

/* Negative tests */
getinfo_test(neg, 1, "Test with non-existent domain name",
		NULL, NULL, 0, hints, invalid_dom, NULL, -FI_ENODATA)

/* Utility provider tests */
getinfo_test(util, 1, "Test if we get utility provider when requested",
		NULL, NULL, 0, hints, NULL, check_util_prov, 0)


static void usage(void)
{
	ft_unit_usage("getinfo_test", "Unit tests for fi_getinfo");
	FT_PRINT_OPTS_USAGE("-e <ep_type>", "Endpoint type: msg|rdm|dgram (default:rdm)");
	ft_addr_usage();
}

static void set_prov(char *prov_name)
{
	const char *util_name;
	const char *core_name;
	size_t len;

	snprintf(old_prov_var, sizeof(old_prov_var) - 1, "FI_PROVIDER=%s",
		 getenv("FI_PROVIDER"));

	util_name = ft_util_name(prov_name, &len);
	core_name = ft_core_name(prov_name, &len);

	if (util_name && !core_name)
		return;

	snprintf(new_prov_var, sizeof(new_prov_var) - 1, "FI_PROVIDER=%s",
		 core_name);

	putenv(new_prov_var);
}

static void reset_prov(void)
{
	putenv(old_prov_var);
}

int main(int argc, char **argv)
{
	int failed;
	int op;
	size_t len;
	const char *util_name;

	struct test_entry no_hint_tests[] = {
		TEST_ENTRY_GETINFO(no_hints1),
		TEST_ENTRY_GETINFO(no_hints2),
		TEST_ENTRY_GETINFO(no_hints3),
		TEST_ENTRY_GETINFO(no_hints4),
		TEST_ENTRY_GETINFO(no_hints5),
		{ NULL, "" }
	};

	struct test_entry hint_tests[] = {
		TEST_ENTRY_GETINFO(src1),
		TEST_ENTRY_GETINFO(src2),
		TEST_ENTRY_GETINFO(src3),
		TEST_ENTRY_GETINFO(src4),
		TEST_ENTRY_GETINFO(src_dest1),
		TEST_ENTRY_GETINFO(src_dest2),
		/* This test has to be last getinfo unit test to be run until we
		 * find a way to reset hints->domain_attr->name*/
		TEST_ENTRY_GETINFO(neg1),
		{ NULL, "" }
	};

	struct test_entry util_prov_tests[] = {
		TEST_ENTRY_GETINFO(util1),
		{ NULL, "" }
	};

	opts = INIT_OPTS;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, ADDR_OPTS INFO_OPTS "h")) != -1) {
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

	if (hints->fabric_attr->prov_name) {
		set_prov(hints->fabric_attr->prov_name);
	} else {
	       FT_WARN("\nTests getinfo1 to getinfo5 may not run exclusively "
		       "for a particular provider since we don't pass hints.\n"
		       "So the failures in any of those tests may not be "
		       "attributable to a single provider.\n");
	}

	failed = run_tests(no_hint_tests, err_buf);

	if (hints->fabric_attr->prov_name) {
		reset_prov();
	}

	if (hints->fabric_attr->prov_name) {
		util_name = ft_util_name(hints->fabric_attr->prov_name, &len);
		if (util_name)
			failed += run_tests(util_prov_tests, err_buf);
	}

	failed += run_tests(hint_tests, err_buf);

	if (failed > 0) {
		printf("\nSummary: %d tests failed\n", failed);
	} else {
		printf("\nSummary: all tests passed\n");
	}

	ft_free_res();
	return (failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
