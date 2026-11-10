/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <inttypes.h>

#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>

#include "shared.h"
#include "unit_common.h"

#define AV_CAPACITY_ADDR_LEN 256

/* 
 * Distinct from FI_ADDR_NOTAVAIL so that a failed insert which leaves fi_addr
 * untouched can be told apart from one where the provider correctly set it to
 * FI_ADDR_NOTAVAIL.
 */
#define AV_CAPACITY_ADDR_POISON ((fi_addr_t) 0xdeadbeefdeadbeefULL)

static char base_addr[AV_CAPACITY_ADDR_LEN];

static int insert_one(struct fid_av *test_av, int index, fi_addr_t *fi_addr)
{
	char addr[AV_CAPACITY_ADDR_LEN];
	int len;

	len = snprintf(addr, sizeof(addr), "%s_%d", base_addr, index);
	if (len < 0 || len >= (int) sizeof(addr))
		return -FI_ETOOSMALL;

	*fi_addr = AV_CAPACITY_ADDR_POISON;
	return fi_av_insert(test_av, addr, 1, fi_addr, 0, NULL);
}

static int get_base_addr(void)
{
	size_t len = sizeof(base_addr) - 16;
	int ret;

	memset(base_addr, 0, sizeof(base_addr));
	ret = fi_getname(&ep->fid, base_addr, &len);
	if (ret) {
		FT_PRINTERR("fi_getname", ret);
		return ret;
	}
	base_addr[sizeof(base_addr) - 1] = '\0';
	return 0;
}

static int av_capacity(int min_addrs, int max_addrs)
{
	struct fi_av_attr attr = { .type = FI_AV_TABLE };
	struct fid_av *test_av;
	fi_addr_t *fi_addrs;
	fi_addr_t extra;
	int inserted;
	int ret, close_ret;

	fi_addrs = calloc(max_addrs, sizeof(*fi_addrs));
	if (!fi_addrs)
		return -FI_ENOMEM;

	ret = fi_av_open(domain, &attr, &test_av, NULL);
	if (ret) {
		FT_PRINTERR("fi_av_open", ret);
		goto out;
	}

	for (inserted = 0; inserted < max_addrs; inserted++) {
		ret = insert_one(test_av, inserted, &fi_addrs[inserted]);
		if (ret != 1)
			break;
	}

	if (inserted == 0) {
		printf("cannot insert generated addresses: %d\n", ret);
		ret = -FI_ENOSYS;
		goto close;
	}

	printf("inserted %d addresses\n", inserted);
	if (inserted < min_addrs) {
		printf("AV accepted %d addresses, expected at least %d\n",
		       inserted, min_addrs);
		ret = -FI_EOTHER;
		goto close;
	}

	if (inserted < max_addrs && fi_addrs[inserted] != FI_ADDR_NOTAVAIL) {
		printf("insert beyond the limit returned %d, fi_addr %" PRIu64
		       "\n", ret, fi_addrs[inserted]);
		ret = -FI_EOTHER;
		goto close;
	}

	ret = fi_av_remove(test_av, &fi_addrs[0], 1, 0);
	if (ret) {
		FT_PRINTERR("fi_av_remove", ret);
		goto close;
	}

	ret = insert_one(test_av, inserted, &extra);
	if (ret != 1 || extra == FI_ADDR_NOTAVAIL ||
	    extra == AV_CAPACITY_ADDR_POISON) {
		printf("insert after remove returned %d, fi_addr %" PRIu64 "\n",
		       ret, extra);
		ret = -FI_EOTHER;
		goto close;
	}
	ret = 0;

close:
	close_ret = fi_close(&test_av->fid);
	if (close_ret) {
		FT_PRINTERR("fi_close", close_ret);
		if (!ret)
			ret = close_ret;
	}
out:
	free(fi_addrs);
	return ret;
}

static void usage(char *name)
{
	ft_unit_usage(name, "Unit test for AV capacity");
	FT_PRINT_OPTS_USAGE("-n <num>",
			    "minimum number of addresses the AV must accept (default 1)");
	FT_PRINT_OPTS_USAGE("-m <num>",
			    "maximum number of addresses to insert (default 65536)");
}

int main(int argc, char **argv)
{
	int min_addrs = 1, max_addrs = 65536;
	int op, ret, cleanup_ret;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, FAB_OPTS "n:m:h")) != -1) {
		switch (op) {
		case 'n':
			min_addrs = atoi(optarg);
			break;
		case 'm':
			max_addrs = atoi(optarg);
			break;
		default:
			ft_parseinfo(op, optarg, hints, &opts);
			break;
		case '?':
		case 'h':
			usage(argv[0]);
			return EXIT_FAILURE;
		}
	}

	if (min_addrs <= 0 || max_addrs <= 0 || min_addrs > max_addrs) {
		usage(argv[0]);
		return EXIT_FAILURE;
	}

	hints->ep_attr->type = FI_EP_RDM;
	hints->mode = ~0;
	hints->domain_attr->mode = ~0;
	hints->domain_attr->mr_mode = ~OFI_MR_DEPRECATED;

	ret = ft_getinfo(hints, &fi);
	if (ret)
		goto out;

	if (fi->addr_format != FI_ADDR_STR) {
		printf("address format %s is not supported\n",
		       fi_tostr(&fi->addr_format, FI_TYPE_ADDR_FORMAT));
		ret = -FI_ENOSYS;
		goto out;
	}

	ret = ft_open_fabric_res();
	if (ret)
		goto out;

	ret = ft_alloc_active_res(fi);
	if (ret)
		goto out;

	ret = ft_enable_ep(ep, eq, av, txcq, rxcq, txcntr, rxcntr, rma_cntr);
	if (ret)
		goto out;

	ret = get_base_addr();
	if (ret)
		goto out;

	ret = av_capacity(min_addrs, max_addrs);
out:
	cleanup_ret = ft_free_res();
	return ft_exit_code(ret ? ret : cleanup_ret);
}
