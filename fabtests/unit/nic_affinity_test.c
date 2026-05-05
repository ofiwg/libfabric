/*
 * Copyright (c) 2013-2015 Intel Corporation.  All rights reserved.
 * Copyright (c) 2014-2017 Cisco Systems, Inc.  All rights reserved.
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
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

#define TEST_ENTRY_NIC_AFFINITY(name) TEST_ENTRY(nic_affinity_ ## name,\
						    nic_affinity_ ## name ## _desc)

typedef int (*ft_nic_affinity_init)(struct fi_info *);
typedef int (*ft_nic_affinity_test)(char *, char *, uint64_t, struct fi_info *,
					struct fi_info **);

static char err_buf[512];
static char new_prov_var[128];

#define OFI_CORE_PROV_ONLY	(1ULL << 59)
#define TEST_PCI_ADDR "0000:00:00.0"
#define TEST_CONFIG_FILE "/tmp/verbs_test_config.conf"

static const char *get_nic_name(struct fi_info *info)
{
	if (info->nic && info->nic->device_attr && info->nic->device_attr->name)
		return info->nic->device_attr->name;
	return NULL;
}

static int get_arbitrary_nic_name(char *nic_name, size_t len)
{
	struct fi_info *info = NULL;
	int ret;

	ret = fi_getinfo(FT_FIVERSION, NULL, NULL, 0, hints, &info);
	if (ret) {
		sprintf(err_buf, "fi_getinfo failed to discover NICs: %s", fi_strerror(-ret));
		return ret;
	}

	if (!info) {
		sprintf(err_buf, "No provider info returned");
		return -FI_ENODATA;
	}

	for (struct fi_info *cur = info; cur; cur = cur->next) {
		const char *name = get_nic_name(cur);
		if (name) {
			snprintf(nic_name, len, "%s", name);
			fi_freeinfo(info);
			return 0;
		}
	}

	fi_freeinfo(info);
	sprintf(err_buf, "No NIC names found in provider info");
	return -FI_ENODATA;
}

static int create_valid_test_config_file(void)
{
	FILE *fp;
	char nic_name[64];
	int ret;

	ret = get_arbitrary_nic_name(nic_name, sizeof(nic_name));
	if (ret) return ret;

	fp = fopen(TEST_CONFIG_FILE, "w");
	if (!fp) {
		sprintf(err_buf, "Failed to open config file for writing");
		return -FI_EIO;
	}

	fprintf(fp, "%s %s\n", TEST_PCI_ADDR, nic_name);
	fclose(fp);

	return 0;
}

static int create_invalid_test_config_file(void)
{
	FILE *fp;

	fp = fopen(TEST_CONFIG_FILE, "w");
	if (!fp) {
		sprintf(err_buf, "Failed to open config file for writing");
		return -FI_EIO;
	}

	fprintf(fp, "invalid_config_line\n");
	fclose(fp);

	return 0;
}

static void cleanup_verbs_affinity_test(void)
{
	unlink(TEST_CONFIG_FILE);
	unsetenv("FI_VERBS_NIC_AFFINITY_POLICY");
	unsetenv("FI_VERBS_AFFINITY_DEVICE");
	unsetenv("FI_VERBS_NIC_AFFINITY_CONFIG");
}

/*
 * Verbs GPU/NIC affinity init functions
 */
static int init_verbs_affinity_manual(struct fi_info *hints)
{
	int ret;

	ret = create_valid_test_config_file();
	if (ret) return ret;

	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "manual", 1);
	setenv("FI_VERBS_NIC_AFFINITY_CONFIG", TEST_CONFIG_FILE, 1);
	return 0;
}

static int init_verbs_affinity_manual_no_device(struct fi_info *hints)
{
	int ret;

	ret = create_valid_test_config_file();
	if (ret) return ret;

	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "manual", 1);
	setenv("FI_VERBS_NIC_AFFINITY_CONFIG", TEST_CONFIG_FILE, 1);
	unsetenv("FI_VERBS_AFFINITY_DEVICE");
	return 0;
}

static int init_verbs_affinity_manual_invalid_device(struct fi_info *hints)
{
	int ret;

	ret = create_valid_test_config_file();
	if (ret) return ret;

	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "manual", 1);
	setenv("FI_VERBS_NIC_AFFINITY_CONFIG", TEST_CONFIG_FILE, 1);
	setenv("FI_VERBS_AFFINITY_DEVICE", "invalid:pci:format:bad", 1);
	return 0;
}

static int init_verbs_affinity_manual_missing_config(struct fi_info *hints)
{
	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "manual", 1);
	setenv("FI_VERBS_NIC_AFFINITY_CONFIG", "/nonexistent/path/to/config.conf", 1);
	return 0;
}

static int init_verbs_affinity_manual_malformed_config(struct fi_info *hints)
{
	int ret;

	ret = create_invalid_test_config_file();
	if (ret) return ret;

	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "manual", 1);
	setenv("FI_VERBS_NIC_AFFINITY_CONFIG", TEST_CONFIG_FILE, 1);
	return 0;
}

static int init_verbs_affinity_auto(struct fi_info *hints)
{
	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "auto", 1);
	return 0;
}

static int init_verbs_affinity_auto_no_device(struct fi_info *hints)
{
	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "auto", 1);
	unsetenv("FI_VERBS_AFFINITY_DEVICE");
	return 0;
}

static int init_verbs_affinity_auto_invalid_device(struct fi_info *hints)
{
	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "auto", 1);
	setenv("FI_VERBS_AFFINITY_DEVICE", "invalid:pci:format:bad", 1);
	return 0;
}

static int init_verbs_affinity_invalid(struct fi_info *hints)
{
	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "invalid_garbage_policy", 1);
	return 0;
}

/*
 * Verbs GPU/NIC affinity check functions
 */
static int check_count_and_grouping(struct fi_info *original_info, struct fi_info *policy_info)
{
	struct fi_info *original_cur;
	struct fi_info *affinity_cur;
	const char *nic_to_find;
	size_t original_count;
	size_t policy_count;

	original_cur = original_info;
	while (original_cur) {
		nic_to_find = get_nic_name(original_cur);
		if (!nic_to_find) {
			original_cur = original_cur->next;
			continue;
		}

		original_count = 0;
		while (original_cur && get_nic_name(original_cur) &&
		       strcmp(get_nic_name(original_cur), nic_to_find) == 0) {
			original_count++;
			original_cur = original_cur->next;
		}

		for (affinity_cur = policy_info; affinity_cur; affinity_cur = affinity_cur->next) {
			if (get_nic_name(affinity_cur) &&
			    strcmp(get_nic_name(affinity_cur), nic_to_find) == 0)
				break;
		}

		policy_count = 0;
		while (affinity_cur && get_nic_name(affinity_cur) &&
		       strcmp(get_nic_name(affinity_cur), nic_to_find) == 0) {
			policy_count++;
			affinity_cur = affinity_cur->next;
		}

		if (original_count != policy_count) {
			sprintf(err_buf, "NIC %s: original has %zu entries, policy has %zu consecutive entries",
				nic_to_find, original_count, policy_count);
			return EXIT_FAILURE;
		}
	}

	return 0;
}

static int compare_lists_same_order(struct fi_info *list1, struct fi_info *list2)
{
	struct fi_info *cur1;
	struct fi_info *cur2;
	const char *name1;
	const char *name2;

	cur1 = list1;
	cur2 = list2;
	while (cur1 && cur2) {
		name1 = get_nic_name(cur1);
		name2 = get_nic_name(cur2);

		if (name1 && name2 && strcmp(name1, name2) != 0) {
			sprintf(err_buf, "Order mismatch: %s != %s", name1, name2);
			return EXIT_FAILURE;
		}

		cur1 = cur1->next;
		cur2 = cur2->next;
	}

	if (cur1 || cur2) {
		sprintf(err_buf, "Different number of entries");
		return EXIT_FAILURE;
	}

	return 0;
}

static int check_verbs_no_interference(char *node, char *service, uint64_t flags,
				       struct fi_info *hints, struct fi_info **info)
{
	struct fi_info *original_info = NULL;
	struct fi_info *policy_info1 = NULL;
	struct fi_info *policy_info2 = NULL;
	int ret;

	ret = fi_getinfo(FT_FIVERSION, node, service, flags | OFI_CORE_PROV_ONLY, hints, &policy_info1);
	if (ret) {
		FT_UNIT_STRERR(err_buf, "fi_getinfo with affinity policy failed", ret);
		return ret;
	}

	ret = fi_getinfo(FT_FIVERSION, node, service, flags | OFI_CORE_PROV_ONLY, hints, &policy_info2);
	if (ret) {
		FT_UNIT_STRERR(err_buf, "fi_getinfo with affinity policy (second call) failed", ret);
		fi_freeinfo(policy_info1);
		return ret;
	}

	ret = compare_lists_same_order(policy_info1, policy_info2);
	if (ret)
		goto cleanup;

	unsetenv("FI_VERBS_NIC_AFFINITY_POLICY");
	unsetenv("FI_VERBS_AFFINITY_DEVICE");

	ret = fi_getinfo(FT_FIVERSION, node, service, flags | OFI_CORE_PROV_ONLY, hints, &original_info);
	if (ret) {
		FT_UNIT_STRERR(err_buf, "fi_getinfo with policy=none failed", ret);
		goto cleanup;
	}

	ret = check_count_and_grouping(original_info, policy_info1);

cleanup:
	fi_freeinfo(original_info);
	fi_freeinfo(policy_info1);
	fi_freeinfo(policy_info2);
	*info = NULL;

	cleanup_verbs_affinity_test();

	return ret;
}

static int check_verbs_identical_list(char *node, char *service, uint64_t flags,
				      struct fi_info *hints, struct fi_info **info)
{
	struct fi_info *original_info = NULL;
	struct fi_info *policy_info = NULL;
	int ret;

	ret = fi_getinfo(FT_FIVERSION, node, service, flags | OFI_CORE_PROV_ONLY, hints, &policy_info);
	if (ret) {
		FT_UNIT_STRERR(err_buf, "fi_getinfo with affinity policy failed", ret);
		cleanup_verbs_affinity_test();
		return ret;
	}

	unsetenv("FI_VERBS_NIC_AFFINITY_POLICY");
	unsetenv("FI_VERBS_AFFINITY_DEVICE");

	ret = fi_getinfo(FT_FIVERSION, node, service, flags | OFI_CORE_PROV_ONLY, hints, &original_info);
	if (ret) {
		FT_UNIT_STRERR(err_buf, "fi_getinfo with policy=none failed", ret);
		fi_freeinfo(policy_info);
		cleanup_verbs_affinity_test();
		return ret;
	}

	ret = compare_lists_same_order(original_info, policy_info);

	fi_freeinfo(original_info);
	fi_freeinfo(policy_info);
	*info = NULL;

	cleanup_verbs_affinity_test();

	return ret;
}

/*
 * nic_affinity test
 */
static int nic_affinity_unit_test(char *node, char *service, uint64_t flags,
			struct fi_info *base_hints, ft_nic_affinity_init init,
			ft_nic_affinity_test test, int ret_exp)
{
	struct fi_info *info = NULL, *test_hints = NULL;
	int ret;

	if (base_hints) {
		test_hints = fi_dupinfo(base_hints);
		if (!test_hints)
			return -FI_ENOMEM;
	}

	if (init) {
		ret = init(test_hints);
		if (ret)
			goto out;
	}

	if (test) {
		ret = test(node, service, flags, test_hints, &info);
	} else {
		ret = fi_getinfo(FT_FIVERSION, node, service, flags,
				 test_hints, &info);
	}
	if (ret) {
		if (ret == ret_exp) {
			ret = 0;
			goto out;
		}
		sprintf(err_buf, "fi_getinfo returned %d - %s",
			-ret, fi_strerror(-ret));
		goto out;
	}

out:
	fi_freeinfo(test_hints);
	fi_freeinfo(info);
	return ret;
}

#define nic_affinity_test(name, num, desc, node, service, flags, hints, 	\
			     init, test, ret_exp)			\
char *nic_affinity_ ## name ## num ## _desc = desc;				\
static int nic_affinity_ ## name ## num(void)				\
{									\
	int ret, testret = FAIL;					\
	ret = nic_affinity_unit_test(node, service, flags, hints, init,	\
					test, ret_exp);			\
	if (ret)							\
		goto fail;						\
	testret = PASS;							\
fail:									\
	return TEST_RET_VAL(ret, testret);				\
}

/*
 * Tests:
 */

/* Verbs GPU/NIC affinity tests */
nic_affinity_test(verbs, 1,
	     "Test verbs manual",
	     NULL, NULL, 0, hints,
	     init_verbs_affinity_manual,
	     check_verbs_no_interference, 0)
nic_affinity_test(verbs, 2,
	     "Test verbs manual without device",
	     NULL, NULL, 0, hints,
	     init_verbs_affinity_manual_no_device,
	     check_verbs_identical_list, 0)
nic_affinity_test(verbs, 3,
	     "Test verbs manual with invalid device",
	     NULL, NULL, 0, hints,
	     init_verbs_affinity_manual_invalid_device,
	     check_verbs_identical_list, 0)
nic_affinity_test(verbs, 4,
	     "Test verbs manual with missing config file",
	     NULL, NULL, 0, hints,
	     init_verbs_affinity_manual_missing_config,
	     check_verbs_identical_list, 0)
nic_affinity_test(verbs, 5,
	     "Test verbs manual with malformed config",
	     NULL, NULL, 0, hints,
	     init_verbs_affinity_manual_malformed_config,
	     check_verbs_identical_list, 0)
nic_affinity_test(verbs, 6,
	     "Test verbs auto",
	     NULL, NULL, 0, hints,
	     init_verbs_affinity_auto,
	     check_verbs_no_interference, 0)
nic_affinity_test(verbs, 7,
	     "Test verbs auto without device",
	     NULL, NULL, 0, hints,
	     init_verbs_affinity_auto_no_device,
	     check_verbs_identical_list, 0)
nic_affinity_test(verbs, 8,
	     "Test verbs auto with invalid device",
	     NULL, NULL, 0, hints,
	     init_verbs_affinity_auto_invalid_device,
	     check_verbs_identical_list, 0)
nic_affinity_test(verbs, 9,
	     "Test verbs invalid fallback to none",
	     NULL, NULL, 0, hints,
	     init_verbs_affinity_invalid,
	     check_verbs_identical_list, 0)

static void usage(char *name)
{
	ft_unit_usage(name, "Unit tests for GPU-NIC affinity");
	FT_PRINT_OPTS_USAGE("-e <ep_type>",
			    "Endpoint type: msg|rdm|dgram (default:rdm)");
	ft_addr_usage();
}

static int set_prov(char *prov_name)
{
	const char *util_name;
	const char *core_name;
	char *core_name_dup;
	size_t len;

	util_name = ft_util_name(prov_name, &len);
	core_name = ft_core_name(prov_name, &len);

	if (util_name && !core_name)
		return 0;

	core_name_dup = strndup(core_name, len);
	if (!core_name_dup)
		return -FI_ENOMEM;

	snprintf(new_prov_var, sizeof(new_prov_var) - 1, "FI_PROVIDER=%s",
		 core_name_dup);

	putenv(new_prov_var);
	free(core_name_dup);
	return 0;
}

int main(int argc, char **argv)
{
	int failed, cleanup_ret;
	int op;

	struct test_entry verbs_nic_affinity_tests[] = {
		TEST_ENTRY_NIC_AFFINITY(verbs1),
		TEST_ENTRY_NIC_AFFINITY(verbs2),
		TEST_ENTRY_NIC_AFFINITY(verbs3),
		TEST_ENTRY_NIC_AFFINITY(verbs4),
		TEST_ENTRY_NIC_AFFINITY(verbs5),
		TEST_ENTRY_NIC_AFFINITY(verbs6),
		TEST_ENTRY_NIC_AFFINITY(verbs7),
		TEST_ENTRY_NIC_AFFINITY(verbs8),
		TEST_ENTRY_NIC_AFFINITY(verbs9),
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
			ft_parseinfo(op, optarg, hints, &opts);
			break;
		case 'h':
		case '?':
			usage(argv[0]);
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
	hints->domain_attr->mr_mode = opts.mr_mode;

	if (hints->fabric_attr->prov_name) {
		if (set_prov(hints->fabric_attr->prov_name))
			return EXIT_FAILURE;
	}

	if (hints->fabric_attr->prov_name &&
	    (strstr(hints->fabric_attr->prov_name, "verbs") != NULL)) {
		setenv("FI_VERBS_AFFINITY_DEVICE", TEST_PCI_ADDR, 1);
		failed = run_tests(verbs_nic_affinity_tests, err_buf);
		unsetenv("FI_VERBS_AFFINITY_DEVICE");
	} else {
		printf("Skipping tests: verbs provider not specified\n");
		failed = 0;
	}

	if (failed > 0) {
		printf("\nSummary: %d tests failed\n", failed);
	} else {
		printf("\nSummary: all tests passed\n");
	}

	cleanup_ret = ft_free_res();
	return cleanup_ret ? ft_exit_code(cleanup_ret) :
		(failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
