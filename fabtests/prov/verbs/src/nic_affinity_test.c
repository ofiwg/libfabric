/*
 * Copyright (c) Intel Corporation.  All rights reserved.
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

#define OFI_CORE_PROV_ONLY      (1ULL << 59)
#define TEST_PCI_ADDR           "0000:00:00.0"
#define TEST_CONFIG_FILE        "/tmp/test_config.conf"
#define NIC_LIST_SIZE           1024

#define TEST_ENTRY_NIC_AFFINITY(name) \
	TEST_ENTRY(nic_affinity_##name, nic_affinity_##name##_desc)

typedef int (*ft_nic_affinity_init)(struct fi_info *);
typedef int (*ft_nic_affinity_test)(struct fi_info *);

static char err_buf[512];
static char *baseline_str = NULL;

static const char *get_nic_name(struct fi_info *info)
{
	if (info->nic && info->nic->device_attr && info->nic->device_attr->name)
		return info->nic->device_attr->name;
	return NULL;
}

static int get_arbitrary_nic_name(char *nic_name, size_t len)
{
	char *baseline_nics;
	char *comma_pos;
	int nic_len;

	baseline_nics = strchr(baseline_str, ':') + 1;
	comma_pos = strchr(baseline_nics, ',');
	nic_len = comma_pos ? (comma_pos - baseline_nics) : strlen(baseline_nics);

	if (nic_len == 0) {
		sprintf(err_buf, "No NICs in baseline");
		return -FI_ENODATA;
	}

	strncpy(nic_name, baseline_nics, nic_len);
	nic_name[nic_len] = '\0';

	return 0;
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

static void cleanup(void)
{
	unlink(TEST_CONFIG_FILE);
	unsetenv("FI_VERBS_NIC_AFFINITY_POLICY");
        unsetenv("FI_VERBS_NIC_AFFINITY_CONFIG");
	unsetenv("FI_VERBS_AFFINITY_DEVICE");
}

/*
 * Init functions
 */
static int init_none(struct fi_info *hints)
{
	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "none", 1);
        setenv("FI_VERBS_AFFINITY_DEVICE", TEST_PCI_ADDR, 1);
	return 0;
}

static int init_manual(struct fi_info *hints)
{
	int ret;

	ret = create_valid_test_config_file();
	if (ret) return ret;

	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "manual", 1);
	setenv("FI_VERBS_NIC_AFFINITY_CONFIG", TEST_CONFIG_FILE, 1);
        setenv("FI_VERBS_AFFINITY_DEVICE", TEST_PCI_ADDR, 1);
	return 0;
}

static int init_manual_no_device(struct fi_info *hints)
{
	int ret;

	ret = create_valid_test_config_file();
	if (ret) return ret;

	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "manual", 1);
	setenv("FI_VERBS_NIC_AFFINITY_CONFIG", TEST_CONFIG_FILE, 1);
	unsetenv("FI_VERBS_AFFINITY_DEVICE");
	return 0;
}

static int init_manual_invalid_device(struct fi_info *hints)
{
	int ret;

	ret = create_valid_test_config_file();
	if (ret) return ret;

	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "manual", 1);
	setenv("FI_VERBS_NIC_AFFINITY_CONFIG", TEST_CONFIG_FILE, 1);
	setenv("FI_VERBS_AFFINITY_DEVICE", "invalid:pci:format:bad", 1);
	return 0;
}

static int init_manual_missing_config(struct fi_info *hints)
{
	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "manual", 1);
	setenv("FI_VERBS_NIC_AFFINITY_CONFIG", "/nonexistent/path/to/config.conf", 1);
        setenv("FI_VERBS_AFFINITY_DEVICE", TEST_PCI_ADDR, 1);
	return 0;
}

static int init_manual_malformed_config(struct fi_info *hints)
{
	int ret;

	ret = create_invalid_test_config_file();
	if (ret) return ret;

	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "manual", 1);
	setenv("FI_VERBS_NIC_AFFINITY_CONFIG", TEST_CONFIG_FILE, 1);
        setenv("FI_VERBS_AFFINITY_DEVICE", TEST_PCI_ADDR, 1);
	return 0;
}

static int init_auto(struct fi_info *hints)
{
	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "auto", 1);
        setenv("FI_VERBS_AFFINITY_DEVICE", TEST_PCI_ADDR, 1);
	return 0;
}

static int init_auto_no_device(struct fi_info *hints)
{
	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "auto", 1);
	unsetenv("FI_VERBS_AFFINITY_DEVICE");
	return 0;
}

static int init_auto_invalid_device(struct fi_info *hints)
{
	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "auto", 1);
	setenv("FI_VERBS_AFFINITY_DEVICE", "invalid:pci:format:bad", 1);
	return 0;
}

static int init_invalid(struct fi_info *hints)
{
	setenv("FI_VERBS_NIC_AFFINITY_POLICY", "invalid_garbage_policy", 1);
        setenv("FI_VERBS_AFFINITY_DEVICE", TEST_PCI_ADDR, 1);
	return 0;
}

/*
 * Check functions
 */
static int print_baseline(struct fi_info *info)
{
	char nics[NIC_LIST_SIZE] = "";
	struct fi_info *cur = NULL;
	const char *name = NULL;
	const char *prev_name = NULL;
	size_t offset = 0;
	int total_count = 0;

	for (cur = info; cur; cur = cur->next) {
		total_count++;

		name = get_nic_name(cur);
		if (!name || (prev_name && strcmp(name, prev_name) == 0))
			continue;

		if (offset > 0)
			offset += snprintf(nics + offset, sizeof(nics) - offset, ",");
		offset += snprintf(nics + offset, sizeof(nics) - offset, "%s", name);
		prev_name = name;
	}

	printf("%d:%s\n", total_count, nics);
	return 0;
}

static int check_identical_list(struct fi_info *info) {
    int baseline_total;
    char *baseline_nics;
    char *baseline_ptr;
    struct fi_info *cur;
    const char *name, *prev_name;
    int current_total;
    
    baseline_total = atoi(baseline_str);
    baseline_nics = strchr(baseline_str, ':') + 1;
    baseline_ptr = baseline_nics;
    prev_name = NULL;
    current_total = 0;
    
    for (cur = info; cur; cur = cur->next) {
        current_total++;
        
        name = get_nic_name(cur);
        if (!name || (prev_name && strcmp(name, prev_name) == 0))
            continue;
        
        if (strncmp(baseline_ptr, name, strlen(name)) != 0) {
            sprintf(err_buf, "Unexpected NIC: %s", name);
            return -FI_EOTHER;
        }
        
        baseline_ptr += strlen(name) + 1;
        prev_name = name;
    }
    
    if (baseline_total != current_total) {
        sprintf(err_buf, "Total mismatch: baseline=%d, current=%d",
                baseline_total, current_total);
        return -FI_EOTHER;
    }
    
    return 0;
}

static int check_no_interference(struct fi_info *info) {
    int baseline_total;
    char *baseline_nics;
    char baseline_copy[NIC_LIST_SIZE];
    struct fi_info *cur;
    const char *name, *prev_name;
    int current_total;
    char *nic_pos;

    baseline_total = atoi(baseline_str);
    baseline_nics = strchr(baseline_str, ':') + 1;
    
    strncpy(baseline_copy, baseline_nics, sizeof(baseline_copy) - 1);
    baseline_copy[sizeof(baseline_copy) - 1] = '\0';
    
    prev_name = NULL;
    current_total = 0;
    
    for (cur = info; cur; cur = cur->next) {
        current_total++;
        
        name = get_nic_name(cur);
        if (!name || (prev_name && strcmp(name, prev_name) == 0))
            continue;

        nic_pos = strstr(baseline_copy, name);
        if (!nic_pos) {
            sprintf(err_buf, "Grouping violation in NIC %s ", name);
            return -FI_EOTHER;
        }
        
        memset(nic_pos, ' ', strlen(name));
        prev_name = name;
    }
    
    /* Verify total count matches */
    if (baseline_total != current_total) {
        sprintf(err_buf, "Total mismatch: baseline=%d, current=%d",
                baseline_total, current_total);
        return -FI_EOTHER;
    }
    
    return 0;
}

static int validate_consistency()
{
        struct fi_info *policy_info1;
	struct fi_info *policy_info2;
	const char *name1;
	const char *name2;
        int ret;

        ret = fi_getinfo(FT_FIVERSION, NULL, NULL, OFI_CORE_PROV_ONLY,
			 hints, &policy_info1);
	if (ret) {
		FT_UNIT_STRERR(err_buf, "fi_getinfo with affinity policy failed", ret);
		goto out;
	}

	ret = fi_getinfo(FT_FIVERSION, NULL, NULL, OFI_CORE_PROV_ONLY | FI_RESCAN,
			 hints, &policy_info2);
	if (ret) {
		FT_UNIT_STRERR(err_buf, "fi_getinfo second call failed", ret);
		goto out;
	}

	while (policy_info1 && policy_info2) {
		name1 = get_nic_name(policy_info1);
		name2 = get_nic_name(policy_info2);

		if (name1 && name2 && strcmp(name1, name2) != 0) {
			sprintf(err_buf, "Order mismatch: %s != %s", name1, name2);
			ret = -FI_EOTHER;
                        goto out;
		}

		policy_info1 = policy_info1->next;
		policy_info2 = policy_info2->next;
	}

	if (policy_info1 || policy_info2) {
		sprintf(err_buf, "Different number of entries");
		ret = -FI_EOTHER;
                goto out;
	}

out:
        if (policy_info1) fi_freeinfo(policy_info1);
	if (policy_info2) fi_freeinfo(policy_info2);
	return ret;
}

/*
 * nic affinity test
 */
static int nic_affinity_unit_test(ft_nic_affinity_init init,
				   ft_nic_affinity_test test)
{
	struct fi_info *info = NULL, *test_hints = NULL;
	int ret;

	test_hints = fi_dupinfo(hints);
	if (!test_hints)
		return -FI_ENOMEM;

	if (init) {
		ret = init(test_hints);
		if (ret) goto out;
	}

	ret = validate_consistency();
	if (ret) goto out;

	ret = fi_getinfo(FT_FIVERSION, NULL, NULL, OFI_CORE_PROV_ONLY,
			 test_hints, &info);
	if (ret) {
		FT_UNIT_STRERR(err_buf, "fi_getinfo failed", ret);
		goto out;
	}

	ret = test(info);

out:
	cleanup();
	fi_freeinfo(test_hints);
	if (info) fi_freeinfo(info);
	return ret;
}

#define nic_affinity_test(name, desc, init, test)			\
char *nic_affinity_ ## name ## _desc = desc;				\
static int nic_affinity_ ## name(void)					\
{									\
	int ret, testret = FAIL;					\
	ret = nic_affinity_unit_test(init, test);			\
	if (ret)							\
		goto fail;						\
	testret = PASS;							\
fail:									\
	return TEST_RET_VAL(ret, testret);				\
}

/*
 * Tests:
 */
nic_affinity_test(baseline, "Test manual policy for sanity",
		  NULL,
		  print_baseline)
nic_affinity_test(none_sanity, "Test manual policy for sanity",
		  init_none,
		  check_identical_list)
nic_affinity_test(manual_sanity, "Test manual policy for sanity",
		  init_manual,
		  check_no_interference)
nic_affinity_test(manual_no_device, "Test manual policy without device",
		  init_manual_no_device,
		  check_identical_list)
nic_affinity_test(manual_invalid_device, "Test manual policy with invalid device",
		  init_manual_invalid_device,
		  check_identical_list)
nic_affinity_test(manual_missing_config, "Test manual policy with missing config file",
		  init_manual_missing_config,
		  check_identical_list)
nic_affinity_test(manual_malformed_config, "Test manual policy with malformed config",
		  init_manual_malformed_config,
		  check_identical_list)
nic_affinity_test(auto_sanity, "Test auto policy for sanity",
		  init_auto,
		  check_no_interference)
nic_affinity_test(auto_no_device, "Test auto policy without device",
		  init_auto_no_device,
		  check_identical_list)
nic_affinity_test(auto_invalid_device, "Test auto policy with invalid device",
		  init_auto_invalid_device,
		  check_identical_list)
nic_affinity_test(invalid_policy, "Test invalid policy fallback to none",
		  init_invalid,
		  check_identical_list)

static void usage(char *name, struct test_entry *tests)
{
	struct test_entry *t;

	fprintf(stderr, "Usage:\n");
	fprintf(stderr, "  %s --test <name> [--baseline <string>]\n\n", name);
	fprintf(stderr, "Unit tests for verbs GPU-NIC affinity feature\n\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -h, --help              Display this help output\n");
	fprintf(stderr, "  --test <name>           Run single test by name\n");
	fprintf(stderr, "  --baseline <string>     Baseline NIC list\n");
	fprintf(stderr, "Available tests:\n");
	for (t = tests; t->test; t++) {
		const char *name = t->name + strlen("nic_affinity_");
		fprintf(stderr, "  %-23s %s\n", name, t->desc);
	}
}

static struct test_entry *find_test_by_name(struct test_entry *tests,
					     const char *name)
{
        struct test_entry *cur;

	for (cur = tests; cur->test; cur++) {
		if (strstr(cur->name, name))
			return cur;
	}

        fprintf(stderr, "Error: Unknown test '%s'\n\n", name);
	return NULL;
}

int main(int argc, char **argv)
{
	int failed, cleanup_ret;
	int op;
	char *test_name = NULL;
	struct test_entry *single_test;
        int ret;

	struct test_entry nic_affinity_tests[] = {
                TEST_ENTRY_NIC_AFFINITY(none_sanity),
		TEST_ENTRY_NIC_AFFINITY(manual_sanity),
		TEST_ENTRY_NIC_AFFINITY(manual_no_device),
		TEST_ENTRY_NIC_AFFINITY(manual_invalid_device),
		TEST_ENTRY_NIC_AFFINITY(manual_missing_config),
		TEST_ENTRY_NIC_AFFINITY(manual_malformed_config),
		TEST_ENTRY_NIC_AFFINITY(auto_sanity),
		TEST_ENTRY_NIC_AFFINITY(auto_no_device),
		TEST_ENTRY_NIC_AFFINITY(auto_invalid_device),
		TEST_ENTRY_NIC_AFFINITY(invalid_policy),
		{ NULL, "" }
	};

	struct option long_options[] = {
		{"baseline", required_argument, 0, 'b'},
		{"test", required_argument, 0, 't'},
		{"help", no_argument, 0, 'h'},
		{0, 0, 0, 0}
	};

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt_long(argc, argv, "h", long_options, NULL)) != -1) {
		switch (op) {
		case 'b':
			baseline_str = optarg;
			break;
		case 't':
			test_name = optarg;
			break;
		case '?':
		case 'h':
		default:
			usage(argv[0], nic_affinity_tests);
			fi_freeinfo(hints);
			return EXIT_FAILURE;
		}
	}

	if (!test_name) {
		fprintf(stderr, "Error: --test is required\n\n");
		usage(argv[0], nic_affinity_tests);
		fi_freeinfo(hints);
		return EXIT_FAILURE;
	}

        hints->mode = ~0;

	hints->fabric_attr->prov_name = strdup("verbs");
	if (!hints->fabric_attr->prov_name) {
		fi_freeinfo(hints);
		return EXIT_FAILURE;
	}

	if (strcmp(test_name, "baseline") == 0) {
		ret = nic_affinity_baseline();
		fi_freeinfo(hints);
		return ret == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
	}

        if (!baseline_str) {
                fprintf(stderr, "Error: --baseline is required\n\n");
                usage(argv[0], nic_affinity_tests);
                fi_freeinfo(hints);
                return EXIT_FAILURE;
        }

	if (!strchr(baseline_str, ':')) {
		fprintf(stderr, "Error: Invalid baseline format.\n\n");
		usage(argv[0], nic_affinity_tests);
		fi_freeinfo(hints);
		return EXIT_FAILURE;
	}

	single_test = find_test_by_name(nic_affinity_tests, test_name);
	if (!single_test) {
		usage(argv[0], nic_affinity_tests);
		fi_freeinfo(hints);
		return EXIT_FAILURE;
	}

	struct test_entry single_test_array[] = {
		{ single_test->test, single_test->name, single_test->desc },
		{ NULL, "" }
	};
	failed = run_tests(single_test_array, err_buf);

	cleanup_ret = ft_free_res();
	return cleanup_ret ? ft_exit_code(cleanup_ret) :
		(failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
