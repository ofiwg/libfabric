#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
# Copyright 2021 Hewlett Packard Enterprise Development LP

# Test DMAC

. ./preamble.sh

test_description="Basic test for CXI DMAC API"

. ./sharness.sh

test_expect_success "Inserting core driver" "
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Inserting DMAC test driver" "
	dmesg --clear &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/tests/test-dmac-api.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Check for test errors" "
	sleep 1 &&
	[ $(dmesg | grep -c 'TESTSUITE_CXI_DMAC_API: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_00: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_00: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_01: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_01: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_02: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_02: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_03: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_03: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_04: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_04: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_05: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_05: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_06: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_06: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_07: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_07: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_08: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_08: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_09: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_09: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_10: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_10: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_11: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_11: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_12: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_DMAC_API_12: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTSUITE_CXI_DMAC_API: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TEST-ERROR') -eq 0 ] &&
	rmmod test-dmac-api &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

dmesg > ../$(basename "$0").dmesg.txt

test_done
