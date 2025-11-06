#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
# Copyright 2021,2024 Hewlett Packard Enterprise Development LP

# Test CXI TELEM API

. ./preamble.sh

test_description="Basic test for CXI TELEM API"

. ./sharness.sh

test_expect_success "Inserting core driver" "
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Inserting TELEM test driver" "
	dmesg --clear &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/tests/test-telem-api.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Check for test errors" "
	sleep 1 &&
	[ $(dmesg | grep -c 'TESTSUITE_CXI_TELEM_API: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_TELEM_API_00: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_CXI_TELEM_API_00: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -E -c -e 'TESTCASE_CXI_TELEM_API_[^:]*: START') -eq 1 ] &&
	[ $(dmesg | grep -E -c -e 'TESTCASE_CXI_TELEM_API_[^:]*: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTSUITE_CXI_TELEM_API: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TEST-ERROR') -eq 0 ] &&
	rmmod test-telem-api &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

dmesg > ../$(basename "$0").dmesg.txt

test_done
