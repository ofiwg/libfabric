#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
# Copyright 2020,2024 Hewlett Packard Enterprise Development LP

# Test DMAC

. ./preamble.sh

test_description="Basic test for DMAC"

. ./sharness.sh

test_expect_success "Inserting core driver" "
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Inserting DMAC test driver" "
	dmesg --clear &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/tests/test-dmac.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Check for test errors" "
	sleep 1 &&
	[ $(dmesg | grep -c 'TESTSUITE_DMAC: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_DMAC_0: START') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTCASE_DMAC_0: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TESTSUITE_DMAC: FINISH') -eq 1 ] &&
	[ $(dmesg | grep -c 'TEST-ERROR') -eq 0 ] &&
	rmmod test-dmac &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

dmesg > ../$(basename "$0").dmesg.txt

test_done
