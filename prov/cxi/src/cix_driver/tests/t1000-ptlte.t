#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
# Copyright 2024 Hewlett Packard Enterprise Development LP

. ./preamble.sh

test_description="PtlTE tests"

. ./sharness.sh

test_expect_success "Inserting core driver" "
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko disable_default_svc=0 &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Inserting PtlTE test driver" "
	insmod ../../../drivers/net/ethernet/hpe/ss1/tests/test-ptlte.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Removing PtlTE test driver" "
	rmmod test-ptlte
"

test_expect_success "Removing core driver" "
	rmmod cxi-ss1 &&
	rmmod cxi-sbl &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Check for oops" "
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

dmesg > ../$(basename "$0").dmesg.txt

test_done
