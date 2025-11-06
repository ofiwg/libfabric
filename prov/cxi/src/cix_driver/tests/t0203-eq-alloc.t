#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
# Copyright 2021 Hewlett Packard Enterprise Development LP

. ./preamble.sh

test_description="EQ alloc tests"

. ./sharness.sh

test_expect_success "Inserting core driver" "
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko disable_default_svc=0 &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Inserting eq alloc test driver" "
	insmod ../../../drivers/net/ethernet/hpe/ss1/tests/test-eq-alloc.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Check for errors" "
	sleep 2 &&
	[ $(dmesg | grep -c 'test_1_realloc_eq_same_reserved_slots succeeded') -eq 1 ]
"

test_expect_success "Removing eq alloc test driver" "
	rmmod test-eq-alloc
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
