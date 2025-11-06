#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
# Copyright 2020,2024 Hewlett Packard Enterprise Development LP

# Test exclusive cp and cp modify functionality

. ./preamble.sh

test_description="Tests to validate exclusive cp and cp modify functionality"

. ./sharness.sh

test_expect_success "Inserting core driver" "
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Inserting exclusive-cp API test driver" "
        dmesg --clear &&
        insmod ../../../drivers/net/ethernet/hpe/ss1/tests/test-exclusive-cp.ko &&
        [ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Removing exclusive-cp API test driver" "
        rmmod test-exclusive-cp
"

test_expect_success "Remove core driver" "
	dmesg --clear &&
	rmmod cxi-ss1 &&
	rmmod cxi-sbl &&
	rmmod cxi-sl &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

dmesg > ../$(basename "$0").dmesg.txt

test_done
