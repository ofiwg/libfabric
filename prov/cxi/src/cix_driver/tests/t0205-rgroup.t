#!/bin/bash

# Test service API

. ./preamble.sh

test_description="Basic tests for service API"

. ./sharness.sh

test_expect_success "Inserting core driver" "
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko disable_default_svc=0 &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Inserting rgroup API test driver" "
	dmesg --clear &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/tests/test-rgroup.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Removing rgroup API test driver" "
	rmmod test-rgroup
"

test_expect_success "Inserting profile API test driver" "
	dmesg --clear &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/tests/test-profiles.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Removing profile API test driver" "
	rmmod test-profiles
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
