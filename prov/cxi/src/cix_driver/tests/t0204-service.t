#!/bin/bash

# Test service API

. ./preamble.sh

test_description="Basic tests for service API"

. ./sharness.sh

test_expect_success "Inserting core driver" "
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Inserting service API test driver" "
	dmesg --clear &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/tests/test-service.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Check for success" "
	[ $(dmesg | grep -c 'Tests passed') -eq 1 ]
"

if [ $(dmesg | grep -c 'Tests passed') -eq 0 ]; then
	dmesg
fi

dmesg > ../$(basename "$0").dmesg.txt

test_expect_success "Removing service API test driver" "
	rmmod test-service &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_done
