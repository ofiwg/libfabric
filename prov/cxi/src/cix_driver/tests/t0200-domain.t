#!/bin/bash

# Test domains

. ./preamble.sh

test_description="Basic tests for domains"

. ./sharness.sh

test_expect_success "Inserting core driver" "
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko disable_default_svc=0 &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Inserting domain test driver" "
	dmesg --clear &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/tests/test-domain.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Check for test errors" "
	sleep 1 &&
	[ $(dmesg | grep -c 'TEST-START: DOMAIN') -eq 1 ] &&
	[ $(dmesg | grep -c 'TEST-END: DOMAIN') -eq 1 ] &&
	[ $(dmesg | grep -c 'TEST-ERROR') -eq 0 ] &&
	rmmod test-domain
"

dmesg > ../$(basename "$0").dmesg.txt

test_done
