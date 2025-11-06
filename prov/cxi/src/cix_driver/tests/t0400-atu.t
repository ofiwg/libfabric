#!/bin/bash

# Test ATU

. ./preamble.sh

test_description="Basic user tests for ATU"

. ./sharness.sh

test_expect_success "Inserting core driver" "
	dmesg --clear &&
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko disable_default_svc=0 &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Inserting CXI User test driver" "
	dmesg --clear &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-user.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Run test program" "
	../../../ucxi/test_ucxi_atu &> ../$(basename "$0").output1 &&
	../../../ucxi/test_ucxi_atu -r -n 27 &> ../$(basename "$0").output2 &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Test error injection" "
	echo 200 > /sys/kernel/debug/cxi/cxi0/atu/error_inject &&
	../../../ucxi/test_ucxi_atu -Drn 15 &> ../$(basename "$0").output3 &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Remove CXI User test driver" "
	dmesg --clear &&
	rmmod cxi-user &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Remove core driver" "
	dmesg --clear &&
	rmmod cxi-ss1 &&
	rmmod cxi-sbl &&
	rmmod cxi-sl &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Inserting core driver" "
	dmesg --clear &&
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko disable_default_svc=0 &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Run ATU test driver" "
	insmod ../../../drivers/net/ethernet/hpe/ss1/tests/test-atu.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

if [ $(dmesg | grep -c 'Test passed') -eq 0 ]; then
	dmesg
fi

# Testing for 'Test passed' inside of previous test_expect_success fails
# but checking again passes
test_expect_success "Check dmesg for 'Test passed'" "
	[ $(dmesg | grep -c 'Test passed') -eq 1 ]
"

test_expect_success "Remove core driver" "
	rmmod test_atu &&
	rmmod cxi-ss1 &&
	rmmod cxi-sbl &&
	rmmod cxi-sl &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

dmesg > ../$(basename "$0").dmesg.txt

test_done
