#!/bin/bash

# Automating the configs test for rgroup, rx and tx profile

. ./preamble.sh

test_description="User tests for configfs interface"

. ./sharness.sh

CONFIG_UTILITY="../../../scripts/cxi_mgmt"
CONFIG_TEST_DIR="../../../tests/tmptests"

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

test_expect_success "Run test program for configfs invalid UID" "
	$CONFIG_UTILITY create --rx-profile rx1 &&
	$CONFIG_UTILITY set-vni --rx-profile rx1 128 127 &&
	$CONFIG_UTILITY add-ac-entry --rx-profile rx1 ac 790 1 &&
	$CONFIG_UTILITY enable --rx-profile rx1 &&
	$CONFIG_UTILITY create --tx-profile tx1 &&
	$CONFIG_UTILITY set-vni --tx-profile tx1 128 127 true &&
	$CONFIG_UTILITY add-ac-entry --tx-profile tx1 ac 790 1 &&
	$CONFIG_UTILITY  set-tc --tx-profile tx1 dedicated true &&
	$CONFIG_UTILITY  set-tc --tx-profile tx1 best_effort true &&
	$CONFIG_UTILITY  set-tc --tx-profile tx1 low_latency true &&
	$CONFIG_UTILITY  set-tc --tx-profile tx1 bulk_data true &&
	$CONFIG_UTILITY enable --tx-profile tx1 &&
	test_expect_code 1 ../../../ucxi/test_ucxi_atu -u 791 -s 1 -v 129 -d cxi0 &> ../$(basename "$0").test.output
"

echo
echo "Display Contents of Test File - Invalid UID Test"
cat $CONFIG_TEST_DIR/$(basename "$0").test.output
echo

test_expect_success "Run test program for configfs invalid GID" "
	$CONFIG_UTILITY create --rx-profile rx2 &&
	$CONFIG_UTILITY set-vni --rx-profile rx2 258 0 &&
	$CONFIG_UTILITY add-ac-entry --rx-profile rx2 ac 790 2 &&
	$CONFIG_UTILITY enable --rx-profile rx2 &&
	$CONFIG_UTILITY create --tx-profile tx2 &&
	$CONFIG_UTILITY set-vni --tx-profile tx2 258 0 true &&
	$CONFIG_UTILITY add-ac-entry --tx-profile tx2 ac 790 2 &&
	$CONFIG_UTILITY  set-tc --tx-profile tx2 dedicated true &&
	$CONFIG_UTILITY  set-tc --tx-profile tx2 best_effort true &&
	$CONFIG_UTILITY  set-tc --tx-profile tx2 low_latency true &&
	$CONFIG_UTILITY  set-tc --tx-profile tx2 bulk_data true &&
	$CONFIG_UTILITY enable --tx-profile tx2 &&
	test_expect_code 1 ../../../ucxi/test_ucxi_atu -g 791 -s 1 -v 258 -d cxi0 &> ../$(basename "$0").test.output
"

echo
echo "Display Contents of Test File - Invalid GID Test"
cat $CONFIG_TEST_DIR/$(basename "$0").test.output
echo

test_expect_success "Run test configfs cleanup" "
	$CONFIG_UTILITY cleanup &> ../$(basename "$0").cleanup.output
"

test_expect_success "Remove core driver" "
	rmmod cxi-user &&
	rmmod cxi-ss1 &&
	rmmod cxi-sbl &&
	rmmod cxi-sl &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Inserting core driver" "
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko disable_default_svc=0 &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "No Oops" "
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

dmesg > ../$(basename "$0").dmesg.txt

test_done
