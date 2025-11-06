#!/bin/bash

# Automating the configs test for rgroup,rx and tx profile

. ./preamble.sh

test_description="User tests for configfs interface"

. ./sharness.sh

CONFIG_UTILITY="../../../scripts/cxi_mgmt"
CONFIG_TEST_DIR="../../../tests/tmptests"
# Test name length = 64, accepted length limit is 63
LONG_NAME="r11111111111111111111111111111111111111111111111111111111111111e"

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

test_expect_success "Add invalid vni range" "
	$CONFIG_UTILITY create --rx-profile rx4321 &&
	$CONFIG_UTILITY set-vni --rx-profile rx4321 1999 1998 &> ../$(basename "$0").add.inv.vni.output
	cat $CONFIG_TEST_DIR/$(basename "$0").add.inv.vni.output | grep 'write error: Invalid argument' &&
	[ $? -eq 0 ]
"

test_expect_success "Teardown test for rgroup" "
	$CONFIG_UTILITY create --rsrc-grp r1 &&
	$CONFIG_UTILITY add-ac-entry --rsrc-grp r1 ac 0 1 &&
	$CONFIG_UTILITY add-resource --rsrc-grp r1 ct 100 100 &&
	$CONFIG_UTILITY add-resource --rsrc-grp r1 tle 100 100 &&
	$CONFIG_UTILITY teardown --rsrc-grp r1
"

test_expect_success "Teardown test for rx" "
	$CONFIG_UTILITY create --rx-profile rx1 &&
	$CONFIG_UTILITY set-vni --rx-profile rx1 128 127 &&
	$CONFIG_UTILITY add-ac-entry --rx-profile rx1 ac 0 1 &&
	$CONFIG_UTILITY teardown --rx-profile rx1 &&
	$CONFIG_UTILITY teardown --rx-profile rx4321
"

test_expect_success "Teardown test for tx" "
	$CONFIG_UTILITY create --tx-profile tx1 &&
	$CONFIG_UTILITY set-vni --tx-profile tx1 128 127 1 &&
	$CONFIG_UTILITY add-ac-entry --tx-profile tx1 ac 0 1 &&
	$CONFIG_UTILITY teardown --tx-profile tx1
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

test_expect_success "Inserting CXI User test driver" "
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-user.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "No Oops" "
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Invalid param test program for configfs" "
	$CONFIG_UTILITY create --rx-profile rx1 &&
	$CONFIG_UTILITY set-vni --rx-profile rx1 128 127 &&
	$CONFIG_UTILITY add-ac-entry --rx-profile rx1 ac 0 1 &&
	$CONFIG_UTILITY enable --rx-profile rx1 &&
	$CONFIG_UTILITY create --tx-profile tx1 &&
	$CONFIG_UTILITY set-vni --tx-profile tx1 128 127 true &&
	$CONFIG_UTILITY add-ac-entry --tx-profile tx1 ac 0 1 &&
	$CONFIG_UTILITY  set-tc --tx-profile tx1 dedicated true &&
	$CONFIG_UTILITY  set-tc --tx-profile tx1 best_effort true &&
	$CONFIG_UTILITY  set-tc --tx-profile tx1 low_latency true &&
	$CONFIG_UTILITY  set-tc --tx-profile tx1 bulk_data true &&
	$CONFIG_UTILITY enable --tx-profile tx1 &&
	test_expect_code 0 ../../../ucxi/test_ucxi_atu -s 1 -v 123 -d cxi0 &> ../$(basename "$0").inv.vni.test.output &&
	test_expect_code 1 ../../../ucxi/test_ucxi_atu -u 791 -s 1 -v 123 -d cxi0 &> ../$(basename "$0").inv.vni.test.output &&
	test_expect_code 1 ../../../ucxi/test_ucxi_atu -s 2 -v 130 -d cxi0 &> ../$(basename "$0").inv.rgroup.test.output &&
	test_expect_code 1 ../../../ucxi/test_ucxi_atu -s 1 -v 134 -d cxi6 &> ../$(basename "$0").inv.device.test.output
"

test_expect_success "Invalid Access Control Type test program for configfs" "
	$CONFIG_UTILITY create --rsrc-grp r1 &&
	$CONFIG_UTILITY add-ac-entry --rsrc-grp r1 ac 0 12 &> ../$(basename "$0").rgp.test.output
	cat $CONFIG_TEST_DIR/$(basename "$0").rgp.test.output | grep 'Invalid argument' &&
	[ $? -eq 0 ]
	$CONFIG_UTILITY create --rx-profile rxp1 &&
	$CONFIG_UTILITY set-vni --rx-profile rxp1 128 127 &&
	$CONFIG_UTILITY add-ac-entry --rx-profile rxp1 ac 0 12 &> ../$(basename "$0").rx.test.output
	cat $CONFIG_TEST_DIR/$(basename "$0").rx.test.output | grep 'Invalid argument' &&
	[ $? -eq 0 ]
	$CONFIG_UTILITY create --tx-profile txp1 &&
	$CONFIG_UTILITY set-vni --tx-profile txp1 128 127 1 &&
	$CONFIG_UTILITY add-ac-entry --tx-profile txp1 ac 0 12 &> ../$(basename "$0").tx.test.output
	cat $CONFIG_TEST_DIR/$(basename "$0").tx.test.output | grep 'Invalid argument' &&
	[ $? -eq 0 ]
"

test_expect_success "Test to add AC Entry when rgroup is enabled" "
        $CONFIG_UTILITY create --rsrc-grp r9999 &&
        $CONFIG_UTILITY enable --rsrc-grp r9999 &&
	$CONFIG_UTILITY add-ac-entry --rsrc-grp r9999 ac1 0 1 &> ../$(basename "$0").rgp9999.test.output
	cat $CONFIG_TEST_DIR/$(basename "$0").rgp9999.test.output | grep 'Device or resource busy' &&
	[ $? -eq 0 ]
"

test_expect_success "Test to add resource when rgroup is enabled" "
	$CONFIG_UTILITY create --rsrc-grp r111 &&
	$CONFIG_UTILITY add-ac-entry --rsrc-grp r111 ac 0 1 &&
        $CONFIG_UTILITY enable --rsrc-grp r111 &&
	$CONFIG_UTILITY add-resource --rsrc-grp r111 ct 100 100 &> ../$(basename "$0").test.111.output
	cat $CONFIG_TEST_DIR/$(basename "$0").test.111.output | grep 'Device or resource busy' &&
	[ $? -eq 0 ]
"

test_expect_success "vni overlap test program for configfs" "
	$CONFIG_UTILITY create --rx-profile rx2 &&
	$CONFIG_UTILITY set-vni --rx-profile rx2 256 255 &&
	$CONFIG_UTILITY create --rx-profile rx3 &&
	$CONFIG_UTILITY set-vni --rx-profile rx3 270 0 &> ../$(basename "$0").overlap.test.output
	cat $CONFIG_TEST_DIR/$(basename "$0").overlap.test.output | grep 'Invalid argument' &&
	[ $? -eq 0 ]
"

test_expect_success "Enable RX profile without vni range" "
	$CONFIG_UTILITY create --rx-profile rx99 &&
	$CONFIG_UTILITY enable --rx-profile rx99 &> ../$(basename "$0").enable.no.99.vni.output
	cat $CONFIG_TEST_DIR/$(basename "$0").enable.no.99.vni.output | grep 'Invalid argument' &&
	[ $? -eq 0 ]
"

test_expect_success "Enable TX profile without vni range" "
	$CONFIG_UTILITY create --tx-profile tx1234 &&
	$CONFIG_UTILITY enable --tx-profile tx1234 &> ../$(basename "$0").enable.no.1234.vni.output
	cat $CONFIG_TEST_DIR/$(basename "$0").enable.no.1234.vni.output | grep 'Invalid argument' &&
	[ $? -eq 0 ]
"

test_expect_success "Test to add an already added resource type" "
	$CONFIG_UTILITY create --rsrc-grp r45 &&
	$CONFIG_UTILITY add-ac-entry --rsrc-grp r45 ac 0 1 &&
	$CONFIG_UTILITY add-resource --rsrc-grp r45 ct 100 100 &&
	$CONFIG_UTILITY add-resource --rsrc-grp r45 ct 100 100 &> ../$(basename "$0").test.45.output
	cat $CONFIG_TEST_DIR/$(basename "$0").test.45.output | grep 'Unable to create directory ct inside resources directory' &&
	[ $? -eq 0 ]
"

test_expect_success "Test to add resource group name length > 63" "
	$CONFIG_UTILITY create --rsrc-grp $LONG_NAME &> ../$(basename "$0").rgp.long.len.test.output
	cat $CONFIG_TEST_DIR/$(basename "$0").rgp.long.len.test.output | grep 'cannot create directory' &&
	[ $? -eq 0 ]
"

test_expect_success "Test to add rx profile name length > 63" "
	$CONFIG_UTILITY create --rx-profile $LONG_NAME &> ../$(basename "$0").rx.long.len.test.output
	cat $CONFIG_TEST_DIR/$(basename "$0").rx.long.len.test.output | grep 'cannot create directory' &&
	[ $? -eq 0 ]
"

test_expect_success "Test to add tx profile name length > 63" "
	$CONFIG_UTILITY create --tx-profile $LONG_NAME &> ../$(basename "$0").tx.long.len.test.output
	cat $CONFIG_TEST_DIR/$(basename "$0").tx.long.len.test.output | grep 'cannot create directory' &&
	[ $? -eq 0 ]
"

test_expect_success "Test to add ac entry name length > 63 to rgroup" "
	$CONFIG_UTILITY create --rsrc-grp r363 &&
	$CONFIG_UTILITY add-ac-entry --rsrc-grp r363 $LONG_NAME 0 1 &> ../$(basename "$0").rgp.long.len.ac.test.output
	cat $CONFIG_TEST_DIR/$(basename "$0").rgp.long.len.ac.test.output | grep 'cannot create directory' &&
	[ $? -eq 0 ]
"

test_expect_success "Test to add ac entry name length > 63 to rx profile" "
	$CONFIG_UTILITY create --rx-profile rx363 &&
	$CONFIG_UTILITY add-ac-entry --rx-profile rx363 $LONG_NAME 0 1 &> ../$(basename "$0").rx.long.len.ac.test.output
	cat $CONFIG_TEST_DIR/$(basename "$0").rx.long.len.ac.test.output | grep 'cannot create directory' &&
	[ $? -eq 0 ]
"

test_expect_success "Test to add ac entry name length > 63 to tx profile" "
	$CONFIG_UTILITY create --tx-profile tx363 &&
	$CONFIG_UTILITY add-ac-entry --tx-profile tx363 $LONG_NAME 0 1 &> ../$(basename "$0").tx.long.len.ac.test.output
	cat $CONFIG_TEST_DIR/$(basename "$0").tx.long.len.ac.test.output | grep 'cannot create directory' &&
	[ $? -eq 0 ]
"

test_expect_success "Remove resource test for resource group" "
	$CONFIG_UTILITY create --rsrc-grp r787 &&
	$CONFIG_UTILITY add-resource --rsrc-grp r787 ct 100 100 &&
	$CONFIG_UTILITY remove-resource --rsrc-grp r787 ct &&
	$CONFIG_UTILITY add-resource --rsrc-grp r787 ct 100 100 &> ../$(basename "$0").test.787.output &&
	[ ! -s $CONFIG_TEST_DIR/$(basename "$0").test.787.output ]
"

test_expect_success "Remove ac entry test for resource group" "
	$CONFIG_UTILITY create --rsrc-grp r687 &&
	$CONFIG_UTILITY add-ac-entry --rsrc-grp r687 ac1 0 1 &&
	$CONFIG_UTILITY remove-ac-entry --rsrc-grp r687 ac1 &&
	$CONFIG_UTILITY add-ac-entry --rsrc-grp r687 ac1 0 1 &> ../$(basename "$0").test.687.output &&
	[ ! -s $CONFIG_TEST_DIR/$(basename "$0").test.687.output ]
"

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
