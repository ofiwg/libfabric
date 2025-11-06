#!/bin/bash

# Automating the configs test for rgroup

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

test_expect_success "Run test program for configfs" "
	../../../scripts/cxi_mgmt create --rx-profile rx1 &&
	../../../scripts/cxi_mgmt set-vni --rx-profile rx1 128 127 &&
	../../../scripts/cxi_mgmt add-ac-entry --rx-profile rx1 ac 0 1 &&
	../../../scripts/cxi_mgmt enable --rx-profile rx1 &&
	../../../scripts/cxi_mgmt create --tx-profile tx1 &&
	../../../scripts/cxi_mgmt set-vni --tx-profile tx1 128 127 true &&
	../../../scripts/cxi_mgmt add-ac-entry --tx-profile tx1 ac 0 1 &&
	../../../scripts/cxi_mgmt  set-tc --tx-profile tx1 dedicated true &&
	../../../scripts/cxi_mgmt  set-tc --tx-profile tx1 best_effort true &&
	../../../scripts/cxi_mgmt  set-tc --tx-profile tx1 low_latency true &&
	../../../scripts/cxi_mgmt  set-tc --tx-profile tx1 bulk_data true &&
	../../../scripts/cxi_mgmt enable --tx-profile tx1 &&
	../../../ucxi/test_ucxi_atu -s 1 -v 129 -d cxi0 &> ../$(basename "$0").test.output
"

test_expect_success "Run test configfs cleanup" "
	dmesg --clear &&
	../../../scripts/cxi_mgmt cleanup &> ../$(basename "$0").cleanup.output
"

test_expect_success "Remove core driver" "
	dmesg --clear &&
	rmmod cxi-user &&
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

dmesg > ../$(basename "$0").dmesg.txt

test_done
