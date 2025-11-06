#!/bin/bash

# Simple test that starts the VM and checks that the QEMU Cassini
# device is present, and that the Ethernet driver loads.

. ./preamble.sh

test_description="Basic ethernet test"

. ./sharness.sh

test_expect_success "One device is present" "
	[ $(lspci -n | grep -c '17db:0501') -eq 1 ]
"

test_expect_success "Inserting driver" "
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Inserting CXI ethernet driver" "
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-eth.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "module removal" "
	rmmod cxi-eth cxi-ss1 cxi-sbl cxi-sl &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "No Oops" "
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

dmesg > ../$(basename "$0").dmesg.txt

test_done
