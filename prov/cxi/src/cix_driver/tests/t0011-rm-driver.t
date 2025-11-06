#!/bin/bash

# Add and remove the device a couple times

. ./preamble.sh

test_description="Device removal"

. ./sharness.sh

test_expect_success "cxi-ss1 sys directories do not exist" "
	[[ ! -d /sys/module/cxi_ss1 ]] && [[ ! -d /sys/class/cxi ]]
"

test_expect_success "module insertion" "
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "cxi-ss1 sys directories are present" "
	[[ -d /sys/module/cxi_ss1 ]] && [[ -d /sys/class/cxi ]]
"

test_expect_success "module removal" "
	rmmod cxi-ss1 &&
	rmmod cxi-sbl &&
	rmmod cxi-sl &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "cxi-ss1 sys directories are gone" "
	[[ ! -d /sys/module/cxi_ss1 ]] && [[ ! -d /sys/class/cxi ]]
"

test_expect_success "module insertion" "
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "cxi-ss1 sys directories are present" "
	[[ -d /sys/module/cxi_ss1 ]] && [[ -d /sys/class/cxi ]]
"

test_expect_success "module removal" "
	rmmod cxi-ss1 &&
	rmmod cxi-sl &&
	rmmod cxi-sbl &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "cxi-ss1 sys directories are gone" "
	[[ ! -d /sys/module/cxi_ss1 ]] && [[ ! -d /sys/class/cxi ]]
"

test_expect_success "No Oops" "
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

dmesg > ../$(basename "$0").dmesg.txt

test_done
