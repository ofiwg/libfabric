#!/bin/bash

# Basic tests for AMO remap sysfs entry

. ./preamble.sh

test_description="Basic tests for AMO remap sysfs entry"

. ./sharness.sh

test_expect_success "Inserting base link driver" "
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	[ $(lsmod | awk '{ print $1 }' | grep -c cxi_sbl) -eq 0 ]
"

test_expect_success "Inserting slingshot link driver" "
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	[ $(lsmod | awk '{ print $1 }' | grep -c cxi-sl) -eq 0 ]
"

test_expect_success "Inserting core driver" "
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko &&
	[ $(lsmod | awk '{ print $1 }' | grep -c cxi_ss1) -eq 0 ]
"

test_expect_success "sys class entry amo_remap_to_pcie_fadd for cxi0 exists" "
	[ -f /sys/class/cxi/cxi0/device/properties/amo_remap_to_pcie_fadd ]
"

test_expect_success "Test valid value -1" "
	echo -1 > /sys/class/cxi/cxi0/device/properties/amo_remap_to_pcie_fadd
"

test_expect_success "Test valid value 0" "
	echo 0 > /sys/class/cxi/cxi0/device/properties/amo_remap_to_pcie_fadd
"

test_expect_success "Test valid value 1" "
	echo 1 > /sys/class/cxi/cxi0/device/properties/amo_remap_to_pcie_fadd
"

test_expect_success "Test valid value 2" "
	echo 2 > /sys/class/cxi/cxi0/device/properties/amo_remap_to_pcie_fadd
"

test_expect_success "Test valid value 4" "
	echo 4 > /sys/class/cxi/cxi0/device/properties/amo_remap_to_pcie_fadd
"

test_expect_success "Test valid value 5" "
	echo 5 > /sys/class/cxi/cxi0/device/properties/amo_remap_to_pcie_fadd
"

test_expect_success "Test valid value 6" "
	echo 6 > /sys/class/cxi/cxi0/device/properties/amo_remap_to_pcie_fadd
"

test_expect_success "Test valid value 7" "
	echo 7 > /sys/class/cxi/cxi0/device/properties/amo_remap_to_pcie_fadd
"

test_expect_success "Test valid value 8" "
	echo 8 > /sys/class/cxi/cxi0/device/properties/amo_remap_to_pcie_fadd
"

test_expect_success "Test valid value 9" "
	echo 9 > /sys/class/cxi/cxi0/device/properties/amo_remap_to_pcie_fadd
"

test_expect_success "Test valid value 10" "
	echo 10 > /sys/class/cxi/cxi0/device/properties/amo_remap_to_pcie_fadd
"

# Sharness doesn't support this...
#test_expect_failure "Test invalid value 11" "
#	echo 11 > /sys/class/cxi/cxi0/device/properties/amo_remap_to_pcie_fadd
#"

test_expect_success "No Oops" "
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

dmesg > ../$(basename "$0").dmesg.txt

test_done
