#!/bin/bash

# Simple test that starts the VM and checks that the QEMU Cassini
# device is present, that the Ethernet driver loads, and that the SBL debugfs
# entry reflects the SBL module is healthy

. ./preamble.sh

test_description="Basic SBL test"

. ./sharness.sh


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

# Note: sleep 3 seconds before checking, because there is an
# unconditional sleep for optical cables in
# sbl_serdes_start()->sbl_serdes_optical_lock_delay()
# This also give time for the device to appear without checkout udev
# events.
sleep 3

# Find the eth device name
for f in /sys/class/net/*
do
    [[ -d "$f/device/cxi" ]] || continue

    ETH_INTF=$(basename "$f")
done

SBL_DEBUGFS=$(find /sys/kernel/debug/cxi -name port)

test_expect_success "Debugfs entry exists" "
	[ $SBL_DEBUGFS != '' ]
"


test_expect_success "Checking SBL debugfs entry: type" "
	cat $SBL_DEBUGFS | grep type | grep ether &&
	[ $? -eq 0 ]
"

test_expect_success "Checking SBL debugfs entry: SerDes" "
	cat /sys/kernel/debug/cxi/cxi0/port | grep serdes | grep running &&
	[ $? -eq 0 ]
"

test_expect_success "Checking SBL debugfs entry: bl" "
	cat /sys/kernel/debug/cxi/cxi0/port | grep 'base link state' | grep up &&
	[ $? -eq 0 ]
"

test_expect_success "Checking SBL debugfs entry: lmon" "
	cat /sys/kernel/debug/cxi/cxi0/port | grep lmon | grep 'dirn up' &&
	[ $? -eq 0 ]
"

test_expect_success "Checking ethtool can set internal loopback" "
        timeout 1 ethtool --set-priv-flags $ETH_INTF internal-loopback on
	[ $? -eq 0 ]
"
sleep 1

test_expect_success "Checking SBL debugfs entry: internal loopback" "
	cat /sys/kernel/debug/cxi/cxi0/port | grep serdes | grep local &&
	[ $? -eq 0 ]
"

test_expect_success "Checking ethtool doesn't hang 1" "
        timeout 1 ethtool --set-priv-flags $ETH_INTF llr on
	[ $? -eq 0 ]
"
sleep 1

test_expect_success "Checking SBL debugfs entry: bl" "
	cat /sys/kernel/debug/cxi/cxi0/port | grep 'base link state' | grep starting &&
	[ $? -eq 0 ]
"

test_expect_success "Checking ethtool doesn't hang 2" "
        timeout 1 ethtool --set-priv-flags $ETH_INTF llr off
	[ $? -eq 0 ]
"
sleep 1

test_expect_success "Checking SBL debugfs entry: bl" "
	cat /sys/kernel/debug/cxi/cxi0/port | grep 'base link state' | grep up &&
	[ $? -eq 0 ]
"

test_expect_success "Checking SBL debugfs entry: internal loopback" "
	cat /sys/kernel/debug/cxi/cxi0/port | grep serdes | grep local &&
	[ $? -eq 0 ]
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
