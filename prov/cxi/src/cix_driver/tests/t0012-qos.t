#!/bin/bash

# Validates QoS Profiles

. ./preamble.sh

test_description="QoS Tests"

. ./sharness.sh

test_expect_success "Load SBL and SL" "
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Load HPC" "
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko active_qos_profile=1 &&
	[ $(lsmod | awk '{ print $1 }' | grep -c cxi_ss1) -eq 0 ]
"

test_expect_success "Validate HPC" "
	[ $(dmesg | grep -c 'QoS Profile: HPC') -eq 1 ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/ ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/best_effort ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/bulk_data ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/low_latency ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/dedicated_access ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/ethernet/ethernet_shared ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/ethernet/ethernet1 ] &&

	[ ! -d /sys/class/cxi/cxi0/device/traffic_classes/ethernet/ethernet2 ]
"

rmmod cxi-ss1

test_expect_success "Load LL_BE_BD_ET" "
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko active_qos_profile=2 &&
	[ $(lsmod | awk '{ print $1 }' | grep -c cxi_ss1) -eq 0 ]
"

test_expect_success "Validate LL_BE_BD_ET" "
	[ $(dmesg | grep -c 'QoS Profile: LL_BE_BD_ET') -eq 1 ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/ ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/best_effort ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/bulk_data ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/low_latency ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/ethernet/ethernet_shared ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/ethernet/ethernet1 ] &&

	[ ! -d /sys/class/cxi/cxi0/device/traffic_classes/dedicated_access ] &&
	[ ! -d /sys/class/cxi/cxi0/device/traffic_classes/ethernet/ethernet2 ]
"

rmmod cxi-ss1

test_expect_success "Load LL_BE_BD_ET1_ET2" "
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko active_qos_profile=3 &&
	[ $(lsmod | awk '{ print $1 }' | grep -c cxi_ss1) -eq 0 ]
"

test_expect_success "Validate LL_BE_BD_ET1_ET2" "
	[ $(dmesg | grep -c 'QoS Profile: LL_BE_BD_ET1_ET2') -eq 1 ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/ ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/best_effort ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/bulk_data ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/low_latency ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/ethernet/ethernet_shared ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/ethernet/ethernet1 ] &&
	[ -d /sys/class/cxi/cxi0/device/traffic_classes/ethernet/ethernet2 ] &&

	[ ! -d /sys/class/cxi/cxi0/device/traffic_classes/dedicated_access ]
"

dmesg > ../$(basename "$0").dmesg.txt

test_done
