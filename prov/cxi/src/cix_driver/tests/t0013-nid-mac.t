#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
# Copyright 2025 Hewlett Packard Enterprise Development LP

# Validate setting and reading NID and MAC address

. ./preamble.sh

test_description="Tests setting and reading NID and MAC address"

. ./sharness.sh

NID_DEV=/sys/class/cxi_user/cxi0/device/properties/nid

# Insert drivers to get to the point where the cxi_user device is present
test_expect_success "Inserting drivers" "
	modprobe ptp &&
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-eth.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-user.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

# Helper function to find CXI ethernet interface
find_cxi_eth_iface() {
	find /sys/class/net -name '*' -type l -exec basename {} \; | while read iface; do
		if [ -e "/sys/class/net/$iface/device/driver" ]; then
			driver=$(readlink /sys/class/net/$iface/device/driver 2>/dev/null | xargs basename)
			if [ "$driver" = "cxi_ss1" ]; then
				echo $iface
				break
			fi
		fi
	done
}

test_expect_success "Finding one Ethernet interface" "
	IFACE=\$(find_cxi_eth_iface) &&
	[ -n \"\$IFACE\" ] &&
	echo \"Found CXI ethernet interface: \$IFACE\"
"

test_expect_success "sys class user entry for cxi0 exists" "
	[ -L /sys/class/cxi_user/cxi0 ]
"

test_expect_success "NID is derived from MAC address if not explicitly set" "
	for n in 1 2 3 4; do
		ip link set dev \$(find_cxi_eth_iface) address \"02:00:00:00:00:0\$n\" &&
		read_nid=\$(cat \$NID_DEV) &&
		expected_nid=\"0x\$n\" &&
		[ \"\$read_nid\" = \"\$expected_nid\" ]
	done
"

test_expect_success "Explicitly setting NID works" "
	for n in 5 6 7 8; do
		echo \$n > \$NID_DEV &&
		read_nid=\$(cat \$NID_DEV) &&
		expected_nid=\"0x\$n\" &&
		[ \"\$read_nid\" = \"\$expected_nid\" ]
	done
"

test_expect_success "After setting NID, changing MAC does not change NID" "
	echo 6 > \$NID_DEV &&
	for n in 1 2 3 4; do
		ip link set dev \$(find_cxi_eth_iface) address \"02:00:00:00:00:0\$n\" &&
		read_nid=\$(cat \$NID_DEV) &&
		expected_nid=\"0x6\" &&
		[ \"\$read_nid\" = \"\$expected_nid\" ]
	done
"

test_done
