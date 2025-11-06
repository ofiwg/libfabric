#!/bin/bash

# Simple test that starts the VM and checks that the QEMU Cassini device is
# present, and that the driver loads.

. ./preamble.sh

test_description="Basic tests for cxi-ss1"

. ./sharness.sh

test_expect_success "One device is present" "
	[ $(lspci -n | grep -c '17db:0501') -eq 1 ]
"

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

test_expect_success "Inserting user driver" "
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-user.ko &&
	[ $(lsmod | awk '{ print $1 }' | grep -c cxi_user) -eq 0 ]
"

test_expect_success "sys class entry for cxi0 exists" "
	[ -L /sys/class/cxi/cxi0 ]
"

PROPDIR="/sys/class/cxi/cxi0/device/properties"
test_expect_success "various $PROPDIR values" "
	[ -d $PROPDIR ] &&
	[ -f $PROPDIR/pid_bits ] &&
	[ $(cat $PROPDIR/pid_bits) -eq 9 ] &&
	[ -f $PROPDIR/pid_count ] &&
	[ $(cat $PROPDIR/pid_count) -eq 511 ] &&
	[ -f $PROPDIR/pid_granule ] &&
	[ $(cat $PROPDIR/pid_granule) -eq 256 ]
"

FRUDIR="/sys/class/cxi/cxi0/device/fru"
test_expect_success "various $FRUDIR values" "
	[ -d $FRUDIR ] &&
	[ -f $FRUDIR/serial_number ] &&
	[ \"$(cat $FRUDIR/serial_number)\" = \"HY19490002\" ]
	[ -f $FRUDIR/part_number ] &&
	[ \"$(cat $FRUDIR/part_number)\" = \"102325100\" ] &&
	[ -f $FRUDIR/asset_tag ] &&
	[ \"x$(cat $FRUDIR/asset_tag)\" = \"x\" ]
"

test_expect_success "sbl debugfs entry exists" "
	[ -f /sys/kernel/debug/cxi/cxi0/port ]
"

test_expect_success "sbl sysfs entry exists" "
	[ -d /sys/class/cxi/cxi0/device/port ]
"

test_expect_success SRIOV "Create 1 VFs" "
	echo 1 > /sys/class/cxi/cxi0/device/sriov_numvfs
"

if test_have_prereq SRIOV; then
	n_devices=2
else
	n_devices=1
fi

test_expect_success SRIOV "$n_devices devices are now present" "
	[ $(lspci  -n | grep -c '17db:0501') -eq $n_devices ]
"

test_expect_success "Intel IOMMU is properly set up" "
	[ $(dmesg | grep -c 'DMAR: IOMMU enabled') -eq 1 ]
"

test_expect_success "MSI-X is present (PF)" "
	[ $(lspci -d 17db: -vv | grep -c 'Capabilities: \[b0\] MSI-X: Enable+ Count=512 Masked-') -eq 1 ]
"

test_expect_success SRIOV "MSI-X is present (VF)" "
	[ $(lspci -d 17db: -vv | grep -c 'Capabilities: \[b0\] MSI-X: Enable+ Count=63 Masked-') -eq 1 ]
"

test_expect_success "PCI capabilities" "
	[ $(lspci -d 17db: -vv | grep -c 'Advanced Error Reporting') -eq $n_devices ] &&
	[ $(lspci -d 17db: -vv | grep -c 'Alternative Routing-ID Interpretation') -eq $n_devices ] &&
	[ $(lspci -d 17db: -vv | grep -c 'Address Translation Service') -eq 1 ] &&
	[ $(lspci -d 17db: -vv | grep -c 'Single Root I/O Virtualization') -eq 1 ]
"

test_expect_success "No Oops" "
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

dmesg > ../$(basename "$0").dmesg.txt

test_done
