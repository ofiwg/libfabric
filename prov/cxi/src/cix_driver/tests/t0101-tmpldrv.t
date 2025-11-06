#!/bin/bash

# Test template driver, and client support

. ./preamble.sh

test_description="Basic tests for cxi-ss1"

. ./sharness.sh

# Disable for now. The kernel on jenkins seems to no have enough
# interrupts to create more than one VFs. TODO.
test_done

# Originally, only the PF device is present, so it should be cxi0
PFDEV=/sys/class/cxi/cxi0/device

# Create a given number of VFs
function create_vfs {
	echo $1 > $PFDEV/sriov_numvfs &&
	[[ $(cat $PFDEV/sriov_numvfs) -eq $1 ]]
}

# Check the number of cxi device is correct
# Arg 1 is the number of expected cxi device
function check_cxi {
	NCXI=$(ls /sys/class/cxi/ | wc -w)
	[[ $NCXI -eq $1 ]]
}

test_expect_success "Inserting core driver" "
	insmod ../../../../slingshot_base_link/cxi-sbl.ko &&
	insmod ../../../../sl-driver/knl/cxi-sl.ko &&
	insmod ../../../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Inserting template driver" "
	insmod ../../../cxi/tests/test-atu.ko &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "Template client is now present" "
	[ $(dmesg | grep -c 'Adding template client for device cxi0') -eq 1 ] &&
	[ $(dmesg | grep -c 'Adding template client for device cxi1') -eq 0 ] &&
	[ $(dmesg | grep -c 'Adding template client for device cxi2') -eq 0 ] &&
	[ $(dmesg | grep -c 'Adding template client for device cxi3') -eq 0 ] &&
	[ $(dmesg | grep -c 'Adding template client for device cxi4') -eq 0 ]
"

test_expect_success "Add 4 more PCI devices" "
    create_vfs 4 && check_cxi 5
"

test_expect_success "More template clients are present" "
	[ $(dmesg | grep -c 'Adding template client for device cxi0') -eq 1 ] &&
	[ $(dmesg | grep -c 'Adding template client for device cxi1') -eq 1 ] &&
	[ $(dmesg | grep -c 'Adding template client for device cxi2') -eq 1 ] &&
	[ $(dmesg | grep -c 'Adding template client for device cxi3') -eq 1 ] &&
	[ $(dmesg | grep -c 'Adding template client for device cxi4') -eq 1 ]
"

test_expect_success "Remove existing VFs" "
	echo 0 > $PFDEV/sriov_numvfs
	[[ $(cat $PFDEV/sriov_numvfs) -ne 0 ]] && check_cxi 1
"

test_expect_success "4 template clients removed" "
	[ $(dmesg | grep -c 'Removing template client for device cxi0') -eq 0 ] &&
	[ $(dmesg | grep -c 'Removing template client for device cxi1') -eq 1 ] &&
	[ $(dmesg | grep -c 'Removing template client for device cxi2') -eq 1 ] &&
	[ $(dmesg | grep -c 'Removing template client for device cxi3') -eq 1 ] &&
	[ $(dmesg | grep -c 'Removing template client for device cxi4') -eq 1 ]
"

test_expect_success "module removal" "
	rmmod test-atu &&
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

test_expect_success "No Oops" "
	[ $(dmesg | grep -c 'Modules linked in') -eq 0 ]
"

dmesg > ../$(basename "$0").dmesg.txt

test_done
