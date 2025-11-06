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

test_expect_success "Run configfs test program" "
	../../../scripts/cxi_mgmt create --rsrc-grp r1 &&
	../../../scripts/cxi_mgmt add-resource --device cxi0 --rsrc-grp r1 ac 100 100 &&
	../../../scripts/cxi_mgmt add-resource --rsrc-grp r1 ct 100 100 &&
	../../../scripts/cxi_mgmt add-resource --rsrc-grp r1 eq 100 100 &&
	../../../scripts/cxi_mgmt add-resource --rsrc-grp r1 tgq 100 100 &&
	../../../scripts/cxi_mgmt add-resource --rsrc-grp r1 txq 100 100 &&
	../../../scripts/cxi_mgmt add-resource --rsrc-grp r1 pe0_le 100 100 &&
	../../../scripts/cxi_mgmt add-resource --rsrc-grp r1 pe1_le 100 100 &&
	../../../scripts/cxi_mgmt add-resource --rsrc-grp r1 pe2_le 100 100 &&
	../../../scripts/cxi_mgmt add-resource --rsrc-grp r1 pe3_le 100 100 &&
	../../../scripts/cxi_mgmt add-resource --rsrc-grp r1 tle 100 100 &&
	../../../scripts/cxi_mgmt add-ac-entry --rsrc-grp r1 ac 0 1 &&
	../../../scripts/cxi_mgmt enable --rsrc-grp r1 &&
	../../../ucxi/test_ucxi_atu -s 2 -d cxi0 &> ../$(basename "$0").test.output
"

test_expect_success "Remove configfs resources" "
	dmesg --clear &&
	../../../scripts/cxi_mgmt disable --rsrc-grp r1 &&
	../../../scripts/cxi_mgmt remove-resource --device cxi0 --rsrc-grp r1 ac &&
	../../../scripts/cxi_mgmt remove-resource --rsrc-grp r1 ct &&
	../../../scripts/cxi_mgmt remove-resource --rsrc-grp r1 eq &&
	../../../scripts/cxi_mgmt remove-resource --rsrc-grp r1 tgq &&
	../../../scripts/cxi_mgmt remove-resource --rsrc-grp r1 txq &&
	../../../scripts/cxi_mgmt remove-resource --rsrc-grp r1 pe0_le &&
	../../../scripts/cxi_mgmt remove-resource --rsrc-grp r1 pe1_le &&
	../../../scripts/cxi_mgmt remove-resource --rsrc-grp r1 pe2_le &&
	../../../scripts/cxi_mgmt remove-resource --rsrc-grp r1 pe3_le &&
	../../../scripts/cxi_mgmt remove-resource --rsrc-grp r1 tle
"

test_expect_success "ConfigFS cleanup" "
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
