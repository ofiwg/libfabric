#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
# Copyright 2020 Hewlett Packard Enterprise Development LP

# Run this script inside a VM.

# Source preamble file
# If not running in netsim, this will re-invoke this script in netsim and exit
# If running in netsim, this will continue
if [[ ! -f ./preamble.sh ]]; then
	if ! cd $(dirname $0); then
		echo "cannot find $(dirname $0) from $(pwd)"
		exit 1
	fi
fi
. ./preamble.sh $@

# We are now inside a VM.  Load the CXI drivers.
# Virtualization arguments have been stripped.
export GIT_DISCOVERY_ACROSS_FILESYSTEM=1
TOP_DIR=$(realpath $(pwd)/../..)
KO_PATH=drivers/net/ethernet/hpe/ss1

modprobe fuse
mkdir -p /run/cxi/cxi0
modprobe configfs
mount -t configfs none /sys/kernel/config
modprobe ptp
modprobe iommu_v2 || modprobe amd_iommu_v2
insmod $TOP_DIR/slingshot_base_link/cxi-sbl.ko
insmod $TOP_DIR/sl-driver/knl/cxi-sl.ko
insmod $TOP_DIR/cxi-driver/$KO_PATH/cxi-ss1.ko disable_default_svc=0 active_qos_profile=1
insmod $TOP_DIR/cxi-driver/$KO_PATH/cxi-user.ko

# Execute the command (if any)
$@
