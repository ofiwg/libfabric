#!/bin/sh

# Setup script run in the VM, used by the testit.sh script.

modprobe configfs
mount -t configfs none /sys/kernel/config
modprobe ptp
modprobe iommu_v2 >/dev/null 2>/dev/null || modprobe amd_iommu_v2 >/dev/null 2>/dev/null
insmod ../../slingshot_base_link/cxi-sbl.ko
insmod ../../sl-driver/drivers/net/ethernet/hpe/sl/cxi-sl.ko
insmod ../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko disable_default_svc=0
insmod ../drivers/net/ethernet/hpe/ss1/cxi-user.ko
