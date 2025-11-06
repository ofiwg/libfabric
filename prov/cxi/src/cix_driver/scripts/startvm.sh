#!/bin/bash

# Start a VM, optionally load the test driver, and exit.

# The parameters are given to netsim. See netsim -h.
#   ./startvm.sh       -> run one instance with 1 NIC
#   ./startvm.sh -N 3  -> run one instance with 3 NICs
#   ./startvm.sh -n 2  -> launch 2 VMs each with 1 NIC
#
#
# Note: When using multiple VMs, it is recommended to set the USE_XTERM
# variable. Each VM will be opened in a new xterm window.
#
# USE_XTERM=1 ./startvm.sh -n 2

DIR=$(dirname $(realpath "$0"))
cd $DIR

VM_INIT_SCRIPT=${VM_INIT_SCRIPT:='./startvm-setup.sh'}

export OMPI_MCA_btl_base_warn_component_unused=0

# Check whether this script is already in a VM or not (ie. running
# under a hypervisor.) If not, we'll need a different setup for nested
# VMs.
HYP=$(grep -c "^flags.* hypervisor" /proc/cpuinfo)

# If the emulator is not running, start it. This script must run under
# its control, so qemu can connect to it. Simply relaunch self under
# netsim's control.
if [[ $HYP -eq 0 ]]; then
	if [[ ! -v NETSIM_ID ]]; then
		exec ../../nic-emu/netsim $@ $(basename $0)
	fi

	if [[ $NETSIM_NICS -eq 1 ]]; then
		CCN_OPTS="-device $NETSIM_CCN,addr=8"
	elif [[ $NETSIM_NICS -eq 2 ]]; then
		CCN_OPTS="-device $NETSIM_CCN,addr=8 -device $NETSIM_CCN,addr=13"
	elif [[ $NETSIM_NICS -eq 4 ]]; then
		CCN_OPTS="-device $NETSIM_CCN,addr=8 -device $NETSIM_CCN,addr=0xd -device $NETSIM_CCN,addr=0x12 -device $NETSIM_CCN,addr=0x17"
	fi
else
	NETSIM_NICS=0
	echo "****** This VM is running inside another VM ******"
fi

. ../tests/framework.sh

# Load vsock modules on host for VF-PF communication
modprobe vsock 2>/dev/null || echo "Warning: Could not load vsock module"
modprobe vhost_vsock 2>/dev/null || echo "Warning: Could not load vhost_vsock module"

# -M q35 = Standard PC (Q35 + ICH9, 2009) (alias of pc-q35-2.10)
# MSI-X needs interrupt remapping enabled to fully work.
# w/ Intel IOMMU. Intremap on requires kernel-irqchip=off OR kernel-irqchip=split
QEMU_OPTS="--qemu-opts -machine q35,kernel-irqchip=split -global q35-pcihost.pci-hole64-size=40G -device intel-iommu,intremap=on,caching-mode=on $CCN_OPTS"

# Example to create a VM with 2 numa nodes with one cpu each
# QEMU_OPTS="$QEMU_OPTS -smp cpus=2 -object memory-backend-ram,size=512M,id=m0 -object memory-backend-ram,size=512M,id=m1 -numa node,memdev=m0,cpus=0,nodeid=0 -numa node,memdev=m1,cpus=1,nodeid=1"

# Example to add a VM with 9 numa nodes, including 7 without a cpu.
#QEMU_OPTS="$QEMU_OPTS -smp cpus=2
#	-object memory-backend-ram,size=1G,id=m0 -numa node,memdev=m0,cpus=0,nodeid=0
#	-object memory-backend-ram,size=128M,id=m1 -numa node,memdev=m1,nodeid=1
#	-object memory-backend-ram,size=128M,id=m2 -numa node,memdev=m2,nodeid=2
#	-object memory-backend-ram,size=128M,id=m3 -numa node,memdev=m3,nodeid=3
#	-object memory-backend-ram,size=128M,id=m4 -numa node,memdev=m4,nodeid=4
#	-object memory-backend-ram,size=128M,id=m5 -numa node,memdev=m5,nodeid=5
#	-object memory-backend-ram,size=128M,id=m6 -numa node,memdev=m6,cpus=1,nodeid=6
#	-object memory-backend-ram,size=128M,id=m7 -numa node,memdev=m7,nodeid=7
#	-object memory-backend-ram,size=128M,id=m8 -numa node,memdev=m8,nodeid=8"

KERN_OPTS="--kopt iommu=pt --kopt intel_iommu=on --kopt iomem=relaxed"
KERN_OPTS="$KERN_OPTS --kopt transparent_hugepage=never --kopt hugepagesz=1g --kopt default_hugepagesz=1g --kopt hugepages=1"

if [[ $HYP -eq 0 ]]; then
	# First VM needs more memory to launch nested VMs
	# Only the first VM will have the Cassini qemu device. Nested VMs will
	# have VFs exported to them
	QEMU_OPTS="$QEMU_OPTS -m 2G"

	if [[ -n $QEMU_MOPTS ]]; then
		QEMU_OPTS="$QEMU_OPTS $QEMU_MOPTS"
	fi
else
	# Nested VM. Use the first PCI VF
	# PCIFN = 0000:00:14.0 or similar

	# Bind cxi1 to get its info
	echo 1 > /sys/class/cxi/cxi0/device/sriov_numvfs
	PCIFN=$(basename $(readlink /sys/class/cxi/cxi0/device/virtfn0))
	VENDOR=$(cat /sys/class/cxi/cxi0/device/virtfn0/vendor)
	DEVICE=$(cat /sys/class/cxi/cxi0/device/virtfn0/device)

	# Unbind VF from cxi core driver. cxi1 no longer exists
	echo $PCIFN > /sys/bus/pci/drivers/cxi_ss1/unbind

	# Bind the VF to vfio driver
	modprobe vfio_pci
	echo ${VENDOR##*x} ${DEVICE##*x} > /sys/bus/pci/drivers/vfio-pci/new_id

	# Tell qemu to bind the VF
	QEMU_OPTS="$QEMU_OPTS -device vfio-pci,host=$PCIFN"

	# Add vhost-vsock device for VF-PF communication
	QEMU_OPTS="$QEMU_OPTS -device vhost-vsock-pci,guest-cid=4"
fi

PATH=$QEMU_DIR:$VIRTME_DIR:/sbin:$PATH
VIRTME_OPTS="--rodir=/lib/firmware=$(pwd)/../../hms-artifacts"

if [[ $TESTING -eq 1 ]]; then
	# Put additional options here...
	VIRTME_OPTS="--rwdir=$(pwd)/../tests/tmptests ${VIRTME_OPTS}"
fi

if [[ -v KDIR ]]; then
    KERNEL="--kdir $KDIR --mods=auto"
else
    KERNEL="--installed-kernel"
fi

# Start the VM, execute the script inside, and exit ...
#virtme-run --installed-kernel --pwd --script-sh $VM_INIT_SCRIPT $KERN_OPTS $QEMU_OPTS

# ... or start a VM and execute the script but don't exit
if [[ $USE_XTERM -eq 1 ]]; then
	xterm -e "virtme-run $KERNEL --pwd --init-sh $VM_INIT_SCRIPT $VIRTME_OPTS $KERN_OPTS $QEMU_OPTS"
else
	virtme-run $KERNEL --pwd --init-sh $VM_INIT_SCRIPT $VIRTME_OPTS $KERN_OPTS $QEMU_OPTS
fi

# ... or just start a clean VM
#virtme-run --installed-kernel --pwd $KERN_OPTS $QEMU_OPTS
