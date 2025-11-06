# SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
# Copyright 2020 Hewlett Packard Enterprise Development LP

# Framework for testing in a VM using virtme
#
# This must be sourced by a test script

# Paths to our tools. They can be overridden from the caller's
# environment.
TOP_DIR=$(realpath $(pwd)/../..)
VIRTME_DIR=${VIRTME_DIR:-$TOP_DIR/virtme}
QEMU_DIR=${QEMU_DIR:-$TOP_DIR/cassini-qemu/x86_64-softmmu}
NETSIM_DIR=${NETSIM_DIR:-$TOP_DIR/nic-emu}

# Arguments are '$noexit [command [args...]]'
# If $noexit != 0, this will run the test as a boot script and stay in the VM.
# Otherwise, this runs the test in the VM, then exits the VM.
function startvm {
	export PATH=$QEMU_DIR:$VIRTME_DIR:/sbin:$PATH

	if [[ $1 -eq 0 ]]; then
	    scriptop='--script-sh'
	else
	    scriptop='--init-sh'
	fi
	shift 1

	# -M q35 = Standard PC (Q35 + ICH9, 2009) (alias of pc-q35-2.10)
	QEMU_OPTS="--qemu-opts -smp cores=2 -device ccn -machine q35,kernel-irqchip=split -device intel-iommu,intremap=on,caching-mode=on -m 8192"
	KERN_OPTS="--kopt iommu=pt --kopt intel_iommu=on --kopt iomem=relaxed"
	KERN_OPTS="$KERN_OPTS --kopt transparent_hugepage=never --kopt hugepagesz=1g --kopt default_hugepagesz=1g --kopt hugepages=1"

	# --script-sh can take arguments in script, but --init-sh cannot
	echo "$@" >./test_cmd.sh
	chmod +x ./test_cmd.sh

	if [[ -v KDIR ]]; then
	    KERNEL="--kdir $KDIR --mods=auto"
	else
	    KERNEL="--installed-kernel"
	fi

	# netsim cannot take arguments to the virtme-run command
	echo "virtme-run $KERNEL --pwd --rwdir=$(pwd) $scriptop $(realpath ./test_cmd.sh) $KERN_OPTS $QEMU_OPTS" >netsim_cmd.sh
	chmod +x ./netsim_cmd.sh
	$NETSIM_DIR/netsim ./netsim_cmd.sh
}
