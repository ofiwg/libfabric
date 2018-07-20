# Framework for testing in a VM using virtme
#
# This must be sourced by a test script

# Paths to our tools. They can be overriden from the caller's
# environment.
TOP_DIR=${TOP_DIR:-$(realpath $(git rev-parse --show-toplevel)/../)}
VIRTME_DIR=${VIRTME_DIR:-$TOP_DIR/virtme}
QEMU_DIR=${QEMU_DIR:-$TOP_DIR/cassini-qemu/x86_64-softmmu}
NETSIM_DIR=${NETSIM_DIR:-$TOP_DIR/nic-emu}

# Start a VM and run a script
# arg 1 = the script to run
# arg 2 = 0 to use --script-sh, 1 to use --init-sh
function startvm {
	export PATH=$QEMU_DIR:$VIRTME_DIR:/sbin:$PATH

	if [[ $2 -eq 0 ]]; then
		scriptop="--script-sh"
	else
		scriptop="--init-sh"
	fi

	# -M q35 = Standard PC (Q35 + ICH9, 2009) (alias of pc-q35-2.10)
	QEMU_OPTS="--qemu-opts -smp cores=2 -device ccn -machine q35,kernel-irqchip=split -device intel-iommu,intremap=on,caching-mode=on -m 1024"
	KERN_OPTS="--kopt iommu=pt --kopt intel_iommu=on --kopt iomem=relaxed"

	echo "virtme-run --installed-kernel --pwd --rwdir=$(pwd) $scriptop $1 $KERN_OPTS $QEMU_OPTS" > test_cmd.sh
	chmod +x test_cmd.sh
	$NETSIM_DIR/netsim ./test_cmd.sh
}
