#!/bin/bash

# Parse arguments.
noexit=0
for parm in $@; do
	case $parm in
	-n|--no-exit) noexit=1;;
	esac
done

# Run this script inside a VM.
cd $(dirname $0)
. ./preamble.sh $noexit

# We are now inside a VM.  Load the CXI drivers.
export GIT_DISCOVERY_ACROSS_FILESYSTEM=1
TOP_DIR=${TOP_DIR:-$(realpath $(git rev-parse --show-toplevel)/../)}

insmod $TOP_DIR/cxi-driver/cxi/cxicore.ko
insmod $TOP_DIR/cxi-driver/cxi/cxi-user.ko

# Run unit tests.  $(CWD) should be writeable.
#
# Tests must be run with -j1 to prevent processes from racing to allocate CXI
# resources.
./cxitest -j1 --verbose --tap=cxitest.tap
