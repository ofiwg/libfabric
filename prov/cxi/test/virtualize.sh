#!/bin/bash

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
TOP_DIR=${TOP_DIR:-$(realpath $(git rev-parse --show-toplevel)/../)}

insmod $TOP_DIR/cxi-driver/cxi/cxicore.ko
insmod $TOP_DIR/cxi-driver/cxi/cxi-user.ko

# Execute the command (if any)
$@
