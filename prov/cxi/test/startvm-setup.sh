#!/bin/sh
#
# Initialize a VM for CXI testing and run a command.

DBS_DIR=$(realpath "../../../..")

if [[ -z $RUNCMD ]]; then
    RUNCMD="$@"
fi

export LC_ALL=en_US.UTF-8

modprobe ptp
modprobe amd_iommu_v2
insmod $DBS_DIR/slingshot_base_link/sbl.ko
insmod $DBS_DIR/cxi-driver/cxi/cxi-core.ko
insmod $DBS_DIR/cxi-driver/cxi/cxi-user.ko

# Add pycxi utilities to path
export PATH=$DBS_DIR/pycxi/utils:$PATH

# Initialize pycxi environment
. $DBS_DIR/pycxi/.venv/bin/activate

if [[ ! -z $RUNCMD ]]; then
    $RUNCMD
fi
