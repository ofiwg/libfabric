#!/bin/bash
#
# If not in a VM, start one and execute self.
#

[[ $(basename $0) = "preamble.sh" ]] &&
	echo "This script is only intended to be run by tests. Exiting." && exit 1

HYP=$(grep -c "^flags.*\ hypervisor" /proc/cpuinfo)
if [[ $HYP -eq 0 ]]; then
	. ./framework.sh

	startvm $(realpath $(basename $0)) $1
	exit 0
fi
