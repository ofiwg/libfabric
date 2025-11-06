#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
# Copyright 2020 Hewlett Packard Enterprise Development LP

# If not in a VM, start one and execute self.
#
[[ $(basename $0) = "preamble.sh" ]] &&
	echo "This script is only intended to be run by tests. Exiting." && exit 1

HYP=$(grep -c "^flags.*\ hypervisor" /proc/cpuinfo)
if [[ $HYP -eq 0 ]]; then
    . ./framework.sh

    noexit=0
    while [[ $# -gt 0 ]]; do
	case "$1" in
	    -n|--no-exit)
		noexit=1
		;;
	    -*)
		echo "option '$1' not recognized"
		exit 1
		;;
	    *)
		break
		;;
	esac
	shift 1
    done

    startvm $noexit $(realpath $(basename $0)) $@
    exit 0
fi
