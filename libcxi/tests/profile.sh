#!/bin/bash
# set -x

MY_DIR="../../cxi-driver/scripts"

# Gives a VNI range of 16-31
match=16
ignore=15
rgroup_name=""
tle=0
exclusive=0

while getopts "i:m:r:t:chx" OPTION; do
        case $OPTION in
        h)      # help
                echo "${0##*/} [-chrx] [-i ignore]  [-m match] [-r name] [-t tles]"
                echo "    Create RX/TX profiles and rgroup"
                echo "    -h This help message"
                echo "    -c clean up the all rgroups and profiles created by configfs"
                echo "    -i ignore value for RX and TX profiles"
                echo "    -m match value for RX and TX profiles"
                echo "    -r Make an rgroup named 'name' with default resources"
                echo "    -x Make the TX profile with the exclusive cp attribute"
                exit 0
                ;;
        c)      # cleanup
                $MY_DIR/cxi_mgmt cleanup
                exit 0
                ;;
        m)      # match
                match="$OPTARG"
                ;;
        i)      # ignore
                ignore="$OPTARG"
                ;;
        r)      # make rgroup
                rgroup_name="$OPTARG"
                ;;
        t)      # tle
                tle="$OPTARG"
                ;;
        x)      # exclusive cp
                exclusive=1
                ;;
        esac
done

if [ ! $rgroup_name = "" ]; then
	$MY_DIR/cxi_mgmt create --rsrc-grp $rgroup_name
	$MY_DIR/cxi_mgmt add-resource --device cxi0 --rsrc-grp $rgroup_name ac 1022 0
	$MY_DIR/cxi_mgmt add-resource --device cxi0 --rsrc-grp $rgroup_name ptlte 1023 0
	$MY_DIR/cxi_mgmt add-resource --device cxi0 --rsrc-grp $rgroup_name txq 1024 0
	$MY_DIR/cxi_mgmt add-resource --device cxi0 --rsrc-grp $rgroup_name tgq 512 0
	$MY_DIR/cxi_mgmt add-resource --device cxi0 --rsrc-grp $rgroup_name eq 2047 0
	$MY_DIR/cxi_mgmt add-resource --device cxi0 --rsrc-grp $rgroup_name ct 2047 0
	$MY_DIR/cxi_mgmt add-resource --device cxi0 --rsrc-grp $rgroup_name pe0_le 16384 0
	$MY_DIR/cxi_mgmt add-resource --device cxi0 --rsrc-grp $rgroup_name pe1_le 16384 0
	$MY_DIR/cxi_mgmt add-resource --device cxi0 --rsrc-grp $rgroup_name pe2_le 16384 0
	$MY_DIR/cxi_mgmt add-resource --device cxi0 --rsrc-grp $rgroup_name pe3_le 16384 0
	if [ $tle -ne 0 ]; then
		# TLEs should be only be added if CTs will be used.
		$MY_DIR/cxi_mgmt add-resource --device cxi0 --rsrc-grp $rgroup_name tle $tle $tle
	fi
	$MY_DIR/cxi_mgmt add-ac-entry --rsrc-grp $rgroup_name ac 0 1
	$MY_DIR/cxi_mgmt enable --rsrc-grp $rgroup_name
fi


$MY_DIR/cxi_mgmt create --rx-profile rx1
$MY_DIR/cxi_mgmt set-vni --rx-profile rx1 $match $ignore
$MY_DIR/cxi_mgmt add-ac-entry --rx-profile rx1 ac 0 4
$MY_DIR/cxi_mgmt enable --rx-profile rx1

$MY_DIR/cxi_mgmt create --tx-profile tx1
$MY_DIR/cxi_mgmt set-vni --tx-profile tx1 $match $ignore $exclusive
$MY_DIR/cxi_mgmt add-ac-entry --tx-profile tx1 ac 0 4
$MY_DIR/cxi_mgmt set-tc --tx-profile tx1 dedicated 1
$MY_DIR/cxi_mgmt set-tc --tx-profile tx1 best_effort 1
$MY_DIR/cxi_mgmt set-tc --tx-profile tx1 low_latency 1
$MY_DIR/cxi_mgmt set-tc --tx-profile tx1 bulk_data 1
$MY_DIR/cxi_mgmt enable --tx-profile tx1

#cat /sys/kernel/debug/cxi/cxi0/services
# /home/chuckf/bin/dump-configfs
