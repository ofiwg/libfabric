#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
# Copyright 2023 Hewlett Packard Enterprise Development LP

# Setup an environment suitable for RoCE testing. Two VMs are started,
# one per xterm, and configured to use rxe. A single IB RC ping is
# done at the end to check the functionality.

# Start 2 VMs, that will execute this script.
HYP=$(grep -c "^flags.* hypervisor" /proc/cpuinfo)
if [[ $HYP -eq 0 ]]; then
    VM_INIT_SCRIPT="$0" USE_XTERM=1 ./startvm.sh -n 2
    exit 0
fi

# Load all the drivers
modprobe ptp
modprobe amd_iommu_v2 || modprobe iommu_v2
modprobe ib_uverbs
modprobe ip6_udp_tunnel
modprobe udp_tunnel
modprobe crc32_generic
insmod ../../rxe/rxe/rdma_rxe.ko
insmod ../../slingshot_base_link/cxi-sbl.ko
insmod ../../sl-driver/drivers/net/ethernet/hpe/sl/cxi-sl.ko
insmod ../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko disable_default_svc=0
insmod ../drivers/net/ethernet/hpe/ss1/cxi-user.ko
insmod ../drivers/net/ethernet/hpe/ss1/cxi-eth.ko

# Wait for the new devices to appear
sleep 1

# Find which node this script is running on. Node 0 has MAC address
# 00:0e:ab:00:00:00 and the other has 00:0e:ab:01:00:00. Set
# appropriate defaults. The MAC address has to be changed so that
# netsim can route the packets between the nodes.
ip a | grep 00:0e:ab:00:00:00
if [[ $? -eq 0 ]]; then
    NAME=node0
    IPADDR=192.168.1.1
    MACADDR=02:0e:ab:00:00:00
    ETHNAME=$(ip -j a | jq -r '.[] | select(.address=="00:0e:ab:00:00:00") | .ifname')
else
    NAME=node1
    IPADDR=192.168.1.2
    MACADDR=02:0e:ab:00:00:01
    ETHNAME=$(ip -j a | jq -r '.[] | select(.address=="00:0e:ab:01:00:00") | .ifname')
fi

hostname $NAME
ethtool --set-priv-flags $ETHNAME roce-opt on
ip link set $ETHNAME a $MACADDR
ip addr add dev $ETHNAME $IPADDR/24
ip link set dev $ETHNAME up
rdma link add rxe_eth0 type rxe netdev $ETHNAME

# Change bash prompt to include the hostname and current directory.
PS1="(VM)\u@$NAME:\W $ "

if [[ $NAME = "node0" ]]; then
    ibv_rc_pingpong -g 1 -n 1
else
    # node 1 might have started faster than node 0. Try until success.
    until ibv_rc_pingpong -g 1 -n 1 192.168.1.1; do sleep 1; done
fi
