#!/bin/bash

# Starts a VM with 2 simulated Cassini devices and run some basic
# tests link ping and iperf.

# For IPv6, use the automatically link local addresses (in fe80::/10)

# Start 2 VMs, that will execute this script.
HYP=$(grep -c "^flags.* hypervisor" /proc/cpuinfo)
if [[ $HYP -eq 0 ]]; then
    VM_INIT_SCRIPT="$0" ./startvm.sh -N 2
    exit 0
fi

# Load all the drivers
modprobe ptp
modprobe amd_iommu_v2 || modprobe iommu_v2
insmod ../../slingshot_base_link/cxi-sbl.ko
insmod ../../sl-driver/drivers/net/ethernet/hpe/sl/cxi-sl.ko
insmod ../drivers/net/ethernet/hpe/ss1/cxi-ss1.ko disable_default_svc=0
insmod ../drivers/net/ethernet/hpe/ss1/cxi-user.ko
insmod ../drivers/net/ethernet/hpe/ss1/cxi-eth.ko

# Wait for the new devices to appear
sleep 1

# Find both interface names. One has MAC address 00:0e:ab:00:00:00 and
# the other has 00:0e:ab:01:00:00.  netsim can route the packets
# between the nodes.
ETH1=$(ip -j a | jq -r '.[] | select(.address=="00:0e:ab:00:00:00") | .ifname')
ETH2=$(ip -j a | jq -r '.[] | select(.address=="00:0e:ab:01:00:00") | .ifname')

# The MAC address has to be changed so that netsim can route the
# packets.
ip link set $ETH1 a 02:0e:ab:00:00:00
ip link set $ETH2 a 02:0e:ab:00:00:01

# Use namespace to force packets to go through the whole stack.
# Setup taken from
# https://serverfault.com/questions/127636/force-local-ip-traffic-to-an-external-interface
ip netns add ns_server
ip link set $ETH1 netns ns_server
ip -n ns_server addr add dev $ETH1 192.168.1.1/24
ip -n ns_server -f inet6 addr add dev $ETH1 fe80::e:abff:fe00:0/64
ip -n ns_server link set dev $ETH1 up

ip netns add ns_client
ip link set $ETH2 netns ns_client
ip -n ns_client addr add dev $ETH2 192.168.1.2/24
ip -n ns_client -f inet6 addr add dev $ETH2 fe80::e:abff:fe00:1/64
ip -n ns_client link set dev $ETH2 up

# Check that going up and down doesn't break things, since resources
# are allocated and then freed every cycle.
ip -n ns_client link set dev $ETH2 up
ip -n ns_client link set dev $ETH2 down
ip -n ns_client link set dev $ETH2 up
ip -n ns_client link set dev $ETH2 down
ip -n ns_client link set dev $ETH2 up

sleep 2

# System is up and running. Test IPv4
ip netns exec ns_server ping -c 1 192.168.1.2
ip netns exec ns_client ping -c 1 192.168.1.1

# System is up and running. Test IPv6
ip netns exec ns_server ping6 -c 1 fe80::e:abff:fe00:1%$ETH1
ip netns exec ns_client ping6 -c 1 fe80::e:abff:fe00:0%$ETH2

# MTU test
ip -n ns_server link set dev $ETH1 mtu 2510
ip -n ns_server link set dev $ETH1 mtu 1000
ip -n ns_server link set dev $ETH1 mtu 1500

# IPERF test - IPv4
ip netns exec ns_server iperf3 -B 192.168.1.1 -s &
ip netns exec ns_client iperf3 -B 192.168.1.2 -c 192.168.1.1

# IPERF test - IPv6
ip netns exec ns_server iperf3 -B fe80::e:abff:fe00:0%$ETH1 -s &
ip netns exec ns_client iperf3 -B fe80::e:abff:fe00:1%$ETH2 -c fe80::e:abff:fe00:0

#ip netns exec ns_client ethtool -i $ETH2

#ip netns exec ns_server ethtool -l $ETH1

#ip netns exec ns_server ethtool -x $ETH1

#ip netns exec ns_server ethtool -X $ETH1 equal 2

#ip netns exec ns_server ethtool -L $ETH1 rx 4
#ip netns exec ns_server ethtool -L $ETH1 rx 2
#ip netns exec ns_server ethtool -L $ETH1 rx 1
#ip netns exec ns_server ethtool -x $ETH1

#ip netns exec ns_server ethtool -n $ETH1 rx-flow-hash udp4
#ip netns exec ns_server ethtool -N $ETH1 rx-flow-hash udp4
#ip netns exec ns_server ethtool -N $ETH1 flow-type ip4
#ip netns exec ns_server ethtool -N $ETH1 flow-type udp6 dst-port 4791

#ip netns exec ns_server ethtool -X $ETH1 equal 3

#ip netns exec ns_server ethtool -X $ETH1 equal 4
