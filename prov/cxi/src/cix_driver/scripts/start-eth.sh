#!/bin/bash
#
# Bring up an Ethernet interface based NIC address. Netsim will route Ethernet
# packets based on the lower 20 bits of the MAC address.
#
# This script can be used to bring up a Ethernet and IP interface used to ping
# across VMs. To start multiple VMs, run the following:
#
# USE_XTERM=1 ./startvm.sh -n 2 -dd
#
# Then run this script on each xterm to bring up the Ethernet and IP interface.

insmod ../drivers/net/ethernet/hpe/ss1/cxi-eth.ko

# Short sleep to let Linux configure Ethernet interface.
sleep 3

echo 8 > /proc/sys/kernel/printk
echo -n 'module cxi_eth +p' > /sys/kernel/debug/dynamic_debug/control

# Locate the first down Ethernet interface and configure it.
regex="eth([0-9]{1}).+DOWN"
eth_id=-1
interfaces="$(ip addr)"
if [[ $interfaces =~ $regex ]]; then
	eth_id=${BASH_REMATCH[1]}
fi

if [ $eth_id -eq -1 ]; then
	echo "Failed to find Ethernet interface"
	exit 1
fi

# Build MAC address.
nid=$(cat /sys/class/cxi/cxi0/device/properties/nid)
octet="$(printf "%02X" $nid)"
mac_addr="00:0E:AB:00:00:$octet"

# Build IP address based on NIC address.
nid=$(printf "%d" $nid)
nid=$(($nid+1))
ip_addr=$(printf "192.168.1.%d/24" $nid)
ip link set eth$eth_id address $mac_addr
ip addr add dev eth$eth_id $ip_addr
ip link set dev eth$eth_id up

# Configure multiple VLAN interfaces each with a different PCP value
for pcp in {0..7}; do
	vlan_id=$(($pcp+1))
	ip link add link eth$eth_id name eth$eth_id.$vlan_id type vlan id $vlan_id egress-qos-map 0:$pcp
	ip link set dev eth$eth_id.$vlan_id up

	vlan_ip_addr=$(printf "192.168.%d.%d/24" $(($pcp+2)) $nid)
	ip addr add dev eth$eth_id.$vlan_id $vlan_ip_addr
done
