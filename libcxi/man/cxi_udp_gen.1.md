---
title: CXI_UDP_GEN(1) Version 1.0.0 | CXI Diagnostics and Utilities
date: 2021-06-21
---

# NAME

cxi_udp_gen - Cassini NIC UDP packet generator


# SYNOPSIS

| **cxi_udp_gen**


# DESCRIPTION

**cxi_udp_gen** is a packet generator, using an Ethernet command queue
to send prepared UDP packet from userspace, bypassing the kernel. As
such, only root can run it.

It is still in development, and its command line arguments are not
definitive.

This man page is not complete.


## How to run

**cxi_udp_gen** runs on one node, and its source IP/MAC and the
destination IP/MAC must be passed on the command line.

The destination may run a listener, for instance netcat or nc, on port
55555, like this:

```
nc -lu 55555
```


## Output

The main information output is the bandwidth the generator was to able
to send at.

  Overall Bandwidth: 9414443306.742228 bytes/s


# OPTIONS

TODO.


# EXAMPLE

```
$ cxi_udp_gen -s 192.168.1.1 -d 192.168.1.2 -n 02:00:00:00:00:12 -m 02:00:00:00:00:13 -b 9014 -t 1 -q 64 -z 1 -w 5 -r 5 -c 0
Running Ethernet test with following arguments
Source IP address: 192.168.1.1
Destination IP address: 192.168.1.2
Source MAC address: 02:00:00:00:00:12
Destination MAC address: 02:00:00:00:00:13
Checksum type: 0
Number of threads: 1
TX queue depth per thread: 64
Packet size: 9014
Batch submit: 1
Warmup seconds: 5
Test seconds: 5
TXQ 0 thread waiting.....
TXQ 0 TX credit count: 64
TXQ 0 thread started!
EQ 0 thread waiting.....
EQ 0 thread started!
Bytes sent including warmup: 94178812840
Packets sent including warmup: 10448060
Overall Bandwidth: 9414443306.742228 bytes/s
```
