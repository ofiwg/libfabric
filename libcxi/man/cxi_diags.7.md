---
title: CXI_DIAGS(7) Version 1.0.0 | CXI Diagnostics and Utilities
date: 2021-08-05
---

# NAME

CXI Diagnostics and Utilities


# SYNOPSIS

The CXI Diagnostics and Utilities package contains a set of basic performance
tests and utilities designed for the Slingshot 200GbE network interface.


# DESCRIPTION

## Device Info

Basic device information and status can be viewed using **cxi_stat**.

## Bandwidth

There are four client/server tests that can be used to measure point-to-point or
loopback bandwidth for various types of network traffic. **cxi_write_bw** and
**cxi_read_bw** measure one-sided RDMA writes and reads, respectively.
**cxi_send_bw** measures two-sided sends. **cxi_atomic_bw** measures atomic
memory operations.

**cxi_gpu_bw_loopback** runs as a single program rather than as a client/server
pair. It measures loopback RDMA write bandwidth with an option to use GPU
memory.

Bandwidth is calculated using only the frame payload. It can be measured
uni-directionally or bi-directionally. Uni-directional bandwidth is measured by
the client, shared with the server, and reported by both. Bi-directional
bandwidth is measured by both the client and server, who share their results and
report the combined value.

## Latency

There are four client/server tests that can be used to measure point-to-point or
loopback latency for various types of network traffic. **cxi_write_lat** and
**cxi_read_lat** measure one-sided RDMA writes and reads, respectively.
**cxi_send_lat** measures two-sided sends. **cxi_atomic_lat** measures atomic
memory operations.

Latency is measured by obtaining the start and end times for individual
transactions. All four tests use the point when software initiates the
transaction as the start time. For **cxi_write_lat**, **cxi_read_lat**, and
**cxi_atomic_lat** the end time is the point when software receives an event from
the device indicating that the data has been written to its target host buffer.
In the case of fetching atomics this is the event indicating that the fetched
data has been written. For **cxi_send_lat** a message is sent first from client
to server, then from server to client. The end time is the point when the second
message has been written, and the latency is estimated by halving the round-trip
time.

## Thermal Diagnostic

**cxi_heatsink_check** is a thermal diagnostic intended to validate that heat is
being dissipated properly. It stresses the chip by generating a large amount of
small RDMA writes. When testing a dual-NIC card, this tool must be run for both
NICs simultaneously.


# SEE ALSO

**cxi_atomic_bw**(1), **cxi_atomic_lat**(1), **cxi_gpu_loopback_bw**(1),
**cxi_heatsink_check**(1), **cxi_read_bw**(1), **cxi_read_lat**(1),
**cxi_send_bw**(1), **cxi_send_lat**(1), **cxi_stat**(1), **cxi_write_bw**(1),
**cxi_write_lat**(1)
