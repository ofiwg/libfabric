---
title: CXI_SEND_LAT(1) Version 2.4.0 | CXI Diagnostics and Utilities
date: 2025-08-05
---

# NAME

cxi_send_lat - Cassini NIC one-sided send latency benchmark using LibCXI


# SYNOPSIS

| **cxi_send_lat** [**-d** *DEV*] [**-p** *PORT*]
| **cxi_send_lat** [*OPTIONS*] *SERVER_ADDR*


# DESCRIPTION

**cxi_send_lat** is a simple benchmark for measuring one-sided send latency.  It
can be configured to run a number of iterations or to run for a duration of
time. Latency is measured for a specific size of sends, or for each size within
a range. Only client-to-server latency is measured. Users have the option to
allocate source and destination buffers from system memory or from an available
GPU device via calls to the HIP, CUDA, or INTEL Level-zero APIs.

## How to run

Two copies of the program must be launched. The first copy is the server. It
accepts optional arguments to specify which CXI device to use, which port to
listen on for the client connection, and whether or not GPU buffers should be
used. The second copy is the client. It must be supplied with the hostname or
IPv4 address of the server, along with the port to connect to if the default
value was not used. The client must also be supplied with all desired run
options.

## Output

Once a connection has been established, both the server and client print a
summary of the specified options along with the NIC fabric addresses of both
endpoints. Only the client prints data beyond the summary. If the
**\-\-report-all** option is used, each measured latency is printed in two
columns.

*SendNum*
: The measurement number.

*Latency[us]*
: The measured latency in microseconds. A message is sent first from client to
server and then from server to client. The time is observed just prior to the
client initiating a send and just after receiving an acknowledgment signifying
that the server's echo has been written into the client buffer. This round-trip
time is then halved to estimate one-way latency. Times are obtained using
**clock_gettime**(2).

After the benchmark runs (or after each run for a range of sizes), a latency
summary is printed in six columns.

*Bytes*
: The size of the send.

*Sends*
: The number of sends performed.

*Min[us]*
: The smallest observed latency in microseconds.

*Max[us]*
: The largest observed latency in microseconds.

*Mean[us]*
: The average latency in microseconds.

*StdDev[us]*
: The standard deviation of the observed latency distribution.


# OPTIONS

**-d, \-\-device**=*DEV*
: The Cassini NIC device name to use. When unspecified, \"cxi0\" is used.

**-v, \-\-svc-id**=*SVC_ID*
: The service ID to use. This option will be used when running as a
non-privileged user. When unspecified the default service ID is used.

**-p, \-\-port**=*PORT*
: The TCP port number to use for the client/server connection. When
unspecified, port 49194 is used.

**-t, \-\-tx-gpu**
: The GPU to use for TX buffer allocation. By default system memory is used.

**-r, \-\-rx-gpu**
: The GPU to use for RX buffer allocation. By default system memory is used.

**-g, \-\-gpu-type**
: The GPU type to use with this test. Valid values are AMD, NVIDIA, or INTEL. By default
the type is determined by discovered GPU files on the system.

**-n, \-\-iters**=*ITERS*
: The number of iterations to perform. An iteration consists of sending a
single send and then waiting for it to be acknowledged. When a range of send
sizes is specified, *ITERS* iterations are performed for each size. When *ITERS*
and *DURATION* are both unspecified, 100 iterations are performed.

**-D, \-\-duration**=*DURATION*
: Continue running iterations until *DURATION* seconds has elapsed. When a
range of send sizes is specified, the benchmark runs for *DURATION* seconds
for each size.

**\-\-warmup**=*WARMUP*
: Perform *WARMUP* additional iterations and discard their latency
measurements. When a range of send sizes is specified, the warmup iterations
are performed for each size. By default 10 warmup iterations are performed.

**\-\-latency-gap**=*GAP*
: Wait *GAP* microseconds between each iteration. By default the gap is 1000
microseconds.

**-s, \-\-size**=*MIN*[:*MAX*]
: The send size, or range of sizes. Valid send sizes are 1 through 4294697295
bytes (4GiB - 1). When a range is specified, the benchmark runs once for each
size that is a power of two from *MIN* to *MAX*. By default the send size is
8 bytes.

**\-\-no-idc**
: Disable the use of Immediate Data Commands. By default IDCs are used for
sizes up to 224 bytes. IDCs are disabled when TX GPU buffers are used.

**\-\-no-ll**
: Disable the Low-Latency command issue mechanism.

**-R, \-\-rdzv**
: Use rendezvous transfers.

**\-\-report-all**
: Print each latency after it is measured.

**\-\-use-hp**=*HP_SIZE*
: The size of huge pages to use when allocating the transmit and target buffers.
This option has no effect when GPU buffers are used. Valid values are 2M and 1G.
By default huge pages are disabled.

**-c, \-\-clock**=*TYPE*
: Clock type used to calculate latency. Valid options are: cycles or clock_gettime.
Default is clock_gettime.

**\-\-ignore-cpu-freq-mismatch**
: Used when clock type is cycles. Ignore CPU frequency mismatch. Mismatch can occur
when ondemand CPU frequency is enabled such as cpufreq_ondemand governor.

**-h, \-\-help**
: Display the help text and exit.

**-V, \-\-version**
: Display the program version and exit.


# EXAMPLES

## Example 1

Running over a range of sizes for 1 second each

*Server*
```
$ cxi_send_lat
Listening on port 49194 for client to connect...
----------------------------------------------------------------------
    CXI RDMA Send Latency Test
Device           : cxi0
Service ID       : 1
Client TX Mem    : System
Server RX Mem    : System
Test Type        : Iteration
Iterations       : 100
Warmup Iters     : 10
Inter-Iter Gap   : 1000 microseconds
Min Send Size    : 1
Max Send Size    : 1024
IDC              : Enabled
LL Cmd Launch    : Enabled
Results Reported : Summary
Hugepages        : Disabled
Local (server)   : NIC 0x12 PID 0 VNI 10
Remote (client)  : NIC 0x13 PID 0
----------------------------------------------------------------------
See client for results.
----------------------------------------------------------------------
```

*Client*
```
$ cxi_send_lat 192.168.1.1 -s 1:1024
----------------------------------------------------------------------
    CXI RDMA Send Latency Test
Device           : cxi0
Service ID       : 1
Client TX Mem    : System
Server RX Mem    : System
Test Type        : Iteration
Iterations       : 100
Warmup Iters     : 10
Inter-Iter Gap   : 1000 microseconds
Min Send Size    : 1
Max Send Size    : 1024
IDC              : Enabled
LL Cmd Launch    : Enabled
Results Reported : Summary
Hugepages        : Disabled
Local (client)   : NIC 0x13 PID 0 VNI 10
Remote (server)  : NIC 0x12 PID 0
----------------------------------------------------------------------
     Bytes       Sends     Min[us]     Max[us]    Mean[us]  StdDev[us]
         1         100        1.60        4.32        1.65        0.27
         2         100        1.60        1.76        1.62        0.05
         4         100        1.60        1.75        1.62        0.05
         8         100        1.60        2.65        1.64        0.13
        16         100        1.60        1.76        1.62        0.05
        32         100        1.61        1.80        1.62        0.05
        64         100        1.61        1.76        1.63        0.04
       128         100        2.20        2.36        2.22        0.06
       256         100        2.26        2.43        2.28        0.06
       512         100        2.31        4.00        2.35        0.17
      1024         100        2.41        2.58        2.43        0.04
----------------------------------------------------------------------
```

## Example 2

Printing all measurements

*Server*
```
$ cxi_send_lat
Listening on port 49194 for client to connect...
----------------------------------------------------------------------
    CXI RDMA Send Latency Test
Device           : cxi0
Service ID       : 1
Client TX Mem    : System
Server RX Mem    : System
Test Type        : Iteration
Iterations       : 5
Warmup Iters     : 10
Inter-Iter Gap   : 1000 microseconds
Send Size        : 8
IDC              : Enabled
LL Cmd Launch    : Enabled
Results Reported : All
Hugepages        : Disabled
Local (server)   : NIC 0x12 PID 0 VNI 10
Remote (client)  : NIC 0x13 PID 0
----------------------------------------------------------------------
See client for results.
----------------------------------------------------------------------
```

*Client*
```
$ cxi_send_lat 192.168.1.1 -n 5 --report-all
----------------------------------------------------------------------
    CXI RDMA Send Latency Test
Device           : cxi0
Service ID       : 1
Client TX Mem    : System
Server RX Mem    : System
Test Type        : Iteration
Iterations       : 5
Warmup Iters     : 10
Inter-Iter Gap   : 1000 microseconds
Send Size        : 8
IDC              : Enabled
LL Cmd Launch    : Enabled
Results Reported : All
Hugepages        : Disabled
Local (client)   : NIC 0x13 PID 0 VNI 10
Remote (server)  : NIC 0x12 PID 0
----------------------------------------------------------------------
   SendNum  Latency[us]
         0        1.688
         1        1.898
         2        1.688
         3        1.693
         4        1.683
----------------------------------------------------------------------
     Bytes       Sends     Min[us]     Max[us]    Mean[us]  StdDev[us]
         8           5        1.68        1.89        1.73        0.08
----------------------------------------------------------------------
```


# SEE ALSO

**cxi_diags**(7)
