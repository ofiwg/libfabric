---
title: CXI_READ_LAT(1) Version 2.4.0 | CXI Diagnostics and Utilities
date: 2025-08-05
---

# NAME

cxi_read_lat - Cassini NIC one-sided RDMA read latency benchmark using LibCXI


# SYNOPSIS

| **cxi_read_lat** [**-d** *DEV*] [**-p** *PORT*]
| **cxi_read_lat** [*OPTIONS*] *SERVER_ADDR*


# DESCRIPTION

**cxi_read_lat** is a simple benchmark for measuring one-sided RDMA read
latency.  It can be configured to run a number of iterations or to run for a
duration of time. Latency is measured for a specific size of reads, or for each
size within a range. Only client-to-server latency is measured. Users have the
option to allocate source and destination buffers from system memory or from an
available GPU device via calls to the HIP, CUDA, or INTEL Level-zero APIs.

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

*ReadNum*
: The measurement number.

*Latency[us]*
: The measured latency in microseconds. Latency is measured by obtaining the
time just prior to initiating a read and just after receiving an
acknowledgement signifying that the data has been read into the target
buffer. These times are obtained using **clock_gettime**(2).

After the benchmark runs (or after each run for a range of sizes), a latency
summary is printed in six columns.

*RDMA Size[B]*
: The size of the RDMA reads.

*Reads*
: The number of reads performed.

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
single read and then waiting for it to be acknowledged. When a range of read
sizes is specified, *ITERS* iterations are performed for each size. When *ITERS*
and *DURATION* are both unspecified, 100 iterations are performed.

**-D, \-\-duration**=*DURATION*
: Continue running iterations until *DURATION* seconds has elapsed. When a
range of read sizes is specified, the benchmark runs for *DURATION* seconds
for each size.

**\-\-warmup**=*WARMUP*
: Perform *WARMUP* additional iterations and discard their latency
measurements. When a range of read sizes is specified, the warmup iterations
are performed for each size. By default 10 warmup iterations are performed.

**\-\-latency-gap**=*GAP*
: Wait *GAP* microseconds between each iteration. By default the gap is 1000
microseconds.

**-s, \-\-size**=*MIN*[:*MAX*]
: The read size, or range of sizes. Valid read sizes are 1 through 4294697295
bytes (4GiB - 1). When a range is specified, the benchmark runs once for each
size that is a power of two from *MIN* to *MAX*. By default the read size are
8 bytes.

**\-\-unrestricted**
: Use unrestricted reads. By default restricted reads are used.

**\-\-no-ll**
: Disable the Low-Latency command issue mechanism.

**\-\-report-all**
: Print each latency after it is measured.

**\-\-use-hp**=*HP_SIZE*
: The size of huge pages to use when allocating the transmit and target buffers.
This option has no effect when GPU buffers are used. Valid values are 2M and 1G.
By default huge pages are disabled.

**-h, \-\-help**
: Display the help text and exit.

**-V, \-\-version**
: Display the program version and exit.


# EXAMPLES

## Example 1

Running over a range of sizes for 1 second each

*Server*
```
$ cxi_read_lat
Listening on port 49194 for client to connect...
------------------------------------------------------------------------
    CXI RDMA Read Latency Test
Device           : cxi0
Service ID       : 1
Client TX Mem    : System
Server RX Mem    : System
Test Type        : Duration
Duration         : 1 seconds
Warmup Iters     : 10
Inter-Iter Gap   : 1000 microseconds
Min Read Size    : 1
Max Read Size    : 1024
Restricted       : Enabled
LL Cmd Launch    : Enabled
Results Reported : Summary
Hugepages        : Disabled
Local (server)   : NIC 0x12 PID 0 VNI 10
Remote (client)  : NIC 0x12 PID 1
------------------------------------------------------------------------
See client for results.
------------------------------------------------------------------------
```

*Client*
```
$ cxi_read_lat 192.168.1.1 -D 1 -s 1:1024
------------------------------------------------------------------------
    CXI RDMA Read Latency Test
Device           : cxi0
Service ID       : 1
Client TX Mem    : System
Server RX Mem    : System
Test Type        : Duration
Duration         : 1 seconds
Warmup Iters     : 10
Inter-Iter Gap   : 1000 microseconds
Min Read Size    : 1
Max Read Size    : 1024
Restricted       : Enabled
LL Cmd Launch    : Enabled
Results Reported : Summary
Hugepages        : Disabled
Local (client)   : NIC 0x12 PID 1 VNI 10
Remote (server)  : NIC 0x12 PID 0
------------------------------------------------------------------------
RDMA Size[B]       Reads     Min[us]     Max[us]    Mean[us]  StdDev[us]
           1       97084        2.62       77.59        2.70        0.25
           2       97135        2.61       13.44        2.70        0.07
           4       97303        2.60       12.75        2.69        0.08
           8       97268        2.59       33.23        2.69        0.12
          16       97205        2.61        4.99        2.70        0.04
          32       97221        2.61        5.65        2.70        0.05
          64       97181        2.61       13.91        2.70        0.07
         128       97080        2.63        5.55        2.71        0.07
         256       96887        2.66        6.17        2.74        0.07
         512       96548        2.71        5.44        2.77        0.05
        1024       95613        2.78       11.57        2.87        0.07
------------------------------------------------------------------------
```

## Example 2

Printing all measurements

*Server*
```
$ cxi_read_lat
Listening on port 49194 for client to connect...
------------------------------------------------------------------------
    CXI RDMA Read Latency Test
Device           : cxi0
Service ID       : 1
Client TX Mem    : System
Server RX Mem    : System
Test Type        : Iteration
Iterations       : 5
Warmup Iters     : 10
Inter-Iter Gap   : 1000 microseconds
Read Size        : 8
Restricted       : Enabled
LL Cmd Launch    : Enabled
Results Reported : All
Hugepages        : Disabled
Local (server)   : NIC 0x12 PID 0 VNI 10
Remote (client)  : NIC 0x12 PID 1
------------------------------------------------------------------------
See client for results.
------------------------------------------------------------------------
```

*Client*
```
$ cxi_read_lat 192.168.1.1 -n 5 --report-all
------------------------------------------------------------------------
    CXI RDMA Read Latency Test
Device           : cxi0
Service ID       : 1
Client TX Mem    : System
Server RX Mem    : System
Test Type        : Iteration
Iterations       : 5
Warmup Iters     : 10
Inter-Iter Gap   : 1000 microseconds
Read Size        : 8
Restricted       : Enabled
LL Cmd Launch    : Enabled
Results Reported : All
Hugepages        : Disabled
Local (client)   : NIC 0x12 PID 1 VNI 10
Remote (server)  : NIC 0x12 PID 0
------------------------------------------------------------------------
   ReadNum  Latency[us]
         0        2.715
         1        2.715
         2        2.725
         3        2.725
         4        2.716
------------------------------------------------------------------------
RDMA Size[B]       Reads     Min[us]     Max[us]    Mean[us]  StdDev[us]
           8           5        2.71        2.72        2.71        0.03
------------------------------------------------------------------------
```


# SEE ALSO

**cxi_diags**(7)
