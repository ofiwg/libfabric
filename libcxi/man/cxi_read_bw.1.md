---
title: CXI_READ_BW(1) Version 2.4.0 | CXI Diagnostics and Utilities
date: 2025-08-05
---

# NAME

cxi_read_bw - Cassini NIC one-sided RDMA read bandwidth benchmark using LibCXI


# SYNOPSIS

| **cxi_read_bw** [**-d** *DEV*] [**-p** *PORT*]
| **cxi_read_bw** [*OPTIONS*] *SERVER_ADDR*


# DESCRIPTION

**cxi_read_bw** is a simple benchmark for measuring one-sided RDMA read
bandwidth. It can be configured to run a number of iterations or to run for a
duration of time. Bandwidth is measured for a specific size of reads, or for
each size within a range. Either bi-directional or uni-directional bandwidth can
be measured. Users have the option to allocate source and destination buffers
from system memory or from an available GPU device via calls to the HIP, CUDA,
or INTEL Level-zero APIs.

## How to run

Two copies of the program must be launched. The first copy is the server. It
accepts optional arguments to specify which CXI device to use, which port to
listen on for the client connection, and whether or not GPU buffers should be
used. The second copy is the client. It must be supplied with the hostname or
IPv4 address of the server, along with the port to connect to if the default
value was not used. The client must also be supplied with all desired run
options.

## Output

Both the server and client print output, much of which is the same between
the two. Once a connection has been established, a summary of the specified
options is printed along with the server and client NIC fabric addresses.
After the benchmark runs (or after each run for a range of sizes), four
columns of data are printed.

*RDMA Size[B]*
: The size of the RDMA reads.

*Reads*
: The number of reads performed. When measuring bi-directional bandwidth, the
server and client only report the number of reads they initiate, not the
combined count.

*BW[MB/s]*
: The measured bandwidth in megabytes per second. When measuring bi-directional
bandwidth, the server and client bandwidths are combined and reported by both
sides.

*PktRate[Mpkt/s]*
: The measured packet rate in millions of packets per second. When measuring
bi-directional bandwidth, the server and client packet rates are combined and
reported by both sides. Reads larger than the Portals MTU (2048 bytes) are
split into multiple packets.


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
: The number of iterations to perform. A single iteration consists of sending
*LIST_SIZE* reads and then waiting for each to be acknowledged. When a range
of read sizes is specified, *ITERS* iterations are performed for each size.
When *ITERS* and *DURATION* are both unspecified, 1000 iterations are
performed.

**-D, \-\-duration**=*DURATION*
: This option configures the benchmark to continue running iterations until
*DURATION* seconds has elapsed. When a range of read sizes is specified,
the benchmark runs for *DURATION* seconds for each size.

**-s, \-\-size**=*MIN*[:*MAX*]
: The read size, or range of sizes. Valid read sizes are 1 through 4294697295
bytes (4GiB - 1). When a range is specified, the benchmark runs once for each
size that is a power of two from *MIN* to *MAX*. By default the read size is
65536 bytes.

**-l, \-\-list-size**=*LIST_SIZE*
: The number of read performed per iteration. The read commands are all
added to the TX command queue before ringing the queue's doorbell to trigger
the reads. By default the list size is 256.

**-b, \-\-bidirectional**
: This option configures the benchmark measure bi-directional bandwidth. The
client and server calculate their individual rates, share these with each
other, and then both report the combined rates. By default the benchmark
performs reads from the client to the server only. In this case, the client
rates are still shared and reported from both client and server. Running
bi-directionally works best when combined with **\-\-duration** rather than
**\-\-iters**.

**\-\-unrestricted**
: Use unrestricted reads. By default restricted reads are used.

**\-\-buf-sz**=*BUF_SIZE*
: The maximum allowed data buffer size for both the transmit and target
buffers, specified in bytes. The benchmark may allocate less space if it is
possible to fit all *LIST_SIZE* reads of a single iteration, without overlap,
aligned to *BUF_ALIGN* boundaries. If *BUF_SIZE* is not a multiple of the page
size, it is rounded up to the nearest page boundary. However, read offsets
within the buffers are still chosen as if the buffer is only *BUF_SIZE* bytes.
If the reads for a single iteration do not fit without overlap, their offsets
are wrapped around.

**\-\-buf-align**=*BUF_ALIGN*
: The alignment of reads in the transmit and target buffers. By default this
is 64 bytes.

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

Running with the default options

*Server*
```
$ cxi_read_bw
Listening on port 49194 for client to connect...
---------------------------------------------------
    CXI RDMA Read Bandwidth Test
Device           : cxi0
Service ID       : 1
Client TX Mem    : System
Server RX Mem    : System
Test Type        : Iteration
Iterations       : 1000
Read Size        : 65536
List Size        : 256
Restricted       : Enabled
Bidirectional    : Disabled
Max RDMA Buf     : 4294967296 (16777216 used)
RDMA Buf Align   : 64
Hugepages        : Disabled
Local (server)   : NIC 0x12 PID 0 VNI 10
Remote (client)  : NIC 0x13 PID 0
---------------------------------------------------
RDMA Size[B]       Reads  BW[MB/s]  PktRate[Mpkt/s]
       65536           -  21479.54        10.488056
---------------------------------------------------
```

*Client*
```
$ cxi_read_bw 192.168.1.1
---------------------------------------------------
    CXI RDMA Read Bandwidth Test
Device           : cxi0
Service ID       : 1
Client TX Mem    : System
Server RX Mem    : System
Test Type        : Iteration
Iterations       : 1000
Read Size        : 65536
List Size        : 256
Restricted       : Enabled
Bidirectional    : Disabled
Max RDMA Buf     : 4294967296 (16777216 used)
RDMA Buf Align   : 64
Hugepages        : Disabled
Local (client)   : NIC 0x13 PID 0 VNI 10
Remote (server)  : NIC 0x12 PID 0
---------------------------------------------------
RDMA Size[B]       Reads  BW[MB/s]  PktRate[Mpkt/s]
       65536      256000  21479.54        10.488056
---------------------------------------------------
```

## Example 2

Running bi-directionally over a range of sizes for 5 seconds each

*Server*
```
$ cxi_read_bw
Listening on port 49194 for client to connect...
---------------------------------------------------
    CXI RDMA Read Bandwidth Test
Device           : cxi0
Service ID       : 1
Client TX Mem    : System
Client RX Mem    : System
Server TX Mem    : System
Server RX Mem    : System
Test Type        : Duration
Duration         : 5 seconds
Min Read Size    : 1024
Max Read Size    : 65536
List Size        : 256
Restricted       : Enabled
Bidirectional    : Enabled
Max RDMA Buf     : 4294967296 (16777216 used)
RDMA Buf Align   : 64
Hugepages        : Disabled
Local (server)   : NIC 0x12 PID 0 VNI 10
Remote (client)  : NIC 0x13 PID 0
---------------------------------------------------
RDMA Size[B]       Reads  BW[MB/s]  PktRate[Mpkt/s]
        1024    65509120  26832.33        26.203444
        2048    40536576  33207.14        16.214422
        4096    22533376  36917.99        18.026362
        8192    11985408  39272.99        19.176267
       16384     6146560  40281.21        19.668559
       32768     3193088  41111.10        20.073777
       65536     1436928  38659.25        18.876588
---------------------------------------------------
```

*Client*
```
$ cxi_read_bw 192.168.1.1 -D 5 -s 1024:65536 -b
---------------------------------------------------
    CXI RDMA Read Bandwidth Test
Device           : cxi0
Service ID       : 1
Client TX Mem    : System
Client RX Mem    : System
Server TX Mem    : System
Server RX Mem    : System
Test Type        : Duration
Duration         : 5 seconds
Min Read Size    : 1024
Max Read Size    : 65536
List Size        : 256
Restricted       : Enabled
Bidirectional    : Enabled
Max RDMA Buf     : 4294967296 (16777216 used)
RDMA Buf Align   : 64
Hugepages        : Disabled
Local (client)   : NIC 0x13 PID 0 VNI 10
Remote (server)  : NIC 0x12 PID 0
---------------------------------------------------
RDMA Size[B]       Reads  BW[MB/s]  PktRate[Mpkt/s]
        1024    65511168  26832.33        26.203444
        2048    40537600  33207.14        16.214422
        4096    22533632  36917.99        18.026362
        8192    11985920  39272.99        19.176267
       16384     6146560  40281.21        19.668559
       32768     3080192  41111.10        20.073777
       65536     1512960  38659.25        18.876588
---------------------------------------------------
```


# SEE ALSO

**cxi_diags**(7)
