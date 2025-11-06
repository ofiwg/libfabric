---
title: CXI_ATOMIC_BW(1) Version 2.4.0 | CXI Diagnostics and Utilities
date: 2025-08-05
---

# NAME

cxi_atomic_bw - Cassini NIC Atomic Memory Operation bandwidth benchmark using
LibCXI


# SYNOPSIS

| **cxi_atomic_bw** [**-d** *DEV*] [**-p** *PORT*]
| **cxi_atomic_bw** [*OPTIONS*] *SERVER_ADDR*


# DESCRIPTION

**cxi_atomic_bw** is a simple benchmark for measuring Atomic Memory Operation
(AMO) bandwidth. It can be configured to run a number of iterations or to run
for a duration of time. Bandwidth is measured for a specified atomic operation
and atomic type. Where possible, target writes are ensured to occur with every
AMO. Either bi-directional or uni-directional bandwidth can be measured.

## How to run

Two copies of the program must be launched. The first copy is the server. It
accepts optional arguments to specify which CXI device to use and which port
to listen on for the client connection. The second copy is the client. It must
be supplied with the hostname or IPv4 address of the server, along with the
port to connect to if the default value was not used. The client must also be
supplied with all desired run options.

## Output

Both the server and client print output, much of which is the same between
the two. Once a connection has been established, a summary of the specified
options is printed along with the server and client NIC fabric addresses.
After the benchmark runs, four columns of data are printed.

*AMO Size[B]*
: The size of the AMOs.

*Ops*
: The number of AMOs performed. When measuring bi-directional bandwidth, the
server and client only report the number of AMOs they initiate, not the
combined count.

*BW[MB/s]*
: The measured bandwidth in megabytes per second. When measuring bi-directional
bandwidth, the server and client bandwidths are combined and reported by both
sides.

*OpRate[M/s]*
: The measured AMO rate in millions of AMOs per second. When measuring
bi-directional bandwidth, the server and client AMO rates are combined and
reported by both sides.


# OPTIONS

**-d, \-\-device**=*DEV*
: The Cassini NIC device name to use. When unspecified, \"cxi0\" is used.

**-v, \-\-svc-id**=*SVC_ID*
: The service ID to use. This option will be used when running as a
non-privileged user. When unspecified the default service ID is used.

**-p, \-\-port**=*PORT*
: The TCP port number to use for the client/server connection. When
unspecified, port 49194 is used.

**-n, \-\-iters**=*ITERS*
: The number of iterations to perform. A single iteration consists of sending
*LIST_SIZE* AMOs and then waiting for each to be acknowledged, or when using
fetching AMOs, for each response to be written to the initiator. When *ITERS*
and *DURATION* are both unspecified, 10000 iterations are performed.

**-D, \-\-duration**=*DURATION*
: This option configures the benchmark to continue running iterations until
*DURATION* seconds has elapsed.

**-l, \-\-list-size**=*LIST_SIZE*
: The number of AMOs performed per iteration. The AMO commands are all added
to the TX command queue before ringing the queue's doorbell to trigger the
AMOs. By default the list size is 4096.

**-A, \-\-atomic-op**=*ATOMIC_OP*
: The atomic operation to use. When unspecified, SUM is used. Supported ops
are: MIN, MAX, SUM, LOR, LAND, BOR, BAND, LXOR, BXOR, SWAP, CSWAP, and AXOR.

**-C, \-\-cswap-op**=*CSWAP_OP*
: The CSWAP operation to use when *ATOMIC_OP* is set to CSWAP. When
unspecified, EQ is used. Supported ops are: EQ, NE, LE, LT, GE, and GT.

**-T, \-\-atomic-type**=*ATOMIC_TYPE*
: The atomic type to use. When unspecified, UINT64 is used. Supported types
are: INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, DOUBLE,
FLOAT_COMPLEX, DOUBLE_COMPLEX, and UINT128.

**-b, \-\-bidirectional**
: This option configures the benchmark to measure bi-directional bandwidth.
The client and server calculate their individual rates, share these with each
other, and then both report the combined rates. By default the benchmark
performs AMOs from the client to the server only. In this case, the client
rates are still shared and reported from both client and server. Running
bi-directionally works best when combined with **\-\-duration** rather than
**\-\-iters**.

**\-\-fetching**
: Use fetching AMOs. By default non-fetching AMOs are used.

**\-\-matching**
: Use matching list entries at the target. By default non-matching list
entries are used.

**\-\-unrestricted**
: Use unrestricted AMOs. By default restricted AMOs are used when
non-matching list entries are used at the target.

**\-\-no-hrp**
: Disable the use of High Rate Puts. By default HRPs are used for restricted,
non-fetching AMOs.

**\-\-no-idc**
: Disable the use of Immediate Data Commands. By default IDCs are used when
non-matching list entries are used at the target.

**-h, \-\-help**
: Display the help text and exit.

**-V, \-\-version**
: Display the program version and exit.


# EXAMPLES

*Server*
```
$ cxi_atomic_bw -p 42000
Listening on port 42000 for client to connect...
------------------------------------------------
    CXI Atomic Memory Operation Bandwidth Test
Device          : cxi0
Service ID      : 1
Test Type       : Duration
Duration        : 2 seconds
Atomic Op       : NON-FETCHING CSWAP EQ
Atomic Type     : UINT128
List Size       : 4096
HRP             : Enabled
IDC             : Enabled
Matching LEs    : Disabled
Restricted      : Enabled
Bidirectional   : Enabled
Local (server)  : NIC 0x12 PID 0 VNI 10
Remote (client) : NIC 0x13 PID 0
------------------------------------------------
AMO Size[B]         Ops  BW[MB/s]    OpRate[M/s]
         16    67584000   1081.16      67.572741
------------------------------------------------
```

*Client*
```
$ cxi_atomic_bw cxi-nid0 -p 42000 -A cswap -C eq -T uint128 -D 2 -b
------------------------------------------------
    CXI Atomic Memory Operation Bandwidth Test
Device          : cxi0
Service ID      : 1
Test Type       : Duration
Duration        : 2 seconds
Atomic Op       : NON-FETCHING CSWAP EQ
Atomic Type     : UINT128
List Size       : 4096
HRP             : Enabled
IDC             : Enabled
Matching LEs    : Disabled
Restricted      : Enabled
Bidirectional   : Enabled
Local (client)  : NIC 0x13 PID 0 VNI 10
Remote (server) : NIC 0x12 PID 0
------------------------------------------------
AMO Size[B]         Ops  BW[MB/s]    OpRate[M/s]
         16    67575808   1081.16      67.572741
------------------------------------------------
```


# SEE ALSO

**cxi_diags**(7)
