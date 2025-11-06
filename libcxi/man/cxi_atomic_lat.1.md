---
title: CXI_ATOMIC_LAT(1) Version 2.4.0 | CXI Diagnostics and Utilities
date: 2025-08-05
---

# NAME

cxi_atomic_lat - Cassini NIC Atomic Memory Operation latency benchmark using
LibCXI


# SYNOPSIS

| **cxi_atomic_lat** [**-d** *DEV*] [**-p** *PORT*]
| **cxi_atomic_lat** [*OPTIONS*] *SERVER_ADDR*


# DESCRIPTION

**cxi_atomic_lat** is a simple benchmark for measuring Atomic Memory Operation
(AMO) latency. It can be configured to run a number of iterations or to run
for a duration of time. Latency is measured for a specified atomic operation
and atomic type. Where possible, target writes are ensured to occur with every
AMO. Only client-to-server latency is measured.

## How to run

Two copies of the program must be launched. The first copy is the server. It
accepts optional arguments to specify which CXI device to use and which port
to listen on for the client connection. The second copy is the client. It must
be supplied with the hostname or IPv4 address of the server, along with the
port to connect to if the default value was not used. The client must also be
supplied with all desired run options.

## Output

Once a connection has been established, both the server and client print a
summary of the specified options along with the NIC fabric addresses of both
endpoints. Only the client prints data beyond the summary. If the
**\-\-report-all** option is used, each measured latency is printed in two
columns.

*OpNum*
: The measurement number.

*Latency[us]*
: The measured latency in microseconds. Latency is measured by obtaining the
time just prior to initiating an AMO and just after receiving an
acknowledgement that either the data has been written into the target buffer,
or when using fetching AMOs, when the fetched data has been written into the
initiator buffer. These times are obtained using **clock_gettime**(2).

After the benchmark runs, a latency summary is printed in six columns.

*AMO Size[B]*
: The size of the AMOs.

*Ops*
: The number of AMOs performed.

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

**-n, \-\-iters**=*ITERS*
: The number of iterations to perform. A single iteration consists of sending
a single AMO and then waiting for it to be acknowledged, or when using
fetching AMOs, for its response to be written to the initiator. When *ITERS*
and *DURATION* are both unspecified, 100 iterations are performed.

**-D, \-\-duration**=*DURATION*
: Continue running iterations until *DURATION* seconds has elapsed.

**\-\-warmup**=*WARMUP*
: Perform *WARMUP* additional iterations and discard their latency
measurements. By default 10 warmup iterations are performed.

**\-\-latency-gap**=*GAP*
: Wait *GAP* microseconds between each iteration. By default the gap is 1000
microseconds.

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

**\-\-fetching**
: Use fetching AMOs. By default non-fetching AMOs are used.

**\-\-matching**
: Use matching list entries at the target. By default non-matching list
entries are used.

**\-\-unrestricted**
: Use unrestricted AMOs. By default restricted AMOs are used when
non-matching list entries are used at the target.

**\-\-no-idc**
: Disable the use of Immediate Data Commands. By default IDCs are used when
non-matching list entries are used at the target.

**\-\-no-ll**
: Disable the Low-Latency command issue mechanism.

**\-\-report-all**
: Print each latency after it is measured.

**-h, \-\-help**
: Display the help text and exit.

**-V, \-\-version**
: Display the program version and exit.


# EXAMPLES

## Example 1

Running with the default options

*Server*
```
$ cxi_atomic_lat
Listening on port 49194 for client to connect...
-----------------------------------------------------------------------
    CXI Atomic Memory Operation Latency Test
Device           : cxi0
Test Type        : Iteration
Iterations       : 100
Warmup Iters     : 10
Inter-Iter Gap   : 1000 microseconds
Atomic Op        : NON-FETCHING SUM
Atomic Type      : UINT64
IDC              : Enabled
Matching LEs     : Disabled
Restricted       : Enabled
LL Cmd Launch    : Enabled
Results Reported : Summary
Local (server)   : NIC 0x12 PID 0 VNI 10
Remote (client)  : NIC 0x13 PID 0
-----------------------------------------------------------------------
See client for results.
-----------------------------------------------------------------------
```

*Client*
```
$ cxi_atomic_lat cxi-nid0
-----------------------------------------------------------------------
    CXI Atomic Memory Operation Latency Test
Device           : cxi0
Test Type        : Iteration
Iterations       : 100
Warmup Iters     : 10
Inter-Iter Gap   : 1000 microseconds
Atomic Op        : NON-FETCHING SUM
Atomic Type      : UINT64
IDC              : Enabled
Matching LEs     : Disabled
Restricted       : Enabled
LL Cmd Launch    : Enabled
Results Reported : Summary
Local (client)   : NIC 0x13 PID 0 VNI 10
Remote (server)  : NIC 0x12 PID 0
-----------------------------------------------------------------------
AMO Size[B]         Ops     Min[us]     Max[us]    Mean[us]  StdDev[us]
          8         100        2.48        2.63        2.50        0.06
-----------------------------------------------------------------------
```

## Example 2

Printing all measurements

*Server*
```
$ cxi_atomic_lat
Listening on port 49194 for client to connect...
-----------------------------------------------------------------------
    CXI Atomic Memory Operation Latency Test
Device           : cxi0
Service ID       : 1
Test Type        : Iteration
Iterations       : 5
Warmup Iters     : 10
Inter-Iter Gap   : 1000 microseconds
Atomic Op        : FETCHING SUM
Atomic Type      : UINT64
IDC              : Enabled
Matching LEs     : Disabled
Restricted       : Enabled
LL Cmd Launch    : Enabled
Results Reported : All
Local (server)   : NIC 0x12 PID 0 VNI 10
Remote (client)  : NIC 0x13 PID 0
-----------------------------------------------------------------------
See client for results.
-----------------------------------------------------------------------
```

*Client*
```
$ cxi_atomic_lat cxi-nid0 --fetching --report-all -n 5
-----------------------------------------------------------------------
    CXI Atomic Memory Operation Latency Test
Device           : cxi0
Service ID       : 1
Test Type        : Iteration
Iterations       : 5
Warmup Iters     : 10
Inter-Iter Gap   : 1000 microseconds
Atomic Op        : FETCHING SUM
Atomic Type      : UINT64
IDC              : Enabled
Matching LEs     : Disabled
Restricted       : Enabled
LL Cmd Launch    : Enabled
Results Reported : All
Local (client)   : NIC 0x13 PID 0 VNI 10
Remote (server)  : NIC 0x12 PID 0
-----------------------------------------------------------------------
     OpNum  Latency[us]
         0        2.554
         1        2.585
         2        2.594
         3        2.565
         4        2.585
-----------------------------------------------------------------------
AMO Size[B]         Ops     Min[us]     Max[us]    Mean[us]  StdDev[us]
          8           5        2.55        2.59        2.57        0.05
-----------------------------------------------------------------------
```


# SEE ALSO

**cxi_diags**(7)
