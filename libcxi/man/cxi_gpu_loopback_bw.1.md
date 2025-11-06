---
title: CXI_GPU_LOOPBACK_BW(1) Version 1.5.1 | CXI Diagnostics and Utilities
date: 2025-08-05
---

# NAME

cxi_gpu_loopback_bw - Cassini with optional GPU loopback bandwidth benchmark
using LibCXI


# SYNOPSIS

| **cxi_gpu_loopback_bw** [*OPTIONS*]


# DESCRIPTION

**cxi_gpu_loopback_bw** is a simple benchmark for measuring loopback write
bandwidth. It can be configured to run some number of iterations or run
for a period of time. Users have the option to allocate source and
destination buffers from system memory or from an available GPU device via
calls to the HIP, CUDA, or INTEL Level-zero APIs.

## How to run

Simply launch the executable on the node under test.

## Output

After the test is launched, a summary of the specified options is printed.
After the benchmark runs, four columns of data are printed.

*RDMA Size[B]*
: The RDMA transfer size in bytes.

*Writes*
: The number of writes performed.

*BW[Gb/s]*
: The measured bandwidth in gigabits per second.

*PktRate[Mpkt/s]*
: The measured packet rate in millions of packets per second.


# OPTIONS

**-d, \-\-device**=*DEV*
: The Cassini NIC device name to use. When unspecified, \"cxi0\" is used.

**-v, \-\-svc-id**=*SVC_ID*
: The service ID to use. This option will be used when running as a
non-privileged user. When unspecified the default service ID is used.

**-t, \-\-tx-gpu**
: The GPU to use for TX buffer allocation. By default system memory is used.

**-r, \-\-rx-gpu**
: The GPU to use for RX buffer allocation. By default system memory is used.

**-g, \-\-gpu-type**
: The GPU type to use with this test. Valid values are AMD, NVIDIA, or INTEL. By default
the type is determined by discovered GPU files on the system.

**-D, \-\-duration**=*DURATION*
: This option configures the benchmark to continue running iterations until
*DURATION* seconds has elapsed. When a range of write sizes is specified,
the benchmark runs for *DURATION* seconds for each size.

**-i, \-\-iters**=*ITERS*
: The number of iterations to perform. A single iteration consists of sending
*NUM_XFERS* writes and then waiting for a counting event to occur after all
writes have completed.  Default = 25 iterations.

**-s, \-\-size**=*SIZE*
: The transfer size in bytes. Valid write sizes are 1 through 4294697295
bytes (4GiB - 1). By default the write size is 524288 bytes.

**-n, \-\-num-xfers**=*NUM_XFERS*
: The number of writes performed per iteration. The write commands are all
added to the TX command queue before ringing the queue's doorbell to trigger
the writes. By default the number of transfers per iteration is 8192.

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

Running with the default options (25 iterations of 8192 512KB transfers using
cxi0 with TX and RX system memory)

```
$ cxi_gpu_loopback_bw
---------------------------------------------------
    CXI Loopback Bandwidth Test
Device           : cxi0
Service ID       : 1
TX Mem           : System
RX Mem           : System
Test Type        : Iteration
Iterations       : 25
Write Size (B)   : 524288
Cmds per iter    : 8192
Hugepages        : Disabled
---------------------------------------------------
RDMA Size[B]      Writes  BW[Gb/s]  PktRate[Mpkt/s]
     524288       204800    188.17        11.487535
---------------------------------------------------
```

## Example 2

Running 10 seconds of 128KB transfers using cxi0 with GPU 0 source memory and
GPU 2 destination memory.

```
$ cxi_gpu_loopback_bw -D 10 -s 131072 --tx-gpu 0 --rx-gpu 2 -g NVIDIA
---------------------------------------------------
    CXI Loopback Bandwidth Test
Device           : cxi0
Service ID       : 1
TX Mem           : GPU 0
RX Mem           : GPU 2
GPU Type         : NVIDIA
Test Type        : Duration
Duration         : 10 seconds
Write Size (B)   : 131072
Cmds per iter    : 8192
Hugepages        : Disabled
Found 4 GPU(s)
---------------------------------------------------
RDMA Size[B]      Writes  BW[Gb/s]  PktRate[Mpkt/s]
      131072     1105920    115.88         7.072772
---------------------------------------------------

```

## Example 3

Running 10 seconds of 1MB transfers using cxi1 with GPU 2 source memory and
system destination memory.

```
$ cxi_gpu_loopback_bw -D 10 -d cxi1 -s 1048576 --tx-gpu 2 -g NVIDIA
---------------------------------------------------
    CXI Loopback Bandwidth Test
Device           : cxi1
Service ID       : 1
TX Mem           : GPU 2
RX Mem           : System
GPU Type         : NVIDIA
Test Type        : Duration
Duration         : 10 seconds
Write Size (B)   : 1048576
Cmds per iter    : 8192
Hugepages        : Disabled
Found 4 GPU(s)
---------------------------------------------------
RDMA Size[B]      Writes  BW[Gb/s]  PktRate[Mpkt/s]
     1048576      229376    186.02        11.353930
---------------------------------------------------
```


# SEE ALSO

**cxi_diags**(7)
